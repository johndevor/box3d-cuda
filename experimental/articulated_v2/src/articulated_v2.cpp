// SPDX-License-Identifier: MIT
#include "articulated_v2.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>
namespace {
bool finite(const double* p,size_t n){if(n&&!p)return false;for(size_t i=0;i<n;i++)if(!std::isfinite(p[i]))return false;return true;}
struct Eval {
 std::vector<double> pose,v,J,M,bias,abias;av1_evaluation d{};
 Eval(uint32_t B,uint32_t N):pose(B*7),v(B*6),J(B*6*N),M(N*N),bias(N),abias(B*6){d={sizeof(d),AV1_ABI,pose.data(),v.data(),J.data(),M.data(),bias.data(),abias.data(),0,0};}
};
bool inverse(uint32_t N,const std::vector<double>& M,std::vector<double>& inv){
 double scale=0;for(uint32_t i=0;i<N;i++)scale=std::max(scale,M[i*N+i]);if(!std::isfinite(scale)||scale<=0)return false;
 std::vector<double>L(N*N);inv.assign(N*N,0);
 for(uint32_t i=0;i<N;i++)for(uint32_t j=0;j<=i;j++){
  if(!std::isfinite(M[i*N+j])||std::abs(M[i*N+j]-M[j*N+i])>2e-12*scale)return false;
  double x=M[i*N+j];for(uint32_t k=0;k<j;k++)x-=L[i*N+k]*L[j*N+k];
  if(i==j){if(!std::isfinite(x)||x<=1e-12*scale)return false;L[i*N+j]=std::sqrt(x);}else L[i*N+j]=x/L[j*N+j];
 }
 for(uint32_t c=0;c<N;c++){
  std::vector<double>y(N),x(N);for(uint32_t i=0;i<N;i++){double t=i==c?1:0;for(uint32_t k=0;k<i;k++)t-=L[i*N+k]*y[k];y[i]=t/L[i*N+i];}
  for(int i=int(N)-1;i>=0;i--){double t=y[i];for(uint32_t k=uint32_t(i)+1;k<N;k++)t-=L[k*N+uint32_t(i)]*x[k];x[i]=t/L[uint32_t(i)*N+uint32_t(i)];if(!std::isfinite(x[i]))return false;inv[uint32_t(i)*N+c]=x[i];}
 }
 for(uint32_t i=0;i<N;i++)if(inv[i*N+i]<=0)return false;return true;
}
struct Model {
 av1_model d;std::vector<av1_body>body;std::vector<av1_hinge>hinge;std::vector<double>reference;std::vector<av2_limit>limit;
 Model(const av2_registration& r):d(*r.model),body(d.body,d.body+d.bodies),reference(d.reference_qpos,d.reference_qpos+7+d.joints){if(d.joints){hinge.assign(d.hinge,d.hinge+d.joints);limit.assign(r.limits,r.limits+d.joints);}d.body=body.data();d.hinge=hinge.data();d.reference_qpos=reference.data();}
};
struct State {std::vector<double>q,v,warm,time,pose,bodyv;std::vector<uint64_t>count;};
uint64_t hash(uint64_t h,const void* ptr,size_t n){const auto* p=static_cast<const unsigned char*>(ptr);for(size_t k=0;k<n;k++){h^=p[k];h*=UINT64_C(1099511628211);}return h;}
template<class T>uint64_t hv(uint64_t h,const std::vector<T>& x){return hash(h,x.data(),x.size()*sizeof(T));}
bool limit_valid(const av2_limit& x){
 const double values[]={x.lower,x.upper,x.margin,x.timeconst,x.dampratio,x.solimp[0],x.solimp[1],x.solimp[2],x.solimp[3],x.solimp[4]};
 return x.enabled<=1&&!x.reserved&&finite(values,10)&&x.lower<x.upper&&x.margin>=0&&x.timeconst>0&&x.dampratio>0&&x.solimp[0]>0&&x.solimp[0]<1&&x.solimp[1]>=x.solimp[0]&&x.solimp[1]<1&&x.solimp[2]>=0&&x.solimp[3]>0&&x.solimp[3]<1&&x.solimp[4]>=1;
}
double bounded_impedance(double x){return std::clamp(x,1e-4,.9999);}
double impedance(const av2_limit& l,double gap){
 double d0=bounded_impedance(l.solimp[0]),dw=bounded_impedance(l.solimp[1]);
 double width=l.solimp[2],mid=bounded_impedance(l.solimp[3]),power=l.solimp[4];
 if(d0==dw||width<=1e-15)return .5*(d0+dw);
 double x=std::abs((gap-l.margin)/width);if(x>=1)return dw;if(x<=0)return d0;
 double y=power==1?x:(x<=mid?std::pow(x,power)/std::pow(mid,power-1):1-std::pow(1-x,power)/std::pow(1-mid,power-1));return d0+(dw-d0)*y;
}
}
struct av2_scene {
 Model model;uint32_t E,B,J,N,R,Q;uint64_t generation=0,binding=UINT64_C(14695981039346656037);
 std::vector<double>gravity,reference_weight;State state,initial;
 explicit av2_scene(const av2_registration& r):model(r),E(r.environments),B(model.d.bodies),J(model.d.joints),N(6+J),R(3*J),Q(7+J),gravity(r.gravity,r.gravity+E*3),reference_weight(J){}
};
struct av2_pre {
 const av2_scene* owner;uint64_t generation;double dt,mtol,jtol;State state;
 std::vector<double>M,inv,bias,actuator,passive,smooth,Jbody,G,target,R,lo,hi,warm,gap,aref;std::vector<uint32_t>kind;std::vector<uint8_t>active;
 av2_pre(const av2_scene& s,const av2_step& step):owner(&s),generation(s.generation),dt(step.dt),mtol(step.momentum_tolerance),jtol(step.joint_impulse_tolerance),state(s.state),M(s.E*s.N*s.N),inv(M.size()),bias(s.E*s.N),actuator(bias.size()),passive(bias.size()),smooth(bias.size()),Jbody(s.E*s.B*6*s.N),G(s.E*s.R*s.N),target(s.E*s.R),R(target.size(),1),lo(target.size()),hi(target.size()),warm(target.size()),gap(target.size()),aref(target.size()),kind(s.R),active(target.size()){}
};
struct av2_stage {const av2_scene* owner;uint64_t generation;bool consumed=false;State state;explicit av2_stage(const av2_scene& s):owner(&s),generation(s.generation){};};
namespace {
int evaluate(const av2_scene& s,const State& st,uint32_t e,Eval& eval){return av1_evaluate(&s.model.d,st.q.data()+e*s.Q,st.v.data()+e*s.N,s.gravity.data()+e*3,&eval.d);}
void armature(const av2_scene& s,std::vector<double>& M){for(uint32_t j=0;j<s.J;j++)M[(j+6)*s.N+j+6]+=s.model.hinge[j].armature;}
int validate_state(const av2_scene& s,State& st){
 if(!finite(st.q.data(),st.q.size())||!finite(st.v.data(),st.v.size())||!finite(st.warm.data(),st.warm.size())||!finite(st.time.data(),st.time.size()))return AV2_INVALID;
 for(uint32_t e=0;e<s.E;e++){
  if(st.time[e]<0)return AV2_INVALID;
  for(uint32_t j=0;j<s.J;j++)if(std::abs(st.warm[e*s.R+j])>s.model.hinge[j].friction_loss||st.warm[e*s.R+s.J+j]<0||st.warm[e*s.R+2*s.J+j]<0)return AV2_INVALID;
  Eval eval(s.B,s.N);int rc=evaluate(s,st,e,eval);if(rc)return rc;armature(s,eval.M);std::vector<double>inv;if(!inverse(s.N,eval.M,inv))return AV2_DYNAMICS;
  std::copy(eval.pose.begin(),eval.pose.end(),st.pose.begin()+e*s.B*7);std::copy(eval.v.begin(),eval.v.end(),st.bodyv.begin()+e*s.B*6);
 }
 return AV2_OK;
}
State allocate(const av2_scene& s){State st;st.q.resize(s.E*s.Q);st.v.resize(s.E*s.N);st.warm.resize(s.E*s.R);st.time.resize(s.E);st.count.resize(s.E);st.pose.resize(s.E*s.B*7);st.bodyv.resize(s.E*s.B*6);return st;}
bool view_valid(const av2_state_view* out){return out&&out->struct_size==sizeof(*out)&&out->version==AV2_ABI;}
void state_view(const av2_scene& s,const State& st,av2_state_view* out){*out={sizeof(*out),AV2_ABI,s.E,s.B,s.J,0,st.q.data(),st.v.data(),st.warm.data(),st.time.data(),st.pose.data(),st.bodyv.data(),st.count.data()};}
bool snapshot_valid(const av2_scene& s,const av2_snapshot* o){return o&&o->struct_size==sizeof(*o)&&o->version==AV2_ABI&&o->environments==s.E&&o->joints==s.J&&o->qpos&&o->velocity&&(s.R==0||o->joint_warm_force)&&o->time&&o->step_count;}
int capture(const av2_scene& s,const State& st,av2_snapshot* o){if(!snapshot_valid(s,o))return AV2_INVALID;std::copy(st.q.begin(),st.q.end(),o->qpos);std::copy(st.v.begin(),st.v.end(),o->velocity);if(s.R)std::copy(st.warm.begin(),st.warm.end(),o->joint_warm_force);std::copy(st.time.begin(),st.time.end(),o->time);std::copy(st.count.begin(),st.count.end(),o->step_count);o->binding=s.binding;return AV2_OK;}
}
extern "C" int av2_create(const av2_registration* r,av2_scene** out){try{
 if(!r||!out||r->struct_size!=sizeof(*r)||r->version!=AV2_ABI||r->reserved||r->environments<1||r->environments>4096||!r->model)return AV2_INVALID;
 const auto& m=*r->model;if(m.joints>26||m.bodies!=m.joints+2||(m.joints&&!r->limits))return AV2_INVALID;
 uint32_t N=m.joints+6;std::vector<double>zero(N);double gzero[3]={0,0,0};Eval ref(m.bodies,N);int rc=av1_evaluate(&m,m.reference_qpos,zero.data(),gzero,&ref.d);if(rc)return rc;
 for(uint32_t j=0;j<m.joints;j++)if(!limit_valid(r->limits[j]))return AV2_INVALID;
 if(!finite(r->initial_qpos,r->environments*(N+1))||!finite(r->initial_velocity,r->environments*N)||!finite(r->gravity,r->environments*3))return AV2_INVALID;
 auto s=std::make_unique<av2_scene>(*r);s->state=allocate(*s);std::copy_n(r->initial_qpos,s->state.q.size(),s->state.q.data());std::copy_n(r->initial_velocity,s->state.v.size(),s->state.v.data());
 armature(*s,ref.M);std::vector<double>inv;if(!inverse(N,ref.M,inv))return AV2_DYNAMICS;for(uint32_t j=0;j<s->J;j++)s->reference_weight[j]=inv[(j+6)*N+j+6];
 rc=validate_state(*s,s->state);if(rc)return rc;s->initial=s->state;
 s->binding=hash(s->binding,&s->E,sizeof(s->E));s->binding=hash(s->binding,&s->J,sizeof(s->J));s->binding=hv(s->binding,s->model.body);
 for(const auto& h:s->model.hinge){s->binding=hash(s->binding,&h.parent,sizeof(h.parent));s->binding=hash(s->binding,&h.motor_enabled,sizeof(h.motor_enabled));s->binding=hash(s->binding,h.parent_anchor,sizeof(h.parent_anchor));s->binding=hash(s->binding,h.child_anchor,sizeof(h.child_anchor));s->binding=hash(s->binding,h.axis_parent,sizeof(h.axis_parent));s->binding=hash(s->binding,h.reference_xyzw,sizeof(h.reference_xyzw));const float c[]={h.armature,h.passive_damping,h.friction_loss,h.stiffness,h.motor_damping,h.maximum_effort,h.friction_d0,h.friction_dwidth,h.friction_timeconst};s->binding=hash(s->binding,c,sizeof(c));}
 for(const auto& l:s->model.limit){s->binding=hash(s->binding,&l.enabled,sizeof(l.enabled));const double x[]={l.lower,l.upper,l.margin,l.timeconst,l.dampratio,l.solimp[0],l.solimp[1],l.solimp[2],l.solimp[3],l.solimp[4]};s->binding=hash(s->binding,x,sizeof(x));}
 s->binding=hash(s->binding,s->model.d.root_source_to_principal,sizeof(s->model.d.root_source_to_principal));s->binding=hv(s->binding,s->model.reference);s->binding=hv(s->binding,s->gravity);s->binding=hv(s->binding,s->state.q);s->binding=hv(s->binding,s->state.v);*out=s.release();return AV2_OK;
}catch(const std::bad_alloc&){return AV2_ALLOCATION;}}
extern "C" void av2_destroy(av2_scene* s){delete s;}
extern "C" int av2_read(const av2_scene* s,av2_state_view* out){if(!s||!view_valid(out))return AV2_INVALID;state_view(*s,s->state,out);return AV2_OK;}
extern "C" int av2_capture(const av2_scene* s,av2_snapshot* out){return s?capture(*s,s->state,out):AV2_INVALID;}
extern "C" int av2_prepare(const av2_scene* s,const av2_step* in,av2_pre** out){try{
 if(!s||!in||!out||in->struct_size!=sizeof(*in)||in->version!=AV2_ABI||!std::isfinite(in->dt)||in->dt<=0||!std::isfinite(in->momentum_tolerance)||in->momentum_tolerance<=0||in->momentum_tolerance>1e-5||!std::isfinite(in->joint_impulse_tolerance)||in->joint_impulse_tolerance<=0||in->joint_impulse_tolerance>1e-5)return AV2_INVALID;
 if(s->generation==UINT64_MAX)return AV2_STALE;
 if(!finite(in->target_position,s->E*s->J)||!finite(in->target_velocity,s->E*s->J)||(in->applied_force&&!finite(in->applied_force,s->E*s->N)))return AV2_INVALID;
 auto pre=std::make_unique<av2_pre>(*s,*in);
 for(uint32_t e=0;e<s->E;e++){
  if(s->state.count[e]==UINT64_MAX)return AV2_STALE;
  Eval ev(s->B,s->N);int rc=evaluate(*s,s->state,e,ev);if(rc)return rc;std::copy(ev.bias.begin(),ev.bias.end(),pre->bias.begin()+e*s->N);std::copy(ev.J.begin(),ev.J.end(),pre->Jbody.begin()+e*s->B*6*s->N);armature(*s,ev.M);std::vector<double>inv;if(!inverse(s->N,ev.M,inv))return AV2_DYNAMICS;
  std::copy(ev.M.begin(),ev.M.end(),pre->M.begin()+e*s->N*s->N);std::copy(inv.begin(),inv.end(),pre->inv.begin()+e*s->N*s->N);
  for(uint32_t j=0;j<s->J;j++){
   uint32_t n=6+j,idx=e*s->N+n;const auto& h=s->model.hinge[j];const auto& l=s->model.limit[j];double target=in->target_position[e*s->J+j];if(l.enabled)target=std::clamp(target,l.lower,l.upper);
   double motor=double(h.stiffness)*(target-s->state.q[e*s->Q+7+j])+double(h.motor_damping)*(in->target_velocity[e*s->J+j]-s->state.v[idx]);if(!std::isfinite(motor))return AV2_DYNAMICS;
   pre->actuator[idx]=h.motor_enabled?std::clamp(motor,-double(h.maximum_effort),double(h.maximum_effort)):0;
   pre->passive[idx]=-double(h.passive_damping)*s->state.v[idx];
   for(uint32_t side=0;side<3;side++){
    uint32_t row=side*s->J+j,k=e*s->R+row;pre->kind[row]=side+1;double sign=side==2?-1:1;
    double gap=side==0?0:(side==1?s->state.q[e*s->Q+7+j]-l.lower:l.upper-s->state.q[e*s->Q+7+j]);pre->gap[k]=gap;
    bool active=side==0?h.friction_loss>0:(l.enabled&&gap<l.margin);if(!active)continue;
    double d=side==0?bounded_impedance(double(h.friction_d0)):impedance(l,gap);double width=bounded_impedance(side==0?double(h.friction_dwidth):l.solimp[1]);double tc=std::max(side==0?double(h.friction_timeconst):l.timeconst,2*in->dt);double B=2/std::max(1e-15,width*tc);double K=side==0?0:1/std::max(1e-15,width*width*tc*tc*l.dampratio*l.dampratio);double rowv=sign*s->state.v[idx];double aref=-B*rowv-(side==0?0:K*d*(gap-l.margin));
    pre->active[k]=1;pre->G[k*s->N+n]=sign;pre->R[k]=std::max(1e-15,(1-d)/d*s->reference_weight[j]);pre->aref[k]=aref;pre->target[k]=rowv+in->dt*aref;
    pre->lo[k]=side==0?-in->dt*double(h.friction_loss):0;pre->hi[k]=side==0?in->dt*double(h.friction_loss):std::numeric_limits<double>::infinity();pre->warm[k]=std::clamp(in->dt*s->state.warm[k],pre->lo[k],pre->hi[k]);
    if(!std::isfinite(pre->R[k])||pre->R[k]<=0||!std::isfinite(aref)||!std::isfinite(pre->target[k])||!std::isfinite(pre->warm[k]))return AV2_DYNAMICS;
   }
  }
  for(uint32_t n=0;n<s->N;n++){
   double dv=0;for(uint32_t k=0;k<s->N;k++){uint32_t idx=e*s->N+k;double f=pre->actuator[idx]+pre->passive[idx]-pre->bias[idx]+(in->applied_force?in->applied_force[idx]:0);if(!std::isfinite(f))return AV2_DYNAMICS;dv+=inv[n*s->N+k]*f;}
   pre->smooth[e*s->N+n]=s->state.v[e*s->N+n]+in->dt*dv;if(!std::isfinite(pre->smooth[e*s->N+n]))return AV2_DYNAMICS;
  }
 }
 *out=pre.release();return AV2_OK;
}catch(const std::bad_alloc&){return AV2_ALLOCATION;}}
extern "C" int av2_pre_read(const av2_pre* p,av2_pre_view* o){if(!p||!o||o->struct_size!=sizeof(*o)||o->version!=AV2_ABI)return AV2_INVALID;const auto& s=*p->owner;*o={sizeof(*o),AV2_ABI,s.E,s.B,s.N,s.J,s.R,0,p->generation,p->dt,p->state.q.data(),p->state.v.data(),p->M.data(),p->inv.data(),p->bias.data(),p->actuator.data(),p->passive.data(),p->smooth.data(),p->state.pose.data(),p->state.bodyv.data(),p->Jbody.data(),p->G.data(),p->target.data(),p->R.data(),p->lo.data(),p->hi.data(),p->warm.data(),p->gap.data(),p->aref.data(),p->kind.data(),p->active.data()};return AV2_OK;}
extern "C" void av2_pre_destroy(av2_pre* p){delete p;}
extern "C" int av2_complete(const av2_pre* p,const av2_solution* in,av2_stage** out){try{
 if(!p||!in||!out||in->struct_size!=sizeof(*in)||in->version!=AV2_ABI)return AV2_INVALID;const auto& s=*p->owner;if(p->generation!=s.generation||s.generation==UINT64_MAX)return AV2_STALE;
 if(!finite(in->velocity,s.E*s.N)||!finite(in->joint_impulse,s.E*s.R)||(in->contact_generalized_impulse&&!finite(in->contact_generalized_impulse,s.E*s.N)))return AV2_INVALID;
 for(uint32_t e=0;e<s.E;e++){
  std::vector<double>impulse(s.N);for(uint32_t n=0;n<s.N;n++)impulse[n]=in->contact_generalized_impulse?in->contact_generalized_impulse[e*s.N+n]:0;
  for(uint32_t row=0;row<s.R;row++){
   uint32_t k=e*s.R+row;double lambda=in->joint_impulse[k];if(lambda<p->lo[k]||lambda>p->hi[k]||(!p->active[k]&&lambda!=0))return AV2_INVALID;
   double residual=p->R[k]*lambda-p->target[k],diag=p->R[k];for(uint32_t n=0;n<s.N;n++){double g=p->G[k*s.N+n];residual+=g*in->velocity[e*s.N+n];impulse[n]+=g*lambda;for(uint32_t m=0;m<s.N;m++)diag+=g*p->inv[(e*s.N+n)*s.N+m]*p->G[k*s.N+m];}
   if(!std::isfinite(residual)||!std::isfinite(diag)||diag<=0)return AV2_DYNAMICS;
   double correction=residual>=0?std::min(residual/diag,lambda-p->lo[k]):std::min(-residual/diag,p->hi[k]-lambda);if(!std::isfinite(correction)||correction>p->jtol)return AV2_NO_CONVERGENCE;
  }
  for(uint32_t n=0;n<s.N;n++){
   double predicted=p->smooth[e*s.N+n];for(uint32_t k=0;k<s.N;k++)predicted+=p->inv[(e*s.N+n)*s.N+k]*impulse[k];if(!std::isfinite(predicted))return AV2_DYNAMICS;if(std::abs(predicted-in->velocity[e*s.N+n])>p->mtol)return AV2_NO_CONVERGENCE;
  }
 }
 auto stage=std::make_unique<av2_stage>(s);stage->state=p->state;std::copy_n(in->velocity,s.E*s.N,stage->state.v.data());
 for(uint32_t e=0;e<s.E;e++){
  int rc=av1_integrate_root(p->state.q.data()+e*s.Q,in->velocity+e*s.N,p->dt,stage->state.q.data()+e*s.Q);if(rc)return rc;
  for(uint32_t j=0;j<s.J;j++)stage->state.q[e*s.Q+7+j]=p->state.q[e*s.Q+7+j]+p->dt*in->velocity[e*s.N+6+j];
  for(uint32_t r=0;r<s.R;r++){
   double warm=in->joint_impulse[e*s.R+r]/p->dt;
   // Division may round one ULP outside the already validated impulse cap.
   if(r<s.J)warm=std::clamp(warm,-double(s.model.hinge[r].friction_loss),double(s.model.hinge[r].friction_loss));
   stage->state.warm[e*s.R+r]=warm;
  }
  stage->state.time[e]+=p->dt;++stage->state.count[e];
 }
 int rc=validate_state(s,stage->state);if(rc)return rc;*out=stage.release();return AV2_OK;
}catch(const std::bad_alloc&){return AV2_ALLOCATION;}}
extern "C" int av2_stage_read(const av2_stage* p,av2_state_view* o){if(!p||p->consumed||!view_valid(o))return AV2_INVALID;state_view(*p->owner,p->state,o);return AV2_OK;}
extern "C" int av2_stage_capture(const av2_stage* p,av2_snapshot* o){return p&&!p->consumed?capture(*p->owner,p->state,o):AV2_INVALID;}
extern "C" int av2_prepare_restore(const av2_scene* s,const av2_snapshot* in,av2_stage** out){try{
 if(!s||!out||!snapshot_valid(*s,in)||in->binding!=s->binding)return AV2_INVALID;if(s->generation==UINT64_MAX)return AV2_STALE;auto stage=std::make_unique<av2_stage>(*s);stage->state=allocate(*s);
 std::copy_n(in->qpos,s->E*s->Q,stage->state.q.data());std::copy_n(in->velocity,s->E*s->N,stage->state.v.data());if(s->R)std::copy_n(in->joint_warm_force,s->E*s->R,stage->state.warm.data());std::copy_n(in->time,s->E,stage->state.time.data());std::copy_n(in->step_count,s->E,stage->state.count.data());int rc=validate_state(*s,stage->state);if(rc)return rc;*out=stage.release();return AV2_OK;
}catch(const std::bad_alloc&){return AV2_ALLOCATION;}}
extern "C" int av2_prepare_reset(const av2_scene* s,const uint8_t* mask,uint32_t count,av2_stage** out){try{
 if(!s||!mask||!out||count!=s->E)return AV2_INVALID;if(s->generation==UINT64_MAX)return AV2_STALE;for(uint32_t e=0;e<s->E;e++)if(mask[e]>1)return AV2_INVALID;auto stage=std::make_unique<av2_stage>(*s);stage->state=s->state;
 for(uint32_t e=0;e<s->E;e++)if(mask[e]){std::copy_n(s->initial.q.data()+e*s->Q,s->Q,stage->state.q.data()+e*s->Q);std::copy_n(s->initial.v.data()+e*s->N,s->N,stage->state.v.data()+e*s->N);if(s->R)std::fill_n(stage->state.warm.data()+e*s->R,s->R,0);stage->state.time[e]=0;stage->state.count[e]=0;}
 int rc=validate_state(*s,stage->state);if(rc)return rc;*out=stage.release();return AV2_OK;
}catch(const std::bad_alloc&){return AV2_ALLOCATION;}}
extern "C" int av2_validate_commit(const av2_scene* s,const av2_stage* p){if(!s||!p)return AV2_INVALID;return p->owner!=s||p->consumed||p->generation!=s->generation||s->generation==UINT64_MAX?AV2_STALE:AV2_OK;}
extern "C" int av2_commit(av2_scene* s,av2_stage* p){int rc=av2_validate_commit(s,p);if(rc)return rc;std::swap(s->state,p->state);++s->generation;p->consumed=true;return AV2_OK;}
extern "C" void av2_stage_destroy(av2_stage* p){delete p;}
