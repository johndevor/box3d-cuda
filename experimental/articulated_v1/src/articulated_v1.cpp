// SPDX-License-Identifier: MIT
#include "articulated_v1.h"
#include "box3d_cuda/experimental_joint_v1.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>
namespace {
using V=std::array<double,3>; using Q=std::array<double,4>;
V vec(const double* a){return {a[0],a[1],a[2]};}
V add(V a,V b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
V sub(V a,V b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
V scale(V a,double t){return {a[0]*t,a[1]*t,a[2]*t};}
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
Q quat(const double* a){return {a[0],a[1],a[2],a[3]};}
Q mul(Q a,Q b){V v=add(add(scale(vec(b.data()),a[3]),scale(vec(a.data()),b[3])),cross(vec(a.data()),vec(b.data())));return {v[0],v[1],v[2],a[3]*b[3]-dot(vec(a.data()),vec(b.data()))};}
Q normalize(Q q){double n=std::hypot(std::hypot(q[0],q[1]),std::hypot(q[2],q[3]));for(double& a:q)a/=n;return q;}
V rotate(Q q,V v){V t=scale(cross(vec(q.data()),v),2);return add(add(v,scale(t,q[3])),cross(vec(q.data()),t));}
Q exponential(V r){double angle=std::hypot(r[0],r[1],r[2]);double s=angle<1e-8 ? .5-angle*angle/48 : std::sin(angle*.5)/angle;return {r[0]*s,r[1]*s,r[2]*s,std::cos(angle*.5)};}
bool finite(const double* p,size_t n){if(!p)return false;for(size_t i=0;i<n;i++)if(!std::isfinite(p[i]))return false;return true;}
bool finite(const float* p,size_t n){if(!p)return false;for(size_t i=0;i<n;i++)if(!std::isfinite(p[i]))return false;return true;}
bool unit(const double* p,size_t n){if(!finite(p,n))return false;double s=0;for(size_t i=0;i<n;i++)s+=p[i]*p[i];return std::abs(s-1)<1e-10;}
void put(double* p,V v){std::copy(v.begin(),v.end(),p);}
V inertia(Q q,const av1_body& b,V w){Q inv={-q[0],-q[1],-q[2],q[3]};V local=rotate(inv,w);for(int k=0;k<3;k++)local[k]*=b.principal_inertia[k];return rotate(q,local);}
bool model_valid(const av1_model* m){
 if(!m||m->struct_size!=sizeof(*m)||m->version!=AV1_ABI||m->flags||m->reserved||m->joints>26||m->bodies!=m->joints+2||!m->body||(m->joints&&!m->hinge))return false;
 if(!finite(m->root_source_to_principal,7)||!unit(m->root_source_to_principal+3,4)||!finite(m->reference_qpos,7+m->joints)||!unit(m->reference_qpos+3,4))return false;
 for(uint32_t b=0;b<m->bodies;b++){
  const auto& x=m->body[b];if(!std::isfinite(x.mass)||!finite(x.principal_inertia,3))return false;
  if(b==0){if(x.mass!=0||x.principal_inertia[0]!=0||x.principal_inertia[1]!=0||x.principal_inertia[2]!=0)return false;}
  else if(x.mass<=0||x.principal_inertia[0]<=0||x.principal_inertia[1]<=0||x.principal_inertia[2]<=0)return false;
 }
 for(uint32_t j=0;j<m->joints;j++){
  const auto& h=m->hinge[j];if(h.parent<1||h.parent>=j+2||h.motor_enabled>1||!finite(h.parent_anchor,3)||!finite(h.child_anchor,3)||!unit(h.axis_parent,3)||!unit(h.reference_xyzw,4))return false;
  const float p[]={h.armature,h.passive_damping,h.friction_loss,h.stiffness,h.motor_damping,h.maximum_effort,h.friction_d0,h.friction_dwidth,h.friction_timeconst};
  if(!finite(p,9))return false;
  for(int k=0;k<6;k++)if(p[k]<0)return false;
  if(h.friction_d0<=0||h.friction_d0>=1||h.friction_dwidth<h.friction_d0||h.friction_dwidth>=1||h.friction_timeconst<=0)return false;
 }
 return true;
}
struct Eval {
 uint32_t B,N;std::vector<double> pose,vel,J,M,bias,abias;double kinetic=0,potential=0;
 Eval(uint32_t b,uint32_t n):B(b),N(n),pose(b*7),vel(b*6),J(b*6*n),M(n*n),bias(n),abias(b*6){}
};
bool output_valid(const av1_evaluation* o){return o&&o->struct_size==sizeof(*o)&&o->version==AV1_ABI&&o->body_pose&&o->body_velocity&&o->jacobian&&o->body_mass&&o->bias&&o->body_bias_acceleration;}
void emit(const Eval& e,av1_evaluation* o){std::copy(e.pose.begin(),e.pose.end(),o->body_pose);std::copy(e.vel.begin(),e.vel.end(),o->body_velocity);std::copy(e.J.begin(),e.J.end(),o->jacobian);std::copy(e.M.begin(),e.M.end(),o->body_mass);std::copy(e.bias.begin(),e.bias.end(),o->bias);std::copy(e.abias.begin(),e.abias.end(),o->body_bias_acceleration);o->kinetic_energy=e.kinetic;o->potential_energy=e.potential;}
int evaluate(const av1_model& m,const double* q,const double* v,const double* gravity,Eval& e){
 const uint32_t N=e.N;
 if(!finite(q,7+m.joints)||!unit(q+3,4)||!finite(v,N)||!finite(gravity,3))return AV1_INVALID;
 std::vector<V> pos(e.B),vel(e.B),w(e.B),acc(e.B),alpha(e.B);std::vector<Q> rot(e.B,Q{0,0,0,1});
 Q root=quat(q+3);V offset=rotate(root,vec(m.root_source_to_principal));pos[1]=add(vec(q),offset);rot[1]=normalize(mul(root,quat(m.root_source_to_principal+3)));w[1]=vec(v+3);vel[1]=add(vec(v),cross(w[1],offset));acc[1]=cross(w[1],cross(w[1],offset));
 auto jv=[&](uint32_t b,uint32_t n){return V{e.J[(b*6)*N+n],e.J[(b*6+1)*N+n],e.J[(b*6+2)*N+n]};};
 auto jw=[&](uint32_t b,uint32_t n){return V{e.J[(b*6+3)*N+n],e.J[(b*6+4)*N+n],e.J[(b*6+5)*N+n]};};
 auto setj=[&](uint32_t b,uint32_t n,V a,V z){for(int k=0;k<3;k++){e.J[(b*6+k)*N+n]=a[k];e.J[(b*6+k+3)*N+n]=z[k];}};
 for(uint32_t k=0;k<3;k++){V basis{};basis[k]=1;setj(1,k,basis,V{});setj(1,k+3,cross(basis,offset),basis);}
 for(uint32_t j=0;j<m.joints;j++){
  uint32_t b=j+2;const auto& h=m.hinge[j];uint32_t p=h.parent;V rp=rotate(rot[p],vec(h.parent_anchor)),s=rotate(rot[p],vec(h.axis_parent));
  rot[b]=normalize(mul(mul(rot[p],exponential(scale(vec(h.axis_parent),q[7+j]))),quat(h.reference_xyzw)));
  V rc=rotate(rot[b],vec(h.child_anchor));pos[b]=sub(add(pos[p],rp),rc);
  w[b]=add(w[p],scale(s,v[6+j]));alpha[b]=add(alpha[p],scale(cross(w[p],s),v[6+j]));
  vel[b]=sub(add(vel[p],cross(w[p],rp)),cross(w[b],rc));
  V aa=add(add(acc[p],cross(alpha[p],rp)),cross(w[p],cross(w[p],rp)));
  acc[b]=sub(sub(aa,cross(alpha[b],rc)),cross(w[b],cross(w[b],rc)));
  for(uint32_t n=0;n<N;n++){V angular=jw(p,n);if(n==6+j)angular=add(angular,s);setj(b,n,sub(add(jv(p,n),cross(jw(p,n),rp)),cross(angular,rc)),angular);}
 }
 for(uint32_t b=0;b<e.B;b++){
  put(e.pose.data()+b*7,pos[b]);std::copy(rot[b].begin(),rot[b].end(),e.pose.begin()+b*7+3);put(e.vel.data()+b*6,vel[b]);put(e.vel.data()+b*6+3,w[b]);put(e.abias.data()+b*6,acc[b]);put(e.abias.data()+b*6+3,alpha[b]);
  if(!b)continue;
  const auto& body=m.body[b];V iw=inertia(rot[b],body,w[b]);V force=scale(sub(acc[b],vec(gravity)),body.mass);V torque=add(inertia(rot[b],body,alpha[b]),cross(w[b],iw));
  e.kinetic+=.5*(body.mass*dot(vel[b],vel[b])+dot(w[b],iw));e.potential-=body.mass*dot(vec(gravity),pos[b]);
  for(uint32_t n=0;n<N;n++){
   e.bias[n]+=dot(jv(b,n),force)+dot(jw(b,n),torque);
   for(uint32_t k=0;k<=n;k++){double x=body.mass*dot(jv(b,n),jv(b,k))+dot(jw(b,n),inertia(rot[b],body,jw(b,k)));e.M[n*N+k]+=x;if(k!=n)e.M[k*N+n]+=x;}
  }
 }
 for(const auto* a:{&e.pose,&e.vel,&e.J,&e.M,&e.bias,&e.abias})if(!finite(a->data(),a->size()))return AV1_DYNAMICS;
 return std::isfinite(e.kinetic)&&std::isfinite(e.potential)?AV1_OK:AV1_DYNAMICS;
}
struct Model {
 av1_model d{};std::vector<av1_body> bodies;std::vector<av1_hinge> hinges;std::vector<double> reference;
 explicit Model(const av1_model& m):d(m),bodies(m.body,m.body+m.bodies),reference(m.reference_qpos,m.reference_qpos+7+m.joints){if(m.joints)hinges.assign(m.hinge,m.hinge+m.joints);d.body=bodies.data();d.hinge=hinges.data();d.reference_qpos=reference.data();}
};
using Joint=std::unique_ptr<box3d_joint_v1_scene,decltype(&box3d_joint_v1_destroy)>;
struct State {
 Joint joint{nullptr,box3d_joint_v1_destroy};std::vector<double> root;
};
struct JointSnapshot {
 std::vector<float> q,v,warm;std::vector<uint64_t> count;box3d_joint_v1_snapshot d{};
 JointSnapshot(uint32_t E,uint32_t N):q(E*N),v(E*N),warm(E*N),count(E){d={sizeof(d),BOX3D_JOINT_V1_ABI,N,E,0,q.data(),v.data(),warm.data(),count.data()};}
};
uint64_t hash_bytes(uint64_t h,const void* p,size_t n){const auto* a=static_cast<const unsigned char*>(p);for(size_t i=0;i<n;i++){h^=a[i];h*=UINT64_C(1099511628211);}return h;}
template<class T>uint64_t hash_vec(uint64_t h,const std::vector<T>& a){return hash_bytes(h,a.data(),a.size()*sizeof(T));}
}
struct av1_scene {
 Model model;uint32_t E,N,Qn;uint64_t binding=UINT64_C(14695981039346656037),generation=0;
 std::vector<double> gravity,initial_q;
 std::vector<float> initial_v,reference_mass,coeff[9];std::vector<uint8_t> revolute,motor;
 State state;
 av1_scene(const av1_registration& r):model(*r.model),E(r.environments),N(6+model.d.joints),Qn(7+model.d.joints),gravity(r.gravity,r.gravity+E*3),initial_q(r.initial_qpos,r.initial_qpos+E*Qn),initial_v(r.initial_velocity,r.initial_velocity+E*N),reference_mass(E*N*N),revolute(N),motor(N){for(auto& c:coeff)c.resize(N);}
};
struct av1_stage {
 const av1_scene* owner;uint64_t generation;bool consumed=false;State state;std::vector<float> acceleration,constraint;
 explicit av1_stage(const av1_scene& s):owner(&s),generation(s.generation){}
};
namespace {
int create_joint(const av1_scene& s,Joint& joint){
 std::vector<float> q(s.E*s.N,0);for(uint32_t e=0;e<s.E;e++)for(uint32_t j=0;j<s.model.d.joints;j++)q[e*s.N+6+j]=float(s.initial_q[e*s.Qn+7+j]);
 box3d_joint_v1_params p{};p.struct_size=sizeof(p);p.version=BOX3D_JOINT_V1_ABI;p.dofs=s.N;p.environments=s.E;p.revolute=s.revolute.data();p.motor_enabled=s.motor.data();
 p.armature=s.coeff[0].data();p.passive_damping=s.coeff[1].data();p.friction_loss=s.coeff[2].data();p.stiffness=s.coeff[3].data();p.motor_damping=s.coeff[4].data();p.maximum_effort=s.coeff[5].data();p.friction_d0=s.coeff[6].data();p.friction_dwidth=s.coeff[7].data();p.friction_timeconst=s.coeff[8].data();p.reference_body_mass=s.reference_mass.data();p.initial_q=q.data();p.initial_velocity=s.initial_v.data();
 box3d_joint_v1_scene* out=nullptr;int rc=box3d_joint_v1_create(&p,&out);if(!rc)joint.reset(out);return rc;
}
int clone_state(const av1_scene& s,const State& from,State& to){
 JointSnapshot snap(s.E,s.N);int rc=box3d_joint_v1_capture(from.joint.get(),&snap.d);if(rc)return rc;
 rc=create_joint(s,to.joint);if(rc)return rc;rc=box3d_joint_v1_restore(to.joint.get(),&snap.d);if(rc)return rc;to.root=from.root;return AV1_OK;
}
int state_eval(const av1_scene& s,const State& state,uint32_t env,Eval& eval){
 if(env>=s.E)return AV1_INVALID;
 JointSnapshot snap(s.E,s.N);int rc=box3d_joint_v1_capture(state.joint.get(),&snap.d);if(rc)return rc;
 std::vector<double> q(s.Qn),v(s.N);std::copy_n(state.root.data()+env*7,7,q.data());for(uint32_t j=0;j<s.model.d.joints;j++)q[7+j]=snap.q[env*s.N+6+j];for(uint32_t n=0;n<s.N;n++)v[n]=snap.v[env*s.N+n];
 rc=evaluate(s.model.d,q.data(),v.data(),s.gravity.data()+env*3,eval);if(rc)return rc;
 // Every published owned state must remain admissible to the f32 joint bridge.
 for(double x:eval.M)if(!std::isfinite(float(x)))return AV1_DYNAMICS;
 for(double x:eval.bias)if(!std::isfinite(float(x)))return AV1_DYNAMICS;
 std::vector<float> mass(eval.M.begin(),eval.M.end()),zero(s.N),response(s.N);float weight=0;
 return box3d_joint_v1_response(s.N,mass.data(),s.coeff[0].data(),zero.data(),response.data(),&weight);
}
bool snapshot_shape(const av1_scene& s,const av1_snapshot* out){return out&&out->struct_size==sizeof(*out)&&out->version==AV1_ABI&&out->environments==s.E&&out->joints==s.model.d.joints&&out->qpos&&out->velocity&&out->friction_warm_force&&out->step_count;}
int capture(const av1_scene& s,const State& state,av1_snapshot* out){
 if(!snapshot_shape(s,out))return AV1_INVALID;JointSnapshot snap(s.E,s.N);int rc=box3d_joint_v1_capture(state.joint.get(),&snap.d);if(rc)return rc;
 for(uint32_t e=0;e<s.E;e++){std::copy_n(state.root.data()+e*7,7,out->qpos+e*s.Qn);for(uint32_t j=0;j<s.model.d.joints;j++)out->qpos[e*s.Qn+7+j]=snap.q[e*s.N+6+j];}
 std::copy(snap.v.begin(),snap.v.end(),out->velocity);std::copy(snap.warm.begin(),snap.warm.end(),out->friction_warm_force);std::copy(snap.count.begin(),snap.count.end(),out->step_count);out->binding=s.binding;return AV1_OK;
}
int restore_state(const av1_scene& s,const av1_snapshot* input,State& result){
 if(!snapshot_shape(s,input)||input->binding!=s.binding||!finite(input->qpos,s.E*s.Qn)||!finite(input->velocity,s.E*s.N)||!finite(input->friction_warm_force,s.E*s.N))return AV1_INVALID;
 for(uint32_t e=0;e<s.E;e++){
  if(!unit(input->qpos+e*s.Qn+3,4))return AV1_INVALID;
  for(uint32_t n=0;n<s.N;n++)if(std::abs(input->friction_warm_force[e*s.N+n])>s.coeff[2][n])return AV1_INVALID;
  for(uint32_t j=0;j<s.model.d.joints;j++){double q=input->qpos[e*s.Qn+7+j];if(double(float(q))!=q)return AV1_INVALID;}
 }
 int rc=create_joint(s,result.joint);if(rc)return rc;JointSnapshot snap(s.E,s.N);rc=box3d_joint_v1_capture(result.joint.get(),&snap.d);if(rc)return rc;
 result.root.resize(s.E*7);for(uint32_t e=0;e<s.E;e++){std::copy_n(input->qpos+e*s.Qn,7,result.root.data()+e*7);for(uint32_t j=0;j<s.model.d.joints;j++)snap.q[e*s.N+6+j]=float(input->qpos[e*s.Qn+7+j]);}
 std::copy_n(input->velocity,s.E*s.N,snap.v.data());std::copy_n(input->friction_warm_force,s.E*s.N,snap.warm.data());std::copy_n(input->step_count,s.E,snap.count.data());
 rc=box3d_joint_v1_restore(result.joint.get(),&snap.d);if(rc)return rc;
 for(uint32_t e=0;e<s.E;e++){Eval check(s.model.d.bodies,s.N);rc=state_eval(s,result,e,check);if(rc)return rc;}
 return AV1_OK;
}
}
extern "C" int av1_evaluate(const av1_model* m,const double* q,const double* v,const double* g,av1_evaluation* o){try{if(!model_valid(m)||!output_valid(o))return AV1_INVALID;Eval e(m->bodies,6+m->joints);int rc=evaluate(*m,q,v,g,e);if(!rc)emit(e,o);return rc;}catch(const std::bad_alloc&){return AV1_ALLOCATION;}}
extern "C" int av1_integrate_root(const double* root,const double* v,double dt,double* out){
 if(!out||!finite(root,7)||!unit(root+3,4)||!finite(v,6)||!std::isfinite(dt)||dt<=0)return AV1_INVALID;
 double result[7];put(result,add(vec(root),scale(vec(v),dt)));Q q=normalize(mul(exponential(scale(vec(v+3),dt)),quat(root+3)));std::copy(q.begin(),q.end(),result+3);if(!finite(result,7))return AV1_DYNAMICS;std::copy_n(result,7,out);return AV1_OK;
}
extern "C" int av1_create(const av1_registration* r,av1_scene** out){try{
 if(!out||!r||r->struct_size!=sizeof(*r)||r->version!=AV1_ABI||r->reserved||r->environments<1||r->environments>4096||!model_valid(r->model))return AV1_INVALID;
 uint32_t E=r->environments,N=6+r->model->joints,Qn=N+1;if(!finite(r->initial_qpos,E*Qn)||!finite(r->initial_velocity,E*N)||!finite(r->gravity,E*3))return AV1_INVALID;
 for(uint32_t e=0;e<E;e++)if(!unit(r->initial_qpos+e*Qn+3,4))return AV1_INVALID;
 auto s=std::make_unique<av1_scene>(*r);s->state.root.resize(E*7);
 for(uint32_t n=0;n<N;n++){s->coeff[6][n]=.9f;s->coeff[7][n]=.95f;s->coeff[8][n]=.02f;}
 for(uint32_t j=0;j<s->model.d.joints;j++){const auto& h=s->model.hinges[j];uint32_t n=6+j;s->revolute[n]=1;s->motor[n]=uint8_t(h.motor_enabled);const float c[]={h.armature,h.passive_damping,h.friction_loss,h.stiffness,h.motor_damping,h.maximum_effort,h.friction_d0,h.friction_dwidth,h.friction_timeconst};for(int k=0;k<9;k++)s->coeff[k][n]=c[k];}
 std::vector<double> zero(N);double g0[3]={0,0,0};Eval ref(s->model.d.bodies,N);int rc=evaluate(s->model.d,s->model.reference.data(),zero.data(),g0,ref);if(rc)return rc;
 for(uint32_t e=0;e<E;e++){std::copy_n(s->initial_q.data()+e*Qn,7,s->state.root.data()+e*7);for(uint32_t k=0;k<N*N;k++)s->reference_mass[e*N*N+k]=float(ref.M[k]);for(uint32_t j=0;j<s->model.d.joints;j++){double& q=s->initial_q[e*Qn+7+j];q=double(float(q));if(!std::isfinite(q))return AV1_INVALID;}}
 // Hash only named fields; never struct padding or pointer addresses.
 s->binding=hash_bytes(s->binding,&E,sizeof(E));s->binding=hash_bytes(s->binding,&N,sizeof(N));
 for(const auto& b:s->model.bodies){s->binding=hash_bytes(s->binding,&b.mass,sizeof(b.mass));s->binding=hash_bytes(s->binding,b.principal_inertia,sizeof(b.principal_inertia));}
 for(const auto& h:s->model.hinges){s->binding=hash_bytes(s->binding,&h.parent,sizeof(h.parent));s->binding=hash_bytes(s->binding,h.parent_anchor,sizeof(h.parent_anchor));s->binding=hash_bytes(s->binding,h.child_anchor,sizeof(h.child_anchor));s->binding=hash_bytes(s->binding,h.axis_parent,sizeof(h.axis_parent));s->binding=hash_bytes(s->binding,h.reference_xyzw,sizeof(h.reference_xyzw));}
 s->binding=hash_bytes(s->binding,s->model.d.root_source_to_principal,sizeof(s->model.d.root_source_to_principal));s->binding=hash_vec(s->binding,s->model.reference);for(const auto& c:s->coeff)s->binding=hash_vec(s->binding,c);s->binding=hash_vec(s->binding,s->motor);s->binding=hash_vec(s->binding,s->gravity);s->binding=hash_vec(s->binding,s->initial_q);s->binding=hash_vec(s->binding,s->initial_v);
 rc=create_joint(*s,s->state.joint);if(rc)return rc;
 for(uint32_t e=0;e<E;e++){
  Eval initial(s->model.d.bodies,N);rc=state_eval(*s,s->state,e,initial);if(rc)return rc;
 }
 *out=s.release();return AV1_OK;
}catch(const std::bad_alloc&){return AV1_ALLOCATION;}}
extern "C" void av1_destroy(av1_scene* s){delete s;}
extern "C" int av1_capture(const av1_scene* s,av1_snapshot* o){try{return s?capture(*s,s->state,o):AV1_INVALID;}catch(const std::bad_alloc&){return AV1_ALLOCATION;}}
extern "C" int av1_restore(av1_scene* s,const av1_snapshot* input){try{if(!s)return AV1_INVALID;if(s->generation==UINT64_MAX)return AV1_STALE;State next;int rc=restore_state(*s,input,next);if(rc)return rc;std::swap(s->state,next);++s->generation;return AV1_OK;}catch(const std::bad_alloc&){return AV1_ALLOCATION;}}
extern "C" int av1_reset_masked(av1_scene* s,const uint8_t* mask,uint32_t count){try{
 if(!s||!mask||count!=s->E)return AV1_INVALID;if(s->generation==UINT64_MAX)return AV1_STALE;for(uint32_t e=0;e<count;e++)if(mask[e]>1)return AV1_INVALID;
 State next;int rc=clone_state(*s,s->state,next);if(rc)return rc;rc=box3d_joint_v1_reset_masked(next.joint.get(),mask,count);if(rc)return rc;
 for(uint32_t e=0;e<count;e++)if(mask[e])std::copy_n(s->initial_q.data()+e*s->Qn,7,next.root.data()+e*7);
 std::swap(s->state,next);++s->generation;return AV1_OK;
}catch(const std::bad_alloc&){return AV1_ALLOCATION;}}
extern "C" int av1_read(const av1_scene* s,uint32_t env,av1_evaluation* out){try{if(!s||!output_valid(out))return AV1_INVALID;Eval e(s->model.d.bodies,s->N);int rc=state_eval(*s,s->state,env,e);if(!rc)emit(e,out);return rc;}catch(const std::bad_alloc&){return AV1_ALLOCATION;}}
extern "C" int av1_prepare(const av1_scene* s,const av1_step* in,av1_stage** out){try{
 if(!s||!in||!out||in->struct_size!=sizeof(*in)||in->version!=AV1_ABI||in->external_rows>32||!std::isfinite(in->dt)||in->dt<=0)return AV1_INVALID;
 if(s->generation==UINT64_MAX)return AV1_STALE;const uint32_t J=s->model.d.joints,N=s->N,E=s->E,K=in->external_rows;
 if(J&&(!finite(in->target_position,E*J)||!finite(in->target_velocity,E*J)))return AV1_INVALID;
 if(in->external_generalized_force&&!finite(in->external_generalized_force,E*N))return AV1_INVALID;
 auto stage=std::make_unique<av1_stage>(*s);int rc=clone_state(*s,s->state,stage->state);if(rc)return rc;
 std::vector<float> mass(E*N*N),force(E*N),target(E*N),targetv(E*N);stage->acceleration.resize(E*N);stage->constraint.resize(E*K);
 for(uint32_t e=0;e<E;e++){
  Eval eval(s->model.d.bodies,N);rc=state_eval(*s,s->state,e,eval);if(rc)return rc;
  for(uint32_t k=0;k<N*N;k++)mass[e*N*N+k]=float(eval.M[k]);
  for(uint32_t n=0;n<N;n++)force[e*N+n]=float((in->external_generalized_force?in->external_generalized_force[e*N+n]:0)-eval.bias[n]);
  for(uint32_t j=0;j<J;j++){target[e*N+6+j]=in->target_position[e*J+j];targetv[e*N+6+j]=in->target_velocity[e*J+j];}
 }
 box3d_joint_v1_step step{};step.struct_size=sizeof(step);step.version=BOX3D_JOINT_V1_ABI;step.external_rows=K;step.max_iterations=in->max_iterations;step.dt=in->dt;step.tolerance=in->tolerance;step.body_mass=mass.data();step.target_position=target.data();step.target_velocity=targetv.data();step.external_generalized_force=force.data();step.constraint_jacobian=in->constraint_jacobian;step.constraint_reference_acceleration=in->constraint_reference_acceleration;step.constraint_regularizer=in->constraint_regularizer;step.constraint_lower=in->constraint_lower;step.constraint_upper=in->constraint_upper;step.constraint_warm_force=in->constraint_warm_force;
 box3d_joint_v1_output output{};output.struct_size=sizeof(output);output.version=BOX3D_JOINT_V1_ABI;output.acceleration=stage->acceleration.data();output.constraint_force=K?stage->constraint.data():nullptr;
 rc=box3d_joint_v1_advance(stage->state.joint.get(),&step,&output);if(rc)return rc;
 JointSnapshot snap(E,N);rc=box3d_joint_v1_capture(stage->state.joint.get(),&snap.d);if(rc)return rc;
 for(uint32_t e=0;e<E;e++){double velocity[6],root[7];for(int n=0;n<6;n++)velocity[n]=snap.v[e*N+n];rc=av1_integrate_root(s->state.root.data()+e*7,velocity,in->dt,root);if(rc)return rc;std::copy_n(root,7,stage->state.root.data()+e*7);Eval check(s->model.d.bodies,N);rc=state_eval(*s,stage->state,e,check);if(rc)return rc;}
 *out=stage.release();return AV1_OK;
}catch(const std::bad_alloc&){return AV1_ALLOCATION;}}
extern "C" int av1_stage_capture(const av1_stage* s,av1_snapshot* out){try{return s&&!s->consumed?capture(*s->owner,s->state,out):AV1_INVALID;}catch(const std::bad_alloc&){return AV1_ALLOCATION;}}
extern "C" int av1_stage_read(const av1_stage* s,uint32_t env,av1_evaluation* out){try{if(!s||s->consumed||!output_valid(out))return AV1_INVALID;Eval e(s->owner->model.d.bodies,s->owner->N);int rc=state_eval(*s->owner,s->state,env,e);if(!rc)emit(e,out);return rc;}catch(const std::bad_alloc&){return AV1_ALLOCATION;}}
extern "C" int av1_stage_diagnostics(const av1_stage* s,float* a,float* f){if(!s||s->consumed||!a||(!s->constraint.empty()&&!f))return AV1_INVALID;std::copy(s->acceleration.begin(),s->acceleration.end(),a);if(f)std::copy(s->constraint.begin(),s->constraint.end(),f);return AV1_OK;}
extern "C" int av1_validate_commit(const av1_scene* s,const av1_stage* t){return !s||!t?AV1_INVALID:((t->owner!=s||t->consumed||t->generation!=s->generation||s->generation==UINT64_MAX)?AV1_STALE:AV1_OK);}
extern "C" int av1_commit(av1_scene* s,av1_stage* t){int rc=av1_validate_commit(s,t);if(rc)return rc;std::swap(s->state,t->state);++s->generation;t->consumed=true;return AV1_OK;}
extern "C" void av1_stage_destroy(av1_stage* s){delete s;}
extern "C" int av1_response(const av1_scene* s,uint32_t env,const float* g,float* dv,double* bodydv,double* effective){try{
 if(!s||!g||!dv||!bodydv||!effective)return AV1_INVALID;Eval e(s->model.d.bodies,s->N);int rc=state_eval(*s,s->state,env,e);if(rc)return rc;
 std::vector<float> mass(e.M.begin(),e.M.end()),response(s->N);float weight=0;rc=box3d_joint_v1_response(s->N,mass.data(),s->coeff[0].data(),g,response.data(),&weight);if(rc)return rc;
 std::vector<double> b(s->model.d.bodies*6);for(size_t i=0;i<b.size();i++)for(uint32_t n=0;n<s->N;n++)b[i]+=e.J[i*s->N+n]*response[n];if(!finite(b.data(),b.size()))return AV1_DYNAMICS;
 std::copy(response.begin(),response.end(),dv);std::copy(b.begin(),b.end(),bodydv);*effective=weight;return AV1_OK;
}catch(const std::bad_alloc&){return AV1_ALLOCATION;}}
