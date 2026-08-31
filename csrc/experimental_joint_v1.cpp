// SPDX-License-Identifier: MIT
#include "box3d_cuda/experimental_joint_v1.h"
#include "experimental_joint_v1_shared.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>
namespace m=box3d_joint_v1;
struct box3d_joint_v1_scene {
 uint32_t n,e;uint64_t binding=UINT64_C(14695981039346656037);
 std::vector<uint8_t> revolute,motor;
 std::vector<float> armature,damping,loss,kp,kv,cap,d0,dwidth,tc,invweight;
 std::vector<float> q,v,warm,q0,v0;
 std::vector<uint64_t> steps;
};
namespace {
bool values(const float* p,size_t n){if(!p)return n==0;for(size_t i=0;i<n;i++)if(!m::finite(p[i]))return false;return true;}
void hash_bytes(uint64_t& h,const void* p,size_t n){auto b=static_cast<const uint8_t*>(p);for(size_t i=0;i<n;i++){h^=b[i];h*=UINT64_C(1099511628211);}}
template<class T> void own(std::vector<T>& out,const T* p,size_t n,uint64_t& h){out.assign(p,p+n);hash_bytes(h,p,n*sizeof(T));}
void publish(float* target,const std::vector<float>& v){if(target)std::copy(v.begin(),v.end(),target);}
bool snapshot_shape(const box3d_joint_v1_scene* s,const box3d_joint_v1_snapshot* p){return s&&p&&p->struct_size==sizeof(*p)&&p->version==BOX3D_JOINT_V1_ABI&&p->dofs==s->n&&p->environments==s->e&&p->q&&p->velocity&&p->friction_warm_force&&p->step_count;}
}
extern "C" int box3d_joint_v1_create(const box3d_joint_v1_params* p,box3d_joint_v1_scene** output){
 if(!p||!output||p->struct_size!=sizeof(*p)||p->version!=BOX3D_JOINT_V1_ABI||p->flags||p->reserved||p->dofs<1||p->dofs>m::max_dofs||p->environments<1||p->environments>4096||!p->revolute||!p->motor_enabled)return BOX3D_JOINT_V1_INVALID;
 const size_t n=p->dofs,e=p->environments,count=n*e;
 const float* arrays[]={p->armature,p->passive_damping,p->friction_loss,p->stiffness,p->motor_damping,p->maximum_effort,p->friction_d0,p->friction_dwidth,p->friction_timeconst};
 for(auto a:arrays)if(!values(a,n))return BOX3D_JOINT_V1_INVALID;
 if(!values(p->reference_body_mass,e*n*n)||!values(p->initial_q,count)||!values(p->initial_velocity,count))return BOX3D_JOINT_V1_INVALID;
 for(size_t i=0;i<n;i++){
  if(p->revolute[i]>1||p->motor_enabled[i]>1)return BOX3D_JOINT_V1_INVALID;
  for(int a=0;a<6;a++)if(arrays[a][i]<0)return BOX3D_JOINT_V1_INVALID;
  if(p->friction_d0[i]<=0||p->friction_dwidth[i]<p->friction_d0[i]||p->friction_dwidth[i]>=1||p->friction_timeconst[i]<=0)return BOX3D_JOINT_V1_INVALID;
  if(!p->revolute[i]&&(p->motor_enabled[i]||p->armature[i]||p->passive_damping[i]||p->friction_loss[i]||p->stiffness[i]||p->motor_damping[i]||p->maximum_effort[i]))return BOX3D_JOINT_V1_INVALID;
 }
 try {
  auto s=std::make_unique<box3d_joint_v1_scene>();s->n=n;s->e=e;
  hash_bytes(s->binding,&s->n,sizeof(s->n));hash_bytes(s->binding,&s->e,sizeof(s->e));
  own(s->revolute,p->revolute,n,s->binding);own(s->motor,p->motor_enabled,n,s->binding);
  std::vector<float>* dest[]={&s->armature,&s->damping,&s->loss,&s->kp,&s->kv,&s->cap,&s->d0,&s->dwidth,&s->tc};
  for(int a=0;a<9;a++)own(*dest[a],arrays[a],n,s->binding);
  own(s->q0,p->initial_q,count,s->binding);own(s->v0,p->initial_velocity,count,s->binding);
  hash_bytes(s->binding,p->reference_body_mass,e*n*n*sizeof(float));
  s->q=s->q0;s->v=s->v0;s->warm.assign(count,0);s->steps.assign(e,0);s->invweight.resize(count);
  float total[m::max_dofs*m::max_dofs],inv[m::max_dofs*m::max_dofs],l[m::max_dofs*m::max_dofs];
  for(size_t env=0;env<e;env++){
   auto body=p->reference_body_mass+env*n*n;
   if(!m::factor(int(n),body,l))return BOX3D_JOINT_V1_DYNAMICS;
   m::armature_mass(int(n),body,p->armature,total);
   if(!m::inverse(int(n),total,inv))return BOX3D_JOINT_V1_DYNAMICS;
   for(size_t i=0;i<n;i++)s->invweight[env*n+i]=inv[i*n+i];
  }
  *output=s.release();return BOX3D_JOINT_V1_OK;
 }catch(const std::bad_alloc&){return BOX3D_JOINT_V1_ALLOCATION;}
}
extern "C" void box3d_joint_v1_destroy(box3d_joint_v1_scene* s){delete s;}
extern "C" int box3d_joint_v1_capture(const box3d_joint_v1_scene* s,box3d_joint_v1_snapshot* p){
 if(!snapshot_shape(s,p))return BOX3D_JOINT_V1_INVALID;
 publish(p->q,s->q);publish(p->velocity,s->v);publish(p->friction_warm_force,s->warm);
 std::copy(s->steps.begin(),s->steps.end(),p->step_count);p->binding=s->binding;return BOX3D_JOINT_V1_OK;
}
extern "C" int box3d_joint_v1_restore(box3d_joint_v1_scene* s,const box3d_joint_v1_snapshot* p){
 if(!snapshot_shape(s,p)||p->binding!=s->binding)return BOX3D_JOINT_V1_INVALID;
 const size_t count=s->n*s->e;
 if(!values(p->q,count)||!values(p->velocity,count)||!values(p->friction_warm_force,count))return BOX3D_JOINT_V1_INVALID;
 for(size_t i=0;i<count;i++)if(std::abs(p->friction_warm_force[i])>s->loss[i%s->n])return BOX3D_JOINT_V1_INVALID;
 std::copy(p->q,p->q+count,s->q.begin());std::copy(p->velocity,p->velocity+count,s->v.begin());std::copy(p->friction_warm_force,p->friction_warm_force+count,s->warm.begin());std::copy(p->step_count,p->step_count+s->e,s->steps.begin());return BOX3D_JOINT_V1_OK;
}
extern "C" int box3d_joint_v1_reset_masked(box3d_joint_v1_scene* s,const uint8_t* mask,uint32_t count){
 if(!s||!mask||count!=s->e)return BOX3D_JOINT_V1_INVALID;
 for(size_t e=0;e<count;e++)if(mask[e]>1)return BOX3D_JOINT_V1_INVALID;
 for(size_t e=0;e<count;e++)if(mask[e]){for(size_t i=e*s->n;i<(e+1)*s->n;i++){s->q[i]=s->q0[i];s->v[i]=s->v0[i];s->warm[i]=0;}s->steps[e]=0;}
 return BOX3D_JOINT_V1_OK;
}
extern "C" int box3d_joint_v1_advance(box3d_joint_v1_scene* s,const box3d_joint_v1_step* p,box3d_joint_v1_output* output){
 if(!s||!p||p->struct_size!=sizeof(*p)||p->version!=BOX3D_JOINT_V1_ABI||!m::finite(p->dt)||p->dt<=0||!m::finite(p->tolerance)||p->tolerance<=0||p->max_iterations<1||p->max_iterations>4096)return BOX3D_JOINT_V1_INVALID;
 if(output&&(output->struct_size!=sizeof(*output)||output->version!=BOX3D_JOINT_V1_ABI))return BOX3D_JOINT_V1_INVALID;
 const size_t n=s->n,e=s->e,k=p->external_rows,count=n*e;
 if(k>32||n+k>m::max_rows||!values(p->body_mass,e*n*n)||!values(p->target_position,count)||!values(p->target_velocity,count)||!values(p->external_generalized_force,count))return BOX3D_JOINT_V1_INVALID;
 if(k&&(!values(p->constraint_jacobian,e*k*n)||!values(p->constraint_reference_acceleration,e*k)||!values(p->constraint_regularizer,e*k)||!values(p->constraint_lower,e*k)||!values(p->constraint_upper,e*k)||(p->constraint_warm_force&&!values(p->constraint_warm_force,e*k))))return BOX3D_JOINT_V1_INVALID;
 for(size_t row=0;row<e*k;row++)if(p->constraint_regularizer[row]<0||p->constraint_lower[row]>p->constraint_upper[row])return BOX3D_JOINT_V1_INVALID;
 for(auto steps:s->steps)if(steps==std::numeric_limits<uint64_t>::max())return BOX3D_JOINT_V1_INVALID;
 try {
  auto q=s->q,v=s->v,warm=s->warm;
  std::vector<float> acceleration(count),smooth(count),actuator(count),passive(count),friction(count,0),regularizer(count,0),aref(count,0),external(e*k),residual(e);
  std::vector<uint32_t> iterations(e);
  for(size_t env=0;env<e;env++){
   const size_t off=env*n;
   float total[m::max_dofs*m::max_dofs],inv[m::max_dofs*m::max_dofs],l[m::max_dofs*m::max_dofs],tau[m::max_dofs],rhs[m::max_dofs];
   const float* body=p->body_mass+env*n*n;
   if(!m::factor(int(n),body,l))return BOX3D_JOINT_V1_DYNAMICS;
   m::armature_mass(int(n),body,s->armature.data(),total);
   if(!m::inverse(int(n),total,inv))return BOX3D_JOINT_V1_DYNAMICS;
   for(size_t i=0;i<n;i++){
    const size_t a=off+i;
    float motor=s->motor[i]?s->kp[i]*(p->target_position[a]-s->q[a])+s->kv[i]*(p->target_velocity[a]-s->v[a]):0;
    if(!m::finite(motor))return BOX3D_JOINT_V1_DYNAMICS;
    actuator[a]=m::clamp(motor,-s->cap[i],s->cap[i]);passive[a]=-s->damping[i]*s->v[a];
    tau[i]=actuator[a]+passive[a]+p->external_generalized_force[a];
   }
   m::matvec(int(n),inv,tau,smooth.data()+off);
   float jac[m::max_rows*m::max_dofs]={},r[m::max_rows],ref[m::max_rows],lo[m::max_rows],hi[m::max_rows],f[m::max_rows],response[m::max_rows*m::max_dofs],h[m::max_rows*m::max_rows],linear[m::max_rows];
   int owners[m::max_rows],nr=0;
   for(size_t i=0;i<n;i++)if(s->loss[i]>0){
    jac[nr*n+i]=1;owners[nr]=int(i);
    m::friction_coefficients(p->dt,s->d0[i],s->dwidth[i],s->tc[i],s->invweight[off+i],s->v[off+i],r[nr],ref[nr]);
    if(!m::finite(r[nr])||r[nr]<=0||!m::finite(ref[nr]))return BOX3D_JOINT_V1_DYNAMICS;
    regularizer[off+i]=r[nr];aref[off+i]=ref[nr];lo[nr]=-s->loss[i];hi[nr]=s->loss[i];f[nr]=s->warm[off+i];nr++;
   }
   int friction_rows=nr;
   for(size_t j=0;j<k;j++,nr++){
    size_t row=env*k+j;
    for(size_t i=0;i<n;i++)jac[nr*n+i]=p->constraint_jacobian[row*n+i];
    r[nr]=p->constraint_regularizer[row];ref[nr]=p->constraint_reference_acceleration[row];lo[nr]=p->constraint_lower[row];hi[nr]=p->constraint_upper[row];f[nr]=p->constraint_warm_force?p->constraint_warm_force[row]:0;owners[nr]=-1;
   }
   for(int row=0;row<nr;row++){
    m::matvec(int(n),inv,jac+row*n,response+row*n);
    linear[row]=-ref[row];for(size_t i=0;i<n;i++)linear[row]+=jac[row*n+i]*smooth[off+i];
    for(int col=0;col<nr;col++){
     float x=row==col?r[row]:0;for(size_t i=0;i<n;i++)x+=jac[col*n+i]*response[row*n+i];
     h[row*nr+col]=x;if(!m::finite(x))return BOX3D_JOINT_V1_DYNAMICS;
    }
    if(!m::finite(linear[row]))return BOX3D_JOINT_V1_DYNAMICS;
   }
   float error=0;int used=0;
   if(nr&&!m::box_qp(nr,h,linear,lo,hi,p->max_iterations,p->tolerance,f,error,used))return BOX3D_JOINT_V1_NO_CONVERGENCE;
   residual[env]=error;iterations[env]=uint32_t(used);
   for(size_t i=0;i<n;i++)rhs[i]=tau[i];
   for(int row=0;row<nr;row++){
    for(size_t i=0;i<n;i++)rhs[i]+=jac[row*n+i]*f[row];
    if(row<friction_rows){friction[off+owners[row]]=f[row];warm[off+owners[row]]=f[row];}
    else external[env*k+row-friction_rows]=f[row];
   }
   m::matvec(int(n),inv,rhs,acceleration.data()+off);
   for(size_t i=0;i<n;i++){
    size_t a=off+i;v[a]+=p->dt*acceleration[a];if(s->revolute[i])q[a]+=p->dt*v[a];
    if(!m::finite(tau[i])||!m::finite(smooth[a])||!m::finite(acceleration[a])||!m::finite(v[a])||!m::finite(q[a]))return BOX3D_JOINT_V1_DYNAMICS;
   }
  }
  s->q.swap(q);s->v.swap(v);s->warm.swap(warm);for(auto& step:s->steps)step++;
  if(output){publish(output->acceleration,acceleration);publish(output->smooth_acceleration,smooth);publish(output->actuator,actuator);publish(output->passive,passive);publish(output->friction,friction);publish(output->regularizer,regularizer);publish(output->reference_acceleration,aref);publish(output->constraint_force,external);publish(output->projected_residual,residual);if(output->iterations)std::copy(iterations.begin(),iterations.end(),output->iterations);}
  return BOX3D_JOINT_V1_OK;
 }catch(const std::bad_alloc&){return BOX3D_JOINT_V1_ALLOCATION;}
}
extern "C" int box3d_joint_v1_response(uint32_t n,const float* body,const float* armature,const float* g,float* response,float* effective){
 if(n<1||n>m::max_dofs||!response||!effective||!values(body,n*n)||!values(armature,n)||!values(g,n))return BOX3D_JOINT_V1_INVALID;
 for(uint32_t i=0;i<n;i++)if(armature[i]<0)return BOX3D_JOINT_V1_INVALID;
 float total[m::max_dofs*m::max_dofs],l[m::max_dofs*m::max_dofs],result[m::max_dofs];
 if(!m::factor(int(n),body,l))return BOX3D_JOINT_V1_DYNAMICS;
 m::armature_mass(int(n),body,armature,total);if(!m::factor(int(n),total,l))return BOX3D_JOINT_V1_DYNAMICS;
 m::solve(int(n),l,g,result);float e=0;for(uint32_t i=0;i<n;i++){if(!m::finite(result[i]))return BOX3D_JOINT_V1_DYNAMICS;e+=g[i]*result[i];}
 if(!m::finite(e))return BOX3D_JOINT_V1_DYNAMICS;
 std::copy(result,result+n,response);*effective=e;return BOX3D_JOINT_V1_OK;
}
extern "C" int box3d_joint_v1_assemble_mass(uint32_t n,uint32_t b,const float* mass,const float* inertia,const float* quat,const float* jac,float* out){
 if(n<1||n>m::max_dofs||b<1||b>32||!out||!values(mass,b)||!values(inertia,b*3)||!values(quat,b*4)||!values(jac,b*6*n))return BOX3D_JOINT_V1_INVALID;
 float result[m::max_dofs*m::max_dofs]={},l[m::max_dofs*m::max_dofs];
 for(uint32_t body=0;body<b;body++){
  if(mass[body]<=0)return BOX3D_JOINT_V1_INVALID;
  for(int k=0;k<3;k++)if(inertia[body*3+k]<=0)return BOX3D_JOINT_V1_INVALID;
  auto q=quat+body*4;float norm=0;for(int k=0;k<4;k++)norm+=q[k]*q[k];
  if(std::abs(norm-1)>1e-5f)return BOX3D_JOINT_V1_INVALID;
  const float x=q[0],y=q[1],z=q[2],w=q[3];
  const float r[9]={1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)};
  m::accumulate_body_mass(int(n),mass[body],inertia+body*3,r,jac+body*6*n,result);
 }
 if(!m::factor(int(n),result,l))return BOX3D_JOINT_V1_DYNAMICS;
 std::copy(result,result+n*n,out);return BOX3D_JOINT_V1_OK;
}
