// SPDX-License-Identifier: MIT
// Stage 7: maximal-coordinate joints and persistent OBB contacts solved in
// one ordered iteration loop. One CUDA lane owns one world. There is no
// attachment state, pose copying, teleportation, or hidden payload force.
#ifndef BOX3D_CUDA_NATIVE_KERNELS_ONLY
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>

#define BOX3D_DEVICE_HELPERS_ONLY
namespace coupled_contact {
#include "manifold.cu"
}
namespace coupled_joint {
#include "joint.cu"
}
#undef BOX3D_DEVICE_HELPERS_ONLY

namespace {
constexpr int STATE_WIDTH = 13;
constexpr int MAX_JOINTS = 16;
constexpr int MAX_CONTACT_PAIRS = 16;
constexpr int POSITION_REPAIR_ITERATIONS = 8;

__device__ inline void warm_joint_rows(
    float *state, const float *inverse_mass, const float *inverse_inertia,
    const int64_t *joint_indices, const int64_t *joint_types,
    const float *parent_anchor, const float *child_anchor,
    const float *axis_parent, const float *reference_quaternion,
    const float *lower_limit, const float *upper_limit, float *joint_cache, float warm_start_factor,
    float *joint_lambda, int world, int bodies, int joints, bool articulation_projection) {
  for (int joint = 0; joint < joints; ++joint) {
    int parent_index = int(joint_indices[joint * 2]);
    int child_index = int(joint_indices[joint * 2 + 1]);
    int parent_flat = world * bodies + parent_index;
    int child_flat = world * bodies + child_index;
    int type = int(joint_types[joint]);
    float *parent = state + parent_flat * STATE_WIDTH;
    float *child = state + child_flat * STATE_WIDTH;
    coupled_joint::G geometry = coupled_joint::geometry(
        parent, child, parent_anchor + joint * 3, child_anchor + joint * 3,
        axis_parent + joint * 3, reference_quaternion + joint * 4, type);
    float world_axes[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    float tangent1[3], tangent2[3];
    coupled_joint::basis(geometry.axis, tangent1, tangent2);
    float *lambda = joint_lambda + joint * 8;
    for (int row = 0; row < 8; ++row) {
      lambda[row] = joint_cache[(world * joints + joint) * 8 + row] * warm_start_factor;
    }
    if (geometry.coord >= lower_limit[joint] && geometry.coord <= upper_limit[joint]) lambda[7] = 0.0f;
    const float *linear_directions[3]; int linear_count = 0;
    const float *angular_directions[3]; int angular_count = 0;
    if (type == coupled_joint::PRISMATIC) {
      linear_directions[0] = tangent1; linear_directions[1] = tangent2; linear_count = 2;
      for (int row=0; row<3; ++row) angular_directions[row] = world_axes[row];
      angular_count = 3;
    } else if (type == coupled_joint::REVOLUTE) {
      for (int row=0; row<3; ++row) linear_directions[row] = world_axes[row];
      linear_count = 3; angular_directions[0] = tangent1; angular_directions[1] = tangent2;
      angular_count = 2;
    } else {
      for (int row=0; row<3; ++row) {
        linear_directions[row] = world_axes[row]; angular_directions[row] = world_axes[row];
      }
      linear_count = angular_count = 3;
    }
    for (int row=0; row<linear_count; ++row) {
      float impulse[3], negative[3];
      for (int k=0; k<3; ++k) { impulse[k]=linear_directions[row][k]*lambda[row]; negative[k]=-impulse[k]; }
      coupled_joint::impulse(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,geometry.rp,negative);
      coupled_joint::impulse(child,inverse_mass[child_flat],inverse_inertia+child_flat*3,geometry.rc,impulse);
    }
    for (int row=0; row<angular_count; ++row) {
      float impulse[3], negative[3];
      for (int k=0; k<3; ++k) { impulse[k]=angular_directions[row][k]*lambda[3+row]; negative[k]=-impulse[k]; }
      coupled_joint::aimpulse(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,negative);
      coupled_joint::aimpulse(child,inverse_mass[child_flat],inverse_inertia+child_flat*3,impulse);
    }
    if (type != coupled_joint::FIXED) for (int slot=6; slot<8; ++slot) {
      // The reduced path warm-starts actuator impulses after constructing its
      // generalized mass matrix. Limit rows remain maximal-coordinate until
      // a bounded active-limit oracle is added.
      if (articulation_projection && slot == 6) continue;
      float impulse[3], negative[3];
      for (int k=0; k<3; ++k) { impulse[k]=geometry.axis[k]*lambda[slot]; negative[k]=-impulse[k]; }
      if (type == coupled_joint::REVOLUTE) {
        coupled_joint::aimpulse(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,negative);
        coupled_joint::aimpulse(child,inverse_mass[child_flat],inverse_inertia+child_flat*3,impulse);
      } else {
        coupled_joint::impulse(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,geometry.rp,negative);
        coupled_joint::impulse(child,inverse_mass[child_flat],inverse_inertia+child_flat*3,geometry.rc,impulse);
      }
    }
  }
}

__device__ inline void solve_joint_rows(
    float *state, const float *inverse_mass, const float *inverse_inertia,
    const int64_t *joint_indices, const int64_t *joint_types,
    const float *parent_anchor, const float *child_anchor, const float *axis_parent,
    const float *reference_quaternion, const float *lower_limit, const float *upper_limit,
    const float *damping, const uint8_t *motor_enabled, const float *target_velocity,
    const float *target_position, const float *stiffness, const float *maximum_effort,
    float *joint_lambda, uint8_t *limit_active, int world, int bodies, int joints, float h,
    bool articulation_projection) {
  for (int joint=0; joint<joints; ++joint) {
    int parent_index=int(joint_indices[joint*2]), child_index=int(joint_indices[joint*2+1]);
    int parent_flat=world*bodies+parent_index, child_flat=world*bodies+child_index;
    int type=int(joint_types[joint]);
    float *parent=state+parent_flat*STATE_WIDTH, *child=state+child_flat*STATE_WIDTH;
    coupled_joint::G geometry=coupled_joint::geometry(parent,child,parent_anchor+joint*3,
        child_anchor+joint*3,axis_parent+joint*3,reference_quaternion+joint*4,type);
    float world_axes[3][3]={{1,0,0},{0,1,0},{0,0,1}}, tangent1[3], tangent2[3];
    coupled_joint::basis(geometry.axis,tangent1,tangent2);
    float *lambda=joint_lambda+joint*8;
    if(type==coupled_joint::PRISMATIC){
      lambda[0]+=coupled_joint::linearrow(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,child,inverse_mass[child_flat],inverse_inertia+child_flat*3,geometry.rp,geometry.rc,tangent1);
      lambda[1]+=coupled_joint::linearrow(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,child,inverse_mass[child_flat],inverse_inertia+child_flat*3,geometry.rp,geometry.rc,tangent2);
      for(int k=0;k<3;k++)lambda[3+k]+=coupled_joint::angularrow(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,child,inverse_mass[child_flat],inverse_inertia+child_flat*3,world_axes[k]);
    }else if(type==coupled_joint::REVOLUTE){
      for(int k=0;k<3;k++)lambda[k]+=coupled_joint::linearrow(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,child,inverse_mass[child_flat],inverse_inertia+child_flat*3,geometry.rp,geometry.rc,world_axes[k]);
      lambda[3]+=coupled_joint::angularrow(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,child,inverse_mass[child_flat],inverse_inertia+child_flat*3,tangent1);
      lambda[4]+=coupled_joint::angularrow(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,child,inverse_mass[child_flat],inverse_inertia+child_flat*3,tangent2);
    }else for(int k=0;k<3;k++){
      lambda[k]+=coupled_joint::linearrow(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,child,inverse_mass[child_flat],inverse_inertia+child_flat*3,geometry.rp,geometry.rc,world_axes[k]);
      lambda[3+k]+=coupled_joint::angularrow(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,child,inverse_mass[child_flat],inverse_inertia+child_flat*3,world_axes[k]);
    }
    if(!articulation_projection&&type!=coupled_joint::FIXED&&maximum_effort[world*joints+joint]>0){
      float vp[3],vc[3],relative[3],speed,denominator;
      if(type==coupled_joint::REVOLUTE){
        for(int k=0;k<3;k++)relative[k]=child[10+k]-parent[10+k];
        speed=coupled_joint::dot3(relative,geometry.axis);
        denominator=coupled_joint::aemass(parent,inverse_inertia+parent_flat*3,child,inverse_inertia+child_flat*3,geometry.axis);
      }else{
        coupled_joint::pointv(parent,geometry.rp,vp); coupled_joint::pointv(child,geometry.rc,vc);
        for(int k=0;k<3;k++)relative[k]=vc[k]-vp[k];
        speed=coupled_joint::dot3(relative,geometry.axis);
        denominator=coupled_joint::lemass(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,geometry.rp,child,inverse_mass[child_flat],inverse_inertia+child_flat*3,geometry.rc,geometry.axis);
      }
      if(denominator>1.0e-12f){
        float old=lambda[6], effort_limit=maximum_effort[world*joints+joint]*h, desired;
        if(stiffness[joint]>0) desired=(stiffness[joint]*(target_position[world*joints+joint]-geometry.coord)-damping[joint]*speed)*h;
        else { desired=old-damping[joint]*speed*h/denominator; if(motor_enabled[joint]) desired+=(target_velocity[world*joints+joint]-speed)/denominator; }
        lambda[6]=fmaxf(-effort_limit,fminf(effort_limit,desired));
        float delta=lambda[6]-old, impulse[3], negative[3];
        for(int k=0;k<3;k++){impulse[k]=geometry.axis[k]*delta;negative[k]=-impulse[k];}
        if(type==coupled_joint::REVOLUTE){coupled_joint::aimpulse(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,negative);coupled_joint::aimpulse(child,inverse_mass[child_flat],inverse_inertia+child_flat*3,impulse);}
        else{coupled_joint::impulse(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,geometry.rp,negative);coupled_joint::impulse(child,inverse_mass[child_flat],inverse_inertia+child_flat*3,geometry.rc,impulse);}
      }
    }
    if(type!=coupled_joint::FIXED && (geometry.coord<lower_limit[joint] || geometry.coord>upper_limit[joint])){
      float sign=geometry.coord<lower_limit[joint]?1.0f:-1.0f;
      float violation=geometry.coord<lower_limit[joint]?lower_limit[joint]-geometry.coord:geometry.coord-upper_limit[joint];
      float vp[3],vc[3],relative[3],speed,denominator;
      if(type==coupled_joint::REVOLUTE){for(int k=0;k<3;k++)relative[k]=child[10+k]-parent[10+k];speed=coupled_joint::dot3(relative,geometry.axis);denominator=coupled_joint::aemass(parent,inverse_inertia+parent_flat*3,child,inverse_inertia+child_flat*3,geometry.axis);}
      else{coupled_joint::pointv(parent,geometry.rp,vp);coupled_joint::pointv(child,geometry.rc,vc);for(int k=0;k<3;k++)relative[k]=vc[k]-vp[k];speed=coupled_joint::dot3(relative,geometry.axis);denominator=coupled_joint::lemass(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,geometry.rp,child,inverse_mass[child_flat],inverse_inertia+child_flat*3,geometry.rc,geometry.axis);}
      float delta=(sign*fminf(2.0f,violation*0.2f/h)-speed)/fmaxf(denominator,1.0e-12f),old=lambda[7];
      float next=sign*fmaxf(0.0f,sign*(old+delta));delta=next-old;lambda[7]=next;
      float impulse[3],negative[3];for(int k=0;k<3;k++){impulse[k]=geometry.axis[k]*delta;negative[k]=-impulse[k];}
      if(type==coupled_joint::REVOLUTE){coupled_joint::aimpulse(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,negative);coupled_joint::aimpulse(child,inverse_mass[child_flat],inverse_inertia+child_flat*3,impulse);}
      else{coupled_joint::impulse(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,geometry.rp,negative);coupled_joint::impulse(child,inverse_mass[child_flat],inverse_inertia+child_flat*3,geometry.rc,impulse);}
      limit_active[world*joints+joint]=1;
    }else lambda[7]=0.0f;
  }
}

struct Articulation2 {
  bool valid;
  int first_joint,second_joint,root,link1,link2;
  float origin1[3],origin2[3],axis1[3],axis2[3];
  float inverse_mass_matrix[3];  // 00, 01, 11
};

__device__ inline float inertia_bilinear(
    const float *body,const float *inverse_inertia,const float *first,const float *second){
  float conjugate[4]={-body[3],-body[4],-body[5],body[6]},local_first[3],local_second[3];
  coupled_joint::rotate3(conjugate,first,local_first);
  coupled_joint::rotate3(conjugate,second,local_second);
  float value=0.0f;
  for(int k=0;k<3;++k){
    if(inverse_inertia[k]<=1.0e-12f)return -1.0f;
    value+=local_first[k]*local_second[k]/inverse_inertia[k];
  }
  return value;
}

__device__ inline Articulation2 build_two_revolute_articulation(
    float *state,const float *inverse_mass,const float *inverse_inertia,
    const int64_t *joint_indices,const int64_t *joint_types,const float *parent_anchor,
    const float *child_anchor,const float *axis_parent,const float *reference_quaternion,
    int world,int bodies,int joints,int contact_body){
  Articulation2 result{};
  // The first production projection is deliberately bounded. Larger trees,
  // branches, and mixed joint types continue through the accepted maximal
  // solver until they receive their own oracle and CUDA gate.
  if(joints!=2)return result;
  int first=-1,second=-1,root=-1,link1=-1,link2=-1;
  for(int a=0;a<2;++a)for(int b=0;b<2;++b)if(a!=b){
    int candidate_root=int(joint_indices[a*2]);
    int candidate_link1=int(joint_indices[a*2+1]);
    int candidate_link2=int(joint_indices[b*2+1]);
    if(int(joint_indices[b*2])==candidate_link1){
      first=a;second=b;root=candidate_root;link1=candidate_link1;link2=candidate_link2;
    }
  }
  if(first<0||int(joint_types[first])!=coupled_joint::REVOLUTE||
      int(joint_types[second])!=coupled_joint::REVOLUTE||
      (contact_body!=link1&&contact_body!=link2))return result;
  int root_flat=world*bodies+root,link1_flat=world*bodies+link1,link2_flat=world*bodies+link2;
  if(inverse_mass[root_flat]!=0.0f||inverse_mass[link1_flat]<=0.0f||inverse_mass[link2_flat]<=0.0f)return result;
  float *root_state=state+root_flat*STATE_WIDTH,*link1_state=state+link1_flat*STATE_WIDTH;
  float *link2_state=state+link2_flat*STATE_WIDTH;
  coupled_joint::G geometry1=coupled_joint::geometry(
      root_state,link1_state,parent_anchor+first*3,child_anchor+first*3,
      axis_parent+first*3,reference_quaternion+first*4,coupled_joint::REVOLUTE);
  coupled_joint::G geometry2=coupled_joint::geometry(
      link1_state,link2_state,parent_anchor+second*3,child_anchor+second*3,
      axis_parent+second*3,reference_quaternion+second*4,coupled_joint::REVOLUTE);
  for(int k=0;k<3;++k){
    result.origin1[k]=root_state[k]+geometry1.rp[k];
    result.origin2[k]=link1_state[k]+geometry2.rp[k];
    result.axis1[k]=geometry1.axis[k];result.axis2[k]=geometry2.axis[k];
  }
  float center1_arm1[3],center2_arm1[3],center2_arm2[3];
  for(int k=0;k<3;++k){
    center1_arm1[k]=link1_state[k]-result.origin1[k];
    center2_arm1[k]=link2_state[k]-result.origin1[k];
    center2_arm2[k]=link2_state[k]-result.origin2[k];
  }
  float velocity11[3],velocity21[3],velocity22[3];
  coupled_joint::cross3(result.axis1,center1_arm1,velocity11);
  coupled_joint::cross3(result.axis1,center2_arm1,velocity21);
  coupled_joint::cross3(result.axis2,center2_arm2,velocity22);
  const float mass1=1.0f/inverse_mass[link1_flat],mass2=1.0f/inverse_mass[link2_flat];
  const float inertia11=inertia_bilinear(link1_state,inverse_inertia+link1_flat*3,result.axis1,result.axis1);
  const float inertia21=inertia_bilinear(link2_state,inverse_inertia+link2_flat*3,result.axis1,result.axis1);
  const float inertia22=inertia_bilinear(link2_state,inverse_inertia+link2_flat*3,result.axis2,result.axis2);
  const float inertia12=inertia_bilinear(link2_state,inverse_inertia+link2_flat*3,result.axis1,result.axis2);
  // The cross inertia term may legitimately be negative for non-parallel
  // axes. Only the diagonal kinetic-energy terms must be strictly positive.
  if(!(inertia11>0.0f&&inertia21>0.0f&&inertia22>0.0f)||!isfinite(inertia12))
    return Articulation2{};
  const float m00=mass1*coupled_joint::dot3(velocity11,velocity11)+inertia11+
      mass2*coupled_joint::dot3(velocity21,velocity21)+inertia21;
  const float m01=mass2*coupled_joint::dot3(velocity21,velocity22)+inertia12;
  const float m11=mass2*coupled_joint::dot3(velocity22,velocity22)+inertia22;
  const float determinant=m00*m11-m01*m01;
  if(!isfinite(m00)||!isfinite(m01)||!isfinite(m11)||
      !isfinite(determinant)||determinant<=1.0e-12f)return Articulation2{};
  result.valid=true;result.first_joint=first;result.second_joint=second;
  result.root=root;result.link1=link1;result.link2=link2;
  result.inverse_mass_matrix[0]=m11/determinant;
  result.inverse_mass_matrix[1]=-m01/determinant;
  result.inverse_mass_matrix[2]=m00/determinant;
  return result;
}

__device__ inline void apply_articulation_velocity_delta(
    float *state,int world,int bodies,const Articulation2 &articulation,
    float delta0,float delta1){
  float *link1=state+(world*bodies+articulation.link1)*STATE_WIDTH;
  float *link2=state+(world*bodies+articulation.link2)*STATE_WIDTH;
  float arm11[3],arm21[3],arm22[3],velocity11[3],velocity21[3],velocity22[3];
  for(int k=0;k<3;++k){
    arm11[k]=link1[k]-articulation.origin1[k];
    arm21[k]=link2[k]-articulation.origin1[k];
    arm22[k]=link2[k]-articulation.origin2[k];
  }
  coupled_joint::cross3(articulation.axis1,arm11,velocity11);
  coupled_joint::cross3(articulation.axis1,arm21,velocity21);
  coupled_joint::cross3(articulation.axis2,arm22,velocity22);
  for(int k=0;k<3;++k){
    link1[7+k]+=velocity11[k]*delta0;
    link1[10+k]+=articulation.axis1[k]*delta0;
    link2[7+k]+=velocity21[k]*delta0+velocity22[k]*delta1;
    link2[10+k]+=articulation.axis1[k]*delta0+articulation.axis2[k]*delta1;
  }
}

__device__ inline void articulation_jacobian(
    const Articulation2 &articulation,int body,const float *point,const float *direction,float *jacobian){
  float arm1[3],velocity1[3];
  for(int k=0;k<3;++k)arm1[k]=point[k]-articulation.origin1[k];
  coupled_joint::cross3(articulation.axis1,arm1,velocity1);
  jacobian[0]=coupled_joint::dot3(direction,velocity1);
  jacobian[1]=0.0f;
  if(body==articulation.link2){
    float arm2[3],velocity2[3];
    for(int k=0;k<3;++k)arm2[k]=point[k]-articulation.origin2[k];
    coupled_joint::cross3(articulation.axis2,arm2,velocity2);
    jacobian[1]=coupled_joint::dot3(direction,velocity2);
  }
}

__device__ inline float articulation_effective_mass(
    const Articulation2 &articulation,int body,const float *point,const float *direction){
  float jacobian[2];articulation_jacobian(articulation,body,point,direction,jacobian);
  return fmaxf(0.0f,
      jacobian[0]*(articulation.inverse_mass_matrix[0]*jacobian[0]+
                   articulation.inverse_mass_matrix[1]*jacobian[1])+
      jacobian[1]*(articulation.inverse_mass_matrix[1]*jacobian[0]+
                   articulation.inverse_mass_matrix[2]*jacobian[1]));
}

__device__ inline void apply_articulation_impulse(
    float *state,int world,int bodies,const Articulation2 &articulation,int body,
    const float *point,const float *impulse){
  float generalized[2];articulation_jacobian(articulation,body,point,impulse,generalized);
  const float delta0=articulation.inverse_mass_matrix[0]*generalized[0]+
      articulation.inverse_mass_matrix[1]*generalized[1];
  const float delta1=articulation.inverse_mass_matrix[1]*generalized[0]+
      articulation.inverse_mass_matrix[2]*generalized[1];
  apply_articulation_velocity_delta(state,world,bodies,articulation,delta0,delta1);
}

__device__ inline void apply_articulation_generalized_impulse(
    float *state,int world,int bodies,const Articulation2 &articulation,
    float impulse0,float impulse1){
  const float delta0=articulation.inverse_mass_matrix[0]*impulse0+
      articulation.inverse_mass_matrix[1]*impulse1;
  const float delta1=articulation.inverse_mass_matrix[1]*impulse0+
      articulation.inverse_mass_matrix[2]*impulse1;
  apply_articulation_velocity_delta(state,world,bodies,articulation,delta0,delta1);
}

__device__ inline void warm_projected_motors(
    float *state,const float *inverse_mass,const float *inverse_inertia,
    const int64_t *joint_indices,const int64_t *joint_types,const float *parent_anchor,
    const float *child_anchor,const float *axis_parent,const float *reference_quaternion,
    float *joint_lambda,int world,int bodies,int joints){
  // link2 identifies the complete chain after validation; either link would
  // be sufficient, but the distal body makes the intended tree unambiguous.
  int distal=int(joint_indices[1]);
  for(int joint=0;joint<joints;++joint)
    if(int(joint_indices[joint*2])==distal)distal=int(joint_indices[joint*2+1]);
  Articulation2 articulation=build_two_revolute_articulation(
      state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,
      axis_parent,reference_quaternion,world,bodies,joints,distal);
  if(!articulation.valid)return;
  apply_articulation_generalized_impulse(
      state,world,bodies,articulation,
      joint_lambda[articulation.first_joint*8+6],
      joint_lambda[articulation.second_joint*8+6]);
}

__device__ inline void solve_projected_motors(
    float *state,const float *inverse_mass,const float *inverse_inertia,
    const int64_t *joint_indices,const int64_t *joint_types,const float *parent_anchor,
    const float *child_anchor,const float *axis_parent,const float *reference_quaternion,
    const float *damping,const uint8_t *motor_enabled,const float *target_velocity,
    const float *target_position,const float *stiffness,const float *maximum_effort,
    float *joint_lambda,int world,int bodies,int joints,float h){
  int distal=int(joint_indices[1]);
  for(int joint=0;joint<joints;++joint)
    if(int(joint_indices[joint*2])==distal)distal=int(joint_indices[joint*2+1]);
  Articulation2 articulation=build_two_revolute_articulation(
      state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,
      axis_parent,reference_quaternion,world,bodies,joints,distal);
  if(!articulation.valid)return;
  const int ordered_joints[2]={articulation.first_joint,articulation.second_joint};
  for(int coordinate_index=0;coordinate_index<2;++coordinate_index){
    const int joint=ordered_joints[coordinate_index];
    if(maximum_effort[world*joints+joint]<=0.0f)continue;
    const int parent_index=int(joint_indices[joint*2]);
    const int child_index=int(joint_indices[joint*2+1]);
    float *parent=state+(world*bodies+parent_index)*STATE_WIDTH;
    float *child=state+(world*bodies+child_index)*STATE_WIDTH;
    coupled_joint::G geometry=coupled_joint::geometry(
        parent,child,parent_anchor+joint*3,child_anchor+joint*3,
        axis_parent+joint*3,reference_quaternion+joint*4,coupled_joint::REVOLUTE);
    float relative_angular[3];
    for(int k=0;k<3;++k)relative_angular[k]=child[10+k]-parent[10+k];
    const float speed=coupled_joint::dot3(relative_angular,geometry.axis);
    const float denominator=coordinate_index==0?
        articulation.inverse_mass_matrix[0]:articulation.inverse_mass_matrix[2];
    if(denominator<=1.0e-12f)continue;
    float *lambda=joint_lambda+joint*8;
    const float old=lambda[6],effort_limit=maximum_effort[world*joints+joint]*h;
    float desired;
    if(stiffness[joint]>0.0f)
      desired=(stiffness[joint]*(target_position[world*joints+joint]-geometry.coord)-
               damping[joint]*speed)*h;
    else{
      desired=old-damping[joint]*speed*h/denominator;
      if(motor_enabled[joint])desired+=(target_velocity[world*joints+joint]-speed)/denominator;
    }
    lambda[6]=fmaxf(-effort_limit,fminf(effort_limit,desired));
    const float delta=lambda[6]-old;
    apply_articulation_generalized_impulse(
        state,world,bodies,articulation,
        coordinate_index==0?delta:0.0f,
        coordinate_index==1?delta:0.0f);
  }
}

__device__ inline float free_body_effective_mass(
    const float *body,float inverse_mass,const float *inverse_inertia,
    const float *point,const float *direction){
  float arm[3],cross[3],angular[3],response[3];
  for(int k=0;k<3;++k)arm[k]=point[k]-body[k];
  coupled_joint::cross3(arm,direction,cross);
  coupled_joint::iworld(body+3,inverse_inertia,cross,angular);
  coupled_joint::cross3(angular,arm,response);
  return inverse_mass+coupled_joint::dot3(response,direction);
}

__device__ inline float projected_pair_effective_mass(
    float *state,const float *inverse_mass,const float *inverse_inertia,
    const int64_t *joint_indices,const int64_t *joint_types,const float *parent_anchor,
    const float *child_anchor,const float *axis_parent,const float *reference_quaternion,
    int world,int bodies,int joints,int left,int right,const float *point,const float *direction,
    Articulation2 *left_articulation,Articulation2 *right_articulation){
  *left_articulation=build_two_revolute_articulation(
      state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,
      axis_parent,reference_quaternion,world,bodies,joints,left);
  *right_articulation=build_two_revolute_articulation(
      state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,
      axis_parent,reference_quaternion,world,bodies,joints,right);
  int left_flat=world*bodies+left,right_flat=world*bodies+right;
  const float left_value=left_articulation->valid?
      articulation_effective_mass(*left_articulation,left,point,direction):
      free_body_effective_mass(state+left_flat*STATE_WIDTH,inverse_mass[left_flat],
          inverse_inertia+left_flat*3,point,direction);
  const float right_value=right_articulation->valid?
      articulation_effective_mass(*right_articulation,right,point,direction):
      free_body_effective_mass(state+right_flat*STATE_WIDTH,inverse_mass[right_flat],
          inverse_inertia+right_flat*3,point,direction);
  return left_value+right_value;
}

__device__ inline void apply_projected_pair_impulse(
    float *state,const float *inverse_mass,const float *inverse_inertia,int world,int bodies,
    int left,int right,const float *point,const float *impulse,
    const Articulation2 &left_articulation,const Articulation2 &right_articulation){
  int left_flat=world*bodies+left,right_flat=world*bodies+right;
  float negative[3]={-impulse[0],-impulse[1],-impulse[2]};
  if(left_articulation.valid)apply_articulation_impulse(
      state,world,bodies,left_articulation,left,point,negative);
  else{
    float arm[3];for(int k=0;k<3;++k)arm[k]=point[k]-state[left_flat*STATE_WIDTH+k];
    coupled_joint::impulse(state+left_flat*STATE_WIDTH,inverse_mass[left_flat],
        inverse_inertia+left_flat*3,arm,negative);
  }
  if(right_articulation.valid)apply_articulation_impulse(
      state,world,bodies,right_articulation,right,point,impulse);
  else{
    float arm[3];for(int k=0;k<3;++k)arm[k]=point[k]-state[right_flat*STATE_WIDTH+k];
    coupled_joint::impulse(state+right_flat*STATE_WIDTH,inverse_mass[right_flat],
        inverse_inertia+right_flat*3,arm,impulse);
  }
}

__device__ inline void warm_projected_contact(
    float *state,const float *inverse_mass,const float *inverse_inertia,
    const int64_t *joint_indices,const int64_t *joint_types,const float *parent_anchor,
    const float *child_anchor,const float *axis_parent,const float *reference_quaternion,
    int world,int bodies,int joints,int left,int right,coupled_contact::MF *manifold){
  for(int point=0;point<manifold->count;++point){
    coupled_contact::MP &contact=manifold->points[point];
    float impulse[3];for(int k=0;k<3;++k)impulse[k]=
        manifold->n[k]*contact.jn+manifold->t1[k]*contact.jt1+manifold->t2[k]*contact.jt2;
    Articulation2 left_articulation=build_two_revolute_articulation(
        state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,
        axis_parent,reference_quaternion,world,bodies,joints,left);
    Articulation2 right_articulation=build_two_revolute_articulation(
        state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,
        axis_parent,reference_quaternion,world,bodies,joints,right);
    apply_projected_pair_impulse(state,inverse_mass,inverse_inertia,world,bodies,left,right,
        contact.p,impulse,left_articulation,right_articulation);
  }
}

__device__ inline void solve_projected_normal(
    float *state,const float *inverse_mass,const float *inverse_inertia,
    const int64_t *joint_indices,const int64_t *joint_types,const float *parent_anchor,
    const float *child_anchor,const float *axis_parent,const float *reference_quaternion,
    int world,int bodies,int joints,int left,int right,coupled_contact::MF *manifold,
    coupled_contact::MP &contact,float restitution,float epsilon){
  int left_flat=world*bodies+left,right_flat=world*bodies+right;
  float left_arm[3],right_arm[3],left_velocity[3],right_velocity[3],relative[3];
  for(int k=0;k<3;++k){left_arm[k]=contact.p[k]-state[left_flat*STATE_WIDTH+k];right_arm[k]=contact.p[k]-state[right_flat*STATE_WIDTH+k];}
  coupled_contact::point_v(state+left_flat*STATE_WIDTH,left_arm,left_velocity);
  coupled_contact::point_v(state+right_flat*STATE_WIDTH,right_arm,right_velocity);
  for(int k=0;k<3;++k)relative[k]=right_velocity[k]-left_velocity[k];
  Articulation2 left_articulation{},right_articulation{};
  float denominator=projected_pair_effective_mass(
      state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,
      axis_parent,reference_quaternion,world,bodies,joints,left,right,contact.p,manifold->n,
      &left_articulation,&right_articulation);
  if(denominator<=epsilon)return;
  float normal_velocity=coupled_joint::dot3(relative,manifold->n);
  float bounce=normal_velocity<-.5f?-restitution*normal_velocity:0.0f;
  float old=contact.jn;contact.jn=fmaxf(0.0f,old+(bounce-normal_velocity)/denominator);
  float impulse[3];for(int k=0;k<3;++k)impulse[k]=manifold->n[k]*(contact.jn-old);
  apply_projected_pair_impulse(state,inverse_mass,inverse_inertia,world,bodies,left,right,
      contact.p,impulse,left_articulation,right_articulation);
}

__device__ inline void solve_projected_friction(
    float *state,const float *inverse_mass,const float *inverse_inertia,
    const int64_t *joint_indices,const int64_t *joint_types,const float *parent_anchor,
    const float *child_anchor,const float *axis_parent,const float *reference_quaternion,
    int world,int bodies,int joints,int left,int right,coupled_contact::MF *manifold,
    coupled_contact::MP &contact,float friction,float epsilon){
  int left_flat=world*bodies+left,right_flat=world*bodies+right;
  float left_arm[3],right_arm[3],left_velocity[3],right_velocity[3],relative[3];
  for(int k=0;k<3;++k){left_arm[k]=contact.p[k]-state[left_flat*STATE_WIDTH+k];right_arm[k]=contact.p[k]-state[right_flat*STATE_WIDTH+k];}
  coupled_contact::point_v(state+left_flat*STATE_WIDTH,left_arm,left_velocity);
  coupled_contact::point_v(state+right_flat*STATE_WIDTH,right_arm,right_velocity);
  for(int k=0;k<3;++k)relative[k]=right_velocity[k]-left_velocity[k];
  float proposed[2]={contact.jt1,contact.jt2};float *directions[2]={manifold->t1,manifold->t2};
  for(int tangent=0;tangent<2;++tangent){
    Articulation2 left_articulation{},right_articulation{};
    float denominator=projected_pair_effective_mass(
        state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,
        axis_parent,reference_quaternion,world,bodies,joints,left,right,contact.p,directions[tangent],
        &left_articulation,&right_articulation);
    if(denominator>epsilon)proposed[tangent]+=-coupled_joint::dot3(relative,directions[tangent])/denominator;
  }
  float limit=friction*contact.jn,magnitude=hypotf(proposed[0],proposed[1]);
  if(magnitude>limit&&magnitude>epsilon){proposed[0]*=limit/magnitude;proposed[1]*=limit/magnitude;}
  float impulse[3];for(int k=0;k<3;++k)impulse[k]=
      manifold->t1[k]*(proposed[0]-contact.jt1)+manifold->t2[k]*(proposed[1]-contact.jt2);
  contact.jt1=proposed[0];contact.jt2=proposed[1];
  Articulation2 left_articulation=build_two_revolute_articulation(
      state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,
      axis_parent,reference_quaternion,world,bodies,joints,left);
  Articulation2 right_articulation=build_two_revolute_articulation(
      state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,
      axis_parent,reference_quaternion,world,bodies,joints,right);
  apply_projected_pair_impulse(state,inverse_mass,inverse_inertia,world,bodies,left,right,
      contact.p,impulse,left_articulation,right_articulation);
}

__device__ inline void solve_contact_rows(
    float *state,const float *inverse_mass,const float *inverse_inertia,
    const int64_t *joint_indices,const int64_t *joint_types,const float *parent_anchor,
    const float *child_anchor,const float *axis_parent,const float *reference_quaternion,
    const int64_t *contact_pairs,coupled_contact::MF *manifolds,const bool *active,
    int world,int bodies,int joints,int pairs,float restitution,float friction,float epsilon,
    bool articulation_projection){
  for(int pair=0;pair<pairs;++pair)if(active[pair]){
    int left=int(contact_pairs[pair*2]),right=int(contact_pairs[pair*2+1]);
    int left_flat=world*bodies+left,right_flat=world*bodies+right;
    for(int point=0;point<manifolds[pair].count;++point){
      if(articulation_projection)solve_projected_normal(
          state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,
          axis_parent,reference_quaternion,world,bodies,joints,left,right,&manifolds[pair],
          manifolds[pair].points[point],restitution,epsilon);
      else coupled_contact::solve_normal(state+left_flat*STATE_WIDTH,inverse_mass[left_flat],
          inverse_inertia+left_flat*3,state+right_flat*STATE_WIDTH,inverse_mass[right_flat],
          inverse_inertia+right_flat*3,&manifolds[pair],manifolds[pair].points[point],restitution,epsilon);
    }
    for(int point=0;point<manifolds[pair].count;++point){
      if(articulation_projection)solve_projected_friction(
          state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,
          axis_parent,reference_quaternion,world,bodies,joints,left,right,&manifolds[pair],
          manifolds[pair].points[point],friction,epsilon);
      else coupled_contact::solve_friction(state+left_flat*STATE_WIDTH,inverse_mass[left_flat],
          inverse_inertia+left_flat*3,state+right_flat*STATE_WIDTH,inverse_mass[right_flat],
          inverse_inertia+right_flat*3,&manifolds[pair],manifolds[pair].points[point],friction,epsilon);
    }
  }
}

__device__ inline bool body_is_articulated(int body,const int64_t *joint_indices,int joints){
  for(int joint=0;joint<joints;++joint)if(int(joint_indices[joint*2])==body||int(joint_indices[joint*2+1])==body)return true;
  return false;
}

__device__ inline void repair_contact_with_articulation_shock(
    float *left,float left_inverse_mass,int left_index,float *right,float right_inverse_mass,int right_index,
    const coupled_contact::MF &manifold,const int64_t *joint_indices,int joints,float slop,float position_correction){
  // Shock propagation is a position-stabilization policy only. Velocity rows
  // above still use both bodies' physical mass and inertia. Treating the
  // articulated tree as the support layer prevents contact projection from
  // tearing links away from their joints before joint repair moves them back.
  const bool left_articulated=body_is_articulated(left_index,joint_indices,joints);
  const bool right_articulated=body_is_articulated(right_index,joint_indices,joints);
  // Against a free payload, keep the articulation as the support layer and
  // project the payload. Against an immovable floor, however, the articulated
  // link must be projected: otherwise both weights are zero and a link can
  // visibly tunnel into the floor. The following joint-repair pass propagates
  // that correction through the chain, and repeated bounded passes converge.
  float left_weight=left_articulated&&right_inverse_mass>0.0f?0.0f:left_inverse_mass;
  float right_weight=right_articulated&&left_inverse_mass>0.0f?0.0f:right_inverse_mass;
  float sum=left_weight+right_weight;if(sum<=0.0f)return;
  float depth=0.0f;for(int point=0;point<manifold.count;++point)depth=fmaxf(depth,manifold.points[point].depth);
  float correction=fminf(0.2f,fmaxf(0.0f,depth-slop)*position_correction)/sum;
  for(int k=0;k<3;++k){left[k]-=manifold.n[k]*correction*left_weight;right[k]+=manifold.n[k]*correction*right_weight;}
}

__device__ inline void repair_joint_constraints(
    float *state,const float *inverse_mass,const float *inverse_inertia,const int64_t *joint_indices,
    const int64_t *joint_types,const float *parent_anchor,const float *child_anchor,const float *axis_parent,
    const float *reference_quaternion,const float *lower_limit,const float *upper_limit,
    int world,int bodies,int joints,float position_correction,float position_slop,float angular_slop,
    float maximum_linear_repair,float maximum_angular_repair){
  for(int joint=0;joint<joints;++joint){
    int parent_index=int(joint_indices[joint*2]),child_index=int(joint_indices[joint*2+1]);
    int parent_flat=world*bodies+parent_index,child_flat=world*bodies+child_index,type=int(joint_types[joint]);
    float *parent=state+parent_flat*STATE_WIDTH,*child=state+child_flat*STATE_WIDTH;
    coupled_joint::G geometry=coupled_joint::geometry(parent,child,parent_anchor+joint*3,child_anchor+joint*3,axis_parent+joint*3,reference_quaternion+joint*4,type);
    float linear_magnitude=coupled_joint::norm3(geometry.lin),sum=inverse_mass[parent_flat]+inverse_mass[child_flat];
    if(linear_magnitude>position_slop&&sum>0){float correction=fminf(maximum_linear_repair,(linear_magnitude-position_slop)*position_correction);for(int k=0;k<3;k++){float delta=geometry.lin[k]*correction/linear_magnitude;parent[k]+=delta*inverse_mass[parent_flat]/sum;child[k]-=delta*inverse_mass[child_flat]/sum;}}
    coupled_joint::angularrepair(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,child,inverse_mass[child_flat],inverse_inertia+child_flat*3,geometry.ang,position_correction,angular_slop,maximum_angular_repair);
    coupled_joint::G after=coupled_joint::geometry(parent,child,parent_anchor+joint*3,child_anchor+joint*3,axis_parent+joint*3,reference_quaternion+joint*4,type);
    if(type==coupled_joint::PRISMATIC&&(after.coord<lower_limit[joint]||after.coord>upper_limit[joint])){float excess=after.coord-(after.coord<lower_limit[joint]?lower_limit[joint]:upper_limit[joint]);float correction=fmaxf(-maximum_linear_repair,fminf(maximum_linear_repair,excess*position_correction));if(sum>0)for(int k=0;k<3;k++){parent[k]+=after.axis[k]*correction*inverse_mass[parent_flat]/sum;child[k]-=after.axis[k]*correction*inverse_mass[child_flat]/sum;}}
    else if(type==coupled_joint::REVOLUTE&&(after.coord<lower_limit[joint]||after.coord>upper_limit[joint])){float target=after.coord<lower_limit[joint]?lower_limit[joint]:upper_limit[joint],error[3];for(int k=0;k<3;k++)error[k]=after.axis[k]*(after.coord-target);coupled_joint::angularrepair(parent,inverse_mass[parent_flat],inverse_inertia+parent_flat*3,child,inverse_mass[child_flat],inverse_inertia+child_flat*3,error,position_correction,angular_slop,maximum_angular_repair);}
  }
}

__global__ void coupled_kernel(
    float *state,const float *inverse_mass,const float *half_extents,const float *inverse_inertia,
    const int64_t *joint_indices,const int64_t *joint_types,const float *parent_anchor,const float *child_anchor,
    const float *axis_parent,const float *reference_quaternion,const float *lower_limit,const float *upper_limit,
    const float *damping,const uint8_t *motor_enabled,const float *target_velocity,const float *target_position,
    const float *stiffness,const float *maximum_effort,float *joint_cache,const int64_t *contact_pairs,
    int64_t *contact_feature_ids,float *contact_impulse_cache,float warm_start_factor,
    float *joint_coordinate,float *joint_anchor_error,float *joint_angular_error,float *joint_limit_error,
    float *motor_impulse,uint8_t *joint_limit_active,uint8_t *contact_ever,float *penetration,
    int32_t *contact_count,float *normal_impulse,int worlds,int bodies,int joints,int pairs,float dt,
    int substeps,float gravity_y,float restitution,float friction,float contact_slop,float position_correction,
    float angular_damping,int solver_iterations,float sat_epsilon,float joint_position_slop,float angular_slop,
    float maximum_linear_repair,float maximum_angular_repair,bool articulation_projection){
  int world=blockIdx.x*blockDim.x+threadIdx.x;if(world>=worlds)return;
  float h=dt/substeps,damping_factor=fmaxf(0.0f,1.0f-angular_damping*h);
  float control_pair_impulse[MAX_CONTACT_PAIRS][3]={};
  for(int substep=0;substep<substeps;++substep){
    for(int body=0;body<bodies;++body){int flat=world*bodies+body;if(inverse_mass[flat]==0)continue;float *value=state+flat*STATE_WIDTH;value[8]+=gravity_y*h;for(int k=0;k<3;k++){value[k]+=value[7+k]*h;value[10+k]*=damping_factor;}coupled_contact::integrate_q(value,h);}
    coupled_contact::MF manifolds[MAX_CONTACT_PAIRS];bool contact_active[MAX_CONTACT_PAIRS]={};
    for(int pair=0;pair<pairs;++pair){int left=int(contact_pairs[pair*2]),right=int(contact_pairs[pair*2+1]);int left_flat=world*bodies+left,right_flat=world*bodies+right,offset=(world*pairs+pair)*4;contact_active[pair]=coupled_contact::manifold(state+left_flat*STATE_WIDTH,half_extents+left_flat*3,state+right_flat*STATE_WIDTH,half_extents+right_flat*3,pair,sat_epsilon,&manifolds[pair]);if(contact_active[pair]){contact_ever[world*pairs+pair]=1;coupled_contact::seed(&manifolds[pair],contact_feature_ids+offset,contact_impulse_cache+offset*3);}}
    float joint_lambda[MAX_JOINTS*8]={};
    warm_joint_rows(state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,axis_parent,reference_quaternion,lower_limit,upper_limit,joint_cache,warm_start_factor,joint_lambda,world,bodies,joints,articulation_projection);
    if(articulation_projection)warm_projected_motors(
        state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,
        axis_parent,reference_quaternion,joint_lambda,world,bodies,joints);
    for(int pair=0;pair<pairs;++pair)if(contact_active[pair]){int left=int(contact_pairs[pair*2]),right=int(contact_pairs[pair*2+1]);int left_flat=world*bodies+left,right_flat=world*bodies+right;if(articulation_projection)warm_projected_contact(state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,axis_parent,reference_quaternion,world,bodies,joints,left,right,&manifolds[pair]);else coupled_contact::warm(state+left_flat*STATE_WIDTH,inverse_mass[left_flat],inverse_inertia+left_flat*3,state+right_flat*STATE_WIDTH,inverse_mass[right_flat],inverse_inertia+right_flat*3,&manifolds[pair]);}
    for(int iteration=0;iteration<solver_iterations;++iteration){
      solve_joint_rows(state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,axis_parent,reference_quaternion,lower_limit,upper_limit,damping,motor_enabled,target_velocity,target_position,stiffness,maximum_effort,joint_lambda,joint_limit_active,world,bodies,joints,h,articulation_projection);
      if(articulation_projection)solve_projected_motors(
          state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,
          axis_parent,reference_quaternion,damping,motor_enabled,target_velocity,target_position,
          stiffness,maximum_effort,joint_lambda,world,bodies,joints,h);
      solve_contact_rows(state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,axis_parent,reference_quaternion,contact_pairs,manifolds,contact_active,world,bodies,joints,pairs,restitution,friction,sat_epsilon,articulation_projection);
    }
    // Accumulate the complete contact-impulse vector over the full control
    // step. PhysX reports the magnitude of normal plus both friction axes;
    // returning only the final substep's normal component was not comparable.
    for(int pair=0;pair<pairs;++pair)if(contact_active[pair]){
      for(int point=0;point<manifolds[pair].count;++point){
        coupled_contact::MP &contact=manifolds[pair].points[point];
        for(int k=0;k<3;++k)
          control_pair_impulse[pair][k]+=
              manifolds[pair].n[k]*contact.jn+
              manifolds[pair].t1[k]*contact.jt1+
              manifolds[pair].t2[k]*contact.jt2;
      }
    }
    for(int pair=0;pair<pairs;++pair){int offset=(world*pairs+pair)*4;if(contact_active[pair])coupled_contact::write_cache(&manifolds[pair],contact_feature_ids+offset,contact_impulse_cache+offset*3);else coupled_contact::write_cache(nullptr,contact_feature_ids+offset,contact_impulse_cache+offset*3);}
    // Split position constraints are coupled too. Rebuilding contact geometry
    // between bounded passes avoids repeatedly applying a stale penetration.
    for(int repair_iteration=0;repair_iteration<POSITION_REPAIR_ITERATIONS;++repair_iteration){
      for(int pair=pairs-1;pair>=0;--pair){int left=int(contact_pairs[pair*2]),right=int(contact_pairs[pair*2+1]);int left_flat=world*bodies+left,right_flat=world*bodies+right;coupled_contact::MF repair_manifold;if(coupled_contact::manifold(state+left_flat*STATE_WIDTH,half_extents+left_flat*3,state+right_flat*STATE_WIDTH,half_extents+right_flat*3,pair,sat_epsilon,&repair_manifold))repair_contact_with_articulation_shock(state+left_flat*STATE_WIDTH,inverse_mass[left_flat],left,state+right_flat*STATE_WIDTH,inverse_mass[right_flat],right,repair_manifold,joint_indices,joints,contact_slop,position_correction);}
      repair_joint_constraints(state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,axis_parent,reference_quaternion,lower_limit,upper_limit,world,bodies,joints,position_correction,joint_position_slop,angular_slop,maximum_linear_repair,maximum_angular_repair);
    }
    for(int joint=0;joint<joints;++joint){float *lambda=joint_lambda+joint*8;motor_impulse[world*joints+joint]+=lambda[6];for(int row=0;row<8;++row)joint_cache[(world*joints+joint)*8+row]=lambda[row];}
  }
  for(int pair=0;pair<pairs;++pair)
    normal_impulse[world*pairs+pair]=coupled_joint::norm3(control_pair_impulse[pair]);
  for(int joint=0;joint<joints;++joint){int parent=int(joint_indices[joint*2]),child=int(joint_indices[joint*2+1]);coupled_joint::G geometry=coupled_joint::geometry(state+(world*bodies+parent)*STATE_WIDTH,state+(world*bodies+child)*STATE_WIDTH,parent_anchor+joint*3,child_anchor+joint*3,axis_parent+joint*3,reference_quaternion+joint*4,int(joint_types[joint]));joint_coordinate[world*joints+joint]=geometry.coord;joint_anchor_error[world*joints+joint]=coupled_joint::norm3(geometry.lin);joint_angular_error[world*joints+joint]=coupled_joint::norm3(geometry.ang);joint_limit_error[world*joints+joint]=fmaxf(0.0f,fmaxf(lower_limit[joint]-geometry.coord,geometry.coord-upper_limit[joint]));}
  for(int pair=0;pair<pairs;++pair){int left=int(contact_pairs[pair*2]),right=int(contact_pairs[pair*2+1]),offset=(world*pairs+pair)*4;coupled_contact::MF manifold;bool active=coupled_contact::manifold(state+(world*bodies+left)*STATE_WIDTH,half_extents+(world*bodies+left)*3,state+(world*bodies+right)*STATE_WIDTH,half_extents+(world*bodies+right)*3,pair,sat_epsilon,&manifold);if(!active){coupled_contact::write_cache(nullptr,contact_feature_ids+offset,contact_impulse_cache+offset*3);continue;}coupled_contact::seed(&manifold,contact_feature_ids+offset,contact_impulse_cache+offset*3);coupled_contact::write_cache(&manifold,contact_feature_ids+offset,contact_impulse_cache+offset*3);contact_count[world*pairs+pair]=manifold.count;float maximum_depth=0;for(int point=0;point<manifold.count;++point)maximum_depth=fmaxf(maximum_depth,manifold.points[point].depth);penetration[world*pairs+pair]=maximum_depth;}
}
}

#ifndef BOX3D_CUDA_NATIVE_KERNELS_ONLY
std::vector<torch::Tensor> box3d_coupled_step_cuda(
    torch::Tensor state,torch::Tensor inverse_mass,torch::Tensor half_extents,torch::Tensor inverse_inertia,
    torch::Tensor joint_indices,torch::Tensor joint_types,torch::Tensor parent_anchor,torch::Tensor child_anchor,
    torch::Tensor axis_parent,torch::Tensor reference_quaternion,torch::Tensor lower_limit,torch::Tensor upper_limit,
    torch::Tensor damping,torch::Tensor motor_enabled,torch::Tensor target_velocity,torch::Tensor target_position,
    torch::Tensor stiffness,torch::Tensor maximum_effort,torch::Tensor joint_cache,torch::Tensor contact_pairs,
    torch::Tensor contact_feature_ids,torch::Tensor contact_impulse_cache,double warm_start_factor,double dt,
    int64_t substeps,double gravity_y,double restitution,double friction,double contact_slop,double position_correction,
    double angular_damping,int64_t solver_iterations,double sat_epsilon,double joint_position_slop,double angular_slop,
    double maximum_linear_repair,double maximum_angular_repair,bool articulation_projection){
  const c10::cuda::CUDAGuard guard(state.device());auto output=state.clone();auto updated_joint_cache=joint_cache.clone();auto updated_feature_ids=contact_feature_ids.clone();auto updated_contact_cache=contact_impulse_cache.clone();int worlds=state.size(0),bodies=state.size(1),joints=joint_indices.size(0),pairs=contact_pairs.size(0);auto joint_scalar=torch::zeros({worlds,joints},state.options());auto coordinate=joint_scalar.clone(),anchor_error=joint_scalar.clone(),angular_error=joint_scalar.clone(),limit_error=joint_scalar.clone(),motor_impulse=joint_scalar.clone();auto limit_active=torch::zeros({worlds,joints},state.options().dtype(torch::kUInt8));auto contact_scalar=torch::zeros({worlds,pairs},state.options());auto penetration=contact_scalar.clone(),normal_impulse=contact_scalar.clone();auto contact_ever=torch::zeros({worlds,pairs},state.options().dtype(torch::kUInt8));auto contact_count=torch::zeros({worlds,pairs},state.options().dtype(torch::kInt32));constexpr int threads=64;coupled_kernel<<<(worlds+threads-1)/threads,threads,0,at::cuda::getDefaultCUDAStream()>>>(output.data_ptr<float>(),inverse_mass.data_ptr<float>(),half_extents.data_ptr<float>(),inverse_inertia.data_ptr<float>(),joint_indices.data_ptr<int64_t>(),joint_types.data_ptr<int64_t>(),parent_anchor.data_ptr<float>(),child_anchor.data_ptr<float>(),axis_parent.data_ptr<float>(),reference_quaternion.data_ptr<float>(),lower_limit.data_ptr<float>(),upper_limit.data_ptr<float>(),damping.data_ptr<float>(),motor_enabled.data_ptr<uint8_t>(),target_velocity.data_ptr<float>(),target_position.data_ptr<float>(),stiffness.data_ptr<float>(),maximum_effort.data_ptr<float>(),updated_joint_cache.data_ptr<float>(),contact_pairs.data_ptr<int64_t>(),updated_feature_ids.data_ptr<int64_t>(),updated_contact_cache.data_ptr<float>(),float(warm_start_factor),coordinate.data_ptr<float>(),anchor_error.data_ptr<float>(),angular_error.data_ptr<float>(),limit_error.data_ptr<float>(),motor_impulse.data_ptr<float>(),limit_active.data_ptr<uint8_t>(),contact_ever.data_ptr<uint8_t>(),penetration.data_ptr<float>(),contact_count.data_ptr<int32_t>(),normal_impulse.data_ptr<float>(),worlds,bodies,joints,pairs,float(dt),int(substeps),float(gravity_y),float(restitution),float(friction),float(contact_slop),float(position_correction),float(angular_damping),int(solver_iterations),float(sat_epsilon),float(joint_position_slop),float(angular_slop),float(maximum_linear_repair),float(maximum_angular_repair),articulation_projection);C10_CUDA_KERNEL_LAUNCH_CHECK();return {output,coordinate,anchor_error,angular_error,limit_error,motor_impulse,limit_active,updated_joint_cache,contact_ever,penetration,updated_feature_ids,updated_contact_cache,contact_count,normal_impulse};
}
#endif
