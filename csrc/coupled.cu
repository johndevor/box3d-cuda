// SPDX-License-Identifier: MIT
// Stage 7: maximal-coordinate joints and persistent OBB contacts solved in
// one ordered iteration loop. One CUDA lane owns one world. There is no
// attachment state, pose copying, teleportation, or hidden payload force.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
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

__device__ inline void warm_joint_rows(
    float *state, const float *inverse_mass, const float *inverse_inertia,
    const int64_t *joint_indices, const int64_t *joint_types,
    const float *parent_anchor, const float *child_anchor,
    const float *axis_parent, const float *reference_quaternion,
    const float *lower_limit, const float *upper_limit, float *joint_cache, float warm_start_factor,
    float *joint_lambda, int world, int bodies, int joints) {
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
    float *joint_lambda, uint8_t *limit_active, int world, int bodies, int joints, float h) {
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
    if(type!=coupled_joint::FIXED && maximum_effort[world*joints+joint]>0){
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

__device__ inline void solve_contact_rows(
    float *state,const float *inverse_mass,const float *inverse_inertia,
    const int64_t *contact_pairs,coupled_contact::MF *manifolds,const bool *active,
    int world,int bodies,int pairs,float restitution,float friction,float epsilon){
  for(int pair=0;pair<pairs;++pair)if(active[pair]){
    int left=int(contact_pairs[pair*2]),right=int(contact_pairs[pair*2+1]);
    int left_flat=world*bodies+left,right_flat=world*bodies+right;
    for(int point=0;point<manifolds[pair].count;++point)
      coupled_contact::solve_normal(state+left_flat*STATE_WIDTH,inverse_mass[left_flat],inverse_inertia+left_flat*3,state+right_flat*STATE_WIDTH,inverse_mass[right_flat],inverse_inertia+right_flat*3,&manifolds[pair],manifolds[pair].points[point],restitution,epsilon);
    for(int point=0;point<manifolds[pair].count;++point)
      coupled_contact::solve_friction(state+left_flat*STATE_WIDTH,inverse_mass[left_flat],inverse_inertia+left_flat*3,state+right_flat*STATE_WIDTH,inverse_mass[right_flat],inverse_inertia+right_flat*3,&manifolds[pair],manifolds[pair].points[point],friction,epsilon);
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
    float maximum_linear_repair,float maximum_angular_repair){
  int world=blockIdx.x*blockDim.x+threadIdx.x;if(world>=worlds)return;
  float h=dt/substeps,damping_factor=fmaxf(0.0f,1.0f-angular_damping*h);
  for(int substep=0;substep<substeps;++substep){
    for(int body=0;body<bodies;++body){int flat=world*bodies+body;if(inverse_mass[flat]==0)continue;float *value=state+flat*STATE_WIDTH;value[8]+=gravity_y*h;for(int k=0;k<3;k++){value[k]+=value[7+k]*h;value[10+k]*=damping_factor;}coupled_contact::integrate_q(value,h);}
    coupled_contact::MF manifolds[MAX_CONTACT_PAIRS];bool contact_active[MAX_CONTACT_PAIRS]={};
    for(int pair=0;pair<pairs;++pair){int left=int(contact_pairs[pair*2]),right=int(contact_pairs[pair*2+1]);int left_flat=world*bodies+left,right_flat=world*bodies+right,offset=(world*pairs+pair)*4;contact_active[pair]=coupled_contact::manifold(state+left_flat*STATE_WIDTH,half_extents+left_flat*3,state+right_flat*STATE_WIDTH,half_extents+right_flat*3,pair,sat_epsilon,&manifolds[pair]);if(contact_active[pair]){contact_ever[world*pairs+pair]=1;coupled_contact::seed(&manifolds[pair],contact_feature_ids+offset,contact_impulse_cache+offset*3);}}
    float joint_lambda[MAX_JOINTS*8]={};
    warm_joint_rows(state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,axis_parent,reference_quaternion,lower_limit,upper_limit,joint_cache,warm_start_factor,joint_lambda,world,bodies,joints);
    for(int pair=0;pair<pairs;++pair)if(contact_active[pair]){int left=int(contact_pairs[pair*2]),right=int(contact_pairs[pair*2+1]);int left_flat=world*bodies+left,right_flat=world*bodies+right;coupled_contact::warm(state+left_flat*STATE_WIDTH,inverse_mass[left_flat],inverse_inertia+left_flat*3,state+right_flat*STATE_WIDTH,inverse_mass[right_flat],inverse_inertia+right_flat*3,&manifolds[pair]);}
    for(int iteration=0;iteration<solver_iterations;++iteration){
      solve_joint_rows(state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,axis_parent,reference_quaternion,lower_limit,upper_limit,damping,motor_enabled,target_velocity,target_position,stiffness,maximum_effort,joint_lambda,joint_limit_active,world,bodies,joints,h);
      solve_contact_rows(state,inverse_mass,inverse_inertia,contact_pairs,manifolds,contact_active,world,bodies,pairs,restitution,friction,sat_epsilon);
    }
    for(int pair=0;pair<pairs;++pair){int offset=(world*pairs+pair)*4;if(contact_active[pair])coupled_contact::write_cache(&manifolds[pair],contact_feature_ids+offset,contact_impulse_cache+offset*3);else coupled_contact::write_cache(nullptr,contact_feature_ids+offset,contact_impulse_cache+offset*3);}
    // Split position constraints are coupled too. Rebuilding contact geometry
    // between bounded passes avoids repeatedly applying a stale penetration.
    for(int repair_iteration=0;repair_iteration<4;++repair_iteration){
      for(int pair=pairs-1;pair>=0;--pair){int left=int(contact_pairs[pair*2]),right=int(contact_pairs[pair*2+1]);int left_flat=world*bodies+left,right_flat=world*bodies+right;coupled_contact::MF repair_manifold;if(coupled_contact::manifold(state+left_flat*STATE_WIDTH,half_extents+left_flat*3,state+right_flat*STATE_WIDTH,half_extents+right_flat*3,pair,sat_epsilon,&repair_manifold))repair_contact_with_articulation_shock(state+left_flat*STATE_WIDTH,inverse_mass[left_flat],left,state+right_flat*STATE_WIDTH,inverse_mass[right_flat],right,repair_manifold,joint_indices,joints,contact_slop,position_correction);}
      repair_joint_constraints(state,inverse_mass,inverse_inertia,joint_indices,joint_types,parent_anchor,child_anchor,axis_parent,reference_quaternion,lower_limit,upper_limit,world,bodies,joints,position_correction,joint_position_slop,angular_slop,maximum_linear_repair,maximum_angular_repair);
    }
    for(int joint=0;joint<joints;++joint){float *lambda=joint_lambda+joint*8;motor_impulse[world*joints+joint]+=lambda[6];for(int row=0;row<8;++row)joint_cache[(world*joints+joint)*8+row]=lambda[row];}
  }
  for(int joint=0;joint<joints;++joint){int parent=int(joint_indices[joint*2]),child=int(joint_indices[joint*2+1]);coupled_joint::G geometry=coupled_joint::geometry(state+(world*bodies+parent)*STATE_WIDTH,state+(world*bodies+child)*STATE_WIDTH,parent_anchor+joint*3,child_anchor+joint*3,axis_parent+joint*3,reference_quaternion+joint*4,int(joint_types[joint]));joint_coordinate[world*joints+joint]=geometry.coord;joint_anchor_error[world*joints+joint]=coupled_joint::norm3(geometry.lin);joint_angular_error[world*joints+joint]=coupled_joint::norm3(geometry.ang);joint_limit_error[world*joints+joint]=fmaxf(0.0f,fmaxf(lower_limit[joint]-geometry.coord,geometry.coord-upper_limit[joint]));}
  for(int pair=0;pair<pairs;++pair){int left=int(contact_pairs[pair*2]),right=int(contact_pairs[pair*2+1]),offset=(world*pairs+pair)*4;coupled_contact::MF manifold;bool active=coupled_contact::manifold(state+(world*bodies+left)*STATE_WIDTH,half_extents+(world*bodies+left)*3,state+(world*bodies+right)*STATE_WIDTH,half_extents+(world*bodies+right)*3,pair,sat_epsilon,&manifold);if(!active){coupled_contact::write_cache(nullptr,contact_feature_ids+offset,contact_impulse_cache+offset*3);continue;}coupled_contact::seed(&manifold,contact_feature_ids+offset,contact_impulse_cache+offset*3);coupled_contact::write_cache(&manifold,contact_feature_ids+offset,contact_impulse_cache+offset*3);contact_count[world*pairs+pair]=manifold.count;float maximum_depth=0,total_normal=0;for(int point=0;point<manifold.count;++point){maximum_depth=fmaxf(maximum_depth,manifold.points[point].depth);total_normal+=manifold.points[point].jn;}penetration[world*pairs+pair]=maximum_depth;normal_impulse[world*pairs+pair]=total_normal;}
}
}

std::vector<torch::Tensor> box3d_coupled_step_cuda(
    torch::Tensor state,torch::Tensor inverse_mass,torch::Tensor half_extents,torch::Tensor inverse_inertia,
    torch::Tensor joint_indices,torch::Tensor joint_types,torch::Tensor parent_anchor,torch::Tensor child_anchor,
    torch::Tensor axis_parent,torch::Tensor reference_quaternion,torch::Tensor lower_limit,torch::Tensor upper_limit,
    torch::Tensor damping,torch::Tensor motor_enabled,torch::Tensor target_velocity,torch::Tensor target_position,
    torch::Tensor stiffness,torch::Tensor maximum_effort,torch::Tensor joint_cache,torch::Tensor contact_pairs,
    torch::Tensor contact_feature_ids,torch::Tensor contact_impulse_cache,double warm_start_factor,double dt,
    int64_t substeps,double gravity_y,double restitution,double friction,double contact_slop,double position_correction,
    double angular_damping,int64_t solver_iterations,double sat_epsilon,double joint_position_slop,double angular_slop,
    double maximum_linear_repair,double maximum_angular_repair){
  const c10::cuda::CUDAGuard guard(state.device());auto output=state.clone();auto updated_joint_cache=joint_cache.clone();auto updated_feature_ids=contact_feature_ids.clone();auto updated_contact_cache=contact_impulse_cache.clone();int worlds=state.size(0),bodies=state.size(1),joints=joint_indices.size(0),pairs=contact_pairs.size(0);auto joint_scalar=torch::zeros({worlds,joints},state.options());auto coordinate=joint_scalar.clone(),anchor_error=joint_scalar.clone(),angular_error=joint_scalar.clone(),limit_error=joint_scalar.clone(),motor_impulse=joint_scalar.clone();auto limit_active=torch::zeros({worlds,joints},state.options().dtype(torch::kUInt8));auto contact_scalar=torch::zeros({worlds,pairs},state.options());auto penetration=contact_scalar.clone(),normal_impulse=contact_scalar.clone();auto contact_ever=torch::zeros({worlds,pairs},state.options().dtype(torch::kUInt8));auto contact_count=torch::zeros({worlds,pairs},state.options().dtype(torch::kInt32));constexpr int threads=64;coupled_kernel<<<(worlds+threads-1)/threads,threads,0,at::cuda::getDefaultCUDAStream()>>>(output.data_ptr<float>(),inverse_mass.data_ptr<float>(),half_extents.data_ptr<float>(),inverse_inertia.data_ptr<float>(),joint_indices.data_ptr<int64_t>(),joint_types.data_ptr<int64_t>(),parent_anchor.data_ptr<float>(),child_anchor.data_ptr<float>(),axis_parent.data_ptr<float>(),reference_quaternion.data_ptr<float>(),lower_limit.data_ptr<float>(),upper_limit.data_ptr<float>(),damping.data_ptr<float>(),motor_enabled.data_ptr<uint8_t>(),target_velocity.data_ptr<float>(),target_position.data_ptr<float>(),stiffness.data_ptr<float>(),maximum_effort.data_ptr<float>(),updated_joint_cache.data_ptr<float>(),contact_pairs.data_ptr<int64_t>(),updated_feature_ids.data_ptr<int64_t>(),updated_contact_cache.data_ptr<float>(),float(warm_start_factor),coordinate.data_ptr<float>(),anchor_error.data_ptr<float>(),angular_error.data_ptr<float>(),limit_error.data_ptr<float>(),motor_impulse.data_ptr<float>(),limit_active.data_ptr<uint8_t>(),contact_ever.data_ptr<uint8_t>(),penetration.data_ptr<float>(),contact_count.data_ptr<int32_t>(),normal_impulse.data_ptr<float>(),worlds,bodies,joints,pairs,float(dt),int(substeps),float(gravity_y),float(restitution),float(friction),float(contact_slop),float(position_correction),float(angular_damping),int(solver_iterations),float(sat_epsilon),float(joint_position_slop),float(angular_slop),float(maximum_linear_repair),float(maximum_angular_repair));C10_CUDA_KERNEL_LAUNCH_CHECK();return {output,coordinate,anchor_error,angular_error,limit_error,motor_impulse,limit_active,updated_joint_cache,contact_ever,penetration,updated_feature_ids,updated_contact_cache,contact_count,normal_impulse};
}
