// SPDX-License-Identifier: MIT
#ifndef BOX3D_EXPERIMENTAL_JOINT_V1_H
#define BOX3D_EXPERIMENTAL_JOINT_V1_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Separate experimental HOST library. Not a v2 scene descriptor/capability.
#define BOX3D_JOINT_V1_ABI UINT32_C(0x4a530001)
enum { BOX3D_JOINT_V1_OK=0, BOX3D_JOINT_V1_INVALID=1,
       BOX3D_JOINT_V1_DYNAMICS=2, BOX3D_JOINT_V1_NO_CONVERGENCE=3,
       BOX3D_JOINT_V1_ALLOCATION=4 };
typedef struct box3d_joint_v1_scene box3d_joint_v1_scene;
typedef struct box3d_joint_v1_params {
 uint32_t struct_size,version,dofs,environments;
 uint32_t flags,reserved; // zero; no legacy/default activation
 const uint8_t* revolute; // [N], 0 unactuated generalized velocity, 1 revolute
 const uint8_t* motor_enabled; // [N], boolean
 const float* armature; // [N], kg m^2, nonrevolute entries must be zero
 const float* passive_damping; // [N], outside motor clamp
 const float* friction_loss; // [N], bounded N m, not contact friction
 const float* stiffness; // [N]
 const float* motor_damping; // [N], within actuator torque clamp
 const float* maximum_effort; // [N]
 const float* friction_d0; // [N], 0<d0<1
 const float* friction_dwidth; // [N], d0<=dwidth<1
 const float* friction_timeconst; // [N], >0; dampratio fixed1
 const float* reference_body_mass; // [E,N,N], SPD at q0; WITHOUT armature
 const float* initial_q; // [E,N], scalar revolute coords, other entries opaque
 const float* initial_velocity; // [E,N]
} box3d_joint_v1_params;
typedef struct box3d_joint_v1_step {
 uint32_t struct_size,version,external_rows,max_iterations;
 float dt,tolerance;
 const float* body_mass; // [E,N,N], freshly assembled at PRE, WITHOUT armature
 const float* target_position; // [E,N]
 const float* target_velocity; // [E,N]
 const float* external_generalized_force; // [E,N], including caller's -bias
 const float* constraint_jacobian; // optional [E,K,N]
 const float* constraint_reference_acceleration; // optional [E,K], includes -Jdot*v
 const float* constraint_regularizer; // optional [E,K], >=0
 const float* constraint_lower; // optional [E,K], finite force bounds
 const float* constraint_upper; // optional [E,K]
 const float* constraint_warm_force; // optional [E,K], caller-owned contact state
} box3d_joint_v1_step;
typedef struct box3d_joint_v1_output {
 uint32_t struct_size,version;
 float* acceleration; // optional [E,N]
 float* smooth_acceleration; // optional [E,N]
 float* actuator; // optional [E,N]
 float* passive; // optional [E,N]
 float* friction; // optional [E,N]
 float* regularizer; // optional [E,N]
 float* reference_acceleration; // optional [E,N]
 float* constraint_force; // optional [E,K]
 float* projected_residual; // optional [E]
 uint32_t* iterations; // optional [E]
} box3d_joint_v1_output;
typedef struct box3d_joint_v1_snapshot {
 uint32_t struct_size,version,dofs,environments;
 uint64_t binding; // create-time parameter/reference/reset-state FNV1a identity
 float* q; // caller-owned [E,N]
 float* velocity; // caller-owned [E,N]
 float* friction_warm_force; // caller-owned [E,N]; no legacy Jx8 cache
 uint64_t* step_count; // caller-owned [E]
} box3d_joint_v1_snapshot;
// Host pointers only, nonoverlapping writable arrays, serialized access by caller.
// Create copies all inputs. Step/restore/reset validate all environments first and
// commit atomically. Failure leaves owned state and output buffers unchanged.
// Only revolute q slots integrate q+=dt*v_new; nonrevolute q slots are opaque.
// This is a velocity/coordinate operator, not a free-root quaternion/FK engine.
int box3d_joint_v1_create(const box3d_joint_v1_params*,box3d_joint_v1_scene**);
void box3d_joint_v1_destroy(box3d_joint_v1_scene*);
int box3d_joint_v1_advance(box3d_joint_v1_scene*,const box3d_joint_v1_step*,box3d_joint_v1_output*);
int box3d_joint_v1_capture(const box3d_joint_v1_scene*,box3d_joint_v1_snapshot*);
int box3d_joint_v1_restore(box3d_joint_v1_scene*,const box3d_joint_v1_snapshot*);
int box3d_joint_v1_reset_masked(box3d_joint_v1_scene*,const uint8_t*,uint32_t);
// Stateless full-coupling seam; g is generalized unit impulse direction.
// Returns M_A^-1*g (ALL DOFs), g'M_A^-1*g. Never apply body-local fallback.
int box3d_joint_v1_response(uint32_t n,const float* body_mass,const float* armature,
                           const float* g,float* velocity_response,float* inverse_effective_mass);
// Assemble M_body=sum spatialJ' diag(mI,I_world) spatialJ, no armature included.
// quaternion [B,4] xyzw, jacobian [B,6,N]; fixed bodies should be omitted.
int box3d_joint_v1_assemble_mass(uint32_t n,uint32_t bodies,const float* mass,
 const float* principal_inertia,const float* xyzw,const float* jacobian,float* body_mass);
#ifdef __cplusplus
}
#endif
#endif
