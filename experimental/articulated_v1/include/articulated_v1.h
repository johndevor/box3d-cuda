// SPDX-License-Identifier: MIT
#ifndef BOX3D_ARTICULATED_V1_H
#define BOX3D_ARTICULATED_V1_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define AV1_ABI UINT32_C(0x41520001)
enum { AV1_OK=0, AV1_INVALID=1, AV1_DYNAMICS=2, AV1_NO_CONVERGENCE=3,
       AV1_ALLOCATION=4, AV1_STALE=5 };
typedef struct av1_body { double mass, principal_inertia[3]; } av1_body;
typedef struct av1_hinge {
 uint32_t parent, motor_enabled; // child is index+2; parent in [1,child)
 double parent_anchor[3], child_anchor[3], axis_parent[3], reference_xyzw[4];
 float armature, passive_damping, friction_loss, stiffness, motor_damping;
 float maximum_effort, friction_d0, friction_dwidth, friction_timeconst;
} av1_hinge;
typedef struct av1_model {
 uint32_t struct_size,version,bodies,joints,flags,reserved; // B=J+2, J<=26
 const av1_body* body; // B; floor0 zero, root1 and other bodies strictly massive
 const av1_hinge* hinge; // J; q=0 reference, no joint-coordinate offsets
 double root_source_to_principal[7]; // COM xyz and principal->source xyzw
 const double* reference_qpos; // [7+J], authored root xyz,xyzw + scalar hinges
} av1_model;
typedef struct av1_evaluation {
 uint32_t struct_size,version;
 double* body_pose; // required [B,7], principal COM xyz + xyzw; floor identity
 double* body_velocity; // required [B,6], world COM linear then angular
 double* jacobian; // required [B,6,N], N=6+J, root WORLD angular columns
 double* body_mass; // required [N,N], excludes armature
 double* bias; // required [N], gravity+Coriolis+gyroscopic; M_A*vdot=tau-bias
 double* body_bias_acceleration; // required [B,6], Jdot*v in this world basis
 double kinetic_energy, potential_energy; // body kinetic energy excludes armature
} av1_evaluation;
// Stateless static query; no time advancement. All outputs staged on failure.
int av1_evaluate(const av1_model*,const double* qpos,const double* velocity,
                 const double* gravity,av1_evaluation*);
// Root world linear/angular convention; left exponential rotation, no COM shift.
int av1_integrate_root(const double* xyz_xyzw,const double* world_velocity6,
                       double dt,double* result_xyz_xyzw);

typedef struct av1_registration {
 uint32_t struct_size,version,environments,reserved;
 const av1_model* model;
 const double* initial_qpos; // [E,7+J], root kept double; hinges become owned f32
 const float* initial_velocity; // [E,N], world root linear/angular + hinge qdot
 const double* gravity; // [E,3]
} av1_registration;
typedef struct av1_snapshot {
 uint32_t struct_size,version,environments,joints;
 uint64_t binding; // model+reference+initial state+gravity FNV1a, not authentication
 double* qpos; // [E,7+J], includes actual root quaternion, NOT opaque root slots
 float* velocity; // [E,N]
 float* friction_warm_force; // [E,N], root slots must be zero
 uint64_t* step_count; // [E]
} av1_snapshot;
typedef struct av1_step {
 uint32_t struct_size,version,external_rows,max_iterations;
 float dt,tolerance;
 const float* target_position; // [E,J], null permitted when J=0
 const float* target_velocity; // [E,J]
 const float* external_generalized_force; // [E,N], optional applied forces; NO -bias
 const float* constraint_jacobian; // optional [E,K,N], WORLD angular basis
 const float* constraint_reference_acceleration; // optional [E,K], includes -Gdot*v
 const float* constraint_regularizer; // optional [E,K]
 const float* constraint_lower; // optional [E,K], finite fixed bounds, NOT a cone
 const float* constraint_upper;
 const float* constraint_warm_force; // caller owns separate contact warm state
} av1_step;
typedef struct av1_scene av1_scene;
typedef struct av1_stage av1_stage;
int av1_create(const av1_registration*,av1_scene**);
void av1_destroy(av1_scene*);
int av1_capture(const av1_scene*,av1_snapshot*);
int av1_restore(av1_scene*,const av1_snapshot*);
int av1_reset_masked(av1_scene*,const uint8_t* mask,uint32_t count);
int av1_read(const av1_scene*,uint32_t environment,av1_evaluation*);
// Prepare clones joint state, assembles current M/bias, advances the frozen joint
// operator, and integrates actual root using v_post. Published scene is unchanged.
int av1_prepare(const av1_scene*,const av1_step*,av1_stage**);
int av1_stage_capture(const av1_stage*,av1_snapshot*);
int av1_stage_read(const av1_stage*,uint32_t environment,av1_evaluation*);
int av1_stage_diagnostics(const av1_stage*,float* acceleration,float* constraint_force);
// Serialized caller: validate both owners first, then fail-free commits are needed
// for future cross-scene atomicity. No combined contact commit is implemented here.
int av1_validate_commit(const av1_scene*,const av1_stage*);
int av1_commit(av1_scene*,av1_stage*);
void av1_stage_destroy(av1_stage*);
// Unit generalized impulse response at the current state, ALL coordinates/bodies.
int av1_response(const av1_scene*,uint32_t environment,const float* direction,
                  float* generalized_dv,double* all_body_dv,double* inverse_effective_mass);
// All pointers host-only, correctly sized, nonoverlapping writable outputs. Live
// handles and serialized calls are caller obligations; destroy stages before owner.
// Operations validate every environment before committing. No CUDA/r3 ABI change.
#ifdef __cplusplus
}
#endif
#endif
