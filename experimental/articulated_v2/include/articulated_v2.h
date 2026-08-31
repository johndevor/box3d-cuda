// SPDX-License-Identifier: MIT
#ifndef BOX3D_ARTICULATED_V2_H
#define BOX3D_ARTICULATED_V2_H
#include "articulated_v1.h"
#ifdef __cplusplus
extern "C" {
#endif
#define AV2_ABI UINT32_C(0x41520002)
enum { AV2_OK=0,AV2_INVALID=1,AV2_DYNAMICS=2,AV2_NO_CONVERGENCE=3,AV2_ALLOCATION=4,AV2_STALE=5 };
enum { AV2_FRICTION=1,AV2_LOWER_LIMIT=2,AV2_UPPER_LIMIT=3 };
typedef struct av2_limit {
 uint32_t enabled,reserved;
 double lower,upper,margin,timeconst,dampratio,solimp[5];
} av2_limit;
typedef struct av2_registration {
 uint32_t struct_size,version,environments,reserved;
 const av1_model* model; // deep-copied; old ABI unchanged, used for static FK only
 const av2_limit* limits; // [J], explicit authored ranges and inherited softlimit law
 const double* initial_qpos; // [E,7+J], actual source root xyz/xyzw + hinges
 const double* initial_velocity; // [E,6+J], WORLD root linear/angular + hinge rates
 const double* gravity; // [E,3]
} av2_registration;
typedef struct av2_scene av2_scene;
typedef struct av2_pre av2_pre;
typedef struct av2_stage av2_stage;
typedef struct av2_step {
 uint32_t struct_size,version;
 double dt,momentum_tolerance,joint_impulse_tolerance; // positive, tolerances <=1e-5
 const double* target_position; // [E,J], actuator commands clamped to enabled ranges
 const double* target_velocity; // [E,J]
 const double* applied_force; // optional [E,N], without -bias
} av2_step;
typedef struct av2_pre_view {
 uint32_t struct_size,version,environments,bodies,dofs,joints,rows,reserved;
 uint64_t generation;
 double dt;
 const double *qpos,*velocity; // [E,7+J], [E,N]
 const double *mass,*inverse_mass; // [E,N,N], FULL M_body+diag(armature)
 const double *bias,*actuator,*passive,*smooth_velocity; // [E,N]
 const double *body_pose,*body_velocity,*spatial_jacobian; // [E,B,7/6/6*N]
 // Fixed R=3*J row slots: friction[0,J), lower[J,2J), upper[2J,3J).
 const double *row_jacobian,*row_target,*row_regularizer; // [E,R,N], [E,R], [E,R]
 const double *row_lower,*row_upper,*row_warm_impulse,*row_gap,*row_aref; // [E,R]
 const uint32_t *row_kind; // [R], stable IDs are slot index
 const uint8_t *row_active; // [E,R]; inactive G0/R1/bounds0, impulse MUST zero
} av2_pre_view;
typedef struct av2_solution {
 uint32_t struct_size,version;
 const double* velocity; // complete solved [E,N], before any pose integration
 const double* joint_impulse; // [E,3J], not force; hook stores force=lambda/dt
 const double* contact_generalized_impulse; // optional [E,N]; contact owner validates cones
} av2_solution;
typedef struct av2_snapshot {
 uint32_t struct_size,version,environments,joints;
 uint64_t binding;
 double *qpos,*velocity,*joint_warm_force,*time; // [E,7+J],[E,N],[E,3J],[E]
 uint64_t* step_count; // [E]
} av2_snapshot;
typedef struct av2_state_view {
 uint32_t struct_size,version,environments,bodies,joints,reserved;
 const double *qpos,*velocity,*joint_warm_force,*time,*body_pose,*body_velocity;
 const uint64_t* step_count;
} av2_state_view;
int av2_create(const av2_registration*,av2_scene**);
void av2_destroy(av2_scene*);
int av2_read(const av2_scene*,av2_state_view*);
int av2_capture(const av2_scene*,av2_snapshot*);
// PRE computes full dynamics and soft friction/limit rows, NEVER integrates or solves.
int av2_prepare(const av2_scene*,const av2_step*,av2_pre**);
int av2_pre_read(const av2_pre*,av2_pre_view*);
void av2_pre_destroy(av2_pre*);
// Independently checks momentum and joint projected KKT residuals on submitted
// complete solution. On success builds POST privately and integrates exactly once.
int av2_complete(const av2_pre*,const av2_solution*,av2_stage**);
int av2_stage_read(const av2_stage*,av2_state_view*);
int av2_stage_capture(const av2_stage*,av2_snapshot*);
// Restore/reset are also private stages, for coherent cross-owner transactions.
int av2_prepare_restore(const av2_scene*,const av2_snapshot*,av2_stage**);
int av2_prepare_reset(const av2_scene*,const uint8_t* mask,uint32_t count,av2_stage**);
int av2_validate_commit(const av2_scene*,const av2_stage*);
int av2_commit(av2_scene*,av2_stage*); // after validation+caller lock: allocation-free swap
void av2_stage_destroy(av2_stage*);
// C pointers/lengths/live handles and serialization are caller obligations.
// Views are read-only and valid while their owner remains alive and unmodified.
// Destroy pre/stages before scene. Both participants MUST validate before either
// commits; no concurrent mutation may intervene. No contact state is owned here.
#ifdef __cplusplus
}
#endif
#endif
