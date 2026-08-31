// SPDX-License-Identifier: MIT
#ifndef BOX3D_DUCK_WORLD_V1_H
#define BOX3D_DUCK_WORLD_V1_H
#include "articulated_v2.h"
#include "contact_v1.h"
#include "coupled_impulse_v1.h"
#ifdef __cplusplus
extern "C" {
#endif
/* duck_world_v1 (dwv1): per-env world = one articulated Open Duck (av2, glued
 * exactly like idv1) + a grid of cube rigid bodies + the static floor plane,
 * batched over E environments. Milestone 1: static cube terrain. Milestone 2:
 * dynamic 6-dof cubes with box-box/box-floor manifolds, uniform-grid
 * broadphase, contact islands, sleeping, per-island civ1 solves.
 * f64 dynamics/state, f32 contact geometry, momentum residual <=1e-8 enforced
 * per island by civ1. Steps are transactional: any failure leaves published
 * state (articulated + contact + cubes) unchanged; the diagnostic array
 * describes the first failure (environment, phase, native status).
 */
enum { DWV1_OK=0,DWV1_INVALID=1,DWV1_ALLOCATION=2,DWV1_CONTACT=3,
       DWV1_ARTICULATED=4,DWV1_SOLVER=5,DWV1_TRANSACTION=6,DWV1_CAPACITY=7 };
enum { DWV1_KIND_DUCK=0,DWV1_KIND_CUBE=1,DWV1_KIND_FLOOR=2 };
typedef struct dwv1_scene dwv1_scene;
typedef struct dwv1_snapshot dwv1_snapshot;
typedef struct dwv1_grid {
 uint32_t nx,nz;        /* lattice counts along world x and world y; nx*nz<=1024 */
 uint32_t dynamic;      /* 0: fixed terrain cubes; 1: free 6-dof rigid bodies */
 uint32_t reserved;     /* must be 0 */
 double cube_size;      /* edge length [0.01,0.5] m */
 double spacing;        /* lattice pitch; 0 selects cube_size; dynamic requires
                           spacing>=cube_size */
 double base_height;    /* z of cube bottoms before jitter, >=0 */
 double height_jitter;  /* per-cube uniform [0,jitter) z offset, from seed, [0,1] */
 double origin_x,origin_y; /* world center of the lattice */
 double cube_mass;      /* dynamic only, (0,100] kg */
 double friction;       /* mu for every cube-involved contact, [0,4] */
 uint64_t seed;         /* deterministic jitter seed */
} dwv1_grid;
typedef struct dwv1_registration {
 const av2_registration* articulation; /* reused av2 registration (deep-copied) */
 uint32_t pairs,reserved;
 const bcv1_shape* shapes;       /* [B] duck bodies; body0 fixed plane z=0 up */
 const bcv1_pair* contact_pairs; /* [P] authored duck pairs (feet-floor etc.) */
 const float* friction;          /* [E,P] */
 dwv1_grid grid;
} dwv1_registration;
typedef struct dwv1_diagnostic {
 uint32_t environment,phase,native_status,iterations,
          contact_points,active_limits,islands,awake_cubes,
          duck_island_cubes,max_island_dofs,reserved0,reserved1;
 double joint_residual,normal_residual,tangent_residual,momentum_residual,
        maximum_normal_impulse,maximum_penetration;
} dwv1_diagnostic;
typedef struct dwv1_contact {
 uint32_t kind_a,index_a,kind_b,index_b; /* kinds DWV1_KIND_*; duck index is body id */
 bcv1_manifold manifold;
} dwv1_contact;
/* Phases: 1 av2 PRE, 2 geometry (bcv1 query + broadphase + cube manifolds +
 * wake propagation), 3 island partition + per-island civ1 solves + cube
 * integration (staged), 4 av2 complete, 5 contact prepare, 6 validate/commit.
 */
int dwv1_create(const dwv1_registration*,dwv1_scene**);
void dwv1_destroy(dwv1_scene*);
int dwv1_step(dwv1_scene*,const av2_step*,uint32_t max_iterations,
              double impulse_tolerance,dwv1_diagnostic* /* [E] */);
/* All outputs optional (NULL skips). cube_pose [E,M,7] xyz+quat xyzw,
 * cube_velocity [E,M,6] world linear+angular, cube_awake [E,M] (static grid
 * reads all zero), foot_contact [E,F] flags of the last accepted step (feet
 * are the convex duck shapes in body-id order; the Open Duck has F=2,
 * bodies 6 and 15). geometry, cache and bodies mirror idv1_read. */
int dwv1_read(dwv1_scene*,double* qpos,double* velocity,double* joint_warm_force,
 double* clock,uint64_t* step_count,bcv1_body*,bcv1_manifold* pre_cache,
 bcv1_manifold* current_geometry,double* cube_pose,double* cube_velocity,
 uint8_t* cube_awake,uint8_t* foot_contact);
/* Current-pose cube-side manifold query for one environment (feet-cube,
 * cube-floor, cube-cube), zero impulses, no side effects. NULL output returns
 * only the count; otherwise capacity must cover the full count. */
int dwv1_query(dwv1_scene*,uint32_t environment,dwv1_contact* output,
               uint32_t capacity,uint32_t* count);
/* Test/setup teleport of one dynamic cube (rejected for static grids):
 * validates pose/velocity, wakes the cube and clears its warm caches.
 * This is the only sanctioned position write outside reset/restore. */
int dwv1_override_cube(dwv1_scene*,uint32_t environment,uint32_t cube,
                       const double pose[7],const double velocity[6]);
int dwv1_capture(dwv1_scene*,dwv1_snapshot**);
void dwv1_snapshot_destroy(dwv1_snapshot*);
/* NULL mask selects all; any nonzero byte selects. Same-scene identity
 * required. Selected environments restore articulated + contact + cube
 * payloads bit-exact; unselected environments unchanged. */
int dwv1_restore(dwv1_scene*,const dwv1_snapshot*,const uint8_t* mask);
int dwv1_reset(dwv1_scene*,const uint8_t* mask);
/* Correctly sized live host arrays/handles are caller obligations. A scene
 * serializes operations internally; destruction cannot race any operation. */
#ifdef __cplusplus
}
#endif
#endif
