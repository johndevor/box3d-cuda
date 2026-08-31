// SPDX-License-Identifier: MIT
#ifndef BOX3D_INTEGRATED_DUCK_V1_H
#define BOX3D_INTEGRATED_DUCK_V1_H
#include "articulated_v2.h"
#include "contact_v1.h"
#include "coupled_impulse_v1.h"
#ifdef __cplusplus
extern "C" {
#endif
enum { IDV1_OK=0,IDV1_INVALID=1,IDV1_ALLOCATION=2,IDV1_CONTACT=3,IDV1_ARTICULATED=4,IDV1_SOLVER=5,IDV1_TRANSACTION=6 };
typedef struct idv1_scene idv1_scene;
typedef struct idv1_snapshot idv1_snapshot;
typedef struct idv1_registration {
 const av2_registration* articulation;
 uint32_t pairs,reserved;
 const bcv1_shape* shapes; // [B], floor0 fixed, all massive links dynamic
 const bcv1_pair* contact_pairs; // [P], frozen explicit set
 const float* friction; // [E,P], copied, fixed per scene
} idv1_registration;
typedef struct idv1_diagnostic {
 uint32_t environment,phase,native_status,iterations,contact_points,active_limits;
 double joint_residual,normal_residual,tangent_residual,momentum_residual;
 double maximum_normal_impulse,maximum_penetration;
} idv1_diagnostic;
/* Stage phases: 1 PRE,2 geometry,3 solve,4 articulated complete,5 contact
 * prepare,6 two-owner validation/commit. On failure, published physics and
 * both clocks/caches remain unchanged; diagnostic describes first failure.
 * No bcv1_step, posthoc body impulses, pose projection or hidden substeps.
 * dt is one internal integration interval. Body outputs use f32 contact
 * storage; generalized/articulated states and solve are f64 throughout.
 */
int idv1_create(const idv1_registration*,idv1_scene**);
void idv1_destroy(idv1_scene*);
int idv1_step(idv1_scene*,const av2_step*,uint32_t max_iterations,
             double impulse_tolerance,idv1_diagnostic* /* [E] */);
// All outputs optional: copied host arrays, no resident-pointer exposure.
int idv1_read(idv1_scene*,double* qpos,double* velocity,double* joint_warm_force,
 double* clock,uint64_t* step_count,bcv1_body*,bcv1_manifold* pre_cache,
 bcv1_manifold* current_geometry);
int idv1_capture(idv1_scene*,idv1_snapshot**);
void idv1_snapshot_destroy(idv1_snapshot*);
// NULL selects all; any nonzero selects. Same scene identity required.
int idv1_restore(idv1_scene*,const idv1_snapshot*,const uint8_t* mask);
int idv1_reset(idv1_scene*,const uint8_t* mask);
// Correctly sized live host arrays/handles are caller obligations. A scene
// serializes operations internally; destruction cannot race any operation.
#ifdef __cplusplus
}
#endif
#endif
