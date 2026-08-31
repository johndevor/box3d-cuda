// SPDX-License-Identifier: MIT
#ifndef BOX3D_EXPERIMENTAL_CONTACT_V1_H
#define BOX3D_EXPERIMENTAL_CONTACT_V1_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Experimental CPU-native ABI 1; not r3, no CUDA capability claims.
 * All arrays are packed host arrays, copied synchronously. No borrowed state.
 * Caller supplies valid live pointers and full stated array capacities;
 * writable outputs must not overlap each other or live scene storage.
 * Body state: COM/principal p xyz, active local-to-world q xyzw, world v, omega.
 * Units m, kg, s. Convex vertices already in principal-COM body frame.
 * Plane solid half-space n.dot(x)<=offset in its body's local frame.
 * Plane normal must satisfy abs(length(n)-1)<2e-5; accepted n AND offset
 * are divided by length(n) together. Arbitrarily scaled normals reject.
 * All handles must remain live; concurrent calls on a handle are unsupported.
 */
enum { BCV1_VERSION=1, BCV1_NONE=0, BCV1_CONVEX=1, BCV1_PLANE=2,
       BCV1_MAX_VERTICES=32, BCV1_MAX_BODIES=32, BCV1_MAX_PAIRS=16,
       BCV1_MAX_ENVIRONMENTS=4096, BCV1_MAX_POINTS=4 };
typedef enum { BCV1_OK=0, BCV1_INVALID=1, BCV1_CAPACITY=2,
               BCV1_NUMERIC=3, BCV1_TOPOLOGY=4, BCV1_INTERNAL=5 } bcv1_status;
typedef struct { float state[13], inverse_mass, inverse_inertia[3]; } bcv1_body;
typedef struct {
  uint32_t caller_id, kind, vertex_count, fixed;
  float vertices[32][3], plane_normal[3], plane_offset;
} bcv1_shape;
typedef struct { uint32_t caller_id, body_a, body_b; } bcv1_pair;
typedef struct {
  uint64_t feature;
  float point[3], depth, normal_impulse, tangent_impulse[2];
} bcv1_point;
typedef struct {
  uint32_t count;
  float normal[3], tangent1[3], tangent2[3];
  bcv1_point points[4];
} bcv1_manifold;
typedef struct {
  uint32_t version, environments, bodies, pairs;
  const bcv1_shape* shapes; /* [B], topology copied */
  const bcv1_pair* contact_pairs; /* [P], no duplicate/self/plane-plane */
  const bcv1_body* initial; /* [E,B] */
  const float* gravity_xyz; /* [E,3] */
  const float* pair_friction; /* [E,P], explicitly authored effective mu */
} bcv1_registration;
typedef struct bcv1_scene bcv1_scene;
typedef struct bcv1_snapshot bcv1_snapshot;
bcv1_status bcv1_create(const bcv1_registration*, bcv1_scene**);
void bcv1_destroy(bcv1_scene*);
/* Optional outputs; read manifolds are the last PRE-integration solve cache
 * while read bodies are POST-integration. query returns current-pose geometry
 * with zero impulses. Accepted near-unit quaternions are normalized for math,
 * without rewriting the caller's stored input during registration. */
bcv1_status bcv1_read(const bcv1_scene*, bcv1_body*, bcv1_manifold*, double* clocks);
bcv1_status bcv1_query(const bcv1_scene*, bcv1_manifold* output /* [E,P] */);
/* Inelastic PGS free-body reference. dt in (0,.01], iterations1..128.
 * Solve at current poses then integrate once; no CCD or position projection.
 * Depth bias=0.2*max(depth-2e-6,0)/dt, capped1m/s; warm cache guarded by
 * feature/basis. Whole operation rolls back on numeric/validation failure.
 * NOT the generalized armature-aware articulated response operator.
 */
bcv1_status bcv1_step(bcv1_scene*, float dt, uint32_t iterations);
bcv1_status bcv1_capture(const bcv1_scene*, bcv1_snapshot**);
void bcv1_snapshot_destroy(bcv1_snapshot*);
/* NULL mask=all; any nonzero byte selects. Full snapshot payload validated
 * before mutation. All selected body parameters/state, gravity, mu, contact
 * caches and clocks restored; unselected environments bit-exact. */
bcv1_status bcv1_restore(bcv1_scene*, const bcv1_snapshot*, const uint8_t* mask);
/* source state velocity is WORLD velocity of authored frame origin, not COM
 * and not a simulator's joint/generalized velocity. q_pc maps principal to
 * authored axes (xyzw). v_com=v_origin+omega cross R_source*com. */
bcv1_status bcv1_to_principal(const float source[13], const float com_local[3],
                             const float q_pc[4], float output[13]);
bcv1_status bcv1_from_principal(const float principal[13], const float com_local[3],
                               const float q_pc[4], float output[13]);
/* Geometry utilities only; no joint constraint solver. Revolute q uses
 * rotvec(conj(parent.q)*child.q*conj(reference)) dot normalized parent axis;
 * qdot=world_axis dot(omega_child-omega_parent). output=[q,qdot,anchor_error_xyz]. */
bcv1_status bcv1_joint_geometry(const float parent[13], const float child[13],
 const float parent_anchor[3], const float child_anchor[3], const float axis[3],
 const float reference_xyzw[4], float output[5]);
bcv1_status bcv1_bake_convex(const float local_pose[7], const float* vertices_xyz,
 uint32_t count, float* output_xyz);
bcv1_status bcv1_support(const bcv1_shape*, const float state[13],
 const float direction[3], float point_value[4], uint32_t* vertex_index);
/* Stateless row adapter. spatialJ[B,6,N] is WORLD vx,vy,vz,wx,wy,wz.
 * outG[N] maps generalized velocities to d.dot(vpointB-vpointA).
 * Joint lane solves full(M_body+armature)^-1 outG and owns propagation.
 */
bcv1_status bcv1_contact_row(uint32_t bodies, uint32_t dofs, const bcv1_body*,
 uint32_t body_a, uint32_t body_b, const float world_point[3],
 const float direction[3], const float* spatial_jacobian, float* out_g);
/* Geometric row extraction is through bcv1_query: signed point wrenches for
 * row direction d are A[-d,-(point-pA)cross d], B[+d,+(point-pB)cross d].
 * Joint-aware integration MUST use coupled generalized inverse inertia and
 * propagate row impulses to ALL body velocities. Not provided by bcv1_step.
 */
#ifdef __cplusplus
}
#endif
#endif
