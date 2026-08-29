#ifndef BOX3D_CUDA_V2_PROPOSAL_H_
#define BOX3D_CUDA_V2_PROPOSAL_H_

/* ABI-v2 draft revision 3. Descriptor layouts and semantics are frozen with
 * the World consumer. The resident lifecycle is implemented; step and raycast
 * remain capability-gated and return UNSUPPORTED. ABI v1 remains stable. */

#include <stddef.h>
#include <stdint.h>

#if UINTPTR_MAX != UINT64_MAX
#  error "The Box3D CUDA ABI-v2 proposal requires a 64-bit process"
#endif

#if defined(_WIN32)
#  if defined(BOX3D_CUDA_BUILD_SHARED)
#    define BOX3D_CUDA_V2_API __declspec(dllexport)
#  else
#    define BOX3D_CUDA_V2_API __declspec(dllimport)
#  endif
#else
#  define BOX3D_CUDA_V2_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define BOX3D_CUDA_ABI_V2_DRAFT_REVISION 3u
#define BOX3D_CUDA_ABI_VERSION_V2 ((2u << 16u) | 0u)
#define BOX3D_CUDA_STATE_WIDTH_V2 13u
#define BOX3D_CUDA_JOINT_CACHE_WIDTH_V2 8u
#define BOX3D_CUDA_MANIFOLD_POINTS_V2 4u
#define BOX3D_CUDA_CONTACT_IMPULSE_WIDTH_V2 3u
#define BOX3D_CUDA_TOPOLOGY_HASH_BYTES_V2 32u
#define BOX3D_CUDA_MAX_BODIES_V2 32u
#define BOX3D_CUDA_MAX_JOINTS_V2 16u
#define BOX3D_CUDA_MAX_CONTACT_PAIRS_V2 16u

typedef uint64_t box3d_cuda_scene_handle_v2;

typedef enum box3d_cuda_status_v2 {
  BOX3D_CUDA_STATUS_V2_SUCCESS = 0,
  BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT = 1,
  BOX3D_CUDA_STATUS_V2_ABI_MISMATCH = 2,
  BOX3D_CUDA_STATUS_V2_CUDA_ERROR = 3,
  BOX3D_CUDA_STATUS_V2_INVALID_HANDLE = 4,
  BOX3D_CUDA_STATUS_V2_UNSUPPORTED = 5,
  BOX3D_CUDA_STATUS_V2_LIMIT_EXCEEDED = 6,
  BOX3D_CUDA_STATUS_V2_ALLOCATION_FAILED = 7,
  BOX3D_CUDA_STATUS_V2_TOPOLOGY_MISMATCH = 8,
  BOX3D_CUDA_STATUS_V2_BUSY = 9
} box3d_cuda_status_v2;

typedef enum box3d_cuda_capability_v2 {
  BOX3D_CUDA_CAP_V2_ORIENTED_BOXES = UINT64_C(1) << 0,
  BOX3D_CUDA_CAP_V2_EXPLICIT_CONTACT_PAIRS = UINT64_C(1) << 1,
  BOX3D_CUDA_CAP_V2_FIXED_JOINTS = UINT64_C(1) << 2,
  BOX3D_CUDA_CAP_V2_REVOLUTE_JOINTS = UINT64_C(1) << 3,
  BOX3D_CUDA_CAP_V2_PRISMATIC_JOINTS = UINT64_C(1) << 4,
  BOX3D_CUDA_CAP_V2_PERSISTENT_CONTACTS = UINT64_C(1) << 5,
  BOX3D_CUDA_CAP_V2_RESIDENT_STATE = UINT64_C(1) << 6,
  BOX3D_CUDA_CAP_V2_DETERMINISTIC_SNAPSHOT = UINT64_C(1) << 7,
  BOX3D_CUDA_CAP_V2_LINEAR_OBB_RAYS = UINT64_C(1) << 8,
  BOX3D_CUDA_CAP_V2_ASYNC_CALLER_STREAM = UINT64_C(1) << 9,
  BOX3D_CUDA_CAP_V2_GLOBAL_MATERIAL = UINT64_C(1) << 10,
  BOX3D_CUDA_CAP_V2_PER_BODY_MATERIAL = UINT64_C(1) << 11,
  BOX3D_CUDA_CAP_V2_PER_PAIR_MATERIAL = UINT64_C(1) << 12,
  BOX3D_CUDA_CAP_V2_KINEMATIC_BODIES = UINT64_C(1) << 13,
  BOX3D_CUDA_CAP_V2_PER_ENVIRONMENT_GRAVITY = UINT64_C(1) << 14,
  BOX3D_CUDA_CAP_V2_ACTUATOR_SPEED_LIMIT = UINT64_C(1) << 15,
  BOX3D_CUDA_CAP_V2_ACTUATOR_ACCELERATION_LIMIT = UINT64_C(1) << 16,
  BOX3D_CUDA_CAP_V2_PARTIAL_ENVIRONMENT_RESTORE = UINT64_C(1) << 17
} box3d_cuda_capability_v2;

typedef enum box3d_cuda_body_motion_v2 {
  BOX3D_CUDA_BODY_FIXED_V2 = 0,
  BOX3D_CUDA_BODY_KINEMATIC_V2 = 1,
  BOX3D_CUDA_BODY_DYNAMIC_V2 = 2
} box3d_cuda_body_motion_v2;

typedef enum box3d_cuda_joint_type_v2 {
  BOX3D_CUDA_JOINT_FIXED_V2 = 0,
  BOX3D_CUDA_JOINT_REVOLUTE_V2 = 1,
  BOX3D_CUDA_JOINT_PRISMATIC_V2 = 2
} box3d_cuda_joint_type_v2;

typedef enum box3d_cuda_control_mode_v2 {
  BOX3D_CUDA_CONTROL_DISABLED_V2 = 0,
  BOX3D_CUDA_CONTROL_POSITION_V2 = 1,
  BOX3D_CUDA_CONTROL_VELOCITY_V2 = 2
} box3d_cuda_control_mode_v2;

typedef enum box3d_cuda_material_binding_v2 {
  BOX3D_CUDA_MATERIAL_GLOBAL_V2 = 0,
  BOX3D_CUDA_MATERIAL_PER_BODY_V2 = 1,
  BOX3D_CUDA_MATERIAL_PER_CONTACT_PAIR_V2 = 2
} box3d_cuda_material_binding_v2;

typedef struct box3d_cuda_api_info_v2 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t implementation_version_major;
  uint32_t implementation_version_minor;
  uint32_t implementation_version_patch;
  uint32_t draft_revision;
  uint64_t capabilities;
  uint32_t maximum_bodies;
  uint32_t maximum_joints;
  uint32_t maximum_contact_pairs;
  uint32_t manifold_points;
  uint32_t joint_cache_width;
  uint32_t reserved_u32[3];
  char implementation_id[64];
} box3d_cuda_api_info_v2;

/* Registration is synchronous. Every pointer below is a naturally aligned
 * CPU pointer. The implementation copies all values before returning and
 * never aliases caller storage. Topology is immutable for the handle's life.
 * Empty joint/pair arrays may be NULL; all other selected-layout arrays are
 * required. All multidimensional arrays are tightly packed C row-major. */
typedef struct box3d_cuda_scene_register_desc_v2 {
  uint32_t struct_size;
  uint32_t abi_version;
  int32_t device_ordinal;
  uint32_t flags;                              /* must be zero */
  uint32_t environments;
  uint32_t bodies;
  uint32_t joints;
  uint32_t contact_pairs;
  uint32_t substeps;
  uint32_t solver_iterations;
  uint32_t material_binding;
  uint32_t reserved_u32;                       /* must be zero */
  float dt;
  float global_gravity_xyz[3];
  float global_friction;
  float global_restitution;
  float warm_start_factor;
  float contact_slop;
  float position_correction;
  float angular_damping;
  float sat_epsilon;
  float joint_position_slop;
  float joint_angular_slop;
  float maximum_linear_repair;
  float maximum_angular_repair;
  const uint32_t* body_caller_ids;             /* [B] */
  const uint32_t* body_motion;                 /* [B], box3d_cuda_body_motion_v2 */
  const uint32_t* joint_caller_ids;            /* [J] */
  const uint32_t* joint_body_indices;          /* [J,2] dense body indices */
  const uint32_t* joint_types;                 /* [J] */
  const float* joint_parent_anchor;            /* [J,3], parent-local metres */
  const float* joint_child_anchor;             /* [J,3], child-local metres */
  const float* joint_axis_parent;              /* [J,3], normalized parent-local */
  const float* joint_reference_xyzw;           /* [J,4], parent^-1 * child */
  const float* joint_lower_limit;              /* [J], rad or m */
  const float* joint_upper_limit;              /* [J], rad or m */
  const float* joint_damping;                  /* [J] */
  const float* joint_stiffness;                /* [J] */
  const uint32_t* joint_control_mode;          /* [J]; fixed joints require DISABLED */
  const uint32_t* contact_pair_caller_ids;     /* [P] */
  const uint32_t* contact_body_indices;        /* [P,2] dense body indices */
  const float* state;                          /* [E,B,13] */
  const float* inverse_mass;                   /* [E,B] */
  const float* inverse_inertia;                /* [E,B,3], body-local */
  const float* half_extents;                   /* [E,B,3] */
  const float* environment_gravity_xyz;        /* optional [E,3] */
  const float* body_friction;                  /* per-body only: [E,B] */
  const float* body_restitution;               /* per-body only: [E,B] */
  const float* pair_friction;                  /* per-pair only: [E,P] */
  const float* pair_restitution;               /* per-pair only: [E,P] */
  const float* joint_cache;                    /* optional [E,J,8], zero if NULL */
  const int64_t* contact_feature_ids;          /* optional [E,P,4], -1 if NULL */
  const float* contact_impulse_cache;          /* optional [E,P,4,3], zero if NULL */
} box3d_cuda_scene_register_desc_v2;

/* topology_sha256 is computed by the engine after validation. The canonical
 * stream and its golden vectors are specified in
 * docs/native-c-abi-v2-proposal.md and topology_digest.py. Mutable episode
 * values are excluded; immutable topology, layout, and solver semantics are
 * included. The representation is 32 digest bytes, never host-endian words. */
typedef struct box3d_cuda_scene_info_v2 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t environments;
  uint32_t bodies;
  uint32_t joints;
  uint32_t contact_pairs;
  uint8_t topology_sha256[BOX3D_CUDA_TOPOLOGY_HASH_BYTES_V2];
  uint64_t reserved_u64[4];                    /* zero on output */
} box3d_cuda_scene_info_v2;

/* Step input/output pointers are CUDA device pointers. NULL diagnostic output
 * pointers mean "not requested". The [E,J] target and limit frame is held for
 * every requested control `steps`; registered `substeps` are internal
 * integration subdivisions. A joint has at most one actuator. Disabled joints
 * ignore targets and limits. Output arrays contain final state diagnostics
 * except motor/contact impulses, which are summed over all requested control
 * steps. `contact_active` is the final control-step state; `contact_ever` is
 * true if the pair was active during any requested step. The call is
 * asynchronous on `stream` (cudaStream_t represented as void*). SUCCESS means
 * argument validation and enqueue succeeded; execution errors may surface only
 * when the caller synchronizes that stream. */
typedef struct box3d_cuda_scene_step_desc_v2 {
  uint32_t struct_size;
  uint32_t abi_version;
  box3d_cuda_scene_handle_v2 scene;
  uint32_t steps;
  uint32_t reserved_u32;                       /* must be zero */
  const float* target_position;                /* [E,J] */
  const float* target_velocity;                /* [E,J] */
  const float* maximum_effort;                 /* [E,J] */
  const float* maximum_speed;                  /* [E,J] */
  const float* maximum_acceleration;           /* [E,J] */
  float* joint_coordinate;                     /* optional [E,J] */
  float* joint_anchor_error;                   /* optional [E,J] */
  float* joint_angular_error;                  /* optional [E,J] */
  float* joint_limit_error;                    /* optional [E,J] */
  float* joint_motor_impulse;                  /* optional [E,J] */
  uint8_t* joint_limit_active;                 /* optional [E,J] */
  uint8_t* contact_active;                     /* optional [E,P] */
  uint8_t* contact_ever;                       /* optional [E,P] */
  uint32_t* contact_count;                     /* optional [E,P], final */
  float* contact_penetration;                  /* optional [E,P] */
  float* contact_normal_impulse;               /* optional [E,P] */
  void* stream;
} box3d_cuda_scene_step_desc_v2;

/* Snapshot buffers are caller-owned CUDA device pointers. Capture and restore
 * are asynchronous with the same enqueue/error rule as step. The supplied
 * SHA-256 must equal scene_info.topology_sha256 or the call fails closed.
 * Material arrays use the registered binding: global [1], per-body [E,B], or
 * per-contact-pair [E,P]. gravity_xyz is always required and uses the selected
 * layout: global [3] or per-environment [E,3]. Every other non-empty array is
 * required. Mutable episode state is intentionally complete so capture/restore
 * is a deterministic RL reset primitive. Restore's optional environment_mask
 * is a tightly packed [E] CUDA-device uint8 array; NULL selects all
 * environments and any nonzero byte selects that environment. A non-NULL mask
 * requires PARTIAL_ENVIRONMENT_RESTORE. The caller owns every input through
 * stream completion. Per-environment arrays and caches update only for selected
 * environments. Global-layout gravity/material values are validated and copied
 * exactly once per call regardless of the mask, including an all-zero mask. */
typedef struct box3d_cuda_scene_capture_desc_v2 {
  uint32_t struct_size;
  uint32_t abi_version;
  box3d_cuda_scene_handle_v2 scene;
  uint8_t topology_sha256[BOX3D_CUDA_TOPOLOGY_HASH_BYTES_V2];
  float* state;                                /* [E,B,13] */
  float* inverse_mass;                         /* [E,B] */
  float* inverse_inertia;                      /* [E,B,3] */
  float* half_extents;                         /* [E,B,3] */
  float* gravity_xyz;                          /* global [3] or per-env [E,3] */
  float* material_friction;                    /* selected layout */
  float* material_restitution;                 /* selected layout */
  float* joint_cache;                          /* [E,J,8] */
  int64_t* contact_feature_ids;                /* [E,P,4] */
  float* contact_impulse_cache;                /* [E,P,4,3] */
  void* stream;
} box3d_cuda_scene_capture_desc_v2;

typedef struct box3d_cuda_scene_restore_desc_v2 {
  uint32_t struct_size;
  uint32_t abi_version;
  box3d_cuda_scene_handle_v2 scene;
  uint8_t topology_sha256[BOX3D_CUDA_TOPOLOGY_HASH_BYTES_V2];
  const float* state;                          /* [E,B,13] */
  const float* inverse_mass;                   /* [E,B] */
  const float* inverse_inertia;                /* [E,B,3] */
  const float* half_extents;                   /* [E,B,3] */
  const float* gravity_xyz;                    /* global [3] or per-env [E,3] */
  const float* material_friction;              /* selected layout */
  const float* material_restitution;           /* selected layout */
  const float* joint_cache;                    /* [E,J,8] */
  const int64_t* contact_feature_ids;          /* [E,P,4] */
  const float* contact_impulse_cache;          /* [E,P,4,3] */
  const uint8_t* environment_mask;             /* optional device [E], NULL=all */
  void* stream;
} box3d_cuda_scene_restore_desc_v2;

/* Linear-scan OBB rays. All query/output pointers are CUDA device pointers.
 * body_index is -1 on miss and otherwise a dense registered body index. */
typedef struct box3d_cuda_ray_query_desc_v2 {
  uint32_t struct_size;
  uint32_t abi_version;
  box3d_cuda_scene_handle_v2 scene;
  uint32_t rays_per_environment;
  uint32_t reserved_u32;                       /* must be zero */
  const float* origins;                        /* [E,R,3] */
  const float* directions;                     /* [E,R,3], normalized */
  const float* maximum_distance;               /* [E,R] */
  float* hit_distance;                         /* [E,R] */
  int32_t* hit_body_index;                     /* [E,R] */
  float* hit_normal;                           /* [E,R,3] */
  void* stream;
} box3d_cuda_ray_query_desc_v2;

/* Exported r3 symbols. Unsupported operations remain present and fail closed. */
BOX3D_CUDA_V2_API uint32_t box3d_cuda_get_abi_version_v2(void);
BOX3D_CUDA_V2_API box3d_cuda_status_v2 box3d_cuda_query_api_v2(
    box3d_cuda_api_info_v2* info);
BOX3D_CUDA_V2_API const char* box3d_cuda_status_string_v2(
    box3d_cuda_status_v2 status);
BOX3D_CUDA_V2_API box3d_cuda_status_v2 box3d_cuda_scene_register_v2(
    const box3d_cuda_scene_register_desc_v2* descriptor,
    box3d_cuda_scene_handle_v2* scene);
BOX3D_CUDA_V2_API box3d_cuda_status_v2 box3d_cuda_scene_get_info_v2(
    box3d_cuda_scene_handle_v2 scene, box3d_cuda_scene_info_v2* info);
BOX3D_CUDA_V2_API box3d_cuda_status_v2 box3d_cuda_scene_step_v2(
    const box3d_cuda_scene_step_desc_v2* descriptor);
BOX3D_CUDA_V2_API box3d_cuda_status_v2 box3d_cuda_scene_capture_v2(
    const box3d_cuda_scene_capture_desc_v2* descriptor);
BOX3D_CUDA_V2_API box3d_cuda_status_v2 box3d_cuda_scene_restore_v2(
    const box3d_cuda_scene_restore_desc_v2* descriptor);
BOX3D_CUDA_V2_API box3d_cuda_status_v2 box3d_cuda_scene_raycast_v2(
    const box3d_cuda_ray_query_desc_v2* descriptor);
BOX3D_CUDA_V2_API box3d_cuda_status_v2 box3d_cuda_scene_unregister_v2(
    box3d_cuda_scene_handle_v2 scene);

#ifdef __cplusplus
}
#endif

#endif
