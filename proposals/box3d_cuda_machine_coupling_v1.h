#ifndef BOX3D_CUDA_MACHINE_COUPLING_V1_H_
#define BOX3D_CUDA_MACHINE_COUPLING_V1_H_

/* Additive machine-coupling extension for the frozen ABI-v2 r3 scene.
 * This header does not modify the r3 descriptor or any existing symbol. */

#include "box3d_cuda_v2.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BOX3D_CUDA_CAP_V2_EXTERNAL_WRENCH_STEP (UINT64_C(1) << 18)
#define BOX3D_CUDA_CAP_V2_JOINT_VELOCITY_OUTPUT (UINT64_C(1) << 19)

/* The first 160 bytes are exactly box3d_cuda_scene_step_desc_v2. All tensor
 * pointers are caller-owned CUDA device pointers on the registered device.
 * external force/torque are tightly packed [E,B,3], environment-major,
 * body-major, xyz-fastest. NULL means an all-zero wrench. Values are held for
 * every requested control step and applied in every registered substep.
 * Forces are world-frame newtons at the COM; torques are world-frame N*m about
 * the COM. Fixed bodies ignore both. joint_velocity is optional [E,J] and
 * receives final signed generalized velocity: revolute rad/s about the
 * parent-local axis transformed to world; prismatic anchor-point relative
 * velocity in m/s along that axis; fixed joints write zero. The engine does
 * not retain any pointer after completion on the caller-owned stream. */
typedef struct box3d_cuda_scene_wrench_step_desc_v1 {
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
  const float* external_force_xyz;             /* optional [E,B,3] */
  const float* external_torque_xyz;            /* optional [E,B,3] */
  float* joint_velocity;                       /* optional [E,J] */
} box3d_cuda_scene_wrench_step_desc_v1;

BOX3D_CUDA_V2_API box3d_cuda_status_v2 box3d_cuda_scene_step_wrench_v1(
    const box3d_cuda_scene_wrench_step_desc_v1* descriptor);

#ifdef __cplusplus
}
static_assert(sizeof(box3d_cuda_scene_wrench_step_desc_v1) == 184u,
              "machine coupling descriptor size");
static_assert(offsetof(box3d_cuda_scene_wrench_step_desc_v1,
                       external_force_xyz) == 160u,
              "external force offset");
static_assert(offsetof(box3d_cuda_scene_wrench_step_desc_v1,
                       external_torque_xyz) == 168u,
              "external torque offset");
static_assert(offsetof(box3d_cuda_scene_wrench_step_desc_v1,
                       joint_velocity) == 176u,
              "joint velocity offset");
#else
_Static_assert(sizeof(box3d_cuda_scene_wrench_step_desc_v1) == 184u,
               "machine coupling descriptor size");
_Static_assert(offsetof(box3d_cuda_scene_wrench_step_desc_v1,
                        external_force_xyz) == 160u,
               "external force offset");
_Static_assert(offsetof(box3d_cuda_scene_wrench_step_desc_v1,
                        external_torque_xyz) == 168u,
               "external torque offset");
_Static_assert(offsetof(box3d_cuda_scene_wrench_step_desc_v1,
                        joint_velocity) == 176u,
               "joint velocity offset");
#endif

#endif
