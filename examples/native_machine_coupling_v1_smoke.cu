#include "../proposals/box3d_cuda_machine_coupling_v1.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

template <typename T>
T* device_copy(const std::vector<T>& values) {
  if (values.empty()) return nullptr;
  T* result = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&result), values.size() * sizeof(T)) !=
          cudaSuccess ||
      cudaMemcpy(result, values.data(), values.size() * sizeof(T),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    std::fprintf(stderr, "device allocation/copy failed\n");
    std::exit(90);
  }
  return result;
}

template <typename T>
T* device_allocate(size_t count) {
  if (count == 0) return nullptr;
  T* result = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&result), count * sizeof(T)) !=
      cudaSuccess) {
    std::fprintf(stderr, "device allocation failed\n");
    std::exit(91);
  }
  return result;
}

template <typename T>
std::vector<T> host_copy(const T* source, size_t count) {
  std::vector<T> result(count);
  if (count != 0 &&
      cudaMemcpy(result.data(), source, count * sizeof(T),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::fprintf(stderr, "device download failed\n");
    std::exit(92);
  }
  return result;
}

bool near(float left, float right, float tolerance = 1.0e-5f) {
  return std::fabs(left - right) <= tolerance;
}

void fill_common_registration(box3d_cuda_scene_register_desc_v2* registration) {
  registration->struct_size = sizeof(*registration);
  registration->abi_version = BOX3D_CUDA_ABI_VERSION_V2;
  registration->device_ordinal = -1;
  registration->environments = 1;
  registration->substeps = 1;
  registration->solver_iterations = 1;
  registration->material_binding = BOX3D_CUDA_MATERIAL_GLOBAL_V2;
  registration->dt = 0.25f;
  registration->global_friction = 0.0f;
  registration->global_restitution = 0.0f;
  registration->warm_start_factor = 0.0f;
  registration->contact_slop = 1.0e-4f;
  registration->position_correction = 0.8f;
  registration->angular_damping = 0.0f;
  registration->sat_epsilon = 1.0e-7f;
  registration->joint_position_slop = 1.0e-5f;
  registration->joint_angular_slop = 1.0e-5f;
  registration->maximum_linear_repair = 0.1f;
  registration->maximum_angular_repair = 0.2f;
}

}  // namespace

int main() {
  box3d_cuda_api_info_v2 api{};
  api.struct_size = sizeof(api);
  api.abi_version = BOX3D_CUDA_ABI_VERSION_V2;
  const uint64_t required = BOX3D_CUDA_CAP_V2_EXTERNAL_WRENCH_STEP |
                            BOX3D_CUDA_CAP_V2_JOINT_VELOCITY_OUTPUT;
  if (box3d_cuda_query_api_v2(&api) != BOX3D_CUDA_STATUS_V2_SUCCESS ||
      (api.capabilities & required) != required) {
    std::fprintf(stderr, "machine-coupling capabilities unavailable\n");
    return 1;
  }

  // Goldens A and B: one dynamic and one fixed identity-orientation body.
  const uint32_t body_ids[] = {10, 20};
  const uint32_t body_motion[] = {BOX3D_CUDA_BODY_DYNAMIC_V2,
                                  BOX3D_CUDA_BODY_FIXED_V2};
  std::vector<float> state(2 * 13, 0.0f);
  state[6] = 1.0f;
  state[13 + 6] = 1.0f;
  const std::vector<float> inverse_mass = {0.5f, 0.0f};
  const std::vector<float> inverse_inertia = {1.0f, 2.0f, 3.0f,
                                               0.0f, 0.0f, 0.0f};
  const std::vector<float> half_extents(2 * 3, 0.1f);
  box3d_cuda_scene_register_desc_v2 body_registration{};
  fill_common_registration(&body_registration);
  body_registration.bodies = 2;
  body_registration.body_caller_ids = body_ids;
  body_registration.body_motion = body_motion;
  body_registration.state = state.data();
  body_registration.inverse_mass = inverse_mass.data();
  body_registration.inverse_inertia = inverse_inertia.data();
  body_registration.half_extents = half_extents.data();
  box3d_cuda_scene_handle_v2 body_scene = 0;
  if (box3d_cuda_scene_register_v2(&body_registration, &body_scene) !=
      BOX3D_CUDA_STATUS_V2_SUCCESS) {
    std::fprintf(stderr, "body scene registration failed\n");
    return 2;
  }
  float* force = device_copy(std::vector<float>{2.0f, 4.0f, 6.0f,
                                                 2.0f, 4.0f, 6.0f});
  float* torque = device_copy(std::vector<float>{2.0f, 4.0f, 6.0f,
                                                  2.0f, 4.0f, 6.0f});
  box3d_cuda_scene_wrench_step_desc_v1 body_step{};
  body_step.struct_size = sizeof(body_step);
  body_step.abi_version = BOX3D_CUDA_ABI_VERSION_V2;
  body_step.scene = body_scene;
  body_step.steps = 1;
  body_step.external_force_xyz = force;
  body_step.external_torque_xyz = torque;
  if (box3d_cuda_scene_step_wrench_v1(&body_step) !=
      BOX3D_CUDA_STATUS_V2_SUCCESS) {
    std::fprintf(stderr, "body wrench step failed\n");
    return 3;
  }
  box3d_cuda_scene_info_v2 body_info{};
  body_info.struct_size = sizeof(body_info);
  body_info.abi_version = BOX3D_CUDA_ABI_VERSION_V2;
  box3d_cuda_scene_get_info_v2(body_scene, &body_info);
  float* captured_state = device_allocate<float>(2 * 13);
  float* captured_mass = device_allocate<float>(2);
  float* captured_inertia = device_allocate<float>(6);
  float* captured_extents = device_allocate<float>(6);
  float* captured_gravity = device_allocate<float>(3);
  float* captured_friction = device_allocate<float>(1);
  float* captured_restitution = device_allocate<float>(1);
  box3d_cuda_scene_capture_desc_v2 capture{};
  capture.struct_size = sizeof(capture);
  capture.abi_version = BOX3D_CUDA_ABI_VERSION_V2;
  capture.scene = body_scene;
  std::memcpy(capture.topology_sha256, body_info.topology_sha256, 32);
  capture.state = captured_state;
  capture.inverse_mass = captured_mass;
  capture.inverse_inertia = captured_inertia;
  capture.half_extents = captured_extents;
  capture.gravity_xyz = captured_gravity;
  capture.material_friction = captured_friction;
  capture.material_restitution = captured_restitution;
  if (box3d_cuda_scene_capture_v2(&capture) != BOX3D_CUDA_STATUS_V2_SUCCESS ||
      cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "body capture failed\n");
    return 4;
  }
  const auto observed = host_copy(captured_state, 2 * 13);
  const float expected_dynamic[] = {0.25f, 0.5f, 0.75f, 0.5f, 2.0f, 4.5f};
  for (int axis = 0; axis < 3; ++axis) {
    if (!near(observed[7 + axis], expected_dynamic[axis]) ||
        !near(observed[10 + axis], expected_dynamic[3 + axis]) ||
        !near(observed[13 + 7 + axis], 0.0f) ||
        !near(observed[13 + 10 + axis], 0.0f)) {
      std::fprintf(stderr, "wrench integration golden failed\n");
      return 5;
    }
  }
  box3d_cuda_scene_unregister_v2(body_scene);

  // Goldens C and D: all-fixed bodies retain the authored velocity field, so
  // the call tests signed generalized-velocity gathering independently.
  const uint32_t joint_body_ids[] = {100, 101, 102, 103};
  const uint32_t fixed_motion[] = {0, 0, 0, 0};
  const uint32_t joint_ids[] = {200, 201};
  const uint32_t joint_bodies[] = {0, 1, 2, 3};
  const uint32_t joint_types[] = {BOX3D_CUDA_JOINT_REVOLUTE_V2,
                                  BOX3D_CUDA_JOINT_PRISMATIC_V2};
  const float root = std::sqrt(0.5f);
  std::vector<float> joint_state(4 * 13, 0.0f);
  for (int body = 0; body < 4; ++body) joint_state[body * 13 + 6] = 1.0f;
  for (int body = 0; body < 2; ++body) {
    joint_state[body * 13 + 5] = root;
    joint_state[body * 13 + 6] = root;
  }
  joint_state[10 + 1] = 1.0f;
  joint_state[13 + 10 + 1] = 4.0f;
  joint_state[3 * 13 + 0] = 2.0f;
  joint_state[3 * 13 + 1] = 2.0f;
  joint_state[2 * 13 + 10 + 2] = 1.0f;
  joint_state[3 * 13 + 7] = 1.0f;
  joint_state[3 * 13 + 10 + 2] = 2.0f;
  const float parent_anchor[] = {0, 0, 0, 0, 1, 0};
  const float child_anchor[] = {0, 0, 0, 0, -1, 0};
  const float axes[] = {1, 0, 0, 1, 0, 0};
  const float references[] = {0, 0, 0, 1, 0, 0, 0, 1};
  const float lower[] = {-10.0f, -10.0f};
  const float upper[] = {10.0f, 10.0f};
  const float zeros2[] = {0.0f, 0.0f};
  const uint32_t disabled[] = {0, 0};
  const std::vector<float> zero_mass(4, 0.0f);
  const std::vector<float> zero_inertia(12, 0.0f);
  const std::vector<float> joint_extents(12, 0.1f);
  box3d_cuda_scene_register_desc_v2 joint_registration{};
  fill_common_registration(&joint_registration);
  joint_registration.bodies = 4;
  joint_registration.joints = 2;
  joint_registration.body_caller_ids = joint_body_ids;
  joint_registration.body_motion = fixed_motion;
  joint_registration.joint_caller_ids = joint_ids;
  joint_registration.joint_body_indices = joint_bodies;
  joint_registration.joint_types = joint_types;
  joint_registration.joint_parent_anchor = parent_anchor;
  joint_registration.joint_child_anchor = child_anchor;
  joint_registration.joint_axis_parent = axes;
  joint_registration.joint_reference_xyzw = references;
  joint_registration.joint_lower_limit = lower;
  joint_registration.joint_upper_limit = upper;
  joint_registration.joint_damping = zeros2;
  joint_registration.joint_stiffness = zeros2;
  joint_registration.joint_control_mode = disabled;
  joint_registration.state = joint_state.data();
  joint_registration.inverse_mass = zero_mass.data();
  joint_registration.inverse_inertia = zero_inertia.data();
  joint_registration.half_extents = joint_extents.data();
  box3d_cuda_scene_handle_v2 joint_scene = 0;
  if (box3d_cuda_scene_register_v2(&joint_registration, &joint_scene) !=
      BOX3D_CUDA_STATUS_V2_SUCCESS) {
    std::fprintf(stderr, "joint scene registration failed\n");
    return 6;
  }
  float* controls = device_copy(std::vector<float>{0.0f, 0.0f});
  float* coordinate = device_allocate<float>(2);
  float* velocity = device_allocate<float>(2);
  box3d_cuda_scene_wrench_step_desc_v1 joint_step{};
  joint_step.struct_size = sizeof(joint_step);
  joint_step.abi_version = BOX3D_CUDA_ABI_VERSION_V2;
  joint_step.scene = joint_scene;
  joint_step.steps = 1;
  joint_step.target_position = controls;
  joint_step.target_velocity = controls;
  joint_step.maximum_effort = controls;
  joint_step.maximum_speed = controls;
  joint_step.maximum_acceleration = controls;
  joint_step.joint_coordinate = coordinate;
  joint_step.joint_velocity = velocity;
  if (box3d_cuda_scene_step_wrench_v1(&joint_step) !=
          BOX3D_CUDA_STATUS_V2_SUCCESS ||
      cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "joint velocity step failed\n");
    return 7;
  }
  const auto q = host_copy(coordinate, 2);
  const auto qdot = host_copy(velocity, 2);
  if (!near(q[0], 0.0f) || !near(qdot[0], 3.0f) ||
      !near(q[1], 2.0f) || !near(qdot[1], 4.0f)) {
    std::fprintf(stderr,
                 "joint velocity golden failed: q=(%g,%g) qdot=(%g,%g)\n",
                 q[0], q[1], qdot[0], qdot[1]);
    return 8;
  }
  box3d_cuda_scene_unregister_v2(joint_scene);
  std::printf("machine coupling v1 goldens passed: mask=0x%llx\n",
              static_cast<unsigned long long>(api.capabilities));
  return 0;
}
