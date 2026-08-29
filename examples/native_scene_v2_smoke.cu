#include "../proposals/box3d_cuda_v2.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
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

struct Snapshot {
  float* state = nullptr;
  float* inverse_mass = nullptr;
  float* inverse_inertia = nullptr;
  float* half_extents = nullptr;
  float* gravity = nullptr;
  float* friction = nullptr;
  float* restitution = nullptr;
  float* joint_cache = nullptr;
  int64_t* feature_ids = nullptr;
  float* impulses = nullptr;

  void release() {
    if (state != nullptr) cudaFree(state);
    if (inverse_mass != nullptr) cudaFree(inverse_mass);
    if (inverse_inertia != nullptr) cudaFree(inverse_inertia);
    if (half_extents != nullptr) cudaFree(half_extents);
    if (gravity != nullptr) cudaFree(gravity);
    if (friction != nullptr) cudaFree(friction);
    if (restitution != nullptr) cudaFree(restitution);
    if (joint_cache != nullptr) cudaFree(joint_cache);
    if (feature_ids != nullptr) cudaFree(feature_ids);
    if (impulses != nullptr) cudaFree(impulses);
  }
};

Snapshot allocate_snapshot(size_t eb, size_t ej, size_t ep) {
  Snapshot result;
  result.state = device_allocate<float>(eb * 13);
  result.inverse_mass = device_allocate<float>(eb);
  result.inverse_inertia = device_allocate<float>(eb * 3);
  result.half_extents = device_allocate<float>(eb * 3);
  result.gravity = device_allocate<float>(3);
  result.friction = device_allocate<float>(1);
  result.restitution = device_allocate<float>(1);
  result.joint_cache = device_allocate<float>(ej * 8);
  result.feature_ids = device_allocate<int64_t>(ep * 4);
  result.impulses = device_allocate<float>(ep * 12);
  return result;
}

bool near(float first, float second) {
  return std::fabs(first - second) <= 1.0e-6f;
}

}  // namespace

int main() {
  box3d_cuda_api_info_v2 api{};
  api.struct_size = sizeof(api);
  api.abi_version = BOX3D_CUDA_ABI_VERSION_V2;
  if (box3d_cuda_get_abi_version_v2() != BOX3D_CUDA_ABI_VERSION_V2 ||
      box3d_cuda_query_api_v2(&api) != BOX3D_CUDA_STATUS_V2_SUCCESS ||
      api.draft_revision != 3 ||
      (api.capabilities & BOX3D_CUDA_CAP_V2_PARTIAL_ENVIRONMENT_RESTORE) == 0 ||
      (api.capabilities & BOX3D_CUDA_CAP_V2_ORIENTED_BOXES) == 0 ||
      (api.capabilities & BOX3D_CUDA_CAP_V2_EXPLICIT_CONTACT_PAIRS) == 0) {
    std::fprintf(stderr, "ABI-v2 lifecycle capability discovery failed\n");
    return 1;
  }

  constexpr uint32_t environments = 4;
  constexpr uint32_t bodies = 2;
  constexpr uint32_t pairs = 1;
  constexpr size_t eb = environments * bodies;
  constexpr size_t ep = environments * pairs;
  const uint32_t body_ids[] = {10, 20};
  const uint32_t body_motion[] = {BOX3D_CUDA_BODY_FIXED_V2,
                                  BOX3D_CUDA_BODY_DYNAMIC_V2};
  const uint32_t pair_ids[] = {30};
  const uint32_t pair_bodies[] = {0, 1};
  std::vector<float> state(eb * 13, 0.0f);
  std::vector<float> inverse_mass(eb, 0.0f);
  std::vector<float> inverse_inertia(eb * 3, 0.0f);
  std::vector<float> half_extents(eb * 3, 0.5f);
  std::vector<int64_t> feature_ids(ep * 4, -1);
  std::vector<float> impulses(ep * 12, 0.0f);
  for (size_t environment = 0; environment < environments; ++environment) {
    for (size_t body = 0; body < bodies; ++body) {
      const size_t flat = environment * bodies + body;
      state[flat * 13 + 0] = static_cast<float>(environment * 10 + body);
      state[flat * 13 + 6] = 1.0f;
    }
    const size_t dynamic = environment * bodies + 1;
    inverse_mass[dynamic] = 0.5f;
    inverse_inertia[dynamic * 3 + 0] = 0.25f;
    inverse_inertia[dynamic * 3 + 1] = 0.25f;
    inverse_inertia[dynamic * 3 + 2] = 0.25f;
    feature_ids[environment * 4] = static_cast<int64_t>(100 + environment);
    impulses[environment * 12] = static_cast<float>(environment + 1);
  }

  box3d_cuda_scene_register_desc_v2 registration{};
  registration.struct_size = sizeof(registration);
  registration.abi_version = BOX3D_CUDA_ABI_VERSION_V2;
  registration.device_ordinal = -1;
  registration.environments = environments;
  registration.bodies = bodies;
  registration.contact_pairs = pairs;
  registration.substeps = 2;
  registration.solver_iterations = 12;
  registration.material_binding = BOX3D_CUDA_MATERIAL_GLOBAL_V2;
  registration.dt = 1.0f / 60.0f;
  registration.global_gravity_xyz[1] = -9.81f;
  registration.global_friction = 0.6f;
  registration.global_restitution = 0.1f;
  registration.warm_start_factor = 0.8f;
  registration.contact_slop = 1.0e-4f;
  registration.position_correction = 0.8f;
  registration.angular_damping = 0.02f;
  registration.sat_epsilon = 1.0e-7f;
  registration.joint_position_slop = 1.0e-5f;
  registration.joint_angular_slop = 1.0e-5f;
  registration.maximum_linear_repair = 0.1f;
  registration.maximum_angular_repair = 0.2f;
  registration.body_caller_ids = body_ids;
  registration.body_motion = body_motion;
  registration.contact_pair_caller_ids = pair_ids;
  registration.contact_body_indices = pair_bodies;
  registration.state = state.data();
  registration.inverse_mass = inverse_mass.data();
  registration.inverse_inertia = inverse_inertia.data();
  registration.half_extents = half_extents.data();
  registration.contact_feature_ids = feature_ids.data();
  registration.contact_impulse_cache = impulses.data();

  box3d_cuda_scene_handle_v2 scene = 0;
  const auto register_status = box3d_cuda_scene_register_v2(&registration, &scene);
  if (register_status != BOX3D_CUDA_STATUS_V2_SUCCESS || scene == 0) {
    std::fprintf(stderr, "scene registration failed: %s\n",
                 box3d_cuda_status_string_v2(register_status));
    return 2;
  }
  box3d_cuda_scene_info_v2 info{};
  info.struct_size = sizeof(info);
  info.abi_version = BOX3D_CUDA_ABI_VERSION_V2;
  if (box3d_cuda_scene_get_info_v2(scene, &info) !=
          BOX3D_CUDA_STATUS_V2_SUCCESS ||
      info.environments != environments || info.bodies != bodies ||
      info.contact_pairs != pairs) {
    std::fprintf(stderr, "scene info failed\n");
    return 3;
  }

  Snapshot captured = allocate_snapshot(eb, 0, ep);
  box3d_cuda_scene_capture_desc_v2 capture{};
  capture.struct_size = sizeof(capture);
  capture.abi_version = BOX3D_CUDA_ABI_VERSION_V2;
  capture.scene = scene;
  std::memcpy(capture.topology_sha256, info.topology_sha256, 32);
  capture.state = captured.state;
  capture.inverse_mass = captured.inverse_mass;
  capture.inverse_inertia = captured.inverse_inertia;
  capture.half_extents = captured.half_extents;
  capture.gravity_xyz = captured.gravity;
  capture.material_friction = captured.friction;
  capture.material_restitution = captured.restitution;
  capture.joint_cache = captured.joint_cache;
  capture.contact_feature_ids = captured.feature_ids;
  capture.contact_impulse_cache = captured.impulses;
  if (box3d_cuda_scene_capture_v2(&capture) != BOX3D_CUDA_STATUS_V2_SUCCESS ||
      cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "initial capture failed\n");
    return 4;
  }

  std::vector<float> replacement_state = state;
  std::vector<float> replacement_mass = inverse_mass;
  std::vector<float> replacement_inertia = inverse_inertia;
  std::vector<float> replacement_extents = half_extents;
  std::vector<int64_t> replacement_features = feature_ids;
  std::vector<float> replacement_impulses = impulses;
  for (size_t environment = 0; environment < environments; ++environment) {
    replacement_state[(environment * bodies + 1) * 13] += 1000.0f;
    replacement_mass[environment * bodies + 1] += 0.1f;
    replacement_inertia[(environment * bodies + 1) * 3] += 0.1f;
    replacement_extents[(environment * bodies + 1) * 3] += 0.1f;
    replacement_features[environment * 4] += 1000;
    replacement_impulses[environment * 12] += 1000.0f;
  }
  Snapshot replacement;
  replacement.state = device_copy(replacement_state);
  replacement.inverse_mass = device_copy(replacement_mass);
  replacement.inverse_inertia = device_copy(replacement_inertia);
  replacement.half_extents = device_copy(replacement_extents);
  replacement.gravity = device_copy(std::vector<float>{1.0f, -3.0f, 2.0f});
  replacement.friction = device_copy(std::vector<float>{0.9f});
  replacement.restitution = device_copy(std::vector<float>{0.3f});
  replacement.feature_ids = device_copy(replacement_features);
  replacement.impulses = device_copy(replacement_impulses);
  uint8_t* mask = device_copy(std::vector<uint8_t>{0, 7, 0, 1});

  box3d_cuda_scene_restore_desc_v2 restore{};
  restore.struct_size = sizeof(restore);
  restore.abi_version = BOX3D_CUDA_ABI_VERSION_V2;
  restore.scene = scene;
  std::memcpy(restore.topology_sha256, info.topology_sha256, 32);
  restore.state = replacement.state;
  restore.inverse_mass = replacement.inverse_mass;
  restore.inverse_inertia = replacement.inverse_inertia;
  restore.half_extents = replacement.half_extents;
  restore.gravity_xyz = replacement.gravity;
  restore.material_friction = replacement.friction;
  restore.material_restitution = replacement.restitution;
  restore.contact_feature_ids = replacement.feature_ids;
  restore.contact_impulse_cache = replacement.impulses;
  restore.environment_mask = mask;
  if (box3d_cuda_scene_restore_v2(&restore) != BOX3D_CUDA_STATUS_V2_SUCCESS ||
      box3d_cuda_scene_capture_v2(&capture) != BOX3D_CUDA_STATUS_V2_SUCCESS ||
      cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "masked restore/capture failed\n");
    return 5;
  }
  const auto observed_state = host_copy(captured.state, eb * 13);
  const auto observed_mass = host_copy(captured.inverse_mass, eb);
  const auto observed_inertia = host_copy(captured.inverse_inertia, eb * 3);
  const auto observed_extents = host_copy(captured.half_extents, eb * 3);
  const auto observed_features = host_copy(captured.feature_ids, ep * 4);
  const auto observed_impulses = host_copy(captured.impulses, ep * 12);
  const auto observed_gravity = host_copy(captured.gravity, 3);
  const auto observed_friction = host_copy(captured.friction, 1);
  const auto observed_restitution = host_copy(captured.restitution, 1);
  for (size_t environment = 0; environment < environments; ++environment) {
    const bool selected = environment == 1 || environment == 3;
    const size_t dynamic = environment * bodies + 1;
    if (!near(observed_state[dynamic * 13],
              selected ? replacement_state[dynamic * 13] : state[dynamic * 13]) ||
        !near(observed_mass[dynamic],
              selected ? replacement_mass[dynamic] : inverse_mass[dynamic]) ||
        !near(observed_inertia[dynamic * 3],
              selected ? replacement_inertia[dynamic * 3]
                       : inverse_inertia[dynamic * 3]) ||
        !near(observed_extents[dynamic * 3],
              selected ? replacement_extents[dynamic * 3]
                       : half_extents[dynamic * 3]) ||
        observed_features[environment * 4] !=
            (selected ? replacement_features[environment * 4]
                      : feature_ids[environment * 4]) ||
        !near(observed_impulses[environment * 12],
              selected ? replacement_impulses[environment * 12]
                       : impulses[environment * 12])) {
      std::fprintf(stderr, "masked environment restore was not exact\n");
      return 6;
    }
  }
  if (!near(observed_gravity[0], 1.0f) ||
      !near(observed_gravity[1], -3.0f) ||
      !near(observed_gravity[2], 2.0f) ||
      !near(observed_friction[0], 0.9f) ||
      !near(observed_restitution[0], 0.3f)) {
    std::fprintf(stderr, "global restore was not applied exactly once\n");
    return 7;
  }

  cudaFree(replacement.gravity);
  cudaFree(replacement.friction);
  cudaFree(replacement.restitution);
  replacement.gravity = device_copy(std::vector<float>{4.0f, -5.0f, 6.0f});
  replacement.friction = device_copy(std::vector<float>{0.4f});
  replacement.restitution = device_copy(std::vector<float>{0.2f});
  cudaFree(mask);
  mask = device_copy(std::vector<uint8_t>{0, 0, 0, 0});
  restore.gravity_xyz = replacement.gravity;
  restore.material_friction = replacement.friction;
  restore.material_restitution = replacement.restitution;
  restore.environment_mask = mask;
  if (box3d_cuda_scene_restore_v2(&restore) != BOX3D_CUDA_STATUS_V2_SUCCESS ||
      box3d_cuda_scene_capture_v2(&capture) != BOX3D_CUDA_STATUS_V2_SUCCESS ||
      cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "all-zero masked restore failed\n");
    return 8;
  }
  const auto zero_mask_state = host_copy(captured.state, eb * 13);
  const auto zero_mask_gravity = host_copy(captured.gravity, 3);
  if (std::memcmp(zero_mask_state.data(), observed_state.data(),
                  zero_mask_state.size() * sizeof(float)) != 0 ||
      !near(zero_mask_gravity[0], 4.0f) ||
      !near(zero_mask_gravity[1], -5.0f) ||
      !near(zero_mask_gravity[2], 6.0f)) {
    std::fprintf(stderr, "all-zero mask changed env state or skipped globals\n");
    return 9;
  }

  uint8_t* contact_active = device_allocate<uint8_t>(ep);
  uint8_t* contact_ever = device_allocate<uint8_t>(ep);
  uint32_t* contact_count = device_allocate<uint32_t>(ep);
  float* contact_penetration = device_allocate<float>(ep);
  float* contact_normal_impulse = device_allocate<float>(ep);
  box3d_cuda_scene_step_desc_v2 step{};
  step.struct_size = sizeof(step);
  step.abi_version = BOX3D_CUDA_ABI_VERSION_V2;
  step.scene = scene;
  step.steps = 2;
  step.contact_active = contact_active;
  step.contact_ever = contact_ever;
  step.contact_count = contact_count;
  step.contact_penetration = contact_penetration;
  step.contact_normal_impulse = contact_normal_impulse;
  if (box3d_cuda_scene_step_v2(&step) != BOX3D_CUDA_STATUS_V2_SUCCESS ||
      box3d_cuda_scene_capture_v2(&capture) != BOX3D_CUDA_STATUS_V2_SUCCESS ||
      cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "resident OBB step failed\n");
    return 10;
  }
  const auto stepped_state = host_copy(captured.state, eb * 13);
  const auto final_active = host_copy(contact_active, ep);
  const auto final_count = host_copy(contact_count, ep);
  bool moved = false;
  for (size_t environment = 0; environment < environments; ++environment) {
    const size_t dynamic = environment * bodies + 1;
    moved = moved ||
            std::memcmp(stepped_state.data() + dynamic * 13,
                        zero_mask_state.data() + dynamic * 13,
                        13 * sizeof(float)) != 0;
    if ((final_active[environment] != 0) !=
            (final_count[environment] != 0) ||
        final_count[environment] > BOX3D_CUDA_MANIFOLD_POINTS_V2) {
      std::fprintf(stderr, "final contact diagnostics disagree\n");
      return 11;
    }
  }
  if (!moved) {
    std::fprintf(stderr, "resident state did not advance\n");
    return 12;
  }
  cudaFree(contact_active);
  cudaFree(contact_ever);
  cudaFree(contact_count);
  cudaFree(contact_penetration);
  cudaFree(contact_normal_impulse);

  captured.release();
  replacement.release();
  cudaFree(mask);
  if (box3d_cuda_scene_unregister_v2(scene) != BOX3D_CUDA_STATUS_V2_SUCCESS ||
      box3d_cuda_scene_get_info_v2(scene, &info) !=
          BOX3D_CUDA_STATUS_V2_INVALID_HANDLE) {
    std::fprintf(stderr, "scene unregister lifecycle failed\n");
    return 13;
  }

  const uint32_t joint_body_ids[] = {100, 101};
  const uint32_t joint_body_motion[] = {BOX3D_CUDA_BODY_FIXED_V2,
                                        BOX3D_CUDA_BODY_DYNAMIC_V2};
  const uint32_t joint_ids[] = {200};
  const uint32_t joint_body_indices[] = {0, 1};
  const uint32_t joint_types[] = {BOX3D_CUDA_JOINT_REVOLUTE_V2};
  const float parent_anchor[] = {0.5f, 0.0f, 0.0f};
  const float child_anchor[] = {-0.5f, 0.0f, 0.0f};
  const float joint_axis[] = {0.0f, 0.0f, 1.0f};
  const float joint_reference[] = {0.0f, 0.0f, 0.0f, 1.0f};
  const float joint_lower[] = {-1.0f};
  const float joint_upper[] = {1.0f};
  const float joint_damping[] = {2.0f};
  const float joint_stiffness[] = {40.0f};
  const uint32_t joint_control_mode[] = {BOX3D_CUDA_CONTROL_POSITION_V2};
  std::vector<float> joint_state(eb * 13, 0.0f);
  std::vector<float> joint_inverse_mass(eb, 0.0f);
  std::vector<float> joint_inverse_inertia(eb * 3, 0.0f);
  std::vector<float> joint_half_extents(eb * 3, 0.2f);
  for (size_t environment = 0; environment < environments; ++environment) {
    const size_t parent = environment * bodies;
    const size_t child = parent + 1;
    joint_state[parent * 13 + 6] = 1.0f;
    joint_state[child * 13 + 0] = 1.0f;
    joint_state[child * 13 + 6] = 1.0f;
    joint_inverse_mass[child] = 1.0f;
    joint_inverse_inertia[child * 3 + 0] = 1.0f;
    joint_inverse_inertia[child * 3 + 1] = 1.0f;
    joint_inverse_inertia[child * 3 + 2] = 1.0f;
  }
  box3d_cuda_scene_register_desc_v2 joint_registration{};
  joint_registration.struct_size = sizeof(joint_registration);
  joint_registration.abi_version = BOX3D_CUDA_ABI_VERSION_V2;
  joint_registration.device_ordinal = -1;
  joint_registration.environments = environments;
  joint_registration.bodies = bodies;
  joint_registration.joints = 1;
  joint_registration.substeps = 2;
  joint_registration.solver_iterations = 12;
  joint_registration.material_binding = BOX3D_CUDA_MATERIAL_GLOBAL_V2;
  joint_registration.dt = 1.0f / 120.0f;
  joint_registration.global_friction = 0.5f;
  joint_registration.warm_start_factor = 0.8f;
  joint_registration.contact_slop = 1.0e-4f;
  joint_registration.position_correction = 0.8f;
  joint_registration.angular_damping = 0.02f;
  joint_registration.sat_epsilon = 1.0e-7f;
  joint_registration.joint_position_slop = 1.0e-5f;
  joint_registration.joint_angular_slop = 1.0e-5f;
  joint_registration.maximum_linear_repair = 0.1f;
  joint_registration.maximum_angular_repair = 0.2f;
  joint_registration.body_caller_ids = joint_body_ids;
  joint_registration.body_motion = joint_body_motion;
  joint_registration.joint_caller_ids = joint_ids;
  joint_registration.joint_body_indices = joint_body_indices;
  joint_registration.joint_types = joint_types;
  joint_registration.joint_parent_anchor = parent_anchor;
  joint_registration.joint_child_anchor = child_anchor;
  joint_registration.joint_axis_parent = joint_axis;
  joint_registration.joint_reference_xyzw = joint_reference;
  joint_registration.joint_lower_limit = joint_lower;
  joint_registration.joint_upper_limit = joint_upper;
  joint_registration.joint_damping = joint_damping;
  joint_registration.joint_stiffness = joint_stiffness;
  joint_registration.joint_control_mode = joint_control_mode;
  joint_registration.state = joint_state.data();
  joint_registration.inverse_mass = joint_inverse_mass.data();
  joint_registration.inverse_inertia = joint_inverse_inertia.data();
  joint_registration.half_extents = joint_half_extents.data();
  box3d_cuda_scene_handle_v2 joint_scene = 0;
  if (box3d_cuda_scene_register_v2(&joint_registration, &joint_scene) !=
      BOX3D_CUDA_STATUS_V2_SUCCESS) {
    std::fprintf(stderr, "revolute scene registration failed\n");
    return 14;
  }
  float* target_position = device_copy(std::vector<float>(environments, 0.35f));
  float* target_velocity = device_copy(std::vector<float>(environments, 0.0f));
  float* maximum_effort = device_copy(std::vector<float>(environments, 50.0f));
  float* maximum_speed = device_copy(std::vector<float>(environments, 10.0f));
  float* maximum_acceleration =
      device_copy(std::vector<float>(environments, 100.0f));
  float* coordinate = device_allocate<float>(environments);
  float* anchor_error = device_allocate<float>(environments);
  float* angular_error = device_allocate<float>(environments);
  float* limit_error = device_allocate<float>(environments);
  float* motor_impulse = device_allocate<float>(environments);
  uint8_t* limit_active = device_allocate<uint8_t>(environments);
  box3d_cuda_scene_step_desc_v2 joint_step{};
  joint_step.struct_size = sizeof(joint_step);
  joint_step.abi_version = BOX3D_CUDA_ABI_VERSION_V2;
  joint_step.scene = joint_scene;
  joint_step.steps = 40;
  joint_step.target_position = target_position;
  joint_step.target_velocity = target_velocity;
  joint_step.maximum_effort = maximum_effort;
  joint_step.maximum_speed = maximum_speed;
  joint_step.maximum_acceleration = maximum_acceleration;
  joint_step.joint_coordinate = coordinate;
  joint_step.joint_anchor_error = anchor_error;
  joint_step.joint_angular_error = angular_error;
  joint_step.joint_limit_error = limit_error;
  joint_step.joint_motor_impulse = motor_impulse;
  joint_step.joint_limit_active = limit_active;
  if (box3d_cuda_scene_step_v2(&joint_step) != BOX3D_CUDA_STATUS_V2_SUCCESS ||
      cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "revolute resident step failed\n");
    return 15;
  }
  const auto observed_coordinate = host_copy(coordinate, environments);
  const auto observed_anchor_error = host_copy(anchor_error, environments);
  const auto observed_motor_impulse = host_copy(motor_impulse, environments);
  for (size_t environment = 0; environment < environments; ++environment) {
    if (!std::isfinite(observed_coordinate[environment]) ||
        !std::isfinite(observed_anchor_error[environment]) ||
        !std::isfinite(observed_motor_impulse[environment]) ||
        std::fabs(observed_coordinate[environment]) <= 1.0e-4f ||
        observed_anchor_error[environment] > 0.1f ||
        std::fabs(observed_motor_impulse[environment]) <= 1.0e-6f) {
      std::fprintf(stderr, "revolute diagnostics/control did not advance\n");
      return 16;
    }
  }
  cudaFree(target_position);
  cudaFree(target_velocity);
  cudaFree(maximum_effort);
  cudaFree(maximum_speed);
  cudaFree(maximum_acceleration);
  cudaFree(coordinate);
  cudaFree(anchor_error);
  cudaFree(angular_error);
  cudaFree(limit_error);
  cudaFree(motor_impulse);
  cudaFree(limit_active);
  if (box3d_cuda_scene_unregister_v2(joint_scene) !=
      BOX3D_CUDA_STATUS_V2_SUCCESS) {
    std::fprintf(stderr, "revolute scene unregister failed\n");
    return 17;
  }
  std::puts("Box3D CUDA ABI-v2 r3 resident lifecycle passed");
  return 0;
}
