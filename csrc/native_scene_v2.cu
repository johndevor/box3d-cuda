// SPDX-License-Identifier: MIT
// ABI-v2 r3 resident scene lifecycle. Physics stepping is enabled separately
// only after the production coupled kernel is wired through this boundary.
#include "box3d_cuda/box3d_cuda.h"
#include "../proposals/box3d_cuda_v2.h"
#include "topology_sha256.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define BOX3D_CUDA_NATIVE_KERNELS_ONLY 1
#include "coupled.cu"

namespace {

constexpr uint64_t kLifecycleCapabilities =
    BOX3D_CUDA_CAP_V2_ORIENTED_BOXES |
    BOX3D_CUDA_CAP_V2_EXPLICIT_CONTACT_PAIRS |
    BOX3D_CUDA_CAP_V2_FIXED_JOINTS |
    BOX3D_CUDA_CAP_V2_REVOLUTE_JOINTS |
    BOX3D_CUDA_CAP_V2_PRISMATIC_JOINTS |
    BOX3D_CUDA_CAP_V2_PERSISTENT_CONTACTS |
    BOX3D_CUDA_CAP_V2_RESIDENT_STATE |
    BOX3D_CUDA_CAP_V2_DETERMINISTIC_SNAPSHOT |
    BOX3D_CUDA_CAP_V2_ASYNC_CALLER_STREAM |
    BOX3D_CUDA_CAP_V2_GLOBAL_MATERIAL |
    BOX3D_CUDA_CAP_V2_PARTIAL_ENVIRONMENT_RESTORE;

template <typename T>
struct DeviceBuffer {
  T* data = nullptr;
  size_t count = 0;

  bool allocate(size_t requested_count) {
    count = requested_count;
    if (count == 0) return true;
    return cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)) ==
           cudaSuccess;
  }

  bool copy_from_cpu(const T* source, size_t requested_count) {
    if (!allocate(requested_count)) return false;
    if (count == 0) return true;
    return cudaMemcpy(data, source, count * sizeof(T), cudaMemcpyHostToDevice) ==
           cudaSuccess;
  }

  bool fill_byte(int value) {
    return count == 0 || cudaMemset(data, value, count * sizeof(T)) == cudaSuccess;
  }

  void release() {
    if (data != nullptr) cudaFree(data);
    data = nullptr;
    count = 0;
  }
};

struct Scene {
  int device = 0;
  uint32_t environments = 0;
  uint32_t bodies = 0;
  uint32_t joints = 0;
  uint32_t contact_pairs = 0;
  uint32_t substeps = 0;
  uint32_t solver_iterations = 0;
  uint32_t material_binding = BOX3D_CUDA_MATERIAL_GLOBAL_V2;
  float dt = 0.0f;
  float warm_start_factor = 0.0f;
  float contact_slop = 0.0f;
  float position_correction = 0.0f;
  float angular_damping = 0.0f;
  float sat_epsilon = 0.0f;
  float joint_position_slop = 0.0f;
  float joint_angular_slop = 0.0f;
  float maximum_linear_repair = 0.0f;
  float maximum_angular_repair = 0.0f;
  uint8_t topology_sha256[BOX3D_CUDA_TOPOLOGY_HASH_BYTES_V2] = {};
  std::mutex operation_mutex;
  bool retired = false;

  std::vector<uint32_t> body_caller_ids;
  std::vector<uint32_t> body_motion;
  std::vector<uint32_t> joint_caller_ids;
  std::vector<uint32_t> contact_pair_caller_ids;
  DeviceBuffer<int64_t> joint_body_indices;
  DeviceBuffer<int64_t> joint_types;
  DeviceBuffer<float> joint_parent_anchor;
  DeviceBuffer<float> joint_child_anchor;
  DeviceBuffer<float> joint_axis_parent;
  DeviceBuffer<float> joint_reference_xyzw;
  DeviceBuffer<float> joint_lower_limit;
  DeviceBuffer<float> joint_upper_limit;
  DeviceBuffer<float> joint_damping;
  DeviceBuffer<float> joint_stiffness;
  DeviceBuffer<uint32_t> joint_control_mode;
  DeviceBuffer<uint8_t> joint_motor_enabled;
  DeviceBuffer<int64_t> contact_body_indices;

  DeviceBuffer<float> state;
  DeviceBuffer<float> inverse_mass;
  DeviceBuffer<float> inverse_inertia;
  DeviceBuffer<float> half_extents;
  DeviceBuffer<float> gravity_xyz;
  DeviceBuffer<float> material_friction;
  DeviceBuffer<float> material_restitution;
  DeviceBuffer<float> joint_cache;
  DeviceBuffer<int64_t> contact_feature_ids;
  DeviceBuffer<float> contact_impulse_cache;
  DeviceBuffer<float> diagnostic_joint_coordinate;
  DeviceBuffer<float> diagnostic_joint_anchor_error;
  DeviceBuffer<float> diagnostic_joint_angular_error;
  DeviceBuffer<float> diagnostic_joint_limit_error;
  DeviceBuffer<float> diagnostic_joint_motor_impulse;
  DeviceBuffer<uint8_t> diagnostic_joint_limit_active;
  DeviceBuffer<uint8_t> diagnostic_contact_ever;
  DeviceBuffer<float> diagnostic_contact_penetration;
  DeviceBuffer<int32_t> diagnostic_contact_count;
  DeviceBuffer<float> diagnostic_contact_normal_impulse;

  ~Scene() {
    cudaSetDevice(device);
    joint_body_indices.release();
    joint_types.release();
    joint_parent_anchor.release();
    joint_child_anchor.release();
    joint_axis_parent.release();
    joint_reference_xyzw.release();
    joint_lower_limit.release();
    joint_upper_limit.release();
    joint_damping.release();
    joint_stiffness.release();
    joint_control_mode.release();
    joint_motor_enabled.release();
    contact_body_indices.release();
    state.release();
    inverse_mass.release();
    inverse_inertia.release();
    half_extents.release();
    gravity_xyz.release();
    material_friction.release();
    material_restitution.release();
    joint_cache.release();
    contact_feature_ids.release();
    contact_impulse_cache.release();
    diagnostic_joint_coordinate.release();
    diagnostic_joint_anchor_error.release();
    diagnostic_joint_angular_error.release();
    diagnostic_joint_limit_error.release();
    diagnostic_joint_motor_impulse.release();
    diagnostic_joint_limit_active.release();
    diagnostic_contact_ever.release();
    diagnostic_contact_penetration.release();
    diagnostic_contact_count.release();
    diagnostic_contact_normal_impulse.release();
  }
};

std::mutex g_registry_mutex;
std::unordered_map<box3d_cuda_scene_handle_v2, std::shared_ptr<Scene>> g_scenes;
std::atomic<uint64_t> g_next_scene{1};

bool finite(float value) { return std::isfinite(value); }

bool finite_array(const float* values, size_t count) {
  if (count != 0 && values == nullptr) return false;
  for (size_t index = 0; index < count; ++index) {
    if (!finite(values[index])) return false;
  }
  return true;
}

bool normalized3(const float* value) {
  const float norm2 = value[0] * value[0] + value[1] * value[1] +
                      value[2] * value[2];
  return finite(norm2) && std::fabs(norm2 - 1.0f) <= 1.0e-3f;
}

bool normalized4(const float* value) {
  const float norm2 = value[0] * value[0] + value[1] * value[1] +
                      value[2] * value[2] + value[3] * value[3];
  return finite(norm2) && std::fabs(norm2 - 1.0f) <= 1.0e-3f;
}

bool unique_ids(const uint32_t* values, size_t count) {
  if (count != 0 && values == nullptr) return false;
  std::unordered_set<uint32_t> seen;
  seen.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    if (!seen.insert(values[index]).second) return false;
  }
  return true;
}

box3d_cuda_status_v2 validate_registration(
    const box3d_cuda_scene_register_desc_v2* descriptor) {
  if (descriptor == nullptr) return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
  if (descriptor->struct_size != sizeof(*descriptor))
    return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
  if (descriptor->abi_version != BOX3D_CUDA_ABI_VERSION_V2)
    return BOX3D_CUDA_STATUS_V2_ABI_MISMATCH;
  if (descriptor->flags != 0 || descriptor->reserved_u32 != 0 ||
      descriptor->device_ordinal < -1)
    return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
  if (descriptor->environments == 0 || descriptor->bodies == 0 ||
      descriptor->substeps == 0 || descriptor->solver_iterations == 0)
    return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
  if (descriptor->bodies > BOX3D_CUDA_MAX_BODIES_V2 ||
      descriptor->joints > BOX3D_CUDA_MAX_JOINTS_V2 ||
      descriptor->contact_pairs > BOX3D_CUDA_MAX_CONTACT_PAIRS_V2)
    return BOX3D_CUDA_STATUS_V2_LIMIT_EXCEEDED;
  if (descriptor->material_binding != BOX3D_CUDA_MATERIAL_GLOBAL_V2 ||
      descriptor->environment_gravity_xyz != nullptr ||
      descriptor->body_friction != nullptr ||
      descriptor->body_restitution != nullptr ||
      descriptor->pair_friction != nullptr ||
      descriptor->pair_restitution != nullptr)
    return BOX3D_CUDA_STATUS_V2_UNSUPPORTED;

  const size_t environment_bodies =
      static_cast<size_t>(descriptor->environments) * descriptor->bodies;
  if (descriptor->body_caller_ids == nullptr ||
      descriptor->body_motion == nullptr || descriptor->state == nullptr ||
      descriptor->inverse_mass == nullptr ||
      descriptor->inverse_inertia == nullptr ||
      descriptor->half_extents == nullptr ||
      !unique_ids(descriptor->body_caller_ids, descriptor->bodies))
    return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;

  const float solver_parameters[] = {
      descriptor->dt,
      descriptor->global_gravity_xyz[0],
      descriptor->global_gravity_xyz[1],
      descriptor->global_gravity_xyz[2],
      descriptor->global_friction,
      descriptor->global_restitution,
      descriptor->warm_start_factor,
      descriptor->contact_slop,
      descriptor->position_correction,
      descriptor->angular_damping,
      descriptor->sat_epsilon,
      descriptor->joint_position_slop,
      descriptor->joint_angular_slop,
      descriptor->maximum_linear_repair,
      descriptor->maximum_angular_repair,
  };
  if (!finite_array(solver_parameters,
                    sizeof(solver_parameters) / sizeof(solver_parameters[0])) ||
      descriptor->dt <= 0.0f || descriptor->global_friction < 0.0f ||
      descriptor->global_restitution < 0.0f ||
      descriptor->warm_start_factor < 0.0f ||
      descriptor->contact_slop < 0.0f ||
      descriptor->position_correction < 0.0f ||
      descriptor->angular_damping < 0.0f || descriptor->sat_epsilon <= 0.0f ||
      descriptor->joint_position_slop < 0.0f ||
      descriptor->joint_angular_slop < 0.0f ||
      descriptor->maximum_linear_repair < 0.0f ||
      descriptor->maximum_angular_repair < 0.0f)
    return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;

  if (!finite_array(descriptor->state, environment_bodies * 13) ||
      !finite_array(descriptor->inverse_mass, environment_bodies) ||
      !finite_array(descriptor->inverse_inertia, environment_bodies * 3) ||
      !finite_array(descriptor->half_extents, environment_bodies * 3))
    return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
  for (size_t flat = 0; flat < environment_bodies; ++flat) {
    const uint32_t motion = descriptor->body_motion[flat % descriptor->bodies];
    if (motion == BOX3D_CUDA_BODY_KINEMATIC_V2)
      return BOX3D_CUDA_STATUS_V2_UNSUPPORTED;
    if (motion != BOX3D_CUDA_BODY_FIXED_V2 &&
        motion != BOX3D_CUDA_BODY_DYNAMIC_V2)
      return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
    if (!normalized4(descriptor->state + flat * 13 + 3))
      return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
    for (size_t axis = 0; axis < 3; ++axis) {
      if (descriptor->half_extents[flat * 3 + axis] <= 0.0f)
        return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
    }
    if (motion == BOX3D_CUDA_BODY_FIXED_V2) {
      if (descriptor->inverse_mass[flat] != 0.0f ||
          descriptor->inverse_inertia[flat * 3] != 0.0f ||
          descriptor->inverse_inertia[flat * 3 + 1] != 0.0f ||
          descriptor->inverse_inertia[flat * 3 + 2] != 0.0f)
        return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
    } else if (descriptor->inverse_mass[flat] <= 0.0f ||
               descriptor->inverse_inertia[flat * 3] <= 0.0f ||
               descriptor->inverse_inertia[flat * 3 + 1] <= 0.0f ||
               descriptor->inverse_inertia[flat * 3 + 2] <= 0.0f) {
      return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
    }
  }

  const size_t joints = descriptor->joints;
  if (joints != 0) {
    if (descriptor->joint_caller_ids == nullptr ||
        descriptor->joint_body_indices == nullptr ||
        descriptor->joint_types == nullptr ||
        descriptor->joint_parent_anchor == nullptr ||
        descriptor->joint_child_anchor == nullptr ||
        descriptor->joint_axis_parent == nullptr ||
        descriptor->joint_reference_xyzw == nullptr ||
        descriptor->joint_lower_limit == nullptr ||
        descriptor->joint_upper_limit == nullptr ||
        descriptor->joint_damping == nullptr ||
        descriptor->joint_stiffness == nullptr ||
        descriptor->joint_control_mode == nullptr ||
        !unique_ids(descriptor->joint_caller_ids, joints) ||
        !finite_array(descriptor->joint_parent_anchor, joints * 3) ||
        !finite_array(descriptor->joint_child_anchor, joints * 3) ||
        !finite_array(descriptor->joint_axis_parent, joints * 3) ||
        !finite_array(descriptor->joint_reference_xyzw, joints * 4) ||
        !finite_array(descriptor->joint_lower_limit, joints) ||
        !finite_array(descriptor->joint_upper_limit, joints) ||
        !finite_array(descriptor->joint_damping, joints) ||
        !finite_array(descriptor->joint_stiffness, joints))
      return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
    for (size_t joint = 0; joint < joints; ++joint) {
      const uint32_t parent = descriptor->joint_body_indices[joint * 2];
      const uint32_t child = descriptor->joint_body_indices[joint * 2 + 1];
      const uint32_t kind = descriptor->joint_types[joint];
      const uint32_t mode = descriptor->joint_control_mode[joint];
      if (parent >= descriptor->bodies || child >= descriptor->bodies ||
          parent == child || kind > BOX3D_CUDA_JOINT_PRISMATIC_V2 ||
          mode > BOX3D_CUDA_CONTROL_VELOCITY_V2 ||
          (kind == BOX3D_CUDA_JOINT_FIXED_V2 &&
           mode != BOX3D_CUDA_CONTROL_DISABLED_V2) ||
          !normalized3(descriptor->joint_axis_parent + joint * 3) ||
          !normalized4(descriptor->joint_reference_xyzw + joint * 4) ||
          descriptor->joint_lower_limit[joint] >
              descriptor->joint_upper_limit[joint] ||
          descriptor->joint_damping[joint] < 0.0f ||
          descriptor->joint_stiffness[joint] < 0.0f)
        return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
    }
  }

  const size_t pairs = descriptor->contact_pairs;
  if (pairs != 0) {
    if (descriptor->contact_pair_caller_ids == nullptr ||
        descriptor->contact_body_indices == nullptr ||
        !unique_ids(descriptor->contact_pair_caller_ids, pairs))
      return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
    for (size_t pair = 0; pair < pairs; ++pair) {
      const uint32_t first = descriptor->contact_body_indices[pair * 2];
      const uint32_t second = descriptor->contact_body_indices[pair * 2 + 1];
      if (first >= descriptor->bodies || second >= descriptor->bodies ||
          first == second)
        return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
    }
  }

  if ((descriptor->joint_cache != nullptr &&
       !finite_array(descriptor->joint_cache,
                     static_cast<size_t>(descriptor->environments) * joints * 8)) ||
      (descriptor->contact_impulse_cache != nullptr &&
       !finite_array(descriptor->contact_impulse_cache,
                     static_cast<size_t>(descriptor->environments) * pairs * 12)))
    return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
  return BOX3D_CUDA_STATUS_V2_SUCCESS;
}

template <typename Destination, typename Source>
bool convert_and_copy(DeviceBuffer<Destination>& destination,
                      const Source* source, size_t count) {
  std::vector<Destination> converted(count);
  for (size_t index = 0; index < count; ++index)
    converted[index] = static_cast<Destination>(source[index]);
  return destination.copy_from_cpu(converted.data(), count);
}

bool initialize_scene(const box3d_cuda_scene_register_desc_v2& descriptor,
                      Scene& scene) {
  scene.environments = descriptor.environments;
  scene.bodies = descriptor.bodies;
  scene.joints = descriptor.joints;
  scene.contact_pairs = descriptor.contact_pairs;
  scene.substeps = descriptor.substeps;
  scene.solver_iterations = descriptor.solver_iterations;
  scene.material_binding = descriptor.material_binding;
  scene.dt = descriptor.dt;
  scene.warm_start_factor = descriptor.warm_start_factor;
  scene.contact_slop = descriptor.contact_slop;
  scene.position_correction = descriptor.position_correction;
  scene.angular_damping = descriptor.angular_damping;
  scene.sat_epsilon = descriptor.sat_epsilon;
  scene.joint_position_slop = descriptor.joint_position_slop;
  scene.joint_angular_slop = descriptor.joint_angular_slop;
  scene.maximum_linear_repair = descriptor.maximum_linear_repair;
  scene.maximum_angular_repair = descriptor.maximum_angular_repair;
  scene.body_caller_ids.assign(descriptor.body_caller_ids,
                               descriptor.body_caller_ids + descriptor.bodies);
  scene.body_motion.assign(descriptor.body_motion,
                           descriptor.body_motion + descriptor.bodies);
  if (descriptor.joints != 0) {
    scene.joint_caller_ids.assign(descriptor.joint_caller_ids,
                                  descriptor.joint_caller_ids + descriptor.joints);
  }
  if (descriptor.contact_pairs != 0) {
    scene.contact_pair_caller_ids.assign(
        descriptor.contact_pair_caller_ids,
        descriptor.contact_pair_caller_ids + descriptor.contact_pairs);
  }

  const size_t eb = static_cast<size_t>(descriptor.environments) * descriptor.bodies;
  const size_t ej = static_cast<size_t>(descriptor.environments) * descriptor.joints;
  const size_t ep =
      static_cast<size_t>(descriptor.environments) * descriptor.contact_pairs;
  if (!convert_and_copy(scene.joint_body_indices,
                        descriptor.joint_body_indices,
                        static_cast<size_t>(descriptor.joints) * 2) ||
      !convert_and_copy(scene.joint_types, descriptor.joint_types,
                        descriptor.joints) ||
      !scene.joint_parent_anchor.copy_from_cpu(
          descriptor.joint_parent_anchor, static_cast<size_t>(descriptor.joints) * 3) ||
      !scene.joint_child_anchor.copy_from_cpu(
          descriptor.joint_child_anchor, static_cast<size_t>(descriptor.joints) * 3) ||
      !scene.joint_axis_parent.copy_from_cpu(
          descriptor.joint_axis_parent, static_cast<size_t>(descriptor.joints) * 3) ||
      !scene.joint_reference_xyzw.copy_from_cpu(
          descriptor.joint_reference_xyzw, static_cast<size_t>(descriptor.joints) * 4) ||
      !scene.joint_lower_limit.copy_from_cpu(descriptor.joint_lower_limit,
                                             descriptor.joints) ||
      !scene.joint_upper_limit.copy_from_cpu(descriptor.joint_upper_limit,
                                             descriptor.joints) ||
      !scene.joint_damping.copy_from_cpu(descriptor.joint_damping,
                                         descriptor.joints) ||
      !scene.joint_control_mode.copy_from_cpu(descriptor.joint_control_mode,
                                              descriptor.joints) ||
      !convert_and_copy(scene.contact_body_indices,
                        descriptor.contact_body_indices,
                        static_cast<size_t>(descriptor.contact_pairs) * 2) ||
      !scene.state.copy_from_cpu(descriptor.state, eb * 13) ||
      !scene.inverse_mass.copy_from_cpu(descriptor.inverse_mass, eb) ||
      !scene.inverse_inertia.copy_from_cpu(descriptor.inverse_inertia, eb * 3) ||
      !scene.half_extents.copy_from_cpu(descriptor.half_extents, eb * 3) ||
      !scene.gravity_xyz.copy_from_cpu(descriptor.global_gravity_xyz, 3) ||
      !scene.material_friction.copy_from_cpu(&descriptor.global_friction, 1) ||
      !scene.material_restitution.copy_from_cpu(&descriptor.global_restitution, 1))
    return false;

  std::vector<uint8_t> motor_enabled(descriptor.joints, 0);
  std::vector<float> solver_stiffness(descriptor.joints, 0.0f);
  for (size_t joint = 0; joint < descriptor.joints; ++joint) {
    motor_enabled[joint] = descriptor.joint_control_mode[joint] !=
                           BOX3D_CUDA_CONTROL_DISABLED_V2;
    if (descriptor.joint_control_mode[joint] ==
        BOX3D_CUDA_CONTROL_POSITION_V2) {
      solver_stiffness[joint] = descriptor.joint_stiffness[joint];
    }
  }
  if (!scene.joint_motor_enabled.copy_from_cpu(motor_enabled.data(),
                                               motor_enabled.size()) ||
      !scene.joint_stiffness.copy_from_cpu(solver_stiffness.data(),
                                           solver_stiffness.size()))
    return false;

  if (descriptor.joint_cache != nullptr) {
    if (!scene.joint_cache.copy_from_cpu(descriptor.joint_cache, ej * 8)) return false;
  } else if (!scene.joint_cache.allocate(ej * 8) ||
             !scene.joint_cache.fill_byte(0)) {
    return false;
  }
  if (descriptor.contact_feature_ids != nullptr) {
    if (!scene.contact_feature_ids.copy_from_cpu(descriptor.contact_feature_ids,
                                                 ep * 4))
      return false;
  } else if (!scene.contact_feature_ids.allocate(ep * 4) ||
             !scene.contact_feature_ids.fill_byte(0xff)) {
    return false;
  }
  if (descriptor.contact_impulse_cache != nullptr) {
    if (!scene.contact_impulse_cache.copy_from_cpu(
            descriptor.contact_impulse_cache, ep * 12))
      return false;
  } else if (!scene.contact_impulse_cache.allocate(ep * 12) ||
             !scene.contact_impulse_cache.fill_byte(0)) {
    return false;
  }
  if (!scene.diagnostic_joint_coordinate.allocate(ej) ||
      !scene.diagnostic_joint_anchor_error.allocate(ej) ||
      !scene.diagnostic_joint_angular_error.allocate(ej) ||
      !scene.diagnostic_joint_limit_error.allocate(ej) ||
      !scene.diagnostic_joint_motor_impulse.allocate(ej) ||
      !scene.diagnostic_joint_limit_active.allocate(ej) ||
      !scene.diagnostic_contact_ever.allocate(ep) ||
      !scene.diagnostic_contact_penetration.allocate(ep) ||
      !scene.diagnostic_contact_count.allocate(ep) ||
      !scene.diagnostic_contact_normal_impulse.allocate(ep)) {
    return false;
  }
  return box3d_cuda_native::compute_topology_sha256(
      descriptor, scene.topology_sha256);
}

std::shared_ptr<Scene> find_scene(box3d_cuda_scene_handle_v2 handle) {
  std::lock_guard<std::mutex> lock(g_registry_mutex);
  const auto iterator = g_scenes.find(handle);
  return iterator == g_scenes.end() ? nullptr : iterator->second;
}

bool is_device_pointer(const void* pointer, int expected_device) {
  if (pointer == nullptr) return false;
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
#if CUDART_VERSION >= 10000
  return (attributes.type == cudaMemoryTypeDevice ||
          attributes.type == cudaMemoryTypeManaged) &&
         (attributes.type == cudaMemoryTypeManaged ||
          attributes.device == expected_device);
#else
  return attributes.memoryType == cudaMemoryTypeDevice &&
         attributes.device == expected_device;
#endif
}

template <typename T>
bool required_device_pointer(const T* pointer, size_t count, int device) {
  return count == 0 || is_device_pointer(pointer, device);
}

bool enqueue_copy(void* destination, const void* source, size_t bytes,
                  cudaStream_t stream) {
  return bytes == 0 ||
         cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice,
                         stream) == cudaSuccess;
}

__global__ void restore_masked_bytes(uint8_t* destination,
                                     const uint8_t* source,
                                     const uint8_t* environment_mask,
                                     size_t bytes_per_environment) {
  const size_t environment = blockIdx.x;
  if (environment_mask[environment] == 0) return;
  destination += environment * bytes_per_environment;
  source += environment * bytes_per_environment;
  for (size_t offset = threadIdx.x; offset < bytes_per_environment;
       offset += blockDim.x) {
    destination[offset] = source[offset];
  }
}

bool enqueue_environment_copy(void* destination, const void* source,
                              size_t bytes_per_environment,
                              uint32_t environments,
                              const uint8_t* environment_mask,
                              cudaStream_t stream) {
  if (bytes_per_environment == 0) return true;
  if (environment_mask == nullptr) {
    return enqueue_copy(destination, source,
                        bytes_per_environment * environments, stream);
  }
  restore_masked_bytes<<<environments, 128, 0, stream>>>(
      static_cast<uint8_t*>(destination), static_cast<const uint8_t*>(source),
      environment_mask, bytes_per_environment);
  return cudaPeekAtLastError() == cudaSuccess;
}

bool enqueue_zero(void* destination, size_t bytes, cudaStream_t stream) {
  return bytes == 0 || cudaMemsetAsync(destination, 0, bytes, stream) == cudaSuccess;
}

__global__ void pack_contact_final(const int32_t* source_count,
                                   uint8_t* destination_active,
                                   uint32_t* destination_count,
                                   size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const uint32_t value = source_count[index] > 0
                             ? static_cast<uint32_t>(source_count[index])
                             : 0u;
  if (destination_active != nullptr) destination_active[index] = value != 0;
  if (destination_count != nullptr) destination_count[index] = value;
}

bool optional_device_pointer(const void* pointer, int device) {
  return pointer == nullptr || is_device_pointer(pointer, device);
}

bool valid_step_pointers(const Scene& scene,
                         const box3d_cuda_scene_step_desc_v2& descriptor) {
  const size_t ej = static_cast<size_t>(scene.environments) * scene.joints;
  return required_device_pointer(descriptor.target_position, ej, scene.device) &&
         required_device_pointer(descriptor.target_velocity, ej, scene.device) &&
         required_device_pointer(descriptor.maximum_effort, ej, scene.device) &&
         required_device_pointer(descriptor.maximum_speed, ej, scene.device) &&
         required_device_pointer(descriptor.maximum_acceleration, ej,
                                 scene.device) &&
         optional_device_pointer(descriptor.joint_coordinate, scene.device) &&
         optional_device_pointer(descriptor.joint_anchor_error, scene.device) &&
         optional_device_pointer(descriptor.joint_angular_error, scene.device) &&
         optional_device_pointer(descriptor.joint_limit_error, scene.device) &&
         optional_device_pointer(descriptor.joint_motor_impulse, scene.device) &&
         optional_device_pointer(descriptor.joint_limit_active, scene.device) &&
         optional_device_pointer(descriptor.contact_active, scene.device) &&
         optional_device_pointer(descriptor.contact_ever, scene.device) &&
         optional_device_pointer(descriptor.contact_count, scene.device) &&
         optional_device_pointer(descriptor.contact_penetration, scene.device) &&
         optional_device_pointer(descriptor.contact_normal_impulse, scene.device);
}

bool enqueue_optional_copy(void* destination, const void* source, size_t bytes,
                           cudaStream_t stream) {
  return destination == nullptr || enqueue_copy(destination, source, bytes, stream);
}

bool valid_snapshot_pointers(const Scene& scene,
                             const box3d_cuda_scene_capture_desc_v2& descriptor) {
  const size_t eb = static_cast<size_t>(scene.environments) * scene.bodies;
  const size_t ej = static_cast<size_t>(scene.environments) * scene.joints;
  const size_t ep = static_cast<size_t>(scene.environments) * scene.contact_pairs;
  return required_device_pointer(descriptor.state, eb * 13, scene.device) &&
         required_device_pointer(descriptor.inverse_mass, eb, scene.device) &&
         required_device_pointer(descriptor.inverse_inertia, eb * 3, scene.device) &&
         required_device_pointer(descriptor.half_extents, eb * 3, scene.device) &&
         required_device_pointer(descriptor.gravity_xyz, 3, scene.device) &&
         required_device_pointer(descriptor.material_friction, 1, scene.device) &&
         required_device_pointer(descriptor.material_restitution, 1, scene.device) &&
         required_device_pointer(descriptor.joint_cache, ej * 8, scene.device) &&
         required_device_pointer(descriptor.contact_feature_ids, ep * 4,
                                 scene.device) &&
         required_device_pointer(descriptor.contact_impulse_cache, ep * 12,
                                 scene.device);
}

bool valid_snapshot_pointers(const Scene& scene,
                             const box3d_cuda_scene_restore_desc_v2& descriptor) {
  const size_t eb = static_cast<size_t>(scene.environments) * scene.bodies;
  const size_t ej = static_cast<size_t>(scene.environments) * scene.joints;
  const size_t ep = static_cast<size_t>(scene.environments) * scene.contact_pairs;
  return required_device_pointer(descriptor.state, eb * 13, scene.device) &&
         required_device_pointer(descriptor.inverse_mass, eb, scene.device) &&
         required_device_pointer(descriptor.inverse_inertia, eb * 3, scene.device) &&
         required_device_pointer(descriptor.half_extents, eb * 3, scene.device) &&
         required_device_pointer(descriptor.gravity_xyz, 3, scene.device) &&
         required_device_pointer(descriptor.material_friction, 1, scene.device) &&
         required_device_pointer(descriptor.material_restitution, 1, scene.device) &&
         required_device_pointer(descriptor.joint_cache, ej * 8, scene.device) &&
         required_device_pointer(descriptor.contact_feature_ids, ep * 4,
                                 scene.device) &&
         required_device_pointer(descriptor.contact_impulse_cache, ep * 12,
                                 scene.device) &&
         (descriptor.environment_mask == nullptr ||
          is_device_pointer(descriptor.environment_mask, scene.device));
}

}  // namespace

extern "C" BOX3D_CUDA_API uint32_t box3d_cuda_get_abi_version_v2(void) {
  return BOX3D_CUDA_ABI_VERSION_V2;
}

extern "C" BOX3D_CUDA_API box3d_cuda_status_v2 box3d_cuda_query_api_v2(
    box3d_cuda_api_info_v2* info) {
  if (info == nullptr || info->struct_size != sizeof(*info))
    return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
  if (info->abi_version != BOX3D_CUDA_ABI_VERSION_V2)
    return BOX3D_CUDA_STATUS_V2_ABI_MISMATCH;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = sizeof(*info);
  info->abi_version = BOX3D_CUDA_ABI_VERSION_V2;
  info->implementation_version_major = 0;
  info->implementation_version_minor = 3;
  info->implementation_version_patch = 0;
  info->draft_revision = BOX3D_CUDA_ABI_V2_DRAFT_REVISION;
  info->capabilities = kLifecycleCapabilities;
  info->maximum_bodies = BOX3D_CUDA_MAX_BODIES_V2;
  info->maximum_joints = BOX3D_CUDA_MAX_JOINTS_V2;
  info->maximum_contact_pairs = BOX3D_CUDA_MAX_CONTACT_PAIRS_V2;
  info->manifold_points = BOX3D_CUDA_MANIFOLD_POINTS_V2;
  info->joint_cache_width = BOX3D_CUDA_JOINT_CACHE_WIDTH_V2;
  std::strncpy(info->implementation_id, "box3d-cuda-native-r3",
               sizeof(info->implementation_id) - 1);
  return BOX3D_CUDA_STATUS_V2_SUCCESS;
}

extern "C" BOX3D_CUDA_API const char* box3d_cuda_status_string_v2(
    box3d_cuda_status_v2 status) {
  switch (status) {
    case BOX3D_CUDA_STATUS_V2_SUCCESS: return "success";
    case BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT: return "invalid argument";
    case BOX3D_CUDA_STATUS_V2_ABI_MISMATCH: return "ABI mismatch";
    case BOX3D_CUDA_STATUS_V2_CUDA_ERROR: return "CUDA error";
    case BOX3D_CUDA_STATUS_V2_INVALID_HANDLE: return "invalid scene handle";
    case BOX3D_CUDA_STATUS_V2_UNSUPPORTED: return "unsupported capability";
    case BOX3D_CUDA_STATUS_V2_LIMIT_EXCEEDED: return "limit exceeded";
    case BOX3D_CUDA_STATUS_V2_ALLOCATION_FAILED: return "allocation failed";
    case BOX3D_CUDA_STATUS_V2_TOPOLOGY_MISMATCH: return "topology mismatch";
    case BOX3D_CUDA_STATUS_V2_BUSY: return "scene busy";
    default: return "unknown status";
  }
}

extern "C" BOX3D_CUDA_API box3d_cuda_status_v2 box3d_cuda_scene_register_v2(
    const box3d_cuda_scene_register_desc_v2* descriptor,
    box3d_cuda_scene_handle_v2* scene_handle) {
  const box3d_cuda_status_v2 validation = validate_registration(descriptor);
  if (validation != BOX3D_CUDA_STATUS_V2_SUCCESS) return validation;
  if (scene_handle == nullptr) return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
  int device = descriptor->device_ordinal;
  if (device < 0 && cudaGetDevice(&device) != cudaSuccess)
    return BOX3D_CUDA_STATUS_V2_CUDA_ERROR;
  if (cudaSetDevice(device) != cudaSuccess) return BOX3D_CUDA_STATUS_V2_CUDA_ERROR;
  try {
    auto scene = std::make_shared<Scene>();
    scene->device = device;
    if (!initialize_scene(*descriptor, *scene)) {
      return BOX3D_CUDA_STATUS_V2_ALLOCATION_FAILED;
    }
    box3d_cuda_scene_handle_v2 handle = g_next_scene.fetch_add(1);
    if (handle == 0) handle = g_next_scene.fetch_add(1);
    {
      std::lock_guard<std::mutex> lock(g_registry_mutex);
      g_scenes.emplace(handle, scene);
    }
    *scene_handle = handle;
    return BOX3D_CUDA_STATUS_V2_SUCCESS;
  } catch (const std::bad_alloc&) {
    return BOX3D_CUDA_STATUS_V2_ALLOCATION_FAILED;
  } catch (...) {
    return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
  }
}

extern "C" BOX3D_CUDA_API box3d_cuda_status_v2 box3d_cuda_scene_get_info_v2(
    box3d_cuda_scene_handle_v2 handle, box3d_cuda_scene_info_v2* info) {
  if (info == nullptr || info->struct_size != sizeof(*info))
    return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
  if (info->abi_version != BOX3D_CUDA_ABI_VERSION_V2)
    return BOX3D_CUDA_STATUS_V2_ABI_MISMATCH;
  const auto scene = find_scene(handle);
  if (scene == nullptr) return BOX3D_CUDA_STATUS_V2_INVALID_HANDLE;
  std::lock_guard<std::mutex> operation_lock(scene->operation_mutex);
  if (scene->retired) return BOX3D_CUDA_STATUS_V2_INVALID_HANDLE;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = sizeof(*info);
  info->abi_version = BOX3D_CUDA_ABI_VERSION_V2;
  info->environments = scene->environments;
  info->bodies = scene->bodies;
  info->joints = scene->joints;
  info->contact_pairs = scene->contact_pairs;
  std::memcpy(info->topology_sha256, scene->topology_sha256,
              sizeof(info->topology_sha256));
  return BOX3D_CUDA_STATUS_V2_SUCCESS;
}

extern "C" BOX3D_CUDA_API box3d_cuda_status_v2 box3d_cuda_scene_capture_v2(
    const box3d_cuda_scene_capture_desc_v2* descriptor) {
  if (descriptor == nullptr || descriptor->struct_size != sizeof(*descriptor))
    return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
  if (descriptor->abi_version != BOX3D_CUDA_ABI_VERSION_V2)
    return BOX3D_CUDA_STATUS_V2_ABI_MISMATCH;
  const auto scene = find_scene(descriptor->scene);
  if (scene == nullptr) return BOX3D_CUDA_STATUS_V2_INVALID_HANDLE;
  std::lock_guard<std::mutex> operation_lock(scene->operation_mutex);
  if (scene->retired) return BOX3D_CUDA_STATUS_V2_INVALID_HANDLE;
  if (std::memcmp(descriptor->topology_sha256, scene->topology_sha256,
                  sizeof(scene->topology_sha256)) != 0)
    return BOX3D_CUDA_STATUS_V2_TOPOLOGY_MISMATCH;
  if (cudaSetDevice(scene->device) != cudaSuccess)
    return BOX3D_CUDA_STATUS_V2_CUDA_ERROR;
  if (!valid_snapshot_pointers(*scene, *descriptor))
    return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(descriptor->stream);
  return enqueue_copy(descriptor->state, scene->state.data,
                      scene->state.count * sizeof(float), stream) &&
                 enqueue_copy(descriptor->inverse_mass, scene->inverse_mass.data,
                              scene->inverse_mass.count * sizeof(float), stream) &&
                 enqueue_copy(descriptor->inverse_inertia,
                              scene->inverse_inertia.data,
                              scene->inverse_inertia.count * sizeof(float), stream) &&
                 enqueue_copy(descriptor->half_extents, scene->half_extents.data,
                              scene->half_extents.count * sizeof(float), stream) &&
                 enqueue_copy(descriptor->gravity_xyz, scene->gravity_xyz.data,
                              3 * sizeof(float), stream) &&
                 enqueue_copy(descriptor->material_friction,
                              scene->material_friction.data, sizeof(float), stream) &&
                 enqueue_copy(descriptor->material_restitution,
                              scene->material_restitution.data, sizeof(float), stream) &&
                 enqueue_copy(descriptor->joint_cache, scene->joint_cache.data,
                              scene->joint_cache.count * sizeof(float), stream) &&
                 enqueue_copy(descriptor->contact_feature_ids,
                              scene->contact_feature_ids.data,
                              scene->contact_feature_ids.count * sizeof(int64_t), stream) &&
                 enqueue_copy(descriptor->contact_impulse_cache,
                              scene->contact_impulse_cache.data,
                              scene->contact_impulse_cache.count * sizeof(float), stream)
             ? BOX3D_CUDA_STATUS_V2_SUCCESS
             : BOX3D_CUDA_STATUS_V2_CUDA_ERROR;
}

extern "C" BOX3D_CUDA_API box3d_cuda_status_v2 box3d_cuda_scene_restore_v2(
    const box3d_cuda_scene_restore_desc_v2* descriptor) {
  if (descriptor == nullptr || descriptor->struct_size != sizeof(*descriptor))
    return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
  if (descriptor->abi_version != BOX3D_CUDA_ABI_VERSION_V2)
    return BOX3D_CUDA_STATUS_V2_ABI_MISMATCH;
  const auto scene = find_scene(descriptor->scene);
  if (scene == nullptr) return BOX3D_CUDA_STATUS_V2_INVALID_HANDLE;
  std::lock_guard<std::mutex> operation_lock(scene->operation_mutex);
  if (scene->retired) return BOX3D_CUDA_STATUS_V2_INVALID_HANDLE;
  if (std::memcmp(descriptor->topology_sha256, scene->topology_sha256,
                  sizeof(scene->topology_sha256)) != 0)
    return BOX3D_CUDA_STATUS_V2_TOPOLOGY_MISMATCH;
  if (cudaSetDevice(scene->device) != cudaSuccess)
    return BOX3D_CUDA_STATUS_V2_CUDA_ERROR;
  if (!valid_snapshot_pointers(*scene, *descriptor))
    return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(descriptor->stream);
  const uint8_t* mask = descriptor->environment_mask;
  const size_t bodies = scene->bodies;
  const size_t joints = scene->joints;
  const size_t pairs = scene->contact_pairs;
  const bool enqueued =
      enqueue_copy(scene->gravity_xyz.data, descriptor->gravity_xyz,
                   3 * sizeof(float), stream) &&
      enqueue_copy(scene->material_friction.data,
                   descriptor->material_friction, sizeof(float), stream) &&
      enqueue_copy(scene->material_restitution.data,
                   descriptor->material_restitution, sizeof(float), stream) &&
      enqueue_environment_copy(scene->state.data, descriptor->state,
                               bodies * 13 * sizeof(float), scene->environments,
                               mask, stream) &&
      enqueue_environment_copy(scene->inverse_mass.data,
                               descriptor->inverse_mass, bodies * sizeof(float),
                               scene->environments, mask, stream) &&
      enqueue_environment_copy(scene->inverse_inertia.data,
                               descriptor->inverse_inertia,
                               bodies * 3 * sizeof(float), scene->environments,
                               mask, stream) &&
      enqueue_environment_copy(scene->half_extents.data,
                               descriptor->half_extents,
                               bodies * 3 * sizeof(float), scene->environments,
                               mask, stream) &&
      enqueue_environment_copy(scene->joint_cache.data,
                               descriptor->joint_cache,
                               joints * 8 * sizeof(float), scene->environments,
                               mask, stream) &&
      enqueue_environment_copy(scene->contact_feature_ids.data,
                               descriptor->contact_feature_ids,
                               pairs * 4 * sizeof(int64_t), scene->environments,
                               mask, stream) &&
      enqueue_environment_copy(scene->contact_impulse_cache.data,
                               descriptor->contact_impulse_cache,
                               pairs * 12 * sizeof(float), scene->environments,
                               mask, stream);
  return enqueued ? BOX3D_CUDA_STATUS_V2_SUCCESS
                  : BOX3D_CUDA_STATUS_V2_CUDA_ERROR;
}

extern "C" BOX3D_CUDA_API box3d_cuda_status_v2 box3d_cuda_scene_step_v2(
    const box3d_cuda_scene_step_desc_v2* descriptor) {
  if (descriptor == nullptr || descriptor->struct_size != sizeof(*descriptor))
    return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
  if (descriptor->abi_version != BOX3D_CUDA_ABI_VERSION_V2)
    return BOX3D_CUDA_STATUS_V2_ABI_MISMATCH;
  if (descriptor->reserved_u32 != 0 || descriptor->steps == 0)
    return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
  const auto scene = find_scene(descriptor->scene);
  if (scene == nullptr) return BOX3D_CUDA_STATUS_V2_INVALID_HANDLE;
  std::lock_guard<std::mutex> operation_lock(scene->operation_mutex);
  if (scene->retired) return BOX3D_CUDA_STATUS_V2_INVALID_HANDLE;
  if (cudaSetDevice(scene->device) != cudaSuccess)
    return BOX3D_CUDA_STATUS_V2_CUDA_ERROR;
  if (!valid_step_pointers(*scene, *descriptor))
    return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(descriptor->stream);
  const size_t ej = static_cast<size_t>(scene->environments) * scene->joints;
  const size_t ep =
      static_cast<size_t>(scene->environments) * scene->contact_pairs;
  const bool cleared =
      enqueue_zero(scene->diagnostic_joint_coordinate.data,
                   ej * sizeof(float), stream) &&
      enqueue_zero(scene->diagnostic_joint_anchor_error.data,
                   ej * sizeof(float), stream) &&
      enqueue_zero(scene->diagnostic_joint_angular_error.data,
                   ej * sizeof(float), stream) &&
      enqueue_zero(scene->diagnostic_joint_limit_error.data,
                   ej * sizeof(float), stream) &&
      enqueue_zero(scene->diagnostic_joint_motor_impulse.data,
                   ej * sizeof(float), stream) &&
      enqueue_zero(scene->diagnostic_joint_limit_active.data,
                   ej * sizeof(uint8_t), stream) &&
      enqueue_zero(scene->diagnostic_contact_ever.data,
                   ep * sizeof(uint8_t), stream) &&
      enqueue_zero(scene->diagnostic_contact_penetration.data,
                   ep * sizeof(float), stream) &&
      enqueue_zero(scene->diagnostic_contact_count.data,
                   ep * sizeof(int32_t), stream) &&
      enqueue_zero(scene->diagnostic_contact_normal_impulse.data,
                   ep * sizeof(float), stream);
  if (!cleared) return BOX3D_CUDA_STATUS_V2_CUDA_ERROR;

  constexpr int threads = 64;
  const int blocks =
      (static_cast<int>(scene->environments) + threads - 1) / threads;
  for (uint32_t step = 0; step < descriptor->steps; ++step) {
    coupled_kernel<<<blocks, threads, 0, stream>>>(
        scene->state.data, scene->inverse_mass.data, scene->half_extents.data,
        scene->inverse_inertia.data, scene->joint_body_indices.data,
        scene->joint_types.data, scene->joint_parent_anchor.data,
        scene->joint_child_anchor.data, scene->joint_axis_parent.data,
        scene->joint_reference_xyzw.data, scene->joint_lower_limit.data,
        scene->joint_upper_limit.data, scene->joint_damping.data,
        scene->joint_motor_enabled.data, descriptor->target_velocity,
        descriptor->target_position, scene->joint_stiffness.data,
        descriptor->maximum_effort, scene->joint_cache.data,
        scene->contact_body_indices.data, scene->contact_feature_ids.data,
        scene->contact_impulse_cache.data, scene->warm_start_factor,
        scene->diagnostic_joint_coordinate.data,
        scene->diagnostic_joint_anchor_error.data,
        scene->diagnostic_joint_angular_error.data,
        scene->diagnostic_joint_limit_error.data,
        scene->diagnostic_joint_motor_impulse.data,
        scene->diagnostic_joint_limit_active.data,
        scene->diagnostic_contact_ever.data,
        scene->diagnostic_contact_penetration.data,
        scene->diagnostic_contact_count.data,
        scene->diagnostic_contact_normal_impulse.data,
        static_cast<int>(scene->environments), static_cast<int>(scene->bodies),
        static_cast<int>(scene->joints), static_cast<int>(scene->contact_pairs),
        scene->dt, static_cast<int>(scene->substeps), scene->gravity_xyz.data,
        scene->material_friction.data, scene->material_restitution.data, 0.0f,
        0.0f, 0.0f, scene->contact_slop, scene->position_correction,
        scene->angular_damping, static_cast<int>(scene->solver_iterations),
        scene->sat_epsilon, scene->joint_position_slop,
        scene->joint_angular_slop, scene->maximum_linear_repair,
        scene->maximum_angular_repair, false);
  }
  if (cudaPeekAtLastError() != cudaSuccess)
    return BOX3D_CUDA_STATUS_V2_CUDA_ERROR;

  const bool copied =
      enqueue_optional_copy(descriptor->joint_coordinate,
                            scene->diagnostic_joint_coordinate.data,
                            ej * sizeof(float), stream) &&
      enqueue_optional_copy(descriptor->joint_anchor_error,
                            scene->diagnostic_joint_anchor_error.data,
                            ej * sizeof(float), stream) &&
      enqueue_optional_copy(descriptor->joint_angular_error,
                            scene->diagnostic_joint_angular_error.data,
                            ej * sizeof(float), stream) &&
      enqueue_optional_copy(descriptor->joint_limit_error,
                            scene->diagnostic_joint_limit_error.data,
                            ej * sizeof(float), stream) &&
      enqueue_optional_copy(descriptor->joint_motor_impulse,
                            scene->diagnostic_joint_motor_impulse.data,
                            ej * sizeof(float), stream) &&
      enqueue_optional_copy(descriptor->joint_limit_active,
                            scene->diagnostic_joint_limit_active.data,
                            ej * sizeof(uint8_t), stream) &&
      enqueue_optional_copy(descriptor->contact_ever,
                            scene->diagnostic_contact_ever.data,
                            ep * sizeof(uint8_t), stream) &&
      enqueue_optional_copy(descriptor->contact_penetration,
                            scene->diagnostic_contact_penetration.data,
                            ep * sizeof(float), stream) &&
      enqueue_optional_copy(descriptor->contact_normal_impulse,
                            scene->diagnostic_contact_normal_impulse.data,
                            ep * sizeof(float), stream);
  if (!copied) return BOX3D_CUDA_STATUS_V2_CUDA_ERROR;
  if (ep != 0 &&
      (descriptor->contact_active != nullptr || descriptor->contact_count != nullptr)) {
    const int contact_blocks = (static_cast<int>(ep) + 127) / 128;
    pack_contact_final<<<contact_blocks, 128, 0, stream>>>(
        scene->diagnostic_contact_count.data, descriptor->contact_active,
        descriptor->contact_count, ep);
    if (cudaPeekAtLastError() != cudaSuccess)
      return BOX3D_CUDA_STATUS_V2_CUDA_ERROR;
  }
  return BOX3D_CUDA_STATUS_V2_SUCCESS;
}

extern "C" BOX3D_CUDA_API box3d_cuda_status_v2 box3d_cuda_scene_raycast_v2(
    const box3d_cuda_ray_query_desc_v2* descriptor) {
  if (descriptor == nullptr || descriptor->struct_size != sizeof(*descriptor))
    return BOX3D_CUDA_STATUS_V2_INVALID_ARGUMENT;
  if (descriptor->abi_version != BOX3D_CUDA_ABI_VERSION_V2)
    return BOX3D_CUDA_STATUS_V2_ABI_MISMATCH;
  return find_scene(descriptor->scene) == nullptr
             ? BOX3D_CUDA_STATUS_V2_INVALID_HANDLE
             : BOX3D_CUDA_STATUS_V2_UNSUPPORTED;
}

extern "C" BOX3D_CUDA_API box3d_cuda_status_v2 box3d_cuda_scene_unregister_v2(
    box3d_cuda_scene_handle_v2 handle) {
  std::shared_ptr<Scene> scene;
  {
    std::lock_guard<std::mutex> registry_lock(g_registry_mutex);
    const auto iterator = g_scenes.find(handle);
    if (iterator == g_scenes.end()) return BOX3D_CUDA_STATUS_V2_INVALID_HANDLE;
    scene = iterator->second;
  }
  {
    std::lock_guard<std::mutex> operation_lock(scene->operation_mutex);
    if (scene->retired) return BOX3D_CUDA_STATUS_V2_INVALID_HANDLE;
    scene->retired = true;
    if (cudaSetDevice(scene->device) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
      scene->retired = false;
      return BOX3D_CUDA_STATUS_V2_CUDA_ERROR;
    }
    std::lock_guard<std::mutex> registry_lock(g_registry_mutex);
    g_scenes.erase(handle);
  }
  return BOX3D_CUDA_STATUS_V2_SUCCESS;
}
