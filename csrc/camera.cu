// SPDX-License-Identifier: MIT
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
constexpr int STATE_WIDTH = 13;

__device__ inline void cross3(
    const float* left, const float* right, float* output) {
  output[0] = left[1] * right[2] - left[2] * right[1];
  output[1] = left[2] * right[0] - left[0] * right[2];
  output[2] = left[0] * right[1] - left[1] * right[0];
}

__device__ inline void rotate3(
    const float* quaternion_xyzw, const float* vector, float* output) {
  float twice_cross[3];
  cross3(quaternion_xyzw, vector, twice_cross);
  for (int axis = 0; axis < 3; ++axis) twice_cross[axis] *= 2.0f;
  float nested_cross[3];
  cross3(quaternion_xyzw, twice_cross, nested_cross);
  for (int axis = 0; axis < 3; ++axis) {
    output[axis] = vector[axis] +
        quaternion_xyzw[3] * twice_cross[axis] + nested_cross[axis];
  }
}

__device__ inline void multiply_quaternion_xyzw(
    const float* left, const float* right, float* output) {
  const float ax = left[0], ay = left[1], az = left[2], aw = left[3];
  const float bx = right[0], by = right[1], bz = right[2], bw = right[3];
  output[0] = aw * bx + ax * bw + ay * bz - az * by;
  output[1] = aw * by - ax * bz + ay * bw + az * bx;
  output[2] = aw * bz + ax * by - ay * bx + az * bw;
  output[3] = aw * bw - ax * bx - ay * by - az * bz;
}

__global__ void camera_ray_kernel(
    const float* state,
    const int64_t* parent_body,
    const float* position_parent,
    const float* quaternion_parent_from_camera,
    const float* intrinsics,
    const int64_t* pixel_camera,
    const int64_t* pixel_xy,
    float* origins,
    float* directions,
    float* maximum_distance,
    float* forward_cosine,
    int worlds,
    int bodies,
    int rays) {
  const int flat_ray = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = worlds * rays;
  if (flat_ray >= total) return;
  const int world = flat_ray / rays;
  const int ray = flat_ray - world * rays;
  const int camera = static_cast<int>(pixel_camera[ray]);
  const int64_t parent = parent_body[camera];
  const float* local_position = position_parent + camera * 3;
  const float* local_quaternion = quaternion_parent_from_camera + camera * 4;
  float world_position[3];
  float world_quaternion[4];
  if (parent < 0) {
    for (int axis = 0; axis < 3; ++axis)
      world_position[axis] = local_position[axis];
    for (int axis = 0; axis < 4; ++axis)
      world_quaternion[axis] = local_quaternion[axis];
  } else {
    const float* body = state + (world * bodies + parent) * STATE_WIDTH;
    float rotated_position[3];
    rotate3(body + 3, local_position, rotated_position);
    for (int axis = 0; axis < 3; ++axis)
      world_position[axis] = body[axis] + rotated_position[axis];
    multiply_quaternion_xyzw(body + 3, local_quaternion, world_quaternion);
  }
  const float* camera_intrinsics = intrinsics + camera * 5;
  const float local_x =
      (static_cast<float>(pixel_xy[ray * 2]) + 0.5f - camera_intrinsics[2]) /
      camera_intrinsics[0];
  const float local_y =
      (static_cast<float>(pixel_xy[ray * 2 + 1]) + 0.5f - camera_intrinsics[3]) /
      camera_intrinsics[1];
  const float inverse_length = rsqrtf(local_x * local_x + local_y * local_y + 1.0f);
  const float local_direction[3] = {
      local_x * inverse_length, local_y * inverse_length, inverse_length};
  float world_direction[3];
  rotate3(world_quaternion, local_direction, world_direction);
  const float world_inverse_length = rsqrtf(
      world_direction[0] * world_direction[0] +
      world_direction[1] * world_direction[1] +
      world_direction[2] * world_direction[2]);
  for (int axis = 0; axis < 3; ++axis) {
    origins[flat_ray * 3 + axis] = world_position[axis];
    directions[flat_ray * 3 + axis] =
        world_direction[axis] * world_inverse_length;
  }
  maximum_distance[flat_ray] = camera_intrinsics[4];
  forward_cosine[flat_ray] = inverse_length;
}

__global__ void camera_depth_kernel(
    const float* distance,
    const int64_t* body_index,
    const float* forward_cosine,
    float* depth_z,
    float* hit_range,
    int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) return;
  if (body_index[index] < 0) {
    depth_z[index] = 0.0f;
    hit_range[index] = 0.0f;
  } else {
    hit_range[index] = distance[index];
    depth_z[index] = distance[index] * forward_cosine[index];
  }
}
}  // namespace

std::vector<torch::Tensor> box3d_camera_rays_cuda(
    torch::Tensor state,
    torch::Tensor parent_body,
    torch::Tensor position_parent,
    torch::Tensor quaternion_parent_from_camera,
    torch::Tensor intrinsics,
    torch::Tensor pixel_camera,
    torch::Tensor pixel_xy) {
  const c10::cuda::CUDAGuard guard(state.device());
  const int worlds = state.size(0);
  const int bodies = state.size(1);
  const int rays = pixel_camera.size(0);
  auto origins = torch::empty({worlds, rays, 3}, state.options());
  auto directions = torch::empty({worlds, rays, 3}, state.options());
  auto maximum_distance = torch::empty({worlds, rays}, state.options());
  auto forward_cosine = torch::empty({worlds, rays}, state.options());
  constexpr int threads = 256;
  const int total = worlds * rays;
  camera_ray_kernel<<<(total + threads - 1) / threads, threads, 0,
                      at::cuda::getDefaultCUDAStream()>>>(
      state.data_ptr<float>(), parent_body.data_ptr<int64_t>(),
      position_parent.data_ptr<float>(),
      quaternion_parent_from_camera.data_ptr<float>(),
      intrinsics.data_ptr<float>(), pixel_camera.data_ptr<int64_t>(),
      pixel_xy.data_ptr<int64_t>(), origins.data_ptr<float>(),
      directions.data_ptr<float>(), maximum_distance.data_ptr<float>(),
      forward_cosine.data_ptr<float>(), worlds, bodies, rays);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {origins, directions, maximum_distance, forward_cosine};
}

std::vector<torch::Tensor> box3d_camera_depth_cuda(
    torch::Tensor distance,
    torch::Tensor body_index,
    torch::Tensor forward_cosine) {
  const c10::cuda::CUDAGuard guard(distance.device());
  auto depth_z = torch::empty_like(distance);
  auto hit_range = torch::empty_like(distance);
  constexpr int threads = 256;
  const int total = distance.numel();
  camera_depth_kernel<<<(total + threads - 1) / threads, threads, 0,
                        at::cuda::getDefaultCUDAStream()>>>(
      distance.data_ptr<float>(), body_index.data_ptr<int64_t>(),
      forward_cosine.data_ptr<float>(), depth_z.data_ptr<float>(),
      hit_range.data_ptr<float>(), total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {depth_z, hit_range};
}
