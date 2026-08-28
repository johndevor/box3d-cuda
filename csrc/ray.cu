// SPDX-License-Identifier: MIT
// Batched nearest-hit rays against fixed-small worlds of oriented boxes.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace {
constexpr int SW = 13;

__device__ inline void cross3(const float* a, const float* b, float* out) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ inline void rotate3(const float* q, const float* value, float* out) {
  float twice[3], cross_twice[3];
  cross3(q, value, twice);
  for (int axis = 0; axis < 3; ++axis) twice[axis] *= 2.0f;
  cross3(q, twice, cross_twice);
  for (int axis = 0; axis < 3; ++axis)
    out[axis] = value[axis] + q[3] * twice[axis] + cross_twice[axis];
}

__device__ inline bool intersect_obb(
    const float* body,
    const float* half,
    const float* origin,
    const float* direction,
    float maximum_distance,
    float* hit_distance,
    float* hit_normal_world) {
  const float conjugate[4] = {-body[3], -body[4], -body[5], body[6]};
  float relative[3] = {
      origin[0] - body[0], origin[1] - body[1], origin[2] - body[2]};
  float local_origin[3], local_direction[3];
  rotate3(conjugate, relative, local_origin);
  rotate3(conjugate, direction, local_direction);

  float near_t = -FLT_MAX, far_t = FLT_MAX;
  int near_axis = 0, far_axis = 0;
  float near_sign = 0.0f, far_sign = 0.0f;
  for (int axis = 0; axis < 3; ++axis) {
    const float component = local_direction[axis];
    if (fabsf(component) <= 1.0e-7f) {
      if (local_origin[axis] < -half[axis] || local_origin[axis] > half[axis])
        return false;
      continue;
    }
    const float first = (-half[axis] - local_origin[axis]) / component;
    const float second = (half[axis] - local_origin[axis]) / component;
    const float slab_near = fminf(first, second);
    const float slab_far = fmaxf(first, second);
    const float slab_near_sign = first <= second ? -1.0f : 1.0f;
    const float slab_far_sign = -slab_near_sign;
    if (slab_near > near_t) {
      near_t = slab_near;
      near_axis = axis;
      near_sign = slab_near_sign;
    }
    if (slab_far < far_t) {
      far_t = slab_far;
      far_axis = axis;
      far_sign = slab_far_sign;
    }
    if (near_t > far_t) return false;
  }
  if (far_t < 0.0f) return false;
  const bool inside = near_t < 0.0f;
  const float distance = inside ? far_t : near_t;
  if (distance < 0.0f || distance > maximum_distance) return false;
  float local_normal[3] = {0.0f, 0.0f, 0.0f};
  local_normal[inside ? far_axis : near_axis] = inside ? far_sign : near_sign;
  rotate3(body + 3, local_normal, hit_normal_world);
  const float normal_length = sqrtf(
      hit_normal_world[0] * hit_normal_world[0] +
      hit_normal_world[1] * hit_normal_world[1] +
      hit_normal_world[2] * hit_normal_world[2]);
  if (normal_length <= 1.0e-12f || !isfinite(normal_length)) return false;
  for (int axis = 0; axis < 3; ++axis) hit_normal_world[axis] /= normal_length;
  *hit_distance = distance;
  return true;
}

__global__ void ray_kernel(
    const float* state,
    const float* half_extents,
    const uint8_t* body_enabled,
    const float* ray_origins,
    const float* ray_directions,
    const float* maximum_distance,
    float* distance,
    int64_t* body_index,
    float* normal,
    int worlds,
    int bodies,
    int rays) {
  const int flat_ray = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = worlds * rays;
  if (flat_ray >= total) return;
  const int world = flat_ray / rays;
  const float* origin = ray_origins + flat_ray * 3;
  const float* direction = ray_directions + flat_ray * 3;
  const float max_t = maximum_distance[flat_ray];
  float best = max_t;
  int64_t best_body = -1;
  float best_normal[3] = {0.0f, 0.0f, 0.0f};
  for (int body = 0; body < bodies; ++body) {
    const int flat_body = world * bodies + body;
    if (!body_enabled[flat_body]) continue;
    float candidate = max_t, candidate_normal[3];
    if (intersect_obb(
            state + flat_body * SW,
            half_extents + flat_body * 3,
            origin,
            direction,
            max_t,
            &candidate,
            candidate_normal) &&
        (best_body < 0 || candidate < best - 1.0e-7f)) {
      best = candidate;
      best_body = body;
      for (int axis = 0; axis < 3; ++axis) best_normal[axis] = candidate_normal[axis];
    }
  }
  distance[flat_ray] = best;
  body_index[flat_ray] = best_body;
  for (int axis = 0; axis < 3; ++axis)
    normal[flat_ray * 3 + axis] = best_normal[axis];
}
}  // namespace

std::vector<torch::Tensor> box3d_ray_cast_cuda(
    torch::Tensor state,
    torch::Tensor half_extents,
    torch::Tensor body_enabled,
    torch::Tensor ray_origins,
    torch::Tensor ray_directions,
    torch::Tensor maximum_distance) {
  const c10::cuda::CUDAGuard guard(state.device());
  const int worlds = state.size(0);
  const int bodies = state.size(1);
  const int rays = ray_origins.size(1);
  auto distance = torch::empty({worlds, rays}, state.options());
  auto body_index = torch::empty(
      {worlds, rays}, state.options().dtype(torch::kInt64));
  auto normal = torch::empty({worlds, rays, 3}, state.options());
  constexpr int threads = 256;
  const int total = worlds * rays;
  ray_kernel<<<(total + threads - 1) / threads, threads, 0,
               at::cuda::getDefaultCUDAStream()>>>(
      state.data_ptr<float>(), half_extents.data_ptr<float>(),
      body_enabled.data_ptr<uint8_t>(), ray_origins.data_ptr<float>(),
      ray_directions.data_ptr<float>(), maximum_distance.data_ptr<float>(),
      distance.data_ptr<float>(), body_index.data_ptr<int64_t>(),
      normal.data_ptr<float>(), worlds, bodies, rays);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {distance, body_index, normal};
}
