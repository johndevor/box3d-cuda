// SPDX-License-Identifier: MIT
// Isolated fixed-base planar two-link contact-response micro. This kernel is
// not called by the production coupled solver until its oracle parity and
// matched-backend impact behavior are independently accepted.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {
constexpr int PROPERTY_WIDTH = 7;
constexpr int RESPONSE_WIDTH = 9;

__device__ inline float dot2(float ax, float ay, float bx, float by) {
  return ax * bx + ay * by;
}

__global__ void articulation_response_kernel(
    const float *base, const float *second, const float *centers,
    const float *contact, const float *normal, const float *properties,
    float *output, int worlds) {
  const int world = blockIdx.x * blockDim.x + threadIdx.x;
  if (world >= worlds) return;
  const float *b = base + world * 2;
  const float *s = second + world * 2;
  const float *c1 = centers + world * 4;
  const float *c2 = c1 + 2;
  const float *p = contact + world * 2;
  const float *n = normal + world * 2;
  const float *value = properties + world * PROPERTY_WIDTH;
  const float mass1 = value[0], mass2 = value[1];
  const float inertia1 = value[2], inertia2 = value[3];
  const float other_inverse = value[4], relative_speed = value[5];
  const float restitution = value[6];

  const float j1x = -(c1[1] - b[1]), j1y = c1[0] - b[0];
  const float j21x = -(c2[1] - b[1]), j21y = c2[0] - b[0];
  const float j22x = -(c2[1] - s[1]), j22y = c2[0] - s[0];
  const float m00 = mass1 * dot2(j1x, j1y, j1x, j1y) + inertia1
      + mass2 * dot2(j21x, j21y, j21x, j21y) + inertia2;
  const float m01 = mass2 * dot2(j21x, j21y, j22x, j22y) + inertia2;
  const float m11 = mass2 * dot2(j22x, j22y, j22x, j22y) + inertia2;
  const float determinant = m00 * m11 - m01 * m01;

  const float contact_j1x = -(p[1] - b[1]), contact_j1y = p[0] - b[0];
  const float contact_j2x = -(p[1] - s[1]), contact_j2y = p[0] - s[0];
  const float jacobian0 = dot2(n[0], n[1], contact_j1x, contact_j1y);
  const float jacobian1 = dot2(n[0], n[1], contact_j2x, contact_j2y);
  const float inverse_jacobian0 = (m11 * jacobian0 - m01 * jacobian1) / determinant;
  const float inverse_jacobian1 = (-m01 * jacobian0 + m00 * jacobian1) / determinant;
  const float articulated_inverse = fmaxf(
      0.0f, jacobian0 * inverse_jacobian0 + jacobian1 * inverse_jacobian1);

  const float offset_x = p[0] - c2[0], offset_y = p[1] - c2[1];
  const float angular_jacobian = offset_x * n[1] - offset_y * n[0];
  const float free_inverse = 1.0f / mass2
      + angular_jacobian * angular_jacobian / inertia2;
  const float numerator = fmaxf(0.0f, -(1.0f + restitution) * relative_speed);
  const float articulated_denominator = articulated_inverse + other_inverse;
  const float free_denominator = free_inverse + other_inverse;
  const float articulated_impulse = articulated_denominator > 1.0e-12f
      ? numerator / articulated_denominator : 0.0f;
  const float free_impulse = free_denominator > 1.0e-12f
      ? numerator / free_denominator : 0.0f;
  const float scale = free_impulse > 0.0f ? articulated_impulse / free_impulse : 1.0f;

  float *result = output + world * RESPONSE_WIDTH;
  result[0] = articulated_inverse;
  result[1] = free_inverse;
  result[2] = other_inverse;
  result[3] = articulated_impulse;
  result[4] = free_impulse;
  result[5] = -inverse_jacobian0 * articulated_impulse;
  result[6] = -inverse_jacobian1 * articulated_impulse;
  result[7] = determinant;
  result[8] = scale;
}
}  // namespace

torch::Tensor box3d_articulation_response_cuda(
    torch::Tensor base_joint_xy, torch::Tensor second_joint_xy,
    torch::Tensor link_centers_xy, torch::Tensor contact_point_xy,
    torch::Tensor normal_xy, torch::Tensor properties) {
  const c10::cuda::CUDAGuard guard(base_joint_xy.device());
  const int worlds = base_joint_xy.size(0);
  auto output = torch::empty({worlds, RESPONSE_WIDTH}, base_joint_xy.options());
  constexpr int threads = 128;
  articulation_response_kernel<<<
      (worlds + threads - 1) / threads, threads, 0,
      at::cuda::getDefaultCUDAStream()>>>(
      base_joint_xy.data_ptr<float>(), second_joint_xy.data_ptr<float>(),
      link_centers_xy.data_ptr<float>(), contact_point_xy.data_ptr<float>(),
      normal_xy.data_ptr<float>(), properties.data_ptr<float>(),
      output.data_ptr<float>(), worlds);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
