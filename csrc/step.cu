// SPDX-License-Identifier: MIT
// Initial CUDA mapping from Box3D commit 30c67b5. This is a deliberately
// bounded port slice: rigid integration, plane contacts and sphere contacts.
#ifndef BOX3D_CUDA_NATIVE_KERNELS_ONLY
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kStateWidth = 13;

__device__ inline void integrate_quaternion(float* body, float h) {
  // CUDA spelling of Box3D b3IntegrateRotation:
  // q2 = normalize(q1 + 0.5 * deltaRotation * q1).
  const float qx = body[3], qy = body[4], qz = body[5], qw = body[6];
  const float dx = body[10] * h, dy = body[11] * h, dz = body[12] * h;
  const float rx = 0.5f * (dx * qw + dy * qz - dz * qy);
  const float ry = 0.5f * (-dx * qz + dy * qw + dz * qx);
  const float rz = 0.5f * (dx * qy - dy * qx + dz * qw);
  const float rw = -0.5f * (dx * qx + dy * qy + dz * qz);
  float x = qx + rx, y = qy + ry, z = qz + rz, w = qw + rw;
  const float inverse_length = rsqrtf(x * x + y * y + z * z + w * w);
  body[3] = x * inverse_length;
  body[4] = y * inverse_length;
  body[5] = z * inverse_length;
  body[6] = w * inverse_length;
}

__global__ void step_worlds(
    float* state,
    const float* inverse_mass,
    const float* radius,
    int worlds,
    int bodies,
    float dt,
    int substeps,
    float gravity_y,
    float restitution,
    float friction) {
  const int world = blockIdx.x * blockDim.x + threadIdx.x;
  if (world >= worlds) return;
  const int material_base = world * bodies;
  float* world_state = state + material_base * kStateWidth;
  const float h = dt / static_cast<float>(substeps);
  constexpr float slop = 1.0e-4f;

  for (int substep = 0; substep < substeps; ++substep) {
    for (int index = 0; index < bodies; ++index) {
      if (inverse_mass[material_base + index] == 0.0f) continue;
      float* body = world_state + index * kStateWidth;
      body[8] += gravity_y * h;
      body[0] += body[7] * h;
      body[1] += body[8] * h;
      body[2] += body[9] * h;
      integrate_quaternion(body, h);
    }

    // Infinite static plane at y=0. Impulses are physical, not attachment logic.
    for (int index = 0; index < bodies; ++index) {
      if (inverse_mass[material_base + index] == 0.0f) continue;
      float* body = world_state + index * kStateWidth;
      const float penetration = radius[material_base + index] - body[1];
      if (penetration <= 0.0f) continue;
      body[1] += fmaxf(0.0f, penetration - slop);
      if (body[8] < 0.0f) {
        const float normal_delta = -(1.0f + restitution) * body[8];
        body[8] += normal_delta;
        const float tangent_speed = hypotf(body[7], body[9]);
        if (tangent_speed > 0.0f) {
          const float reduction = fminf(tangent_speed, friction * normal_delta);
          const float scale = (tangent_speed - reduction) / tangent_speed;
          body[7] *= scale;
          body[9] *= scale;
        }
      }
    }

    // Fixed-small-world narrow phase. One thread owns a world, avoiding atomics
    // while the topology is tiny; broad phase and graph coloring are later ports.
    for (int a = 0; a < bodies; ++a) {
      float* body_a = world_state + a * kStateWidth;
      for (int b = a + 1; b < bodies; ++b) {
        float* body_b = world_state + b * kStateWidth;
        const float dx = body_b[0] - body_a[0];
        const float dy = body_b[1] - body_a[1];
        const float dz = body_b[2] - body_a[2];
        const float distance2 = dx * dx + dy * dy + dz * dz;
        const float target = radius[material_base + a] + radius[material_base + b];
        if (distance2 >= target * target || distance2 <= 1.0e-16f) continue;
        const float distance = sqrtf(distance2);
        const float nx = dx / distance, ny = dy / distance, nz = dz / distance;
        const float inverse_sum = inverse_mass[material_base + a] + inverse_mass[material_base + b];
        if (inverse_sum == 0.0f) continue;
        const float correction = fmaxf(0.0f, target - distance - slop) / inverse_sum;
        const float mass_a = inverse_mass[material_base + a];
        const float mass_b = inverse_mass[material_base + b];
        body_a[0] -= nx * correction * mass_a;
        body_a[1] -= ny * correction * mass_a;
        body_a[2] -= nz * correction * mass_a;
        body_b[0] += nx * correction * mass_b;
        body_b[1] += ny * correction * mass_b;
        body_b[2] += nz * correction * mass_b;
        const float relative_normal =
            (body_b[7] - body_a[7]) * nx +
            (body_b[8] - body_a[8]) * ny +
            (body_b[9] - body_a[9]) * nz;
        if (relative_normal >= 0.0f) continue;
        const float impulse = -(1.0f + restitution) * relative_normal / inverse_sum;
        body_a[7] -= nx * impulse * mass_a;
        body_a[8] -= ny * impulse * mass_a;
        body_a[9] -= nz * impulse * mass_a;
        body_b[7] += nx * impulse * mass_b;
        body_b[8] += ny * impulse * mass_b;
        body_b[9] += nz * impulse * mass_b;
      }
    }
  }
}

}  // namespace

#ifndef BOX3D_CUDA_NATIVE_KERNELS_ONLY
torch::Tensor box3d_step_cuda(
    torch::Tensor state,
    torch::Tensor inverse_mass,
    torch::Tensor radius,
    double dt,
    int64_t substeps,
    double gravity_y,
    double restitution,
    double friction) {
  const c10::cuda::CUDAGuard device_guard(state.device());
  torch::Tensor output = state.clone();
  const int worlds = static_cast<int>(state.size(0));
  const int bodies = static_cast<int>(state.size(1));
  constexpr int threads = 128;
  const int blocks = (worlds + threads - 1) / threads;
  step_worlds<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      output.data_ptr<float>(), inverse_mass.data_ptr<float>(), radius.data_ptr<float>(),
      worlds, bodies, static_cast<float>(dt), static_cast<int>(substeps),
      static_cast<float>(gravity_y), static_cast<float>(restitution),
      static_cast<float>(friction));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
#endif
