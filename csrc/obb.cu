// SPDX-License-Identifier: MIT
// Stage-2 oriented-box contact slice: quaternion rotation, world-space inertia,
// vertex/plane manifolds, angular impulses and two-axis Coulomb friction.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kStateWidth = 13;

__device__ inline void cross3(const float* a, const float* b, float* out) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ inline float dot3(const float* a, const float* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ inline void rotate3(const float* q, const float* vector, float* out) {
  const float qv[3] = {q[0], q[1], q[2]};
  float twice[3];
  cross3(qv, vector, twice);
  twice[0] *= 2.0f; twice[1] *= 2.0f; twice[2] *= 2.0f;
  float again[3];
  cross3(qv, twice, again);
  out[0] = vector[0] + q[3] * twice[0] + again[0];
  out[1] = vector[1] + q[3] * twice[1] + again[1];
  out[2] = vector[2] + q[3] * twice[2] + again[2];
}

__device__ inline void inverse_inertia_world(
    const float* q, const float* diagonal, const float* vector, float* out) {
  const float conjugate[4] = {-q[0], -q[1], -q[2], q[3]};
  float local[3];
  rotate3(conjugate, vector, local);
  local[0] *= diagonal[0]; local[1] *= diagonal[1]; local[2] *= diagonal[2];
  rotate3(q, local, out);
}

__device__ inline void integrate_quaternion(float* body, float h) {
  const float qx = body[3], qy = body[4], qz = body[5], qw = body[6];
  const float dx = body[10] * h, dy = body[11] * h, dz = body[12] * h;
  const float rx = 0.5f * (dx * qw + dy * qz - dz * qy);
  const float ry = 0.5f * (-dx * qz + dy * qw + dz * qx);
  const float rz = 0.5f * (dx * qy - dy * qx + dz * qw);
  const float rw = -0.5f * (dx * qx + dy * qy + dz * qz);
  float x = qx + rx, y = qy + ry, z = qz + rz, w = qw + rw;
  const float inverse_length = rsqrtf(x * x + y * y + z * z + w * w);
  body[3] = x * inverse_length; body[4] = y * inverse_length;
  body[5] = z * inverse_length; body[6] = w * inverse_length;
}

__device__ inline float effective_mass(
    const float* q, float inverse_mass, const float* inverse_inertia,
    const float* r, const float* direction) {
  float torque_axis[3], angular[3], induced[3];
  cross3(r, direction, torque_axis);
  inverse_inertia_world(q, inverse_inertia, torque_axis, angular);
  cross3(angular, r, induced);
  return inverse_mass + dot3(induced, direction);
}

__device__ inline void apply_impulse(
    float* body, const float* q, float inverse_mass, const float* inverse_inertia,
    const float* r, const float* impulse) {
  body[7] += impulse[0] * inverse_mass;
  body[8] += impulse[1] * inverse_mass;
  body[9] += impulse[2] * inverse_mass;
  float torque[3], angular[3];
  cross3(r, impulse, torque);
  inverse_inertia_world(q, inverse_inertia, torque, angular);
  body[10] += angular[0]; body[11] += angular[1]; body[12] += angular[2];
}

__device__ inline void solve_plane(
    float* body,
    float inverse_mass,
    const float* half,
    const float* inverse_inertia,
    float restitution,
    float friction,
    float slop,
    int solver_iterations,
    uint8_t* touched,
    float* minimum_clearance) {
  const float q[4] = {body[3], body[4], body[5], body[6]};
  float corners[8][3];
  float minimum = 1.0e30f;
  int corner_index = 0;
  for (int sx = -1; sx <= 1; sx += 2) {
    for (int sy = -1; sy <= 1; sy += 2) {
      for (int sz = -1; sz <= 1; sz += 2) {
        const float local[3] = {
            sx * half[0], sy * half[1], sz * half[2]};
        rotate3(q, local, corners[corner_index]);
        minimum = fminf(minimum, body[1] + corners[corner_index][1]);
        ++corner_index;
      }
    }
  }
  *minimum_clearance = fminf(*minimum_clearance, minimum);
  if (minimum >= 0.0f) return;
  body[1] += fmaxf(0.0f, -minimum - slop);
  const float normal[3] = {0.0f, 1.0f, 0.0f};
  bool active[8];
  for (int corner = 0; corner < 8; ++corner) {
    active[corner] = body[1] + corners[corner][1] <= slop * 2.0f;
  }
  for (int iteration = 0; iteration < solver_iterations; ++iteration) {
    for (int corner = 0; corner < 8; ++corner) {
      if (!active[corner]) continue;
      const float* r = corners[corner];
      const float omega[3] = {body[10], body[11], body[12]};
      float rotational[3];
      cross3(omega, r, rotational);
      float point_velocity[3] = {
          body[7] + rotational[0], body[8] + rotational[1], body[9] + rotational[2]};
      if (point_velocity[1] >= 0.0f) continue;
      const float denominator = effective_mass(q, inverse_mass, inverse_inertia, r, normal);
      const float normal_impulse = -(1.0f + restitution) * point_velocity[1] / denominator;
      const float normal_vector[3] = {0.0f, normal_impulse, 0.0f};
      apply_impulse(body, q, inverse_mass, inverse_inertia, r, normal_vector);
      *touched = 1;

      const float updated_omega[3] = {body[10], body[11], body[12]};
      cross3(updated_omega, r, rotational);
      point_velocity[0] = body[7] + rotational[0];
      point_velocity[2] = body[9] + rotational[2];
      const float tangent_speed = hypotf(point_velocity[0], point_velocity[2]);
      if (tangent_speed <= 1.0e-12f) continue;
      const float direction[3] = {
          point_velocity[0] / tangent_speed, 0.0f, point_velocity[2] / tangent_speed};
      const float tangent_mass = effective_mass(q, inverse_mass, inverse_inertia, r, direction);
      const float tangent_impulse = fminf(tangent_speed / tangent_mass, friction * normal_impulse);
      const float friction_vector[3] = {
          -direction[0] * tangent_impulse, 0.0f, -direction[2] * tangent_impulse};
      apply_impulse(body, q, inverse_mass, inverse_inertia, r, friction_vector);
    }
  }
}

__global__ void step_obb_worlds(
    float* state,
    const float* inverse_mass,
    const float* half_extents,
    const float* inverse_inertia,
    uint8_t* contacts,
    float* minimum_clearance,
    int worlds,
    int bodies,
    float dt,
    int substeps,
    float gravity_y,
    float restitution,
    float friction,
    float slop,
    float angular_damping,
    int solver_iterations) {
  const int world = blockIdx.x * blockDim.x + threadIdx.x;
  if (world >= worlds) return;
  const float h = dt / static_cast<float>(substeps);
  const float damping = fmaxf(0.0f, 1.0f - angular_damping * h);
  float world_minimum = 1.0e30f;
  for (int substep = 0; substep < substeps; ++substep) {
    for (int body_index = 0; body_index < bodies; ++body_index) {
      const int flat = world * bodies + body_index;
      if (inverse_mass[flat] == 0.0f) continue;
      float* body = state + flat * kStateWidth;
      body[8] += gravity_y * h;
      body[0] += body[7] * h; body[1] += body[8] * h; body[2] += body[9] * h;
      body[10] *= damping; body[11] *= damping; body[12] *= damping;
      integrate_quaternion(body, h);
      solve_plane(
          body, inverse_mass[flat], half_extents + flat * 3, inverse_inertia + flat * 3,
          restitution, friction, slop, solver_iterations, contacts + flat, &world_minimum);
    }
  }
  minimum_clearance[world] = world_minimum;
}

}  // namespace

std::vector<torch::Tensor> box3d_obb_step_cuda(
    torch::Tensor state,
    torch::Tensor inverse_mass,
    torch::Tensor half_extents,
    torch::Tensor inverse_inertia,
    double dt,
    int64_t substeps,
    double gravity_y,
    double restitution,
    double friction,
    double slop,
    double angular_damping,
    int64_t solver_iterations) {
  const c10::cuda::CUDAGuard device_guard(state.device());
  torch::Tensor output = state.clone();
  torch::Tensor contacts = torch::zeros(
      {state.size(0), state.size(1)}, state.options().dtype(torch::kUInt8));
  torch::Tensor minimum = torch::empty({state.size(0)}, state.options());
  const int worlds = static_cast<int>(state.size(0));
  const int bodies = static_cast<int>(state.size(1));
  constexpr int threads = 128;
  const int blocks = (worlds + threads - 1) / threads;
  step_obb_worlds<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      output.data_ptr<float>(), inverse_mass.data_ptr<float>(), half_extents.data_ptr<float>(),
      inverse_inertia.data_ptr<float>(), contacts.data_ptr<uint8_t>(), minimum.data_ptr<float>(),
      worlds, bodies, static_cast<float>(dt), static_cast<int>(substeps),
      static_cast<float>(gravity_y), static_cast<float>(restitution), static_cast<float>(friction),
      static_cast<float>(slop), static_cast<float>(angular_damping),
      static_cast<int>(solver_iterations));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, contacts, minimum};
}
