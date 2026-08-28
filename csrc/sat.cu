// SPDX-License-Identifier: MIT
// Fixed-small-world oriented box pairs: all 15 SAT axes, world-space inertia,
// angular contact velocity, restitution, Coulomb friction, and position repair.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kStateWidth = 13;

__device__ inline float dot3(const float* a, const float* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ inline void cross3(const float* a, const float* b, float* out) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
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

__device__ inline void rotation_axes(const float* q, float axes[3][3]) {
  const float x[3] = {1.0f, 0.0f, 0.0f};
  const float y[3] = {0.0f, 1.0f, 0.0f};
  const float z[3] = {0.0f, 0.0f, 1.0f};
  rotate3(q, x, axes[0]);
  rotate3(q, y, axes[1]);
  rotate3(q, z, axes[2]);
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
  const float length_squared = x * x + y * y + z * z + w * w;
  if (length_squared <= 1.0e-20f) {
    body[3] = 0.0f; body[4] = 0.0f; body[5] = 0.0f; body[6] = 1.0f;
    return;
  }
  const float inverse_length = rsqrtf(length_squared);
  body[3] = x * inverse_length; body[4] = y * inverse_length;
  body[5] = z * inverse_length; body[6] = w * inverse_length;
}

__device__ inline float projection_radius(
    const float axes[3][3], const float* half, const float* direction) {
  return half[0] * fabsf(dot3(axes[0], direction)) +
         half[1] * fabsf(dot3(axes[1], direction)) +
         half[2] * fabsf(dot3(axes[2], direction));
}

// Returns false only when this candidate is a separating axis. Degenerate
// edge-edge cross products are deliberately ignored rather than normalized.
__device__ inline bool test_axis(
    const float* candidate,
    bool cross_axis,
    const float* center_delta,
    const float axes_a[3][3],
    const float axes_b[3][3],
    const float* half_a,
    const float* half_b,
    float sat_epsilon,
    float* minimum_overlap,
    float* minimum_axis) {
  const float length_squared = dot3(candidate, candidate);
  const float epsilon_squared = sat_epsilon * sat_epsilon;
  if (cross_axis && length_squared <= epsilon_squared) return true;
  if (length_squared <= 1.0e-20f) return true;
  const float inverse_length = rsqrtf(length_squared);
  float axis[3] = {
      candidate[0] * inverse_length,
      candidate[1] * inverse_length,
      candidate[2] * inverse_length};
  const float signed_distance = dot3(center_delta, axis);
  const float overlap = projection_radius(axes_a, half_a, axis) +
                        projection_radius(axes_b, half_b, axis) - fabsf(signed_distance);
  if (overlap < -sat_epsilon) return false;
  const float clamped_overlap = fmaxf(0.0f, overlap);
  if (clamped_overlap < *minimum_overlap) {
    *minimum_overlap = clamped_overlap;
    const float sign = signed_distance < 0.0f ? -1.0f : 1.0f;
    minimum_axis[0] = axis[0] * sign;
    minimum_axis[1] = axis[1] * sign;
    minimum_axis[2] = axis[2] * sign;
  }
  return true;
}

__device__ inline bool sat_overlap(
    const float* body_a,
    const float* half_a,
    const float* body_b,
    const float* half_b,
    float sat_epsilon,
    float* penetration,
    float* normal,
    float axes_a[3][3],
    float axes_b[3][3]) {
  rotation_axes(body_a + 3, axes_a);
  rotation_axes(body_b + 3, axes_b);
  const float center_delta[3] = {
      body_b[0] - body_a[0], body_b[1] - body_a[1], body_b[2] - body_a[2]};
  float minimum_overlap = 1.0e30f;
  float minimum_axis[3] = {1.0f, 0.0f, 0.0f};

  // Six face normals.
  for (int axis = 0; axis < 3; ++axis) {
    if (!test_axis(axes_a[axis], false, center_delta, axes_a, axes_b,
                   half_a, half_b, sat_epsilon, &minimum_overlap, minimum_axis)) return false;
  }
  for (int axis = 0; axis < 3; ++axis) {
    if (!test_axis(axes_b[axis], false, center_delta, axes_a, axes_b,
                   half_a, half_b, sat_epsilon, &minimum_overlap, minimum_axis)) return false;
  }
  // Nine edge cross products. Near-parallel pairs have no reliable direction
  // and are skipped using the explicit SAT epsilon threshold.
  for (int axis_a = 0; axis_a < 3; ++axis_a) {
    for (int axis_b = 0; axis_b < 3; ++axis_b) {
      float candidate[3];
      cross3(axes_a[axis_a], axes_b[axis_b], candidate);
      if (!test_axis(candidate, true, center_delta, axes_a, axes_b,
                     half_a, half_b, sat_epsilon, &minimum_overlap, minimum_axis)) return false;
    }
  }
  *penetration = minimum_overlap;
  normal[0] = minimum_axis[0]; normal[1] = minimum_axis[1]; normal[2] = minimum_axis[2];
  return true;
}

__device__ inline void support_point(
    const float* body,
    const float axes[3][3],
    const float* half,
    const float* direction,
    float epsilon,
    float* out) {
  out[0] = body[0]; out[1] = body[1]; out[2] = body[2];
  for (int axis = 0; axis < 3; ++axis) {
    const float alignment = dot3(axes[axis], direction);
    const float sign = alignment > epsilon ? 1.0f : (alignment < -epsilon ? -1.0f : 0.0f);
    out[0] += axes[axis][0] * half[axis] * sign;
    out[1] += axes[axis][1] * half[axis] * sign;
    out[2] += axes[axis][2] * half[axis] * sign;
  }
}

__device__ inline void point_velocity(const float* body, const float* r, float* out) {
  const float omega[3] = {body[10], body[11], body[12]};
  float rotational[3];
  cross3(omega, r, rotational);
  out[0] = body[7] + rotational[0];
  out[1] = body[8] + rotational[1];
  out[2] = body[9] + rotational[2];
}

__device__ inline float angular_effective_mass(
    const float* q,
    const float* inverse_inertia,
    const float* r,
    const float* direction) {
  float torque_axis[3], angular[3], induced[3];
  cross3(r, direction, torque_axis);
  inverse_inertia_world(q, inverse_inertia, torque_axis, angular);
  cross3(angular, r, induced);
  return dot3(induced, direction);
}

__device__ inline void apply_impulse(
    float* body,
    float inverse_mass,
    const float* inverse_inertia,
    const float* r,
    const float* impulse) {
  if (inverse_mass == 0.0f) return;
  body[7] += impulse[0] * inverse_mass;
  body[8] += impulse[1] * inverse_mass;
  body[9] += impulse[2] * inverse_mass;
  float torque[3], angular[3];
  cross3(r, impulse, torque);
  inverse_inertia_world(body + 3, inverse_inertia, torque, angular);
  body[10] += angular[0]; body[11] += angular[1]; body[12] += angular[2];
}

__device__ inline bool solve_pair(
    float* body_a,
    float inverse_mass_a,
    const float* half_a,
    const float* inverse_inertia_a,
    float* body_b,
    float inverse_mass_b,
    const float* half_b,
    const float* inverse_inertia_b,
    float restitution,
    float friction,
    float slop,
    float position_correction,
    float sat_epsilon) {
  if (inverse_mass_a == 0.0f && inverse_mass_b == 0.0f) return false;
  float penetration, normal[3], axes_a[3][3], axes_b[3][3];
  if (!sat_overlap(body_a, half_a, body_b, half_b, sat_epsilon,
                   &penetration, normal, axes_a, axes_b)) return false;

  float support_a[3], support_b[3];
  support_point(body_a, axes_a, half_a, normal, sat_epsilon, support_a);
  const float opposite[3] = {-normal[0], -normal[1], -normal[2]};
  support_point(body_b, axes_b, half_b, opposite, sat_epsilon, support_b);
  const float contact[3] = {
      0.5f * (support_a[0] + support_b[0]),
      0.5f * (support_a[1] + support_b[1]),
      0.5f * (support_a[2] + support_b[2])};
  const float r_a[3] = {
      contact[0] - body_a[0], contact[1] - body_a[1], contact[2] - body_a[2]};
  const float r_b[3] = {
      contact[0] - body_b[0], contact[1] - body_b[1], contact[2] - body_b[2]};
  float velocity_a[3], velocity_b[3], relative[3];
  point_velocity(body_a, r_a, velocity_a);
  point_velocity(body_b, r_b, velocity_b);
  relative[0] = velocity_b[0] - velocity_a[0];
  relative[1] = velocity_b[1] - velocity_a[1];
  relative[2] = velocity_b[2] - velocity_a[2];
  const float normal_speed = dot3(relative, normal);
  const float total_inverse_mass = inverse_mass_a + inverse_mass_b;
  if (normal_speed < 0.0f) {
    const float normal_mass = total_inverse_mass +
        angular_effective_mass(body_a + 3, inverse_inertia_a, r_a, normal) +
        angular_effective_mass(body_b + 3, inverse_inertia_b, r_b, normal);
    if (normal_mass > sat_epsilon) {
      const float normal_impulse_magnitude =
          fmaxf(0.0f, -(1.0f + restitution) * normal_speed / normal_mass);
      const float impulse[3] = {
          normal[0] * normal_impulse_magnitude,
          normal[1] * normal_impulse_magnitude,
          normal[2] * normal_impulse_magnitude};
      const float negative_impulse[3] = {-impulse[0], -impulse[1], -impulse[2]};
      apply_impulse(body_a, inverse_mass_a, inverse_inertia_a, r_a, negative_impulse);
      apply_impulse(body_b, inverse_mass_b, inverse_inertia_b, r_b, impulse);

      point_velocity(body_a, r_a, velocity_a);
      point_velocity(body_b, r_b, velocity_b);
      relative[0] = velocity_b[0] - velocity_a[0];
      relative[1] = velocity_b[1] - velocity_a[1];
      relative[2] = velocity_b[2] - velocity_a[2];
      const float updated_normal_speed = dot3(relative, normal);
      float tangent[3] = {
          relative[0] - normal[0] * updated_normal_speed,
          relative[1] - normal[1] * updated_normal_speed,
          relative[2] - normal[2] * updated_normal_speed};
      const float tangent_speed_squared = dot3(tangent, tangent);
      if (tangent_speed_squared > sat_epsilon * sat_epsilon && friction > 0.0f) {
        const float inverse_tangent_speed = rsqrtf(tangent_speed_squared);
        tangent[0] *= inverse_tangent_speed;
        tangent[1] *= inverse_tangent_speed;
        tangent[2] *= inverse_tangent_speed;
        const float tangent_mass = total_inverse_mass +
            angular_effective_mass(body_a + 3, inverse_inertia_a, r_a, tangent) +
            angular_effective_mass(body_b + 3, inverse_inertia_b, r_b, tangent);
        if (tangent_mass > sat_epsilon) {
          float tangent_impulse_magnitude = -dot3(relative, tangent) / tangent_mass;
          const float friction_limit = friction * normal_impulse_magnitude;
          tangent_impulse_magnitude =
              fmaxf(-friction_limit, fminf(friction_limit, tangent_impulse_magnitude));
          const float friction_impulse[3] = {
              tangent[0] * tangent_impulse_magnitude,
              tangent[1] * tangent_impulse_magnitude,
              tangent[2] * tangent_impulse_magnitude};
          const float negative_friction[3] = {
              -friction_impulse[0], -friction_impulse[1], -friction_impulse[2]};
          apply_impulse(body_a, inverse_mass_a, inverse_inertia_a, r_a, negative_friction);
          apply_impulse(body_b, inverse_mass_b, inverse_inertia_b, r_b, friction_impulse);
        }
      }
    }
  }

  if (total_inverse_mass > 0.0f) {
    const float repair = fminf(
        0.2f,
        fmaxf(0.0f, penetration - slop) * position_correction);
    const float correction = repair / total_inverse_mass;
    body_a[0] -= normal[0] * correction * inverse_mass_a;
    body_a[1] -= normal[1] * correction * inverse_mass_a;
    body_a[2] -= normal[2] * correction * inverse_mass_a;
    body_b[0] += normal[0] * correction * inverse_mass_b;
    body_b[1] += normal[1] * correction * inverse_mass_b;
    body_b[2] += normal[2] * correction * inverse_mass_b;
  }
  return true;
}

__global__ void step_sat_worlds(
    float* state,
    const float* inverse_mass,
    const float* half_extents,
    const float* inverse_inertia,
    const int64_t* pair_indices,
    uint8_t* contacts,
    float* penetration,
    int worlds,
    int bodies,
    int pairs,
    float dt,
    int substeps,
    float gravity_y,
    float restitution,
    float friction,
    float slop,
    float position_correction,
    float angular_damping,
    int solver_iterations,
    float sat_epsilon) {
  const int world = blockIdx.x * blockDim.x + threadIdx.x;
  if (world >= worlds) return;
  const float h = dt / static_cast<float>(substeps);
  const float damping = fmaxf(0.0f, 1.0f - angular_damping * h);
  for (int substep = 0; substep < substeps; ++substep) {
    for (int body_index = 0; body_index < bodies; ++body_index) {
      const int flat = world * bodies + body_index;
      if (inverse_mass[flat] == 0.0f) continue;
      float* body = state + flat * kStateWidth;
      body[8] += gravity_y * h;
      body[0] += body[7] * h; body[1] += body[8] * h; body[2] += body[9] * h;
      body[10] *= damping; body[11] *= damping; body[12] *= damping;
      integrate_quaternion(body, h);
    }
    for (int iteration = 0; iteration < solver_iterations; ++iteration) {
      for (int pair = 0; pair < pairs; ++pair) {
        const int64_t index_a = pair_indices[pair * 2];
        const int64_t index_b = pair_indices[pair * 2 + 1];
        // The Python contract validates pair contents before launch. Keep a
        // device guard as defense in depth against direct native calls.
        if (index_a < 0 || index_b < 0 || index_a >= bodies || index_b >= bodies || index_a == index_b) {
          continue;
        }
        const int flat_a = world * bodies + static_cast<int>(index_a);
        const int flat_b = world * bodies + static_cast<int>(index_b);
        const bool touched = solve_pair(
            state + flat_a * kStateWidth,
            inverse_mass[flat_a],
            half_extents + flat_a * 3,
            inverse_inertia + flat_a * 3,
            state + flat_b * kStateWidth,
            inverse_mass[flat_b],
            half_extents + flat_b * 3,
            inverse_inertia + flat_b * 3,
            restitution,
            friction,
            slop,
            position_correction,
            sat_epsilon);
        if (touched) {
          contacts[world * pairs + pair] = 1;
        }
      }
    }
  }
  // Match the CPU oracle: report final, not peak, minimum-translation depth.
  for (int pair = 0; pair < pairs; ++pair) {
    const int64_t index_a = pair_indices[pair * 2];
    const int64_t index_b = pair_indices[pair * 2 + 1];
    if (index_a < 0 || index_b < 0 || index_a >= bodies || index_b >= bodies || index_a == index_b) {
      continue;
    }
    const int flat_a = world * bodies + static_cast<int>(index_a);
    const int flat_b = world * bodies + static_cast<int>(index_b);
    float depth, normal[3], axes_a[3][3], axes_b[3][3];
    if (sat_overlap(
            state + flat_a * kStateWidth,
            half_extents + flat_a * 3,
            state + flat_b * kStateWidth,
            half_extents + flat_b * 3,
            sat_epsilon,
            &depth,
            normal,
            axes_a,
            axes_b)) {
      penetration[world * pairs + pair] = depth;
    }
  }
}

}  // namespace

std::vector<torch::Tensor> box3d_sat_step_cuda(
    torch::Tensor state,
    torch::Tensor inverse_mass,
    torch::Tensor half_extents,
    torch::Tensor inverse_inertia,
    torch::Tensor pair_indices,
    double dt,
    int64_t substeps,
    double gravity_y,
    double restitution,
    double friction,
    double slop,
    double position_correction,
    double angular_damping,
    int64_t solver_iterations,
    double sat_epsilon) {
  const c10::cuda::CUDAGuard device_guard(state.device());
  torch::Tensor output = state.clone();
  torch::Tensor contacts = torch::zeros(
      {state.size(0), pair_indices.size(0)}, state.options().dtype(torch::kUInt8));
  torch::Tensor penetration = torch::zeros(
      {state.size(0), pair_indices.size(0)}, state.options());
  const int worlds = static_cast<int>(state.size(0));
  const int bodies = static_cast<int>(state.size(1));
  const int pairs = static_cast<int>(pair_indices.size(0));
  constexpr int threads = 128;
  const int blocks = (worlds + threads - 1) / threads;
  step_sat_worlds<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      output.data_ptr<float>(),
      inverse_mass.data_ptr<float>(),
      half_extents.data_ptr<float>(),
      inverse_inertia.data_ptr<float>(),
      pair_indices.data_ptr<int64_t>(),
      contacts.data_ptr<uint8_t>(),
      penetration.data_ptr<float>(),
      worlds,
      bodies,
      pairs,
      static_cast<float>(dt),
      static_cast<int>(substeps),
      static_cast<float>(gravity_y),
      static_cast<float>(restitution),
      static_cast<float>(friction),
      static_cast<float>(slop),
      static_cast<float>(position_correction),
      static_cast<float>(angular_damping),
      static_cast<int>(solver_iterations),
      static_cast<float>(sat_epsilon));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, contacts, penetration};
}
