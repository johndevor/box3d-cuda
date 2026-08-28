// SPDX-License-Identifier: MIT
// Stage-1 fixed-topology manipulation slice: a dynamic AABB between two
// kinematic AABB fingers. Contact impulses and Coulomb friction carry the box;
// there is no attachment state or pose-copy path.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kStateWidth = 13;

__device__ inline void solve_floor(
    float* body, float cube_half_y, float restitution, float friction, float slop) {
  const float penetration = cube_half_y - body[1];
  if (penetration <= 0.0f) return;
  body[1] += fmaxf(0.0f, penetration - slop);
  if (body[8] >= 0.0f) return;
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

__global__ void step_gripper_worlds(
    float* cube_state,
    float* finger_positions,
    const float* finger_velocity,
    uint8_t* contacts,
    int worlds,
    float dt,
    int substeps,
    float gravity_y,
    float restitution,
    float friction,
    float slop,
    float position_correction,
    float cube_half_x,
    float cube_half_y,
    float cube_half_z,
    float finger_half_x,
    float finger_half_y,
    float finger_half_z) {
  const int world = blockIdx.x * blockDim.x + threadIdx.x;
  if (world >= worlds) return;
  float* body = cube_state + world * kStateWidth;
  float* fingers = finger_positions + world * 6;
  uint8_t* world_contacts = contacts + world * 2;
  const float cube_half[3] = {cube_half_x, cube_half_y, cube_half_z};
  const float finger_half[3] = {finger_half_x, finger_half_y, finger_half_z};
  const float h = dt / static_cast<float>(substeps);

  for (int substep = 0; substep < substeps; ++substep) {
    for (int finger = 0; finger < 2; ++finger) {
      for (int axis = 0; axis < 3; ++axis) {
        fingers[finger * 3 + axis] += finger_velocity[finger * 3 + axis] * h;
      }
    }
    body[8] += gravity_y * h;
    body[0] += body[7] * h;
    body[1] += body[8] * h;
    body[2] += body[9] * h;
    solve_floor(body, cube_half_y, restitution, friction, slop);

    bool active[2] = {false, false};
    float normals[2][3] = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
    float penetration[2] = {0.0f, 0.0f};
    for (int finger = 0; finger < 2; ++finger) {
      float overlap[3];
      float delta[3];
      bool touching = true;
      for (int axis = 0; axis < 3; ++axis) {
        delta[axis] = body[axis] - fingers[finger * 3 + axis];
        overlap[axis] = cube_half[axis] + finger_half[axis] - fabsf(delta[axis]);
        touching = touching && overlap[axis] > 0.0f;
      }
      if (!touching) continue;
      int contact_axis = overlap[1] < overlap[0] ? 1 : 0;
      contact_axis = overlap[2] < overlap[contact_axis] ? 2 : contact_axis;
      active[finger] = true;
      penetration[finger] = overlap[contact_axis];
      normals[finger][contact_axis] = delta[contact_axis] >= 0.0f ? 1.0f : -1.0f;
      world_contacts[finger] = 1;
    }

    const float incoming[3] = {body[7], body[8], body[9]};
    float total_normal_impulse = 0.0f;
    float velocity_delta[3] = {0.0f, 0.0f, 0.0f};
    float position_delta[3] = {0.0f, 0.0f, 0.0f};
    float target[3] = {0.0f, 0.0f, 0.0f};
    int active_count = 0;
    for (int finger = 0; finger < 2; ++finger) {
      if (!active[finger]) continue;
      ++active_count;
      float normal_speed = 0.0f;
      for (int axis = 0; axis < 3; ++axis) {
        normal_speed +=
            (incoming[axis] - finger_velocity[finger * 3 + axis]) * normals[finger][axis];
        target[axis] += finger_velocity[finger * 3 + axis];
      }
      const float depth = fmaxf(0.0f, penetration[finger] - slop);
      const float bias_speed = position_correction * depth / h;
      const float normal_delta = fmaxf(0.0f, -normal_speed + bias_speed);
      total_normal_impulse += normal_delta;
      for (int axis = 0; axis < 3; ++axis) {
        velocity_delta[axis] += normals[finger][axis] * normal_delta;
        position_delta[axis] += normals[finger][axis] * position_correction * depth;
      }
    }
    if (active_count > 0) {
      for (int axis = 0; axis < 3; ++axis) {
        body[7 + axis] += velocity_delta[axis];
        body[axis] += position_delta[axis];
        target[axis] /= static_cast<float>(active_count);
      }
      float tangent[3] = {
          body[7] - target[0], body[8] - target[1], body[9] - target[2]};
      for (int finger = 0; finger < 2; ++finger) {
        if (!active[finger]) continue;
        float component = 0.0f;
        for (int axis = 0; axis < 3; ++axis) component += tangent[axis] * normals[finger][axis];
        for (int axis = 0; axis < 3; ++axis) tangent[axis] -= component * normals[finger][axis];
      }
      const float tangent_speed = sqrtf(
          tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]);
      if (tangent_speed > 0.0f) {
        const float friction_delta = fminf(tangent_speed, friction * total_normal_impulse);
        for (int axis = 0; axis < 3; ++axis) {
          body[7 + axis] -= tangent[axis] * friction_delta / tangent_speed;
        }
      }
    }
    solve_floor(body, cube_half_y, restitution, friction, slop);
  }
}

}  // namespace

std::vector<torch::Tensor> box3d_gripper_step_cuda(
    torch::Tensor cube_state,
    torch::Tensor finger_positions,
    torch::Tensor finger_velocity,
    double dt,
    int64_t substeps,
    double gravity_y,
    double restitution,
    double friction,
    double slop,
    double position_correction,
    double cube_half_x,
    double cube_half_y,
    double cube_half_z,
    double finger_half_x,
    double finger_half_y,
    double finger_half_z) {
  const c10::cuda::CUDAGuard device_guard(cube_state.device());
  torch::Tensor output = cube_state.clone();
  torch::Tensor fingers = finger_positions.clone();
  torch::Tensor contacts = torch::zeros(
      {cube_state.size(0), 2}, cube_state.options().dtype(torch::kUInt8));
  const int worlds = static_cast<int>(cube_state.size(0));
  constexpr int threads = 128;
  const int blocks = (worlds + threads - 1) / threads;
  step_gripper_worlds<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      output.data_ptr<float>(), fingers.data_ptr<float>(), finger_velocity.data_ptr<float>(),
      contacts.data_ptr<uint8_t>(), worlds, static_cast<float>(dt), static_cast<int>(substeps),
      static_cast<float>(gravity_y), static_cast<float>(restitution), static_cast<float>(friction),
      static_cast<float>(slop), static_cast<float>(position_correction),
      static_cast<float>(cube_half_x), static_cast<float>(cube_half_y), static_cast<float>(cube_half_z),
      static_cast<float>(finger_half_x), static_cast<float>(finger_half_y), static_cast<float>(finger_half_z));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, fingers, contacts};
}
