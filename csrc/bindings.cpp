// SPDX-License-Identifier: MIT
#include <torch/extension.h>

torch::Tensor box3d_step_cuda(
    torch::Tensor state,
    torch::Tensor inverse_mass,
    torch::Tensor radius,
    double dt,
    int64_t substeps,
    double gravity_y,
    double restitution,
    double friction);

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
    double finger_half_z);

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
    int64_t solver_iterations);

torch::Tensor box3d_step(
    torch::Tensor state,
    torch::Tensor inverse_mass,
    torch::Tensor radius,
    double dt,
    int64_t substeps,
    double gravity_y,
    double restitution,
    double friction) {
  TORCH_CHECK(state.is_cuda(), "state must be CUDA");
  TORCH_CHECK(inverse_mass.is_cuda() && radius.is_cuda(), "material tensors must be CUDA");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32, "state must be float32");
  TORCH_CHECK(inverse_mass.scalar_type() == torch::kFloat32, "inverse_mass must be float32");
  TORCH_CHECK(radius.scalar_type() == torch::kFloat32, "radius must be float32");
  TORCH_CHECK(state.dim() == 3 && state.size(2) == 13, "state must have shape [worlds,bodies,13]");
  TORCH_CHECK(inverse_mass.sizes() == state.sizes().slice(0, 2), "inverse_mass shape mismatch");
  TORCH_CHECK(radius.sizes() == state.sizes().slice(0, 2), "radius shape mismatch");
  TORCH_CHECK(substeps > 0 && dt > 0.0, "dt and substeps must be positive");
  return box3d_step_cuda(
      state.contiguous(), inverse_mass.contiguous(), radius.contiguous(), dt,
      substeps, gravity_y, restitution, friction);
}

std::vector<torch::Tensor> box3d_gripper_step(
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
  TORCH_CHECK(cube_state.is_cuda(), "cube_state must be CUDA");
  TORCH_CHECK(finger_positions.is_cuda() && finger_velocity.is_cuda(), "finger tensors must be CUDA");
  TORCH_CHECK(cube_state.scalar_type() == torch::kFloat32, "cube_state must be float32");
  TORCH_CHECK(finger_positions.scalar_type() == torch::kFloat32, "finger_positions must be float32");
  TORCH_CHECK(finger_velocity.scalar_type() == torch::kFloat32, "finger_velocity must be float32");
  TORCH_CHECK(cube_state.dim() == 2 && cube_state.size(1) == 13, "cube_state must have shape [worlds,13]");
  TORCH_CHECK(finger_positions.dim() == 3 && finger_positions.size(0) == cube_state.size(0) &&
              finger_positions.size(1) == 2 && finger_positions.size(2) == 3,
              "finger_positions must have shape [worlds,2,3]");
  TORCH_CHECK(finger_velocity.dim() == 2 && finger_velocity.size(0) == 2 && finger_velocity.size(1) == 3,
              "finger_velocity must have shape [2,3]");
  TORCH_CHECK(substeps > 0 && dt > 0.0, "dt and substeps must be positive");
  return box3d_gripper_step_cuda(
      cube_state.contiguous(), finger_positions.contiguous(), finger_velocity.contiguous(),
      dt, substeps, gravity_y, restitution, friction, slop, position_correction,
      cube_half_x, cube_half_y, cube_half_z, finger_half_x, finger_half_y, finger_half_z);
}

std::vector<torch::Tensor> box3d_obb_step(
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
  TORCH_CHECK(state.is_cuda(), "state must be CUDA");
  TORCH_CHECK(inverse_mass.is_cuda() && half_extents.is_cuda() && inverse_inertia.is_cuda(),
              "material tensors must be CUDA");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32 &&
              inverse_mass.scalar_type() == torch::kFloat32 &&
              half_extents.scalar_type() == torch::kFloat32 &&
              inverse_inertia.scalar_type() == torch::kFloat32,
              "all tensors must be float32");
  TORCH_CHECK(state.dim() == 3 && state.size(2) == 13, "state must have shape [worlds,bodies,13]");
  TORCH_CHECK(inverse_mass.sizes() == state.sizes().slice(0, 2), "inverse_mass shape mismatch");
  TORCH_CHECK(half_extents.dim() == 3 && half_extents.size(0) == state.size(0) &&
              half_extents.size(1) == state.size(1) && half_extents.size(2) == 3,
              "half_extents must have shape [worlds,bodies,3]");
  TORCH_CHECK(inverse_inertia.sizes() == half_extents.sizes(), "inverse_inertia shape mismatch");
  TORCH_CHECK(dt > 0.0 && substeps > 0 && solver_iterations > 0, "invalid step configuration");
  return box3d_obb_step_cuda(
      state.contiguous(), inverse_mass.contiguous(), half_extents.contiguous(),
      inverse_inertia.contiguous(), dt, substeps, gravity_y, restitution,
      friction, slop, angular_damping, solver_iterations);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("step", &box3d_step, "Box3D-derived fixed-world CUDA step");
  module.def("gripper_step", &box3d_gripper_step, "Physical parallel-jaw box grasp CUDA step");
  module.def("obb_step", &box3d_obb_step, "Oriented box plane-contact CUDA step");
}
