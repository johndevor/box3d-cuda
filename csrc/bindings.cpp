// SPDX-License-Identifier: MIT
#include <torch/extension.h>
#include <cmath>

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
    double sat_epsilon);

std::vector<torch::Tensor> box3d_manifold_step_cuda(
    torch::Tensor state,
    torch::Tensor inverse_mass,
    torch::Tensor half_extents,
    torch::Tensor inverse_inertia,
    torch::Tensor pair_indices,
    torch::Tensor cache_feature_ids,
    torch::Tensor cache_impulses,
    double dt,
    int64_t substeps,
    double gravity_y,
    double restitution,
    double friction,
    double slop,
    double position_correction,
    double angular_damping,
    int64_t solver_iterations,
    double sat_epsilon);

std::vector<torch::Tensor> box3d_joint_step_cuda(
    torch::Tensor state,
    torch::Tensor inverse_mass,
    torch::Tensor inverse_inertia,
    torch::Tensor joint_indices,
    torch::Tensor joint_types,
    torch::Tensor parent_anchor_local,
    torch::Tensor child_anchor_local,
    torch::Tensor axis_parent,
    torch::Tensor reference_quaternion,
    torch::Tensor lower_limit,
    torch::Tensor upper_limit,
    torch::Tensor damping,
    torch::Tensor motor_enabled,
    torch::Tensor motor_target_velocity,
    torch::Tensor motor_target_position,
    torch::Tensor stiffness,
    torch::Tensor maximum_effort,
    torch::Tensor warm_start_cache,
    double warm_start_factor,
    double dt,
    int64_t substeps,
    double gravity_y,
    int64_t solver_iterations,
    double position_correction,
    double position_slop,
    double angular_slop,
    double maximum_linear_repair_m,
    double maximum_angular_repair_rad);

std::vector<torch::Tensor> box3d_ray_cast_cuda(
    torch::Tensor state,
    torch::Tensor half_extents,
    torch::Tensor body_enabled,
    torch::Tensor ray_origins,
    torch::Tensor ray_directions,
    torch::Tensor maximum_distance);

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

std::vector<torch::Tensor> box3d_sat_step(
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
  TORCH_CHECK(state.is_cuda(), "state must be CUDA");
  TORCH_CHECK(inverse_mass.is_cuda() && half_extents.is_cuda() && inverse_inertia.is_cuda() &&
              pair_indices.is_cuda(), "all SAT tensors must be CUDA");
  TORCH_CHECK(state.device() == inverse_mass.device() && state.device() == half_extents.device() &&
              state.device() == inverse_inertia.device() && state.device() == pair_indices.device(),
              "all SAT tensors must be on the same CUDA device");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32 &&
              inverse_mass.scalar_type() == torch::kFloat32 &&
              half_extents.scalar_type() == torch::kFloat32 &&
              inverse_inertia.scalar_type() == torch::kFloat32,
              "state and SAT material tensors must be float32");
  TORCH_CHECK(pair_indices.scalar_type() == torch::kInt64, "pair_indices must be int64");
  TORCH_CHECK(state.dim() == 3 && state.size(2) == 13,
              "state must have shape [worlds,bodies,13]");
  TORCH_CHECK(state.size(0) > 0 && state.size(1) >= 2 && state.size(1) <= 32,
              "SAT requires positive worlds and between 2 and 32 bodies per fixed-small world");
  TORCH_CHECK(inverse_mass.sizes() == state.sizes().slice(0, 2), "inverse_mass shape mismatch");
  TORCH_CHECK(half_extents.dim() == 3 && half_extents.size(0) == state.size(0) &&
              half_extents.size(1) == state.size(1) && half_extents.size(2) == 3,
              "half_extents must have shape [worlds,bodies,3]");
  TORCH_CHECK(inverse_inertia.sizes() == half_extents.sizes(), "inverse_inertia shape mismatch");
  TORCH_CHECK(pair_indices.dim() == 2 && pair_indices.size(1) == 2 &&
              pair_indices.size(0) > 0 && pair_indices.size(0) <= 64,
              "pair_indices must have shape [pairs,2] with between 1 and 64 pairs");
  TORCH_CHECK(std::isfinite(dt) && dt > 0.0 && dt <= 1.0,
              "dt must be finite and in (0,1]");
  TORCH_CHECK(substeps > 0 && substeps <= 64, "substeps must be in [1,64]");
  TORCH_CHECK(std::isfinite(gravity_y) && std::abs(gravity_y) <= 1000.0,
              "gravity_y must be finite and bounded");
  TORCH_CHECK(std::isfinite(restitution) && restitution >= 0.0 && restitution <= 1.0,
              "restitution must be finite and in [0,1]");
  TORCH_CHECK(std::isfinite(friction) && friction >= 0.0 && friction <= 10.0,
              "friction must be finite and in [0,10]");
  TORCH_CHECK(std::isfinite(slop) && slop >= 0.0 && slop <= 0.1,
              "position_slop must be finite and in [0,0.1]");
  TORCH_CHECK(std::isfinite(position_correction) && position_correction >= 0.0 &&
              position_correction <= 1.0,
              "position_correction must be finite and in [0,1]");
  TORCH_CHECK(std::isfinite(angular_damping) && angular_damping >= 0.0 &&
              angular_damping <= 100.0,
              "angular_damping must be finite and in [0,100]");
  TORCH_CHECK(solver_iterations > 0 && solver_iterations <= 64,
              "solver_iterations must be in [1,64]");
  TORCH_CHECK(std::isfinite(sat_epsilon) && sat_epsilon > 0.0 && sat_epsilon <= 0.01,
              "sat_epsilon must be finite and in (0,0.01]");
  return box3d_sat_step_cuda(
      state.contiguous(), inverse_mass.contiguous(), half_extents.contiguous(),
      inverse_inertia.contiguous(), pair_indices.contiguous(), dt, substeps,
      gravity_y, restitution, friction, slop, position_correction,
      angular_damping, solver_iterations, sat_epsilon);
}

std::vector<torch::Tensor> box3d_manifold_step(
    torch::Tensor state,
    torch::Tensor inverse_mass,
    torch::Tensor half_extents,
    torch::Tensor inverse_inertia,
    torch::Tensor pair_indices,
    torch::Tensor cache_feature_ids,
    torch::Tensor cache_impulses,
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
  TORCH_CHECK(state.is_cuda() && inverse_mass.is_cuda() && half_extents.is_cuda() &&
              inverse_inertia.is_cuda() && pair_indices.is_cuda() &&
              cache_feature_ids.is_cuda() && cache_impulses.is_cuda(),
              "all manifold tensors must be CUDA");
  TORCH_CHECK(state.device() == inverse_mass.device() && state.device() == half_extents.device() &&
              state.device() == inverse_inertia.device() && state.device() == pair_indices.device() &&
              state.device() == cache_feature_ids.device() && state.device() == cache_impulses.device(),
              "all manifold tensors must be on the same CUDA device");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32 &&
              inverse_mass.scalar_type() == torch::kFloat32 &&
              half_extents.scalar_type() == torch::kFloat32 &&
              inverse_inertia.scalar_type() == torch::kFloat32 &&
              cache_impulses.scalar_type() == torch::kFloat32,
              "manifold state, material, and impulse tensors must be float32");
  TORCH_CHECK(pair_indices.scalar_type() == torch::kInt64 &&
              cache_feature_ids.scalar_type() == torch::kInt64,
              "manifold pair indices and feature IDs must be int64");
  TORCH_CHECK(state.dim() == 3 && state.size(2) == 13 && state.size(0) > 0 &&
              state.size(1) >= 2 && state.size(1) <= 32,
              "state must have shape [worlds,2..32,13]");
  TORCH_CHECK(inverse_mass.sizes() == state.sizes().slice(0, 2), "inverse_mass shape mismatch");
  TORCH_CHECK(half_extents.dim() == 3 && half_extents.size(0) == state.size(0) &&
              half_extents.size(1) == state.size(1) && half_extents.size(2) == 3,
              "half_extents must have shape [worlds,bodies,3]");
  TORCH_CHECK(inverse_inertia.sizes() == half_extents.sizes(), "inverse_inertia shape mismatch");
  TORCH_CHECK(pair_indices.dim() == 2 && pair_indices.size(1) == 2 &&
              pair_indices.size(0) > 0 && pair_indices.size(0) <= 16,
              "pair_indices must have shape [1..16,2]");
  TORCH_CHECK(cache_feature_ids.dim() == 3 && cache_feature_ids.size(0) == state.size(0) &&
              cache_feature_ids.size(1) == pair_indices.size(0) && cache_feature_ids.size(2) == 4,
              "cache_feature_ids must have shape [worlds,pairs,4]");
  TORCH_CHECK(cache_impulses.dim() == 4 && cache_impulses.size(0) == state.size(0) &&
              cache_impulses.size(1) == pair_indices.size(0) && cache_impulses.size(2) == 4 &&
              cache_impulses.size(3) == 3,
              "cache_impulses must have shape [worlds,pairs,4,3]");
  TORCH_CHECK(std::isfinite(dt) && dt > 0.0 && dt <= 1.0,
              "dt must be finite and in (0,1]");
  TORCH_CHECK(substeps > 0 && substeps <= 64 && solver_iterations > 0 && solver_iterations <= 64,
              "substeps and solver_iterations must be in [1,64]");
  TORCH_CHECK(std::isfinite(gravity_y) && std::abs(gravity_y) <= 1000.0,
              "gravity_y must be finite and bounded");
  TORCH_CHECK(std::isfinite(restitution) && restitution >= 0.0 && restitution <= 1.0,
              "restitution must be finite and in [0,1]");
  TORCH_CHECK(std::isfinite(friction) && friction >= 0.0 && friction <= 10.0,
              "friction must be finite and in [0,10]");
  TORCH_CHECK(std::isfinite(slop) && slop >= 0.0 && slop <= 0.1,
              "position_slop must be finite and in [0,0.1]");
  TORCH_CHECK(std::isfinite(position_correction) && position_correction >= 0.0 &&
              position_correction <= 1.0,
              "position_correction must be finite and in [0,1]");
  TORCH_CHECK(std::isfinite(angular_damping) && angular_damping >= 0.0 &&
              angular_damping <= 100.0,
              "angular_damping must be finite and in [0,100]");
  TORCH_CHECK(std::isfinite(sat_epsilon) && sat_epsilon > 0.0 && sat_epsilon <= 0.01,
              "sat_epsilon must be finite and in (0,0.01]");
  return box3d_manifold_step_cuda(
      state.contiguous(), inverse_mass.contiguous(), half_extents.contiguous(),
      inverse_inertia.contiguous(), pair_indices.contiguous(),
      cache_feature_ids.contiguous(), cache_impulses.contiguous(), dt, substeps,
      gravity_y, restitution, friction, slop, position_correction,
      angular_damping, solver_iterations, sat_epsilon);
}

std::vector<torch::Tensor> box3d_joint_step(
    torch::Tensor state,
    torch::Tensor inverse_mass,
    torch::Tensor inverse_inertia,
    torch::Tensor joint_indices,
    torch::Tensor joint_types,
    torch::Tensor parent_anchor_local,
    torch::Tensor child_anchor_local,
    torch::Tensor axis_parent,
    torch::Tensor reference_quaternion,
    torch::Tensor lower_limit,
    torch::Tensor upper_limit,
    torch::Tensor damping,
    torch::Tensor motor_enabled,
    torch::Tensor motor_target_velocity,
    torch::Tensor motor_target_position,
    torch::Tensor stiffness,
    torch::Tensor maximum_effort,
    torch::Tensor warm_start_cache,
    double warm_start_factor,
    double dt,
    int64_t substeps,
    double gravity_y,
    int64_t solver_iterations,
    double position_correction,
    double position_slop,
    double angular_slop,
    double maximum_linear_repair_m,
    double maximum_angular_repair_rad) {
  TORCH_CHECK(state.is_cuda() && inverse_mass.is_cuda() && inverse_inertia.is_cuda() &&
              joint_indices.is_cuda() && joint_types.is_cuda() && parent_anchor_local.is_cuda() &&
              child_anchor_local.is_cuda() && axis_parent.is_cuda() && reference_quaternion.is_cuda() &&
              lower_limit.is_cuda() && upper_limit.is_cuda() && damping.is_cuda() &&
              motor_enabled.is_cuda() && motor_target_velocity.is_cuda() &&
              motor_target_position.is_cuda() && stiffness.is_cuda() && maximum_effort.is_cuda() &&
              warm_start_cache.is_cuda(),
              "all joint tensors must be CUDA");
  const auto device = state.device();
  TORCH_CHECK(inverse_mass.device() == device && inverse_inertia.device() == device &&
              joint_indices.device() == device && joint_types.device() == device &&
              parent_anchor_local.device() == device && child_anchor_local.device() == device &&
              axis_parent.device() == device && reference_quaternion.device() == device &&
              lower_limit.device() == device && upper_limit.device() == device && damping.device() == device &&
              motor_enabled.device() == device && motor_target_velocity.device() == device &&
              motor_target_position.device() == device && stiffness.device() == device &&
              maximum_effort.device() == device && warm_start_cache.device() == device,
              "all joint tensors must share one CUDA device");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32 && inverse_mass.scalar_type() == torch::kFloat32 &&
              inverse_inertia.scalar_type() == torch::kFloat32 && parent_anchor_local.scalar_type() == torch::kFloat32 &&
              child_anchor_local.scalar_type() == torch::kFloat32 && axis_parent.scalar_type() == torch::kFloat32 &&
              reference_quaternion.scalar_type() == torch::kFloat32 && lower_limit.scalar_type() == torch::kFloat32 &&
              upper_limit.scalar_type() == torch::kFloat32 && damping.scalar_type() == torch::kFloat32 &&
              motor_target_velocity.scalar_type() == torch::kFloat32 &&
              motor_target_position.scalar_type() == torch::kFloat32 && stiffness.scalar_type() == torch::kFloat32 &&
              maximum_effort.scalar_type() == torch::kFloat32 &&
              warm_start_cache.scalar_type() == torch::kFloat32,
              "joint state, topology, and control tensors must be float32");
  TORCH_CHECK(joint_indices.scalar_type() == torch::kInt64 && joint_types.scalar_type() == torch::kInt64,
              "joint indices and types must be int64");
  TORCH_CHECK(motor_enabled.scalar_type() == torch::kUInt8, "motor_enabled must be uint8");
  TORCH_CHECK(state.dim() == 3 && state.size(0) > 0 && state.size(1) >= 2 &&
              state.size(1) <= 32 && state.size(2) == 13,
              "state must have shape [worlds,2..32,13]");
  const int64_t worlds = state.size(0), bodies = state.size(1);
  TORCH_CHECK(inverse_mass.dim() == 2 && inverse_mass.size(0) == worlds && inverse_mass.size(1) == bodies,
              "inverse_mass shape mismatch");
  TORCH_CHECK(inverse_inertia.dim() == 3 && inverse_inertia.size(0) == worlds &&
              inverse_inertia.size(1) == bodies && inverse_inertia.size(2) == 3,
              "inverse_inertia must have shape [worlds,bodies,3]");
  TORCH_CHECK(joint_indices.dim() == 2 && joint_indices.size(1) == 2 &&
              joint_indices.size(0) > 0 && joint_indices.size(0) <= 16,
              "joint_indices must have shape [1..16,2]");
  const int64_t joints = joint_indices.size(0);
  TORCH_CHECK(joint_types.dim() == 1 && joint_types.size(0) == joints &&
              parent_anchor_local.sizes() == torch::IntArrayRef({joints, 3}) &&
              child_anchor_local.sizes() == torch::IntArrayRef({joints, 3}) &&
              axis_parent.sizes() == torch::IntArrayRef({joints, 3}) &&
              reference_quaternion.sizes() == torch::IntArrayRef({joints, 4}),
              "joint topology row shapes mismatch");
  TORCH_CHECK(lower_limit.sizes() == torch::IntArrayRef({joints}) &&
              upper_limit.sizes() == torch::IntArrayRef({joints}) &&
              damping.sizes() == torch::IntArrayRef({joints}) &&
              stiffness.sizes() == torch::IntArrayRef({joints}) &&
              motor_enabled.sizes() == torch::IntArrayRef({joints}),
              "joint scalar topology shapes mismatch");
  TORCH_CHECK(motor_target_velocity.sizes() == torch::IntArrayRef({worlds, joints}) &&
              motor_target_position.sizes() == torch::IntArrayRef({worlds, joints}) &&
              maximum_effort.sizes() == torch::IntArrayRef({worlds, joints}),
              "joint controls must have shape [worlds,joints]");
  TORCH_CHECK(warm_start_cache.sizes() == torch::IntArrayRef({worlds, joints, 8}),
              "warm_start_cache must have shape [worlds,joints,8]");
  TORCH_CHECK(std::isfinite(dt) && dt > 0.0 && dt <= 1.0 && substeps > 0 && substeps <= 64 &&
              solver_iterations > 0 && solver_iterations <= 64, "invalid joint step configuration");
  TORCH_CHECK(std::isfinite(gravity_y) && std::abs(gravity_y) <= 1000.0,
              "gravity_y must be finite and bounded");
  TORCH_CHECK(std::isfinite(position_correction) && position_correction >= 0.0 && position_correction <= 1.0 &&
              std::isfinite(position_slop) && position_slop >= 0.0 &&
              std::isfinite(angular_slop) && angular_slop >= 0.0 &&
              std::isfinite(maximum_linear_repair_m) && maximum_linear_repair_m >= 0.0 &&
              std::isfinite(maximum_angular_repair_rad) && maximum_angular_repair_rad >= 0.0,
              "joint repair parameters must be finite and bounded");
  TORCH_CHECK(std::isfinite(warm_start_factor) && warm_start_factor >= 0.0 &&
              warm_start_factor <= 1.0,
              "warm_start_factor must be finite and in [0,1]");
  return box3d_joint_step_cuda(
      state.contiguous(), inverse_mass.contiguous(), inverse_inertia.contiguous(),
      joint_indices.contiguous(), joint_types.contiguous(), parent_anchor_local.contiguous(),
      child_anchor_local.contiguous(), axis_parent.contiguous(), reference_quaternion.contiguous(),
      lower_limit.contiguous(), upper_limit.contiguous(), damping.contiguous(),
      motor_enabled.contiguous(), motor_target_velocity.contiguous(),
      motor_target_position.contiguous(), stiffness.contiguous(), maximum_effort.contiguous(),
      warm_start_cache.contiguous(), warm_start_factor, dt, substeps, gravity_y,
      solver_iterations, position_correction, position_slop,
      angular_slop, maximum_linear_repair_m, maximum_angular_repair_rad);
}

std::vector<torch::Tensor> box3d_ray_cast(
    torch::Tensor state,
    torch::Tensor half_extents,
    torch::Tensor body_enabled,
    torch::Tensor ray_origins,
    torch::Tensor ray_directions,
    torch::Tensor maximum_distance) {
  TORCH_CHECK(state.is_cuda() && half_extents.is_cuda() && body_enabled.is_cuda() &&
              ray_origins.is_cuda() && ray_directions.is_cuda() && maximum_distance.is_cuda(),
              "all ray tensors must be CUDA");
  const auto device = state.device();
  TORCH_CHECK(half_extents.device() == device && body_enabled.device() == device &&
              ray_origins.device() == device && ray_directions.device() == device &&
              maximum_distance.device() == device,
              "all ray tensors must share one CUDA device");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32 &&
              half_extents.scalar_type() == torch::kFloat32 &&
              ray_origins.scalar_type() == torch::kFloat32 &&
              ray_directions.scalar_type() == torch::kFloat32 &&
              maximum_distance.scalar_type() == torch::kFloat32,
              "ray state, geometry, and query tensors must be float32");
  TORCH_CHECK(body_enabled.scalar_type() == torch::kUInt8,
              "body_enabled must be uint8");
  TORCH_CHECK(state.dim() == 3 && state.size(0) > 0 && state.size(1) >= 1 &&
              state.size(1) <= 32 && state.size(2) == 13,
              "state must have shape [worlds,1..32,13]");
  const int64_t worlds = state.size(0), bodies = state.size(1);
  TORCH_CHECK(half_extents.sizes() == torch::IntArrayRef({worlds, bodies, 3}),
              "half_extents must have shape [worlds,bodies,3]");
  TORCH_CHECK(body_enabled.sizes() == torch::IntArrayRef({worlds, bodies}),
              "body_enabled must have shape [worlds,bodies]");
  TORCH_CHECK(ray_origins.dim() == 3 && ray_origins.size(0) == worlds &&
              ray_origins.size(1) > 0 && ray_origins.size(2) == 3,
              "ray_origins must have shape [worlds,rays,3]");
  const int64_t rays = ray_origins.size(1);
  TORCH_CHECK(ray_directions.sizes() == torch::IntArrayRef({worlds, rays, 3}),
              "ray_directions must have shape [worlds,rays,3]");
  TORCH_CHECK(maximum_distance.sizes() == torch::IntArrayRef({worlds, rays}),
              "maximum_distance must have shape [worlds,rays]");
  TORCH_CHECK(rays <= 262144 && worlds <= 1048576 / rays,
              "ray batch is too large");
  return box3d_ray_cast_cuda(
      state.contiguous(), half_extents.contiguous(), body_enabled.contiguous(),
      ray_origins.contiguous(), ray_directions.contiguous(),
      maximum_distance.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("step", &box3d_step, "Box3D-derived fixed-world CUDA step");
  module.def("gripper_step", &box3d_gripper_step, "Physical parallel-jaw box grasp CUDA step");
  module.def("obb_step", &box3d_obb_step, "Oriented box plane-contact CUDA step");
  module.def("sat_step", &box3d_sat_step, "Oriented box pair SAT contact CUDA step");
  module.def("manifold_step", &box3d_manifold_step, "Persistent clipped OBB manifold CUDA step");
  module.def("joint_step", &box3d_joint_step, "Fixed-topology articulated joint CUDA step");
  module.def("ray_cast", &box3d_ray_cast, "Batched nearest-hit OBB ray query");
}
