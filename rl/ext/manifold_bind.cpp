// SPDX-License-Identifier: MIT
// A binding of csrc/manifold.cu alone (the peg-insert env needs only manifold_step): a much smaller, faster build than
// csrc/bindings.cpp with every kernel. Argument checks are the caller's (rl/skills/peg_insert/env.py builds the tensors).
#include <torch/extension.h>

std::vector<torch::Tensor> box3d_manifold_step_cuda(
    torch::Tensor state, torch::Tensor inverse_mass, torch::Tensor half_extents, torch::Tensor inverse_inertia,
    torch::Tensor pair_indices, torch::Tensor cache_feature_ids, torch::Tensor cache_impulses, double dt,
    int64_t substeps, double gravity_y, double restitution, double friction, double slop, double position_correction,
    double angular_damping, int64_t solver_iterations, double sat_epsilon);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("manifold_step", [](torch::Tensor state, torch::Tensor im, torch::Tensor half, torch::Tensor ii, torch::Tensor pairs,
                                 torch::Tensor ids, torch::Tensor imp, double dt, int64_t sub, double gy, double rest, double mu,
                                 double slop, double pc, double damp, int64_t iters, double eps) {
    TORCH_CHECK(state.is_cuda() && state.dim() == 3 && state.size(2) == 13 && state.size(1) <= 32 && pairs.size(0) <= 16, "manifold_step shapes");
    return box3d_manifold_step_cuda(state.contiguous(), im.contiguous(), half.contiguous(), ii.contiguous(), pairs.contiguous(),
                                    ids.contiguous(), imp.contiguous(), dt, sub, gy, rest, mu, slop, pc, damp, iters, eps);
  });
}
