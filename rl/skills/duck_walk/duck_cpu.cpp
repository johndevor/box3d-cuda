// SPDX-License-Identifier: MIT
// CPU build of the batched duck step (debugging, parity checks against World2).
#include "duck_bind.h"

void step(std::vector<torch::Tensor> mf, std::vector<torch::Tensor> mi, int64_t imu_b, double dt, int64_t substeps, int64_t iterations,
          double baumgarte, double slop, double contact_beta, double contact_slop, double max_bias,
          std::vector<torch::Tensor> ef, torch::Tensor rng, torch::Tensor goal_delay, torch::Tensor torque_on, int64_t n_outer) {
  duck::Model M = duck::make_model(mf, mi, imu_b, dt, substeps, iterations, baumgarte, slop, contact_beta, contact_slop, max_bias);
  duck::Env E = duck::make_env(ef, rng, goal_delay, torque_on);
  M.nb = (int)ef[0].size(1);
  TORCH_CHECK(M.nj <= duck::MAXJ && M.nf <= duck::MAXF, "too many joints or contact boxes for the kernel");
  const int n = (int)ef[0].size(0);
  for (int e = 0; e < n; e++) duck::step_env(e, M, E, (int)n_outer);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("step", &step, "batched duck step (CPU)"); }
