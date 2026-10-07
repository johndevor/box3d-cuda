// SPDX-License-Identifier: MIT
// CUDA build of the batched duck step: one thread per environment.
#include "duck_bind.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void step_kernel(duck::Model M, duck::Env E, int n, int n_outer) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e < n) duck::step_env(e, M, E, n_outer);
}

void step(std::vector<torch::Tensor> mf, std::vector<torch::Tensor> mi, int64_t imu_b, double dt, int64_t substeps, int64_t iterations,
          double baumgarte, double slop, double contact_beta, double contact_slop, double max_bias,
          std::vector<torch::Tensor> ef, torch::Tensor rng, torch::Tensor goal_delay, torch::Tensor torque_on, int64_t n_outer) {
  const c10::cuda::CUDAGuard guard(ef[0].device());
  duck::Model M = duck::make_model(mf, mi, imu_b, dt, substeps, iterations, baumgarte, slop, contact_beta, contact_slop, max_bias);
  duck::Env E = duck::make_env(ef, rng, goal_delay, torque_on);
  const int n = (int)ef[0].size(0), T = 64;
  step_kernel<<<(n + T - 1) / T, T, 0, at::cuda::getCurrentCUDAStream()>>>(M, E, n, (int)n_outer);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("step", &step, "batched duck step (CUDA)"); }
