// SPDX-License-Identifier: MIT
// The binding of rl/ext/detent.cu (an optional extension: a prismatic detent per world, used by rl/skills/connector_mate).
#include <torch/extension.h>

std::vector<torch::Tensor> detent_step_cuda(torch::Tensor x, torch::Tensor v, torch::Tensor tilt, torch::Tensor push, torch::Tensor m,
                                            torch::Tensor params, torch::Tensor state, double h);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("detent_step", [](torch::Tensor x, torch::Tensor v, torch::Tensor tilt, torch::Tensor push, torch::Tensor m,
                               torch::Tensor params, torch::Tensor state, double h) {
    auto f = [](torch::Tensor t) { return t.contiguous().to(torch::kFloat32); };
    return detent_step_cuda(f(x), f(v), f(tilt), f(push), f(m), f(params), f(state), h);
  });
}
