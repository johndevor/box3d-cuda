// SPDX-License-Identifier: MIT
// Torch bindings shared by the CPU and CUDA builds: tensors -> Model/Env views.
#pragma once
#include <torch/extension.h>
#include "duck_sim.h"

namespace duck {
inline const float *F(const torch::Tensor &t) { return t.data_ptr<float>(); }
inline float *FW(torch::Tensor &t) { return t.data_ptr<float>(); }

// model: [pa, ca, axis, ref, lo, hi, foot_c, foot_h, foot_q, imu_p, imu_q] float tensors; jp, jc, foot_b int32 tensors
inline Model make_model(std::vector<torch::Tensor> mf, std::vector<torch::Tensor> mi, int64_t imu_b, double dt, int64_t substeps, int64_t iterations,
                        double baumgarte, double slop, double contact_beta, double contact_slop, double max_bias) {
  Model M;
  M.pa = F(mf[0]); M.ca = F(mf[1]); M.axis = F(mf[2]); M.ref = F(mf[3]); M.lo = F(mf[4]); M.hi = F(mf[5]);
  M.foot_c = F(mf[6]); M.foot_h = F(mf[7]); M.foot_q = F(mf[8]); M.imu_p = F(mf[9]); M.imu_q = F(mf[10]);
  M.jp = mi[0].data_ptr<int>(); M.jc = mi[1].data_ptr<int>(); M.foot_b = mi[2].data_ptr<int>();
  M.foot_mu = mf.size() > 11 && mf[11].numel() ? F(mf[11]) : nullptr;
  M.nj = (int)mf[4].numel(); M.nf = (int)mi[2].numel();
  M.imu_b = (int)imu_b; M.dt = (float)dt; M.substeps = (int)substeps; M.iterations = (int)iterations;
  M.baumgarte = (float)baumgarte; M.slop = (float)slop; M.contact_beta = (float)contact_beta; M.contact_slop = (float)contact_slop; M.max_bias = (float)max_bias;
  return M;
}
// env: [state, inv_mass, inv_inertia, servo, sp, jl, cl, mu, damp, imu, imu_par, imu_bias, goal_new, push, out] float; rng int32; goal_delay int32; torque_on uint8
inline Env make_env(std::vector<torch::Tensor> ef, torch::Tensor rng, torch::Tensor goal_delay, torch::Tensor torque_on) {
  Env E;
  TORCH_CHECK(ef[0].size(1) <= MAXB, "too many bodies for the kernel (MAXB)");
  E.state = FW(ef[0]); E.inv_mass = F(ef[1]); E.inv_inertia = F(ef[2]); E.servo = FW(ef[3]); E.sp = F(ef[4]); E.jl = FW(ef[5]); E.cl = FW(ef[6]);
  E.mu = F(ef[7]); E.damp = F(ef[8]); E.imu = FW(ef[9]); E.imu_par = F(ef[10]); E.imu_bias = F(ef[11]); E.goal_new = F(ef[12]); E.push = F(ef[13]); E.out = FW(ef[14]);
  E.rng = reinterpret_cast<uint32_t *>(rng.data_ptr<int>()); E.goal_delay = goal_delay.data_ptr<int>(); E.torque_on = torque_on.data_ptr<uint8_t>();
  return E;
}
}  // namespace duck
