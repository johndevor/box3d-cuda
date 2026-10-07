// SPDX-License-Identifier: MIT
// A connector's mate as an interaction, not geometry: one prismatic detent per world, stepped once per controller
// substep before the contact solve (rl/skills/connector_mate/env.py). The housing's sliding fit and lead-in funnel are box
// contacts (csrc/manifold.cu); this kernel adds what boxes cannot say about a latching connector:
//   - the latch's force-vs-depth curve: a rise from x_ramp to the peak F_peak at x_click, a drop over drop_w as the latch
//     snaps over (the click), sliding friction F_res once the contacts engage;
//   - latched (passed x_click + drop_w / 2): a withdrawal hold up to F_wd (an outward velocity below the hold point is
//     stopped by an impulse capped at F_wd * h; more than that pulls it out and unlatches);
//   - jam: engaged deeper than x_jam with the plug tilted over theta_jam, insertion locks up to F_jam (an impulse that
//     stops the inward velocity, capped at F_jam * h); it frees when the tilt falls under 0.6 theta_jam while drawing out;
//   - bent pin: engaged past x_pin, tilted over theta_jam and pushed with more than F_bend: a permanent fault.
// Inputs per world (SI, along the insertion axis): x (the plug's leading face past the mouth), v (its axial velocity,
// positive inward, after the controller's impulse), tilt (rad), push (the controller's axial force, N, positive inward),
// m (the axial virtual mass, kg), params[11], state[3] (latched, jammed, bent as 0/1 floats).
// Output: the axial impulse J (N s, positive outward) and the new state; clicked[n] = 1 on the substep the latch closes.
// CPU oracle: rl/ext/detent_reference.py (the same arithmetic in torch, float32); test: rl/ext/test_detent.py.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// params columns
enum { P_XRAMP = 0, P_XCLICK, P_DROPW, P_FPEAK, P_FRES, P_FWD, P_THJAM, P_XJAM, P_FJAM, P_FBEND, P_XPIN, P_COUNT };

__global__ void detent_kernel(const float* __restrict__ x, const float* __restrict__ v, const float* __restrict__ tilt,
                              const float* __restrict__ push, const float* __restrict__ m, const float* __restrict__ P,
                              float* __restrict__ state, float* __restrict__ J, float* __restrict__ clicked, int n, float h) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float* p = P + (size_t)i * P_COUNT;
  float* s = state + (size_t)i * 3;
  float xi = x[i], vi = v[i], th = tilt[i], f = push[i], mi = m[i];
  float latched = s[0], jammed = s[1], bent = s[2], click = 0.f;
  // the curve (a spring: it pushes out whatever the velocity) and the contacts' sliding friction
  float R = 0.f;
  float x0 = p[P_XRAMP], xc = p[P_XCLICK], dw = p[P_DROPW];
  if (latched < 0.5f) {
    if (xi >= x0 && xi < xc) R = p[P_FPEAK] * (xi - x0) / fmaxf(xc - x0, 1e-6f);
    else if (xi >= xc && xi < xc + dw) R = p[P_FPEAK] * (1.f - (xi - xc) / fmaxf(dw, 1e-6f));
  }
  if (xi > x0) R += p[P_FRES] * tanhf(vi / 2e-3f);
  float Jv = R * h;
  float vn = vi - Jv / mi;
  // latch closes past the drop's middle
  if (latched < 0.5f && xi >= xc + 0.5f * dw) { latched = 1.f; click = 1.f; }
  // withdrawal hold: below the hold point and moving out, stopped up to F_wd
  if (latched > 0.5f && xi < xc + 0.5f * dw && vn < 0.f) {
    float need = -vn * mi, cap = p[P_FWD] * h;
    if (need <= cap) { Jv -= need; vn = 0.f; }
    else { Jv -= cap; vn += cap / mi; latched = 0.f; }
  }
  // jam: engaged and tilted over the angle
  if (xi > p[P_XJAM] && th > p[P_THJAM]) jammed = 1.f;
  if (jammed > 0.5f && (th < 0.6f * p[P_THJAM] && vn < 0.f || xi <= 0.f)) jammed = 0.f;
  if (jammed > 0.5f && vn > 0.f) {
    float need = vn * mi, cap = p[P_FJAM] * h;
    float take = fminf(need, cap);
    Jv += take; vn -= take / mi;
  }
  // bent pin: forced while misaligned
  if (xi > p[P_XPIN] && th > p[P_THJAM] && f > p[P_FBEND]) bent = 1.f;
  s[0] = latched; s[1] = jammed; s[2] = bent;
  J[i] = Jv;
  clicked[i] = click;
}

std::vector<torch::Tensor> detent_step_cuda(torch::Tensor x, torch::Tensor v, torch::Tensor tilt, torch::Tensor push, torch::Tensor m,
                                            torch::Tensor params, torch::Tensor state, double h) {
  TORCH_CHECK(x.is_cuda() && params.size(1) == P_COUNT && state.size(1) == 3, "detent_step shapes");
  int n = x.size(0);
  auto st = state.clone();
  auto J = torch::zeros_like(x), clicked = torch::zeros_like(x);
  int threads = 256, blocks = (n + threads - 1) / threads;
  detent_kernel<<<blocks, threads>>>(x.data_ptr<float>(), v.data_ptr<float>(), tilt.data_ptr<float>(), push.data_ptr<float>(),
                                     m.data_ptr<float>(), params.data_ptr<float>(), st.data_ptr<float>(), J.data_ptr<float>(),
                                     clicked.data_ptr<float>(), n, (float)h);
  return {J, st, clicked};
}
