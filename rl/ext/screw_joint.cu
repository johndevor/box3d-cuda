// SPDX-License-Identifier: MIT
// Screw/thread interaction joint for box3d-cuda (rl/screw_drive): a helical joint plus engagement and failure state,
// one CUDA thread per world, run after manifold_step each substep. Body 0 of each world is the screw. The model, the
// layouts of P (params), J (joint state) and C (commands) and every branch are those of the CPU oracle
// rl/ext/screw_reference.py (read it for the physics); tests/test in rl/ext/test_screw_joint.py.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {
constexpr int SW = 13, NP = 24, NJ = 24, NC = 2, QSTEPS = 4;
constexpr float TWO_PI = 6.283185307179586f, DEG1 = 0.017453292519943295f, PI_F = 3.141592653589793f;
enum { FREE = 0, ENGAGED = 1, SEATED = 2, FAILED = 3 };
enum { P_PITCH = 0, P_D, P_RCAP, P_THX, P_TRUN, P_KX, P_TSET, P_KJ, P_K, P_TSTRIP, P_PHIL, P_CAMC, P_HEXCAP, P_TMOT, P_TRAVEL,
       P_MODEB, P_MOUTH, P_UINS = 19, P_DROP = 22, P_L = 23 };
enum { J_MODE = 0, J_THETA, J_OMEGA, J_SSTRIP, J_U, J_ANCHOR = 7, J_CROSSED = 10, J_PHASE, J_CLICKED, J_ARMED, J_THSEAT, J_CLAMP,
       J_TORQUE, J_PEAK, J_STRIPPED, J_FAIL, J_SUCCESS, J_SPIN, J_FAX, J_CATCHES };
enum { FAIL_CLUTCH = 1, FAIL_STRIP = 2, FAIL_CAM = 3, FAIL_SLIP = 4 };

__device__ inline float dot3(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
__device__ inline void rot(const float* q, const float* v, float* o) {
  float x = q[0], y = q[1], z = q[2], w = q[3];
  float tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  o[0] = v[0] + w * tx + (y * tz - z * ty); o[1] = v[1] + w * ty + (z * tx - x * tz); o[2] = v[2] + w * tz + (x * ty - y * tx);
}
__device__ inline void qmul(const float* a, const float* b, float* o) {
  o[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1]; o[1] = a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0];
  o[2] = a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3]; o[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2];
}
__device__ inline void between(const float* p, const float* q, float* o) {
  float c = dot3(p, q), x[3] = {p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]}, w = 1.f + c;
  if (w < 1e-9f) { o[0] = 1; o[1] = o[2] = o[3] = 0; return; }
  float n = sqrtf(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + w * w);
  o[0] = x[0] / n; o[1] = x[1] / n; o[2] = x[2] / n; o[3] = w / n;
}
__device__ inline void qnorm(float* q) {
  float n = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (n > 1e-20f) for (int k = 0; k < 4; k++) q[k] /= n; else { q[0] = q[1] = q[2] = 0; q[3] = 1; }
}

__global__ void screw_kernel(float* state, const float* Pall, float* Jall, const float* Call, int W, int B, float h) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= W) return;
  float* b = state + (size_t)w * B * SW;
  const float* P = Pall + (size_t)w * NP;
  float* J = Jall + (size_t)w * NJ;
  const float* C = Call + (size_t)w * NC;
  int mode = int(J[J_MODE]);
  if (mode == FAILED || J[J_SUCCESS] > 0.5f) { J[J_OMEGA] = 0.f; return; }
  float pitch = P[P_PITCH], d = P[P_D], L = P[P_L], w_cmd = C[0], fax = fmaxf(0.f, C[1]);
  float q[4] = {b[3], b[4], b[5], b[6]}, down[3] = {0, 0, -1}, ax_b[3];
  rot(q, down, ax_b);
  float half_mm = L * 500.f, tip[3];
  for (int k = 0; k < 3; k++) tip[k] = b[k] + ax_b[k] * half_mm;
  const float* uins = P + P_UINS;
  J[J_TORQUE] = 0.f;
  J[J_FAX] = fax;
  if (mode == FREE) {
    J[J_OMEGA] = w_cmd;
    J[J_SPIN] += w_cmd * h;
    float rel[3];
    for (int k = 0; k < 3; k++) rel[k] = (tip[k] - P[P_MOUTH + k]) * 1e-3f;
    float along = dot3(rel, uins), lat = sqrtf(fmaxf(0.f, dot3(rel, rel) - along * along));
    if (along > -2e-4f && fax > 0.f) J[J_TORQUE] = 0.2f * fax * d * 0.25f * (w_cmd > 0.f ? 1.f : (w_cmd < 0.f ? -1.f : 0.f));
    bool inside = lat <= P[P_RCAP] && along >= -1.5e-4f && along <= 0.3f * pitch;
    if (!inside) { J[J_ARMED] = 1.f; return; }
    if (J[J_ARMED] < 0.5f) return;
    bool caught = false;
    if (P[P_MODEB] > 0.5f) caught = true;
    else if (fax >= 1.f) {
      if (J[J_CLICKED] > 0.5f && w_cmd > 0.f) caught = true;
      else {
        J[J_PHASE] += w_cmd * h;
        if (J[J_PHASE] >= TWO_PI) caught = true;
        else if (J[J_PHASE] < 0.f) { J[J_CLICKED] = 1.f; J[J_PHASE] += TWO_PI; J[J_TORQUE] -= 0.3f * P[P_TRUN]; }
      }
    }
    if (!caught) return;
    float c = fmaxf(-1.f, fminf(1.f, dot3(ax_b, uins))), tilt = acosf(c);
    float thr = P[P_THX] + (J[J_CLICKED] > 0.5f ? DEG1 : 0.f);
    bool crossed = tilt > thr;
    for (int k = 0; k < 3; k++) {
      J[J_U + k] = crossed ? ax_b[k] : uins[k];
      J[J_ANCHOR + k] = crossed ? tip[k] : P[P_MOUTH + k] + uins[k] * along * 1e3f;
    }
    J[J_CROSSED] = crossed ? 1.f : 0.f;
    J[J_MODE] = ENGAGED; J[J_THETA] = 0.f; J[J_THSEAT] = -1.f; J[J_CLAMP] = 0.f; J[J_CLICKED] = 0.f; J[J_ARMED] = 0.f;
    J[J_PHASE] = 0.f; J[J_CATCHES] += 1.f;
  }
  float theta = J[J_THETA];
  bool crossed = J[J_CROSSED] > 0.5f, released = false;
  float kd = P[P_K] * d * P[P_KJ] * pitch / TWO_PI, hs = h / QSTEPS, omega = 0.f, torque = 0.f;
  for (int it = 0; it < QSTEPS; it++) {
    bool seated = int(J[J_MODE]) == SEATED, stripped = J[J_STRIPPED] > 0.5f;
    float bind = crossed ? P[P_KX] * theta / TWO_PI : 0.f, clampT = seated ? kd * (theta - J[J_THSEAT]) : 0.f;
    if (w_cmd > 0.f) {
      if (stripped) { theta += w_cmd * hs; omega = w_cmd; torque = 0.3f * J[J_PEAK]; continue; }
      float tres = P[P_TRUN] + 0.05f * d * fax + bind + clampT;
      float capbit = P[P_PHIL] > 0.5f ? fax * P[P_CAMC] : P[P_HEXCAP];
      float cap = fminf(P[P_TSET], fminf(capbit, P[P_TMOT]));
      if (tres > P[P_TSTRIP] && P[P_TSTRIP] < cap) {
        J[J_STRIPPED] = 1.f; J[J_PEAK] = fmaxf(J[J_PEAK], P[P_TSTRIP]);
        J[J_SSTRIP] = fminf(pitch * theta / TWO_PI, P[P_TRAVEL]);
        omega = w_cmd; torque = 0.3f * P[P_TSTRIP]; theta += w_cmd * hs;
        continue;
      }
      if (tres >= cap) {
        omega = 0.f; torque = cap;
        if (cap >= P[P_TSET]) { if (seated) J[J_SUCCESS] = 1.f; else { J[J_FAIL] = FAIL_CLUTCH; J[J_MODE] = FAILED; } }
        else if (cap >= P[P_TMOT]) { J[J_FAIL] = FAIL_CLUTCH; J[J_MODE] = FAILED; }
        else { J[J_FAIL] = P[P_PHIL] > 0.5f ? FAIL_CAM : FAIL_SLIP; J[J_MODE] = FAILED; }
        break;
      }
      theta += w_cmd * hs; omega = w_cmd; torque = tres;
    } else if (w_cmd < 0.f) {
      float tl = P[P_TRUN] + bind + (seated ? 0.8f * clampT : 0.f);
      if (stripped) tl = 0.3f * J[J_PEAK];
      theta += w_cmd * hs; omega = w_cmd; torque = -tl;
    } else { omega = 0.f; torque = 0.f; }
    if (!stripped) {
      float s = pitch * theta / TWO_PI;
      if (!seated && s >= P[P_TRAVEL]) { J[J_MODE] = SEATED; J[J_THSEAT] = P[P_TRAVEL] * TWO_PI / pitch; }
      else if (seated && theta < J[J_THSEAT]) J[J_MODE] = ENGAGED;
    }
    if (theta < 0.f) { released = true; break; }
  }
  J[J_SPIN] += released ? w_cmd * h : omega * h;
  J[J_TORQUE] = torque;
  J[J_PEAK] = fmaxf(J[J_PEAK], torque);
  J[J_OMEGA] = omega;
  bool seatedNow = int(J[J_MODE]) == SEATED;
  J[J_CLAMP] = seatedNow ? P[P_KJ] * pitch * (theta - J[J_THSEAT]) / TWO_PI : 0.f;
  J[J_THETA] = theta;
  if (released) {
    J[J_MODE] = FREE; J[J_THETA] = 0.f; J[J_ARMED] = 0.f; J[J_PHASE] = PI_F; J[J_CROSSED] = 0.f; J[J_STRIPPED] = 0.f;
    return;
  }
  if (int(J[J_MODE]) == FAILED) return;
  const float* u = J + J_U;
  float s_eff = J[J_STRIPPED] > 0.5f ? J[J_SSTRIP] : fminf(pitch * theta / TWO_PI, P[P_TRAVEL]);
  float qa[4], qn[4], axn[3];
  between(ax_b, u, qa);
  qmul(qa, q, qn);
  qnorm(qn);
  rot(qn, down, axn);
  for (int k = 0; k < 3; k++) b[k] = J[J_ANCHOR + k] + u[k] * s_eff * 1e3f - axn[k] * half_mm;
  for (int k = 0; k < 4; k++) b[3 + k] = qn[k];
  bool moving = int(J[J_MODE]) == ENGAGED && J[J_STRIPPED] < 0.5f;
  float vax = moving ? pitch * omega / TWO_PI * 1e3f : 0.f;
  for (int k = 0; k < 3; k++) { b[7 + k] = u[k] * vax; b[10 + k] = 0.f; }
}
}  // namespace

std::vector<torch::Tensor> screw_joint_step(torch::Tensor state, torch::Tensor params, torch::Tensor joint, torch::Tensor cmd, double h) {
  TORCH_CHECK(state.is_cuda() && state.dim() == 3 && state.size(2) == SW, "state [W, B, 13] cuda");
  TORCH_CHECK(params.size(1) == NP && joint.size(1) == NJ && cmd.size(1) == NC, "P [W, 24], J [W, 24], C [W, 2]");
  const c10::cuda::CUDAGuard guard(state.device());
  auto out = state.contiguous().clone(), J = joint.contiguous().clone();
  auto P = params.contiguous(), C = cmd.contiguous();
  int W = state.size(0), B = state.size(1);
  constexpr int T = 128;
  screw_kernel<<<(W + T - 1) / T, T, 0, at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<float>(), P.data_ptr<float>(), J.data_ptr<float>(),
                                                                         C.data_ptr<float>(), W, B, float(h));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, J};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("screw_joint_step", &screw_joint_step, "screw/thread interaction joint step (rl/screw_drive)");
}
