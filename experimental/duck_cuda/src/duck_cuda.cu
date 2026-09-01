// SPDX-License-Identifier: MIT
// Real CUDA driver for the batched duck lane, WARP-PER-ENV: 32 lanes
// cooperate on one environment (DW_WARP_LANES=32; build with
// -DDW_WARP_LANES=1 for the legacy thread-per-env layout). State lives in
// one device buffer indexed by env (DwState[E]) plus one DwWork workspace
// slice per env in global memory. Host-memory copies at the ABI boundary;
// the PPO driver keeps everything device-resident between calls.
//
// Cannot be compiled or tested on the development Mac (no nvcc); the serial
// build src/duck_cuda_serial.cpp shares every physics line via
// duck_cuda_kernel.h and is the tested parity vehicle. Cross-build numerics
// are gated by tests/remote_gpu_parity.py (windowed tolerances + bitwise
// on-device determinism). Keep this file thin.
#include <cuda.h>
#include <cuda_runtime.h>

#define DW_HD __host__ __device__
#ifndef DW_WARP_LANES
#define DW_WARP_LANES 32
#endif
#include "duck_cuda_kernel.h"

#include <cstddef>
#include <memory>
#include <new>
#include <vector>

// Occupancy floor: without __launch_bounds__ the fully-inlined tick kernel
// compiles to ~max registers/thread, capping residency at ONE 256-thread
// block per SM (~8 envs/SM, ~1.4k envs in flight on a 5090) -- exactly the
// flat-throughput-from-E~1024 signature the first hardware run showed.
// Forcing >= DW_MIN_BLOCKS_PER_SM blocks/SM clamps registers (4 blocks @
// 256 threads => <= 64 regs/thread) and trades a little spill traffic
// (L1/L2-cached, tiny next to the DwWork working set) for 4x cross-env
// parallelism. Tunable at build time: -DDW_MIN_BLOCKS_PER_SM=2 relaxes the
// clamp to <= 128 regs if the diag sweep shows spill-bound per-env latency.
#ifndef DW_MIN_BLOCKS_PER_SM
#define DW_MIN_BLOCKS_PER_SM 4
#endif

namespace {
constexpr int kLanesPerEnv = DW_WARP_LANES;   // 32 (warp) or 1 (legacy)
constexpr int kThreadsPerBlock = 256;         // 8 envs per block at 32 lanes
// Per-thread stack is small now: every tick-sized array lives in the per-env
// DwWork global slice; locals are quat/vec temps, the lane-0 manifold reduce
// (~1 KB) and the lane-0 policy FK. NOTE: the stack limit only sizes the
// local-memory pool (max-resident-threads x limit bytes of device memory);
// it does NOT gate occupancy. 8 KB is ample and halves the reservation.
constexpr size_t kStackBytes = 8 * 1024;

__global__ void __launch_bounds__(kThreadsPerBlock, DW_MIN_BLOCKS_PER_SM)
dw_step_kernel(DwState* states, const float* targets,
                               uint32_t n_ticks, DwParams params,
                               DwWork* work, dwc1_diagnostic* diags,
                               uint32_t E) {
  const uint32_t stride = (blockDim.x * gridDim.x) / kLanesPerEnv;
  for (uint32_t e = (blockIdx.x * blockDim.x + threadIdx.x) / kLanesPerEnv;
       e < E; e += stride) {
    dwc1_diagnostic d = {};
    d.environment = e;
    dw_step_env(&states[e], targets + (size_t)e * DW_J, n_ticks, &params,
                &work[e], &d);
    DW_LANE0 diags[e] = d;   // lane 0 holds the authoritative diag scalars
  }
}

__global__ void __launch_bounds__(kThreadsPerBlock, DW_MIN_BLOCKS_PER_SM)
dw_step_policy_kernel(DwState* states, const float* actions,
                                      uint32_t n_ticks, DwParams params,
                                      DwWork* work, float* obs, float* rew,
                                      uint8_t* done, dwc1_diagnostic* diags,
                                      uint32_t E) {
  const uint32_t stride = (blockDim.x * gridDim.x) / kLanesPerEnv;
  for (uint32_t e = (blockIdx.x * blockDim.x + threadIdx.x) / kLanesPerEnv;
       e < E; e += stride) {
    dwc1_diagnostic d = {};
    d.environment = e;
    dw_step_policy_env(&states[e], actions + (size_t)e * DW_J, n_ticks,
                       &params, &work[e],
                       obs + (size_t)e * DWP_OBS, rew + e, done + e, &d);
    DW_LANE0 diags[e] = d;
  }
}
}  // namespace

struct dwc1_scene {
  uint32_t E = 0;
  DwParams params{};
  dwc1_randomization rand{};  // creation-time ranges (all zero = off)
  DwState* device_state = nullptr;      // [E], authoritative
  float* device_targets = nullptr;      // [E, J] targets or actions
  DwWork* device_scratch = nullptr;     // [E] per-env cooperative workspaces
  float* device_obs = nullptr;          // [E, 58]
  float* device_reward = nullptr;       // [E]
  uint8_t* device_done = nullptr;       // [E]
  dwc1_diagnostic* device_diag = nullptr;  // [E]
  std::vector<DwState> initial, host;   // creation state + read scratch
  ~dwc1_scene() {
    cudaFree(device_state);
    cudaFree(device_targets);
    cudaFree(device_scratch);
    cudaFree(device_obs);
    cudaFree(device_reward);
    cudaFree(device_done);
    cudaFree(device_diag);
  }
};

namespace {
int pull_states(const dwc1_scene* s) {
  auto* mutable_scene = const_cast<dwc1_scene*>(s);
  cudaError_t err = cudaMemcpy(mutable_scene->host.data(), s->device_state,
                               sizeof(DwState) * s->E, cudaMemcpyDeviceToHost);
  return err == cudaSuccess ? DWC1_OK : DWC1_NUMERIC;
}
bool rand_config_valid(const dwc1_randomization* r) {
  if (!r) return true;
  const double ranges[4] = {r->r_mass, r->r_friction, r->r_kp, r->r_damping};
  for (double x : ranges)
    if (!(x == x) || x < 0.0 || x > 0.5) return false;
  return r->max_latency_steps <= DWC1_MAX_LATENCY && r->reserved == 0;
}
bool env_random_valid(const dwc1_randomization& cfg, const dwc1_env_random& v) {
  const double scales[4] = {v.mass_scale, v.friction_scale, v.kp_scale,
                            v.damping_scale};
  const double ranges[4] = {cfg.r_mass, cfg.r_friction, cfg.r_kp,
                            cfg.r_damping};
  for (int k = 0; k < 4; k++) {
    if (!(scales[k] == scales[k])) return false;
    if (fabs(scales[k] - 1.0) > ranges[k] + 1e-12) return false;
  }
  return v.latency_steps <= cfg.max_latency_steps;
}
}  // namespace

extern "C" {

int dwc1_abi_version(void) { return DWC1_ABI_VERSION; }

int dwc1_create(uint32_t environments, const float* joint_offsets,
                const dwc1_randomization* randomization, dwc1_scene** out) {
  if (!out || environments < 1 || environments > 1048576) return DWC1_INVALID;
  *out = nullptr;
  if (joint_offsets && !dw_finite(joint_offsets, (int)environments * DW_J))
    return DWC1_INVALID;
  if (!rand_config_valid(randomization)) return DWC1_INVALID;
  try {
    auto s = new dwc1_scene;
    s->E = environments;
    if (randomization) s->rand = *randomization;
    {
      auto tmp = std::make_unique<DwWork>();   // host-side workspace
      if (!dw_reference_weights(s->params.refweight, tmp.get())) {
        delete s;
        return DWC1_DYNAMICS;
      }
    }
    s->params.tolerance = DW_SOLVE_TOLERANCE;
    s->params.max_iterations = DW_MAX_ITERATIONS;
    s->initial.resize(environments);
    s->host.resize(environments);
    for (uint32_t e = 0; e < environments; e++)
      dw_init_state(&s->initial[e],
                    joint_offsets ? joint_offsets + (size_t)e * DW_J : nullptr);
    if (cudaDeviceSetLimit(cudaLimitStackSize, kStackBytes) != cudaSuccess
        || cudaMalloc(&s->device_state, sizeof(DwState) * environments) != cudaSuccess
        || cudaMalloc(&s->device_targets, sizeof(float) * environments * DW_J) != cudaSuccess
        || cudaMalloc(&s->device_scratch,
                      sizeof(DwWork) * (size_t)environments) != cudaSuccess
        || cudaMalloc(&s->device_obs,
                      sizeof(float) * (size_t)environments * DWP_OBS) != cudaSuccess
        || cudaMalloc(&s->device_reward, sizeof(float) * environments) != cudaSuccess
        || cudaMalloc(&s->device_done, sizeof(uint8_t) * environments) != cudaSuccess
        || cudaMalloc(&s->device_diag, sizeof(dwc1_diagnostic) * environments) != cudaSuccess
        || cudaMemcpy(s->device_state, s->initial.data(),
                      sizeof(DwState) * environments,
                      cudaMemcpyHostToDevice) != cudaSuccess) {
      delete s;
      return DWC1_ALLOCATION;
    }
    *out = s;
    return DWC1_OK;
  } catch (const std::bad_alloc&) {
    return DWC1_ALLOCATION;
  }
}

void dwc1_destroy(dwc1_scene* s) { delete s; }

int dwc1_info_get(const dwc1_scene* s, dwc1_info* info) {
  if (!s || !info) return DWC1_INVALID;
  dw_fill_info(s->E, info);
  return DWC1_OK;
}

int dwc1_device_info_get(const dwc1_scene* s, dwc1_device_info* info) {
  if (!s || !info) return DWC1_INVALID;
  *info = dwc1_device_info{};
  info->lanes_per_env = (uint32_t)kLanesPerEnv;
  info->threads_per_block = (uint32_t)kThreadsPerBlock;
  info->min_blocks_per_sm = (uint32_t)DW_MIN_BLOCKS_PER_SM;
  info->workspace_bytes_per_env = sizeof(DwWork);
  int sms = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0)
      != cudaSuccess)
    return DWC1_NUMERIC;
  info->sm_count = (uint32_t)sms;
  size_t stack = 0;
  if (cudaDeviceGetLimit(&stack, cudaLimitStackSize) != cudaSuccess)
    return DWC1_NUMERIC;
  info->stack_limit_bytes = stack;
  cudaFuncAttributes attr{};
  int blocks = 0;
  if (cudaFuncGetAttributes(&attr, (const void*)dw_step_kernel) != cudaSuccess
      || cudaOccupancyMaxActiveBlocksPerMultiprocessor(
             &blocks, (const void*)dw_step_kernel, kThreadsPerBlock, 0)
             != cudaSuccess)
    return DWC1_NUMERIC;
  info->step_regs_per_thread = (uint32_t)attr.numRegs;
  info->step_local_bytes = (uint32_t)attr.localSizeBytes;
  info->step_blocks_per_sm = (uint32_t)blocks;
  if (cudaFuncGetAttributes(&attr, (const void*)dw_step_policy_kernel)
          != cudaSuccess
      || cudaOccupancyMaxActiveBlocksPerMultiprocessor(
             &blocks, (const void*)dw_step_policy_kernel, kThreadsPerBlock, 0)
             != cudaSuccess)
    return DWC1_NUMERIC;
  info->policy_regs_per_thread = (uint32_t)attr.numRegs;
  info->policy_local_bytes = (uint32_t)attr.localSizeBytes;
  info->policy_blocks_per_sm = (uint32_t)blocks;
  const uint32_t envs_per_block =
      (uint32_t)(kThreadsPerBlock / kLanesPerEnv);
  info->resident_envs_estimate =
      info->step_blocks_per_sm * info->sm_count * envs_per_block;
  return DWC1_OK;
}

int dwc1_step(dwc1_scene* s, const float* targets, uint32_t n_ticks,
              dwc1_diagnostic* diagnostics) {
  if (!s || !targets || !diagnostics || n_ticks < 1 || n_ticks > 1000)
    return DWC1_INVALID;
  if (!dw_finite(targets, (int)s->E * DW_J)) return DWC1_INVALID;
  if (cudaMemcpy(s->device_targets, targets, sizeof(float) * s->E * DW_J,
                 cudaMemcpyHostToDevice) != cudaSuccess)
    return DWC1_NUMERIC;
  const size_t lanes_total = (size_t)s->E * kLanesPerEnv;
  const int blocks = (int)((lanes_total + kThreadsPerBlock - 1) / kThreadsPerBlock);
  dw_step_kernel<<<blocks, kThreadsPerBlock>>>(
      s->device_state, s->device_targets, n_ticks, s->params,
      s->device_scratch, s->device_diag, s->E);
  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
    return DWC1_NUMERIC;
  if (cudaMemcpy(diagnostics, s->device_diag, sizeof(dwc1_diagnostic) * s->E,
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    return DWC1_NUMERIC;
  int rc = DWC1_OK;
  for (uint32_t e = 0; e < s->E; e++)
    if (diagnostics[e].status != DWC1_OK && rc == DWC1_OK)
      rc = (int)diagnostics[e].status;
  return rc;
}

int dwc1_read(const dwc1_scene* s, float* qpos, float* velocity, float* warm,
              double* time, uint64_t* count, float* body_state,
              uint8_t* foot_contact, float* sole_height, dwc1_manifold* cache,
              uint32_t* contact_ticks) {
  if (!s) return DWC1_INVALID;
  int rc = pull_states(s);
  if (rc) return rc;
  for (uint32_t e = 0; e < s->E; e++) {
    const DwState* x = &s->host[e];
    if (contact_ticks)
      for (int p = 0; p < DW_PAIRS; p++)
        contact_ticks[(size_t)e * DW_PAIRS + p] = x->contact_ticks[p];
    if (qpos) for (int k = 0; k < DW_Q; k++) qpos[(size_t)e * DW_Q + k] = x->q[k];
    if (velocity) for (int k = 0; k < DW_N; k++) velocity[(size_t)e * DW_N + k] = x->v[k];
    if (warm) for (int k = 0; k < DW_JROWS; k++) warm[(size_t)e * DW_JROWS + k] = x->warm[k];
    if (time) time[e] = double(x->count) * 0.002;
    if (count) count[e] = x->count;
    if (body_state || sole_height) {
      float bodies[DW_B][13];
      dw_body_states(x->q, x->v, bodies);
      if (body_state)
        for (int b = 0; b < DW_B; b++)
          for (int k = 0; k < 13; k++)
            body_state[((size_t)e * DW_B + b) * 13 + k] = bodies[b][k];
      if (sole_height) dw_sole_heights(bodies, sole_height + (size_t)e * 2);
    }
    if (foot_contact)
      for (int p = 0; p < DW_PAIRS; p++)
        foot_contact[(size_t)e * DW_PAIRS + p] = x->cache[p].count > 0 ? 1 : 0;
    if (cache)
      for (int p = 0; p < DW_PAIRS; p++) cache[(size_t)e * DW_PAIRS + p] = x->cache[p];
  }
  return DWC1_OK;
}

int dwc1_query(const dwc1_scene* s, dwc1_manifold* out) {
  if (!s || !out) return DWC1_INVALID;
  int rc = pull_states(s);
  if (rc) return rc;
  for (uint32_t e = 0; e < s->E; e++) {
    float bodies[DW_B][13];
    dw_body_states(s->host[e].q, s->host[e].v, bodies);
    for (int p = 0; p < DW_PAIRS; p++) {
      int foot = (int)DW_PAIR_BODY_A[p];
      dw_plane_manifold(p, &bodies[foot][0], &bodies[foot][3],
                        &out[(size_t)e * DW_PAIRS + p]);
    }
  }
  return DWC1_OK;
}

int dwc1_reset_policy(dwc1_scene* s, const uint8_t* mask,
                      const double* commands, const double* phase_offsets) {
  if (!s) return DWC1_INVALID;
  for (uint32_t e = 0; e < s->E; e++) {
    if (commands && !(commands[e] == commands[e])) return DWC1_INVALID;
    if (phase_offsets && !(phase_offsets[e] == phase_offsets[e]))
      return DWC1_INVALID;
  }
  for (uint32_t e = 0; e < s->E; e++) {
    if (mask && !mask[e]) continue;
    DwState next = s->initial[e];
    if (commands) {
      next.command = commands[e];
    } else {  // keep the env's previous command (creation default: 0)
      if (cudaMemcpy(&next.command,
                     (const char*)(s->device_state + e)
                         + offsetof(DwState, command),
                     sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
        return DWC1_NUMERIC;
    }
    if (phase_offsets) {
      next.phase0 = phase_offsets[e];
    } else {  // keep the env's previous phase offset (creation default: 0)
      if (cudaMemcpy(&next.phase0,
                     (const char*)(s->device_state + e)
                         + offsetof(DwState, phase0),
                     sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
        return DWC1_NUMERIC;
    }
    if (cudaMemcpy(s->device_state + e, &next, sizeof(DwState),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      return DWC1_NUMERIC;
  }
  return DWC1_OK;
}

int dwc1_reset(dwc1_scene* s, const uint8_t* mask) {
  return dwc1_reset_policy(s, mask, nullptr, nullptr);
}

int dwc1_set_randomization(dwc1_scene* s, const uint8_t* mask,
                           const dwc1_env_random* randoms) {
  if (!s || !randoms) return DWC1_INVALID;
  for (uint32_t e = 0; e < s->E; e++)
    if ((!mask || mask[e]) && !env_random_valid(s->rand, randoms[e]))
      return DWC1_INVALID;
  for (uint32_t e = 0; e < s->E; e++) {
    if (mask && !mask[e]) continue;
    DwState st{};
    if (cudaMemcpy(&st, s->device_state + e, sizeof(DwState),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      return DWC1_NUMERIC;
    dw_policy_set_random(&st, &randoms[e]);
    if (cudaMemcpy(s->device_state + e, &st, sizeof(DwState),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      return DWC1_NUMERIC;
  }
  return DWC1_OK;
}

int dwc1_step_policy(dwc1_scene* s, const float* actions, uint32_t n_ticks,
                     float* obs, float* reward, uint8_t* done,
                     dwc1_diagnostic* diagnostics) {
  if (!s || !actions || !obs || !reward || !done || !diagnostics
      || n_ticks < 1 || n_ticks > 1000)
    return DWC1_INVALID;
  if (!dw_finite(actions, (int)s->E * DW_J)) return DWC1_INVALID;
  if (cudaMemcpy(s->device_targets, actions, sizeof(float) * s->E * DW_J,
                 cudaMemcpyHostToDevice) != cudaSuccess)
    return DWC1_NUMERIC;
  const size_t lanes_total = (size_t)s->E * kLanesPerEnv;
  const int blocks = (int)((lanes_total + kThreadsPerBlock - 1) / kThreadsPerBlock);
  dw_step_policy_kernel<<<blocks, kThreadsPerBlock>>>(
      s->device_state, s->device_targets, n_ticks, s->params,
      s->device_scratch, s->device_obs, s->device_reward, s->device_done,
      s->device_diag, s->E);
  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
    return DWC1_NUMERIC;
  if (cudaMemcpy(obs, s->device_obs, sizeof(float) * (size_t)s->E * DWP_OBS,
                 cudaMemcpyDeviceToHost) != cudaSuccess
      || cudaMemcpy(reward, s->device_reward, sizeof(float) * s->E,
                    cudaMemcpyDeviceToHost) != cudaSuccess
      || cudaMemcpy(done, s->device_done, sizeof(uint8_t) * s->E,
                    cudaMemcpyDeviceToHost) != cudaSuccess
      || cudaMemcpy(diagnostics, s->device_diag,
                    sizeof(dwc1_diagnostic) * s->E,
                    cudaMemcpyDeviceToHost) != cudaSuccess)
    return DWC1_NUMERIC;
  return DWC1_OK;  // per-env faults surface via diagnostics + done flags
}

int dwc1_observe(const dwc1_scene* s, float* obs) {
  if (!s || !obs) return DWC1_INVALID;
  int rc = pull_states(s);
  if (rc) return rc;
  for (uint32_t e = 0; e < s->E; e++)
    dw_policy_observe(&s->host[e], obs + (size_t)e * DWP_OBS);
  return DWC1_OK;
}

int dwc1_set_command(dwc1_scene* s, const double* commands) {
  if (!s || !commands) return DWC1_INVALID;
  for (uint32_t e = 0; e < s->E; e++) {
    if (!(commands[e] == commands[e])) return DWC1_INVALID;  // NaN
    if (cudaMemcpy((char*)(s->device_state + e) + offsetof(DwState, command),
                   &commands[e], sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      return DWC1_NUMERIC;
  }
  return DWC1_OK;
}

int dwc1_set_state(dwc1_scene* s, uint32_t environment, const float* qpos21,
                   const float* velocity20, const float* warm42,
                   const dwc1_manifold* cache2, uint64_t count) {
  if (!s || environment >= s->E || !qpos21 || !velocity20 || !warm42)
    return DWC1_INVALID;
  if (!dw_finite(qpos21, DW_Q) || !dw_finite(velocity20, DW_N)
      || !dw_finite(warm42, DW_JROWS))
    return DWC1_INVALID;
  DwState next{};
  {  // neutral randomization + reset latency ring (new fields must not be 0)
    dwc1_env_random neutral{1.0, 1.0, 1.0, 1.0, 0, 0};
    dw_policy_set_random(&next, &neutral);
  }
  for (int k = 0; k < DW_Q; k++) next.q[k] = qpos21[k];
  dw_qnormalize(next.q + 3);
  for (int k = 0; k < DW_N; k++) next.v[k] = velocity20[k];
  for (int j = 0; j < DW_J; j++) {
    next.warm[j] = dw_clampf(warm42[j], -DW_FRICTION_LOSS, DW_FRICTION_LOSS);
    next.warm[DW_J + j] = fmaxf(0.0f, warm42[DW_J + j]);
    next.warm[2 * DW_J + j] = fmaxf(0.0f, warm42[2 * DW_J + j]);
  }
  if (cache2)
    for (int p = 0; p < DW_PAIRS; p++) {
      if (cache2[p].count > DW_MAXPOINTS) return DWC1_INVALID;
      next.cache[p] = cache2[p];
    }
  next.count = count;
  return cudaMemcpy(s->device_state + environment, &next, sizeof(DwState),
                    cudaMemcpyHostToDevice) == cudaSuccess ? DWC1_OK
                                                           : DWC1_NUMERIC;
}

}  // extern "C"
