// SPDX-License-Identifier: MIT
// Serial parity driver: the single-source kernel run as a plain loop over
// environments on the host. This build is the local test vehicle (no nvcc on
// this machine) and shares every line of physics with src/duck_cuda.cu.
//
//   /usr/bin/clang++ -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off \
//       -Iinclude -fPIC -shared src/duck_cuda_serial.cpp -o libduck_cuda_serial.dylib
#include "cuda_compat.h"
#include "duck_cuda_kernel.h"

#include <new>
#include <vector>

struct dwc1_scene {
  uint32_t E = 0;
  DwParams params{};
  std::vector<DwState> state, initial;
};

extern "C" {

int dwc1_create(uint32_t environments, const float* joint_offsets,
                dwc1_scene** out) {
  if (!out || environments < 1 || environments > 65536) return DWC1_INVALID;
  *out = nullptr;
  if (joint_offsets && !dw_finite(joint_offsets, (int)environments * DW_J))
    return DWC1_INVALID;
  try {
    auto s = new dwc1_scene;
    s->E = environments;
    if (!dw_reference_weights(s->params.refweight)) { delete s; return DWC1_DYNAMICS; }
    s->params.tolerance = DW_SOLVE_TOLERANCE;
    s->params.max_iterations = DW_MAX_ITERATIONS;
    s->state.resize(environments);
    for (uint32_t e = 0; e < environments; e++)
      dw_init_state(&s->state[e],
                    joint_offsets ? joint_offsets + (size_t)e * DW_J : nullptr);
    s->initial = s->state;
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

int dwc1_step(dwc1_scene* s, const float* targets, uint32_t n_ticks,
              dwc1_diagnostic* diagnostics) {
  if (!s || !targets || !diagnostics || n_ticks < 1 || n_ticks > 1000)
    return DWC1_INVALID;
  if (!dw_finite(targets, (int)s->E * DW_J)) return DWC1_INVALID;
  int rc = DWC1_OK;
  for (uint32_t e = 0; e < s->E; e++) {
    dwc1_diagnostic* d = &diagnostics[e];
    *d = dwc1_diagnostic{};
    d->environment = e;
    dw_step_env(&s->state[e], targets + (size_t)e * DW_J, n_ticks, &s->params, d);
    if (d->status != DWC1_OK && rc == DWC1_OK) rc = (int)d->status;
  }
  return rc;
}

int dwc1_read(const dwc1_scene* s, float* qpos, float* velocity, float* warm,
              double* time, uint64_t* count, float* body_state,
              uint8_t* foot_contact, float* sole_height, dwc1_manifold* cache) {
  if (!s) return DWC1_INVALID;
  for (uint32_t e = 0; e < s->E; e++) {
    const DwState* x = &s->state[e];
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
  for (uint32_t e = 0; e < s->E; e++) {
    float bodies[DW_B][13];
    dw_body_states(s->state[e].q, s->state[e].v, bodies);
    for (int p = 0; p < DW_PAIRS; p++) {
      int foot = (int)DW_PAIR_BODY_A[p];
      dw_plane_manifold(p, &bodies[foot][0], &bodies[foot][3],
                        &out[(size_t)e * DW_PAIRS + p]);
    }
  }
  return DWC1_OK;
}

int dwc1_reset(dwc1_scene* s, const uint8_t* mask) {
  if (!s) return DWC1_INVALID;
  for (uint32_t e = 0; e < s->E; e++)
    if (!mask || mask[e]) s->state[e] = s->initial[e];
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
  s->state[environment] = next;
  return DWC1_OK;
}

// Test-only introspection (serial build): expose the fp32 dynamics terms so
// the parity tests can localize any divergence against the f64 oracle.
int dwc1_debug_eval(const float* q, const float* v, float* mass /*[N,N]*/,
                    float* bias /*[N]*/, float* pose /*[B,7]*/,
                    float* jac /*[B,6,N]*/, float* smooth /*[N]*/,
                    const float* target /*[J] or NULL*/) {
  if (!q || !v) return DWC1_INVALID;
  DwEval e;
  if (!dw_evaluate(q, v, DW_GRAVITY_Z, &e)) return DWC1_DYNAMICS;
  if (mass) memcpy(mass, &e.M[0][0], sizeof(e.M));
  if (bias) memcpy(bias, e.bias, sizeof(e.bias));
  if (pose)
    for (int b = 0; b < DW_B; b++) {
      for (int k = 0; k < 3; k++) pose[b * 7 + k] = e.pos[b][k];
      for (int k = 0; k < 4; k++) pose[b * 7 + 3 + k] = e.rot[b][k];
    }
  if (jac) memcpy(jac, &e.J[0][0][0], sizeof(e.J));
  if (smooth && target) {
    float L[DW_N][DW_N];
    if (!dw_chol(e.M, L)) return DWC1_DYNAMICS;
    for (int n = 0; n < DW_N; n++) smooth[n] = -e.bias[n];
    for (int j = 0; j < DW_J; j++) {
      float tj = dw_clampf(target[j], DW_LIMIT_LOWER[j], DW_LIMIT_UPPER[j]);
      float motor = DW_KP * (tj - q[7 + j]) + DW_KV * (0.0f - v[6 + j]);
      smooth[6 + j] += dw_clampf(motor, -DW_EFFORT_CAP, DW_EFFORT_CAP)
                     - DW_DAMPING * v[6 + j];
    }
    dw_chol_solve(L, smooth);
    for (int n = 0; n < DW_N; n++) smooth[n] = v[n] + DW_DT * smooth[n];
  }
  return DWC1_OK;
}

}  // extern "C"
