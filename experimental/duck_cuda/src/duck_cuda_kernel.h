// SPDX-License-Identifier: MIT
// Single-source fp32 physics for the batched CUDA duck lane. WARP-PER-ENV:
// 32 lanes cooperate on one environment on device (DW_WARP_LANES=32); the
// serial build and the legacy thread-per-env build run the same source with
// one lane (see the execution-model section below for the lane helpers and
// the reduction-order contract). Physics: free-root articulated dynamics
// (port of articulated_v1/v2),
// foot-vs-floor plane manifolds with stable feature ids (port of contact_v1's
// plane path), joint + contact row assembly (port of integrated_duck_v1) and
// the dense PGS solver with the degenerate-contact repairs (port of
// coupled_impulse_v1: conditional two-normal solve, Richardson extrapolation
// and the Tresca fixed-point acceleration; the rank-1 null-direction move is
// intentionally not ported -- it needs an m x m Jacobi eigensolver and the
// Tresca acceleration certifies the same rank-deficient flat-foot blocks).
//
// No dynamic allocation anywhere; fixed per-thread arrays sized by
// duck_model.h. Compiled two ways from this one header:
//   - src/duck_cuda_serial.cpp via include/cuda_compat.h (plain clang++,
//     the local parity/test vehicle),
//   - src/duck_cuda.cu under nvcc (DW_HD = __host__ __device__).
// Certificates are fp32-scaled: solver impulse tolerance DW_SOLVE_TOLERANCE,
// momentum-residual gate DW_MOMENTUM_TOLERANCE (the CPU oracle uses 1e-8 for
// both in f64).
#ifndef DUCK_CUDA_KERNEL_H
#define DUCK_CUDA_KERNEL_H
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "duck_cuda.h"
#include "duck_model.h"

#ifndef DW_HD
#error "include cuda_compat.h (serial) or compile with nvcc defining DW_HD"
#endif

#ifndef DW_SOLVE_TOLERANCE
#define DW_SOLVE_TOLERANCE 5e-6f
#endif
#ifndef DW_MOMENTUM_TOLERANCE
#define DW_MOMENTUM_TOLERANCE 2e-4f
#endif
#ifndef DW_MAX_ITERATIONS
#define DW_MAX_ITERATIONS 4096u
#endif
#define DW_CONTACT_EPS 2e-6f
#ifndef DW_APGD_BUDGET
#define DW_APGD_BUDGET 65536u
#endif
// Exhaustion-tier certificate ceiling for the degenerate CONTACT block,
// mirroring civ1: a provably-stuck solve with the full Tresca arsenal spent
// is accepted at the ceiling rather than faulted (joint rows stay strict;
// ratio ceiling/tolerance matches civ1's 1e-5/1e-8 certificate range scaled
// to fp32). Duck-scale problems converge strictly in tens of sweeps and
// never reach it (bit-identical).
#ifndef DW_TIER_CEILING
#define DW_TIER_CEILING 1e-3f
#endif

// ==================== execution model: warp-per-env ========================
// DW_WARP_LANES lanes cooperate on ONE environment. Three build modes off the
// same source:
//   serial (clang++, DW_WARP_LANES=1): DW_FOR = plain loops, DW_SYNC = no-op;
//   CUDA warp-per-env (nvcc, DW_WARP_LANES=32): DW_FOR strides by lane,
//     DW_SYNC = __syncwarp(); one warp per env, fixed-order reductions;
//   CUDA thread-per-env (legacy, -DDW_WARP_LANES=1): identical to serial
//     semantics, one thread per env.
//
// REDUCTION-ORDER CONTRACT: independent per-element work (rows of K, entries
// of residual updates, jacobian columns, mass-matrix entries) keeps its exact
// serial per-element arithmetic in every mode -- only WHICH lane computes an
// element changes. True cross-element float reductions (the Richardson /
// Tresca repair paths, certificate maxima, momentum maxima) use ONE fixed
// strided-partial + butterfly-tree order (DW_TREE lanes) in ALL builds, so
// the serial build reproduces the device reduction order too. Max reductions
// are exactly order-independent; sum reductions differ from a plain
// sequential sum only in the (rare) stalled-repair paths. Cross-BUILD bit
// equality is still not guaranteed (device sinf/expf/hypotf differ from libm
// by ULPs); tests/remote_gpu_parity.py's windowed tolerance gates plus its
// bitwise on-device determinism gate are the cross-build contract. Within a
// build, results are bit-identical across runs: fixed schedule, fixed-order
// reductions, no atomics.
#ifndef DW_WARP_LANES
#define DW_WARP_LANES 1
#endif
#if defined(__CUDA_ARCH__) && DW_WARP_LANES > 1
#define DW_LANE ((int)(threadIdx.x & (DW_WARP_LANES - 1)))
#define DW_STRIDE DW_WARP_LANES
#define DW_SYNC() __syncwarp()
#else
#define DW_LANE 0
#define DW_STRIDE 1
#define DW_SYNC() ((void)0)
#endif
#define DW_FOR(i, n) for (int i = DW_LANE; i < (n); i += DW_STRIDE)
#define DW_LANE0 if (DW_LANE == 0)
#define DW_TREE 32  // reduction tree width shared by every build

// Warp-uniform boolean OR of per-lane flags (call from converged lanes only).
static DW_HD bool dw_any(bool v) {
#if defined(__CUDA_ARCH__) && DW_WARP_LANES > 1
  return __any_sync(0xffffffffu, v) != 0;
#else
  return v;
#endif
}
// Fixed-order sum reduction: strided per-lane partials combined by an
// xor-butterfly tree. The serial build emulates the exact 32-lane order.
template <class F>
static DW_HD float dw_sum_over(int n, F f) {
#if defined(__CUDA_ARCH__) && DW_WARP_LANES > 1
  float v = 0;
  for (int i = DW_LANE; i < n; i += DW_WARP_LANES) v += f(i);
  for (int off = DW_WARP_LANES / 2; off; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
#else
  float p[DW_TREE];
  for (int l = 0; l < DW_TREE; l++) {
    p[l] = 0;
    for (int i = l; i < n; i += DW_TREE) p[l] += f(i);
  }
  for (int off = DW_TREE / 2; off; off >>= 1)
    for (int l = 0; l < off; l++) p[l] += p[l ^ off];
  return p[0];
#endif
}
// Fixed-order max reduction of NON-NEGATIVE per-element values (max is
// exactly associative, so this equals the plain sequential max bit for bit).
template <class F>
static DW_HD float dw_max_over(int n, F f) {
#if defined(__CUDA_ARCH__) && DW_WARP_LANES > 1
  float v = 0;
  for (int i = DW_LANE; i < n; i += DW_WARP_LANES) v = fmaxf(v, f(i));
  for (int off = DW_WARP_LANES / 2; off; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
#else
  float v = 0;
  for (int i = 0; i < n; i++) v = fmaxf(v, f(i));
  return v;
#endif
}
// Butterfly max of an already-computed per-lane partial (device); identity
// in single-lane builds.
static DW_HD float dw_lane_max(float v) {
#if defined(__CUDA_ARCH__) && DW_WARP_LANES > 1
  for (int off = DW_WARP_LANES / 2; off; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
#endif
  return v;
}

// ---------------------------------------------------------------- small math
static DW_HD void dw_add3(const float* a, const float* b, float* o) {
  o[0] = a[0] + b[0]; o[1] = a[1] + b[1]; o[2] = a[2] + b[2];
}
static DW_HD void dw_sub3(const float* a, const float* b, float* o) {
  o[0] = a[0] - b[0]; o[1] = a[1] - b[1]; o[2] = a[2] - b[2];
}
static DW_HD void dw_scale3(const float* a, float t, float* o) {
  o[0] = a[0] * t; o[1] = a[1] * t; o[2] = a[2] * t;
}
static DW_HD float dw_dot3(const float* a, const float* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
static DW_HD void dw_cross3(const float* a, const float* b, float* o) {
  o[0] = a[1] * b[2] - a[2] * b[1];
  o[1] = a[2] * b[0] - a[0] * b[2];
  o[2] = a[0] * b[1] - a[1] * b[0];
}
static DW_HD void dw_qmul(const float* a, const float* b, float* o) {
  float v[3], t1[3], t2[3];
  dw_scale3(b, a[3], t1); dw_scale3(a, b[3], t2); dw_add3(t1, t2, v);
  dw_cross3(a, b, t1); dw_add3(v, t1, v);
  o[0] = v[0]; o[1] = v[1]; o[2] = v[2];
  o[3] = a[3] * b[3] - dw_dot3(a, b);
}
static DW_HD void dw_qnormalize(float* q) {
  float n = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  q[0] /= n; q[1] /= n; q[2] /= n; q[3] /= n;
}
static DW_HD void dw_rotate(const float* q, const float* v, float* o) {
  float t[3], u[3], c[3];
  dw_cross3(q, v, t); dw_scale3(t, 2.0f, t);
  dw_scale3(t, q[3], u); dw_add3(v, u, u);
  dw_cross3(q, t, c); dw_add3(u, c, o);
}
static DW_HD void dw_qexp(const float* r, float* o) {  // rotation vector -> quat
  float angle = sqrtf(dw_dot3(r, r));
  float s = angle < 1e-8f ? 0.5f - angle * angle / 48.0f
                          : sinf(angle * 0.5f) / angle;
  o[0] = r[0] * s; o[1] = r[1] * s; o[2] = r[2] * s; o[3] = cosf(angle * 0.5f);
}
// world inertia response: rot * diag(I_principal) * rot^-1 * w
static DW_HD void dw_inertia(const float* q, const float* principal,
                             const float* w, float* o) {
  float inv[4] = {-q[0], -q[1], -q[2], q[3]}, local[3];
  dw_rotate(inv, w, local);
  local[0] *= principal[0]; local[1] *= principal[1]; local[2] *= principal[2];
  dw_rotate(q, local, o);
}
static DW_HD bool dw_finite(const float* p, int n) {
  for (int i = 0; i < n; i++) if (!isfinite(p[i])) return false;
  return true;
}
static DW_HD float dw_clampf(float x, float lo, float hi) {
  return fminf(hi, fmaxf(lo, x));
}

// -------------------------------------------------------------------- state
typedef struct DwState {
  float q[DW_Q];               // root xyz, root quat xyzw, 14 hinge angles
  float v[DW_N];               // world root linear, world root angular, qdot
  float warm[DW_JROWS];        // joint warm forces: friction/lower/upper slots
  dwc1_manifold cache[DW_PAIRS];  // previous solve manifolds (warm start)
  uint64_t count;              // accepted ticks
  // Per-foot contact tick counters: zeroed at the start of every dwc1_step
  // call, +1 per accepted tick whose solve manifold has contact points
  // (same predicate as the foot_contact read flag). Lets the reward's
  // flicker penalty read per-tick contact with ONE read per policy step.
  uint32_t contact_ticks[DW_PAIRS];
  // ---- device policy layer (walk/env/flat.py + reward.py contract) ----
  // The policy chain runs in f64, mirroring the numpy env operation for
  // operation: this makes the effective PD targets (and hence the physics
  // trajectory) BIT-IDENTICAL to FlatFloorDuckEnv driving the same lane
  // build, and the tracker/threshold comparisons decide identically. Cost:
  // ~300 f64 ops per env per policy step vs ~1M f32 physics ops.
  double targets[DW_J];        // persistent slew reference (pre-limit-clip)
  double command;              // commanded forward velocity (m/s)
  double phase0;               // per-episode gait-phase offset (flat.py v10)
  double v_avg, double_support;            // GaitTracker scalars
  double air_time[2], stance_time[2];      // GaitTracker per foot (L, R)
  double pre_swing_stance[2], opp_support[2], liftoff_x[2];
  float prev_action[DW_J];        // obs[28:42] (updated only while live)
  float prev_state_action[DW_J];  // reward action-rate reference (all envs)
  int32_t last_foot;              // last qualified footfall (-1 none)
  uint32_t t;                     // policy steps this episode
  uint8_t done, prev_contact[2], policy_pad;
  // ---- per-episode domain randomization + actuation latency (ABI v6) ----
  // Neutral values (1.0 scales, latency 0) reproduce today's behavior BIT
  // EXACTLY: every scale applies at a consumption point as
  // (float)((double)constant * scale), which is the identity at 1.0, and
  // latency 0 reads back the just-written ring slot. Values are drawn
  // host-side per episode (walk/env/cuda_lane.py documents the exact RNG
  // stream) and applied via dwc1_set_randomization; any reset returns them
  // to neutral until the next set_randomization.
  double mass_scale;              // scales body masses AND principal inertias
  double friction_scale;          // scales contact-pair mu
  double kp_scale;                // scales PD stiffness (physics + reward est)
  double damping_scale;           // scales passive joint damping
  double eff_ring[DWC1_MAX_LATENCY + 1][DW_J];  // effective-target history
  uint32_t latency_steps, rand_pad;  // PD consumes eff[t - latency_steps]
} DwState;

typedef struct DwParams {
  float refweight[DW_J];       // reference-pose inverse-mass diagonal (av2)
  float tolerance;             // solver certificate tolerance (impulse units)
  uint32_t max_iterations;
  // Training-lane throughput switch (dwc1_set_fast_termination, default 0):
  // when set, dwc1_step_policy latches the termination predicate after
  // every accepted tick (a fallen env stops ticking mid-block) and skips
  // physics entirely for envs already done at block entry. Post-fall
  // impact piles peg the solver at DW_MAX_ITERATIONS plus the full Tresca
  // arsenal every tick (measured ~25-100x a live solve), so they dominate
  // whole launches; their remaining ticks have zero training value. LIVE
  // env arithmetic is untouched and the flag defaults OFF, keeping the
  // physics path, the duck fingerprint and the python-env parity gates
  // bit-identical.
  uint32_t fast_termination;
} DwParams;

typedef struct DwEval {
  float pos[DW_B][3], rot[DW_B][4];
  float velb[DW_B][6];                 // world COM linear then angular
  float J[DW_B][6][DW_N];              // spatial jacobian, linear rows first
  float M[DW_N][DW_N];                 // generalized mass incl. armature
  float bias[DW_N];                    // gravity + Coriolis + gyroscopic
} DwEval;

// Compacted row problem (only active rows enter the solve; skipped rows have
// identically zero jacobian and pinned zero impulse on the CPU oracle too).
typedef struct DwRows {
  int R, C;
  float G[DW_MAXROWS][DW_N];
  float target[DW_MAXROWS], reg[DW_MAXROWS];
  float lo[DW_MAXROWS], hi[DW_MAXROWS], warm[DW_MAXROWS];
  int kind[DW_MAXROWS];   // 0 scalar joint row, 1 normal, 2/3 tangents
  int slot[DW_MAXROWS];   // joint warm slot [0,42) or -1 for contact rows
  int cfirst[DW_PAIRS * DW_MAXPOINTS];
  float cmu[DW_PAIRS * DW_MAXPOINTS];
} DwRows;

// Per-env workspace: EVERYTHING the tick touches cooperatively lives here
// (lanes of a warp share it), sliced out of ONE global buffer allocated at
// scene creation (device) / a heap vector (serial). Per-lane registers stay
// lean; nothing tick-sized remains in thread-local storage.
typedef struct DwWork {
  DwEval e;
  DwRows rows;
  float L[DW_N][DW_N];                    // mass Cholesky factor
  float K[DW_MAXROWS][DW_MAXROWS];        // row response matrix
  float response[DW_MAXROWS][DW_N];       // M^-1 G^T rows
  float base[DW_MAXROWS], residual[DW_MAXROWS], lambda[DW_MAXROWS];
  float wprev[DW_MAXROWS], wprev2[DW_MAXROWS];
  float snapl[DW_MAXROWS], snapr[DW_MAXROWS];   // repair snapshots
  float ax[DW_MAXROWS], ay[DW_MAXROWS], axn[DW_MAXROWS];  // APGD iterates
  float agrad[DW_MAXROWS], abestx[DW_MAXROWS];
  float acap[DW_PAIRS * DW_MAXPOINTS];
  float smooth[DW_N], vnew[DW_N], qnew[DW_Q];
  float warmnew[DW_JROWS];
  float eff[DW_J];                        // effective PD targets (policy)
  double a64[DW_J], eff64[DW_J];          // policy-layer f64 chain (lane 0)
  double app64[DW_J];                     // APPLIED (latency-delayed) targets
  dwc1_manifold manifolds[DW_PAIRS];
  float bodies13[DW_B][13];               // policy-layer FK (lane 0)
  // dw_evaluate cooperative workspace
  float w3[DW_B][3], acc3[DW_B][3], alpha3[DW_B][3];
  float jrp[DW_J][3], jax[DW_J][3], jrc[DW_J][3];  // per-joint FK anchors/axis
  float forceb[DW_B][3], torqueb[DW_B][3];
  float Ijw[DW_B][DW_N][3];               // world inertia * Jw column
  uint32_t colmask[DW_B];                 // dof-column bitmask per body
  int iflag;                              // lane0 -> warp status broadcasts
} DwWork;
#define DW_SCRATCH_FLOATS ((int)(sizeof(DwWork) / sizeof(float)))

// ------------------------------------------------------------ forward kinem.
// Port of articulated_v1's evaluate(): FK, body velocities/accelerations,
// spatial jacobians, generalized mass and bias, plus av2's armature add.
// Lane-cooperative: the sequential FK chain runs on lane 0; jacobian columns,
// mass entries and bias entries are independent and lane-parallel with the
// exact per-element serial arithmetic (mass/bias entries accumulate over
// bodies in ascending order, matching the original body-major loops).
// mass_scale (domain randomization, neutral 1.0) scales every body mass AND
// principal inertia at the consumption points as (float)((double)X * scale):
// bit-identical to the unscaled path at 1.0, physically consistent otherwise
// (inertia is linear in mass for fixed geometry).
static DW_HD bool dw_evaluate(const float* q, const float* v, float gz,
                              double mass_scale, DwWork* w) {
  DwEval* e = &w->e;
  DW_FOR(i, DW_B * 6 * DW_N) (&e->J[0][0][0])[i] = 0;
  DW_SYNC();
  // ---- lane 0: sequential FK chain (identical to the scalar port) ----
  DW_LANE0 {
    for (int k = 0; k < 3; k++) {
      e->pos[0][k] = 0; w->w3[0][k] = 0; w->acc3[0][k] = 0; w->alpha3[0][k] = 0;
    }
    e->rot[0][0] = e->rot[0][1] = e->rot[0][2] = 0; e->rot[0][3] = 1;
    for (int k = 0; k < 6; k++) e->velb[0][k] = 0;
    w->colmask[0] = 0;
    float offset[3];
    dw_rotate(q + 3, DW_ROOT_COM, offset);
    dw_add3(q, offset, e->pos[1]);
    dw_qmul(q + 3, DW_ROOT_QPC, e->rot[1]);
    dw_qnormalize(e->rot[1]);
    w->w3[1][0] = v[3]; w->w3[1][1] = v[4]; w->w3[1][2] = v[5];
    float wxo[3];
    dw_cross3(w->w3[1], offset, wxo);
    dw_add3(v, wxo, e->velb[1]);
    dw_cross3(w->w3[1], wxo, w->acc3[1]);
    w->alpha3[1][0] = w->alpha3[1][1] = w->alpha3[1][2] = 0;
    w->colmask[1] = 0x3fu;
    for (int k = 0; k < 3; k++) {
      float basis[3] = {0, 0, 0}, cb[3];
      basis[k] = 1;
      e->J[1][k][k] = 1;
      dw_cross3(basis, offset, cb);
      for (int a = 0; a < 3; a++) {
        e->J[1][a][3 + k] = cb[a];
        e->J[1][3 + a][3 + k] = basis[a];
      }
    }
    e->velb[1][3] = w->w3[1][0]; e->velb[1][4] = w->w3[1][1];
    e->velb[1][5] = w->w3[1][2];
    for (int j = 0; j < DW_J; j++) {
      int b = j + 2, p = (int)DW_HINGE_PARENT[j];
      float ax[3], eq[4], t[4];
      dw_rotate(e->rot[p], DW_HINGE_AP[j], w->jrp[j]);
      dw_rotate(e->rot[p], DW_HINGE_AXIS[j], w->jax[j]);
      dw_scale3(DW_HINGE_AXIS[j], q[7 + j], ax);
      dw_qexp(ax, eq);
      dw_qmul(e->rot[p], eq, t);
      dw_qmul(t, DW_HINGE_REF[j], e->rot[b]);
      dw_qnormalize(e->rot[b]);
      dw_rotate(e->rot[b], DW_HINGE_AC[j], w->jrc[j]);
      dw_add3(e->pos[p], w->jrp[j], e->pos[b]);
      dw_sub3(e->pos[b], w->jrc[j], e->pos[b]);
      float qd = v[6 + j], tmp[3], tmp2[3];
      dw_scale3(w->jax[j], qd, tmp); dw_add3(w->w3[p], tmp, w->w3[b]);
      dw_cross3(w->w3[p], w->jax[j], tmp); dw_scale3(tmp, qd, tmp);
      dw_add3(w->alpha3[p], tmp, w->alpha3[b]);
      dw_cross3(w->w3[p], w->jrp[j], tmp); dw_add3(e->velb[p], tmp, e->velb[b]);
      dw_cross3(w->w3[b], w->jrc[j], tmp); dw_sub3(e->velb[b], tmp, e->velb[b]);
      float aa[3];
      dw_cross3(w->alpha3[p], w->jrp[j], tmp); dw_add3(w->acc3[p], tmp, aa);
      dw_cross3(w->w3[p], w->jrp[j], tmp); dw_cross3(w->w3[p], tmp, tmp2);
      dw_add3(aa, tmp2, aa);
      dw_cross3(w->alpha3[b], w->jrc[j], tmp); dw_sub3(aa, tmp, w->acc3[b]);
      dw_cross3(w->w3[b], w->jrc[j], tmp); dw_cross3(w->w3[b], tmp, tmp2);
      dw_sub3(w->acc3[b], tmp2, w->acc3[b]);
      e->velb[b][3] = w->w3[b][0]; e->velb[b][4] = w->w3[b][1];
      e->velb[b][5] = w->w3[b][2];
      w->colmask[b] = w->colmask[p] | (1u << (6 + j));
    }
  }
  DW_SYNC();
  // ---- jacobian columns: one lane propagates one column down the chain ----
  DW_FOR(nn, DW_N) {
    for (int j = 0; j < DW_J; j++) {
      int b = j + 2, p = (int)DW_HINGE_PARENT[j];
      if (!((w->colmask[b] >> nn) & 1u)) continue;  // untouched exact zeros
      float ang[3] = {e->J[p][3][nn], e->J[p][4][nn], e->J[p][5][nn]};
      float lin[3] = {e->J[p][0][nn], e->J[p][1][nn], e->J[p][2][nn]};
      float full_ang[3] = {ang[0], ang[1], ang[2]};
      if (nn == 6 + j) dw_add3(full_ang, w->jax[j], full_ang);
      float tmp[3];
      dw_cross3(ang, w->jrp[j], tmp); dw_add3(lin, tmp, lin);
      dw_cross3(full_ang, w->jrc[j], tmp); dw_sub3(lin, tmp, lin);
      for (int a = 0; a < 3; a++) {
        e->J[b][a][nn] = lin[a];
        e->J[b][3 + a][nn] = full_ang[a];
      }
    }
  }
  DW_SYNC();
  // ---- per-body force/torque and inertia-weighted Jw columns ----
  const float gravity[3] = {0, 0, gz};
  DW_FOR(b2, DW_B - 1) {
    int b = b2 + 1;
    float iw[3], tmp[3];
    float msf = (float)((double)DW_BODY_MASS[b] * mass_scale);
    float in3[3] = {(float)((double)DW_BODY_INERTIA[b][0] * mass_scale),
                    (float)((double)DW_BODY_INERTIA[b][1] * mass_scale),
                    (float)((double)DW_BODY_INERTIA[b][2] * mass_scale)};
    dw_inertia(e->rot[b], in3, w->w3[b], iw);
    dw_sub3(w->acc3[b], gravity, w->forceb[b]);
    dw_scale3(w->forceb[b], msf, w->forceb[b]);
    dw_inertia(e->rot[b], in3, w->alpha3[b], w->torqueb[b]);
    dw_cross3(w->w3[b], iw, tmp); dw_add3(w->torqueb[b], tmp, w->torqueb[b]);
  }
  DW_FOR(idx, (DW_B - 1) * DW_N) {
    int b = 1 + idx / DW_N, n = idx % DW_N;
    if (!((w->colmask[b] >> n) & 1u)) continue;
    float in3[3] = {(float)((double)DW_BODY_INERTIA[b][0] * mass_scale),
                    (float)((double)DW_BODY_INERTIA[b][1] * mass_scale),
                    (float)((double)DW_BODY_INERTIA[b][2] * mass_scale)};
    float jw[3] = {e->J[b][3][n], e->J[b][4][n], e->J[b][5][n]};
    dw_inertia(e->rot[b], in3, jw, w->Ijw[b][n]);
  }
  DW_SYNC();
  // ---- mass entries (n >= m) and bias entries, bodies in ascending order --
  DW_FOR(p2, DW_N * (DW_N + 1) / 2) {
    int n = 0, acc = 0;
    while (acc + n + 1 <= p2) { acc += n + 1; n++; }
    int m = p2 - acc;
    uint32_t need = (1u << n) | (1u << m);
    float x = 0;
    for (int b = 1; b < DW_B; b++) {
      if ((w->colmask[b] & need) != need) continue;
      float jv[3] = {e->J[b][0][n], e->J[b][1][n], e->J[b][2][n]};
      float jw[3] = {e->J[b][3][n], e->J[b][4][n], e->J[b][5][n]};
      float jv2[3] = {e->J[b][0][m], e->J[b][1][m], e->J[b][2][m]};
      x += (float)((double)DW_BODY_MASS[b] * mass_scale) * dw_dot3(jv, jv2)
         + dw_dot3(jw, w->Ijw[b][m]);
    }
    e->M[n][m] = x;
    if (m != n) e->M[m][n] = x;
  }
  DW_FOR(n, DW_N) {
    float bx = 0;
    for (int b = 1; b < DW_B; b++) {
      if (!((w->colmask[b] >> n) & 1u)) continue;
      float jv[3] = {e->J[b][0][n], e->J[b][1][n], e->J[b][2][n]};
      float jw[3] = {e->J[b][3][n], e->J[b][4][n], e->J[b][5][n]};
      bx += dw_dot3(jv, w->forceb[b]) + dw_dot3(jw, w->torqueb[b]);
    }
    e->bias[n] = bx;
  }
  DW_SYNC();
  DW_FOR(j, DW_J) e->M[6 + j][6 + j] += DW_ARMATURE;
  DW_SYNC();
  bool bad = false;
  DW_FOR(i, DW_N * DW_N) bad |= !isfinite((&e->M[0][0])[i]);
  DW_FOR(i, DW_N) bad |= !isfinite(e->bias[i]);
  DW_FOR(i, DW_B * 3) bad |= !isfinite((&e->pos[0][0])[i]);
  DW_FOR(i, DW_B * 6) bad |= !isfinite((&e->velb[0][0])[i]);
  return !dw_any(bad);
}

// Cholesky of the generalized mass (in place lower factor); false if not SPD.
static DW_HD bool dw_chol(const float M[DW_N][DW_N], float L[DW_N][DW_N]) {
  for (int i = 0; i < DW_N; i++)
    for (int j = 0; j < DW_N; j++) L[i][j] = 0;
  for (int i = 0; i < DW_N; i++) {
    for (int j = 0; j <= i; j++) {
      float x = M[i][j];
      for (int k = 0; k < j; k++) x -= L[i][k] * L[j][k];
      if (i == j) {
        if (!(x > 0) || !isfinite(x)) return false;
        L[i][j] = sqrtf(x);
      } else {
        L[i][j] = x / L[j][j];
      }
    }
  }
  return true;
}
static DW_HD void dw_chol_solve(const float L[DW_N][DW_N], float* x) {
  for (int i = 0; i < DW_N; i++) {
    float s = x[i];
    for (int k = 0; k < i; k++) s -= L[i][k] * x[k];
    x[i] = s / L[i][i];
  }
  for (int i = DW_N - 1; i >= 0; i--) {
    float s = x[i];
    for (int k = i + 1; k < DW_N; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
}

// av2 reference weights: inverse-mass diagonal at the reference pose, zero
// velocity and zero gravity (feeds the soft-row regularizers). Host-side
// (scene creation) only; the caller provides a workspace.
static DW_HD bool dw_reference_weights(float out[DW_J], DwWork* w) {
  float zero[DW_N] = {};
  if (!dw_evaluate(DW_REFERENCE_QPOS, zero, 0.0f, 1.0, w)) return false;
  if (!dw_chol(w->e.M, w->L)) return false;
  for (int j = 0; j < DW_J; j++) {
    float x[DW_N] = {};
    x[6 + j] = 1;
    dw_chol_solve(w->L, x);
    out[j] = x[6 + j];
    if (!(out[j] > 0) || !isfinite(out[j])) return false;
  }
  return true;
}

// ------------------------------------------------------- contact (contact_v1)
static DW_HD void dw_basis(const float* n, float* u, float* v) {
  float ax = fabsf(n[0]), ay = fabsf(n[1]), az = fabsf(n[2]);
  float s[3] = {0, 0, 0};
  if (ax <= ay && ax <= az) s[0] = 1;
  else if (ay <= az) s[1] = 1;
  else s[2] = 1;
  dw_cross3(s, n, u);
  float len = sqrtf(dw_dot3(u, u));
  u[0] /= len; u[1] /= len; u[2] /= len;
  dw_cross3(n, u, v);
}

typedef struct DwCandidate { float p[3]; float depth; uint64_t id; } DwCandidate;

// contact_v1 reduce(): dedupe, keep <=4 (deepest + max-min-spread), id order.
static DW_HD void dw_reduce(const float* n, DwCandidate* pts, int count,
                            dwc1_manifold* m) {
  memset(m, 0, sizeof(*m));
  if (!count) return;
  float u[3], v[3];
  dw_basis(n, u, v);
  for (int k = 0; k < 3; k++) { m->normal[k] = n[k]; m->tangent1[k] = u[k]; m->tangent2[k] = v[k]; }
  // insertion sort by id (plane candidates are already ordered; keep exact)
  for (int i = 1; i < count; i++) {
    DwCandidate x = pts[i];
    int j = i - 1;
    while (j >= 0 && pts[j].id > x.id) { pts[j + 1] = pts[j]; j--; }
    pts[j + 1] = x;
  }
  DwCandidate unique[DW_FOOT_VERTS];
  int nu = 0;
  for (int i = 0; i < count; i++) {
    bool duplicate = false;
    for (int k = 0; k < nu; k++) {
      float d[3];
      dw_sub3(pts[i].p, unique[k].p, d);
      if (sqrtf(dw_dot3(d, d)) < DW_CONTACT_EPS * 0.5f) { duplicate = true; break; }
    }
    if (!duplicate) unique[nu++] = pts[i];
  }
  int chosen[4], nc = 0;
  if (nu <= 4) {
    for (int i = 0; i < nu; i++) chosen[nc++] = i;
  } else {
    int deep = 0;
    for (int i = 1; i < nu; i++) if (unique[i].depth > unique[deep].depth) deep = i;
    chosen[nc++] = deep;
    while (nc < 4) {
      float best = -1; int bi = 0;
      for (int i = 0; i < nu; i++) {
        bool taken = false;
        for (int k = 0; k < nc; k++) if (chosen[k] == i) { taken = true; break; }
        if (taken) continue;
        float score = 1e30f;
        for (int k = 0; k < nc; k++) {
          float d[3];
          dw_sub3(unique[i].p, unique[chosen[k]].p, d);
          score = fminf(score, dw_dot3(d, d));
        }
        if (score > best) { best = score; bi = i; }
      }
      chosen[nc++] = bi;
    }
  }
  for (int i = 1; i < nc; i++) {  // sort chosen by candidate id
    int x = chosen[i], j = i - 1;
    while (j >= 0 && unique[chosen[j]].id > unique[x].id) { chosen[j + 1] = chosen[j]; j--; }
    chosen[j + 1] = x;
  }
  m->count = (uint32_t)nc;
  for (int i = 0; i < nc; i++) {
    const DwCandidate* p = &unique[chosen[i]];
    m->points[i].feature = p->id;
    for (int k = 0; k < 3; k++) m->points[i].point[k] = p->p[k];
    m->points[i].depth = p->depth;
  }
}

// Foot (convex, body_a) vs static floor plane z=0 (body_b). Matches
// contact_v1's plane_contact with plane_is_a=false: manifold normal is -up.
static DW_HD void dw_plane_manifold(int pair, const float pos[3],
                                    const float rot[4], dwc1_manifold* m) {
  const float n[3] = {0, 0, 1};
  DwCandidate pts[DW_FOOT_VERTS];
  int count = 0;
  for (int i = 0; i < DW_FOOT_VERTS; i++) {
    float world[3];
    dw_rotate(rot, DW_FOOT_VERTICES[pair][i], world);
    dw_add3(pos, world, world);
    float sep = world[2];  // dot(n, world) - offset, offset = 0
    if (sep <= DW_CONTACT_EPS) {
      DwCandidate* c = &pts[count++];
      float shift[3];
      dw_scale3(n, sep * 0.5f, shift);
      dw_sub3(world, shift, c->p);
      c->depth = fmaxf(0.0f, -sep);
      c->id = 0x100000000ull + (uint64_t)i + 1;
    }
  }
  const float mn[3] = {0, 0, -1};
  dw_reduce(mn, pts, count, m);
}

// idv1 row() lives inline in dw_tick's lane-parallel contact-jacobian fill
// (one lane per (row, column) work item, per-column arithmetic unchanged).

// --------------------------------------------------------------- joint rows
static DW_HD float dw_bounded_impedance(float x) {
  return dw_clampf(x, 1e-4f, 0.9999f);
}
static DW_HD float dw_impedance(float gap) {  // av2 impedance() with our solimp
  float d0 = dw_bounded_impedance(DW_LIMIT_SOLIMP[0]);
  float dwv = dw_bounded_impedance(DW_LIMIT_SOLIMP[1]);
  float width = DW_LIMIT_SOLIMP[2];
  float mid = dw_bounded_impedance(DW_LIMIT_SOLIMP[3]);
  float power = DW_LIMIT_SOLIMP[4];
  if (d0 == dwv || width <= 1e-15f) return 0.5f * (d0 + dwv);
  float x = fabsf((gap - DW_LIMIT_MARGIN) / width);
  if (x >= 1) return dwv;
  if (x <= 0) return d0;
  float y = power == 1 ? x
      : (x <= mid ? powf(x, power) / powf(mid, power - 1)
                  : 1 - powf(1 - x, power) / powf(1 - mid, power - 1));
  return d0 + (dwv - d0) * y;
}

// (DwRows and the DwWork workspace are defined next to DwEval above.)

// -------------------------------------------------- dense solver (civ1 port)
typedef struct DwDisk { float x, y; } DwDisk;
static DW_HD bool dw_disk(float a, float b, float c, float fx, float fy,
                          float cap, DwDisk* out) {
  if (!(isfinite(cap) && cap >= 0)) return false;
  if (cap == 0) { out->x = out->y = 0; return true; }
  float determinant = a * c - b * b;
  if (!(a > 0 && c > 0 && determinant > 0 && isfinite(determinant))) return false;
  bool ok = true;
  auto at = [&](float l) {
    DwDisk v = {0, 0};
    float d = (a + l) * (c + l) - b * b;
    if (!(isfinite(d) && d > 0)) { ok = false; return v; }
    float nx = -(c + l) * fx + b * fy, ny = b * fx - (a + l) * fy;
    if (!(isfinite(nx) && isfinite(ny))) { ok = false; return v; }
    v.x = nx / d; v.y = ny / d;
    if (!(isfinite(v.x) && isfinite(v.y))) ok = false;
    return v;
  };
  DwDisk fr = at(0);
  if (!ok) return false;
  if (hypotf(fr.x, fr.y) <= cap) { *out = fr; return true; }
  float lo = 0;
  float hi = fmaxf(fmaxf(a, c), fmaxf(hypotf(fx, fy) / cap, 1e-30f));
  if (!isfinite(hi)) return false;
  for (int k = 0; k < 128; k++) {
    DwDisk v = at(hi);
    if (!ok) return false;
    if (hypotf(v.x, v.y) <= cap) break;
    hi *= 2;
    if (!isfinite(hi)) return false;
  }
  {
    DwDisk v = at(hi);
    if (!ok || hypotf(v.x, v.y) > cap) return false;
  }
  for (int k = 0; k < 48; k++) {
    float mid = lo + (hi - lo) * 0.5f;
    DwDisk v = at(mid);
    if (!ok) return false;
    if (hypotf(v.x, v.y) > cap) lo = mid; else hi = mid;
  }
  *out = at(hi);
  return ok;
}

// Full generalized non-associated Coulomb solve; see coupled_impulse_v1.cpp.
// Lane-cooperative: the PGS row sweep stays strictly sequential (Gauss-Seidel
// ordering is part of the physics contract); each row's rank-1 residual
// update, the K build, the response rows, the certificates and the repair
// paths are lane-parallel. Per-element arithmetic is serial-identical; the
// only cross-element float sums (Richardson/Tresca repair reductions) use
// the fixed dw_sum_over tree in every build. Small scalar work (2x2 disk
// solves, stall bookkeeping) runs redundantly on all lanes from shared
// inputs, which keeps control flow warp-uniform by construction.
// Returns a dwc1 status; on success w->vnew holds the velocity and
// w->lambda the row impulses.
static DW_HD int dw_solve(DwWork* w, float tolerance, uint32_t max_iterations,
                          dwc1_diagnostic* diag) {
  const DwRows* rows = &w->rows;
  const int R = rows->R, C = rows->C;
  static_assert(DW_MAXROWS == DW_JROWS + 3 * DW_PAIRS * DW_MAXPOINTS, "rows");
  float (*K)[DW_MAXROWS] = w->K;
  float (*response)[DW_N] = w->response;
  float* lambda = w->lambda;
  float* base = w->base;
  float* residual = w->residual;
  if (R == 0) {
    DW_FOR(k, DW_N) w->vnew[k] = w->smooth[k];
    DW_LANE0 {
      diag->iterations = 0;
      diag->joint_residual = diag->normal_residual = diag->tangent_residual = 0;
      diag->momentum_residual = 0;
    }
    DW_SYNC();
    return DWC1_OK;
  }
  // per-lane numeric flag; consolidated with dw_any at warp-uniform points
  // (the scalar-vs-fp32 abort now happens at those checkpoints instead of
  // mid-loop -- reachable only on numeric blow-up, same DWC1_NUMERIC result).
  bool numeric = false;
  DW_FOR(r, R) {
    float b0 = -rows->target[r];
    for (int j = 0; j < DW_N; j++) {
      float x = rows->G[r][j];
      b0 += x * w->smooth[j];
      response[r][j] = x;
    }
    base[r] = b0;
    dw_chol_solve(w->L, response[r]);
    lambda[r] = dw_clampf(rows->warm[r], rows->lo[r], rows->hi[r]);
  }
  DW_SYNC();
  DW_FOR(p, R * R) {
    int i = p / R, j = p % R;
    float x = i == j ? rows->reg[i] : 0.0f;
    for (int k = 0; k < DW_N; k++) x += rows->G[i][k] * response[j][k];
    if (!isfinite(x)) numeric = true;
    K[i][j] = x;
  }
  DW_SYNC();
  DW_FOR(i, R)
    if (!(K[i][i] > 0 || (rows->lo[i] == 0 && rows->hi[i] == 0)))
      numeric = true;
  numeric = dw_any(numeric);
  if (numeric) return DWC1_NUMERIC;
  DW_LANE0 for (int c = 0; c < C; c++) {
    int r = rows->cfirst[c];
    float cap = rows->cmu[c] * lambda[r];
    float norm = hypotf(lambda[r + 1], lambda[r + 2]);
    if (norm > cap) { lambda[r + 1] *= cap / norm; lambda[r + 2] *= cap / norm; }
  }
  DW_SYNC();
  auto residuals = [&]() {
    DW_FOR(i, R) {
      float x = base[i];
      for (int j = 0; j < R; j++) x += K[i][j] * lambda[j];
      residual[i] = x;
      if (!isfinite(x)) numeric = true;
    }
    DW_SYNC();
  };
  auto scalar = [&](int r) {
    if (rows->lo[r] == rows->hi[r]) return rows->lo[r];
    float candidate = lambda[r] - residual[r] / K[r][r];
    if (!isfinite(candidate)) numeric = true;
    return dw_clampf(candidate, rows->lo[r], rows->hi[r]);
  };
  auto scalar_error = [&](int r) {
    if (rows->lo[r] == rows->hi[r]) return fabsf(lambda[r] - rows->lo[r]);
    if (residual[r] == 0) return 0.0f;
    float correction = residual[r] / K[r][r];
    if (!isfinite(correction)) { numeric = true; return 0.0f; }
    float distance = correction > 0 ? lambda[r] - rows->lo[r]
                                    : rows->hi[r] - lambda[r];
    if (!(distance >= 0)) { numeric = true; return 0.0f; }
    return fminf(fabsf(correction), distance);
  };
  auto update = [&](int r, float x) {
    float delta = x - lambda[r];
    if (!(isfinite(x) && isfinite(delta))) numeric = true;
    DW_SYNC();  // every lane must read the old lambda[r] before the store
    DW_FOR(i, R) {
      residual[i] += K[i][r] * delta;
      if (!isfinite(residual[i])) numeric = true;
    }
    DW_LANE0 lambda[r] = x;
    DW_SYNC();
  };
  auto tangent_error = [&](int r, float mu) {
    int a = r + 1, b = r + 2;
    float cap = mu * lambda[r], norm = hypotf(lambda[a], lambda[b]);
    if (!(isfinite(cap) && isfinite(norm))) { numeric = true; return 0.0f; }
    if (cap == 0) return norm;
    float aa = K[a][a], ab = 0.5f * (K[a][b] + K[b][a]), bb = K[b][b];
    float det = aa * bb - ab * ab;
    float largest = 0.5f * (aa + bb + hypotf(aa - bb, 2 * ab));
    float smallest = det / largest;
    if (!(isfinite(smallest) && smallest > 0)) { numeric = true; return 0.0f; }
    float gx = residual[a], gy = residual[b], stationarity = 0, complement = 0;
    if (norm == 0) {
      stationarity = hypotf(gx, gy) / smallest;
    } else {
      float ux = lambda[a] / norm, uy = lambda[b] / norm;
      float radial = gx * ux + gy * uy;
      if (!isfinite(radial)) { numeric = true; return 0.0f; }
      float m2 = fmaxf(0.0f, -radial);
      stationarity = hypotf(gx + m2 * ux, gy + m2 * uy) / smallest;
      complement = (m2 / smallest) * (fmaxf(0.0f, cap - norm) / norm);
    }
    if (!(isfinite(stationarity) && isfinite(complement))) { numeric = true; return 0.0f; }
    return fmaxf(fmaxf(stationarity, complement), fmaxf(0.0f, norm - cap));
  };
  float jr = 0, nr = 0, tr = 0;
  auto certify = [&]() {
    float pj = 0, pn = 0, pt = 0;
    DW_FOR(r, R)
      if (rows->kind[r] == 0) pj = fmaxf(pj, scalar_error(r));
    DW_FOR(c, C) {
      int r = rows->cfirst[c];
      pn = fmaxf(pn, scalar_error(r));
      pt = fmaxf(pt, tangent_error(r, rows->cmu[c]));
    }
    jr = dw_lane_max(pj); nr = dw_lane_max(pn); tr = dw_lane_max(pt);
    numeric = dw_any(numeric);
    return fmaxf(fmaxf(jr, nr), tr);
  };
  auto correlation = [&](int i, int j) {
    float d2 = K[i][i] * K[j][j];
    if (!(d2 > 0 && isfinite(d2))) return 0.0f;
    float s = 0.5f * (K[i][j] + K[j][i]) / sqrtf(d2);
    return isfinite(s) ? s : 0.0f;
  };
  auto snap_take = [&]() {
    DW_FOR(i, R) { w->snapl[i] = lambda[i]; w->snapr[i] = residual[i]; }
    DW_SYNC();
  };
  auto snap_restore = [&]() {
    DW_FOR(i, R) { lambda[i] = w->snapl[i]; residual[i] = w->snapr[i]; }
    DW_SYNC();
  };
  // Joint nonnegative solve of two nearly dependent normal rows (civ1's
  // coupled_pair), tangents held fixed; kept whichever candidate has the
  // smallest complementarity error, restored on any numeric trouble.
  auto coupled_pair = [&](int i, int j) {
    const float aii = K[i][i], ajj = K[j][j], aij = 0.5f * (K[i][j] + K[j][i]);
    const float qi = residual[i] - aii * lambda[i] - aij * lambda[j];
    const float qj = residual[j] - aij * lambda[i] - ajj * lambda[j];
    if (!(aii > 0 && ajj > 0 && isfinite(aij) && isfinite(qi) && isfinite(qj)))
      return;
    float bi = lambda[i], bj = lambda[j], be = INFINITY;
    auto consider = [&](float xi, float xj) {
      if (!(isfinite(xi) && isfinite(xj))) return;
      xi = fmaxf(0.0f, xi); xj = fmaxf(0.0f, xj);
      float ri = qi + aii * xi + aij * xj, rj = qj + aij * xi + ajj * xj;
      if (!(isfinite(ri) && isfinite(rj))) return;
      float e2 = fmaxf(fabsf(fminf(xi, ri)), fabsf(fminf(xj, rj)));
      if (isfinite(e2) && e2 < be) { be = e2; bi = xi; bj = xj; }
    };
    consider(0.0f, 0.0f);
    consider(-qi / aii, 0.0f);
    consider(0.0f, -qj / ajj);
    const float det = aii * ajj - aij * aij;
    if (det > 1e-6f * aii * ajj) {
      consider((aij * qj - ajj * qi) / det, (aij * qi - aii * qj) / det);
    } else {
      float ui = sqrtf(aii), uj = aij < 0 ? -sqrtf(ajj) : sqrtf(ajj);
      float uu = aii + ajj;
      if (uu > 0) { float al = -(ui * qi + uj * qj) / (uu * uu); consider(al * ui, al * uj); }
    }
    const float cap = 1e6f * (1 + fabsf(lambda[i]) + fabsf(lambda[j])
                              + fabsf(qi) / aii + fabsf(qj) / ajj);
    if (isfinite(be) && bi <= cap && bj <= cap) { update(i, bi); update(j, bj); }
  };
  int nwindows = 0;
  // Richardson extrapolation of window-to-window creep; kept only when the
  // unchanged certificate strictly improves (civ1's extrapolate).
  auto extrapolate = [&]() {
    if (nwindows < 2) return;
    snap_take();
    const float sj0 = jr, sn0 = nr, st0 = tr, cert0 = fmaxf(fmaxf(jr, nr), tr);
    bool numeric0 = numeric, kept = false;
    float n1 = dw_sum_over(R, [&](int i) {
      float d1 = lambda[i] - w->wprev[i]; return d1 * d1; });
    float n2 = dw_sum_over(R, [&](int i) {
      float d2 = w->wprev[i] - w->wprev2[i]; return d2 * d2; });
    float dot = dw_sum_over(R, [&](int i) {
      return (lambda[i] - w->wprev[i]) * (w->wprev[i] - w->wprev2[i]); });
    n1 = sqrtf(n1); n2 = sqrtf(n2);
    if (n1 > 0 && n2 > 0 && isfinite(n1) && isfinite(n2) && isfinite(dot)) {
      const float rho = fminf(n1 / n2, 0.9999f), ca = dot / (n1 * n2);
      if (ca > 0.5f && rho > 0.3f) {
        const float f = rho / (1 - rho);
        bool bad = false;
        DW_FOR(i, R) {
          const float x = lambda[i] + f * (lambda[i] - w->wprev[i]);
          if (!isfinite(x)) { bad = true; continue; }
          lambda[i] = dw_clampf(x, rows->lo[i], rows->hi[i]);
        }
        bad = dw_any(bad);
        DW_SYNC();
        if (!bad) {
          DW_LANE0 for (int c = 0; c < C; c++) {
            const int r = rows->cfirst[c];
            const float cp = rows->cmu[c] * lambda[r];
            const float n3 = hypotf(lambda[r + 1], lambda[r + 2]);
            if (n3 > cp) {
              const float sc = cp > 0 ? cp / n3 : 0.0f;
              lambda[r + 1] *= sc; lambda[r + 2] *= sc;
            }
          }
          DW_SYNC();
          residuals();
          const float e2 = certify();
          if (!numeric && isfinite(e2) && e2 < cert0) kept = true;
        }
      }
    }
    if (!kept) {
      snap_restore();
      jr = sj0; nr = sn0; tr = st0; numeric = numeric0;
    }
  };
  // Tresca fixed-point acceleration (civ1's block_accelerate): freeze the
  // friction caps at the current normals, solve the resulting convex box/disk
  // QP by accelerated projected gradient with adaptive restart, refresh the
  // caps; keep the trial only when the unchanged certificate strictly
  // improves. This is the proven repair for the exactly rank-deficient
  // flat-foot contact blocks in the fault corpus. Mirrors civ1: the inner
  // budget is granted FRESH per call (a shared pool let the first call eat
  // everything, leaving the remaining arsenal as no-ops on degenerate
  // duck-scale grid islands that ~10 full-budget calls solve strictly).
  auto block_accelerate = [&](uint32_t apgd_budget = DW_APGD_BUDGET,
                              int damping = 0) {
    if (R < 1 || !apgd_budget) return;
    snap_take();
    const float sj0 = jr, sn0 = nr, st0 = tr, cert0 = fmaxf(fmaxf(jr, nr), tr);
    bool numeric0 = numeric, kept = false;
    float pl = 0;
    DW_FOR(i, R) {
      float s = 0;
      for (int j = 0; j < R; j++) s += fabsf(0.5f * (K[i][j] + K[j][i]));
      pl = fmaxf(pl, s);
    }
    float Lc = dw_lane_max(pl);
    if (isfinite(Lc) && Lc > 0) {
      const float step = 1 / Lc;
      float* x = w->ax; float* y = w->ay; float* xn = w->axn;
      float* grad = w->agrad; float* bestx = w->abestx; float* cap = w->acap;
      DW_FOR(i, R) { x[i] = lambda[i]; bestx[i] = lambda[i]; }
      DW_SYNC();
      float beste = cert0;
      bool bad = false;
      for (int outer = 0; outer < 64 && apgd_budget && !bad; outer++) {
        // Cap-refresh damping schedule mirrors civ1: 0 = classical undamped
        // (always used by the first call, preserving old trajectories
        // bit-exactly), 1 = half-averaged, 2 = Cesaro; later calls rotate to
        // break outer-map limit cycles on degenerate multi-support blocks.
        DW_LANE0 for (int c = 0; c < C; c++) {
          const float cw = rows->cmu[c] * fmaxf(0.0f, x[rows->cfirst[c]]);
          const float th = !outer ? 1.0f
                                  : damping == 1 ? 0.5f
                                  : damping == 2 ? 1.0f / (outer + 1) : 1.0f;
          cap[c] = (1 - th) * cap[c] + th * cw;
        }
        float t = 1;
        DW_FOR(i, R) y[i] = x[i];
        DW_SYNC();
        for (int k = 0; k < 4096 && apgd_budget; k++) {
          apgd_budget--;
          DW_FOR(i, R) {
            float s = base[i];
            for (int j = 0; j < R; j++) s += K[i][j] * y[j];
            grad[i] = s;
          }
          DW_SYNC();
          bool badl = false;
          DW_FOR(i, R) {
            xn[i] = dw_clampf(y[i] - step * grad[i], rows->lo[i], rows->hi[i]);
            if (!isfinite(xn[i])) badl = true;
          }
          if (dw_any(badl)) { bad = true; break; }
          DW_SYNC();
          DW_LANE0 for (int c = 0; c < C; c++) {
            const int r = rows->cfirst[c];
            const float n2 = hypotf(xn[r + 1], xn[r + 2]);
            if (n2 > cap[c]) {
              const float f = cap[c] > 0 ? cap[c] / n2 : 0.0f;
              xn[r + 1] *= f; xn[r + 2] *= f;
            }
          }
          DW_SYNC();
          float dot = dw_sum_over(R, [&](int i) {
            return (y[i] - xn[i]) * (xn[i] - x[i]); });
          float dn = dw_max_over(R, [&](int i) { return fabsf(xn[i] - x[i]); });
          float xs = dw_max_over(R, [&](int i) { return fabsf(xn[i]); });
          if (dot > 0) t = 1;                              // adaptive restart
          const float tn = 0.5f * (1 + sqrtf(1 + 4 * t * t)), mo = (t - 1) / tn;
          DW_FOR(i, R) { y[i] = xn[i] + mo * (xn[i] - x[i]); x[i] = xn[i]; }
          DW_SYNC();
          t = tn;
          if (!(dn > 1.2e-7f * (1 + xs))) break;           // fixed point (f32)
        }
        if (bad) break;
        DW_FOR(i, R) lambda[i] = x[i];
        DW_SYNC();
        residuals();
        const float e2 = certify();
        if (!numeric && isfinite(e2) && e2 < beste) {
          beste = e2;
          DW_FOR(i, R) bestx[i] = x[i];
        }
        snap_restore();
        numeric = numeric0;
        if (beste <= tolerance) break;
      }
      if (!bad && beste < cert0) {
        DW_FOR(i, R) lambda[i] = bestx[i];
        DW_SYNC();
        residuals();
        certify();
        if (!numeric) kept = true;
      }
    }
    if (!kept) {
      snap_restore();
      jr = sj0; nr = sn0; tr = st0; numeric = numeric0;
    }
  };
  uint32_t iterations = 0;
  bool converged = false, stalled = false;
  int pair_i = R, pair_j = R;
  uint32_t accelerations = 16;
  float reference = INFINITY;
  residuals();
  numeric = dw_any(numeric);
  if (numeric) return DWC1_NUMERIC;
  for (uint32_t it = 0; it < max_iterations && !converged; it++) {
    // civ1 attempts its null-direction move once at sweep 256; not ported
    // (see the header comment), the Tresca acceleration below covers it.
    int c = 0;
    bool invalid = false;
    for (int r = 0; r < R; r++) {
      if (rows->kind[r] == 2 || rows->kind[r] == 3) continue;
      update(r, scalar(r));
      if (rows->kind[r] == 1) {
        if (!(c < C && rows->cfirst[c] == r)) { invalid = true; break; }
        int a = r + 1, b = r + 2;
        float aa = K[a][a], ab = 0.5f * (K[a][b] + K[b][a]), bb = K[b][b];
        float fx = residual[a] - aa * lambda[a] - ab * lambda[b];
        float fy = residual[b] - ab * lambda[a] - bb * lambda[b];
        DwDisk d2;
        if (!dw_disk(aa, ab, bb, fx, fy, rows->cmu[c] * lambda[r], &d2))
          return DWC1_NUMERIC;  // warp-uniform: redundant identical inputs
        c++;
        update(r + 1, d2.x);
        update(r + 2, d2.y);
      }
    }
    if (invalid) return DWC1_INVALID;
    if (stalled && pair_i < R && pair_j < R) {
      snap_take();
      bool numeric0 = numeric;
      coupled_pair(pair_i, pair_j);
      numeric = dw_any(numeric);
      if (numeric) { snap_restore(); numeric = numeric0; }
    }
    residuals();          // recompute each sweep; hides no roundoff drift
    float err = certify();
    if (numeric) return DWC1_NUMERIC;
    iterations = it + 1;
    converged = err <= tolerance;
    if (!converged && (it & 31u) == 31u) {
      if (pair_i < R && err > 0.5f * reference) { pair_i = pair_j = R; }
      if (it >= 63 && !stalled && err > 0.25f * reference) {
        stalled = true;
        float best = 0.99f;
        for (int a2 = 0; a2 < C; a2++)
          for (int b2 = a2 + 1; b2 < C; b2++) {
            int i2 = rows->cfirst[a2], j2 = rows->cfirst[b2];
            float corr = correlation(i2, j2);
            if (corr >= best) { best = corr; pair_i = i2; pair_j = j2; }
          }
      }
      if (stalled) {
        extrapolate();
        converged = fmaxf(fmaxf(jr, nr), tr) <= tolerance;
#ifdef DW_STALL_TIER_EAGER
        // EAGER tier (model-header opt-in, e.g. the H0 humanoid training
        // lane; never defined for the duck, keeping it bit-identical):
        // accept a STALLED solve at the tier ceiling immediately instead
        // of spending the Tresca arsenal chasing the strict tolerance
        // first. The humanoid's exactly-coplanar box soles stall nearly
        // every contact tick; measured on a flail workload, chasing the
        // last decades costs 20-150 APGD bursts per policy step and is
        // then accepted at this SAME ceiling anyway once the arsenal is
        // spent. Certificate FORM unchanged; joint rows stay strict.
        if (!converged)
          converged = jr <= tolerance && fmaxf(nr, tr) <= DW_TIER_CEILING;
#endif
        if (!converged && accelerations && err > 0.25f * reference) {
          accelerations--;
          block_accelerate(DW_APGD_BUDGET, (int)((15u - accelerations) % 3u));
          converged = fmaxf(fmaxf(jr, nr), tr) <= tolerance;
        }
        // Exhaustion tier (mirrors civ1): the full repair arsenal is spent
        // on a stalled degenerate block and this window made no meaningful
        // progress -> accept the degenerate CONTACT block at the ceiling
        // instead of faulting (joint rows stay at the strict tolerance:
        // regularized, diagonal, always convergent). Same error measures;
        // duck-scale problems never reach this branch.
        // civ1's best-iterate memory and last-window certificate polish are
        // deliberately NOT mirrored: they close the band between 1e-5 and
        // the residual limit cycle (~1e-4) in f64, which lies entirely BELOW
        // this kernel's fp32 tier ceiling (1e-3); the tier alone already
        // contains that family here, and a sub-1e-5 walk is beneath fp32
        // certificate resolution at these impulse scales.
        if (!converged && !accelerations && err > 0.99f * reference)
          converged = jr <= tolerance && fmaxf(nr, tr) <= DW_TIER_CEILING;
        if (numeric) return DWC1_NUMERIC;
      }
      DW_FOR(i, R) { w->wprev2[i] = w->wprev[i]; w->wprev[i] = lambda[i]; }
      DW_SYNC();
      nwindows++;
      reference = err;
    }
  }
  if (!converged) {
    DW_LANE0 {
      diag->iterations = iterations;
      diag->joint_residual = jr; diag->normal_residual = nr;
      diag->tangent_residual = tr;
    }
    DW_SYNC();
    return DWC1_NO_CONVERGENCE;
  }
  bool badv = false;
  DW_FOR(k, DW_N) {
    float x = w->smooth[k];
    for (int r = 0; r < R; r++) x += response[r][k] * lambda[r];
    if (!isfinite(x)) badv = true;
    w->vnew[k] = x;
  }
  DW_SYNC();
  if (dw_any(badv)) return DWC1_NUMERIC;
  float pm = 0;
  bool badm = false;
  DW_FOR(k, DW_N) {
    float x = 0;
    for (int j = 0; j < DW_N; j++)
      x += w->e.M[k][j] * (w->vnew[j] - w->smooth[j]);
    for (int r = 0; r < R; r++) x -= rows->G[r][k] * lambda[r];
    if (!isfinite(x)) badm = true;
    pm = fmaxf(pm, fabsf(x));
  }
  float mr = dw_lane_max(pm);
  if (dw_any(badm)) return DWC1_NUMERIC;
  if (!(mr <= DW_MOMENTUM_TOLERANCE)) return DWC1_NUMERIC;
  DW_LANE0 {
    diag->iterations = iterations;
    diag->joint_residual = jr; diag->normal_residual = nr;
    diag->tangent_residual = tr; diag->momentum_residual = mr;
  }
  DW_SYNC();
  return DWC1_OK;
}

// ------------------------------------------------------------------- tick
// One 0.002 s step of one environment, executed cooperatively by every lane
// of the env's warp (plain sequential code in single-lane builds). On any
// failure the state is left exactly as it was (per-env rollback); the
// warp-uniform return value reports it. diag scalars are lane-0-written.
static DW_HD int dw_tick(DwState* s, const float* target,
                         const DwParams* params, DwWork* w,
                         dwc1_diagnostic* diag) {
  const float dt = DW_DT;
  DW_LANE0 {
    diag->contact_points = 0;
    diag->active_limits = 0;
    diag->maximum_normal_impulse = 0;
    diag->maximum_penetration = 0;
  }
  if (!dw_evaluate(s->q, s->v, DW_GRAVITY_Z, s->mass_scale, w))
    return DWC1_DYNAMICS;
  DW_LANE0 w->iflag = dw_chol(w->e.M, w->L) ? 0 : 1;
  DW_SYNC();
  if (w->iflag) return DWC1_DYNAMICS;
  // PD actuator (clip(target) per limits, effort cap) + passive damping; the
  // smooth velocity is v + dt * M^-1 (actuator + passive - bias).
  bool bad = false;
  DW_FOR(n, DW_N) w->smooth[n] = -w->e.bias[n];
  DW_SYNC();
  // domain randomization: scaled gains, identity at neutral (1.0) scales
  const float kpf = (float)((double)DW_KP * s->kp_scale);
  const float dampf = (float)((double)DW_DAMPING * s->damping_scale);
  DW_FOR(j, DW_J) {
    float tj = dw_clampf(target[j], DW_LIMIT_LOWER[j], DW_LIMIT_UPPER[j]);
    float motor = kpf * (tj - s->q[7 + j]) + DW_KV * (0.0f - s->v[6 + j]);
    if (!isfinite(motor)) bad = true;
    // per-joint effort caps (H0 tiers 180/140/70); the duck table is a
    // uniform broadcast of DW_EFFORT_CAP, keeping duck builds bit-identical
    w->smooth[6 + j] += dw_clampf(motor, -DW_EFFORT_CAP_TABLE[j],
                                  DW_EFFORT_CAP_TABLE[j])
                      - dampf * s->v[6 + j];
  }
  DW_SYNC();
  if (dw_any(bad)) return DWC1_DYNAMICS;
  DW_LANE0 dw_chol_solve(w->L, w->smooth);   // triangular: inherently serial
  DW_SYNC();
  DW_FOR(n, DW_N) {
    w->smooth[n] = s->v[n] + dt * w->smooth[n];
    if (!isfinite(w->smooth[n])) bad = true;
  }
  DW_SYNC();
  if (dw_any(bad)) return DWC1_DYNAMICS;
  // joint rows (av2_prepare): friction always active (loss > 0); soft limit
  // rows activate when the gap dips under the (zero) margin. The compacted
  // row-index assignment is sequential bookkeeping: lane 0 (tiny).
  DW_FOR(i, DW_JROWS * DW_N) (&w->rows.G[0][0])[i] = 0;
  DW_SYNC();
  DW_LANE0 {
    w->rows.R = 0;
    w->rows.C = 0;
    w->iflag = 0;
    for (int j = 0; j < DW_J; j++) {
      for (int side = 0; side < 3; side++) {
        float sign = side == 2 ? -1.0f : 1.0f;
        float gap = side == 0 ? 0.0f
            : (side == 1 ? s->q[7 + j] - DW_LIMIT_LOWER[j]
                         : DW_LIMIT_UPPER[j] - s->q[7 + j]);
        bool active = side == 0 ? true : gap < DW_LIMIT_MARGIN;
        if (!active) continue;
        float d = side == 0 ? dw_bounded_impedance(DW_FRICTION_D0)
                            : dw_impedance(gap);
        float width = dw_bounded_impedance(side == 0 ? DW_FRICTION_DWIDTH
                                                     : DW_LIMIT_SOLIMP[1]);
        float tc = fmaxf(side == 0 ? DW_FRICTION_TIMECONST : DW_LIMIT_TIMECONST,
                         2 * dt);
        float B = 2 / fmaxf(1e-15f, width * tc);
        float Ks = side == 0 ? 0.0f
            : 1 / fmaxf(1e-15f, width * width * tc * tc
                                * DW_LIMIT_DAMPRATIO * DW_LIMIT_DAMPRATIO);
        float rowv = sign * s->v[6 + j];
        float aref = -B * rowv - Ks * d * (gap - DW_LIMIT_MARGIN);
        int r = w->rows.R++;
        w->rows.G[r][6 + j] = sign;
        w->rows.reg[r] = fmaxf(1e-15f, (1 - d) / d * params->refweight[j]);
        w->rows.target[r] = rowv + dt * aref;
        w->rows.lo[r] = side == 0 ? -dt * DW_FRICTION_LOSS : 0.0f;
        w->rows.hi[r] = side == 0 ? dt * DW_FRICTION_LOSS : INFINITY;
        w->rows.warm[r] = dw_clampf(dt * s->warm[side * DW_J + j],
                                    w->rows.lo[r], w->rows.hi[r]);
        w->rows.kind[r] = 0;
        w->rows.slot[r] = side * DW_J + j;
        if (side) diag->active_limits++;
        if (!(isfinite(w->rows.reg[r]) && w->rows.reg[r] > 0 && isfinite(aref)
              && isfinite(w->rows.target[r]) && isfinite(w->rows.warm[r])))
          w->iflag = 1;
      }
    }
    // contact manifolds at current poses + idv1's warm-start matching by
    // feature id (normal agreement > .98, point displacement < .02 m); the
    // sequential reduce/warm-match stays on lane 0, the 20-column contact
    // jacobians are filled lane-parallel below.
    for (int pair = 0; pair < DW_PAIRS; pair++) {
      int foot = (int)DW_PAIR_BODY_A[pair];
      dw_plane_manifold(pair, w->e.pos[foot], w->e.rot[foot],
                        &w->manifolds[pair]);
      dwc1_manifold* m = &w->manifolds[pair];
      const dwc1_manifold* previous = &s->cache[pair];
      bool normal_ok = dw_dot3(m->normal, previous->normal) > 0.98f;
      for (uint32_t k = 0; k < m->count; k++) {
        dwc1_point* x = &m->points[k];
        float warm[3] = {0, 0, 0};
        if (normal_ok) {
          for (uint32_t q2 = 0; q2 < previous->count; q2++) {
            const dwc1_point* y = &previous->points[q2];
            float dvec[3];
            dw_sub3(x->point, y->point, dvec);
            if (x->feature == y->feature && dw_dot3(dvec, dvec) < 0.0004f) {
              warm[0] = y->normal_impulse;
              float t[3];
              for (int a = 0; a < 3; a++)
                t[a] = previous->tangent1[a] * y->tangent_impulse[0]
                     + previous->tangent2[a] * y->tangent_impulse[1];
              warm[1] = dw_dot3(t, m->tangent1);
              warm[2] = dw_dot3(t, m->tangent2);
              break;
            }
          }
        }
        w->rows.cfirst[w->rows.C] = w->rows.R;
        w->rows.cmu[w->rows.C] =
            (float)((double)DW_PAIR_MU[pair] * s->friction_scale);
        w->rows.C++;
        for (int a = 0; a < 3; a++) {
          int r = w->rows.R++;
          w->rows.target[r] = a == 0
              ? fminf(1.0f, 0.2f * fmaxf(0.0f, x->depth - DW_CONTACT_EPS) / dt)
              : 0.0f;
          w->rows.reg[r] = 0;
          w->rows.lo[r] = a == 0 ? 0.0f : -INFINITY;
          w->rows.hi[r] = INFINITY;
          w->rows.warm[r] = warm[a];
          w->rows.kind[r] = a + 1;
          w->rows.slot[r] = -1;
        }
        diag->contact_points++;
        diag->maximum_penetration =
            fmaxf(diag->maximum_penetration, x->depth);
      }
    }
  }
  DW_SYNC();
  if (w->iflag) return DWC1_DYNAMICS;
  {  // contact-row jacobians: one lane per (row, column) work item, with the
     // exact per-column arithmetic of the original dw_contact_row.
    int Rall = w->rows.R, Call = w->rows.C;
    int first = Call > 0 ? w->rows.cfirst[0] : Rall;
    int m0 = (int)w->manifolds[0].count;
    DW_FOR(idx, (Rall - first) * DW_N) {
      int cr = idx / DW_N, n = idx % DW_N;
      int r = first + cr;
      int point = cr / 3, dir = cr % 3;
      int pair = point < m0 ? 0 : 1;
      int k = pair ? point - m0 : point;
      const dwc1_manifold* m = &w->manifolds[pair];
      const float* direction = dir == 0 ? m->normal
                             : (dir == 1 ? m->tangent1 : m->tangent2);
      const float* pt = m->points[k].point;
      const int bodies[2] = {(int)DW_PAIR_BODY_A[pair],
                             (int)DW_PAIR_BODY_B[pair]};
      const float sign[2] = {-1.0f, 1.0f};
      float out = 0;
      for (int ii = 0; ii < 2; ii++) {
        int b = bodies[ii];
        float arm[3], torque[3];
        dw_sub3(pt, w->e.pos[b], arm);
        dw_cross3(arm, direction, torque);
        float x = 0;
        for (int kk = 0; kk < 3; kk++)
          x += direction[kk] * w->e.J[b][kk][n]
             + torque[kk] * w->e.J[b][3 + kk][n];
        out += sign[ii] * x;
      }
      w->rows.G[r][n] = out;
    }
  }
  DW_SYNC();
  // dense coupled solve
  int rc = dw_solve(w, params->tolerance, params->max_iterations, diag);
  if (rc != DWC1_OK) return rc;
  // integrate (av1_integrate_root + hinge Euler) on lane 0, commit in parallel
  DW_LANE0 {
    w->iflag = 0;
    for (int k = 0; k < 3; k++) w->qnew[k] = s->q[k] + dt * w->vnew[k];
    {
      float wdt[3] = {dt * w->vnew[3], dt * w->vnew[4], dt * w->vnew[5]}, eq[4];
      dw_qexp(wdt, eq);
      dw_qmul(eq, s->q + 3, w->qnew + 3);
      dw_qnormalize(w->qnew + 3);
    }
    for (int j = 0; j < DW_J; j++)
      w->qnew[7 + j] = s->q[7 + j] + dt * w->vnew[6 + j];
    if (!dw_finite(w->qnew, DW_Q) || !dw_finite(w->vnew, DW_N)) w->iflag = 1;
    if (!w->iflag) {
      for (int k = 0; k < DW_JROWS; k++) w->warmnew[k] = 0;
      for (int r = 0; r < w->rows.R; r++) {
        if (w->rows.slot[r] < 0) continue;
        float wf = w->lambda[r] / dt;
        if (w->rows.slot[r] < DW_J)
          wf = dw_clampf(wf, -DW_FRICTION_LOSS, DW_FRICTION_LOSS);
        w->warmnew[w->rows.slot[r]] = wf;
      }
      int c = 0;
      for (int pair = 0; pair < DW_PAIRS; pair++) {
        dwc1_manifold* m = &w->manifolds[pair];
        for (uint32_t k = 0; k < m->count; k++) {
          int fr = w->rows.cfirst[c++];
          m->points[k].normal_impulse = w->lambda[fr];
          m->points[k].tangent_impulse[0] = w->lambda[fr + 1];
          m->points[k].tangent_impulse[1] = w->lambda[fr + 2];
          diag->maximum_normal_impulse =
              fmaxf(diag->maximum_normal_impulse, w->lambda[fr]);
        }
      }
    }
  }
  DW_SYNC();
  if (w->iflag) return DWC1_DYNAMICS;
  DW_FOR(k, DW_Q) s->q[k] = w->qnew[k];
  DW_FOR(k, DW_N) s->v[k] = w->vnew[k];
  DW_FOR(k, DW_JROWS) s->warm[k] = w->warmnew[k];
  DW_LANE0 {
    for (int pair = 0; pair < DW_PAIRS; pair++) {
      s->cache[pair] = w->manifolds[pair];
      if (w->manifolds[pair].count > 0) s->contact_ticks[pair]++;
    }
    s->count++;
  }
  DW_SYNC();
  return DWC1_OK;
}

// The policy termination predicate (height / tilt on the model's up-axis /
// nonfinite), shared by the policy-step boundary check and the optional
// mid-block fast-termination latch. Every lane computes it from the shared
// env state, so branching on it stays warp-uniform.
static DW_HD bool dw_policy_fell(const DwState* s) {
  bool finite = dw_finite(s->q, DW_Q) && dw_finite(s->v, DW_N);
  // world-Z component of the model's body up axis: duck body +Z ->
  // R[2][2] = 1-2(qx^2+qy^2); H0 humanoid body +Y -> R[2][1] =
  // 2(qy*qz + qx*qw) (humanoid_native_lane.tilt). q[3..6] is root xyzw.
#if DW_ENV_UP_AXIS == 1
  double up = 2.0 * ((double)s->q[4] * (double)s->q[5]
                     + (double)s->q[3] * (double)s->q[6]);
#else
  double up = 1.0 - 2.0 * ((double)s->q[3] * (double)s->q[3]
                           + (double)s->q[4] * (double)s->q[4]);
#endif
  return ((double)s->q[2] < DW_ENV_MIN_HEIGHT_FRACTION
                            * (double)DW_INITIAL_QPOS[2])
      || (up < DW_ENV_COS_MAX_TILT) || !finite;
}

// n_ticks with the targets held, all inside ONE call (one kernel launch on
// device): a failing environment freezes (state and cache untouched from its
// last accepted tick) and keeps its failure diag. The per-foot contact tick
// counters cover exactly this call's accepted ticks. Returns the final
// warp-uniform status (also written to diag->status by lane 0).
// latch_termination (policy path only, dwc1_set_fast_termination): stop
// ticking as soon as the termination predicate holds after an accepted
// tick -- the boundary check then latches done from the same predicate, so
// the env terminates identically, just without solving its post-fall ticks.
static DW_HD int dw_step_env(DwState* s, const float* target, uint32_t n_ticks,
                             const DwParams* params, DwWork* w,
                             dwc1_diagnostic* diag,
                             bool latch_termination = false) {
  DW_LANE0 {
    diag->ticks = 0;
    for (int pair = 0; pair < DW_PAIRS; pair++) s->contact_ticks[pair] = 0;
  }
  DW_SYNC();
  for (uint32_t t = 0; t < n_ticks; t++) {
    int st = dw_tick(s, target, params, w, diag);
    DW_LANE0 diag->status = (uint32_t)st;
    if (st != DWC1_OK) return st;
    DW_LANE0 diag->ticks++;
    if (latch_termination && dw_policy_fell(s)) break;
  }
  DW_SYNC();
  return DWC1_OK;
}

// ------------------------------------------------------------- init / read
static DW_HD void dw_init_state(DwState* s, const float* joint_offsets) {
  memset(s, 0, sizeof(*s));
  for (int k = 0; k < DW_Q; k++) s->q[k] = DW_INITIAL_QPOS[k];
  for (int k = 0; k < DW_N; k++) s->v[k] = DW_INITIAL_VEL[k];
  if (joint_offsets)
    for (int j = 0; j < DW_J; j++)
      s->q[7 + j] = dw_clampf(s->q[7 + j] + joint_offsets[j],
                              DW_LIMIT_LOWER[j], DW_LIMIT_UPPER[j]);
  for (int j = 0; j < DW_J; j++) s->targets[j] = DW_HOME_TARGETS_F64[j];
  s->last_foot = -1;
  s->mass_scale = s->friction_scale = s->kp_scale = s->damping_scale = 1.0;
  for (int k = 0; k <= DWC1_MAX_LATENCY; k++)   // ring = reset targets
    for (int j = 0; j < DW_J; j++)
      s->eff_ring[k][j] = fmin((double)DW_LIMIT_UPPER[j],
                               fmax((double)DW_LIMIT_LOWER[j],
                                    DW_HOME_TARGETS_F64[j]));
}

// Apply per-env randomization values (host-drawn) and reset the actuation
// latency ring to the reset targets; call right after resetting the env.
static DW_HD void dw_policy_set_random(DwState* s, const dwc1_env_random* r) {
  s->mass_scale = r->mass_scale;
  s->friction_scale = r->friction_scale;
  s->kp_scale = r->kp_scale;
  s->damping_scale = r->damping_scale;
  s->latency_steps = r->latency_steps;
  for (int k = 0; k <= DWC1_MAX_LATENCY; k++)
    for (int j = 0; j < DW_J; j++)
      s->eff_ring[k][j] = fmin((double)DW_LIMIT_UPPER[j],
                               fmax((double)DW_LIMIT_LOWER[j],
                                    DW_HOME_TARGETS_F64[j]));
}

// Light FK for reads: body poses and velocities only (no mass/bias/jacobian).
static DW_HD void dw_body_states(const float* q, const float* v,
                                 float out[DW_B][13]) {
  float pos[DW_B][3], rot[DW_B][4], w[DW_B][3], vel[DW_B][3];
  for (int k = 0; k < 3; k++) { pos[0][k] = 0; w[0][k] = 0; vel[0][k] = 0; }
  rot[0][0] = rot[0][1] = rot[0][2] = 0; rot[0][3] = 1;
  float offset[3];
  dw_rotate(q + 3, DW_ROOT_COM, offset);
  dw_add3(q, offset, pos[1]);
  dw_qmul(q + 3, DW_ROOT_QPC, rot[1]);
  dw_qnormalize(rot[1]);
  w[1][0] = v[3]; w[1][1] = v[4]; w[1][2] = v[5];
  float wxo[3];
  dw_cross3(w[1], offset, wxo);
  dw_add3(v, wxo, vel[1]);
  for (int j = 0; j < DW_J; j++) {
    int b = j + 2, p = (int)DW_HINGE_PARENT[j];
    float rp[3], s[3], ax[3], eq[4], t[4], rc[3], tmp[3];
    dw_rotate(rot[p], DW_HINGE_AP[j], rp);
    dw_rotate(rot[p], DW_HINGE_AXIS[j], s);
    dw_scale3(DW_HINGE_AXIS[j], q[7 + j], ax);
    dw_qexp(ax, eq);
    dw_qmul(rot[p], eq, t);
    dw_qmul(t, DW_HINGE_REF[j], rot[b]);
    dw_qnormalize(rot[b]);
    dw_rotate(rot[b], DW_HINGE_AC[j], rc);
    dw_add3(pos[p], rp, pos[b]); dw_sub3(pos[b], rc, pos[b]);
    dw_scale3(s, v[6 + j], tmp); dw_add3(w[p], tmp, w[b]);
    dw_cross3(w[p], rp, tmp); dw_add3(vel[p], tmp, vel[b]);
    dw_cross3(w[b], rc, tmp); dw_sub3(vel[b], tmp, vel[b]);
  }
  for (int b = 0; b < DW_B; b++) {
    for (int k = 0; k < 3; k++) out[b][k] = pos[b][k];
    for (int k = 0; k < 4; k++) out[b][3 + k] = rot[b][k];
    for (int k = 0; k < 3; k++) out[b][7 + k] = vel[b][k];
    for (int k = 0; k < 3; k++) out[b][10 + k] = w[b][k];
  }
}

// Whole-sole heights: min world z over the baked sole vertices per foot.
static DW_HD void dw_sole_heights(const float body[DW_B][13], float out[2]) {
  for (int pair = 0; pair < DW_PAIRS; pair++) {
    int b = (int)DW_PAIR_BODY_A[pair];
    float lowest = INFINITY;
    for (int i = 0; i < DW_FOOT_VERTS; i++) {
      float world[3];
      dw_rotate(&body[b][3], DW_FOOT_VERTICES[pair][i], world);
      lowest = fminf(lowest, body[b][2] + world[2]);
    }
    out[pair] = lowest;
  }
}

static DW_HD void dw_fill_info(uint32_t environments, dwc1_info* info) {
  info->environments = environments;
  info->bodies = DW_B; info->joints = DW_J; info->dofs = DW_N;
  info->dt = DW_DT; info->kp = DW_KP; info->kv = DW_KV;
  info->effort_cap = DW_EFFORT_CAP;
  info->home_root_height = DW_INITIAL_QPOS[2];
  for (int k = 0; k < DW_Q; k++) info->home_qpos[k] = DW_INITIAL_QPOS[k];
  for (int j = 0; j < DW_J; j++) {
    info->joint_lower[j] = DW_LIMIT_LOWER[j];
    info->joint_upper[j] = DW_LIMIT_UPPER[j];
    // DW_HOME_TARGETS equals the home joint pose by construction; keep the
    // constant referenced so model drift in either is caught at compile time.
    (void)DW_HOME_TARGETS[j];
  }
}

// ==================== device policy layer ==================================
// Verbatim f64 port of the flat-floor env contract (action->target slew
// chain, 3*J+16 observation, termination) and its reward (EMA velocity
// tracking, alive, lateral/yaw, action-rate, torque, phase-gated qualified
// steps with placement/opposite-support/stance gates, chatter, flicker via
// the device contact-tick counters, clearance, phase-match, double-support,
// alternation/same-foot, optional self-imitation). EVERY env constant comes
// from the generated model header's DW_ENV_* contract block, so the same
// kernel source serves the duck (walk/env/flat.py + reward.py, OBS 58 /
// ACT 14) and the H0 humanoid (walk/env/humanoid_flat.py +
// humanoid_reward.py, OBS 52 / ACT 12) purely by header selection; the
// python env pair for the selected header is the contract and the parity
// tests run them side by side.
#define DWP_OBS DW_ENV_OBS
#define DWP_PI 3.141592653589793
#define DWP_CONTROL_DT DW_ENV_CONTROL_DT
#define DWP_ACTION_SCALE DW_ENV_ACTION_SCALE
#define DWP_MAX_TARGET_INCREMENT DW_ENV_MAX_TARGET_INCREMENT
#define DWP_HORIZON DW_ENV_HORIZON_STEPS
#define DWP_COS_MAX_TILT DW_ENV_COS_MAX_TILT     // tilt > max  <=>  up < cos
// reward.py weights/constants: GENERATED into duck_model.h from reward.py
// itself (DW_RW_*), so any python-side weight change fails the header-drift
// test until the header is regenerated. The kernel keeps its DWP_ aliases.
#define DWP_W_TRACK DW_RW_W_TRACK
#define DWP_TRACK_SIGMA_SQ DW_RW_TRACK_SIGMA_SQ
#define DWP_TRACK_EMA_COEF (DWP_CONTROL_DT / DW_RW_TRACK_EMA_S)
#define DWP_W_ALIVE DW_RW_W_ALIVE
#define DWP_W_LATERAL DW_RW_W_LATERAL
#define DWP_W_ACTION_RATE DW_RW_W_ACTION_RATE
#define DWP_W_TORQUE DW_RW_W_TORQUE
#define DWP_W_AIR_TIME DW_RW_W_AIR_TIME
#define DWP_AIR_TIME_MIN DW_RW_AIR_TIME_MIN
#define DWP_AIR_TIME_MAX DW_RW_AIR_TIME_MAX
#define DWP_PLACEMENT_MIN_M DW_RW_PLACEMENT_MIN_M
#define DWP_OPP_SUPPORT_FRAC DW_RW_OPP_SUPPORT_FRAC
#define DWP_W_CHATTER DW_RW_W_CHATTER
#define DWP_CHATTER_MAX_S DW_RW_CHATTER_MAX_S
#define DWP_W_FLICKER DW_RW_W_FLICKER
#define DWP_TICKS_FULL DW_RW_TICKS_FULL
#define DWP_STANCE_MIN_S DW_RW_STANCE_MIN_S
#define DWP_W_CLEARANCE DW_RW_W_CLEARANCE
#define DWP_CLEARANCE_M DW_RW_CLEARANCE_M
#define DWP_W_DOUBLE_SUPPORT DW_RW_W_DOUBLE_SUPPORT
#define DWP_DOUBLE_SUPPORT_GRACE DW_RW_DOUBLE_SUPPORT_GRACE
#define DWP_W_ALTERNATE DW_RW_W_ALTERNATE
#define DWP_W_SAME_FOOT DW_RW_W_SAME_FOOT
#define DWP_W_PHASE DW_RW_W_PHASE

// flat.py AFFINE gait clock, exact numpy association: the per-episode random
// offset (resampled at reset, host-side RNG) plus
// ((2.0*pi) * (PHASE_HZ_BASE + PHASE_HZ_PER_MPS*command)) * t * CONTROL_DT,
// all IEEE f64. Both clock constants are GENERATED into duck_model.h from
// walk.env.flat itself (which reads the DUCK_PHASE_HZ_* sweep env vars at
// import), so the header-drift test pins env <-> kernel agreement.
static DW_HD double dw_policy_phase(double phase0, uint32_t t, double command) {
  return phase0
       + ((2.0 * DWP_PI) * (DW_PHASE_HZ_BASE + DW_PHASE_HZ_PER_MPS * command))
         * (double)t * DW_ENV_CONTROL_DT;
}

// native_lane.quat_to_rot in f64 (body->world rotation from xyzw quat).
static DW_HD void dw_quat_rot_f64(const float* q, double R[3][3]) {
  double x = q[0], y = q[1], z = q[2], w = q[3];
  R[0][0] = 1.0 - 2.0 * (y * y + z * z);
  R[0][1] = 2.0 * (x * y - z * w);
  R[0][2] = 2.0 * (x * z + y * w);
  R[1][0] = 2.0 * (x * y + z * w);
  R[1][1] = 1.0 - 2.0 * (x * x + z * z);
  R[1][2] = 2.0 * (y * z - x * w);
  R[2][0] = 2.0 * (x * z - y * w);
  R[2][1] = 2.0 * (y * z + x * w);
  R[2][2] = 1.0 - 2.0 * (x * x + y * y);
}

// env _observe(): the DW_ENV_OBS (= 3*J + 16) observation at the CURRENT
// state (uses the post-increment step counter, exactly like the python env).
// Tail block base T = 3*DW_J: duck 42..57, humanoid 36..51 -- identical
// relative layout on both models.
static DW_HD void dw_policy_observe(const DwState* s, float* obs) {
  const int T = 3 * DW_J;
  double R[3][3];
  dw_quat_rot_f64(s->q + 3, R);
  for (int j = 0; j < DW_J; j++) {
    obs[j] = (float)((double)s->q[7 + j] - DW_HOME_TARGETS_F64[j]);
    obs[DW_J + j] = (float)(DW_ENV_QDOT_OBS_SCALE * (double)s->v[6 + j]);
    obs[2 * DW_J + j] = s->prev_action[j];
  }
  for (int i = 0; i < 3; i++) {
    obs[T + i] = (float)(-R[2][i]);
    double wv = 0, lv = 0;                      // einsum("eji,ej->ei"): R^T v
    for (int j = 0; j < 3; j++) {
      wv += R[j][i] * (double)s->v[3 + j];
      lv += R[j][i] * (double)s->v[j];
    }
    obs[T + 3 + i] = (float)wv;
    obs[T + 6 + i] = (float)lv;
  }
  obs[T + 9] = (float)s->command;
  obs[T + 10] = 0.0f;
  obs[T + 11] = 0.0f;
  obs[T + 12] = s->cache[0].count > 0 ? 1.0f : 0.0f;
  obs[T + 13] = s->cache[1].count > 0 ? 1.0f : 0.0f;
  double phase = dw_policy_phase(s->phase0, s->t, s->command);
  obs[T + 14] = (float)sin(phase);
  obs[T + 15] = (float)cos(phase);
}

// reward.py reward(): updates the tracker fields in place (for every env,
// live or done, exactly like the python env) and returns the f32 reward.
// `a` is the clipped f64 action, `eff` the f64 effective targets (the torque
// estimate uses the pre-cast f64 values like flat.py's _torque).
static DW_HD float dw_policy_reward(DwState* s, const double* a,
                                    const double* eff, const float sole[2],
                                    const float foot_x[2]) {
  const double dt = DWP_CONTROL_DT;
  bool contact[2] = {s->cache[0].count > 0, s->cache[1].count > 0};
  bool prevc[2] = {s->prev_contact[0] != 0, s->prev_contact[1] != 0};
  double vx = (double)s->v[0], vy = (double)s->v[1], wz = (double)s->v[5];
  // 1. EMA forward-velocity tracking
  s->v_avg += DWP_TRACK_EMA_COEF * (vx - s->v_avg);
  double d1 = s->v_avg - s->command;
  double r = DWP_W_TRACK * exp(-(d1 * d1) / DWP_TRACK_SIGMA_SQ);
  // 2. alive
  r += DWP_W_ALIVE;
  // 3. lateral / yaw
  r -= DWP_W_LATERAL * (vy * vy + wz * wz);
  // 4. action rate (vs the previous step's action, all-env memory)
  double ar = 0;
  for (int j = 0; j < DW_J; j++) {
    double d = a[j] - (double)s->prev_state_action[j];
    ar += d * d;
  }
  r -= DWP_W_ACTION_RATE * ar;
  // 5. torque (boundary PD estimate at the post-step state; `eff` is the
  // APPLIED, latency-delayed targets, and kp carries the per-env scale)
  double tq = 0;
  for (int j = 0; j < DW_J; j++) {
    double m = ((double)DW_KP * s->kp_scale) * (eff[j] - (double)s->q[7 + j])
             - (double)DW_KV * (double)s->v[6 + j];
    m = fmin((double)DW_EFFORT_CAP_TABLE[j],
             fmax(-(double)DW_EFFORT_CAP_TABLE[j], m));
    tq += m * m;
  }
  r -= DWP_W_TORQUE * tq;
  // 6. qualified steps (evaluator-mirroring gates) + chatter
  double phase = dw_policy_phase(s->phase0, s->t, s->command);  // pre-incr t
  bool stance_left = sin(phase) >= 0.0;
  bool touchdown[2], liftoff[2], qualified[2];
  int nq = 0, nchat = 0;
  for (int f = 0; f < 2; f++) {
    touchdown[f] = !prevc[f] && contact[f];
    liftoff[f] = prevc[f] && !contact[f];
    bool duration_ok = s->air_time[f] >= DWP_AIR_TIME_MIN
                    && s->air_time[f] <= DWP_AIR_TIME_MAX;
    bool placement_ok =
        ((double)foot_x[f] - s->liftoff_x[f]) >= DWP_PLACEMENT_MIN_M;
    bool opp_ok = s->opp_support[f]
               >= DWP_OPP_SUPPORT_FRAC * fmax(s->air_time[f], dt);
    bool stance_ok = s->pre_swing_stance[f] >= DWP_STANCE_MIN_S;
    bool phase_ok = f == 0 ? stance_left : !stance_left;
    qualified[f] = touchdown[f] && duration_ok && placement_ok && opp_ok
                && stance_ok && phase_ok;
    if (qualified[f]) nq++;
    if (touchdown[f] && s->air_time[f] < DWP_CHATTER_MAX_S) nchat++;
  }
  r += DWP_W_AIR_TIME * nq;
  r -= DWP_W_CHATTER * nchat;
  // per-foot sequential updates (foot 0 then 1, exactly like the python loop:
  // foot 1's alternation check sees foot 0's last_foot update)
  for (int f = 0; f < 2; f++) {
    if (qualified[f] && s->last_foot == 1 - f) r += DWP_W_ALTERNATE;
    if (qualified[f] && s->last_foot == f) r -= DWP_W_SAME_FOOT;
    if (qualified[f]) s->last_foot = f;
    if (liftoff[f]) s->liftoff_x[f] = (double)foot_x[f];
    if (liftoff[f]) s->opp_support[f] = 0.0;
    if (!contact[f]) s->opp_support[f] += (contact[1 - f] ? 1.0 : 0.0) * dt;
    if (liftoff[f]) s->pre_swing_stance[f] = s->stance_time[f];
  }
  // reward.py v9: stance credit accrues only on FULL-contact steps (all
  // native ticks in contact); tick-scale flicker inside a stance resets it,
  // mirroring the evaluator's continuous-support requirement.
  for (int f = 0; f < 2; f++) {
    bool solid = contact[f] && s->contact_ticks[f] >= DWP_TICKS_FULL;
    s->stance_time[f] = solid ? s->stance_time[f] + dt : 0.0;
  }
  // 6c. flicker: stance at both boundaries but partial tick contact
  int nflick = 0;
  for (int f = 0; f < 2; f++)
    if (prevc[f] && contact[f] && s->contact_ticks[f] < DWP_TICKS_FULL) nflick++;
  r -= DWP_W_FLICKER * nflick;
  for (int f = 0; f < 2; f++)
    s->air_time[f] = contact[f] ? 0.0 : s->air_time[f] + dt;
  // 7. clearance
  int ncl = 0;
  for (int f = 0; f < 2; f++)
    if (!contact[f] && (double)sole[f] >= DWP_CLEARANCE_M) ncl++;
  r += DWP_W_CLEARANCE * ncl;
  // 8a. self-imitation (reward.py v12): joint pose near the phase-indexed
  // reference cycle while commanded. Exact numpy mirror: bin =
  // int(mod(phase/(2pi), 1) * BINS) % BINS (np.mod follows the divisor
  // sign; the cast truncates), err = sum(sq)/J, bonus = W*exp((-err)/s2).
  // DW_REF_GAIT and the constants are generated from the model's reward
  // module (drift-test pinned). The whole block sits behind a
  // compile-time-constant weight test so a model with DW_IMIT_W 0.0 (the
  // H0 humanoid: no reference gait exists, all-zero DW_REF_GAIT
  // placeholder) pays ZERO cost -- the table lookup and the exp() are
  // dead-code-eliminated -- while any nonzero weight (duck 0.5) keeps the
  // arithmetic bit-identical to the ungated original.
  if (DW_IMIT_W != 0.0) {
    double frac = fmod(phase / (2.0 * DWP_PI), 1.0);
    if (frac < 0.0) frac += 1.0;
    int bin = ((int)(frac * (double)DW_REF_BINS)) % DW_REF_BINS;
    double serr = 0;
    for (int j = 0; j < DW_J; j++) {
      double dref = (double)s->q[7 + j] - DW_REF_GAIT[bin][j];
      serr += dref * dref;
    }
    double ierr = serr / (double)DW_J;
    if (fabs(s->command) > 0.0)
      r += DW_IMIT_W * exp((-ierr) / DW_IMIT_SIGMA_SQ);
  }
  // 8b. phase-locked stance while commanded -- SIGNED (reward.py v7):
  // mismatched contact pays -W_PHASE so planting one foot through both
  // clock windows nets zero instead of the standing subsidy.
  double match = (contact[0] == stance_left ? 1.0 : -1.0)
               + (contact[1] == !stance_left ? 1.0 : -1.0);
  if (fabs(s->command) > 0.0) r += DWP_W_PHASE * match;
  // 8. double support beyond the grace while commanded
  bool both = contact[0] && contact[1];
  s->double_support = both ? s->double_support + dt : 0.0;
  if (fabs(s->command) > 0.0 && s->double_support > DWP_DOUBLE_SUPPORT_GRACE)
    r -= DWP_W_DOUBLE_SUPPORT;
  return (float)r;
}

// flat.py step() on device: action -> slew-limited targets -> n_ticks physics
// -> reward -> termination -> observation. A solver fault (flat.py raises
// SolverFault) marks the environment done with its state frozen at the last
// accepted tick; the diagnostic carries the failure.
static DW_HD void dw_step_policy_env(DwState* s, const float* action,
                                     uint32_t n_ticks, const DwParams* params,
                                     DwWork* w, float* obs,
                                     float* reward_out, uint8_t* done_out,
                                     dwc1_diagnostic* diag) {
  const bool live = s->done == 0;   // warp-uniform read (pre-step state)
  // Fast-termination short-circuit for envs ALREADY done at block entry:
  // no physics, no tracker mutation, reward 0, frozen observation. The
  // python env keeps ticking fallen robots (and its tracker) until the
  // trainer resets them; under the training flag those solves -- pegged
  // post-fall impact piles -- are skipped outright. Reward for done envs
  // is 0 on both paths and the trainer resets them at the next boundary,
  // so nothing training-visible changes.
  if (params->fast_termination && !live) {
    DW_LANE0 {
      *reward_out = 0.0f;
      *done_out = 1;
      dw_policy_observe(s, obs);
      diag->status = DWC1_OK;
    }
    DW_SYNC();
    return;
  }
  // The f64 policy chain stays bit-exact vs the python env: per-env scalar
  // sequential ops on lane 0 only; physics below is lane-cooperative.
  DW_LANE0 {
    for (int j = 0; j < DW_J; j++)
      w->a64[j] = fmin(1.0, fmax(-1.0, (double)action[j]));
    if (live)
      for (int j = 0; j < DW_J; j++) {
        double requested = DW_HOME_TARGETS_F64[j] + DWP_ACTION_SCALE * w->a64[j];
        double lo = s->targets[j] - DWP_MAX_TARGET_INCREMENT;
        double hi = s->targets[j] + DWP_MAX_TARGET_INCREMENT;
        s->targets[j] = fmin(hi, fmax(lo, requested));
      }
    for (int j = 0; j < DW_J; j++)
      w->eff64[j] = fmin((double)DW_LIMIT_UPPER[j],
                         fmax((double)DW_LIMIT_LOWER[j], s->targets[j]));
    // actuation latency: the ring stores the COMPUTED effective targets by
    // step index; physics consumes eff[t - latency] (reset targets before
    // step `latency`). latency 0 reads back the just-written slot, so the
    // feature-off path is bit-identical. The slew chain above stays
    // undelayed; done envs keep t frozen (same slot rewritten, applied
    // targets pinned) exactly like the python spec.
    {
      const uint32_t P = (uint32_t)(DWC1_MAX_LATENCY + 1);
      uint32_t slot = s->t % P;
      for (int j = 0; j < DW_J; j++) s->eff_ring[slot][j] = w->eff64[j];
      uint32_t aslot = (s->t + P - s->latency_steps) % P;
      for (int j = 0; j < DW_J; j++) {
        w->app64[j] = s->eff_ring[aslot][j];
        w->eff[j] = (float)w->app64[j];
      }
    }
  }
  DW_SYNC();
  int st = dw_step_env(s, w->eff, n_ticks, params, w, diag,
                       params->fast_termination != 0);
  DW_LANE0 {
    if (st != DWC1_OK) s->done = 1;
    dw_body_states(s->q, s->v, w->bodies13);
    float sole[2];
    dw_sole_heights(w->bodies13, sole);
    float foot_x[2] = {w->bodies13[DW_PAIR_BODY_A[0]][0],
                       w->bodies13[DW_PAIR_BODY_A[1]][0]};
    float r = dw_policy_reward(s, w->a64, w->app64, sole, foot_x);
    if (live) s->t += 1;
    bool fell = dw_policy_fell(s);   // height / up-axis tilt / nonfinite
    if (live && (fell || s->t >= DWP_HORIZON)) s->done = 1;
    *reward_out = live ? r : 0.0f;
    for (int j = 0; j < DW_J; j++) {
      s->prev_state_action[j] = (float)w->a64[j];      // all-env reward memory
      if (live) s->prev_action[j] = (float)w->a64[j];  // live-only obs memory
    }
    for (int p = 0; p < DW_PAIRS; p++)
      s->prev_contact[p] = s->cache[p].count > 0 ? 1 : 0;
    dw_policy_observe(s, obs);
    *done_out = s->done;
  }
  DW_SYNC();
}

// Masked policy reset: physics + policy/tracker state back to the creation
// state; the command and per-episode phase offset are taken from the host
// (resampled exactly like flat.py's counter-based _episode_rng -- command
// drawn first, then phase0 = 2*pi*rng.random() -- which is not reproducible
// in-kernel) or kept when the caller passes none.
static DW_HD void dw_policy_reset_env(DwState* s, const DwState* initial,
                                      const double* command,
                                      const double* phase0) {
  double keep_command = s->command, keep_phase0 = s->phase0;
  *s = *initial;
  s->command = command ? *command : keep_command;
  s->phase0 = phase0 ? *phase0 : keep_phase0;
}

#endif  // DUCK_CUDA_KERNEL_H
