// SPDX-License-Identifier: MIT
// Batched fp32 duck worlds: one flat-floor Open Duck (plain-14) per lane,
// E independent environments, one CUDA thread per environment (or a plain
// loop in the serial parity build). The f64 CPU lane (integrated_duck_v1)
// is the physics oracle.
//
// PARITY CONTRACT vs the f64 CPU lane (verified by tests/test_serial_parity.py)
// -----------------------------------------------------------------------------
// All state and arithmetic are float32; divergence from the f64 oracle grows
// with simulated time and with contact activity. The gates are:
//  (a) home-hold, 500 ticks (1 s): |root position - CPU| < 2 mm,
//      tilt difference < 1 deg;
//  (b) seeded random actions (clip 0.5, 10-tick holds), 300 ticks: bounded
//      per-tick divergence, no NaN, unit quaternions, ground penetration
//      <= 5 mm;
//  (c) all recorded fault-corpus states step 10 ticks without fault/NaN.
// Solver certificates are fp32-scaled: impulse tolerance 1e-6 (CPU: 1e-8),
// momentum residual gate 2e-4 absolute (CPU: 1e-8). Determinism holds within
// a build: identical inputs give bit-identical trajectories.
#ifndef DUCK_CUDA_H
#define DUCK_CUDA_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

// Bumped whenever an exported signature or output layout changes.
// v2: dwc1_read gained the trailing contact_ticks output; dwc1_step now
//     resets and accumulates per-foot contact tick counters.
// v3: device-side policy path (walk/env/flat.py + reward.py in-kernel):
//     dwc1_step_policy, dwc1_observe, dwc1_set_command, dwc1_reset_policy.
#define DWC1_ABI_VERSION 3
int dwc1_abi_version(void);

enum {
  DWC1_OK = 0,
  DWC1_INVALID = 1,
  DWC1_DYNAMICS = 2,
  DWC1_NO_CONVERGENCE = 3,
  DWC1_ALLOCATION = 4,
  DWC1_NUMERIC = 5
};

typedef struct dwc1_point {
  uint64_t feature;
  float point[3], depth, normal_impulse, tangent_impulse[2];
} dwc1_point;

typedef struct dwc1_manifold {
  uint32_t count;
  float normal[3], tangent1[3], tangent2[3];
  dwc1_point points[4];
} dwc1_manifold;

// Mirrors idv1_diagnostic field-for-field where meaningful; float residuals.
typedef struct dwc1_diagnostic {
  uint32_t environment, status, iterations, contact_points, active_limits, ticks;
  float joint_residual, normal_residual, tangent_residual, momentum_residual;
  float maximum_normal_impulse, maximum_penetration;
} dwc1_diagnostic;

typedef struct dwc1_info {
  uint32_t environments, bodies, joints, dofs;
  float dt, kp, kv, effort_cap, home_root_height;
  float home_qpos[21];            // root xyz + quat xyzw + 14 joint angles
  float joint_lower[14], joint_upper[14];
} dwc1_info;

typedef struct dwc1_scene dwc1_scene;

// joint_offsets: optional [E,14] perturbations added to the home joint pose
// (clipped to the joint limits), matching NativeDuckLane(joint_offsets=...).
int dwc1_create(uint32_t environments, const float* joint_offsets,
                dwc1_scene** out);
void dwc1_destroy(dwc1_scene*);
int dwc1_info_get(const dwc1_scene*, dwc1_info*);

// Hold `targets` [E,14] for n_ticks ticks of 0.002 s, all inside ONE device
// kernel launch. The PD torque is recomputed every tick from the current
// q/qdot exactly like the CPU lane: clip(kp*(clip(target,limits)-q) -
// kv*qdot, +-cap). An environment whose solve fails keeps its pre-tick
// state, freezes for the remaining ticks and reports the failure in its
// diagnostic; other environments are unaffected. Per-foot contact tick
// counters are zeroed at the start of the call and incremented once per
// accepted tick whose solve manifold has contact points, so one dwc1_read
// after the call replaces a per-tick read loop.
// Returns DWC1_OK when every environment completed all ticks.
int dwc1_step(dwc1_scene*, const float* targets, uint32_t n_ticks,
              dwc1_diagnostic* diagnostics /* [E] */);

// All outputs optional (NULL skips). Host copies; the CUDA build keeps the
// authoritative state resident on the device and copies once per call.
//   qpos [E,21], velocity [E,20], warm [E,42], time [E] (seconds, count*dt),
//   count [E], body_state [E,16,13] (p3 q_xyzw4 v3 omega3, principal COM),
//   foot_contact [E,2] (solve-cache manifold count > 0; left, right),
//   sole_height [E,2] (min world z over the 18 sole vertices), cache [E,2],
//   contact_ticks [E,2] (per-foot accepted contact ticks of the most recent
//   dwc1_step call; left, right).
int dwc1_read(const dwc1_scene*, float* qpos, float* velocity, float* warm,
              double* time, uint64_t* count, float* body_state,
              uint8_t* foot_contact, float* sole_height, dwc1_manifold* cache,
              uint32_t* contact_ticks);

// ---- device-side policy path (obs + reward + termination in-kernel) -------
// One policy step entirely on device: clip(actions,+-1) -> targets =
// clip(HOME + 0.25*a, previous targets +- 0.1048) (persistent per-env slew
// reference, frozen while done) -> joint-limit clip -> n_ticks physics ->
// reward.py (tracker state lives in the env state) -> termination (root
// height < 0.7*home, tilt > 45 deg, nonfinite, 400-step horizon) -> 58-dim
// observation. The policy chain runs in f64 mirroring the python env exactly
// (walk/env/flat.py and walk/env/reward.py are the contract). Done
// environments keep stepping physics with frozen targets, return reward 0
// and stay done until reset (no auto-reset), like FlatFloorDuckEnv. A solver
// fault (where the python env raises SolverFault) freezes the env at its
// last accepted tick, marks it done and reports via its diagnostic; the call
// still returns DWC1_OK so training batches survive per-env faults.
//   actions [E,14] f32 in [-1,1]; obs [E,58] f32; reward [E] f32;
//   done [E] u8; diagnostics [E].
int dwc1_step_policy(dwc1_scene*, const float* actions, uint32_t n_ticks,
                     float* obs, float* reward, uint8_t* done,
                     dwc1_diagnostic* diagnostics);

// The 58-dim observation of the current state (no stepping): what
// FlatFloorDuckEnv.reset()/set_command() return.
int dwc1_observe(const dwc1_scene*, float* obs /* [E,58] */);

// Commanded forward velocity (m/s) for every env, f64 to match the python
// command values exactly.
int dwc1_set_command(dwc1_scene*, const double* commands /* [E] */);

// Masked policy reset: physics AND policy/tracker state back to the creation
// state; commands[e] (f64, [E]) is applied to selected envs -- the caller
// resamples it like flat.py's _episode_rng (counter-based numpy PCG64, not
// reproducible in-kernel; walk/env/cuda_lane.py implements the exact
// sampling). NULL commands keeps each selected env's previous command.
int dwc1_reset_policy(dwc1_scene*, const uint8_t* mask,
                      const double* commands);

// Current-pose contact geometry with zero impulses (mirrors bcv1_query).
int dwc1_query(const dwc1_scene*, dwc1_manifold* out /* [E,2] */);

// NULL mask selects all; any nonzero byte selects. Restores the creation
// state (including per-env joint offsets), zero warm start, empty cache.
int dwc1_reset(dwc1_scene*, const uint8_t* mask);

// Full single-env state injection (fault-corpus replay / forensics).
// warm42 is the CPU lane layout: friction rows [0,14), lower [14,28),
// upper [28,42). cache2 may be NULL (empty cache). Quaternion is normalized.
int dwc1_set_state(dwc1_scene*, uint32_t environment, const float* qpos21,
                   const float* velocity20, const float* warm42,
                   const dwc1_manifold* cache2, uint64_t count);

#ifdef __cplusplus
}
#endif
#endif
