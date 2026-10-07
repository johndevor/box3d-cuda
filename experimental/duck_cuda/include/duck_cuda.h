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
// v4: per-episode gait-phase offsets (flat.py v10): dwc1_reset_policy gained
//     the trailing phase_offsets input.
// v5: warp-per-env occupancy telemetry: dwc1_device_info struct +
//     dwc1_device_info_get (launch geometry, registers/local-mem per thread,
//     occupancy blocks/SM, stack limit, resident-env estimate).
// v6: per-episode domain randomization + actuation latency (off by default):
//     dwc1_create gained the dwc1_randomization config, new
//     dwc1_set_randomization applies per-env draws (host-side RNG).
// v7: per-episode GRAVITY randomization: dwc1_randomization gained r_gravity
//     and dwc1_env_random gained gravity_scale (both inserted before the
//     latency fields -- struct layouts changed, hence the bump). Absent /
//     0 / neutral 1.0 is bit-identical to v6 (fingerprint-pinned:
//     tests/test_duck_fingerprint.py).
// v8: generated-header-selected ENVIRONMENT KIND. DW_ENV_KIND in the model
//     header picks the device policy layer at compile time:
//       DWC1_ENV_KIND_LOCOMOTION (default when the header omits it): the
//         duck/humanoid 3*J+16 gait contract -- today's exact code path,
//         byte-identical (fingerprint-pinned);
//       DWC1_ENV_KIND_REACH: the fixed-base arm reach contract
//         (walk/env/arm_reach.py + arm_reward.py): OBS 27, host-drawn
//         target sequence, in-kernel acquisition/reward/termination and
//         judge-shadow counters for the frozen arm judge's clauses.
//     New exports: dwc1_env_kind, dwc1_obs_width, dwc1_reach_set_targets,
//     dwc1_reach_get (+ dwc1_reach_state). dwc1_reset_policy keeps its
//     signature; its two f64 slots carry per-kind semantics (below).
#define DWC1_ABI_VERSION 8
int dwc1_abi_version(void);

// Which policy layer this build compiled (DW_ENV_KIND of its model header)
// and the observation width it writes (DW_ENV_OBS).
enum { DWC1_ENV_KIND_LOCOMOTION = 0, DWC1_ENV_KIND_REACH = 1 };
int dwc1_env_kind(void);
int dwc1_obs_width(void);
// REACH action contract this build compiled (DW_ENV_ACTION_MODE, additive
// export): 0 = ABS (limit-scaled absolute targets, slew-limited), 1 =
// DELTA (target += a * MAX_TARGET_INCREMENT, clamped to the limits;
// a = 0 holds). Locomotion builds report 0 (no meaning).
enum { DWC1_ACTION_MODE_ABS = 0, DWC1_ACTION_MODE_DELTA = 1 };
int dwc1_action_mode(void);

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

// Launch/occupancy telemetry (v5). GPU builds fill everything from the CUDA
// runtime (cudaFuncGetAttributes / cudaOccupancyMaxActiveBlocksPerMulti-
// processor / device limits); serial builds report lanes_per_env=1 and zero
// for the device-only fields. resident_envs_estimate =
// step_blocks_per_sm * sm_count * envs_per_block: throughput should scale
// with E until roughly this many environments are in flight.
typedef struct dwc1_device_info {
  uint32_t lanes_per_env, threads_per_block, min_blocks_per_sm, sm_count;
  uint32_t step_regs_per_thread, step_local_bytes, step_blocks_per_sm;
  uint32_t policy_regs_per_thread, policy_local_bytes, policy_blocks_per_sm;
  uint64_t stack_limit_bytes, workspace_bytes_per_env;
  uint32_t resident_envs_estimate, reserved;
} dwc1_device_info;

// ---- per-episode domain randomization + actuation latency (v6) ------------
// Creation-time config: RANGES for the uniform [1-r, 1+r] multipliers and
// the maximum command latency in policy steps. All zero (or a NULL pointer)
// = feature off = today's exact behavior (bit-identical: neutral multipliers
// apply as (float)((double)X * 1.0) == X and latency 0 reads back the
// just-written targets). Per-env VALUES are drawn host-side at reset (the
// counter-based numpy stream is not reproducible in-kernel) and applied via
// dwc1_set_randomization. Scales affect physics consumption points only
// (mass AND principal inertia together, pair mu, PD kp, passive damping,
// gravity) -- never the model tables, observations or the reference-weight
// regularizers.
//
// GRAVITY (v7) is ONE-SIDED: the authored DW_GRAVITY_Z magnitude is the
// MAXIMUM and gravity_scale in [1 - r_gravity, 1] draws lighter worlds
// (gravity_scale = 1 - r_gravity * u, u ~ U[0,1)). Rationale: the humanoid
// is authored at -20 m/s^2 (2 g); with r_gravity = 1 - 9.81/20 = 0.5095 a
// single binary trains across gravity in (9.81, 20] m/s^2 -- Earth at the
// light end, the authored 2 g at the top -- so the 2g-vs-Earth question
// dissolves into the policy's implicit system ID. A symmetric [1-r, 1+r]
// scale on the authored magnitude cannot express that range (its maximum
// would exceed the authored value). Applied at the single consumption point
// (dw_tick -> dw_evaluate's gravity vector) as
// gz = (float)((double)DW_GRAVITY_Z * gravity_scale): identity at 1.0.
#define DWC1_MAX_LATENCY 4
#define DWC1_MAX_R_GRAVITY 0.9   // gravity never below 10% of authored
typedef struct dwc1_randomization {
  double r_mass, r_friction, r_kp, r_damping;  // each in [0, 0.5]
  double r_gravity;                            // [0, DWC1_MAX_R_GRAVITY]
  uint32_t max_latency_steps, reserved;        // 0..DWC1_MAX_LATENCY
} dwc1_randomization;
typedef struct dwc1_env_random {
  double mass_scale, friction_scale, kp_scale, damping_scale;  // [0.5, 1.5]
  double gravity_scale;                        // [1 - r_gravity, 1]
  uint32_t latency_steps, reserved;            // <= config max_latency_steps
} dwc1_env_random;

typedef struct dwc1_scene dwc1_scene;

// joint_offsets: optional [E,14] perturbations added to the home joint pose
// (clipped to the joint limits), matching NativeDuckLane(joint_offsets=...).
// randomization: creation-time ranges (NULL = off); when off, only neutral
// per-env values (1.0 scales, latency 0) are accepted later.
int dwc1_create(uint32_t environments, const float* joint_offsets,
                const dwc1_randomization* randomization, dwc1_scene** out);

// Apply per-env randomization values to selected envs (NULL mask = all) and
// reset their actuation-latency buffer to the reset targets. Values must lie
// inside the creation config's ranges. Call right after a reset of the same
// envs (dwc1_reset_policy or dwc1_reset), before stepping.
int dwc1_set_randomization(dwc1_scene*, const uint8_t* mask,
                           const dwc1_env_random* randoms /* [E] */);
void dwc1_destroy(dwc1_scene*);
int dwc1_info_get(const dwc1_scene*, dwc1_info*);
int dwc1_device_info_get(const dwc1_scene*, dwc1_device_info*);

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
// One policy step entirely on device. LOCOMOTION kind: clip(actions,+-1)
// -> targets =
// clip(HOME + DW_ENV_ACTION_SCALE*a, previous targets +-
// DW_ENV_MAX_TARGET_INCREMENT) (persistent per-env slew reference, frozen
// while done) -> joint-limit clip -> n_ticks physics -> reward (tracker
// state lives in the env state) -> termination (root height <
// DW_ENV_MIN_HEIGHT_FRACTION*home, tilt > DW_ENV_MAX_TILT_RAD on the
// model's up-axis, nonfinite, DW_ENV_HORIZON_STEPS) -> DW_ENV_OBS-dim
// observation. Every constant comes from the generated model header's
// DW_ENV_* contract block: duck builds mirror walk/env/flat.py +
// walk/env/reward.py (OBS 58 / ACT 14), humanoid builds mirror
// walk/env/humanoid_flat.py + walk/env/humanoid_reward.py (OBS 52 /
// ACT 12); the selected header's python env pair is the contract.
// REACH kind (v8): targets = clip(lower + (a+1)/2*(upper-lower), previous
// +- per-joint DW_ENV_MAX_TARGET_INCREMENT_F64) -> n_ticks physics ->
// tip/wrist/elbow FK -> acquisition hold -> arm_reward.py (DW_RW_* block)
// -> termination (judge proxy clause, nonfinite, horizon) -> OBS 27 =
// [q, QDOT_SCALE*qd, target, tip, target-tip, prev action] mirroring
// walk/env/arm_reach.py. The
// policy chain runs in f64 mirroring the python env exactly. Done
// environments keep stepping physics with frozen targets, return reward 0
// and stay done until reset (no auto-reset), like the python envs. A solver
// fault (where the python env raises SolverFault) freezes the env at its
// last accepted tick, marks it done and reports via its diagnostic; the call
// still returns DWC1_OK so training batches survive per-env faults.
//   actions [E,DW_J] f32 in [-1,1]; obs [E,DW_ENV_OBS] f32; reward [E] f32;
//   done [E] u8; diagnostics [E].
int dwc1_step_policy(dwc1_scene*, const float* actions, uint32_t n_ticks,
                     float* obs, float* reward, uint8_t* done,
                     dwc1_diagnostic* diagnostics);

// The DW_ENV_OBS-dim observation of the current state (no stepping): what
// the python env's reset()/set_command() return.
int dwc1_observe(const dwc1_scene*, float* obs /* [E,DW_ENV_OBS] */);

// Commanded forward velocity (m/s) for every env, f64 to match the python
// command values exactly. LOCOMOTION kind only (REACH: DWC1_INVALID).
int dwc1_set_command(dwc1_scene*, const double* commands /* [E] */);

// gate_proxy_* judge-shadow gait counters (METRICS ONLY; additive export,
// ABI version unchanged). Per-env continuous approximation of the frozen
// walking judge's core footfall clauses -- qualified swings (duration in
// the judge's window, whole-sole contiguous clearance, forward placement
// at touchdown), per foot, plus consecutive-same-foot (alternation
// violation) count -- computed in-kernel per accepted tick from state the
// kernel already has, thresholds from the generated DW_GATE_* header
// block. qualified_*/alternation_violations cover the CURRENT episode so
// far; episode_* fields are the snapshot of the last COMPLETED episode
// (taken at policy reset). HONESTY: raw tick resolution, no 20 ms contact
// debounce, no support/slip clauses -- a cheap culling/monitoring shadow,
// never a substitute for the frozen CPU judge. Counters never feed reward
// or termination; certified paths are bit-identical (fingerprint-proven).
// Why an env last became done (gate-proxy readback; metrics only).
// REACH kind (v8): DWC1_TERM_FELL = proxy crash / non-finite state (the
// arm env's `crashed`), DWC1_TERM_REACH_STARVED = acquisition with no
// queued target (host contract violation, see dwc1_reach_set_targets).
enum { DWC1_TERM_NONE = 0, DWC1_TERM_FELL = 1, DWC1_TERM_GATE_DEADLINE = 2,
       DWC1_TERM_ALTERNATION = 3, DWC1_TERM_HORIZON = 4, DWC1_TERM_FAULT = 5,
       DWC1_TERM_REACH_STARVED = 6 };
// REACH kind mapping of this struct (so chain fitness / curriculum code
// reading gate_proxy works unchanged): qualified_left = targets acquired
// this episode (target_index), qualified_right = 0, alternation_violations
// = judge-clause violating ticks (limit + speed + proxy); episode_* the
// same for the last completed episode. dwc1_reach_get has the detail.
typedef struct dwc1_gate_proxy {
  uint32_t qualified_left, qualified_right, alternation_violations;
  uint32_t episode_qualified_left, episode_qualified_right;
  uint32_t episode_alternation_violations;
  uint32_t termination_reason;          // DWC1_TERM_* (current/last done)
  uint32_t episode_termination_reason;  // last COMPLETED episode's reason
} dwc1_gate_proxy;
int dwc1_gate_proxy_get(const dwc1_scene*, dwc1_gate_proxy* out /* [E] */);

// OPT-IN judge-aligned termination rules driven by the gate_proxy_*
// counters (additive export; BOTH knobs default 0 = OFF, so all existing
// gates and the duck fingerprint hold bit-identically). Runtime per-scene
// values -- a training curriculum can tighten them per leg without
// recompiling; they compose with dwc1_set_fast_termination (a
// gate-terminated env freezes the same way). first_deadline_ticks: live
// envs terminate at the policy boundary once this many accepted episode
// ticks pass without ANY gate-qualified swing (the judge's first-step
// clause as a pruning rule). max_alternation_violations: terminate when
// the cumulative same-foot consecutive qualified-touchdown count reaches
// this value. Reasons surface as dwc1_gate_proxy.termination_reason.
// REACH kind (v8): the same two knobs read the reach counters --
// first_deadline_ticks = no target acquired within that many accepted
// ticks; max_alternation_violations = judge-clause violating ticks
// (limit + speed + proxy) reaching that count -- so a tech-tree ladder
// authored against gate_proxy semantics drives the arm unchanged.
int dwc1_set_gate_termination(dwc1_scene*, uint32_t first_deadline_ticks,
                              uint32_t max_alternation_violations);

// Reference State Initialization (DeepMimic-style; additive export,
// default OFF/0.0 keeps every reset bit-identical). fraction in [0, 1]:
// each policy-reset env independently starts, with this probability, ON
// the DW_REF_GAIT reference cycle at the bin aligned with its freshly
// drawn phase offset (joint q from the table row, joint qdot from the
// table's finite difference at the gait clock rate, root at the reset
// pose) -- so the imitation reward term is consistent at t = 0. The
// scene-level draw stream is deterministic (fixed seed at creation). An
// all-zero reference table makes enabled RSI a no-op by construction.
// LOCOMOTION kind only (REACH accepts 0.0, rejects any other fraction).
int dwc1_set_rsi(dwc1_scene*, double fraction);

// Training-lane throughput switch (default OFF; additive export, ABI
// version unchanged). When enabled, dwc1_step_policy (a) skips physics
// entirely for envs already done at block entry (frozen state, reward 0,
// frozen observation) and (b) latches the termination predicate after
// every accepted tick, so a robot that falls mid-block stops ticking
// immediately instead of solving its post-fall impact pile -- those solves
// peg the fp32 solver at its full iteration + Tresca budget (measured
// 25-100x a live solve) and dominate whole launches while contributing
// nothing to training. LIVE-env physics arithmetic is untouched; with the
// flag off, behavior is bit-identical to previous builds. The plain
// physics path (dwc1_step / tick_block) never consults the flag.
int dwc1_set_fast_termination(dwc1_scene*, uint32_t enable);

// Masked policy reset: physics AND policy/tracker state back to the creation
// state; the two f64 [E] slots are applied to selected envs, NULL keeps each
// selected env's previous value. Their meaning is per ENV KIND (v8):
//   LOCOMOTION: commands[e] (m/s) and phase_offsets[e] -- the caller
//     resamples them like flat.py's _episode_rng (counter-based numpy PCG64,
//     not reproducible in-kernel; per episode the command is drawn FIRST,
//     then phase0 = 2*pi*rng.random() -- walk/env/cuda_lane.py implements
//     the exact stream). Unchanged from v3/v4.
//   REACH: slot a = the episode's difficulty TIER (integer-valued f64, the
//     judge's tier index) and slot b = the TARGET-SEQUENCE KEY (the host's
//     per-episode stream index, i.e. arm_reach._episode_rng(seed, env,
//     episode)'s `episode`), both stored for readback/forensics only; the
//     targets themselves are host-drawn from that stream (the numpy PCG64
//     rejection sampler is not reproducible in-kernel) and pushed with
//     dwc1_reach_set_targets. A reach reset CLEARS the env's target slots:
//     the host must push the active AND the queued target before stepping
//     (walk/env/arm_cuda_lane.py implements the exact arm_reach.py stream:
//     tier draw, then targets in acquisition order).
int dwc1_reset_policy(dwc1_scene*, const uint8_t* mask,
                      const double* commands, const double* phase_offsets);

// ---- REACH kind (v8; every entry returns DWC1_INVALID on other kinds) ----
// Per-env target queue: the ACTIVE target and ONE queued NEXT target (world
// xyz, f64, exactly the host's draws). On acquisition (tip within
// DW_ENV_ACQ_RADIUS_M at DW_ENV_ACQ_HOLD_STEPS consecutive policy
// boundaries, the frozen judge's rule) the kernel promotes next -> active,
// bumps target_index and clears next_valid; the host reads target_index
// (dwc1_reach_get) after each step and pushes the following draw for the
// envs that advanced, so a queued target is always present (an env whose
// acquisition finds NO queued target is terminated with
// DWC1_TERM_REACH_STARVED and counted in `starved` -- a host contract
// violation made loud rather than a silently repeated target). active /
// next may be NULL (slot untouched); NULL mask = all envs.
int dwc1_reach_set_targets(dwc1_scene*, const uint8_t* mask,
                           const double* active /* [E,3] or NULL */,
                           const double* next /* [E,3] or NULL */);
#define DWC1_REACH_TARGETS 5   // judged targets per episode (arm judge N_TARGETS)
typedef struct dwc1_reach_state {
  double target[3], next_target[3];   // active / queued (world m)
  double tier, key;                   // reset slots a / b (host semantics)
  uint32_t target_index, hold, next_valid, valid;
  // judge-shadow counters, CURRENT episode (metrics only, never read by
  // reward or termination -- except the opt-in gate rules, which are
  // the same knobs the humanoid uses, see dwc1_set_gate_termination):
  //   acquire_step[k]: 1-based policy step at which target k (k < 5) was
  //     acquired, 0 = not yet (judge time_s = step * CONTROL_DT);
  //   *_violation_ticks: accepted ticks violating the judge's joint-limit
  //     (LIMIT_TOL 0.01 rad), joint-speed (URDF limits) and
  //     self-collision/floor proxy clauses (clause 5 counts EXACTLY the
  //     judge's violating_ticks).
  uint32_t acquire_step[DWC1_REACH_TARGETS];
  uint32_t limit_violation_ticks, speed_violation_ticks;
  uint32_t proxy_violation_ticks, starved;
  // last COMPLETED episode snapshot (taken at policy reset)
  uint32_t episode_acquired, episode_acquire_step[DWC1_REACH_TARGETS];
  uint32_t episode_limit_violation_ticks, episode_speed_violation_ticks;
  uint32_t episode_proxy_violation_ticks, reserved;
} dwc1_reach_state;
int dwc1_reach_get(const dwc1_scene*, dwc1_reach_state* out /* [E] */);

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
