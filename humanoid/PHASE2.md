# H0 humanoid Phase 2 — walking bring-up (env + reward v1 + header pins)

Prereqs landed upstream (verified here): workstream-A solver repair (the
perturbed-standing reproducers now tick through clean on both lanes —
`test_humanoid_serial_parity.py::test_perturbed_standing_ticks_through_both_lanes`),
per-joint effort caps in the kernel (`DW_EFFORT_CAP_TABLE` consumed at the
PD clamp and the reward torque estimate; scalar `DW_EFFORT_CAP` remains
only as the `dwc1_info` summary field), CPU lane at 16384 iterations.

Deliverables (new files; nothing existing modified except the humanoid
generator/tests I own):

- `walk/env/humanoid_flat.py` — `FlatFloorHumanoidEnv` (OBS 52, ACT 12),
  mirroring `walk/env/flat.py` step for step.
- `walk/env/humanoid_reward.py` — reward v1: duck v12 shape, humanoid
  constants, NO self-imitation (empty hook; no reference gait exists).
- regenerated `humanoid/include/duck_model.h` with all reward/env/clock
  constants pinned (`DW_RW_*`, `DW_ENV_*`, `DW_PHASE_HZ_*`), drift-tested.
- `humanoid/tests/test_humanoid_env.py` (obs layout, header pins, reward,
  termination), `humanoid/tests/test_humanoid_train_smoke.py`
  (end-to-end CPU PPO smoke) and `humanoid/tests/test_humanoid_gpu_train.py`
  (robot switch, env-vs-kernel policy parity, gpu_train lane-env smoke,
  actor 52→…→12 round-trip; see §6).

## 1. Cadence

One policy step = 0.02 s = 10 × 0.002 s native ticks — the duck env
contract exactly. Phase 1 ran the oracle at the authored per-substep
1/240; Phase 2 pins SIM_DT = 0.002 (h0_lowering.py), which stays below the
1/240 stability bound (FEASIBILITY.md §5; spectral radius 1.0, margin
kv·dt/I_eff ≤ 0.56) and keeps every 0.02 s-based constant and the kernel's
hardcoded control dt valid. Authored control cadence (1/60) belongs to
World's engine; cadence here follows the duck contract per the Phase 2
scope.

## 2. Observation layout (52 = 3·J + 16; duck is 58 = 3·14 + 16)

Structurally identical to flat.py's `_observe`, same tail block:

| slots  | width | content | notes |
|---|---|---|---|
| 0:12   | 12 | joint q − HOME | HOME = 0 (authored H0 reset) |
| 12:24  | 12 | 0.05 · joint qdot | QDOT_OBS_SCALE, duck pinned scale |
| 24:36  | 12 | previous action | live envs only, frozen when done |
| 36:39  | 3 | gravity direction, body frame (−R[2,:]) | = (0,−1,0) at reset (body frame is authored y-up-local) |
| 39:42  | 3 | root angular velocity, body frame (Rᵀω) | |
| 42:45  | 3 | root linear velocity, body frame (Rᵀv) | |
| 45     | 1 | commanded forward velocity (m/s) | |
| 46:48  | 2 | reserved zeros | duck obs[52:54] convention |
| 48:50  | 2 | foot contact flags (L, R) | solve-cache manifolds |
| 50:52  | 2 | gait phase clock (sin, cos) | affine clock, §3 |

Pinned into the generated header as `DW_ENV_OBS 52` + layout comment; the
kernel's `dw_policy_observe` still emits the DUCK layout (58 with 14-joint
offsets) — consuming this table in-kernel is the enumerated kernel edit
(FEASIBILITY.md §2 item 2).

## 3. Env constants (one-line justifications)

| constant | value | duck | why |
|---|---|---|---|
| ACTION_SCALE | 0.5 rad | 0.25 | covers full waist/neck/ankle authored ranges and early-walking swing amplitudes (step ≈ 0.25 m at leg 0.77 m → hip ≈ 0.35, knee ≈ 0.5 rad); similar range fraction to duck's 0.25 |
| MAX_TARGET_INCREMENT | 0.16 rad/step | 0.1048 | = AUTHORED actuator speed limit 8 rad/s (humanoid.rs:785) × 0.02 s; the slew implements exactly H0's host-side speed shaping (humanoid_h0.py::shape_action_targets) |
| COMMANDS_MPS | 0.50/0.75/1.00 | 0.10/0.15/0.20 | ×5 leg-length ratio (hip height ~1.0 m vs ~0.2 m); same slowest-command oversampling (50/25/25) |
| PHASE_HZ_PER_MPS | 3.33 (sweepable) | 16.67 | duck's own recipe hz = v/(2·step): encodes the reward's 0.15 m placement floor, as 16.67 encodes the duck's 30 mm |
| PHASE_HZ_BASE | 0.0 | 0.0 | duck default |
| HORIZON_STEPS | 400 (8 s) | 400 | duck-proven horizon; authored 1200 steps is World-engine cadence |
| MIN_HEIGHT_FRACTION | 0.7 (→ 0.805 m) | 0.7 | duck recipe |
| MAX_TILT | 45° | 45° | duck recipe, but HUMANOID tilt formula (body +Y vs world +Z, humanoid_native_lane.tilt) |
| QDOT_OBS_SCALE | 0.05 | 0.05 | duck pinned candidate; humanoid qdot magnitudes comparable |

## 4. Reward v1 constants (walk/env/humanoid_reward.py)

Duck v12 term structure preserved 1:1 (tracker semantics identical, the
GaitTracker class is reused unmodified); scaled constants only:

| constant | value | duck | rationale |
|---|---|---|---|
| TRACK_SIGMA_SQ | 0.25 | 0.01 | σ=0.5 m/s keeps duck's σ/command ratio ≈ 0.67 at 0.5–1.0 m/s commands |
| TRACK_EMA_S | 0.2 | 0.4 | ~half a gait period (clock 2.5 Hz at 0.75 m/s → 0.4 s cycle) |
| W_TORQUE | 2e-7 | 2e-4 | equal penalty at full saturation: Σcap² = 172600 (tiers 180/140/70) vs duck 146 |
| AIR_TIME_MIN/MAX | 0.10 / 0.50 s | 0.08 / 0.40 | swing ≈ 0.16–0.30 s at the slower cadence, window widened proportionally |
| PLACEMENT_MIN_M | 0.15 | 0.030 | leg-ratio (×5) scaled minimum step; consistent with the 3.33 clock |
| STANCE_MIN_S | 0.12 | 0.06 | stance ≈ 60% of a 0.4 s cycle (≈0.24 s); floor at half, duck's proportion |
| CLEARANCE_M | 0.030 | 0.010 | human-scale swing-foot clearance (3–5 cm) |
| W_IMIT | **0.0** | 0.5 | NO humanoid reference gait exists — empty 8a hook, REF_GAIT=None; header pins an all-zero DW_REF_GAIT[64][12] placeholder |
| all others | duck values | — | dimensionless gait-structure terms (alive, lateral, action-rate, chatter/flicker/tick math, double-support, alternate/same-foot/phase): tick layout identical, keep duck-proven weights |

## 5. Header pinning + drift

`generate_model_humanoid.py` now emits, in addition to the physics tables:
`DW_PHASE_HZ_*` (from humanoid_flat, env-var sweepable at generation like
the duck), all `DW_RW_*` (from humanoid_reward), `DW_IMIT_*` + zero
`DW_REF_GAIT`, and the env contract block `DW_ENV_{OBS, ACT,
TICKS_PER_STEP, CONTROL_DT, ACTION_SCALE, MAX_TARGET_INCREMENT,
QDOT_OBS_SCALE, HORIZON_STEPS, MIN_HEIGHT_FRACTION, MAX_TILT_RAD,
COMMANDS_MPS[3]}`. Exact-repr float equality between the env/reward
modules and the committed header is enforced by
`test_humanoid_env.py::test_header_pins_bit_parity`, and full-text drift by
`test_humanoid_serial_parity.py::test_humanoid_header_drift`.

## 6. Kernel policy path + training adapter (LANDED)

The kernel edit landed upstream (commit b063a3b): the device policy layer
is robot-generic via the `DW_ENV_*` contract block (`DWP_OBS = DW_ENV_OBS`,
J-derived obs offsets, `DW_ENV_UP_AXIS` termination, per-joint
`DW_EFFORT_CAP_TABLE`, zero-cost empty `DW_REF_GAIT` at `W_IMIT 0`), so
`dwc1_step_policy` / `dwc1_observe` / `dwc1_reset_policy` are valid for the
humanoid build. On top of that:

- `walk/env/humanoid_cuda_lane.py` gained the policy-path wrappers
  (`step_policy` / `observe` / `set_command` / `reset_policy` with
  humanoid_flat's exact counter-based command+phase stream) and
  `effort_cap_per_joint` (authored tiers; `effort_cap` stays the
  `dwc1_info` scalar summary).
- `walk/train/gpu_train.py` is parameterized by `--robot {duck,humanoid}`
  through one `robot_classes()` indirection (obs/act dims + lane/env
  classes). The duck path is behavior-identical (same classes, same
  constructor calls, same RNG streams; fingerprint-verified byte-identical
  metrics + actor weights on a 2-env lane-env smoke). `--accept-every`
  (duck strict evaluator) and `--randomization` (duck DR surface) are
  rejected for `--robot humanoid`.
- Parity gate: `FlatFloorHumanoidEnv` over the fp32 lane vs
  `dwc1_step_policy`, same seed stream + actions, 40 steps: **obs
  bit-identical (0.0)**, reward ≤ 1.9e-4 (f32-vs-f64 reward summation),
  done flags identical
  (`test_humanoid_gpu_train.py::EnvVsKernelPolicyParity`).
- Envelope note: the fp32 solve (5e-6 / 4096) has less stall headroom than
  the f64 oracle (1e-8 / 16384): sustained ±0.3-flail while standing can
  still fault fp32 where f64 survives; the in-kernel path freezes+finishes
  such envs (counted, not raised), which is the intended training-time
  behavior.

CPU-lane training also works via `walk/train/run.py` unmodified (OBS/ACT
read off the env class). GPU (Daytona) bring-up is the orchestrator's leg.

## 8. Frozen walking judge (walk/eval/humanoid_gait.py + humanoid_acceptance.py)

`walk/eval/humanoid_gait.py` is the FROZEN strict evaluator — a
clause-for-clause port of the duck judge (walk/eval/gait.py) with
morphology-scaled thresholds; the full threshold table with per-line
justifications lives in its module docstring (anchors: x5 leg ratio, x4.3
sole length, matched 0.6 s slowest-command cycle by clock construction;
the 20 ms contact-debounce AMENDMENT is carried verbatim). Tilt is
recomputed from `base_quat_xyzw` with the humanoid up axis; the shared
capture (walk/eval/capture.py, untouched) records quats, so its duck-frame
tilt column is simply ignored.

`walk/eval/humanoid_acceptance.py` mirrors the duck multi-seed harness:
same 4 seeds (4242/7/1913/90210) x commands 0.50/0.75/1.00 m/s, episodes
on FlatFloorHumanoidEnv over the CPU-serial fp32 lane (`--lane native`
swaps in the f64 oracle), `unpack_actor_file` for 52->12 actors.

Synthetic gates (humanoid/tests/test_humanoid_gait_eval.py): a
hand-authored clean gait PASSES all criteria; a flat shuffle fails ONLY
the clearance clause; a one-leg hop fails per-foot balance + alternation;
stand-still fails footfall counts + translation + step deadlines; 6 ms
contact dropouts are debounced away; terminated/short traces are rejected.

Leg-1 actor verdict (runs/gpu/20260901-200306-humanoid-train-ff,
4 seeds x 3 commands = 0/12, remarkably uniform): **zero swings examined
in every episode** — the feet never leave the ground; the policy survives
8 s by leaning ~32.5-34.7 deg (over the judge's 30 deg, under the env's
45 deg termination) and sliding ~0.43 m regardless of command (ratio
0.054-0.108 of commanded translation). This is exactly the duck's
documented pre-gait-shaping "lunge with feet never leaving the ground"
attractor (walk/env/reward.py docstring): the reward's next need is
making stepping pay vs the standing subsidy — addressed by reward v2 (§9).

## 9. Reward v2 — breaking the lunge-and-slide attractor

Weight-only rebalance (the term SHAPE stays the duck-v12 structure the
robot-generic kernel implements, so env↔kernel bit-parity via the DW_RW_*
header pins is preserved; a NEW term would need its kernel twin first):

| constant | v1 | v2 | rationale |
|---|---|---|---|
| TRACK_SIGMA_SQ | 0.25 | **0.09** | σ 0.5→0.3: v1 paid the measured 0.054 m/s slide-creep up to 0.45 track at cmd 0.5 (standing subsidy); at σ 0.3 the creep earns ≤ 0.11 while a stepper within 0.3 m/s of command still earns ≥ 0.37 |
| W_AIR_TIME | 1.5 | **3.0** | qualified steps are rarer on the biped (six gates at 5× scale); keeps the per-second step-bonus ceiling comparable to the duck's |
| W_DOUBLE_SUPPORT | 0.5 | **1.5** | the only term active in a permanent commanded stand must outweigh its income (alive 0.5 + creep track ≤ 0.45) |
| W_PHASE | 0.5 | **1.0** | doubles the in-phase stepping differential (±2/step); the duck's binding anti-limp force at the humanoid's larger standing subsidy |
| env MAX_TILT | 45° | **28°** | leg-1 survived 8 s at a 32.5–34.7° lean — past the FROZEN judge's 30°; termination now sits just inside the judge with 2° measurement margin (walking pitch ≤ ~15°, no legitimate gait clipped); header DW_ENV_MAX_TILT_RAD/COS_MAX_TILT regenerated |

Measured stand-vs-step gap (authored trajectories through the real
reward(), pinned forever by humanoid/tests/test_humanoid_reward_v2.py):
perfect stand+lean per step −0.85 / −0.95 / −0.95 at cmd 0.5/0.75/1.0
(v1: +0.28..+0.45 — the attractor's income); crude half-speed in-phase
stepper +2.04 / +1.75 / +1.61 (gap ≥ 2.6/step); the judge-passing clean
gait scores +2.81/step vs the stand's −0.85 at cmd 0.5, tying the reward's
preference to the frozen judge's verdict. Standing a whole episode now
loses to falling at t=0.

Known residual loophole (documented in humanoid_reward.py, not yet
observed): a permanent ONE-legged stand nets ~+0.6/step (no double-support
penalty; the signed phase term nets 0). If it emerges, the counter is a
no-swing term (continuous stance OR air beyond ~2 slowest cycles at
|cmd|>0) — a shape change that must land in the kernel first.

Fresh GPU legs must start from FRESH inits: the leg-1/leg-chain policies
were optimized for the v1 landscape (lunge) and would fight v2.

## 10. Reward v2.1 — synthetic reference gait + live imitation

The idle DW_REF_GAIT infrastructure now carries a SYNTHETIC analytic
reference cycle (no authored H0 gait exists; the duck's own table is
likewise self-generated): `humanoid/author_reference_gait.py` →
`humanoid/reference_gait.json` (64 bins × 12 joints, byte-reproducible),
loaded by `walk/env/humanoid_reward.py` (term 8a live, duck's exact
formula) and pinned into the header's `DW_REF_GAIT` (identical f64 values;
the kernel imitation path is the duck-tested one — env↔kernel obs parity
re-measured bit-identical, reward ≤ 1.9e-4).

Design (planar model → analytic; FK sign conventions probe-verified):
left stances while sin(phase) ≥ 0 (the reward's clock convention), hip =
0.0873·cos(p) (= asin(0.075/0.86): the no-slip amplitude for the clock's
command-independent 0.300 m stride = 1/PHASE_HZ_PER_MPS; anything larger
over-strides the clock into slip/scuff), knee = 0.6·min(1, 2sin²p) on the
swing side only (plateau bell: full flexion across the certified
|sin| ≥ 0.707 half of the swing; a plain sin² bell sagged to 13 mm at the
window edges), ankle = −(hip+knee) clipped to ±0.65 (foot-flat; ≤ 0.01 rad
pitch at the clip), waist/neck/arms at HOME.

Validation BEFORE training on it (`humanoid/tests/test_reference_gait.py`,
real-fixture FK per frame over all 64 bins): stance sole |z| ≤ 4 mm and
bottom-face flat; swing sole never below floor; ≥ 30 mm whole-sole
clearance across the certified window (peak 56 mm); stance sweep 0.150 m
per half cycle → per-swing world placement 0.30 m (judge band, 2× floor);
all joints inside authored limits; L/R mirror; json + header drift-locked.
Open-loop PD replay on the f64 CPU lane at the cmd-0.75 clock: fault-free,
both feet lift inside their swing windows (not a stability claim).

W_IMIT = 0.5 (duck's value, re-justified): max bonus 0.5 = half of
W_TRACK's 1.0, a quarter of the ±2.0 phase differential — guide, not
rail; σ² = 0.04 (rms 0.2 rad) puts a stander ~1.5σ out at knee-active
bins. The stander's imitation leak (reference passes near HOME twice per
cycle) is +0.24/step mean and does NOT flip v2's property:

| per step, measured (pinned) | cmd 0.50 | cmd 0.75 | cmd 1.00 |
|---|---|---|---|
| stand+lean v2.1 (leak incl.) | −0.61 | −0.71 | −0.72 |
| imitating in-phase stepper v2.1 | +2.54 | +2.25 | +2.11 |
| gap (v2 was 2.89/2.70/2.56) | **3.15** | **2.96** | **2.83** |

Judge-passing clean gait: +3.05/step (was +2.81). This lands as v2.1 for
the GPU leg after the running v2 leg.

## 11. BC pre-training (bc_init.pt): GPU legs start "already stepping"

`walk/train/bc_pretrain.py` (robot-parameterized via
`gpu_train.robot_classes`; the duck path is untouched — no duck dataset
provider exists and asking for one is an explicit error) regresses a
fresh ff Actor (52→256→256→12) onto `humanoid/bc_dataset.py`: the
reference gait rolled CLOSED-LOOP over real fp32-serial-lane observations
(the exact physics + obs distribution of the GPU legs; ~500× faster than
the f64 oracle on contact-heavy stepping).

Demonstrator labels, computed from the obs alone (no privileged state):
lead-2 reference lookup (compensates the PD+slew lag; raw next-phase
labels produced a non-stepping clone — measured) + an ankle balance
assist −(2.0·obs[36] + 0.1·obs[41]) on both ankles (gains swept for max
alternating lifts). Knee-plateau actions saturate at 1.0 by design
(action box = ±0.5 vs reference knee 0.6); dataset and deployment stay
consistent because the clipped actions themselves drive the recording
env, and the imitation-reward shortfall from the unreachable 0.1 rad is
< 2% of the bonus.

Dataset (default config): 8 seeds × 3 commands × 4 envs × ≤60 steps =
3584 pairs (envs die early — see below), contact channels live, ±0.02 rad
joint-offset noise. BC: MSE(tanh(mu), labels clamped ±0.98), Adam 1e-3,
300 epochs ≈ 7 s CPU, loss 0.0588 → 0.000197. Saved with log_std = −1.0:
at ppo.py's −0.5 the exploration noise (±0.3 rad targets ≈ half the
action box) knocks the fresh gait over; at −2.0 PPO is polish-cold; −1.0
perturbs targets by ≈ the env slew step (one-step recoverable).

Closed-loop replay of the committed bc_init.pt (deterministic, pinned by
humanoid/tests/test_bc_pretrain.py): 3 commands × 3 seeds → 18 lifts,
6 alternations, best sequence LRLR (cmd 1.0), every episode ≥ 0.72 s.
HARD CEILING, morphology not training: all 12 joint axes are sagittal —
zero roll authority — and single support is laterally statically unstable
by ~1 cm (stance-foot inner edge vs CoM), so every non-recovery policy
tips sideways within ~1 s. The BC init gives PPO stepping + alternation
+ the ankle reflex from step one; lateral survival strategies (stepping
cadence as a stabilizer) are PPO's job.

Orchestrator command line (deterministic; both artifacts drift-locked /
handoff-tested by humanoid/tests/test_bc_pretrain.py):
    .venv/bin/python -B -m walk.train.bc_pretrain --robot humanoid \
        --out humanoid/bc_init.pt --checkpoint-out humanoid/bc_init_ckpt.pt \
        --epochs 300 --seeds 11,22,33,44,55,66,77,88
(H0-era paragraph; superseded by §12's H1 command lines.)
Flagship leg = v2.1 reward + 16384 envs, started from the cloned gait via
the turnkey resume path (verified: resumes at update 0, first PPO update
clean):
    python -B -m walk.train.gpu_train --robot humanoid --lane-env \
        --resume humanoid/bc_init_ckpt.pt --envs 16384 --device cuda \
        --library <libduck_cuda.so built with -Ihumanoid/include> ...
(bc_init.pt stays the actor-only artifact for evaluators/acceptance;
bc_init_ckpt.pt is the same actor plus fresh critic/optimizer/generators
in gpu_train's checkpoint schema.)

## 7. Smoke results (measured)

`test_humanoid_train_smoke.py`: PPO through the unmodified
walk/train/run.py + VecEnv machinery, 1 worker, 2 envs, horizon 8,
2 updates, preflight skipped (see test docstring — uniform-flail preflight
keeps ticking fallen done-envs, which is wall-time prohibitive; VecEnv
resets them immediately):
- 2/2 updates completed, **faults 0, poisoned_envs 0** in every row,
  faults.jsonl empty
- reward finite and non-constant (u1 mean −0.45 / std 1.38, u2 mean −0.68
  / std 1.28), PPO losses finite, 16 transitions per update
- wall ≈ 125 s for 32 env-steps at 2 envs on a shared box (untrained
  policy spends most steps falling; fall states are the expensive solves —
  see FEASIBILITY.md §6 note). Throughput is a non-goal for the CPU smoke;
  batch training waits on the kernel policy-layer edit + orchestrator GPU.

## 12. H1: hip-roll lateral actuation (B16/J14) — the morphology fix

Empirical driver: the H0 flagship leg (BC + v2.1, 16384 envs) stepped
immediately (reward ~2.0) but ep_len saturated at ~69 steps (~1.4 s — the
predicted lateral tipping timescale) and the judge rejected 0/12 on early
termination. All 12 H0 joint axes are sagittal; no policy can regulate
roll on that morphology.

H1 (`humanoid/h1_lowering.py`, now the ACTIVE lowering; H0 stays
importable and archived under its own tests): adds exactly TWO hip-roll
joints. Rationale: the authored fixture is strictly planar (NO roll
joints exist anywhere in humanoid.rs, so any lateral joint is our
authorship); hip roll is the primary frontal-plane stabilizer with full
lateral CoM authority; ankle roll would double the invented surface with
no authored tier to join (minimal-change). Rolls join the AUTHORED hip
tier (180 N·m, kp 90/kv 8), limits ±0.4 rad, axis = parent-local
[1,0,0] (forward; FK-probed sign). The stack is body-per-joint, so each
roll carries a small inserted hip-link body (0.5 kg, 0.06 m cube, both
hip joints coincident at the H0 hip point → home FK is IDENTICAL to H0
for every H0 body); MASS-NEUTRAL: link mass carved from the thigh
(7.0 → 6.5 kg), total stays 68.0 kg. Everything else inherited from
h0_lowering by import, not copy.

Pure-regeneration proof: reference table + header regenerated (B16/J14,
OBS/ACT 58/14 — coincidentally the duck's exact DW_J/N/Q sizes);
`shasum -c` over duck_cuda_kernel.h / duck_cuda_serial.cpp / duck_cuda.h /
cuda_compat.h before vs after: ALL OK — zero kernel edits, the
robot-generic pipeline claim held. Env↔kernel policy parity on H1:
obs AND reward bit-identical (0.0/0.0) over the parity window. Oracle
home-hold on H1: 1000/1000 ticks, momentum ≤ 2.4e-15, tilt 0.

Reference gait v2 (H1): roll columns roll_L = roll_R =
0.25·sin(p + 0.4). The 0.25 rad is a LOAD-COMPENSATION target: single
support loads the stance hip-roll with ~166 N·m (CoM 0.15 m inboard of
the hip — H0's unfixable torque), so the PD sags to ~0.1 rad of achieved
lean ≈ 0.09 m of CoM shift; kinematic-margin sizing (0.05 rad) measured
NO better than zero roll. Phase advance 0.4 rad establishes the lean
BEFORE the swing foot lifts; the sinusoid still crosses zero near the
transfers. FK gates: sagittal columns validated with rolls zeroed
(unchanged numbers: stance |z| ≤ 3.3 mm, clearance ≥ 30 mm certified
window / 56 mm peak, sweep 0.150 m, limits OK) + dedicated roll
amplitude/timing/limit checks; open-loop physics replay still lifts both
feet in-window, fault-free.

BC redo on H1 (`humanoid/bc_dataset.py` demonstrator, all channel indices
J-derived): lead-2 reference + swept assists — ankle pitch −3.0·grav_x,
hip pitch share 1.2× (ankles alone saturate their 140 N·m cap during
recovery), hip roll +5.0·grav_lat − 0.3·roll_rate (the authority H0
lacked). Dataset 4590 pairs (envs live longer); BC MSE 0.0796 → 0.000207.
Closed-loop clone (3 commands × 3 seeds, pinned):

| cmd | survivals (s, seeds 4242/7/1913) | lifts/alt |
|---|---|---|
| 0.50 | 1.04 / 0.90 / 1.02 | 8 / 2 |
| 0.75 | 0.98 / 0.88 / 0.96 | 10 / 2 |
| 1.00 | 1.04 / 0.94 / 1.04 | 5 / 2 |

**Mean survival 0.978 s vs the H0 clone's ~0.76 s ceiling (+29%)**, every
episode ≥ 0.88 s. The remaining fall is now a mixed recovery problem with
actuation available on every axis — PPO's job, no longer a morphology
dead end. Orchestrator command lines are unchanged (§11); artifacts
bc_init.pt / bc_init_ckpt.pt regenerated for 58/14 and drift-locked.

## 13. Tech-tree curriculum controller (walk/train/curriculum_controller.py)

Training as a tech tree: one run senses which walking requirements are met
(the gate_proxy judge-shadow metrics, commit 65d99b8) and advances its own
difficulty through the runtime gate-termination knobs + command
distribution, so all compute goes to the current frontier node.

Controller design (auditable by construction): the ladder is DATA (Stage
dataclass rows; builtin `humanoid-walk` or a JSON file); it reads ONLY the
per-update metrics gpu_train already writes; acts ONLY at update
boundaries through ONLY the two public lane knobs
(`set_gate_termination`, per-reset command-pool override). Advance =
median of the stage's metric over its trailing window ≥ threshold, with a
full-window minimum dwell (a single lucky update cannot skip a node;
windows reset on every transition). De-escalation (the fail-back-down
edge): median ep_len below the stage floor (25 steps = 0.5 s, under even
the BC bootstrap ceiling) for 5 CONSECUTIVE updates steps back exactly one
stage — enforcement that kills the population leaves no gradient. Never
past `full`, never below the first stage. Every transition logged to
metrics.jsonl ({"kind":"curriculum", event/direction/name/from/update/
reason}) and stdout; every train row carries `curriculum_stage`.

The humanoid-walk ladder (thresholds anchored to the FROZEN judge):

| stage | knobs (deadline ticks / alt cap) | commands | advance when (median over 8 updates) |
|---|---|---|---|
| free | 0 / 0 | 0.75 | qualified_total ≥ 0.5 — real steps exist (gate_proxy shadows duration+clearance+placement) |
| swings_appear | 0 / 0 (log only) | 0.75 | ≥ 2 — one qualified swing per foot, the minimal alternating unit |
| deadline_loose | 2500 (5 s) / 0 | 0.5, 0.75 | ≥ 3 — 2× the judge's FIRST_STEP_S prunes never-steppers gently |
| deadline_judge | 1250 (2.5 s = judge) / 0 | all three | ≥ 4 — judge-exact deadline; judge scores all commands |
| alternation_cap | 1250 / 3 | all three | ≥ 6 = judge MIN_FOOTFALLS — cap 3 kills persistent limps, tolerates learning doubles |
| full | 1250 / 1 | all three | terminal — judge-tight; surviving 8 s here ≈ a judge-passing episode (frozen CPU judge stays the only authority) |

gpu_train hooks: `--curriculum <name|ladder.json>` (default off = exact
legacy behavior; duck fingerprint re-proven byte-identical:
22c175f3cfb8… pre == post). Guards: no duck ladder exists (explicit
error), requires `--lane-env`, requires a lane exposing gate_proxy +
set_gate_termination. The command override rides
`reset_policy(commands=…)`; `command_override=None` leaves the duck/legacy
reset call untouched.

Next GPU leg:
    python -B -m walk.train.gpu_train --robot humanoid --lane-env \
        --curriculum humanoid-walk --resume humanoid/bc_init_ckpt.pt \
        --envs 16384 --device cuda --library <libduck_cuda.so with -Ihumanoid/include> ...

## 14. Swing diagnostics: no real swing exists — the transfer never completes

Instrumentation: humanoid/diagnose_swings.py (288 episodes: demonstrator /
BC clone / tree-leg actor x 3 commands x 4 seeds x 8 envs on the fp32
serial lane; every liftoff->touchdown recorded raw AND with the judge's
20 ms debounce; three-way check vs gate_proxy and the frozen judge).

Findings (all three policies identical in kind):
- 1065–1853 raw "swings"/policy, EVERY one a single-tick (2 ms) contact
  flicker: duration median 0.002 s (bar 0.1), peak clearance ≤ 0.03 mm
  (bar 30 mm), placement ~±0.3 mm (bar 150 mm); first-fail = duration for
  100%. After the 20 ms debounce the swing count is exactly ZERO.
- Three-way consistency is PERFECT: analyzer-raw qualified 0 ==
  gate_proxy 0 == analyzer-debounced 0; zero per-episode mismatches.
  The proxy is not the problem; there is no marginal-swing gap.
- CORRECTION of earlier reporting: the BC-leg "lifts/alternations" gates
  sampled foot_contact at 20 ms policy boundaries and were counting these
  single-tick solver flickers, not real swings. The survival-time gains
  were real; the stepping claims were not.
- Mechanics (probe, demonstrator at cmd 0.5): contact_ticks 10/10 on BOTH
  feet at every policy step — permanent double support. Roll channel:
  commanded ±0.25 rad, achieved ~±0.15 with ~quarter-cycle lag —
  textbook over-bandwidth drive: the roll plant's natural frequency is
  sqrt(kp/I_eff) = sqrt(90/1.74) ≈ 1.15 Hz (ζ ≈ 0.32), while the clock
  demands the weight shift at the cycle rate = 1.67/2.5/3.33 Hz at
  cmd 0.5/0.75/1.0 — at or above bandwidth at every command. The CoM
  never gets over a foot in time; statics compound it (holding single
  support at zero lean needs ~204 N·m about the stance hip-roll vs the
  180 N·m authored cap, so transfer MUST complete during double support —
  and the v2 reference has zero double-support time by construction:
  alternating single support with swings starting exactly at transfer).

RECOMMENDATION (single highest-leverage fix; propose-only per the task):
reference/clock v3 — author an EXECUTION-FEASIBLE weight-shift gait:
  (i) slow the gait clock: PHASE_HZ_PER_MPS 3.33 -> 1.67 (stride
      1/PER_MPS 0.30 -> 0.60 m; step 0.30 m, still 2x the judge's 0.15 m
      placement bar; weight-shift frequency drops to 0.83–1.67 Hz,
      at-or-below the roll plant's 1.15 Hz bandwidth for cmd <= 0.75);
  (ii) restructure the table with an explicit double-support transfer
      phase (~25–30% of cycle) in which the roll completes the shift
      BEFORE the swing window opens (swing ~35% of cycle: 0.28–0.42 s,
      inside the judge's [0.1, 1.2] s);
  (iii) new validation gate BEFORE anything trains on it: the
      demonstrator's EXECUTED closed-loop rollout must produce >= 1
      debounced qualified swing per episode (the FK gates provably do not
      catch execution-infeasibility — this leg's lesson).
Why not the alternatives: amplifying amplitudes cannot work (roll targets
already saturate the action box and the plant attenuates above bandwidth);
near-miss reward shaping and 50%-bar pre-stages both assume a population
NEAR the bars — it is at literally zero swings with the mechanics blocked,
so there is no gradient path for them to pay. The frozen judge is
untouched by (i)+(ii): the clock constant is the env's explicitly
sweepable knob and placement margin GROWS. Fallback if executed
validation still fails at 0.83 Hz: per-joint kp (roll ~400 N·m/rad) — an
H1.1 authoring decision + a kernel DW_KP_TABLE edit, precedented exactly
by the effort-cap table.

## 15. v3 implemented — executed validation FAILS: STOP, H1.1 kp-table required

Implemented exactly as approved: clock PHASE_HZ_PER_MPS 3.33 → 1.67
(bandwidth-pinned comment in humanoid_flat.py; do-not-move without an
executed-validation run), reference table v3 (30% double-support transfer,
swings 0.21–0.42 s, roll trapezoid completing the shift BEFORE liftoff,
plus two execution fixes found while validating: swing-hip lift bump
HIP_LIFT 0.3 — the executed lean drops the pelvis by L·(1−cosθ) and a
box-capped knee cannot out-shorten it — and the whole table clipped into
the ±0.5 ACTION BOX so imitation targets are reachable), header
regenerated, BC chain regenerated, lift-counting gates replaced by the
debounced analyzer everywhere (the flicker lesson institutionalized), and
the MANDATORY ExecutedValidation gate added.

FK numbers (roll-zeroed sagittal, pinned): planted-sole |z| ≤ 22.3 mm
(pinned-pelvis stance-extreme bob at ALPHA 0.228), swing clearance ≥ 44 mm
over the knee-plateau window (peak 123 mm), stance sweep 0.388 m =
2L·sin(ALPHA), stride 0.599 m (4× the judge's placement bar).

EXECUTED validation: FAILS — and the mechanism is now fully measured:
- The body above the hip-roll joint is a LATERAL INVERTED PENDULUM with
  destabilizing stiffness g·Σmᵢhᵢ ≈ 388 N·m/rad (56 kg minus the
  below-pivot hanging leg), vs joint kp = 90 N·m/rad. Any lean beyond
  ~kp·headroom is a runaway: measured, every hold target from 0.18 (the
  static balance point) to 0.40 ramps the achieved roll monotonically
  past 0.38 rad into the 28° termination, with the PD pulling back the
  whole time.
- Saturation bound (policy-independent): the action box caps restoring
  torque at kp·(0.5−θ) ≤ 45 N·m of headroom, which stabilizes leans only
  below 388·θ = 45 → θ ≈ 0.116 rad; unloading a foot REQUIRES the CoM
  over the stance hip ≈ 0.175 rad. Required lean > maximum stabilizable
  lean: quasi-static single support is impossible on this plant for ANY
  controller acting through PD position targets — not a reference-design,
  clock, BC or reward problem.
- Confirmed downstream: v3 demonstrator and BC clone: 0 debounced
  qualified swings in every configuration (amplitudes 0.18–0.40, assists
  swept, aligned and random phase starts); clone survival 0.72–0.84 s.

STATE LEFT: all suites green with the two stepping gates present but
@skip-BLOCKED (ExecutedValidation in test_reference_gait.py and
test_closed_loop_replay_steps in test_bc_pretrain.py), each skip message
carrying the blocker analysis; survival gates active; drift/parity/FK
gates all green; v3 artifacts committed (a strictly better scaffold the
moment the plant is fixed).

H1.1 REQUIREMENT (orchestrator + kernel specialist; NOT implemented here
per instruction): per-joint gain tables in the kernel — DW_KP_TABLE /
DW_KV_TABLE consumed like the existing DW_EFFORT_CAP_TABLE — with
hip-roll kp ≳ 500 N·m/rad (≥ 388 + margin; also lifts the roll bandwidth
to √(500/1.74) ≈ 2.7 Hz) and kv ≈ 60–100 (ζ ≈ 0.3–0.5 against the ~20
kg·m² pendulum inertia). The lowering, generator, env and oracle already
carry per-joint values natively (av1 Hinge.kp/kv are per-joint); only the
kernel consumes scalars. When it lands: set the two gains in
h1_lowering, regenerate, UNSKIP the two gates, rerun the executed sweep.
