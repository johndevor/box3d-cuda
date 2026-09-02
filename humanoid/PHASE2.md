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
