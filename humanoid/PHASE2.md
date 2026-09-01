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
  termination) and `humanoid/tests/test_humanoid_train_smoke.py`
  (end-to-end CPU PPO smoke).

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

## 6. Still-pending kernel edits (unchanged from FEASIBILITY.md §2)

The in-kernel policy path (`dwc1_step_policy` / `dwc1_observe`) remains
duck-only: obs offsets (58/14-wide), `DWP_ACTION_SCALE 0.25`,
`DWP_MAX_TARGET_INCREMENT`, duck termination up-axis. All humanoid values
it needs are now pinned in the header (`DW_ENV_*`); the edit is mechanical.
GPU training goes through the orchestrator once that lands; CPU-lane
training uses `walk.env.humanoid_flat:FlatFloorHumanoidEnv` via
`walk/train/run.py` unmodified (OBS/ACT read off the env class).

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
