# walk/env — batched Open Duck environments (workstream C)

`FlatFloorDuckEnv` (walk/env/flat.py) implements the `DuckEnvBatch` contract
(walk/env/contract.py) over the native idv1 flat-floor lane. All native-lane
specifics live in one adapter module, walk/env/native_lane.py; the class is
backend-agnostic otherwise.

## Construction

```python
from walk.env.flat import FlatFloorDuckEnv
env = FlatFloorDuckEnv(environments=16, seed=0)          # builds/caches the dylib
obs = env.reset()                                        # [E, 58] float32
obs, reward, done, info = env.step(action)               # action [E, 14] in [-1, 1]
```

- The combined native library is compiled with the exact clang++ command from
  `experimental/integrated_duck_v1/run_local.py` and cached as
  `build/libintegrated_duck-<source-hash>.dylib` (rebuilt only when the native
  sources change).
- One policy step = 10 native ticks x 0.002 s = 0.02 s. Target semantics per
  the pinned plain-14 candidate: `requested = HOME + 0.25 * action`, slew
  limited to 0.1048 rad per policy step against the stored previous targets,
  then joint-limit clipped; the native lane applies PD per 2 ms tick against
  the current q/qdot with kp = 13.37, kv = 0, effort cap 3.23 N m (identical
  to run_home_hold.py).
- `reset(mask, seed)` performs a masked native restore to the initial
  snapshot. `seed` (when changed) rebuilds the lane so the snapshot embeds
  fresh deterministic per-env joint perturbations.
- `perturbation_rad` (constructor, default **0.0**, bound 0.02): deterministic
  per-env uniform joint-q perturbations drawn from the seed. Solver repairs
  are included; tests cover a formerly failing perturbed reset and a separate
  injected fault. This is not a universal convergence claim.
- Solver faults: if `idv1_step` rejects a tick, the failing env's complete
  state (idv1_read fields incl. both manifold sets), the step inputs and all
  diagnostics are persisted to `runs/faults/<timestamp>-env<i>.json` and
  `SolverFault(env_index, saved_problem_path)` is raised. Post-fault
  observations are never returned. Envs are never auto-reset: `done[i]`
  stays terminal (frozen targets, reward 0) until `reset(mask)`.

### Swapping in the cube-grid backend (workstream B)

Pass `lane_factory=lambda E, joint_offsets: MyWorldLane(...)` to the
constructor. A lane must provide the duck-typed surface documented at the top
of walk/env/native_lane.py (`tick`, `read` -> `LaneState`, `restore`,
`state_dump`, `close`, plus `E/J/joint_limits/home_joint_q/home_root_height/
kp/kv/effort_cap`). Nothing else in flat.py touches the backend.

## Observation layout (OBS = 58, float32)

| index | width | content |
|-------|-------|---------|
| 0:14  | 14 | joint q minus HOME pose (rad), canonical plain-14 order |
| 14:28 | 14 | joint qdot x 0.05 (matches the pinned candidate scaling) |
| 28:42 | 14 | previous action (zeros right after reset) |
| 42:45 | 3  | gravity direction in body frame (unit; [0,0,-1] when upright) |
| 45:48 | 3  | root angular velocity, body frame (rad/s) |
| 48:51 | 3  | root linear velocity, body frame (m/s) |
| 51:54 | 3  | commanded velocity [vx, vy=0, wyaw=0]; vx sampled per episode from {0.10, 0.15, 0.20} m/s (deterministic per seed/env/episode) |
| 54:56 | 2  | foot contact flags (left, right) from the foot-vs-floor solve-cache manifolds |
| 56:58 | 2  | phase clock: sin/cos of a fixed 2.5 Hz clock over episode time |

Body indices (verified against the pinned geometry goldens): left foot =
contact body 6, right foot = contact body 15; foot-vs-floor manifolds are
pairs 0 and 1. Whole-sole height = min world z over the 18 baked sole
vertices per foot; the floor is the z = 0 plane.

## Reward (walk/env/reward.py, evaluated per 0.02 s policy step)

| term | weight | detail |
|------|--------|--------|
| forward-velocity tracking | `W_TRACK = 1.0` | `exp(-(vx - cmd)^2 / 0.01)`, world-frame vx |
| alive bonus | `W_ALIVE = 0.5` | constant while not terminated |
| lateral/yaw penalty | `W_LATERAL = 0.5` | `-(vy^2 + wz^2)` |
| action-rate penalty | `W_ACTION_RATE = 0.01` | `-sum((a - a_prev)^2)` |
| torque penalty | `W_TORQUE = 2e-4` | `-sum(tau^2)`, tau = boundary PD estimate (clipped at 3.23) |
| air-time bonus | `W_AIR_TIME = 1.5` | qualified touchdown with duration, clearance, advance, support and phase checks |
| foot-clearance bonus | `W_CLEARANCE = 0.1` | per swing foot per step whose whole sole clears >= 10 mm |
| double-support penalty | `W_DOUBLE_SUPPORT = 0.5` | after > 0.25 s continuous double support while `cmd != 0` |
| alternation bonus | `W_ALTERNATE = 0.5` | qualified touchdown on the opposite foot to the previous one |
| same-foot repeat penalty | `W_SAME_FOOT = 2.0` | qualified touchdown repeats the previous foot |
| phase shaping | `W_PHASE = 0.5` | stance matches the observed 2.5 Hz clock |
| chatter / flicker penalties | `W_CHATTER = 0.2`, `W_FLICKER = 0.3` | short swings and tick-level contact flicker |

The v6 implementation in `reward.py` is authoritative. Velocity tracking uses
the rolling velocity estimate, not instantaneous speed. Reward alone is not
accepted walking; the separate gait evaluator still applies.

The gait terms exist because prior runs proved survival-only rewards produce
a lunge with feet never leaving the ground. `GaitTracker` keeps per-env air
time / double-support / last-footfall state and is reset with the env mask.

## Termination (no auto-reset)

- root height < 0.7 x HOME root height (0.7 x 0.16789 m = 0.1175 m), or
- tilt > 45 degrees (angle between body z and world z), or
- non-finite native state, or
- 8 s horizon (400 policy steps).

## Evaluation (walk/eval)

- `walk/eval/capture.py` runs a policy (`obs -> action`) and records a
  per-tick JSON trace (`duckgridwalk.episode/1`: base pose, tilt, per-foot
  COM position, whole-sole height, contact flags every 2 ms).
- `walk/eval/gait.py` is the STRICT acceptance evaluator per PLAN.md (three
  8 s episodes at +0.10/+0.15/+0.20 m/s); per-criterion pass/fail JSON;
  stance-slip bound documented there (25 mm per adjacent stance phase).

## Tests

```
.venv/bin/python -B -m unittest discover -s walk
```
