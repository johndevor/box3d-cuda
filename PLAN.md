# duck-grid-walk: Open Duck learns to walk on a grid of cube rigid bodies (CPU-first)

Goal: the Open Duck plain-14 biped learns to walk across a grid of cube rigid
bodies, trained fast via parallel PPO on this Mac (Apple M5 Pro, no CUDA).
Higher body/joint/contact limits than the upstream 32/16/16. CUDA port later.

## Stack (all local, committed at baseline)

- `experimental/contact_v1` — contact manifolds/transactions (bcv1/bcx1), f32 geometry.
- `experimental/articulated_v1/v2` — generalized-coordinate articulation (free
  root + up to 26 hinges), f64, PRE/stage/commit transactions.
- `experimental/integrated_duck_v1` — glue lane (idv1): per-env joint rows +
  contact rows -> civ1 dense solve -> av2_complete -> contact stage -> commit.
  `native.py` has ctypes bindings and `duck_scene(lib, environments=E)` which
  lowers the Open Duck (14 hinges, 16 bodies incl. fixed floor) natively.
- `coupled_impulse_v1.cpp` (civ1) — generic dense PGS solver: SPD mass, jacobian
  rows, box bounds, friction disks. Caps now 256 dofs / 1536 rows / 512 contacts.
- `duck_model/` — Open Duck MuJoCo XML + inertia scripts + plain-14 candidate
  constants (HOME pose, CONTROL_DT=.02, SIMULATION_DT=.002, ACTION_SCALE=.25).
- Gate runner: `.venv/bin/python -B experimental/integrated_duck_v1/run_local.py
  --output /abs/new-dir` must stay `passed-local-native-cpu` (14 jobs).

## Frozen task semantics (keep)

14 actions in [-1,1]; target = HOME + 0.25*action, clipped to joint limits,
slew-limited to 0.1048 rad per policy decision; 1 policy step = 10 native ticks
x 0.002 s = 0.02 s. PD per joint: kp=13.37, kv=0, effort cap 3.23, armature
.027, damping .56, friction loss .068 (see duck_model/scripts and hinge ref).

## Strict walking acceptance (evaluator must implement exactly)

Eight-second episodes at commanded +0.10/+0.15/+0.20 m/s, all three must pass:
>=6 alternating qualified footfalls, >=3 per foot; whole sole clears 10 mm for
>=20 ms during a 60 ms–1.2 s swing; forward placement >=30 mm; support 40 ms
before/after; opposite-foot support >=90% of swing; bounded stance slip;
commanded translation 60–150% of command x 8 s; tilt <=30 deg; no reset or
physical failure; first step within 2.5 s; final step within 1.5 s of end.
No reset concatenation; incomplete prefixes never qualify.

## Workstreams and file ownership (no agent runs git commit)

- **A: solver robustness** — owns `experimental/integrated_duck_v1/src/coupled_impulse_v1.cpp`
  (+ header if needed) and new test files. Known failure modes from prior runs:
  (1) training: two highly coupled normal rows (correlation ~0.99457), scalar
  sweep stalls ~1.05e-8 after 4096 sweeps; (2) evaluation: rank-5 six-row
  contact block with a self-stress direction, slow friction redistribution.
  Proven repair (reimplement; original patch inaccessible): a conditional
  two-normal nonnegative solve with tangents held fixed, plus one bounded
  feasible null-direction boundary move at sweep 256; verify nullness against
  the original jacobian and all response rows; apply unchanged certificates;
  unsafe/nonsingular/more-degenerate cases keep the ordinary path.
- **B: cube-grid world** — owns new `experimental/duck_world_v1/` only.
  One articulated duck (reuse av2) + M cube rigid bodies + static floor per env.
  Milestone 1: static cubes (fixed terrain grid, feet-vs-cube + feet-vs-floor
  manifolds). Milestone 2: dynamic cubes (6-dof free bodies, semi-implicit
  integration, box-box/box-floor manifolds, uniform-grid broadphase, contact
  islands, sleeping; islands solved via civ1 with block mass = duck N x N plus
  6x6 per cube). C API `dwv1_*` mirroring idv1 + ctypes wrapper `world.py`.
- **C: environment + evaluator** — owns new `walk/env/` and `walk/eval/`.
  Batched env per the contract below, flat-floor backend first (native.py
  duck_scene), cube-grid backend behind the same interface once B lands.
  Reward: velocity-command tracking + alive - action/torque penalties + gait
  shaping (foot clearance, air time, alternating single support) — prior runs
  proved survival-only rewards produce lunging with feet never leaving ground.
  Strict evaluator exactly as above, plus JSON episode capture for replay.
- **D: parallel trainer** — owns new `walk/train/`. torch (in `.venv`) CPU PPO,
  multiprocess env workers (shared-memory numpy), deterministic seeding,
  checkpoint/resume, JSONL metrics, stop-on-solver-fault preserving the exact
  failing tick inputs. Builds/tests against `walk/env/contract.py` stub.

## Batched env contract (C implements, D consumes)

```python
class DuckEnvBatch:                      # walk/env/contract.py defines the ABC + a StubEnv
    E: int                               # environments
    OBS: int                             # observation width (C documents layout)
    ACT: int = 14
    def reset(self, mask=None, seed=None) -> obs[E,OBS] float32
    def step(self, action[E,14] float32) -> (obs, reward[E] f32, done[E] bool, info dict)
    # info carries per-env solver diagnostics; a solver fault raises
    # SolverFault(env_index, saved_problem_path) after persisting inputs.
```

Observation must include: joint q/qdot (28), previous action (14), root
orientation (gravity vector in body frame, 3), root angular velocity (3), root
linear velocity (3), commanded velocity (3), foot contact flags (2), phase
clock (2). C may extend; document layout in `walk/env/README.md`.

## Rules

- Never weaken gate tests to pass; `run_local.py` gates green before handing back.
- f64 dynamics, momentum residual <=1e-8 stays enforced.
- No provider/GPU calls, no Doppler, no network beyond pip.
- Determinism: same seed -> same trajectory (single-threaded per env).
- Preserve failure artifacts; never train on partial/faulted rollouts.
