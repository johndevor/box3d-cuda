# duck-grid-walk — results

## What this is

An Open Duck plain-14 biped (14 actuated hinge joints, 16 rigid bodies
including the floor) learns to walk on a flat floor under strict,
pre-registered acceptance criteria, using a physics engine, contact solver,
environment, evaluator, and PPO trainer that were all written in this
repository. There is no MuJoCo, Isaac, or third-party physics in the
training loop — MuJoCo appears only once, as an external cross-check of the
finished policy. Training began on CPU (Apple M5 Pro, multiprocess) and
moved to a CUDA port of the same engine running on rented, ephemeral
RTX 5090 sandboxes. The task semantics are frozen in `PLAN.md`: 14 actions in
[-1, 1], targets = HOME + 0.25·action with a 0.1048 rad slew limit per
decision, PD control (kp = 13.37, kv = 0, 3.23 N·m effort cap), one policy
step = 10 physics ticks × 0.002 s = 0.02 s.

The accepted policy is a 58-input, 256×256-hidden, 14-output feed-forward
network with 84,508 parameters: `runs/actor-walking-v1.pt` (identical bytes
in `evidence/walking-accepted-20260901/actor-walking-v1.pt` and in
`runs/gpu/20260901-120136-continue-ff-short/artifacts/train/gpu-train-out/accepted/actor_accepted.pt`).

## Headline results

Every number below is read from a committed file; the path follows each
claim.

- **Strict walking acceptance passed in-run at update 6245**, after
  1,637,089,280 environment steps and 27,957.7 s (7.77 h) of accumulated GPU
  training wall clock. All six probe episodes (seeds 4242 and 7 × commands
  0.10 / 0.15 / 0.20 m/s) pass every criterion, and all three 11-second
  confirmation episodes on seed 4242 survive with their 8 s prefixes still
  passing. The relay had also passed once 5 updates earlier (update 6240,
  27,940.4 s).
  `runs/gpu/20260901-120136-continue-ff-short/artifacts/train/gpu-train-out/accepted/acceptance.json`,
  `runs/gpu/20260901-115435-continue-ff-short/artifacts/train/gpu-train-out/accepted/acceptance.json`
- **A wider 4-seed audit passes 10 of 12 episodes.** Re-running the strict
  evaluator on seeds 4242, 7, 1913, 90210 × three commands: seeds 4242, 7
  and 90210 pass everything; seed 1913 passes at 0.15 m/s but fails the
  footfall-alternation criterion at 0.10 and 0.20 m/s (25 and 51 qualified
  footfalls with a near-even left/right split, but at least one out-of-order
  step each). Under that file's all-seeds rule the record therefore says
  `"accepted": false` — the milestone claim rests on the 2-seed in-run
  protocol above, and this stricter audit is published as-is.
  `evidence/walking-accepted-20260901/acceptance.json` (byte-identical to
  `runs/acceptance-FINAL/acceptance.json`)
- **GPU physics matches the f64 CPU oracle.** First bring-up on an RTX 5090
  (32,607 MiB, driver 580.76.05, nvcc 12.8.93): windowed 100-tick max
  joint-position divergence 4.52e-4 (bound 1e-3), velocity divergence 0.106
  (bound 0.25), max ground penetration 1.55e-4 m (bound 5e-3), all values
  finite, bit-identical determinism across repeat runs — 64 envs, 800
  ticks, all gates pass. The same gates pass with the same values at the
  release commit. `evidence/gpu-bringup-20260831/gpu-parity.json`,
  `evidence/gpu-bringup-20260831/gpu-info.txt`,
  `runs/gpu/20260901-141100-parity-bench-duck-cuda/artifacts/gpu-parity/gpu-parity.json`
- **Raw physics throughput: 826,835 ticks/s at bring-up, 5,311,784 ticks/s
  at the release commit** (both best-of-run at 8192 environments, fast
  build, 1000-tick benchmark; the 1M ticks/s target flag flips from `false`
  to `true` between the two files). The two runs are not the same kernel:
  in between came the warp-per-env cooperative kernel, a
  `launch_bounds` occupancy fix, and batched benchmark launches (commits
  `66783a8`, `ef22c6f`, `c30e7a5`).
  `evidence/gpu-bringup-20260831/gpu-bench.json`,
  `runs/gpu/20260901-141100-parity-bench-duck-cuda/artifacts/gpu-bench/gpu-bench.json`
- **End-to-end training throughput reached 140k+ env-steps/s.** In the
  preserved metrics of leg 6 of the day's first GPU relay (warm-started
  from the CPU checkpoint; 247 PPO updates, 4096 envs × 64-step horizon),
  per-update rollout throughput climbs from 37,786 env-steps/s at the head
  of the leg to a 120k–153k band late in the leg, peaking at 153,025 and
  logging 140,430 env-steps/s (≈1.40M physics ticks/s, since each env-step
  is 10 ticks) on the final update; the median over the whole leg including
  the ramp is ≈100k. For scale, the CPU trainer ran at ≈6,000–6,200
  env-steps/s (192 envs × 12 workers).
  `evidence/gpu-training-relay/leg6-metrics.jsonl`,
  `runs/flat-002/metrics.jsonl`, `runs/flat-003/metrics.jsonl`
- **The accepted lineage is one continuous run executed as 52 sandbox
  legs, most of them 13 minutes long** (fresh-ff → continue-ff →
  continue-ff-short, checkpoint-resumed across ephemeral sandboxes, with
  domain randomization and command latency on from the start), reaching
  update 7298 and 1,913,126,912 environment steps by the end of the day.
  `runs/gpu/*/artifacts/train/gpu-train-out/metrics.jsonl`, charted in
  `runs/progress.html`
- **Cross-simulator check (MuJoCo 3.12.0):** the accepted actor, dropped
  into the original MuJoCo model with native position servos, walks 1.008 m
  in 8 s at the 0.15 m/s command (84% of commanded distance, 19 qualified
  footfalls, first step at 0.41 s, max tilt 11.2°). It does not pass the
  strict evaluator there: alternation degrades into left-dominant stepping
  late in the episode. Stated plainly: the gait transfers as locomotion,
  not yet as a criterion-clean gait. `runs/mujoco-xval-final/mujoco-xval-result-cmd0.15.json`

## How it works

**Engine.** The physics is a generalized-coordinate articulated rigid-body
simulation (free root + 14 hinges) with contact manifolds and a dense
projected-Gauss-Seidel impulse solver, all local code under `experimental/`
(`articulated_v1/v2`, `contact_v1`, `coupled_impulse_v1`, glued by
`integrated_duck_v1`). The CPU lane is f64 with a momentum-residual gate of
1e-8 and is the oracle for everything else (`PLAN.md`). The CUDA port,
`experimental/duck_cuda/`, is a single-source kernel: one implementation in
`src/duck_cuda_kernel.h` compiled both as `duck_cuda.cu` (GPU, fp32,
warp-per-env 32-lane cooperative solver) and `duck_cuda_serial.cpp` (serial
reference used by the parity tests). Its header (`include/duck_cuda.h`)
documents the parity contract quantitatively (fp32-scaled certificates:
impulse tolerance 1e-6 vs CPU 1e-8, momentum residual 2e-4 vs 1e-8;
bit-identical determinism within a build) and the ABI history through v6:
per-tick PD recomputation inside one kernel launch, per-foot contact-tick
counters (v2), a device-side policy step (v3), per-episode gait-phase
offsets (v4), occupancy telemetry (v5), and per-episode domain
randomization of mass/friction/kp/damping plus a command-latency ring,
off-by-default and bit-identical when off (v6). Physics state is fp32, but
the in-kernel policy chain — observation, reward, termination — runs in
f64, mirroring `walk/env/flat.py` and `walk/env/reward.py` exactly, which
is what makes the CPU env and the GPU lane cross-checkable.

**Training loop.** PPO over 4096 environments with a 64-step horizon,
feed-forward policy, domain randomization (±10% mass and kp, ±20% friction
and damping, up to 1 policy step of command latency) — the exact flags are
in `gpu/specs/continue-ff-short.json`, executed by
`walk/train/gpu_train.py`. One `dwc1_step_policy` call advances all envs a
full policy step on-device — one kernel launch, with observation, reward,
and termination computed in-kernel so the per-tick full-state readback
disappears (`walk/env/cuda_lane.py`) — and the Python side only runs the
optimizer. Sandboxes are ephemeral RTX 5090s (spot by default; the
accepted lineage's spec ran on-demand) driven by `gpu/run_daytona.py` with
hard budget caps, verified deletion, and checkpoint relay between legs; a solver fault freezes only
the affected env and preserves its exact failing state
(`runs/faults/`).

**Judges.** The strict evaluator (`walk/eval/gait.py`) consumes raw
0.002 s-tick episode traces and implements the acceptance criteria from
`PLAN.md` verbatim (see next section). During training, an acceptance probe
runs every 5 updates (`--accept-every 5` in the spec): it captures fresh
8 s episodes at all three commands, evaluates them strictly, and on a full
pass re-confirms with 11 s episodes before writing the accepted checkpoint.
Multi-seed re-evaluation is a separate offline tool
(`walk/eval/acceptance.py`). The reward (`walk/env/reward.py`) is
velocity-tracking against a 0.4 s rolling average plus explicit
gait-shaping — per-touchdown step bonuses that mirror the evaluator's
qualification rules, alternation bonus/same-foot penalty, contact-chatter
and single-tick "flicker" penalties, whole-sole clearance bonus,
phase-locked stance, and self-imitation against a phase-indexed reference
cycle extracted from the project's own earlier best gait — each constant
carries its rationale in a comment, including the failures that forced it
(survival-only rewards produced lunging; W_SAME_FOOT at 0.5 left repeats
net-positive; touchdown bounces at 2 ms scale were the last alternation
breaker).

## The strict acceptance protocol, and the one amendment

Acceptance requires three 8-second episodes at commanded +0.10, +0.15,
+0.20 m/s, each passing all of: ≥6 alternating qualified footfalls with ≥3
per foot; each qualified footfall is a 60 ms–1.2 s swing whose whole sole
(minimum over 18 sole vertices) clears 10 mm for a contiguous ≥20 ms, with
≥30 mm forward placement, ≥40 ms of continuous support before liftoff and
after touchdown, opposite-foot support ≥90% of the swing, and stance slip
bounded at 25 mm; total forward translation within 60–150% of
command × 8 s; tilt ≤30° throughout; no reset, termination, or solver
fault (spliced or incomplete traces are rejected outright); first step
within 2.5 s and last step within the final 1.5 s (`PLAN.md`,
`walk/eval/gait.py`).

One change was made to the evaluator after the protocol was frozen, and it
is documented in the code rather than buried. Quoting
`walk/eval/gait.py` in full:

> AMENDMENT (authorized by John, 2026-09-01): contact debounce for SUPPORT
> continuity. The raw flag reports single-tick (2 ms) impulse dropouts
> during soft touchdowns as "lost support"; no physical force sensor
> resolves 2 ms dropouts, and real contact estimation is filtered over tens
> of ms. Gaps of <= 20 ms bracketed by contact are treated as continuous
> contact BEFORE swing and support-window analysis. Real swings (>= 60 ms
> minimum) are unaffected. Every other criterion, threshold, and semantics
> is unchanged.

Concretely (`CONTACT_DEBOUNCE_S = 0.020`): a contact gap of up to 20 ms
that is bracketed by contact on both sides is filled before the evaluator
segments swings and support windows. This cannot manufacture a qualified
step — those need a ≥60 ms swing with 10 mm whole-sole clearance held
≥20 ms — but it stops a single 2 ms solver-impulse dropout during a soft
touchdown from voiding an otherwise continuous 40 ms support window. The
amendment landed with a regression test (commit `447bcc0`), and the
acceptance results above were produced with it in effect.

## Reproducing

Local, no GPU or network needed:

```sh
# physics gates (must stay passed-local-native-cpu, 14 jobs)
.venv/bin/python -B experimental/integrated_duck_v1/run_local.py --output /abs/new-dir

# strict 4-seed acceptance of the shipped actor
.venv/bin/python -B -m walk.eval.acceptance --actor runs/actor-walking-v1.pt

# regenerate the training dashboard (runs/progress.html)
.venv/bin/python -B scripts/make_progress_dashboard.py
```

GPU jobs run on ephemeral Daytona sandboxes; the API key stays in Doppler
and is never written to disk (full policy in `gpu/README.md`):

```sh
# compile + unit gates on a sandbox
doppler run --project hallway --config dev --only-secrets DAYTONA_API_KEY --no-fallback -- \
  /Users/john/.cache/box3d-cuda-host-runtime-0.207.0/bin/python -B \
  gpu/run_daytona.py run --spec gpu/specs/compile-duck-cuda.json

# parity gates + throughput benchmark (writes gpu-parity.json / gpu-bench.json)
doppler run --project hallway --config dev --only-secrets DAYTONA_API_KEY --no-fallback -- \
  /Users/john/.cache/box3d-cuda-host-runtime-0.207.0/bin/python -B \
  gpu/run_daytona.py run --spec gpu/specs/parity-bench-duck-cuda.json

# one 13-minute training relay leg (resumes the checkpoint lineage)
doppler run --project hallway --config dev --only-secrets DAYTONA_API_KEY --no-fallback -- \
  /Users/john/.cache/box3d-cuda-host-runtime-0.207.0/bin/python -B \
  gpu/run_daytona.py run --spec gpu/specs/continue-ff-short.json
```

`--dry-run` on any spec validates it and prints the plan with zero provider
calls. Outputs land under `runs/gpu/<timestamp>-<spec>/` with redacted
logs, per-job artifacts, a manifest, and a verified deletion receipt.

## What's next

The next milestone inside this repo is the one in its name: walking across
a grid of cube rigid bodies. The world backend already exists —
`experimental/duck_world_v1` simulates the duck among dynamic cubes with
broadphase, contact islands, and sleeping, and `walk/env/grid.py` exposes
it behind the same environment contract as the flat floor — so the work is
training and acceptance, not infrastructure. Beyond that, the same
pattern (single-source f64-oracle/fp32-device engine, strict pre-registered
judges, relay training on ephemeral GPUs) is the intended template for
harder domains: manipulation on a humanoid arm (grasping, where contact
richness is the whole problem), additional physics domains beyond
rigid-body walking, and a standing "foundry" benchmark that runs the full
gate-parity-train-accept pipeline end to end so that engine and trainer
changes are measured the way this milestone was — against fixed criteria,
with every number traceable to an artifact.
