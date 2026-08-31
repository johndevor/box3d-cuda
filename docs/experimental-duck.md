# Experimental Open Duck / cube-grid track

This track is additive. The existing PyTorch package, native ABI-v2 r3,
machine-coupling v1 and their source files are preserved. Experimental
`av1_*`, `av2_*`, `idv1_*`, `dwv1_*` and `dwc1_*` interfaces are separate;
do not substitute one library for another or infer their capabilities from
the frozen native capability mask.

## What is here

- `experimental/integrated_duck_v1`: double-precision articulated CPU dynamics
  plus contact. All original physical certificates remain enforced. The
  solver's stall accelerators have an additional bounded inner-work budget;
  its reported outer sweep count is not a count of all numerical operations.
- `experimental/duck_world_v1`: one articulated robot and at most 1,024 cubes
  per environment (262,144 total cube slots across a batch), with bounded
  contact islands, sleeping and transactional reset. CPU, not a demonstrated
  200k-active-body CUDA world.
- `experimental/duck_cuda`: specialized float32 robot implementation, with
  separate serial and CUDA drivers. This is not the double-precision CPU
  solver compiled unchanged for GPU, nor the frozen rigid native ABI.
- `walk`: batched environments, reward, CPU/CUDA-lane PPO trainers, strict
  gait evaluator and failure quarantine. Solver faults reject a rollout;
  they are not valid terminal transitions for PPO.

## Reproduce locally

Use Python 3.10+ with NumPy, pytest, PyTorch and an existing clang toolchain.
This verification uses no provider, credentials, CUDA execution or simulator
installation. The serial CUDA-source tests execute on the CPU.

```sh
python -B scripts/verify_duck_cpu.py --output /absolute/fresh/check-directory
```

`--without-training` omits trainer tests/PyTorch-dependent walking discovery.
The default also enforces the original 10 ms/capacity-tick benchmark.
Shared Linux CI uses `--correctness-only`: all physics/capacity assertions
remain enforced, while measured timing and a missed-target flag are retained
as hardware-dependent telemetry. This mode cannot establish a performance pass.
The runner records commands, stdout/stderr hashes and outcomes, stops at the
first failure, and has a ten-minute total cap. Runtime-only native builds go
under ignored `build/`; do not load historical checked-in binaries.

The model's recorded CPU reference and XML are included under
`duck_model/reference`, checked against their existing SHA-256 identities.
The live environment no longer needs the original author's checkout. The
reference has 501 historical MuJoCo frames used for static translation
checks; loading it does not run MuJoCo or establish native walking. See
`duck_model/reference/README.md` for provenance and asset notices.

To run just the native algebra/translation/transaction/sanitizer gates:

```sh
python -B experimental/integrated_duck_v1/run_local.py \
  --output /absolute/fresh/native-check-directory
```

## Evidence limits

`evidence/gpu-bringup-20260831` is retained historical evidence for the exact
source/artifacts in its manifest. Its GPU comparison resynchronizes state and
clears warm starts every 100 ticks, with 1e-3 position/coordinate and 0.25
velocity-array bounds. It is a bounded, windowed float32 comparison—not
bit-exact CPU/GPU agreement or an uninterrupted walking evaluation. Later
device-policy and scratch-layout commits are not automatically validated by
that earlier manifest. No GPU rerun was performed for this integration.

The fault-corpus tests require separately captured training artifacts. They
report skipped when that corpus is absent. Two small, immutable saved
nonconvergence problems are included for reproducible independent KKT,
momentum, deterministic repeat and failure-atomicity regression tests.

The fast policy interface returns structured native diagnostics. Any solver
fault is now saved and raised before PPO can consume the transition. Its
artifact records the last accepted tick after a possibly partial policy
repeat; it explicitly does **not** claim to be an exact full-policy snapshot.

Historical gate logs and machine-specific binaries are kept in the imported
Git history, not distributed as current build products. Older local voxel,
terrain and experimental worktrees are not silently promoted by this merge.

## Remote work is explicit

`gpu/run_daytona.py` is an opt-in paid-resource launcher; unit tests use fake
providers. Review resources and limits before invoking it. The compile spec
now builds actual sources and fails if they are absent; compilation neither
runs a smoke test nor implies functional acceptance. Parity and training are
separate specs. Hardware/model/library identities must match the evidence
being claimed. No default command in this guide starts paid work.

Full CAD replay exports additionally need the explicitly supplied historical
CAD/FK asset bundle; the physics reference package is not a full mesh bundle.
The replay must retain its recorded/scripted/learned label and asset notices.
