# Experimental complete articulated/contact CPU solve

This new candidate combines the sealed contact-v1 and articulated/joint-v1
packages through the additive articulated-v2 staging seam. It does not change
public r3/machine-v1, CUDA capabilities, or older source/evidence packages.

One step:

1. Compute PRE full generalized mass (including armature), bias, smooth velocity,
   actuator/passive forces, and authored joint-friction/soft-limit rows.
2. Query real convex foot manifolds; form world point Jacobians against all DOFs.
3. Solve joint scalar rows and non-associated Coulomb contact blocks together.
   Factor full mass, not the potentially rank-deficient contact matrix.
4. Validate the complete velocity/impulse momentum and joint residuals; integrate
   root and joints exactly once, then regenerate every principal-COM body state.
5. Privately stage contact POST bodies with PRE impulse caches. Validate both
   generations under one lock, then swap both without allocation or callbacks.

Capture, reset, and masked restore serialize both owners. There is no free-body
fallback, position clipping, posthoc foot impulse, or second integration.
Global matrix convergence is **not** guaranteed: first nonconvergence rejects
the entire batch without changing either owner's state, caches, or clock.

Contact geometry/cache is f32. Articulated state, dynamics, impulse solve and
clocks are f64 in this local native lane. Contact cache impulses are rounded to
f32 only after acceptance; warm normals are reused by stable feature with
normal agreement>.98 and point displacement<.02m, tangents transported through
the world basis. Current friction disks are revalidated/projected during solve.

Residual semantics: scalar rows use stable distance-to-bound corrections;
tangent rows use minimum-response-eigenvalue-scaled stationarity/complementarity
and disk feasibility. This is an explicit conditional KKT certificate, **not**
a proof of distance to the global solution. Contact normals do not include the
associated-cone radius derivative (no artificial dilatancy term).

## Local commands

Use an existing Python with NumPy and clang; no installation is performed:

```sh
python -B experimental/integrated_duck_v1/run_local.py --output /absolute/new-local-gates
```

The independently reviewed tests cover full-mass impulse algebra, anisotropic
friction, simultaneous soft limits/contact, numerical false-success cases,
fixed-floor immutability, two-owner later-environment rollback, and bit-exact
nonzero-cache masked replay. Model queries before the home-hold are static only.

Only after that gate and independent source review, the separately authorized
one-shot experiment is described in [HOME_HOLD_PROTOCOL.md](HOME_HOLD_PROTOCOL.md).
Use `run_home_hold.py --gates /absolute/new-local-gates/result.json --output
/absolute/new-attempt`. It rejects source/artifact drift, retains full raw logs,
and imposes a60-second external cap. No automatic retry. `export_replay.py`
renders only captured native states; it never advances physics.

These are CPU correctness/integration tools. They are not CUDA throughput,
200k-body performance, robot learning, or accepted walking evidence.
