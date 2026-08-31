# Articulated complete-solve hook v2 (experimental CPU)

This additive C ABI (`0x41520002`) separates PRE dynamics from POST integration.
It is intended for the independently owned `integrated_duck_v1` contact driver.
No legacy API, joint-v1 source, AV1 source, default build, or default engine path
is changed. This component alone does not solve contact cones or advance a
complete contacting robot. It uses the frozen AV1 stateless FK/dynamics and
root integration primitives, never AV1's stateful solve/integration path.

## Build and verify

From a repository containing the sealed joint-v1 and AV1 additive sources:

```sh
python experimental/articulated_v2/run_local.py --output /absolute/fresh/results
```

Uses existing clang/clang++ and Python/NumPy; installs nothing. Builds a shared
library, executes synthetic native tests, compiles C11 ABI assertions, runs
ASAN/UBSAN, the 15 AV2 tests and frozen 17 AV1/18 joint tests. Exact-duck checks
evaluate saved static references or PRE only: no duck integration and no
MuJoCo execution. All command output and hashes are retained. Standalone CMake
configuration is supplied for convenience but is not the tested build path.

## State and equations

All AV2 generalized states, matrices, impulses and clocks are doubles. Model
inertial and controller constants retain the frozen AV1 representation.
`qpos=[root xyz, root xyzw, hinge positions]`; generalized velocity is root
origin WORLD linear velocity, WORLD angular velocity, then hinge rates.
Body output and spatial Jacobians use principal COM frames with world vectors,
linear components before angular components. The root is not fixed.

PRE owns a snapshot and returns full `M=M_body(q)+diag(armature)`, its inverse,
gravity/Coriolis/gyroscopic bias, body poses/Jacobians and smooth velocity:

```
v_smooth = v_pre + dt * inverse(M) * (motor + passive + applied - bias)
```

Commands are clamped to enabled authored ranges before the PD motor law, then
motor effort is capped; passive damping remains outside the motor cap.
The plain14 source uses position-control inheritrange=1. The adapter takes the
explicit authored range endpoints; no claim of bitwise compiler-derived
control-range endpoint equivalence is made.

Fixed `3*J` row slots contain friction, lower limits, then upper limits. An
inactive row has G=0, regularizer=1, bounds=0 and requires exactly zero impulse.
The complete solver must include every active joint row in the same solve as
contact; solving friction first and adding contact or limits later is invalid.

For friction, `G=+e_j`, bounds are `[-dt*loss,+dt*loss]`, K=0 and d=d0. For
limits, gaps are `q-lower` and `upper-q`, G signs are + and -. A limit is active
strictly when `gap < margin`, with nonnegative impulse. Exact boundaries are
inactive even when velocity points outwards. This is the authored soft-limit
law, not predictive position projection; penetration can persist after a step.

For the supported positive solref/increasing solimp format:

```
t_eff = max(timeconst, 2*dt)
B = 2 / max(1e-15, dwidth*t_eff)
K = 1 / max(1e-15, dwidth^2*t_eff^2*dampratio^2)  # limits only
aref = -B*(G*v_pre) - K*d*(gap-margin)
target = G*v_pre + dt*aref
regularizer = max(1e-15, (1-d)/d * reference_inverse_weight)
```

`reference_inverse_weight` is the hinge diagonal of the FULL inverse of
`M_body(qpos0)+armature`, including free-root coupling. It is not reciprocal
diagonal inertia, current/reset-pose inertia, or a fixed-root reduction.
Limit d follows the authored piecewise-power impedance ramp in
`abs((gap-margin)/width)`. Impedance endpoints and midpoint clamp to
[0.0001,0.9999]; width<=1e-15 uses the mean endpoint impedance; denominator
floors match the source guards. Registration rejects negative width, power<1,
decreasing impedance, nonpositive solref, and out-of-domain parameters. This is
not support for all MuJoCo constraint formats. Plain14 defaults are
solref=(.02,1), solimp=(.9,.95,.001,.5,2), margin=0.

Equations were independently checked against the already captured official
[MuJoCo 3.3.7 constraint source](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_constraint.c).
No new simulator trajectory or parameter query is required by this package.

## Complete solution and transaction

1. `av2_prepare` computes PRE only; it neither solves nor integrates.
2. The contact owner assembles point rows with PRE Jacobians and solves all
   joint/contact impulses with the full mass matrix. It validates its own
   normal/tangent law and contact cache. Pass complete velocity, joint impulse
   and total contact generalized impulse to `av2_complete`.
3. `av2_complete` checks velocity momentum balance against
   `v_smooth + inverse(M)*(G_joint^T*lambda_joint + contact_impulse)` and every
   joint bound/projected KKT correction. These are absolute requested
   tolerances, both positive and <=1e-5. Stable direct correction calculation
   avoids falsely reporting zero from subtraction of large nearby impulses.
4. Only a valid solution creates private POST: root translation uses v_post,
   root quaternion receives left world `Exp(dt*omega_post)`, and each hinge
   advances `dt*v_post` exactly once. Warm force is impulse/dt; saturated
   friction cache values are canonicalized to the already validated force
   bound to avoid a one-ULP division overshoot. Time and count advance once.
5. The caller builds the contact participant's private stage from this POST.
   Under ONE caller lock, validate BOTH participants before either commit.
   AV2 commit then only swaps owned storage and advances generation. Do not
   expose either owner's pointers or allow concurrent mutations between
   validation and commits. AV2 alone cannot make unrelated ownership atomic.

`prepare_restore` and `prepare_reset` also return private validated stages.
Snapshots include q/v, all joint warm forces, per-environment time/count and a
model/reset binding (identity check, not cryptographic authentication). Reset
masks contain exactly E values in {0,1}. Stale/consumed/wrong-owner stages are
rejected. Generation never rewinds on restore. Pre/stages must be destroyed
before their scene; views are borrowed and read-only. Live handles, array
lengths, nonaliasing output buffers and serialization remain C caller duties.
Failed operations leave published state and output handles unchanged.

## Verification limits

Native tests cover both violated limits, strict boundary activation, impedance
and timestep guards, mixed friction/limit coupling, external contact impulse,
quaternion integration once, atomic batch failure, stale stages, masked reset,
bound snapshot restoration and the numerical review regressions. The AV1
static fixture checks do not establish dynamic equivalence. No CUDA/GPU,
complete-robot health, walking, learning, provider, merge or deployment claim.
