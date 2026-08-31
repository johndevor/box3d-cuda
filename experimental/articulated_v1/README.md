# Native free-root articulated adapter v1

This opt-in **host CPU** adapter supplies the missing articulated-model layer for
exact plain14 OpenDuck. It consumes the accepted principal-COM topology and
feeds the sealed joint-v1 coupled operator. It implements FK, all-body spatial
Jacobians, current body mass, gravity/Coriolis/gyroscopic bias, source-root
position/quaternion integration, joint mapping, and owned transactional state.
It is not wired into production CUDA/r3 or the contact scene.

The C ABI is `0x41520001`, separate from joint-v1 `0x4a530001` and r3 `0x20000`.
All new files are under `experimental/articulated_v1/`. No old root CMake,
capabilities, layouts, machine API, defaults, or joint/contact warm cache changes.
The frozen joint-v1 implementation is a required, unchanged dependency.

## Exact model and frames

The fixture consumes the same sealed `geometry-goldens.json` as CUDA's asset
translation: SHA `e52ba7d0f79434499d8fb6c2d611eb46ee12e2f32cb36258b38cd22959d0b08b`.
No alternate XML importer, eigendecomposition, mass fit or collision conversion
is introduced. Body 0 is the fixed floor; bodies 1..15 are all authored massive
links (total 2.1071407 kg). The massless base aliases trunk body 1, without an
invented extra inertia. Fourteen ordered hinges have child id `j+2` and a
preceding massive parent. Anchors, axes and zero-q reference orientations are
already in principal COM frames. The bounded C model also admits small synthetic
trees for independent testing: B=J+2, 0<=J<=26, N=6+J<=32.

Generalized velocity is

```
v = [root origin WORLD linear xyz, root WORLD angular xyz, 14 scalar joint rates]
q = [root authored-origin xyz, root source-frame xyzw quaternion, 14 joint angles]
```

Spatial J is `[B,6,N]`: WORLD principal-COM linear xyz, then angular xyz. Body
poses are COM xyz and principal-to-world xyzw. The root state is the authored
base/trunk **origin**, not the trunk COM. Its actual COM offset is
`(-.0483259,-9.97823e-5,.0384971)` in the source frame; the accepted principal
orientation is retained. Root/kinematic calculations use double. Owned hinge
coordinates and generalized velocity are f32 to match joint-v1; registration
explicitly rounds hinge q once, and snapshots expose those exact stored values.

MuJoCo's free root uses world linear and **local** angular velocity. At every
retained checkpoint we explicitly rotate angular velocity by source-root R.
For `v_mj=T*v_world`, `T=diag(I3,R^T,I14)`, compare
`J_world=J_mj*T`, `M_world=T^T*M_mj*T`, `bias_world=T^T*bias_mj`.
The apparent angular basis derivative contributes zero here because
`omega x omega=0`. No ordinary coordinate-Christoffel formula is applied to
world angular quasi-coordinates. See the official
[MuJoCo floating-object convention](https://mujoco.readthedocs.io/en/3.3.7/overview.html#floating-objects).

## Kinematics, mass and bias

For principal-frame parent p and hinge child c, with local anchors ap/ac,
parent axis s and zero-q orientation qref:

```
Rc = Rp * Exp(s*qj) * Rref
world_anchor = pc_parent + Rp*ap
pc_child = world_anchor - Rc*ac
```

Root columns include the COM lever arm; child Jacobians inherit every ancestor
column. The hinge column is `[s_world x (COM-anchor), s_world]`. All massive
bodies therefore respond to a coupled generalized impulse, including bodies
outside the directly actuated branch when the free root moves.

The zero-generalized-acceleration recursion keeps root-origin world linear
and angular velocities fixed. Root-origin bias acceleration and angular
acceleration are zero; root COM acceleration includes `omega x (omega x r)`.
For parent/child COMs and world anchor offsets rp/rc:

```
omega_c = omega_p + s_world*qdot
alpha_c = alpha_p + (omega_p x s_world)*qdot
v_anchor = v_p + omega_p x rp
v_c = v_anchor - omega_c x rc
a_anchor = a_p + alpha_p x rp + omega_p x (omega_p x rp)
a_c = a_anchor - alpha_c x rc - omega_c x (omega_c x rc)
```

The native adapter then computes, over all massive bodies:

```
M_body = sum(m*Jv^T*Jv + Jw^T*I_world*Jw)
bias = sum(Jv^T*m*(a_bias-gravity)
           + Jw^T*(I_world*alpha_bias + omega x (I_world*omega)))
M_A = M_body + diag([0,0,0,0,0,0, .027 x 14])
M_A*vdot = actuator + passive + caller_applied_force + G^T*constraint_force - bias
```

There is no extra `omega x v_root`, no omitted gyroscopic term, no added second
Mdot term, no body-mass substitution for armature, and no assumption that the
root is fixed. Friction reference inverse weights use the complete free-root
reference mass at authored qpos0 (zero hinge angles), **not the home/reset pose**
or a fixed-root joint matrix. The reference pose is retained as a registration
input; captured reference mass, bias or Jacobians never feed runtime calculation.

The duck model maps each authored hinge's `.027` armature, `.56` passive damping,
`.068` friction loss, `13.37` stiffness, zero motor damping and `3.23` cap to slots
6..19. Solref/solimp-derived friction coefficients remain the sealed joint-v1
law (`d0=.9,dwidth=.95,timeconst=.02`). Root slots own no actuator, armature or
joint friction. Bias is subtracted by the adapter; callers supply applied forces
without subtracting bias again. Fixed external rows use the same world basis and
must include their own `-Gdot*v` reference term.

Every runtime state evaluates double M/bias, checks f32 representability and
the frozen operator's body/armature factorization, before publication. Double
static evaluation is distinct and may represent quantities beyond the f32
solver's range. Dynamics cast current M and `(applied_force-bias)` to f32, then
execute actual joint-v1 code. This precision boundary is explicit; the near-double
static reference errors are not a claim of identical f32 trajectory results.

## Root integration and transactional ownership

`av1_prepare` clones the owned joint state, assembles current mass and full bias,
executes the frozen coupled operator, and integrates the actual source root once
using the accepted post-step velocity:

```
p_post = p_pre + dt*v_root_post
qroot_post = normalize(Exp(dt*omega_world_post) * qroot_pre)
qjoint_post = qjoint_pre + dt*qdot_post
```

The quaternion exponential is left multiplied because omega is world-frame.
It is neither Euler integration of four quaternion components nor a scalar-root
placeholder. POST all-body FK and velocity are regenerated from the new root
and joint state, validated before the stage can be published. The scheme is
first-order semi-implicit integration, not a high-order energy-preserving solver.

Create copies topology, physical coefficients, reference pose, initial state and
gravity. Snapshots contain the actual root quaternion, joint q/qdot, friction warm
force and per-environment uint64 step count. An FNV1a model/reference/reset-state
binding prevents accidental cross-model replay; it is not authentication. Restore
requires canonical f32-exact hinge q, unit double root quaternion (squared norm
within 1e-10), finite state, valid warm-force bounds, and an admissible M/bias
bridge in every environment. Selected reset returns owned initial root/joint
state and zero warm/count, preserving peers bit-for-bit.

Prepare and restore failures leave all published state unchanged. A prepared
stage is private; read/capture can inspect it before commit. Commit checks owner,
generation and consumption, then swaps already allocated state with no further
computation/allocation. Every restore/reset/commit advances a non-rewinding
generation, rejecting stale and already consumed stages. Step count can be
restored, but generation cannot. Counter exhaustion fails explicitly.

The caller must serialize scene operations, provide correctly sized live host
buffers with nonoverlapping writable outputs, and destroy stages before their
owner. Python wrappers validate array dimensions before handing pointers to C.
No GPU device pointer, asynchronous stream or arbitrary pointer-length validation
is claimed by this host C ABI.

## Contact ownership boundary

CUDA retains `experimental/contact_v1/`, exact collision/material assets and
Coulomb cone orchestration. CUDA3 owns this runtime articulated adapter. Its
stateless `av1_response` evaluates current M_A and returns ALL generalized and
all-body velocity responses. CUDA's signed contact wrenches map through this J:
A `[-d,-rA x d]`, B `[+d,+rB x d]`. The source-root world basis must be used
consistently for G, effective mass and every velocity update.

The joint-v1 external fixed-box QP does not implement a Coulomb disk whose
radius depends on unknown normal force. No joint-aware contact step or combined
contact/adapter commit is wired here. A future combined transaction must prepare
both owners, validate both generations under serialization, then commit both
without remaining failure paths. The current adapter atomically owns its root
and joint scene only. Body poses must be integrated once, not once per contact
impulse. A free-body contact solve plus local denominators is not equivalent.

## Evidence and reproduction

Run the local build with an existing Python+NumPy and clang/clang++:

```sh
/path/to/existing/python -B experimental/articulated_v1/run_local.py \
  --output /absolute/new/validation-directory
```

This builds C++17, executes the synthetic native smoke, checks C11 ABI/layouts,
executes AddressSanitizer+UndefinedBehaviorSanitizer smoke, runs 17 focused adapter
tests and the existing 18 joint-v1 tests. The output directory must be fresh.
Each child is bounded to 60 seconds, with complete stdout/stderr files. It never
imports MuJoCo, runs the separate query script, or advances the duck model.
An optional standalone CMake target is included but its configure/CTest path is
unverified because CMake was unavailable locally.

A runnable native exact-duck static inspection is also provided:

```sh
/path/to/existing/python -B experimental/articulated_v1/scripts/inspect_plain14.py \
  --library /absolute/validation-directory/libarticulated_v1.dylib
```

Use `.so` on Linux. It instantiates the exact model, evaluates all three retained
states and returns all-body impulse responses without changing state or time.

The separately authorized MuJoCo reference used only retained checkpoints
0/250/500 and pinned source/runtime 3.3.7/NumPy 2.4.6. Exact inputs, outputs,
source/model/runtime pins and process records are retained. The first extraction
incorrectly read legacy qM after `mj_crb`, yielding zeros; it is preserved as
rejected evidence. The corrected extraction uses `mj_makeM` and verifies SPD.
Both processes queried the same three states, with combined wall time
0.301116625 seconds, below the 60-second total allowance. Neither called
mj_step, integrated time, solved contacts, changed source evidence or installed
anything. The corrected output SHA is
`58535c1e36728ced2c69b87c504a11116757313a6fd556c2435f3515a9f6e5a1`.
See [MuJoCo mass and bias subcomponents](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APIfunctions.html#mj-makem).
Do not rerun the explicit query script as part of ordinary validation.

The native static comparisons cover all 15 bodies at the three checkpoints,
with maximum errors: pose 3.34e-16 m, rotation matrix 5.45e-15, J 1.86e-15,
velocity 7.59e-19, mass 2.78e-16, bias 3.56e-15. These checkpoints have small
velocities; they alone would be weak evidence for Coriolis behavior. Separate
nontrivial static rotations/rates, geometric-ancestor and finite-difference J,
Jdot*v, potential gradients, free fall, kinetic power, Galilean invariance and
rigid-aggregate gyroscopic oracles cover that gap. Only synthetic 0/1/2-hinge
models are advanced by tests. OpenDuck receives static evaluation, registration,
capture and nonmutating impulse-response queries only.

No full OpenDuck dynamics trajectory, joint-aware contact cone solve, CUDA
compile/runtime, GPU work, training, walking, performance, merged code or
production acceptance is claimed. nvcc and CMake were unavailable locally;
no dependencies/providers were installed or provisioned. Earlier sealed
joint/contact/experiment packages remain unchanged.
