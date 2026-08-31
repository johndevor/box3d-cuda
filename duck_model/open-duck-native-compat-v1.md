# Combined native compatibility references, v1

Scope: executable geometry/frame and isolated-hinge goldens. No importer,
native physics/ABI, full-robot rollout, GPU/provider, dependencies or policy
change. Original robot evidence remains immutable. Reference matches do not
mean the frozen native backend supports this model.

## Geometry half

Use pinned source XML/CAD, compiled mass/principal-frame/mesh data already
captured, and all501 recorded states. Retain the source massless base as a
virtual frame alias of its welded trunk, not an invented massive/fixed link.
Map authored COM/full tensors, joint anchors/axes/reference and exact convex
foot support into principal-COM frames. The floor is an infinite plane, not a
finite box. Floor-foot and foot-foot effective materials are explicit and
distinct; no guessed global combine rule. Wrong-frame/axis/COM/reference,
discarded inertia off-diagonal and OBB-substitution controls must discriminate.

## Hinge half: frozen pre-execution matrix

Exactly16 cases, three integration steps each at .002s: maximum48 total steps.
One child process under60s wall cap, stop at first component/health failure,
retain terminal PRE/POST record. No retry/sweep/tuning. Synthetic1kg rotor with
diagonal inertia (.01,.02,.02)kg m², COM at a world-fixed +Z hinge; gravity0,
no collision geometry or joint limits. This isolates torque/inertia semantics,
not the robot's complete mass distribution. Fresh state per declared case.

Motor: kp13.37 N·m/rad, kv0, actuator limit±3.23N·m, no control-position clamp
in this isolated fixture. Tests use q=0 except the final q=.1 case. Parameters
below are explicit ablations of source armature .027kg m², passive damping
.56N·m·s/rad and load-independent frictionloss .068N·m—not model tuning.

| Cases | Initial velocity | Target | Armature | Damping | Frictionloss |
|---|---:|---:|---:|---:|---:|
| bare positive / negative | 0 | ±.1 | 0 | 0 | 0 |
| armature positive / negative | 0 | ±.1 | .027 | 0 | 0 |
| damping positive / negative | ±2 | 0 | .027 | .56 | 0 |
| cap + damping positive / negative | ±2 | ±1 | .027 | .56 | 0 |
| friction exact rest | 0 | 0 | .027 | .56 | .068 |
| friction sub-threshold positive / negative | 0 | ±(.02/13.37) | .027 | .56 | .068 |
| friction slip positive / negative | ±2 | 0 | .027 | .56 | .068 |
| friction above-threshold positive / negative | 0 | ±(.136/13.37) | .027 | .56 | .068 |
| position error negative (q=.1) | 0 | 0 | .027 | .56 | .068 |

MuJoCo3.3.7 cached runtime; Newton iterations1/line-search5, Euler with implicit
joint damping disabled as in the original model. Explicit default friction
solref(.02,1), solimp(.9,.95,.001,.5,2); solver coefficients captured per step.
No extra forward pass: force/mass/solver outputs correspond to PRE-integration
state; q/v/time are POST-integration state.

## Frozen checks and honest solver-dependent comparisons

Analytic laws: M=.02+armature; actuator=clamp(kp*(target−q)−kv*v,±3.23);
passive=−damping*v separately; M*a=actuator+passive+constraint. Explicit damping
with semi-implicit Euler gives v'=v+.002*a, q'=q+.002*v'. Check these with fixed
absolute tolerances: inertia1e-12, torque1e-11, acceleration1e-9, state1e-11,
clock1e-12. Require finite state/forces, zero warnings, no contacts, and friction
force bounded by authored frictionloss and opposing nonzero velocity.

For a scalar friction row J=1, compare runtime force to the clipped quadratic
optimum `(aref−a_smooth)/(1/M+R)`. R and aref come from captured solver outputs:
this tests the scalar optimization conditional on them, not their derivation.
A one-iteration solve is not assumed converged. Difference and1e-9N·m comparison
are diagnostics, not a secretly imposed rigid-stiction law or relaxed gate.
Sub-threshold motion is measured; only zero-input rest should stay exactly
at rest. No claim of general soft-contact or fullrobot MuJoCo/native parity.

MuJoCo documents actuator clamping, passive forces, semi-implicit integration
and friction-loss constraints separately: [3.3.7 computation](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html),
[pinned forward source](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_forward.c),
[pinned solver source](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_solver.c).

## Process evidence

Pure reference and mocked harness tests precede execution. Freeze source and
case-XML hashes before the sole attempt. Reuse prior installed-runtime/source
audit; verify current distribution identities and native runtime binary pins,
not repeat a broad audit. A separate launcher sends stdout/stderr directly to
exclusive files, records hashes/exit/time, enforces timeout and refuses an
existing run directory. This fixes the prior tool-output truncation boundary
without rerunning or rewriting that older evidence.
