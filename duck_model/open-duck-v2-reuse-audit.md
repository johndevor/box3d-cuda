# Open Duck Mini V2: source-only reuse audit

Status: **exact native import rejected; no physics or policy executed.**
This is an additive local compatibility deliverable, not a new walking result.
It does not alter the active humanoid experiment or native ABI-v2 r3/machine-v1.
The long-term objective remains a reusable open-source CUDA engine with real
robot contact, large resident-body worlds, and independently evaluated learned
locomotion. A scripted duck viewer does not satisfy that objective.

## Exact sources

| Source | Pinned commit | Role |
| --- | --- | --- |
| [mertcookimg/Open_Duck_Mini_Viewer](https://github.com/mertcookimg/Open_Duck_Mini_Viewer/tree/af077653305cb003bf45ae817130dbeaa1779b74) | `af077653305cb003bf45ae817130dbeaa1779b74` | Browser/CAD frontend |
| [apirrone/Open_Duck_Mini, v2](https://github.com/apirrone/Open_Duck_Mini/tree/b23317a485b3cec7d8417f352478778b3475173c) | `b23317a485b3cec7d8417f352478778b3475173c` | Hardware/CAD and older published policy files |
| [apirrone/Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground/tree/b9be205ac64488c23504ca42e5ec790337adeec3) | `b9be205ac64488c23504ca42e5ec790337adeec3` | Current MJX task, model, export and CPU MuJoCo inference sources |

These are **not Pollen Microduck**. Prior Microduck analysis, its 61-observation
policy, BAM configuration and model-license caveat must not be applied here.

Viewer `src/robot/Robot.ts` generates sine joint angles, directly integrates
commanded odometry, and synthesizes IMU/foot/battery telemetry. It is reusable
presentation code, not a learned policy, contact solver or training benchmark.
Viewer LICENSE/NOTICE and its asset-specific LICENSE/NOTICE identify the
viewer and bundled URDF/STLs as Apache-2.0. Preserve notices if reused. The
hardware repository also has an Apache-2.0 LICENSE. Playground core files
carry Apache-2.0 headers, but there is no root LICENSE in this pinned tree:
review each copied file's provenance, especially data and imported assets.
This audit vendors no upstream meshes, weights, pickle data or source code.

## Model identity and native limits

The default Playground task is `flat_terrain`, loading
`scene_flat_terrain.xml` and its included `open_duck_mini_v2.xml`.
Its 16 XML body nodes comprise a massless floating base, a welded massive
trunk, and 14 articulated descendants. After explicit fixed-frame fusion,
there are 15 massive rigid components (2.1071407 kg), 14 hinges and 14 controls;
a floor would make native B16/J14. That fits native B32/J16/P16 **counts only**.
No compiled MuJoCo model was created to certify the massless-base treatment.

The hardware repository's XML instead has 22 body nodes and 16 actuators,
including antennas. Never mix that model, current 14-action training code,
and an older `BEST_WALK_ONNX*.onnx` based only on the shared V2 name.
Neither published ONNX binary was downloaded or executed in this audit;
their exact input/output graphs and matching historical runtime remain unverified.

Action order for the plain training model:

| Index | Joint | Child body |
| --- | --- | --- |
| 0 | left_hip_yaw | hip_roll_assembly |
| 1 | left_hip_roll | left_roll_to_pitch_assembly |
| 2 | left_hip_pitch | knee_and_ankle_assembly |
| 3 | left_knee | knee_and_ankle_assembly_2 |
| 4 | left_ankle | foot_assembly |
| 5 | neck_pitch | neck_pitch_assembly |
| 6 | head_pitch | head_pitch_to_yaw |
| 7 | head_yaw | neck_yaw_assembly |
| 8 | head_roll | head_assembly |
| 9 | right_hip_yaw | hip_roll_assembly_2 |
| 10 | right_hip_roll | right_roll_to_pitch_assembly |
| 11 | right_hip_pitch | knee_and_ankle_assembly_3 |
| 12 | right_knee | knee_and_ankle_assembly_4 |
| 13 | right_ankle | foot_assembly_2 |

Home controls, radians:
`[.002,.053,-.63,1.368,-.784,0,0,0,0,-.003,-.065,.635,1.379,-.796]`.
Axes default to child-local +Z in MJCF and must be transformed into native
parent-local axes. Upstream quaternions are wxyz, native xyzw. Z-up can remain
Z-up with native global XYZ gravity; no arbitrary coordinate swap is required.

### Unsupported without changing or extending the represented physics

- Two independently offset/rotated foot collision meshes, versus native's one
  COM-centred/body-axis OBB per body. The other 44 robot geoms are visual-only.
  STL hulls/bounds were not inspected. Never silently fit boxes or make the
  torso/shins collide with ground.
- Nonzero generalized rotor armature and dry joint friction. Body inertia is
  not a substitute for joint armature. External torques alone do not supply a
  matching static-friction constraint or modified generalized mass matrix.
- An infinite ground plane versus a bounded native OBB. A finite test arena is
  a separately authored approximation, not the unchanged upstream model.
- Backlash task: 24 generalized hinges (14 driven plus ten passive leg hinges),
  exceeding J16, with two coaxial hinges on each leg body. Duplicating two
  native binary constraints between the same bodies does not reproduce this.

All 15 massive links have offset COMs and full non-diagonal inertia. Exact
principal-frame rebasing can preserve those tensors, but must also transform
anchors, axes, references, sensor frames and independent collider poses.
Dropping products of inertia or substituting uniform-box masses is prohibited.
Ground contact priority, effective friction and MuJoCo constraint regularization
also need matching analysis; solver iteration counts are not interchangeable.
Native r3 does not expose the same site accelerometer API or enforce actuator
speed/acceleration limits. Sensor/controller state must live in the adapter and
participate in resets, not be inferred from a synthetic viewer.

## Motor constants: use the loaded XML

| Source | kp | damping | frictionloss | armature | torque limit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain embedded XML | 13.37 | .56 | .068 | .027 | ±3.23 Nm |
| Backlash embedded XML | 17.11 | .56 | .068 | .027 | ±3.23 Nm |
| Standalone joints_properties.xml (not loaded) | 17.8 | .60 | .052 | .028 | ±3.35 Nm |

Plain `kv=0`; its joint damping is separate from the actuator. These embedded
defaults are already flattened into the robot XML; do not include the standalone
file again. Current Playground uses these MuJoCo position actuators, not the
Pollen Microduck voltage/BAM path. An older hardware experiment optionally uses
BAM, but that is a different configuration and must be pinned separately.

## Current policy interface and early mismatches

Source-derived actor input is **101 scalars**, output **14 actions**:

| Half-open offsets | Field |
| --- | --- |
| 0:3 | site gyro |
| 3:6 | site accelerometer |
| 6:13 | vx, vy, yaw-rate, four head commands |
| 13:27 | actuator q minus home (plus passive backlash q when selected) |
| 27:41 | actuator qdot × .05 |
| 41:55, 55:69, 69:83 | three raw-action history frames |
| 83:97 | previous/current held motor-target state at the observation point |
| 97:99 | left/right foot–floor contact booleans |
| 99:101 | imitation phase cos/sin |

Control dt=.02 s, simulation dt=.002 s, ten substeps. Position targets are
home + .25×delayed action, with target slew bounded to 5.24 rad/s
(max .1048 rad/control step). This bounds **target** speed, not actual joint
velocity. Delay draws use an exclusive upper bound: 0,1,2 control frames
(0–40 ms), not an inclusive three-frame delay. Gravity delay exists in the
task code but gravity is not in the current actor observation.

The source explicitly sets `USE_IMITATION_REWARD=True`. Its reference motion
loads a pickle, which this audit neither deserializes nor executes. Turning
imitation off is a new experiment configuration, not reproduction of default
training. Current reward combines linear/yaw tracking, alive, torque, action
change, stand-still and imitation terms, scaled by dt and clipped nonnegative.
Its upside-down/NaN termination is weaker than our independent walking and
physics-health gates. The README's current-win command requests 300M timesteps;
there is no inspected source evidence for a sub-two-minute training recipe.

Static source checks expose these prelaunch issues:

1. `base.py` asks for `trunk_assembly_freejoint`, but both selected models
   declare `floating_base`. Resolve deliberately in a separate upstream adapter;
   do not silently rename the source or claim a runtime pass.
2. Training calls `accelerometer.at[0].set(accelerometer[0]+1.3)` without assigning
   the immutable JAX result. CPU inference really applies that +1.3 offset.
3. Training builds the next observation before rotating raw-action history;
   CPU inference updates its history after the preceding policy evaluation.
   Freeze a timestamped fixed-vector sequence before declaring the two inputs
   equivalent. Sensor, motor-target, contact and imitation-phase timing matter.
4. CPU inference does not reproduce training's random action/IMU delay or
   observation noise. Backlash position observations also need explicit review.
5. ONNX export includes normalizer and tanh(mean). It transfers weights through
   a TensorFlow reconstruction and can continue after a missing layer. Require
   strict layer coverage plus fixed-vector JAX/export parity, not just export success.
6. Dependency ranges float and there is no lockfile in this pinned tree;
   PPO parameters are inherited from an external BerkeleyHumanoid config.
   Source commit alone does not freeze a runnable environment or training profile.

## Smallest next experiment (proposal, not launched)

1. Keep the current native humanoid run independent. Reuse the viewer/CAD later
   as a presentation layer driven by real states; keep scripted demo mode labelled.
2. Choose one exact plain 14-action model and a matching policy artifact—not
   an arbitrary older 16-action model. Freeze runtime/model/action/observation
   identities, controller histories, sensor frames and solver settings. Fix
   constructor and inference discrepancies only in a separate reviewed adapter.
   Safely transcribe required reference metadata rather than loading untrusted pickle.
3. First physics check: one CPU MuJoCo environment, reset→one step→eight steps→
   a bounded 10-second replay of the **unchanged** matching policy, with finite
   state, contact, joint limits, penetration and actual body displacement recorded.
   No retuning or weight update during this comparison. That would be upstream
   reproduction, not native CUDA validation or from-scratch learning.
4. Native decision remains explicit: either design additive shape/armature/friction
   support with CPU goldens, or author a separately named simplified OBB duck and
   train it from scratch. Do not label the latter an exact Open Duck port or use
   it to claim unchanged-policy parity. No broad importer or engine rewrite is
   authorized by this audit.
5. Only after flat-ground native contact and locomotion pass should that policy
   face uneven terrain and movable blocks in the large-resident-body world.
   Separately report resident count, active count, training throughput, and
   real held-out walking behavior. Static voxels alone do not prove 200k dynamic bodies.

## Reproduce the local source-only check

`scripts/audit_open_duck_v2.py` verifies eleven exact text SHA-256 pins, inventories
XML and AST-parses Python without executing it. It is deliberately NOT a general
MJCF parser/importer, a MuJoCo compiler, or a complete model-admission validator.
It always emits `native_import_accepted=false`, `physics_executed=false` and
`policy_executed=false`. Binary assets need not be present.

After acquiring the exact text blobs in a local Git database:

```sh
AUDIT_OPEN_DUCK_GIT_DIR=/absolute/playground.git python3 -B -m unittest discover -s tests -p test_open_duck_source_audit.py -v
python3 -B scripts/audit_open_duck_v2.py --git-dir /absolute/playground.git --output /fresh/absolute/source-audit.json
```

The script disables Git lazy fetching during inspection and creates evidence
exclusively, refusing to overwrite an existing output. The four source-backed
tests skip explicitly when no local source is supplied; they must be present
and pass for this audit's evidence. Eleven tests passed with the pinned source
(seven portable checks, four source-backed checks); that is CPU text validation
only, not a full engine regression run. No upstream module, model weight, pickle,
MuJoCo/JAX runtime, CUDA build, provider, GPU or hardware was invoked.
