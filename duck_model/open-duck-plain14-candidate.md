# Plain-14 executable source candidate v1

This is a bounded observation/controller adapter, **not a robot importer or a
physics implementation**. Source bundle: Open_Duck_Playground
`b9be205ac64488c23504ca42e5ec790337adeec3`, plain `scene_flat_terrain.xml` and its
included `open_duck_mini_v2.xml`. Fourteen actuator joints, 101 actor scalars.
The upstream files remain unmodified. This candidate is separately named
`box3d.open_duck_plain14.source_candidate/v1`; it is not represented as an
unchanged upstream runtime or a verified released-policy contract.

The previous [source audit](open-duck-v2-reuse-audit.md) remains immutable.
Its exact-native-import rejection remains in force. All source-identified
motor/geometry settings are retained, not replaced with native approximations.
Generated source-binding metadata verifies seventeen pinned original text
files and compares joint order, home, limits and embedded motor settings to
the candidate. It also retains the nested `eulerdamp="disable"` option.
This binding is not a MuJoCo compilation or proof of model initialization.

## Runnable files

- `open_duck_plain14_candidate.py`: dependency-free scalar adapter.
- `scripts/open_duck_plain14_fixture.py`: source-bound manifest generator;
  refuses source/candidate drift and output overwrites.
- `tests/test_open_duck_plain14_candidate.py`: independently authored fixed
  expected vectors, separate from implementation ownership.
- `tests/test_open_duck_plain14_binding.py`: exact source/candidate binding checks.

No upstream Python is imported. No model weights, pickle data, JAX, Torch,
MuJoCo, ONNX runtime, GPU or provider is used. Core fixture tests require only
Python's standard library. Source-backed checks need the exact text blobs in
a local Git database and explicitly skip when they are absent.

```sh
python3 -B -m unittest discover -s tests -p test_open_duck_plain14_candidate.py -v
AUDIT_OPEN_DUCK_GIT_DIR=/absolute/playground.git python3 -B -m unittest discover -s tests -p 'test_open_duck_plain14*.py' -v
python3 -B scripts/open_duck_plain14_fixture.py --git-dir /absolute/playground.git --output /fresh/absolute/source-binding.json
```

Expected numerical values were independently transcribed/evaluated from the
pinned source. Tests do not call candidate functions to generate their expected
answers. The full 101-scalar vector uses distinct segments to catch reordering,
scaling or missing fields. Controller fixtures cover ramp, reversal, all three
delay choices, changing delays, history timing, reset and inherited knee clipping.
They also reject wrong joint sets, bad widths, nonfinite data and invalid actions.
This is float64 scalar parity within 1e-12, **not JAX/native float32 bit parity**.

## Explicit corrections and timeline

The source model declares exactly `floating_base`. The candidate records that
name instead of the original base constructor's undeclared
`trunk_assembly_freejoint`. It does not patch that constructor or run it.
A future simulator wrapper must load the exact XML directly, verify its compiled
model, and bind the declared free joint and named actuator DOFs. Missing/extra
or antenna/backlash joints must reject rather than being silently truncated.

Controller state stores three raw-action history frames and held motor targets.
At reset: histories zero, targets source HOME. The caller owns immutable snapshots
of this state; no simulator/body state is stored here.

For control frame t:

1. Evaluate the policy on the previously prepared observation (policy execution
   is not implemented in this candidate).
2. `advance(state, action, delay_frames)` inserts raw action at history age zero;
   selects age 0/1/2 explicitly. These are 0/20/40 ms delays, not 60 ms.
3. Requested target is HOME + .25 × selected action. Slew from previous stored
   target is capped at .1048 rad per .02 s. Store this un-clipped-by-joint-range
   value as `motor_targets` and use it in observations.
4. `effective_controls` additionally clamps each target to the loaded actuator's
   inherited joint range. For example a fully raised left-knee target can be
   1.618 rad while effective control is 1.5707963267948966 rad. This is **target
   shaping**, not actual joint-speed enforcement or a contact solver.
5. Future simulator wrapper holds effective controls for ten .002 s substeps.
6. Encode returned physical site measurements using **PRE-shift raw histories**
   from `StepResult.observation_history`, plus the new motor targets. This matches
   the effective training-source order: `_get_obs` precedes raw-history rotation.
   Do not substitute the already shifted `StepResult.state.raw_history` here.
7. Retain `StepResult.state` for the next control frame.

The encoder accepts gyro and accelerometer in the source IMU site's frame,
not synthesized signals or the COM frame. Site offset is (-.08,0,.05) in the
floating-base frame. Acceleration must be the simulator's proper site sensor
measurement, including its specific-force convention; this adapter does not
infer it from simple position differences. There is no added +1.3 X offset,
matching effective training code rather than the divergent CPU playback code.
Contacts are measured left/right foot–floor booleans. Joint q and qdot must be
bound by actuator joint names, not assumed contiguous generalized-coordinate IDs.

Raw q−HOME and qdot×.05 precede three raw-action histories, absolute motor targets,
contacts and caller-supplied phase. The actor has no projected-gravity field.
The encoder performs no hidden normalization, tanh, RNG, observation clipping or
joint-limit clipping of measured q. Noise/pushes/randomization are disabled for
the first proposed deterministic comparison, a labelled departure from training.
Delay vectors remain explicit inputs. The phase reset vector (0,0) is allowed;
subsequent phase period must come from verified safe metadata, not a guessed gait.

## Motor/physics settings are declarations, not implemented dynamics

Loaded plain settings are kp=13.37, kv=0, joint damping=.56,
frictionloss=.068, armature=.027 and effort ±3.23 Nm. The candidate neither
integrates a PD law nor substitutes joint damping for actuator kv. It does not
remove rotor inertia, dry friction, mesh feet, full link inertia or an infinite
plane. These remain unsupported by an exact frozen native import.

## Released policy contract: not established

Bounded source inspection did **not** establish a released ONNX policy tied to
this exact plain-14/O101 bundle. The older hardware repository lists two
`BEST_WALK_ONNX` binaries and contains a 16-action morphology and legacy inference
variants. Their binary graphs were not loaded, downloaded or executed, so their
actual compatibility is unknown—not assumed incompatible solely from names.
The selected Playground source contains the export code but no matching released
policy manifest/checkpoint identity in the inspected tree.

Still missing before any unchanged-policy comparison:

- Exact ONNX bytes/full SHA and safe graph input/output metadata for O101/A14.
- Evidence tying model, action order, controller timing and checkpoint to this
  artifact, including the embedded normalizer and deterministic output transform.
- Reference phase period/fps in safe text metadata. Source computes its period
  from `period*fps` stored inside reference pickle; no value was guessed or loaded.
- A pinned simulator/runtime dependency set and fixed-vector export-output check.

The observation/controller fixtures work without those missing inputs, but do
not manufacture a policy identity. No broad repository-history search is planned.

## Smallest honest CPU MuJoCo comparison proposal

Not authorized or launched by this source step. Once the missing artifact
contract is provided, use one fresh isolated CPU process and the exact plain XML:

1. Compile the model, verify mass/COM/inertias, freejoint, sites, joints, actuators,
   control ranges, collision masks and loaded constants; preserve a compiled
   metadata digest. Stop on any mismatch, including massless-base initialization.
2. Run current scalar fixtures against the simulator-facing adapter before physics.
   Verify ONNX input/output metadata and fixed vectors without retraining;
   do not silently double-normalize or change tanh/output scaling.
3. With no randomization/pushes, fixed command and explicit phase/delay schedules,
   run reset→1 control frame→8 frames, stopping on nonfinite/limit/contact defects.
   Compare adapter observation and controls to independently calculated vectors
   at each captured frame. Proposed float32 adapter tolerance: 2e-6 absolute.
4. Only after those gates pass, propose one capped 10-second CPU policy replay.
   Log actual root displacement, foot contacts/slip, tilt, joint errors, controller
   histories and hashes. Preserve failures; no weight/controller retuning within
   the comparison. Walking promotion uses separately frozen behavioral gates,
   not merely surviving or achieving reward.

This would establish a coherent upstream reproduction baseline, **not** native
CUDA walking, all-active 200k-body performance, or training from scratch. Current
humanoid results remain independently owned and unchanged.

## Provenance

The new scalar adapter and fixtures are original project code. Source constants
and interface were reviewed against Apache-labelled Open Duck Playground task
files credited to DeepMind Technologies Limited, Antoine Pirrone and Steve Nguyen.
The audit preserves source hashes and credits without copying upstream model,
weights, reference data or runtime files into this candidate. Retain upstream
LICENSE/NOTICE with any later redistribution of their viewer/CAD/source assets.
