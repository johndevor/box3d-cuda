# Ten-second stationary CPU contact/controller admission

One new attempt only, after immutable floor-clear eight-frame evidence. This is
MuJoCo 3.3.7 CPU, not Box3D CUDA, learned standing, walking or policy transfer.

## Frozen experiment

- Exact original plain14 XML/STL and cached CPython/NumPy/MuJoCo runtime.
- Same minimal floor-clear root-Z reset helper; no later reset or teleport.
- 500 control frames at .02 s, ten .002 s integration steps each: 10 seconds.
- Fourteen zero actions, commands seven zeros, original home position targets.
- Repeat the short fixture's delay cycle `[0,1,2,0,1,2,0,1]` to 500 entries.
- Observation phase `[0,0]` at all 501 checkpoints is explicitly synthetic.
- Same source armature, damping, friction, gains, efforts, timestep and solver.
- 60-second external subprocess cap; stop at first physical/harness failure;
  preserve terminal POST snapshot and raw process outputs. No retry or tuning.

## Unchanged admission gates

All state/sensors/controls/forces finite, no solver warnings; clock matches
`frame*.02` within `1e-10` seconds. Maximum joint limit violation .05 rad,
penetration .01 m, joint speed 250 rad/s, base linear speed 20 m/s, base angular
speed 250 rad/s. These are CONTROL-CHECKPOINT gates, not every-internal-step
maxima. None is a standing-height or uprightness criterion. They extend without
reinterpretation; health acceptance must not be promoted to standing.

## Capture and behavioral limits

Record actual qpos/qvel/qacc, actuator force, contacts, effective controls,
targets, raw action history, gyro/accelerometer and complete O101 observation.
Contacts/sensors preserve the actual returned MuJoCo computation stage.

For post-integration root/torso pose and foot clearance, copy configuration to a
separate MjData and call ONLY `mj_kinematics` there. Never call extra forward,
collision, constraints or dynamics on live data. Report world +Z up, active
local-to-world wxyz quaternion, tilt, height, horizontal drift and orientation
change from reset. Foot clearance uses the compiled convex-mesh vertices and
the frozen scalar plane-support routine. It can differ from returned contact
distance because stages differ; do not mix those measurements.

The source has only two foot collision meshes. Torso/head/limbs are visual-only:
a fall is not necessarily stopped by body-floor contacts. Report quantitative
behavior honestly even if all unchanged numerical health gates pass. No new
uprightness threshold is silently substituted into this experiment.

## Native follow-on boundary

No native implementation is authorized by this run. Independent inspection
finds counts fit (15 massive links + floor, 14 hinges, three eligible pairs),
but exact mesh feet/plane, COM/principal-inertia frame transformations, virtual
massless root, joint armature, separate passive damping/friction-loss and pair
materials are not represented faithfully by frozen r3/machine-v1. The smallest
next step is geometry/frame and isolated-hinge goldens, not importing a silently
simplified robot or claiming CPU evidence validates native execution.
