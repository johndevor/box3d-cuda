# Native machine-coupling extension v1

This additive extension lets a caller couple resident rigid scenes to custom
CUDA machine, fluid, thermal, or electrical kernels without exposing an
engine-owned pointer. The frozen ABI-v2 r3 header and all existing symbols are
unchanged. The extension is declared in
`box3d_cuda_machine_coupling_v1.h`.

An implementation advertises capability bit 18 for external wrench stepping
and bit 19 for final joint-velocity output. With every currently implemented
r3 capability, the complete mask is `0xE07FF`. The implementation version is
`0.5.0`; older `0.4.0` artifacts remain the frozen `0x207FF` baseline.

`box3d_cuda_scene_step_wrench_v1` accepts the exact 160-byte r3 step prefix,
then three pointers at fixed offsets:

- `external_force_xyz` at 160: optional device f32 `[E,B,3]`.
- `external_torque_xyz` at 168: optional device f32 `[E,B,3]`.
- `joint_velocity` at 176: optional device f32 `[E,J]` output.

The descriptor is 184 bytes and naturally 8-byte aligned. NULL wrench inputs
mean zero. Inputs are world-frame force at the center of mass and world-frame
torque about the center of mass. They are held for every requested control
step and integrated in every registered substep after gravity and angular
damping, before warm-start and constraint solving. Fixed bodies ignore them.

Revolute velocity is relative child-minus-parent angular velocity projected on
the normalized parent-local axis transformed to world. Prismatic velocity is
relative joint-anchor-point velocity, including each `omega cross r` term,
projected on the same world axis. Fixed joints write zero.

All pointers remain caller-owned and valid through completion on the supplied
stream. The engine neither retains nor aliases them. Device scalar contents
must be finite and bounded by the caller; enqueue-time validation covers
descriptor layout, dimensions, pointer device, flags, handle, and stream
ordering without introducing a host synchronization.

## Frozen goldens

1. With inverse mass `0.5`, local inverse inertia `(1,2,3)`, identity
   orientation, `h=0.25`, force `(2,4,6)`, and torque `(2,4,6)`, the wrench
   integration point produces `dv=(0.25,0.5,0.75)` and
   `domega=(0.5,2,4.5)`.
2. The same wrench leaves a fixed body unchanged.
3. A revolute parent-local `+X` axis rotated to world `+Y`, with parent and
   child angular velocities `(0,1,0)` and `(0,4,0)`, produces `q=0` and
   `qdot=+3 rad/s`.
4. The offset-anchor prismatic golden documented with World produces `q=+2 m`
   and `qdot=+4 m/s`.
