# Native C ABI v2 proposal

`proposals/box3d_cuda_v2.h` is draft revision 2 of the resident rigid-scene
boundary. It is compile-checked and concrete enough for foreign-function review,
but it is deliberately outside the installed `include/` tree. ABI v1 remains
the only implemented and stable native ABI.

The proposal fixes these semantics:

- 64-bit opaque scene handles; registration copies CPU buffers synchronously;
- immutable topology, engine-owned resident GPU state, and dense `uint32_t`
  indices mapped from caller IDs;
- packed environment-major FP32 state in position xyz, quaternion xyzw, linear
  velocity xyz, angular velocity xyz order;
- joint kinds fixed=0, revolute=1, prismatic=2; local anchors, normalized
  parent-local axes, and `conjugate(parent) * child` reference quaternions;
- signed 64-bit contact feature IDs with `-1` as the empty sentinel;
- distinct joint, feature-ID, and contact-impulse caches;
- explicit material binding and capability discovery. The first implementation
  may advertise only global material and reject heterogeneous inputs; it must
  never invent a body-to-pair combine rule;
- fixed bodies are supported. A kinematic body has inverse mass/inertia zero
  and a caller-prescribed velocity. Kinematics, per-environment gravity,
  non-global materials, and speed/acceleration enforcement are independently
  discoverable capabilities and fail closed when absent;
- naturally aligned device pointers, no implicit strides or padded views, and
  exact packed shapes beside every pointer;
- flags and reserved input fields must be zero. Bounds, unsupported layouts,
  non-finite values, duplicate actuator assignments, and missing capabilities
  fail closed before work is enqueued;
- each joint has at most one actuator. Disabled joints ignore targets. The
  `[E,J]` position, velocity, effort, speed, and acceleration frame is held
  across every requested control step; registered substeps are internal
  integration subdivisions;
- successful asynchronous calls report that validation and enqueue succeeded.
  Execution failures can surface when the caller synchronizes its stream;
- diagnostics expose both final `contact_active` and accumulated
  `contact_ever`. Contact counts are unsigned final-step values; motor and
  contact impulses accumulate across the requested control steps.

## Deterministic reset boundary

Capture and restore have separate const-correct descriptors. They include all
per-environment mutable physics values: state, inverse mass and inertia, half
extents, gravity in its selected layout, the selected material layout, joint
cache, contact feature IDs, and contact impulse cache. The immutable graph,
motion kinds, dimensions, and layout choices remain handle topology.

Material snapshot arrays use the registered binding: global `[1]`, per-body
`[E,B]`, or explicitly authored per-contact-pair `[E,P]`. A global-only engine
must reject heterogeneous registration and restore data. Box3D CUDA does not
invent a per-body material combine rule. Gravity is always present: `[3]` for
global binding or `[E,3]` when registration selected per-environment gravity.
This keeps gravity values out of topology identity without making deterministic
reset depend on the destination handle's pre-restore gravity.

Registration is CPU-source-only, synchronous, and copying. Snapshot, restore,
step, and ray buffers are device pointers. A future device-source registration
path or trajectory rollout call requires a separate capability and function.
Unregister is synchronous and must wait for scene work or return `BUSY`.

## Canonical topology SHA-256

The engine returns `uint8_t topology_sha256[32]`. It validates every source
first, rejects non-finite values, and hashes this mechanical byte stream:

1. the domain bytes `world.box3d-cuda.native-topology/v2\0`, encoded as a
   little-endian `u64` byte count followed by those bytes;
2. little-endian `u32` values in order: ABI version, proposal draft revision,
   `E`, `B`, `J`, `P`, substeps, solver iterations, and material binding;
3. one byte for per-environment-gravity presence;
4. `dt` as one FP32;
5. the nine FP32 solver values in header order: warm-start factor, contact
   slop, position correction, angular damping, SAT epsilon, joint position
   slop, joint angular slop, maximum linear repair, maximum angular repair;
6. topology arrays in registration order: body caller IDs and motion; joint
   caller IDs, body indices, kinds, parent anchors, child anchors, parent axes,
   reference quaternions, lower limits, upper limits, damping, stiffness, and
   control modes; contact-pair caller IDs and body indices.

Every array has a little-endian `u64` scalar-count prefix. Integers are
little-endian. Floats are their IEEE-754 binary32 bits with either sign of zero
normalized to `0x00000000`. Mutable state, mass/inertia/extents, gravity values,
materials, and caches are excluded because snapshots carry them.

The dependency-free executable encoder is `topology_digest.py`. For the shared
two-body World vector, draft revision 1 hashes to
`06ab2f776dbf99c4dbf1a72220e6e56947fd24c1baeaa2d8068e837decaae0c4`.
Changing only the included draft-revision scalar to revision 2 correctly gives
`d664dfee5af110dc61e9ab8c3cf8568fdca1c23d501260e2c47d96f15271b34c`.
This deliberate change prevents cross-revision snapshot restore.

No proposed symbol is compiled into the shared library yet.
