# Native C ABI v2 proposal

`proposals/box3d_cuda_v2.h` is draft revision 1 of the resident rigid-scene
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
- fixed bodies are supported. Kinematics, per-environment gravity, per-body or
  per-pair materials, and speed/acceleration limit enforcement are separately
  discoverable capabilities and fail closed when absent;
- asynchronous step, snapshot, restore, and linear-scan OBB ray calls on a
  caller-owned CUDA stream. Registration copies before returning; unregister is
  synchronous and must wait for scene work or return `BUSY`;
- naturally aligned device pointers, no implicit strides or padded views, and
  exact shapes documented beside every pointer.

Still requiring joint review before implementation:

1. whether snapshot buffers should include randomized mass, inertia, extents,
   gravity, and materials or those values remain immutable scene data;
2. the exact topology hash algorithm (the proposal reserves four `uint64_t`
   words without claiming SHA-256 semantics);
3. whether registration should additionally accept CUDA-device source buffers
   for zero-copy initialization;
4. whether multi-step calls need per-step target tensors rather than one target
   frame held for `steps`.

No proposed symbol is compiled into the shared library yet.
