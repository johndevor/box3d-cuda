# Experimental native contact v1 — opt-in CPU implementation

This separately built C++17/C ABI library implements convex/convex and
convex/infinite-plane contacts, explicit per-pair materials, frame/velocity
translation, contact rows and owned batched snapshots. It does not alter or
link into frozen r3/machine-v1. No published capability mask is changed.
The old per-pair enum2/bit12 is not advertised by this experimental CPU ABI.

## Implemented

- Up to32 input convex vertices; registration constructs bounded hull faces
  and genuine edges. SAT uses face normals/edge crosses and signed exit depth;
  clipping retains the winning reference face. Edge contact uses closest
  actual support-edge segments, not an arbitrary extreme-point midpoint.
- An oriented infinite half-space `normal.dot(x)<=offset`. Near-unit input
  normals are normalized together with the offset, preserving the plane.
  Convex vertices are baked with their exact local pose into principal COM
  coordinates. No OBB or finite-floor proxy.
- At most4 spatially spread manifold points, deterministic feature IDs and
  order. Inelastic normal impulses; a conditional anisotropic 2D friction
  quadratic constrained to `|jt|<=mu*jn` is solved by bounded scalar bisection.
  Global contact solving is fixed-iteration sequential impulses, not an exact
  simultaneous or MuJoCo soft-contact solve. Per-pair effective mu is copied
  independently for each environment; there is no guessed material combine.
- State `[E,B,13]`: COM/principal xyz/xyzw plus WORLD linear/angular velocity.
  Native conversion uses `vCOM=vOrigin+omega cross R*localCOM`; rotations are
  normalized after validating a near-unit input. Inverse conversion included.
  Pure joint coordinate/qdot/anchor geometry utilities do not solve joints.
- Synchronous copied topology/state, full capture and atomic full/nonzero-byte
  masked restore. Snapshot includes body state/inverse mass/inertia, gravity,
  mu, feature/impulse caches and clocks. Query and failed step/restore do not
  partially publish output or owned state. Immutable topology equality compares
  canonical scalar words, including caller IDs/order, not a lossy numeric hash.
- Shared math uses host/device annotations for vectors, inertia and point
  response. Only host compilation is verified. Hull construction, orchestration
  and lifecycle currently use host STL and are not GPU-resident implementations.

The convenience step solves at PRE poses, then integrates once. Cache returned
by read belongs to PRE; `query` computes current POST geometry. Free rotation
updates orientation and preserves world angular momentum. Contact depth bias
is0.2*max(depth−2e−6,0)/dt capped1m/s. No restitution, CCD, articulated solve,
joint limits, gyro-accurate high-order integrator, or full robot rollout is
claimed. Existing velocity can cross a contact between steps; this is not a
tunnelling-proof integrator. Warm-start impulses are transported to the new
tangent basis or invalidated after normal/point changes.

## Build and reproduce

Use existing clang and Python; no Torch, MuJoCo, NumPy or new dependency is
needed for these native tests. CMake target is provided independently but is
not required on a Mac without CMake/CUDA. From the repository root:

```
python3 -B experimental/contact_v1/run_local.py --output /absolute/new/run-directory
```

The file-backed runner compiles the shared library, C++ tests, C11 layout
consumer and sanitized test executable; runs independent response/geometry
tests and exact recorded-model translation; stops at the first failure. It
refuses existing output and records all sources/commands/logs/artifact hashes.
The model translator currently uses the separately preserved, hash-pinned
reference directory from the local OpenDuck worktree; its Model constructor
accepts an explicit alternate reference path. This package does not download
assets or bundle the historical501-state recording. This dependency is not
permission to rewrite that sealed evidence.

## Admission limits and f32 meaning

E1..4096, B1..32, P0..16; one collider per body, no self/duplicate or plane-plane
pairs, no fixed-fixed pairs. Dynamic inverse mass positive≤1e6; local diagonal
inverse inertia positive≤1e8. Fixed mass/inertia/velocities must be zero.
No kinematic-body semantics. Convex local vertices norm≤100m, diameter
1e−4..100m; duplicate/degenerate/nonclosed/ill-conditioned hulls reject, maxima
60faces/90edges. Query clipping scratch≤64points. Body p/v/omega norms≤1e4,
gravity≤100m/s², mu0..4, dt(0,.01], iterations1..128. Row adapter N≤64,
finite spatial Jacobian entries≤1e6. All callers own valid sized host buffers
and serialize handle access; stale pointers/concurrent destruction are outside
this experimental contract.

2e−6m is a contact classification/merge tolerance, NOT a global world-space
accuracy promise at coordinates1e4m. Measured robot gates use its actual
sub-metre scale. f32 test bounds use unit roundoff2^-24 and conservative
128/256-operation error envelopes on input magnitudes, distinct from the
f64 frozen1e−9 reference checks. No robot health threshold is changed.
Source velocity validation uses analytic FK and independent central-difference
kinematic derivatives (not another physics rollout). The MuJoCo3.3.7 convention
is [world free linear, local free angular](https://mujoco.readthedocs.io/en/3.3.7/overview.html#floating-objects).

## Exact joint-lane seam

`bcv1_query` supplies contact point, A→B normal/tangents, depth and feature.
`bcv1_contact_row` consumes `[B,6,N]` WORLD COM spatial Jacobians (linear first)
and produces generalized row `g=JA^T[-d,-rA×d]+JB^T[d,rB×d]`. Virtual-work and
point-velocity tests cover this map, including the pinned robot's20-DOF
kinematic Jacobians. This is not an integrated articulated simulation.

CUDA3's response operator must use `(M_body+diag(armature))^-1*g` and propagate
the result to ALL DOFs/bodies. Default `bcv1_step` uses standalone free-body
response and MUST NOT be used to claim armature-aware robot parity. Fixed
independent box bounds for external tangent rows do not represent a Coulomb
disk whose radius depends on the unknown normal force. Coupled cone/soft
friction orchestration, joint-state reset coordination and a full robot step
remain unverified. Neither lane silently mutates the other's state/cache.
