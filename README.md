# Box3D CUDA

This standalone repository is an early, measured port of selected Box3D ideas to CUDA for
thousands of small independent reinforcement-learning worlds. It is not a
drop-in replacement for Box3D yet.

The Python package preserves the original `box3d_cuda` API and lazily builds
its PyTorch extension. A separate versioned native C ABI has no PyTorch
dependency. ABI v1 intentionally exposes only the accepted Stage-0 sphere
step. The frozen ABI-v2 revision 3 boundary adds capability-gated resident
fixed-small OBB scenes, coupled joint/contact stepping, deterministic snapshot
and masked environment restore, and linear OBB rays; unsupported capabilities
fail closed. See [docs/native-c-abi.md](docs/native-c-abi.md) and
[docs/native-c-abi-v2-proposal.md](docs/native-c-abi-v2-proposal.md).

The upstream source is pinned in `UPSTREAM.json`. Stage 0 maps Box3D's
quaternion integration to CUDA and implements physical plane and sphere-pair
impulses. A dependency-free CPU oracle checks the GPU result before any timing
is accepted.

Stage 1 adds a fixed-topology parallel-jaw workload: two driven box fingers
close on a dynamic box, lift it through Coulomb friction, open, and prove that
the released box falls. There is no attachment flag, weld, or cube pose-copy.

Stage 2 adds truly oriented boxes against a plane: rotated vertices,
world-space inertia, contact-point velocity, angular impulses, a compact
multi-point manifold, two-axis Coulomb friction, quaternion integration, and
angular damping. The CPU oracle and CUDA kernel execute the same contract.

Stage 3 adds oriented box-to-box collision with all 15 separating axes. The
CPU oracle and CUDA kernel share the same explicit-pair contract, including
near-parallel-axis handling, world-space inertia, angular contact velocity,
restitution, Coulomb friction, bounded position repair, and a one-point
support-midpoint contact.

Stage 4 adds persistent oriented-box manifolds with stable feature IDs, up to
four points per explicit pair, warm-started normal and two-axis friction
impulses, and bounded split position repair. Its fixed scene is a four-box
stack plus an independent friction slider over 720 control steps.

Stage 5 adds maximal-coordinate fixed, revolute, and prismatic joint rows,
bounded velocity and position motors, joint limits, split anchor/orientation
repair, and an explicit eight-row warm-start cache per world and joint. The
matched throughput workload is a fixed base plus six revolute links with no
collision shapes; generic fixed/prismatic behavior is covered by separate
CPU/CUDA micro-gates.

Stage 6 adds a native batched nearest-hit ray query for up to 32 oriented boxes
per fixed-small world. It returns range, stable body ID, and an outward unit
normal with deterministic miss and tie semantics. A CUDA camera compiler now
generates calibrated rays for up to 64 heterogeneous world- or body-attached
pinhole cameras and converts hit range to optical-axis depth without a host
pixel loop. The scalar CPU oracle independently covers the same camera pose,
ray, range, miss, and optical-depth semantics, plus planes and hit positions.

Stage 7 couples articulated joint rows and persistent contact rows in one
fixed-small-world solver. Its CPU/CUDA physical correctness gate is accepted.
Strict instantaneous output parity against the matched ManiSkill/PhysX impact
trace is still rejected, so no Stage-7 speedup claim is accepted.

## Measured stages

- State: position, quaternion, linear velocity, angular velocity
- Shapes: dynamic spheres and an infinite static plane
- Contacts: normal restitution, tangential Coulomb friction, position repair
- Batching: one CUDA thread per fixed-small world
- Validation: finite state, quaternion norm, ground penetration, CPU/GPU error

Stage 1 additionally measures:

- Shapes: one dynamic axis-aligned box and two driven axis-aligned box fingers
- Gates: touch, bilateral contact, lift, release, post-release fall
- Validation: CPU-oracle parity and independent PhysX contact-force evidence
- RTX 5090 result: 43.83M world-steps/s versus ManiSkill's 1.212M on the same
  4,096-world, 344-step contract (36.16×)
- Negative control: identical geometry and motion at zero friction touches the
  cube but does not lift it

Stage 2 additionally measures:

- Shapes: eight independently tumbling oriented boxes per world
- Gates: every box contacts, every box gains angular velocity, finite state,
  normalized quaternions, final rotated-corner clearance, and CPU/CUDA parity
- RTX 5090 result: 25.65M world-steps/s (205.16M body-steps/s) versus
  ManiSkill/PhysX's 0.428M world-steps/s on the same 4,096-world, 500-step
  contract (59.92×)
- CPU/CUDA maximum absolute state error: 4.92e-7

Stage 3 additionally measures:

- Shapes: six independently moving oriented boxes in three explicit pairs
- Scenarios: face, rotated-face, and edge-dominant box impacts
- Gates: all 15 SAT axes, all pairs contact, zero final penetration, finite
  state, normalized quaternions, rotated/edge angular response, and CPU/CUDA
  state/contact/penetration parity
- RTX 5090 result: 65.64M world-steps/s (393.85M body-steps/s) versus
  ManiSkill/PhysX's 0.490M world-steps/s on the same 4,096-world, 500-step
  contract (133.95x)
- CPU/CUDA maximum absolute state error: 2.83e-6; maximum penetration error:
  2.21e-8 m

Stage 4 additionally measures:

- Shapes: a static floor, four dynamic stack boxes, and one independent slider
  with five explicit persistent contact pairs per world
- Gates: stable feature IDs and contact counts, cached normal/tangent impulses,
  CPU/CUDA state and penetration parity, settled stack contacts, slider friction,
  and independent PhysX contact-force evidence
- RTX 5090 result: 7.612M world-steps/s (45.67M body-steps/s) versus
  ManiSkill/PhysX's 0.382M world-steps/s on the same 4,096-world, 720-step
  contract (19.93x)
- CPU/CUDA maximum absolute state error: 1.37e-6; maximum penetration error:
  1.58e-7 m; maximum cached-impulse error: 7.64e-8 N·s
- Zero failing CUDA worlds; minimum final contact counts per pair: 2/3/3/3/4

Stage 5 additionally measures:

- Topology: one fixed base, six revolute joints, seven rigid bodies, bounded PD
  drives, explicit inertia/limits, and no collision shapes
- Gates: CPU/CUDA state, diagnostic, and warm-cache parity; deterministic
  state/cache replay; cross-world isolation; limits; quaternion normalization;
  anchor drift; and warm-start utility under an adversarial one-iteration solve
- RTX 5090 result: 6.091M world-steps/s (42.64M body-steps/s) versus
  ManiSkill/PhysX's 0.185M world-steps/s on the same 4,096-world, 720-step
  contract (32.85x)
- CPU/CUDA state and cache errors: exactly zero; maximum diagnostic error:
  5.96e-8
- Warm-start stress drift: 14.57 mm versus 22.78 mm cold, with bounded energy
  and finite explicit cache rows

Stage 6 additionally measures:

- Scene: eight exact finite OBBs, including a ground slab and rotated boxes
- Sensor layout: two ordered 8×16 calibrated ray rigs, 256 rays per world
- Gates: 100% CPU/CUDA hit-ID and miss agreement, 1.0 minimum normal cosine,
  all eight IDs observed, deterministic replay, and bit-exact world isolation
- RTX 5090 result: 21.192B native first-hit OBB ray queries/s across 1,024
  worlds and 240 query steps; maximum CPU/CUDA range error 1.91e-6 m
- Post-extraction RTX 5090 camera gate: two heterogeneous scene/wrist cameras
  across two isolated worlds passed CPU/CUDA origin, direction, hit-ID, range,
  optical-depth, body-attachment, miss-zero, normal, and deterministic-replay
  checks; maximum depth error was 4.77e-7 m and range error 9.54e-7 m
- End-to-end RTX 5090 analytic camera result: 1,024 worlds, two 16×16 cameras,
  eight OBB slots, and 240 frames completed in 0.007688 s: 16.366B depth
  pixels/s or 63.930M camera frames/s. The timed interval includes CUDA camera
  pose/ray compilation, linear nearest-hit queries, and optical-depth conversion
- No PhysX speedup is claimed: PhysX 5's documented batch-query extension wraps
  scene raycasts and ManiSkill exposes no comparable native CUDA tensor batch

The Stage 6 browser view is a bounded CPU-oracle debug artifact linked by
SHA-256 to the measured CUDA result. It visualizes both camera bodies, sampled
rays, hit points, normals, and range previews; it is not live CUDA, RGB, or
training evidence.

Stage 7 currently establishes:

- A fixed-base two-link arm physically pushes a free payload on a floor using
  motor drives and contact impulses only
- No attachment state, pose copying, teleportation, or direct payload force
- Deterministic replay, world isolation, real contact, release, friction
  negative controls, finite state, normalized quaternions, and bounded caches
- CPU/CUDA correctness for the bounded solver implementation
- The strict PhysX parity gate remains failed at the first impact; iteration
  sweeps from 4 through 32 did not resolve the structural response mismatch
- No throughput comparison is accepted until that parity gate passes

Not implemented: a broad phase, general convex collision, the complete Box3D
Soft Step constraint set, reduced-coordinate articulation, CCD, sleep/islands,
meshes, or GPU raster rendering. Stage 6 analytic camera queries currently test
OBBs with a linear per-world body scan; there is no acceleration structure,
RGB, textures, lens distortion, rolling shutter, or sensor noise model.
Stage 7 is a bounded maximal-coordinate joint/contact workload, not general
robot dynamics or validated sim-to-real physics.

## Repository boundaries

- `csrc/`: CUDA kernels and the PyTorch and native-C entry points
- `include/box3d_cuda/`: the versioned public C ABI
- `*_reference.py`: dependency-free CPU oracles
- `contracts/`: exact correctness and cross-backend parity contracts
- `benchmarking/` and `benchmark_*.py`: fail-closed evidence and workloads
- `tests/`: standalone CPU, source-contract, and optional CUDA gates
- `UPSTREAM.json` and `LICENSE.box3d`: pinned Box3D provenance

Factory OS scene orchestration, assets, GPU-provider lifecycle, ManiSkill
adapters, dashboards, and browser presentation intentionally remain in
`box3d-arm-lab`, which consumes this repository at an exact Git commit.

## Port order

1. Resolve Stage-7 impact response parity without loosening the accepted gate.
2. Constraint graph coloring and parallel contact/joint rows.
3. Add a broad phase for larger per-world scenes, then profile CCD and sleep.
4. Feed calibrated multi-rig depth packets into the staged RL observation API.
5. Touch, grasp, lift and move parity against ManiSkill before RL training.

The benchmark intentionally refuses a speedup claim when two result files do
not share a `contract_id`.

See [docs/verification.md](docs/verification.md) for the extraction regression
record and exact GPU run IDs.
