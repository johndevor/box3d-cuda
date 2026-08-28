# Box3D CUDA port

This directory is an early, measured port of selected Box3D ideas to CUDA for
thousands of small independent reinforcement-learning worlds. It is not a
drop-in replacement for Box3D yet.

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

Not implemented: a broad phase, general convex collision, the complete Box3D
Soft Step constraint set, articulated joints, CCD, sleep/islands, ray queries,
or GPU rendering. Stage 4 is a fixed-topology explicit-pair stacking solver;
the browser shows a deterministic CPU-oracle replay rather than live CUDA.

## Port order

1. Fixed-topology joint rows for robot links and grippers.
2. Batched ray queries for foveated depth observations.
3. Constraint graph coloring and parallel contact/joint rows.
4. Broad phase, CCD and sleep only where profiling shows they are required.
5. Touch, grasp, lift and move parity against ManiSkill before RL training.

The benchmark intentionally refuses a speedup claim when two result files do
not share a `contract_id`.
