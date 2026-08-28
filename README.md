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

Not implemented: oriented/general convex collision, angular contact response,
broad phase, full Box3D Soft Step contact constraints, articulated joints, CCD,
sleep/islands, ray queries or rendering.

## Port order

1. Fixed-topology joint rows for robot links and grippers.
2. Oriented box/capsule support and a feature-pair narrow phase.
3. Batched ray queries for foveated depth observations.
4. Constraint graph coloring and parallel contact/joint rows.
5. Broad phase, CCD and sleep only where profiling shows they are required.
6. Touch, grasp, lift and move parity against ManiSkill before RL training.

The benchmark intentionally refuses a speedup claim when two result files do
not share a `contract_id`.
