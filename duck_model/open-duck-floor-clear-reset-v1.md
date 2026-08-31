# Floor-clear reset candidate v1 — not unchanged upstream transfer

Bounded authorization: one separately named reset candidate, then one same-model
CPU reset→1→8 controller-fixture attempt under a parent 60-second hard cap.
No policy, weights, new dependencies, GPU, native ABI or World changes.

The original home key remains z=0.15 m. Its r3 failure (16.888275 mm floor
penetration) is immutable, not reclassified by this candidate. Root Z is the
only reset input changed. Every joint angle, quaternion, velocity, control,
model parameter, timestep, solver setting and physical-health gate is retained.

## Derivation fixed before execution

For every collision-enabled robot mesh, transform **compiled** `mesh_vert`
vertices by `data.geom_xmat` and `data.geom_xpos` at exact source home FK.
Subtract the fixed floor origin and project onto its unit normal. Take the
minimum. Linear support on the vertices equals support on their convex hull.
Do not use rendered bounds/raw STL coordinates, and do not reapply compiler
mesh transforms. Check every collider, not just whichever foot looks lowest.
This exact model has two foot collision meshes and a plane; all other 44 geoms
are noncolliding visuals. Any added/changed collider topology fails closed.

Fixed clearance: **0.001 m**, declared before any candidate simulation.
For the admitted horizontal +Z floor:

`lift = max(0, 0.001 - minimum_signed_distance)`

This is one closed-form minimal nonnegative vertical translation, not a search,
optimization or tuning of the gait. FK/contact queries before/after the change
do not advance time. The original model key is never overwritten.

Both support distances must agree with MuJoCo 3.3.7 `mj_geomDistance` before and
after translation within 1e-9 m, a numerical geometric agreement bound separate
from the unchanged 0.01 m physical penetration gate. Query cutoff1m; require
analytic distance strictly within0.5m and no cutoff/nonfinite result. Also
check foot-to-foot distance stays positive and invariant. A root shift cannot
repair self-collision. Fixed floor, +Z normal, zero margins/gaps, default contact
filter, no explicit pairs/exclusions/flex objects, root ancestry and exact source
home are required. Subsequent forward must retain qpos/qvel/ctrl/time and the
original key; sensor/acceleration/contact outputs may change physically.

Pinned upstream source support: [plane/convex collision](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_convex.c#L1041)
and [geom distance dispatch](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_support.c#L447).
The former selects the negative-normal support vertex and measures its signed
distance to the plane. The latter returns the minimum contact distance using
the same collision function with a query cutoff; it does not change geometry.

The candidate report preserves compiled vertices, transforms, supporting vertex
IDs, per-foot analytic/native distances and original/candidate complete qpos.
An independent offline calculation can therefore test the recorded arithmetic.
Geometry validation is not dynamic admission. The fixture must separately pass
all unchanged physical, controller, sensor, warning and clock checks at reset
and each of eight 20ms frames. Stop on first failure without another retry.

## Ownership

Root owns `scripts/open_duck_floor_clear_reset.py`, the separately named runner,
delta tests, documentation and evidence. Rawls owns independent scalar geometry
tests; Boole reviews source/geometry/model/runner boundaries without executing
a simulator. No original r1/r2/r3 file or evidence may be changed.
