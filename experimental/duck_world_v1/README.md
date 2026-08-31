# duck_world_v1 (dwv1)

Per-environment world: one articulated Open Duck (av2, glued exactly like
idv1) + a seeded grid of cube rigid bodies + the static floor plane, batched
over E environments. C ABI prefix `dwv1_`, header
`include/duck_world_v1.h`, Python wrapper `walk/env/world.py`.

## Architecture

One `dwv1_step` = one transactional tick, mirroring idv1's phases:

1. **av2 PRE** — full duck dynamics (mass, bias, smooth velocity, spatial
   jacobians, 3J soft joint rows). av2 is reused byte-for-byte as in idv1.
2. **Geometry** — bcv1 query for the authored duck pairs (feet-floor,
   foot-foot), plus dwv1's own manifolds: feet-vs-cube, cube-vs-floor and
   cube-vs-cube (SAT with face clipping ported from contact_v1 conventions:
   f32 geometry, stable feature ids, <=4 points, see `src/dwv1_geometry.h`).
   Candidates come from a uniform 2D grid over cube centers (O(1) per foot).
   Sleeping cubes touched by an awake body wake, with fixpoint propagation
   through touching chains. Near-duplicate support points whose rows touch
   identical dynamic columns (e.g. a foot straddling two coplanar static
   cubes) are merged, keeping the deeper point — civ1 stalls on duplicated
   normal rows.
3. **Islands + solve** — union-find per env over {duck} ∪ awake cubes;
   islands merge only through dynamic bodies (cube-floor / cube-static
   contact never merges; resting non-touching cubes are singletons). Each
   island is one civ1 dense solve: block-diagonal mass = duck NxN (from av2
   PRE, duck island only) + 6x6 per awake cube (cube inertia is isotropic
   m*s^2/6, so world inertia is constant diagonal and the gyroscopic term is
   exactly zero). Joint rows + contact rows exactly as idv1; the normal
   target is the bounded penetration-repair term
   `min(1, .2*max(0, depth-2e-6)/dt)`; warm starts come from per-pair
   manifold caches keyed by stable feature ids. civ1 enforces the momentum
   residual <=1e-8 per island. Rows against static/sleeping bodies only
   touch the awake side's columns. Cubes then integrate semi-implicit Euler
   in f64 (staged), and sleep bookkeeping runs (velocity + warm-impulse-delta
   thresholds, 50-tick hysteresis; sleeping zeroes velocity).
4. **av2 complete** — duck velocity/joint impulses/contact generalized
   impulse verified independently by av2 (momentum mtol=1e-8 pinned; joint
   KKT jtol selectable, see below).
5. **Contact prepare** — bcx1 stage with post bodies + solved duck-pair
   manifolds; clock equality checked.
6. **Commit** — both foreign owners validate, then commit; the staged cube
   payload (poses, velocities, sleep state, warm caches, foot flags) swaps in
   the same critical section. Any failure anywhere leaves everything
   unchanged and the diagnostic names the first failing environment/phase.

Static grids (`dynamic=0`) are permanently-asleep fixed cubes: only feet-cube
manifolds are generated, rows touch duck columns only, one island per env.

Cube layout: lattice `nx x nz` (world x/y), pitch `spacing`, centered on
(`origin_x`,`origin_y`), cube center z = `base_height + jitter + cube_size/2`
with `jitter = height_jitter * u(seed, ix, iz)` from splitmix64 — fully
deterministic per seed and identical across envs.

## API

See `include/duck_world_v1.h` for the full contract. Summary:

- `dwv1_create(const dwv1_registration*, dwv1_scene**)` — av2 registration
  reused verbatim + duck bcv1 shapes/pairs/friction (as idv1) + `dwv1_grid`
  {nx, nz, dynamic, cube_size, spacing, base_height, height_jitter, origin,
  cube_mass, friction, seed}. Feet are auto-detected as the convex non-fixed
  duck shapes (Open Duck: bodies 6 and 15). Floor must be shape 0, plane
  z=0, +z up. Caps: nx*nz <= 1024 cubes, E*M <= 262144.
- `dwv1_step(scene, const av2_step*, max_iterations, impulse_tolerance,
  dwv1_diagnostic[E])` — diagnostic per env: phase, native_status,
  iterations, contact_points, active_limits, islands, awake_cubes,
  duck_island_cubes, max_island_dofs, residuals (joint/normal/tangent/
  momentum), maximum normal impulse and penetration.
- `dwv1_read(...)` — duck qpos/velocity/warm/time/step_count + bcv1
  bodies/cache/geometry (as idv1) + cube poses [E,M,7], velocities [E,M,6],
  awake flags [E,M], foot contact flags [E,F] from the last accepted step.
- `dwv1_query(scene, env, out, capacity, count)` — current-pose cube-side
  manifolds (foot-cube, cube-floor, cube-cube), zero impulses, no side
  effects.
- `dwv1_override_cube(scene, env, cube, pose[7], velocity[6])` — test/setup
  teleport for dynamic cubes only; wakes the cube, clears its warm caches.
- `dwv1_capture / dwv1_restore / dwv1_reset` — masked, bit-exact, same
  two-owner discipline as idv1 plus the cube payload.

## Build

Exactly run_local.py's flags (also wrapped by `walk.env.world.build()`):

```
clang++ -std=c++17 -Wall -Wextra -Werror -ffp-contract=off -O2 -fPIC -shared \
  -I experimental/duck_world_v1/include -I experimental/integrated_duck_v1/include \
  -I experimental/contact_v1/include -I experimental/articulated_v1/include \
  -I experimental/articulated_v2/include -I include -I csrc \
  experimental/duck_world_v1/src/duck_world_v1.cpp \
  experimental/integrated_duck_v1/src/coupled_impulse_v1.cpp \
  experimental/integrated_duck_v1/src/integrated_duck_v1.cpp \
  experimental/contact_v1/src/contact_v1.cpp \
  experimental/articulated_v1/src/articulated_v1.cpp \
  experimental/articulated_v2/src/articulated_v2.cpp \
  csrc/experimental_joint_v1.cpp \
  -o libduck_world.dylib
```

## Tests

```
.venv/bin/python -B experimental/duck_world_v1/tests/test_static_grid.py
.venv/bin/python -B experimental/duck_world_v1/tests/test_dynamic_cubes.py
.venv/bin/python -B experimental/duck_world_v1/tests/test_capacity.py
```

- static: rest within 1s + 2s upright hold (tilt < 2 deg), jittered-grid
  contact + seed determinism, bit-identical trajectories, masked
  restore/reset exactness, transactional failed-step no-op; momentum
  residual <= 1e-8 asserted every accepted step.
- dynamic: 5x5 drop settles and all cubes sleep; pushed cube slides
  ~v^2/(2 mu g) and stops; two-cube stack stable 2s and sleeps; island
  partition counts on a constructed scene; duck standing on dynamic cubes
  pulls them into its island and stays upright.
- capacity: 225 cubes static and dynamic-sleeping, < 10 ms/tick
  single-threaded, island dof caps asserted far below civ1's 256/1536/512.

## Known limitations / interim choices

- **civ1 stall (workstream A)**: islands with redundant nearly-parallel
  normal rows (duck resting across stepped cube tops; duck+cube+floor
  couplings) can stall the civ1 sweep just above 1e-8. Static milestone-1
  configurations in the tests converge at impulse tolerance 1e-8; coupled
  dynamic scenarios currently run at 1e-6 with av2 jtol=1e-6 (av2_step
  legally accepts up to 1e-5). The momentum residual stays <= 1e-8
  (machine-precision in practice) and is asserted per step. Expected to
  return to 1e-8 across the board once workstream A's civ1 repair lands.
- **Sleep churn under load**: a sleeping cube touched by the duck wakes the
  next tick (contact-wake rule), so load-bearing cubes cycle sleep/wake with
  a ~51-tick period instead of staying asleep. Physically consistent, keeps
  islands small; just not maximally efficient.
- **Geometry fallbacks** (ported SAT hardened for hull-vs-cube): grazing
  edge-edge axes without support-band witnesses and empty face-clips fall
  back to widened support search / deepest-vertex witness, and only
  penetrations deeper than 8*EPS without any witness remain hard failures.
- Milestone-2 items implemented but only lightly covered by tests: warm-start
  quality across sleep/wake transitions, tall stacks (>2), dense dynamic
  rubble piles (spacing == cube_size dynamic grids are validated only via
  the 225-cube resting/holding capacity gate).
