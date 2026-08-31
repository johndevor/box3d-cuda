# Pinned plain-14 compiled metadata audit — r3

Scope: every numeric comparison in `open_duck_mujoco_admission_r2.py`, SHA
`6c581d973e8ee67aa8ba33aa8140361a00d84a78dc2cb066307c25208ba25521`.
Keep model/geometry/controller/solver values and runtime physical-health limits
unchanged. Original r1/r2 source and failed attempts are immutable. This audit
does not load a model or execute upstream Python, weights or pickle.

## Exact structure versus arithmetic

| Metadata checked | Compiler treatment and r3 decision |
| --- | --- |
| XML SHA/include; dimensions; body/joint/actuator/geom/site/sensor IDs, order, parent/owner; address/dimension arrays; kinds, masks, flags | Discrete structure. Keep exact equality, including actuator→joint and sensor→site maps. No numeric tolerance can hide remapping. |
| Simulation dt | Explicit candidate assignment of binary64 .002. Keep exact. |
| Gravity; body position/COM; joint anchor; site position | Parsed doubles/direct copies for this source (no `<frame>` or `fromto`). Existing 1e-9 metre / acceleration comparison unchanged. Exact copy would also satisfy it; wrong unit/material translation remains rejected. |
| Body/site quaternion-derived matrices; joint axes | Source `mjuu_normvec` normalizes, possibly skipping a norm within 1e-14 of one; checker normalizes before matrix conversion. Existing 1e-9 dimensionless envelope retained, not widened. Compare matrices/axes, never quaternion sign alone. |
| Positive masses, total authored and compiled mass | Source positive masses copied, no fusion/substitution. Keep individual 1e-10+1e-8 relative and total tolerances; the 15-term nonnegative sum has gamma14 relative rounding, far below the existing limit. Massless body/massless inertia stay exactly zero. |
| Full inertia tensors | Retain independently reviewed r2 pre-sort reproduction, domain, positivity/triangle and two-layer comparison unchanged. |
| Free damping/friction/armature; hinge damping/friction/armature | Direct double/default copies .56/.068/.027 or zeros. Keep exact. |
| Joint ranges; force limits | Source explicitly `angle=radian`; no degree conversion, range doubles copied. Keep exact values and ordering. Force limits ±3.23 are direct copies. |
| Gain/bias arrays | Fixed position gain13.37 and affine bias−13.37, kv0; direct copy/unary sign. Preserve current 1e-12 check, including every zero parameter. |
| Gear | Exact default [1,0,0,0,0,0]. No tolerance change. |
| Inherited actuator control ranges | NOT a direct joint-range copy: source builds midpoint and half-width, then endpoints. Replace only this incorrect exact-copy expectation with the bounded operation envelope below. Preserve exact inheritrange=1 and joint mapping. |
| Geom friction | Direct default copies: floor .6/.005/.0001, others1/.005/.0001. Retain 1e-12 check. |
| Mesh geom position/orientation | Source normalizes geom quaternion and composes mesh pose. For actual mesh geoms `center=0`; no primitive fit. Checker composes source geom with compiled mesh_pos/mesh_quat. Keep its current 1e-9 position/matrix tolerance; do not compare against untransformed raw STL coordinates. |
| Sensor metadata | Integer kinds, sites, order, dimensions and packed addresses; exact. Runtime sensor values are not forged or supplied by this audit. |
| Home qpos, ctrl, qvel | Source doubles copied; free quaternion [1,0,0,0] remains exact under both normalization passes. All other qpos values are scalar hinges. Keep exact. No body-pose reset or control change. |

For normalized quaternions, the skipped-normalization scale is <=1e-14;
quadratic matrix entries and short pose compositions add scalar rounding.
Dot3 uses gamma5; position error scales with operand lengths. At the pinned
robot's metre-or-smaller local scales these rounding sources are comfortably
below the existing 1e-9 engineering threshold. This audit does not claim a
new universal proof for those retained legacy tolerances. No broad
epsilon is added to direct/discrete invariants. Wrong frames/units are separate
negative controls; equality of an inertia spectrum or quaternion sign is not
permission to accept the wrong physical frame.

## Only new numeric rule: inherited range operation envelope

MuJoCo 3.3.7 `src/user/user_objects.cc:6844-6847`:

```
mean = 0.5*(high + low)
radius = 0.5*(high - low)*inheritrange
range = [mean - radius, mean + radius]
```

The pinned factor is exactly 1. Each sum/difference, half multiply, and final
sum/subtraction is interval-evaluated with one outward `nextafter` neighbour
on each bound. This encloses binary64 rounding at the operation's own scale,
including cancellation. Multiplication by one is exact. No number of ULPs is
chosen from the recorded failed endpoint. Unsupported factors, malformed/
nonfinite/overflowing or unordered source ranges reject. Returned intervals
are only metadata comparison envelopes, never new limits sent to physics.

For the recorded left-hip pitch endpoints the formula predicts the captured
control range exactly. The maximum absolute envelope deviation from source
endpoints is <7e-16 radians; material changes such as 1e-6 radians, degree/radian conversion,
wrong joint ranges and swapped endpoints must fail. Independent Fraction
arithmetic tests cover adjacent representable values and values just outside
each computed envelope, rather than testing only the previously failed field.

## Pinned source provenance and execution boundary

Relevant source excerpts (with original line selections) are archived in
`evidence/open-duck-metadata-r3/`. Whole-file hashes:

- user_objects.cc: `a1fedaace694c5b8ba364213cead4d7da4693698e4d8ce00a25f2df433fe3695`
- user_model.cc: `7c79def1c714ce7884ce85167e61696a1f7f1433d2bf930bd343f8fb9871c946`
- user_util.cc: `f9d5ef77317707039f12658e620ff3393f1eb8a0e7087111e4740207f2ee2522` (full source retained in r2)

After independent audit/tests and the exact r2→r3 delta check: ONE same-model,
same-runtime reset→1→8 CPU fixture under a 60-second parent cap. No new deps,
policy/weights, GPU/provider, native/World edits or physical-health threshold
changes. Stop on first failure; no automatic retry beyond r3.
