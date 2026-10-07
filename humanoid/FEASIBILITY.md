# H0 humanoid on the duck stack — Phase 1 feasibility memo

Status: COMPLETE (Phase 1 record; see humanoid/PHASE2.md for the Phase 2
bring-up). Verdict: **pure regeneration** for the Phase 1 physics lane
(both the f64 CPU oracle and the fp32 serial kernel build) — zero
kernel-source edits; all gates pass (section 7). Since then, upstream
landed the workstream-A solver repair (section 6, now resolved) and the
per-joint effort-cap kernel edit (section 2 item 1, done); the remaining
kernel edits (section 2 items 2-4, in-kernel policy layer) are still
pending — the Phase 2 env/reward run on the CPU lane meanwhile.

Scope: can the frozen H0 humanoid (14 bodies incl. floor, 12 revolute
joints, 2 box-foot/floor contact pairs — the B14/J12/P2 contract of
`/Users/john/Code/box3d-arm-lab/factory_os/independent_validation/humanoid_h0.py`,
line 94: `(registration["bodies"], registration["joints"],
registration["contact_pairs"]) != (14, 12, 2)`) run on this repo's stack
(CPU oracle lane `integrated_duck_v1` and fp32 kernel
`experimental/duck_cuda/src/duck_cuda_kernel.h`) by pure model
regeneration, or does it need kernel edits?

## 1. What is duck-specific in the fp32 kernel

Sources read: `experimental/duck_cuda/src/duck_cuda_kernel.h` (1801 lines),
`experimental/duck_cuda/include/duck_model.h`,
`experimental/duck_cuda/include/duck_cuda.h`,
`experimental/duck_cuda/src/duck_cuda_serial.cpp`,
`experimental/duck_cuda/tools/generate_model.py`.

### 1.1 Compile-time dimensions (all from the generated `duck_model.h`)

| macro | duck | H0 humanoid | consumed by |
|---|---|---|---|
| `DW_B`  | 16 | 14 | every FK/mass/bias loop |
| `DW_J`  | 14 | 12 | joint rows, PD, policy layer |
| `DW_N`  | 20 (6+J) | 18 | solver, jacobians |
| `DW_Q`  | 21 (7+J) | 19 | state, integrate |
| `DW_JROWS` | 42 (3J) | 36 | warm-start slots |
| `DW_PAIRS` | 2 | 2 (identical) | contact loops |
| `DW_MAXPOINTS` | 4 | 4 | manifold reduce |
| `DW_MAXROWS` | 66 | 60 | `static_assert(DW_MAXROWS == DW_JROWS + 3*DW_PAIRS*DW_MAXPOINTS)` at kernel line 707 — holds by construction |
| `DW_FOOT_VERTS` | 18 (baked convex sole) | 8 (box corners) | `dw_plane_manifold`, `dw_sole_heights`, `dw_reduce` scratch — all loop over the macro; `contact_v1` and the kernel accept any convex with 4..32 vertices |

Every physics loop in the kernel is written against these macros; none of
the physics hardcodes 14/16/20/21/18. Duck numerals appear only in
comments ("14 hinge angles", "20-column") and in the *policy layer* (below).

### 1.2 Structural assumptions of the kernel physics (must hold for H0 — they do)

1. **Topology**: body 0 = fixed floor, body 1 = single free 6-dof root,
   child of hinge j is body j+2 (`int b = j + 2` in `dw_evaluate`,
   `dw_body_states`); parent per joint is a free table (`DW_HINGE_PARENT`).
   The same rule is enforced by the CPU lane
   (`articulated_v1.cpp:31  m->bodies!=m->joints+2`,
   `articulated_v2.cpp:82`, joints <= 26, bodies <= 32 in idv1).
   H0's frozen ordering satisfies it *exactly*: `BODY_NAMES[i+2]` is the
   child of `JOINT_NAMES[i]` for all 12 joints (humanoid_h0.py lines 24-53:
   waist→torso(2), neck→head(3), left_hip→left_upper_leg(4), left_knee→(5),
   left_ankle→left_foot(6), right_hip→(7), right_knee→(8),
   right_ankle→right_foot(9), left_shoulder→(10), left_elbow→(11),
   right_shoulder→(12), right_elbow→(13)). No renumbering needed.
2. **Revolute-only joints**: `Hinge` is the only joint record in
   `articulated_v1.py`/`duck_model.h`; H0 is all-revolute
   (humanoid_h0.py line 141: `"kind": "revolute"`). OK.
3. **Floor plane hardcoded z-up at z=0**: `dw_plane_manifold` uses
   `n = {0,0,1}`, separation `world[2]`; gravity is `DW_GRAVITY_Z` on the
   z axis only. The H0 bundle is authored **+Y-up** (humanoid_h0.py
   line 113: `"up_axis": "+Y"`). Resolution: rotate the *reset states only*
   by Rx(+90°) (y→z, z→−y; forward +X preserved) at lowering time.
   Joint anchors, parent-local axes, reference quats, principal inertias
   and half-extents are all body-frame-local and are **unchanged** by a
   world-frame rotation; only the root reset pose/velocity and gravity
   direction live in world frame. Pure data transformation, no kernel edit.
4. **Exactly 2 foot-floor pairs, body_b fixed**: `DW_PAIR_BODY_A/B` are
   tables; H0 pairs are floor(0)-left_foot(6), floor(0)-right_foot(9)
   (humanoid_h0.py lines 163-166). The duck's third pair (foot-vs-foot)
   exists only in the CPU lane; the kernel already models exactly 2
   foot-vs-floor pairs. H0 matches the kernel *more* closely than the duck
   does (no self-collision: humanoid_h0.py line 162).
5. **Box inertia from half-extents**: H0 validates
   `I = m/3 * (h_j^2 + h_k^2)` per body (humanoid_h0.py lines 284-291),
   i.e. bodies are principal-axis-aligned boxes; the fixture's
   `Body.inertia[3]` principal triple and identity `root_inertia`
   (source frame == principal frame, `[0,0,0, 0,0,0,1]`) represent this
   exactly. The duck needed a nontrivial source→principal transform
   (`DW_ROOT_COM`, `DW_ROOT_QPC`); the humanoid's is identity.

### 1.3 The C ABI (`duck_cuda.h`) — fits without edits

`dwc1_info` hardcodes `home_qpos[21]`, `joint_lower[14]`, `joint_upper[14]`
(duck_cuda.h lines 70-75). These are **max-size fixed arrays**; a J=12,
Q=19 build writes prefixes and never overruns. `dwc1_set_state`'s
`qpos21/velocity20/warm42` names are duck-lengths in name only; the
implementations use `DW_Q/DW_N/DW_JROWS`. The ctypes wrapper for the
humanoid must read dims from `dwc1_info_get` instead of assuming 14/21.

### 1.4 What is genuinely duck-specific: the in-kernel policy layer

`dw_policy_observe` / `dw_policy_reward` / `dw_step_policy_env`
(kernel lines 1494-1799) are a verbatim f64 port of `walk/env/flat.py` +
`walk/env/reward.py` **for the duck**:

- `DWP_OBS 58` and hardcoded observation offsets 14/28/42/45/48/51/54/56
  (kernel lines 1502, 1565-1591) — the layout is 3*J+16 with J=14 baked
  into the indices. With J=12 the code still compiles and writes 58 floats,
  but slots 12-13/26-27/40-41 are stale and the layout is wrong for a
  12-joint robot.
- `DW_REF_GAIT[64][DW_J]` self-imitation table (duck reference gait from
  `walk/env/reference_gait.json`) — no humanoid equivalent exists.
- Termination thresholds (0.7×home height, 45° tilt, 400-step horizon),
  `DWP_ACTION_SCALE 0.25`, slew 0.1048 rad — duck-tuned flat.py constants.
- Reward weights `DW_RW_*` — generic scalars, but tuned for the duck gait.

**None of this is needed for Phase 1**: home-hold parity uses only
`dwc1_create/dwc1_step/dwc1_read/dwc1_reset/dwc1_set_state/dwc1_query`,
which never touch the policy layer. The generated humanoid header just has
to *define* the `DW_RW_*`/`DW_REF_GAIT`/`DW_PHASE_HZ_*` symbols so the
translation unit compiles (we emit the reward.py scalars unchanged and an
all-zero 64×12 `DW_REF_GAIT`, clearly marked "policy layer NOT valid for
humanoid — Phase 2 work").

## 2. Verdict

**Pure regeneration is sufficient for the Phase 1 physics lane** (CPU
oracle *and* fp32 serial kernel): generate a humanoid `duck_model.h`
(same macro names, humanoid values, `DW_FOOT_VERTS 8`) into
`humanoid/include/` and compile `duck_cuda_serial.cpp` with
`-Ihumanoid/include` ahead of `-Iexperimental/duck_cuda/include` so the
humanoid header shadows the duck one. Zero kernel-source edits; the duck
build and its committed header are untouched.

**Kernel edits ARE required for Phase 2** (training on device):
1. **Per-joint effort caps** — DONE upstream: the kernel now consumes
   `DW_EFFORT_CAP_TABLE[DW_J]` at the PD clamp and the reward torque
   estimate (duck bit-identity proven there); the scalar `DW_EFFORT_CAP`
   remains only as the `dwc1_info` summary field. (Original finding: H0
   authors tiers 180/140/70, humanoid.rs:778-784; Phase 1 baked the MIN
   and proved the clamp never binds in its gate regimes. The CPU oracle
   always honored per-joint caps natively, av1 Hinge.cap.)
2. `dw_policy_observe`: obs width 3*J+16 = 52 and J-derived offsets
   (currently hardcoded duck offsets 14/28/42… and `DWP_OBS 58`).
3. Humanoid `DW_REF_GAIT` (no humanoid reference gait exists — Phase 2
   must author one) or compile-out of the imitation term.
4. flat.py-equivalent humanoid env constants (action scale, slew,
   termination height/tilt, commands, phase clock) decided and generated —
   including the tilt/termination formula: the humanoid's up axis is body
   +Y (up = 2*(qy*qz + qx*qw)), not the duck's body-Z
   `1 − 2(qx²+qy²)` baked into dw_step_policy_env.
5. Cosmetic: `dwc1_read`'s `time` output hardcodes `count * 0.002` in
   duck_cuda_serial.cpp (physics itself uses DW_DT); the humanoid lane
   recomputes time host-side meanwhile.
6. `dwc1_info`'s fixed 14/21 arrays are max-size and already fit J=12 —
   no change, but wrappers must read dims from dwc1_info (the humanoid
   lane does).

The kernel specialist owns 1-4 when Phase 2 starts; they are localized to
the PD/actuator lines, the policy-layer section (lines 1494-1799) and
`generate_model_humanoid.py` emission.

## 3. Constant sourcing

There is exactly ONE authoring site for the H0 morphology:
`/Users/john/Code/world/crates/sim/src/humanoid.rs`
(`world.humanoid.planar_13_link-v1`, lines 28-31). Everything else is a
consumer, validator or materialization of it. Per-constant citations live
in `humanoid/h0_lowering.py`; summary:

| category | authoritative source | value |
|---|---|---|
| bodies: centers, box scales, masses | humanoid.rs:50-162 | 13 links + floor; 68.0 kg total dynamic (invariant test humanoid.rs:934-955) |
| principal inertias | DERIVED, never authored: world_slice.rs:2496-2521 solid box `m*(s_j^2+s_k^2)/12` on full extents == `m*(h_j^2+h_k^2)/3` on half-extents (independently re-derived by humanoid_h0.py:284-291) | computed in the lowering |
| foot geometry | humanoid.rs:99-106/123-130 + 709-713 (single OBB per body) | half-extents [0.23, 0.07, 0.14], centers [0.12, 0.07, ±0.15], 1.5 kg |
| floor | humanoid.rs:51-58 (finite box, top face y=0) | lowered to the stack's infinite plane z=0 (documented) |
| joints: parent/child, anchors, limits | humanoid.rs:185-318 | 12 `joint(...)` rows; child(j) = body j+2 |
| joint axes / reference quats | humanoid.rs:764-765 | ALL `[0,0,1]` / identity — strictly planar sagittal model |
| PD gains | humanoid.rs:787-788 | kp=90, kv=8 uniform; drive law torque = clamp(kp·(t−q) − kv·q̇, ±effort) per native-flatfloor-readiness.md:26-33, identical to av2's motor law (articulated_v2.cpp:107-109) |
| effort limits | humanoid.rs:778-784 | per-joint tiers 180 (hip/waist) / 140 (knee/ankle) / 70 (else) |
| speed/accel limits | humanoid.rs:785-786 | 8 rad/s, 40 rad/s² — HOST-side action shaping, not solver-enforced (rapid-walking-benchmark.md:17-24); Phase 2 policy-layer concern |
| gravity | world/crates/sim/src/lib.rs:48 | **20.0 m/s²** (authored, NOT 9.81; confirmed by readiness doc:17-19 and the bundle) |
| friction / restitution | humanoid.rs:734-735 | 0.8 / 0.0, one material everywhere |
| dt / episode | humanoid.rs:181-182 | 1/120 s, 1200 steps; action_repeat 2 (humanoid_h0.py:124) |
| home/reset pose | humanoid.rs:50-162 + the frozen golden box3d-arm-lab/factory_os/artifacts/contracts/humanoid-independent-h0-v1.json | all joints q=0, identity orientations, zero velocity, soles exactly at the floor |
| naming / ordering / frames | humanoid_h0.py:24-55, 111-126 | +Y up, +X forward, xyzw, state13; frozen body/joint order |

Materialized cross-check: `/Users/john/Code/world/evidence/
humanoid-balance-r4-preflight-20260830/humanoid-cuda-training-bundle-v2.json`
(schema `world.humanoid-cuda-training-bundle` v2, E=4096) — every
registered anchor/axis/limit/gain/effort/gravity/friction value matches the
lowering (`humanoid/tests/test_h0_lowering.py::test_bundle_cross_check`).
Caveat: the bundle's `initial` block carries the authored per-env mass
randomization on pelvis+torso only (humanoid.rs:397-408), so authored
masses come from humanoid.rs, not from bundle env slices. Note this v2
bundle's package sha (fc8b53…) differs from humanoid_h0.py's pinned v1
PACKAGE_SHA256 (4d8e25…) — same morphology, newer bundle revision.

### Constants that exist NOWHERE (verified absent; NOT invented)

1. **Armature / passive joint damping / joint friction loss** — SliceJoint
   and SliceActuator have no such fields. Lowered as 0 (= feature off in
   this stack), not as a guess.
2. **Non-box (anatomical) inertias, COM offsets** — inertia is always the
   solid-box derivation; COM offsets hardcoded zero (humanoid.rs:721).
3. **A crouch/nominal-stance pose** — the authored reset IS the home pose
   (all q=0, fully extended); no other pose vector exists anywhere.
4. **Non-planar axes** (hip ab/adduction, ankle roll), per-joint gain
   tables, foot-specific friction, multi-point sole geometry — none exist.
5. **Humanoid reference gait / reward weights / env shaping constants** —
   nothing to transfer; Phase 2 must author them (the O121 objective and
   gait gates in the world repo are the closest prior art).
6. World-engine solver internals (substeps 2, iterations 12, warm start
   0.8, angular damping 0.02 — box3d-cuda-client/src/lib.rs:117-147) are
   engine parameters of an explicitly non-equivalent solver
   (humanoid_h0.py:184-188); this stack's to-tolerance impulse solver
   replaces them, keeping its own d0/dw/tc + solimp defaults.

## 4. CPU oracle path

The av2/idv1 machinery is fixture-generic:

- `experimental/articulated_v1/articulated_v1.py` `Fixture(J)` builds
  B=J+2 bodies with per-joint `Hinge{parent, ap, ac, axis, reference,
  armature, damping, loss, kp, kv, cap, d0, dw, tc}` — `av.duck()`
  (lines 53-64) just fills it from pinned JSON; the humanoid fills it from
  `humanoid/h0_lowering.py` (implemented: `h0_lowering.fixture()`).
- `experimental/integrated_duck_v1/native.py` `Scene(lib, fixture, q, v,
  shapes, pairs, friction)` takes any fixture + contact shape table;
  `duck_scene()` (lines 126-134) is 8 lines mirrored as
  `h0_lowering.scene()`, wrapped by
  `walk/env/humanoid_native_lane.py::NativeHumanoidLane`.
- Contact shapes: `contact_v1` accepts convex hulls of 4..32 vertices
  (contact_v1.cpp line 44); the H0 box feet lower to their exact 8 corners
  (single-OBB feet — no baked multi-point sole exists for H0, so this is
  exact, unlike a simplification of the duck's 18-vertex convention).

Gates (mirroring `run_home_hold.py` / the mission):
zero-action home-hold 2 s (480 × 1/240 s ticks — see section 5 for why the
tick is 1/240), tilt < 5°, both feet in contact at every post-step read,
per-step `momentum_residual <= 1e-8` (idv1 diagnostic), solver rc == 0
throughout. Results in section 7.

## 5. Integration-cadence finding (dt = 1/240, not 1/120)

This stack applies the PD drive once per tick; World applies it once per
SUBSTEP (h = dt/substeps, readiness doc:23-33) with dt=1/120, substeps=2.
At a raw 1/120 tick the explicit drive is numerically UNSTABLE here: the
linearized one-tick map at the reset pose has kv·dt/I_eff = 2.33 (elbow) /
2.07 (ankle) and spectral radius 1.79 — measured blow-up: arms oscillate,
elbow slams its −0.10 limit and the solve stalls ~0.6 s in. At the
authored per-substep cadence 1/240 the spectral radius is 1.0 (free-root
modes only) and the 2 s home-hold holds to ~1e-15 momentum residual.
Phase 1 ran the oracle at 1/240; **Phase 2 pinned SIM_DT = 0.002 s** (the
duck stack's tick, strictly below the 1/240 stability bound, spectral
radius 1.0) so one policy step is exactly the duck env contract's 0.02 s =
10 ticks and every 0.02 s-based constant stays valid — see
humanoid/PHASE2.md §1. The instability finding is pinned by
`test_humanoid_oracle.py::test_authored_dt_would_be_unstable_documented`.

## 6. Pre-existing solver-robustness gap (workstream A) — RESOLVED upstream

STATUS UPDATE: the workstream-A repair landed (civ1 + fp32 kernel:
per-call APGD budget, damping schedule, best-iterate polish, load-aware
exhaustion ceiling; CPU lane raised to 16384 iterations). The reproducers
below now tick through clean on both lanes — the pinning test was updated
accordingly (`test_perturbed_standing_ticks_through_both_lanes`: fp32
240/240, f64 240/240). The original Phase 1 finding is kept for the
record:

The humanoid home-hold and all in-air dynamics solve cleanly at the duck's
own fp32 certificates. But ANY perturbed **standing** target (even a held
0.05 rad waist lean, or joint offsets of 0.002 rad) stalls the coupled
solve with NO_CONVERGENCE on BOTH lanes — f64 oracle (civ1) and fp32
kernel. This is the duck stack's documented degenerate flat-foot
contact-block failure (walk/env/flat.py:55-61: "ANY nonzero joint
perturbation (even 1e-4 rad) makes civ1 stall … until the workstream-A
solver robustness repair lands"; duck_cuda_kernel.h header notes the
un-ported rank-1 null-direction repair). The humanoid amplifies it: two
exactly-coplanar 4-corner box soles (8 rank-deficient contact rows) and a
~413× larger impulse scale (68 kg × 20 m/s² / 240 Hz = 5.67 N·s per
weight-tick vs the duck's 1.37e-2). The fault is detected and contained
identically on both lanes (clean status 3, state frozen finite). Falling /
piled states remain EXPENSIVE (up to the 16384-iteration ceiling per 2 ms
tick) even though they no longer fault — training infrastructure resets
done envs immediately (walk/train/vec.py) so only the falling step itself
pays; anything that keeps ticking a fallen humanoid (e.g. flat.py-style
frozen-target stepping of done envs outside VecEnv) will crawl.

## 7. Gate results (all enforced by committed tests)

(Figures below re-measured at the Phase 2 tick SIM_DT = 0.002; the Phase 1
run at 1/240 had the same character: momentum ~3.7e-15, drift 0.13 mm.)

CPU oracle (humanoid/tests/test_humanoid_oracle.py), zero-action home-hold
2 s = 1000 ticks @ 0.002, E=1:
- 1000/1000 ticks accepted (rc=0, native_status=0)
- momentum residual ≤ 2.2e-15 every step (gate 1e-8)
- both feet in contact at every post-step read
- final tilt 0.00° (gate < 5°), height 1.15 m held
- max |PD torque| 1.8e-4 N·m (min effort tier 70 → scalar-cap licensing;
  moot since the kernel now consumes DW_EFFORT_CAP_TABLE)
- health bounds (duck run_home_hold GATES) all clear; solver ≤ 64 iters

fp32 serial parity (humanoid/tests/test_humanoid_serial_parity.py), duck
flags AND duck default certificates (5e-6 / 2e-4), zero -D overrides:
- header drift: committed humanoid header == fresh regeneration; duck
  header regenerates byte-identical (untouched)
- home-hold 2 s: root drift < 2 mm gate, tilt diff < 1° gate, both lanes
  both-feet-in-contact (Phase 1 measured 0.13 mm / 0.0°)
- in-air dynamics parity (random joint poses dropped from +0.5 m, PD to
  home, 40 ticks, no contact): root 2.3e-4 mm, joints 1.8e-8 rad
- bit-identical determinism across two scenes in one build
- perturbed standing ticks through clean on both lanes (section 6)

Lowering gates (humanoid/tests/test_h0_lowering.py): topology, 68 kg,
effort tiers, FK reset == authored centers to 1e-12 (anchors close
exactly), soles at z=0 exactly, bundle cross-check, H0-contract
convention cross-check (their revolute_q_qdot recovers our FK's joint
angle to 1e-10).
