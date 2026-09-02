# Fixed-base 6-axis arm on the robot-generic duck stack — feasibility memo

Status: COMPLETE for the CPU/serial lanes (both variants); GPU lane
untested locally (no CUDA on this machine; `gpu/specs/arm-reach-*.json`
carry the remote recipe). Verdict: **pure regeneration** — zero edits under
`experimental/duck_cuda/src/**`, zero edits to duck/humanoid files; the only
shared-file change is the `arm` branch in `walk/train/gpu_train.robot_classes`
(+ the `--lane-env` guard). Every gate below is enforced by `arm/tests/*.py`.

Scope: a new robot CATEGORY (fixed-base arm, reach task) with TWO variants
lowered by ONE builder (`arm/arm_lowering.py`), so a policy can be trained
across a distribution of robots of the same category:

| variant | source | moving mass | base | reach (flange, stretched) | effort a1..a6 (N·m) |
|---|---|---|---|---|---|
| `kr240` | pinned KR240 URDF (box3d-arm-lab `kr240r2900.physics.urdf`) | 690 kg | 430 kg | 3.462 m | 12000/12000/9000/3000/2000/1200 |
| `lite`  | `scaled(KR240, 0.5, 1/8)` — Froude scaling | 86.25 kg | 53.75 kg | 1.731 m | 750/750/562.5/187.5/125/75 |

Lite scaling laws (dynamic similarity): lengths ×0.5, masses ×1/8, inertia
×1/32, torque ×1/16, angular rates ×√2, joint damping ×(1/8)(1/2)^1.5. The
spec's "~30 kg UR10-class" would be ~1/23 mass; the lite is the geometric
1/8-mass member the spec's own numbers imply (a third, lighter member is a
one-line `scaled(...)` call).

## 1. Fixed base on a free-root kernel — decision and proof

The kernel (`duck_cuda_kernel.h`) always carries a 6-dof free root (body 1)
and `DW_N = 6 + J`; there is no root pin, no external force, no per-body
gravity switch. Three candidates were evaluated:

| path | kernel | f64 oracle (idv1/av1) | drift | verdict |
|---|---|---|---|---|
| (A) joint 0 parented on the STATIC FLOOR body 0 | expressible as-is: `dw_evaluate` initialises body 0 to identity pose / zero motion / `colmask[0]=0` before the FK loop and reads parents through `DW_HINGE_PARENT`; the arm's colmasks never contain a root bit, so the root block of M is exactly decoupled | **rejected**: `articulated_v1.cpp:39` validates `h.parent<1 -> invalid` | exact (0) | **kernel: chosen** |
| (B) heavy root box resting on the floor with friction | contact rows every tick; two exactly coplanar 4-corner blocks = the humanoid's documented degenerate stall/Tresca regime (78 env-steps/s before repairs); rest position noise ~ DW_CONTACT_EPS 2e-6 m; arm reaction torques (6.7 kN·m at full extension) rock a 1120 kg box on a ~1 m footprint (tipping margin 11 kN × 0.5 m = 5.5 kN·m — marginal) | same | ≥1e-6 m, plus cost | rejected |
| (C) root at rest, gravity-compensated | no force hook in the kernel | **expressible**: `av2_step.applied_force [E,N]` is plumbed through `idv1_step` (`native.Scene.step(force=...)`) | see below | **oracle: chosen** (virtual weld) |

**Kernel = structural weld (A).** Bodies: 0 floor (+ fused base_link), 1
phantom root carrying base_link's mass/inertia (never joined to anything),
2..7 = link_1..link_6 (child(j) = j+2 as the stack requires),
`DW_HINGE_PARENT = {0,2,3,4,5,6}`. The two kernel-mandated contact pairs
point at link_1 with a small box ~0.7 m above the floor (link_1 only yaws),
so no contact row is ever generated (`contact_points == 0` gate). The
phantom free-falls, decoupled: after 8 s it sits at z = −g·dt²·n(n+1)/2 =
−313.998 m (semi-implicit Euler, matched to 1e-2 by the test) while the arm
joints are bit-for-bit unaffected (its mass block is block-diagonal; no
joint/contact row has a root column).

**Oracle = virtual weld (C).** Root body 1 IS base_link with mass
1e6 × moving mass (6.9e8 kg kr240; av2's Cholesky pivot check
`x <= 1e-12·max(diag)` bounds the ratio at ~1e11 — lite's a6 pivot is
3.8e-3 vs a 8.6e-5 threshold), inertia m·(1 m)². `arm_lowering.weld_force`
applies, every tick, (i) the EXACT static gravity generalized force of all
6 root dofs — av1's root angular dofs act about the root SOURCE origin, so
the weld mass's own 6.8e9 N at its 0.14 m COM lever is a 9.6e8 N·m moment
that must be cancelled, not PD-resisted (measured 1.4e-4 rad of tilt when
it was not) — and (ii) an explicit PD on the root pose with ω·dt = 0.2,
ζ = 1 (stable: bound 2), absorbing the arm's dynamic reactions
(7 kN → 1e-9 m deflection).

Proof (`arm/tests/test_arm_oracle.py`, 8 s = 4000 ticks, both variants):
base translation drift **1.6e-19 m / 1.2e-19 m**, rotation **1.2e-17 /
7.1e-19 rad** (gate 1e-6), momentum residual 0.0 (gate 1e-8), 1 solver
sweep per tick, joints at rest, sag ≤ 0.0089 rad (bound 0.01), zero
contacts. `test_weld_cancels_static_gravity_exactly` pins
`weld_force == av1 root bias` at rest.

Parity kernel vs oracle (`arm/tests/test_arm_serial_parity.py`): home-hold
8 s joint |Δq| **8.2e-7 / 7.0e-7 rad**, flange 0.003 / 0.001 mm; scripted
3 s multi-joint sinusoids (speeds to 21 rad/s) **1.2e-6 / 7.2e-7 rad**,
flange < 0.005 mm; a3 driven into its soft limit: excess 1.6e-5 (kernel
1.7e-5), Δq 1.5e-6 rad; bit-identical determinism; env-level parity
(ArmReachEnv over both lanes, 60 mixed steps) obs 2e-6, reward 3e-6.

Kernel need (none required): a first-class fixed base would be a
`#define DW_ROOT_FIXED 1` header switch that (a) skips body 1 in the
gravity/bias loop of `dw_evaluate` (or scales its gravity by 0) and (b)
zeros `v[0:6]` after `dw_solve` — ~6 lines. It would only make the phantom
sit still (cosmetic for readers of `body_state[1]`); physics is already
exact without it.

## 2. Obs / act contract (walk/env/arm_reach.py) — identical for both variants

OBS 27 = `[q(6), 0.25·qd(6), target xyz(3), tip xyz(3), target−tip(3),
prev action(6)]` (m, rad, world frame with base_link at the origin).
ACT 6: `requested_j = lower_j + (a_j+1)/2·(upper_j−lower_j)`, slew-limited
per 0.02 s step to `velocity_j·CONTROL_DT` (URDF speed limit), held for the
10 physics ticks; PD per joint from the tables below. Termination: proxy
violation (judge clause 5), non-finite, 400-step horizon. A learned policy
must emit sag-compensated absolute targets (the baseline controller stalls
4 cm short when it re-anchors to the measured q every step, section 6).

## 3. Gravity

Authored **−9.81 m/s²** for the arm (the KR240 contract's 9.81 magnitude);
the humanoid lowering is authored at **−20** (world/crates/sim/src/lib.rs:48).
Deliberately NOT unified: each robot carries its authoring engine's gravity;
`DW_GRAVITY_Z` is per generated header.

## 4. Per-joint kp/kv/effort tables (arm/feasibility_check.py)

Derivation (`arm_lowering.gains`): `kp_j = min( max(bandwidth, sag,
authority), cap )` with bandwidth `sqrt(kp/I_max) ≥ 3·ω_cmd`, ω_cmd = 2π·5
targets/8 s = 3.93 rad/s (the judge's target-change rate); sag
`τ_hold/kp ≤ 0.01 rad` at full horizontal extension; authority
`cap/0.1 rad`; discrete cap `sqrt(kp/I_min)·dt ≤ 0.25`;
`kv = 2·0.7·sqrt(kp·I_max) + URDF damping` (the kernel's DW_DAMPING is a
single scalar, so damping is folded into the kv table); URDF Coulomb
friction (3-12 N·m ≤ 0.1 % of effort) dropped (scalar DW_FRICTION_LOSS).
Values are float32-exact (3 significant digits) so oracle (`c_float`
Hinge), header and env agree bit for bit.

| kr240 | cap N·m | τ_hold (q=0) | ratio | +240 kg @ q=0 | +240 kg @ HOME | I_max | I_min | kp | kv | ω rad/s | ω/req | sag rad | ω·dt | kv·dt/I |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a1 | 12000 | 0 | 0.00 | 0.00 | 0.00 | 2251 | 191.9 | 312000 | 37200 | 11.8 | 1.00 | 0 | 0.08 | 0.39 |
| a2 | 12000 | 6707 | 0.56 | **1.15** | 0.67 | 1493 | 559.9 | 671000 | 44400 | 21.2 | 1.80 | 0.0100 | 0.07 | 0.16 |
| a3 | 9000 | 1858 | 0.21 | 0.61 | 0.56 | 246.6 | 220.9 | 186000 | 9540 | 27.5 | 2.33 | 0.0100 | 0.06 | 0.09 |
| a4 | 3000 | 7 | 0.00 | 0.00 | 0.00 | 3.23 | 1.567 | 24500 | 419 | 87.1 | 7.39 | 0.0003 | 0.25 | 0.53 |
| a5 | 2000 | 77 | 0.04 | 0.36 | 0.36 | 2.35 | 2.352 | 20000 | 322 | 92.2 | 7.83 | 0.0039 | 0.18 | 0.27 |
| a6 | 1200 | 0 | 0.00 | 0.00 | 0.00 | 0.120 | 0.120 | 1880 | 33 | 125 | 10.6 | 0 | 0.25 | 0.55 |

| lite | cap | τ_hold | ratio | +30 kg @ q=0 | +30 kg @ HOME | I_max | I_min | kp | kv | ω | ω/req | sag | ω·dt | kv·dt/I |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a1 | 750 | 0 | 0.00 | 0.00 | 0.00 | 70.3 | 6.00 | 9760 | 1160 | 11.8 | 1.00 | 0 | 0.08 | 0.39 |
| a2 | 750 | 419 | 0.56 | **1.15** | 0.67 | 46.7 | 17.5 | 41900 | 1960 | 30.0 | 2.54 | 0.0100 | 0.10 | 0.22 |
| a3 | 562.5 | 116 | 0.21 | 0.61 | 0.56 | 7.71 | 6.90 | 11600 | 421 | 38.8 | 3.29 | 0.0100 | 0.08 | 0.12 |
| a4 | 187.5 | 0 | 0.00 | 0.00 | 0.00 | 0.101 | 0.049 | 765 | 13.4 | 87.0 | 7.39 | 0.0006 | 0.25 | 0.55 |
| a5 | 125 | 5 | 0.04 | 0.36 | 0.36 | 0.073 | 0.0735 | 1150 | 13.7 | 125 | 10.6 | 0.0042 | 0.25 | 0.37 |
| a6 | 75 | 0 | 0.00 | 0.00 | 0.00 | 0.004 | 0.0038 | 58.7 | 1.19 | 125 | 10.6 | 0 | 0.25 | 0.63 |

FINDING (pinned by `test_gains_and_feasibility_rows`): the URDF's "bounded
approximate" 12 kN·m A2 effort cannot lift the published 240 kg rated
payload at FULL horizontal extension (13.8 kN·m needed, ratio 1.15); it
holds it at the HOME pose (0.67). The reach task is unloaded; the real
KR240's load diagram is not authored anywhere in the sources.

fp32 certificates are SCALE-AWARE like the humanoid's (same relative pins
on the reference impulse max(effort)·dt): kr240 solve 1.54e-3 / momentum
6.18e-2 / tier ceiling 0.212 on 24 N·m·s; lite 9.65e-5 / 3.86e-3 / 1.32e-2
on 1.5 N·m·s. Measured momentum residual 0.0 (joint rows only).

## 5. Frozen judge (walk/eval/arm_reach_judge.py) and acceptance

8 s episode, 5 sequential targets uniformly drawn (seeded, counter-based
per (seed, env, episode)) from the reachable workspace inside a ball of
`TIER_RADIUS_FRAC[tier]·reach` = (0.15, 0.30, 0.40) around the HOME tip
((2.41, 0, 1.68) m kr240). Reachability BY CONSTRUCTION: FK of a uniform
joint-box draw (a3 ≥ 0.4 rad keeps the elbow bent — the straight arm is the
singular workspace boundary where paths sweep the floor), rejected unless
inside the ball and proxy-clear (tip, wrist, elbow at the sampled pose).
Acquired = tip within 2 cm (kr240) / 1.5 cm (lite) at 14 consecutive
policy-step boundaries (0.26 s span ≥ 0.25 s; identical to the env's
target-advance rule so verdict and presented sequence cannot disagree).
Pass = all 5 acquired in order within 8 s AND joint limits (tol 0.01 rad)
AND joint speeds ≤ URDF limits at every tick AND no proxy violation
(floor: tip/wrist/elbow z ≥ 0.05·reach; base column: tip/wrist horizontal
radius ≥ 0.20·reach while z < 0.40·reach) AND integrity (single 8 s
episode, no fault/termination). Acceptance: seeds (4242, 7, 1913, 90210) ×
3 tiers = 12/12 (`walk/eval/arm_acceptance.py`).

Calibration with the scripted proxy-aware IK baseline (0.8 × URDF speeds):
tiers 0/1 16/16 for both variants (acquisitions every ~1.0-1.5 s); tier 2
17/24 on first-episode draws and 4/8 on the acceptance-sequence draws
(kr240 1/4, lite 3/4), EVERY failure a 4-of-5 time miss (no crash, no
other clause) — hard-but-feasible for a learned policy at full speed
(`test_arm_judge.py` pins the structural half and ≥ 1 tier-2 pass).

Reward (`walk/env/arm_reward.py`): `+exp(−(d/0.1R)²) − 0.5·d/R +
10·acquired − 0.05·|Δa|² − 0.05·mean(τ/cap)² − 0.5·Σmax(0,|qd|/vlim−0.9)²
− 5·proxy`; all constants emitted as `DW_ARM_RW_*` header pins (drift test).

## 6. Device policy path — not expressible; exact kernel block needed

> **UPDATE 2026-09-02 — IMPLEMENTED.** The device policy path described below now exists: `DW_ENV_KIND_REACH` (ABI v8), host-drawn target queue via `dwc1_reach_set_targets`, gate-proxy mapping (acquired targets / violating ticks), bit-exact device-vs-python parity (`arm/tests/test_arm_device_policy.py`). Reward pins are `DW_RW_*`. `--lane-env` is the default arm training path.

`dw_policy_observe` / `dw_step_policy_env` implement the duck/humanoid
`3·J+16` locomotion observation (gravity/velocity/contacts/phase clock) and
gait reward through the `DW_ENV_*` block; none of {target xyz, tip FK,
target−tip, per-episode target sequence, acquisition counter} exists there.
The arm therefore runs obs/reward python-side over `dwc1_step` +
`dwc1_read(body_state)` (measured 22 k env-steps/s serial at E=8; 15 k
env-steps/s inside the PPO loop), and `--lane-env` is rejected for
`--robot arm`. The extension that would enable a device path, stated
precisely (kernel owner's call):

```
// generated header, arm builds only
#define DW_ENV_KIND_REACH 1            // selects the reach policy layer
#define DW_ENV_TIP_BODY 7              // link_6
DW_MODEL_CONST float DW_ENV_TIP_OFFSET[3];   // tool_xyz - com_link6 (body frame)
#define DW_ENV_ACQ_RADIUS, DW_ENV_ACQ_HOLD_STEPS, DW_ENV_REACH_M, DW_ENV_OBS 27
// DwState additions: double target[3]; uint32_t hold, target_index;
// dwc1_set_targets(scene, const double* targets /*[E,3]*/, const uint8_t* mask)
// -- host draws targets (numpy counter stream, like commands/phase0) and
//    pushes the next one when the kernel reports acquired (done-style flag)
// dw_policy_observe_reach(): q, QDOT_SCALE*qd, target, tip = FK(body 7)
//    + R*offset, target-tip, prev_action;  dw_policy_reward_reach(): the
//    DW_ARM_RW_* terms; termination: proxy(tip, link_5/link_3 origins).
```

Until then, `dwc1_step_policy` on an arm header would write the 34-wide
locomotion layout (the header pins `DW_ENV_OBS 34` for memory safety) and
is guarded off in `CudaArmLane.step_policy`.

## 7. Gate results (all enforced by arm/tests/*.py)

- `test_arm_lowering.py` (6): FK vs pinned KR240 runtime trace 0.34 mm at
  step 0 (< 3 mm over the trace's own PD sag), topology / hinge
  convention, Froude scaling laws, fk_batch == fk (1e-12), gains f32-exact
  and all feasibility rows, HOME tip pin.
- `test_arm_oracle.py` (4): 8 s home-hold both variants (section 1
  numbers), restore, weld == root gravity bias.
- `test_arm_serial_parity.py` (9): header drift both variants + duck
  header untouched, home-hold parity + phantom free-fall, scripted +
  limit-hit parity, determinism, info/tables, set_state.
- `test_arm_env.py` (7): obs/act contract, slew, sampler reachability +
  seeding, deterministic resets, IK acquires + bonus bookkeeping, proxy
  termination, env-level oracle-vs-kernel parity.
- `test_arm_judge.py` (6): every clause on synthetic traces; scripted
  baseline through the full 4-seed × 3-tier acceptance on both variants.
- `test_arm_gpu_train.py` (5): robot_classes arm rows (default kr240, bad
  variant rejected), duck identity, lane-env guard, make_env binding,
  gpu_train.train 8 envs × 2 updates on cpu zero-fault + actor round trip.
- CLI smoke: `gpu_train --robot arm --variant kr240 --device cpu --envs 8
  --horizon 32 --max-wall-s 30`: 577 updates, 147 712 env-steps, 0 faults,
  reward −0.36 (random) → +0.5.
- Duck + humanoid suites green after the gpu_train edit:
  `experimental/duck_cuda/tests/test_serial_parity.py` (19),
  `walk/train/tests/test_gpu_train.py` (9),
  `humanoid/tests/test_humanoid_gpu_train.py` (7),
  `humanoid/tests/test_humanoid_serial_parity.py` (8); duck header
  regenerates byte-identically.

## 8. Commands

```
# headers (committed under arm/include/<variant>/duck_model.h)
.venv/bin/python -B experimental/duck_cuda/tools/generate_model_arm.py --variant kr240
.venv/bin/python -B experimental/duck_cuda/tools/generate_model_arm.py --variant lite
# feasibility tables + gates
.venv/bin/python -B arm/feasibility_check.py
# local CPU PPO smoke (serial fp32 kernel lane, python-side obs/reward)
.venv/bin/python -B -m walk.train.gpu_train --robot arm --variant kr240 --device cpu --envs 8 --max-wall-s 30 --out runs/arm-smoke
# GPU (remote): gpu/specs/arm-reach-kr240.json, gpu/specs/arm-reach-lite.json
#   nvcc ... -Iarm/include/<variant> -Iexperimental/duck_cuda/include duck_cuda.cu -> libarm_<variant>_cuda.so
#   gpu_train --robot arm --variant <v> --envs 4096 --horizon 64 --device cuda --library <so> --policy ff --max-wall-s 900
# acceptance (frozen judge, 4 seeds x 3 tiers)
.venv/bin/python -B -m walk.eval.arm_acceptance --variant kr240 --actor <actor_final.pt> [--lane native]
.venv/bin/python -B -m walk.eval.arm_acceptance --variant kr240 --policy ik   # scripted baseline
# gates
for t in arm/tests/test_arm_*.py; do .venv/bin/python -B $t; done
```
