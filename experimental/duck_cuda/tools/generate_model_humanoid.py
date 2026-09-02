"""Generate humanoid/include/duck_model.h from the pinned H0 lowering.

The humanoid twin of generate_model.py (which is UNTOUCHED and still owns
the duck header): emits the same DW_* macro/table surface the single-source
kernel (src/duck_cuda_kernel.h) consumes, with H0 humanoid values from
humanoid/h0_lowering.py -- every constant rounded once to float32. The
emitted header is committed and drift-checked by
humanoid/tests/test_humanoid_serial_parity.py.

The file is deliberately named duck_model.h: the kernel includes
"duck_model.h", and the humanoid serial build (see
walk/env/humanoid_cuda_lane.py) passes -Ihumanoid/include AHEAD of
-Iexperimental/duck_cuda/include so this header shadows the duck's without
touching any kernel source. The duck build never sees humanoid/include.

PHYSICS + PINNED POLICY CONSTANTS. The physics tables drive the working
dwc1 physics path. Phase 2 additionally pins the HUMANOID reward v1
(walk/env/humanoid_reward.py) and env contract (walk/env/humanoid_flat.py)
constants into DW_RW_* / DW_ENV_* / DW_PHASE_HZ_* so python-side changes
fail the drift test until regenerated. The kernel's device policy layer
(dw_policy_observe / dw_step_policy_env) still hardcodes the DUCK's 58-wide
obs offsets and action constants, so dwc1_step_policy/dwc1_observe remain
INVALID for the humanoid until the enumerated kernel edit lands
(humanoid/FEASIBILITY.md section 2). Since v2.1, DW_REF_GAIT carries the
SYNTHETIC analytic reference cycle (humanoid/reference_gait.json) and
DW_IMIT_W is live -- the kernel imitation path is the duck-tested one.

Effort caps: the kernel consumes the per-joint DW_EFFORT_CAP_TABLE
(authored H0 tiers 180/140/70, world/crates/sim/src/humanoid.rs:778-784);
the scalar DW_EFFORT_CAP is still emitted as the MINIMUM tier (70) for
diagnostics/back-compat (dwc1_info reports it). The duck generator emits a
uniform table equal to its scalar cap, so duck builds are bit-identical.

FAMILY VARIANTS (humanoid/h1_family.py): --lowering {h1,h1_tall,h1_stocky}
selects the member (default h1 = the accepted base, output byte-identical
to before the switch existed). A variant header goes to
humanoid/variants/<name>/include/duck_model.h and carries the VARIANT's
reference table (humanoid/variants/<name>/reference_gait.json) in
DW_REF_GAIT; the humanoid lanes select the header dir by the same name.

Usage: .venv/bin/python -B experimental/duck_cuda/tools/generate_model_humanoid.py \
           [--lowering NAME] [--output PATH]
           (default output: <repo>/humanoid/include/duck_model.h for h1,
            <repo>/humanoid/variants/<name>/include/duck_model.h otherwise)
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "humanoid" / "include" / "duck_model.h"


def f32(x) -> str:
    """Exact float32 C literal (9 significant digits round-trips binary32)."""
    v = float(np.float32(x))
    if v == int(v) and abs(v) < 1e9:
        return f"{v:.1f}f"
    return f"{np.float32(x):.9g}f"


def row(values) -> str:
    return "{" + ",".join(f32(v) for v in values) + "}"


def _family():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(ROOT / "humanoid") not in sys.path:
        sys.path.insert(0, str(ROOT / "humanoid"))
    import h1_family  # noqa: PLC0415
    return h1_family


def load_lowering(name: str | None = None):
    """The lowering module for a family member (default "h1": the exact
    h1_lowering module, as before the family existed)."""
    return _family().load_lowering(name)


def default_output(name: str | None = None) -> Path:
    return _family().header_path(name)


def emit(h0, ref_gait=None) -> str:
    """Header text for lowering `h0`; `ref_gait` = [bins, J] f64 table for
    DW_REF_GAIT (default: the H1 table in humanoid_reward, i.e. exactly the
    pre-family output; variants pass their own table)."""
    from walk.env import humanoid_flat as env_mod  # noqa: PLC0415
    from walk.env import humanoid_reward as reward_mod  # noqa: PLC0415
    from walk.eval import humanoid_gait as judge  # noqa: PLC0415 (frozen judge)

    J, B = h0.J, h0.B
    assert (J, B) == (14, 16), "active lowering is H1 (hip roll)"
    ref_table = np.asarray(reward_mod.REF_GAIT if ref_gait is None
                           else ref_gait, dtype=np.float64)
    assert ref_table.shape == (int(reward_mod.REF_BINS), J), "REF_GAIT shape drift"
    variant = getattr(h0, "VARIANT", "h1")
    title = ("// H1 humanoid (B16/J14/P2: H0 + hip roll, humanoid/h1_lowering.py)"
             if variant == "h1" else
             f"// {variant} humanoid (B16/J14/P2 family member of H1, "
             f"humanoid/{variant}_lowering.py; PROFILE {h0.PROFILE})")
    n, q = 6 + J, 7 + J
    jrows, maxrows = 3 * J, 3 * J + 3 * 2 * 4
    masses = [b[3] if i else 0.0 for i, b in enumerate(h0.BODIES)]
    inertias = [(0.0, 0.0, 0.0) if i == 0 else h0.box_inertia(b[3], b[2])
                for i, b in enumerate(h0.BODIES)]
    reset_q = h0.reset_qpos()
    verts = h0.foot_vertices()
    caps = list(h0.EFFORT)

    lines = [
        "// GENERATED by experimental/duck_cuda/tools/generate_model_humanoid.py"
        " -- DO NOT EDIT.",
        title,
        "// lowering, z-up frame; every constant",
        "// rounded once to float32. Sources cited in humanoid/h0_lowering.py.",
        "// Shadows the duck header by include order (humanoid builds only);",
        "// PHYSICS ONLY -- the kernel's policy layer is NOT valid for this",
        "// model (see generate_model_humanoid.py docstring).",
        "// Drift-checked by humanoid/tests/test_humanoid_serial_parity.py.",
        "#ifndef DUCK_MODEL_H",
        "#define DUCK_MODEL_H",
        "#if defined(__CUDACC__)",
        "#define DW_MODEL_CONST static __constant__ const",
        "#else",
        "#define DW_MODEL_CONST static const",
        "#endif",
        f"#define DW_B {B}   // bodies: floor 0, pelvis root 1, hinge children j+2",
        f"#define DW_J {J}   // hinge joints",
        f"#define DW_N {n}   // generalized dofs (6 root + J)",
        f"#define DW_Q {q}   // qpos width (root xyz + quat xyzw + J)",
        f"#define DW_JROWS {jrows}        // 3*J joint row slots",
        "#define DW_PAIRS 2         // foot-vs-floor contact pairs (L, R)",
        "#define DW_MAXPOINTS 4     // manifold points per pair",
        f"#define DW_MAXROWS {maxrows}      // 3*J + 3*DW_PAIRS*DW_MAXPOINTS",
        "#define DW_FOOT_VERTS 8    // exact box corners (single-OBB feet)",
        f"#define DW_DT {f32(h0.SIM_DT)}  // duck-stack tick; 10 per 0.02 s step",
        f"#define DW_GRAVITY_Z {f32(-h0.GRAVITY)}  // authored -20 (y-up) -> z-up",
        "// Humanoid gait clock (walk/env/humanoid_flat.py, sweepable via",
        "// HUMANOID_PHASE_HZ_* env vars at generation time, duck mechanism):",
        f"#define DW_PHASE_HZ_BASE {float(env_mod.PHASE_HZ_BASE)!r}",
        f"#define DW_PHASE_HZ_PER_MPS {float(env_mod.PHASE_HZ_PER_MPS)!r}",
        f"#define DW_ARMATURE {f32(h0.ARMATURE)}       // not authored for H0",
        f"#define DW_DAMPING {f32(h0.PASSIVE_DAMPING)} // H0 damping is kv, not passive",
        f"#define DW_FRICTION_LOSS {f32(h0.FRICTION_LOSS)}  // not authored for H0",
        "// fp32 training-lane certificates + repair economics, SCALE-AWARE:",
        "// the duck lane is certified at ABSOLUTE 5e-6 / 2e-4 on a 0.0413",
        "// N*s per-tick weight impulse; H0 carries 2.72 N*s (65.8x). These",
        "// pins keep the humanoid at least 1.9x STRICTER in RELATIVE terms",
        "// than the certified duck lane (x35 = 53% of the exact impulse",
        "// ratio). Chasing duck-ABSOLUTE certificates on H0 made nearly",
        "// every contact tick stall into the Tresca arsenal (measured: 78",
        "// env-steps/s serial, 20-150 APGD bursts per policy step, 500 ms",
        "// straggler steps); with these pins + the eager stall tier the",
        "// same flail workload runs 3.6x faster with fewer contained",
        "// faults (10 -> 1 per 960 env-steps).",
        f"#define DW_SOLVE_TOLERANCE {f32(1.75e-4)}",
        f"#define DW_MOMENTUM_TOLERANCE {f32(7e-3)}",
        "// per-call APGD budget: bounds one Tresca call at 8192 inner",
        "// iterations so a full 16-call arsenal costs 131k, not 1M -- the",
        "// launch straggler bound; the duck default (65536) is untouched.",
        "#define DW_APGD_BUDGET 8192u",
        "// stalled solves accept at the tier ceiling IMMEDIATELY (see",
        "// DW_STALL_TIER_EAGER in duck_cuda_kernel.h); duck: never defined.",
        "#define DW_STALL_TIER_EAGER 1",
        "// stall-tier ceiling, scale-aware like the tolerances: the duck's",
        "// certified 1e-3f is 2.42e-2 RELATIVE to its 0.0413 N*s per-tick",
        "// weight impulse; 2.4e-2f absolute here is 8.8e-3 relative to",
        "// H0's 2.72 N*s -- 2.7x STRICTER than the certified duck ceiling.",
        "// Measured on the flail workload: 78 -> 1159 env-steps/s serial",
        "// overall (this pin: 281 -> 1159), contained faults 10 -> 0.",
        "#define DW_TIER_CEILING 2.4e-2f",
        f"#define DW_KP {f32(h0.KP)}",
        f"#define DW_KV {f32(h0.KV)}",
        "// per-joint PD gain tables consumed by the kernel (H1.1 spec,",
        "// PHASE2.md section 15: quasi-static single support needs",
        "// hip-roll kp >> the destabilizing gravitational stiffness).",
        "// AUTHORED BY THE LOWERING: emitted from KP_TABLE / KV_TABLE",
        "// when the lowering defines them (tuples of length J), else",
        "// uniform broadcasts of the scalar KP / KV -- the humanoid agent",
        "// coordinates gain values through the lowering, never here.",
        "DW_MODEL_CONST float DW_KP_TABLE[DW_J] = "
        + row(list(getattr(h0, "KP_TABLE", None) or [h0.KP] * J)) + ";",
        "DW_MODEL_CONST float DW_KV_TABLE[DW_J] = "
        + row(list(getattr(h0, "KV_TABLE", None) or [h0.KV] * J)) + ";",
        "// Scalar cap = MIN of the authored per-joint tiers (180/140/70),",
        "// kept for diagnostics/back-compat; the kernel clamps with the",
        "// per-joint DW_EFFORT_CAP_TABLE below.",
        f"#define DW_EFFORT_CAP {f32(min(caps))}",
        "DW_MODEL_CONST float DW_EFFORT_CAP_TABLE[DW_J] = " + row(caps) + ";",
        f"#define DW_FRICTION_D0 {f32(0.9)}",
        f"#define DW_FRICTION_DWIDTH {f32(0.95)}",
        f"#define DW_FRICTION_TIMECONST {f32(0.02)}",
        f"#define DW_LIMIT_MARGIN {f32(0.0)}",
        f"#define DW_LIMIT_TIMECONST {f32(0.02)}",
        f"#define DW_LIMIT_DAMPRATIO {f32(1.0)}",
        "DW_MODEL_CONST float DW_LIMIT_SOLIMP[5] = "
        + row([0.9, 0.95, 0.001, 0.5, 2.0]) + ";",
        "DW_MODEL_CONST float DW_BODY_MASS[DW_B] = " + row(masses) + ";",
        "DW_MODEL_CONST float DW_BODY_INERTIA[DW_B][3] = {"
        + ",".join(row(i3) for i3 in inertias) + "};",
        "DW_MODEL_CONST unsigned DW_HINGE_PARENT[DW_J] = {"
        + ",".join(str(j[1]) for j in h0.JOINTS) + "};",
        "DW_MODEL_CONST float DW_HINGE_AP[DW_J][3] = {"
        + ",".join(row(j[2]) for j in h0.JOINTS) + "};",
        "DW_MODEL_CONST float DW_HINGE_AC[DW_J][3] = {"
        + ",".join(row(j[3]) for j in h0.JOINTS) + "};",
        "DW_MODEL_CONST float DW_HINGE_AXIS[DW_J][3] = {"
        + ",".join(row(j[7]) for j in h0.JOINTS) + "};",  # per-joint (H1 roll)
        "DW_MODEL_CONST float DW_HINGE_REF[DW_J][4] = {"
        + ",".join(row(h0.REFERENCE_XYZW) for _ in h0.JOINTS) + "};",
        "DW_MODEL_CONST float DW_LIMIT_LOWER[DW_J] = "
        + row([j[4] for j in h0.JOINTS]) + ";",
        "DW_MODEL_CONST float DW_LIMIT_UPPER[DW_J] = "
        + row([j[5] for j in h0.JOINTS]) + ";",
        "// identity source->principal: H0 bodies are zero-offset principal",
        "// boxes (humanoid.rs:710-721), unlike the duck.",
        "DW_MODEL_CONST float DW_ROOT_COM[3] = " + row([0.0, 0.0, 0.0]) + ";",
        "DW_MODEL_CONST float DW_ROOT_QPC[4] = "
        + row([0.0, 0.0, 0.0, 1.0]) + ";",
        "DW_MODEL_CONST float DW_REFERENCE_QPOS[DW_Q] = " + row(reset_q) + ";",
        "DW_MODEL_CONST float DW_INITIAL_QPOS[DW_Q] = " + row(reset_q) + ";",
        "DW_MODEL_CONST float DW_INITIAL_VEL[DW_N] = "
        + row([0.0] * n) + ";",
        "DW_MODEL_CONST float DW_HOME_TARGETS[DW_J] = "
        + row(h0.HOME_TARGETS) + ";",
        "DW_MODEL_CONST double DW_HOME_TARGETS_F64[DW_J] = {"
        + ",".join(f"{float(x):.17g}" for x in h0.HOME_TARGETS) + "};",
        "// HUMANOID reward v1 pins (walk/env/humanoid_reward.py, duck v12",
        "// shape, NO imitation term -- empty hook, W_IMIT 0): generated so",
        "// any python-side change fails the drift test until regenerated.",
        *[f"#define DW_RW_{name} {float(getattr(reward_mod, name))!r}"
          for name in [
              "W_TRACK", "TRACK_SIGMA_SQ", "TRACK_EMA_S", "W_ALIVE",
              "W_LATERAL", "W_ACTION_RATE", "W_TORQUE", "W_AIR_TIME",
              "AIR_TIME_MIN", "AIR_TIME_MAX", "PLACEMENT_MIN_M",
              "OPP_SUPPORT_FRAC", "W_CHATTER", "CHATTER_MAX_S", "W_FLICKER",
              "STANCE_MIN_S", "W_CLEARANCE", "CLEARANCE_M",
              "W_DOUBLE_SUPPORT", "DOUBLE_SUPPORT_GRACE", "W_ALTERNATE",
              "W_SAME_FOOT", "W_PHASE"]],
        f"#define DW_RW_TICKS_FULL {int(reward_mod.TICKS_FULL)}u",
        f"#define DW_REF_BINS {int(reward_mod.REF_BINS)}",
        f"#define DW_IMIT_W {float(reward_mod.W_IMIT)!r}",
        f"#define DW_IMIT_SIGMA_SQ {float(reward_mod.IMIT_SIGMA_SQ)!r}",
        "// Synthetic analytic reference cycle (v2.1):",
        "// humanoid/author_reference_gait.py -> humanoid/reference_gait.json,",
        "// FK-validated (humanoid/tests/test_reference_gait.py); identical",
        "// f64 values to walk/env/humanoid_reward.REF_GAIT (bit-parity).",
        "// HUMANOID env contract pins (walk/env/humanoid_flat.py). The",
        "// kernel's device policy layer consumes these directly (obs",
        "// width/offsets, action->target chain, termination up-axis), so",
        "// dwc1_step_policy/observe are first-class on this build.",
        "// Obs layout (52 = 3*J + 16):",
        "//   [0:12] q-HOME  [12:24] 0.05*qdot  [24:36] prev action",
        "//   [36:39] gravity body (-R[2])  [39:42] R^T omega  [42:45] R^T v",
        "//   [45] command  [46:48] zeros  [48:50] contacts  [50:52] phase",
        f"#define DW_ENV_OBS {int(env_mod.OBS)}",
        f"#define DW_ENV_ACT {int(env_mod.ACT)}",
        f"#define DW_ENV_TICKS_PER_STEP {int(env_mod.TICKS_PER_STEP)}u",
        f"#define DW_ENV_CONTROL_DT {float(env_mod.CONTROL_DT)!r}",
        f"#define DW_ENV_ACTION_SCALE {float(env_mod.ACTION_SCALE)!r}",
        "#define DW_ENV_MAX_TARGET_INCREMENT "
        + f"{float(env_mod.MAX_TARGET_INCREMENT)!r}",
        f"#define DW_ENV_QDOT_OBS_SCALE {float(env_mod.QDOT_OBS_SCALE)!r}",
        f"#define DW_ENV_HORIZON_STEPS {int(env_mod.HORIZON_STEPS)}u",
        "#define DW_ENV_MIN_HEIGHT_FRACTION "
        + f"{float(env_mod.MIN_HEIGHT_FRACTION)!r}",
        f"#define DW_ENV_MAX_TILT_RAD {float(env_mod.MAX_TILT_RAD)!r}",
        "// termination up-scalar = cos(MAX_TILT), pinned in f64 (no libm",
        "// cos() in the kernel).",
        "#define DW_ENV_COS_MAX_TILT "
        + f"{float(math.cos(env_mod.MAX_TILT_RAD))!r}",
        "// tilt tests BODY +Y against world +Z (authored y-up-local root:",
        "// up = R[2][1] = 2*(qy*qz + qx*qw), humanoid_native_lane.tilt).",
        "#define DW_ENV_UP_AXIS 1",
        "DW_MODEL_CONST double DW_ENV_COMMANDS_MPS[3] = {"
        + ",".join(f"{float(c)!r}" for c in env_mod.COMMANDS_MPS) + "};",
        "// gate_proxy_* thresholds: the FROZEN humanoid judge's footfall",
        "// clauses (walk/eval/humanoid_gait.py), generation-time imported",
        "// so judge and kernel shadow counters cannot silently diverge",
        "// (drift-pinned). HONESTY: the in-kernel counters approximate the",
        "// judge's swing-duration / whole-sole-clearance / placement",
        "// clauses at raw tick resolution WITHOUT the 20 ms contact-",
        "// debounce sensor model or the support/slip clauses -- a cheap",
        "// culling/monitoring shadow, never a substitute for the frozen",
        "// CPU judge.",
        f"#define DW_GATE_SWING_MIN_S {float(judge.SWING_MIN_S)!r}",
        f"#define DW_GATE_SWING_MAX_S {float(judge.SWING_MAX_S)!r}",
        f"#define DW_GATE_CLEARANCE_M {f32(judge.CLEARANCE_M)}",
        f"#define DW_GATE_CLEARANCE_MIN_S {float(judge.CLEARANCE_MIN_S)!r}",
        f"#define DW_GATE_PLACEMENT_MIN_M {f32(judge.PLACEMENT_MIN_M)}",
        "DW_MODEL_CONST double DW_REF_GAIT[DW_REF_BINS][DW_J] = {"
        + ",".join("{" + ",".join(repr(float(x)) for x in row_vals) + "}"
                   for row_vals in ref_table) + "};",
        "DW_MODEL_CONST unsigned DW_PAIR_BODY_A[DW_PAIRS] = {"
        + ",".join(f"{b}u" for b in h0.FOOT_BODIES) + "};",
        "DW_MODEL_CONST unsigned DW_PAIR_BODY_B[DW_PAIRS] = {0u,0u};",
        "DW_MODEL_CONST float DW_PAIR_MU[DW_PAIRS] = "
        + row([h0.FRICTION, h0.FRICTION]) + ";",
        "DW_MODEL_CONST float DW_FOOT_VERTICES[DW_PAIRS][DW_FOOT_VERTS][3] = {"
        + ",".join("{" + ",".join(row(v) for v in verts) + "}"
                   for _ in range(2)) + "};",
        "#endif",
        "",
    ]
    return "\n".join(lines)


def emit_variant(name: str | None = None) -> str:
    """Header for a family member with ITS reference table (h1: identical
    to emit(load_lowering()))."""
    h1_family = _family()
    from walk.env import humanoid_reward as reward_mod  # noqa: PLC0415
    h0 = load_lowering(name)
    if h1_family.is_base(name):
        return emit(h0)
    return emit(h0, reward_mod.load_reference(
        h1_family.reference_gait_path(name)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lowering", default="h1",
                        help="family member: h1 (default) | h1_tall | h1_stocky")
    parser.add_argument("--output", type=Path, default=None,
                        help="default: humanoid/include/duck_model.h (h1) or "
                             "humanoid/variants/<name>/include/duck_model.h")
    args = parser.parse_args()
    if args.output is None:
        args.output = default_output(args.lowering)
    text = emit_variant(args.lowering)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f"wrote {args.output} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
