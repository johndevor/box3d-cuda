"""H1 humanoid FAMILY: parametrized morphology builder over the H1.1 tables.

WHY (John, 2026-09-02): a generalist trained across a DISTRIBUTION of
robots of one category system-IDs the whole family, and sim2real becomes
"just another sample". The base H1.1 (humanoid/h1_lowering.py, WALKING
ACCEPTED 12/12 by the frozen judge) is the family's reference member; the
variants are derived from its tables PARAMETRICALLY here, never forked.

INVARIANT (proved by humanoid/tests/test_variants.py): building the
identity morphology `H1` reproduces h1_lowering's BODIES / JOINTS / KP /
KV / EFFORT / reset tables BIT-IDENTICALLY, and the generated header for
"h1" is byte-identical to the committed humanoid/include/duck_model.h.
h1_lowering.py itself is NOT modified: every default code path still
imports that module directly (load_lowering("h1") returns it), so the
accepted base is behavior-identical by construction.

TRANSFORM (authored y-up frame, identity orientations at home; every
factor is exactly 1.0 / 0.0 for the base so the identity build is exact
in IEEE arithmetic):
  - link LENGTH scale s_b (thigh / shank / torso): half-extent y *= s_b,
    and -- density preserved -- mass *= s_b when `link_mass_with_length`;
    every joint anchor expressed in body b's frame gets its y component
    *= s_b (anchors sit on the link axis at +-half_y, so this keeps them
    on the scaled link's ends: hip/knee/ankle anchors on the legs, the
    waist/neck/shoulder anchors on the torso);
  - `mass_scale`: every dynamic body mass *= mass_scale (proportional);
  - `sole_half_width_add_m`: foot half-extent z (LATERAL in the authored
    frame) += add; the ankle anchor and foot length/height are untouched;
  - `effort_scale`: every authored effort cap *= effort_scale;
  - KP_TABLE / KV_TABLE: H1.1 values with per-joint-group overrides
    (roll / knee / hip_ankle / other), authored per variant after the
    plant feasibility checklist (humanoid/feasibility_check.py);
  - body CENTERS are re-derived from the kinematic chain: center[child] =
    center[parent] + ap - ac at zero joint angles, and the root is
    re-seated so the soles sit exactly on the floor (z = 0). Implemented
    as center_h1 + (chain_new - chain_h1) so the identity build is exact.
Hip anchors stay at +-0.15 m laterally for every member (the family
shares the hip spacing; check (e) of the checklist reports the lateral
static margin per member).

Consumers address a member through `load_lowering(name)`; "h1" is the
h1_lowering module itself, every other name is the module
humanoid/<name>_lowering.py (built by `build(MORPHOLOGIES[name])`).
Per-variant artifacts live under humanoid/variants/<name>/:
  include/duck_model.h   (generate_model_humanoid.py --lowering <name>)
  reference_gait.json    (author_reference_gait.py --lowering <name>)
  bc_init.pt, bc_init_ckpt.pt (walk.train.bc_pretrain --variant <name>)
"""
from __future__ import annotations

import dataclasses
import importlib
import math
import sys
import types
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import h0_lowering as h0  # noqa: E402
import h1_lowering as h1  # noqa: E402  (the accepted base; untouched)

ROOT = _HERE.parent
VARIANTS_DIR = _HERE / "variants"

# joint-name groups for the per-joint gain tables (h1_lowering._gain order)
_GROUPS = ("roll", "knee", "hip_ankle", "other")


def _group(name: str) -> str:
    if "hip_roll" in name:
        return "roll"
    if "knee" in name:
        return "knee"
    if "hip" in name or "ankle" in name:
        return "hip_ankle"
    return "other"


# H1.1 gain groups (h1_lowering.KP_TABLE / KV_TABLE, PHASE2.md s15/16)
H11_KP = {"roll": 500.0, "knee": 800.0, "hip_ankle": 300.0, "other": h0.KP}
H11_KV = {"roll": 60.0, "knee": 30.0, "hip_ankle": 20.0, "other": h0.KV}


@dataclasses.dataclass(frozen=True)
class Morphology:
    """One family member. Defaults == the accepted H1.1 (identity build)."""
    name: str = "h1"
    profile: str = h1.PROFILE
    thigh_scale: float = 1.0          # upper-leg length factor (both legs)
    shank_scale: float = 1.0          # lower-leg length factor (both legs)
    torso_scale: float = 1.0          # torso length factor
    link_mass_with_length: bool = True  # scaled links keep their density
    mass_scale: float = 1.0           # proportional total-mass factor
    sole_half_width_add_m: float = 0.0  # + lateral half-extent of each foot
    effort_scale: float = 1.0         # every authored effort cap
    kp: tuple = tuple(H11_KP[g] for g in _GROUPS)   # per _GROUPS order
    kv: tuple = tuple(H11_KV[g] for g in _GROUPS)
    rationale: str = ""               # WHY the gains/caps were authored so
    # Per-variant REWARD CONSTANT overrides (walk/env/humanoid_reward.py
    # names -> values), emitted into the member's header as DW_RW_<NAME>
    # (the kernel's device reward reads those macros) and applied by the
    # python env through reward(..., overrides) -- env<->kernel parity is
    # gated per variant. Empty for the base (byte-identical header).
    reward_overrides: tuple = ()      # (("CLEARANCE_M", 0.045), ...)

    def kp_of(self, joint: str) -> float:
        return self.kp[_GROUPS.index(_group(joint))]

    def kv_of(self, joint: str) -> float:
        return self.kv[_GROUPS.index(_group(joint))]


def _gains(kp_roll, kp_knee, kp_hip_ankle, kv_roll, kv_knee, kv_hip_ankle):
    return (tuple([kp_roll, kp_knee, kp_hip_ankle, h0.KP]),
            tuple([kv_roll, kv_knee, kv_hip_ankle, h0.KV]))


H1 = Morphology()

# FAMILY LAWS (measured 2026-09-02 on the TALL 6/12 GPU actor vs the accepted
# H1.1 actor, cmd 1.0, 4 seeds, humanoid/PHASE2.md s18):
#  * GAIT CLOCK: NOT Froude-scaled. At +12 % leg the TALL policy locks onto
#    the base clock exactly like H1.1 (step period 0.299 s == clock half-
#    cycle for both; touchdown phase error +0.070 +- 0.017 cyc vs H1.1
#    +0.074 +- 0.014), zero double-steps. A per-variant clock IS available
#    with zero kernel edits (DW_PHASE_HZ_PER_MPS is generated and read by
#    the kernel) but the data says leave it; members share the clock.
#  * SWING CLEARANCE MARGIN: the reward's clearance bar (CLEARANCE_M 0.030)
#    equals the frozen judge's bar, so a policy has no gradient toward
#    margin above it. TALL's PPO optimum settled at 41 mm median / 31 mm
#    p10 whole-sole clearance (H1.1: 62 / 58), 15 % of swings failing the
#    judge's 30 mm x 30 ms clause -> qualified-sequence dropouts -> the
#    "alternation" failures and the curriculum's alternation terminations
#    (which then unlearn stepping at speed). Variants get CLEARANCE_M
#    raised to 1.5x the judge bar (0.045) so the optimum sits above it;
#    base H1.1 keeps 0.030 (accepted, byte-identical).
CLEARANCE_MARGIN_FACTOR = 1.5
VARIANT_REWARD_OVERRIDES = (("CLEARANCE_M", round(0.030 * CLEARANCE_MARGIN_FACTOR, 6)),)

# H1-TALL: legs +12 %, torso +5 %, link masses scale with length (density
# preserved: 68.0 -> 71.52 kg). Gains: see the checklist rationale.
_TALL_KP, _TALL_KV = _gains(620.0, 1070.0, 350.0, 74.0, 40.0, 23.0)
H1_TALL = Morphology(
    name="h1_tall", profile="duckgridwalk.humanoid.h1_tall-v1",
    thigh_scale=1.12, shank_scale=1.12, torso_scale=1.05,
    kp=_TALL_KP, kv=_TALL_KV, reward_overrides=VARIANT_REWARD_OVERRIDES,
    rationale=(
        "Longer legs raise every leg joint's apparent inertia I_eff = "
        "1/(M^-1)_jj (hip roll 1.743 -> 2.150, hip 0.210 -> 0.242, knee "
        "0.074 -> 0.098 kg m^2) and the destabilizing stiffness about the "
        "stance knee (1028 -> 1142 N*m/rad) / hip roll (388 -> 393). kp is "
        "scaled by the I_eff ratio -- roll 500->620, knee 800->1070, "
        "hip/ankle 300->350 -- so every checklist-(b) bandwidth ratio and "
        "the (c) margins are >= the accepted H1.1's; kv scaled by "
        "sqrt(kp*I_eff) (60->74, 30->40, 20->23) keeps the base damping "
        "ratios and kv*dt/I_eff < 2. Caps unchanged (authored): (a) 4.5x."))

# H1-STOCKY: total mass +20 % (proportional), legs -6 %, soles +3 cm wider
# (full width; +0.015 m per half-extent), effort caps +15 % (bigger
# actuators for the bigger body).
_STOCKY_KP, _STOCKY_KV = _gains(620.0, 1000.0, 360.0, 70.0, 35.0, 23.0)
H1_STOCKY = Morphology(
    name="h1_stocky", profile="duckgridwalk.humanoid.h1_stocky-v1",
    thigh_scale=0.94, shank_scale=0.94, mass_scale=1.2,
    link_mass_with_length=False,        # mass is set by mass_scale here
    sole_half_width_add_m=0.015, effort_scale=1.15,
    kp=_STOCKY_KP, kv=_STOCKY_KV,
    rationale=(
        "+20 % mass at -6 % leg length raises g*sum(m*h) about the stance "
        "hip roll 388 -> 474 and knee 1028 -> 1196 N*m/rad (checklist (c)) "
        "and I_eff x~1.1 (roll 1.743 -> 1.922): kp_roll 500->620 keeps the "
        "(c) margin >= 1.2x AND the H1.1 bandwidth ratio; kp_knee 800->1000 "
        "and kp_hip/ankle 300->360 keep every (b)/(c) ratio >= H1.1's; kv "
        "by sqrt(kp*I_eff) (60->70, 30->35, 20->23). Caps x1.15 (207/161/"
        "80.5) are the authored bigger actuators: (a) hold 3.8x > 1.3x."))

MORPHOLOGIES = {m.name: m for m in (H1, H1_TALL, H1_STOCKY)}


# ---------------------------------------------------------------- builder
def _body_index(name: str) -> int:
    return h1.BODY_NAMES.index(name)


def _length_scales(spec: Morphology) -> dict[int, float]:
    s = {}
    for side in ("left", "right"):
        s[_body_index(f"{side}_upper_leg")] = float(spec.thigh_scale)
        s[_body_index(f"{side}_lower_leg")] = float(spec.shank_scale)
    s[_body_index("torso")] = float(spec.torso_scale)
    return s


def _chain_offsets(joints) -> list[np.ndarray]:
    """Zero-angle offset of every body center from the pelvis center:
    off[child] = off[parent] + ap - ac (body 0 = floor: zeros)."""
    off = [np.zeros(3) for _ in range(h1.B)]
    for j, jt in enumerate(joints):
        parent, ap, ac = jt[1], np.asarray(jt[2], float), np.asarray(jt[3], float)
        off[j + 2] = off[parent] + ap - ac
    return off


def _scaled(x: float, s: float) -> float:
    """x * s, exact identity for s == 1.0, else rounded to 12 decimals
    (keeps derived constants printable: 180 * 1.15 -> 207.0, not
    206.99999999999997)."""
    return x if s == 1.0 else round(x * s, 12)


def _derive_tables(spec: Morphology):
    """(BODIES, JOINTS, EFFORT, KP_TABLE, KV_TABLE) for the member."""
    scales = _length_scales(spec)
    foot_ids = set(h1.FOOT_BODIES)
    bodies = []
    for b, (name, center, half, mass) in enumerate(h1.BODIES):
        hx, hy, hz = half
        if b in scales:
            hy = _scaled(hy, scales[b])
            if spec.link_mass_with_length:
                mass = _scaled(mass, scales[b])
        if b in foot_ids and spec.sole_half_width_add_m:
            hz = round(hz + spec.sole_half_width_add_m, 12)
        if b != h1.FLOOR_BODY:
            mass = _scaled(mass, spec.mass_scale)
        bodies.append([name, tuple(center), (hx, hy, hz), mass])
    joints = []
    for j, jt in enumerate(h1.JOINTS):
        name, parent, ap, ac, lower, upper, effort, axis = jt
        child = j + 2
        sp, sc = scales.get(parent, 1.0), scales.get(child, 1.0)
        ap = (ap[0], _scaled(ap[1], sp), ap[2])
        ac = (ac[0], _scaled(ac[1], sc), ac[2])
        joints.append((name, parent, ap, ac, lower, upper,
                       _scaled(effort, spec.effort_scale), axis))
    # re-seat: centers = authored + (new chain - base chain) + root shift,
    # root shift chosen so the sole plane returns to y = 0 (identity build:
    # every delta is exactly 0.0, so the authored centers pass through).
    base_off = _chain_offsets(h1.JOINTS)
    new_off = _chain_offsets(joints)
    lf = h1.FOOT_BODIES[0]
    d_sole = (new_off[lf][1] - base_off[lf][1]) \
        - (bodies[lf][2][1] - h1.BODIES[lf][2][1])
    root_shift = -d_sole
    for b in range(1, h1.B):
        c = np.asarray(h1.BODIES[b][1], float)
        d = new_off[b] - base_off[b] + np.array([0.0, root_shift, 0.0])
        bodies[b][1] = tuple(float(c[k]) if d[k] == 0.0
                             else round(float(c[k] + d[k]), 12)
                             for k in range(3))
    bodies = tuple(tuple(b) for b in bodies)
    joints = tuple(joints)
    effort = tuple(jt[6] for jt in joints)
    kp = tuple(spec.kp_of(n) for n in h1.JOINT_NAMES)
    kv = tuple(spec.kv_of(n) for n in h1.JOINT_NAMES)
    return bodies, joints, effort, kp, kv


def leg_geometry(lowering) -> dict:
    """{'leg_length_m', 'hip_half_spacing_m', 'hip_height_m', 'thigh_m',
    'shank_m', 'sole_half_length_m', 'sole_half_width_m'} from the
    lowering's JOINTS/BODIES (authored frame; ankle -> hip along the leg)."""
    jn = list(lowering.JOINT_NAMES)
    J = lowering.JOINTS
    hip, knee, ankle = (J[jn.index(n)] for n in ("left_hip", "left_knee",
                                                 "left_ankle"))
    thigh = hip[3][1] - knee[2][1]          # ac_hip_y - ap_knee_y (0.27+0.27)
    shank = knee[3][1] - ankle[2][1]        # ac_knee_y - ap_ankle_y
    foot = lowering.BODIES[lowering.FOOT_BODIES[0]]
    roll = J[jn.index("left_hip_roll")]
    r = lambda x: round(float(x), 12)          # noqa: E731  (0.54+0.32 -> 0.86)
    return {"leg_length_m": r(thigh + shank), "thigh_m": r(thigh),
            "shank_m": r(shank), "hip_half_spacing_m": r(abs(roll[2][2])),
            "ankle_height_m": r(foot[1][1] + ankle[3][1]),   # foot center + ac
            "hip_height_m": r(foot[1][1] + ankle[3][1] + shank + thigh),
            "sole_half_length_m": r(foot[2][0]), "sole_half_width_m": r(foot[2][2])}


def build(spec: Morphology) -> types.SimpleNamespace:
    """A lowering NAMESPACE with h1_lowering's full public surface
    (constants, tables, reset_qpos/fixture/contact_tables/limits/scene/
    foot_vertices/symmetry_spec) for the given member."""
    bodies, joints, effort, kp_table, kv_table = _derive_tables(spec)
    ns = types.SimpleNamespace()
    # inherited scalars (identical for every member)
    for k in ("AUTHORED_DT", "AUTHORED_SUBSTEPS", "SIM_DT", "CONTROL_DT",
              "TICKS_PER_CONTROL", "MAX_EPISODE_STEPS", "GRAVITY", "FRICTION",
              "RESTITUTION", "KP", "KV", "SPEED_LIMIT", "ACCELERATION_LIMIT",
              "ARMATURE", "PASSIVE_DAMPING", "FRICTION_LOSS", "QX90",
              "y_up_to_z_up", "box_inertia", "HIP_LINK_MASS", "HIP_LINK_HALF",
              "HIP_ROLL_LIMIT", "ROLL_AXIS", "BODY_NAMES", "JOINT_NAMES",
              "J", "B", "N", "Q", "FOOT_BODIES", "FLOOR_BODY",
              "REFERENCE_XYZW", "HOME_TARGETS", "H11_GAINS_ENABLED",
              "symmetry_spec"):
        setattr(ns, k, getattr(h1, k))
    ns.VARIANT = spec.name
    ns.MORPHOLOGY = spec
    ns.PROFILE = spec.profile
    ns.BODIES = bodies
    ns.JOINTS = joints
    ns.EFFORT = effort
    ns.HIP_ROLL_EFFORT = joints[h1.JOINT_NAMES.index("left_hip_roll")][6]
    ns.KP_TABLE = kp_table
    ns.KV_TABLE = kv_table
    ns.REWARD_OVERRIDES = dict(spec.reward_overrides)
    ns.TOTAL_DYNAMIC_MASS = sum(b[3] for b in bodies[1:])
    geo = leg_geometry(ns)
    ns.LEG_LENGTH_M = geo["leg_length_m"]
    ns.HIP_HALF_SPACING_M = geo["hip_half_spacing_m"]
    ns.GEOMETRY = geo
    J, Q, N, B = h1.J, h1.Q, h1.N, h1.B
    foot_half = bodies[h1.FOOT_BODIES[0]][2]

    def foot_vertices(half=None):
        return h0.foot_vertices(half if half is not None else foot_half)

    def reset_qpos() -> np.ndarray:
        q = np.zeros(Q)
        q[:3] = h0.y_up_to_z_up(bodies[1][1])
        q[3:7] = h0.QX90
        return q

    def reset_vel() -> np.ndarray:
        return np.zeros(N)

    def fixture(h11_gains: bool | None = None):
        import api as av  # noqa: PLC0415
        from articulated_v1 import Body as _Body  # noqa: PLC0415
        use_tables = (h1.H11_GAINS_ENABLED if h11_gains is None
                      else bool(h11_gains))
        f = av.Fixture(J)
        for b, (_, _, half, mass) in enumerate(bodies):
            if b == 0:
                f.body[b] = _Body(0.0, (np.ctypeslib.ctypes.c_double * 3)(0, 0, 0))
            else:
                f.body[b] = _Body(mass, (np.ctypeslib.ctypes.c_double * 3)(
                    *h0.box_inertia(mass, half)))
        for j, (jt, hinge) in enumerate(zip(joints, f.hinge)):
            name, parent, ap, ac, lower, upper, cap, axis = jt
            hinge.parent = parent
            hinge.ap[:] = ap
            hinge.ac[:] = ac
            hinge.axis[:] = axis
            hinge.reference[:] = h1.REFERENCE_XYZW
            hinge.armature = h1.ARMATURE
            hinge.damping = h1.PASSIVE_DAMPING
            hinge.loss = h1.FRICTION_LOSS
            hinge.kp = kp_table[j] if use_tables else h1.KP
            hinge.kv = kv_table[j] if use_tables else h1.KV
            hinge.cap = cap
            hinge.motor_enabled = 1
        f.model.root_inertia[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        f.reference[:] = reset_qpos()
        f.mapping = {"joints": [{"lower": jt[4], "upper": jt[5]}
                                for jt in joints]}
        return f

    def contact_tables():
        import model_translation as contact  # noqa: PLC0415
        shapes = (contact.Shape * B)()
        for b, shape in enumerate(shapes):
            shape.caller_id = b
            shape.fixed = int(b == h1.FLOOR_BODY)
        shapes[h1.FLOOR_BODY].kind = 2
        shapes[h1.FLOOR_BODY].plane_normal[:] = (0.0, 0.0, 1.0)
        shapes[h1.FLOOR_BODY].plane_offset = 0.0
        for b in h1.FOOT_BODIES:
            shapes[b].kind = 1
            shapes[b].vertex_count = 8
            for i, v in enumerate(foot_vertices()):
                shapes[b].vertices[i][:] = v
        pairs = (contact.Pair * 2)()
        for i, (foot, pair) in enumerate(zip(h1.FOOT_BODIES, pairs)):
            pair.caller_id = i
            pair.body_a = foot
            pair.body_b = h1.FLOOR_BODY
        return shapes, pairs, [h1.FRICTION, h1.FRICTION]

    def limits(f=None):
        import api as av  # noqa: PLC0415
        return av.limits(f if f is not None else fixture())

    def scene(lib, environments: int = 1, joint_offsets=None,
              root_lift: float = 0.0, h11_gains: bool | None = None):
        lane = ROOT / "experimental" / "integrated_duck_v1"
        if str(lane) not in sys.path:
            sys.path.insert(0, str(lane))
        import native  # noqa: PLC0415
        f = fixture(h11_gains=h11_gains)
        q = np.tile(reset_qpos(), (environments, 1))
        q[:, 2] += float(root_lift)
        if joint_offsets is not None:
            off = np.asarray(joint_offsets, dtype="d")
            if off.shape != (environments, J):
                raise ValueError("joint_offsets requires shape [E, J]")
            lim = np.array([(jt[4], jt[5]) for jt in joints])
            q[:, 7:] = np.clip(q[:, 7:] + off, lim[:, 0], lim[:, 1])
        v = np.tile(reset_vel(), (environments, 1))
        shapes, pairs, mu = contact_tables()
        return native.Scene(
            lib, f, q, v, shapes, pairs, np.tile(mu, (environments, 1)),
            gravity=[[0.0, 0.0, -h1.GRAVITY]] * environments,
            limits=limits(f)), f

    ns.foot_vertices = foot_vertices
    ns.reset_qpos = reset_qpos
    ns.reset_vel = reset_vel
    ns.fixture = fixture
    ns.contact_tables = contact_tables
    ns.limits = limits
    ns.scene = scene
    return ns


def export(spec: Morphology, module_globals: dict) -> None:
    """Populate a variant module's globals from build(spec) (the variant
    lowering file is then a plain importable module like h1_lowering)."""
    ns = build(spec)
    module_globals.update({k: v for k, v in vars(ns).items()
                           if not k.startswith("__")})


# ------------------------------------------------------------ registry
def variant_names() -> tuple[str, ...]:
    return tuple(MORPHOLOGIES)


def is_base(name: str | None) -> bool:
    return name in (None, "", "h1")


def load_lowering(name: str | None = None):
    """The lowering MODULE for a family member: "h1"/None -> h1_lowering
    (the exact module the default paths import); else humanoid/<name>_lowering."""
    if is_base(name):
        return h1
    if name not in MORPHOLOGIES:
        raise ValueError(f"unknown humanoid variant {name!r}; "
                         f"known: {sorted(MORPHOLOGIES)}")
    return importlib.import_module(f"{name}_lowering")


def variant_dir(name: str | None) -> Path:
    return _HERE if is_base(name) else VARIANTS_DIR / name


def header_dir(name: str | None) -> Path:
    return variant_dir(name) / "include"


def header_path(name: str | None) -> Path:
    return header_dir(name) / "duck_model.h"


def reference_gait_path(name: str | None) -> Path:
    return variant_dir(name) / "reference_gait.json"


def bc_init_path(name: str | None) -> Path:
    return variant_dir(name) / "bc_init.pt"


def bc_ckpt_path(name: str | None) -> Path:
    return variant_dir(name) / "bc_init_ckpt.pt"


def canonical(name: str | None) -> str:
    return "h1" if is_base(name) else str(name)


def tables_equal(a, b) -> bool:
    """Bit-exact comparison of two lowerings' morphology tables."""
    def flat(x):
        if isinstance(x, (tuple, list)):
            return tuple(flat(y) for y in x)
        if isinstance(x, float):
            return x.hex()
        return x
    keys = ("BODIES", "JOINTS", "EFFORT", "KP_TABLE", "KV_TABLE",
            "HOME_TARGETS", "REFERENCE_XYZW", "FOOT_BODIES", "J", "B")
    return all(flat(getattr(a, k)) == flat(getattr(b, k)) for k in keys) \
        and a.reset_qpos().tobytes() == b.reset_qpos().tobytes() \
        and flat(a.foot_vertices()) == flat(b.foot_vertices()) \
        and math.isclose(a.GRAVITY, b.GRAVITY, rel_tol=0, abs_tol=0)
