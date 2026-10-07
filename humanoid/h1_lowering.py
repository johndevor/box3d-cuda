"""H1 humanoid lowering: H0 + hip-roll lateral actuation (B16/J14/P2).

WHY H1 EXISTS (empirical): the flagship BC+v2.1 GPU leg on H0 stepped
immediately (reward ~2.0) but ep_len saturated at ~69 steps (~1.4 s) and
the judge rejected 0/12 on early termination -- exactly the predicted
lateral instability (humanoid/PHASE2.md sections 8/11): ALL of H0's 12
authored joint axes are sagittal, single support is laterally statically
unstable by ~1 cm, and NO policy on that morphology can regulate roll.
8-second walking requires lateral actuation.

WHAT H1 ADDS (minimal-change): exactly TWO hip-roll joints. Rationale:
- The authored fixture (world/crates/sim/src/humanoid.rs) is strictly
  planar -- it authors NO roll joints anywhere, so any lateral joint is an
  H1 authorship decision; we add the fewest that solve the failure.
- Hip roll (frontal-plane hip strategy) is the primary human lateral
  stabilizer at walking speeds and gives full lateral CoM authority
  (+-0.4 rad sweeps the CoM +-0.34 m at leg length 0.86 m, vs the 1 cm
  deficit). Ankle roll would add 2 more joints, has no authored effort
  tier to join (ankles are the 140 PITCH tier; a roll tier would be
  invented twice over) and is redundant for the failure at hand.
- Effort/gains: hip-roll joins the AUTHORED hip tier (180 N*m, kp 90,
  kv 8, speed 8, accel 40 -- humanoid.rs:778-788 applies its actuator
  constants uniformly per joint class).
- Limits +-0.4 rad: symmetric ab/adduction, ~ human range at gait,
  laterally sweeps +-0.34 m >> the required centimeters, and keeps the
  authored home pose (0) centered.

STRUCTURE: this stack's articulation is strictly body-per-joint
(bodies == joints + 2, child(j) == body j+2 -- articulated_v1.cpp:31), so
each hip-roll joint carries a small inserted "hip link" body between the
pelvis and the upper leg, coincident with the H0 hip point (the standard
2-dof hip decomposition; both joints anchor at the same point, so H1's
home FK is IDENTICAL to H0's for every H0 body). MASS-NEUTRAL: each 0.5 kg
hip link (0.06 m half-extent cube, the roll actuator housing) is carved
out of its thigh's 7.0 kg (-> 6.5 kg); total dynamic mass stays 68.0 kg.
Floor, soles, every other H0 body, anchor, limit, gain, dt, gravity and
friction are inherited UNCHANGED from humanoid/h0_lowering.py (imported,
not copied -- single source of truth).

Body order (child(j) = j+2):   Joint order:
  0 floor          8  right_hip_link     0 waist          7  right_hip
  1 pelvis         9  right_upper_leg    1 neck           8  right_knee
  2 torso         10  right_lower_leg    2 left_hip_roll  9  right_ankle
  3 head          11  right_foot         3 left_hip      10  left_shoulder
  4 left_hip_link 12  left_upper_arm     4 left_knee     11  left_elbow
  5 left_upper_leg 13 left_forearm       5 left_ankle    12  right_shoulder
  6 left_lower_leg 14 right_upper_arm    6 right_hip_roll 13 right_elbow
  7 left_foot     15  right_forearm

Roll axis: parent-local [1, 0, 0] (the forward axis in the authored y-up
body frames; world +x at reset) -- positive roll swings the leg toward
world +y (FK-probed; the "left"-named leg stands at world y = -0.15 after
the z-up lowering rotation).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import h0_lowering as h0  # noqa: E402  (H0 stays importable and untouched)

# ---- re-exported H0 constants (unchanged; see h0_lowering citations) -------
AUTHORED_DT = h0.AUTHORED_DT
AUTHORED_SUBSTEPS = h0.AUTHORED_SUBSTEPS
SIM_DT = h0.SIM_DT
CONTROL_DT = h0.CONTROL_DT
TICKS_PER_CONTROL = h0.TICKS_PER_CONTROL
MAX_EPISODE_STEPS = h0.MAX_EPISODE_STEPS
GRAVITY = h0.GRAVITY
FRICTION = h0.FRICTION
RESTITUTION = h0.RESTITUTION
KP = h0.KP
KV = h0.KV
SPEED_LIMIT = h0.SPEED_LIMIT
ACCELERATION_LIMIT = h0.ACCELERATION_LIMIT
ARMATURE = h0.ARMATURE
PASSIVE_DAMPING = h0.PASSIVE_DAMPING
FRICTION_LOSS = h0.FRICTION_LOSS
QX90 = h0.QX90
y_up_to_z_up = h0.y_up_to_z_up
box_inertia = h0.box_inertia
TOTAL_DYNAMIC_MASS = h0.TOTAL_DYNAMIC_MASS          # 68.0, preserved

PROFILE = "duckgridwalk.humanoid.h1_hip_roll-v1"    # H1 authorship marker

# ---- H1 additions (authored HERE; everything else inherited) ---------------
HIP_LINK_MASS = 0.5            # kg, carved from the thigh (7.0 -> 6.5)
HIP_LINK_HALF = (0.06, 0.06, 0.06)   # m, roll-actuator housing cube
HIP_ROLL_LIMIT = 0.4           # rad, symmetric (docstring rationale)
HIP_ROLL_EFFORT = 180.0        # authored hip tier (humanoid.rs:778-784)
ROLL_AXIS = (1.0, 0.0, 0.0)    # parent-local forward axis (docstring)

# ---- H1.1: per-joint PD gains (PHASE2.md sections 15/16) --------------------
# The body above a hip-roll joint is a lateral inverted pendulum with
# destabilizing stiffness g*sum(m_i*h_i) ~= 388 N*m/rad (56 kg above the
# pivot minus the hanging swing leg below it). kp_roll = 500 = 388 + 29%
# margin: locally stable holds (net stiffness 112 N*m/rad), roll bandwidth
# sqrt(500/1.74) ~= 2.7 Hz >= the 1.67 Hz/mps gait clock, and with the
# authored 180 N*m effort cap the max-restoring stabilizable lean rises to
# 180/388 ~= 0.46 rad >> the 0.175 rad transfer requirement (kp 90 managed
# only 0.116 -- the measured runaway). kv_roll = 60: zeta = kv/(2*sqrt(
# kp*I)) ~= 0.30 on the ~20 kg*m^2 upper-body pendulum, ~1.0 on the 1.74
# kg*m^2 double-support joint inertia; discrete-stability margin
# kv*dt/I_eff = 0.069 << 2. All OTHER joints keep the authored uniform
# 90/8 (sagittal single-support balance runs through the ankle CoP on the
# 0.46 m foot, which kp 90 serves -- minimal-change principle).
# (KP_TABLE/KV_TABLE/H11_GAINS_ENABLED defined after JOINT_NAMES below)

# H0 hip anchor on the pelvis (h0.JOINTS left/right hip parent anchors)
_H0 = {name: j for j, name in enumerate(h0.JOINT_NAMES)}
_L_HIP_ANCHOR = h0.JOINTS[_H0["left_hip"]][2]        # (0, -0.15, +0.15)
_R_HIP_ANCHOR = h0.JOINTS[_H0["right_hip"]][2]       # (0, -0.15, -0.15)
# authored hip-point centers (pelvis center + anchor, y-up frame)
_L_HIP_POINT = (0.0, 1.0, 0.15)
_R_HIP_POINT = (0.0, 1.0, -0.15)

BODY_NAMES = (
    "floor", "pelvis", "torso", "head",
    "left_hip_link", "left_upper_leg", "left_lower_leg", "left_foot",
    "right_hip_link", "right_upper_leg", "right_lower_leg", "right_foot",
    "left_upper_arm", "left_forearm", "right_upper_arm", "right_forearm",
)
JOINT_NAMES = (
    "waist", "neck",
    "left_hip_roll", "left_hip", "left_knee", "left_ankle",
    "right_hip_roll", "right_hip", "right_knee", "right_ankle",
    "left_shoulder", "left_elbow", "right_shoulder", "right_elbow",
)
J = 14
B = 16
N = 6 + J
Q = 7 + J
FOOT_BODIES = (7, 11)
FLOOR_BODY = 0


def _h0_body(name):
    return h0.BODIES[[b[0] for b in h0.BODIES].index(name)]


def _thigh(name):
    n, pos, half, mass = _h0_body(name)
    return (n, pos, half, mass - HIP_LINK_MASS)          # 7.0 -> 6.5


# (name, authored_center_y_up, half_extents, mass) -- H0 rows inherited,
# hip links inserted, thigh mass carved (module docstring).
BODIES = (
    _h0_body("floor"), _h0_body("pelvis"), _h0_body("torso"), _h0_body("head"),
    ("left_hip_link", _L_HIP_POINT, HIP_LINK_HALF, HIP_LINK_MASS),
    _thigh("left_upper_leg"), _h0_body("left_lower_leg"), _h0_body("left_foot"),
    ("right_hip_link", _R_HIP_POINT, HIP_LINK_HALF, HIP_LINK_MASS),
    _thigh("right_upper_leg"), _h0_body("right_lower_leg"),
    _h0_body("right_foot"),
    _h0_body("left_upper_arm"), _h0_body("left_forearm"),
    _h0_body("right_upper_arm"), _h0_body("right_forearm"),
)

_Z3 = (0.0, 0.0, 0.0)


def _h0_joint(name):
    return h0.JOINTS[_H0[name]]


def _row(name, parent, ap, ac, lower, upper, effort, axis=h0.AXIS):
    return (name, parent, ap, ac, lower, upper, effort, axis)


# (name, parent_body, parent_anchor, child_anchor, lower, upper, effort,
#  axis_parent). H0 rows keep their anchors/limits/tiers verbatim; only
# parent INDICES are remapped to the H1 body order, and the hip pitch
# joints re-parent onto the hip links (anchor (0,0,0): both hip joints
# are coincident with the H0 hip point).
JOINTS = (
    _row("waist", 1, *_h0_joint("waist")[2:7]),
    _row("neck", 2, *_h0_joint("neck")[2:7]),
    _row("left_hip_roll", 1, _L_HIP_ANCHOR, _Z3,
         -HIP_ROLL_LIMIT, HIP_ROLL_LIMIT, HIP_ROLL_EFFORT, ROLL_AXIS),
    _row("left_hip", 4, _Z3, *_h0_joint("left_hip")[3:7]),
    _row("left_knee", 5, *_h0_joint("left_knee")[2:7]),
    _row("left_ankle", 6, *_h0_joint("left_ankle")[2:7]),
    _row("right_hip_roll", 1, _R_HIP_ANCHOR, _Z3,
         -HIP_ROLL_LIMIT, HIP_ROLL_LIMIT, HIP_ROLL_EFFORT, ROLL_AXIS),
    _row("right_hip", 8, _Z3, *_h0_joint("right_hip")[3:7]),
    _row("right_knee", 9, *_h0_joint("right_knee")[2:7]),
    _row("right_ankle", 10, *_h0_joint("right_ankle")[2:7]),
    _row("left_shoulder", 2, *_h0_joint("left_shoulder")[2:7]),
    _row("left_elbow", 12, *_h0_joint("left_elbow")[2:7]),
    _row("right_shoulder", 2, *_h0_joint("right_shoulder")[2:7]),
    _row("right_elbow", 14, *_h0_joint("right_elbow")[2:7]),
)
REFERENCE_XYZW = h0.REFERENCE_XYZW      # identity for every joint
EFFORT = tuple(j[6] for j in JOINTS)
HOME_TARGETS = tuple(0.0 for _ in range(J))
# v3.2 executed-sweep finding (PHASE2.md section 16): the SAME static-
# instability class recurs down the leg -- mass-above-joint x g x CoM
# height exceeds kp 90 at the knee too (~62 kg x 20 x 0.55 ~= 680 N*m/rad
# buckling stiffness; the measured single-support pelvis sink ate the
# swing clearance), and the stance hip pitch / ankle need matching
# authority to hold posture through the transfer. The unlocking set,
# validated by the first executed QUALIFIED swing: knee 800/30 (> 680 +
# margin), hip pitch and ankle 300/20 (support roles; empirically
# sufficient, larger values not needed), hip roll 500/60 (the section-15
# pendulum spec), everything else authored 90/8.
def _gain(name, roll, knee, hip_ankle, other):
    if "hip_roll" in name:
        return roll
    if "knee" in name:
        return knee
    if "hip" in name or "ankle" in name:
        return hip_ankle
    return other


KP_TABLE = tuple(_gain(n, 500.0, 800.0, 300.0, h0.KP) for n in JOINT_NAMES)
KV_TABLE = tuple(_gain(n, 60.0, 30.0, 20.0, h0.KV) for n in JOINT_NAMES)
# ACTIVE since the kernel's DW_KP_TABLE/DW_KV_TABLE consumption landed
# (the generator emits these tables straight from KP_TABLE/KV_TABLE, so
# the fp32 lane and the f64 oracle apply identical per-joint gains --
# parity preserved).
H11_GAINS_ENABLED = True


def foot_vertices(half=None):
    return h0.foot_vertices(half if half is not None
                            else _h0_body("left_foot")[2])


def reset_qpos() -> np.ndarray:
    q = np.zeros(Q)
    q[:3] = y_up_to_z_up(BODIES[1][1])
    q[3:7] = QX90
    return q


def reset_vel() -> np.ndarray:
    return np.zeros(N)


def fixture(h11_gains: bool | None = None):
    """H1 as an articulated_v2 fixture (same lowering rules as h0.fixture)."""
    import api as av  # noqa: PLC0415  (path set up by h0_lowering import)
    from articulated_v1 import Body as _Body  # noqa: PLC0415
    use_tables = H11_GAINS_ENABLED if h11_gains is None else bool(h11_gains)
    f = av.Fixture(J)
    for b, (_, _, half, mass) in enumerate(BODIES):
        if b == 0:
            f.body[b] = _Body(0.0, (np.ctypeslib.ctypes.c_double * 3)(0, 0, 0))
        else:
            f.body[b] = _Body(mass, (np.ctypeslib.ctypes.c_double * 3)(
                *box_inertia(mass, half)))
    for j, (jt, hinge) in enumerate(zip(JOINTS, f.hinge)):
        name, parent, ap, ac, lower, upper, effort, axis = jt
        hinge.parent = parent
        hinge.ap[:] = ap
        hinge.ac[:] = ac
        hinge.axis[:] = axis
        hinge.reference[:] = REFERENCE_XYZW
        hinge.armature = ARMATURE
        hinge.damping = PASSIVE_DAMPING
        hinge.loss = FRICTION_LOSS
        hinge.kp = KP_TABLE[j] if use_tables else KP
        hinge.kv = KV_TABLE[j] if use_tables else KV
        hinge.cap = effort
        hinge.motor_enabled = 1
    f.model.root_inertia[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    f.reference[:] = reset_qpos()
    f.mapping = {"joints": [{"lower": jt[4], "upper": jt[5]} for jt in JOINTS]}
    return f


def contact_tables():
    """(shapes[B], pairs[2], mu[2]): floor plane + the two H0 box feet."""
    import model_translation as contact  # noqa: PLC0415
    shapes = (contact.Shape * B)()
    for b, shape in enumerate(shapes):
        shape.caller_id = b
        shape.fixed = int(b == FLOOR_BODY)
    shapes[FLOOR_BODY].kind = 2
    shapes[FLOOR_BODY].plane_normal[:] = (0.0, 0.0, 1.0)
    shapes[FLOOR_BODY].plane_offset = 0.0
    for b in FOOT_BODIES:
        shapes[b].kind = 1
        shapes[b].vertex_count = 8
        for i, v in enumerate(foot_vertices()):
            shapes[b].vertices[i][:] = v
    pairs = (contact.Pair * 2)()
    for i, (foot, pair) in enumerate(zip(FOOT_BODIES, pairs)):
        pair.caller_id = i
        pair.body_a = foot
        pair.body_b = FLOOR_BODY
    return shapes, pairs, [FRICTION, FRICTION]


def limits(f=None):
    import api as av  # noqa: PLC0415
    return av.limits(f if f is not None else fixture())


def scene(lib, environments: int = 1, joint_offsets=None,
          root_lift: float = 0.0, h11_gains: bool | None = None):
    """Registered idv1 Scene at the H1 floor-clear reset (h0.scene twin)."""
    lane = _HERE.parent / "experimental" / "integrated_duck_v1"
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
        lim = np.array([(jt[4], jt[5]) for jt in JOINTS])
        q[:, 7:] = np.clip(q[:, 7:] + off, lim[:, 0], lim[:, 1])
    v = np.tile(reset_vel(), (environments, 1))
    shapes, pairs, mu = contact_tables()
    return native.Scene(
        lib, f, q, v, shapes, pairs, np.tile(mu, (environments, 1)),
        gravity=[[0.0, 0.0, -GRAVITY]] * environments, limits=limits(f)), f


# ---- L/R mirror symmetry spec (PPO symmetry augmentation) -------------------
# Mirror about the sagittal (x-z world) plane, y -> -y. Joint mapping:
# left_* <-> right_*; SAGITTAL (pitch, local-z axis) joints keep sign;
# ROLL (local-x axis) joints FLIP sign (a left lean mirrors to a right
# lean). Body-frame obs vectors under the mirror (body x=fwd, y=up,
# z=world -y): true vectors (gravity, linear velocity) flip their z
# component; pseudo-vectors (angular velocity) flip x and y components.
# The phase clock's left/right semantics swap under mirror = phase + pi:
# (sin, cos) -> (-sin, -cos). Contacts swap L/R; command/zeros unchanged.
def symmetry_spec():
    """{'obs_perm','obs_sign','act_perm','act_sign'} numpy arrays for the
    walk/env/humanoid_flat.py observation layout (3J + 16). Verified by
    humanoid/tests/test_symmetry.py: involution + mirrored-physics."""
    names = list(JOINT_NAMES)

    def mirror_joint(n):
        if n.startswith("left_"):
            return names.index("right_" + n[5:])
        if n.startswith("right_"):
            return names.index("left_" + n[6:])
        return names.index(n)                     # waist / neck
    act_perm = np.array([mirror_joint(n) for n in names])
    act_sign = np.array([-1.0 if "roll" in n else 1.0 for n in names])
    T = 3 * J
    obs_perm = np.arange(T + 16)
    obs_sign = np.ones(T + 16)
    for block in range(3):                        # q, qdot, prev action
        base = block * J
        obs_perm[base:base + J] = base + act_perm
        obs_sign[base:base + J] = act_sign
    obs_sign[T + 2] = -1.0                        # gravity body-z
    obs_sign[T + 3] = -1.0                        # omega body-x
    obs_sign[T + 4] = -1.0                        # omega body-y
    obs_sign[T + 8] = -1.0                        # linear velocity body-z
    obs_perm[[T + 12, T + 13]] = [T + 13, T + 12]  # contacts swap L/R
    obs_sign[T + 14] = -1.0                       # sin(phase + pi)
    obs_sign[T + 15] = -1.0                       # cos(phase + pi)
    return {"obs_perm": obs_perm, "obs_sign": obs_sign,
            "act_perm": act_perm, "act_sign": act_sign}
