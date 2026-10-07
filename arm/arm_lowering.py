"""Fixed-base 6-axis arm lowering for the robot-generic duck stack.

ONE builder, TWO variants (the sim2real "distribution of robots of one
category" point): every constant of both variants is derived here from the
pinned KR240 URDF by pure data transformation -- no simulation, no invented
morphology. Consumers: the f64 CPU oracle lane (walk/env/arm_native_lane.py,
over experimental/integrated_duck_v1), the fp32 kernel header generator
(experimental/duck_cuda/tools/generate_model_arm.py) and the reach env /
judge (walk/env/arm_reach.py, walk/eval/arm_reach_judge.py).

SOURCES (every number is cited; the lite variant is a documented scaling):
  [URDF] /Users/john/Code/box3d-arm-lab/factory_os/assets/generated/
         kr240r2900.physics.urdf  (sha256 26e37d3081f1..., pinned by
         kr240r2900.physics.asset.json): link masses (430/260/180/125/65/38/
         22 kg = 1120 kg total), COMs, diagonal inertias, joint origins/axes/
         limits/efforts/velocities/damping/friction, flange offset.
         "calibration_class": "bounded_engineering_approximation" -- NOT
         manufacturer values (asset json "unresolved_blockers").
  [JSON] .../physics_comparisons/daytona-kr240-joints-20260829-r70/
         kr240-joints.json: the pinned KR240 joint-runtime contract
         (7 bodies, 6 joints, y-up frame, gravity 9.81 magnitude); its
         world0_body_state rows cross-check the FK below
         (arm/tests/test_arm_lowering.py).
  [KIN]  kernel/oracle hinge convention: experimental/duck_cuda/src/
         duck_cuda_kernel.h dw_evaluate (pos_b = pos_p + R_p*AP - R_b*AC,
         R_b = R_p * exp(axis_parent*q) * REF) == articulated_v1.cpp:63-64.

FRAME CONVENTION. The URDF is z-up with base_link at the world origin, the
same frame as this stack (floor plane z = 0, gravity on -z), so NO world
rotation is needed (unlike the y-up humanoid). URDF joint j: child frame =
parent frame * T(origin_xyz, rpy) * Rot(axis_child, q). The stack's hinge
is R_b = R_p * exp(axis_parent q) * REF, so axis_parent = R_rpy @ axis_child
and REF = quat(R_rpy) (exp(R a q) R = R exp(a q)); anchors are expressed in
the bodies' principal COM frames: AP = origin_xyz - com_parent,
AC = -com_child (inertias are diagonal with rpy 0, so each link's principal
frame is its link frame translated to the COM).

FIXED BASE (arm/FEASIBILITY.md section 1). The kernel welds structurally:
joint 0's parent is the STATIC FLOOR body 0 (dw_evaluate initialises body 0
to identity pose / zero motion / colmask 0 before the FK loop and indexes
parents through the DW_HINGE_PARENT table), so the base_link is fused into
the world and the kernel's mandatory free root (body 1) becomes a PHANTOM
carrying the base_link's mass/inertia whose 6-dof block is exactly
decoupled from the arm (no arm body's colmask has a root bit). The f64
oracle (articulated_v1.cpp:39 rejects parent < 1) welds virtually: the root
IS base_link, scaled WELD_MASS_FACTOR heavier, with its weight cancelled
exactly and a stiff explicit PD on its 6 root dofs applied through
av2_step.applied_force (native.Scene.step(force=...)). Both welds are
proven by arm/tests (home-hold drift < 1e-6 m over 8 s).

GRAVITY: authored -9.81 m/s^2 here, matching the KR240 contract's 9.81
magnitude ([JSON] gravity); the humanoid lowering is authored at -20
(world/crates/sim/src/lib.rs:48). The two are NOT unified on purpose: each
robot carries its authoring engine's gravity (FEASIBILITY.md section 3).
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "experimental" / "articulated_v2" / "tests"),
           str(ROOT / "experimental" / "contact_v1"),
           str(ROOT / "experimental" / "integrated_duck_v1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

J = 6                                  # revolute joints (both variants)
B = J + 2                              # floor 0, root/base 1, links 2..7
N = 6 + J
Q = 7 + J
FLOOR_BODY = 0
ROOT_BODY = 1
LINK_BODIES = tuple(range(2, 2 + J))   # link_k is body k+1
TIP_BODY = 1 + J                       # link_6

# ------------------------------------------------------------- integration
SIM_DT = 0.002                         # duck-stack tick (walk/env/flat.py)
CONTROL_DT = 0.02                      # policy step = 10 ticks
TICKS_PER_CONTROL = 10
GRAVITY = 9.81                         # [JSON] magnitude; authored (0,0,-9.81)
GRAVITY_VEC = (0.0, 0.0, -GRAVITY)

# stack solver constants (NOT morphology; the same defaults every fixture in
# this repo uses: articulated_v2/tests/api.py:24, h0_lowering.py)
FRICTION_D0, FRICTION_DWIDTH, FRICTION_TIMECONST = 0.9, 0.95, 0.02
LIMIT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
LIMIT_TIMECONST, LIMIT_DAMPRATIO, LIMIT_MARGIN = 0.02, 1.0, 0.0
ARMATURE = 0.0                         # not authored by [URDF]
PASSIVE_DAMPING = 0.0                  # URDF damping is folded into kv (below)
FRICTION_LOSS = 0.0                    # URDF Coulomb friction 3-12 N*m (<=0.1%
#                                        of effort) dropped: the kernel has ONE
#                                        scalar DW_FRICTION_LOSS, so a per-joint
#                                        value cannot be lowered faithfully

# ------------------------------------------------------ gain derivation
# kp_j = max(bandwidth, sag, authority) rounded to 3 significant digits:
#   bandwidth: sqrt(kp/I_max) >= BANDWIDTH_FACTOR * OMEGA_CMD, where
#              OMEGA_CMD = 2*pi * (JUDGE_TARGETS / JUDGE_EPISODE_S) is the
#              target-change rate the reach judge demands (5 targets / 8 s),
#              I_max = joint-space inertia M_jj at the fully stretched pose;
#   sag:       gravity torque at full horizontal extension / SAG_MAX_RAD;
#   authority: effort_cap / FULL_TORQUE_ERR_RAD (full torque within 0.1 rad).
# kv_j = 2*ZETA*sqrt(kp_j*I_max_j) + URDF damping_j (folded: kernel DAMPING is
#        a single scalar, kv is a per-joint table).
BANDWIDTH_FACTOR = 3.0
JUDGE_TARGETS = 5
JUDGE_EPISODE_S = 8.0
OMEGA_CMD = 2.0 * math.pi * JUDGE_TARGETS / JUDGE_EPISODE_S
SAG_MAX_RAD = 0.01
FULL_TORQUE_ERR_RAD = 0.1
ZETA = 0.7
# discrete-time cap: sqrt(kp/I_min)*dt <= OMEGA_MAX_DT (8 ticks per radian of
# the stiffest mode; the explicit-PD stability bound is 2). Binds only on the
# light wrist joints (a4/a6), whose authored efforts are "bounded
# approximations" far above what their inertia can use.
OMEGA_MAX_DT = 0.25

# fp32 certificates, SCALE-AWARE (humanoid/include/duck_model.h rationale):
# the humanoid pins 1.75e-4 / 7e-3 / 2.4e-2 on a 2.72 N*s per-tick weight
# impulse. The arm's reference impulse is max(effort)*dt (the largest joint
# impulse a tick can carry); the same RELATIVE pins apply.
SOLVE_TOL_REL = 1.75e-4 / 2.72
MOMENTUM_TOL_REL = 7e-3 / 2.72
TIER_CEILING_REL = 2.4e-2 / 2.72

# f64 oracle virtual weld (root = base_link)
WELD_MASS_FACTOR = 1.0e6               # root mass = factor * moving mass
WELD_PIN_OMEGA_DT = 0.2                # explicit PD: omega*dt (stable < 2)
WELD_PIN_ZETA = 1.0

# kernel contact placeholder: the kernel hard-codes 2 foot/floor pairs; the
# arm has none, so both pairs point at link_1 (body 2) with a small box that
# sits ~0.7 m above the floor for every q (link_1 only yaws) -> never a
# contact row, never a solver stall. Verified by the tests (contact_points 0).
PLACEHOLDER_PAIR_BODY = 2
PLACEHOLDER_HALF_FRAC = 0.1            # box half-extent = frac * reach scale


@dataclasses.dataclass(frozen=True)
class Link:
    name: str
    mass: float
    com: tuple            # link-frame COM offset [URDF inertial origin]
    inertia: tuple        # diagonal (ixx, iyy, izz) about the COM


@dataclasses.dataclass(frozen=True)
class Joint:
    name: str
    xyz: tuple            # origin in parent link frame
    rpy: tuple            # origin rotation (URDF roll-pitch-yaw)
    axis: tuple           # in the child/joint frame
    lower: float
    upper: float
    effort: float         # N*m cap
    velocity: float       # rad/s limit
    damping: float        # N*m*s/rad (URDF dynamics)
    friction: float       # N*m (URDF dynamics; dropped, see FRICTION_LOSS)


@dataclasses.dataclass(frozen=True)
class ArmSpec:
    variant: str
    base: Link
    links: tuple          # 6 Links (link_1..link_6)
    joints: tuple         # 6 Joints (joint_a1..a6)
    tool_xyz: tuple       # flange origin in the link_6 frame
    home_q: tuple         # reset / HOME joint pose (rad)
    length_scale: float   # 1.0 for the pinned KR240
    mass_scale: float


# --------------------------------------------------------- KR240 (pinned)
_KR240_BASE = Link("base_link", 430.0, (-0.141742, 0.0, 0.237581),
                   (48.205008, 70.730475, 102.754591))
_KR240_LINKS = (
    Link("link_1", 260.0, (-0.029873, -0.002869, 0.155299),
         (24.986381, 57.501733, 50.373232)),
    Link("link_2", 180.0, (0.709669, -0.266732, -0.003029),
         (11.600606, 62.861755, 60.634472)),
    Link("link_3", 125.0, (0.232041, 0.076180, -0.044416),
         (6.950178, 29.116040, 29.823052)),
    Link("link_4", 65.0, (0.106082, 0.0, -0.000035),
         (0.718546, 1.405056, 1.405089)),
    Link("link_5", 38.0, (0.058640, 0.018541, -0.000030),
         (0.714865, 0.707022, 0.959171)),
    Link("link_6", 22.0, (-0.256887, 0.0, -0.000042),
         (0.120170, 0.062249, 0.062266)),
)
_PI = 3.1415927                        # the URDF's literal
_KR240_JOINTS = (
    Joint("joint_a1", (0.0, 0.0, 0.860625), (_PI, 0.0, _PI), (0.0, 0.0, 1.0),
          -3.2288591, 3.2288591, 12000.0, 1.797689, 70.0, 12.0),
    Joint("joint_a2", (-0.44625, 0.0, 0.0), (_PI, 0.0, _PI), (0.0, 1.0, 0.0),
          -2.7052601, 0.6108652, 12000.0, 1.640610, 70.0, 12.0),
    Joint("joint_a3", (1.46625, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
          -2.268928, 2.687807, 9000.0, 1.745329, 55.0, 10.0),
    Joint("joint_a4", (1.2750001, 0.0, -0.052275), (0.0, 0.0, _PI),
          (1.0, 0.0, 0.0), -6.1086524, 6.1086524, 3000.0, 2.967060, 25.0, 5.0),
    Joint("joint_a5", (0.0, 0.0, 0.0), (0.0, 0.0, _PI), (0.0, 1.0, 0.0),
          -2.268928, 2.268928, 2000.0, 2.251475, 18.0, 4.0),
    Joint("joint_a6", (0.0, 0.0, 0.0), (0.0, 0.0, _PI), (1.0, 0.0, 0.0),
          -6.1086524, 6.1086524, 1200.0, 3.595378, 12.0, 3.0),
)
# HOME: a mid-workspace "ready" pose (shoulder raised 69 deg, elbow bent,
# wrist pitched level) whose flange sits at (2.41, 0, 1.68) m for the KR240
# -- 70 % of the stretched reach, elbow at z 2.23 m -- with every joint far
# from its limits (FK-pinned in arm/tests/test_arm_lowering.py). The URDF's
# q = 0 is the fully stretched horizontal arm: the worst-case gravity pose,
# used for the feasibility tables, never for reset.
_HOME_Q = (0.0, -1.2, 1.6, 0.0, -0.4, 0.0)

KR240 = ArmSpec("kr240", _KR240_BASE, _KR240_LINKS, _KR240_JOINTS,
                (-0.274, 0.0, 0.0), _HOME_Q, 1.0, 1.0)


def scaled(spec: ArmSpec, variant: str, length_scale: float,
           mass_scale: float) -> ArmSpec:
    """Dynamically similar (Froude) scaling of a spec: lengths x s_L, masses
    x s_m, inertia x s_m*s_L^2, torque (effort) x s_m*s_L, angular rates x
    s_L^-1/2 (time scales with sqrt(L)), joint damping x s_m*s_L^1.5;
    limits, axes and rpy are dimensionless and unchanged."""
    sL, sm = float(length_scale), float(mass_scale)
    s_t = math.sqrt(sL)

    def link(l: Link) -> Link:
        return Link(l.name, l.mass * sm, tuple(c * sL for c in l.com),
                    tuple(i * sm * sL * sL for i in l.inertia))

    def joint(j: Joint) -> Joint:
        return Joint(j.name, tuple(x * sL for x in j.xyz), j.rpy, j.axis,
                     j.lower, j.upper, j.effort * sm * sL, j.velocity / s_t,
                     j.damping * sm * sL * sL / s_t, j.friction * sm * sL)
    return ArmSpec(variant, link(spec.base), tuple(map(link, spec.links)),
                   tuple(map(joint, spec.joints)),
                   tuple(x * sL for x in spec.tool_xyz), spec.home_q,
                   spec.length_scale * sL, spec.mass_scale * sm)


# "lite": half-scale, 1/8-mass KR240 (UR10-class reach 1.6 m to the wrist
# centre / 1.73 m to the flange; 86 kg moving mass -- a geometric 1/8 of the
# KR240's 690 kg moving mass, NOT the 33 kg of a real UR10, which would be
# ~1/23 mass; see FEASIBILITY.md section 2).
LITE = scaled(KR240, "lite", 0.5, 0.125)
VARIANTS = {"kr240": KR240, "lite": LITE}


def spec(variant: str) -> ArmSpec:
    try:
        return VARIANTS[variant]
    except KeyError:
        raise ValueError(f"unknown arm variant {variant!r}; "
                         f"choose from {sorted(VARIANTS)}") from None


# ------------------------------------------------------------ rotations
def rpy_to_rot(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """xyzw unit quaternion of a rotation matrix (Shepperd, w >= 0)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(1e-300, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2
        q = np.zeros(4)
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    if q[3] < 0:
        q = -q
    return q / np.linalg.norm(q)


def axis_angle_rot(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def quat_to_rot(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# --------------------------------------------------------- hinge tables
def hinge_rows(s: ArmSpec, kernel: bool):
    """Per-joint (parent_body, AP, AC, axis_parent, REF_xyzw) in the stack's
    hinge convention (module docstring). kernel=True parents joint 0 on the
    static floor (body 0, world frame == base_link frame); kernel=False
    parents it on the root body 1 (base_link, principal COM frame)."""
    rows = []
    for j, jt in enumerate(s.joints):
        R_o = rpy_to_rot(jt.rpy)
        if j == 0:
            parent = FLOOR_BODY if kernel else ROOT_BODY
            com_p = np.zeros(3) if kernel else np.asarray(s.base.com)
        else:
            parent = LINK_BODIES[j - 1]
            com_p = np.asarray(s.links[j - 1].com)
        com_c = np.asarray(s.links[j].com)
        ap = np.asarray(jt.xyz) - com_p
        ac = -com_c
        axis_p = R_o @ np.asarray(jt.axis, float)
        rows.append((int(parent), ap, ac, axis_p, rot_to_quat(R_o)))
    return rows


def hinge_parents(s: ArmSpec, kernel: bool) -> tuple:
    return tuple(r[0] for r in hinge_rows(s, kernel))


def joint_limits(s: ArmSpec) -> np.ndarray:
    return np.array([(j.lower, j.upper) for j in s.joints])


def effort(s: ArmSpec) -> np.ndarray:
    return np.array([j.effort for j in s.joints])


def velocity_limits(s: ArmSpec) -> np.ndarray:
    return np.array([j.velocity for j in s.joints])


def body_masses(s: ArmSpec) -> list:
    return [0.0, s.base.mass] + [l.mass for l in s.links]


def body_inertias(s: ArmSpec) -> list:
    return [(0.0, 0.0, 0.0), tuple(s.base.inertia)] + [tuple(l.inertia) for l in s.links]


# ------------------------------------------------------------------- FK
@dataclasses.dataclass
class FK:
    joint_pos: np.ndarray   # [J,3] world joint origins
    link_pos: np.ndarray    # [J,3] world link COMs (bodies 2..7)
    link_rot: np.ndarray    # [J,3,3] world link rotations
    axis: np.ndarray        # [J,3] world joint axes
    tip: np.ndarray         # [3] flange origin (world)


def fk(s: ArmSpec, q) -> FK:
    """URDF-style forward kinematics from the world-fixed base_link frame."""
    q = np.asarray(q, float).reshape(J)
    R = np.eye(3)
    p = np.zeros(3)
    jp, lp, lr, ax = [], [], [], []
    for j, jt in enumerate(s.joints):
        p = p + R @ np.asarray(jt.xyz)
        R = R @ rpy_to_rot(jt.rpy)
        jp.append(p.copy())
        R = R @ axis_angle_rot(jt.axis, q[j])
        ax.append(R @ np.asarray(jt.axis, float))
        lp.append(p + R @ np.asarray(s.links[j].com))
        lr.append(R.copy())
    tip = p + R @ np.asarray(s.tool_xyz)
    return FK(np.array(jp), np.array(lp), np.array(lr), np.array(ax), tip)


def _rodrigues_batch(axis, angles: np.ndarray) -> np.ndarray:
    """[E,3,3] rotations about a fixed unit axis by [E] angles."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    sn = np.sin(angles)[:, None, None]
    cs = np.cos(angles)[:, None, None]
    return np.eye(3)[None] + sn * K[None] + (1.0 - cs) * (K @ K)[None]


def fk_batch(s: ArmSpec, q: np.ndarray):
    """Vectorised FK: (tip [E,3], joint origins [E,J,3]) for [E,J] poses.
    Same chain as fk(); used by the reach env's batched target sampler
    (arm/tests pin fk_batch == fk to 1e-12)."""
    q = np.asarray(q, float).reshape(-1, J)
    E = q.shape[0]
    R = np.broadcast_to(np.eye(3), (E, 3, 3)).copy()
    p = np.zeros((E, 3))
    origins = np.zeros((E, J, 3))
    for j, jt in enumerate(s.joints):
        p = p + R @ np.asarray(jt.xyz)
        R = R @ rpy_to_rot(jt.rpy)[None]
        origins[:, j] = p
        R = R @ _rodrigues_batch(jt.axis, q[:, j])
    return p + R @ np.asarray(s.tool_xyz), origins


def fk_batch_tip(s: ArmSpec, q: np.ndarray) -> np.ndarray:
    """[E,3] tip positions for [E,J] joint poses."""
    return fk_batch(s, q)[0]


def tip_from_body_state(s: ArmSpec, body_state: np.ndarray) -> np.ndarray:
    """Tip (flange) world position from a lane body_state [E,B,13] read:
    tip = p_link6 + R_link6 @ (tool_xyz - com_link6)."""
    b = np.asarray(body_state, float)[:, TIP_BODY, :]
    rot = np.stack([quat_to_rot(qq) for qq in b[:, 3:7]])
    off = np.asarray(s.tool_xyz) - np.asarray(s.links[-1].com)
    return b[:, :3] + rot @ off


def mass_matrix_and_gravity(s: ArmSpec, q):
    """Joint-space inertia M [J,J] and generalized gravity torque tau_g [J]
    (the PD must supply -tau_g to hold), from the FK jacobians:
    M = sum_b m_b Jv^T Jv + Jw^T I_w Jw, tau_g = sum_b m_b Jv^T g."""
    f = fk(s, q)
    M = np.zeros((J, J))
    tau = np.zeros(J)
    g = np.asarray(GRAVITY_VEC)
    for b in range(J):
        Jv = np.zeros((3, J))
        Jw = np.zeros((3, J))
        for k in range(b + 1):
            Jv[:, k] = np.cross(f.axis[k], f.link_pos[b] - f.joint_pos[k])
            Jw[:, k] = f.axis[k]
        Iw = f.link_rot[b] @ np.diag(s.links[b].inertia) @ f.link_rot[b].T
        M += s.links[b].mass * Jv.T @ Jv + Jw.T @ Iw @ Jw
        tau += s.links[b].mass * Jv.T @ g
    return M, tau


def reach(s: ArmSpec) -> float:
    """Nominal reach: horizontal distance of the flange from the a1 axis at
    the fully stretched URDF pose q = 0 (KR240: 3.46 m; KUKA quotes 2.9 m
    to the wrist centre for the physical robot)."""
    tip = fk(s, np.zeros(J)).tip
    return float(math.hypot(tip[0], tip[1]))


def home_tip(s: ArmSpec) -> np.ndarray:
    return fk(s, s.home_q).tip


# ------------------------------------------------------------- gains
def _round_sig(x: float, sig: int = 3) -> float:
    if x == 0:
        return 0.0
    e = math.floor(math.log10(abs(x)))
    return round(x, sig - 1 - e)


_POSE_GRID = None


def pose_grid() -> np.ndarray:
    """Deterministic joint-pose sample for min/max inertia scans: the
    stretched pose, the home pose and a 3^3 grid over a2/a3/a5 (the
    inertia-shaping joints) at their 10/50/90 % limit fractions."""
    global _POSE_GRID
    if _POSE_GRID is None:
        lim = joint_limits(KR240)
        rows = [np.zeros(J), np.asarray(_HOME_Q)]
        for f2 in (0.1, 0.5, 0.9):
            for f3 in (0.1, 0.5, 0.9):
                for f5 in (0.1, 0.5, 0.9):
                    q = np.zeros(J)
                    q[1] = lim[1, 0] + f2 * (lim[1, 1] - lim[1, 0])
                    q[2] = lim[2, 0] + f3 * (lim[2, 1] - lim[2, 0])
                    q[4] = lim[4, 0] + f5 * (lim[4, 1] - lim[4, 0])
                    rows.append(q)
        _POSE_GRID = np.array(rows)
    return _POSE_GRID


def inertia_scan(s: ArmSpec):
    """(I_max[J], I_min[J], tau_hold[J]) over pose_grid(): diagonal joint-
    space inertias and the max |gravity torque| per joint (the stretched
    pose q = 0 is in the grid, so tau_hold covers full horizontal
    extension)."""
    diag = []
    tau = []
    for q in pose_grid():
        M, t = mass_matrix_and_gravity(s, q)
        diag.append(np.diag(M))
        tau.append(np.abs(t))
    diag = np.array(diag)
    return diag.max(0), diag.min(0), np.array(tau).max(0)


def gains(s: ArmSpec):
    """(kp[J], kv[J]) per the module-docstring derivation, 3 significant
    digits (exactly representable in the fp32 header)."""
    i_max, i_min, tau_hold = inertia_scan(s)
    cap = effort(s)
    kp = np.maximum.reduce([
        (BANDWIDTH_FACTOR * OMEGA_CMD) ** 2 * i_max,
        tau_hold / SAG_MAX_RAD,
        cap / FULL_TORQUE_ERR_RAD,
    ])
    kp = np.minimum(kp, (OMEGA_MAX_DT / SIM_DT) ** 2 * i_min)
    kp = np.array([_round_sig(x) for x in kp])
    kv = 2.0 * ZETA * np.sqrt(kp * i_max) + np.array([j.damping for j in s.joints])
    kv = np.array([_round_sig(x) for x in kv])
    # float32-exact: the oracle's av1_hinge stiffness/damping ARE c_float and
    # the header rounds to f32 once; every consumer (env torque estimate
    # included) must see the identical values.
    return (kp.astype(np.float32).astype(np.float64),
            kv.astype(np.float32).astype(np.float64))


def reference_impulse(s: ArmSpec) -> float:
    return float(effort(s).max() * SIM_DT)


def certificates(s: ArmSpec) -> dict:
    ref = reference_impulse(s)
    return {"solve_tolerance": _round_sig(SOLVE_TOL_REL * ref, 3),
            "momentum_tolerance": _round_sig(MOMENTUM_TOL_REL * ref, 3),
            "tier_ceiling": _round_sig(TIER_CEILING_REL * ref, 3),
            "reference_impulse": ref}


def moving_mass(s: ArmSpec) -> float:
    return float(sum(l.mass for l in s.links))


def weld(s: ArmSpec) -> dict:
    """Oracle virtual-weld parameters (root = base_link)."""
    m = WELD_MASS_FACTOR * moving_mass(s)
    k = (WELD_PIN_OMEGA_DT / SIM_DT) ** 2 * m
    d = 2.0 * WELD_PIN_ZETA * math.sqrt(k * m)
    # rotational pin uses the same numbers on a unit-lever inertia (I = m*1m^2)
    return {"mass": m, "inertia": (m, m, m), "k_lin": k, "d_lin": d,
            "k_ang": k, "d_ang": d}


def placeholder_vertices(s: ArmSpec) -> list:
    h = PLACEHOLDER_HALF_FRAC * s.length_scale
    return [tuple(h * (1.0 if i & (1 << k) else -1.0) for k in range(3))
            for i in range(8)]


# ----------------------------------------------------------- reset state
def reset_qpos(s: ArmSpec) -> np.ndarray:
    """[Q] root xyz (base_link frame at the world origin), quat identity,
    home joint pose. Identical for both lanes (the kernel's phantom root
    starts here and free-falls, decoupled; the oracle's welded root stays)."""
    q = np.zeros(Q)
    q[6] = 1.0
    q[7:] = s.home_q
    return q


def reset_vel() -> np.ndarray:
    return np.zeros(N)


# ---------------------------------------------------------- oracle fixture
def fixture(s: ArmSpec):
    """articulated_v2 Fixture of the arm with the WELDED root (f64 oracle)."""
    import api as av  # noqa: PLC0415
    from articulated_v1 import Body as _Body  # noqa: PLC0415
    import ctypes as C  # noqa: PLC0415
    f = av.Fixture(J)
    w = weld(s)
    masses = body_masses(s)
    inertias = body_inertias(s)
    masses[ROOT_BODY] = w["mass"]
    inertias[ROOT_BODY] = w["inertia"]
    for b in range(B):
        f.body[b] = _Body(masses[b], (C.c_double * 3)(*inertias[b]))
    kp, kv = gains(s)
    for j, (h, row) in enumerate(zip(f.hinge, hinge_rows(s, kernel=False))):
        parent, ap, ac, axis_p, ref = row
        h.parent = parent
        h.ap[:] = ap.tolist()
        h.ac[:] = ac.tolist()
        h.axis[:] = axis_p.tolist()
        h.reference[:] = ref.tolist()
        h.armature = ARMATURE
        h.damping = PASSIVE_DAMPING
        h.loss = FRICTION_LOSS
        h.kp = float(kp[j])
        h.kv = float(kv[j])
        h.cap = float(s.joints[j].effort)
        h.motor_enabled = 1
        h.d0, h.dw, h.tc = FRICTION_D0, FRICTION_DWIDTH, FRICTION_TIMECONST
    # root source->principal: base_link COM offset, no rotation
    f.model.root_inertia[:] = [*s.base.com, 0.0, 0.0, 0.0, 1.0]
    f.reference[:] = reset_qpos(s)
    f.mapping = {"joints": [{"lower": jt.lower, "upper": jt.upper}
                            for jt in s.joints]}
    return f


def limits(s: ArmSpec, f=None):
    import api as av  # noqa: PLC0415
    return av.limits(f if f is not None else fixture(s))


def contact_tables():
    """(shapes[B], pairs (empty), mu (empty)): floor plane only, NO contact
    pairs -- the arm never touches the floor in this lowering (targets and
    the judge's floor proxy keep it clear)."""
    import model_translation as contact  # noqa: PLC0415
    shapes = (contact.Shape * B)()
    for b, shape in enumerate(shapes):
        shape.caller_id = b
        shape.fixed = int(b == FLOOR_BODY)
    shapes[FLOOR_BODY].kind = 2
    shapes[FLOOR_BODY].plane_normal[:] = (0.0, 0.0, 1.0)
    shapes[FLOOR_BODY].plane_offset = 0.0
    return shapes, (contact.Pair * 0)(), []


def scene(lib, s: ArmSpec, environments: int = 1, joint_offsets=None):
    """Registered idv1 Scene at the arm's home pose (welded root)."""
    import native  # noqa: PLC0415
    f = fixture(s)
    q = np.tile(reset_qpos(s), (environments, 1))
    if joint_offsets is not None:
        off = np.asarray(joint_offsets, dtype="d")
        if off.shape != (environments, J):
            raise ValueError("joint_offsets requires shape [E, J]")
        lim = joint_limits(s)
        q[:, 7:] = np.clip(q[:, 7:] + off, lim[:, 0], lim[:, 1])
    v = np.tile(reset_vel(), (environments, 1))
    shapes, pairs, mu = contact_tables()
    sc = native.Scene(lib, f, q, v, shapes, pairs,
                      np.zeros((environments, 0), "f"),
                      gravity=[list(GRAVITY_VEC)] * environments,
                      limits=limits(s, f))
    return sc, f


def weld_force(s: ArmSpec, q: np.ndarray, v: np.ndarray,
               link_pos: np.ndarray) -> np.ndarray:
    """[E,N] applied generalized force for the oracle's virtual weld.

    Exact cancellation of the STATIC gravity generalized force on the 6 root
    dofs -- av1's root angular dofs act about the root SOURCE origin (the
    base_link frame), so the weight of every body b produces the moment
    (r_b - o_root) x (0, 0, m_b g) there; the welded root's own 6.8e9 N at
    its 0.14 m COM lever is the dominant term and MUST be cancelled, not
    PD-resisted (measured: 1.4e-4 rad of root tilt when it was not) -- plus
    a stiff explicit PD on the root pose (small-angle quaternion error,
    world rates) that absorbs the DYNAMIC reactions of the moving arm.
    link_pos: [E,J,3] world link COMs (from the lane read; f32 precision is
    ample for the ~1e4 N*m link moments). Joint dofs get 0."""
    q = np.asarray(q, float)
    v = np.asarray(v, float)
    link_pos = np.asarray(link_pos, float)
    E = q.shape[0]
    w = weld(s)
    f = np.zeros((E, N))
    x0 = reset_qpos(s)
    f[:, 0:3] = -w["k_lin"] * (q[:, 0:3] - x0[0:3]) - w["d_lin"] * v[:, 0:3]
    f[:, 2] += (w["mass"] + moving_mass(s)) * GRAVITY
    # root COM lever in f64 (the 6.8e9 N term): R(q) @ com_base
    rot = np.stack([quat_to_rot(qq) for qq in q[:, 3:7]])
    lever_root = rot @ np.asarray(s.base.com)
    moment = np.cross(lever_root, np.array([0.0, 0.0, w["mass"] * GRAVITY]))
    for b, l in enumerate(s.links):
        r = link_pos[:, b, :] - q[:, 0:3]
        moment += np.cross(r, np.array([0.0, 0.0, l.mass * GRAVITY]))
    sign = np.where(q[:, 6] < 0, -1.0, 1.0)[:, None]
    theta = 2.0 * sign * q[:, 3:6]
    f[:, 3:6] = moment - w["k_ang"] * theta - w["d_ang"] * v[:, 3:6]
    return f
