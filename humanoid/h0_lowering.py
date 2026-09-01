"""H0 humanoid (B14/J12/P2) lowering into the duck-grid-walk native stack.

Single source of truth for every humanoid constant used by the CPU oracle
lane (walk/env/humanoid_native_lane.py) and the fp32 kernel header generator
(experimental/duck_cuda/tools/generate_model_humanoid.py). No simulation.

SOURCES (every number below is cited; nothing is invented):
  [HR]  /Users/john/Code/world/crates/sim/src/humanoid.rs
        (profile "world.humanoid.planar_13_link-v1", lines 28-31; the ONLY
        authoring site of the H0 morphology)
  [WS]  /Users/john/Code/world/crates/sim/src/world_slice.rs:2496-2521
        (inertia is DERIVED there: solid box, I = m*(s_j^2+s_k^2)/12 on the
        full extents == m*(h_j^2+h_k^2)/3 on half-extents; never authored)
  [LIB] /Users/john/Code/world/crates/sim/src/lib.rs:48 (GRAVITY = 20.0,
        applied as [0,-20,0]; confirmed by [BND] global_gravity_xyz and by
        /Users/john/Code/box3d-cuda-voxel-gate-c1/docs/
        native-flatfloor-readiness.md:17-19 "authored -20 m/s^2, not -9.81")
  [BND] /Users/john/Code/world/evidence/humanoid-balance-r4-preflight-20260830/
        humanoid-cuda-training-bundle-v2.json (materialized registration;
        cross-checked by humanoid/tests/test_h0_lowering.py when present.
        NOTE: its `initial` block carries per-env mass domain randomization
        on pelvis+torso ([HR]:397-408), so authored masses come from [HR],
        not from the bundle's env slices)
  [H0]  /Users/john/Code/box3d-arm-lab/factory_os/independent_validation/
        humanoid_h0.py (frozen naming/ordering/frame contract; line 94 pins
        B14/J12/P2, lines 24-53 pin body/joint order, line 113 pins +Y up /
        +X forward, lines 284-291 pin the box-inertia formula)
  [RDY] /Users/john/Code/box3d-cuda-voxel-gate-c1/docs/
        native-flatfloor-readiness.md:26-33 (drive law: torque =
        stiffness*(target-q) - damping*qdot, clamped to +-effort; position
        mode has NO target-velocity feed-forward -> lower it as kp=stiffness,
        kv=damping, cap=effort in this stack's identical PD law, see
        experimental/articulated_v2/src/articulated_v2.cpp:107-109)

FRAME CONVENTION. [H0] authors the world +Y-up, +X-forward. This stack
(gravity on z in walk/env, floor plane z=0 in the duck_cuda kernel's
dw_plane_manifold) is +Z-up. The lowering rotates the WORLD by
Rx(+90 deg): (x, y, z) -> (x, -z, y), so up +Y -> +Z and forward +X -> +X.
Body-LOCAL quantities (joint anchors, parent-local axes, reference
quaternions, principal inertia triples, half-extents, foot vertices) are
frame-local and therefore UNCHANGED; the rotation only touches the reset
root pose (every body's reset world orientation becomes QX90) and the
gravity vector. Hinge reference quaternions stay identity because parent
and child reset orientations are both QX90 (relative rotation identity),
exactly as authored ([HR]:765).

STACK-SIDE SOLVER CONSTANTS (not part of the H0 morphology; H0's engine
uses substeps=2 / solver_iterations=12 / warm_start 0.8 / angular_damping
0.02 per world/crates/box3d-cuda-client/src/lib.rs:117-147, which this
stack's to-tolerance impulse solver replaces -- an explicitly non-equivalent
solver per [H0] lines 184-188): joint friction-cone impedance d0/dw/tc and
limit solimp keep the stack defaults used by every fixture here
(experimental/articulated_v2/tests/api.py:24, articulated_v1.py:39).
"""
from __future__ import annotations

import ctypes as C
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "experimental" / "articulated_v2" / "tests"),
           str(ROOT / "experimental" / "contact_v1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import api as av                      # noqa: E402  (articulated_v2 fixture ABI)
import model_translation as contact   # noqa: E402  (contact_v1 ctypes structs)

# ---------------------------------------------------------------- identity
PROFILE = "world.humanoid.planar_13_link-v1"        # [HR]:28
# [H0]:24-53 -- frozen dense ordering; child of joint j is body j+2, the
# exact topology rule this stack's articulation requires
# (articulated_v1.cpp:31  bodies == joints+2, kernel `int b = j + 2`).
BODY_NAMES = (
    "floor", "pelvis", "torso", "head",
    "left_upper_leg", "left_lower_leg", "left_foot",
    "right_upper_leg", "right_lower_leg", "right_foot",
    "left_upper_arm", "left_forearm", "right_upper_arm", "right_forearm",
)
JOINT_NAMES = (
    "waist", "neck",
    "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle",
    "left_shoulder", "left_elbow", "right_shoulder", "right_elbow",
)
J = 12                                # [H0]:94, [HR]:30
B = 14                                # [H0]:94, [HR]:29 (13 links + floor)
N = 6 + J
Q = 7 + J
FOOT_BODIES = (6, 9)                  # [H0]:164-165 (floor-left/right_foot)
FLOOR_BODY = 0

# ------------------------------------------------------------- integration
AUTHORED_DT = 1.0 / 120.0             # [HR]:181 step_seconds (World engine)
AUTHORED_SUBSTEPS = 2                 # world/crates/box3d-cuda-client/src/
#                                       lib.rs:117-147 RegistrationConfig
# This stack applies the PD drive once per tick (no substeps), so its tick
# must be at most the authored engine's effective PER-SUBSTEP drive cadence
# (native-flatfloor-readiness.md:23-33: drive applied per substep with
# h = dt/substeps = 1/240). At the raw 1/120 the explicit per-tick PD is
# numerically UNSTABLE here: kv*dt / I_effective = 2.33 (elbow) / 2.07
# (ankle) > 2, linearized spectral radius 1.79 (measured; blows up ~0.6 s
# in). Any tick <= 1/240 is stable (spectral radius 1.0 = free-root modes;
# measured at 1/240, 1/480 and 0.002). See humanoid/FEASIBILITY.md sec. 5.
#
# Phase 2 pins SIM_DT to the duck stack's proven 0.002 s so ONE policy step
# is exactly the duck env contract's 0.02 s (10 ticks) -- every 0.02 s-based
# reward/env constant and the kernel policy layer's hardcoded control dt
# stay valid, and stability margin strictly improves (kv*dt/I_eff max 0.56).
# The authored control cadence (1/60, action_repeat 2 [H0]:124) belongs to
# World's engine; control cadence here follows the duck env contract.
SIM_DT = 0.002                        # duck stack tick (walk/env/flat.py)
CONTROL_DT = 0.02                     # duck env contract policy step
TICKS_PER_CONTROL = 10                # CONTROL_DT / SIM_DT
MAX_EPISODE_STEPS = 1200              # [HR]:182
GRAVITY = 20.0                        # [LIB]:48; z-up lowering: (0, 0, -20)
FRICTION = 0.8                        # [HR]:734 (humanoid-rubberized)
RESTITUTION = 0.0                     # [HR]:735 (this stack has none: match)

# ---------------------------------------------------------------- actuators
KP = 90.0                             # [HR]:787 stiffness (uniform)
KV = 8.0                              # [HR]:788 damping; [RDY] drive law:
#                                       clamp(kp*(t-q) - kv*qdot, +-effort),
#                                       identical to av2's motor law with
#                                       target_velocity = 0
SPEED_LIMIT = 8.0                     # [HR]:785 (host-side action shaping,
ACCELERATION_LIMIT = 40.0             # [HR]:786  NOT enforced by the solver
#                                       per rapid-walking-benchmark.md:17-24;
#                                       Phase 2 policy-layer concern)
ARMATURE = 0.0                        # NOT AUTHORED anywhere for H0
PASSIVE_DAMPING = 0.0                 # (SliceJoint/SliceActuator have no such
FRICTION_LOSS = 0.0                   #  fields); zero == feature off in this
#                                        stack, NOT an invented value.

# --------------------------------------------------- bodies [HR]:50-162
# (name, authored_center_y_up[m], half_extents[m] (= scale/2), mass[kg])
# Floor: authored as a finite box, center [0,-0.1,0], half [20,0.1,4]
# ([HR]:51-58) whose TOP FACE is the plane y=0; lowered to this stack's
# infinite floor plane z=0 like the duck's (identical contact behavior for
# |x|<20, |lateral|<4; H0 terminates episodes at x>20 anyway, [HR]:446-460).
BODIES = (
    ("floor",           (0.0, -0.1, 0.0),   (20.0, 0.1, 4.0),    0.0),
    ("pelvis",          (0.0, 1.15, 0.0),   (0.25, 0.15, 0.18),  10.0),
    ("torso",           (0.0, 1.70, 0.0),   (0.33, 0.40, 0.17),  20.0),
    ("head",            (0.0, 2.27, 0.0),   (0.17, 0.17, 0.17),  5.0),
    ("left_upper_leg",  (0.0, 0.73, 0.15),  (0.11, 0.27, 0.11),  7.0),
    ("left_lower_leg",  (0.0, 0.30, 0.15),  (0.09, 0.16, 0.09),  4.0),
    ("left_foot",       (0.12, 0.07, 0.15), (0.23, 0.07, 0.14),  1.5),
    ("right_upper_leg", (0.0, 0.73, -0.15), (0.11, 0.27, 0.11),  7.0),
    ("right_lower_leg", (0.0, 0.30, -0.15), (0.09, 0.16, 0.09),  4.0),
    ("right_foot",      (0.12, 0.07, -0.15), (0.23, 0.07, 0.14), 1.5),
    ("left_upper_arm",  (0.0, 1.62, 0.46),  (0.09, 0.27, 0.09),  2.5),
    ("left_forearm",    (0.0, 1.14, 0.46),  (0.08, 0.21, 0.08),  1.5),
    ("right_upper_arm", (0.0, 1.62, -0.46), (0.09, 0.27, 0.09),  2.5),
    ("right_forearm",   (0.0, 1.14, -0.46), (0.08, 0.21, 0.08),  1.5),
)
TOTAL_DYNAMIC_MASS = 68.0             # [HR]:934-955 invariant test

# ------------------------------------------------- joints [HR]:185-318
# (name, parent_body, parent_anchor[m], child_anchor[m], lower[rad],
#  upper[rad], effort_limit[N*m]); child body is j+2 by the frozen ordering
# ([BND] joint_body_indices confirms). Anchors are body-local (COM == box
# center, [HR]:721 zero COM offsets), so they survive the frame rotation
# unchanged. axis_parent = [0,0,1] and reference = identity for ALL joints
# ([HR]:764-765): the model is strictly planar/sagittal.
# Effort tiers [HR]:778-784: 180 hip/waist, 140 knee/ankle, 70 otherwise.
JOINTS = (
    ("waist",          1, (0.0, 0.15, 0.0),   (0.0, -0.40, 0.0), -0.35, 0.35, 180.0),
    ("neck",           2, (0.0, 0.40, 0.0),   (0.0, -0.17, 0.0), -0.30, 0.30, 70.0),
    ("left_hip",       1, (0.0, -0.15, 0.15), (0.0, 0.27, 0.0),  -1.10, 0.85, 180.0),
    ("left_knee",      4, (0.0, -0.27, 0.0),  (0.0, 0.16, 0.0),  -0.10, 1.55, 140.0),
    ("left_ankle",     5, (0.0, -0.16, 0.0),  (-0.12, 0.07, 0.0), -0.65, 0.65, 140.0),
    ("right_hip",      1, (0.0, -0.15, -0.15), (0.0, 0.27, 0.0), -1.10, 0.85, 180.0),
    ("right_knee",     7, (0.0, -0.27, 0.0),  (0.0, 0.16, 0.0),  -0.10, 1.55, 140.0),
    ("right_ankle",    8, (0.0, -0.16, 0.0),  (-0.12, 0.07, 0.0), -0.65, 0.65, 140.0),
    ("left_shoulder",  2, (0.0, 0.19, 0.46),  (0.0, 0.27, 0.0),  -1.40, 1.40, 70.0),
    ("left_elbow",     10, (0.0, -0.27, 0.0), (0.0, 0.21, 0.0),  -0.10, 1.50, 70.0),
    ("right_shoulder", 2, (0.0, 0.19, -0.46), (0.0, 0.27, 0.0),  -1.40, 1.40, 70.0),
    ("right_elbow",    12, (0.0, -0.27, 0.0), (0.0, 0.21, 0.0),  -0.10, 1.50, 70.0),
)
AXIS = (0.0, 0.0, 1.0)                # [HR]:764 (all joints)
REFERENCE_XYZW = (0.0, 0.0, 0.0, 1.0)  # [HR]:765 (all joints)
EFFORT = tuple(j[6] for j in JOINTS)  # == [BND] control.maximum_effort

# ------------------------------------------------------- home / reset pose
# The authored reset IS the home pose: every joint q = 0, every controller
# target 0 (verified against [BND] controller_initial, all zeros), every
# body at its authored center with identity orientation, zero velocity
# ([HR]:50-162 positions + rot [0,0,0,1]; [H0] contract golden
# box3d-arm-lab/factory_os/artifacts/contracts/humanoid-independent-h0-v1.json
# reset_observation_env0). No crouch pose exists anywhere -- see
# humanoid/FEASIBILITY.md "constants not found".
HOME_TARGETS = tuple(0.0 for _ in range(J))

# Rx(+90 deg) world rotation, y-up -> z-up (see module docstring).
_S = math.sqrt(0.5)
QX90 = (_S, 0.0, 0.0, _S)             # xyzw


def y_up_to_z_up(p) -> tuple[float, float, float]:
    """Rotate an authored +Y-up world point into this stack's +Z-up world."""
    return (p[0], -p[2], p[1])


def box_inertia(mass: float, half) -> tuple[float, float, float]:
    """Solid-box principal inertia, [WS]:2496-2521 (== [H0]:284-291)."""
    hx, hy, hz = half
    return (mass * (hy * hy + hz * hz) / 3.0,
            mass * (hx * hx + hz * hz) / 3.0,
            mass * (hx * hx + hy * hy) / 3.0)


def foot_vertices(half=BODIES[6][2]) -> list[tuple[float, float, float]]:
    """The 8 box corners of a foot collider, body/COM frame.

    H0 feet are single OBBs ([HR]:709-713 + scale) -- no baked multi-point
    sole like the duck's 18-vertex convention; the corner set is exact, not
    an approximation. Vertex order follows native.box()
    (experimental/integrated_duck_v1/native.py:121-123) for determinism.
    """
    return [tuple(half[k] * (1.0 if i & (1 << k) else -1.0) for k in range(3))
            for i in range(8)]


def reset_qpos() -> np.ndarray:
    """[Q] root xyz (z-up), root quat xyzw (= QX90), 12 joint q (= 0)."""
    q = np.zeros(Q)
    q[:3] = y_up_to_z_up(BODIES[1][1])           # pelvis center [HR]:62
    q[3:7] = QX90
    return q


def reset_vel() -> np.ndarray:
    """[N] zero ([HR] reset states carry zero velocity; [BND] initial.state)."""
    return np.zeros(N)


def fixture() -> av.Fixture:
    """The H0 humanoid as an articulated_v2 fixture (this stack's model ABI).

    Mirrors av.duck() (articulated_v1.py:53-64) with H0 constants. The root
    source->principal transform is IDENTITY: H0 bodies are principal-axis
    boxes with zero COM offset ([HR]:710-721), unlike the duck.
    """
    f = av.Fixture(J)
    from articulated_v1 import Body as _Body  # noqa: PLC0415 (av re-exports)
    for b, (_, _, half, mass) in enumerate(BODIES):
        if b == 0:
            f.body[b] = _Body(0.0, (C.c_double * 3)(0.0, 0.0, 0.0))
        else:
            f.body[b] = _Body(mass, (C.c_double * 3)(*box_inertia(mass, half)))
    for j, (h, (_, parent, ap, ac, lower, upper, effort)) in enumerate(
            zip(f.hinge, JOINTS)):
        h.parent = parent
        h.ap[:] = ap
        h.ac[:] = ac
        h.axis[:] = AXIS
        h.reference[:] = REFERENCE_XYZW
        h.armature = ARMATURE
        h.damping = PASSIVE_DAMPING
        h.loss = FRICTION_LOSS
        h.kp = KP
        h.kv = KV
        h.cap = effort                      # per-joint tier [HR]:778-784
        h.motor_enabled = 1
        # d0/dw/tc keep the Fixture ctor's stack defaults (.9/.95/.02);
        # unused while loss == 0 (av2 emits no friction rows then,
        # articulated_v2.cpp:114).
        assert (h.d0, h.dw, h.tc) == (np.float32(0.9), np.float32(0.95),
                                      np.float32(0.02)), (j, "stack defaults")
    # identity source->principal for the root (pelvis): COM offset 0,
    # principal frame == body frame ([HR]:721, [WS] diagonal box inertia).
    f.model.root_inertia[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    f.reference[:] = reset_qpos()          # refweight pose == reset pose
    # av.limits() consumes this mapping (api.py:25-26).
    f.mapping = {"joints": [{"lower": jt[4], "upper": jt[5]} for jt in JOINTS]}
    return f


def contact_tables():
    """(shapes[B], pairs[P], mu[P]) for the contact/idv1 registration.

    Floor: infinite plane z=0 (top face of the authored floor box, see
    BODIES note). Feet: exact 8-corner OBB hulls. Pair order (foot, floor)
    matches the duck's (native_lane.py:22-24) and the kernel's
    DW_PAIR_BODY_A/B convention; [H0]:163-166 lists the same two pairs as
    (floor, foot) -- order within a pair is a backend convention, the
    manifold normal handling is contact_v1's plane_is_a=False path either
    way (contact_v1.cpp:172-173).
    """
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
    """Per-joint limit rows (av2 Limit ABI) with H0 bounds [HR]:194-316."""
    return av.limits(f if f is not None else fixture())


def scene(lib, environments: int = 1, joint_offsets=None, root_lift: float = 0.0):
    """Registered idv1 Scene at the floor-clear H0 reset (mirrors
    native.duck_scene, experimental/integrated_duck_v1/native.py:126-134).

    The reset is exactly floor-touching: foot sole plane at z == 0
    (foot center z 0.07 - half height 0.07, [HR]:102/126), zero penetration,
    zero clearance. Returns (Scene, fixture).
    """
    lane = ROOT / "experimental" / "integrated_duck_v1"
    if str(lane) not in sys.path:
        sys.path.insert(0, str(lane))
    import native  # noqa: PLC0415
    f = fixture()
    q = np.tile(reset_qpos(), (environments, 1))
    q[:, 2] += float(root_lift)      # test hook: in-air (contact-free) starts
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
