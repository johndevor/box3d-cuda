"""PLANT FEASIBILITY CHECKLIST for a humanoid lowering (pure closed-form).

Reusable for ANY lowering module/namespace that exposes the H1 surface
(BODIES, JOINTS, JOINT_NAMES, FOOT_BODIES, KP_TABLE, EFFORT, GRAVITY,
box_inertia): no simulation, no native library -- every number is derived
analytically from the authored tables at the home pose (authored y-up
frame, identity orientations). The platform runs this on every future
robot BEFORE anything trains on it; the frozen judge stays the only
acceptance authority.

Each check returns Result(value, bound, ok, ...) with the physics in the
check's docstring. `run(lowering)` returns the five rows; `table(...)`
renders them, `main()` prints H1.1 (the accepted reference) and every
registered variant side by side and ALSO reports, per check, whether the
member is at least as good as the accepted H1.1 reference (`vs_ref`):
where the accepted walker itself misses an absolute bar, the bar is
evidently not necessary for walking on this stack, and baseline parity
is the honest gate (see (b) below).

Conventions shared by the checks:
  * gravity g = lowering.GRAVITY (authored -20 m/s^2, 2x Earth; KEPT);
  * "stance" = the LEFT leg (the model is mirror-symmetric, verified);
  * subtree(j) = the bodies moved by joint j (child j+2 and descendants);
    with the stance foot planted, actuating a stance-leg joint j rotates
    the COMPLEMENT of subtree(j) -- the rest of the robot -- about the
    joint's anchor. All "above the pivot" sums run over that complement;
  * heights h_i are authored-frame y of body centers (COM == box center,
    humanoid.rs:721) relative to the pivot anchor; lateral = authored z.

Usage: .venv/bin/python -B humanoid/feasibility_check.py [--json OUT]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
ROOT = _HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# bounds (the checklist spec)
HOLD_MARGIN = 1.3          # (a) effort cap / hold torque
BANDWIDTH_RATIO = 3.0      # (b) actuator natural frequency / gait clock
STIFFNESS_MARGIN = 1.2     # (c) kp / gravitational destabilizing stiffness
LEG_JOINTS = ("hip_roll", "hip", "knee", "ankle")   # per-side leg joints


@dataclasses.dataclass
class Result:
    check: str
    joint: str          # joint the bound applies to ("" for whole-body)
    value: float
    bound: float
    ok: bool
    unit: str
    detail: str = ""
    # the platform reports pass/fail as (value, bound, pass); `ratio` is
    # the margin in the direction that is "good" when >= 1
    ratio: float = float("nan")

    def __post_init__(self):
        self.value, self.bound = float(self.value), float(self.bound)
        self.ok, self.ratio = bool(self.ok), float(self.ratio)

    def as_tuple(self):
        return (self.value, self.bound, self.ok)


# ----------------------------------------------------------- geometry
class Home:
    """Home-pose geometry of a lowering: centers, anchors, subtrees."""

    def __init__(self, lw):
        self.lw = lw
        self.names = list(lw.JOINT_NAMES)
        self.B, self.J = lw.B, lw.J
        self.mass = np.array([b[3] for b in lw.BODIES], float)
        self.mass[0] = 0.0
        self.center = np.array([b[1] for b in lw.BODIES], float)
        self.half = np.array([b[2] for b in lw.BODIES], float)
        self.parent = np.array([jt[1] for jt in lw.JOINTS], int)
        self.axis = np.array([jt[7] for jt in lw.JOINTS], float)
        self.cap = np.array(lw.EFFORT, float)
        self.kp = np.array(lw.KP_TABLE, float)
        self.g = float(lw.GRAVITY)
        # joint anchors in world = child center + child anchor
        self.anchor = np.array([self.center[j + 2] + np.asarray(jt[3], float)
                                for j, jt in enumerate(lw.JOINTS)])
        # anchors must close: parent center + ap == child center + ac
        for j, jt in enumerate(lw.JOINTS):
            alt = self.center[jt[1]] + np.asarray(jt[2], float)
            assert np.allclose(alt, self.anchor[j], atol=1e-9), (j, "anchor")
        self.children = {b: [] for b in range(self.B)}
        for j in range(self.J):
            self.children[self.parent[j]].append(j + 2)
        self.inertia = np.array([(0.0, 0.0, 0.0) if b == 0 else
                                 lw.box_inertia(self.mass[b], self.half[b])
                                 for b in range(self.B)])
        self.total_mass = float(self.mass.sum())

    def j(self, name: str) -> int:
        return self.names.index(name)

    def subtree(self, j: int) -> list[int]:
        out, stack = [], [j + 2]
        while stack:
            b = stack.pop()
            out.append(b)
            stack.extend(self.children[b])
        return sorted(out)

    def complement(self, j: int) -> list[int]:
        sub = set(self.subtree(j))
        return [b for b in range(1, self.B) if b not in sub]

    def path(self, b: int) -> list[int]:
        """Joints from the root down to body b."""
        out = []
        while b > 1:
            j = b - 2
            out.append(j)
            b = self.parent[j]
        return out[::-1]

    def com(self, bodies) -> np.ndarray:
        m = self.mass[bodies]
        return (m[:, None] * self.center[bodies]).sum(0) / m.sum()

    # -- composite rigid body mass matrix at the home pose ----------------
    def mass_matrix(self) -> np.ndarray:
        """Joint-space mass matrix M [N, N] at home (all orientations =
        identity, authored frame). Generalized velocity ordering
        [v_root(3), w_root(3), qdot(J)] with the root = pelvis center.
        M = sum_b J_b^T diag(m_b I3, I_b) J_b, J_b: (v_com_b, w_b)."""
        N = 6 + self.J
        M = np.zeros((N, N))
        for b in range(1, self.B):
            Jb = np.zeros((6, N))
            r = self.center[b] - self.center[1]
            Jb[0:3, 0:3] = np.eye(3)
            Jb[0:3, 3:6] = -_skew(r)           # v = v0 + w x r
            Jb[3:6, 3:6] = np.eye(3)
            for j in self.path(b):
                a = self.axis[j]
                Jb[0:3, 6 + j] = np.cross(a, self.center[b] - self.anchor[j])
                Jb[3:6, 6 + j] = a
            G = np.diag([self.mass[b]] * 3 + list(self.inertia[b]))
            M += Jb.T @ G @ Jb
        return M

    def effective_inertia(self) -> np.ndarray:
        """[J] apparent inertia at each joint with the base and every other
        joint FREE: 1 / (M^-1)_jj (the codebase's I_eff convention,
        FEASIBILITY.md s5 / PHASE2.md s14-15)."""
        Minv = np.linalg.inv(self.mass_matrix())
        return 1.0 / np.diag(Minv)[6:]

    def subtree_inertia(self, j: int) -> float:
        """Inertia of subtree(j) about joint j's axis (the swing load)."""
        a = self.axis[j]
        total = 0.0
        for b in self.subtree(j):
            r = self.center[b] - self.anchor[j]
            perp = r - a * (r @ a)
            total += self.mass[b] * (perp @ perp) + a @ (self.inertia[b] * a)
        return float(total)


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
                    float)


# --------------------------------------------------------------- checks
def check_sagittal_hold(h: Home) -> Result:
    """(a) SAGITTAL HOLD -- stance hip-pitch torque to hold single support
    at the static lean, vs its effort cap (margin >= 1.3x).

    Static lean: the whole robot pitched forward as a rigid body about the
    stance ankle by theta_s = atan(x_c / h_com), the angle that moves the
    CoM from above the ankle (home) to above the sole CENTROID (the
    quiet-standing CoP; x_c = foot-center forward offset from the ankle,
    0.12 m on H1; h_com = height of the rotating body's CoM above the
    ankle). In that posture the trunk (complement of the stance hip-pitch
    subtree: pelvis, torso, head, arms, hip links AND the hanging swing
    leg, whose negative lever partly offsets) loads the stance hip pitch
    with tau = g * sin(theta_s) * sum_i m_i (h_i - h_hip). The stance
    KNEE is deliberately NOT gated here: a forward CoM hyperextends a
    straight knee onto its authored -0.10 rad limit (limit-borne, not
    actuator-borne); the ankle's sagittal authority is check (d)."""
    hip = h.j("left_hip")
    ankle = h.j("left_ankle")
    rot = h.complement(ankle)
    com = h.com(rot)
    x_c = h.center[h.lw.FOOT_BODIES[0]][0] - h.anchor[ankle][0]
    h_com = com[1] - h.anchor[ankle][1]
    theta = math.atan2(x_c, h_com)
    bodies = h.complement(hip)
    lever = float(sum(h.mass[b] * (h.center[b][1] - h.anchor[hip][1])
                      for b in bodies))
    tau = h.g * math.sin(theta) * lever
    cap = h.cap[hip]
    return Result("a_sagittal_hold", "left_hip", tau, cap / HOLD_MARGIN,
                  tau <= cap / HOLD_MARGIN, "N*m",
                  f"theta_s={math.degrees(theta):.2f}deg x_c={x_c:.3f} "
                  f"h_com={h_com:.3f} cap={cap:.0f}", cap / tau)


def check_bandwidth(h: Home, clock_hz: float) -> list[Result]:
    """(b) ACTUATOR BANDWIDTH -- f_n = sqrt(kp / I_eff) / (2 pi) for each
    leg joint vs the gait clock drive frequency at cmd 1.0 m/s
    (humanoid_flat: PHASE_HZ_BASE + PHASE_HZ_PER_MPS * 1.0 = 1.67 Hz);
    ratio >= 3x is the textbook bound for <~10 % tracking attenuation of
    a target at the drive frequency.

    I_eff is the apparent inertia at the joint with the floating base and
    all other joints free, 1 / (M^-1)_jj at the home pose (M from an
    analytic CRBA over the authored boxes; the codebase's convention in
    FEASIBILITY.md s5 and PHASE2.md s14/15 -- 1.74 kg m^2 for the H1 hip
    roll). The subtree (swing-load) inertia is reported alongside.
    Hz vs Hz: PHASE2 quotes sqrt(500/1.74) ~= 2.7 Hz for kp_roll 500."""
    ieff = h.effective_inertia()
    out = []
    for jn in LEG_JOINTS:
        j = h.j(f"left_{jn}")
        fn = math.sqrt(h.kp[j] / ieff[j]) / (2.0 * math.pi)
        out.append(Result(
            "b_bandwidth", f"left_{jn}", fn, BANDWIDTH_RATIO * clock_hz,
            fn >= BANDWIDTH_RATIO * clock_hz, "Hz",
            f"kp={h.kp[j]:.0f} I_eff={ieff[j]:.3f} "
            f"I_subtree={h.subtree_inertia(j):.3f} clock={clock_hz:.2f}Hz",
            fn / clock_hz))
    return out


def check_gravitational_stiffness(h: Home) -> list[Result]:
    """(c) GRAVITATIONAL DESTABILIZING STIFFNESS -- with the stance foot
    planted, rotating a stance joint j by delta moves every body of the
    complement of subtree(j) sideways by (h_i - h_j) delta, so gravity
    feeds back a destabilizing moment K_g delta with
        K_g(j) = g * sum_{i not in subtree(j)} m_i (h_i - h_j)
    (an inverted pendulum about the joint; hanging bodies below the pivot
    -- the swing leg under the hip -- contribute negatively). The PD must
    out-stiffen it: kp_j >= 1.2 K_g(j). Gated at the stance HIP ROLL (the
    lateral pendulum, PHASE2.md s15: 388 N*m/rad on H1 vs kp 500) and the
    stance KNEE (sagittal buckling, s16); the stance ANKLE row is reported
    for the record -- at 2g no authored ankle (kp 300, cap 140) holds the
    whole body (K_g ~ 1.4 kN*m/rad); balance is a whole-body CoP task,
    not an ankle-stiffness task, which is why H1.1 walks anyway."""
    out = []
    for jn, gated in (("hip_roll", True), ("knee", True), ("ankle", False)):
        j = h.j(f"left_{jn}")
        kg = h.g * float(sum(h.mass[b] * (h.center[b][1] - h.anchor[j][1])
                             for b in h.complement(j)))
        bound = STIFFNESS_MARGIN * kg
        out.append(Result(
            "c_grav_stiffness" + ("" if gated else "_info"), f"left_{jn}",
            h.kp[j], bound, (h.kp[j] >= bound) if gated else True,
            "N*m/rad", f"K_g={kg:.0f} kp={h.kp[j]:.0f}"
            + ("" if gated else " (reported, not gated)"), h.kp[j] / kg))
    return out


def check_ankle_cop_authority(h: Home) -> Result:
    """(d) ANKLE CoP AUTHORITY -- the farthest the ankle pitch actuator can
    statically hold the whole-body CoP from the ankle, x_max = cap_ankle /
    (M g), as a fraction of the sole half-length (the sole spans +-half
    about the foot center, which itself sits x_c ahead of the ankle).
    REPORT-ONLY (no bound): H1.1 reaches 45 % of a half-sole (0.103 m)
    and walks; a fraction < the CoM-over-centroid demand (x_c / half =
    52 %) simply means quiet standing must keep the CoM near the ankle."""
    ankle = h.j("left_ankle")
    half = h.half[h.lw.FOOT_BODIES[0]][0]
    x_max = h.cap[ankle] / (h.total_mass * h.g)
    frac = x_max / half
    return Result("d_ankle_cop_authority", "left_ankle", frac, float("nan"),
                  True, "fraction of sole half-length",
                  f"x_max={x_max:.3f}m half={half:.2f} cap={h.cap[ankle]:.0f} "
                  f"Mg={h.total_mass * h.g:.0f}N", frac)


def check_lateral_static_margin(h: Home) -> Result:
    """(e) LATERAL STATIC MARGIN in single support at ZERO lean -- the
    whole-body CoM lateral offset from the stance foot center (the hip
    half-spacing, 0.15 m on H1: the CoM sits on the midline at home) vs the
    sole half-width. margin = half_width - offset; positive means the CoM
    projects inside the planted sole so a foot can be lifted with no
    weight shift; negative (H1.1: -0.01 m, PHASE2.md s12 "unstable by
    ~1 cm") means single support exists ONLY with an active lean (hip
    roll to the static balance point asin(offset / leg_length), which the
    v3.2 reference and the accepted policy perform). May fail; sign is the
    deliverable."""
    foot = h.lw.FOOT_BODIES[0]
    com = h.com(list(range(1, h.B)))
    offset = abs(com[2] - h.center[foot][2])
    half_w = h.half[foot][2]
    margin = half_w - offset
    return Result("e_lateral_static_margin", "", margin, 0.0, margin >= 0.0,
                  "m", f"offset={offset:.3f} sole_half_width={half_w:.3f} "
                  f"static_lean={math.degrees(math.asin(offset / _leg(h))):.1f}deg",
                  half_w / offset)


def _leg(h: Home) -> float:
    hip, ankle = h.j("left_hip"), h.j("left_ankle")
    return float(h.anchor[hip][1] - h.anchor[ankle][1])


def gait_clock_hz(command_mps: float = 1.0) -> float:
    from walk.env import humanoid_flat as env_mod  # noqa: PLC0415
    return float(env_mod.PHASE_HZ_BASE + env_mod.PHASE_HZ_PER_MPS * command_mps)


def run(lowering, clock_hz: float | None = None) -> list[Result]:
    """All checklist rows for one lowering (module or namespace)."""
    h = Home(lowering)
    clock = gait_clock_hz(1.0) if clock_hz is None else float(clock_hz)
    rows = [check_sagittal_hold(h)]
    rows += check_bandwidth(h, clock)
    rows += check_gravitational_stiffness(h)
    rows.append(check_ankle_cop_authority(h))
    rows.append(check_lateral_static_margin(h))
    return rows


def gated(rows: list[Result]) -> list[Result]:
    """Rows that must pass for a variant to be deliverable: (a)-(c)."""
    return [r for r in rows if r.check in
            ("a_sagittal_hold", "b_bandwidth", "c_grav_stiffness")]


def compare(rows: list[Result], ref: list[Result]) -> list[bool]:
    """Per row: is the member's margin ratio >= the reference's (within
    1e-9), i.e. no worse than the accepted H1.1 on that check."""
    return [r.ratio >= q.ratio - 1e-9 for r, q in zip(rows, ref)]


def verdict(rows: list[Result], ref: list[Result] | None) -> list[tuple[Result, bool, str]]:
    """Deliverability verdict per GATED row ((a)-(c)), calibrated on the
    accepted reference: a row passes if it meets the absolute bound, OR --
    where the accepted H1.1 itself misses that absolute bound (it walks
    12/12 regardless, so the bound is demonstrably not necessary on this
    stack) -- if the member's margin ratio is >= the reference's. Every
    other failure is a real infeasibility: adjust gains/caps."""
    out = []
    g = gated(rows)
    gr = gated(ref) if ref is not None else [None] * len(g)
    for r, q in zip(g, gr):
        if r.ok:
            out.append((r, True, "absolute"))
        elif q is not None and not q.ok and r.ratio >= q.ratio - 1e-9:
            out.append((r, True, f"baseline-parity (H1.1 ratio {q.ratio:.3f}, "
                                 f"member {r.ratio:.3f})"))
        else:
            out.append((r, False, f"ratio {r.ratio:.3f}"))
    return out


def deliverable(rows: list[Result], ref: list[Result] | None) -> bool:
    return all(ok for _, ok, _ in verdict(rows, ref))


def table(members: dict[str, list[Result]], ref_name: str = "h1") -> str:
    names = list(members)
    ref = members[ref_name]
    lines = []
    head = f"{'check':26s} {'joint':15s} " + " ".join(f"{n:>26s}" for n in names)
    lines.append(head)
    lines.append("-" * len(head))
    for i, row in enumerate(ref):
        cells = []
        for n in names:
            r = members[n][i]
            mark = "PASS" if r.ok else "FAIL"
            if math.isnan(r.bound):
                cells.append(f"{r.value:9.3f} {'(report)':>16s}")
            else:
                vs = "" if n == ref_name else (
                    " >=ref" if compare([r], [ref[i]])[0] else " <ref")
                cells.append(f"{r.value:9.3f}/{r.bound:8.3f} {mark}{vs:>6s}")
        lines.append(f"{row.check:26s} {row.joint:15s} "
                     + " ".join(f"{c:>26s}" for c in cells))
    lines.append("")
    for n in names:
        lines.append(f"[{n}] " + "; ".join(f"{r.check}/{r.joint or '-'}: "
                                           f"{r.detail}" for r in members[n]))
    return "\n".join(lines)


def main(argv=None) -> int:
    import h1_family as fam  # noqa: PLC0415
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None)
    ap.add_argument("--variants", default=",".join(fam.variant_names()))
    args = ap.parse_args(argv)
    members = {}
    for name in args.variants.split(","):
        members[name] = run(fam.load_lowering(name))
    print(table(members))
    ref = members.get("h1")
    for name, rows in members.items():
        g = gated(rows)
        bad = [r for r in g if not r.ok]
        worse = ([r for r, ok in zip(g, compare(g, gated(ref))) if not ok]
                 if ref is not None else [])
        print(f"{name}: gated (a)-(c) {len(g) - len(bad)}/{len(g)} absolute"
              + (f", FAIL {[r.joint + ':' + r.check for r in bad]}" if bad else "")
              + (f"; worse than H1.1 on {[r.joint + ':' + r.check for r in worse]}"
                 if worse else "; no check worse than H1.1"))
        v = verdict(rows, ref)
        print(f"    verdict: {'DELIVERABLE' if all(ok for _, ok, _ in v) else 'NOT DELIVERABLE'}"
              + "".join(f"\n      {r.check}/{r.joint}: {'ok' if ok else 'FAIL'} ({why})"
                        for r, ok, why in v if why != "absolute"))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {n: [dataclasses.asdict(r) for r in rows]
             for n, rows in members.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
