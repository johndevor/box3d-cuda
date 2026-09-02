"""Author the synthetic H1 reference gait table (humanoid/reference_gait.json).

64-phase x 14-joint kinematic walking cycle for the reward's self-imitation
term (duck v12 term 8a; DW_REF_GAIT kernel infra). Deterministic analytic
construction -- rerunning this script reproduces the json byte-for-byte
(drift-checked by humanoid/tests/test_reference_gait.py).

V3 (execution-feasible weight transfer -- PHASE2.md section 14): v2 had
ZERO double-support time (alternating single support, swings starting
exactly at transfer) and drove the lateral weight shift at the cycle rate,
above the hip-roll plant's sqrt(kp/I_eff) ~= 1.15 Hz bandwidth -- executed
result: the shift never completed, no foot ever unloaded, 0 real swings in
288 episodes across three policies. V3 restructures the cycle around the
transfer and rides the slowed clock (PHASE_HZ_PER_MPS 3.33 -> 1.67,
walk/env/humanoid_flat.py):

  cycle fraction s = phase/(2*pi), DS_FRACTION d = 0.15 per transfer
  (30% total double support, mid-range of human walking's 20-30%):

    s in [0, d)        DS1: both feet down; roll ramps toward the LEFT
                       foot (half-cosine) -- the shift COMPLETES here,
                       before anything lifts;
    s in [d, 0.5)      right swing (35% of cycle); roll HOLDS full-left;
    s in [0.5, 0.5+d)  DS2: ramp toward the RIGHT foot;
    s in [0.5+d, 1)    left swing; roll HOLDS full-right.

  Clock convention preserved: left touchdown at s=0 (sin >= 0 window),
  right touchdown at s=0.5 (sin < 0 window) -- both phase_ok for the
  reward's qualification windows. Swing duration = 0.35/(1.67*v) s:
  0.42 / 0.28 / 0.21 s at cmd 0.5 / 0.75 / 1.0, inside the judge's
  [0.1, 1.2] s band (the executed-validation gate covers cmd <= 0.75).

Sagittal (planar sum rules as v1/v2; FK signs probe-verified):
- hip: stance (0.65 of cycle) sweeps +ALPHA -> -ALPHA as alpha*cos(pi*
  s/0.65); swing recovers -ALPHA -> +ALPHA as -alpha*cos(pi*u). No-slip
  stance sweep must equal the pelvis advance during stance:
  2*L*sin(ALPHA) = 0.65*STRIDE with STRIDE = 1/PHASE_HZ_PER_MPS =
  0.599 m (clock-encoded, command-free; step 0.30 m = 2x the judge's
  0.15 m placement bar) -> ALPHA = 0.2283 rad.
- knee: 0 in stance; swing plateau bell K*min(1, 2*sin^2(pi*u)),
  KNEE_PEAK 0.5 -> 39 mm FK clearance (>= the 30 mm bar; the EXECUTED
  clearance is what the mandatory validation gate now measures).
- ankle: clip(-(hip+knee), +-0.65) foot-flat; the clip binds only near
  the late-swing edge (<= 0.012 rad of pitch).

Roll (the v3 core): roll_L = roll_R = ROLL_AMPLITUDE * lambda(s) with
lambda the ramp/hold trapezoid above (+1 = lean toward the LEFT foot at
world y = -0.15; FK-probed: positive roll on both hips leans the pelvis
toward -y when the left foot is the planted one). ROLL_AMPLITUDE is a
LOAD-COMPENSATION target sized by the executed sweep (the achieved lean
sags under the single-support gravity moment; see PHASE2.md sections
12/14); the ramp gets a full DS window and the hold gets the whole swing
to converge -- unlike v2's above-bandwidth sinusoid.

Verified EXECUTED numbers live in PHASE2.md section 15 and are pinned by
the mandatory ExecutedValidation gate (>= 1 debounced qualified swing per
episode at cmd <= 0.75) -- the institutionalized lesson that FK gates
alone cannot catch execution infeasibility.

FAMILY VARIANTS (humanoid/h1_family.py): the kinematics scale with the
member's leg length -- `payload(lowering)` derives LEG_LENGTH_M from the
lowering (hip->ankle at home) and ROLL_HOLD from its static balance point
asin(hip_half_spacing / leg_length) + the same 0.005 hair, rounded to 2
decimals (H1: 0.18 exactly as authored); every other constant is shared.
`--lowering NAME` writes humanoid/variants/<name>/reference_gait.json; the
default (h1) output is byte-identical to before the switch existed.

Usage: .venv/bin/python -B humanoid/author_reference_gait.py [--lowering NAME]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

BINS = 64
JOINTS = ("waist", "neck",
          "left_hip_roll", "left_hip", "left_knee", "left_ankle",
          "right_hip_roll", "right_hip", "right_knee", "right_ankle",
          "left_shoulder", "left_elbow", "right_shoulder", "right_elbow")
LEG_LENGTH_M = 0.86            # hip->knee 0.54 + knee->ankle 0.32 (humanoid.rs)
PHASE_HZ_PER_MPS = 1.67        # MUST match walk/env/humanoid_flat.py (pinned)
STRIDE_M = 1.0 / PHASE_HZ_PER_MPS      # clock-encoded, command-free
DS_FRACTION = 0.22             # per-transfer double support (44% total --
#                                the 180 N*m-capped lateral shift needs
#                                ~0.25 s; fp32-executed-swept vs 0.15/
#                                0.25/0.28/0.30)
STANCE_FRACTION = 0.5 + DS_FRACTION    # 0.65 of the cycle per foot
SWING_FRACTION = 0.5 - DS_FRACTION     # 0.35 of the cycle per foot
HIP_AMPLITUDE = math.asin(STANCE_FRACTION * STRIDE_M / (2.0 * LEG_LENGTH_M))
KNEE_PEAK = 0.5                # rad; = the ACTION-BOX ceiling (0.5*1.0)
HIP_LIFT = 0.42                # rad; swing-hip flexion bump: the executed
#                                lean drops the pelvis by L*(1-cos(lean))
#                                and a box-capped knee alone cannot out-
#                                shorten it (measured kiss-drag); the hip
#                                bump raises the ankle ~0.12 m at mid-swing
ACTION_BOX = 0.5               # rad; every table target must be reachable
#                                through requested = HOME + 0.5*a, a in
#                                [-1,1] -- an unreachable imitation target
#                                is a permanent error term (ankle was -0.65)
ROLL_DRIVE = 0.4               # rad; TRANSFER OVERDRIVE: commanding far
#                                past the target runs the roll motor at
#                                its 180 N*m cap during DS (the 0.15 m
#                                shift physically needs ~0.25 s at full
#                                torque, a = tau/(m*h); a tracking-scale
#                                ramp crawls at the 2.4 rad/s closed-loop
#                                pendulum rate and arrives a swing late
#                                -- measured, PHASE2.md section 16)
ROLL_HOLD = 0.18               # rad; the STATIC BALANCE POINT asin(0.15/
#                                0.86) = 0.175 + a hair: the executed probe
#                                showed 0.35 holds blow PAST the balance
#                                point into sideways tipping (achieved
#                                0.38 rad, 28 deg termination); once the
#                                transfer completes the gravity moment -> 0
#                                and droop vanishes, so the hold target IS
#                                the desired lean
ROLL_TAPER = 0.06              # rad; the hold DECAYS to this across the
#                                swing (real-gait lean release): the swing
#                                foot has no ankle-roll dof, so body lean
#                                tilts it and its low EDGE eats
#                                0.14*sin(lean) of whole-sole clearance --
#                                25 mm at a constant 0.18 hold (measured
#                                29 mm plateau vs the 30 mm bar); tapering
#                                restored 68 mm executed clearance
ROLL_ADVANCE = 0.05            # cycle fraction; ramp starts late in the
#                                preceding swing so slew+lag finish in DS
OUT = Path(__file__).resolve().parent / "reference_gait.json"
HIP_HALF_SPACING_M = 0.15      # authored hip anchor lateral offset (H1)
ROLL_HOLD_HAIR = 0.005         # ROLL_HOLD = round(asin(d/L) + hair, 2)


class Kinematics:
    """Leg-length-dependent constants of one family member (the module
    constants above ARE the H1 member; `for_lowering` re-derives them)."""

    def __init__(self, leg_length_m: float = LEG_LENGTH_M,
                 hip_half_spacing_m: float = HIP_HALF_SPACING_M,
                 roll_hold: float | None = None, member: str = "h1"):
        self.member = member
        self.leg_length_m = float(leg_length_m)
        self.hip_half_spacing_m = float(hip_half_spacing_m)
        self.hip_amplitude = math.asin(
            STANCE_FRACTION * STRIDE_M / (2.0 * self.leg_length_m))
        self.static_lean = math.asin(self.hip_half_spacing_m / self.leg_length_m)
        self.roll_hold = (round(self.static_lean + ROLL_HOLD_HAIR, 2)
                          if roll_hold is None else float(roll_hold))

    @classmethod
    def for_lowering(cls, lowering=None) -> "Kinematics":
        if lowering is None or getattr(lowering, "VARIANT", "h1") == "h1":
            return cls()                       # H1: the authored constants
        return cls(lowering.LEG_LENGTH_M, lowering.HIP_HALF_SPACING_M,
                   member=lowering.VARIANT)


_H1 = Kinematics()
assert _H1.roll_hold == ROLL_HOLD and _H1.hip_amplitude == HIP_AMPLITUDE


def roll_target(s: float, k: Kinematics = _H1) -> float:
    """v3.2 bang-settle-taper roll: overdrive in DS, tapering hold in swing
    (+ toward the LEFT foot; phase-advanced by ROLL_ADVANCE)."""
    s = (s + ROLL_ADVANCE) % 1.0
    if s < DS_FRACTION:                          # DS1 -> left: overdrive
        return ROLL_DRIVE
    if s < 0.5:                                  # right swing: tapering hold
        u = (s - DS_FRACTION) / (0.5 - DS_FRACTION)
        return k.roll_hold + (ROLL_TAPER - k.roll_hold) * u
    if s < 0.5 + DS_FRACTION:                    # DS2 -> right: overdrive
        return -ROLL_DRIVE
    u = (s - 0.5 - DS_FRACTION) / (0.5 - DS_FRACTION)
    return -(k.roll_hold + (ROLL_TAPER - k.roll_hold) * u)


def leg(s: float, k: Kinematics = _H1) -> tuple[float, float, float]:
    """(hip, knee, ankle) for the LEFT-convention leg (stance from s=0)."""
    s = s % 1.0
    if s < STANCE_FRACTION:
        hip = k.hip_amplitude * math.cos(math.pi * s / STANCE_FRACTION)
        knee = 0.0
    else:
        u = (s - STANCE_FRACTION) / SWING_FRACTION
        hip = (-k.hip_amplitude * math.cos(math.pi * u)
               + HIP_LIFT * math.sin(math.pi * u))
        knee = KNEE_PEAK * min(1.0, 2.0 * math.sin(math.pi * u) ** 2)
    ankle = max(-ACTION_BOX, min(ACTION_BOX, -(hip + knee)))
    return hip, knee, ankle


def table(k: Kinematics = _H1) -> list[list[float]]:
    rows = []
    for b in range(BINS):
        s = (b + 0.5) / BINS
        lh, lk, la = leg(s, k)                 # left: stance from s=0
        rh, rk, ra = leg(s + 0.5, k)           # right: half-cycle shift
        roll = roll_target(s, k)               # both hips, pelvis -> stance
        rows.append([0.0, 0.0, roll, lh, lk, la, roll, rh, rk, ra,
                     0.0, 0.0, 0.0, 0.0])
    return rows


def payload(lowering=None) -> dict:
    """The json payload; `lowering` (module/namespace with VARIANT,
    LEG_LENGTH_M, HIP_HALF_SPACING_M) selects the family member. None or
    the H1 lowering reproduces the committed humanoid/reference_gait.json
    byte-for-byte."""
    k = Kinematics.for_lowering(lowering)
    out = {
        "schema": "duckgridwalk.humanoid_reference_gait/3.2",
        "generator": "humanoid/author_reference_gait.py (analytic; rerun "
                     "reproduces byte-identically)",
        "bins": BINS,
        "joints": list(JOINTS),
        "constants": {
            "phase_hz_per_mps": PHASE_HZ_PER_MPS,
            "stride_m": STRIDE_M,
            "ds_fraction": DS_FRACTION,
            "stance_fraction": STANCE_FRACTION,
            "hip_amplitude_rad": k.hip_amplitude,
            "roll_drive_rad": ROLL_DRIVE,
            "roll_hold_rad": k.roll_hold,
            "roll_taper_rad": ROLL_TAPER,
            "roll_advance_cycle": ROLL_ADVANCE,
            "hip_lift_rad": HIP_LIFT,
            "action_box_rad": ACTION_BOX,
            "knee_peak_rad": KNEE_PEAK,
            "leg_length_m": k.leg_length_m,
            "phase_convention": "left stance from s=0 (sin>=0 window), "
                                "double-support transfers on s in [0,0.15) "
                                "and [0.5,0.65); bin b sampled at "
                                "s=(b+0.5)/64",
        },
        "table": table(k),
    }
    if k.member != "h1":
        out["variant"] = k.member          # family member marker (h1: absent)
        out["constants"]["hip_half_spacing_m"] = k.hip_half_spacing_m
        out["constants"]["static_lean_rad"] = k.static_lean
    return out


def output_path(name: str | None = None) -> Path:
    if name in (None, "", "h1"):
        return OUT
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import h1_family  # noqa: PLC0415
    return h1_family.reference_gait_path(name)


def load_lowering(name: str | None):
    if name in (None, "", "h1"):
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import h1_family  # noqa: PLC0415
    return h1_family.load_lowering(name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="author the reference gait")
    ap.add_argument("--lowering", default="h1",
                    help="family member: h1 (default) | h1_tall | h1_stocky")
    args = ap.parse_args(argv)
    out = output_path(args.lowering)
    text = json.dumps(payload(load_lowering(args.lowering)), indent=1) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
