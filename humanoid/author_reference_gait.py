"""Author the synthetic H1 reference gait table (humanoid/reference_gait.json).

64-phase x 14-joint kinematic walking cycle for the reward's self-imitation
term (duck v12 term 8a; DW_REF_GAIT kernel infra). Deterministic analytic
construction -- rerunning this script reproduces the json byte-for-byte
(drift-checked by humanoid/tests/test_reference_gait.py).

V2 (H1): adds the two hip-roll columns -- a lateral weight shift toward
the stance side, timed to the same phase clock:

    roll_L(p) = roll_R(p) = ROLL_AMPLITUDE * sin(p + ROLL_PHASE_ADVANCE)

Equal values keep the legs parallel; with the stance foot planted the
pelvis leans toward the stance side (left stance on sin >= 0, pelvis
toward the left foot at world y = -0.15; FK-probed sign: positive roll
swings the FREE leg toward +y, i.e. the planted-foot reaction leans the
pelvis toward -y). ROLL_AMPLITUDE = 0.25 rad is a LOAD-COMPENSATION
target, not a kinematic pose: during single support the stance hip-roll
carries ~55 kg x 20 x 0.15 m = 166 N*m (the CoM sits 0.15 m inboard of
the hip -- H0's unfixable tipping torque), so the PD (kp 90, cap 180)
sags well below any commanded angle; commanding 0.25 rad yields ~0.1 rad
of ACHIEVED lean = ~0.09 m of CoM shift over the stance foot. Swept
empirically (0.05 kinematic-margin sizing survived no longer than roll
zero; 0.25 with ROLL_PHASE_ADVANCE 0.4 -- the lean must be established
BEFORE the swing foot lifts -- maximized demonstrator survival x
alternations). Because the roll channel is load compensation, the
kinematic FK gates validate the SAGITTAL columns with rolls zeroed and
the roll columns get amplitude/timing/limit checks of their own
(humanoid/tests/test_reference_gait.py).

DESIGN (planar model: every joint axis is parallel, so foot pitch is the
plain sum hip+knee+ankle; FK sign conventions verified against the real
fixture -- positive q swings the distal segment FORWARD (+x), positive
ankle is toes-up):

- Phase convention: the reward's clock (walk/env/humanoid_reward.py 8b,
  kernel dw_policy_reward): LEFT foot stances while sin(phase) >= 0
  (bins 0..31), swings while sin < 0 (bins 32..63); right is the mirror.
  Left touchdown lands at phase 0, right at pi -- inside their phase_ok
  windows. Bin b is evaluated at its center p = 2*pi*(b + 0.5)/64.

- Stride consistency with the gait clock: cycle_hz = 3.33 * v, so the
  clock encodes a COMMAND-INDEPENDENT stride of v / cycle_hz =
  1/3.33 = 0.3003 m per cycle (0.15 m per step) -- also the judge's and
  reward's placement floor (PLACEMENT_MIN_M = 0.15; per-swing placement =
  one full stride 0.30 m >= the floor with 2x margin). A no-slip stance
  requires the grounded foot to sweep exactly half a stride relative to
  the pelvis per half-cycle: 2 * L_leg * sin(alpha) = 0.15 with
  L_leg = 0.86 m (hip->knee 0.54 + knee->ankle 0.32, humanoid.rs anchors)
  giving HIP_AMPLITUDE alpha = asin(0.075/0.86) = 0.08734 rad. Anything
  larger over-strides the clock and forces foot slip or scuffing.

- hip_L(p) = alpha * cos(p): +alpha at touchdown (foot ahead), -alpha at
  liftoff, linear-ish sweep through stance, forward recovery in swing.
- knee_L(p) = KNEE_PEAK * min(1, 2*sin(p)^2) while sin(p) < 0 (swing
  only), else 0: a plateau bell -- full flexion across the whole
  |sin| >= 0.707 half of the swing, ramping to 0 at liftoff/touchdown.
  KNEE_PEAK = 0.6 rad gives 56 mm whole-sole clearance over the entire
  plateau (>= the judge/reward 30 mm with margin for the <= 3.3 mm stance
  pelvis bob and fp32 tracking error; a plain sin^2 bell sagged to 13 mm
  at the plateau edges), inside the +1.55 knee limit; the foot-flat ankle
  clips at its -0.65 limit for at most 0.01 rad of foot pitch (~1 mm heel
  drop).
- ankle_L(p) = clip(-(hip+knee), ankle limits): foot-flat throughout
  (planar sum), so touchdown and stance are full-sole.
- right leg = left shifted by pi. waist/neck/shoulders/elbows = HOME (0):
  the imitation term should teach the LEG cycle; posture stays the
  reward/termination's business.

Verified numbers (humanoid/tests/test_reference_gait.py, real-fixture FK):
stance-foot sole |z| <= 3.3 mm at every stance bin; swing clearance
>= 30 mm on the certified mid-swing window (peak 56 mm); stance-foot
sweep = 0.150 m per half cycle; all joints within authored limits.

Usage: .venv/bin/python -B humanoid/author_reference_gait.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

BINS = 64
JOINTS = ("waist", "neck",
          "left_hip_roll", "left_hip", "left_knee", "left_ankle",
          "right_hip_roll", "right_hip", "right_knee", "right_ankle",
          "left_shoulder", "left_elbow", "right_shoulder", "right_elbow")
ROLL_AMPLITUDE = 0.25          # rad; rationale in the module docstring
ROLL_PHASE_ADVANCE = 0.4       # rad; lean established BEFORE liftoff
LEG_LENGTH_M = 0.86            # hip->knee 0.54 + knee->ankle 0.32 (humanoid.rs)
STRIDE_M = 0.15 / 0.5          # v / (3.33*v): clock-encoded, command-free
HIP_AMPLITUDE = math.asin((STRIDE_M / 4.0) / LEG_LENGTH_M)  # 0.0873.. rad
KNEE_PEAK = 0.6                # rad; 56 mm mid-swing clearance (docstring)
ANKLE_LIMIT = 0.65             # authored ankle range (humanoid.rs:238/271)
OUT = Path(__file__).resolve().parent / "reference_gait.json"


def leg(p: float) -> tuple[float, float, float]:
    """(hip, knee, ankle) for a leg whose stance window is sin(p) >= 0."""
    hip = HIP_AMPLITUDE * math.cos(p)
    knee = (KNEE_PEAK * min(1.0, 2.0 * math.sin(p) ** 2)
            if math.sin(p) < 0.0 else 0.0)
    ankle = max(-ANKLE_LIMIT, min(ANKLE_LIMIT, -(hip + knee)))
    return hip, knee, ankle


def table() -> list[list[float]]:
    rows = []
    for b in range(BINS):
        p = 2.0 * math.pi * (b + 0.5) / BINS
        lh, lk, la = leg(p)                    # left: stance on sin >= 0
        rh, rk, ra = leg(p + math.pi)          # right: half-cycle shift
        roll = ROLL_AMPLITUDE * math.sin(p + ROLL_PHASE_ADVANCE)
        rows.append([0.0, 0.0, roll, lh, lk, la, roll, rh, rk, ra,
                     0.0, 0.0, 0.0, 0.0])
    return rows


def payload() -> dict:
    return {
        "schema": "duckgridwalk.humanoid_reference_gait/2",
        "generator": "humanoid/author_reference_gait.py (analytic; rerun "
                     "reproduces byte-identically)",
        "bins": BINS,
        "joints": list(JOINTS),
        "constants": {
            "hip_amplitude_rad": HIP_AMPLITUDE,
            "roll_amplitude_rad": ROLL_AMPLITUDE,
            "roll_phase_advance_rad": ROLL_PHASE_ADVANCE,
            "knee_peak_rad": KNEE_PEAK,
            "stride_m": STRIDE_M,
            "leg_length_m": LEG_LENGTH_M,
            "phase_convention": "left stance while sin(phase) >= 0; bin b "
                                "sampled at phase 2*pi*(b+0.5)/64",
        },
        "table": table(),
    }


def main() -> int:
    text = json.dumps(payload(), indent=1) + "\n"
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
