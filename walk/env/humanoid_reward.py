"""Reward v2 for velocity-commanded flat-floor H0 humanoid walking.

V2 (anti-attractor rebalance, weights ONLY -- the term SHAPE is frozen to
the duck-v12 structure the robot-generic kernel implements, so env and
in-kernel training stay bit-comparable through the DW_RW_* header pins):
the frozen judge's leg-1 verdict showed the reward's optimum was a
lunge-and-slide -- 8 s survival with ZERO swings, ~33 deg lean, ~0.43 m of
sliding (humanoid/PHASE2.md section 8), i.e. v1's standing subsidy
(alive + wide-sigma tracking creep = up to ~+0.95/step) beat its only
standing penalty (double-support -0.5). V2 makes commanded standing
STRICTLY LOSING and doubles the stepping differential:

  measured per policy step at |cmd| > 0 (pinned on authored trajectories
  by humanoid/tests/test_humanoid_reward_v2.py):
    perfect stand+lean (v1):    +0.28 .. +0.45  <- the observed attractor
    perfect stand+lean (v2):    -0.89 .. -1.00  (strictly worse than
                                                 falling at t=0, ~0)
    perfect stand+lean (v2.1):  -0.61 .. -0.72  (imitation leak +0.24
                                                 included; still strictly
                                                 negative every step)
    imitating in-phase stepper (v2.1): +2.11 .. +2.54  (gap 2.8-3.2/step,
                                                 WIDER than v2's 2.6-2.9)

The four changed weights (rationale one-liners at each constant):
TRACK_SIGMA_SQ 0.25 -> 0.09, W_AIR_TIME 1.5 -> 3.0,
W_DOUBLE_SUPPORT 0.5 -> 1.5, W_PHASE 0.5 -> 1.0. Everything else is v1.

Known residual loophole (documented, not yet observed): a PERMANENT
one-legged stand pays no double-support penalty and nets ~+0.6/step
(~+0.85 with the v2.1 imitation leak); the
signed phase term nets it 0 rather than negative (same as the duck). If a
one-foot-lean attractor emerges, the counter is a no-swing term (penalize
any foot with continuous stance OR air beyond ~2 slowest cycles at
|cmd|>0) -- that is a SHAPE change requiring the kernel twin first; do not
add it python-side alone or env/kernel rewards silently diverge.

Base structure (v1, unchanged): a port of walk/env/reward.py's duck v12
SHAPE (rolling-EMA velocity
tracking, alive, lateral/yaw, action-rate, torque, evaluator-style
qualified steps with placement/opposite-support/stance gates, chatter,
flicker, clearance, phase-locked stance, double-support, alternation /
same-foot) with humanoid-scaled constants. Term numbering and tracker
semantics match reward.py line for line so the future in-kernel port can
diff the two files; walk/env/reward.py itself is untouched and remains the
duck's contract.

SELF-IMITATION (duck term 8a) is LIVE since v2.1: no AUTHORED reference
gait exists anywhere for H0 (humanoid/FEASIBILITY.md section 3), so the
reference is a SYNTHETIC analytic cycle we author ourselves
(humanoid/author_reference_gait.py -> humanoid/reference_gait.json,
FK-validated before use) -- the duck's own REF_GAIT is likewise
self-generated (its best verified cycle). The generated kernel header pins
the same table into DW_REF_GAIT.

State dict fields are identical to reward.py's (J=12-wide where the duck
is 14-wide); the GaitTracker is reused unchanged from reward.py (it is
J-independent: per-foot arrays only).

Constant scaling rationale (one line each) -- duck value in [brackets]:
leg length ratio humanoid/duck ~= 5 (hip height 1.0 m vs ~0.20 m), commands
scale with it, torques with the effort caps, and the gait clock is chosen
by the duck's own recipe step_length = v / (2 * phase_hz).
"""
from __future__ import annotations

import json as _json
from pathlib import Path as _Path

import numpy as np

from .reward import GaitTracker  # noqa: F401  (re-exported; J-independent)

CONTROL_DT = 0.02                # duck env contract policy step (flat.py)

# ---- self-imitation (v2.1: the 8a hook is now LIVE) -------------------------
# Synthetic analytic reference cycle authored by
# humanoid/author_reference_gait.py (design + FK-verified numbers in its
# docstring; kinematic per-frame validation pinned by
# humanoid/tests/test_reference_gait.py). Same table feeds DW_REF_GAIT in
# the generated kernel header, so env and in-kernel imitation stay
# bit-comparable (the kernel imitation path is the duck-tested one).
_REF = _json.loads((_Path(__file__).resolve().parents[2] / "humanoid"
                    / "reference_gait.json").read_text())
REF_GAIT = np.asarray(_REF["table"], dtype=np.float64)       # [64, 12]
REF_BINS = int(_REF["bins"])
W_IMIT = 0.5             # [0.5] duck's proven weight, re-justified against
                         # the v2 magnitudes: max bonus 0.5 = half of
                         # W_TRACK's 1.0 and a quarter of the +-2.0 phase
                         # differential -> a guide, not a rail; a stander
                         # at HOME still leaks ~+0.17/step from it (the
                         # reference passes near HOME twice per cycle),
                         # which does NOT flip v2's stand-strictly-negative
                         # property (re-pinned in test_humanoid_reward_v2).
IMIT_SIGMA_SQ = 0.04     # [0.04] rms 0.2 rad tolerance: the table's own
                         # amplitudes (hip 0.087, knee 0.6) put a stander
                         # ~1.5 sigma out at the knee-active bins --
                         # gradient everywhere, saturation nowhere.

# ---- weights (one-line rationale each; duck v12 value in [brackets]) -------
W_TRACK = 1.0            # [1.0] primary objective, dimensionless bonus.
TRACK_SIGMA_SQ = 0.09    # [0.01] v2 (was 0.25): sigma 0.3 m/s. v1's 0.5
                         # sigma paid the lunge-and-slide 0.054 m/s creep up
                         # to 0.45 track at cmd 0.5 (the standing subsidy);
                         # at 0.3 the creep earns <= 0.11 while a stepper
                         # within 0.3 m/s of command still earns >= 0.37.
TRACK_EMA_S = 0.2        # [0.4] ~half a gait period: humanoid clock is
                         # 2.5 Hz at 0.75 m/s (period 0.4 s) vs duck 0.8 s.
W_ALIVE = 0.5            # [0.5] survival bonus, dimensionless.
W_LATERAL = 0.5          # [0.5] vy^2 + wz^2 penalty; magnitudes comparable
                         # at these commands, keep duck weight for v1.
W_ACTION_RATE = 0.01     # [0.01] actions are dimensionless in both robots.
W_TORQUE = 2e-7          # [2e-4] equal penalty at full saturation: duck
                         # sum(cap^2)=146 vs humanoid 172600 (tiers
                         # 180/140/70) -> 2e-4 * 146 / 172600 ~= 1.7e-7.
W_AIR_TIME = 3.0         # [1.5] v2 (was 1.5): qualified steps are rarer on
                         # the biped (six gates at 5x scale); doubling keeps
                         # the per-second step-bonus ceiling comparable to
                         # the duck's and makes attempting steps beat the
                         # (now negative) standing baseline in expectation.
AIR_TIME_MIN = 0.10      # [0.08] swing at 2.5 steps/s is ~0.16-0.30 s;
AIR_TIME_MAX = 0.50      # [0.40] window scaled with the slower cadence.
PLACEMENT_MIN_M = 0.15   # [0.030] leg-ratio (~5x) scaled minimum step.
OPP_SUPPORT_FRAC = 0.90  # [0.90] evaluator-recipe fraction, dimensionless.
W_CHATTER = 1.0          # [1.0] per sub-CHATTER_MAX_S micro-swing touchdown.
CHATTER_MAX_S = 0.06     # [0.06] tick-scale bounce threshold; tick layout
                         # (10 x 0.002 s) is identical, keep.
W_FLICKER = 2.5          # [2.5] partial-tick stance penalty, same tick math.
TICKS_FULL = 10          # [10] native ticks per policy step (0.02 / 0.002).
STANCE_MIN_S = 0.12      # [0.06] humanoid stance is ~60% of a 0.4 s cycle
                         # (~0.24 s); floor at half of that, like the duck.
W_CLEARANCE = 0.1        # [0.1] per swing foot clearing CLEARANCE_M.
CLEARANCE_M = 0.030      # [0.010] human-scale swing-foot clearance ~3-5 cm.
W_DOUBLE_SUPPORT = 1.5   # [0.5] v2 (was 0.5): the ONLY term active during a
                         # permanent commanded stand must outweigh the
                         # stand's income (alive 0.5 + track creep <= 0.45);
                         # at 1.5 a perfect stand+lean nets -0.89..-1.0 per
                         # step -- strictly worse than falling immediately.
DOUBLE_SUPPORT_GRACE = 0.25  # [0.25] ~ double-support share of a cycle;
                         # right order for 0.4-0.8 s humanoid cycles, keep.
W_ALTERNATE = 0.5        # [0.5] anti-limp structure, dimensionless.
W_SAME_FOOT = 2.0        # [2.0] duck-proven repeat-foot pricing.
W_PHASE = 1.0            # [0.5] v2 (was 0.5): doubles the in-phase stepping
                         # differential (+-2/step at full match/mismatch);
                         # the duck's binding anti-limp force, scaled to the
                         # humanoid's larger standing subsidy. A permanent
                         # double-stand still nets exactly 0 here (signed).


def reward(prev_state: dict, state: dict, action: np.ndarray,
           command: np.ndarray, tracker: GaitTracker,
           dt: float = CONTROL_DT) -> np.ndarray:
    """Per-env float32 reward; updates `tracker` in place.

    Structure mirrors walk/env/reward.py::reward term for term; the only
    removed term is 8a (self-imitation -- empty hook, module docstring).
    """
    a = np.asarray(action, dtype=np.float64)
    cmd = np.asarray(command, dtype=np.float64)
    vx = state["root_lin_vel"][:, 0]
    vy = state["root_lin_vel"][:, 1]
    wz = state["root_ang_vel"][:, 2]
    contact = np.asarray(state["foot_contact"], bool)
    prev_contact = np.asarray(prev_state["foot_contact"], bool)
    sole = state["sole_height"]

    # 1. forward-velocity tracking against a rolling average
    tracker.v_avg += (dt / TRACK_EMA_S) * (vx - tracker.v_avg)
    r = W_TRACK * np.exp(-np.square(tracker.v_avg - cmd) / TRACK_SIGMA_SQ)
    # 2. alive bonus
    r += W_ALIVE
    # 3. lateral / yaw velocity penalty
    r -= W_LATERAL * (np.square(vy) + np.square(wz))
    # 4. action-rate penalty
    r -= W_ACTION_RATE * np.square(a - np.asarray(prev_state["action"])).sum(1)
    # 5. torque penalty
    r -= W_TORQUE * np.square(np.asarray(state["torque"])).sum(1)

    # 6. qualified step bonus + 9. alternation (evaluator-mirroring gates)
    foot_x = state.get("foot_x")
    touchdown = ~prev_contact & contact                       # [E,2]
    liftoff = prev_contact & ~contact
    duration_ok = ((tracker.air_time >= AIR_TIME_MIN)
                   & (tracker.air_time <= AIR_TIME_MAX))
    if foot_x is not None:
        placement_ok = (np.asarray(foot_x) - tracker.liftoff_x) >= PLACEMENT_MIN_M
        opp_ok = tracker.opp_support >= OPP_SUPPORT_FRAC * np.maximum(
            tracker.air_time, dt)
    else:                                     # older callers/tests
        placement_ok = np.ones_like(touchdown)
        opp_ok = np.ones_like(touchdown)
    stance_ok = tracker.pre_swing_stance >= STANCE_MIN_S
    phase = state.get("phase")
    if phase is not None:
        stance_left = np.sin(np.asarray(phase, np.float64)) >= 0.0
        phase_ok = np.stack([stance_left, ~stance_left], axis=1)
    else:
        phase_ok = np.ones_like(touchdown)
    qualified = (touchdown & duration_ok & placement_ok & opp_ok
                 & stance_ok & phase_ok)
    r += W_AIR_TIME * qualified.sum(1)
    r -= W_CHATTER * (touchdown & (tracker.air_time < CHATTER_MAX_S)).sum(1)
    for foot in (0, 1):
        hit = qualified[:, foot]
        r += W_ALTERNATE * (hit & (tracker.last_foot == 1 - foot))
        r -= W_SAME_FOOT * (hit & (tracker.last_foot == foot))
        tracker.last_foot[hit] = foot
        if foot_x is not None:
            tracker.liftoff_x[liftoff[:, foot], foot] = \
                np.asarray(foot_x)[liftoff[:, foot], foot]
        airborne = ~contact[:, foot]
        tracker.opp_support[liftoff[:, foot], foot] = 0.0
        tracker.opp_support[airborne, foot] += contact[airborne, 1 - foot] * dt
        tracker.pre_swing_stance[liftoff[:, foot], foot] = \
            tracker.stance_time[liftoff[:, foot], foot]
    # stance credit only on FULL-contact steps (tick-scale flicker resets)
    ct = state.get("contact_ticks")
    solid = contact if ct is None else contact & (np.asarray(ct) >= TICKS_FULL)
    tracker.stance_time = np.where(solid, tracker.stance_time + dt, 0.0)

    # 6c. flicker penalty: stance at both boundaries, partial tick contact
    if ct is not None:
        flicker = prev_contact & contact & (np.asarray(ct) < TICKS_FULL)
        r -= W_FLICKER * flicker.sum(1)
    tracker.air_time = np.where(contact, 0.0, tracker.air_time + dt)

    # 7. foot-clearance bonus
    r += W_CLEARANCE * ((~contact) & (sole >= CLEARANCE_M)).sum(1)

    # 8a. self-imitation: joint pose near the synthetic reference cycle at
    # own phase (exactly reward.py lines 193-199; kernel twin is the
    # duck-tested DW_REF_GAIT path fed by the same table).
    jq = state.get("joint_q")
    if jq is not None and phase is not None:
        bins = (np.mod(np.asarray(phase, np.float64) / (2.0 * np.pi), 1.0)
                * REF_BINS).astype(int) % REF_BINS
        err = np.mean(np.square(np.asarray(jq, np.float64) - REF_GAIT[bins]),
                      axis=1)
        r += W_IMIT * np.exp(-err / IMIT_SIGMA_SQ) * (np.abs(cmd) > 0)

    # 8b. phase-locked stance while commanded (signed, anti-limp)
    if phase is not None:
        match = np.where(contact[:, 0] == stance_left, 1.0, -1.0) \
            + np.where(contact[:, 1] == ~stance_left, 1.0, -1.0)
        r += W_PHASE * match * (np.abs(cmd) > 0)

    # 8. double-support penalty beyond the grace while commanded
    both = contact.all(1)
    tracker.double_support = np.where(both, tracker.double_support + dt, 0.0)
    r -= W_DOUBLE_SUPPORT * ((np.abs(cmd) > 0)
                             & (tracker.double_support > DOUBLE_SUPPORT_GRACE))
    return r.astype(np.float32)
