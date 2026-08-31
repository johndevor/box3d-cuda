"""STRICT walking-acceptance evaluator (exactly per PLAN.md).

Input: a captured episode trace (walk/eval/capture.py JSON schema
``duckgridwalk.episode/1``: per-0.002 s-tick base poses, per-foot COM
positions, whole-sole heights and contact flags). Output: per-criterion
pass/fail JSON.

Acceptance (all three commanded episodes +0.10/+0.15/+0.20 m/s, 8 s each):
  - >= 6 alternating qualified footfalls, >= 3 per foot;
  - a qualified footfall is the touchdown of a swing whose duration is in
    [60 ms, 1.2 s], whose WHOLE sole (min over the 18 sole vertices) clears
    >= 10 mm for a contiguous >= 20 ms, with forward (world +x) placement
    >= 30 mm from liftoff to touchdown, continuous support >= 40 ms
    immediately before liftoff and after touchdown, opposite-foot support for
    >= 90% of the swing, and bounded stance slip;
  - stance slip bound (documented choice): during each stance phase adjacent
    to the swing, the foot COM must stay within STANCE_SLIP_BOUND_M = 25 mm
    horizontally of where that stance began — a planted foot may roll/settle
    a little but must not slide;
  - commanded translation: forward displacement in [60%, 150%] of
    command x 8 s;
  - tilt <= 30 degrees throughout;
  - no reset, termination, or solver fault; a reset-concatenated or
    incomplete trace is rejected outright (incomplete prefixes never qualify);
  - first qualified touchdown <= 2.5 s; last within the final 1.5 s.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA = "duckgridwalk.episode/1"
EPISODE_SECONDS = 8.0
COMMANDS_MPS = (0.10, 0.15, 0.20)
MIN_FOOTFALLS = 6
MIN_FOOTFALLS_PER_FOOT = 3
SWING_MIN_S = 0.060
SWING_MAX_S = 1.2
CLEARANCE_M = 0.010
CLEARANCE_MIN_S = 0.020
PLACEMENT_MIN_M = 0.030
SUPPORT_S = 0.040
OPPOSITE_SUPPORT_FRACTION = 0.90
STANCE_SLIP_BOUND_M = 0.025   # documented bound, see module docstring
TRANSLATION_RANGE = (0.60, 1.50)
TILT_MAX_DEG = 30.0
FIRST_STEP_S = 2.5
LAST_STEP_WINDOW_S = 1.5
FEET = ("left", "right")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [start, end) index runs where mask is True."""
    idx = np.flatnonzero(np.diff(np.r_[0, mask.view(np.int8), 0]))
    return list(zip(idx[0::2].tolist(), idx[1::2].tolist()))


def _integrity(trace: dict) -> tuple[bool, str]:
    if trace.get("schema") != SCHEMA:
        return False, "unknown schema"
    if int(trace.get("resets", 1)) != 0:
        return False, "reset-concatenated trace"
    if trace.get("solver_fault"):
        return False, "solver fault"
    if trace.get("terminated"):
        return False, "terminated before the horizon"
    t = np.asarray(trace["ticks"]["time_s"], float)
    dt = float(trace["dt"])
    if t.size < 2 or dt <= 0:
        return False, "empty trace"
    steps = np.diff(t)
    if (steps <= 0).any() or np.abs(steps - dt).max() > 1e-6:
        return False, "non-uniform or non-monotonic tick clock (reset splice?)"
    if t[-1] - t[0] + dt < EPISODE_SECONDS - 1e-9:
        return False, "episode shorter than 8 s"
    return True, "single continuous 8 s episode"


def _qualified_footfalls(trace: dict) -> tuple[list[dict], list[dict]]:
    """Return (qualified footfalls, all bounded swings with reasons)."""
    ticks = trace["ticks"]
    dt = float(trace["dt"])
    t = np.asarray(ticks["time_s"], float)
    contact = np.asarray(ticks["contact"], bool)            # [n, 2]
    sole = np.asarray(ticks["sole_height"], float)          # [n, 2]
    foot_pos = np.asarray(ticks["foot_pos"], float)         # [n, 2, 3]
    n = len(t)
    support_ticks = int(np.ceil(SUPPORT_S / dt))
    clear_ticks = int(np.ceil(CLEARANCE_MIN_S / dt))
    t0 = t[0] - dt                                           # episode start
    qualified, swings = [], []
    for f in range(2):
        for i0, i1 in _runs(~contact[:, f]):
            # Incomplete prefixes/suffixes never qualify: the swing must be
            # bracketed by in-trace contact.
            if i0 == 0 or i1 == n:
                continue
            duration = (i1 - i0) * dt
            info = {"foot": FEET[f], "liftoff_s": float(t[i0 - 1]),
                    "touchdown_s": float(t[i1]), "swing_s": float(duration)}
            why = []
            if not SWING_MIN_S - 1e-9 <= duration <= SWING_MAX_S + 1e-9:
                why.append("swing duration outside [60 ms, 1.2 s]")
            clear = sole[i0:i1, f] >= CLEARANCE_M
            best = max([(b - a) for a, b in _runs(clear)], default=0)
            info["clearance_s"] = float(best * dt)
            info["max_sole_clearance_m"] = float(sole[i0:i1, f].max(initial=0.0))
            if best < clear_ticks:
                why.append("whole-sole 10 mm clearance held < 20 ms")
            placement = float(foot_pos[i1, f, 0] - foot_pos[i0 - 1, f, 0])
            info["placement_m"] = placement
            if placement < PLACEMENT_MIN_M - 1e-12:
                why.append("forward placement < 30 mm")
            if i0 < support_ticks or not contact[i0 - support_ticks:i0, f].all():
                why.append("< 40 ms support before liftoff")
            if i1 + support_ticks > n or not contact[i1:i1 + support_ticks, f].all():
                why.append("< 40 ms support after touchdown")
            opposite = contact[i0:i1, 1 - f].mean()
            info["opposite_support_fraction"] = float(opposite)
            if opposite < OPPOSITE_SUPPORT_FRACTION - 1e-12:
                why.append("opposite-foot support < 90% of swing")
            slip = 0.0
            for s0, s1 in _runs(contact[:, f]):
                if s1 == i0 or s0 == i1:  # stance phases adjacent to the swing
                    drift = foot_pos[s0:s1, f, :2] - foot_pos[s0, f, :2]
                    slip = max(slip, float(np.linalg.norm(drift, axis=1).max()))
            info["stance_slip_m"] = slip
            if slip > STANCE_SLIP_BOUND_M + 1e-12:
                why.append("stance slip beyond 25 mm bound")
            info["qualified"] = not why
            info["disqualified_because"] = why
            info["liftoff_rel_s"] = float(t[i0 - 1] - t0)
            info["touchdown_rel_s"] = float(t[i1] - t0)
            swings.append(info)
            if not why:
                qualified.append(info)
    qualified.sort(key=lambda x: x["touchdown_s"])
    return qualified, swings


def evaluate_episode(trace: dict | str | Path) -> dict:
    """Evaluate one 8 s episode trace; returns per-criterion pass/fail JSON."""
    if not isinstance(trace, dict):
        trace = json.loads(Path(trace).read_text())
    criteria: dict[str, dict] = {}

    ok, detail = _integrity(trace)
    criteria["single_episode_no_reset_or_failure"] = {"pass": ok, "detail": detail}
    if not ok:
        # Rejected outright: a spliced/incomplete trace never qualifies.
        return {"schema": "duckgridwalk.gait_eval/1", "rejected": True,
                "passed": False, "command_mps": trace.get("command_mps"),
                "criteria": criteria, "footfalls": []}

    ticks = trace["ticks"]
    t = np.asarray(ticks["time_s"], float)
    dt = float(trace["dt"])
    t0 = t[0] - dt
    cmd = float(trace["command_mps"])
    base = np.asarray(ticks["base_pos"], float)
    tilt = np.asarray(ticks["tilt_deg"], float)

    footfalls, swings = _qualified_footfalls(trace)
    per_foot = {name: sum(1 for x in footfalls if x["foot"] == name) for name in FEET}
    alternating = len(footfalls) >= 2 and all(
        a["foot"] != b["foot"] for a, b in zip(footfalls, footfalls[1:]))

    criteria["at_least_6_qualified_footfalls"] = {
        "pass": len(footfalls) >= MIN_FOOTFALLS, "detail": len(footfalls)}
    criteria["at_least_3_per_foot"] = {
        "pass": all(v >= MIN_FOOTFALLS_PER_FOOT for v in per_foot.values()),
        "detail": per_foot}
    criteria["footfalls_alternate"] = {
        "pass": alternating, "detail": [x["foot"] for x in footfalls]}
    displacement = float(base[-1, 0] - base[0, 0])
    expected = cmd * EPISODE_SECONDS
    ratio = displacement / expected if expected else float("nan")
    criteria["translation_60_to_150_percent"] = {
        "pass": bool(TRANSLATION_RANGE[0] - 1e-12 <= ratio <= TRANSLATION_RANGE[1] + 1e-12),
        "detail": {"displacement_m": displacement, "expected_m": expected,
                   "ratio": ratio}}
    criteria["tilt_within_30_degrees"] = {
        "pass": bool(tilt.max() <= TILT_MAX_DEG + 1e-9),
        "detail": {"max_tilt_deg": float(tilt.max())}}
    first = footfalls[0]["touchdown_rel_s"] if footfalls else None
    last = footfalls[-1]["touchdown_rel_s"] if footfalls else None
    criteria["first_step_within_2p5_s"] = {
        "pass": bool(footfalls) and first <= FIRST_STEP_S + 1e-9, "detail": first}
    criteria["last_step_within_final_1p5_s"] = {
        "pass": bool(footfalls)
        and last >= (t[-1] - t0) - LAST_STEP_WINDOW_S - 1e-9, "detail": last}

    return {"schema": "duckgridwalk.gait_eval/1", "rejected": False,
            "command_mps": cmd,
            "passed": all(c["pass"] for c in criteria.values()),
            "criteria": criteria, "footfalls": footfalls,
            "swings_examined": len(swings)}


def evaluate_run(traces: list[dict | str | Path]) -> dict:
    """Strict acceptance over exactly the three commanded 8 s episodes."""
    episodes = [evaluate_episode(x) for x in traces]
    commands = sorted(round(e["command_mps"] or -1, 6) for e in episodes)
    complete = commands == sorted(COMMANDS_MPS)
    return {"schema": "duckgridwalk.gait_eval_run/1",
            "commands_complete": complete,
            "passed": complete and all(e["passed"] for e in episodes),
            "episodes": episodes}


if __name__ == "__main__":
    import sys
    result = evaluate_run(sys.argv[1:]) if len(sys.argv) > 2 \
        else evaluate_episode(sys.argv[1])
    print(json.dumps(result, indent=2, sort_keys=True))
