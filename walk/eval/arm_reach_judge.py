"""STRICT fixed-base arm reach-acceptance evaluator. FROZEN.

FROZEN in the same sense as walk/eval/gait.py and walk/eval/humanoid_gait.py:
thresholds and semantics may not drift with training; any amendment needs an
explicit authorization note. The judge consumes a captured per-tick trace
(schema ``duckgridwalk.arm_reach_episode/1``, produced by
walk/eval/arm_acceptance.capture_arm_episodes over walk/env/arm_reach.py) and
decides one 8 s episode.

THE TASK (per episode, per difficulty tier): five sequential targets are
presented, uniformly sampled (seeded) in the reachable workspace inside a
ball of radius TIER_RADIUS_FRAC[tier] * reach around the HOME tip; target k+1
is presented once target k is acquired.

ACCEPTANCE (all clauses; an episode passes iff every clause passes):
  1. integrity: single continuous 8 s episode at the 0.002 s tick, no reset,
     no solver fault, no termination (a proxy crash terminates the env);
  2. all 5 targets ACQUIRED within 8 s: target k counts as acquired at the
     first policy-step boundary (every CONTROL_TICKS = 10 ticks, i.e. the
     state the controller sees at 50 Hz) at which the tip has been within
     ACQ_RADIUS_M[variant] of it (2 cm KR240 / 1.5 cm lite) at
     ACQ_HOLD_STEPS = 14 CONSECUTIVE boundaries -- a continuous hold
     spanning 0.26 s >= ACQ_HOLD_S (0.25 s) at the control cadence -- while
     k was the active target; acquisitions must occur in order. Sampling
     at the control boundaries makes the judge's count IDENTICAL to the
     env's target-advance rule (walk/env/arm_reach.py), so a presented
     target sequence and its verdict can never disagree;
  3. no joint-limit violation: every joint stays inside its URDF limits
     with LIMIT_TOL_RAD (0.01 rad) of tolerance for the solver's soft
     limit rows, at every tick;
  4. joint speed within the URDF velocity limits at every tick (per-tick
     qdot from the lane; no filtering, SPEED_TOL_FRAC 1.0);
  5. no self-collision / floor proxy violation at any tick (a simple,
     conservative geometric proxy; the URDF ships collision meshes but no
     validated self-collision set -- asset json "self_collision_validated":
     false -- so a mesh check would be no more authoritative):
       floor:  tip, wrist (a5 origin) and elbow (a3 origin) world z >=
               FLOOR_MARGIN_FRAC * reach;
       column: tip and wrist keep a horizontal distance >= COLUMN_RADIUS_FRAC
               * reach from the a1 axis whenever their z <
               COLUMN_HEIGHT_FRAC * reach (the base + link_1 column).
Multi-seed acceptance (walk/eval/arm_acceptance.py, mirroring
humanoid_acceptance.py): SEEDS (4242, 7, 1913, 90210) x 3 tiers = 12/12.

Variant scaling: every length threshold except the acquisition radius is a
fraction of the variant's reach (arm_lowering.reach: 3.46 m KR240, 1.73 m
lite); the acquisition radii are the authored absolute values above --
tighter relative precision for the smaller arm (0.87 % vs 0.58 % of reach).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "arm") not in sys.path:
    sys.path.insert(0, str(ROOT / "arm"))
import arm_lowering as al  # noqa: E402  (frozen physics constants: limits)

SCHEMA = "duckgridwalk.arm_reach_episode/1"
EPISODE_SECONDS = 8.0
SIM_DT = al.SIM_DT
N_TARGETS = 5
ACQ_RADIUS_M = {"kr240": 0.020, "lite": 0.015}
ACQ_HOLD_S = 0.25
CONTROL_TICKS = al.TICKS_PER_CONTROL                     # 10 ticks = 0.02 s
ACQ_HOLD_STEPS = int(math.ceil(ACQ_HOLD_S / al.CONTROL_DT)) + 1   # 14 (0.26 s span)
# difficulty tiers (x reach). Calibration (walk/eval/arm_acceptance.py
# ScriptedIKPolicy, a proxy-aware straight-line IK at 0.8 x the URDF speed
# limits, 2026-09-02): tiers 0/1 pass 16/16 (both variants, 4 seeds); tier 2
# at 0.40 passes 17/24 first-episode draws (4/8 on the acceptance-sequence
# draws) with EVERY failure a 4-of-5 TIME miss (~1.4 s per transfer against
# a1's 1.8 rad/s limit), none a crash -- hard but feasible; at 0.45-0.50
# straight-line paths start crashing into the proxies.
TIER_RADIUS_FRAC = (0.15, 0.30, 0.40)
TIERS = tuple(range(len(TIER_RADIUS_FRAC)))
LIMIT_TOL_RAD = 0.01
SPEED_TOL_FRAC = 1.0
FLOOR_MARGIN_FRAC = 0.05
COLUMN_RADIUS_FRAC = 0.20
COLUMN_HEIGHT_FRAC = 0.40
SEEDS = (4242, 7, 1913, 90210)               # identical to the duck harness's


def tier_radius(variant: str, tier: int) -> float:
    return TIER_RADIUS_FRAC[int(tier)] * al.reach(al.spec(variant))


def proxy_violation(variant: str, tip: np.ndarray, wrist: np.ndarray,
                    elbow: np.ndarray) -> np.ndarray:
    """[E] bool: the self-collision / floor proxy (clause 5) for batched
    world positions [E,3] of tip, wrist (a5 origin) and elbow (a3 origin).
    Shared with the env's termination and reward so they cannot diverge."""
    r = al.reach(al.spec(variant))
    zmin = FLOOR_MARGIN_FRAC * r
    rcol = COLUMN_RADIUS_FRAC * r
    hcol = COLUMN_HEIGHT_FRAC * r
    floor = (tip[:, 2] < zmin) | (wrist[:, 2] < zmin) | (elbow[:, 2] < zmin)

    def column(p):
        return (p[:, 2] < hcol) & (np.hypot(p[:, 0], p[:, 1]) < rcol)
    return floor | column(tip) | column(wrist)


def _runs(mask: np.ndarray) -> list:
    idx = np.flatnonzero(np.diff(np.r_[0, mask.view(np.int8), 0]))
    return list(zip(idx[0::2].tolist(), idx[1::2].tolist()))


def _integrity(trace: dict) -> tuple:
    if trace.get("schema") != SCHEMA:
        return False, "unknown schema"
    if trace.get("variant") not in ACQ_RADIUS_M:
        return False, "unknown variant"
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


def acquisitions(trace: dict) -> list:
    """Ordered acquisition records for the first N_TARGETS targets: for
    target k, the first policy-step boundary (ticks CONTROL_TICKS-1,
    2*CONTROL_TICKS-1, ...) while k is active that ends a run of
    ACQ_HOLD_STEPS consecutive in-radius boundaries. Returns dicts with
    'target', 'acquired', 'time_s' (None if not acquired) and 'min_dist_m'
    (over ALL ticks while active)."""
    ticks = trace["ticks"]
    dt = float(trace["dt"])
    t = np.asarray(ticks["time_s"], float)
    tip = np.asarray(ticks["tip"], float)
    target = np.asarray(ticks["target"], float)
    active = np.asarray(ticks["target_index"], int)
    radius = ACQ_RADIUS_M[trace["variant"]]
    t0 = t[0] - dt
    boundary = np.arange(CONTROL_TICKS - 1, len(t), CONTROL_TICKS)
    out = []
    for k in range(N_TARGETS):
        sel = active == k
        rec = {"target": k, "acquired": False, "time_s": None,
               "min_dist_m": None}
        if sel.any():
            d_all = np.linalg.norm(tip[sel] - target[sel], axis=1)
            rec["min_dist_m"] = float(d_all.min())
            bsel = boundary[sel[boundary]]
            d = np.linalg.norm(tip[bsel] - target[bsel], axis=1)
            inside = d <= radius + 1e-12
            for a, b in _runs(inside):
                if b - a >= ACQ_HOLD_STEPS:
                    rec["acquired"] = True
                    rec["time_s"] = float(t[bsel[a + ACQ_HOLD_STEPS - 1]] - t0)
                    break
        out.append(rec)
    return out


def evaluate_episode(trace: dict | str | Path) -> dict:
    """Evaluate one 8 s reach episode; per-criterion pass/fail JSON."""
    if not isinstance(trace, dict):
        trace = json.loads(Path(trace).read_text())
    criteria: dict = {}
    ok, detail = _integrity(trace)
    criteria["single_episode_no_reset_or_failure"] = {"pass": ok, "detail": detail}
    if not ok:
        return {"schema": "duckgridwalk.arm_reach_eval/1", "rejected": True,
                "passed": False, "variant": trace.get("variant"),
                "tier": trace.get("tier"), "criteria": criteria,
                "acquisitions": []}
    variant = trace["variant"]
    s = al.spec(variant)
    ticks = trace["ticks"]
    q = np.asarray(ticks["q"], float)
    qd = np.asarray(ticks["qd"], float)
    tip = np.asarray(ticks["tip"], float)
    wrist = np.asarray(ticks["wrist"], float)
    elbow = np.asarray(ticks["elbow"], float)

    acq = acquisitions(trace)
    times = [a["time_s"] for a in acq]
    all_acq = all(a["acquired"] for a in acq)
    in_order = all_acq and all(a < b for a, b in zip(times, times[1:]))
    criteria["all_5_targets_acquired_in_order"] = {
        "pass": bool(all_acq and in_order),
        "detail": {"acquired": [a["acquired"] for a in acq],
                   "times_s": times,
                   "min_dist_m": [a["min_dist_m"] for a in acq]}}

    lim = al.joint_limits(s)
    excess = np.maximum(np.maximum(lim[:, 0] - q, q - lim[:, 1]), 0.0)
    criteria["joint_limits_respected"] = {
        "pass": bool(excess.max() <= LIMIT_TOL_RAD + 1e-12),
        "detail": {"max_excess_rad": float(excess.max())}}

    vlim = al.velocity_limits(s)
    ratio = np.abs(qd) / vlim
    criteria["joint_speed_within_urdf_limits"] = {
        "pass": bool(ratio.max() <= SPEED_TOL_FRAC + 1e-9),
        "detail": {"max_speed_ratio": float(ratio.max()),
                   "per_joint": ratio.max(0).tolist()}}

    viol = proxy_violation(variant, tip, wrist, elbow)
    criteria["no_self_collision_or_floor_proxy_violation"] = {
        "pass": bool(not viol.any()),
        "detail": {"violating_ticks": int(viol.sum()),
                   "min_tip_z_m": float(tip[:, 2].min()),
                   "min_wrist_z_m": float(wrist[:, 2].min()),
                   "min_elbow_z_m": float(elbow[:, 2].min())}}

    return {"schema": "duckgridwalk.arm_reach_eval/1", "rejected": False,
            "variant": variant, "tier": trace.get("tier"),
            "passed": all(c["pass"] for c in criteria.values()),
            "criteria": criteria, "acquisitions": acq}


def evaluate_run(traces: list) -> dict:
    """Strict acceptance over exactly the three tiers of one seed."""
    episodes = [evaluate_episode(x) for x in traces]
    tiers = sorted(int(e["tier"]) if e["tier"] is not None else -1
                   for e in episodes)
    complete = tiers == sorted(TIERS)
    return {"schema": "duckgridwalk.arm_reach_eval_run/1",
            "tiers_complete": complete,
            "passed": complete and all(e["passed"] for e in episodes),
            "episodes": episodes}


if __name__ == "__main__":
    result = evaluate_run(sys.argv[1:]) if len(sys.argv) > 2 \
        else evaluate_episode(sys.argv[1])
    print(json.dumps(result, indent=2, sort_keys=True))
