"""The sim-match gate: the same scripted motion in box3d-cuda and in World2's Rapier must agree within stated
tolerances before a skill may train (rl/train.py refuses otherwise).

World2 runs the motion (world2: scripts/sim-match.mjs <skill> --out trace.json) through the runtime a trained policy
runs on and writes its trace and its case (sizes, masses, sensor). Here the skill's probe pins every randomization to
that case, replays the motion in the skill's env and returns named checks:
    {name, value, tolerance, pass, note}
The report (python -m rl.simmatch <skill> --world2 trace.json --out report.json) records the box3d-cuda sources it ran
on; training accepts it only for the same sources.

Shared helpers for the probes:
    insert_descend(env, world2)   the insert contract's constant-action probe (peg_insert, connector_mate)
    check(name, value, tol, note) one check row (pass when |value| <= tol, or value >= tol for ("min", tol))
    rms, lag_ticks                trace comparisons
"""
from __future__ import annotations

import math

import numpy as np
import torch


def check(name, value, tol, note=""):
    if isinstance(tol, tuple) and tol[0] == "min":
        ok = value is not None and value >= tol[1]
    elif isinstance(tol, tuple) and tol[0] == "equal":
        ok = value == tol[1]
    else:
        ok = value is not None and abs(value) <= tol
    v = None if value is None else (value if isinstance(value, bool) else round(float(value), 4) if isinstance(value, (int, float, np.floating)) else value)
    return {"name": name, "value": v, "tolerance": list(tol) if isinstance(tol, tuple) else tol, "pass": bool(ok), "note": note}


def rms(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2))) if n else None


def lag_ticks(ref, x, max_lag=4):
    """The shift (ticks, sub-tick by a parabola through the peak) that best aligns x to ref: positive = x lags."""
    ref, x = np.asarray(ref, float), np.asarray(x, float)
    n = min(len(ref), len(x))
    ref, x = ref[:n] - ref[:n].mean(), x[:n] - x[:n].mean()
    lags = list(range(-max_lag, max_lag + 1))
    c = [float(np.dot(ref[max(0, -L):n - max(0, L)], x[max(0, L):n - max(0, -L)])) / (n - abs(L)) for L in lags]
    i = int(np.argmax(c))
    if 0 < i < len(c) - 1:
        den = c[i - 1] - 2 * c[i] + c[i + 1]
        return lags[i] + (0.5 * (c[i - 1] - c[i + 1]) / den if den else 0.0)
    return float(lags[i])


def insert_descend(env, w2, progress_index=45, wrench_index=12, max_ticks=None):
    """Replay World2's constant-action insert probe in a one-world env pinned to its case. Returns the box3d trace in
    World2's profile form (per tick, the observation before the action: t, believed progress mm, push N, lateral N)
    and the end."""
    a = torch.tensor([w2["action"]], dtype=torch.float32, device=env.dev)
    obs, priv = env.observe()
    T = dict(t=[], progress_mm=[], push_n=[], lateral_n=[])
    end = None
    for k in range(max_ticks or len(w2["trace"]["t"]) + 30):
        o = obs[0].double().cpu()
        T["t"].append(round(k / 30, 3))
        T["progress_mm"].append(float(o[progress_index]) * 1000)
        T["push_n"].append(-float(o[wrench_index + 2]))
        T["lateral_n"].append(float(math.hypot(o[wrench_index], o[wrench_index + 1])))
        obs, priv, r, done, info = env.step(a, autoreset=False)
        if bool(done[0]):
            o = obs[0].double().cpu()
            T["t"].append(round((k + 1) / 30, 3))
            T["progress_mm"].append(float(o[progress_index]) * 1000)
            T["push_n"].append(-float(o[wrench_index + 2]))
            T["lateral_n"].append(float(math.hypot(o[wrench_index], o[wrench_index + 1])))
            end = dict(code=env.code_names.get(int(info["code"][0]), str(int(info["code"][0]))), success=bool(info["success"][0]), ticks=k + 1,
                       true_depth_mm=round(float(info["true_depth"][0]) * 1000, 3))
            break
    return T, end


def insert_checks(w2, T, end, depth_mm, seated_codes=("depth", "policy", "stop"), need_success=True):
    """The insert probe's checks and their tolerances (the reasons are in the notes)."""
    W = w2["trace"]
    wp, bp = np.array(W["progress_mm"]), np.array(T["progress_mm"])
    n = min(len(wp), len(bp))
    free = [i for i in range(n) if wp[i] < 0 and bp[i] < 0]
    inside = [i for i in range(n) if 0 <= wp[i] < depth_mm - 0.3 and 0 <= bp[i] < depth_mm - 0.3]
    reach = lambda p: next((i for i, x in enumerate(p) if x >= depth_mm - 0.05), None)
    rw, rb = reach(wp), reach(bp)
    moving = [i for i in range(n) if wp[i] < depth_mm - 0.05 and bp[i] < depth_mm - 0.05 and i > 0]
    seated_w = bool(w2.get("end", {}).get("ok"))
    seated_b = bool(end and end["code"] in seated_codes and (end["success"] or not need_success))
    mean = lambda arr, idx: float(np.mean([arr[i] for i in idx])) if idx else None
    # force tolerances: 1 N of physics allowance plus 3 sigma of the difference of two independent noisy means over the
    # window (World2's stated F/T sigma per tick; a lateral magnitude's spread is 0.655 sigma)
    sig = ((w2.get("ft_sigma") or {}).get("force_n") or [2.5])
    sig = 0.0 if (w2.get("ft") or {}).get("noise") == 0 else float(sig[0] if isinstance(sig, list) else sig)   # (noise-free sensors: 1 N)
    nm = max(1, len(moving))
    tol_push, tol_lat = round(1.0 + 3 * sig * math.sqrt(2 / nm), 2), round(1.0 + 3 * 0.655 * sig * math.sqrt(2 / nm), 2)
    return [
        check("free-space progress RMS (mm)", rms(wp[free], bp[free]) if free else None, 0.15,
              "the contract's impedance law on the same virtual mass and stiffness, the same 30 Hz reference ramp: a missed tick, a "
              "different gain or a controller lag shows here (a one-tick delay at 15 mm/s is 0.5 mm)"),
        check("in-hole progress RMS (mm)", rms(wp[inside], bp[inside]) if inside else None, 0.3,
              "the part sliding down an aligned hole with clearance: contacts barely load it; 0.3 mm allows the crude box geometry"),
        check("ticks to depth (box3d - World2)", (rb - rw) if rb is not None and rw is not None else None, 2,
              "both reach the planned depth within two 30 Hz ticks"),
        check("seated in both", seated_w and seated_b, ("equal", True), "World2: insert ok; box3d: ended at depth (or stop/policy) with success"),
        check("mean push while moving (N, box3d - World2)", (mean(T["push_n"], moving) - mean(W["push_n"], moving)) if moving else None, tol_push,
              f"the wrist sensor's axial reading while the reference leads the part (the spring's reaction) plus contact drag, over {nm} ticks; "
              f"tolerance 1 N + 3 sigma of the noise ({sig} N per tick)"),
        check("mean lateral force while moving (N, box3d - World2)", (mean(T["lateral_n"], moving) - mean(W["lateral_n"], moving)) if moving else None, tol_lat,
              "aligned: both read only the sensor's noise (a Rayleigh mean of about 1.25 sigma); tolerance 1 N + 3 sigma of its spread"),
    ]
