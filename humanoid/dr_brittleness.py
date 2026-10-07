"""Zero-shot brittleness of a humanoid actor across FIXED randomization draws.

    .venv/bin/python -B humanoid/dr_brittleness.py \
        [--actor evidence/humanoid-accepted-20260902/actor-humanoid-accepted.pt] \
        [--seeds 4242,7] [--commands 0.5,0.75,1.0] [--out runs/dr-brittleness] \
        [--workers 6]

Motivation (yacine's sim2real recipe): the ACCEPTED feed-forward walker is a
specialist for one set of dynamics. Before turning it into a GRU generalist
trained over a per-env dynamics distribution, measure how brittle it is:
score it with the FROZEN judge (walk/eval/humanoid_gait.py via the read-only
walk/eval/humanoid_acceptance.py helpers) at a handful of PINNED per-env
randomization values inside the proposed generalist ranges

    {"r_mass":0.15, "r_friction":0.3, "r_kp":0.15, "r_damping":0.3,
     "max_latency_steps":2, "r_gravity":0.5095}

(r_gravity = 1 - 9.81/20: the authored -20 m/s^2 is the maximum, so the
one-sided gravity scale spans Earth 9.81 .. 2 g 20.0). Every configuration
runs the same seeds x commands protocol as the acceptance harness; the
env's pin_randomization hook fixes the draws while the command/phase0
stream stays identical to an unpinned episode. NOTE r_damping is a no-op on
the humanoid (authored passive joint damping is 0; its damping is kv), so
no damping rows are scored.

Output: <out>/brittleness.json + a markdown table on stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "humanoid"))

ACCEPTED = ROOT / "evidence" / "humanoid-accepted-20260902" / "actor-humanoid-accepted.pt"
RANGES = {"r_mass": 0.15, "r_friction": 0.3, "r_kp": 0.15, "r_damping": 0.3,
          "max_latency_steps": 2, "r_gravity": 0.5095}
G_AUTHORED = 20.0

# name -> pinned values (anything omitted is neutral)
CONFIGS = {
    "nominal":        {},
    "gravity_9.81":   {"gravity_scale": 9.81 / G_AUTHORED},
    "gravity_12":     {"gravity_scale": 12.0 / G_AUTHORED},
    "gravity_15":     {"gravity_scale": 15.0 / G_AUTHORED},
    "gravity_18":     {"gravity_scale": 18.0 / G_AUTHORED},
    "mass_x0.85":     {"mass_scale": 0.85},
    "mass_x1.15":     {"mass_scale": 1.15},
    "friction_x0.7":  {"friction_scale": 0.7},
    "friction_x1.3":  {"friction_scale": 1.3},
    "kp_x0.85":       {"kp_scale": 0.85},
    "kp_x1.15":       {"kp_scale": 1.15},
    "latency_1":      {"latency_steps": 1},
    "latency_2":      {"latency_steps": 2},
    "earth_light_soft": {"gravity_scale": 9.81 / G_AUTHORED, "mass_scale": 0.85,
                         "kp_scale": 0.85, "friction_scale": 0.7,
                         "latency_steps": 1},
    "2g_heavy_stiff_lag": {"mass_scale": 1.15, "kp_scale": 1.15,
                           "friction_scale": 1.3, "latency_steps": 2},
}


def _score(args):
    name, pins, actor_path, seeds, commands = args
    import numpy as np  # noqa: F401
    from walk.env import humanoid_flat as hf
    from walk.env.humanoid_cuda_lane import CudaHumanoidLane
    from walk.eval.capture import capture_episodes
    from walk.eval.humanoid_acceptance import load_actor, make_policy
    from walk.eval.humanoid_gait import evaluate_episode
    arch, actor = load_actor(actor_path)
    rows = {}
    for seed in seeds:
        env = hf.FlatFloorHumanoidEnv(
            environments=1, seed=seed, perturbation_rad=0.0,
            randomization=dict(RANGES),
            lane_factory=lambda E, off: CudaHumanoidLane(
                E, joint_offsets=off, randomization=dict(RANGES)))
        try:
            env.pin_randomization(**pins)
            for cmd in commands:
                t0 = time.perf_counter()
                trace = capture_episodes(env, make_policy(arch, actor),
                                         command=cmd, seconds=8.0, seed=seed)[0]
                r = evaluate_episode(trace)
                q = [f for f in r.get("footfalls", []) if f.get("qualified")]
                fails = [k for k, v in r["criteria"].items()
                         if str(v.get("pass")) != "True"]
                pos = trace["ticks"]["base_pos"]
                rows[f"seed{seed}-cmd{cmd:.2f}"] = {
                    "passed": bool(r["passed"]), "qualified": len(q),
                    "left": sum(1 for f in q if f["foot"] == "left"),
                    "alive_s": round(len(trace["ticks"]["time_s"]) * 0.002, 2),
                    "distance_m": round(pos[-1][0] - pos[0][0], 3) if pos else None,
                    "terminated": bool(trace["terminated"]),
                    "solver_fault": bool(trace["solver_fault"]),
                    "failed_criteria": fails,
                    "wall_s": round(time.perf_counter() - t0, 1)}
        finally:
            env.close()
    return name, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--actor", default=str(ACCEPTED))
    ap.add_argument("--seeds", default="4242,7")
    ap.add_argument("--commands", default="0.5,0.75,1.0")
    ap.add_argument("--out", default=str(ROOT / "runs" / "dr-brittleness"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--configs", default=None,
                    help="comma-separated subset of config names")
    a = ap.parse_args(argv)
    seeds = tuple(int(s) for s in a.seeds.split(","))
    commands = tuple(float(c) for c in a.commands.split(","))
    names = list(CONFIGS) if not a.configs else a.configs.split(",")
    jobs = [(n, CONFIGS[n], a.actor, seeds, commands) for n in names]
    t0 = time.perf_counter()
    results = {}
    with ProcessPoolExecutor(max_workers=a.workers,
                             mp_context=get_context("spawn")) as ex:
        for name, rows in ex.map(_score, jobs):
            results[name] = rows
            n_pass = sum(1 for v in rows.values() if v["passed"])
            print(f"[{time.perf_counter() - t0:6.0f}s] {name:20s} "
                  f"{n_pass}/{len(rows)} pass", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "duckgridwalk.humanoid-dr-brittleness/1",
               "actor": a.actor, "ranges": RANGES, "seeds": list(seeds),
               "commands": list(commands),
               "configs": {n: CONFIGS[n] for n in names}, "results": results}
    (out / "brittleness.json").write_text(json.dumps(payload, indent=1) + "\n")
    # markdown table
    cols = [f"s{s}/c{c:.2f}" for s in seeds for c in commands]
    lines = ["| config | pins | pass | " + " | ".join(cols) + " |",
             "|---|---|---|" + "---|" * len(cols)]
    for n in names:
        rows = results[n]
        cells = []
        for s in seeds:
            for c in commands:
                v = rows[f"seed{s}-cmd{c:.2f}"]
                mark = "PASS" if v["passed"] else "fail"
                cells.append(f"{mark} q{v['qualified']} {v['alive_s']}s "
                             f"{v['distance_m']}m")
        n_pass = sum(1 for v in rows.values() if v["passed"])
        pins = ", ".join(f"{k}={v:.3g}" for k, v in CONFIGS[n].items()) or "authored"
        lines.append(f"| {n} | {pins} | {n_pass}/{len(rows)} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    (out / "brittleness.md").write_text(table + "\n")
    print(table)
    print(f"wrote {out}/brittleness.json ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
