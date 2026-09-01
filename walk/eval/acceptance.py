#!/usr/bin/env python3
"""Multi-seed strict walking acceptance.

Single-seed evaluation can be overfit by fast iteration (we evaluated on seed
4242 all day). Acceptance now requires the strict evaluator to pass at ALL
three commands on EVERY seed in SEEDS.

Usage:
  .venv/bin/python -B -m walk.eval.acceptance --actor <actor_state_dict.pt> \
      [--library build/libintegrated_duck-pinned-97c3d37.dylib] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from walk.env.flat import FlatFloorDuckEnv
from walk.eval.capture import capture_episodes
from walk.eval.gait import evaluate_episode
from walk.train.ppo import Actor

SEEDS = (4242, 7, 1913, 90210)
COMMANDS = (0.10, 0.15, 0.20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", required=True)
    ap.add_argument("--library", default="build/libintegrated_duck-pinned-97c3d37.dylib")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sd = torch.load(a.actor, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "actor" in sd:
        sd = sd["actor"]
    actor = Actor(58, 14)
    actor.load_state_dict(sd)
    actor.eval()

    @torch.no_grad()
    def policy(obs):
        return actor.deterministic(
            torch.from_numpy(np.ascontiguousarray(obs))).numpy()

    results, all_pass = {}, True
    for seed in SEEDS:
        env = FlatFloorDuckEnv(environments=1, seed=seed, perturbation_rad=0.0,
                               library_path=a.library)
        for cmd in COMMANDS:
            t = capture_episodes(env, policy, command=cmd, seconds=8.0,
                                 seed=seed)[0]
            r = evaluate_episode(t)
            q = [f for f in r.get("footfalls", []) if f.get("qualified")]
            L = sum(1 for f in q if f["foot"] == "left")
            fails = [k for k, v in r["criteria"].items()
                     if str(v.get("pass")) != "True"]
            results[f"seed{seed}-cmd{cmd:.2f}"] = {
                "passed": bool(r["passed"]), "qualified": len(q),
                "left": L, "right": len(q) - L, "failed_criteria": fails}
            mark = "PASS" if r["passed"] else "fail"
            print(f"seed {seed} cmd {cmd:.2f}: {mark} q={len(q)} (L{L}/R{len(q)-L})"
                  + (f"  <- {fails}" if fails else ""))
            all_pass &= bool(r["passed"])
        env.close()

    n_pass = sum(1 for v in results.values() if v["passed"])
    print(f"\n{n_pass}/{len(results)} episodes pass")
    print("WALKING ACCEPTED (all seeds, all commands)" if all_pass
          else "not accepted")
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "acceptance.json").write_text(json.dumps(
            {"schema": "duckgridwalk.multiseed-acceptance/1",
             "actor": a.actor, "seeds": SEEDS, "commands": COMMANDS,
             "episodes": results, "accepted": all_pass}, indent=1))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
