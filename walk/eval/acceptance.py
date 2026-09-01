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
from walk.train.ppo import Actor, RecurrentActor, unpack_actor_file

SEEDS = (4242, 7, 1913, 90210)
COMMANDS = (0.10, 0.15, 0.20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", required=True)
    ap.add_argument("--library", default="build/libintegrated_duck-pinned-97c3d37.dylib")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    raw = torch.load(a.actor, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "actor" in raw:        # full checkpoint
        raw = raw["actor"]
    arch, sd = unpack_actor_file(raw)
    actor = (RecurrentActor(58, 14) if arch == "gru" else Actor(58, 14))
    actor.load_state_dict(sd)
    actor.eval()

    def make_policy():
        if arch == "ff":
            @torch.no_grad()
            def policy(obs):
                return actor.deterministic(
                    torch.from_numpy(np.ascontiguousarray(obs))).numpy()
            return policy
        state = {"h": None}

        @torch.no_grad()
        def policy(obs):
            o = torch.from_numpy(np.ascontiguousarray(obs))
            if state["h"] is None:
                state["h"] = actor.initial_state(o.shape[0])
            act, state["h"] = actor.deterministic(o, state["h"])
            return act.numpy()
        return policy

    results, all_pass = {}, True
    for seed in SEEDS:
        env = FlatFloorDuckEnv(environments=1, seed=seed, perturbation_rad=0.0,
                               library_path=a.library)
        for cmd in COMMANDS:
            t = capture_episodes(env, make_policy(), command=cmd, seconds=8.0,
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
