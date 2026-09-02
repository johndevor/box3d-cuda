#!/usr/bin/env python3
"""Multi-seed strict HUMANOID walking acceptance.

The humanoid twin of walk/eval/acceptance.py (the duck harness, read as the
model): the strict evaluator (walk/eval/humanoid_gait.py, FROZEN) must pass
at ALL three commands (0.50/0.75/1.00 m/s, the env contract's
COMMANDS_MPS) on EVERY seed in SEEDS -- the same four seeds as the duck's,
so overfitting one evaluation seed can't fake acceptance.

Episodes run on FlatFloorHumanoidEnv over the CPU-serial fp32 humanoid
lane (walk/env/humanoid_cuda_lane.CudaHumanoidLane -- the same physics the
GPU trains on; --lane native swaps in the f64 idv1 oracle lane). Traces
come from the shared walk/eval/capture.py recorder; tilt is recomputed
inside the evaluator with the humanoid up axis, so the capture's
duck-frame tilt column is irrelevant here.

Usage:
  .venv/bin/python -B -m walk.eval.humanoid_acceptance \
      --actor <actor_final.pt> [--lane {serial,native}] [--library PATH] \
      [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from walk.env.humanoid_flat import COMMANDS_MPS, FlatFloorHumanoidEnv
from walk.eval.capture import capture_episodes
from walk.eval.humanoid_gait import evaluate_episode
from walk.train.ppo import Actor, RecurrentActor, unpack_actor_file

SEEDS = (4242, 7, 1913, 90210)        # identical to the duck harness's
COMMANDS = COMMANDS_MPS               # (0.50, 0.75, 1.00) m/s
from walk.env.humanoid_flat import OBS, ACT  # active lowering dims (H1: 58/14)


def load_actor(path: str):
    """(arch, actor) from an actor file/checkpoint; 52 -> ... -> 12."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "actor" in raw:        # full checkpoint
        raw = raw["actor"]
    arch, sd = unpack_actor_file(raw)
    actor = (RecurrentActor(OBS, ACT) if arch == "gru" else Actor(OBS, ACT))
    actor.load_state_dict(sd)
    actor.eval()
    return arch, actor


def make_policy(arch: str, actor):
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


def make_env(seed: int, lane: str, library: str | None,
             variant: str | None = None):
    """`variant`: humanoid family member (humanoid/h1_family.py); None = the
    accepted H1.1 base (unchanged construction)."""
    vk = {} if variant in (None, "", "h1") else {"variant": variant}
    if lane == "serial":
        from walk.env.humanoid_cuda_lane import CudaHumanoidLane
        return FlatFloorHumanoidEnv(
            environments=1, seed=seed, perturbation_rad=0.0,
            lane_factory=lambda E, offsets: CudaHumanoidLane(
                E, joint_offsets=offsets, library_path=library, **vk), **vk)
    if lane == "native":
        return FlatFloorHumanoidEnv(environments=1, seed=seed,
                                    perturbation_rad=0.0,
                                    library_path=library, **vk)
    raise SystemExit(f"--lane must be serial or native, got {lane!r}")


def run_acceptance(actor_path: str, lane: str = "serial",
                   library: str | None = None,
                   seeds=SEEDS, commands=COMMANDS,
                   quiet: bool = False, variant: str | None = None) -> dict:
    """Full multi-seed multi-command acceptance; returns the result dict."""
    arch, actor = load_actor(actor_path)
    results, all_pass = {}, True
    for seed in seeds:
        env = make_env(seed, lane, library, variant)
        try:
            for cmd in commands:
                t = capture_episodes(env, make_policy(arch, actor),
                                     command=cmd, seconds=8.0, seed=seed)[0]
                r = evaluate_episode(t)
                q = [f for f in r.get("footfalls", []) if f.get("qualified")]
                left = sum(1 for f in q if f["foot"] == "left")
                fails = [k for k, v in r["criteria"].items()
                         if str(v.get("pass")) != "True"]
                results[f"seed{seed}-cmd{cmd:.2f}"] = {
                    "passed": bool(r["passed"]), "qualified": len(q),
                    "left": left, "right": len(q) - left,
                    "failed_criteria": fails,
                    "criteria": {k: v for k, v in r["criteria"].items()},
                    "swings_examined": r.get("swings_examined", 0)}
                if not quiet:
                    mark = "PASS" if r["passed"] else "fail"
                    print(f"seed {seed} cmd {cmd:.2f}: {mark} q={len(q)} "
                          f"(L{left}/R{len(q) - left})"
                          + (f"  <- {fails}" if fails else ""))
                all_pass &= bool(r["passed"])
        finally:
            env.close()
    n_pass = sum(1 for v in results.values() if v["passed"])
    if not quiet:
        print(f"\n{n_pass}/{len(results)} episodes pass")
        print("HUMANOID WALKING ACCEPTED (all seeds, all commands)"
              if all_pass else "not accepted")
    return {"schema": "duckgridwalk.humanoid-multiseed-acceptance/1",
            "actor": str(actor_path), "arch": arch, "lane": lane,
            "seeds": list(seeds), "commands": list(commands),
            "variant": variant or "h1",
            "episodes": results, "accepted": all_pass}


def _json_default(o):
    """numpy scalars/bools -> python (the harness crashed on np.bool_)."""
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", required=True)
    ap.add_argument("--lane", choices=("serial", "native"), default="serial",
                    help="serial = fp32 humanoid dwc1 build (training "
                         "physics, default); native = f64 idv1 oracle")
    ap.add_argument("--library", default=None,
                    help="optional pinned dylib path for the chosen lane")
    ap.add_argument("--out", default=None)
    ap.add_argument("--variant", default=None,
                    help="humanoid family member (h1_tall | h1_stocky); "
                         "unset = the accepted H1.1 base")
    a = ap.parse_args()
    result = run_acceptance(a.actor, lane=a.lane, library=a.library,
                            variant=a.variant)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "acceptance.json").write_text(
            json.dumps(result, indent=1, default=_json_default))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
