#!/usr/bin/env python3
"""Multi-seed strict walking acceptance on the cube-terrain grid.

The cube-grid counterpart of walk/eval/acceptance.py: the SAME strict episode
protocol (capture_episodes + gait.evaluate_episode, both reused unchanged —
grid_lane already reports sole_height above the SUPPORTING surface, which is
exactly what the evaluator's clearance criterion needs), run on
CubeGridDuckEnv instead of FlatFloorDuckEnv, over seeds (4242, 7, 1913, 90210)
x commands (0.10, 0.15, 0.20) m/s.

The grid ladder is defined here as named stages (single source of truth for
the curriculum runner walk/train/grid_curriculum.py):

  flush   45x10 static grid, spacing == cube_size, zero height jitter
  rough4  same, 4 mm height jitter, terrain seed follows the env seed
  rough8  same, 8 mm height jitter, terrain seed follows the env seed

The terrain seed is deliberately left unset in every stage spec so
grid_lane.resolve_grid pins it to the env seed: each acceptance seed walks
its own deterministic terrain, and re-running is bit-reproducible.

Geometry: dwv1 places cube [ix, iz] at
x = origin_x + (ix - (nx-1)/2) * pitch, y = origin_y + (iz - (nz-1)/2) *
pitch (duck_world_v1.cpp), so nx runs along world +x — the commanded travel
direction — and the lattice is centered on (origin_x, origin_y); the duck's
base starts at (0, 0). The stages must cover the strict protocol's whole
travel envelope (worst case 0.20 m/s x 8 s x 150% translation bound = 2.4 m
ahead, plus margin behind and laterally), otherwise the episode ends in a
60 mm step-down off the edge instead of cube walking — exactly what the
first 8x8 baseline measured. With cube_size = spacing = 0.06 m:
nx=45, nz=10, origin_x=1.05 covers x in [-0.30, +2.40] and y in
[-0.30, +0.30] (cube edges included: centers span +-(n-1)/2*pitch, plus a
half cube each side). 450 static cubes is above the dwv1 capacity gate's
validated 225 (test_capacity.py) but well under the 1024 header cap; a
50-step static hold at this size passes with ~13.8 ms/tick for E=1 —
indistinguishable from the 8x8 grid's cost.

Usage:
  .venv/bin/python -B -m walk.eval.grid_acceptance \
      --actor evidence/walking-accepted-20260901/actor-walking-v1.pt \
      --stage flush --out runs/grid-baseline/flush [--jobs 4]

Exit code 0 iff every seed x command episode passes every criterion.
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import get_context
from pathlib import Path

SEEDS = (4242, 7, 1913, 90210)
COMMANDS = (0.10, 0.15, 0.20)
EPISODE_SECONDS = 8.0

# Named curriculum stages. cube_size/spacing/static follow the known-good
# static-grid specs of walk/env/tests/test_grid.py (FLUSH_GRID); the lattice
# covers the full strict-protocol travel envelope (see module docstring:
# x in [-0.30, +2.40], y in [-0.30, +0.30] around the duck's start). Jitter
# stages add uniform per-cube height jitter. No "seed" key: the terrain seed
# follows the env seed (grid_lane.resolve_grid default_seed).
_STAGE_BASE = dict(nx=45, nz=10, cube_size=0.06, spacing=0.06,
                   origin_x=1.05, origin_y=0.0, dynamic=False)
STAGES = {
    "flush": dict(_STAGE_BASE, height_jitter=0.0),
    "rough4": dict(_STAGE_BASE, height_jitter=0.004),
    "rough8": dict(_STAGE_BASE, height_jitter=0.008),
}


def _json_default(o):
    """gait.py's criteria details can carry numpy scalars (e.g. np.bool_ from
    float-vs-np.float64 comparisons); serialize them as their Python values."""
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def stage_grid(stage: str) -> dict:
    """The (unresolved) grid spec of a named stage; raises on unknown names."""
    try:
        return dict(STAGES[stage])
    except KeyError:
        raise ValueError(f"unknown stage {stage!r}; stages: {sorted(STAGES)}")


def load_actor(path: str):
    """Load an actor file OR a full trainer checkpoint; returns (arch, actor).

    Accepts the accepted-policy format {"arch","state_dict"} (via
    walk.train.ppo.unpack_actor_file), a legacy plain ff state_dict, and
    walk.train.run checkpoints ({"actor": state_dict, ...}) so the curriculum
    can judge latest.pt directly."""
    import torch

    from walk.train.ppo import Actor, RecurrentActor, unpack_actor_file

    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "actor" in raw:        # full checkpoint
        raw = raw["actor"]
    arch, sd = unpack_actor_file(raw)
    actor = (RecurrentActor(58, 14) if arch == "gru" else Actor(58, 14))
    actor.load_state_dict(sd)
    actor.eval()
    return arch, actor


def make_policy(arch: str, actor):
    """Fresh deterministic policy closure (per episode: GRU state resets)."""
    import numpy as np
    import torch

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


def _episode_record(r: dict, trace: dict) -> dict:
    """acceptance.py's per-episode summary row from an evaluate_episode dict,
    plus how far the episode got (duration/travel/fault) so a failing
    baseline stays diagnosable without re-capturing."""
    q = [f for f in r.get("footfalls", []) if f.get("qualified")]
    left = sum(1 for f in q if f["foot"] == "left")
    fails = [k for k, v in r["criteria"].items() if str(v.get("pass")) != "True"]
    times = trace["ticks"]["time_s"]
    base = trace["ticks"]["base_pos"]
    return {"passed": bool(r["passed"]), "qualified": len(q),
            "left": left, "right": len(q) - left, "failed_criteria": fails,
            "duration_s": round(len(times) * float(trace["dt"]), 4),
            "base_x_travel_m": (round(base[-1][0] - base[0][0], 4)
                                if base else 0.0),
            "terminated": bool(trace.get("terminated")),
            "solver_fault": bool(trace.get("solver_fault"))}


def judge_seed(actor_path: str, stage: str, seed: int,
               commands=COMMANDS, seconds: float = EPISODE_SECONDS,
               library: str | None = None,
               impulse_tolerance: float | None = None) -> dict:
    """One env seed: build a 1-env CubeGridDuckEnv on the stage terrain and
    run every command episode serially — same env-reuse protocol as
    acceptance.py, so per-command episode-counter/phase draws match a serial
    run bitwise regardless of --jobs. Returns
    {"grid": resolved spec, "episodes": {key: record}, "details": {...}}."""
    from walk.env.grid import CubeGridDuckEnv
    from walk.eval.capture import capture_episodes
    from walk.eval.gait import evaluate_episode

    arch, actor = load_actor(actor_path)
    env = CubeGridDuckEnv(environments=1, seed=int(seed),
                          grid=stage_grid(stage), perturbation_rad=0.0,
                          library_path=library,
                          impulse_tolerance=impulse_tolerance)
    episodes, details = {}, {}
    try:
        resolved = env.grid
        for cmd in commands:
            trace = capture_episodes(env, make_policy(arch, actor),
                                     command=cmd, seconds=seconds,
                                     seed=int(seed))[0]
            r = evaluate_episode(trace)
            key = f"seed{seed}-cmd{cmd:.2f}"
            episodes[key] = _episode_record(r, trace)
            details[key] = r
    finally:
        env.close()
    return {"grid": resolved, "episodes": episodes, "details": details}


def _judge_seed_star(args):
    return judge_seed(*args)


def judge(actor_path: str, stage: str, seeds=SEEDS, commands=COMMANDS,
          seconds: float = EPISODE_SECONDS, jobs: int = 1,
          library: str | None = None, impulse_tolerance: float | None = None,
          verbose: bool = True) -> dict:
    """Run the full stage judgement; returns the acceptance record dict."""
    tasks = [(actor_path, stage, s, tuple(commands), seconds, library,
              impulse_tolerance) for s in seeds]
    if jobs <= 1:
        per_seed = [_judge_seed_star(t) for t in tasks]
    else:
        ctx = get_context("spawn")
        with ctx.Pool(min(int(jobs), len(tasks))) as pool:
            per_seed = pool.map(_judge_seed_star, tasks)

    episodes, details, grids = {}, {}, {}
    for seed, res in zip(seeds, per_seed):
        grids[str(seed)] = res["grid"]
        episodes.update(res["episodes"])
        details.update(res["details"])
    all_pass = all(v["passed"] for v in episodes.values())
    if verbose:
        for seed in seeds:
            for cmd in commands:
                v = episodes[f"seed{seed}-cmd{cmd:.2f}"]
                mark = "PASS" if v["passed"] else "fail"
                extra = f"  <- {v['failed_criteria']}" if v["failed_criteria"] else ""
                how = ("solver_fault" if v["solver_fault"]
                       else "terminated" if v["terminated"] else "survived")
                print(f"[{stage}] seed {seed} cmd {cmd:.2f}: {mark} "
                      f"q={v['qualified']} (L{v['left']}/R{v['right']}) "
                      f"{how}@{v['duration_s']:.2f}s "
                      f"x+{v['base_x_travel_m']:.3f}m{extra}")
        n_pass = sum(1 for v in episodes.values() if v["passed"])
        print(f"\n{n_pass}/{len(episodes)} episodes pass on stage '{stage}'")
        print(f"GRID STAGE '{stage}' ACCEPTED (all seeds, all commands)"
              if all_pass else f"stage '{stage}' not accepted")
    return {
        "schema": "duckgridwalk.grid-acceptance/1",
        "actor": str(actor_path), "stage": stage,
        "stage_grid": stage_grid(stage),
        "resolved_grids_by_seed": grids,
        "seeds": list(seeds), "commands": list(commands),
        "episode_seconds": seconds,
        "episodes": episodes,
        "episode_details": details,
        "accepted": all_pass,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="walk.eval.grid_acceptance")
    ap.add_argument("--actor", required=True)
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                    help="parallel seed workers (protocol is per-seed serial "
                         "over commands, so results are jobs-invariant)")
    ap.add_argument("--library", default=None,
                    help="prebuilt libduck_world path (default: world.build())")
    ap.add_argument("--impulse-tolerance", type=float, default=None,
                    help="override the grid lane's civ1 impulse tolerance")
    ap.add_argument("--seconds", type=float, default=EPISODE_SECONDS,
                    help=argparse.SUPPRESS)  # test hook; <8 s never passes
    a = ap.parse_args(argv)

    record = judge(a.actor, a.stage, jobs=a.jobs, seconds=a.seconds,
                   library=a.library, impulse_tolerance=a.impulse_tolerance)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "grid_acceptance.json").write_text(
            json.dumps(record, indent=1, sort_keys=True,
                       default=_json_default) + "\n")
    return 0 if record["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
