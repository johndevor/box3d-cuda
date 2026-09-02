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
      [--out DIR] [--variant NAME] [--batched]

BATCHED PROBE (in-training acceptance, walk/train/gpu_train.py
--accept-every for --robot humanoid): run_batched_probe() judges the SAME
12 cells (SEEDS x COMMANDS, 8 s, this module's protocol) as run_acceptance,
but as ONE E=12 batch on the training lane class (CudaHumanoidLane over the
CUDA build on the sandbox, the serial build locally) with domain
randomization, gate terminations and RSI all off. Every cell reproduces
the harness's initial conditions exactly (BatchedCellEnv pins the per-env
command and the gait-phase offset harness_phase0 that the E=1 harness's
counter-based stream would have drawn), so on the serial lane the probe's
per-tick traces -- and therefore the frozen judge's verdict -- are
bit-identical to run_acceptance's (humanoid/tests/test_humanoid_accept_probe.py).
--batched runs that path from the CLI for cross-checking.
"""
from __future__ import annotations

import argparse
import json
import math
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


def make_policy(arch: str, actor, rowwise: bool = False):
    """Deterministic policy closure obs [E,OBS] -> action [E,ACT].

    rowwise=True evaluates every env row as its OWN batch of 1 -- the shape
    the E=1 harness feeds the net. torch's batched CPU matmul is not
    bitwise equal to the row-by-row product (measured 5e-7 on the accepted
    actor), and that difference alone flipped a marginal stocky cell
    (seed 90210 / 0.50 m/s: q=12 L7/R5 fail vs q=13 L7/R6 pass), so a
    batched probe that must agree with the harness uses rowwise."""
    if arch == "ff":
        @torch.no_grad()
        def policy(obs):
            o = torch.from_numpy(np.ascontiguousarray(obs))
            if rowwise:
                return torch.cat([actor.deterministic(o[i:i + 1])
                                  for i in range(o.shape[0])]).numpy()
            return actor.deterministic(o).numpy()
        return policy
    state = {"h": None}

    @torch.no_grad()
    def policy(obs):
        o = torch.from_numpy(np.ascontiguousarray(obs))
        if rowwise:
            if state["h"] is None:
                state["h"] = [actor.initial_state(1) for _ in range(o.shape[0])]
            acts = []
            for i in range(o.shape[0]):
                a, state["h"][i] = actor.deterministic(o[i:i + 1], state["h"][i])
                acts.append(a)
            return torch.cat(acts).numpy()
        if state["h"] is None:
            state["h"] = actor.initial_state(o.shape[0])
        act, state["h"] = actor.deterministic(o, state["h"])
        return act.numpy()
    return policy


def make_env(seed: int, lane: str, library: str | None,
             variant: str | None = None, environments: int = 1):
    """`variant`: humanoid family member (humanoid/h1_family.py); None = the
    accepted H1.1 base (unchanged construction). `environments` > 1 only
    for the batched probe (run_batched_probe)."""
    vk = {} if variant in (None, "", "h1") else {"variant": variant}
    if lane == "serial":
        from walk.env.humanoid_cuda_lane import CudaHumanoidLane
        return FlatFloorHumanoidEnv(
            environments=environments, seed=seed, perturbation_rad=0.0,
            lane_factory=lambda E, offsets: CudaHumanoidLane(
                E, joint_offsets=offsets, library_path=library, **vk), **vk)
    if lane == "native":
        return FlatFloorHumanoidEnv(environments=environments, seed=seed,
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
                rec = cell_record(evaluate_episode(t))
                results[f"seed{seed}-cmd{cmd:.2f}"] = rec
                if not quiet:
                    _print_cell(seed, cmd, rec)
                all_pass &= rec["passed"]
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


def cell_record(r: dict) -> dict:
    """Per-cell acceptance record from one frozen-judge result (the exact
    shape run_acceptance has always written; the batched probe reuses it so
    both paths' records are comparable field for field)."""
    q = [f for f in r.get("footfalls", []) if f.get("qualified")]
    left = sum(1 for f in q if f["foot"] == "left")
    fails = [k for k, v in r["criteria"].items()
             if str(v.get("pass")) != "True"]
    return {"passed": bool(r["passed"]), "qualified": len(q),
            "left": left, "right": len(q) - left,
            "failed_criteria": fails,
            "criteria": {k: v for k, v in r["criteria"].items()},
            "swings_examined": r.get("swings_examined", 0)}


def _print_cell(seed: int, cmd: float, rec: dict) -> None:
    mark = "PASS" if rec["passed"] else "fail"
    fails = rec["failed_criteria"]
    print(f"seed {seed} cmd {cmd:.2f}: {mark} q={rec['qualified']} "
          f"(L{rec['left']}/R{rec['right']})"
          + (f"  <- {fails}" if fails else ""))


# ---------------------------------------------------------------- batched probe
def protocol_cells(seeds=SEEDS, commands=COMMANDS) -> list:
    """The harness's cell order: for each seed, each command -> [(seed, cmd)]."""
    return [(int(s), float(c)) for s in seeds for c in commands]


def harness_phase0(seed: int, command_index: int) -> float:
    """Gait-phase offset (rad) the CPU harness's episode for (seed, k-th
    command) starts from. run_acceptance builds ONE E=1 env per seed -- its
    constructor reset consumes episode 1 of humanoid_flat's counter-based
    (seed, env, episode) stream -- then resets once per command, so command
    k runs as episode k+2 of env 0. The stream draws the command first
    (overridden by set_command) and phase0 = 2*pi*random() second. Pinning
    this value into a batched env reproduces the harness's initial state
    exactly (perturbation 0, randomization off: phase0 is the ONLY
    seed-dependent quantity)."""
    from walk.env.humanoid_flat import _episode_rng  # noqa: PLC0415
    rng = _episode_rng(int(seed), 0, int(command_index) + 2)
    rng.random()                     # the command draw (overridden)
    return 2.0 * math.pi * rng.random()


class BatchedCellEnv:
    """walk/eval/capture.capture_episodes adapter: one env per protocol
    cell. reset() resets the wrapped FlatFloorHumanoidEnv, then pins every
    env's command and phase offset to its cell's harness values; capture
    is called with command=None so each trace records its own command
    (env.command). Everything else delegates unchanged."""

    def __init__(self, env, commands, phases):
        self._env = env
        self.E = env.E
        self.OBS, self.ACT = env.OBS, env.ACT
        if len(commands) != self.E or len(phases) != self.E:
            raise ValueError("one command and one phase per env")
        self._commands = np.asarray(commands, np.float64)
        self._phases = np.asarray(phases, np.float64)

    def reset(self, mask=None, seed=None):
        self._env.reset(mask=mask, seed=seed)
        self._env.set_command(self._commands)
        return self._env.pin_phase(self._phases)

    @property
    def command(self):
        return self._env.command

    def step(self, action, on_tick=None):
        return self._env.step(action, on_tick=on_tick)

    def close(self):
        self._env.close()


def run_batched_probe(env_factory, policy_factory, seeds=SEEDS,
                      commands=COMMANDS, variant: str | None = None,
                      quiet: bool = True, actor_label: str = "") -> dict:
    """Judge all SEEDS x COMMANDS cells in ONE batch; same result schema as
    run_acceptance plus cells_passed / cells_total / failed_cells.

    env_factory(n_envs, seed) -> an n_envs FlatFloorHumanoidEnv over the
    lane class under test (randomization off, perturbation 0);
    policy_factory() -> a fresh deterministic policy closure obs [E,OBS]
    -> action [E,ACT] (fresh so a recurrent policy starts from h = 0).
    Solver faults / terminations are recorded in the traces by
    capture_episodes and fail their cell; nothing is raised."""
    cells = protocol_cells(seeds, commands)
    batch_seed = int(cells[0][0])
    env = env_factory(len(cells), batch_seed)
    try:
        wrapped = BatchedCellEnv(
            env, commands=[c for _, c in cells],
            phases=[harness_phase0(s, list(commands).index(c))
                    for s, c in cells])
        traces = capture_episodes(wrapped, policy_factory(), command=None,
                                  seconds=8.0, seed=batch_seed)
    finally:
        env.close()
    results, all_pass = {}, True
    for (seed, cmd), trace in zip(cells, traces):
        trace["seed"] = seed              # capture stamped the batch seed
        rec = cell_record(evaluate_episode(trace))
        results[f"seed{seed}-cmd{cmd:.2f}"] = rec
        if not quiet:
            _print_cell(seed, cmd, rec)
        all_pass &= rec["passed"]
    n_pass = sum(1 for v in results.values() if v["passed"])
    if not quiet:
        print(f"\n{n_pass}/{len(results)} episodes pass")
        print("HUMANOID WALKING ACCEPTED (all seeds, all commands)"
              if all_pass else "not accepted")
    return {"schema": "duckgridwalk.humanoid-multiseed-acceptance/1",
            "actor": actor_label, "lane": "batched",
            "seeds": list(seeds), "commands": list(commands),
            "variant": variant or "h1",
            "episodes": results, "accepted": all_pass,
            "cells_passed": n_pass, "cells_total": len(results),
            "failed_cells": [k for k, v in results.items() if not v["passed"]]}


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
    ap.add_argument("--batched", action="store_true",
                    help="judge the same 12 cells as ONE E=12 batch "
                         "(the in-training probe's path, run_batched_probe) "
                         "instead of 12 sequential E=1 episodes")
    a = ap.parse_args()
    if a.batched:
        arch, actor = load_actor(a.actor)
        result = run_batched_probe(
            lambda n, seed: make_env(seed, a.lane, a.library, a.variant,
                                     environments=n),
            lambda: make_policy(arch, actor, rowwise=True),
            variant=a.variant, quiet=False, actor_label=str(a.actor))
        result["arch"] = arch
    else:
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
