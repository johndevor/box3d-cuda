"""Run a policy on a batched duck env and record full per-tick episode traces.

The trace is a plain-JSON dict (schema ``duckgridwalk.episode/1``) holding one
row per 0.002 s native tick — base pose, tilt, per-foot COM position,
whole-sole height and contact flag — which is exactly what the strict gait
evaluator (walk/eval/gait.py) consumes and what a renderer can replay later.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from walk.env.contract import SolverFault

SCHEMA = "duckgridwalk.episode/1"
SIM_DT = 0.002
CONTROL_DT = 0.02


def _new_trace(command: float, seed, env_index: int) -> dict:
    return {
        "schema": SCHEMA,
        "dt": SIM_DT,
        "policy_dt": CONTROL_DT,
        "command_mps": float(command),
        "seed": seed,
        "env_index": int(env_index),
        "resets": 0,                # single continuous episode by construction
        "terminated": False,        # done before the requested horizon
        "truncated_at_horizon": False,
        "solver_fault": False,
        "ticks": {
            "time_s": [], "base_pos": [], "base_quat_xyzw": [], "tilt_deg": [],
            "foot_pos": [], "sole_height": [], "contact": [],
        },
    }


def _append_tick(trace: dict, state, e: int) -> None:
    t = trace["ticks"]
    q = state.q[e]
    up = 1.0 - 2.0 * (q[3] * q[3] + q[4] * q[4])
    t["time_s"].append(float(state.time[e]))
    t["base_pos"].append([float(x) for x in q[0:3]])
    t["base_quat_xyzw"].append([float(x) for x in q[3:7]])
    t["tilt_deg"].append(math.degrees(math.acos(min(1.0, max(-1.0, up)))))
    t["foot_pos"].append([[float(x) for x in state.foot_pos[e, f]] for f in (0, 1)])
    t["sole_height"].append([float(x) for x in state.sole_height[e]])
    t["contact"].append([bool(x) for x in state.foot_contact[e]])


def capture_episodes(env, policy, command: float | None = None,
                     seconds: float = 8.0, out_dir: str | Path | None = None,
                     seed: int | None = None) -> list[dict]:
    """Reset `env`, run `policy` (obs [E,OBS] -> action [E,14]) and record.

    Returns one trace dict per env. If `command` is given the env must expose
    `set_command` (FlatFloorDuckEnv does); every env is pinned to it. Recording
    for an env stops when it terminates; a solver fault marks all still-live
    traces and stops the run (post-fault observations are never produced).
    """
    obs = env.reset(seed=seed)
    if command is not None:
        obs = env.set_command(command)
        commands = np.full(env.E, float(command))
    else:
        commands = env.command if hasattr(env, "command") else np.full(env.E, np.nan)
    traces = [_new_trace(commands[e], seed, e) for e in range(env.E)]
    steps = int(round(seconds / CONTROL_DT))
    recording = np.ones(env.E, bool)

    def on_tick(state):
        for e in np.flatnonzero(recording):
            _append_tick(traces[e], state, e)

    for step in range(steps):
        action = np.asarray(policy(obs), dtype=np.float32)
        try:
            obs, _, done, _ = env.step(action, on_tick=on_tick)
        except SolverFault as fault:
            for e in np.flatnonzero(recording):
                traces[e]["solver_fault"] = True
                traces[e]["terminated"] = True
                traces[e]["fault_path"] = fault.saved_problem_path
            break
        for e in np.flatnonzero(recording & done):
            if step == steps - 1:
                traces[e]["truncated_at_horizon"] = True
            else:
                traces[e]["terminated"] = True
        recording &= ~done
        if not recording.any():
            break
    for e in range(env.E):
        if not traces[e]["terminated"] and not traces[e]["truncated_at_horizon"]:
            traces[e]["truncated_at_horizon"] = len(traces[e]["ticks"]["time_s"]) >= \
                int(round(seconds / SIM_DT))

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for e, trace in enumerate(traces):
            path = out / f"episode-cmd{trace['command_mps']:.2f}-env{e}.json"
            path.write_text(json.dumps(trace) + "\n")
    return traces
