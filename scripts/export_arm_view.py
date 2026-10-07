#!/usr/bin/env python3
"""Record 3D replay clips of a trained ARM reach actor (KR240 / lite).

Rolls the actor out closed-loop on ArmReachEnv over the CPU-serial fp32 arm
lane (walk/env/arm_cuda_lane.CudaArmLane -- the same physics the GPU trains
on, deterministic), following the acceptance harness's exact cell protocol
(one env per seed, constructor reset = episode 1, then one reset per tier in
order, so "seed S / tier k" here is the SAME episode the frozen judge scores
in walk/eval/arm_acceptance.py). Per 20 ms policy step it records the pose
of every arm body, the flange tip, the active target and its acquisition
state; per 2 ms tick it captures the judge trace and scores it with the
frozen judge, so every clip carries its own verdict.

Geometry for the viewer is derived from the lowering (arm/arm_lowering.py):
each link is a shell (capsule-like segment) from its joint origin to the next
joint origin, expressed in the body's principal COM frame that the lane
reports poses in; the static base column and the judge's floor / base-column
proxy volumes are emitted too. The kernel's body 1 (base_link) is a
decoupled phantom that free-falls -- readers ignore it; the base is drawn
static at the origin.

Programmatic use (the dashboard builder imports these):
    geometry = arm_geometry("kr240")
    clip = record_arm_clip(actor_path, tier=0, seed=4242)
    clips = record_arm_cells(actor_path, seed=4242, tiers=(0, 1, 2))

CLI:
  .venv/bin/python -B scripts/export_arm_view.py \
      --actor runs/gpu/20260902-160543-arm-reach-kr240/artifacts/train/gpu-train-out/actor_final.pt \
      --seed 4242 --tiers 0,1,2 --out dashboard/data/arm
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "arm"))

import arm_lowering as al  # noqa: E402
from walk.env.arm_reach import (  # noqa: E402
    CONTROL_DT, HORIZON_STEPS, link_origins_from_body_state)
from walk.env.contract import SolverFault  # noqa: E402
from walk.eval import arm_reach_judge as judge  # noqa: E402
from walk.eval.arm_acceptance import (  # noqa: E402
    _new_trace, load_actor, make_actor_policy, make_env)

# shell radii per link (m) for the pinned KR240; scaled by length_scale
_LINK_RADIUS = (0.30, 0.17, 0.14, 0.10, 0.09, 0.07)
_BASE_RADIUS = 0.42


def arm_geometry(variant: str = "kr240") -> dict:
    """Viewer geometry for `variant`: per-body segments in the body's
    principal COM frame (the frame lane body_state poses are given in),
    plus the static base and the frozen judge's proxy volumes."""
    s = al.spec(variant)
    sL = s.length_scale
    links = []
    for j, link in enumerate(s.links):
        com = np.asarray(link.com)
        p0 = -com                                   # link origin in COM frame
        nxt = (np.asarray(s.joints[j + 1].xyz) if j + 1 < al.J
               else np.asarray(s.tool_xyz))
        p1 = nxt - com
        links.append({"name": link.name, "body": 2 + j,
                      "radius": round(_LINK_RADIUS[j] * sL, 4),
                      "p0": [round(float(x), 5) for x in p0],
                      "p1": [round(float(x), 5) for x in p1],
                      "mass_kg": link.mass})
    reach = al.reach(s)
    a1_z = float(s.joints[0].xyz[2])
    return {
        "variant": s.variant, "reach_m": round(reach, 4),
        "home_tip": [round(float(x), 4) for x in al.home_tip(s)],
        "base": {"radius": round(_BASE_RADIUS * sL, 4), "height": round(a1_z, 4),
                 "mass_kg": s.base.mass},
        "links": links,
        "tip_body": al.TIP_BODY,
        "tool_offset": [round(float(x), 5) for x in
                        (np.asarray(s.tool_xyz) - np.asarray(s.links[-1].com))],
        "acq_radius_m": judge.ACQ_RADIUS_M[s.variant],
        "tier_radius_m": [round(judge.tier_radius(s.variant, t), 4)
                          for t in judge.TIERS],
        "proxies": {
            "floor_margin_m": round(judge.FLOOR_MARGIN_FRAC * reach, 4),
            "column_radius_m": round(judge.COLUMN_RADIUS_FRAC * reach, 4),
            "column_height_m": round(judge.COLUMN_HEIGHT_FRAC * reach, 4),
        },
        "notes": "body 1 (base_link) is the kernel's decoupled phantom root "
                 "and is not drawn; the base is static at the origin.",
    }


def _frame(env, state) -> dict:
    bs = state.body_state[0]
    tip = al.tip_from_body_state(env.spec, state.body_state)[0]
    return {"bodies": np.round(bs[:, :7], 4).tolist(),
            "tip": np.round(tip, 4).tolist()}


def _run_episode(env, policy, tier: int, seed: int, seconds: float):
    """One harness-protocol episode: returns (frames, trace, events)."""
    env.pin_tier(int(tier))
    obs = env.reset(seed=seed)
    trace = _new_trace(env, int(tier), seed, 0)
    steps = int(round(seconds / CONTROL_DT))
    if steps > HORIZON_STEPS:
        raise ValueError("seconds beyond the env horizon")

    def on_tick(state, env_ref):
        tip = al.tip_from_body_state(env_ref.spec, state.body_state)
        wrist = link_origins_from_body_state(env_ref.spec, state.body_state, 5)
        elbow = link_origins_from_body_state(env_ref.spec, state.body_state, 3)
        t = trace["ticks"]
        t["time_s"].append(float(state.time[0]))
        t["q"].append([float(x) for x in state.q[0, 7:]])
        t["qd"].append([float(x) for x in state.v[0, 6:]])
        t["tip"].append([float(x) for x in tip[0]])
        t["wrist"].append([float(x) for x in wrist[0]])
        t["elbow"].append([float(x) for x in elbow[0]])
        t["target"].append([float(x) for x in env_ref.target[0]])
        t["target_index"].append(int(env_ref.target_index[0]))

    frames, targets, tidx, acquired_steps = [], [], [], []
    holds = []
    ended_at = None
    state = env._lane.read()
    frames.append(_frame(env, state))
    targets.append(np.round(env.target[0], 4).tolist())
    tidx.append(int(env.target_index[0]))
    holds.append(int(env._hold[0]))
    try:
        for step in range(steps):
            a = np.asarray(policy(obs), np.float32)
            obs, _r, done, info = env.step(a, on_tick=on_tick)
            state = env._lane.read()
            frames.append(_frame(env, state))
            targets.append(np.round(env.target[0], 4).tolist())
            tidx.append(int(env.target_index[0]))
            holds.append(int(env._hold[0]))
            if info["acquired"][0]:
                acquired_steps.append(step + 1)
            if done[0]:
                if info["episode_time"][0] < HORIZON_STEPS * CONTROL_DT - 1e-6:
                    trace["terminated"] = True
                    ended_at = round((step + 1) * CONTROL_DT, 2)
                break
    except SolverFault as fault:
        trace["solver_fault"] = True
        trace["fault_path"] = fault.saved_problem_path
        ended_at = round(len(frames) * CONTROL_DT, 2)
    if not trace["terminated"] and not trace["solver_fault"]:
        trace["truncated_at_horizon"] = True
    return frames, targets, tidx, holds, acquired_steps, ended_at, trace


def _verdict(result: dict) -> dict:
    acq = result.get("acquisitions", [])
    fails = [k for k, v in result["criteria"].items()
             if str(v.get("pass")) != "True"]
    return {"passed": bool(result["passed"]),
            "acquired": sum(1 for a in acq if a["acquired"]),
            "acquisition_times_s": [a["time_s"] for a in acq],
            "failed_criteria": fails,
            "max_speed_ratio": result["criteria"].get(
                "joint_speed_within_urdf_limits", {}).get(
                "detail", {}).get("max_speed_ratio")}


def record_arm_cells(actor_path: str, seed: int = 4242,
                     tiers=judge.TIERS, seconds: float = 8.0,
                     variant: str = "kr240", library: str | None = None,
                     lane: str = "serial") -> list[dict]:
    """Harness-exact cells for one seed: one env, tiers in order."""
    arch, actor = load_actor(actor_path)
    env = make_env(variant, int(seed), lane, library)
    clips = []
    try:
        for tier in tiers:
            policy = make_actor_policy(arch, actor)     # fresh GRU state
            (frames, targets, tidx, holds, acq_steps, ended_at,
             trace) = _run_episode(env, policy, int(tier), int(seed), seconds)
            result = judge.evaluate_episode(trace)
            clips.append({
                "schema": "duckgridwalk.dashboard-arm-clip/1",
                "robot": "arm", "variant": variant, "arch": arch,
                "actor": str(actor_path), "seed": int(seed), "tier": int(tier),
                "tier_radius_m": round(judge.tier_radius(variant, int(tier)), 4),
                "acq_radius_m": judge.ACQ_RADIUS_M[variant],
                "dt": CONTROL_DT, "seconds": seconds,
                "physics": f"CudaArmLane serial fp32 ({lane} lane), "
                           f"{env.action_mode} action contract",
                "ended_at": ended_at,
                "frames": frames, "targets": targets, "target_index": tidx,
                "hold_steps": holds, "acquired_steps": acq_steps,
                "verdict": _verdict(result),
            })
    finally:
        env.close()
    return clips


def record_arm_clip(actor_path: str, tier: int, seed: int = 4242, **kw) -> dict:
    """One clip (NOTE: harness cell identity only holds for tier order
    0..k via record_arm_cells; this records a single pinned tier)."""
    return record_arm_cells(actor_path, seed=seed, tiers=(tier,), **kw)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", required=True)
    ap.add_argument("--variant", default="kr240")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--tiers", default="0,1,2")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--library", default=None)
    ap.add_argument("--out", required=True, help="output directory (JSON clips)")
    a = ap.parse_args()
    tiers = tuple(int(x) for x in a.tiers.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"geometry-{a.variant}.json").write_text(
        json.dumps(arm_geometry(a.variant), indent=1))
    for c in record_arm_cells(a.actor, seed=a.seed, tiers=tiers,
                              seconds=a.seconds, variant=a.variant,
                              library=a.library):
        p = out / f"arm-{a.variant}-seed{a.seed}-tier{c['tier']}.json"
        p.write_text(json.dumps(c, separators=(",", ":")))
        v = c["verdict"]
        print(f"{p}  frames={len(c['frames'])}  "
              f"{'PASS' if v['passed'] else 'fail'} acquired {v['acquired']}/5"
              + (f"  <- {v['failed_criteria']}" if v["failed_criteria"] else "")
              + f"  ({p.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
