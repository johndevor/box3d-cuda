#!/usr/bin/env python3
"""Multi-seed strict ARM reach acceptance + per-tick trace capture.

The arm twin of walk/eval/humanoid_acceptance.py: the FROZEN judge
(walk/eval/arm_reach_judge.py) must pass at ALL three difficulty tiers on
EVERY seed in judge.SEEDS (4242, 7, 1913, 90210) -- 12/12 -- so
overfitting one evaluation seed can't fake acceptance.

Episodes run on ArmReachEnv over the CPU-serial fp32 arm lane
(walk/env/arm_cuda_lane.CudaArmLane -- the same physics the GPU trains
on; --lane native swaps in the f64 idv1 oracle lane). Traces follow the
``duckgridwalk.arm_reach_episode/1`` schema: one row per 0.002 s tick
with q, qdot, tip / wrist / elbow world positions, the active target and
its index.

Policies: a trained actor (--actor, walk.train.ppo Actor/RecurrentActor,
27 -> ... -> 6) or the scripted damped-least-squares IK baseline
(--policy ik): closed-loop on the observed tip, it exists to prove the task
+ judge are consistent and passable (arm/tests/test_arm_judge.py pins
12/12 for the baseline on both variants) -- it is NOT a learned result.

Usage:
  .venv/bin/python -B -m walk.eval.arm_acceptance --variant kr240 \
      (--actor <actor_final.pt> | --policy ik) [--lane {serial,native}] \
      [--library PATH] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "arm") not in sys.path:
    sys.path.insert(0, str(ROOT / "arm"))
import arm_lowering as al  # noqa: E402

from walk.env.arm_reach import (  # noqa: E402
    ACT, OBS, CONTROL_DT, HORIZON_STEPS, SIM_DT, ArmReachEnv,
    link_origins_from_body_state)
from walk.env.contract import SolverFault  # noqa: E402
from walk.eval import arm_reach_judge as judge  # noqa: E402

SEEDS = judge.SEEDS
TIERS = judge.TIERS


# ---------------------------------------------------------------- capture
def _new_trace(env: ArmReachEnv, tier: int, seed, env_index: int) -> dict:
    return {
        "schema": judge.SCHEMA,
        "variant": env.variant,
        "dt": SIM_DT,
        "policy_dt": CONTROL_DT,
        "tier": int(tier),
        "seed": seed,
        "env_index": int(env_index),
        "resets": 0,
        "terminated": False,
        "truncated_at_horizon": False,
        "solver_fault": False,
        "ticks": {"time_s": [], "q": [], "qd": [], "tip": [], "wrist": [],
                  "elbow": [], "target": [], "target_index": []},
    }


def capture_arm_episodes(env: ArmReachEnv, policy, tier: int,
                         seconds: float = 8.0, seed: int | None = None,
                         out_dir: str | Path | None = None) -> list:
    """Pin `tier`, reset `env` (fresh seed if given), run `policy`
    (obs [E,27] -> action [E,6]) for `seconds` and record per-tick traces
    (one per env). Terminations and solver faults are recorded, never
    raised."""
    env.pin_tier(tier)
    obs = env.reset(seed=seed) if seed is not None else env.reset()
    traces = [_new_trace(env, tier, seed, e) for e in range(env.E)]
    steps = int(round(seconds / CONTROL_DT))
    if steps > HORIZON_STEPS:
        raise ValueError("seconds beyond the env horizon")

    def on_tick(state, env_ref):
        tip = al.tip_from_body_state(env_ref.spec, state.body_state)
        wrist = link_origins_from_body_state(env_ref.spec, state.body_state, 5)
        elbow = link_origins_from_body_state(env_ref.spec, state.body_state, 3)
        tgt = env_ref.target
        idx = env_ref.target_index
        for e, tr in enumerate(traces):
            if tr["terminated"]:
                continue
            t = tr["ticks"]
            t["time_s"].append(float(state.time[e]))
            t["q"].append([float(x) for x in state.q[e, 7:]])
            t["qd"].append([float(x) for x in state.v[e, 6:]])
            t["tip"].append([float(x) for x in tip[e]])
            t["wrist"].append([float(x) for x in wrist[e]])
            t["elbow"].append([float(x) for x in elbow[e]])
            t["target"].append([float(x) for x in tgt[e]])
            t["target_index"].append(int(idx[e]))

    done_prev = np.zeros(env.E, bool)
    horizon_s = HORIZON_STEPS * CONTROL_DT
    try:
        for _ in range(steps):
            a = np.asarray(policy(obs), np.float32)
            obs, _r, done, info = env.step(a, on_tick=on_tick)
            for e in np.flatnonzero(done & ~done_prev):
                # the env's horizon `done` is a truncation, not a failure
                if info["episode_time"][e] < horizon_s - 1e-6:
                    traces[e]["terminated"] = True
            done_prev = done.copy()
            if done.all():
                break
    except SolverFault as fault:
        traces[fault.env_index]["solver_fault"] = True
        traces[fault.env_index]["fault_path"] = fault.saved_problem_path
    for tr in traces:
        if not tr["terminated"] and not tr["solver_fault"]:
            tr["truncated_at_horizon"] = True
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for e, tr in enumerate(traces):
            (out / f"arm-{env.variant}-tier{tier}-seed{seed}-env{e}.json"
             ).write_text(json.dumps(tr))
    return traces


# ------------------------------------------------------------ policies
class ScriptedIKPolicy:
    """Closed-loop damped-least-squares IK on the observed tip, expressed as
    the env's absolute limit-scaled action. Uses only obs channels (q,
    target - tip). It INTEGRATES its own commanded reference q_cmd (one
    Newton step per policy step, Jacobian at the measured q): re-anchoring
    the request to the measured q every step stalls ~4 cm short, because
    the PD's gravity sag (~0.006-0.009 rad on a2/a3) cancels each step --
    a learned policy must likewise emit sag-compensated absolute targets.
    Anti-windup: the reference may lead the measured q by at most
    LEAD_j = max(LEAD_MIN, SPEED_FRAC * vmax_j * kv_j / kp_j) -- the lead at
    which the (unsaturated) PD tracks at SPEED_FRAC of the URDF speed limit
    (steady-state tracking speed = (kp/kv) * lead), so the judge's speed
    clause holds with margin while the sag offset (<= 0.01 rad) fits
    inside the lead."""

    SPEED_FRAC = 0.8      # reference ramp rate (x URDF velocity limit)
    LEAD_FRAC = 0.7       # PD steady-state tracking speed at full lead
    LEAD_MIN = 0.02
    AVOID_MARGIN_FRAC = 0.10   # proxy standoff (x reach) for the repulsion
    AVOID_GAIN = 0.5

    def __init__(self, variant: str, gain: float = 0.8, damping: float = 1e-3):
        self.spec = al.spec(variant)
        self.lim = al.joint_limits(self.spec)
        kp, kv = al.gains(self.spec)
        vmax = al.velocity_limits(self.spec)
        self.max_dq = self.SPEED_FRAC * vmax * CONTROL_DT
        self.lead = np.maximum(self.LEAD_MIN, self.LEAD_FRAC * vmax * kv / kp)
        self.gain, self.damping = gain, damping
        r = al.reach(self.spec)
        self.z_safe = (judge.FLOOR_MARGIN_FRAC + self.AVOID_MARGIN_FRAC) * r
        self.r_safe = (judge.COLUMN_RADIUS_FRAC + self.AVOID_MARGIN_FRAC) * r
        self.h_safe = (judge.COLUMN_HEIGHT_FRAC + self.AVOID_MARGIN_FRAC) * r
        self.q_cmd = None

    def _avoid(self, f) -> np.ndarray:
        """Repulsive tip correction (m) keeping the predicted tip / wrist /
        elbow a margin clear of the judge's floor and base-column proxies
        (a straight Cartesian path can otherwise sweep the forearm through
        them on the way to a legal target)."""
        corr = np.zeros(3)
        for p in (f.tip, f.joint_pos[4], f.joint_pos[2]):
            if p[2] < self.z_safe:
                corr[2] += self.AVOID_GAIN * (self.z_safe - p[2])
        for p in (f.tip, f.joint_pos[4]):
            rad = np.hypot(p[0], p[1])
            if p[2] < self.h_safe and rad < self.r_safe:
                out = np.array([p[0], p[1], 0.0]) / max(rad, 1e-6)
                corr += self.AVOID_GAIN * (self.r_safe - rad) * out
        return corr

    def _jacobian(self, q: np.ndarray) -> np.ndarray:
        f = al.fk(self.spec, q)
        Jv = np.zeros((3, al.J))
        for k in range(al.J):
            Jv[:, k] = np.cross(f.axis[k], f.tip - f.joint_pos[k])
        return Jv

    KI = 0.15        # integral gain (per step) on the MEASURED tip error
    BIAS_MAX_M = 0.2  # integrator clamp (m)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, float)
        E = obs.shape[0]
        if self.q_cmd is None or self.q_cmd.shape[0] != E:
            self.q_cmd = obs[:, 0:6].copy()
            self.bias = np.zeros((E, 3))
        out = np.zeros((E, ACT))
        for e in range(E):
            q = obs[e, 0:6]
            target = obs[e, 12:15]
            delta_meas = obs[e, 18:21]                # target - measured tip
            # servo the REFERENCE's own FK (no plant lag inside the loop --
            # servoing the measured tip overshoots by lead x reach) toward
            # target + bias, where bias integrates the measured error and
            # thereby removes the PD's gravity sag exactly at equilibrium
            self.bias[e] = np.clip(self.bias[e] + self.KI * delta_meas,
                                   -self.BIAS_MAX_M, self.BIAS_MAX_M)
            qc = self.q_cmd[e]
            f = al.fk(self.spec, qc)
            delta_cmd = target + self.bias[e] - f.tip + self._avoid(f)
            Jv = self._jacobian(qc)
            JJt = Jv @ Jv.T + self.damping * np.eye(3)
            dq = Jv.T @ np.linalg.solve(JJt, self.gain * delta_cmd)
            dq = np.clip(dq, -self.max_dq, self.max_dq)
            lead = np.clip(qc + dq - q, -self.lead, self.lead)
            self.q_cmd[e] = np.clip(q + lead, self.lim[:, 0], self.lim[:, 1])
            out[e] = (2.0 * (self.q_cmd[e] - self.lim[:, 0])
                      / (self.lim[:, 1] - self.lim[:, 0]) - 1.0)
        return np.clip(out, -1.0, 1.0)


def load_actor(path: str):
    """(arch, actor) from an actor file/checkpoint; 27 -> ... -> 6."""
    import torch  # noqa: PLC0415
    from walk.train.ppo import Actor, RecurrentActor, unpack_actor_file  # noqa: PLC0415
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "actor" in raw:
        raw = raw["actor"]
    arch, sd = unpack_actor_file(raw)
    actor = (RecurrentActor(OBS, ACT) if arch == "gru" else Actor(OBS, ACT))
    actor.load_state_dict(sd)
    actor.eval()
    return arch, actor


def make_actor_policy(arch: str, actor):
    import torch  # noqa: PLC0415
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


# ------------------------------------------------------------ harness
def make_env(variant: str, seed: int, lane: str, library: str | None):
    if lane == "serial":
        from walk.env.arm_cuda_lane import CudaArmLane  # noqa: PLC0415
        return ArmReachEnv(
            environments=1, seed=seed, perturbation_rad=0.0, variant=variant,
            lane_factory=lambda E, offsets: CudaArmLane(
                E, variant=variant, joint_offsets=offsets, library_path=library))
    if lane == "native":
        return ArmReachEnv(environments=1, seed=seed, perturbation_rad=0.0,
                           variant=variant, library_path=library)
    raise SystemExit(f"--lane must be serial or native, got {lane!r}")


def run_acceptance(variant: str, policy_factory, lane: str = "serial",
                   library: str | None = None, seeds=SEEDS, tiers=TIERS,
                   quiet: bool = False, policy_name: str = "actor") -> dict:
    """Full multi-seed x multi-tier acceptance; returns the result dict.
    policy_factory() must return a fresh policy closure per episode."""
    results, all_pass = {}, True
    for seed in seeds:
        env = make_env(variant, seed, lane, library)
        try:
            for tier in tiers:
                t = capture_arm_episodes(env, policy_factory(), tier=tier,
                                         seconds=judge.EPISODE_SECONDS,
                                         seed=seed)[0]
                r = judge.evaluate_episode(t)
                fails = [k for k, v in r["criteria"].items()
                         if str(v.get("pass")) != "True"]
                acq = [a["time_s"] for a in r.get("acquisitions", [])]
                results[f"seed{seed}-tier{tier}"] = {
                    "passed": bool(r["passed"]), "failed_criteria": fails,
                    "acquisition_times_s": acq,
                    "criteria": r["criteria"]}
                if not quiet:
                    mark = "PASS" if r["passed"] else "fail"
                    times = ", ".join("-" if x is None else f"{x:.2f}" for x in acq)
                    print(f"{variant} seed {seed} tier {tier}: {mark} "
                          f"acq=[{times}]" + (f"  <- {fails}" if fails else ""))
                all_pass &= bool(r["passed"])
        finally:
            env.close()
    n_pass = sum(1 for v in results.values() if v["passed"])
    if not quiet:
        print(f"\n{n_pass}/{len(results)} episodes pass")
        print(f"ARM REACH ACCEPTED ({variant}, all seeds, all tiers)"
              if all_pass else "not accepted")
    return {"schema": "duckgridwalk.arm-multiseed-acceptance/1",
            "variant": variant, "policy": policy_name, "lane": lane,
            "seeds": list(seeds), "tiers": list(tiers),
            "episodes": results, "accepted": all_pass}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(al.VARIANTS), default="kr240")
    ap.add_argument("--actor", default=None)
    ap.add_argument("--policy", choices=("actor", "ik"), default="actor")
    ap.add_argument("--lane", choices=("serial", "native"), default="serial")
    ap.add_argument("--library", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.policy == "ik":
        factory = lambda: ScriptedIKPolicy(a.variant)  # noqa: E731
        name = "scripted-ik"
    else:
        if not a.actor:
            raise SystemExit("--actor is required unless --policy ik")
        arch, actor = load_actor(a.actor)
        factory = lambda: make_actor_policy(arch, actor)  # noqa: E731
        name = str(a.actor)
    result = run_acceptance(a.variant, factory, lane=a.lane, library=a.library,
                            policy_name=name)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "acceptance.json").write_text(json.dumps(result, indent=1))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
