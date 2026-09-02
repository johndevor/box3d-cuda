"""Swing-clause diagnostics: which judge clause binds at the tree frontier?

Runs a policy closed-loop on the fp32 CPU-serial humanoid lane across
commands x seeds x envs, records EVERY swing (liftoff -> touchdown, per
foot) with its raw stats, and reports per-clause distributions plus the
first-failing clause histogram against the gate_proxy/judge bars
(walk/eval/humanoid_gait.py: duration in [0.1, 1.2] s, whole-sole
clearance >= 30 mm held contiguously >= 30 ms, forward placement
>= 0.15 m).

Three-way consistency on the SAME trajectories (env tick path and the
in-kernel policy path are bit-identical, parity-proven):
  (a) this analyzer, RAW tick contact  <-> (b) dwc1 gate_proxy counters
      (both raw-tick, same 3 clauses; should agree exactly modulo
      end-of-episode bracketing),
  (a') this analyzer with the judge's 20 ms contact debounce -> quantifies
      the debounce/tick-resolution gap,
  (c) the FROZEN judge for any episode that survives the full 8 s.

Policies:
  tree  <actor.pt>   -- e.g. the 20260902 tree leg's actor_final.pt
  bc                 -- the committed humanoid/bc_init.pt clone
  demo               -- the BC demonstrator itself (reference gait +
                        balance assists, humanoid/bc_dataset.py) => the
                        reference's EXECUTED numbers, not FK numbers

Usage:
  .venv/bin/python -B humanoid/diagnose_swings.py \
      [--tree runs/gpu/.../actor_final.pt] [--seeds 4] [--envs 8] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "humanoid"))

import torch  # noqa: E402

import bc_dataset  # noqa: E402
from walk.env import humanoid_flat as hf  # noqa: E402
from walk.env.humanoid_cuda_lane import CudaHumanoidLane  # noqa: E402
from walk.eval import humanoid_gait as judge  # noqa: E402
from walk.eval.capture import capture_episodes  # noqa: E402
from walk.train.ppo import Actor, unpack_actor_file  # noqa: E402

DT = 0.002
SEED_BASE = (4242, 7, 1913, 90210, 31, 62, 93, 124)


def _runs(mask: np.ndarray):
    idx = np.flatnonzero(np.diff(np.r_[0, mask.view(np.int8), 0]))
    return list(zip(idx[0::2].tolist(), idx[1::2].tolist()))


def _debounce(contact: np.ndarray, seconds: float = 0.020) -> np.ndarray:
    out = contact.copy()
    ticks = int(round(seconds / DT))
    for f in range(2):
        for g0, g1 in _runs(~out[:, f]):
            if g0 > 0 and g1 < len(out) and (g1 - g0) <= ticks:
                out[g0:g1, f] = True
    return out


def swings_from_trace(trace: dict, debounce: bool) -> list[dict]:
    """Every bracketed swing with raw stats + first-failing clause."""
    t = trace["ticks"]
    contact = np.asarray(t["contact"], bool)
    if debounce:
        contact = _debounce(contact)
    sole = np.asarray(t["sole_height"], float)
    foot_x = np.asarray(t["foot_pos"], float)[:, :, 0]
    n = len(contact)
    hold_ticks = int(np.ceil(judge.CLEARANCE_MIN_S / DT))
    out = []
    for f in range(2):
        for i0, i1 in _runs(~contact[:, f]):
            if i0 == 0 or i1 == n:
                continue                       # unbracketed: never a swing
            duration = (i1 - i0) * DT
            clear = sole[i0:i1, f]
            hold = max([(b - a) for a, b in
                        _runs(clear >= judge.CLEARANCE_M)], default=0)
            placement = float(foot_x[i1, f] - foot_x[i0 - 1, f])
            fails = []
            if not (judge.SWING_MIN_S - 1e-9 <= duration
                    <= judge.SWING_MAX_S + 1e-9):
                fails.append("duration")
            if hold < hold_ticks:
                fails.append("clearance")
            if placement < judge.PLACEMENT_MIN_M - 1e-12:
                fails.append("placement")
            out.append({
                "foot": "LR"[f], "duration_s": duration,
                "peak_clearance_m": float(clear.max(initial=0.0)),
                "clearance_hold_s": hold * DT,
                "placement_m": placement,
                "qualified": not fails,
                "first_fail": fails[0] if fails else None,
            })
    return out


# ---------------------------------------------------------------- policies
def load_policy(kind: str, tree_path: str | None):
    if kind == "demo":
        return lambda obs: bc_dataset.reference_actions(obs).astype(np.float32)
    path = (ROOT / "humanoid" / "bc_init.pt") if kind == "bc" else Path(tree_path)
    arch, sd = unpack_actor_file(torch.load(path, map_location="cpu",
                                            weights_only=False))
    assert arch == "ff", arch
    actor = Actor(hf.OBS, hf.ACT)
    actor.load_state_dict(sd)
    actor.eval()

    @torch.no_grad()
    def policy(obs):
        return actor.deterministic(
            torch.from_numpy(np.ascontiguousarray(obs))).numpy()
    return policy


def run_episodes(policy, cmd: float, seed: int, envs: int):
    """(traces, gate_proxy rows) for one batch, identical trajectories:
    tick-path capture for the trace, policy-path rerun for the proxy."""
    env = hf.FlatFloorHumanoidEnv(
        environments=envs, seed=seed,
        lane_factory=lambda E, off: CudaHumanoidLane(E, joint_offsets=off))
    try:
        traces = capture_episodes(env, policy, command=cmd, seconds=8.0,
                                  seed=seed)
    finally:
        env.close()
    lane = CudaHumanoidLane(envs)
    try:
        obs = lane.reset_policy(seed=seed, commands=np.full(envs, cmd))
        for _ in range(400):
            obs, _, done, _ = lane.step_policy(
                np.asarray(policy(obs), np.float32))
            if done.all():
                break
        proxy = lane.gate_proxy()
    finally:
        lane.close()
    return traces, proxy


def analyze(kind: str, tree_path: str | None, seeds: int, envs: int):
    policy = load_policy(kind, tree_path)
    all_raw, all_deb, per_ep = [], [], []
    proxy_total = np.zeros(3)          # qualified L, R, alt violations
    analyzer_q_raw = 0
    judged = []
    for cmd in hf.COMMANDS_MPS:
        for seed in SEED_BASE[:seeds]:
            traces, proxy = run_episodes(policy, cmd, seed, envs)
            proxy_total += [proxy["qualified_left"].sum(),
                            proxy["qualified_right"].sum(),
                            proxy["alternation_violations"].sum()]
            for e, tr in enumerate(traces):
                raw = swings_from_trace(tr, debounce=False)
                deb = swings_from_trace(tr, debounce=True)
                all_raw.extend(raw)
                all_deb.extend(deb)
                analyzer_q_raw += sum(s["qualified"] for s in raw)
                per_ep.append({
                    "cmd": cmd, "seed": seed, "env": e,
                    "alive_s": len(tr["ticks"]["time_s"]) * DT,
                    "swings_raw": len(raw), "swings_deb": len(deb),
                    "q_raw": sum(s["qualified"] for s in raw),
                    "proxy_q": int(proxy["qualified_left"][e]
                                   + proxy["qualified_right"][e]),
                })
                if not tr["terminated"] and tr["truncated_at_horizon"]:
                    judged.append(judge.evaluate_episode(tr))
    return {"kind": kind, "raw": all_raw, "deb": all_deb, "per_ep": per_ep,
            "proxy_total": proxy_total.tolist(),
            "analyzer_q_raw": analyzer_q_raw, "judged": judged}


def dist(values):
    if not values:
        return "n=0"
    v = np.asarray(values, float)
    return (f"n={len(v)} median={np.median(v):.4g} p90={np.percentile(v, 90):.4g} "
            f"max={v.max():.4g}")


def report(result: dict) -> dict:
    raw, deb, per_ep = result["raw"], result["deb"], result["per_ep"]
    print(f"\n===== {result['kind']} =====")
    alive = [e["alive_s"] for e in per_ep]
    print(f"episodes {len(per_ep)}, alive {dist(alive)} s")
    print(f"swings: raw {len(raw)}, debounced {len(deb)}")
    summary = {"kind": result["kind"], "episodes": len(per_ep),
               "swings_raw": len(raw), "swings_deb": len(deb)}
    for name, swings in (("RAW", raw), ("DEBOUNCED-20ms", deb)):
        if not swings:
            continue
        print(f"-- {name} swing stats (bars: duration>={judge.SWING_MIN_S}s, "
              f"clearance {judge.CLEARANCE_M*1000:.0f}mm held "
              f">={judge.CLEARANCE_MIN_S}s, placement "
              f">={judge.PLACEMENT_MIN_M}m)")
        print(f"   duration_s        {dist([s['duration_s'] for s in swings])}")
        print(f"   peak_clearance_m  {dist([s['peak_clearance_m'] for s in swings])}")
        print(f"   clearance_hold_s  {dist([s['clearance_hold_s'] for s in swings])}")
        print(f"   placement_m       {dist([s['placement_m'] for s in swings])}")
        ff = Counter(s["first_fail"] or "QUALIFIED" for s in swings)
        print(f"   first-fail histogram: {dict(ff)}")
        # per-clause pass rates (independent, not first-fail)
        rates = {
            "duration_ok": np.mean([judge.SWING_MIN_S <= s["duration_s"]
                                    <= judge.SWING_MAX_S for s in swings]),
            "clearance_ok": np.mean([s["clearance_hold_s"]
                                     >= judge.CLEARANCE_MIN_S - 1e-9
                                     for s in swings]),
            "placement_ok": np.mean([s["placement_m"]
                                     >= judge.PLACEMENT_MIN_M
                                     for s in swings]),
        }
        print(f"   independent pass rates: "
              + " ".join(f"{k}={v:.2%}" for k, v in rates.items()))
        summary[name] = {"first_fail": dict(ff), "rates": rates}
    pl, pr, alt = result["proxy_total"]
    q_deb = sum(s["qualified"] for s in deb)
    print(f"-- three-way: analyzer RAW qualified={result['analyzer_q_raw']}  "
          f"gate_proxy qualified={int(pl + pr)} (L{int(pl)}/R{int(pr)}, "
          f"alt_violations {int(alt)})  analyzer DEBOUNCED qualified={q_deb}")
    mism = [e for e in result["per_ep"] if e["q_raw"] != e["proxy_q"]]
    print(f"   per-episode analyzer-vs-proxy mismatches: {len(mism)}"
          + (f" (e.g. {mism[:3]})" if mism else ""))
    print(f"-- full-8s episodes judged: {len(result['judged'])}"
          + (f", passed {sum(j['passed'] for j in result['judged'])}"
             if result["judged"] else ""))
    summary["three_way"] = {"analyzer_raw_q": result["analyzer_q_raw"],
                            "proxy_q": int(pl + pr),
                            "analyzer_deb_q": q_deb,
                            "episode_mismatches": len(mism)}
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=str(
        ROOT / "runs/gpu/20260902-000009-humanoid-tree/artifacts/train/"
               "gpu-train-out/actor_final.pt"))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--json", default=None)
    ap.add_argument("--policies", default="demo,bc,tree")
    args = ap.parse_args()
    summaries = []
    for kind in args.policies.split(","):
        result = analyze(kind, args.tree, args.seeds, args.envs)
        summaries.append(report(result))
    if args.json:
        Path(args.json).write_text(json.dumps(summaries, indent=1,
                                              default=float))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
