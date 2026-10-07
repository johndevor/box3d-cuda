"""The two-step chain in box3d-cuda: step 1 places the housing (place.py; end states snapshotted), step 2 mates the plug
into the placed housing with the socket's true pose restored from those snapshots (env cfg chain_states). Reports the
chained yield for the base student (trained on independent pose errors) and for the student fine-tuned on step 1's end
states (PPO, the privileged critic kept).
  python -m rl.skills.connector_mate.chain --run runs/connector_mate --out runs/chain [--finetune-min 10]
"""
import argparse
import json
import time
from pathlib import Path

import torch

from rl.common import models
from rl.common.evaluate import evaluate, model_act
from rl.common.ppo import PPOConfig, best_or_last, ppo_phase
from . import place
from .env import ConnectorMateBatch
from .skill import SKILL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/connector_mate")
    ap.add_argument("--out", default="runs/chain")
    ap.add_argument("--n", type=int, default=32768)
    ap.add_argument("--eval-n", type=int, default=8192)
    ap.add_argument("--finetune-min", type=float, default=10)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda row: print(json.dumps(row), flush=True)
    # ---- step 1: place, snapshot
    t0 = time.time()
    r = place.run(a.n, a.device, seed=7)
    states = out / "place_states.pt"
    torch.save(r, states)
    ok = r["ok"]
    q = lambda t: [round(float(x), 3) for x in torch.quantile(t.float(), torch.tensor([0.5, 0.9, 0.99]))]
    s1 = dict(n=a.n, success=round(ok.float().mean().item(), 4), off_mm=q(r["off_mm"][ok].norm(dim=-1)), yaw_deg=q(torch.rad2deg(r["yaw_rad"][ok]).abs()),
              tilt_fail=int((r["tilt_deg"] >= 1.0).sum()), wall_s=round(time.time() - t0, 1))
    log(dict(event="step1", **s1))
    cfg = dict(chain_states=str(states), chain_sigma_mm=0.3, physical_p=0.0)
    # ---- step 2 with the base student (the run's policy.pt: actor + the teacher's privileged critic)
    ck = torch.load(Path(a.run) / "student_best.pt", map_location=a.device)
    student = models.ActorCritic(**{k: v for k, v in ck["sizes"].items() if k not in ("actor_in", "critic_in", "act")},
                                 actor_in=ck["sizes"]["actor_in"], critic_in=ck["sizes"]["critic_in"], act=ck["sizes"]["act"]).to(a.device)
    student.load_state_dict(ck["model"])
    s_in = lambda o, p: o
    ev = lambda c: evaluate(SKILL, model_act(student, s_in), c, a.eval_n, 40_000, a.device)
    base = ev(cfg)
    log(dict(event="step2_base", **base))
    base_indep = ev(dict(physical_p=0.0))
    log(dict(event="step2_base_independent_errors", **base_indep))
    # ---- fine-tune on step 1's end states
    student.obs_norm.frozen = True
    env = ConnectorMateBatch(a.n, device=a.device, seed=11, cfg=cfg)
    with torch.no_grad():
        student.log_std.fill_(-1.8)
        student.log_std[8] = -2.5
    pc = PPOConfig(horizon=32, lr=1e-4, min_iters=20, max_minutes=a.finetune_min, min_episodes=2000)
    ft = ppo_phase(env, student, s_in, pc, out, "chain", log)
    student.load_state_dict(torch.load(best_or_last(out, "chain"), map_location=a.device)["model"])
    tuned = ev(cfg)
    log(dict(event="step2_chain_tuned", **tuned))
    torch.save(dict(model=student.state_dict(), sizes=student.sizes), out / "student.pt")
    res = dict(step1=s1, step2_base=base, step2_base_independent=base_indep, step2_tuned=tuned, finetune=dict(steps=ft["steps"], wall_s=ft["wall_s"], stop=ft["stop"]),
               chained_base=round(s1["success"] * base["success"], 4), chained_tuned=round(s1["success"] * tuned["success"], 4), cfg=cfg)
    (out / "chain.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "eval.json").write_text(json.dumps(dict(cfg=cfg, finetune=res["finetune"], eval_nominal=tuned), default=str))
    log(dict(event="done", chained_base=res["chained_base"], chained_tuned=res["chained_tuned"]))


if __name__ == "__main__":
    main()
