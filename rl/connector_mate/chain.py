"""The two-step chain in box3d-cuda: step 1 places the housing (place.py; end states snapshotted), step 2 mates the plug
into the placed housing with the socket's true pose restored from those snapshots (env cfg chain_states). Reports the
chained yield for the base student (trained on independent pose errors) and for the student fine-tuned on step 1's end
states (PPO, the privileged critic kept).
  python -m rl.connector_mate.chain --run runs/conn --out runs/chain [--finetune-min 10]
"""
import argparse
import json
import time
from pathlib import Path

import torch

from rl.peg_insert.env import OBS_SIZE
from . import place
from .env import PRIV_SIZE
from .train import Actor, Critic, evaluate, ppo_phase
from .env import ConnectorMateBatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/conn")
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
    # ---- step 2 with the base student
    ck = torch.load(Path(a.run) / "student_best.pt", map_location=a.device)
    student = Actor(OBS_SIZE).to(a.device)
    student.load_state_dict(ck["actor"])
    s_in = lambda o, p: o
    base = evaluate(cfg, student, s_in, a.eval_n, 40_000, a.device)
    log(dict(event="step2_base", **base))
    base_indep = evaluate(dict(physical_p=0.0), student, s_in, a.eval_n, 40_000, a.device)
    log(dict(event="step2_base_independent_errors", **base_indep))
    # ---- fine-tune on step 1's end states
    critic = Critic(OBS_SIZE + PRIV_SIZE).to(a.device)
    critic.load_state_dict(ck["critic"])
    student.norm.frozen = True
    env = ConnectorMateBatch(a.n, device=a.device, seed=11, cfg=cfg)

    class A:  # the trainer's arguments
        horizon, lr, gamma, lam, clip, ent, epochs, minibatches, plateau_iters, max_steps = 32, 1e-4, 0.99, 0.95, 0.2, 0.0, 4, 4, 40, 3e9
    with torch.no_grad():
        student.log_std.fill_(-1.8)
        student.log_std[8] = -2.5
    ft = ppo_phase(env, student, critic, s_in, A, out, "chain", a.finetune_min, log, min_iters=20)
    best = out / "chain_best.pt"
    student.load_state_dict(torch.load(best if best.exists() else out / "chain.pt", map_location=a.device)["actor"])
    tuned = evaluate(cfg, student, s_in, a.eval_n, 40_000, a.device)
    log(dict(event="step2_chain_tuned", **tuned))
    torch.save(dict(actor=student.state_dict(), obs_size=OBS_SIZE), out / "student.pt")
    res = dict(step1=s1, step2_base=base, step2_base_independent=base_indep, step2_tuned=tuned, finetune=dict(steps=ft["steps"], wall_s=ft["wall_s"], stop=ft["stop"]),
               chained_base=round(s1["success"] * base["success"], 4), chained_tuned=round(s1["success"] * tuned["success"], 4), cfg=cfg)
    (out / "chain.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "eval.json").write_text(json.dumps(dict(cfg=cfg, finetune=res["finetune"], eval_nominal=tuned), default=str))
    log(dict(event="done", chained_base=res["chained_base"], chained_tuned=res["chained_tuned"]))


if __name__ == "__main__":
    main()
