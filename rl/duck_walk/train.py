"""PPO for the Open Duck walk on box3d-cuda (env.py): an asymmetric actor-critic. The actor (the exported student) sees only
the robot's sensors (IMU, servo encoders, its previous actions, the command: env.OBS_SIZE); the critic also sees privileged
state (trunk velocity, height, contacts, friction, mass, ...). Own implementation (torch only).

  python -m rl.duck_walk.train --envs 8192 --steps 300e6 --out runs/duck-walk [--device cuda] [--init ckpt.pt]
Writes runs/<out>/progress.jsonl, ckpt.pt (latest), best.pt (best mean walking score), config.json.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from .env import DuckWalkEnv, OBS_SIZE, LEGS


class RunningNorm(nn.Module):
    def __init__(self, n, clip=5.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n)); self.register_buffer("var", torch.ones(n)); self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x):
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d = b_mean - self.mean; tot = self.count + b_n
        self.mean += d * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + d ** 2 * self.count * b_n / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


def mlp(i, o, hidden=(256, 256, 128)):
    layers, n = [], i
    for h in hidden: layers += [nn.Linear(n, h), nn.ELU()]; n = h
    layers.append(nn.Linear(n, o))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs, priv, act):
        super().__init__()
        self.obs_norm, self.crit_norm = RunningNorm(obs), RunningNorm(obs + priv)
        self.actor = mlp(obs, act)
        self.critic = mlp(obs + priv, 1, (512, 256, 128))
        self.log_std = nn.Parameter(torch.full((act,), math.log(0.35)))

    def act_mean(self, obs):
        return self.actor(self.obs_norm(obs))

    def value(self, obs, priv):
        return self.critic(self.crit_norm(torch.cat([obs, priv], -1))).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=8192)
    ap.add_argument("--steps", type=float, default=300e6)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--out", default="runs/duck-walk")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--init", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--minutes", type=float, default=90)
    ap.add_argument("--dr", default="{}", help="JSON overrides of env.default_dr")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    env = DuckWalkEnv(a.envs, device=a.device, seed=a.seed, dr=json.loads(a.dr), substeps=a.substeps, iterations=a.iterations)
    ac = ActorCritic(OBS_SIZE, env.PRIV_SIZE, LEGS).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=a.lr)
    if a.init:
        ck = torch.load(a.init, map_location=dev); ac.load_state_dict(ck["model"]); print(json.dumps({"event": "init", "from": a.init}))
    cfg = dict(vars(a), dr_full=env.dr, obs=OBS_SIZE, priv=env.PRIV_SIZE, act=LEGS, device_name=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu")
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps({"event": "start", **{k: v for k, v in cfg.items() if k != "dr_full"}}), flush=True)
    E, Hn = a.envs, a.horizon
    gamma, lam, clip, epochs, minibatches = 0.99, 0.95, 0.2, 5, 4
    obs = env.obs(); priv = env.priv()
    buf = {k: torch.zeros(Hn, E, *s, device=dev) for k, s in dict(obs=(OBS_SIZE,), priv=(env.PRIV_SIZE,), act=(LEGS,), logp=(), val=(), rew=(), done=(), tout=()).items()}
    steps, it, t0, best = 0, 0, time.time(), -1e9
    ep_stats = []
    desired_kl = 0.01
    while steps < a.steps and (time.time() - t0) < a.minutes * 60:
        it += 1
        with torch.no_grad():
            for t in range(Hn):
                ac.obs_norm.update(obs); ac.crit_norm.update(torch.cat([obs, priv], -1))
                mean = ac.act_mean(obs); std = ac.log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                act = dist.sample()
                buf["obs"][t] = obs; buf["priv"][t] = priv; buf["act"][t] = act
                buf["logp"][t] = dist.log_prob(act).sum(-1); buf["val"][t] = ac.value(obs, priv)
                obs2, priv2, rew, done, info = env.step(act)
                buf["rew"][t] = rew; buf["done"][t] = done.float(); buf["tout"][t] = info["timeout"].float()
                if done.any():
                    idx = done.nonzero().flatten()
                    for i in idx[:256].tolist():
                        ep_stats.append((info["ep_ret"][i].item(), info["ep_len"][i].item(), bool(info["timeout"][i]), env.cmd[i, 0].item()))
                    env.reset(done)
                    obs2 = env.obs(); priv2 = env.priv()
                obs, priv = obs2, priv2
            last_val = ac.value(obs, priv)
            # GAE with bootstrapping through timeouts
            adv = torch.zeros(E, device=dev); advs = torch.zeros(Hn, E, device=dev)
            for t in reversed(range(Hn)):
                nv = last_val if t == Hn - 1 else buf["val"][t + 1]
                r = buf["rew"][t] + gamma * buf["val"][t] * buf["tout"][t]
                nonterm = 1.0 - buf["done"][t]
                delta = r + gamma * nv * nonterm - buf["val"][t]
                adv = delta + gamma * lam * nonterm * adv
                advs[t] = adv
            rets = advs + buf["val"]
        flat = lambda x: x.reshape(Hn * E, *x.shape[2:])
        B = {k: flat(v) for k, v in buf.items()}; A = flat(advs); R = flat(rets)
        A = (A - A.mean()) / (A.std() + 1e-8)
        n = Hn * E; mb = n // minibatches
        kls = []
        for ep in range(epochs):
            perm = torch.randperm(n, device=dev)
            for k in range(minibatches):
                j = perm[k * mb:(k + 1) * mb]
                mean = ac.act_mean(B["obs"][j]); std = ac.log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(B["act"][j]).sum(-1)
                ratio = torch.exp(logp - B["logp"][j])
                pg = -torch.min(ratio * A[j], ratio.clamp(1 - clip, 1 + clip) * A[j]).mean()
                v = ac.value(B["obs"][j], B["priv"][j])
                vl = ((v - R[j]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg + 1.0 * vl - float(__import__("os").environ.get("DUCK_ENT", "0.006")) * ent + 0.001 * (mean ** 2).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ac.parameters(), 1.0); opt.step()
                with torch.no_grad():
                    kl = (B["logp"][j] - logp).mean().item(); kls.append(kl)
            # adaptive learning rate on the KL
            mk = sum(kls[-minibatches:]) / minibatches
            for g in opt.param_groups:
                if mk > desired_kl * 2: g["lr"] = max(1e-5, g["lr"] / 1.5)
                elif mk < desired_kl / 2: g["lr"] = min(1e-3, g["lr"] * 1.5)
        with torch.no_grad(): ac.log_std.clamp_(math.log(0.15), math.log(1.0))
        steps += n
        recent = ep_stats[-2000:]
        if recent:
            walkers = [e for e in recent if e[3] > 0.05]
            mean_ret = sum(e[0] for e in recent) / len(recent); mean_len = sum(e[1] for e in recent) / len(recent)
            surv = sum(e[2] for e in recent) / len(recent)
        else: mean_ret = mean_len = surv = 0.0
        T = env.trunk()
        row = dict(it=it, steps=steps, wall_s=round(time.time() - t0, 1), sps=int(steps / (time.time() - t0)), ret=round(mean_ret, 3), len=round(mean_len, 1), survive=round(surv, 3),
                   vx=round(T["vx"].mean().item(), 3), cmd=round(env.cmd[:, 0].mean().item(), 3), kl=round(sum(kls) / len(kls), 4), lr=opt.param_groups[0]["lr"], std=round(ac.log_std.exp().mean().item(), 3))
        if it % 5 == 1 or it < 5:
            # terms (last step) for diagnosis
            row["terms"] = {k: round(v.mean().item(), 3) for k, v in info["terms"].items()}
        print(json.dumps(row), flush=True)
        with open(out / "progress.jsonl", "a") as f: f.write(json.dumps(row) + "\n")
        if it % 20 == 0 or steps >= a.steps:
            torch.save({"model": ac.state_dict(), "steps": steps, "cfg": cfg}, out / "ckpt.pt")
        score = mean_ret if recent else -1e9
        if it > 20 and score > best:
            best = score; torch.save({"model": ac.state_dict(), "steps": steps, "cfg": cfg, "score": score}, out / "best.pt")
    torch.save({"model": ac.state_dict(), "steps": steps, "cfg": cfg}, out / "ckpt.pt")
    print(json.dumps({"event": "done", "steps": steps, "wall_s": round(time.time() - t0, 1), "best": best}), flush=True)


if __name__ == "__main__":
    main()
