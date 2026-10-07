"""PPO on any rl/ env (the env protocol in rl/common/skill.py), all on one device.

One implementation, two schedules the skills use:
  - kl="stop": fixed learning rate, an epoch loop that stops once the mean KL passes kl_target (manipulation skills);
  - kl="adaptive": the learning rate follows the KL toward kl_target (x1.5 / /1.5, in [1e-5, 1e-3]) (the duck).
Normalisers update either once per rollout on its whole batch (norm="batch") or every step (norm="step").
Timeouts can be bootstrapped (r + gamma V(s) where an episode ran out of time rather than ended).
The best checkpoint is chosen by the windowed success rate (score="success", at least min_episodes finished) or the
mean episode return (score="return"); a phase stops on its budget (minutes, env steps) or a plateau of the score.
"""
from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    horizon: int = 32
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatches: int = 4
    ent: float = 0.0
    vf_coef: float = 0.5
    mean_reg: float = 0.0
    max_grad_norm: float = 1.0
    kl: str = "stop"               # "stop" | "adaptive"
    kl_target: float = 0.03
    log_std_bounds: tuple = (-4.0, 0.0)
    norm: str = "batch"            # "batch" | "step"
    bootstrap_timeouts: bool = False
    score: str = "success"         # "success" | "return"
    min_episodes: int = 2000
    plateau_iters: int = 40        # 0: no plateau stop
    min_iters: int = 30
    max_minutes: float = 30.0
    max_steps: float = 3e9
    stats_window: int = 4000
    save_every: int = 10

    def update(self, over: dict | None):
        for k, v in (over or {}).items():
            if not hasattr(self, k):
                raise KeyError(f"unknown PPO setting {k!r}")
            setattr(self, k, tuple(v) if isinstance(getattr(self, k), tuple) else v)
        return self


class Stats:
    """Finished episodes in a window: success, return, length, end codes."""

    def __init__(self, code_names=None, window=4000, per_step_cap=None):
        self.code_names = code_names or {}
        self.succ, self.codes, self.ret, self.len, self.tout = (deque(maxlen=window) for _ in range(5))
        self.cap = per_step_cap

    def add(self, done, info, ep_ret, ep_len):
        idx = done.nonzero().squeeze(-1)
        if self.cap:
            idx = idx[: self.cap]
        if len(idx) == 0:
            return
        self.ret.extend(ep_ret[idx].tolist())
        self.len.extend(ep_len[idx].tolist())
        if "success" in info:
            self.succ.extend(info["success"][idx].tolist())
        if "code" in info:
            self.codes.extend(self.code_names.get(int(x), str(x)) for x in info["code"][idx].tolist())
        if "timeout" in info:
            self.tout.extend(info["timeout"][idx].tolist())

    def row(self):
        n = max(1, len(self.ret))
        r = dict(episodes=len(self.ret), ret=round(sum(self.ret) / n, 3), len=round(sum(self.len) / n, 1))
        if self.succ:
            r["success"] = round(sum(self.succ) / max(1, len(self.succ)), 4)
        if self.tout:
            r["survive"] = round(sum(self.tout) / max(1, len(self.tout)), 3)
        if self.codes:
            r["codes"] = dict(Counter(self.codes).most_common(6))
        return r


def ppo_phase(env, ac, actor_in, cfg: PPOConfig, out, phase, log, extra_row=None):
    """Train `ac` with PPO on env. actor_in(obs, priv) -> the actor's input. Saves <phase>.pt and <phase>_best.pt in
    `out`; returns a summary (steps, wall_s, final row, best score, why it stopped)."""
    dev = env.dev
    N, H = env.n, cfg.horizon
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr)
    obs, priv = env.observe()
    ep_ret = torch.zeros(N, device=dev)
    ep_len = torch.zeros(N, device=dev)
    stats = Stats(getattr(env, "code_names", None), cfg.stats_window, per_step_cap=256 if cfg.score == "return" else None)
    a_in = actor_in(obs, priv)
    buf = dict(ain=torch.zeros(H, N, a_in.shape[1], device=dev), cin=torch.zeros(H, N, obs.shape[1] + priv.shape[1], device=dev),
               act=torch.zeros(H, N, env.act_size, device=dev))
    buf.update({k: torch.zeros(H, N, device=dev) for k in ("logp", "val", "rew", "done", "tout")})
    t0, steps, it = time.time(), 0, 0
    best, hist = -math.inf, []
    while True:
        it += 1
        env.on_iteration(phase=phase, it=it, frac=(time.time() - t0) / max(1e-9, cfg.max_minutes * 60))
        with torch.no_grad():
            for t in range(H):
                x, c = actor_in(obs, priv), torch.cat([obs, priv], -1)
                if cfg.norm == "step":
                    ac.obs_norm.update(x)
                    ac.crit_norm.update(c)
                d = ac.dist(x)
                a = d.sample()
                buf["ain"][t], buf["cin"][t], buf["act"][t] = x, c, a
                buf["logp"][t], buf["val"][t] = d.log_prob(a).sum(-1), ac.value(c)
                obs, priv, r, done, info = env.step(a)
                ep_ret += r
                ep_len += 1
                stats.add(done, info, ep_ret, ep_len)
                buf["rew"][t], buf["done"][t] = r, done.float()
                buf["tout"][t] = info["timeout"].float() if "timeout" in info else 0.0
                ep_ret = torch.where(done, torch.zeros_like(ep_ret), ep_ret)
                ep_len = torch.where(done, torch.zeros_like(ep_len), ep_len)
            steps += N * H
            last_v = ac.value(torch.cat([obs, priv], -1))
            adv = torch.zeros(H, N, device=dev)
            gae = torch.zeros(N, device=dev)
            for t in reversed(range(H)):
                nv = last_v if t == H - 1 else buf["val"][t + 1]
                nd = 1 - buf["done"][t]
                rew = buf["rew"][t] + (cfg.gamma * buf["val"][t] * buf["tout"][t] if cfg.bootstrap_timeouts else 0.0)
                delta = rew + cfg.gamma * nv * nd - buf["val"][t]
                gae = delta + cfg.gamma * cfg.lam * nd * gae
                adv[t] = gae
            ret = adv + buf["val"]
        flat = lambda x: x.reshape(H * N, *x.shape[2:])
        X, C, A, LP, ADV, RET = flat(buf["ain"]), flat(buf["cin"]), flat(buf["act"]), flat(buf["logp"]), flat(adv), flat(ret)
        if cfg.norm == "batch":
            ac.obs_norm.update(X)
            ac.crit_norm.update(C)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        M = H * N
        mb = M // cfg.minibatches
        kls = []
        for ep in range(cfg.epochs):
            perm = torch.randperm(M, device=dev)
            for i in range(cfg.minibatches):
                j = perm[i * mb:(i + 1) * mb]
                mean = ac.mean(X[j])
                d = torch.distributions.Normal(mean, ac.log_std.exp().expand(mean.shape[0], -1))
                lp = d.log_prob(A[j]).sum(-1)
                ratio = torch.exp(lp - LP[j])
                pl = -torch.min(ratio * ADV[j], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ADV[j]).mean()
                vl = (ac.value(C[j]) - RET[j]).pow(2).mean()
                loss = pl + cfg.vf_coef * vl - cfg.ent * d.entropy().sum(-1).mean()
                if cfg.mean_reg:
                    loss = loss + cfg.mean_reg * (mean ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    kls.append((LP[j] - lp).mean().item())
            if cfg.kl == "stop" and sum(kls) / len(kls) > cfg.kl_target:
                break
            if cfg.kl == "adaptive":
                mk = sum(kls[-cfg.minibatches:]) / cfg.minibatches
                for g in opt.param_groups:
                    if mk > cfg.kl_target * 2:
                        g["lr"] = max(1e-5, g["lr"] / 1.5)
                    elif mk < cfg.kl_target / 2:
                        g["lr"] = min(1e-3, g["lr"] * 1.5)
        with torch.no_grad():
            ac.log_std.clamp_(*cfg.log_std_bounds)
        wall = time.time() - t0
        row = dict(phase=phase, it=it, steps=steps, wall_s=round(wall, 1), sps=round(steps / max(wall, 1e-9)), kl=round(sum(kls) / max(1, len(kls)), 4),
                   lr=opt.param_groups[0]["lr"], std=round(ac.log_std.exp().mean().item(), 3), **stats.row(), **env.iteration_info(), **(extra_row(it) if extra_row else {}))
        log(row)
        score = row.get("success", -math.inf) if cfg.score == "success" else (row["ret"] if row["episodes"] else -math.inf)
        enough = row["episodes"] >= min(cfg.min_episodes, N) if cfg.score == "success" else it > 20
        hist.append(score)
        ck = lambda: dict(model=ac.state_dict(), sizes=ac.sizes, row=row, steps=steps)
        if enough and env.best_allowed() and score > best:
            best = score
            torch.save(ck(), out / f"{phase}_best.pt")
        if it % cfg.save_every == 0:
            torch.save(ck(), out / f"{phase}.pt")
        W = cfg.plateau_iters // 2   # plateau: the last W iterations' mean score no better than the W before (+0.003)
        plateau = cfg.plateau_iters > 0 and it >= cfg.min_iters and env.best_allowed() and len(hist) >= 2 * W and sum(hist[-W:]) / W - sum(hist[-2 * W:-W]) / W < 0.003
        if plateau or wall > cfg.max_minutes * 60 or steps >= cfg.max_steps:
            torch.save(ck(), out / f"{phase}.pt")
            return dict(steps=steps, wall_s=round(wall, 1), final=row, best=best, stop="plateau" if plateau else "budget", config=asdict(cfg))


def best_or_last(out, phase):
    b = out / f"{phase}_best.pt"
    return b if b.exists() else out / f"{phase}.pt"
