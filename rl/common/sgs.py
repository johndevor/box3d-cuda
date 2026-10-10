"""Success-Guided Sampling (SGS): an opt-in reset sampler for any rl/ env whose reset draws its episode through
`_sample(idx) -> {name: tensor[m, ...]}` (the task configuration: initial state, goal, environment parameters).

Implemented from the description in "A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for
Robot Control" (arXiv 2610.12465, sgs-rl.github.io; no code released, none copied), sections 3.2 and B.2:
  - before training, a fixed set of `configs` task configurations is drawn from the task distribution (here: the env's
    own `_sample` on a separate generator, so the bank does not depend on the training seed's stream);
  - per configuration, the last `window` Boolean outcomes in a circular buffer; p_i = their mean;
  - a Beta-shaped kernel with mode `target` (t) and concentration `kappa`:
        w_i = (p_i + eps)^(kappa t) (1 - p_i + eps)^(kappa (1 - t)),  w_i <- max(w_i, floor),  l_i = log(w_i + eps)
  - at each episode reset the configuration is drawn i.i.d. from softmax(l_i / T_eff), T_eff = max(temperature, 1).
  Paper settings for manipulation: N = 32768, H = 100, T = 2, t = 0.5, kappa = 1, eps = 1e-4.
Choices of ours (the paper does not state them): a configuration not yet seen counts as p = t (the kernel's mode) so
every configuration is visited early; `floor` defaults to eps; a `uniform_frac` of the worlds (the first ones) draw
their configuration uniformly from the bank, and their episodes are the PPO phase's score (best checkpoint, plateau),
since the SGS-weighted success rate is pulled toward t by design and says nothing about the task distribution.

Not part of a configuration: what the env draws outside `_sample` (per-iteration friction groups, sensor noise and
its per-episode gain/tare draws); those stay as random as without SGS.

Opt in from a recipe: R["sgs"] = {"enabled": true, "configs": 4096, "window": 50, "floor": 1e-4, ...}
(python -m rl.train <skill> --set sgs.enabled=true --set sgs.configs=4096). Off by default.
"""
from __future__ import annotations

import math

import torch

DEFAULTS = dict(enabled=False, configs=4096, window=100, floor=1e-4, eps=1e-4, target=0.5, kappa=1.0, temperature=2.0,
                uniform_frac=0.125, seed=7919)


def settings(over: dict | None) -> dict:
    s = dict(DEFAULTS)
    for k, v in (over or {}).items():
        if k not in s:
            raise KeyError(f"unknown SGS setting {k!r} (known: {', '.join(DEFAULTS)})")
        s[k] = v
    return s


class SuccessGuidedSampler:
    """The outcome table and the sampling distribution over K configurations (all on one device, no host syncs)."""

    def __init__(self, K, window, device, floor=1e-4, eps=1e-4, target=0.5, kappa=1.0, temperature=2.0):
        self.K, self.H, self.dev = int(K), int(window), device
        self.floor, self.eps, self.t, self.kappa, self.T = floor, eps, target, kappa, max(1.0, temperature)
        self.buf = torch.zeros(self.K, self.H, device=device)
        self.ptr = torch.zeros(self.K, dtype=torch.long, device=device)
        self.count = torch.zeros(self.K, dtype=torch.long, device=device)

    def success(self):
        """p_i: the mean of the last `window` outcomes (the kernel's mode for a configuration not yet seen)."""
        n = self.count.clamp(max=self.H)
        return torch.where(n > 0, self.buf.sum(1) / n.clamp(min=1), torch.full_like(self.buf[:, 0], self.t))

    def probs(self):
        p, e = self.success(), self.eps
        w = (p + e) ** (self.kappa * self.t) * (1 - p + e) ** (self.kappa * (1 - self.t))
        logit = torch.log(torch.clamp(w, min=self.floor) + e)
        return torch.softmax(logit / self.T, 0)

    def draw(self, m, g):
        return torch.multinomial(self.probs(), m, replacement=True, generator=g)

    def record(self, ids, outcome):
        """ids[m] (configurations, -1 = none), outcome[m] (bool): appended to each configuration's circular buffer;
        several outcomes of one configuration in one call take consecutive slots."""
        keep = ids >= 0
        ids, outcome = ids[keep], outcome[keep].float()
        if ids.numel() == 0:
            return
        c, order = torch.sort(ids)
        o = outcome[order]
        pos = torch.arange(len(c), device=c.device)
        rank = pos - torch.searchsorted(c, c)                 # (the position within its configuration's group)
        slot = (self.ptr[c] + rank) % self.H
        self.buf[c, slot] = o
        n = torch.bincount(c, minlength=self.K)
        self.ptr.add_(n)
        self.count.add_(n)

    def info(self):
        p, seen = self.success(), self.count > 0
        q = self.probs()
        ps = p[seen]
        return dict(sgs_seen=round(seen.float().mean().item(), 3),
                    sgs_p_mean=round(ps.mean().item(), 4) if ps.numel() else None,
                    sgs_zone=round(((ps > 0.1) & (ps < 0.9)).float().mean().item(), 3) if ps.numel() else None,
                    sgs_solved=round((ps >= 0.9).float().mean().item(), 3) if ps.numel() else None,
                    sgs_ess=round((1.0 / (q ** 2).sum()).item() / self.K, 3))


def attach(env, over: dict | None, log=None):
    """Turn SGS on for this env (in place): its resets draw from a fixed bank of configurations weighted by the sampler;
    its steps record each finished episode's success against the configuration it ran. Returns the sampler."""
    s = settings(over)
    if not hasattr(env, "_sample"):
        raise SystemExit(f"SGS: {type(env).__name__} has no _sample(idx) (the per-episode configuration draw SGS replaces)")
    dev, n, K = env.dev, env.n, int(s["configs"])
    # the bank: the env's own draw for K worlds, on its own generator (then the env's generator is put back)
    g_train = env.g
    gb = torch.Generator(device=dev)
    gb.manual_seed(int(s["seed"]))
    object.__setattr__(env, "g", gb)
    with torch.no_grad():
        bank = {k: v.clone() for k, v in env._sample(torch.arange(K, device=dev)).items()}
    object.__setattr__(env, "g", g_train)
    sampler = SuccessGuidedSampler(K, s["window"], dev, s["floor"], s["eps"], s["target"], s["kappa"], s["temperature"])
    ids = torch.full((n,), -1, dtype=torch.long, device=dev)
    n_uni = int(round(n * float(s["uniform_frac"])))
    uniform = torch.arange(n, device=dev) < n_uni
    orig_sample, orig_step, orig_info = env._sample, env.step, env.iteration_info

    def _sample(idx):
        m = len(idx)
        pick = sampler.draw(m, env.g)
        if n_uni:
            u = torch.randint(0, K, (m,), generator=env.g, device=dev)
            pick = torch.where(uniform[idx], u, pick)
        ids[idx] = pick
        return {k: v[pick] for k, v in bank.items()}

    def step(action, autoreset=True):
        prev = ids.clone()
        out = orig_step(action, autoreset)
        done, info = out[3], out[4]
        if "success" in info:
            sampler.record(torch.where(done, prev, torch.full_like(prev, -1)), info["success"])
        return out

    def iteration_info():
        return {**orig_info(), **sampler.info()}

    object.__setattr__(env, "_sample", _sample)
    object.__setattr__(env, "step", step)
    object.__setattr__(env, "iteration_info", iteration_info)
    object.__setattr__(env, "score_mask", uniform if n_uni else None)
    object.__setattr__(env, "sgs", sampler)
    env.reset(torch.ones(n, dtype=torch.bool, device=dev))          # (every world starts on a bank configuration)
    if log:
        log(dict(event="sgs", **{k: v for k, v in s.items()}, uniform_worlds=n_uni, bank_keys=len(bank)))
    return sampler
