"""Throughput benchmark for the multiprocess vec env + PPO update.

  .venv/bin/python -B -m walk.train.bench
  .venv/bin/python -B -m walk.train.bench --sim-us 200   # simulate physics cost

Measures, for each (--workers x --total-envs) combination:
  - total env steps/s through VecEnv (random actions, no policy in the loop)
  - per-worker steps/s and scaling efficiency vs the 1-worker row
and, once per total-envs value, the PPO update time for a horizon-64 batch.

StubEnv steps in ~1 us/env, far cheaper than the real duck env (10 native
ticks/policy step). --sim-us adds a busy-wait of that many microseconds per
env per step inside each worker to model real physics cost; use it to judge
scaling at realistic step costs.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from walk.env.contract import StubEnv
from walk.train.ppo import PPOConfig, make_nets, ppo_update, tanh_gaussian_log_prob
from walk.train.vec import VecEnv


class LoadedStubEnv(StubEnv):
    """StubEnv plus a busy-wait of sim_us microseconds per env per step."""

    def __init__(self, environments: int = 16, seed: int = 0, sim_us: float = 0.0, **kw):
        self._sim_us = float(sim_us)
        super().__init__(environments=environments, seed=seed, **kw)

    def step(self, action):
        if self._sim_us > 0:
            t_end = time.perf_counter() + self._sim_us * 1e-6 * self.E
            while time.perf_counter() < t_end:
                pass
        return super().step(action)


def bench_env_steps(workers: int, total_envs: int, steps: int, sim_us: float, seed: int = 0) -> float:
    if sim_us > 0:
        spec, kwargs = "walk.train.bench:LoadedStubEnv", {"sim_us": sim_us}
    else:
        spec, kwargs = "walk.env.contract:StubEnv", {}
    vec = VecEnv(spec, kwargs, workers=workers, total_envs=total_envs, seed=seed)
    try:
        vec.reset(seed)
        rng = np.random.default_rng(seed)
        actions = rng.uniform(-1, 1, (total_envs, vec.act_dim)).astype(np.float32)
        for _ in range(max(5, steps // 10)):  # warmup
            vec.step(actions)
        t0 = time.perf_counter()
        for _ in range(steps):
            vec.step(actions)
        dt = time.perf_counter() - t0
    finally:
        vec.close()
    return total_envs * steps / dt


def bench_ppo_update(total_envs: int, horizon: int, obs_dim: int = 58, act_dim: int = 14,
                     repeats: int = 3, seed: int = 0) -> float:
    torch.manual_seed(seed)
    cfg = PPOConfig()
    actor, critic = make_nets(obs_dim, act_dim, cfg)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=cfg.lr)
    gen = torch.Generator()
    gen.manual_seed(seed)
    B = total_envs * horizon
    obs = torch.randn(B, obs_dim)
    with torch.no_grad():
        mu, std = actor.dist(obs)
        raw = mu + std * torch.randn(mu.shape, generator=gen)
        logp = tanh_gaussian_log_prob(mu, std, raw)
        val = critic(obs)
    batch = {"obs": obs, "raw_act": raw, "logp": logp, "val": val,
             "adv": torch.randn(B, generator=gen), "ret": val + 0.1 * torch.randn(B, generator=gen)}
    ppo_update(actor, critic, opt, batch, cfg, gen)  # warmup
    t0 = time.perf_counter()
    for _ in range(repeats):
        ppo_update(actor, critic, opt, batch, cfg, gen)
    return (time.perf_counter() - t0) / repeats


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="walk.train.bench")
    p.add_argument("--workers", type=int, nargs="+", default=[1, 6, 12, 16])
    p.add_argument("--total-envs", type=int, nargs="+", default=[96, 192, 384])
    p.add_argument("--steps", type=int, default=200, help="timed vec steps per combo")
    p.add_argument("--horizon", type=int, default=64, help="for the PPO update timing")
    p.add_argument("--sim-us", type=float, default=0.0, help="busy-wait us per env per step in workers")
    p.add_argument("--torch-threads", type=int, default=4)
    args = p.parse_args(argv)

    torch.set_num_threads(args.torch_threads)
    print(f"# StubEnv bench: sim_us={args.sim_us}, steps={args.steps}, horizon={args.horizon}, "
          f"torch_threads={args.torch_threads}")
    upd = {n: bench_ppo_update(n, args.horizon) for n in args.total_envs}

    hdr = f"{'workers':>7} {'envs':>5} {'steps/s':>10} {'steps/s/wkr':>11} {'scale-eff':>9} {'ppo_upd_s':>9} {'roll+upd_s':>10}"
    print(hdr)
    print("-" * len(hdr))
    base: dict[int, float] = {}
    for n in args.total_envs:
        for w in args.workers:
            if n % w != 0:
                print(f"{w:>7} {n:>5} {'skip (envs % workers != 0)':>32}")
                continue
            sps = bench_env_steps(w, n, args.steps, args.sim_us)
            if w == min(args.workers):
                base[n] = sps / min(args.workers)
            eff = sps / (w * base[n]) if n in base else float("nan")
            rollupd = n * args.horizon / sps + upd[n]
            print(f"{w:>7} {n:>5} {sps:>10.0f} {sps / w:>11.0f} {eff:>9.2f} {upd[n]:>9.3f} {rollupd:>10.3f}")


if __name__ == "__main__":
    main()
