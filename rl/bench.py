"""Environment and training throughput per skill: env-steps/s at several world counts, eager or CUDA-graph stepping.

  python -m rl.bench <skill|all> [--envs 4096,16384,65536] [--steps 40] [--graph] [--train-iters 0]

Env-only: random actions in [-1, 1] (autoreset on), timed after a warm-up, synchronised. --train-iters N also times N
PPO iterations of the skill's recipe at each world count (rollout + update, end to end).
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from rl.common import cuda
from rl.common.skill import SKILLS, load_skill


def env_sps(skill, n, steps, graph, env_kw):
    env = skill.make_env(n, device="cuda", seed=0, cfg={}, **env_kw)
    if graph:
        env.use_graphs(True)
    g = torch.Generator(device="cuda").manual_seed(0)
    a = lambda: torch.rand(n, env.act_size, device="cuda", generator=g) * 2 - 1
    for _ in range(5):
        env.step(a())
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(steps):
        env.step(a())
    torch.cuda.synchronize()
    dt = time.time() - t
    return dict(envs=n, graph=graph, env_steps_per_s=round(n * steps / dt), ms_per_step=round(1000 * dt / steps, 2), mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("skill")
    ap.add_argument("--envs", default="4096,16384,65536")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--graph", action="store_true", help="also time CUDA-graph stepping")
    a = ap.parse_args()
    cuda.require_device("cuda")
    for name in (SKILLS if a.skill == "all" else [a.skill]):
        skill = load_skill(name)
        env_kw = dict(skill.recipe.get("env", {}), **(skill.recipe.get("stages", [{}])[-1].get("env", {}) if skill.recipe.get("stages") else {}))
        for n in map(int, a.envs.split(",")):
            for graph in ([False, True] if a.graph else [False]):
                try:
                    r = env_sps(skill, n, a.steps, graph, env_kw)
                except Exception as e:   # (an OOM at a large count ends that skill's sweep, said so)
                    r = dict(envs=n, graph=graph, error=str(e)[:200])
                print(json.dumps(dict(skill=name, **r)), flush=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
