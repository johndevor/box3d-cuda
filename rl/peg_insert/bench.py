"""Build check + throughput of PegInsertBatch on the GPU, and a scripted-policy sanity run.
  python -m rl.peg_insert.bench [--sizes 4096,16384,32768]
"""
import argparse
import json
import time

import torch

from .env import CODE_NAMES, PegInsertBatch


def scripted(o):
    a = torch.zeros(o.shape[0], 9, device=o.device)
    a[:, 0:3] = (o[:, 0:3] / 0.001).clamp(-1, 1) * 0.5
    a[:, 2] = 0.6
    a[:, 3:6] = (o[:, 3:6] / 0.01).clamp(-1, 1) * 0.5
    a[:, 8] = -1
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="4096,16384,32768")
    ap.add_argument("--ticks", type=int, default=60)
    a = ap.parse_args()
    t0 = time.time()
    env = PegInsertBatch(64, device="cuda", seed=0)
    print(json.dumps(dict(event="built", s=round(time.time() - t0, 1), gpu=torch.cuda.get_device_name(0))), flush=True)
    for n in map(int, a.sizes.split(",")):
        env = PegInsertBatch(n, device="cuda", seed=1)
        o = env.obs(update=False)
        for _ in range(3):
            o, r, d, i = env.step(scripted(o))
        torch.cuda.synchronize()
        t = time.time()
        succ = fin = 0
        codes = {}
        for _ in range(a.ticks):
            o, r, d, info = env.step(scripted(o))
            succ += int(info["success"].sum())
            fin += int(d.sum())
            for c in info["code"][d].tolist():
                codes[CODE_NAMES[c]] = codes.get(CODE_NAMES[c], 0) + 1
            if d.any():
                env.reset(d)
                o = torch.where(d[:, None], env.obs(update=False), o)
        torch.cuda.synchronize()
        dt = time.time() - t
        print(json.dumps(dict(event="throughput", envs=n, env_steps_per_s=round(n * a.ticks / dt), substeps_per_s=round(n * a.ticks * 4 * env.sub / dt),
                              scripted_success=round(succ / max(1, fin), 3), episodes=fin, codes=codes,
                              mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))), flush=True)


if __name__ == "__main__":
    main()
