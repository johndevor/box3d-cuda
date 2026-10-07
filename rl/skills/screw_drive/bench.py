"""Build check, throughput and a scripted-policy sanity run of ScrewDriveBatch.
  python -m rl.skills.screw_drive.bench [--sizes 4096,16384,32768] [--device cpu --sizes 2 --ticks 40]
The scripted policy is the World2 scripted path's logic: aim at the believed seat and axis, push, run forward; no
back-out (a cross-thread runs into the clutch); slows to 100 rpm two pitches before the believed seat.
"""
import argparse
import json
import time
from collections import Counter

import torch

from .env import CODE_NAMES, ScrewDriveBatch


def scripted(o):
    a = torch.zeros(o.shape[0], 7, device=o.device)
    a[:, 0:2] = (o[:, 0:2] / 0.0005).clamp(-1, 1) * 0.5
    a[:, 2:4] = (o[:, 3:5] / 0.004).clamp(-1, 1)
    a[:, 4] = 0.5
    a[:, 5] = torch.where(o[:, 2] > 2 * o[:, 29], 1.0, 0.25)   # slow (100 rpm) for the last two pitches before the seat
    a[:, 6] = -1.0
    return a


def run(env, ticks):
    obs = env.obs(update=False)
    n = env.n
    fin = torch.zeros(n, dtype=torch.bool, device=env.dev)
    codes = torch.zeros(n, dtype=torch.long, device=env.dev)
    for _ in range(ticks):
        obs, _, r, done, info = env.step(scripted(obs), autoreset=False)
        new = done & ~fin
        codes = torch.where(new, info["code"], codes)
        fin |= done
        if fin.all():
            break
    return Counter(CODE_NAMES.get(int(c), "unfinished") for c in codes.tolist()), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="4096,16384,32768")
    ap.add_argument("--ticks", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    t0 = time.time()
    for n in map(int, a.sizes.split(",")):
        env = ScrewDriveBatch(n, device=a.device, seed=0)
        if a.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        codes, info = run(env, a.ticks)
        if a.device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t1
        print(json.dumps(dict(n=n, build_s=round(t1 - t0, 1), ticks=a.ticks, s=round(dt, 2), env_steps_per_s=round(n * a.ticks / dt),
                              scripted_codes=dict(codes.most_common()))), flush=True)


if __name__ == "__main__":
    main()
