"""Evaluate any contract ONNX insert policy (57 observations) in the box3d connector env: one episode per world, deterministic.
  python -m rl.connector_mate.eval_onnx <skill dir> [--n 256 --seed 30000 --pose-scale 1 --device cpu]
"""
import argparse
import json
from collections import Counter

import numpy as np
import onnxruntime as ort
import torch

from .env import CODE_NAMES, ConnectorMateBatch as PegInsertBatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("skill")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=30000)
    ap.add_argument("--pose-scale", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cfg", default="{}")
    a = ap.parse_args()
    sess = ort.InferenceSession(f"{a.skill}/policy.onnx")
    cfg = dict(json.loads(a.cfg), pose_scale=a.pose_scale)
    env = PegInsertBatch(a.n, device=a.device, seed=a.seed, cfg=cfg)
    obs = env.obs(update=False)
    fin = torch.zeros(a.n, dtype=torch.bool)
    succ = torch.zeros(a.n, dtype=torch.bool)
    codes = torch.zeros(a.n, dtype=torch.long)
    for _ in range(200):
        act = torch.tensor(sess.run(["action"], {"obs": obs.cpu().numpy().astype(np.float32)})[0], device=env.dev)
        obs, r, done, info = env.step(act)
        done, s, c = done.cpu(), info["success"].cpu(), info["code"].cpu()
        new = done & ~fin
        succ |= new & s
        codes = torch.where(new, c, codes)
        fin |= done
        if fin.all():
            break
    print(json.dumps(dict(skill=a.skill, n=a.n, pose_scale=a.pose_scale, success=round(succ.float().mean().item(), 4),
                          codes=dict(Counter(CODE_NAMES.get(int(x), "unfinished") for x in codes.tolist()).most_common()))))


if __name__ == "__main__":
    main()
