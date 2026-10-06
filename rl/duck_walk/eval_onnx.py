"""Run an exported walk policy (policy.onnx) in the batched duck env (CPU or CUDA build) and report what World2's judge
measures: forward distance in a window, falls, sole lifts over 4 mm per foot.
  python -m rl.duck_walk.eval_onnx <policy dir> [--envs 16] [--seconds 10] [--dr off|on] [--vx 0.15] [--dump obs.json]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .env import DuckWalkEnv, RATE_HZ

ap = argparse.ArgumentParser()
ap.add_argument("policy"); ap.add_argument("--envs", type=int, default=16); ap.add_argument("--seconds", type=float, default=10)
ap.add_argument("--dr", default="off"); ap.add_argument("--vx", type=float, default=0.15); ap.add_argument("--dump", default=None)
ap.add_argument("--device", default="cpu"); ap.add_argument("--settle", type=float, default=1.0)
a = ap.parse_args()
sess = ort.InferenceSession(str(Path(a.policy) / "policy.onnx"))
env = DuckWalkEnv(a.envs, device=a.device, seed=123, dr_on=(a.dr == "on"), dr=dict(push_p=0.0, cmd_zero_p=0.0, cmd_vx=(a.vx, a.vx), episode_s=1e9, init_yaw=0.0))
env.cmd[:, 0] = a.vx
from .env import LEGS
# settle (hold) like World2's program, then walk
for _ in range(int(a.settle * RATE_HZ)):
    obs, priv, r, done, info = env.step(torch.zeros(a.envs, LEGS))
env.prev_action.zero_()
x0 = env.trunk()["p"][:, 0].clone(); z0 = env.trunk()["p"][:, 2].clone(); yaw0 = env.trunk()["yaw"].clone(); fell = torch.zeros(a.envs, dtype=torch.bool)
on = torch.ones(a.envs, 2, dtype=torch.bool); lifts = torch.zeros(a.envs, 2); maxh = torch.zeros(a.envs, 2)
dumps = []
obs = env.obs()
for k in range(int(a.seconds * RATE_HZ)):
    act = torch.from_numpy(np.concatenate([sess.run(None, {"obs": obs[i:i + 1].cpu().numpy()})[0] for i in range(a.envs)]))
    if a.dump and k < 40: dumps.append({"k": k, "obs": obs[0].tolist(), "act": act[0].tolist()})
    obs, priv, r, done, info = env.step(act.to(env.device))
    T = env.trunk(); fell |= (T["h"] < 0.7 * env.trunk_y0) | (T["tilt"] > math.radians(40))
    h = env.out[:, 7:9].cpu(); maxh = torch.maximum(maxh, h)
    lift = on & (h > 0.004); lifts += lift.float(); on = torch.where(lift, torch.zeros_like(on), on) | (h < 0.001)
fwd = (env.trunk()["p"][:, 0] - x0).cpu(); lat = (env.trunk()["p"][:, 2] - z0).cpu(); dyaw = (env.trunk()["yaw"] - yaw0).cpu()
print(json.dumps({"forward_m_mean": round(fwd.mean().item(), 3), "forward_min": round(fwd.min().item(), 3), "falls": int(fell.sum()), "lateral_mean": round(lat.mean().item(), 3), "lateral_absmax": round(lat.abs().max().item(), 3), "yaw_change_deg_mean": round(math.degrees(dyaw.mean().item()), 1), "lifts_mean": lifts.mean(0).tolist(),
                  "lifts_min_foot": lifts.min(-1).values.min().item(), "max_sole_mm": (maxh.mean(0) * 1000).tolist()}))
if a.dump: Path(a.dump).write_text(json.dumps(dumps))
