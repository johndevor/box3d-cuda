"""Export the trained actor (the student: sensors only) as a World2 learned skill (contract world2-learned-skill/1, skill
'walk', action kind leg-joint-targets): policy.onnx (observation normalisation inside, the action mean out) + manifest.json
recording this repository's commit and the training config.

  python -m rl.duck_walk.export_onnx --ckpt runs/duck-walk/best.pt --out <world2>/skills/learned/duck-walk-box3d-v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import torch.nn as nn

from .env import OBS_SIZE, LEGS, HISTORY, RATE_HZ, ACTION_SCALE
from .train import ActorCritic


class Student(nn.Module):
    def __init__(self, ac):
        super().__init__()
        self.norm, self.actor = ac.obs_norm, ac.actor

    def forward(self, obs):
        return self.actor(self.norm(obs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--id", default="duck-walk-box3d-v1")
    ap.add_argument("--notes", default="")
    ap.add_argument("--heading-hold", type=float, default=0.0, help="gain of the robot-side heading loop the policy was trained to follow (0: none)")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    cfg = ck["cfg"]
    ac = ActorCritic(cfg["obs"], cfg["priv"], cfg["act"])
    ac.load_state_dict(ck["model"]); ac.eval()
    st = Student(ac).eval()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, OBS_SIZE)
    torch.onnx.export(st, x, str(out / "policy.onnx"), input_names=["obs"], output_names=["action"], opset_version=17, dynamo=False)
    sha = hashlib.sha256((out / "policy.onnx").read_bytes()).hexdigest()
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(["git", "-C", str(here), "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "-C", str(here), "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        commit, branch = cfg.get("box3d_cuda_commit", "unknown"), "duck-walk-world2"
    manifest = {
        "id": a.id, "contract": "world2-learned-skill/1", "skill": "walk", "policy": "policy.onnx", "sha256": sha, "rate_hz": RATE_HZ,
        "observation": ["imu_gyro", "imu_gravity", "leg_joint_pos", "leg_joint_vel", "prev_action", "command"], "history": HISTORY,
        "action": {"kind": "leg-joint-targets", "scale": {"rad": ACTION_SCALE}}, "input": "obs", "output": "action",
        "families": ["open-duck-mini-v2"],
        **({"heading_hold": {"gain": a.heading_hold, "max": 0.3}} if a.heading_hold > 0 else {}),
        "robot": {"model": "parts/open-duck-mini (World2), exported by parts/open-duck-mini/export-cuda-model.mjs", "legs": ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]},
        "training": {
            "engine": "box3d-cuda (MIT, github johndevor/box3d-cuda) rl/duck_walk/duck_sim.h: maximal-coordinate rigid bodies, revolute joints and sole-floor contacts by sequential impulses (float32, one CUDA thread per world), World2's bam-m1 servo step and IMU signal chain step for step at 1/120 s",
            "box3d_cuda_commit": commit, "box3d_cuda_branch": branch, "project": "box3d-cuda rl/duck_walk (env.py, train.py)",
            "algorithm": "PPO, asymmetric actor-critic: the actor (this policy) sees only the IMU, the servo encoders, its previous actions and the command; the critic also sees the trunk's true velocity, height, contacts, friction, mass (own torch implementation)",
            "steps": ck.get("steps"), "score": ck.get("score"), "config": {k: v for k, v in cfg.items() if k not in ("init",)}, "notes": a.notes,
        },
        "licence": "MIT (box3d-cuda); robot model derived from Open_Duck_Mini (Apache-2.0)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    # parity: onnxruntime against torch on random observations
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(str(out / "policy.onnx"))
        xs = torch.randn(16, 1, OBS_SIZE)
        err = max(float(abs(s.run(None, {"obs": x.numpy()})[0] - st(x).detach().numpy()).max()) for x in xs)
        print("onnx parity max abs", err)
    except Exception as e:
        print("onnx parity skipped:", e)
    print(json.dumps({"out": str(out), "sha256": sha, "commit": commit}))


if __name__ == "__main__":
    main()
