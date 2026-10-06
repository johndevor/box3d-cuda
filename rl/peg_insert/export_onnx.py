"""Export the trained student (World2's 57 observations) to a contract `world2-learned-skill/1` skill directory.
The graph: clamp((obs - mean) / sqrt(var + 1e-8), +-10) -> MLP (ELU) -> tanh: raw SI observations in, actions in [-1, 1].
  python -m rl.peg_insert.export_onnx runs/peg --id peg-hole-box3d-v1 --dest <world2>/skills/learned [--extra training.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .env import ACTION_SCALE, LIMITS, OBS_FIELDS, OBS_SIZE, RATE_HZ
from .train import Actor


class Exported(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.net = actor.net
        self.register_buffer("mean", actor.norm.mean.clone())
        self.register_buffer("inv_std", 1.0 / torch.sqrt(actor.norm.var + 1e-8))

    def forward(self, obs):
        return torch.tanh(self.net(torch.clamp((obs - self.mean) * self.inv_std, -10.0, 10.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--id", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--ckpt", default="student.pt")
    ap.add_argument("--extra", default=None, help="JSON file merged into manifest.training")
    a = ap.parse_args()
    run = Path(a.run)
    actor = Actor(OBS_SIZE)
    actor.load_state_dict(torch.load(run / a.ckpt, map_location="cpu")["actor"])
    net = Exported(actor).eval()
    dest = Path(a.dest) / a.id
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "policy.onnx"
    torch.onnx.export(net, torch.zeros(1, OBS_SIZE), str(path), input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path))
    m, s = net.mean.numpy(), (1 / net.inv_std).numpy()
    x = (m + s * np.random.default_rng(0).normal(0, 1, (256, OBS_SIZE))).astype(np.float32)
    with torch.no_grad():
        ref = net(torch.tensor(x)).numpy()
    err = float(np.abs(ref - sess.run(["action"], {"obs": x})[0]).max())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    ev = json.loads((run / "eval.json").read_text()) if (run / "eval.json").exists() else {}
    training = {
        "engine": "box3d-cuda (MIT, github johndevor/box3d-cuda) manifold_step: persistent OBB manifolds, float32, one CUDA thread per world",
        "box3d_cuda_commit": commit, "box3d_cuda_branch": "peg-insert-dr", "project": "box3d-cuda rl/peg_insert (env.py, train.py)",
        "algorithm": "PPO teacher (privileged state) -> DAgger student (these 57 observations only) -> PPO fine-tune with the teacher's privileged critic (own torch implementation, BSD/MIT deps only)",
        "geometry": "crude on purpose: a square box peg (side = the family's diameter) into a square hole of box walls with 45-degree chamfer boxes; no part chamfer; cylinder features reported",
        "config": ev.get("cfg"), "args": ev.get("args"),
        "steps": sum((ev.get(k) or {}).get("steps", 0) for k in ("teacher", "distill", "finetune")),
        "success_rate": (ev.get("eval_nominal") or {}).get("success"), "eval_box3d": {k: ev.get(k) for k in ("teacher_eval", "distill_eval", "eval_nominal", "eval_hard_1.5x")},
        "onnx_parity_max_abs": err,
    }
    if a.extra:
        training.update(json.loads(Path(a.extra).read_text()))
    manifest = {
        "id": a.id, "contract": "world2-learned-skill/1", "skill": "insert", "policy": "policy.onnx", "sha256": sha, "rate_hz": RATE_HZ,
        "observation": OBS_FIELDS,
        "action": {"kind": "delta-pose-impedance", "scale": {"pos_m": ACTION_SCALE["pos_m"], "rot_rad": ACTION_SCALE["rot_rad"],
                                                            "k_trans_n_per_m": list(ACTION_SCALE["k_trans_n_per_m"]), "k_rot_nm_per_rad": list(ACTION_SCALE["k_rot_nm_per_rad"])}},
        "input": "obs", "output": "action", "families": ["peg-hole"], "limits": dict(LIMITS), "termination": {"min_s": 0.1, "verify": "insertion"},
        "training": training, "licence": "project",
        "basis": "PPO in box3d-cuda (crude box peg/hole, heavy domain randomisation: clearance, pose error, friction, mass, F/T noise/delay/drift/gain, action latency, motor lag); World2's Rapier referees",
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"onnx": str(path), "sha256": sha[:12], "parity_max_abs": err, "commit": commit[:10]}))


if __name__ == "__main__":
    main()
