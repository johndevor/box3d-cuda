"""Export the trained student (World2's 49 screw-drive observations) to a contract `world2-learned-skill/1` skill
directory (skill 'screw', action kind 'screw-drive', docs/learned-skills.md section 11).
The graph: clamp((obs - mean) / sqrt(var + 1e-8), +-10) -> MLP (ELU) -> tanh: raw SI observations in, actions in [-1, 1].
  python -m rl.screw_drive.export_onnx runs/screw --id screw-drive-box3d-v1 --dest <world2>/skills/learned [--extra training.json]
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
        "engine": "box3d-cuda (MIT, github johndevor/box3d-cuda) manifold_step (persistent OBB manifolds, float32) + rl/screw_drive/screw_joint.cu "
                  "(helical joint, thread engagement, cross-thread / strip / cam-out state; CPU oracle screw_reference.py), one CUDA thread per world",
        "box3d_cuda_commit": commit, "box3d_cuda_branch": "screw-thread", "project": "box3d-cuda rl/screw_drive (screw_joint.cu, env.py, train.py)",
        "algorithm": "PPO teacher (privileged state) -> DAgger student (these 49 observations only) -> PPO fine-tune with the teacher's privileged critic (own torch implementation, BSD/MIT deps only)",
        "model": "the interaction, not the threads: a square box screw on a compliant bit through a plate's square clearance hole (box walls, chamfer boxes) "
                 "onto an insert slab; the thread starts within the capture (0.1-0.2 d) of the insert axis (catch mode A: forward spin through the "
                 "thread start, a back-turn clicks; mode B (30 %): on capture, as World2's referee); over theta_x of tilt at the catch it is cross-threaded "
                 "(binding k_x per turn); helical joint (pitch per turn); quasi-static spindle (speed command, clutch, motor limit, Phillips cam-out, hex "
                 "bit slip); run-down, seat, clamp K d F, strip; reverse backs it out and releases it past the thread start",
        "config": ev.get("cfg"), "args": ev.get("args"),
        "steps": sum((ev.get(k) or {}).get("steps", 0) for k in ("teacher", "distill", "finetune")),
        "success_rate": (ev.get("eval_nominal") or {}).get("success"),
        "eval_box3d": {k: ev.get(k) for k in ("teacher_eval", "distill_eval", "eval_nominal", "eval_hard_1.5x")},
        "onnx_parity_max_abs": err,
    }
    if a.extra:
        training.update(json.loads(Path(a.extra).read_text()))
    S = ACTION_SCALE
    manifest = {
        "id": a.id, "contract": "world2-learned-skill/1", "skill": "screw", "policy": "policy.onnx", "sha256": sha, "rate_hz": RATE_HZ,
        "observation": OBS_FIELDS,
        "action": {"kind": "screw-drive", "scale": {"pos_m": S["pos_m"], "rot_rad": S["rot_rad"], "force_n": list(S["force_n"]),
                                                    "descend_m_s": S["descend_m_s"], "lift_m_s": S["lift_m_s"], "max_rpm": S["max_rpm"],
                                                    "k_n_per_m": S["k_n_per_m"], "k_rot_nm_per_rad": S["k_rot_nm_per_rad"]}},
        "input": "obs", "output": "action", "families": ["screw"], "limits": dict(LIMITS),
        "termination": {"min_s": 0.1, "verify": "screw", "seat_rpm_max": 120},
        "training": training, "licence": "project",
        "basis": "PPO in box3d-cuda (crude box screw, a helical joint with engagement and failure state, heavy domain randomisation: M2-M5 pitch, "
                 "run-down torque, set torque, joint stiffness, strip, Phillips/hex, cross-thread angle and binding, belief tilt/offset (D405-class sigma), "
                 "bit compliance, wobble, torque and F/T noise/delay/drift, action latency, motor lag); World2's Rapier referees",
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"onnx": str(path), "sha256": sha, "parity_max_abs": err, "commit": commit[:10]}))


if __name__ == "__main__":
    main()
