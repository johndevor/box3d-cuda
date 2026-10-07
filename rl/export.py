"""Export a trained run (python -m rl.train) to a World2 learned-skill directory (contract world2-learned-skill/1).

  python -m rl.export runs/<skill> --id <skill-id> --dest <world2>/skills/learned [--extra extra.json] [--allow-cpu]

Writes <dest>/<id>/policy.onnx (observation normaliser inside: raw SI observations in, the action out) and
manifest.json: the skill's contract fields (rl/skills/<skill>/skill.py), the ONNX sha256, and `training` recording the
box3d-cuda commit (and whether the tree was dirty), the recipe and randomization config, the device it trained on, the
sim-match result, the box3d evaluation and the ONNX/torch parity. Refuses a run that did not train on CUDA unless
--allow-cpu (local smoke tests), and a policy whose ONNX disagrees with torch by more than 1e-4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rl.common.models import Exported, from_actor_state
from rl.common.skill import load_skill

ALGORITHMS = {
    "teacher_student": "PPO teacher (privileged state) -> DAgger student (the policy observations only) -> PPO fine-tune with the teacher's privileged critic (rl/common, own torch implementation, BSD/MIT deps only)",
    "asymmetric": "PPO, asymmetric actor-critic: the actor (this policy) sees only the policy observations; the critic also sees privileged state (rl/common, own torch implementation)",
}


def export(run_dir, skill_id, dest, extra=None, allow_cpu=False):
    run_dir = Path(run_dir)
    ev = json.loads((run_dir / "eval.json").read_text())
    run, S = ev["run"], ev["summary"]
    skill = load_skill(run["skill"])
    if run["device"]["type"] != "cuda" and not allow_cpu:
        raise SystemExit(f"{run_dir}: trained on {run['device']['type']}, not CUDA: not exported (--allow-cpu for a smoke test)")
    pol = torch.load(run_dir / "policy.pt", map_location="cpu")
    ac = from_actor_state(pol["actor"]).eval()
    net = Exported(ac).eval()
    n_in = pol["obs_size"]
    out = Path(dest) / skill_id
    out.mkdir(parents=True, exist_ok=True)
    path = out / "policy.onnx"
    torch.onnx.export(net, torch.zeros(1, n_in), str(path), input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path))
    m, s = net.mean.numpy(), (1 / net.inv_std).numpy()
    x = (m + s * np.random.default_rng(0).normal(0, 1, (256, n_in))).astype(np.float32)
    with torch.no_grad():
        ref = net(torch.tensor(x)).numpy()
    err = float(np.abs(ref - sess.run(["action"], {"obs": x})[0]).max())
    if err > 1e-4:
        raise SystemExit(f"ONNX parity {err} > 1e-4")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    contract = skill.manifest(skill.spec, pol.get("env_settings") or {})
    text = contract.pop("training_text", {})
    R, prov = run["recipe"], run["provenance"]
    kind = R.get("kind")
    phases = {k: (S.get(k) or {}).get("wall_s") for k in ("teacher", "distill", "finetune", "train")}
    if S.get("stages"):
        phases = {name: st.get("train", {}).get("wall_s") for name, st in S["stages"].items()}
    nominal = (S.get("nominal") or {}).get("success")
    training = dict(
        **text,
        pipeline="box3d-cuda rl/common (python -m rl.train %s): one trainer, recipe %s%s" % (skill.name, kind, f", stages {[s.get('name') for s in R['stages']]}" if R.get("stages") else ""),
        box3d_cuda_commit=prov.get("commit"), box3d_cuda_branch=prov.get("branch"), box3d_cuda_dirty=prov.get("dirty"), rl_sources=prov.get("rl_sources"),
        project=f"box3d-cuda rl/skills/{skill.name} (env.py, skill.py) on rl/common",
        algorithm=ALGORITHMS.get(kind, kind),
        device=run["device"], sim_match=run.get("sim_match"),
        config=dict(recipe=R, randomization=run["randomization"], randomization_overrides=run.get("randomization_overrides"), env=pol.get("env_settings"), seed=run["args"].get("seed")),
        steps=S.get("steps"), success_rate=nominal,
        eval_box3d={k: S.get(k) for k in ("teacher_eval", "distill_eval", *skill.eval_sets) if S.get(k) is not None},
        wall_s=dict(phases, total=S.get("wall_s_total")), extensions=S.get("extensions"),
        onnx_parity_max_abs=err,
    )
    if extra:
        training.update(json.loads(Path(extra).read_text()) if isinstance(extra, (str, Path)) else extra)
    manifest = dict(id=skill_id, contract="world2-learned-skill/1", policy="policy.onnx", sha256=sha, **contract, training=training)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    return dict(dir=str(out), sha256=sha, parity_max_abs=err, commit=(prov.get("commit") or "")[:10], success_rate=nominal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--id", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--extra", default=None, help="JSON file merged into manifest.training")
    ap.add_argument("--allow-cpu", action="store_true")
    a = ap.parse_args()
    print(json.dumps(export(a.run, a.id, a.dest, a.extra, a.allow_cpu)))


if __name__ == "__main__":
    main()
