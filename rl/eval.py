"""Evaluate a World2 learned-skill directory (policy.onnx + manifest.json) in its box3d-cuda env, on held-out seeds,
with the same harness training uses (rl/common/evaluate.py): any version, ours or shipped, side by side.

  python -m rl.eval <skill> <skill dir> [<skill dir> ...] [--sets nominal,hard_1.5x] [--n 1024] [--seed 30000] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl.common.evaluate import evaluate, onnx_act
from rl.common.skill import load_skill


def env_settings(skill, manifest):
    env = dict((manifest.get("training", {}).get("config", {}) or {}).get("env") or {})
    if not env:
        env = dict(skill.recipe.get("env", {}))
        if skill.name == "duck_walk":
            env.update(clock_hz=manifest.get("gait_clock_hz", 0.0))
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("skill")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--sets", default=None)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=30_000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    skill = load_skill(a.skill)
    sets = a.sets.split(",") if a.sets else list(skill.eval_sets)
    for d in a.dirs:
        man = json.loads((Path(d) / "manifest.json").read_text())
        act = onnx_act(Path(d) / man.get("policy", "policy.onnx"))
        for s in sets:
            r = evaluate(skill, act, dict(skill.eval_sets[s]), a.n, a.seed, a.device, env=env_settings(skill, man))
            print(json.dumps(dict(policy=man["id"], set=s, **r)), flush=True)


if __name__ == "__main__":
    main()
