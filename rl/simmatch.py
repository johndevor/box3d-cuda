"""The sim-match gate (rl/common/simmatch.py): replay World2's scripted motion for a skill and compare.

  python -m rl.simmatch <skill> --world2 trace.json [--out report.json] [--device cuda|cpu]

Exit 0 when every check passes, 1 when one fails (the report says which, with the values and tolerances), 2 when
World2 had no probe for this skill.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from rl.common.provenance import provenance
from rl.common.skill import load_skill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("skill")
    ap.add_argument("--world2", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--env", default="{}", help="JSON env settings for the probe (e.g. a generated robot model: {\"model_file\": ...})")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    skill = load_skill(a.skill)
    w2 = json.loads(Path(a.world2).read_text())
    if w2.get("skill") != skill.name:
        raise SystemExit(f"{a.world2} is World2's trace for {w2.get('skill')}, not {skill.name}")
    t0 = time.time()
    if w2.get("unavailable") or skill.sim_match is None:
        rep = {"skill": skill.name, "pass": False, "unavailable": w2.get("unavailable") or "this skill has no sim-match probe"}
        code = 2
    else:
        env = json.loads(a.env)
        checks, box3d = skill.sim_match(w2, a.device, env=env) if env else skill.sim_match(w2, a.device)
        rep = {"skill": skill.name, "pass": all(c["pass"] for c in checks), "checks": checks, "box3d": box3d, "device": a.device}
        # (failing checks that all carry `needs` are conditional: rl/train.py allows a run that does not use them)
        cond = [c["name"] for c in checks if not c["pass"] and c.get("needs")]
        rep["conditional_fail"] = cond
        code = 0 if rep["pass"] or len(cond) == sum(not c["pass"] for c in checks) else 1
    rep.update(env=json.loads(a.env), provenance=provenance(), world2={k: w2.get(k) for k in ("world2", "probe", "case", "end", "wall_s")}, wall_s=round(time.time() - t0, 1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    for c in rep.get("checks", []):
        print(f"  {'ok  ' if c['pass'] else 'FAIL'} {c['name']}: {c['value']} (tolerance {c['tolerance']})", file=sys.stderr)
    print(json.dumps({"skill": skill.name, "pass": rep["pass"], "conditional_fail": rep.get("conditional_fail"), "failed": [c["name"] for c in rep.get("checks", []) if not c["pass"]], "unavailable": rep.get("unavailable")}))
    sys.exit(code)


if __name__ == "__main__":
    main()
