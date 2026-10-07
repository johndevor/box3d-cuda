"""Train any skill with the shared pipeline (rl/common): one command for every skill.

  python -m rl.train <skill> --out runs/<skill> --sim-match runs/<skill>/simmatch.json [--device cuda]
        [--cfg '{"clearance_mm": [0.05, 0.5]}'] [--set teacher.max_minutes=20 --set envs=16384] [--time-scale 0.5]

Refuses to start unless the sim-match report (python -m rl.simmatch) for this skill passed at these sources, or
--no-sim-match is given (recorded as skipped). Refuses a CPU device unless --cpu-ok (local smoke tests): training
never falls back to the CPU on its own.
Writes in --out: run.json (skill, provenance, device, recipe, randomization spec, extensions), metrics.jsonl (one row
per iteration), the phases' checkpoints, policy.pt (the exported actor) and eval.json (the summary export.py records).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from rl.common import cuda, recipes
from rl.common.provenance import provenance
from rl.common.skill import load_skill


def scale_minutes(R, f):
    """Multiply every max_minutes in the recipe (and its stages) by f."""
    for k, v in list(R.items()):
        if isinstance(v, dict):
            scale_minutes(v, f)
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, dict):
                    scale_minutes(x, f)
        elif k == "max_minutes":
            R[k] = v * f


def check_sim_match(path, skill, prov, uses=lambda k: True):
    """uses(setting) -> whether this training turns a randomization setting on: a failing check that `needs` a setting
    the run does not use is recorded, not blocking."""
    rep = json.loads(Path(path).read_text())
    problems, excused = [], []
    if rep.get("skill") != skill.name:
        problems.append(f"report is for {rep.get('skill')}")
    for c in rep.get("checks", []):
        if not c.get("pass"):
            if c.get("needs") and not uses(c["needs"]):
                excused.append(f"{c['name']} (needs {c['needs']}, which this run does not use)")
            else:
                problems.append(c["name"] + (f" (needs {c['needs']}, which this run uses)" if c.get("needs") else ""))
    if not rep.get("checks") and not rep.get("pass"):
        problems.append("it did not pass")
    if rep.get("provenance", {}).get("rl_sources") != prov["rl_sources"]:
        problems.append(f"it ran on other sources ({rep.get('provenance', {}).get('rl_sources')} vs {prov['rl_sources']})")
    if problems:
        raise SystemExit(f"sim-match gate: {path}: " + "; ".join(problems) + " -> training not allowed (rerun python -m rl.simmatch, or --no-sim-match for a dev run)")
    return {"report": str(path), "pass": True, "excused": excused, "checks": [{k: c[k] for k in ("name", "value", "tolerance", "pass")} for c in rep["checks"]], "world2": rep.get("world2", {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("skill")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu-ok", action="store_true", help="allow --device cpu (local smoke tests only)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cfg", default="{}", help="JSON randomization overrides (rl/common/randomization.py format)")
    ap.add_argument("--set", action="append", default=[], help="recipe setting, dotted path=value (JSON), e.g. teacher.max_minutes=20")
    ap.add_argument("--time-scale", type=float, default=1.0, help="multiply every phase's max_minutes")
    ap.add_argument("--sim-match", default=None, help="the passing sim-match report for this skill (python -m rl.simmatch)")
    ap.add_argument("--no-sim-match", action="store_true")
    ap.add_argument("--recipe", default="default", help="the skill's default recipe or one of its alternatives (Skill.recipes)")
    a = ap.parse_args()
    skill = load_skill(a.skill)
    out = Path(a.out or f"runs/{skill.name}")
    out.mkdir(parents=True, exist_ok=True)
    if a.device != "cuda" and not a.cpu_ok:
        raise SystemExit(f"--device {a.device}: training runs on CUDA (use --cpu-ok for a local smoke test)")
    dev = cuda.require_device(a.device)
    prov = provenance()
    R = copy.deepcopy(skill.recipe if a.recipe == "default" else skill.recipes[a.recipe])
    for kv in a.set:
        k, v = kv.split("=", 1)
        recipes.set_path(R, k, v)
    if a.time_scale != 1.0:
        scale_minutes(R, a.time_scale)
    R.update(device=a.device)
    cfg = json.loads(a.cfg)

    def uses(k):
        on = lambda v: v is not None and v is not False and v != 0 and tuple(v if isinstance(v, (list, tuple)) else [v]) not in ((0, 0), (0,))
        v = cfg.get(k, R.get("cfg", {}).get(k, skill.spec[k] if k in skill.spec else None))
        return on(v) or any(on(dict(R.get("cfg", {}), **st.get("cfg", {}), **cfg).get(k)) for st in R.get("stages", []))
    if a.no_sim_match:
        sm = dict(skipped=True)
    elif a.sim_match:
        sm = check_sim_match(a.sim_match, skill, prov, uses)
    else:
        raise SystemExit("sim-match gate: pass --sim-match <report> (python -m rl.simmatch <skill> --world2 <trace>) or --no-sim-match")
    torch.manual_seed(a.seed)
    mf = open(out / "metrics.jsonl", "a")

    def log(row):
        print(json.dumps(row, default=str), flush=True)
        mf.write(json.dumps(row, default=str) + "\n")
        mf.flush()

    def make_env(n, cfg_, env_kw):
        env = skill.make_env(n, device=a.device, seed=a.seed, cfg=cfg_, **(env_kw or {}))
        if a.device == "cuda":
            assert env.dev.type == "cuda", "env is not on CUDA"
        return env

    run = dict(skill=skill.name, contract_skill=skill.contract_skill, provenance=prov, device=cuda.device_record(dev), args=vars(a), recipe=R,
               randomization=skill.spec.override(dict(R.get("cfg", {}), **cfg)).to_json(), randomization_overrides=dict(R.get("cfg", {}), **cfg), sim_match=sm, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (out / "run.json").write_text(json.dumps(run, indent=1, default=str))
    log(dict(event="start", skill=skill.name, device=run["device"], commit=prov.get("commit"), dirty=prov.get("dirty"), sim_match="skipped" if a.no_sim_match else "pass"))
    t0 = time.time()
    ac, a_in, S, env = recipes.run(skill, R, cfg, out, log, make_env)
    S["wall_s_total"] = round(time.time() - t0, 1)
    S["extensions"] = cuda.LOAD_LOG
    S["env_settings"] = dict(R.get("env", {}), **(R["stages"][-1].get("env", {}) if R.get("stages") else {}))
    torch.save(dict(actor=ac.actor_state(), skill=skill.name, obs_size=env.obs_size, act_size=env.act_size, env_settings=S["env_settings"]), out / "policy.pt")
    (out / "eval.json").write_text(json.dumps(dict(run=run, summary=S), indent=1, default=str))
    log(dict(event="done", wall_s=S["wall_s_total"], **{k: (S.get(k) or {}).get("success") for k in skill.eval_sets}))


if __name__ == "__main__":
    main()
