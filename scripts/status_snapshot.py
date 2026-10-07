#!/usr/bin/env python3
"""One-screen status across every training/eval lane in this workspace.

Pull-based on purpose: background runs keep writing their own logs and this
script reads them, so nothing has to remember to push status anywhere.

  .venv/bin/python -B scripts/status_snapshot.py          # human table
  .venv/bin/python -B scripts/status_snapshot.py --json   # machine form
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tail_jsonl(path: Path, n: int = 50) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines()[-n:]:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def curriculum_status() -> dict:
    rows = tail_jsonl(ROOT / "runs/grid-curriculum/curriculum.jsonl")
    if not rows:
        return {"state": "no-log"}
    last = rows[-1]
    judges = [r for r in rows if r.get("event") == "judge"]
    halted = any(r.get("event") == "stage_halt_solver_faults" for r in rows)
    return {
        "state": ("halted-solver-faults" if halted else last.get("event")),
        "stage": last.get("stage"),
        "update": last.get("update", last.get("to_update")),
        "last_judge_passed": judges[-1].get("passed") if judges else None,
        "age_min": round((time.time() - last.get("time", 0)) / 60, 1),
    }


def fault_rate() -> dict:
    fault_dir = ROOT / "runs/faults"
    if not fault_dir.is_dir():
        return {"recent_hour": 0}
    cutoff = time.time() - 3600
    files = [p for p in fault_dir.glob("*.json") if p.stat().st_mtime >= cutoff]
    return {"recent_hour": len(files)}


def latest_gpu_probe() -> dict:
    runs = sorted((ROOT / "runs/gpu").glob("*/logs/train.log"),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        return {"state": "no-runs"}
    log = runs[-1]
    probe = None
    for line in log.read_text().splitlines()[::-1]:
        if "[accept " in line:
            probe = line.strip()
            break
    return {"run": log.parents[1].name, "last_probe": probe,
            "age_min": round((time.time() - log.stat().st_mtime) / 60, 1)}


def milestones() -> dict:
    out = {}
    acc = ROOT / "evidence/walking-accepted-20260901/acceptance.json"
    if acc.exists():
        # This file is the STRETCH 4-seed x 3-command gate, not the accepted
        # in-run probe protocol; report its episode score, not a boolean.
        eps = json.loads(acc.read_text()).get("episodes", {})
        n = sum(1 for e in eps.values() if e.get("passed"))
        out["four_seed_gate"] = f"{n}/{len(eps)}"
        out["walking_accepted_probe"] = True
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    snap = {
        "taken_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "grid_curriculum": curriculum_status(),
        "solver_faults": fault_rate(),
        "gpu": latest_gpu_probe(),
        "milestones": milestones(),
    }
    if a.json:
        print(json.dumps(snap, indent=1))
        return 0
    for section, body in snap.items():
        if section == "taken_at":
            print(f"status @ {body}")
            continue
        print(f"\n[{section}]")
        for k, v in (body.items() if isinstance(body, dict) else []):
            print(f"  {k:<18} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
