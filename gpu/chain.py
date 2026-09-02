#!/usr/bin/env python3
"""Warm-started GPU leg chain with an early-stop plateau rule.

The 12/12 polish chain taught the lesson this encodes: six identical legs
hovered at the same probe score, so retrying without changing a variable
was pure spend. A chain now stops itself when the best probe score has not
improved for --plateau-legs consecutive legs (or on full acceptance).

  .venv/bin/python -B gpu/chain.py --spec gpu/specs/continue-ff-short.json \
      --legs 6 --plateau-legs 2

Each leg: run gpu/run_daytona.py with the spec, copy the returned
actor/checkpoint into runs/warmstart.pt (the spec uploads it), then score
the leg from its train.log accept-probe lines.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WARMSTART = ROOT / "runs/warmstart.pt"
# "[accept u6999] stage1=False (6/7 episodes) confirmed=False ..."
PROBE = re.compile(
    r"\[accept u(\d+)\] stage1=(True|False) \((\d+)/(\d+) episodes\)"
    r" confirmed=(True|False)")


def leg_score(run_dir: Path) -> tuple[float, int, bool]:
    """(score, probes seen, confirmed?) for one leg.

    With accept probes (duck) the score is the best passed-episode count.
    Without probes (humanoid: no gait evaluator yet) fall back to mean
    ep_len over the last 10 updates of metrics.jsonl — the plateau rule
    then tracks survival instead of gate progress."""
    best, probes, confirmed = -1.0, 0, False
    for log in (run_dir / "logs").glob("*.log"):
        for m in PROBE.finditer(log.read_text(errors="replace")):
            probes += 1
            best = max(best, float(m.group(3)))
            confirmed |= m.group(5) == "True"
    if probes:
        return best, probes, confirmed
    metrics = sorted(run_dir.glob("artifacts/**/metrics.jsonl"))
    if metrics:
        rows = [json.loads(l) for l in
                metrics[-1].read_text().splitlines() if l.strip()]
        tail = [r["ep_len_mean"] for r in rows[-10:] if "ep_len_mean" in r]
        if tail:
            best = round(sum(tail) / len(tail), 1)
    return best, 0, False


def latest_run_dir(output: str) -> Path | None:
    m = re.search(r"run dir: (\S+)", output)
    return Path(m.group(1)) if m else None


def find_actor(run_dir: Path) -> Path | None:
    for name in ("latest.pt", "actor_final.pt"):
        hits = sorted(run_dir.glob(f"artifacts/**/{name}"))
        if hits:
            return hits[-1]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--legs", type=int, default=6)
    ap.add_argument("--plateau-legs", type=int, default=2,
                    help="stop after this many consecutive legs with no "
                         "improvement in best probe episodes")
    a = ap.parse_args()

    best_ever, flat_legs, failed_legs = -1, 0, 0
    for leg in range(1, a.legs + 1):
        print(f"=== chain leg {leg}/{a.legs} ===", flush=True)
        proc = subprocess.run(
            [sys.executable, "-B", str(ROOT / "gpu/run_daytona.py"),
             "run", "--spec", a.spec],
            cwd=ROOT, capture_output=True, text=True)
        sys.stdout.write(proc.stdout[-2000:])
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr[-2000:])
            failed_legs += 1
            print(f"leg {leg} failed (rc={proc.returncode}); "
                  "not counting toward plateau", flush=True)
            if failed_legs >= 3:
                print("3 consecutive leg failures — launcher or provider "
                      "problem, aborting chain")
                return 4
            continue
        failed_legs = 0
        run_dir = latest_run_dir(proc.stdout)
        if run_dir is None:
            print("no run dir in launcher output; stopping"); return 1

        best, probes, confirmed = leg_score(run_dir)
        print(f"leg {leg}: best={best} probes={probes} confirmed={confirmed}",
              flush=True)
        if confirmed:
            print("ACCEPTED — chain complete"); return 0

        actor = find_actor(run_dir)
        if actor is not None:
            shutil.copy2(actor, WARMSTART)
            print(f"warmstart <- {actor.relative_to(ROOT)}")

        if best > best_ever:
            best_ever, flat_legs = best, 0
        else:
            flat_legs += 1
            if flat_legs >= a.plateau_legs:
                print(f"PLATEAU: no improvement for {flat_legs} legs "
                      f"(best={best_ever}); stopping — change a variable "
                      "(lr, seeds, DR, policy) instead of re-rolling",
                      flush=True)
                return 2
    print(f"chain exhausted {a.legs} legs (best={best_ever})")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
