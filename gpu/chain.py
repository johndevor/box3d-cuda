#!/usr/bin/env python3
"""Warm-started GPU leg chain with an early-stop plateau rule.

The 12/12 polish chain taught the lesson this encodes: six identical legs
hovered at the same probe score, so retrying without changing a variable
was pure spend. A chain now stops itself when the best judge score has not
improved for --plateau-legs consecutive legs (or on full acceptance).

  .venv/bin/python -B gpu/chain.py --spec gpu/specs/continue-ff-short.json \
      --legs 6 --plateau-legs 2

Each leg: run gpu/run_daytona.py with the spec, copy the returned
actor/checkpoint into runs/warmstart.pt (the spec uploads it), then score
the leg.

ACCEPTANCE STOP. A leg is accepted when the trainer's in-run probe
(walk/train/gpu_train.py --accept-every) passed: its log carries
"... confirmed=True" / a "<ROBOT> ACCEPTED at update N" line (WALKING
ACCEPTED, HUMANOID WALKING ACCEPTED, ARM REACH ACCEPTED) and the run dir
holds artifacts/**/accepted/{acceptance.json, actor_accepted.pt} (the specs
list them as artifacts). Any of those -> print ACCEPTED and stop (rc 0).

LOCAL JUDGE. After every leg the leg's actor_final.pt is judged LOCALLY
with the robot's CPU acceptance harness (walk/eval/{acceptance,
humanoid_acceptance,arm_acceptance}.py -- the frozen judges' authority;
--robot/--variant are inferred from the spec's train command), the cells
passed are recorded in the leg line, and the best actor BY JUDGE CELLS is
kept at runs/best-<spec-name>.pt (+ .json provenance), so a chain's "best"
is the real judge, not the in-kernel proxy or the probe on the device
lane. The plateau rule scores a leg by its judge cells when the judge ran
(else the best probe cells, else the gate-proxy fallback).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WARMSTART = ROOT / "runs/warmstart.pt"
# "[accept u6999] stage1=False (6/7 episodes) confirmed=False ..." -- the
# trainer prints this line for EVERY robot (gpu_train.do_probe); humanoid /
# arm probes append " failed=[...]" after probe_wall, which this ignores.
PROBE = re.compile(
    r"\[accept u(\d+)\] stage1=(True|False) \((\d+)/(\d+) episodes\)"
    r" confirmed=(True|False)")
# "WALKING ACCEPTED at update 6245 after 27957.7 s (...)" and the humanoid /
# arm twins (gpu_train.ACCEPTED_LINE).
ACCEPTED = re.compile(
    r"^(WALKING|HUMANOID WALKING|ARM REACH) ACCEPTED at update (\d+)",
    re.MULTILINE)
# the CPU harness module per robot (module, extra argv builder)
HARNESS_MODULE = {"duck": "walk.eval.acceptance",
                  "humanoid": "walk.eval.humanoid_acceptance",
                  "arm": "walk.eval.arm_acceptance"}


def accepted_dir(run_dir: Path) -> Path | None:
    """artifacts/**/accepted/ holding the trainer's acceptance record."""
    for hit in sorted(run_dir.glob("artifacts/**/accepted")):
        if hit.is_dir() and ((hit / "acceptance.json").is_file()
                             or (hit / "actor_accepted.pt").is_file()):
            return hit
    return None


def leg_score(run_dir: Path) -> tuple[float, int, bool]:
    """(score, probes seen, accepted?) for one leg.

    With accept probes the score is the best passed-cell count over the
    leg's probes (duck: episodes, humanoid/arm: the 12 harness cells);
    accepted = a probe confirmed, an ACCEPTED line was printed, or the
    accepted/ artifacts came back. Without probes fall back to the mean
    gate-proxy qualified swings (then ep_len) over the last 10 updates of
    metrics.jsonl -- the plateau rule then tracks the proxy instead."""
    best, probes, confirmed = -1.0, 0, False
    for log in (run_dir / "logs").glob("*.log"):
        text = log.read_text(errors="replace")
        for m in PROBE.finditer(text):
            probes += 1
            best = max(best, float(m.group(3)))
            confirmed |= m.group(5) == "True"
        confirmed |= ACCEPTED.search(text) is not None
    confirmed |= accepted_dir(run_dir) is not None
    if probes:
        return best, probes, confirmed
    metrics = sorted(run_dir.glob("artifacts/**/metrics.jsonl"))
    if metrics:
        rows = [json.loads(l) for l in
                metrics[-1].read_text().splitlines() if l.strip()]
        # Preferred fitness: judge-aligned qualified swings (ungameable);
        # fall back to ep_len only when gate metrics are absent.
        qual = [(r.get("gate_proxy_ep_qualified_l") or 0)
                + (r.get("gate_proxy_ep_qualified_r") or 0)
                for r in rows[-10:]
                if r.get("gate_proxy_ep_qualified_l") is not None]
        if qual:
            best = round(sum(qual) / len(qual), 3)
        else:
            tail = [r["ep_len_mean"] for r in rows[-10:]
                    if r.get("ep_len_mean") is not None]
            if tail:
                best = round(sum(tail) / len(tail), 1)
    return best, 0, confirmed


def latest_run_dir(output: str) -> Path | None:
    m = re.search(r"run dir: (\S+)", output)
    return Path(m.group(1)) if m else None


def find_actor(run_dir: Path) -> Path | None:
    for name in ("latest.pt", "actor_final.pt"):
        hits = sorted(run_dir.glob(f"artifacts/**/{name}"))
        if hits:
            return hits[-1]
    return None


def find_file(run_dir: Path, name: str) -> Path | None:
    hits = sorted(run_dir.glob(f"artifacts/**/{name}"))
    return hits[-1] if hits else None


# ---------------------------------------------------------------- local judge
def spec_train_command(spec_path: Path) -> str:
    """The spec's training job command ('train', else the first command
    mentioning walk.train.gpu_train, else '')."""
    spec = json.loads(Path(spec_path).read_text())
    jobs = spec.get("jobs", [])
    for job in jobs:
        if job.get("name") == "train":
            return str(job.get("command", ""))
    for job in jobs:
        if "walk.train.gpu_train" in str(job.get("command", "")):
            return str(job["command"])
    return ""


def infer_robot(train_command: str) -> tuple[str, str | None]:
    """(--robot, --variant) of a gpu_train invocation; duck when absent."""
    robot = re.search(r"--robot[ =]+(\w+)", train_command)
    variant = re.search(r"--variant[ =]+([\w.-]+)", train_command)
    return ((robot.group(1) if robot else "duck"),
            (variant.group(1) if variant else None))


def judge_command(robot: str, variant: str | None, actor: Path,
                  out_dir: Path) -> list[str]:
    """argv of the robot's CPU acceptance harness for `actor`."""
    if robot not in HARNESS_MODULE:
        raise ValueError(f"no CPU harness for robot {robot!r}")
    cmd = [sys.executable, "-B", "-m", HARNESS_MODULE[robot],
           "--actor", str(actor), "--out", str(out_dir)]
    if robot == "humanoid" and variant not in (None, "", "h1"):
        cmd += ["--variant", variant]
    if robot == "arm":
        cmd += ["--variant", variant or "kr240"]
    return cmd


def summarize_acceptance(record: dict) -> dict:
    """{cells_passed, cells_total, failed_cells, accepted} from a harness
    acceptance.json (all three harnesses share the episodes/accepted shape)."""
    episodes = record.get("episodes", {}) or {}
    failed = [k for k, v in episodes.items() if not v.get("passed")]
    return {"cells_passed": len(episodes) - len(failed),
            "cells_total": len(episodes), "failed_cells": failed,
            "accepted": bool(record.get("accepted"))}


def judge_actor(robot: str, variant: str | None, actor: Path, out_dir: Path,
                timeout_s: float = 1800.0) -> dict | None:
    """Run the CPU harness on `actor` (rc 1 = not accepted is a normal
    outcome); returns summarize_acceptance(...) + wall seconds + paths, or
    None when the harness produced no record (logged, never raised)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = judge_command(robot, variant, actor, out_dir)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"judge timed out after {timeout_s:.0f} s: {' '.join(cmd)}",
              flush=True)
        return None
    (out_dir / "judge.log").write_text(proc.stdout + proc.stderr)
    record_path = out_dir / "acceptance.json"
    if not record_path.is_file():
        print(f"judge wrote no acceptance.json (rc={proc.returncode}); "
              f"see {out_dir / 'judge.log'}", flush=True)
        return None
    summary = summarize_acceptance(json.loads(record_path.read_text()))
    summary.update({"judge_wall_s": round(time.perf_counter() - t0, 1),
                    "actor": str(actor), "acceptance_json": str(record_path)})
    return summary


def best_paths(spec_name: str, runs_dir: Path = ROOT / "runs"):
    return (runs_dir / f"best-{spec_name}.pt",
            runs_dir / f"best-{spec_name}.json")


def update_best(spec_name: str, judged: dict, actor: Path, run_dir: Path,
                leg: int, runs_dir: Path = ROOT / "runs") -> bool:
    """Keep runs/best-<spec>.pt as the actor with the most judge cells seen
    (ties keep the incumbent). Provenance in the sibling .json. A prior
    chain's record is honoured, so consecutive chains accumulate."""
    best_pt, best_json = best_paths(spec_name, runs_dir)
    incumbent = -1
    if best_json.is_file() and best_pt.is_file():
        try:
            incumbent = int(json.loads(best_json.read_text())["cells_passed"])
        except (ValueError, KeyError, TypeError):
            incumbent = -1
    if judged["cells_passed"] <= incumbent:
        return False
    runs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(actor, best_pt)
    best_json.write_text(json.dumps({
        "spec": spec_name, "leg": leg, "run_dir": str(run_dir),
        "source_actor": str(actor), **judged,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1) + "\n")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--legs", type=int, default=6)
    ap.add_argument("--plateau-legs", type=int, default=2,
                    help="stop after this many consecutive legs with no "
                         "improvement in the leg score (judge cells)")
    ap.add_argument("--warmstart", default=str(WARMSTART),
                    help="checkpoint relay path the spec resumes from "
                         "(distinct per concurrent chain)")
    ap.add_argument("--label-suffix", default=None,
                    help="passed through to run_daytona.py (concurrent chains)")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the local CPU-harness judge of each leg's "
                         "actor_final.pt (default: judge, record cells, keep "
                         "runs/best-<spec>.pt by judge cells)")
    ap.add_argument("--judge-timeout", type=float, default=1800.0)
    a = ap.parse_args()

    spec_path = Path(a.spec)
    spec_name = json.loads(spec_path.read_text()).get("name", spec_path.stem)
    robot, variant = infer_robot(spec_train_command(spec_path))
    print(f"chain: spec {spec_name} robot={robot} variant={variant or '-'} "
          f"judge={'off' if a.no_judge else HARNESS_MODULE.get(robot, '?')}",
          flush=True)

    best_ever, flat_legs, failed_legs = -1, 0, 0
    for leg in range(1, a.legs + 1):
        print(f"=== chain leg {leg}/{a.legs} ===", flush=True)
        proc = subprocess.run(
            [sys.executable, "-B", str(ROOT / "gpu/run_daytona.py"),
             "run", "--spec", a.spec]
            + (["--label-suffix", a.label_suffix] if a.label_suffix else []),
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
        acc = accepted_dir(run_dir)

        # local judge of the leg's final actor (the frozen judge's authority)
        judged = None
        final_actor = find_file(run_dir, "actor_final.pt")
        if not a.no_judge and final_actor is not None \
                and robot in HARNESS_MODULE:
            judged = judge_actor(robot, variant, final_actor,
                                 run_dir / "judge", a.judge_timeout)
            if judged is not None and update_best(spec_name, judged,
                                                  final_actor, run_dir, leg):
                print(f"best-by-judge {best_paths(spec_name)[0].relative_to(ROOT)}"
                      f" <- leg {leg} ({judged['cells_passed']}/"
                      f"{judged['cells_total']} cells)", flush=True)
        judge_note = ("judge=off" if judged is None else
                      f"judge={judged['cells_passed']}/{judged['cells_total']}"
                      + (f" failed={judged['failed_cells']}"
                         if judged["failed_cells"] else "")
                      + f" ({judged['judge_wall_s']}s)")
        score = judged["cells_passed"] if judged is not None else best
        print(f"leg {leg}: best={best} probes={probes} confirmed={confirmed} "
              f"{judge_note} score={score}", flush=True)
        if confirmed:
            where = f" ({acc.relative_to(ROOT)})" if acc else ""
            print(f"ACCEPTED — chain complete{where}"); return 0

        actor = find_actor(run_dir)
        if actor is not None:
            shutil.copy2(actor, a.warmstart)
            print(f"warmstart {a.warmstart} <- {actor.relative_to(ROOT)}")

        if score > best_ever:
            best_ever, flat_legs = score, 0
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
