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

ACCEPTANCE = CPU-CONFIRMED. The trainer's in-run probe (walk/train/
gpu_train.py --accept-every) runs on the fp32 CUDA lane, which is NOT the
frozen judge's lane: the first real use (stocky leg 20260902-155451) wrote
an on-device 12/12 that the CPU harness scored 10/12. So an
artifacts/**/accepted/ directory (actor_accepted.pt + candidate_<u>.pt,
the specs list them) is a set of CANDIDATES, never a verdict: every
accepted/*.pt is re-judged locally with the robot's CPU acceptance harness
(walk/eval/{acceptance,humanoid_acceptance,arm_acceptance}.py; --robot/
--variant inferred from the spec's train command), both numbers are
printed per candidate, and the chain stops -- "ACCEPTED — chain complete
(CPU-confirmed ...)" -- only when a candidate scores 12/12 on the CPU
harness. Unconfirmed candidates are logged as false positives, kept in
the best-by-judge bookkeeping (they may still be the best actor) and the
leg counts normally toward the plateau. Legacy duck legs (the duck's own
probe + 11 s confirmation, RESULTS.md) keep their log-line acceptance.

LOCAL JUDGE, ALWAYS. After every leg the leg's actor_final.pt is judged
with the CPU harness too, the cells are recorded in the leg line, and the
best actor BY CPU CELLS over final actors and candidates is kept at
runs/best-<spec-name>.pt (+ .json provenance), so a chain's "best" is the
real judge, not the in-kernel proxy or the device-lane probe. The plateau
rule scores a leg by its best CPU cells (else the best probe cells, else
the gate-proxy fallback).
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
# arm twins (gpu_train.ACCEPTED_LINE): the trainer's own verdict (cpu lane or
# the legacy duck path). "... ACCEPTED-CANDIDATE at update N ..." is the cuda
# lane's nomination (matched separately; never a stop by itself).
ACCEPTED = re.compile(
    r"^(WALKING|HUMANOID WALKING|ARM REACH) ACCEPTED at update (\d+)",
    re.MULTILINE)
CANDIDATE = re.compile(
    r"^(WALKING|HUMANOID WALKING|ARM REACH) ACCEPTED-CANDIDATE at update (\d+)",
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
    """(score, probes seen, device-confirmed?) for one leg.

    With accept probes the score is the best passed-cell count over the
    leg's probes (duck: episodes, humanoid/arm: the 12 harness cells);
    device-confirmed = a probe line says confirmed=True, an ACCEPTED /
    ACCEPTED-CANDIDATE line was printed, or accepted/ artifacts came back.
    That is the TRAINER's lane speaking -- main() only stops on it for the
    legacy duck; humanoid/arm candidates go through the CPU harness
    (judge_candidates). Without probes fall back to the mean gate-proxy
    qualified swings (then ep_len) over the last 10 updates of
    metrics.jsonl -- the plateau rule then tracks the proxy instead."""
    best, probes, confirmed = -1.0, 0, False
    for log in (run_dir / "logs").glob("*.log"):
        text = log.read_text(errors="replace")
        for m in PROBE.finditer(text):
            probes += 1
            best = max(best, float(m.group(3)))
            confirmed |= m.group(5) == "True"
        confirmed |= ACCEPTED.search(text) is not None
        confirmed |= CANDIDATE.search(text) is not None
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
    # The chain may run under the launcher's host runtime (no numpy/torch);
    # the judge must run under the project venv (2026-09-02: every STOCKY
    # candidate came back "judge failed: No module named numpy").
    venv_py = ROOT / ".venv/bin/python"
    py = str(venv_py) if venv_py.is_file() else sys.executable
    cmd = [py, "-B", "-m", HARNESS_MODULE[robot],
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


def candidate_actors(acc: Path) -> list[Path]:
    """Every actor the trainer nominated: actor_accepted.pt (first
    candidate) + candidate_<update>.pt (later ones), oldest first."""
    return sorted(p for p in acc.glob("*.pt") if p.is_file())


def device_record(candidate: Path) -> dict | None:
    """The trainer's on-device record next to a candidate (acceptance.json
    for actor_accepted.pt, candidate_<u>.json otherwise)."""
    side = (candidate.parent / "acceptance.json"
            if candidate.name == "actor_accepted.pt"
            else candidate.with_suffix(".json"))
    if not side.is_file():
        return None
    try:
        return json.loads(side.read_text())
    except ValueError:
        return None


def judge_candidates(acc: Path, robot: str, variant: str | None,
                     judge_dir: Path, timeout_s: float = 1800.0,
                     judge=judge_actor) -> list[dict]:
    """CPU-harness verdict for every candidate in accepted/: one dict per
    candidate {actor, device_cells, cpu_cells, cells_total, cpu_confirmed,
    judged (the judge_actor summary or None)}, printed as it goes."""
    out = []
    for cand in candidate_actors(acc):
        rec = device_record(cand) or {}
        dev_cells = rec.get("cells_passed")
        judged = judge(robot, variant, cand, judge_dir / cand.stem, timeout_s)
        cpu_cells = judged["cells_passed"] if judged else None
        total = (judged or rec).get("cells_total")
        confirmed = bool(judged and judged["accepted"]
                         and judged["cells_passed"] == judged["cells_total"])
        mark = ("CPU-CONFIRMED" if confirmed
                else "FALSE POSITIVE" if judged else "judge failed")
        print(f"candidate {cand.name} (update {rec.get('update', '?')}): "
              f"on-device {dev_cells}/{total} vs CPU harness "
              f"{cpu_cells}/{total} -> {mark}"
              + (f" failed={judged['failed_cells']}"
                 if judged and judged["failed_cells"] else ""), flush=True)
        out.append({"actor": cand, "device_cells": dev_cells,
                    "cpu_cells": cpu_cells, "cells_total": total,
                    "cpu_confirmed": confirmed, "judged": judged})
    return out


def process_leg(run_dir: Path, robot: str, variant: str | None,
                spec_name: str, leg: int, judge_enabled: bool = True,
                timeout_s: float = 1800.0, runs_dir: Path = ROOT / "runs",
                judge=judge_actor) -> dict:
    """Everything the chain decides from one finished leg: the log score,
    the ALWAYS-run CPU judge of actor_final.pt, the CPU re-judge of every
    accepted/ candidate, best-by-CPU-cells bookkeeping, and the stop
    decision. Returns {score, stop, confirmed_actor, judged_final,
    candidates, probes, best, device_confirmed}."""
    best, probes, device_confirmed = leg_score(run_dir)
    acc = accepted_dir(run_dir)
    judged_final, candidates, confirmed_actor = None, [], None
    scores = []

    final_actor = find_file(run_dir, "actor_final.pt")
    if judge_enabled and robot in HARNESS_MODULE:
        if final_actor is not None:
            judged_final = judge(robot, variant, final_actor,
                                 run_dir / "judge" / "actor_final", timeout_s)
            if judged_final is not None:
                scores.append(judged_final["cells_passed"])
                if update_best(spec_name, judged_final, final_actor, run_dir,
                               leg, runs_dir):
                    print(f"best-by-judge {best_paths(spec_name, runs_dir)[0]}"
                          f" <- leg {leg} actor_final.pt "
                          f"({judged_final['cells_passed']}/"
                          f"{judged_final['cells_total']} cells)", flush=True)
        if acc is not None and robot != "duck":
            candidates = judge_candidates(acc, robot, variant,
                                          run_dir / "judge", timeout_s, judge)
            for c in candidates:
                if c["judged"] is None:
                    continue
                scores.append(c["cpu_cells"])
                if update_best(spec_name, c["judged"], c["actor"], run_dir,
                               leg, runs_dir):
                    print(f"best-by-judge {best_paths(spec_name, runs_dir)[0]}"
                          f" <- leg {leg} {c['actor'].name} "
                          f"({c['cpu_cells']}/{c['cells_total']} cells)",
                          flush=True)
            confirmed = [c for c in candidates if c["cpu_confirmed"]]
            if confirmed:
                confirmed_actor = confirmed[-1]["actor"]
            elif candidates:
                print(f"leg {leg}: {len(candidates)} on-device candidate(s), "
                      "none CPU-confirmed (false positive of the cuda lane); "
                      "chain continues", flush=True)

    if judged_final is None:
        judge_note = "judge=off" if not judge_enabled else "judge=failed"
    else:
        judge_note = (f"judge={judged_final['cells_passed']}/"
                      f"{judged_final['cells_total']}"
                      + (f" failed={judged_final['failed_cells']}"
                         if judged_final["failed_cells"] else "")
                      + f" ({judged_final['judge_wall_s']}s)")
    if candidates:
        judge_note += (" candidates=" + ",".join(
            f"{c['actor'].name}:{c['device_cells']}->{c['cpu_cells']}"
            for c in candidates))
    score = max(scores) if scores else best
    # legacy duck: its own probe + 11 s confirmation is the accepted lineage
    stop = confirmed_actor is not None or (robot == "duck" and device_confirmed)
    print(f"leg {leg}: best={best} probes={probes} "
          f"device_confirmed={device_confirmed} {judge_note} score={score}",
          flush=True)
    if stop:
        if confirmed_actor is not None:
            print(f"ACCEPTED — chain complete (CPU-confirmed 12/12: "
                  f"{confirmed_actor})", flush=True)
        else:
            where = f" ({acc})" if acc else ""
            print(f"ACCEPTED — chain complete (duck lineage probe "
                  f"confirmed{where})", flush=True)
    return {"score": score, "stop": stop, "confirmed_actor": confirmed_actor,
            "judged_final": judged_final, "candidates": candidates,
            "probes": probes, "best": best,
            "device_confirmed": device_confirmed}


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
                         "actor_final.pt AND of accepted/ candidates (then "
                         "no humanoid/arm leg can ever stop the chain as "
                         "accepted; default: judge, record cells, keep "
                         "runs/best-<spec>.pt by CPU cells)")
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

        result = process_leg(run_dir, robot, variant, spec_name, leg,
                             judge_enabled=not a.no_judge,
                             timeout_s=a.judge_timeout)
        if result["stop"]:
            return 0
        score = result["score"]

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
