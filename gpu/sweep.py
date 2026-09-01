"""Sequential gait-clock sweep driver over ephemeral Daytona GPU legs.

A sweep config is one (PHASE_HZ_BASE, PHASE_HZ_PER_MPS) pair of the affine
gait clock phase_hz = BASE + PER_MPS * command (walk/env/flat.py reads
DUCK_PHASE_HZ_BASE / DUCK_PHASE_HZ_PER_MPS at import; the CUDA kernel bakes
the same two constants through the GENERATED header duck_model.h).

Per config, sequentially:
  1. regenerate experimental/duck_cuda's duck_model.h into runs/sweep-tmp/
     with the config's env vars set (the generator imports flat.py, so the
     regenerated header bakes exactly the swept clock);
  2. stage the warmstart checkpoint next to it and launch one (or more)
     15-minute training legs via gpu/run_daytona.py with the sweep-train
     spec, whose first remote job copies the uploaded header over
     experimental/duck_cuda/include/duck_model.h BEFORE the build;
  3. run the STRICT local evaluation (walk/eval/capture + walk/eval/gait,
     commands 0.10/0.15/0.20, seed 4242, pinned CPU idv1 dylib) on the
     downloaded actor_final.pt, in a fresh .venv subprocess with the config's
     env vars set so the local clock matches the trained clock;
  4. append the record to <out>/sweep-results.jsonl and print a table row.

A config whose leg fails (spot capacity exit 3, job failure exit 2, ...)
records the failure and the sweep continues; overall exit is 0 iff at least
one config completed.

Documented invocation (the Daytona key must come from Doppler, never argv):

    doppler run --project hallway --config dev --only-secrets DAYTONA_API_KEY \
        --no-fallback -- \
        /Users/john/.cache/box3d-cuda-host-runtime-0.207.0/bin/python -B \
        gpu/sweep.py --configs '[[0,10],[0,16.67],[0.8,6],[1.2,4]]' \
        --warmstart runs/warmstart-sweep.pt --out runs/sweep-<name> \
        [--legs-per-config 1] [--parallel N] [--train-wall-s 780] \
        [--fast-build] [--runs-dir runs/gpu] [--dry-run]

--parallel N runs configs through a thread pool of N concurrent sandboxes:
each config gets a unique launcher label suffix (cfg<i>, scoping the
launcher's one-sandbox concurrency guard), staging dir (runs/sweep-tmp-<i>/)
and a derived spec written under <out>/specs/ whose upload_extra and job
commands point at that config's paths. Live table rows print as configs
finish. Default remains fully sequential.

Secrets discipline: the environment (which carries DAYTONA_API_KEY under
doppler) is inherited by the launcher subprocess and never printed or logged.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"
GENERATOR = REPO_ROOT / "experimental" / "duck_cuda" / "tools" / "generate_model.py"
LAUNCHER = REPO_ROOT / "gpu" / "run_daytona.py"
SPEC_PATH = REPO_ROOT / "gpu" / "specs" / "sweep-train.json"

# Fixed repo-relative staging paths referenced by the BASE spec's
# upload_extra. In --parallel mode each config i gets its own staging dir
# runs/sweep-tmp-<i>/ and a derived spec whose paths point there.
DEFAULT_STAGE_REL = "runs/sweep-tmp"
STAGE_HEADER_REL = f"{DEFAULT_STAGE_REL}/duck_model.h"
STAGE_WARMSTART_REL = f"{DEFAULT_STAGE_REL}/warmstart.pt"
REMOTE_HEADER_DEST = "experimental/duck_cuda/include/duck_model.h"

RUNS_GPU_DIR = REPO_ROOT / "runs" / "gpu"
RUN_DIR_GLOB = "*sweep-train*"

DEFAULT_TRAIN_WALL_S = 780          # 15-minute ranking legs by default
TRAIN_TIMEOUT_SLACK_S = 240         # remote job timeout = wall + slack
LAUNCHER_WALL_CAP_S = 3300          # run_daytona.py per-job timeout ceiling

# Header regeneration shells out to the model generator, which touches the
# shared native build cache; serialize it so --parallel N never races there.
_GEN_LOCK = threading.Lock()
ACTOR_REL = Path("artifacts/train/gpu-train-out/actor_final.pt")
LATEST_REL = Path("artifacts/train/gpu-train-out/latest.pt")

EVAL_LIBRARY = REPO_ROOT / "build" / "libintegrated_duck-pinned-97c3d37.dylib"
EVAL_COMMANDS = (0.10, 0.15, 0.20)
EVAL_SEED = 4242
EVAL_SECONDS = 8.0

ENV_BASE = "DUCK_PHASE_HZ_BASE"
ENV_PER_MPS = "DUCK_PHASE_HZ_PER_MPS"

# Strict local evaluation, run in a FRESH .venv subprocess because
# walk/env/flat.py reads the phase env vars at import time. The CPU idv1 lane
# itself has no phase dependence (only the env obs/reward clock does), so the
# pinned dylib is valid for every config; the obs the actor sees must use the
# trained clock, hence the env vars.
EVAL_SCRIPT = r"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from walk.env.contract import ACT, OBS
from walk.env.flat import FlatFloorDuckEnv
from walk.eval.capture import capture_episodes
from walk.eval.gait import evaluate_episode
from walk.train.ppo import Actor

actor_path, library, out_path, seed_s, commands_s, seconds_s = sys.argv[1:7]
seed = int(seed_s)
commands = [float(x) for x in commands_s.split(",")]
actor = Actor(OBS, ACT)
actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
actor.eval()

def policy(obs):
    with torch.no_grad():
        t = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        return actor.deterministic(t).numpy()

env = FlatFloorDuckEnv(environments=1, seed=seed, library_path=library)
episodes = []
try:
    for cmd in commands:
        traces = capture_episodes(env, policy, command=cmd,
                                  seconds=float(seconds_s), seed=seed)
        episodes.append(evaluate_episode(traces[0]))
finally:
    env.close()
Path(out_path).write_text(json.dumps(episodes, indent=2, sort_keys=True) + "\n")
"""

EXIT_OK = 0
EXIT_ERROR = 1


class SweepError(Exception):
    """A per-config (or setup) failure with a printable reason."""


# --------------------------------------------------------------------------
# Config parsing and spec validation
# --------------------------------------------------------------------------

def parse_configs(text):
    """'[[0,10],[0.8,6]]' -> [(0.0, 10.0), (0.8, 6.0)]."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise SweepError(f"--configs is not valid JSON: {e}") from e
    if not isinstance(raw, list) or not raw:
        raise SweepError("--configs must be a non-empty JSON list of [base, per_mps] pairs")
    configs = []
    for i, pair in enumerate(raw):
        if (not isinstance(pair, list) or len(pair) != 2
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                           for x in pair)):
            raise SweepError(f"--configs[{i}] must be a [base, per_mps] pair of numbers")
        base, per = float(pair[0]), float(pair[1])
        if base < 0 or per < 0 or base + per <= 0:
            raise SweepError(f"--configs[{i}] must be non-negative with base+per_mps > 0")
        configs.append((base, per))
    return configs


def validate_spec(spec_path=None):
    """Check the sweep spec references the fixed staging paths and injects
    the header before the build. (Full schema validation is the launcher's.)"""
    spec_path = Path(spec_path or SPEC_PATH)
    if not spec_path.is_file():
        raise SweepError(f"spec not found: {spec_path}")
    try:
        raw = json.loads(spec_path.read_text())
    except json.JSONDecodeError as e:
        raise SweepError(f"spec is not valid JSON: {e}") from e
    extra = raw.get("upload_extra", [])
    for rel in (STAGE_HEADER_REL, STAGE_WARMSTART_REL):
        if rel not in extra:
            raise SweepError(f"spec upload_extra must include {rel}")
    jobs = raw.get("jobs", [])
    if not jobs:
        raise SweepError("spec has no jobs")
    first = jobs[0].get("command", "")
    if STAGE_HEADER_REL not in first or REMOTE_HEADER_DEST not in first:
        raise SweepError(
            "spec's first job must copy the uploaded header "
            f"{STAGE_HEADER_REL} over {REMOTE_HEADER_DEST} before the build")
    train = [j for j in jobs if STAGE_WARMSTART_REL in j.get("command", "")]
    if not train:
        raise SweepError(f"spec's train job must --resume {STAGE_WARMSTART_REL}")
    return raw


def stage_rel_for(cfg_index=None):
    """Repo-relative staging dir: shared by default, per-config in parallel."""
    return DEFAULT_STAGE_REL if cfg_index is None else f"{DEFAULT_STAGE_REL}-{cfg_index}"


def derive_spec(base_raw, cfg_index=None, train_wall_s=DEFAULT_TRAIN_WALL_S,
                fast_build=False):
    """Per-config spec derived in-memory from the base sweep-train spec.

    cfg_index is not None (parallel mode): unique spec name (-> unique run
    dir stamp suffix) and per-config staging paths runs/sweep-tmp-<i>/ in
    upload_extra and every job command. train_wall_s flows into the train
    job's --max-wall-s and its remote timeout. fast_build prepends
    SWEEP_FAST_BUILD=1 so the build job compiles only libduck_cuda_fast.so.
    """
    raw = copy.deepcopy(base_raw)
    if not 60 <= int(train_wall_s) <= LAUNCHER_WALL_CAP_S - TRAIN_TIMEOUT_SLACK_S:
        raise SweepError(f"--train-wall-s must be in [60, "
                         f"{LAUNCHER_WALL_CAP_S - TRAIN_TIMEOUT_SLACK_S}]")
    if cfg_index is not None:
        raw["name"] = f"{base_raw['name']}-cfg{cfg_index}"
        old, new = DEFAULT_STAGE_REL + "/", stage_rel_for(cfg_index) + "/"
        raw["upload_extra"] = [x.replace(old, new) for x in raw.get("upload_extra", [])]
        for job in raw["jobs"]:
            job["command"] = job["command"].replace(old, new)
    for job in raw["jobs"]:
        if job["name"] == "train":
            if not re.search(r"--max-wall-s \d+", job["command"]):
                raise SweepError("base spec's train job lacks --max-wall-s")
            job["command"] = re.sub(r"--max-wall-s \d+",
                                    f"--max-wall-s {int(train_wall_s)}",
                                    job["command"])
            job["timeout_s"] = min(int(train_wall_s) + TRAIN_TIMEOUT_SLACK_S,
                                   LAUNCHER_WALL_CAP_S)
        if fast_build and job["name"] == "build":
            if "SWEEP_FAST_BUILD" not in job["command"]:
                raise SweepError("base spec's build job lacks the "
                                 "SWEEP_FAST_BUILD guard")
            job["command"] = "export SWEEP_FAST_BUILD=1; " + job["command"]
    return raw


# --------------------------------------------------------------------------
# Per-config plumbing
# --------------------------------------------------------------------------

def config_env(base, per_mps, base_env=None):
    """Inherited environment plus the two phase-clock vars. Never printed."""
    env = dict(os.environ if base_env is None else base_env)
    env[ENV_BASE] = repr(float(base))
    env[ENV_PER_MPS] = repr(float(per_mps))
    return env


def regenerate_header(base, per_mps, out_path, runner=None, log=print):
    """Run the model generator in a .venv subprocess with the config's env
    vars set (flat.py reads them at import, so the header bakes them)."""
    runner = runner or subprocess.run
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(VENV_PY), "-B", str(GENERATOR), "--output", str(out_path)]
    log(f"[sweep] regenerating header ({ENV_BASE}={float(base)!r} "
        f"{ENV_PER_MPS}={float(per_mps)!r}) -> {out_path}")
    with _GEN_LOCK:   # generator shares the native build cache; no races
        proc = runner(cmd, env=config_env(base, per_mps), cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise SweepError(f"header generation failed (exit {proc.returncode})")
    if not out_path.is_file():
        raise SweepError(f"generator did not write {out_path}")
    text = out_path.read_text()
    for macro, value in ((f"#define DW_PHASE_HZ_BASE {float(base)!r}", base),
                         (f"#define DW_PHASE_HZ_PER_MPS {float(per_mps)!r}", per_mps)):
        if macro not in text:
            raise SweepError(f"regenerated header missing expected line: {macro}")
    return out_path


def stage_warmstart(src, stage_rel=DEFAULT_STAGE_REL):
    src = Path(src)
    if not src.is_file():
        raise SweepError(f"warmstart checkpoint not found: {src}")
    dest = REPO_ROOT / stage_rel / "warmstart.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return dest


def launch_leg(spec_path=None, dry_run=False, label_suffix=None,
               runs_dir=None, runner=None, log=print):
    """Invoke gpu/run_daytona.py with the SAME interpreter this process was
    started with and the inherited environment (Doppler-provided key stays in
    env only; nothing env-related is ever printed). Returns the exit code."""
    runner = runner or subprocess.run
    cmd = [sys.executable, "-B", str(LAUNCHER), "run",
           "--spec", str(spec_path or SPEC_PATH)]
    if label_suffix:
        cmd += ["--label-suffix", str(label_suffix)]
    if runs_dir:
        cmd += ["--runs-dir", str(runs_dir)]
    if dry_run:
        cmd.append("--dry-run")
    log("[sweep] launching: " + " ".join(cmd))
    proc = runner(cmd, cwd=str(REPO_ROOT))
    return int(proc.returncode)


def list_run_dirs(runs_dir=None, pattern=RUN_DIR_GLOB):
    runs_dir = Path(runs_dir or RUNS_GPU_DIR)
    if not runs_dir.is_dir():
        return set()
    return {p for p in runs_dir.glob(pattern) if p.is_dir()}


def find_new_run_dir(before, runs_dir=None, pattern=RUN_DIR_GLOB):
    """Newest <runs_dir>/<stamp>-<spec-name> dir not present before launch."""
    new = list_run_dirs(runs_dir, pattern) - set(before)
    if not new:
        return None
    return max(new, key=lambda p: p.name)


def run_eval(base, per_mps, actor_path, out_json, runner=None, log=print):
    """Strict local eval in a fresh .venv subprocess (env vars = config)."""
    runner = runner or subprocess.run
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(VENV_PY), "-B", "-c", EVAL_SCRIPT,
           str(actor_path), str(EVAL_LIBRARY), str(out_json),
           str(EVAL_SEED), ",".join(f"{c:g}" for c in EVAL_COMMANDS),
           str(EVAL_SECONDS)]
    log(f"[sweep] strict eval: {actor_path}")
    proc = runner(cmd, env=config_env(base, per_mps), cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise SweepError(f"strict eval failed (exit {proc.returncode})")
    if not out_json.is_file():
        raise SweepError(f"strict eval wrote no output: {out_json}")
    return json.loads(out_json.read_text())


# --------------------------------------------------------------------------
# Metrics and ranking
# --------------------------------------------------------------------------

def longest_alternating(feet):
    """Longest run of consecutive qualified footfalls on alternating feet."""
    if not feet:
        return 0
    best = cur = 1
    for a, b in zip(feet, feet[1:]):
        cur = cur + 1 if a != b else 1
        best = max(best, cur)
    return best


def summarize_eval(episodes):
    """Per-command pass + qualified L/R + longest alternating run, plus
    config-level aggregates used for ranking."""
    per_command = {}
    agg = {"commands_passed": 0, "qualified_left": 0, "qualified_right": 0,
           "total_qualified": 0, "alt_sum": 0}
    for ep in episodes:
        feet = [f["foot"] for f in ep.get("footfalls", [])]
        row = {
            "passed": bool(ep.get("passed")),
            "qualified_left": feet.count("left"),
            "qualified_right": feet.count("right"),
            "longest_alternating": longest_alternating(feet),
            "total_qualified": len(feet),
        }
        cmd = ep.get("command_mps")
        per_command[f"{float(cmd):.2f}" if cmd is not None else "?"] = row
        agg["commands_passed"] += int(row["passed"])
        agg["qualified_left"] += row["qualified_left"]
        agg["qualified_right"] += row["qualified_right"]
        agg["total_qualified"] += row["total_qualified"]
        agg["alt_sum"] += row["longest_alternating"]
    return dict(per_command=per_command, **agg)


def rank_results(records):
    """Completed configs ranked by (#commands passed, sum of longest
    alternating runs) descending; failed configs after, in sweep order."""
    done = [r for r in records if r.get("status") == "ok" and r.get("metrics")]
    done_ids = {id(r) for r in done}
    rest = [r for r in records if id(r) not in done_ids]
    done.sort(key=lambda r: (-r["metrics"]["commands_passed"],
                             -r["metrics"]["alt_sum"]))
    return done + rest


# --------------------------------------------------------------------------
# Table output
# --------------------------------------------------------------------------

_TABLE_HEADER = (f"{'config (base,per)':<20} {'status':<14} "
                 f"{'0.10':<5} {'0.15':<5} {'0.20':<5} "
                 f"{'qL/qR':<7} {'altSum':<7} {'totalQ':<6}")


def table_row(rec):
    base, per = rec["config"]
    cfg = f"({base:g}, {per:g})"
    m = rec.get("metrics")
    if not m:
        return f"{cfg:<20} {rec.get('status', '?'):<14} {'-':<5} {'-':<5} {'-':<5} {'-':<7} {'-':<7} {'-':<6}"
    marks = []
    for c in EVAL_COMMANDS:
        row = m["per_command"].get(f"{c:.2f}")
        marks.append("-" if row is None else ("PASS" if row["passed"] else "fail"))
    ql_qr = f"{m['qualified_left']}/{m['qualified_right']}"
    return (f"{cfg:<20} {rec['status']:<14} {marks[0]:<5} {marks[1]:<5} "
            f"{marks[2]:<5} {ql_qr:<7} {m['alt_sum']:<7} {m['total_qualified']:<6}")


# --------------------------------------------------------------------------
# Per-config driver
# --------------------------------------------------------------------------

def run_config(base, per_mps, warmstart, out_dir, legs=1, dry_run=False,
               runner=None, log=print, *, spec_path=None, spec_name="sweep-train",
               stage_rel=DEFAULT_STAGE_REL, label_suffix=None, runs_dir=None):
    """Header -> stage -> leg(s) -> strict eval; returns the config record.
    Any leg failure is recorded and ends this config without raising.

    In --parallel mode each config gets its own spec_path / spec_name /
    stage_rel / label_suffix so N configs can run in N concurrent sandboxes
    without sharing staging files, run-dir detection globs, or the
    launcher's one-sandbox-per-label concurrency guard."""
    tag = f"base{base:g}-per{per_mps:g}"
    rec = {"schema": "duckgridwalk.sweep_config/1",
           "config": [base, per_mps], "tag": tag,
           "status": None, "legs": [], "metrics": None,
           "label_suffix": label_suffix, "spec": str(spec_path or SPEC_PATH)}
    run_pattern = f"*-{spec_name}"
    header_path = REPO_ROOT / stage_rel / "duck_model.h"
    try:
        regenerate_header(base, per_mps, header_path, runner=runner, log=log)
    except SweepError as e:
        rec["status"], rec["error"] = "header_failed", str(e)
        return rec

    resume_src = Path(warmstart)
    last_success = None
    for leg in range(int(legs)):
        leg_rec = {"leg": leg, "resume": str(resume_src)}
        rec["legs"].append(leg_rec)
        try:
            stage_warmstart(resume_src, stage_rel)
        except SweepError as e:
            rec["status"], leg_rec["error"] = "leg_failed", str(e)
            return rec
        if dry_run:
            rc = launch_leg(spec_path, dry_run=True, label_suffix=label_suffix,
                            runs_dir=runs_dir, runner=runner, log=log)
            leg_rec.update(dry_run=True, launch_exit=rc)
            if rc != 0:
                rec["status"], rec["error"] = "dry_run_failed", f"launcher dry-run exit {rc}"
                return rec
            continue
        before = list_run_dirs(runs_dir, run_pattern)
        rc = launch_leg(spec_path, label_suffix=label_suffix,
                        runs_dir=runs_dir, runner=runner, log=log)
        run_dir = find_new_run_dir(before, runs_dir, run_pattern)
        leg_rec.update(launch_exit=rc,
                       run_dir=str(run_dir) if run_dir else None)
        if rc != 0 or run_dir is None:
            leg_rec["error"] = (f"launcher exit {rc}" if run_dir else
                                f"launcher exit {rc}; no run dir found")
            log(f"[sweep] config {tag} leg {leg} FAILED ({leg_rec['error']}); "
                "continuing with the next config")
            break
        actor = run_dir / ACTOR_REL
        if not actor.is_file():
            leg_rec["error"] = f"missing artifact {ACTOR_REL}"
            log(f"[sweep] config {tag} leg {leg} FAILED ({leg_rec['error']})")
            break
        last_success = run_dir
        latest = run_dir / LATEST_REL
        if latest.is_file():
            resume_src = latest  # chain legs from the full checkpoint

    if dry_run:
        rec["status"] = "dry_run_ok"
        return rec
    if last_success is None:
        rec["status"] = "leg_failed"
        return rec

    try:
        episodes = run_eval(base, per_mps, last_success / ACTOR_REL,
                            Path(out_dir) / f"eval-{tag}.json",
                            runner=runner, log=log)
        rec["metrics"] = summarize_eval(episodes)
        rec["eval_json"] = str(Path(out_dir) / f"eval-{tag}.json")
        rec["status"] = "ok"
    except SweepError as e:
        rec["status"], rec["error"] = "eval_failed", str(e)
    return rec


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _build_parser():
    ap = argparse.ArgumentParser(
        prog="sweep.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", required=True,
                    help="JSON list of [phase_hz_base, phase_hz_per_mps] pairs")
    ap.add_argument("--warmstart", required=True,
                    help="full training checkpoint to resume each config from")
    ap.add_argument("--out", required=True,
                    help="output dir for sweep-results.jsonl / summary.json")
    ap.add_argument("--legs-per-config", type=int, default=1,
                    help="sequential 15-min training legs per config (default 1)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="run N configs in N concurrent sandboxes (unique label "
                         "suffix, staging dir and derived spec per config; "
                         "default 1 = sequential)")
    ap.add_argument("--train-wall-s", type=int, default=DEFAULT_TRAIN_WALL_S,
                    help="per-leg training wall clock (flows into the derived "
                         f"spec's --max-wall-s and job timeout; default "
                         f"{DEFAULT_TRAIN_WALL_S})")
    ap.add_argument("--fast-build", action="store_true",
                    help="remote build compiles only libduck_cuda_fast.so "
                         "(SWEEP_FAST_BUILD=1 in the build job command)")
    ap.add_argument("--runs-dir", default=None,
                    help="base dir for launcher run dirs (default: runs/gpu)")
    ap.add_argument("--dry-run", action="store_true",
                    help="per config: regenerate header + validate spec via the "
                         "launcher's --dry-run; no sandbox, no eval")
    return ap


def main(argv=None, runner=None, log=print):
    args = _build_parser().parse_args(argv)
    try:
        configs = parse_configs(args.configs)
        base_raw = validate_spec()
        if not Path(args.warmstart).is_file():
            raise SweepError(f"warmstart checkpoint not found: {args.warmstart}")
        if args.legs_per_config < 1:
            raise SweepError("--legs-per-config must be >= 1")
        if args.parallel < 1:
            raise SweepError("--parallel must be >= 1")
    except SweepError as e:
        log(f"[sweep] error: {e}")
        return EXIT_ERROR

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "sweep-results.jsonl"
    mode = "dry-run" if args.dry_run else "live"
    parallel_mode = args.parallel > 1
    runs_dir = (REPO_ROOT / args.runs_dir) if args.runs_dir else None

    # One derived spec file per config (written under <out>/specs). In
    # sequential mode it is content-identical to the base spec unless
    # --train-wall-s / --fast-build override it.
    spec_dir = out_dir / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    try:
        for i, (base, per_mps) in enumerate(configs):
            idx = i if parallel_mode else None
            raw = derive_spec(base_raw, cfg_index=idx,
                              train_wall_s=args.train_wall_s,
                              fast_build=args.fast_build)
            spec_path = spec_dir / f"cfg{i}-{raw['name']}.json"
            spec_path.write_text(json.dumps(raw, indent=2) + "\n")
            tasks.append({"index": i, "base": base, "per_mps": per_mps,
                          "spec_path": spec_path, "spec_name": raw["name"],
                          "stage_rel": stage_rel_for(idx),
                          "label_suffix": f"cfg{i}" if parallel_mode else None})
    except SweepError as e:
        log(f"[sweep] error: {e}")
        return EXIT_ERROR

    log(f"[sweep] {len(configs)} config(s), {args.legs_per_config} leg(s) each, "
        f"mode={mode}, parallel={args.parallel}, train_wall_s={args.train_wall_s}, "
        f"fast_build={args.fast_build}, out={out_dir}")
    log("[sweep] " + _TABLE_HEADER)

    io_lock = threading.Lock()

    def emit(msg):
        with io_lock:
            log(msg)

    def one(task):
        return run_config(task["base"], task["per_mps"], args.warmstart, out_dir,
                          legs=args.legs_per_config, dry_run=args.dry_run,
                          runner=runner, log=emit,
                          spec_path=task["spec_path"],
                          spec_name=task["spec_name"],
                          stage_rel=task["stage_rel"],
                          label_suffix=task["label_suffix"],
                          runs_dir=runs_dir)

    by_index = {}
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(one, t): t for t in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:  # a crashed config must not sink the others
                rec = {"schema": "duckgridwalk.sweep_config/1",
                       "config": [task["base"], task["per_mps"]],
                       "tag": f"base{task['base']:g}-per{task['per_mps']:g}",
                       "status": "error", "error": f"{type(e).__name__}: {e}",
                       "legs": [], "metrics": None,
                       "label_suffix": task["label_suffix"]}
            rec["at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            by_index[task["index"]] = rec
            with io_lock:   # live row + jsonl append as each config finishes
                with results_path.open("a") as f:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
                log("[sweep] " + table_row(rec))
    records = [by_index[i] for i in sorted(by_index)]

    ranked = rank_results(records)
    log("[sweep] " + "=" * len(_TABLE_HEADER))
    log("[sweep] RANKED SUMMARY (by #commands passed, then sum of longest "
        "alternating runs)")
    log("[sweep] " + _TABLE_HEADER)
    for rec in ranked:
        log("[sweep] " + table_row(rec))

    completed = [r for r in records if r["status"] in ("ok", "dry_run_ok")]
    summary = {
        "schema": "duckgridwalk.sweep_summary/1",
        "mode": mode,
        "legs_per_config": args.legs_per_config,
        "parallel": args.parallel,
        "train_wall_s": args.train_wall_s,
        "fast_build": args.fast_build,
        "warmstart": str(args.warmstart),
        "configs_total": len(records),
        "configs_completed": len(completed),
        "rank_key": ["commands_passed", "alt_sum"],
        "ranked": ranked,
        "results_jsonl": str(results_path),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    log(f"[sweep] wrote {out_dir / 'summary.json'} "
        f"({len(completed)}/{len(records)} config(s) completed)")
    return EXIT_OK if completed else EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
