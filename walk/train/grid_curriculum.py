#!/usr/bin/env python3
"""Cube-terrain curriculum runner (CPU, local Mac cores).

Fine-tunes the accepted flat-floor walker across the grid ladder defined in
walk/eval/grid_acceptance.py (flush -> rough4 -> rough8) by resuming
walk.train.run on CubeGridDuckEnv, one stage at a time:

  1. Bootstrap: wrap the accepted actor file ({"arch","state_dict"}) into a
     full walk.train.run checkpoint (fresh critic/optimizer/generator) so the
     trainer's --resume contract applies unchanged.
  2. Per stage: run the trainer in bounded chunks of --poll-updates PPO
     updates (each chunk is one walk.train.run subprocess resuming
     <stage>/latest.pt; run.py's checkpoint-boundary reset contract makes the
     chunked run reproduce a single long run bitwise). After each chunk the
     strict grid judge (walk.eval.grid_acceptance) runs on a snapshot of
     latest.pt in a subprocess. Pass => copy accepted.pt, warm-start the next
     stage from it. --max-hours per stage bounds the loop.
  3. Every event is appended to <out>/curriculum.jsonl.

Why chunked instead of a concurrent judge: the grid lane is CPU-heavy
(~0.15 s per env policy-step), so an in-flight judge would fight the 8
trainer workers for cores; chunking keeps the machine usable and keeps the
train stream deterministic per run.py's resume contract.

Usage (stage 1 from the accepted walker):
  .venv/bin/python -B -m walk.train.grid_curriculum \
      --init-actor evidence/walking-accepted-20260901/actor-walking-v1.pt \
      --out runs/grid-curriculum --workers 8 --max-hours 12
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from walk.eval.grid_acceptance import STAGES, stage_grid

DEFAULT_STAGES = ("flush", "rough4", "rough8")
DEFAULT_INIT_ACTOR = "evidence/walking-accepted-20260901/actor-walking-v1.pt"


# ---------------------------------------------------------------------------
# Warm-start bootstrap
# ---------------------------------------------------------------------------
def make_warmstart_checkpoint(actor_path: str | Path, out_path: str | Path,
                              seed: int, lr: float) -> Path:
    """Wrap an actor file into a walk.train.run-resumable checkpoint.

    Accepts the accepted-policy format ({"arch":"ff","state_dict":...}), a
    legacy plain state_dict, or a full trainer checkpoint (whose actor is
    reused as-is). Critic and optimizer start fresh; the torch generator is
    seeded exactly like a fresh run.py run (derive_seed(seed, 0x22))."""
    import torch

    from walk.train.ppo import PPOConfig, make_nets, unpack_actor_file
    from walk.train.vec import derive_seed

    raw = torch.load(str(actor_path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "actor" in raw:        # full checkpoint
        raw = raw["actor"]
    arch, sd = unpack_actor_file(raw)
    if arch != "ff":
        raise ValueError(f"grid curriculum warm start requires an ff actor, got {arch!r}")
    actor, critic = make_nets(58, 14, PPOConfig())
    actor.load_state_dict(sd)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=lr)
    gen = torch.Generator()
    gen.manual_seed(derive_seed(seed, 0x22))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    torch.save({
        "update": 0,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "gen_state": gen.get_state(),
        "env_steps": 0,
        "faults_total": 0,
        "config": {"warmstart_from": str(actor_path)},
    }, tmp)
    tmp.replace(out_path)
    return out_path


def checkpoint_update(path: str | Path) -> int:
    """The PPO update counter stored in a trainer checkpoint."""
    import torch
    return int(torch.load(str(path), map_location="cpu",
                          weights_only=False)["update"])


# ---------------------------------------------------------------------------
# Subprocess command builders (pure; unit-tested)
# ---------------------------------------------------------------------------
def build_env_kwargs(stage: str, environments: int,
                     impulse_tolerance: float | None = None) -> dict:
    kw = {"environments": int(environments), "grid": stage_grid(stage)}
    if impulse_tolerance is not None:
        kw["impulse_tolerance"] = float(impulse_tolerance)
    return kw


def trainer_cmd(stage: str, stage_dir: Path, target_updates: int, args) -> list[str]:
    """walk.train.run invocation for one chunk, resuming <stage_dir>/latest.pt."""
    kw = build_env_kwargs(stage, args.envs_per_worker, args.impulse_tolerance)
    return [
        sys.executable, "-B", "-m", "walk.train.run",
        "--env", "walk.env.grid:CubeGridDuckEnv",
        "--env-kwargs", json.dumps(kw),
        "--workers", str(args.workers),
        "--horizon", str(args.horizon),
        "--updates", str(target_updates),
        "--seed", str(args.seed),
        "--out", str(stage_dir),
        "--resume",                       # bare flag => <out>/latest.pt
        "--checkpoint-every", str(args.checkpoint_every),
        "--eval-every", "0",
        "--preflight-steps", "0",         # resumed runs skip it anyway
        "--torch-threads", str(args.torch_threads),
        "--lr", str(args.lr),
        *(["--target-kl", str(args.target_kl)] if args.target_kl else []),
    ]


def judge_cmd(actor_ckpt: Path, stage: str, out_dir: Path, args) -> list[str]:
    return [
        sys.executable, "-B", "-m", "walk.eval.grid_acceptance",
        "--actor", str(actor_ckpt),
        "--stage", stage,
        "--out", str(out_dir),
        "--jobs", str(args.judge_jobs),
    ]


# ---------------------------------------------------------------------------
# Stage loop (subprocess runners injectable for tests)
# ---------------------------------------------------------------------------
def _log_event(log_path: Path, event: dict) -> None:
    event = dict(event, time=time.time())
    with log_path.open("a") as f:
        f.write(json.dumps(event) + "\n")


def _run_trainer_chunk(cmd: list[str]) -> int:
    return subprocess.run(cmd).returncode


def _run_judge(cmd: list[str]) -> int:
    return subprocess.run(cmd).returncode


def run_stage(stage: str, stage_dir: Path, start_ckpt: Path, args,
              log_path: Path, run_trainer=_run_trainer_chunk,
              run_judge=_run_judge, clock=time.monotonic,
              read_update=checkpoint_update) -> bool:
    """Train `stage` until the grid judge passes it or --max-hours elapses.

    Returns True iff the stage was accepted; on success
    <stage_dir>/accepted.pt is the passing checkpoint."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    latest = stage_dir / "latest.pt"
    if not latest.exists():
        shutil.copy2(start_ckpt, latest)
    deadline = clock() + args.max_hours * 3600.0
    _log_event(log_path, {"event": "stage_start", "stage": stage,
                          "start_checkpoint": str(start_ckpt),
                          "update": read_update(latest),
                          "grid": stage_grid(stage),
                          "max_hours": args.max_hours})

    # Judge the warm-start first: a stage the incoming policy already passes
    # (e.g. after flat training, or when re-entering a finished stage) must
    # advance without burning a training chunk.
    chunk = 0
    while True:
        update = read_update(latest)
        snapshot = stage_dir / f"judge-u{update:06d}.pt"
        shutil.copy2(latest, snapshot)
        judge_out = stage_dir / f"judge-u{update:06d}"
        rc = run_judge(judge_cmd(snapshot, stage, judge_out, args))
        passed = rc == 0
        _log_event(log_path, {"event": "judge", "stage": stage,
                              "update": update, "passed": passed,
                              "judge_out": str(judge_out / "grid_acceptance.json")})
        if passed:
            shutil.copy2(snapshot, stage_dir / "accepted.pt")
            _log_event(log_path, {"event": "stage_pass", "stage": stage,
                                  "update": update,
                                  "accepted": str(stage_dir / "accepted.pt")})
            return True
        snapshot.unlink(missing_ok=True)   # keep only passing snapshots
        if clock() >= deadline:
            _log_event(log_path, {"event": "stage_timeout", "stage": stage,
                                  "update": update,
                                  "hours": args.max_hours})
            return False

        chunk += 1
        target = update + args.poll_updates
        cmd = trainer_cmd(stage, stage_dir, target, args)
        _log_event(log_path, {"event": "train_chunk", "stage": stage,
                              "chunk": chunk, "from_update": update,
                              "to_update": target, "cmd": cmd})
        rc = run_trainer(cmd)
        if rc != 0:
            _log_event(log_path, {"event": "trainer_failed", "stage": stage,
                                  "chunk": chunk, "returncode": rc})
            raise RuntimeError(f"trainer chunk failed (rc={rc}) on stage {stage}")


def run_curriculum(args, run_trainer=_run_trainer_chunk, run_judge=_run_judge,
                   clock=time.monotonic, read_update=checkpoint_update) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "curriculum.jsonl"

    # Warm the dwv1 build cache once in this process: worker processes all
    # call world.build() at env construction, and racing cold compiles of
    # the same cached .dylib would corrupt it.
    from walk.env import world
    print(f"[curriculum] libduck_world: {world.build()}")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; available: {sorted(STAGES)}")

    warmstart = out / "warmstart.pt"
    if not warmstart.exists():
        make_warmstart_checkpoint(args.init_actor, warmstart, args.seed, args.lr)
        _log_event(log_path, {"event": "warmstart", "from": str(args.init_actor),
                              "checkpoint": str(warmstart)})

    start_ckpt = warmstart
    for stage in stages:
        stage_dir = out / stage
        ok = run_stage(stage, stage_dir, start_ckpt, args, log_path,
                       run_trainer=run_trainer, run_judge=run_judge,
                       clock=clock, read_update=read_update)
        if not ok:
            print(f"[curriculum] stage '{stage}' NOT passed within "
                  f"{args.max_hours} h; stopping. Resume with the same "
                  f"command to continue from {stage_dir / 'latest.pt'}.")
            return 1
        print(f"[curriculum] stage '{stage}' ACCEPTED "
              f"({stage_dir / 'accepted.pt'})")
        start_ckpt = stage_dir / "accepted.pt"
    print("[curriculum] all stages accepted")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="walk.train.grid_curriculum",
                                description=__doc__.splitlines()[0])
    p.add_argument("--init-actor", default=DEFAULT_INIT_ACTOR,
                   help="actor file or checkpoint to warm-start stage 1 from")
    p.add_argument("--out", default="runs/grid-curriculum")
    p.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                   help="comma list; subset/reorder to resume mid-ladder")
    p.add_argument("--workers", type=int, default=8,
                   help="CPU trainer workers (keep <= 8 on the Mac)")
    p.add_argument("--envs-per-worker", type=int, default=2,
                   help="small shards on purpose: a SolverFault poisons its "
                        "whole worker shard's rollout window, and the "
                        "zero-shot baseline shows the flat policy faults "
                        "roughly every ~150 grid env-steps, so big shards "
                        "would train on almost nothing")
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--seed", type=int, default=917)
    p.add_argument("--lr", type=float, default=7.5e-5,
                   help="fine-tune lr (the GPU continue legs' value, not "
                        "run.py's fresh-run 3e-4)")
    p.add_argument("--target-kl", type=float, default=0.05,
                   help="PPO early-stop KL guard; protects the warm-started "
                        "walker while the fresh critic's advantages are "
                        "still noise (0 disables)")
    p.add_argument("--poll-updates", type=int, default=20, metavar="M",
                   help="judge latest.pt every M PPO updates")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--max-hours", type=float, default=12.0,
                   help="wall-clock budget PER STAGE")
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--judge-jobs", type=int, default=4)
    p.add_argument("--impulse-tolerance", type=float, default=None,
                   help="grid-lane civ1 tolerance override (default: lane "
                        "defaults, 1e-8 static)")
    return p


def main(argv=None) -> int:
    return run_curriculum(build_argparser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
