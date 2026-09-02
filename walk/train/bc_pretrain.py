"""Behavior-cloning pre-training: regress the ff Actor onto a reference
gait dataset so GPU PPO legs start from "already stepping".

Robot-parameterized through walk.train.gpu_train.robot_classes (the same
indirection the GPU trainer uses); the duck path is UNTOUCHED -- no duck
dataset provider exists, and requesting one is a clear error, not a
behavior change. Currently the only provider is the humanoid's
(humanoid/bc_dataset.py: the synthetic reference gait rolled closed-loop
over real fp32-lane observations).

    .venv/bin/python -B -m walk.train.bc_pretrain --robot humanoid \
        --out humanoid/bc_init.pt [--checkpoint-out humanoid/bc_init_ckpt.pt] \
        [--epochs 200] [--seed 0] [--seeds 11,22,33,44] \
        [--envs-per-config 4] [--steps 60] [--log-std -1.0] [--quiet]

--checkpoint-out additionally writes a gpu_train --resume-compatible
update-0 checkpoint (BC actor + fresh critic/optimizer/generators): the
turnkey handoff for the flagship GPU leg.

Training: full-batch-shuffled minibatch MSE between tanh(mu(obs)) and the
recorded actions, labels clamped to +-0.98 (the reference knee plateau
saturates at exactly 1.0, which tanh can only reach at mu=inf; the 2%
shave bounds the targets while changing the commanded knee by 0.01 rad).
Adam, deterministic given --seed.

log_std handoff (--log-std, default -1.0): the saved actor carries
log_std for PPO to start from. At the ppo.py default -0.5 (pre-tanh std
0.61) exploration noise moves PD targets by ~0.3 rad rms -- half the
action box, which knocks over a fresh BC gait immediately; at -2.0 (std
0.14) exploration is final-polish cold and PPO cannot escape BC's local
flaws. -1.0 (std 0.37) perturbs targets by roughly the env's own slew
step (0.16 rad), i.e. one-step-recoverable exploration around the cloned
gait; the v2.1 phase/imitation terms re-anchor it.

Output: the standard self-describing actor file
{"arch": "ff", "state_dict": ...} (walk.train.ppo.unpack_actor_file), so
gpu_train --resume-style tooling, the acceptance harness and the
evaluators all load it unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from walk.train.gpu_train import robot_classes
from walk.train.ppo import PPOConfig, make_nets
from walk.train.vec import derive_seed

ROOT = Path(__file__).resolve().parents[2]


def humanoid_dataset(args) -> dict:
    if str(ROOT / "humanoid") not in sys.path:
        sys.path.insert(0, str(ROOT / "humanoid"))
    import bc_dataset  # noqa: PLC0415
    seeds = tuple(int(s) for s in str(args.seeds).split(","))
    return bc_dataset.build_dataset(
        seeds=seeds, envs_per_config=args.envs_per_config, steps=args.steps,
        variant=getattr(args, "variant", None))


_DATASET_BUILDERS = {"humanoid": humanoid_dataset}


def train_bc(robot: str, dataset: dict, epochs: int = 200,
             batch_size: int = 256, lr: float = 1e-3, seed: int = 0,
             log_std: float = -1.0, quiet: bool = False,
             variant: str | None = None):
    """(actor, history): MSE-regress the robot's ff actor on the dataset."""
    obs_dim, act_dim, _, _ = robot_classes(robot, variant)
    obs = torch.from_numpy(np.ascontiguousarray(dataset["obs"], np.float32))
    act = torch.from_numpy(np.ascontiguousarray(
        np.clip(dataset["act"], -0.98, 0.98), np.float32))
    if obs.shape[1] != obs_dim or act.shape[1] != act_dim:
        raise SystemExit(f"dataset dims {tuple(obs.shape[1:])}/"
                         f"{tuple(act.shape[1:])} != robot {obs_dim}/{act_dim}")
    torch.manual_seed(derive_seed(seed, 0xBC))
    actor, _ = make_nets(obs_dim, act_dim, PPOConfig(log_std_init=log_std))
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(derive_seed(seed, 0xBC1))
    n = len(obs)
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            mu, _ = actor.dist(obs[idx])
            loss = torch.nn.functional.mse_loss(torch.tanh(mu), act[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)
        history.append(total / n)
        if not quiet and (epoch % max(1, epochs // 10) == 0
                          or epoch == epochs - 1):
            print(f"[bc] epoch {epoch:4d} mse {history[-1]:.6f}")
    return actor, history


def save_actor(actor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": "ff", "state_dict": actor.cpu().state_dict()}, path)


def save_checkpoint(robot: str, actor, path: Path, seed: int,
                    log_std: float, variant: str | None = None) -> None:
    """gpu_train --resume-compatible update-0 checkpoint: BC actor + FRESH
    critic/optimizer/generators, so the flagship leg starts PPO directly
    from the cloned gait (`gpu_train --robot <robot> --resume <path>`).
    The generator states are cpu-seeded; on a cuda host gpu_train's resume
    falls back to its documented per-update reseed (handled there)."""
    from walk.train.ppo import PPOConfig as _Cfg  # noqa: PLC0415
    obs_dim, act_dim, _, _ = robot_classes(robot, variant)
    _, critic = make_nets(obs_dim, act_dim, _Cfg(log_std_init=log_std))
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=_Cfg().lr)
    sample_gen = torch.Generator().manual_seed(derive_seed(seed, 0x22))
    perm_gen = torch.Generator().manual_seed(derive_seed(seed, 0x33))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "update": 0,
        "actor": actor.cpu().state_dict(),
        "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sample_gen_state": sample_gen.get_state(),
        "sample_gen_device": "cpu",
        "perm_gen_state": perm_gen.get_state(),
        "env_steps": 0,
        "faults_total": 0,
        "train_wall_s": 0.0,
        "probe_wall_s": 0.0,
        "config": {"policy": "ff", "robot": robot,
                   "bc_pretrained": True, "bc_log_std": log_std,
                   **({"variant": variant} if variant else {})},
    }, path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="walk.train.bc_pretrain",
        description="Behavior-clone the reference gait into a fresh actor")
    p.add_argument("--robot", choices=sorted(_DATASET_BUILDERS), required=True)
    p.add_argument("--variant", default=None,
                   help="humanoid family member (h1_tall | h1_stocky); the "
                        "dataset rolls on the variant's lane with its "
                        "reference table; unset = H1.1 base")
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint-out", default=None,
                   help="also write a gpu_train --resume-compatible "
                        "update-0 checkpoint (BC actor + fresh critic/"
                        "optimizer) at this path")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-std", type=float, default=-1.0,
                   help="log_std carried into PPO (see module docstring)")
    p.add_argument("--seeds", default="11,22,33,44",
                   help="comma-separated dataset env seeds")
    p.add_argument("--envs-per-config", type=int, default=4)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    t0 = time.monotonic()
    dataset = _DATASET_BUILDERS[args.robot](args)
    if not args.quiet:
        print(f"[bc] dataset: {json.dumps(dataset['meta'])}")
    actor, history = train_bc(args.robot, dataset, epochs=args.epochs,
                              batch_size=args.batch_size, lr=args.lr,
                              seed=args.seed, log_std=args.log_std,
                              quiet=args.quiet, variant=args.variant)
    save_actor(actor, Path(args.out))
    if args.checkpoint_out:
        save_checkpoint(args.robot, actor, Path(args.checkpoint_out),
                        args.seed, args.log_std, variant=args.variant)
        print(f"[bc] wrote resume checkpoint {args.checkpoint_out}")
    print(f"[bc] wrote {args.out}: {dataset['meta']['pairs']} pairs, "
          f"mse {history[0]:.5f} -> {history[-1]:.6f}, "
          f"log_std {args.log_std}, wall {time.monotonic() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
