"""Write a gpu_train-resumable GRU checkpoint warm-started from an accepted
FEED-FORWARD actor/checkpoint (the generalist recipe's step 0).

    .venv/bin/python -B -m walk.train.gru_warmstart --robot humanoid \
        --source runs/gpu/<accepted-leg>/artifacts/train/gpu-train-out/latest.pt \
        --out runs/humanoid-generalist-warmstart.pt [--actor-out PATH] \
        [--seed 917] [--log-std FLOAT] [--lr 1e-4]

Method (walk.train.ppo.warm_start_recurrent_from_ff): the RecurrentActor /
RecurrentCritic are built with residual feed-forward trunks of the FF nets'
exact sizes; the FF weights (and log_std) are copied into the trunks and the
GRU heads' output layers are zeroed, so mu(obs, h) == FF(obs) for every h --
the warm-started policy IS the accepted specialist at step 0, and the GRU
pathway learns a zero-initialized, history-dependent correction (implicit
system ID) under domain randomization. No distillation rollouts, no BC. The
written checkpoint is a normal update-0 GRU checkpoint (fresh Adam over all
parameters, seeded generators, config.policy = gru), so
`gpu_train --robot <robot> --policy gru --resume <out>` resumes it like any
other GRU leg, and the acceptance harness loads the actor file unchanged
(the trunk is rebuilt from the state dict's ff.* keys on load).

The script verifies the equivalence before writing: deterministic actions
of the GRU (any hidden state) vs the FF actor on random observations must
match exactly (max |diff| printed; asserted <= 1e-6, measured 0.0).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from walk.train.gpu_train import robot_classes
from walk.train.ppo import (
    Actor,
    PPOConfig,
    make_recurrent_nets,
    trunk_hidden_from_state_dict,
    unpack_actor_file,
    warm_start_recurrent_from_ff,
)
from walk.train.vec import derive_seed


def load_ff_source(path: Path):
    """(actor_sd, critic_sd|None, meta) from an FF actor file or checkpoint."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "actor" in raw and "critic" in raw:
        arch = str(raw.get("config", {}).get("policy", "ff"))
        if arch != "ff":
            raise SystemExit(f"{path} is a {arch} checkpoint, not feed-forward")
        return raw["actor"], raw["critic"], {
            "update": int(raw.get("update", 0)),
            "env_steps": int(raw.get("env_steps", 0))}
    arch, sd = unpack_actor_file(raw)
    if arch != "ff":
        raise SystemExit(f"{path} is a {arch} actor, not feed-forward")
    return sd, None, {}


def build_warm_started(robot: str, actor_sd: dict, critic_sd: dict | None,
                       seed: int, log_std: float | None = None):
    """(actor, critic, ff_hidden): residual GRU nets equal to the FF source."""
    obs_dim, act_dim, _, _ = robot_classes(robot)
    ff_hidden = trunk_hidden_from_state_dict(actor_sd, "mu_net.")
    if ff_hidden is None:
        raise SystemExit("source has no mu_net.* feed-forward layers")
    torch.manual_seed(derive_seed(seed, 0x11))
    actor, critic = make_recurrent_nets(obs_dim, act_dim, PPOConfig(),
                                        ff_hidden=ff_hidden)
    warm_start_recurrent_from_ff(actor, actor_sd, critic, critic_sd)
    if log_std is not None:
        with torch.no_grad():
            actor.log_std.fill_(float(log_std))
    return actor, critic, ff_hidden


def verify_equivalence(robot: str, actor, actor_sd: dict, n: int = 256) -> float:
    obs_dim, act_dim, _, _ = robot_classes(robot)
    ff = Actor(obs_dim, act_dim, trunk_hidden_from_state_dict(actor_sd, "mu_net."))
    ff.load_state_dict(actor_sd)
    g = torch.Generator().manual_seed(0)
    obs = torch.randn(n, obs_dim, generator=g)
    h = torch.randn(n, actor.gru_hidden, generator=g)     # ANY hidden state
    with torch.no_grad():
        a_gru, _ = actor.deterministic(obs, h)
        a_ff = ff.deterministic(obs)
    return float((a_gru - a_ff).abs().max())


def save_resume_checkpoint(robot: str, actor, critic, path: Path, seed: int,
                           lr: float, source: str, ff_hidden, meta: dict) -> None:
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=lr)
    sample_gen = torch.Generator().manual_seed(derive_seed(seed, 0x22))
    perm_gen = torch.Generator().manual_seed(derive_seed(seed, 0x33))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "update": 0,
        "actor": actor.cpu().state_dict(),
        "critic": critic.cpu().state_dict(),
        "optimizer": optimizer.state_dict(),
        "sample_gen_state": sample_gen.get_state(),
        "sample_gen_device": "cpu",
        "perm_gen_state": perm_gen.get_state(),
        "env_steps": 0,
        "faults_total": 0,
        "train_wall_s": 0.0,
        "probe_wall_s": 0.0,
        "config": {"policy": "gru", "robot": robot,
                   "warm_start_from": str(source),
                   "warm_start_source_update": meta.get("update"),
                   "warm_start_source_env_steps": meta.get("env_steps"),
                   "gru_ff_trunk": list(ff_hidden),
                   "critic_warm": critic.ff is not None},
    }, path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="walk.train.gru_warmstart",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--robot", choices=("duck", "humanoid"), required=True)
    p.add_argument("--source", required=True,
                   help="FF actor file or full FF checkpoint (latest.pt)")
    p.add_argument("--out", required=True, help="GRU resume checkpoint path")
    p.add_argument("--actor-out", default=None,
                   help="also write the self-describing GRU actor file")
    p.add_argument("--seed", type=int, default=917)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Adam lr recorded in the fresh optimizer state")
    p.add_argument("--log-std", type=float, default=None,
                   help="override the copied FF log_std (default: keep)")
    a = p.parse_args(argv)
    actor_sd, critic_sd, meta = load_ff_source(Path(a.source))
    actor, critic, ff_hidden = build_warm_started(a.robot, actor_sd, critic_sd,
                                                  a.seed, a.log_std)
    diff = verify_equivalence(a.robot, actor, actor_sd)
    if diff > 1e-6:
        raise SystemExit(f"warm start is NOT equivalent to the FF actor: {diff}")
    save_resume_checkpoint(a.robot, actor, critic, Path(a.out), a.seed, a.lr,
                           a.source, ff_hidden, meta)
    if a.actor_out:
        Path(a.actor_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"arch": "gru", "state_dict": actor.cpu().state_dict()},
                   a.actor_out)
    n_params = sum(int(np.prod(v.shape)) for v in actor.state_dict().values())
    print(json.dumps({"out": a.out, "actor_out": a.actor_out,
                      "ff_trunk": list(ff_hidden),
                      "critic_warm": critic_sd is not None,
                      "max_abs_action_diff_vs_ff": diff,
                      "actor_params": n_params,
                      "log_std_mean": float(actor.log_std.detach().mean()),
                      "source": meta}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
