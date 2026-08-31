"""Single-process GPU PPO trainer for the duck over the batched CUDA lane.

One FlatFloorDuckEnv holding E environments in a single CudaDuckLane (physics
on the GPU via libduck_cuda*.so; the identical serial dylib locally), with the
Actor/Critic from walk.train.ppo on a torch device (cuda on the GPU host, cpu
locally). PPO semantics match walk.train.run: same nets, same tanh-squashed
Gaussian sampling, same GAE(0.99, 0.95), same clipped update via
walk.train.ppo.ppo_update with the PPOConfig defaults.

    python -B -m walk.train.gpu_train --envs 4096 --horizon 32 --updates 300 \
        --seed 917 --device cuda --library <path-to-libduck_cuda.so> \
        --out <dir> [--resume <ckpt>] [--max-wall-s 900] [--lr 3e-4] \
        [--perturbation 0.02]

Solver faults: a SolverFault anywhere in a rollout poisons that whole rollout
window (no update is applied), the fault artifact is copied into <out>/faults/
and logged to <out>/faults.jsonl, every env is reset, and training continues.

Checkpoints: <out>/ckpt_NNNNNN.pt every --checkpoint-every updates, plus
<out>/latest.pt at every checkpoint and at exit. <out>/actor_final.pt (a plain
cpu state_dict of the actor) is ALWAYS written at exit for local evaluation.
--max-wall-s stops the update loop cleanly (checkpoint + actor_final, exit 0).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from walk.env.contract import ACT, OBS, SolverFault
from walk.env.cuda_lane import CudaDuckLane
from walk.env.flat import FlatFloorDuckEnv
from walk.train.ppo import (
    PPOConfig,
    compute_gae,
    make_nets,
    ppo_update,
    tanh_gaussian_log_prob,
)
from walk.train.vec import derive_seed


@dataclasses.dataclass
class GpuTrainConfig:
    envs: int = 4096
    horizon: int = 32
    updates: int = 300
    seed: int = 917
    device: str = "cpu"
    library: str | None = None
    out: str = "runs/gpu-train"
    resume: str | None = None
    max_wall_s: float = 0.0          # 0 disables the wall-clock stop
    lr: float = 3e-4
    perturbation: float = 0.0
    checkpoint_every: int = 25
    preflight_steps: int = 100
    torch_threads: int = 4
    quiet: bool = False


@dataclasses.dataclass
class Rollout:
    obs: torch.Tensor       # [T, E, OBS] on device
    raw_act: torch.Tensor   # [T, E, ACT] pre-tanh actions
    logp: torch.Tensor      # [T, E]
    val: torch.Tensor       # [T, E]
    rew: torch.Tensor       # [T, E]
    done: torch.Tensor      # [T, E] float32
    last_obs: torch.Tensor  # [E, OBS]
    next_obs_np: np.ndarray
    episodes: list          # list[(return, length)] finished this window


def make_env(cfg: GpuTrainConfig) -> FlatFloorDuckEnv:
    return FlatFloorDuckEnv(
        environments=cfg.envs,
        seed=cfg.seed,
        perturbation_rad=cfg.perturbation,
        lane_factory=lambda E, offsets: CudaDuckLane(
            E, joint_offsets=offsets, library_path=cfg.library),
    )


def sample_actions(actor, obs_t: torch.Tensor, gen: torch.Generator, device: torch.device):
    """Device-generator twin of ppo.Actor.sample (identical math + RNG use)."""
    mu, std = actor.dist(obs_t)
    u = mu + std * torch.randn(mu.shape, generator=gen, dtype=mu.dtype, device=device)
    return u, torch.tanh(u), tanh_gaussian_log_prob(mu, std, u)


def collect_rollout(env, actor, critic, obs_np: np.ndarray, horizon: int,
                    gen: torch.Generator, device: torch.device,
                    ep_ret: np.ndarray, ep_len: np.ndarray) -> Rollout:
    """Roll `horizon` steps with auto-reset of done envs via env.reset(mask).

    A SolverFault propagates to the caller, which drops the whole window.
    ep_ret/ep_len are mutated in place so partial episodes span rollouts.
    """
    T, E = horizon, env.E
    obs_b = torch.zeros((T, E, env.OBS), dtype=torch.float32, device=device)
    raw_b = torch.zeros((T, E, env.ACT), dtype=torch.float32, device=device)
    logp_b = torch.zeros((T, E), dtype=torch.float32, device=device)
    val_b = torch.zeros((T, E), dtype=torch.float32, device=device)
    rew_b = torch.zeros((T, E), dtype=torch.float32, device=device)
    done_b = torch.zeros((T, E), dtype=torch.float32, device=device)
    episodes: list[tuple[float, int]] = []
    with torch.no_grad():
        for t in range(T):
            obs_t = torch.from_numpy(np.ascontiguousarray(obs_np)).to(device)
            u, a, logp = sample_actions(actor, obs_t, gen, device)
            obs_b[t] = obs_t
            raw_b[t] = u
            logp_b[t] = logp
            val_b[t] = critic(obs_t)
            next_obs, rew, done, _info = env.step(a.cpu().numpy())
            rew_b[t] = torch.from_numpy(np.ascontiguousarray(rew)).to(device)
            done_b[t] = torch.from_numpy(done.astype(np.float32)).to(device)
            ep_ret += rew
            ep_len += 1
            if done.any():
                for e in np.flatnonzero(done):
                    episodes.append((float(ep_ret[e]), int(ep_len[e])))
                ep_ret[done] = 0.0
                ep_len[done] = 0
                next_obs = env.reset(mask=done)  # fresh obs for reset envs
            obs_np = next_obs
    last_obs = torch.from_numpy(np.ascontiguousarray(obs_np)).to(device)
    return Rollout(obs=obs_b, raw_act=raw_b, logp=logp_b, val=val_b, rew=rew_b,
                   done=done_b, last_obs=last_obs, next_obs_np=obs_np,
                   episodes=episodes)


def make_batch(ro: Rollout, critic, gamma: float, lam: float) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        last_val = critic(ro.last_obs)
    adv, ret = compute_gae(ro.rew, ro.done, ro.val, last_val, gamma, lam)

    def flat(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, *x.shape[2:])

    return {"obs": flat(ro.obs), "raw_act": flat(ro.raw_act), "logp": flat(ro.logp),
            "val": flat(ro.val), "adv": flat(adv), "ret": flat(ret)}


def record_fault(out: Path, ff, update: int, fault: SolverFault) -> str | None:
    """Copy the fault artifact into <out>/faults/ and append a faults.jsonl line."""
    saved_copy = None
    try:
        src = Path(fault.saved_problem_path)
        if src.is_file():
            dst = out / "faults" / f"u{update:06d}_{src.name}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            saved_copy = str(dst)
    except Exception:
        pass
    ff.write(json.dumps({
        "update": update, "env_index": fault.env_index,
        "saved_problem_path": fault.saved_problem_path,
        "copied_to": saved_copy, "message": str(fault), "time": time.time(),
    }) + "\n")
    ff.flush()
    return saved_copy


def preflight_reward_check(env, steps: int, seed: int, out: Path, ff,
                           quiet: bool) -> tuple[float, float, int]:
    """~`steps` random-action steps; abort (SystemExit) on a flat reward."""
    rng = np.random.default_rng(derive_seed(seed, 0xF11))
    rewards, faults = [], 0
    for _ in range(steps):
        a = rng.uniform(-1.0, 1.0, (env.E, env.ACT)).astype(np.float32)
        try:
            _obs, rew, done, _info = env.step(a)
        except SolverFault as fault:
            faults += 1
            record_fault(out, ff, 0, fault)
            env.reset()
            continue
        live = ~np.asarray(done, bool)
        if live.any():
            rewards.append(np.asarray(rew, np.float64)[live])
        if done.any():
            env.reset(mask=done)
    r = np.concatenate(rewards) if rewards else np.zeros(0)
    mean = float(r.mean()) if r.size else 0.0
    std = float(r.std()) if r.size else 0.0
    if r.size < 10 or std < 1e-7 or np.allclose(r, r.flat[0]):
        raise SystemExit(
            "PREFLIGHT FAILED: reward shows no variation under random actions "
            f"({r.size} samples, mean={mean:.6g}, std={std:.3g}). The reward is "
            "flat/constant, so PPO has no gradient signal. Fix the env reward "
            "before training. (--preflight-steps 0 overrides.)"
        )
    if not quiet:
        print(f"[preflight] OK: random-action reward mean={mean:.4f} "
              f"std={std:.4f} faults={faults}")
    return mean, std, faults


def save_checkpoint(path: Path, update: int, actor, critic, optimizer,
                    sample_gen: torch.Generator, perm_gen: torch.Generator,
                    env_steps: int, faults_total: int, cfg: GpuTrainConfig) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "update": update,
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sample_gen_state": sample_gen.get_state(),
            "sample_gen_device": str(sample_gen.device),
            "perm_gen_state": perm_gen.get_state(),
            "env_steps": env_steps,
            "faults_total": faults_total,
            "config": dataclasses.asdict(cfg),
        },
        tmp,
    )
    tmp.replace(path)


def train(cfg: GpuTrainConfig) -> list[dict]:
    torch.set_num_threads(max(1, cfg.torch_threads))
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2, default=str) + "\n")

    ppo_cfg = PPOConfig(lr=cfg.lr)
    torch.manual_seed(derive_seed(cfg.seed, 0x11))       # net init (matches run.py)
    actor, critic = make_nets(OBS, ACT, ppo_cfg)
    actor.to(device)
    critic.to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=ppo_cfg.lr)
    # Action-noise generator lives on the compute device; the minibatch
    # permutation generator stays on cpu because torch.randperm inside
    # ppo_update draws from a cpu generator (indices cross devices fine).
    sample_gen = torch.Generator(device=device)
    sample_gen.manual_seed(derive_seed(cfg.seed, 0x22))
    perm_gen = torch.Generator()
    perm_gen.manual_seed(derive_seed(cfg.seed, 0x33))

    start_update, env_steps, faults_total = 0, 0, 0
    if cfg.resume:
        ck_path = out / "latest.pt" if cfg.resume == "auto" else Path(cfg.resume)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
        optimizer.load_state_dict(ck["optimizer"])
        start_update = int(ck["update"])
        env_steps = int(ck["env_steps"])
        faults_total = int(ck["faults_total"])
        try:
            if ck.get("sample_gen_device", "cpu") == str(sample_gen.device):
                sample_gen.set_state(ck["sample_gen_state"])
            else:
                raise ValueError("generator device changed")
            perm_gen.set_state(ck["perm_gen_state"])
        except Exception:
            sample_gen.manual_seed(derive_seed(cfg.seed, 0x22, start_update))
            perm_gen.manual_seed(derive_seed(cfg.seed, 0x33, start_update))
        if not cfg.quiet:
            print(f"[gpu_train] resumed from {ck_path} at update {start_update}")

    env = make_env(cfg)
    E = env.E
    metrics: list[dict] = []
    metrics_path = out / "metrics.jsonl"
    faults_path = out / "faults.jsonl"
    t_start = time.perf_counter()
    stopped_by_wall = False
    last_update = start_update
    try:
        with metrics_path.open("a") as mf, faults_path.open("a") as ff:
            if cfg.preflight_steps > 0 and not cfg.resume:
                mean, std, pf_faults = preflight_reward_check(
                    env, cfg.preflight_steps, cfg.seed, out, ff, cfg.quiet)
                faults_total += pf_faults

            obs = env.reset(seed=cfg.seed)   # clean deterministic start
            ep_ret = np.zeros(E, np.float64)
            ep_len = np.zeros(E, np.int64)

            for u in range(start_update + 1, cfg.updates + 1):
                if cfg.max_wall_s and time.perf_counter() - t_start >= cfg.max_wall_s:
                    stopped_by_wall = True
                    if not cfg.quiet:
                        print(f"[gpu_train] --max-wall-s {cfg.max_wall_s} reached "
                              f"after {last_update} updates; stopping cleanly")
                    break
                t0 = time.perf_counter()
                try:
                    ro = collect_rollout(env, actor, critic, obs, cfg.horizon,
                                         sample_gen, device, ep_ret, ep_len)
                except SolverFault as fault:
                    # Poison the whole rollout window: no update, full reset.
                    faults_total += 1
                    record_fault(out, ff, u, fault)
                    obs = env.reset()
                    ep_ret[:] = 0.0
                    ep_len[:] = 0
                    line = {
                        "kind": "train", "update": u, "skipped": "solver_fault",
                        "env_steps": env_steps, "faults": 1,
                        "faults_total": faults_total,
                        "wall_update_s": round(time.perf_counter() - t0, 4),
                        "wall_total_s": round(time.perf_counter() - t_start, 3),
                    }
                    mf.write(json.dumps(line) + "\n")
                    mf.flush()
                    metrics.append(line)
                    last_update = u
                    if not cfg.quiet:
                        print(f"[u{u:4d}] SOLVER FAULT: rollout poisoned, "
                              f"envs reset (faults_total={faults_total})")
                    continue
                t_roll = time.perf_counter() - t0
                obs = ro.next_obs_np
                env_steps += E * cfg.horizon

                t1 = time.perf_counter()
                batch = make_batch(ro, critic, ppo_cfg.gamma, ppo_cfg.lam)
                stats = ppo_update(actor, critic, optimizer, batch, ppo_cfg, perm_gen)
                t_upd = time.perf_counter() - t1

                ep_rets = [r for r, _l in ro.episodes]
                ep_lens = [l for _r, l in ro.episodes]
                line = {
                    "kind": "train",
                    "update": u,
                    "wall_update_s": round(time.perf_counter() - t0, 4),
                    "wall_total_s": round(time.perf_counter() - t_start, 3),
                    "env_steps": env_steps,
                    "rollout_s": round(t_roll, 4),
                    "ppo_s": round(t_upd, 4),
                    "steps_per_s": round(E * cfg.horizon / max(t_roll, 1e-9), 1),
                    "reward_mean": float(ro.rew.mean()),
                    "reward_std": float(ro.rew.std()),
                    "ep_return_mean": float(np.mean(ep_rets)) if ep_rets else None,
                    "ep_len_mean": float(np.mean(ep_lens)) if ep_lens else None,
                    "episodes": len(ro.episodes),
                    "pi_loss": stats.get("pi_loss"),
                    "v_loss": stats.get("v_loss"),
                    "entropy": stats.get("entropy"),
                    "approx_kl": stats.get("approx_kl"),
                    "clip_frac": stats.get("clip_frac"),
                    "faults": 0,
                    "faults_total": faults_total,
                }
                mf.write(json.dumps(line) + "\n")
                mf.flush()
                metrics.append(line)
                last_update = u
                if not cfg.quiet:
                    print(
                        f"[u{u:4d}] sps={line['steps_per_s']:>9.0f} "
                        f"rew={line['reward_mean']:+.4f} "
                        f"pi={stats.get('pi_loss', float('nan')):+.4f} "
                        f"v={stats.get('v_loss', float('nan')):.4f} "
                        f"kl={stats.get('approx_kl', float('nan')):.4f} "
                        f"faults={faults_total}"
                    )

                if u % cfg.checkpoint_every == 0:
                    ck = out / f"ckpt_{u:06d}.pt"
                    save_checkpoint(ck, u, actor, critic, optimizer, sample_gen,
                                    perm_gen, env_steps, faults_total, cfg)
                    shutil.copy2(ck, out / "latest.pt")
    finally:
        # Always checkpoint at exit and always write the cpu actor for local
        # evaluation, even after a wall-clock stop or an exception.
        try:
            save_checkpoint(out / "latest.pt", last_update, actor, critic,
                            optimizer, sample_gen, perm_gen, env_steps,
                            faults_total, cfg)
            torch.save({k: v.detach().cpu() for k, v in actor.state_dict().items()},
                       out / "actor_final.pt")
        finally:
            env.close()
    if not cfg.quiet:
        print(f"[gpu_train] done: updates={last_update} env_steps={env_steps} "
              f"faults={faults_total} wall={time.perf_counter() - t_start:.1f}s "
              f"stopped_by_wall={stopped_by_wall}")
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="walk.train.gpu_train",
        description="Single-process PPO trainer over the batched CUDA duck lane")
    d = GpuTrainConfig()
    p.add_argument("--envs", type=int, default=d.envs)
    p.add_argument("--horizon", type=int, default=d.horizon)
    p.add_argument("--updates", type=int, default=d.updates)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device, help="torch device (cuda on the GPU host)")
    p.add_argument("--library", default=None,
                   help="path to libduck_cuda*.so / serial dylib (default: "
                        "DUCK_CUDA_LIBRARY env or local serial build)")
    p.add_argument("--out", required=True)
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="checkpoint path, or bare flag for <out>/latest.pt")
    p.add_argument("--max-wall-s", type=float, default=0.0,
                   help="stop cleanly (checkpoint + actor_final.pt) after this many seconds")
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--perturbation", type=float, default=d.perturbation,
                   help="per-env joint perturbation bound in rad (<= 0.02)")
    p.add_argument("--checkpoint-every", type=int, default=d.checkpoint_every)
    p.add_argument("--preflight-steps", type=int, default=d.preflight_steps,
                   help="random-action reward-sensitivity steps; 0 skips")
    p.add_argument("--torch-threads", type=int, default=d.torch_threads)
    p.add_argument("--quiet", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> GpuTrainConfig:
    return GpuTrainConfig(
        envs=args.envs, horizon=args.horizon, updates=args.updates, seed=args.seed,
        device=args.device, library=args.library, out=args.out, resume=args.resume,
        max_wall_s=args.max_wall_s, lr=args.lr, perturbation=args.perturbation,
        checkpoint_every=args.checkpoint_every, preflight_steps=args.preflight_steps,
        torch_threads=args.torch_threads, quiet=args.quiet,
    )


def main(argv: list[str] | None = None) -> list[dict]:
    args = build_argparser().parse_args(argv)
    return train(config_from_args(args))


if __name__ == "__main__":
    main()
