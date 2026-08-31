"""PPO training entry point.

  .venv/bin/python -B -m walk.train.run \
      --env walk.env.contract:StubEnv --env-kwargs '{"environments":16}' \
      --workers 12 --total-envs 192 --horizon 64 --updates 100 --seed 917 \
      --out runs/stub

Loop: rollout `horizon` steps across all envs -> PPO update (minibatched,
multi-epoch) -> JSONL metrics line -> checkpoint every K updates.

Determinism / resume contract:
  - All learner randomness flows through one torch.Generator whose state is
    checkpointed, plus deterministic per-worker env seeds.
  - Env shards hold native state that cannot be pickled, so at every
    checkpoint boundary (and at resume) the vec env is reset with a seed
    derived from (seed, update). A resumed run therefore reproduces the
    original run bitwise from the checkpoint onward.

Solver faults: a SolverFault in any worker poisons that shard's entire
rollout window (those transitions are never trained on), the fault artifact
path is recorded in <out>/faults.jsonl (artifact copied into <out>/faults/ if
readable), the shard auto-resets, and training continues.
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

from walk.env.contract import SolverFault

from walk.train.ppo import PPOConfig, compute_gae, make_nets, ppo_update
from walk.train.vec import VecEnv, derive_seed, load_factory


@dataclasses.dataclass
class TrainConfig:
    env: str = "walk.env.contract:StubEnv"
    env_kwargs: dict = dataclasses.field(default_factory=dict)
    workers: int = 4
    total_envs: int | None = None
    horizon: int = 64
    updates: int = 100
    seed: int = 0
    out: str = "runs/dev"
    resume: str | None = None
    checkpoint_every: int = 10
    eval_every: int = 0
    eval_envs: int = 4
    eval_steps: int = 400
    preflight_steps: int = 300
    torch_threads: int = 4
    obs_dim: int | None = None
    env_count_key: str = "environments"
    quiet: bool = False
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)


@dataclasses.dataclass
class Rollout:
    obs: torch.Tensor       # [T, N, OBS]
    raw_act: torch.Tensor   # [T, N, ACT] pre-tanh actions
    logp: torch.Tensor      # [T, N]
    val: torch.Tensor       # [T, N]
    rew: torch.Tensor       # [T, N]
    done: torch.Tensor      # [T, N] float32
    last_obs: torch.Tensor  # [N, OBS]
    next_obs_np: np.ndarray # [N, OBS] rollout continuation point
    poisoned: np.ndarray    # [N] bool, True => never train on this column
    faults: list
    episodes: list[tuple[float, int]]


def collect_rollout(vec: VecEnv, actor, critic, obs_np: np.ndarray, horizon: int,
                    gen: torch.Generator) -> Rollout:
    T, N = horizon, vec.total_envs
    obs_b = torch.zeros((T, N, vec.obs_dim), dtype=torch.float32)
    raw_b = torch.zeros((T, N, vec.act_dim), dtype=torch.float32)
    logp_b = torch.zeros((T, N), dtype=torch.float32)
    val_b = torch.zeros((T, N), dtype=torch.float32)
    rew_b = torch.zeros((T, N), dtype=torch.float32)
    done_b = torch.zeros((T, N), dtype=torch.float32)
    poisoned = np.zeros(N, bool)
    faults, episodes = [], []
    with torch.no_grad():
        for t in range(T):
            obs_t = torch.from_numpy(np.ascontiguousarray(obs_np))
            u, a, logp = actor.sample(obs_t, gen)
            obs_b[t] = obs_t
            raw_b[t] = u
            logp_b[t] = logp
            val_b[t] = critic(obs_t)
            next_obs, rew, done, info = vec.step(a.numpy())
            rew_b[t] = torch.from_numpy(rew)
            done_b[t] = torch.from_numpy(done.astype(np.float32))
            for f in info["faults"]:
                poisoned[vec.slices[f.worker]] = True
                faults.append(f)
            episodes.extend(info["episodes"])
            obs_np = next_obs
    return Rollout(
        obs=obs_b, raw_act=raw_b, logp=logp_b, val=val_b, rew=rew_b, done=done_b,
        last_obs=torch.from_numpy(np.ascontiguousarray(obs_np)), next_obs_np=obs_np,
        poisoned=poisoned, faults=faults, episodes=episodes,
    )


def make_batch(ro: Rollout, critic, gamma: float, lam: float) -> dict[str, torch.Tensor] | None:
    """GAE + flatten, dropping poisoned columns. None if everything is poisoned."""
    valid = ~ro.poisoned
    if not valid.any():
        return None
    with torch.no_grad():
        last_val = critic(ro.last_obs)
    adv, ret = compute_gae(ro.rew, ro.done, ro.val, last_val, gamma, lam)
    vidx = torch.from_numpy(np.nonzero(valid)[0])

    def flat(x: torch.Tensor) -> torch.Tensor:
        x = x.index_select(1, vidx)
        return x.reshape(-1, *x.shape[2:])

    return {
        "obs": flat(ro.obs),
        "raw_act": flat(ro.raw_act),
        "logp": flat(ro.logp),
        "val": flat(ro.val),
        "adv": flat(adv),
        "ret": flat(ret),
    }


def preflight_reward_check(env, steps: int, seed: int) -> tuple[float, float]:
    """Fail fast before update 1 if random-action rewards carry no signal.

    Rolls `steps` random-action steps and aborts (SystemExit) when reward is
    constant / has ~zero variance across states, instead of burning hours of
    PPO on a flat objective.
    """
    rng = np.random.default_rng(derive_seed(seed, 0xF11))
    env.reset(seed=derive_seed(seed, 0xF12))
    rewards = []
    for k in range(steps):
        a = rng.uniform(-1.0, 1.0, (env.E, env.ACT)).astype(np.float32)
        _obs, rew, done, _info = env.step(a)
        live = ~np.asarray(done, bool)
        # done envs report degenerate rewards by contract; only count live ones
        if live.any():
            rewards.append(np.asarray(rew, np.float64)[live])
        else:
            env.reset(seed=derive_seed(seed, 0xF13, k))
    r = np.concatenate(rewards) if rewards else np.zeros(0)
    mean = float(r.mean()) if r.size else 0.0
    std = float(r.std()) if r.size else 0.0
    if r.size < 10 or std < 1e-7 or np.allclose(r, r.flat[0]):
        raise SystemExit(
            "PREFLIGHT FAILED: reward shows no variation under random actions "
            f"({r.size} samples, mean={mean:.6g}, std={std:.3g}). The reward is flat/constant, "
            "so PPO has no gradient signal. Fix the env reward before training. "
            "(--preflight-steps 0 overrides, but you almost certainly should not.)"
        )
    return mean, std


@torch.no_grad()
def evaluate(env, actor, max_steps: int, seed: int) -> dict:
    """Deterministic-policy (tanh of mean) eval on a held-out env instance."""
    obs = env.reset(seed=seed)
    E = env.E
    ret = np.zeros(E, np.float64)
    length = np.zeros(E, np.int64)
    finished = np.zeros(E, bool)
    steps = 0
    for _ in range(max_steps):
        a = actor.deterministic(torch.from_numpy(np.ascontiguousarray(obs))).numpy()
        obs, rew, done, _info = env.step(a)
        live = ~finished
        ret[live] += rew[live]
        length[live] += 1
        finished |= np.asarray(done, bool)
        steps += 1
        if finished.all():
            break
    return {
        "eval_return_mean": float(ret.mean()),
        "eval_return_std": float(ret.std()),
        "eval_len_mean": float(length.mean()),
        "eval_finished_frac": float(finished.mean()),
        "eval_steps": steps,
    }


def reset_seed(seed: int, update: int) -> int:
    return derive_seed(seed, 0xC0FFE, update)


def save_checkpoint(path: Path, update: int, actor, critic, optimizer, gen: torch.Generator,
                    env_steps: int, faults_total: int, cfg: TrainConfig) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "update": update,
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optimizer": optimizer.state_dict(),
            "gen_state": gen.get_state(),
            "env_steps": env_steps,
            "faults_total": faults_total,
            "config": _cfg_to_dict(cfg),
        },
        tmp,
    )
    tmp.replace(path)


def _cfg_to_dict(cfg: TrainConfig) -> dict:
    d = dataclasses.asdict(cfg)
    return d


def train(cfg: TrainConfig) -> list[dict]:
    torch.set_num_threads(max(1, cfg.torch_threads))
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(_cfg_to_dict(cfg), indent=2, default=str) + "\n")
    metrics_path = out / "metrics.jsonl"
    faults_path = out / "faults.jsonl"

    torch.manual_seed(derive_seed(cfg.seed, 0x11))  # net init
    actor, critic = make_nets(
        cfg.obs_dim or getattr(load_factory(cfg.env), "OBS", 58),
        getattr(load_factory(cfg.env), "ACT", 14),
        cfg.ppo,
    )
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=cfg.ppo.lr)
    gen = torch.Generator()
    gen.manual_seed(derive_seed(cfg.seed, 0x22))

    start_update, env_steps, faults_total = 0, 0, 0
    if cfg.resume:
        ck_path = out / "latest.pt" if cfg.resume == "auto" else Path(cfg.resume)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
        optimizer.load_state_dict(ck["optimizer"])
        gen.set_state(ck["gen_state"])
        start_update = int(ck["update"])
        env_steps = int(ck["env_steps"])
        faults_total = int(ck["faults_total"])
        if not cfg.quiet:
            print(f"[run] resumed from {ck_path} at update {start_update}")

    # Held-out env instance in the learner process, for preflight + eval.
    eval_env = None
    if (cfg.preflight_steps > 0 and not cfg.resume) or cfg.eval_every > 0:
        # The held-out instance takes the worker slot one past the fleet so
        # "$WORKER"-seeded envs never share a stream with training shards.
        eval_kwargs = {k: (cfg.workers if v == "$WORKER" else v)
                       for k, v in cfg.env_kwargs.items()}
        eval_kwargs[cfg.env_count_key] = cfg.eval_envs
        eval_env = load_factory(cfg.env)(**eval_kwargs)

    if cfg.preflight_steps > 0 and not cfg.resume:
        try:
            mean, std = preflight_reward_check(eval_env, cfg.preflight_steps, cfg.seed)
        except SystemExit:
            eval_env.close()
            raise
        if not cfg.quiet:
            print(f"[preflight] OK: random-action reward mean={mean:.4f} std={std:.4f}")

    vec = VecEnv(
        cfg.env, cfg.env_kwargs, workers=cfg.workers, total_envs=cfg.total_envs,
        seed=cfg.seed, env_count_key=cfg.env_count_key, obs_dim=cfg.obs_dim,
    )
    N = vec.total_envs
    metrics: list[dict] = []
    t_start = time.perf_counter()
    try:
        obs = vec.reset(reset_seed(cfg.seed, start_update))
        with metrics_path.open("a") as mf, faults_path.open("a") as ff:
            for u in range(start_update + 1, cfg.updates + 1):
                t0 = time.perf_counter()
                ro = collect_rollout(vec, actor, critic, obs, cfg.horizon, gen)
                t_roll = time.perf_counter() - t0
                obs = ro.next_obs_np
                env_steps += N * cfg.horizon

                for f in ro.faults:
                    faults_total += 1
                    saved_copy = None
                    try:
                        src = Path(f.saved_problem_path)
                        if src.is_file():
                            dst = out / "faults" / f"u{u:06d}_w{f.worker}_{src.name}"
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src, dst)
                            saved_copy = str(dst)
                    except Exception:
                        pass
                    ff.write(json.dumps({
                        "update": u, "worker": f.worker, "env_index": f.env_index,
                        "saved_problem_path": f.saved_problem_path,
                        "copied_to": saved_copy, "message": f.message,
                        "time": time.time(),
                    }) + "\n")
                    ff.flush()

                t1 = time.perf_counter()
                batch = make_batch(ro, critic, cfg.ppo.gamma, cfg.ppo.lam)
                stats = ppo_update(actor, critic, optimizer, batch, cfg.ppo, gen) if batch else {}
                t_upd = time.perf_counter() - t1

                valid = ~ro.poisoned
                rew_valid = ro.rew[:, torch.from_numpy(np.nonzero(valid)[0])] if valid.any() else None
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
                    "steps_per_s": round(N * cfg.horizon / max(t_roll, 1e-9), 1),
                    "steps_per_s_per_worker": round(N * cfg.horizon / max(t_roll, 1e-9) / cfg.workers, 1),
                    "reward_mean": float(rew_valid.mean()) if rew_valid is not None else None,
                    "reward_std": float(rew_valid.std()) if rew_valid is not None else None,
                    "ep_return_mean": float(np.mean(ep_rets)) if ep_rets else None,
                    "ep_len_mean": float(np.mean(ep_lens)) if ep_lens else None,
                    "episodes": len(ro.episodes),
                    "pi_loss": stats.get("pi_loss"),
                    "v_loss": stats.get("v_loss"),
                    "entropy": stats.get("entropy"),
                    "approx_kl": stats.get("approx_kl"),
                    "clip_frac": stats.get("clip_frac"),
                    "batch_transitions": int(batch["obs"].shape[0]) if batch else 0,
                    "poisoned_envs": int(ro.poisoned.sum()),
                    "faults": len(ro.faults),
                    "faults_total": faults_total,
                }
                mf.write(json.dumps(line) + "\n")
                mf.flush()
                metrics.append(line)
                if not cfg.quiet:
                    rm = line["reward_mean"]
                    print(
                        f"[u{u:4d}] sps={line['steps_per_s']:>9.0f} rew={rm if rm is None else f'{rm:+.4f}'} "
                        f"pi={stats.get('pi_loss', float('nan')):+.4f} v={stats.get('v_loss', float('nan')):.4f} "
                        f"kl={stats.get('approx_kl', float('nan')):.4f} faults={faults_total}"
                    )

                if cfg.eval_every > 0 and u % cfg.eval_every == 0:
                    # The held-out env hits the same native solver as workers; a
                    # fault must skip this eval pass, not kill the training run.
                    try:
                        ev = evaluate(eval_env, actor, cfg.eval_steps, derive_seed(cfg.seed, 0xE7A1, u))
                    except SolverFault as fault:
                        ev = None
                        ev_line = {"kind": "eval", "update": u, "skipped": "solver_fault",
                                   "fault": str(fault)}
                        eval_env.reset(seed=derive_seed(cfg.seed, 0xE7A2, u))
                    else:
                        ev_line = {"kind": "eval", "update": u, **ev}
                    mf.write(json.dumps(ev_line) + "\n")
                    mf.flush()
                    metrics.append(ev_line)
                    if not cfg.quiet and ev is not None:
                        print(f"[eval u{u}] return={ev['eval_return_mean']:+.4f} len={ev['eval_len_mean']:.1f}")
                    elif not cfg.quiet:
                        print(f"[eval u{u}] skipped: solver fault")

                if u % cfg.checkpoint_every == 0 or u == cfg.updates:
                    ck = out / f"ckpt_{u:06d}.pt"
                    save_checkpoint(ck, u, actor, critic, optimizer, gen, env_steps, faults_total, cfg)
                    shutil.copy2(ck, out / "latest.pt")
                    if u != cfg.updates:
                        # deterministic env boundary so a resume from this
                        # checkpoint continues bitwise-identically
                        obs = vec.reset(reset_seed(cfg.seed, u))
    finally:
        vec.close()
        if eval_env is not None:
            eval_env.close()
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="walk.train.run", description="Parallel PPO trainer for DuckEnvBatch envs")
    d = TrainConfig()
    pd = PPOConfig()
    p.add_argument("--env", default=d.env, help="dotted factory spec, e.g. walk.env.flat:FlatFloorDuckEnv")
    p.add_argument("--env-kwargs", default="{}", help='JSON kwargs for the factory, e.g. \'{"environments":16}\'')
    p.add_argument("--workers", type=int, default=d.workers)
    p.add_argument("--total-envs", type=int, default=None, help="default: env-kwargs count * workers")
    p.add_argument("--horizon", type=int, default=d.horizon)
    p.add_argument("--updates", type=int, default=d.updates)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="checkpoint path, or bare flag for <out>/latest.pt")
    p.add_argument("--checkpoint-every", type=int, default=d.checkpoint_every, metavar="K")
    p.add_argument("--eval-every", type=int, default=d.eval_every, help="0 disables eval")
    p.add_argument("--eval-envs", type=int, default=d.eval_envs)
    p.add_argument("--eval-steps", type=int, default=d.eval_steps)
    p.add_argument("--preflight-steps", type=int, default=d.preflight_steps, help="0 skips the flat-reward preflight")
    p.add_argument("--torch-threads", type=int, default=d.torch_threads, help="learner threads (workers always use 1)")
    p.add_argument("--obs-dim", type=int, default=None, help="override obs width if env class attr is wrong")
    p.add_argument("--env-count-key", default=d.env_count_key)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--lr", type=float, default=pd.lr)
    p.add_argument("--gamma", type=float, default=pd.gamma)
    p.add_argument("--lam", type=float, default=pd.lam)
    p.add_argument("--clip", type=float, default=pd.clip)
    p.add_argument("--clip-value", type=float, default=pd.clip_value)
    p.add_argument("--epochs", type=int, default=pd.epochs)
    p.add_argument("--minibatches", type=int, default=pd.minibatches)
    p.add_argument("--ent-coef", type=float, default=pd.ent_coef)
    p.add_argument("--vf-coef", type=float, default=pd.vf_coef)
    p.add_argument("--max-grad-norm", type=float, default=pd.max_grad_norm)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--hidden", type=int, nargs=2, default=list(pd.hidden), metavar=("H1", "H2"))
    p.add_argument("--log-std-init", type=float, default=pd.log_std_init)
    return p


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    ppo = PPOConfig(
        lr=args.lr, gamma=args.gamma, lam=args.lam, clip=args.clip, clip_value=args.clip_value,
        epochs=args.epochs, minibatches=args.minibatches, ent_coef=args.ent_coef,
        vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm, target_kl=args.target_kl,
        hidden=tuple(args.hidden), log_std_init=args.log_std_init,
    )
    return TrainConfig(
        env=args.env, env_kwargs=json.loads(args.env_kwargs), workers=args.workers,
        total_envs=args.total_envs, horizon=args.horizon, updates=args.updates, seed=args.seed,
        out=args.out, resume=args.resume, checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every, eval_envs=args.eval_envs, eval_steps=args.eval_steps,
        preflight_steps=args.preflight_steps, torch_threads=args.torch_threads,
        obs_dim=args.obs_dim, env_count_key=args.env_count_key, quiet=args.quiet, ppo=ppo,
    )


def main(argv: list[str] | None = None) -> list[dict]:
    args = build_argparser().parse_args(argv)
    return train(config_from_args(args))


if __name__ == "__main__":
    main()
