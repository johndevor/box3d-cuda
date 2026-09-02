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
        [--perturbation 0.02] [--policy {ff,gru}] [--gamma G] [--gae-lambda L] \
        [--robot {duck,humanoid}] [--variant NAME]

--variant NAME (humanoid only) selects a FAMILY member (humanoid/h1_family.py:
h1_tall, h1_stocky): robot_classes binds the humanoid lane/env classes to the
variant (its header/dylib, gains, reference table); --library must then be a
build against humanoid/variants/<name>/include. Unset = the accepted H1.1.

--robot selects the env/lane contract (robot_classes): duck (default) is
byte-identical to the pre-switch trainer (same classes, same OBS/ACT 58/14,
same RNG streams); humanoid wires FlatFloorHumanoidEnv / CudaHumanoidLane
(OBS/ACT 3*J+16 / J) over the robot-generic dwc1 kernel. --accept-every is
duck-only (strict gait evaluator) and rejected for other robots.
--randomization (the shared dwc1 DR contract, walk/env/cuda_lane.py: mass,
friction, kp, damping, command latency and the ABI-v7 one-sided gravity
scale) is accepted for every robot on the device policy path (--lane-env).

--policy gru swaps in the tiny recurrent policy (walk.train.ppo
RecurrentActor/RecurrentCritic: 1-layer GRU + reduced MLP head) for implicit
system ID: hidden state [E, H] is carried across steps, zeroed for envs that
reset, and the update is recurrent PPO with truncated BPTT over the rollout
window using the stored window-initial hidden states (sequence minibatches =
env slices). --policy ff (default) is byte-identical to the pre-gru trainer.

FF -> GRU WARM START (generalist recipe): with --policy gru, an --init-actor
or --resume source that is a FEED-FORWARD actor/checkpoint is not an error:
the recurrent nets are built with a residual feed-forward trunk of the FF
nets' sizes and ppo.warm_start_recurrent_from_ff copies the FF weights into
it and zeroes the GRU heads (policy bit-identical to the FF specialist at
step 0; critic likewise when the source is a full checkpoint). For --resume
the update/env_steps/wall counters continue and the optimizer/generators
start fresh (an FF Adam state cannot drive GRU parameters). Resuming a GRU
checkpoint that carries a trunk rebuilds the trunk automatically
(ppo.trunk_hidden_from_state_dict). walk/train/gru_warmstart.py writes such
a warm-started update-0 GRU checkpoint offline.

Solver faults: a SolverFault anywhere in a rollout poisons that whole rollout
window (no update is applied), the fault artifact is copied into <out>/faults/
and logged to <out>/faults.jsonl, every env is reset, and training continues.

Checkpoints: <out>/ckpt_NNNNNN.pt every --checkpoint-every updates, plus
<out>/latest.pt at every checkpoint and at exit. <out>/actor_final.pt (a plain
cpu state_dict of the actor) is ALWAYS written at exit for local evaluation.
--max-wall-s stops the update loop cleanly (checkpoint + actor_final, exit 0).

--accept-every N (0 = off): every N updates (and once at the very end) the
current deterministic policy is judged by the STRICT gait evaluator on a
fresh non-randomized E=1 env — seeds (4242, 7) x commands (0.10/0.15/0.20)
m/s, 8 s each; if all 6 pass, a stability confirmation runs 11 s episodes per
command at seed 4242 (no fall through 11 s AND the exact-8 s prefix still
passes). On confirmed pass the trainer writes <out>/accepted/
{actor_accepted.pt, acceptance.json}, prints the WALKING ACCEPTED line with
cumulative training wall seconds (probe time tracked and excluded) and exits
0. Probe time never counts toward the reported training wall.
"""
from __future__ import annotations

import argparse
import contextlib
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
from walk.eval.capture import capture_episodes
from walk.eval.gait import evaluate_episode
from walk.train.ppo import (
    PPOConfig,
    compute_gae,
    make_nets,
    make_recurrent_nets,
    ppo_update,
    recurrent_ppo_update,
    tanh_gaussian_log_prob,
    trunk_hidden_from_state_dict,
    unpack_actor_file,
    warm_start_recurrent_from_ff,
)
from walk.train.vec import derive_seed


@dataclasses.dataclass
class GpuTrainConfig:
    robot: str = "duck"              # "duck" (default, byte-identical) or "humanoid"
    variant: str | None = None       # humanoid family member (None = H1.1 base)
    curriculum: str | None = None    # tech-tree ladder (builtin name or JSON
    #                                  path); None = today's exact behavior
    envs: int = 4096
    lane_env: bool = False
    randomization: dict | None = None
    horizon: int = 32
    updates: int = 300
    seed: int = 917
    device: str = "cpu"
    library: str | None = None
    out: str = "runs/gpu-train"
    resume: str | None = None
    init_actor: str | None = None    # actor-only warm start (e.g. BC pretrain)
    rsi_fraction: float = 0.0        # reference-state-init reset fraction
    max_wall_s: float = 0.0          # 0 disables the wall-clock stop
    policy: str = "ff"               # "ff" (feed-forward) or "gru" (recurrent)
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    perturbation: float = 0.0
    accept_every: int = 0            # 0 = off; N = strict-acceptance probe every N updates
    checkpoint_every: int = 25
    preflight_steps: int = 100
    torch_threads: int = 4
    quiet: bool = False
    validate_only: bool = False      # config/flag validation, then exit 0


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


def robot_classes(robot: str, variant: str | None = None):
    """(obs_dim, act_dim, lane_cls, flat_env_cls) for a robot.

    The duck row returns exactly the module-level contract dims and the
    classes the pre-robot-switch trainer used, so --robot duck (the default)
    is behavior-identical to the old hardcoded path. Humanoid classes import
    lazily (they build/load the humanoid serial dylib on first use).

    `variant` (humanoid only): a family member name (humanoid/h1_family.py)
    binds the lane/env classes to that member via functools.partial
    (variant=...); None/"h1" returns the bare classes exactly as before."""
    if robot == "duck":
        if variant not in (None, "", "h1"):
            raise SystemExit("--variant is humanoid-only (family variants)")
        return OBS, ACT, CudaDuckLane, FlatFloorDuckEnv
    if robot == "humanoid":
        from walk.env import humanoid_flat
        from walk.env.humanoid_cuda_lane import CudaHumanoidLane
        if variant in (None, "", "h1"):
            return (humanoid_flat.OBS, humanoid_flat.ACT, CudaHumanoidLane,
                    humanoid_flat.FlatFloorHumanoidEnv)
        import functools
        import h1_family
        if variant not in h1_family.MORPHOLOGIES:
            raise SystemExit(f"--variant must be one of "
                             f"{sorted(h1_family.MORPHOLOGIES)}, got {variant!r}")
        lane_cls = functools.partial(CudaHumanoidLane, variant=variant)
        env_cls = functools.partial(humanoid_flat.FlatFloorHumanoidEnv,
                                    variant=variant)
        return humanoid_flat.OBS, humanoid_flat.ACT, lane_cls, env_cls
    if robot == "arm":
        # fixed-base 6-axis reach family (arm/arm_lowering.py): the variant
        # ("kr240" default | "lite") is bound into both classes; obs/reward
        # run python-side over dwc1_step (no device policy path, see
        # arm/FEASIBILITY.md section 6), so --lane-env is rejected below.
        import functools
        from walk.env import arm_reach
        from walk.env.arm_cuda_lane import CudaArmLane
        import arm_lowering
        variant = variant or "kr240"
        if variant not in arm_lowering.VARIANTS:
            raise SystemExit(f"--variant must be one of "
                             f"{sorted(arm_lowering.VARIANTS)}, got {variant!r}")
        return (arm_reach.OBS, arm_reach.ACT,
                functools.partial(CudaArmLane, variant=variant),
                functools.partial(arm_reach.ArmReachEnv, variant=variant))
    raise SystemExit(f"--robot must be duck, humanoid or arm, got {robot!r}")


class LanePolicyEnv:
    """Duck-typed drop-in for the flat env over the ABI-v3 device policy
    path: one kernel launch + tiny transfers per policy step. Solver faults
    freeze+finish the env in-kernel (no exception); counted in fault_count."""

    OBS, ACT = 58, 14                # legacy duck defaults (class-level)

    def __init__(self, cfg):
        robot = getattr(cfg, "robot", "duck")
        if robot == "arm":
            raise SystemExit("--lane-env needs the in-kernel device policy "
                             "path, which cannot express the arm's obs "
                             "(arm/FEASIBILITY.md section 6); run --robot arm "
                             "without --lane-env")
        self.OBS, self.ACT, lane_cls, _ = robot_classes(
            robot, getattr(cfg, "variant", None))
        self.E = cfg.envs
        kwargs = dict(
            joint_offsets=_perturbation_offsets(cfg) if cfg.perturbation else None,
            library_path=cfg.library)
        # every dwc1 lane shares the duck's DR contract (cuda_lane.py):
        # None = off = bit-identical physics
        kwargs["randomization"] = cfg.randomization
        if robot != "duck":          # training lane: freeze fallen envs
            kwargs["fast_termination"] = True
        self._lane = lane_cls(self.E, **kwargs)
        rsi = float(getattr(cfg, "rsi_fraction", 0.0) or 0.0)
        if rsi > 0.0:
            if not hasattr(self._lane, "set_rsi"):
                raise SystemExit("--rsi-fraction requires a lane with RSI "
                                 "support (humanoid)")
            self._lane.set_rsi(rsi)
        self._seed = cfg.seed
        self.fault_count = 0
        # curriculum hook: a stage may pin resets to a command pool. None
        # (the default, and always for the duck) leaves reset_policy's own
        # counter-stream draw untouched -- byte-identical behavior.
        self.command_override = None
        self._cmd_rng = np.random.default_rng(derive_seed(cfg.seed, 0xC0DE))

    def reset(self, mask=None, seed=None):
        if self.command_override is None:
            return self._lane.reset_policy(mask=mask, seed=seed
                                           if seed is not None else self._seed)
        pool = np.asarray(self.command_override, np.float64)
        commands = pool[self._cmd_rng.integers(0, len(pool), self.E)]
        return self._lane.reset_policy(
            mask=mask, seed=seed if seed is not None else self._seed,
            commands=commands)

    def step(self, action):
        obs, reward, done, diag = self._lane.step_policy(
            np.asarray(action, np.float32))
        n_fault = int((diag["status"] != 0).sum())
        if n_fault:
            self.fault_count += n_fault
        return obs, reward, done.astype(bool), {"faults": n_fault}

    def set_command(self, commands):
        return self._lane.set_command(commands)

    def close(self):
        self._lane.close()


def _perturbation_offsets(cfg):
    import numpy as _np
    act = robot_classes(getattr(cfg, "robot", "duck"),
                        getattr(cfg, "variant", None))[1]   # duck: 14, as before
    return _np.stack([
        _np.random.default_rng([cfg.seed & 0xFFFFFFFF, e, 0]).uniform(
            -cfg.perturbation, cfg.perturbation, act) for e in range(cfg.envs)])


def make_env(cfg: GpuTrainConfig):
    if cfg.lane_env:
        return LanePolicyEnv(cfg)
    _, _, lane_cls, flat_env_cls = robot_classes(
        getattr(cfg, "robot", "duck"), getattr(cfg, "variant", None))
    return flat_env_cls(
        environments=cfg.envs,
        seed=cfg.seed,
        perturbation_rad=cfg.perturbation,
        lane_factory=lambda E, offsets: lane_cls(
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


# ---------------------------------------------------------------------------
# Recurrent (--policy gru) path. The feed-forward path above is untouched.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RecurrentRollout(Rollout):
    h0_actor: torch.Tensor = None    # [E, H] hidden at the START of the window
    h0_critic: torch.Tensor = None   # [E, H]
    h_actor: torch.Tensor = None     # [E, H] hidden AFTER the window (carried)
    h_critic: torch.Tensor = None    # [E, H]


def collect_rollout_recurrent(env, actor, critic, obs_np: np.ndarray,
                              h_a: torch.Tensor, h_c: torch.Tensor,
                              horizon: int, gen: torch.Generator,
                              device: torch.device,
                              ep_ret: np.ndarray, ep_len: np.ndarray) -> RecurrentRollout:
    """Recurrent twin of collect_rollout: carries actor/critic hidden state
    [E, H] across steps, zeroing the rows of envs that reset (done mask) so a
    fresh episode always starts from h = 0. Stores the window-initial hidden
    states for the truncated-BPTT update."""
    T, E = horizon, env.E
    obs_b = torch.zeros((T, E, env.OBS), dtype=torch.float32, device=device)
    raw_b = torch.zeros((T, E, env.ACT), dtype=torch.float32, device=device)
    logp_b = torch.zeros((T, E), dtype=torch.float32, device=device)
    val_b = torch.zeros((T, E), dtype=torch.float32, device=device)
    rew_b = torch.zeros((T, E), dtype=torch.float32, device=device)
    done_b = torch.zeros((T, E), dtype=torch.float32, device=device)
    episodes: list[tuple[float, int]] = []
    h0_a, h0_c = h_a.clone(), h_c.clone()
    with torch.no_grad():
        for t in range(T):
            obs_t = torch.from_numpy(np.ascontiguousarray(obs_np)).to(device)
            mu, std, h_a = actor.dist(obs_t, h_a)
            u = mu + std * torch.randn(mu.shape, generator=gen,
                                       dtype=mu.dtype, device=device)
            a = torch.tanh(u)
            obs_b[t] = obs_t
            raw_b[t] = u
            logp_b[t] = tanh_gaussian_log_prob(mu, std, u)
            val_b[t], h_c = critic(obs_t, h_c)
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
                next_obs = env.reset(mask=done)
                keep = torch.from_numpy((~done).astype(np.float32)).to(device)
                h_a = h_a * keep.unsqueeze(1)   # fresh episodes start at h = 0
                h_c = h_c * keep.unsqueeze(1)
            obs_np = next_obs
    last_obs = torch.from_numpy(np.ascontiguousarray(obs_np)).to(device)
    return RecurrentRollout(obs=obs_b, raw_act=raw_b, logp=logp_b, val=val_b,
                            rew=rew_b, done=done_b, last_obs=last_obs,
                            next_obs_np=obs_np, episodes=episodes,
                            h0_actor=h0_a, h0_critic=h0_c,
                            h_actor=h_a, h_critic=h_c)


def make_batch_recurrent(ro: RecurrentRollout, critic, gamma: float,
                         lam: float) -> dict[str, torch.Tensor]:
    """GAE identical to the ff path; keeps [T, N] sequence layout plus the
    window-initial hidden states (minibatches are env slices)."""
    with torch.no_grad():
        last_val, _h = critic(ro.last_obs, ro.h_critic)
    adv, ret = compute_gae(ro.rew, ro.done, ro.val, last_val, gamma, lam)
    return {"obs": ro.obs, "raw_act": ro.raw_act, "logp": ro.logp, "val": ro.val,
            "adv": adv, "ret": ret, "done": ro.done,
            "h0_actor": ro.h0_actor, "h0_critic": ro.h0_critic}


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


# ---------------------------------------------------------------------------
# In-training strict-acceptance probe (--accept-every N): the time-to-walking
# stopwatch. Every N updates the current deterministic policy is judged by the
# STRICT evaluator (walk/eval/gait.py) on a fresh, non-randomized E=1 env:
#   stage 1: seeds (4242, 7) x commands (0.10, 0.15, 0.20) m/s, 8 s each —
#            all 6 episodes must pass;
#   stage 2 (stability confirmation): per command at seed 4242, an 11 s
#            episode must survive with no termination AND its exact-8 s tick
#            prefix must still pass the evaluator.
# On confirmed pass the trainer writes <out>/accepted/{actor_accepted.pt,
# acceptance.json}, checkpoints, prints the WALKING ACCEPTED line and exits 0.
# ---------------------------------------------------------------------------

PROBE_SEEDS = (4242, 7, 1913, 90210)  # full anti-overfit gate
PROBE_COMMANDS = (0.10, 0.15, 0.20)
PROBE_EPISODE_SECONDS = 8.0
CONFIRM_SEED = 4242
CONFIRM_SECONDS = 11.0
POLICY_DT = 0.02


def _make_probe_env(cfg: GpuTrainConfig, seed: int):
    """Judging env: E=1, zero perturbation, randomization OFF, same library."""
    return FlatFloorDuckEnv(
        environments=1, seed=int(seed), perturbation_rad=0.0, randomization=None,
        lane_factory=lambda E, offsets: CudaDuckLane(
            E, joint_offsets=offsets, library_path=cfg.library))


def _probe_policy(actor, arch: str, device: torch.device):
    """Deterministic policy closure for capture_episodes. gru carries hidden
    state within the episode; a fresh closure (fresh h) is made per episode."""
    if arch == "ff":
        @torch.no_grad()
        def policy(obs):
            o = torch.from_numpy(np.ascontiguousarray(obs)).to(device)
            return actor.deterministic(o).cpu().numpy()
        return policy
    state = {"h": None}

    @torch.no_grad()
    def policy(obs):
        o = torch.from_numpy(np.ascontiguousarray(obs)).to(device)
        if state["h"] is None:
            state["h"] = actor.initial_state(o.shape[0], device)
        act, state["h"] = actor.deterministic(o, state["h"])
        return act.cpu().numpy()
    return policy


@contextlib.contextmanager
def _extended_horizon(steps: int):
    """Temporarily raise the flat env's episode horizon (module global read at
    step time) so the 11 s stability episodes are not cut off at 8 s. The
    trainer's own env never steps while a probe runs, and the old value is
    always restored."""
    from walk.env import flat as flat_mod
    old = flat_mod.HORIZON_STEPS
    flat_mod.HORIZON_STEPS = int(steps)
    try:
        yield
    finally:
        flat_mod.HORIZON_STEPS = old


def truncate_trace_to_8s(trace: dict) -> dict:
    """Exact-8 s prefix of a longer trace: the strict evaluator's translation
    and last-step criteria are calibrated to 8 s, so re-scoring uses exactly
    int(round(8 s / dt)) ticks, marked as a clean horizon truncation."""
    dt = float(trace["dt"])
    n8 = int(round(PROBE_EPISODE_SECONDS / dt))
    out = {k: v for k, v in trace.items() if k != "ticks"}
    out["ticks"] = {k: list(v[:n8]) for k, v in trace["ticks"].items()}
    out["terminated"] = False
    out["truncated_at_horizon"] = True
    return out


def _episode_summary(r: dict) -> dict:
    q = [f for f in r.get("footfalls", []) if f.get("qualified")]
    fails = [k for k, v in r.get("criteria", {}).items() if not v.get("pass")]
    return {"passed": bool(r.get("passed")), "qualified": len(q),
            "failed_criteria": fails}


def run_acceptance_probe(cfg: GpuTrainConfig, actor, device: torch.device) -> dict:
    """Run the 6-episode strict probe + 3-episode stability confirmation.

    Short-circuits on the first failing episode (probe cost control: at most
    6 + 3 episodes on E=1 with per-tick reads). A SolverFault anywhere is a
    failed probe, never a crash. Returns stage results + probe wall seconds.
    """
    t0 = time.perf_counter()
    episodes: dict[str, dict] = {}
    confirmation: dict[str, dict] = {}
    stage1 = True
    confirmed = False
    try:
        for seed in PROBE_SEEDS:
            if not stage1:
                break
            env = _make_probe_env(cfg, seed)
            try:
                for cmd in PROBE_COMMANDS:
                    trace = capture_episodes(
                        env, _probe_policy(actor, cfg.policy, device),
                        command=cmd, seconds=PROBE_EPISODE_SECONDS, seed=seed)[0]
                    r = evaluate_episode(trace)
                    episodes[f"seed{seed}-cmd{cmd:.2f}"] = _episode_summary(r)
                    if not r.get("passed"):
                        stage1 = False
                        break
            finally:
                env.close()
        if stage1:
            confirmed = True
            env = _make_probe_env(cfg, CONFIRM_SEED)
            try:
                with _extended_horizon(int(round(CONFIRM_SECONDS / POLICY_DT))):
                    for cmd in PROBE_COMMANDS:
                        trace = capture_episodes(
                            env, _probe_policy(actor, cfg.policy, device),
                            command=cmd, seconds=CONFIRM_SECONDS,
                            seed=CONFIRM_SEED)[0]
                        survived = (bool(trace.get("truncated_at_horizon"))
                                    and not trace.get("terminated")
                                    and not trace.get("solver_fault"))
                        r8 = (evaluate_episode(truncate_trace_to_8s(trace))
                              if survived else None)
                        ok = survived and bool(r8 and r8.get("passed"))
                        confirmation[f"seed{CONFIRM_SEED}-cmd{cmd:.2f}-11s"] = {
                            "survived_11s": survived,
                            "prefix_8s": _episode_summary(r8) if r8 else None,
                            "passed": ok,
                        }
                        if not ok:
                            confirmed = False
                            break
            finally:
                env.close()
    except SolverFault as fault:
        stage1 = confirmed = False
        episodes["solver_fault"] = {"passed": False,
                                    "detail": str(fault),
                                    "saved_problem_path": fault.saved_problem_path}
    return {"stage1_passed": stage1, "confirmed": confirmed,
            "episodes": episodes, "confirmation": confirmation,
            "probe_wall_s": time.perf_counter() - t0}


def write_acceptance(out: Path, cfg: GpuTrainConfig, actor, update: int,
                     env_steps: int, train_wall_s: float, probe_wall_s: float,
                     probe: dict) -> Path:
    acc = out / "accepted"
    acc.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"arch": cfg.policy,
         "state_dict": {k: v.detach().cpu()
                        for k, v in actor.state_dict().items()}},
        acc / "actor_accepted.pt")
    (acc / "acceptance.json").write_text(json.dumps({
        "schema": "duckgridwalk.training_acceptance/1",
        "accepted": True,
        "update": int(update),
        "env_steps": int(env_steps),
        "train_wall_s": round(float(train_wall_s), 3),   # excludes probe time
        "probe_wall_s": round(float(probe_wall_s), 3),   # cumulative probing
        "probe_wall_last_s": round(float(probe["probe_wall_s"]), 3),
        "policy": cfg.policy,
        "seeds": list(PROBE_SEEDS),
        "commands": list(PROBE_COMMANDS),
        "confirm_seed": CONFIRM_SEED,
        "confirm_seconds": CONFIRM_SECONDS,
        "episodes": probe["episodes"],
        "confirmation": probe["confirmation"],
    }, indent=1) + "\n")
    return acc


def _inspect_source(cfg: GpuTrainConfig) -> dict | None:
    """Load the --init-actor / --resume source once and classify it:
    {"path", "raw", "arch" ("ff"|"gru"), "actor" (state dict),
    "critic" (state dict or None), "checkpoint" (bool)}. An actor file is
    {"arch", "state_dict"} or a legacy plain FF state dict
    (ppo.unpack_actor_file); a full checkpoint carries actor/critic/
    optimizer and its config's policy."""
    init_actor = getattr(cfg, "init_actor", None)
    if init_actor and cfg.resume:
        raise SystemExit("--init-actor and --resume are mutually exclusive")
    if init_actor:
        path = Path(init_actor)
    elif cfg.resume:
        path = (Path(cfg.out) / "latest.pt" if cfg.resume == "auto"
                else Path(cfg.resume))
    else:
        return None
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "actor" in raw and "critic" in raw:
        return {"path": path, "raw": raw, "checkpoint": True,
                "arch": str(raw.get("config", {}).get("policy", "ff")),
                "actor": raw["actor"], "critic": raw["critic"]}
    arch, sd = unpack_actor_file(raw)
    return {"path": path, "raw": raw, "checkpoint": False, "arch": arch,
            "actor": sd, "critic": None}


def save_checkpoint(path: Path, update: int, actor, critic, optimizer,
                    sample_gen: torch.Generator, perm_gen: torch.Generator,
                    env_steps: int, faults_total: int, cfg: GpuTrainConfig,
                    train_wall_s: float = 0.0, probe_wall_s: float = 0.0) -> None:
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
            "train_wall_s": float(train_wall_s),   # cumulative, excl. probes
            "probe_wall_s": float(probe_wall_s),   # cumulative probe time
            "config": dataclasses.asdict(cfg),
        },
        tmp,
    )
    tmp.replace(path)


def train(cfg: GpuTrainConfig) -> list[dict]:
    torch.set_num_threads(max(1, cfg.torch_threads))
    device = torch.device(cfg.device)
    if (device.type == "cuda" and not cfg.validate_only
            and not torch.cuda.is_available()):
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2, default=str) + "\n")

    if cfg.policy not in ("ff", "gru"):
        raise SystemExit(f"--policy must be ff or gru, got {cfg.policy!r}")
    obs_dim, act_dim, _, _ = robot_classes(
        cfg.robot, getattr(cfg, "variant", None))   # duck: exactly OBS, ACT
    if cfg.robot != "duck":
        if cfg.accept_every > 0:
            raise SystemExit("--accept-every uses the duck-only strict gait "
                             "evaluator; not available for --robot humanoid")
        if cfg.randomization and not cfg.lane_env:
            raise SystemExit("--randomization for --robot humanoid runs on "
                             "the device policy path; add --lane-env")
    controller = None
    if cfg.curriculum:
        # tech-tree curriculum (walk/train/curriculum_controller.py):
        # imported lazily so the default path stays import-identical.
        from walk.train.curriculum_controller import (  # noqa: PLC0415
            CurriculumController, load_ladder)
        if not cfg.lane_env:
            raise SystemExit("--curriculum drives lane-level gate knobs; "
                             "run with --lane-env")
        controller = CurriculumController(
            load_ladder(cfg.curriculum, cfg.robot), quiet=cfg.quiet)
    if cfg.validate_only:
        # Launcher preflight: every flag/robot/curriculum check above has
        # passed from the archived payload; also require the relay files.
        for label, path in (("--resume", cfg.resume), ("--init-actor", cfg.init_actor)):
            if path and not Path(path).is_file():
                raise SystemExit(f"{label} file not found: {path}")
        print("[gpu_train] validate-only: configuration OK")
        return []
    recurrent = cfg.policy == "gru"
    ppo_cfg = PPOConfig(lr=cfg.lr, gamma=cfg.gamma, lam=cfg.gae_lambda)
    # Peek at the warm-start / resume source BEFORE building the nets: a GRU
    # policy may need a residual feed-forward trunk (FF -> GRU warm start,
    # or a GRU checkpoint that already carries one), and the optimizer must
    # see every parameter at construction.
    source = _inspect_source(cfg)
    gru_trunk = None
    if recurrent and source is not None:
        if source["arch"] == "ff":
            gru_trunk = trunk_hidden_from_state_dict(source["actor"], "mu_net.")
        else:
            gru_trunk = trunk_hidden_from_state_dict(source["actor"], "ff.")
    torch.manual_seed(derive_seed(cfg.seed, 0x11))       # net init (matches run.py)
    if recurrent:
        actor, critic = make_recurrent_nets(obs_dim, act_dim, ppo_cfg,
                                            ff_hidden=gru_trunk)
    else:
        actor, critic = make_nets(obs_dim, act_dim, ppo_cfg)
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
    prev_train_wall, prev_probe_wall = 0.0, 0.0
    if getattr(cfg, "init_actor", None):
        if cfg.resume:
            raise SystemExit("--init-actor and --resume are mutually exclusive")
        arch, sd = source["arch"], source["actor"]
        if arch == cfg.policy:
            actor.load_state_dict(sd)
            note = "(critic/optimizer fresh)"
        elif arch == "ff" and recurrent:
            # FF -> residual GRU warm start (policy bit-identical at step 0)
            warm_start_recurrent_from_ff(actor, sd, critic, source.get("critic"))
            note = ("via FF->GRU residual warm start (trunk "
                    f"{gru_trunk}, critic {'warm' if source.get('critic') else 'fresh'})")
        else:
            raise SystemExit(f"--init-actor is a {arch} actor but --policy "
                             f"{cfg.policy} was requested")
        if not cfg.quiet:
            print(f"[gpu_train] actor initialized from {cfg.init_actor} {note}")
    if cfg.resume:
        ck_path = source["path"]
        ck = source["raw"]
        ck_policy = source["arch"]
        ck_variant = ck.get("config", {}).get("variant") or "h1"
        if cfg.robot == "humanoid" and ck_variant != (
                getattr(cfg, "variant", None) or "h1"):
            raise SystemExit(
                f"checkpoint {ck_path} belongs to humanoid variant "
                f"{ck_variant!r}, but --variant {cfg.variant!r} was requested")
        if ck_policy == cfg.policy:
            actor.load_state_dict(ck["actor"])
            critic.load_state_dict(ck["critic"])
            optimizer.load_state_dict(ck["optimizer"])
        elif ck_policy == "ff" and recurrent:
            # cross-arch: FF checkpoint -> residual GRU. Counters continue
            # (lineage accounting); optimizer + generators start fresh.
            warm_start_recurrent_from_ff(actor, ck["actor"], critic, ck["critic"])
            if not cfg.quiet:
                print(f"[gpu_train] cross-arch warm start: FF checkpoint "
                      f"{ck_path} -> residual GRU (trunk {gru_trunk}; "
                      "optimizer/generators fresh)")
        else:
            raise SystemExit(
                f"checkpoint {ck_path} was trained with --policy {ck_policy}, "
                f"but --policy {cfg.policy} was requested")
        start_update = int(ck["update"])
        env_steps = int(ck["env_steps"])
        faults_total = int(ck["faults_total"])
        prev_train_wall = float(ck.get("train_wall_s", 0.0))
        prev_probe_wall = float(ck.get("probe_wall_s", 0.0))
        try:
            if ck_policy != cfg.policy:
                raise ValueError("cross-arch warm start: fresh generators")
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
    if controller is not None:
        lane = getattr(env, "_lane", None)
        if lane is None or not hasattr(lane, "set_gate_termination") \
                or not hasattr(lane, "gate_proxy"):
            env.close()
            raise SystemExit("--curriculum requires a lane exposing "
                             "gate_proxy + set_gate_termination (this lane "
                             "predates commit 65d99b8?)")
        controller.apply(env)
        if not cfg.quiet:
            print(f"[curriculum] stage '{controller.stage.name}' "
                  f"(entry, update {start_update})")
    E = env.E
    metrics: list[dict] = []
    metrics_path = out / "metrics.jsonl"
    faults_path = out / "faults.jsonl"
    t_start = time.perf_counter()
    stopped_by_wall = False
    last_update = start_update
    probe_wall_proc = 0.0            # probe seconds spent in THIS process
    last_probe_update = -1
    accepted = False

    def train_wall() -> float:
        """Cumulative training wall seconds, probes excluded."""
        return prev_train_wall + (time.perf_counter() - t_start) - probe_wall_proc

    def probe_wall() -> float:
        """Cumulative probe wall seconds."""
        return prev_probe_wall + probe_wall_proc

    try:
        with metrics_path.open("a") as mf, faults_path.open("a") as ff:

            def do_probe(u: int) -> bool:
                """Acceptance probe at update u; returns True on CONFIRMED
                pass (training must stop). Time is booked as probe wall."""
                nonlocal probe_wall_proc, last_probe_update, accepted
                probe = run_acceptance_probe(cfg, actor, device)
                probe_wall_proc += probe["probe_wall_s"]
                last_probe_update = u
                line = {
                    "kind": "accept", "update": u, "env_steps": env_steps,
                    "stage1_passed": probe["stage1_passed"],
                    "accepted": probe["confirmed"],
                    "probe_wall_s": round(probe["probe_wall_s"], 3),
                    "probe_wall_total_s": round(probe_wall(), 3),
                    "train_wall_s": round(train_wall(), 3),
                    "episodes": probe["episodes"],
                    "confirmation": probe["confirmation"],
                }
                mf.write(json.dumps(line) + "\n")
                mf.flush()
                metrics.append(line)
                if not cfg.quiet:
                    n_pass = sum(1 for v in probe["episodes"].values()
                                 if v.get("passed"))
                    print(f"[accept u{u}] stage1={probe['stage1_passed']} "
                          f"({n_pass}/{len(probe['episodes'])} episodes) "
                          f"confirmed={probe['confirmed']} "
                          f"probe_wall={probe['probe_wall_s']:.1f}s")
                if not probe["confirmed"]:
                    return False
                accepted = True
                write_acceptance(out, cfg, actor, u, env_steps,
                                 train_wall(), probe_wall(), probe)
                save_checkpoint(out / "latest.pt", u, actor, critic, optimizer,
                                sample_gen, perm_gen, env_steps, faults_total,
                                cfg, train_wall(), probe_wall())
                print(f"WALKING ACCEPTED at update {u} after "
                      f"{train_wall():.1f} s ({probe_wall():.1f} s probing)")
                return True

            if cfg.preflight_steps > 0 and not cfg.resume:
                mean, std, pf_faults = preflight_reward_check(
                    env, cfg.preflight_steps, cfg.seed, out, ff, cfg.quiet)
                faults_total += pf_faults

            obs = env.reset(seed=cfg.seed)   # clean deterministic start
            ep_ret = np.zeros(E, np.float64)
            ep_len = np.zeros(E, np.int64)
            if recurrent:                    # fresh episodes start at h = 0
                h_a = actor.initial_state(E, device)
                h_c = critic.initial_state(E, device)

            for u in range(start_update + 1, cfg.updates + 1):
                if cfg.max_wall_s and time.perf_counter() - t_start >= cfg.max_wall_s:
                    stopped_by_wall = True
                    if not cfg.quiet:
                        print(f"[gpu_train] --max-wall-s {cfg.max_wall_s} reached "
                              f"after {last_update} updates; stopping cleanly")
                    break
                t0 = time.perf_counter()
                try:
                    if recurrent:
                        ro = collect_rollout_recurrent(
                            env, actor, critic, obs, h_a, h_c, cfg.horizon,
                            sample_gen, device, ep_ret, ep_len)
                        h_a, h_c = ro.h_actor, ro.h_critic
                    else:
                        ro = collect_rollout(env, actor, critic, obs, cfg.horizon,
                                             sample_gen, device, ep_ret, ep_len)
                except SolverFault as fault:
                    # Poison the whole rollout window: no update, full reset.
                    faults_total += 1
                    record_fault(out, ff, u, fault)
                    obs = env.reset()
                    ep_ret[:] = 0.0
                    ep_len[:] = 0
                    if recurrent:            # every env restarted: zero hidden
                        h_a = actor.initial_state(E, device)
                        h_c = critic.initial_state(E, device)
                    line = {
                        "kind": "train", "update": u, "skipped": "solver_fault",
                        "env_steps": env_steps, "faults": 1,
                        "faults_total": faults_total + getattr(env, "fault_count", 0),
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
                if recurrent:
                    batch = make_batch_recurrent(ro, critic, ppo_cfg.gamma, ppo_cfg.lam)
                    stats = recurrent_ppo_update(actor, critic, optimizer, batch,
                                                 ppo_cfg, perm_gen)
                else:
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
                    "faults_total": faults_total + getattr(env, "fault_count", 0),
                }
                # gate_proxy_*: in-kernel judge-shadow gait counters (means
                # over each env's LAST COMPLETED episode; robot-generic --
                # any lane exposing gate_proxy()). Metrics only; see the
                # honesty note in duck_cuda.h: tick-resolution clause
                # shadow, never a substitute for the frozen judge.
                lane = getattr(env, "_lane", None)
                if lane is not None and hasattr(lane, "gate_proxy"):
                    try:
                        gp = lane.gate_proxy()
                        line["gate_proxy_ep_qualified_l"] = round(
                            float(gp["episode_qualified_left"].mean()), 3)
                        line["gate_proxy_ep_qualified_r"] = round(
                            float(gp["episode_qualified_right"].mean()), 3)
                        line["gate_proxy_ep_alt_violations"] = round(
                            float(gp["episode_alternation_violations"].mean()),
                            3)
                    except Exception:
                        pass                 # metrics must never kill training
                if controller is not None:
                    line["curriculum_stage"] = controller.stage.name
                    event = controller.observe(line)
                    if event is not None:
                        controller.apply(env)     # knobs at update boundary
                        stage_line = {"kind": "curriculum", **event,
                                      "env_steps": env_steps}
                        mf.write(json.dumps(stage_line) + "\n")
                        metrics.append(stage_line)
                        if not cfg.quiet:
                            print(f"[curriculum] {event['direction']} -> "
                                  f"'{event['name']}' at u{u} "
                                  f"({event['reason']})")
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
                                    perm_gen, env_steps, faults_total, cfg,
                                    train_wall(), probe_wall())
                    shutil.copy2(ck, out / "latest.pt")

                if cfg.accept_every > 0 and u % cfg.accept_every == 0:
                    if do_probe(u):
                        break        # WALKING ACCEPTED: stop training, exit 0

            # One last probe at the very end (wall stop or updates exhausted),
            # unless the final update was already probed or already accepted.
            if (cfg.accept_every > 0 and not accepted
                    and last_update > start_update
                    and last_probe_update != last_update):
                do_probe(last_update)
    finally:
        # Always checkpoint at exit and always write the cpu actor for local
        # evaluation, even after a wall-clock stop or an exception.
        try:
            save_checkpoint(out / "latest.pt", last_update, actor, critic,
                            optimizer, sample_gen, perm_gen, env_steps,
                            faults_total, cfg, train_wall(), probe_wall())
            # Self-describing actor file: {"arch": "ff"|"gru", "state_dict": ...}
            # (legacy consumers of plain ff state_dicts: see ppo.unpack_actor_file)
            torch.save(
                {"arch": cfg.policy,
                 "state_dict": {k: v.detach().cpu()
                                for k, v in actor.state_dict().items()}},
                out / "actor_final.pt")
        finally:
            env.close()
    if not cfg.quiet:
        print(f"[gpu_train] done: updates={last_update} env_steps={env_steps} "
              f"faults={faults_total} wall={time.perf_counter() - t_start:.1f}s "
              f"stopped_by_wall={stopped_by_wall} accepted={accepted}")
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="walk.train.gpu_train",
        description="Single-process PPO trainer over the batched CUDA lane")
    d = GpuTrainConfig()
    p.add_argument("--robot", choices=("duck", "humanoid", "arm"), default=d.robot,
                   help="robot contract: env/lane classes and OBS/ACT dims "
                        "(duck = the original byte-identical path)")
    p.add_argument("--variant", default=d.variant, metavar="NAME",
                   help="robot FAMILY member (validated per robot by "
                        "robot_classes): humanoid -> h1_tall | h1_stocky "
                        "(unset = the accepted H1.1 base, humanoid/"
                        "h1_family.py); arm -> kr240 | lite; duck: none")
    p.add_argument("--curriculum", default=d.curriculum, metavar="LADDER",
                   help="tech-tree curriculum: builtin ladder name (e.g. "
                        "humanoid-walk) or a JSON ladder file; requires "
                        "--lane-env; off by default (exact legacy behavior)")
    p.add_argument("--envs", type=int, default=d.envs)
    p.add_argument("--horizon", type=int, default=d.horizon)
    p.add_argument("--updates", type=int, default=d.updates)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device, help="torch device (cuda on the GPU host)")
    p.add_argument("--library", default=None,
                   help="path to libduck_cuda*.so / serial dylib (default: "
                        "DUCK_CUDA_LIBRARY env or local serial build)")
    p.add_argument("--lane-env", action="store_true",
                   help="use the ABI-v3 device policy path (1 launch/step)")
    p.add_argument("--randomization", default=None,
                   help='JSON DR config, e.g. {"r_mass":0.1,"max_latency_steps":1}')
    p.add_argument("--out", required=True)
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="checkpoint path, or bare flag for <out>/latest.pt")
    p.add_argument("--init-actor", default=None,
                   help="actor file (or full checkpoint) for weights-only "
                        "init; critic/optimizer start fresh. With --policy "
                        "gru an FF source triggers the residual FF->GRU "
                        "warm start (see module docstring)")
    p.add_argument("--rsi-fraction", type=float, default=0.0,
                   help="fraction of resets initialized from reference-gait "
                        "states (DeepMimic RSI); requires a lane with set_rsi")
    p.add_argument("--max-wall-s", type=float, default=0.0,
                   help="stop cleanly (checkpoint + actor_final.pt) after this many seconds")
    p.add_argument("--policy", choices=("ff", "gru"), default=d.policy,
                   help="ff: feed-forward MLP (default); gru: tiny recurrent "
                        "policy (implicit system ID), trained with truncated BPTT")
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--gamma", type=float, default=d.gamma)
    p.add_argument("--gae-lambda", type=float, default=d.gae_lambda)
    p.add_argument("--perturbation", type=float, default=d.perturbation,
                   help="per-env joint perturbation bound in rad (<= 0.02)")
    p.add_argument("--accept-every", type=int, default=d.accept_every, metavar="N",
                   help="0 = off; every N updates run the strict-acceptance "
                        "probe (6 x 8 s episodes + 3 x 11 s stability "
                        "confirmation) and STOP with exit 0 on confirmed pass")
    p.add_argument("--checkpoint-every", type=int, default=d.checkpoint_every)
    p.add_argument("--preflight-steps", type=int, default=d.preflight_steps,
                   help="random-action reward-sensitivity steps; 0 skips")
    p.add_argument("--torch-threads", type=int, default=d.torch_threads)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--validate-only", action="store_true",
                   help="validate flags/robot/curriculum/relay files, then exit 0 "
                        "(launcher preflight from the archived payload)")
    return p


def config_from_args(args: argparse.Namespace) -> GpuTrainConfig:
    return GpuTrainConfig(
        robot=args.robot,
        variant=args.variant,
        curriculum=args.curriculum,
        envs=args.envs, horizon=args.horizon, updates=args.updates, seed=args.seed,
        device=args.device, library=args.library, lane_env=args.lane_env, randomization=(json.loads(args.randomization) if args.randomization else None), out=args.out, resume=args.resume, init_actor=args.init_actor, rsi_fraction=args.rsi_fraction, validate_only=args.validate_only,
        max_wall_s=args.max_wall_s, policy=args.policy, lr=args.lr,
        gamma=args.gamma, gae_lambda=args.gae_lambda, perturbation=args.perturbation,
        accept_every=args.accept_every,
        checkpoint_every=args.checkpoint_every, preflight_steps=args.preflight_steps,
        torch_threads=args.torch_threads, quiet=args.quiet,
    )


def main(argv: list[str] | None = None) -> list[dict]:
    args = build_argparser().parse_args(argv)
    return train(config_from_args(args))


if __name__ == "__main__":
    main()
