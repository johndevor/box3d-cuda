"""PPO core for the duck trainer (CPU, float32).

Tanh-squashed Gaussian actor + value critic as two separate small MLPs
(2 x 256 by default), GAE(lambda), clipped surrogate objective, clipped
value loss, entropy bonus, per-batch advantage normalization.

All stochastic operations (action sampling, minibatch permutation) draw from
an explicit torch.Generator so training is deterministic given a seed and the
generator state can be checkpointed bitwise.
"""
from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn

_LOG2 = math.log(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


@dataclasses.dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    clip_value: float = 0.2
    epochs: int = 4
    minibatches: int = 8
    ent_coef: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None  # early-stop epochs when exceeded (1.5x)
    hidden: tuple[int, int] = (256, 256)
    log_std_init: float = -0.5


def _mlp(in_dim: int, hidden, out_dim: int, out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        d = h
    out = nn.Linear(d, out_dim)
    nn.init.orthogonal_(out.weight, gain=out_gain)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """State-dependent mean, state-independent log-std, tanh squash to [-1,1]."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256), log_std_init: float = -0.5):
        super().__init__()
        self.mu_net = _mlp(obs_dim, hidden, act_dim, out_gain=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def dist(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std.clamp(-5.0, 2.0))
        return mu, std

    def sample(self, obs: torch.Tensor, generator: torch.Generator):
        """Return (raw pre-tanh action u, squashed action a, log-prob of a)."""
        mu, std = self.dist(obs)
        u = mu + std * torch.randn(mu.shape, generator=generator, dtype=mu.dtype)
        return u, torch.tanh(u), tanh_gaussian_log_prob(mu, std, u)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mu_net(obs))


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden=(256, 256)):
        super().__init__()
        self.v_net = _mlp(obs_dim, hidden, 1, out_gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v_net(obs).squeeze(-1)


def tanh_gaussian_log_prob(mu: torch.Tensor, std: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """log p(tanh(u)) where u ~ N(mu, std). Uses the raw sample (no atanh)."""
    logp_u = (-0.5 * ((u - mu) / std) ** 2 - torch.log(std) - _LOG_SQRT_2PI).sum(-1)
    # log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u)), numerically stable
    corr = (2.0 * (_LOG2 - u - F.softplus(-2.0 * u))).sum(-1)
    return logp_u - corr


def gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
    """Entropy of the pre-squash Gaussian (standard PPO entropy bonus proxy)."""
    return (0.5 + _LOG_SQRT_2PI + torch.log(std)).sum(-1)


def make_nets(obs_dim: int, act_dim: int, cfg: PPOConfig) -> tuple[Actor, Critic]:
    actor = Actor(obs_dim, act_dim, cfg.hidden, cfg.log_std_init)
    critic = Critic(obs_dim, cfg.hidden)
    return actor, critic


@torch.no_grad()
def compute_gae(
    rew: torch.Tensor,      # [T, N] float32
    done: torch.Tensor,     # [T, N] float32 (1.0 where transition ended episode)
    val: torch.Tensor,      # [T, N] V(s_t)
    last_val: torch.Tensor, # [N]    V(s_T)
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (advantages [T,N], returns [T,N]); never bootstraps across done."""
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    lastgae = torch.zeros_like(last_val)
    for t in range(T - 1, -1, -1):
        nonterm = 1.0 - done[t]
        next_val = val[t + 1] if t + 1 < T else last_val
        delta = rew[t] + gamma * next_val * nonterm - val[t]
        lastgae = delta + gamma * lam * nonterm * lastgae
        adv[t] = lastgae
    return adv, adv + val


def ppo_update(
    actor: Actor,
    critic: Critic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],  # obs, raw_act, logp, adv, ret, val (flat [B,...])
    cfg: PPOConfig,
    generator: torch.Generator,
) -> dict[str, float]:
    obs, raw_act = batch["obs"], batch["raw_act"]
    old_logp, old_val = batch["logp"], batch["val"]
    ret = batch["ret"]
    adv = batch["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    B = obs.shape[0]
    mb_size = max(1, B // max(1, cfg.minibatches))
    params = list(actor.parameters()) + list(critic.parameters())

    pi_losses, v_losses, entropies, kls, clip_fracs = [], [], [], [], []
    stop = False
    for _ in range(cfg.epochs):
        perm = torch.randperm(B, generator=generator)
        for start in range(0, B, mb_size):
            idx = perm[start : start + mb_size]
            mu, std = actor.dist(obs[idx])
            logp = tanh_gaussian_log_prob(mu, std, raw_act[idx])
            log_ratio = logp - old_logp[idx]
            ratio = log_ratio.exp()

            a = adv[idx]
            pg1 = ratio * a
            pg2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * a
            pi_loss = -torch.min(pg1, pg2).mean()

            entropy = gaussian_entropy(std).mean()

            v = critic(obs[idx])
            v_clipped = old_val[idx] + (v - old_val[idx]).clamp(-cfg.clip_value, cfg.clip_value)
            v_loss = 0.5 * torch.max((v - ret[idx]) ** 2, (v_clipped - ret[idx]) ** 2).mean()

            loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                kl = ((ratio - 1.0) - log_ratio).mean()  # k3 estimator, >= 0
                clip_frac = ((ratio - 1.0).abs() > cfg.clip).float().mean()
            pi_losses.append(float(pi_loss.detach()))
            v_losses.append(float(v_loss.detach()))
            entropies.append(float(entropy.detach()))
            kls.append(float(kl))
            clip_fracs.append(float(clip_frac))
            if cfg.target_kl is not None and kls[-1] > 1.5 * cfg.target_kl:
                stop = True
                break
        if stop:
            break

    n = max(1, len(pi_losses))
    return {
        "pi_loss": sum(pi_losses) / n,
        "v_loss": sum(v_losses) / n,
        "entropy": sum(entropies) / n,
        "approx_kl": sum(kls) / n,
        "clip_frac": sum(clip_fracs) / n,
        "minibatches_done": len(pi_losses),
    }
