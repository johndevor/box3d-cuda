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
    # recurrent (--policy gru) sizes; ignored by the feed-forward nets
    gru_hidden: int = 128
    gru_head: int = 64


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


# ---------------------------------------------------------------------------
# Recurrent policy (RAPTOR-style implicit system ID): a small 1-layer GRU
# followed by a reduced MLP head. Hidden state is always explicit so the
# trainer owns reset/carry semantics; sequences are recomputed through the
# GRU during the update (truncated BPTT over the rollout window).
# ---------------------------------------------------------------------------


def trunk_hidden_from_state_dict(sd, prefix: str = "ff."):
    """Hidden sizes of an _mlp saved under `prefix` (e.g. (256, 256) from
    ff.0.weight [256, obs], ff.2.weight [256, 256], ff.4.weight [act, 256]);
    None when no such trunk is present in the state dict."""
    layers = sorted(int(k[len(prefix):].split(".")[0]) for k in sd
                    if k.startswith(prefix) and k.endswith(".weight"))
    if not layers:
        return None
    return tuple(int(sd[f"{prefix}{i}.weight"].shape[0]) for i in layers[:-1])


def _lazy_trunk_hook(module, state_dict, prefix, *_args):
    """load_state_dict pre-hook (nn.Module.register_load_state_dict_pre_hook,
    signature (module, state_dict, prefix, ...)): a RecurrentActor/RecurrentCritic built
    WITHOUT a feed-forward trunk grows one when the incoming state dict
    carries `ff.*` keys (sizes inferred), so generic loaders that construct
    `RecurrentActor(OBS, ACT)` (acceptance harness, probes) load a
    warm-started residual policy unchanged. Runs before the module's own
    keys are consumed and before children are recursed into, so the new
    child is loaded normally by the same call."""
    if getattr(module, "ff", None) is None:
        hidden = trunk_hidden_from_state_dict(state_dict, prefix + "ff.")
        if hidden is not None:
            module.ff = module._make_trunk(hidden).to(module.log_std.device
                                                     if hasattr(module, "log_std")
                                                     else next(module.parameters()).device)


class RecurrentActor(nn.Module):
    """GRU(obs->H) -> Linear(H->head)+Tanh -> mu; state-independent log-std,
    tanh squash to [-1,1]. Same distribution math as Actor.

    Optional RESIDUAL feed-forward trunk (`ff_hidden`, e.g. (256, 256)):
    mu = ff(obs) + head(GRU(obs, h)). This is the distillation-free warm
    start from an accepted feed-forward Actor (warm_start_recurrent_from_ff):
    the trunk receives the FF weights and the GRU head's output layer is
    ZEROED, so at init the recurrent correction is exactly 0 and the policy
    (deterministic action AND sampling distribution, log_std copied) is
    bit-identical to the FF specialist; PPO then learns the history-dependent
    correction (implicit system ID under domain randomization) on top of a
    walker that already walks. ff_hidden=None is the original plain GRU
    policy (byte-identical state dict and behavior)."""

    arch = "gru"

    def __init__(self, obs_dim: int, act_dim: int, gru_hidden: int = 128,
                 head_hidden: int = 64, log_std_init: float = -0.5,
                 ff_hidden=None):
        super().__init__()
        self.obs_dim, self.act_dim = int(obs_dim), int(act_dim)
        self.gru_hidden = int(gru_hidden)
        self.gru = nn.GRU(obs_dim, self.gru_hidden, num_layers=1)
        self.mu_net = _mlp(self.gru_hidden, (int(head_hidden),), act_dim, out_gain=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self.ff = self._make_trunk(ff_hidden) if ff_hidden else None
        self.register_load_state_dict_pre_hook(_lazy_trunk_hook)

    def _make_trunk(self, hidden) -> nn.Sequential:
        return _mlp(self.obs_dim, tuple(int(h) for h in hidden), self.act_dim,
                    out_gain=0.01)

    def initial_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.gru_hidden, device=device)

    def _std(self) -> torch.Tensor:
        return torch.exp(self.log_std.clamp(-5.0, 2.0))

    def _mu(self, feats: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        mu = self.mu_net(feats)
        return mu + self.ff(obs) if self.ff is not None else mu

    def dist(self, obs: torch.Tensor, h: torch.Tensor):
        """One step. obs [B, OBS], h [B, H] -> (mu [B, A], std [A], h_next [B, H])."""
        y, h1 = self.gru(obs.unsqueeze(0), h.unsqueeze(0))
        return self._mu(y.squeeze(0), obs), self._std(), h1.squeeze(0)

    def dist_seq(self, obs_seq: torch.Tensor, h0: torch.Tensor,
                 done_seq: torch.Tensor | None = None):
        """Whole window with BPTT. obs_seq [T, B, OBS], h0 [B, H],
        done_seq [T, B] float (1.0 zeroes h AFTER step t, mirroring the
        rollout's env auto-reset). Returns (mu [T, B, A], std [A], h_T [B, H])."""
        h = h0.unsqueeze(0)
        feats = []
        for t in range(obs_seq.shape[0]):
            y, h = self.gru(obs_seq[t].unsqueeze(0), h)
            feats.append(y.squeeze(0))
            if done_seq is not None:
                h = h * (1.0 - done_seq[t]).view(1, -1, 1)
        return self._mu(torch.stack(feats), obs_seq), self._std(), h.squeeze(0)

    def sample(self, obs: torch.Tensor, h: torch.Tensor, generator: torch.Generator):
        """Return (raw u, squashed a, log-prob, h_next)."""
        mu, std, h1 = self.dist(obs, h)
        u = mu + std * torch.randn(mu.shape, generator=generator,
                                   dtype=mu.dtype, device=mu.device)
        return u, torch.tanh(u), tanh_gaussian_log_prob(mu, std, u), h1

    def deterministic(self, obs: torch.Tensor, h: torch.Tensor):
        """Return (action, h_next) for evaluation (tanh of the mean)."""
        mu, _std, h1 = self.dist(obs, h)
        return torch.tanh(mu), h1


class RecurrentCritic(nn.Module):
    """GRU value net; optional residual feed-forward trunk (ff_hidden) so a
    warm start can carry the FF checkpoint's critic too: v = ff(obs) +
    head(GRU(obs, h)), head output zeroed at warm start."""

    arch = "gru"

    def __init__(self, obs_dim: int, gru_hidden: int = 128, head_hidden: int = 64,
                 ff_hidden=None):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.gru_hidden = int(gru_hidden)
        self.gru = nn.GRU(obs_dim, self.gru_hidden, num_layers=1)
        self.v_net = _mlp(self.gru_hidden, (int(head_hidden),), 1, out_gain=1.0)
        self.ff = self._make_trunk(ff_hidden) if ff_hidden else None
        self.register_load_state_dict_pre_hook(_lazy_trunk_hook)

    def _make_trunk(self, hidden) -> nn.Sequential:
        return _mlp(self.obs_dim, tuple(int(h) for h in hidden), 1, out_gain=1.0)

    def initial_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.gru_hidden, device=device)

    def _v(self, feats: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        v = self.v_net(feats)
        return (v + self.ff(obs) if self.ff is not None else v).squeeze(-1)

    def forward(self, obs: torch.Tensor, h: torch.Tensor):
        """One step. obs [B, OBS], h [B, H] -> (v [B], h_next [B, H])."""
        y, h1 = self.gru(obs.unsqueeze(0), h.unsqueeze(0))
        return self._v(y.squeeze(0), obs), h1.squeeze(0)

    def value_seq(self, obs_seq: torch.Tensor, h0: torch.Tensor,
                  done_seq: torch.Tensor | None = None):
        """Whole window with BPTT: (v [T, B], h_T [B, H])."""
        h = h0.unsqueeze(0)
        feats = []
        for t in range(obs_seq.shape[0]):
            y, h = self.gru(obs_seq[t].unsqueeze(0), h)
            feats.append(y.squeeze(0))
            if done_seq is not None:
                h = h * (1.0 - done_seq[t]).view(1, -1, 1)
        return self._v(torch.stack(feats), obs_seq), h.squeeze(0)


def make_recurrent_nets(obs_dim: int, act_dim: int, cfg: PPOConfig,
                        ff_hidden=None) -> tuple[RecurrentActor, RecurrentCritic]:
    """ff_hidden=None: the original plain GRU nets (unchanged). A tuple
    (e.g. cfg.hidden) adds the residual feed-forward trunks used by the
    FF -> GRU warm start."""
    actor = RecurrentActor(obs_dim, act_dim, cfg.gru_hidden, cfg.gru_head,
                           cfg.log_std_init, ff_hidden=ff_hidden)
    critic = RecurrentCritic(obs_dim, cfg.gru_hidden, cfg.gru_head,
                             ff_hidden=ff_hidden)
    return actor, critic


@torch.no_grad()
def warm_start_recurrent_from_ff(actor: RecurrentActor, ff_actor_sd: dict,
                                 critic: RecurrentCritic | None = None,
                                 ff_critic_sd: dict | None = None) -> None:
    """Distillation-free FF -> residual-GRU warm start.

    actor: a RecurrentActor built with ff_hidden == the FF Actor's hidden
    sizes (trunk_hidden_from_state_dict(ff_actor_sd, "mu_net.")). Copies
    mu_net.* -> ff.* and log_std, then ZEROES the GRU head's output layer
    (mu_net[-1]) so mu == ff(obs) exactly: the warm-started policy is
    bit-identical to the FF specialist at step 0 and the GRU pathway starts
    as a pure zero correction (its gradient enters through the zeroed layer
    first -- standard residual-policy init). The GRU's own weights keep
    torch's default init: with the output layer at zero they are
    irrelevant to the initial behavior and only shape how fast the
    correction can be learned. Same for the critic when both are given:
    v_net.* -> ff.*, v_net[-1] zeroed, so the value baseline starts at the
    FF critic's estimate instead of noise."""
    if actor.ff is None:
        raise ValueError("warm start needs a RecurrentActor with an ff trunk")
    trunk = {k[len("mu_net."):]: v for k, v in ff_actor_sd.items()
             if k.startswith("mu_net.")}
    actor.ff.load_state_dict(trunk)            # strict: shapes must match
    actor.log_std.copy_(ff_actor_sd["log_std"].to(actor.log_std.device))
    nn.init.zeros_(actor.mu_net[-1].weight)
    nn.init.zeros_(actor.mu_net[-1].bias)
    if critic is not None and ff_critic_sd is not None:
        if critic.ff is None:
            raise ValueError("warm start needs a RecurrentCritic with an ff trunk")
        critic.ff.load_state_dict({k[len("v_net."):]: v for k, v in
                                   ff_critic_sd.items() if k.startswith("v_net.")})
        nn.init.zeros_(critic.v_net[-1].weight)
        nn.init.zeros_(critic.v_net[-1].bias)


def unpack_actor_file(obj):
    """Return (arch, state_dict) from a saved actor file.

    New self-describing format: {"arch": "ff"|"gru", "state_dict": {...}}.
    Legacy actor_final.pt files are a plain feed-forward state_dict."""
    if isinstance(obj, dict) and "arch" in obj and "state_dict" in obj:
        return str(obj["arch"]), obj["state_dict"]
    return "ff", obj


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


def recurrent_ppo_update(
    actor: RecurrentActor,
    critic: RecurrentCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    cfg: PPOConfig,
    generator: torch.Generator,
) -> dict[str, float]:
    """PPO update for the GRU policy with truncated BPTT over the rollout
    window. Minibatches are ENV SLICES (whole [T]-length sequences), never
    flat transitions: each epoch re-runs the sequences through the GRU from
    the stored window-initial hidden states (h0_actor / h0_critic), zeroing
    hidden rows at each stored done exactly as the rollout did.

    batch (all [T, N, ...] on one device): obs, raw_act, logp, val, adv, ret,
    done, plus h0_actor [N, H] and h0_critic [N, H]. Loss formulas, advantage
    normalization, clipping, entropy bonus, KL estimator and target-kl early
    stop are identical to ppo_update.
    """
    obs, raw_act = batch["obs"], batch["raw_act"]
    old_logp, old_val = batch["logp"], batch["val"]
    ret, done = batch["ret"], batch["done"]
    h0_a, h0_c = batch["h0_actor"], batch["h0_critic"]
    adv = batch["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N = obs.shape[1]
    mb_envs = max(1, N // max(1, cfg.minibatches))
    params = list(actor.parameters()) + list(critic.parameters())

    pi_losses, v_losses, entropies, kls, clip_fracs = [], [], [], [], []
    stop = False
    for _ in range(cfg.epochs):
        perm = torch.randperm(N, generator=generator)
        for start in range(0, N, mb_envs):
            idx = perm[start : start + mb_envs].to(obs.device)
            o, d = obs[:, idx], done[:, idx]
            mu, std, _h = actor.dist_seq(o, h0_a[idx], d)
            logp = tanh_gaussian_log_prob(mu, std, raw_act[:, idx])
            log_ratio = logp - old_logp[:, idx]
            ratio = log_ratio.exp()

            a = adv[:, idx]
            pg1 = ratio * a
            pg2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * a
            pi_loss = -torch.min(pg1, pg2).mean()

            entropy = gaussian_entropy(std).mean()

            v, _hc = critic.value_seq(o, h0_c[idx], d)
            ov = old_val[:, idx]
            v_clipped = ov + (v - ov).clamp(-cfg.clip_value, cfg.clip_value)
            r = ret[:, idx]
            v_loss = 0.5 * torch.max((v - r) ** 2, (v_clipped - r) ** 2).mean()

            loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                kl = ((ratio - 1.0) - log_ratio).mean()
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
