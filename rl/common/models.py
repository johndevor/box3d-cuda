"""The networks every skill trains: an actor-critic with its own observation normalisers.

ActorCritic(actor_in, critic_in, act): obs_norm + actor MLP (+ tanh when squash) -> the action mean; crit_norm + critic
MLP -> the value; a state-independent log_std. The actor's input is the policy's observation (a student, an asymmetric
actor) or observation + privileged state (a teacher); the critic always sees both. The parameter names (obs_norm,
crit_norm, actor, critic, log_std) are those of rl/skills/duck_walk's earlier checkpoints, which load unchanged.
The exported graph is obs -> clamp((obs - mean) / sqrt(var + 1e-8), +-clip) -> actor -> (tanh): raw observations in.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    def __init__(self, n, clip=10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("var", torch.ones(n))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip
        self.frozen = False

    @torch.no_grad()
    def update(self, x):
        if self.frozen:
            return
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot

    def forward(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)


def mlp(i, o, hidden=(256, 256)):
    layers, last = [], i
    for k in hidden:
        layers += [nn.Linear(last, k), nn.ELU()]
        last = k
    layers.append(nn.Linear(last, o))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, actor_in, critic_in, act, actor_hidden=(256, 256), critic_hidden=(256, 256), squash=True, log_std=-0.5,
                 norm_clip=10.0, init_bias=None, init_log_std=None):
        super().__init__()
        self.sizes = dict(actor_in=actor_in, critic_in=critic_in, act=act, actor_hidden=tuple(actor_hidden), critic_hidden=tuple(critic_hidden),
                          squash=squash, norm_clip=norm_clip)
        self.obs_norm, self.crit_norm = RunningNorm(actor_in, norm_clip), RunningNorm(critic_in, norm_clip)
        self.actor = mlp(actor_in, act, actor_hidden)
        self.critic = mlp(critic_in, 1, critic_hidden)
        self.log_std = nn.Parameter(torch.full((act,), float(log_std)))
        self.squash = squash
        with torch.no_grad():
            for i, b in (init_bias or {}).items():
                self.actor[-1].bias[int(i)] = b
            for i, s in (init_log_std or {}).items():
                self.log_std[int(i)] = s

    def mean(self, x):
        y = self.actor(self.obs_norm(x))
        return torch.tanh(y) if self.squash else y

    def dist(self, x):
        return torch.distributions.Normal(self.mean(x), self.log_std.exp().expand(x.shape[0], -1))

    def value(self, x):
        return self.critic(self.crit_norm(x)).squeeze(-1)

    def actor_state(self):
        """What an exported policy needs (the actor, its normaliser, log_std) plus the sizes to rebuild it."""
        return dict(sizes=self.sizes, state={k: v for k, v in self.state_dict().items() if not k.startswith(("critic.", "crit_norm."))})


def build(spec: dict, actor_in, critic_in, act):
    """An ActorCritic from a skill's model spec (rl/common/skill.py Skill.model)."""
    return ActorCritic(actor_in, critic_in, act, **spec)


def from_actor_state(d):
    s = d["sizes"]
    ac = ActorCritic(s["actor_in"], s["critic_in"], s["act"], s["actor_hidden"], s["critic_hidden"], s["squash"], norm_clip=s["norm_clip"])
    ac.load_state_dict(d["state"], strict=False)
    return ac


class Exported(nn.Module):
    """The deployable graph: raw observations in, the action mean out (World2's contract runs it)."""

    def __init__(self, ac: ActorCritic):
        super().__init__()
        self.net = ac.actor
        self.squash, self.clip = ac.squash, ac.obs_norm.clip
        self.register_buffer("mean", ac.obs_norm.mean.clone())
        self.register_buffer("inv_std", 1.0 / torch.sqrt(ac.obs_norm.var + 1e-8))

    def forward(self, obs):
        y = self.net(torch.clamp((obs - self.mean) * self.inv_std, -self.clip, self.clip))
        return torch.tanh(y) if self.squash else y
