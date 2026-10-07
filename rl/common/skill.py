"""What a skill module gives the shared pipeline: an env, its reward (inside the env), its randomization spec, and a
small descriptor. Kept small on purpose: a model generated from a World2 design (item 3) plugs in by providing the same.

The env protocol (batched, one device, autoreset):
    n, dev, obs_size, priv_size, act_size          ints / torch.device
    observe() -> (obs[N, obs_size], priv[N, priv_size])     the current observation (fresh worlds included)
    step(action[N, act_size], autoreset=True) -> (obs, priv, reward[N], done[N], info)
        info: tensors per world; at a done world: "success" (bool), optionally "code" (int), "timeout" (bool, the
        episode ran out of time rather than ended), and whatever the skill's eval splits read. With autoreset the
        done worlds start a new episode before obs/priv are returned.
    on_iteration(phase, it, frac)                   once per PPO iteration (resample friction groups, a curriculum)
    iteration_info() -> dict                        extra numbers for the training log
    best_allowed() -> bool                          False while a curriculum is still easier than the real task
    code_names: {int: str}                          names of the end codes
    p: dict of per-world parameter tensors          what eval splits read (clone at reset)
EnvBase supplies the defaults.

Skill descriptor (rl/skills/<name>/skill.py defines SKILL = Skill(...)):
    name, contract_skill             World2's skill name ('insert', 'screw', 'walk')
    make_env(n, device, seed, cfg)   -> env; cfg: randomization overrides (the env's Spec.override)
    spec                             the default randomization Spec
    model                            ActorCritic settings (hidden sizes, squash, init biases, log_std, norm clip)
    recipe                           which recipe (rl/common/recipes.py) and its settings
    eval_sets                        name -> cfg overrides of the held-out box3d evaluation sets
    eval_splits(p0, succ, final)     -> dict of extra rates (optional)
    evaluate(...)                    a custom evaluation (optional; else rl/common/evaluate.py's episodic harness)
    manifest(env, run)               -> the contract fields of World2's manifest.json (id/sha/training are added)
    sim_match                        rl/common/simmatch.py Probe for this skill (optional)
    extensions                       native extensions the env loads (rl/common/cuda.py EXTENSIONS names)
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


class EnvBase:
    code_names: dict = {}

    def on_iteration(self, phase=None, it=0, frac=0.0):
        pass

    def iteration_info(self):
        return {}

    def best_allowed(self):
        return True


@dataclass
class Skill:
    name: str
    contract_skill: str
    make_env: Callable
    spec: Any
    model: dict
    recipe: dict
    manifest: Callable
    eval_sets: dict = field(default_factory=lambda: {"nominal": {}})
    eval_splits: Callable | None = None
    evaluate: Callable | None = None
    sim_match: Any = None
    extensions: tuple = ()
    eval_ticks: int = 200
    description: str = ""


SKILLS = ("peg_insert", "connector_mate", "screw_drive", "duck_walk")


def load_skill(name: str) -> Skill:
    name = name.replace("-", "_")
    try:
        mod = importlib.import_module(f"rl.skills.{name}.skill")
    except ModuleNotFoundError as e:
        raise SystemExit(f"unknown skill {name!r} (rl/skills/<name>/skill.py; known: {', '.join(SKILLS)}): {e}")
    return mod.SKILL
