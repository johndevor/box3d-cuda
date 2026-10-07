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


_CONST = {}


def cvec(values, device):
    """A constant float tensor, cached per (values, device): no host-to-device copy inside a captured step."""
    key = (tuple(float(x) for x in values), str(device))
    t = _CONST.get(key)
    if t is None:
        t = _CONST[key] = torch.tensor(key[0], dtype=torch.float32, device=device)
    return t


def clone_out(t):
    if torch.is_tensor(t):
        return t.clone()
    if isinstance(t, dict):
        return {k: clone_out(v) for k, v in t.items()}
    if isinstance(t, (tuple, list)):
        return type(t)(clone_out(v) for v in t)
    return t


class InPlaceDict(dict):
    """Per-world state: assigning a tensor of the same shape and dtype to an existing key copies into it, so the tensors
    a captured CUDA graph reads and writes stay the same objects."""

    def __setitem__(self, k, val):
        old = self.get(k)
        if torch.is_tensor(val) and torch.is_tensor(old) and old.shape == val.shape and old.dtype == val.dtype and old.device == val.device:
            if old.data_ptr() != val.data_ptr():
                old.copy_(val)
        else:
            super().__setitem__(k, val)


class EnvBase:
    """Defaults of the env protocol, and CUDA-graph stepping: with use_graphs(True) the env's `_step(action)` (pure tensor
    work: no host syncs, no data-dependent Python branches, kernels on the current stream) is captured once and
    replayed; resets stay eager between replays. Tensor attributes and InPlaceDict entries are updated in place, so the
    graph's tensors are the env's own. `graph_stale()` marks the graph for recapture (e.g. new per-group friction)."""
    code_names: dict = {}

    def __setattr__(self, k, val):
        old = self.__dict__.get(k)
        if torch.is_tensor(val) and torch.is_tensor(old) and old.shape == val.shape and old.dtype == val.dtype and old.device == val.device:
            if old.data_ptr() != val.data_ptr():
                old.copy_(val)
        else:
            object.__setattr__(self, k, val)

    def on_iteration(self, phase=None, it=0, frac=0.0):
        pass

    def iteration_info(self):
        return {}

    def best_allowed(self):
        return True

    # ------------------------------------------------------------ CUDA graphs
    def use_graphs(self, on=True):
        object.__setattr__(self, "_graphs_on", bool(on) and self.dev.type == "cuda")
        object.__setattr__(self, "_graph", None)

    def graph_stale(self):
        object.__setattr__(self, "_graph", None)

    def graph_generators(self):
        """The CUDA generators the step draws from (registered with the graph so replays advance them)."""
        g = getattr(self, "gs", None) or getattr(self, "g", None)
        return [g] if isinstance(g, torch.Generator) and g.device.type == "cuda" else []

    def _run_step(self, action):
        if not getattr(self, "_graphs_on", False):
            return self._step(action)
        G = getattr(self, "_graph", None)
        if G is not None:
            G["a"].copy_(action)
            G["graph"].replay()
            # (the graph's outputs are its static buffers, overwritten by the next replay: callers get copies)
            return clone_out(tuple(G["out"]))
        # an eager step on a side stream (it is this step, and the warm-up the capture needs), then the capture
        a_static = action.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            out = self._step(a_static)
        torch.cuda.current_stream().wait_stream(s)
        res = clone_out(tuple(out))
        graph = torch.cuda.CUDAGraph()
        for gen in self.graph_generators():
            graph.register_generator_state(gen)
        pool = getattr(self, "_graph_pool", None) or torch.cuda.graph_pool_handle()
        object.__setattr__(self, "_graph_pool", pool)
        snap = self._graph_snapshot()
        try:
            with torch.cuda.graph(graph, pool=pool):
                gout = self._step(a_static)
        except Exception as e:
            # a step that cannot be captured runs eager from now on, said once (training goes on, slower)
            self._graph_restore(snap)
            torch.cuda.synchronize()
            object.__setattr__(self, "_graphs_on", False)
            import json
            import sys
            print(json.dumps(dict(event="graph_capture_failed", env=type(self).__name__, error=str(e).splitlines()[0][:300])), file=sys.stderr, flush=True)
            return res
        self._graph_restore(snap)          # (capture records without running; the state is the eager step's)
        object.__setattr__(self, "_graph", dict(graph=graph, a=a_static, out=gout))
        return res

    # (the host-side counters a step advances; capturing runs the Python code once more, so they are put back)
    def _graph_snapshot(self):
        return {k: v for k, v in self.__dict__.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}

    def _graph_restore(self, snap):
        for k, v in snap.items():
            object.__setattr__(self, k, v)


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
    recipes: dict = field(default_factory=dict)     # alternative recipes by name (python -m rl.train --recipe NAME)


SKILLS = ("peg_insert", "connector_mate", "screw_drive", "duck_walk")


def load_skill(name: str) -> Skill:
    name = name.replace("-", "_")
    try:
        mod = importlib.import_module(f"rl.skills.{name}.skill")
    except ModuleNotFoundError as e:
        raise SystemExit(f"unknown skill {name!r} (rl/skills/<name>/skill.py; known: {', '.join(SKILLS)}): {e}")
    return mod.SKILL
