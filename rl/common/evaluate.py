"""The evaluation harness: one deterministic episode per world (the action mean, no autoreset), on held-out seeds.

evaluate(skill, act_fn, cfg, n, seed, device) -> {n, success, codes, splits...}
    act_fn(obs, priv) -> action. Built from a torch model (model_act) or an ONNX file (onnx_act: what World2 runs).
Each world's first ending is recorded: success, its end code and every info tensor at that moment ("final");
skill.eval_splits(p0, succ, final) adds per-subset rates (p0: the worlds' parameters at the start).
A skill with its own measure (the duck's walking metrics) provides skill.evaluate and is called instead.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch


def model_act(ac, actor_in):
    def f(obs, priv):
        with torch.no_grad():
            return ac.mean(actor_in(obs, priv))
    return f


def onnx_act(path):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path))
    name_in, name_out = sess.get_inputs()[0].name, sess.get_outputs()[0].name

    one = sess.get_inputs()[0].shape[0] == 1          # (a graph exported with a fixed batch of 1)

    def f(obs, priv):
        x = obs.detach().cpu().numpy().astype(np.float32)
        a = np.concatenate([sess.run([name_out], {name_in: x[i:i + 1]})[0] for i in range(len(x))]) if one else sess.run([name_out], {name_in: x})[0]
        return torch.as_tensor(a, device=obs.device)
    return f


def episodic(skill, act_fn, cfg, n, seed, device, max_ticks=None, env=None):
    env = skill.make_env(n, device=device, seed=seed, cfg=cfg, **(env or {}))
    obs, priv = env.observe()
    p0 = {k: v.clone() for k, v in getattr(env, "p", {}).items() if torch.is_tensor(v)}
    finished = torch.zeros(n, dtype=torch.bool, device=env.dev)
    succ = torch.zeros(n, dtype=torch.bool, device=env.dev)
    codes = torch.zeros(n, dtype=torch.long, device=env.dev)
    final = {}
    for _ in range(max_ticks or skill.eval_ticks):
        a = act_fn(obs, priv)
        obs, priv, r, done, info = env.step(a, autoreset=False)
        new = done & ~finished
        if "success" in info:
            succ |= new & info["success"]
        if "code" in info:
            codes = torch.where(new, info["code"], codes)
        for k, v in info.items():
            if torch.is_tensor(v) and v.shape[:1] == (n,):
                if k not in final:
                    final[k] = torch.zeros_like(v)
                final[k] = torch.where(new.view(-1, *([1] * (v.dim() - 1))), v, final[k])
        finished |= done
        if finished.all():
            break
    names = getattr(env, "code_names", {})
    res = dict(n=n, success=round(succ.float().mean().item(), 4),
               codes=dict(Counter(names.get(int(c), "unfinished") if f else "unfinished" for c, f in zip(codes.tolist(), finished.tolist())).most_common()))
    if skill.eval_splits:
        res.update(skill.eval_splits(p0, succ, final))
    return res


def evaluate(skill, act_fn, cfg, n, seed, device, max_ticks=None, env=None):
    """env: the env's non-random settings (e.g. the duck's gait clock), as training used them."""
    if skill.evaluate:
        return skill.evaluate(act_fn, cfg, n, seed, device, env=env or {})
    return episodic(skill, act_fn, cfg, n, seed, device, max_ticks, env)


def rate(succ, m):
    return round(succ[m].float().mean().item(), 4) if m.any() else None
