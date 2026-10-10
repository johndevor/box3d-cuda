"""Training recipes: how the shared pieces (PPO, DAgger, evaluation) are put together for a skill.

teacher_student (manipulation): a teacher PPO on observations + privileged state, evaluated; a student by DAgger on
    the policy observations only (normaliser = the teacher's, frozen), evaluated; PPO fine-tune of the student with the
    teacher's privileged critic; the best student evaluated on every eval set.
asymmetric (locomotion): PPO with the actor on the policy observations and the critic on observations + privileged
    state; then the eval sets.
stages: a list of recipe runs with their own env settings, each warm-started from the previous stage's policy
    through the skill's `transfer(prev_ac, new_env) -> ac` (e.g. new observation columns at zero weight).

sgs: {"enabled": true, ...} turns on Success-Guided Sampling of the training env's resets (rl/common/sgs.py); off by
    default. Evaluation envs are never SGS envs (held-out seeds, the task distribution as is).

Every recipe returns a summary for eval.json and leaves out/policy.pt (the student/actor that export.py writes).
Settings come from the skill's recipe dict, overridable from the command line (--set key=value, dotted paths).
"""
from __future__ import annotations

import copy
import json
import time

import torch

from . import models
from .distill import dagger
from .evaluate import evaluate, model_act
from .ppo import PPOConfig, best_or_last, ppo_phase


def _model(skill, env, actor_in, extra=None, R=None):
    spec = dict(skill.model, **((R or {}).get("model") or {}), **(extra or {}))      # (a recipe may give its own model settings, e.g. a prior)
    return models.build(spec, actor_in, env.obs_size + env.priv_size, env.act_size).to(env.dev)


def _load(ac, path):
    ac.load_state_dict(torch.load(path, map_location=ac.log_std.device)["model"])
    return ac


def _evals(skill, ac, actor_in, base_cfg, R):
    """The final evaluation sets, all on the same held-out seed (eval_seed + 10000; training never used it)."""
    return {name: evaluate(skill, model_act(ac, actor_in), dict(base_cfg, **over), R["eval_n"], R["eval_seed"] + 10_000, R["device"], R.get("eval_ticks"), R.get("env"))
            for name, over in skill.eval_sets.items()}


def teacher_student(skill, env, R, cfg, out, log, init=None):
    dev = env.dev
    S = {}
    t_in = lambda o, p: torch.cat([o, p], -1)
    s_in = lambda o, p: o
    teacher = _model(skill, env, env.obs_size + env.priv_size, R=R)
    tcfg = PPOConfig().update(R.get("ppo")).update(R.get("teacher"))
    if R.get("resume_teacher"):
        _load(teacher, R["resume_teacher"])
    else:
        S["teacher"] = ppo_phase(env, teacher, t_in, tcfg, out, "teacher", log)
        _load(teacher, best_or_last(out, "teacher"))
    getattr(env, "end_curriculum", lambda: None)()
    S["teacher_eval"] = evaluate(skill, model_act(teacher, t_in), cfg, R["eval_n"], R["eval_seed"], R["device"], R.get("eval_ticks"))
    log(dict(event="teacher_eval", **S["teacher_eval"]))
    student = _model(skill, env, env.obs_size, dict(log_std=R.get("student_log_std", -1.5)), R=R)
    student.critic.load_state_dict(teacher.critic.state_dict())
    student.crit_norm.load_state_dict(teacher.crit_norm.state_dict())
    D = {**dict(iters=150, lr=1e-3, epochs=4, minibatches=4, beta_frac=0.4), **R.get("distill", {})}
    S["distill"] = dagger(env, teacher, student, D["iters"], tcfg.horizon, out, log, D["lr"], D["epochs"], D["minibatches"], D["beta_frac"])
    S["distill_eval"] = evaluate(skill, model_act(student, s_in), cfg, R["eval_n"], R["eval_seed"], R["device"], R.get("eval_ticks"))
    log(dict(event="distill_eval", **S["distill_eval"]))
    with torch.no_grad():
        student.log_std.fill_(R.get("finetune_log_std", -1.5))
        for i, v in (R.get("finetune_log_std_at") or {}).items():
            student.log_std[int(i)] = v
    fcfg = PPOConfig().update(R.get("ppo")).update(R.get("finetune"))
    S["finetune"] = ppo_phase(env, student, s_in, fcfg, out, "student", log)
    _load(student, best_or_last(out, "student"))
    S.update(_evals(skill, student, s_in, cfg, R))
    for k in skill.eval_sets:
        log(dict(event=f"eval_{k}", **S[k]))
    S["steps"] = sum((S.get(k) or {}).get("steps", 0) for k in ("teacher", "distill", "finetune"))
    return student, s_in, S


def asymmetric(skill, env, R, cfg, out, log, init=None):
    a_in = lambda o, p: o
    ac = _model(skill, env, env.obs_size, R=R)
    if init is not None:
        ac = init
    pcfg = PPOConfig().update(R.get("ppo"))
    S = dict(train=ppo_phase(env, ac, a_in, pcfg, out, "actor", log, extra_row=getattr(env, "extra_row", None)))
    _load(ac, best_or_last(out, "actor"))
    S.update(_evals(skill, ac, a_in, cfg, R))
    for k in skill.eval_sets:
        log(dict(event=f"eval_{k}", **S[k]))
    S["steps"] = S["train"]["steps"]
    return ac, a_in, S


RECIPES = dict(teacher_student=teacher_student, asymmetric=asymmetric)


def run(skill, R, cfg, out, log, make_env):
    """R: the recipe settings (kind, its phases, device, eval_n, eval_seed); cfg: randomization overrides."""
    if getattr(skill, "prepare", None):                  # (a skill may settle recipe settings on the device first, e.g. a probed prior)
        skill.prepare(R, cfg, out, log)
    stages = R.get("stages")
    cfg = dict(R.get("cfg", {}), **cfg)                 # (the recipe's own randomization, under the run's overrides)
    graphs = R.get("graphs", False)
    make_env0 = make_env

    def make_env(n, c, e):
        env = make_env0(n, c, e)
        if graphs and env.dev.type == "cuda":
            env.use_graphs(True)
        if (R.get("sgs") or {}).get("enabled"):
            from .sgs import attach
            attach(env, R["sgs"], log)
        return env
    if not stages:
        env = make_env(R["envs"], cfg, R.get("env", {}))
        ac, a_in, S = RECIPES[R["kind"]](skill, env, R, cfg, out, log)
        return ac, a_in, S, env
    S, ac, env, steps = {"stages": {}}, None, None, 0
    for i, st in enumerate(stages):
        Ri = merge(R, st)
        Ri.pop("stages", None)
        sub = out / f"stage{i + 1}-{st.get('name', i + 1)}"
        sub.mkdir(parents=True, exist_ok=True)
        cfg_i = dict(cfg, **st.get("cfg", {}))           # a stage's own randomization (a curriculum), over the run's
        env = make_env(Ri["envs"], cfg_i, Ri.get("env", {}))
        init = skill.transfer(ac, env) if ac is not None else None
        log(dict(event="stage", stage=i + 1, name=st.get("name"), env=Ri.get("env", {}), cfg=st.get("cfg", {}), init=init is not None))
        ac, a_in, Si = RECIPES[Ri["kind"]](skill, env, Ri, cfg_i, sub, log, init=init)
        S["stages"][st.get("name", str(i + 1))] = Si
        steps += Si.get("steps", 0)
    S.update({k: v for k, v in Si.items() if k in skill.eval_sets})
    S["steps"] = steps
    return ac, a_in, S, env


def merge(a, b):
    """Recursive dict merge (b wins)."""
    o = copy.deepcopy(a)
    for k, v in (b or {}).items():
        o[k] = merge(o[k], v) if isinstance(v, dict) and isinstance(o.get(k), dict) else copy.deepcopy(v)
    return o


def set_path(d, path, value):
    """path: dotted keys; a number indexes a list (stages.1.ppo.max_minutes)."""
    keys = path.split(".")
    for k in keys[:-1]:
        d = d[int(k)] if isinstance(d, list) else d.setdefault(k, {})
    try:
        value = json.loads(value)
    except (TypeError, ValueError):
        pass
    last = keys[-1]
    if isinstance(d, list):
        d[int(last)] = value
    else:
        d[last] = value
