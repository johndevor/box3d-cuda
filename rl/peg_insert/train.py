"""PPO on PegInsertBatch, all on the GPU: a teacher with privileged state, a student on World2's 63 observations only
(DAgger from the teacher), then PPO fine-tuning of the student with the teacher's privileged critic.

  python -m rl.peg_insert.train --out runs/peg --envs 16384 [--teacher-max-min 60 --distill-iters 150 --finetune-max-min 40]

Writes metrics.jsonl (one line per iteration), teacher.pt, student_distilled.pt, student.pt (actor + its observation
normaliser: what export_onnx.py turns into the contract's ONNX), eval.json.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, deque
from pathlib import Path

import torch
import torch.nn as nn

from .env import CODE_NAMES, OBS_SIZE, PRIV_SIZE, PegInsertBatch


class RunningNorm(nn.Module):
    def __init__(self, n, clip=10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("var", torch.ones(n))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip
        self.frozen = False

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


def mlp(i, o, h=(256, 256)):
    layers, last = [], i
    for k in h:
        layers += [nn.Linear(last, k), nn.ELU()]
        last = k
    layers.append(nn.Linear(last, o))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, n_in, n_act=9, log_std=-0.5):
        super().__init__()
        self.norm = RunningNorm(n_in)
        self.net = mlp(n_in, n_act)
        self.log_std = nn.Parameter(torch.full((n_act,), log_std))
        with torch.no_grad():  # start pushing in gently and never terminating (a[8] > 0 ends the skill)
            self.net[-1].bias[2] = 0.3
            self.net[-1].bias[8] = -3.0
            self.log_std[8] = -2.0

    def mean(self, x):
        return torch.tanh(self.net(self.norm(x)))

    def dist(self, x):
        return torch.distributions.Normal(self.mean(x), self.log_std.exp().expand(x.shape[0], -1))


class Critic(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.norm = RunningNorm(n_in)
        self.net = mlp(n_in, 1)

    def forward(self, x):
        return self.net(self.norm(x)).squeeze(-1)


class Stats:
    def __init__(self):
        self.succ, self.codes, self.ret = deque(maxlen=4000), deque(maxlen=4000), deque(maxlen=4000)

    def add(self, done, info, ep_ret):
        idx = done.nonzero().squeeze(-1)
        if len(idx) == 0:
            return
        s, c, r = info["success"][idx].tolist(), info["code"][idx].tolist(), ep_ret[idx].tolist()
        self.succ.extend(s)
        self.codes.extend(CODE_NAMES.get(int(x), str(x)) for x in c)
        self.ret.extend(r)

    def row(self):
        n = max(1, len(self.succ))
        return dict(success=round(sum(self.succ) / n, 4), episodes=len(self.succ), ret=round(sum(self.ret) / max(1, len(self.ret)), 3),
                    codes=dict(Counter(self.codes).most_common(6)))


def rollout(env, act_fn, horizon, obs, priv, ep_ret, stats, store):
    """act_fn(obs, priv) -> (action, extra dict). store(t, obs, priv, action, extra, reward, done)."""
    for t in range(horizon):
        with torch.no_grad():
            a, extra = act_fn(obs, priv)
        nobs, r, done, info = env.step(a)
        ep_ret += r
        stats.add(done, info, ep_ret)
        store(t, obs, priv, a, extra, r, done)
        if done.any():
            env.reset(done)
            fresh = env.obs(update=False)
            nobs = torch.where(done[:, None], fresh, nobs)
            ep_ret = torch.where(done, torch.zeros_like(ep_ret), ep_ret)
        obs, priv = nobs, env.priv()
    return obs, priv, ep_ret


def ppo_phase(env, actor, critic, actor_in, args, out, phase, max_min, log, min_iters=30):
    dev = env.dev
    N, H = env.n, args.horizon
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    obs, priv = env.obs(update=False), env.priv()
    ep_ret = torch.zeros(N, device=dev)
    stats = Stats()
    buf = {k: torch.zeros(H, N, s, device=dev) for k, s in (("obs", OBS_SIZE), ("priv", PRIV_SIZE), ("act", 9))}
    buf.update({k: torch.zeros(H, N, device=dev) for k in ("logp", "val", "rew", "done")})
    t0, steps, it = time.time(), 0, 0
    best, hist = -1.0, []
    while True:
        env.resample_friction()

        def store(t, o, p, a, ex, r, done):
            buf["obs"][t], buf["priv"][t], buf["act"][t] = o, p, a
            buf["logp"][t], buf["val"][t], buf["rew"][t], buf["done"][t] = ex["logp"], ex["val"], r, done.float()
        # (the sampled, unclamped action is what logp refers to; the env clamps)
        def act_fn_raw(o, p):
            x = actor_in(o, p)
            d = actor.dist(x)
            a = d.sample()
            return a, dict(logp=d.log_prob(a).sum(-1), val=critic(torch.cat([o, p], -1)))
        obs, priv, ep_ret = rollout(env, act_fn_raw, H, obs, priv, ep_ret, stats, store)
        steps += N * H
        with torch.no_grad():
            last_v = critic(torch.cat([obs, priv], -1))
            adv = torch.zeros(H, N, device=dev)
            gae = torch.zeros(N, device=dev)
            for t in reversed(range(H)):
                nv = last_v if t == H - 1 else buf["val"][t + 1]
                nd = 1 - buf["done"][t]
                delta = buf["rew"][t] + args.gamma * nv * nd - buf["val"][t]
                gae = delta + args.gamma * args.lam * nd * gae
                adv[t] = gae
            ret = adv + buf["val"]
        flat = lambda x: x.reshape(H * N, *x.shape[2:])
        O, Pv, A, LP, ADV, RET = flat(buf["obs"]), flat(buf["priv"]), flat(buf["act"]), flat(buf["logp"]), flat(adv), flat(ret)
        actor.norm.update(actor_in(O, Pv))
        critic.norm.update(torch.cat([O, Pv], -1))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        M = H * N
        mb = M // args.minibatches
        kl_sum, n_mb = 0.0, 0
        for ep in range(args.epochs):
            perm = torch.randperm(M, device=dev)
            for i in range(args.minibatches):
                j = perm[i * mb:(i + 1) * mb]
                d = actor.dist(actor_in(O[j], Pv[j]))
                lp = d.log_prob(A[j]).sum(-1)
                ratio = torch.exp(lp - LP[j])
                pl = -torch.min(ratio * ADV[j], ratio.clamp(1 - args.clip, 1 + args.clip) * ADV[j]).mean()
                vl = (critic(torch.cat([O[j], Pv[j]], -1)) - RET[j]).pow(2).mean()
                loss = pl + 0.5 * vl - args.ent * d.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                with torch.no_grad():
                    kl_sum += (LP[j] - lp).mean().item()
                    n_mb += 1
            if kl_sum / n_mb > 0.03:
                break
        with torch.no_grad():
            actor.log_std.clamp_(-4.0, 0.0)
        it += 1
        row = dict(phase=phase, it=it, steps=steps, wall_s=round(time.time() - t0, 1), sps=round(steps / (time.time() - t0)),
                   kl=round(kl_sum / max(1, n_mb), 4), std=round(actor.log_std.exp().mean().item(), 3), **stats.row())
        log(row)
        hist.append(row["success"])
        if row["episodes"] >= 2000 and row["success"] > best:
            best = row["success"]
            torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), row=row), out / f"{phase}_best.pt")
        if it % 10 == 0:
            torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), row=row), out / f"{phase}.pt")
        W = args.plateau_iters // 2   # plateau: the last W iterations' mean success no better than the W before (+0.003)
        plateau = it >= min_iters and len(hist) >= 2 * W and sum(hist[-W:]) / W - sum(hist[-2 * W:-W]) / W < 0.003
        if plateau or time.time() - t0 > max_min * 60 or steps >= args.max_steps:
            torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), row=row), out / f"{phase}.pt")
            return dict(steps=steps, wall_s=time.time() - t0, final=row, best=best, stop="plateau" if plateau else "budget")


def distill(env, teacher, student, args, out, log):
    """DAgger: the student acts (teacher with probability beta, decaying), the teacher labels every visited state."""
    dev, N, H = env.dev, env.n, args.horizon
    opt = torch.optim.Adam(student.net.parameters(), lr=1e-3)
    obs, priv = env.obs(update=False), env.priv()
    ep_ret = torch.zeros(N, device=dev)
    stats = Stats()
    t0, steps = time.time(), 0
    student.norm.load_state_dict(teacher.norm_student_view)
    student.norm.frozen = True
    for it in range(1, args.distill_iters + 1):
        beta = max(0.0, 1.0 - it / (0.4 * args.distill_iters))
        Os, Ts = [], []

        def act_fn(o, p):
            ta = teacher.mean(torch.cat([o, p], -1))
            sa = student.mean(o)
            use_t = (torch.rand(o.shape[0], 1, device=dev) < beta)
            return torch.where(use_t, ta, sa), dict(ta=ta)

        def store(t, o, p, a, ex, r, done):
            Os.append(o)
            Ts.append(ex["ta"])
        obs, priv, ep_ret = rollout(env, act_fn, H, obs, priv, ep_ret, stats, store)
        steps += N * H
        O, T = torch.cat(Os), torch.cat(Ts)
        for _ in range(4):
            perm = torch.randperm(O.shape[0], device=dev)
            for i in range(4):
                j = perm[i * O.shape[0] // 4:(i + 1) * O.shape[0] // 4]
                loss = (student.mean(O[j]) - T[j]).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        log(dict(phase="distill", it=it, steps=steps, beta=round(beta, 3), loss=round(loss.item(), 5), wall_s=round(time.time() - t0, 1), **stats.row()))
    torch.save(dict(actor=student.state_dict()), out / "student_distilled.pt")
    return dict(steps=steps, wall_s=time.time() - t0, final=stats.row())


def evaluate(env_cfg, actor, actor_in, n, seed, device, max_ticks=200):
    env = PegInsertBatch(n, device=device, seed=seed, cfg=env_cfg)
    obs, priv = env.obs(update=False), env.priv()
    finished = torch.zeros(n, dtype=torch.bool, device=env.dev)
    succ = torch.zeros(n, dtype=torch.bool, device=env.dev)
    codes = torch.zeros(n, dtype=torch.long, device=env.dev)
    clear = env.p["clear"].clone()
    for _ in range(max_ticks):
        with torch.no_grad():
            a = actor.mean(actor_in(obs, priv))
        obs, r, done, info = env.step(a)
        new = done & ~finished
        succ |= new & info["success"]
        codes = torch.where(new, info["code"], codes)
        finished |= done
        if finished.all():
            break
        priv = env.priv()
    bins = {}
    for lo, hi in ((0, 50e-6), (50e-6, 200e-6), (200e-6, 1.1e-3)):
        m = (clear >= lo) & (clear < hi)
        bins[f"{int(lo*1e6)}-{int(hi*1e6)}um"] = round(succ[m].float().mean().item(), 4) if m.any() else None
    return dict(n=n, success=round(succ.float().mean().item(), 4), by_clearance=bins,
                codes=dict(Counter(CODE_NAMES.get(int(c), "unfinished") for c in codes.tolist()).most_common()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/peg")
    ap.add_argument("--envs", type=int, default=16384)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--plateau-iters", type=int, default=40)
    ap.add_argument("--max-steps", type=float, default=3e9)
    ap.add_argument("--teacher-max-min", type=float, default=60)
    ap.add_argument("--distill-iters", type=int, default=150)
    ap.add_argument("--finetune-max-min", type=float, default=40)
    ap.add_argument("--eval-n", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cfg", default="{}", help="JSON overrides of env.default_cfg")
    ap.add_argument("--resume-teacher", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    cfg = json.loads(a.cfg)
    mf = open(out / "metrics.jsonl", "a")

    def log(row):
        print(json.dumps(row), flush=True)
        mf.write(json.dumps(row) + "\n")
        mf.flush()

    t_all = time.time()
    env = PegInsertBatch(a.envs, device=a.device, seed=a.seed, cfg=cfg)
    log(dict(event="env", envs=a.envs, cfg=env.cfg, device=str(env.dev), cuda=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    summary = dict(args=vars(a), cfg=env.cfg)
    # ---- teacher: privileged actor and critic
    teacher = Actor(OBS_SIZE + PRIV_SIZE).to(env.dev)
    critic = Critic(OBS_SIZE + PRIV_SIZE).to(env.dev)
    t_in = lambda o, p: torch.cat([o, p], -1)
    if a.resume_teacher:
        ck = torch.load(a.resume_teacher, map_location=env.dev)
        teacher.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
    else:
        summary["teacher"] = ppo_phase(env, teacher, critic, t_in, a, out, "teacher", a.teacher_max_min, log)
        ck = torch.load(out / "teacher_best.pt", map_location=env.dev)
        teacher.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
    summary["teacher_eval"] = evaluate(cfg, teacher, t_in, a.eval_n, 10_000, a.device)
    log(dict(event="teacher_eval", **summary["teacher_eval"]))
    # ---- student by DAgger, its observation normaliser = the teacher's on the 63 shared fields
    teacher.norm_student_view = {"mean": teacher.norm.mean[:OBS_SIZE].clone(), "var": teacher.norm.var[:OBS_SIZE].clone(), "count": teacher.norm.count.clone()}
    student = Actor(OBS_SIZE, log_std=-1.5).to(env.dev)
    s_in = lambda o, p: o
    summary["distill"] = distill(env, teacher, student, a, out, log)
    summary["distill_eval"] = evaluate(cfg, student, s_in, a.eval_n, 10_000, a.device)
    log(dict(event="distill_eval", **summary["distill_eval"]))
    # ---- PPO fine-tune of the student (its normaliser frozen: the exported graph carries it), privileged critic
    with torch.no_grad():
        student.log_std.fill_(-1.5); student.log_std[8] = -2.5
    summary["finetune"] = ppo_phase(env, student, critic, s_in, a, out, "student", a.finetune_max_min, log, min_iters=20)
    ck = torch.load(out / "student_best.pt", map_location=env.dev)
    student.load_state_dict(ck["actor"])
    for name, c in (("eval_nominal", cfg), ("eval_hard_1.5x", dict(cfg, pose_scale=1.5))):
        summary[name] = evaluate(c, student, s_in, a.eval_n, 20_000, a.device)
        log(dict(event=name, **summary[name]))
    summary["distilled_eval_hard_1.5x"] = None
    torch.save(dict(actor=student.state_dict(), obs_fields="world2 insert v2 layout", obs_size=OBS_SIZE), out / "student.pt")
    summary["wall_s_total"] = time.time() - t_all
    (out / "eval.json").write_text(json.dumps(summary, indent=1, default=str))
    log(dict(event="done", wall_s=summary["wall_s_total"]))


if __name__ == "__main__":
    main()
