"""Teacher -> student by DAgger: the student (policy observations only) acts, the teacher (observations + privileged
state) acting instead with probability beta (1 -> 0 over the first 40 % of the iterations), and the teacher labels
every visited state; the student regresses the teacher's action mean. The student's observation normaliser is the
teacher's on the shared columns, frozen (the exported graph carries it).
"""
from __future__ import annotations

import time

import torch

from .ppo import Stats


def dagger(env, teacher, student, iters, horizon, out, log, lr=1e-3, epochs=4, minibatches=4, beta_frac=0.4):
    dev, N, H = env.dev, env.n, horizon
    obs_n = env.obs_size
    student.obs_norm.load_state_dict({"mean": teacher.obs_norm.mean[:obs_n].clone(), "var": teacher.obs_norm.var[:obs_n].clone(),
                                      "count": teacher.obs_norm.count.clone()})
    student.obs_norm.frozen = True
    opt = torch.optim.Adam(student.actor.parameters(), lr=lr)
    obs, priv = env.observe()
    ep_ret, ep_len = torch.zeros(N, device=dev), torch.zeros(N, device=dev)
    stats = Stats(getattr(env, "code_names", None))
    t0, steps, loss = time.time(), 0, torch.tensor(0.0)
    for it in range(1, iters + 1):
        beta = max(0.0, 1.0 - it / (beta_frac * iters))
        Os, Ts = [], []
        for t in range(H):
            with torch.no_grad():
                ta = teacher.mean(torch.cat([obs, priv], -1))
                sa = student.mean(obs)
                use_t = torch.rand(N, 1, device=dev) < beta
                a = torch.where(use_t, ta, sa)
            Os.append(obs)
            Ts.append(ta)
            obs, priv, r, done, info = env.step(a)
            ep_ret += r
            ep_len += 1
            stats.add(done, info, ep_ret, ep_len)
            ep_ret = torch.where(done, torch.zeros_like(ep_ret), ep_ret)
            ep_len = torch.where(done, torch.zeros_like(ep_len), ep_len)
        steps += N * H
        O, T = torch.cat(Os), torch.cat(Ts)
        for _ in range(epochs):
            perm = torch.randperm(O.shape[0], device=dev)
            for i in range(minibatches):
                j = perm[i * O.shape[0] // minibatches:(i + 1) * O.shape[0] // minibatches]
                loss = (student.mean(O[j]) - T[j]).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        log(dict(phase="distill", it=it, steps=steps, beta=round(beta, 3), loss=round(loss.item(), 5), wall_s=round(time.time() - t0, 1), **stats.row()))
    torch.save(dict(model=student.state_dict(), sizes=student.sizes), out / "student_distilled.pt")
    return dict(steps=steps, wall_s=round(time.time() - t0, 1), final=stats.row())
