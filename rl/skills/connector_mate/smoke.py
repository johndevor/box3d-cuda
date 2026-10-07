"""CPU smoke of the connector env with a scripted policy (centre on the believed target, push in at a fixed stiffness).
  python -m rl.skills.connector_mate.smoke [--n 16 --device cpu]
"""
import argparse
import time
from collections import Counter

import torch

from rl.skills.peg_insert.env import OBS_SIZE
from .env import CODE_NAMES, PRIV_SIZE, ConnectorMateBatch


def scripted(o, push=0.6):
    a = torch.zeros(o.shape[0], 9, device=o.device)
    a[:, 0:2] = (o[:, 0:2] / 1e-3).clamp(-1, 1)
    a[:, 3:6] = (o[:, 3:6] / 0.01).clamp(-1, 1)
    a[:, 2] = push
    a[:, 7] = 0.5
    a[:, 8] = -1
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--push", type=float, default=0.6)
    a = ap.parse_args()
    env = ConnectorMateBatch(a.n, device=a.device, seed=a.seed)
    o, p = env.obs(update=False), env.priv()
    assert o.shape[1] == OBS_SIZE and p.shape[1] == PRIV_SIZE, (o.shape, p.shape)
    n = a.n
    fin = torch.zeros(n, dtype=torch.bool, device=env.dev)
    succ = fin.clone()
    codes = torch.zeros(n, dtype=torch.long, device=env.dev)
    t0 = time.time()
    for it in range(200):
        o, _, r, d, info = env.step(scripted(o, a.push), autoreset=False)
        new = d & ~fin
        codes = torch.where(new, info["code"], codes)
        succ |= new & info["success"]
        fin |= d
        if it % 15 == 0:
            print(it, round(time.time() - t0, 1), "x_mm", [round(v, 2) for v in (-env._tip()[:4, 2] * 1000).tolist()],
                  "state", env.v["det_state"][:4].tolist(), "push", [round(v, 1) for v in (-env.v["reading"][:4, 2]).tolist()], flush=True)
        if fin.all():
            break
    print("success", succ.float().mean().item(), dict(Counter(CODE_NAMES.get(int(c), "unfinished") for c in codes.tolist())),
          "phys", env.p["phys"].tolist()[:8])
    diag(env, codes)


def diag(env, codes):
    p, x = env.p, -env._tip()[:, 2]
    r = lambda t, s=1e3, k=2: round(t.item() * s, k)
    for i in range(env.n):
        print(CODE_NAMES.get(int(codes[i]), "?"), dict(x=r(x[i]), depth=r(p["depth"][i]), floor=r(p["floor"][i]), xclick=r(p["xclick"][i]),
              state=env.v["det_state"][i].tolist(), tilt=r(env._tilt()[i], 57.3), thjam=r(p["thjam"][i], 57.3), clear=r(p["clear"][i], 1e3, 3),
              lat=r(env._tip()[i, :2].norm()), fpk=r(p["fpk"][i], 1, 1), maxF=r(p["maxF"][i], 1, 1), phys=int(p["phys"][i].item()),
              mu=r(env.group_mu[env.group_of[i]], 1)))


if __name__ == "__main__":
    main()
