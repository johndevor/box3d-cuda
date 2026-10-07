"""A GIF of a few batched training worlds (side view, x-z section through the hole's axis), CPU reference physics.
  python -m rl.skills.peg_insert.viz [--ckpt runs/peg_insert/policy.pt] --out peg.gif [--n 6 --ticks 90]
Without --ckpt a scripted move-to-the-believed-target policy drives the pegs.
"""
import argparse
import itertools
import math

import torch
from PIL import Image, ImageDraw

from .env import CODE_NAMES, OBS_SIZE, PegInsertBatch, qrot


def corners(pos, q, half):
    out = []
    for s in itertools.product((-1, 1), repeat=3):
        out.append(pos + qrot(q, half * torch.tensor(s, dtype=torch.float32)))
    return torch.stack(out)


def hull(pts):
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="peg.gif")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--ticks", type=int, default=90)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    env = PegInsertBatch(a.n, device="cpu", seed=a.seed, cfg=dict(friction_groups=1))
    actor = None
    if a.ckpt:
        from rl.common.models import from_actor_state
        actor = from_actor_state(torch.load(a.ckpt, map_location="cpu")["actor"])
    o = env.obs(update=False)
    cols, cell = 3, 260
    rows = math.ceil(a.n / cols)
    frames, status = [], ["" for _ in range(a.n)]
    for t in range(a.ticks):
        img = Image.new("RGB", (cols * cell, rows * cell), (245, 247, 250))
        for w in range(a.n):
            sub = Image.new("RGB", (cell, cell), (245, 247, 250))
            dr = ImageDraw.Draw(sub)
            D = float(env.p["depth"][w]) * 1000
            W = float(env.half[w, 12][0])
            sc = (cell - 20) / max(2 * W, D + 5.0 + 14.0)
            cx, cz = cell / 2, 34 + 12.0 * sc
            order = [b for b in range(1, env.state.shape[1]) if b not in (4, 5, 6, 10, 11, 12)] + [0]
            for b in order:
                st = env.state[w, b]
                cs = corners(st[0:3], st[3:7], env.half[w, b])
                pts = [(cx + float(c[0]) * sc, cz - float(c[2]) * sc) for c in cs]
                hp = hull([(round(x, 2), round(y, 2)) for x, y in pts])
                dr.polygon(hp, fill=(70, 130, 220) if b == 0 else (150, 158, 170), outline=(40, 40, 50))
            dr.rectangle([0, 0, cell, 30], fill=(245, 247, 250))
            info = f"world {w}: d={env.p['d'][w]*1e3:.1f} mm, clearance {env.p['clear'][w]*1e6:.0f} um, mu {float(env.group_mu[env.group_of[w]]):.2f}"
            dr.text((6, 4), info, fill=(20, 20, 30))
            dr.text((6, 16), status[w] or f"t={float(env.v['t'][w]):.2f} s", fill=(0, 110, 0) if "depth" in status[w] or "stop" in status[w] else (160, 0, 0))
            dr.rectangle([0, 0, cell - 1, cell - 1], outline=(200, 205, 212))
            img.paste(sub, ((w % cols) * cell, (w // cols) * cell))
        frames.append(img)
        with torch.no_grad():
            if actor is not None:
                act = actor.mean(o)
            else:
                act = torch.zeros(a.n, 9)
                act[:, 0:3] = (o[:, 0:3] / 0.001).clamp(-1, 1) * 0.5
                act[:, 2] = 0.6
                act[:, 8] = -1
        o, _, r, done, info = env.step(act, autoreset=False)
        for w in done.nonzero().squeeze(-1).tolist():
            status[w] = f"ended: {CODE_NAMES.get(int(info['code'][w]))} t={float(env.v['t'][w]):.2f}s"
        if done.any():
            env.reset(done)
            o = torch.where(done[:, None], env.obs(update=False), o)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=66, loop=0)
    print(a.out, len(frames))


if __name__ == "__main__":
    main()
