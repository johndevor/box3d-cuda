"""Open-loop gait-prior probe: World2's walk judge (skill.evaluate's loop) on a grid of parametric stepping patterns, one
group of worlds per pattern, no policy. Picks the GaitPrior settings the duck's residual policy starts from (a prior that
falls by itself teaches the policy to cancel it, not to walk).

    python3 -m rl.skills.duck_walk.prior_probe [--per 256] [--clock 1.5,2.5] [--top 12] [--out out/prior_probe.json]

Pattern (per leg, policy joint order hip_yaw, hip_roll, hip_pitch, knee, ankle): the swing leg flexes like the squat
(hip -/+0.5, knee 1, ankle -0.5) by `amp`, both hip rolls shift the weight by `roll` (same sign or mirrored), the swing
hip pitches forward and the stance hip back by `stride`; all on the gait clock's sine (left on its positive half).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os

import torch

from rl.common.cuda import require_device
from rl.skills.duck_walk.env import LEGS, RATE_HZ, DuckWalkEnv
from rl.skills.duck_walk.skill import CRITERIA, FAST, SPEC


def pattern(amp, roll, mirror, stride):
    L, R = [0.0] * LEGS, [0.0] * LEGS
    L[2], L[3], L[4] = -0.5 * amp, amp, -0.5 * amp
    R[7], R[8], R[9] = 0.5 * amp, amp, -0.5 * amp
    L[1], L[6] = roll, (-roll if mirror else roll)
    R[1], R[6] = -roll, (roll if mirror else -roll)
    SL, SR = [0.0] * LEGS, [0.0] * LEGS
    SL[2], SL[7] = -stride, -stride          # left swing: left thigh forward, right thigh back
    SR[2], SR[7] = stride, stride
    return L, R, SL, SR


def run(clock, grid, per, dev, seed=0, dr=False, settle=True):
    C = CRITERIA
    n = len(grid) * per
    env = {**FAST["env"], "clock_hz": clock}
    over = dict(push_p=0.0, cmd_zero_p=0.0, cmd_vx=(C["vx"], C["vx"]), episode_s=1e9, init_yaw=0.0, cmd_wz=0.0)
    e = DuckWalkEnv(n, device=dev, seed=seed, dr=SPEC.override(over) if dr else over, dr_on=dr, **env)
    e.cmd[:, 0] = C["vx"]
    P = [pattern(*g) for g in grid]
    rep = lambda k: torch.tensor([p[k] for p in P], dtype=torch.float32, device=e.dev).repeat_interleave(per, 0)
    Ll, Lr, Sl, Sr = rep(0), rep(1), rep(2), rep(3)
    for _ in range(int(C["settle_s"] * RATE_HZ) if settle else 0):
        e.step(torch.zeros(n, LEGS, device=e.dev), autoreset=False)
    e.prev_action.zero_(); e.t.zero_(); e.hist[:] = e._frame()[:, None]
    T0 = e.trunk()
    x0, z0 = T0["p"][:, 0].clone(), T0["p"][:, 2].clone()
    fell = torch.zeros(n, dtype=torch.bool, device=e.dev)
    on = torch.ones(n, 2, dtype=torch.bool, device=e.dev)
    lifts = torch.zeros(n, 2, device=e.dev)
    t_fall = torch.full((n,), C["window_s"], device=e.dev)
    obs, priv = e.observe()
    yaw = torch.zeros(n, device=e.dev)
    for k in range(int(C["window_s"] * RATE_HZ)):
        yaw += obs[:, 1] / RATE_HZ
        e.cmd[:, 2] = (-C["heading_gain"] * yaw).clamp(-C["heading_max"], C["heading_max"])
        s = obs[:, 41:42]
        sl, sr = s.clamp(min=0), (-s).clamp(min=0)
        a = sl * (Ll + Sl) + sr * (Lr + Sr)
        obs, priv, r, done, info = e.step(a, autoreset=False)
        T = e.trunk()
        nf = (T["h"] < 0.7 * e.trunk_y0) | (T["tilt"] > math.radians(40)) | ~torch.isfinite(T["h"])
        t_fall = torch.where(nf & ~fell, torch.full_like(t_fall, k / RATE_HZ), t_fall)
        fell |= nf
        h = e.out[:, 7:9]
        lift = on & (h > C["sole_clearance_m"])
        lifts += lift.float()
        on = torch.where(lift, torch.zeros_like(on), on) | (h < C["sole_down_m"])
    T = e.trunk()
    fwd, lat = T["p"][:, 0] - x0, T["p"][:, 2] - z0
    ok = ~fell & (fwd >= C["min_forward_m"]) & (lat.abs() <= C["max_lateral_m"]) & (lifts.min(-1).values >= C["min_liftoffs"])
    g = lambda t: t.float().view(len(grid), per)
    rows = []
    for i, (amp, roll, mirror, stride) in enumerate(grid):
        alive = ~g(fell)[i].bool()
        rows.append(dict(dr=dr, settle=settle, clock=clock, amp=amp, roll=roll, mirror=mirror, stride=stride, success=round(g(ok)[i].mean().item(), 3),
                         survive=round(alive.float().mean().item(), 3), t_fall=round(g(t_fall)[i].mean().item(), 2),
                         fwd=round(g(fwd)[i].median().item(), 3), lat=round(g(lat)[i].abs().median().item(), 3),
                         lifts=round(g(lifts.min(-1).values)[i].median().item(), 1)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=256)
    ap.add_argument("--clock", default="1.5,2.5")
    ap.add_argument("--amp", default="0.15,0.3,0.5")
    ap.add_argument("--roll", default="-0.3,-0.15,0,0.15,0.3")
    ap.add_argument("--stride", default="-0.3,-0.15,0,0.15,0.3")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default="out/prior_probe.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dr", action="store_true", help="the training randomization (the robot drawn per world)")
    ap.add_argument("--no-settle", action="store_true", help="step from the spawn, as training episodes start")
    a = ap.parse_args()
    dev = require_device(a.device)
    fl = lambda s: [float(x) for x in s.split(",")]
    grid = [g for g in itertools.product(fl(a.amp), fl(a.roll), (False, True), fl(a.stride)) if not (g[1] == 0 and g[2])]
    rows = []
    for c in fl(a.clock):
        rows += run(c, grid, a.per, dev, dr=a.dr, settle=not a.no_settle)
    key = lambda r: (r["success"], r["survive"] * min(1.0, max(r["fwd"], 0) / 0.5) * min(1.0, r["lifts"] / 4), r["survive"])
    rows.sort(key=key, reverse=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    for r in rows[: a.top]:
        print(json.dumps(r))
    zero = [r for r in rows if r["amp"] == min(fl(a.amp)) and r["roll"] == 0 and r["stride"] == 0]
    print(json.dumps(dict(event="probe_done", patterns=len(rows), smallest=zero[:2])))


if __name__ == "__main__":
    main()
