"""Step 1 of the chain test: "place the housing". A housing block (a dynamic box, gravity) released by a gripper over a
nest (a pocket of box walls with a 45-degree lead-in, the same builder as the socket), with the release's pose error; it
falls, the lead-in guides it, it settles. The end states are snapshotted (box3d state rows) for step 2, which restores
them as the socket's true pose against the program's belief (the nest's layout pose): rl/connector_mate/env.py cfg
`chain_states`.

  python -m rl.connector_mate.place --n 32768 --out runs/chain/place_states.pt [--device cuda]

Step-1 success (what a look at the placed housing confirms): resting flat (tilt under 1 degree) in the nest (centre within
the clearance + 0.1 mm). Physics in mm, z up, the nest floor at z = 0.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

from rl.peg_insert.env import qmul, qrot, quat_of, uni
from .env import load_cuda_ext, pocket_statics, qyaw

N_STATIC = 13


def default_place_cfg(**over):
    c = dict(size_mm=(20.0, 14.0, 12.0), nest_clear_mm=(0.15, 0.6), nest_chamfer_mm=(0.8, 2.0), nest_depth_mm=4.0,
             release_h_mm=(0.5, 4.0), release_sigma_mm=(0.2, 0.8), yaw_sigma_deg=(0.5, 2.5), tilt_sigma_deg=(0.2, 1.2),
             friction=(0.2, 0.6), density=1400.0, settle_s=0.6, substep_hz=480, solver_iterations=8)
    c.update(over)
    return c


def run(n, device="cuda", seed=0, cfg=None):
    c = default_place_cfg(**(cfg or {}))
    dev = torch.device(device)
    g = torch.Generator(device=dev).manual_seed(seed)
    sx, sy, sz = c["size_mm"]
    clear = uni(g, n, c["nest_clear_mm"], dev)
    ch = uni(g, n, c["nest_chamfer_mm"], dev)
    D = torch.full((n,), c["nest_depth_mm"], device=dev)
    B = 1 + N_STATIC
    state = torch.zeros(n, B, 13, device=dev)
    state[:, :, 6] = 1
    half = torch.ones(n, B, 3, device=dev)
    inv_mass = torch.zeros(n, B, device=dev)
    inv_inertia = torch.zeros(n, B, 3, device=dev)
    pos, hf, quat = pocket_statics(sx / 2 + clear, sy / 2 + clear, D, ch, torch.zeros(n, device=dev))
    # (the pocket builder's floor sits at -D: lift the whole nest so its floor is z = 0)
    pos[..., 2] += D[:, None]
    state[:, 1:, 0:3], state[:, 1:, 3:7], half[:, 1:] = pos, quat, hf
    # the housing: released centred over the nest's layout pose with the release's error
    rs, ys, ts = uni(g, n, c["release_sigma_mm"], dev), torch.deg2rad(uni(g, n, c["yaw_sigma_deg"], dev)), torch.deg2rad(uni(g, n, c["tilt_sigma_deg"], dev))
    rn = lambda *s: torch.randn(*s, generator=g, device=dev)
    off = torch.stack([rn(n) * rs, rn(n) * rs], -1)
    yaw = rn(n) * ys
    tilt = torch.stack([rn(n) * ts, rn(n) * ts, torch.zeros(n, device=dev)], -1)
    q0 = qmul(quat_of(tilt), qyaw(yaw))
    h0 = uni(g, n, c["release_h_mm"], dev)
    state[:, 0, 0:2] = off
    state[:, 0, 2] = sz / 2 + h0
    state[:, 0, 3:7] = q0
    half[:, 0] = torch.tensor([sx / 2, sy / 2, sz / 2], device=dev)
    mass_kg = c["density"] * sx * sy * sz * 1e-9
    inv_mass[:, 0] = 1.0 / mass_kg
    I = mass_kg / 12 * torch.tensor([sy ** 2 + sz ** 2, sx ** 2 + sz ** 2, sx ** 2 + sy ** 2], device=dev)   # kg mm^2
    inv_inertia[:, 0] = 1.0 / I
    pairs = torch.tensor([[0, j] for j in range(1, B)], dtype=torch.int64, device=dev)
    cache_ids = torch.zeros(n, B - 1, 4, dtype=torch.int64, device=dev)
    cache_imp = torch.zeros(n, B - 1, 4, 3, device=dev)
    mu = float(uni(g, 1, c["friction"], dev))
    h = 1.0 / c["substep_hz"]
    steps = int(c["settle_s"] * c["substep_hz"])
    if dev.type == "cuda":
        ext = load_cuda_ext()
    else:
        from .env import _base
        mref, sref = _base.load_cpu_reference()
    t0 = time.time()
    for _ in range(steps):
        state[:, 0, 9] -= 9810.0 * h          # gravity (mm/s^2) along -z, applied as the controller's impulses are
        if dev.type == "cuda":
            out = ext.manifold_step(state, inv_mass, half, inv_inertia, pairs, cache_ids, cache_imp, h, 1, 0.0, 0.0, mu, 1e-3, 0.2, 0.0, c["solver_iterations"], 1e-4)
            state, cache_ids, cache_imp = out[0], out[3], out[4]
        else:
            cfgc = sref.SATConfig(dt=h, substeps=1, gravity_y=0.0, restitution=0.0, friction=mu, position_slop=1e-3, position_correction=0.2,
                                  angular_damping=0.0, solver_iterations=c["solver_iterations"], sat_epsilon=1e-4)
            out = mref.step_manifold_reference(state.tolist(), inv_mass.tolist(), half.tolist(), inv_inertia.tolist(), pairs.tolist(), cache_ids.tolist(), cache_imp.tolist(), cfgc)
            state, cache_ids, cache_imp = torch.tensor(out[0]), torch.tensor(out[3], dtype=torch.int64), torch.tensor(out[4])
    st = state[:, 0]
    up = qrot(st[:, 3:7], torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))
    tilt_end = torch.rad2deg(torch.atan2(up[:, :2].norm(dim=-1), up[:, 2]))
    xv = qrot(st[:, 3:7], torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3))
    yaw_end = torch.atan2(xv[:, 1], xv[:, 0])
    off_end = st[:, 0:2]
    resting = st[:, 7:10].norm(dim=-1) < 2.0          # mm/s
    inside = (off_end.abs() <= torch.stack([clear, clear], -1) + 0.1).all(-1)
    ok = (tilt_end < 1.0) & resting & inside & torch.isfinite(st).all(-1)
    return dict(state=st.cpu(), ok=ok.cpu(), off_mm=off_end.cpu(), yaw_rad=yaw_end.cpu(), tilt_deg=tilt_end.cpu(), clear_mm=clear.cpu(),
                z_mm=(st[:, 2] - sz / 2).cpu(), wall_s=time.time() - t0, cfg=c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32768)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/chain/place_states.pt")
    a = ap.parse_args()
    r = run(a.n, a.device, a.seed)
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(r, a.out)
    ok = r["ok"]
    q = lambda t: [round(float(x), 3) for x in torch.quantile(t.float(), torch.tensor([0.5, 0.9, 0.99]))]
    print(json.dumps(dict(event="place", n=a.n, success=round(ok.float().mean().item(), 4), wall_s=round(r["wall_s"], 1),
                          off_mm_p50_90_99=q(r["off_mm"][ok].norm(dim=-1)), yaw_deg_p50_90_99=q(torch.rad2deg(r["yaw_rad"][ok]).abs()),
                          tilt_deg_p50_90_99=q(r["tilt_deg"]), fail_tilted=int((r["tilt_deg"] >= 1.0).sum()))))


if __name__ == "__main__":
    main()
