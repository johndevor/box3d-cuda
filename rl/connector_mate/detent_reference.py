"""CPU oracle of rl/connector_mate/detent.cu: the same arithmetic, elementwise in torch (float32), on any device.
detent_step(x, v, tilt, push, m, params[N,11], state[N,3], h) -> (J, state', clicked). See detent.cu for the model.
"""
from __future__ import annotations

import torch

P_XRAMP, P_XCLICK, P_DROPW, P_FPEAK, P_FRES, P_FWD, P_THJAM, P_XJAM, P_FJAM, P_FBEND, P_XPIN = range(11)
P_COUNT = 11


def detent_step(x, v, tilt, push, m, P, state, h):
    x, v, tilt, push, m = (t.float() for t in (x, v, tilt, push, m))
    latched, jammed, bent = state[:, 0].clone(), state[:, 1].clone(), state[:, 2].clone()
    x0, xc, dw = P[:, P_XRAMP], P[:, P_XCLICK], P[:, P_DROPW]
    unl = latched < 0.5
    ramp = unl & (x >= x0) & (x < xc)
    drop = unl & (x >= xc) & (x < xc + dw)
    R = torch.zeros_like(x)
    R = torch.where(ramp, P[:, P_FPEAK] * (x - x0) / torch.clamp(xc - x0, min=1e-6), R)
    R = torch.where(drop, P[:, P_FPEAK] * (1 - (x - xc) / torch.clamp(dw, min=1e-6)), R)
    R = R + torch.where(x > x0, P[:, P_FRES] * torch.tanh(v / 2e-3), torch.zeros_like(x))
    J = R * h
    vn = v - J / m
    close = unl & (x >= xc + 0.5 * dw)
    click = close.float()
    latched = torch.where(close, torch.ones_like(latched), latched)
    hold = (latched > 0.5) & (x < xc + 0.5 * dw) & (vn < 0)
    need, cap = -vn * m, P[:, P_FWD] * h
    ok = hold & (need <= cap)
    over = hold & (need > cap)
    J = torch.where(ok, J - need, torch.where(over, J - cap, J))
    vn = torch.where(ok, torch.zeros_like(vn), torch.where(over, vn + cap / m, vn))
    latched = torch.where(over, torch.zeros_like(latched), latched)
    th_j = P[:, P_THJAM]
    jammed = torch.where((x > P[:, P_XJAM]) & (tilt > th_j), torch.ones_like(jammed), jammed)
    free = (jammed > 0.5) & (((tilt < 0.6 * th_j) & (vn < 0)) | (x <= 0))
    jammed = torch.where(free, torch.zeros_like(jammed), jammed)
    lock = (jammed > 0.5) & (vn > 0)
    take = torch.minimum(vn * m, P[:, P_FJAM] * h)
    J = torch.where(lock, J + take, J)
    bent = torch.where((x > P[:, P_XPIN]) & (tilt > th_j) & (push > P[:, P_FBEND]), torch.ones_like(bent), bent)
    return J, torch.stack([latched, jammed, bent], -1), click
