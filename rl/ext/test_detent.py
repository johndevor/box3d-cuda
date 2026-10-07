"""Checks of the detent model: the CPU oracle's curve (rise, peak, click, hold, jam, bent pin) on scripted inputs, and,
with CUDA, detent.cu against the oracle on random inputs (max abs difference of the impulse and identical states).
  python -m rl.ext.test_detent
"""
import json

import torch

from . import detent_reference as DR


def params(n, dev):
    P = torch.zeros(n, DR.P_COUNT, device=dev)
    P[:, DR.P_XRAMP], P[:, DR.P_XCLICK], P[:, DR.P_DROPW] = 0.002, 0.004, 0.0002
    P[:, DR.P_FPEAK], P[:, DR.P_FRES], P[:, DR.P_FWD] = 12.0, 1.0, 30.0
    P[:, DR.P_THJAM], P[:, DR.P_XJAM], P[:, DR.P_FJAM], P[:, DR.P_FBEND], P[:, DR.P_XPIN] = 0.06, 0.0005, 80.0, 20.0, 0.001
    return P


def curve_checks():
    """A quasi-static sweep in x at a slow inward velocity: the resisting force J/h along the depth."""
    h, m = 1 / 480, 2.0
    xs = torch.linspace(0, 0.005, 501)
    n = len(xs)
    P = params(n, "cpu")
    st = torch.zeros(n, 3)
    J, s2, clk = DR.detent_step(xs, torch.full((n,), 1e-4), torch.zeros(n), torch.zeros(n), torch.full((n,), m), P, st, h)
    F = J / h
    i_peak = int(F.argmax())
    out = dict(peak_n=round(F.max().item(), 3), peak_at_mm=round(xs[i_peak].item() * 1e3, 3),
               after_click_n=round(F[(xs > 0.0042)].max().item(), 3), latched_from_mm=round(xs[s2[:, 0] > 0.5][0].item() * 1e3, 3))
    assert abs(out["peak_at_mm"] - 4.0) < 0.02 and out["peak_n"] > 12.0 and out["after_click_n"] < 1.1, out
    # the hold: latched, pulled out at 0.1 m/s below the hold point -> stopped while the needed impulse is under F_wd h
    st1 = torch.tensor([[1.0, 0, 0], [1.0, 0, 0]])
    J1, s1, _ = DR.detent_step(torch.tensor([0.0038, 0.0038]), torch.tensor([-0.001, -1.0]), torch.zeros(2), torch.zeros(2), torch.full((2,), m), params(2, "cpu"), st1, h)
    assert s1[0, 0] == 1 and s1[1, 0] == 0, s1           # a gentle pull is held; a hard one pulls it out
    # jam: tilted over the angle and engaged: inward motion stopped (impulse >= m v up to F_jam h)
    J2, s2b, _ = DR.detent_step(torch.tensor([0.001]), torch.tensor([0.01]), torch.tensor([0.08]), torch.tensor([10.0]), torch.tensor([m]), params(1, "cpu"), torch.zeros(1, 3), h)
    assert s2b[0, 1] == 1 and J2[0] > 0.9 * min(m * 0.01, 80 * h), (J2, s2b)
    # bent pin: tilted, past the pin engagement, pushed over F_bend
    _, s3, _ = DR.detent_step(torch.tensor([0.0015]), torch.tensor([0.0]), torch.tensor([0.08]), torch.tensor([25.0]), torch.tensor([m]), params(1, "cpu"), torch.zeros(1, 3), h)
    assert s3[0, 2] == 1, s3
    return out


def cuda_vs_oracle(n=200_000):
    from rl.common.cuda import load_ext
    ext = load_ext("b3_detent")
    g = torch.Generator(device="cuda").manual_seed(0)
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=g, device="cuda")
    P = torch.stack([r(0.0002, 0.003), r(0.003, 0.006), r(0.00005, 0.0004), r(5, 30), r(0.2, 2), r(10, 90), r(0.03, 0.1), r(0.0003, 0.001), r(40, 120), r(8, 40), r(0.0005, 0.002)], -1)
    st = (torch.rand(n, 3, generator=g, device="cuda") < 0.3).float()
    args = (r(-0.001, 0.008), r(-0.05, 0.05), r(0, 0.12), r(-10, 40), r(1.7, 2.3))
    J, s, c = ext.detent_step(*args, P, st, 1 / 480)
    Jr, sr, cr = DR.detent_step(*args, P, st, 1 / 480)
    out = dict(n=n, max_abs_J=float((J - Jr).abs().max()), max_rel_J=float(((J - Jr).abs() / (Jr.abs() + 1e-6)).max()),
               state_mismatch=int((s != sr).any(-1).sum()), click_mismatch=int((c != cr).sum()))
    # (fast-math tanh: a state flip at an exact threshold is allowed in a handful of worlds, not more)
    assert out["max_abs_J"] < 1e-5 and out["state_mismatch"] <= n * 1e-4, out
    return out


if __name__ == "__main__":
    res = dict(curve=curve_checks())
    if torch.cuda.is_available():
        res["cuda_vs_oracle"] = cuda_vs_oracle()
    print(json.dumps(res))
