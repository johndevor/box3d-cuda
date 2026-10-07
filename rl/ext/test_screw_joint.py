"""Screw joint: oracle unit checks (CPU) and the CUDA kernel against the oracle (needs CUDA).
  python -m pytest rl/ext/test_screw_joint.py -q
"""
from __future__ import annotations

import math
import random

try:
    import pytest
    skip_no_cuda = pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs CUDA")
except ImportError:      # (run directly: python -m rl.ext.test_screw_joint)
    skip_no_cuda = lambda f: f

from . import screw_reference as SR

TWO_PI = 2 * math.pi
H = 1.0 / 480


def rot_about(axis_xy, ang):
    """Quaternion for a rotation by ang about the horizontal axis (x, y, 0)."""
    n = math.hypot(*axis_xy) or 1.0
    s = math.sin(ang / 2)
    return [axis_xy[0] / n * s, axis_xy[1] / n * s, 0.0, math.cos(ang / 2)]


def params(pitch=0.5e-3, d=3e-3, tset=0.6, travel=6e-3, modeb=0, phil=0, camc=1e3, tstrip=1e3, kx=0.24, tilt=0.0, L=10e-3, thx=math.radians(2.5)):
    P = [0.0] * SR.NP
    P[SR.P_PITCH], P[SR.P_D], P[SR.P_RCAP], P[SR.P_THX] = pitch, d, 0.15 * d, thx
    P[SR.P_TRUN], P[SR.P_KX], P[SR.P_TSET], P[SR.P_KJ] = 0.02, kx, tset, 2e7
    P[SR.P_K], P[SR.P_TSTRIP], P[SR.P_PHIL], P[SR.P_CAMC] = 0.2, tstrip, phil, camc
    P[SR.P_HEXCAP], P[SR.P_TMOT], P[SR.P_TRAVEL], P[SR.P_MODEB] = 3.8, 1.9, travel, modeb
    P[SR.P_MOUTH:SR.P_MOUTH + 3] = [0.0, 0.0, -3.0]
    q = rot_about((1.0, 0.0), tilt)
    P[SR.P_UINS:SR.P_UINS + 3] = SR._rot(q, [0.0, 0.0, -1.0])
    P[SR.P_DROP], P[SR.P_L] = 0.4 * pitch, L
    return P


def body_at_tip(tip_mm, q, L):
    ax = SR._rot(q, [0.0, 0.0, -1.0])
    return [tip_mm[k] - ax[k] * L * 500 for k in range(3)] + list(q) + [0.0] * 6


def fresh_joint(phase=0.5):
    J = [0.0] * SR.NJ
    J[SR.J_ARMED], J[SR.J_THSEAT], J[SR.J_PHASE] = 1.0, -1.0, phase
    return J


def tip_of(b, L):
    ax = SR._rot(b[3:7], [0.0, 0.0, -1.0])
    return [b[k] + ax[k] * L * 500 for k in range(3)]


def drive(b, P, J, w, fax, steps):
    for _ in range(steps):
        SR.step_world(b, P, J, [w, fax], H)


def test_mode_b_catches_on_capture_and_advances_one_pitch_per_turn():
    P = params(modeb=1)
    b = body_at_tip([0.1, 0.0, -3.0], [0, 0, 0, 1], P[SR.P_L])
    J = fresh_joint()
    SR.step_world(b, P, J, [0.0, 5.0], H)
    assert J[SR.J_MODE] == SR.ENGAGED and J[SR.J_CROSSED] == 0.0
    z0 = tip_of(b, P[SR.P_L])[2]
    w = TWO_PI * 2          # 2 rev/s
    drive(b, P, J, w, 5.0, 240)    # 0.5 s = 1 turn
    z1 = tip_of(b, P[SR.P_L])[2]
    assert abs((z0 - z1) - 0.5) < 1e-3     # mm per turn = pitch
    assert abs(tip_of(b, P[SR.P_L])[0]) < 1e-6   # centred on the hole's axis


def test_mode_a_needs_forward_spin_and_a_back_turn_clicks():
    P = params(modeb=0)
    b = body_at_tip([0.0, 0.0, -3.0], [0, 0, 0, 1], P[SR.P_L])
    J = fresh_joint(phase=1.0)
    drive(b, P, J, 0.0, 5.0, 50)
    assert J[SR.J_MODE] == SR.FREE
    drive(b, P, J, -TWO_PI, 5.0, 300)          # back-turn past the thread start: a click
    assert J[SR.J_CLICKED] == 1.0 and J[SR.J_MODE] == SR.FREE
    SR.step_world(b, P, J, [TWO_PI, 5.0], H)   # the next forward turn catches at once
    assert J[SR.J_MODE] == SR.ENGAGED


def test_cross_thread_binds_and_the_clutch_fails_before_the_seat():
    P = params(modeb=1, tilt=math.radians(4.0))
    b = body_at_tip([0.0, 0.0, -3.0], [0, 0, 0, 1], P[SR.P_L])
    J = fresh_joint()
    SR.step_world(b, P, J, [0.0, 5.0], H)
    assert J[SR.J_CROSSED] == 1.0
    drive(b, P, J, TWO_PI * 4, 5.0, 2000)
    assert J[SR.J_MODE] == SR.FAILED and J[SR.J_FAIL] == SR.FAIL_CLUTCH
    turns = J[SR.J_THETA] / TWO_PI
    assert abs(turns - (0.6 - 0.02 - 0.05 * 3e-3 * 5) / 0.24) < 0.05


def test_back_out_releases_and_must_leave_the_capture():
    P = params(modeb=1, tilt=math.radians(4.0))
    b = body_at_tip([0.0, 0.0, -3.0], [0, 0, 0, 1], P[SR.P_L])
    J = fresh_joint()
    SR.step_world(b, P, J, [0.0, 5.0], H)
    drive(b, P, J, TWO_PI * 4, 5.0, 100)
    assert J[SR.J_MODE] == SR.ENGAGED
    drive(b, P, J, -TWO_PI * 4, 5.0, 200)
    assert J[SR.J_MODE] == SR.FREE and J[SR.J_ARMED] == 0.0
    SR.step_world(b, P, J, [0.0, 5.0], H)
    assert J[SR.J_MODE] == SR.FREE         # still in the capture: no catch until it leaves


def test_seat_then_clutch_is_success_and_strip_and_cam_out_fail():
    P = params(modeb=1, travel=1e-3)
    b = body_at_tip([0.0, 0.0, -3.0], [0, 0, 0, 1], P[SR.P_L])
    J = fresh_joint()
    drive(b, P, J, TWO_PI * 4, 5.0, 400)
    assert J[SR.J_SUCCESS] == 1.0 and abs(J[SR.J_TORQUE] - 0.6) < 1e-9
    P = params(modeb=1, travel=1e-3, tstrip=0.4)
    b = body_at_tip([0.0, 0.0, -3.0], [0, 0, 0, 1], P[SR.P_L])
    J = fresh_joint()
    drive(b, P, J, TWO_PI * 4, 5.0, 400)
    assert J[SR.J_STRIPPED] == 1.0 and J[SR.J_SUCCESS] == 0.0 and J[SR.J_TORQUE] < 0.2
    P = params(modeb=1, travel=1e-3, phil=1, camc=0.0133)
    b = body_at_tip([0.0, 0.0, -3.0], [0, 0, 0, 1], P[SR.P_L])
    J = fresh_joint()
    drive(b, P, J, TWO_PI * 4, 15.0, 400)      # 15 N holds 0.2 N m: cams out before 0.6
    assert J[SR.J_MODE] == SR.FAILED and J[SR.J_FAIL] == SR.FAIL_CAM


def random_case(rng):
    pitch, d = rng.choice([(0.4e-3, 2e-3), (0.45e-3, 2.5e-3), (0.5e-3, 3e-3), (0.7e-3, 4e-3), (0.8e-3, 5e-3)])
    tset = rng.uniform(0.2, 3.0)
    P = params(pitch=pitch, d=d, tset=tset, travel=rng.uniform(2e-3, 9e-3), modeb=float(rng.random() < 0.4), phil=float(rng.random() < 0.5),
               camc=rng.uniform(0.005, 0.03), tstrip=rng.choice([1e3, tset * rng.uniform(0.7, 1.3)]), kx=rng.uniform(0.25, 0.8) * tset,
               tilt=math.radians(rng.uniform(0, 4)), L=rng.uniform(5e-3, 15e-3), thx=math.radians(rng.uniform(1.8, 3.5)))
    P[SR.P_RCAP] = rng.uniform(0.1, 0.2) * d
    P[SR.P_TMOT] = 1.875 * tset * 1.2
    tilt = math.radians(rng.uniform(0, 5))
    q = rot_about((rng.uniform(-1, 1), rng.uniform(-1, 1)), tilt)
    tip = [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), -3.0 + rng.uniform(-0.2, 0.3)]
    b = body_at_tip(tip, q, P[SR.P_L])
    b[7:13] = [rng.uniform(-5, 5) for _ in range(3)] + [rng.uniform(-0.2, 0.2) for _ in range(3)]
    J = fresh_joint(rng.uniform(0, TWO_PI))
    J[SR.J_ARMED] = float(rng.random() < 0.8)
    J[SR.J_CLICKED] = float(rng.random() < 0.2)
    r = rng.random()
    if r < 0.4:
        J[SR.J_MODE] = rng.choice([SR.ENGAGED, SR.SEATED])
        J[SR.J_U:SR.J_U + 3] = P[SR.P_UINS:SR.P_UINS + 3]
        J[SR.J_ANCHOR:SR.J_ANCHOR + 3] = tip
        J[SR.J_CROSSED] = float(rng.random() < 0.3)
        th_seat = P[SR.P_TRAVEL] * TWO_PI / pitch
        if J[SR.J_MODE] == SR.SEATED:
            J[SR.J_THSEAT], J[SR.J_THETA] = th_seat, th_seat + rng.uniform(0, 0.2)
        else:
            J[SR.J_THETA] = rng.uniform(0.01, th_seat)
        J[SR.J_STRIPPED] = float(rng.random() < 0.05)
        J[SR.J_PEAK] = rng.uniform(0, tset)
    return b, P, J


@skip_no_cuda
def test_cuda_matches_oracle():
    import torch
    from rl.common.cuda import load_ext
    ext = load_ext("b3_screw")
    rng = random.Random(7)
    W, Bn, steps = 4096, 2, 200
    cases = [random_case(rng) for _ in range(W)]
    state = torch.zeros(W, Bn, 13)
    state[:, :, 6] = 1
    state[:, 0] = torch.tensor([c[0] for c in cases])
    P = torch.tensor([c[1] for c in cases])
    J = torch.tensor([c[2] for c in cases])
    st, Pc, Jc = state.cuda(), P.cuda(), J.cuda()
    mismatched, compared, worst = 0, 0, dict(pos=0.0, quat=0.0, vel=0.0, theta=0.0, torque=0.0)
    for s in range(steps):
        C = torch.tensor([[rng.choice([-1.0, 0.0, 1.0, 1.0]) * rng.uniform(0, 42), rng.uniform(0, 40)] for _ in range(W)])
        out_s, out_j = ext.screw_joint_step(st, Pc, Jc, C.cuda(), H)
        ref_s, ref_j = SR.step_reference(st.cpu().tolist(), P.tolist(), Jc.cpu().tolist(), C.tolist(), H)
        cs, cj = out_s.cpu(), out_j.cpu()
        rs, rj = torch.tensor(ref_s), torch.tensor(ref_j)
        branch = (cj[:, SR.J_MODE] == rj[:, SR.J_MODE]) & (cj[:, SR.J_FAIL] == rj[:, SR.J_FAIL]) & (cj[:, SR.J_SUCCESS] == rj[:, SR.J_SUCCESS]) \
            & (cj[:, SR.J_STRIPPED] == rj[:, SR.J_STRIPPED]) & (cj[:, SR.J_CROSSED] == rj[:, SR.J_CROSSED])
        mismatched += int((~branch).sum())
        compared += W
        m = branch
        worst["pos"] = max(worst["pos"], float((cs[m, 0, 0:3] - rs[m, 0, 0:3]).abs().max()))
        worst["quat"] = max(worst["quat"], float((cs[m, 0, 3:7] - rs[m, 0, 3:7]).abs().max()))
        worst["vel"] = max(worst["vel"], float((cs[m, 0, 7:13] - rs[m, 0, 7:13]).abs().max()))
        worst["theta"] = max(worst["theta"], float((cj[m, SR.J_THETA] - rj[m, SR.J_THETA]).abs().max()))
        worst["torque"] = max(worst["torque"], float((cj[m, SR.J_TORQUE] - rj[m, SR.J_TORQUE]).abs().max()))
        st, Jc = out_s, out_j
        # keep the population alive: worlds that ended start a new random case
        ended = (cj[:, SR.J_MODE] == SR.FAILED) | (cj[:, SR.J_SUCCESS] > 0.5)
        if s % 20 == 19 and ended.any():
            for i in ended.nonzero().squeeze(-1).tolist():
                b, p, j = random_case(rng)
                st[i, 0] = torch.tensor(b, device="cuda")
                P[i] = torch.tensor(p)
                Jc[i] = torch.tensor(j, device="cuda")
            Pc = P.cuda()
    print(dict(compared=compared, branch_mismatch=mismatched, rate=mismatched / compared, **worst))
    assert mismatched / compared < 5e-3
    assert worst["pos"] < 5e-3 and worst["quat"] < 1e-4 and worst["vel"] < 0.05 and worst["theta"] < 1e-3 and worst["torque"] < 1e-3


if __name__ == "__main__":
    import torch
    for name, fn in list(globals().items()):
        if name.startswith("test_") and (name != "test_cuda_matches_oracle" or torch.cuda.is_available()):
            fn()
            print("ok", name, flush=True)
