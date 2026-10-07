"""Dependency-free scalar CPU oracle of rl/screw_drive/screw_joint.cu: the screw/thread interaction joint.

A screw is modelled by its interaction, not its threads. One dynamic body (the screw, a square box: crude on purpose)
against a static threaded hole, per world:

  FREE      the screw rides the bit (the env's compliance controller drives it; box3d manifold contacts with the
            plate's clearance hole). Its spin is a scalar DOF (the box body never turns about its own axis).
            Engagement: the tip within the capture radius of the hole's axis at the mouth plane (along in
            [-0.15 mm, +0.3 pitch]) and armed (the tip has been outside that window since the last release), then
              catch mode A: pushing >= 1 N and spinning forward through the thread-start phase (one pass per turn);
                a back-turn through the phase while pushing is a 'click' (a torque blip): the next forward turn
                catches at once and the cross-thread threshold is 1 degree more forgiving;
              catch mode B: at once (World2's referee: operations/driven-fastener.mjs starts the thread on capture).
            At the catch the tilt between the screw axis and the hole axis decides: over theta_x the screw is
            CROSS-THREADED (its helix follows its own axis, binding torque rises k_x per turn), else it is centred.
  ENGAGED   helical joint: the tip held on the engagement axis, the axis held to it, axial travel = pitch x turns
            (a hard position projection plus the matching velocity: one body against static ground). The spindle (a
            quasi-static drive train as World2's operations/spindle.mjs: speed command, clutch at the set torque,
            motor limit, Phillips cam-out capacity F_ax r_eff (1 + mu tan b) / (tan b - mu), hex bit-slip capacity)
            turns it against T_res = T_run + 0.05 d F_ax + crossed k_x turns (+ the clamp once seated).
            Reverse spin backs it out; past the thread start (theta < 0) it is FREE again (disarmed).
  SEATED    the head bears (travel = L - grip): clamp F = kJ pitch dtheta / 2 pi, T_res += K d F; the clutch slipping
            here is success; the clutch slipping before the seat is a failure (cross-thread or jam).
  strip     T_res over T_strip below the clutch: the thread strips (torque falls to 0.3 x, no more advance).

Units: body state as box3d (positions mm, velocities mm/s, angular rad/s, quaternion xyzw); parameters SI except
the mouth point (mm). Layouts of P, J, C below are the kernel's.
"""
from __future__ import annotations

import math

TWO_PI = 2.0 * math.pi
FREE, ENGAGED, SEATED, FAILED = 0, 1, 2, 3
NP, NJ, NC = 24, 24, 2
QSTEPS = 4          # quasi-static sub-iterations of the drive train per substep
DEG1 = math.pi / 180.0

# P (per world, float32)
P_PITCH, P_D, P_RCAP, P_THX, P_TRUN, P_KX, P_TSET, P_KJ, P_K, P_TSTRIP, P_PHIL, P_CAMC, P_HEXCAP, P_TMOT, P_TRAVEL, \
    P_MODEB, P_MOUTH, P_UINS, P_DROP, P_L = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 22, 23
# J (per world joint state)
J_MODE, J_THETA, J_OMEGA, J_SSTRIP, J_U, J_ANCHOR, J_CROSSED, J_PHASE, J_CLICKED, J_ARMED, J_THSEAT, J_CLAMP, J_TORQUE, \
    J_PEAK, J_STRIPPED, J_FAIL, J_SUCCESS, J_SPIN, J_FAX, J_CATCHES = 0, 1, 2, 3, 4, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, \
    19, 20, 21, 22, 23
# C (per world command): spindle speed (rad/s, + tightening), axial push (N, >= 0)
C_W, C_FAX = 0, 1
FAIL_CLUTCH, FAIL_STRIP, FAIL_CAM, FAIL_SLIP = 1, 2, 3, 4


def _rot(q, v):
    x, y, z, w = q
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)]


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]


def _between(p, q):
    """Shortest rotation taking unit p onto unit q (xyzw)."""
    c = p[0] * q[0] + p[1] * q[1] + p[2] * q[2]
    x = [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]]
    w = 1.0 + c
    if w < 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    n = math.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + w * w)
    return [x[0] / n, x[1] / n, x[2] / n, w / n]


def _qnorm(q):
    n = math.sqrt(sum(t * t for t in q))
    return [t / n for t in q] if n > 1e-20 else [0.0, 0.0, 0.0, 1.0]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def step_world(b, P, J, C, h):
    """One substep of one world, in place: b body 0 state (13), P params (NP), J joint state (NJ), C command (NC)."""
    mode = int(J[J_MODE])
    if mode == FAILED or J[J_SUCCESS] > 0.5:
        J[J_OMEGA] = 0.0
        return
    pitch, d, L = P[P_PITCH], P[P_D], P[P_L]
    w_cmd, fax = C[C_W], max(0.0, C[C_FAX])
    q = b[3:7]
    ax_b = _rot(q, [0.0, 0.0, -1.0])
    half_mm = L * 500.0
    tip = [b[k] + ax_b[k] * half_mm for k in range(3)]
    uins = P[P_UINS:P_UINS + 3]
    J[J_TORQUE] = 0.0
    J[J_FAX] = fax
    if mode == FREE:
        J[J_OMEGA] = w_cmd
        J[J_SPIN] += w_cmd * h
        rel = [(tip[k] - P[P_MOUTH + k]) * 1e-3 for k in range(3)]
        along = _dot(rel, uins)
        lat = math.sqrt(max(0.0, _dot(rel, rel) - along * along))
        if along > -2e-4 and fax > 0.0:     # the tip rubbing on the mouth
            J[J_TORQUE] = 0.2 * fax * d * 0.25 * (1.0 if w_cmd > 0 else -1.0 if w_cmd < 0 else 0.0)
        inside = lat <= P[P_RCAP] and along >= -1.5e-4 and along <= 0.3 * pitch
        if not inside:
            J[J_ARMED] = 1.0
            return
        if J[J_ARMED] < 0.5:
            return
        catch = False
        if P[P_MODEB] > 0.5:
            catch = True
        elif fax >= 1.0:
            if J[J_CLICKED] > 0.5 and w_cmd > 0.0:
                catch = True
            else:
                J[J_PHASE] += w_cmd * h
                if J[J_PHASE] >= TWO_PI:
                    catch = True
                elif J[J_PHASE] < 0.0:
                    J[J_CLICKED] = 1.0
                    J[J_PHASE] += TWO_PI
                    J[J_TORQUE] -= 0.3 * P[P_TRUN]
        if not catch:
            return
        c = max(-1.0, min(1.0, _dot(ax_b, uins)))
        tilt = math.acos(c)
        thr = P[P_THX] + (DEG1 if J[J_CLICKED] > 0.5 else 0.0)
        crossed = tilt > thr
        u = ax_b if crossed else uins
        if crossed:
            anchor = tip[:]
        else:
            anchor = [P[P_MOUTH + k] + uins[k] * along * 1e3 for k in range(3)]
        for k in range(3):
            J[J_U + k] = u[k]
            J[J_ANCHOR + k] = anchor[k]
        J[J_CROSSED] = 1.0 if crossed else 0.0
        J[J_MODE] = ENGAGED
        J[J_THETA] = 0.0
        J[J_THSEAT] = -1.0
        J[J_CLAMP] = 0.0
        J[J_CLICKED] = 0.0
        J[J_ARMED] = 0.0
        J[J_PHASE] = 0.0
        J[J_CATCHES] += 1.0
        mode = ENGAGED
    # ---- engaged or seated: the drive train against the thread, QSTEPS quasi-static sub-iterations
    theta, crossed = J[J_THETA], J[J_CROSSED] > 0.5
    kd = P[P_K] * d * P[P_KJ] * pitch / TWO_PI          # clamp torque rate, N m per rad
    hs = h / QSTEPS
    omega, torque = 0.0, 0.0
    released = False
    for _ in range(QSTEPS):
        seated = int(J[J_MODE]) == SEATED
        stripped = J[J_STRIPPED] > 0.5
        bind = P[P_KX] * theta / TWO_PI if crossed else 0.0
        clampT = kd * (theta - J[J_THSEAT]) if seated else 0.0
        if w_cmd > 0.0:
            if stripped:
                tres = 0.3 * J[J_PEAK]
                theta += w_cmd * hs
                omega, torque = w_cmd, tres
                continue
            tres = P[P_TRUN] + 0.05 * d * fax + bind + clampT
            capbit = fax * P[P_CAMC] if P[P_PHIL] > 0.5 else P[P_HEXCAP]
            cap = min(P[P_TSET], capbit, P[P_TMOT])
            if tres > P[P_TSTRIP] and P[P_TSTRIP] < cap:
                J[J_STRIPPED] = 1.0
                J[J_PEAK] = max(J[J_PEAK], P[P_TSTRIP])
                J[J_SSTRIP] = min(pitch * theta / TWO_PI, P[P_TRAVEL])
                omega, torque = w_cmd, 0.3 * P[P_TSTRIP]
                theta += w_cmd * hs
                continue
            if tres >= cap:
                omega, torque = 0.0, cap
                if cap >= P[P_TSET]:
                    if seated:
                        J[J_SUCCESS] = 1.0
                    else:
                        J[J_FAIL] = FAIL_CLUTCH
                        J[J_MODE] = FAILED
                elif cap >= P[P_TMOT]:
                    J[J_FAIL] = FAIL_CLUTCH
                    J[J_MODE] = FAILED
                else:
                    J[J_FAIL] = FAIL_CAM if P[P_PHIL] > 0.5 else FAIL_SLIP
                    J[J_MODE] = FAILED
                break
            theta += w_cmd * hs
            omega, torque = w_cmd, tres
        elif w_cmd < 0.0:
            tl = P[P_TRUN] + bind + (0.8 * clampT if seated else 0.0)
            if stripped:
                tl = 0.3 * J[J_PEAK]
            theta += w_cmd * hs
            omega, torque = w_cmd, -tl
        else:
            omega, torque = 0.0, 0.0
        if not stripped:
            s = pitch * theta / TWO_PI
            if not seated and s >= P[P_TRAVEL]:
                J[J_MODE] = SEATED
                J[J_THSEAT] = P[P_TRAVEL] * TWO_PI / pitch
            elif seated and theta < J[J_THSEAT]:
                J[J_MODE] = ENGAGED
        if theta < 0.0:
            released = True
            break
    J[J_SPIN] += omega * h if not released else w_cmd * h
    J[J_TORQUE] = torque
    J[J_PEAK] = max(J[J_PEAK], torque)
    J[J_OMEGA] = omega
    seated = int(J[J_MODE]) == SEATED
    J[J_CLAMP] = P[P_KJ] * pitch * (theta - J[J_THSEAT]) / TWO_PI if seated else 0.0
    J[J_THETA] = theta
    if released:
        J[J_MODE] = FREE
        J[J_THETA] = 0.0
        J[J_ARMED] = 0.0
        J[J_PHASE] = math.pi
        J[J_CROSSED] = 0.0
        J[J_STRIPPED] = 0.0
        return
    if int(J[J_MODE]) == FAILED:
        return
    # ---- the helical joint: tip on the engagement axis at its travel, axis along it (position projection + velocity)
    u = J[J_U:J_U + 3]
    if J[J_STRIPPED] > 0.5:
        s_eff = J[J_SSTRIP]
    else:
        s_eff = min(pitch * theta / TWO_PI, P[P_TRAVEL])
    ptip = [J[J_ANCHOR + k] + u[k] * s_eff * 1e3 for k in range(3)]
    qn = _qnorm(_qmul(_between(ax_b, u), q))
    axn = _rot(qn, [0.0, 0.0, -1.0])
    for k in range(3):
        b[k] = ptip[k] - axn[k] * half_mm
    for k in range(4):
        b[3 + k] = qn[k]
    moving = int(J[J_MODE]) == ENGAGED and J[J_STRIPPED] < 0.5
    vax = pitch * omega / TWO_PI * 1e3 if moving else 0.0
    for k in range(3):
        b[7 + k] = u[k] * vax
        b[10 + k] = 0.0


def step_reference(state, P, J, C, h):
    """Batched lists: state[W][B][13] (only body 0 is touched), P[W][NP], J[W][NJ], C[W][NC]. Returns copies."""
    import copy
    st, jj = copy.deepcopy(state), copy.deepcopy(J)
    for w in range(len(st)):
        step_world(st[w][0], P[w], jj[w], C[w], h)
    return st, jj
