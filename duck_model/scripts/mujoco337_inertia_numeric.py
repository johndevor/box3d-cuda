# Copyright 2021 DeepMind Technologies Limited
# Python diagnostic adaptation, 2026 Box3D CUDA contributors.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Pure numerical reproduction of MuJoCo 3.3.7 user_util.cc mjuu_eig3.

Original: https://github.com/google-deepmind/mujoco/blob/3.3.7/src/user/user_util.cc
SHA256 f9d5ef77317707039f12658e620ff3393f1eb8a0e7087111e4740207f2ee2522.
This is not a physics engine and imports no MuJoCo or device runtime. Diagnostic
traces are additive; scalar evaluation order mirrors the upstream source.
"""
import math

EIG_EPS = 1e-12
NORMALIZE_EPS = 1e-14


def _finite(values):
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite eigensolver intermediate")


def _normalize(q):
    norm = 0.
    for item in q:
        norm += item*item
    _finite([norm])
    if norm < NORMALIZE_EPS:
        return list(q)
    norm = math.sqrt(norm)
    return [item/norm for item in q] if abs(norm-1) > NORMALIZE_EPS else list(q)


def _multiply_quaternion(a, b):
    return _normalize([
        a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
        a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
        a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
        a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]])


def _rotation(q):
    a,b,c,d = q
    return [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c),
            2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b),
            2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]


def _transpose(a):
    return [a[j*3+i] for i in range(3) for j in range(3)]


def _multiply(a, b):
    return [a[i*3]*b[j]+a[i*3+1]*b[3+j]+a[i*3+2]*b[6+j]
            for i in range(3) for j in range(3)]


def compiler_eig3(tensor):
    try:
        if len(tensor) != 3 or any(len(row) != 3 for row in tensor):
            raise ValueError("expected 3x3 tensor")
        matrix = [float(item) for row in tensor for item in row]
    except (TypeError, OverflowError) as exc:
        raise ValueError("expected numeric 3x3 tensor") from exc
    if not all(math.isfinite(x) for x in matrix):
        raise ValueError("nonfinite tensor")
    if any(matrix[i*3+j] != matrix[j*3+i] for i in range(3) for j in range(3)):
        raise ValueError("expected exactly symmetric source tensor")
    q, trace, reason = [1., 0., 0., 0.], [], None
    for iteration in range(500):
        frame = _rotation(q)
        diagonalized = _multiply(_multiply(_transpose(frame), matrix), frame)
        _finite(frame + diagonalized)
        values = [diagonalized[0], diagonalized[4], diagonalized[8]]
        if abs(diagonalized[1]) > abs(diagonalized[2]) and abs(diagonalized[1]) > abs(diagonalized[5]):
            row, col, axis = 0, 1, 2
        elif abs(diagonalized[2]) > abs(diagonalized[5]):
            row, col, axis = 0, 2, 1
        else:
            row, col, axis = 1, 2, 0
        offdiag = diagonalized[3*row+col]
        entry = {"iteration": iteration, "max_offdiag": abs(offdiag),
                 "diagonal_gap": abs(values[col]-values[row]), "row": row, "column": col}
        _finite([entry["max_offdiag"], entry["diagonal_gap"]])
        trace.append(entry)
        if abs(offdiag) < EIG_EPS:
            reason = "absolute_offdiagonal"
            break
        _finite([values[col]-values[row], 2*offdiag])
        tau = (values[col]-values[row])/(2*offdiag)
        _finite([tau, tau*tau])
        t = 1/(tau+math.sqrt(1+tau*tau)) if tau >= 0 else -1/(-tau+math.sqrt(1+tau*tau))
        cosine = 1/math.sqrt(1+t*t)
        entry.update(tau=tau, tangent=t, cosine=cosine)
        if cosine > 1-EIG_EPS:
            reason = "rotation_cosine"
            break
        rotation = [0., 0., 0., 0.]
        rotation[axis+1] = -math.sqrt(.5-.5*cosine) if tau >= 0 else math.sqrt(.5-.5*cosine)
        if axis == 1:
            rotation[axis+1] = -rotation[axis+1]
        rotation[0] = math.sqrt(1-rotation[axis+1]*rotation[axis+1])
        q = _normalize(_multiply_quaternion(q, _normalize(rotation)))
    if reason is None:
        raise ValueError("compiler reproduction hit 500-iteration cap")
    pre_sort_frame, pre_sort_values = list(frame), list(values)
    pre_sort_diagonal = [values[i//3] if i//3 == i%3 else 0. for i in range(9)]
    pre_sort_reconstructed = _multiply(_multiply(frame, pre_sort_diagonal), _transpose(frame))
    for j in range(3):
        lead = j % 2
        if values[lead]+EIG_EPS < values[lead+1]:
            values[lead], values[lead+1] = values[lead+1], values[lead]
            rotation = [.707106781186548, 0., 0., 0.]
            rotation[(lead+2)%3+1] = rotation[0]
            q = _normalize(_multiply_quaternion(q, rotation))
    frame = _rotation(q)
    diagonal = [values[i//3] if i//3 == i%3 else 0. for i in range(9)]
    reconstructed = _multiply(_multiply(frame, diagonal), _transpose(frame))
    _finite(values + q + reconstructed)
    return {"stop_reason": reason, "trace": trace, "values": values,
            "quaternion_wxyz": q,
            "pre_sort_frame": [pre_sort_frame[i:i+3] for i in (0, 3, 6)],
            "pre_sort_D": [diagonalized[i:i+3] for i in (0, 3, 6)],
            "pre_sort_values": pre_sort_values,
            "pre_sort_reconstructed": [pre_sort_reconstructed[i:i+3] for i in (0, 3, 6)],
            "reconstructed": [reconstructed[i:i+3] for i in (0, 3, 6)]}


def numerical_comparison(actual, expected):
    """Two-layer r2 tensor gate, limited to resolved SPD inertia tensors.

    See docs/open-duck-inertia-bound-r2.md. The original elementwise constants
    are applied to PRE-sort compiler reconstruction, with an additional 1e-8
    relative-Frobenius agreement gate. No non-inertia model check is affected.
    """
    import sys
    import numpy as np

    u = 2.**-53
    def up(value):
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid numerical bound intermediate")
        return math.nextafter(value, math.inf)

    def norm(matrix):
        return math.hypot(*(float(v) for v in np.asarray(matrix).flat))

    try:
        a, source = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    except (TypeError, OverflowError) as exc:
        raise ValueError("expected numeric tensor") from exc
    if a.shape != (3,3) or source.shape != (3,3) or not np.isfinite(a).all() or not np.isfinite(source).all():
        raise ValueError("expected finite 3x3 tensors")
    scale = up(norm(source))
    if scale < sys.float_info.min or scale > sys.float_info.max/64:
        raise ValueError("unsupported tensor scale")
    if not np.array_equal(source, source.T) or norm(a-a.T) > 32*u*scale:
        raise ValueError("expected symmetric tensors")
    se, ae = np.linalg.eigvalsh(source/scale)*scale, np.linalg.eigvalsh(a/scale)*scale
    report = {"schema": "box3d.mujoco337.inertia_admission/r2", "accepted": False,
              "failure": None, "reference": "pre-sort compiler reconstruction",
              "source_norm": scale, "original_atol": 1e-10, "original_rtol": 1e-8}
    def reject(message):
        report["failure"] = message
        return report
    if not np.isfinite(se).all() or not np.isfinite(ae).all() or min(se) < NORMALIZE_EPS or min(ae) < NORMALIZE_EPS:
        return reject("nonpositive or unresolved principal inertia")
    # Rounding allowance only, not a physical triangle relaxation.
    if se[0]+se[1]-se[2] < -64*u*scale or ae[0]+ae[1]-ae[2] < -64*u*scale:
        return reject("nonphysical principal-inertia triangle")
    reproduction = compiler_eig3(source.tolist())
    frame = np.asarray(reproduction["pre_sort_frame"])
    d = np.asarray(reproduction["pre_sort_D"])
    reference = np.asarray(reproduction["pre_sort_reconstructed"])
    post = np.asarray(reproduction["reconstructed"])
    gamma5 = up(5*u/(1-5*u))
    r_squared = up(up(norm(frame))**2)
    # Dot3: three multiplies + two adds. Every iteration recomputes from A.
    eta = up(up(gamma5*(2+gamma5))*up(r_squared*scale))
    delta = up(up(norm(frame@frame.T-np.eye(3))) + up(gamma5*r_squared))
    gap = up(up((1+delta)*scale) + up(2*eta))
    epsilon = up(EIG_EPS+32*u)
    tangent_max = up(math.sqrt(up(epsilon*(2-epsilon)))/(1-epsilon))
    cosine_bound = up(up(gap*tangent_max)/(1-up(tangent_max*tangent_max)))
    if EIG_EPS > cosine_bound:
        return reject("unsupported absolute-stop-dominated tensor scale")
    maximum_offdiag = max(abs(float(d[0,1])), abs(float(d[0,2])), abs(float(d[1,2])))
    if maximum_offdiag > cosine_bound:
        return reject("reproduced stopping trace exceeds analytic bound")
    pre_bound = up(up(delta*(2+delta)*scale) + up((1+delta)*up(math.sqrt(6)*cosine_bound+6*eta)))
    reconstruction_eta = up(up(gamma5*(2+gamma5))*up(r_squared*up(norm(np.diag(np.diag(d))))))
    source_bound = up(pre_bound+reconstruction_eta)
    source_error = norm(reference-source)
    post_error = norm(post-reference)
    actual_error = norm(a-reference)
    tight_bound = 1e-8*scale  # Existing relative tolerance, now also scale-invariant.
    report.update(source_frobenius_bound=source_bound, source_frobenius_error=source_error,
                  actual_frobenius_error=actual_error, actual_relative_error=actual_error/scale,
                  post_sort_frobenius_error=post_error, supplemental_bound=tight_bound,
                  source_to_actual_bound=up(source_bound+tight_bound),
                  delta=delta, eta=eta, cosine_offdiagonal_bound=cosine_bound,
                  trace_max_offdiagonal=maximum_offdiag, reproduction=reproduction)
    if source_error > source_bound:
        return reject("source reconstruction exceeds algorithm-derived bound")
    if post_error > tight_bound:
        return reject("source sorting violates tight reconstruction consistency")
    if not np.all(np.abs(a-reference) <= 1e-10+1e-8*np.abs(reference)) or actual_error > tight_bound:
        return reject("compiled tensor does not match pinned numerical reproduction")
    report["accepted"] = True
    return report
