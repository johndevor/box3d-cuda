#!/usr/bin/env python3
"""GPU-vs-serial parity gate, run ON the GPU host (needs numpy only).

Env: DUCK_CUDA_LIBRARY (CUDA .so), DUCK_CUDA_SERIAL_LIBRARY (serial .so).
Writes gpu-parity.json and exits nonzero on any gate failure.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from walk.env.cuda_lane import CudaDuckLane  # noqa: E402

E = 64
GATES = {}
FAIL = []


def gate(name, value, bound):
    ok = bool(value <= bound)
    GATES[name] = {"value": float(value), "bound": float(bound), "pass": ok}
    if not ok:
        FAIL.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {value:.3e} (bound {bound:.1e})")


def run(lane, actions):
    states = []
    home = lane.home_joint_q
    lim = lane.joint_limits
    for a in actions:
        target = np.clip(home + 0.25 * a, lim[:, 0], lim[:, 1])
        rc, diags = lane.tick(np.tile(target, (lane.E, 1)).astype(np.float64))
        if rc:
            raise SystemExit(f"native fault rc={rc}")
        states.append(lane.read())
    return states


def main():
    gpu = CudaDuckLane(E, library_path=os.environ["DUCK_CUDA_LIBRARY"])
    ser = CudaDuckLane(E, library_path=os.environ["DUCK_CUDA_SERIAL_LIBRARY"])
    rng = np.random.default_rng(917)
    acts = [np.zeros(14)] * 500 + list(np.clip(rng.normal(0, 0.5, (300, 14)), -1, 1))

    # Windowed numerics gate: chaotic contact dynamics amplify fp32 rounding
    # exponentially, so free-running long-horizon agreement is not a numerics
    # test. Every WINDOW ticks both lanes are resynced to the serial state
    # (zeroed warm start on both sides) and per-window drift is bounded.
    WINDOW = 100
    wq = wv = 0.0
    sg = []
    for w0 in range(0, len(acts), WINDOW):
        chunk = acts[w0:w0 + WINDOW]
        s_states = run(ser, chunk)
        g_states = run(gpu, chunk)
        sg.extend(g_states)
        wq = max(wq, max(float(np.abs(a.q - b.q).max())
                         for a, b in zip(g_states, s_states)))
        wv = max(wv, max(float(np.abs(a.v - b.v).max())
                         for a, b in zip(g_states, s_states)))
        sync = s_states[-1]
        zero_w = np.zeros(42, np.float32)
        for e in range(E):
            ser.set_state(e, sync.q[e], sync.v[e], zero_w)
            gpu.set_state(e, sync.q[e], sync.v[e], zero_w)
    # Bounds rationale: GPU and glibc libm differ by ULPs (notably hypot), and
    # contact impulses amplify that at impact ticks. Measured on RTX 5090:
    # q 4.5e-4 rad / 100 ticks, v 0.107 at touchdown ticks, with bitwise GPU
    # determinism and penetration identical to serial. Correctness is carried
    # by the f64-oracle gates (test_serial_parity.py) + the gates below; these
    # bounds catch implementation divergence, not fp32 impact noise.
    gate("windowed_max_q_diff_100t", wq, 1e-3)
    gate("windowed_max_v_diff_100t", wv, 2.5e-1)
    pen = max(float((-s.sole_height).max(initial=0.0)) for s in sg)
    gate("gpu_max_penetration_m", pen, 5e-3)
    finite = all(bool(s.finite().all()) for s in sg)
    GATES["gpu_all_finite"] = {"pass": finite}
    if not finite:
        FAIL.append("gpu_all_finite")

    # determinism: two fresh free-running GPU lanes must agree bit-for-bit
    ga = run(CudaDuckLane(E, library_path=os.environ["DUCK_CUDA_LIBRARY"]), acts)
    gb = run(CudaDuckLane(E, library_path=os.environ["DUCK_CUDA_LIBRARY"]), acts)
    det = all(np.array_equal(a.q, b.q) and np.array_equal(a.v, b.v)
              for a, b in zip(ga, gb))
    GATES["gpu_determinism_bitwise"] = {"pass": bool(det)}
    if not det:
        FAIL.append("gpu_determinism_bitwise")

    result = {"schema": "duckgridwalk.gpu-parity/1", "environments": E,
              "ticks": len(acts), "gates": GATES, "failures": FAIL,
              "passed": not FAIL}
    Path("gpu-parity.json").write_text(json.dumps(result, indent=1))
    print("PARITY:", "PASSED" if not FAIL else f"FAILED {FAIL}")
    sys.exit(0 if not FAIL else 1)


if __name__ == "__main__":
    main()
