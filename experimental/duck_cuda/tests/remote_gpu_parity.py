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

    sg = run(gpu, acts)
    ss = run(ser, acts)
    dq = max(float(np.abs(a.q - b.q).max()) for a, b in zip(sg, ss))
    dv = max(float(np.abs(a.v - b.v).max()) for a, b in zip(sg, ss))
    gate("gpu_vs_serial_max_q_diff", dq, 1e-3)
    gate("gpu_vs_serial_max_v_diff", dv, 5e-2)
    pen = max(float((-s.sole_height).max(initial=0.0)) for s in sg)
    gate("gpu_max_penetration_m", pen, 5e-3)
    finite = all(bool(s.finite().all()) for s in sg)
    GATES["gpu_all_finite"] = {"pass": finite}
    if not finite:
        FAIL.append("gpu_all_finite")

    # determinism: second GPU run bit-identical
    gpu2 = CudaDuckLane(E, library_path=os.environ["DUCK_CUDA_LIBRARY"])
    sg2 = run(gpu2, acts)
    det = all(np.array_equal(a.q, b.q) and np.array_equal(a.v, b.v)
              for a, b in zip(sg, sg2))
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
