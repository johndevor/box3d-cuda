#!/usr/bin/env python3
"""GPU throughput benchmark, run ON the GPU host.

Env: DUCK_CUDA_LIBRARY (parity build, fmad off); optional DUCK_CUDA_FAST_LIBRARY.
Writes gpu-bench.json. Random PD targets, resident stepping, timed via
wall clock around synchronous dwc1_step batches.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from walk.env.cuda_lane import CudaDuckLane  # noqa: E402

RESULTS = []


def bench(label, path, envs, warm_ticks=200, timed_ticks=1000):
    lane = CudaDuckLane(envs, library_path=path)
    rng = np.random.default_rng(1)
    home = lane.home_joint_q
    lim = lane.joint_limits

    def step_block(n):
        # One tick_block launch per 10-tick decision (the training fast
        # path). Per-tick lane.tick() calls under-read badly here: 10x the
        # launch/sync/readback overhead per decision dominates the kernel.
        for _ in range(n // 10):
            a = np.clip(rng.normal(0, 0.5, 14), -1, 1)
            t = np.clip(home + 0.25 * a, lim[:, 0], lim[:, 1])
            targets = np.tile(t, (envs, 1)).astype(np.float64)
            rc, _ = lane.tick_block(targets, 10)
            if rc:
                raise SystemExit(f"fault rc={rc} at {label} E={envs}")

    step_block(warm_ticks)
    t0 = time.perf_counter()
    step_block(timed_ticks)
    dt = time.perf_counter() - t0
    tps = envs * timed_ticks / dt
    row = {"label": label, "environments": envs, "ticks": timed_ticks,
           "wall_s": round(dt, 3), "ticks_per_s": round(tps)}
    RESULTS.append(row)
    print(f"{label:8s} E={envs:6d}  {tps/1e6:7.3f}M ticks/s  ({dt:.2f}s wall)")
    lane.close()


def main():
    libs = [("parity", os.environ["DUCK_CUDA_LIBRARY"])]
    fast = os.environ.get("DUCK_CUDA_FAST_LIBRARY")
    if fast and Path(fast).exists():
        libs.append(("fast", fast))
    for label, path in libs:
        for envs in (1024, 4096, 8192, 16384):
            bench(label, path, envs)
    best = max(r["ticks_per_s"] for r in RESULTS)
    out = {"schema": "duckgridwalk.gpu-bench/1", "results": RESULTS,
           "best_ticks_per_s": best, "target_ticks_per_s": 1_000_000,
           "target_met": best >= 1_000_000}
    Path("gpu-bench.json").write_text(json.dumps(out, indent=1))
    print(f"BEST: {best/1e6:.3f}M ticks/s (target 1M: {'MET' if out['target_met'] else 'NOT MET'})")


if __name__ == "__main__":
    main()
