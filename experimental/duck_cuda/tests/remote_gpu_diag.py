#!/usr/bin/env python3
"""GPU occupancy diagnostic, run ON the GPU host (needs numpy only).

Env: DUCK_CUDA_LIBRARY (the CUDA .so). Writes gpu-diag.json.

Prints the warp-per-env launch geometry, per-kernel registers/thread and
local memory/thread (cudaFuncGetAttributes), achieved occupancy in blocks/SM
(cudaOccupancyMaxActiveBlocksPerMultiprocessor), the stack limit, the
resident-env estimate, and a scaling micro-sweep over E so the knee where
wall time turns linear is visible next to the estimate.

Reading the sweep: throughput should grow with E until roughly
resident_envs_estimate environments are in flight and flatten after. A knee
far BELOW the estimate means something else caps residency; a knee AT ~8 x
sm_count envs with step_blocks_per_sm == 1 is the register-bound signature
(rebuild with a different -DDW_MIN_BLOCKS_PER_SM).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from walk.env.cuda_lane import CudaDuckLane  # noqa: E402

SWEEP_ENVS = (256, 512, 1024, 2048, 4096, 8192)
WARM_TICKS = 100
TIMED_TICKS = 400


def sweep_point(path, envs):
    lane = CudaDuckLane(envs, library_path=path)
    rng = np.random.default_rng(1)
    home, lim = lane.home_joint_q, lane.joint_limits

    def run(ticks):
        for _ in range(ticks // 10):
            a = np.clip(rng.normal(0, 0.5, 14), -1, 1)
            t = np.clip(home + 0.25 * a, lim[:, 0], lim[:, 1])
            rc, _ = lane.tick_block(np.tile(t, (envs, 1)), 10)
            if rc:
                raise SystemExit(f"fault rc={rc} at E={envs}")

    run(WARM_TICKS)
    t0 = time.perf_counter()
    run(TIMED_TICKS)
    dt = time.perf_counter() - t0
    lane.close()
    return dt


def main():
    path = os.environ["DUCK_CUDA_LIBRARY"]
    probe = CudaDuckLane(SWEEP_ENVS[0], library_path=path)
    info = probe.device_info()
    probe.close()
    envs_per_block = max(1, info["threads_per_block"] // max(1, info["lanes_per_env"]))
    print("=== launch geometry / occupancy ===")
    for key, value in info.items():
        print(f"  {key:26s} = {value}")
    print(f"  envs_per_block             = {envs_per_block}")
    if info["step_blocks_per_sm"]:
        print(f"  -> resident envs (step)    = "
              f"{info['step_blocks_per_sm'] * info['sm_count'] * envs_per_block}")

    print("=== scaling micro-sweep (%d timed ticks each) ===" % TIMED_TICKS)
    rows = []
    prev_tps = None
    for envs in SWEEP_ENVS:
        dt = sweep_point(path, envs)
        tps = envs * TIMED_TICKS / dt
        scale = "" if prev_tps is None else f"  x{tps / prev_tps:.2f} vs prev E"
        prev_tps = tps
        # grid/block dims actually used by dwc1_step for this E
        lanes_total = envs * info["lanes_per_env"]
        blocks = (lanes_total + info["threads_per_block"] - 1) \
            // info["threads_per_block"]
        print(f"  E={envs:6d}  grid={blocks:5d}x{info['threads_per_block']}"
              f"  {tps / 1e6:7.3f}M ticks/s  ({dt:.2f}s){scale}")
        rows.append({"environments": envs, "wall_s": round(dt, 3),
                     "ticks_per_s": round(tps), "grid_blocks": int(blocks),
                     "block_threads": info["threads_per_block"]})

    out = {"schema": "duckgridwalk.gpu-diag/1", "device_info": info,
           "envs_per_block": envs_per_block, "sweep": rows,
           "timed_ticks": TIMED_TICKS}
    Path("gpu-diag.json").write_text(json.dumps(out, indent=1))
    print("wrote gpu-diag.json")


if __name__ == "__main__":
    main()
