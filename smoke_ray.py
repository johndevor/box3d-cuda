"""Small measured CUDA correctness gate for Stage 6 batched OBB rays."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .extension import load_extension, ray_cast


def _state(torch):
    # Two worlds, two bodies. World 0 exercises nearest/tie/disable behavior;
    # world 1 proves world isolation with a rotated OBB.
    state = torch.zeros((2, 2, 13), dtype=torch.float32, device="cuda")
    state[..., 6] = 1.0
    state[0, 0, 0] = 2.0
    state[0, 1, 0] = 2.0
    state[1, 0, 0] = 3.0
    angle = math.pi / 2.0
    state[1, 0, 5] = math.sin(angle / 2.0)
    state[1, 0, 6] = math.cos(angle / 2.0)
    half = torch.tensor(
        [[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
         [[1.0, 0.25, 0.5], [0.5, 0.5, 0.5]]],
        dtype=torch.float32,
        device="cuda",
    )
    enabled = torch.tensor([[1, 1], [1, 0]], dtype=torch.uint8, device="cuda")
    return state, half, enabled


def run() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 6 smoke requires a visible CUDA device")
    load_extension()
    state, half, enabled = _state(torch)
    origins = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 1.1, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=torch.float32,
        device="cuda",
    )
    directions = torch.tensor(
        [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]],
        dtype=torch.float32,
        device="cuda",
    )
    maximum = torch.tensor(
        [[10.0, 10.0, 10.0, 1.5], [10.0, 10.0, 10.0, 10.0]],
        dtype=torch.float32,
        device="cuda",
    )
    distance, body_index, normal = ray_cast(
        state, half, enabled, origins, directions, maximum
    )
    torch.cuda.synchronize()
    distance_cpu = distance.cpu()
    body_cpu = body_index.cpu()
    normal_cpu = normal.cpu()
    checks = {
        "nearest_tie_uses_lower_body": int(body_cpu[0, 0]) == 0 and abs(float(distance_cpu[0, 0]) - 1.5) <= 1.0e-6,
        "inside_origin_uses_exit": int(body_cpu[0, 1]) == 0 and abs(float(distance_cpu[0, 1]) - 0.5) <= 1.0e-6,
        "parallel_miss": int(body_cpu[0, 2]) == -1 and float(distance_cpu[0, 2]) == 10.0,
        "maximum_distance_is_inclusive": int(body_cpu[0, 3]) == 0 and abs(float(distance_cpu[0, 3]) - 1.5) <= 1.0e-6,
        "rotated_obb_hit": int(body_cpu[1, 0]) == 0 and abs(float(distance_cpu[1, 0]) - 2.75) <= 1.0e-5,
        "rotated_obb_parallel_miss": int(body_cpu[1, 1]) == -1,
        "disabled_body_ignored": int(body_cpu[1, 3]) == -1,
        "hit_normal": torch.allclose(normal_cpu[0, 0], torch.tensor([-1.0, 0.0, 0.0]), atol=1.0e-6),
        "miss_normal_zero": torch.equal(normal_cpu[0, 2], torch.zeros(3)),
    }
    # Re-run the exact packet to require deterministic distances, IDs, normals.
    replay = ray_cast(state, half, enabled, origins, directions, maximum)
    torch.cuda.synchronize()
    deterministic = all(torch.equal(left, right) for left, right in zip((distance, body_index, normal), replay))
    checks["deterministic_replay"] = deterministic
    if not all(checks.values()):
        raise RuntimeError("CUDA ray smoke failed: " + json.dumps(checks, sort_keys=True))
    return {
        "schema_version": "box3d.ray-smoke/v1",
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "extension": "factory_box3d_cuda_v10",
        "checks": checks,
        "distances": distance_cpu.tolist(),
        "body_indices": body_cpu.tolist(),
        "normals": normal_cpu.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
