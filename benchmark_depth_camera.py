"""End-to-end CUDA throughput gate for analytic OBB depth cameras."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .benchmarking import BenchmarkResult, CapabilitySet
from .extension import load_extension
from .smoke_camera import run as run_camera_correctness


CONTRACT_ID = "box3d.analytic-depth-camera/v1"
WORLDS = 1024
CAMERAS = 2
WIDTH = 16
HEIGHT = 16
RAYS_PER_WORLD = CAMERAS * WIDTH * HEIGHT
STEPS = 240
BODY_COUNT = 8
MAXIMUM_DISTANCE_M = 10.0


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=WORLDS)
    parser.add_argument("--steps", type=int, default=STEPS)
    return parser.parse_args()


def _fixture(torch, worlds: int):
    device = "cuda"
    state = torch.zeros((worlds, BODY_COUNT, 13), dtype=torch.float32, device=device)
    state[..., 6] = 1.0
    world = torch.arange(worlds, dtype=torch.float32, device=device)
    yaw = 0.15 * torch.sin(world * 0.017)
    state[:, 0, 0] = 0.75
    state[:, 0, 4] = torch.sin(yaw * 0.5)
    state[:, 0, 6] = torch.cos(yaw * 0.5)
    for body in range(1, BODY_COUNT):
        column = (body - 1) % 3
        row = (body - 1) // 3
        state[:, body, 0] = (column - 1) * 1.4
        state[:, body, 1] = (row - 1) * 1.1
        state[:, body, 2] = 2.5 + body * 0.75
        body_yaw = (body - 4) * 0.11
        state[:, body, 4] = math.sin(body_yaw * 0.5)
        state[:, body, 6] = math.cos(body_yaw * 0.5)
    half = torch.full(
        (worlds, BODY_COUNT, 3), 0.4, dtype=torch.float32, device=device
    )
    half[:, 0] = 0.1
    enabled = torch.ones((worlds, BODY_COUNT), dtype=torch.uint8, device=device)
    enabled[:, 0] = 0
    parent = torch.tensor([-1, 0], dtype=torch.int64, device=device)
    position = torch.tensor(
        [[-0.75, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    quaternion = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    intrinsics = torch.tensor(
        [
            [12.0, 12.0, WIDTH / 2.0, HEIGHT / 2.0, MAXIMUM_DISTANCE_M],
            [12.0, 12.0, WIDTH / 2.0, HEIGHT / 2.0, MAXIMUM_DISTANCE_M],
        ],
        dtype=torch.float32,
        device=device,
    )
    pixel_camera = []
    pixel_xy = []
    for camera in range(CAMERAS):
        for y in range(HEIGHT):
            for x in range(WIDTH):
                pixel_camera.append(camera)
                pixel_xy.append((x, y))
    return (
        state,
        half,
        enabled,
        parent,
        position,
        quaternion,
        intrinsics,
        torch.tensor(pixel_camera, dtype=torch.int64, device=device),
        torch.tensor(pixel_xy, dtype=torch.int64, device=device),
    )


def _launch(extension, fixture):
    (
        state,
        half,
        enabled,
        parent,
        position,
        quaternion,
        intrinsics,
        pixel_camera,
        pixel_xy,
    ) = fixture
    origins, directions, maximum, forward = extension.camera_rays(
        state,
        parent,
        position,
        quaternion,
        intrinsics,
        pixel_camera,
        pixel_xy,
    )
    distance, body_index, normal = extension.ray_cast(
        state, half, enabled, origins, directions, maximum
    )
    depth_z, hit_range = extension.camera_depth(
        distance, body_index, forward
    )
    return depth_z, hit_range, body_index, normal, origins, directions


def main() -> int:
    args = arguments()
    if args.worlds != WORLDS or args.steps != STEPS:
        raise ValueError(
            f"depth-camera contract requires exactly {WORLDS} worlds and {STEPS} steps"
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("depth-camera throughput requires a visible CUDA device")
    extension = load_extension()
    correctness = run_camera_correctness()
    fixture = _fixture(torch, args.worlds)
    warmup = _launch(extension, fixture)
    torch.cuda.synchronize()
    depth_z, hit_range, body_index, normal, _, directions = warmup
    hit = body_index >= 0
    miss = ~hit
    direction_error = float(
        (torch.linalg.vector_norm(directions, dim=-1) - 1.0).abs().max().item()
    )
    output_gate = {
        "finite_depth": bool(torch.isfinite(depth_z).all().item()),
        "finite_range": bool(torch.isfinite(hit_range).all().item()),
        "valid_body_indices": bool(
            torch.all((body_index >= -1) & (body_index < BODY_COUNT)).item()
        ),
        "misses_are_zero": bool(
            torch.equal(depth_z[miss], torch.zeros_like(depth_z[miss]))
            and torch.equal(hit_range[miss], torch.zeros_like(hit_range[miss]))
        ),
        "depth_not_greater_than_range": bool(
            torch.all(depth_z[hit] <= hit_range[hit] + 1.0e-6).item()
        ),
        "range_within_far": bool(
            torch.all(hit_range[hit] <= MAXIMUM_DISTANCE_M).item()
        ),
        "directions_normalized": direction_error <= 2.0e-5,
        "nontrivial_hit_and_miss_population": bool(hit.any().item() and miss.any().item()),
    }
    if not all(output_gate.values()):
        raise RuntimeError(
            "depth-camera output gate failed: " + json.dumps(output_gate, sort_keys=True)
        )
    replay = _launch(extension, fixture)
    torch.cuda.synchronize()
    deterministic = all(torch.equal(left, right) for left, right in zip(warmup, replay))
    if not deterministic:
        raise RuntimeError("depth-camera replay was not bit-exact")
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    start.record()
    final = None
    for _ in range(args.steps):
        final = _launch(extension, fixture)
    finish.record()
    torch.cuda.synchronize()
    duration = start.elapsed_time(finish) / 1000.0
    if final is None or not bool(torch.isfinite(final[0]).all().item()):
        raise RuntimeError("timed depth-camera output was not finite")
    result = BenchmarkResult(
        backend="box3d_cuda_depth_camera",
        backend_version="box3d-cuda-camera-v1",
        workload=(
            "two calibrated 16x16 scene/wrist cameras over eight OBB slots; "
            "camera rays + nearest hit + optical depth"
        ),
        contract_id=CONTRACT_ID,
        device=torch.cuda.get_device_name(0),
        worlds=args.worlds,
        bodies_per_world=BODY_COUNT,
        steps=args.steps,
        duration_seconds=duration,
        capabilities=CapabilitySet(
            rigid_body_integration=False,
            static_plane_contacts=False,
            dynamic_contacts=False,
            articulated_joints=False,
            continuous_collision=False,
            ray_queries=True,
            camera_rendering=False,
            robot_manipulation=False,
        ),
        correctness={
            "passed": True,
            "cpu_cuda_micro": correctness,
            "timed_output_gate": output_gate,
            "deterministic_replay_passed": deterministic,
            "maximum_direction_norm_error": direction_error,
            "hit_ratio": float(hit.float().mean().item()),
            "analytic_depth_camera": True,
            "reported_rgb_or_raster_pixels": False,
        },
        peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
        notes=(
            "Timed interval includes CUDA camera pose/ray compilation, linear OBB ray queries, and optical-depth conversion.",
            "This is analytic OBB depth, not rasterization, RGB, materials, textures, lens distortion, rolling shutter, or noise.",
            "No matched PhysX/ManiSkill speedup is claimed by this result.",
        ),
    )
    payload = result.to_dict()
    pixels = args.worlds * RAYS_PER_WORLD * args.steps
    frames = args.worlds * CAMERAS * args.steps
    payload.update(
        {
            "cameras_per_world": CAMERAS,
            "camera_width": WIDTH,
            "camera_height": HEIGHT,
            "rays_per_world": RAYS_PER_WORLD,
            "depth_pixels_per_second": pixels / duration,
            "camera_frames_per_second": frames / duration,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "device": result.device,
                "duration_seconds": duration,
                "depth_pixels_per_second": payload["depth_pixels_per_second"],
                "camera_frames_per_second": payload["camera_frames_per_second"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
