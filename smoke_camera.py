"""Measured CUDA correctness gate for calibrated multi-camera depth queries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .extension import depth_camera_query, load_extension
from .ray_reference import (
    CameraRig,
    PinholeCamera,
    depth_images_from_hits,
    make_camera_rays,
    query_rays,
)


def _body(position, quaternion=(0.0, 0.0, 0.0, 1.0)):
    return [*position, *quaternion, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _fixture():
    half_angle = math.pi / 4.0
    states = [
        [_body((0.0, 0.0, 0.0)), _body((0.0, 0.0, 5.0))],
        [
            _body(
                (1.0, 0.0, 0.0),
                (0.0, math.sin(half_angle), 0.0, math.cos(half_angle)),
            ),
            _body((5.0, 0.0, 0.0)),
        ],
    ]
    half_extents = [
        [(0.25, 0.25, 0.25), (10.0, 10.0, 0.5)],
        [(0.25, 0.25, 0.25), (0.5, 10.0, 10.0)],
    ]
    rig = CameraRig(
        (
            PinholeCamera(
                "scene",
                3,
                1,
                2.0,
                2.0,
                1.5,
                0.5,
                10.0,
                position_parent_m=(0.0, 3.0, 0.0),
            ),
            PinholeCamera(
                "wrist",
                2,
                2,
                2.0,
                2.0,
                1.0,
                1.0,
                10.0,
                parent_body=0,
                position_parent_m=(0.0, 0.0, 1.0),
            ),
        )
    )
    return states, half_extents, rig


def run() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("depth-camera smoke requires a visible CUDA device")
    load_extension()
    states, half_extents, rig = _fixture()
    cpu_rays = make_camera_rays(rig, states)
    cpu_hits = query_rays(
        cpu_rays.origins_m,
        cpu_rays.directions,
        cpu_rays.maximum_distance_m,
        states,
        half_extents,
        [[], []],
        [[], []],
    )
    cpu_images = depth_images_from_hits(rig, cpu_rays, cpu_hits)
    expected_depth = []
    expected_range = []
    expected_body = []
    for world_images in cpu_images:
        depth_row, range_row, body_row = [], [], []
        for image in world_images:
            depth_row.extend(value for row in image.depth_z_m for value in row)
            range_row.extend(value for row in image.range_m for value in row)
            body_row.extend(value for row in image.body_index for value in row)
        expected_depth.append(depth_row)
        expected_range.append(range_row)
        expected_body.append(body_row)
    device = "cuda"
    camera_rows = rig.cameras
    pixel_camera = []
    pixel_xy = []
    for camera_index, camera in enumerate(camera_rows):
        for y in range(camera.height):
            for x in range(camera.width):
                pixel_camera.append(camera_index)
                pixel_xy.append((x, y))
    state = torch.tensor(states, dtype=torch.float32, device=device)
    half = torch.tensor(half_extents, dtype=torch.float32, device=device)
    enabled = torch.ones((2, 2), dtype=torch.uint8, device=device)
    parent = torch.tensor(
        [camera.parent_body for camera in camera_rows],
        dtype=torch.int64,
        device=device,
    )
    position = torch.tensor(
        [camera.position_parent_m for camera in camera_rows],
        dtype=torch.float32,
        device=device,
    )
    quaternion = torch.tensor(
        [camera.quaternion_parent_from_camera_xyzw for camera in camera_rows],
        dtype=torch.float32,
        device=device,
    )
    intrinsics = torch.tensor(
        [
            (camera.fx, camera.fy, camera.cx, camera.cy, camera.maximum_distance_m)
            for camera in camera_rows
        ],
        dtype=torch.float32,
        device=device,
    )
    pixel_camera_tensor = torch.tensor(
        pixel_camera, dtype=torch.int64, device=device
    )
    pixel_xy_tensor = torch.tensor(pixel_xy, dtype=torch.int64, device=device)
    result = depth_camera_query(
        state,
        half,
        enabled,
        parent,
        position,
        quaternion,
        intrinsics,
        pixel_camera_tensor,
        pixel_xy_tensor,
    )
    replay = depth_camera_query(
        state,
        half,
        enabled,
        parent,
        position,
        quaternion,
        intrinsics,
        pixel_camera_tensor,
        pixel_xy_tensor,
    )
    torch.cuda.synchronize()
    depth_z, hit_range, body_index, normal, origins, directions = result
    expected_depth_tensor = torch.tensor(
        expected_depth, dtype=torch.float32, device=device
    )
    expected_range_tensor = torch.tensor(
        expected_range, dtype=torch.float32, device=device
    )
    expected_body_tensor = torch.tensor(
        expected_body, dtype=torch.int64, device=device
    )
    expected_origins = torch.tensor(
        cpu_rays.origins_m, dtype=torch.float32, device=device
    )
    expected_directions = torch.tensor(
        cpu_rays.directions, dtype=torch.float32, device=device
    )
    hit = body_index >= 0
    checks = {
        "cpu_cuda_camera_origins": torch.allclose(
            origins, expected_origins, atol=2.0e-6, rtol=0.0
        ),
        "cpu_cuda_camera_directions": torch.allclose(
            directions, expected_directions, atol=2.0e-6, rtol=0.0
        ),
        "cpu_cuda_hit_ids": torch.equal(body_index, expected_body_tensor),
        "cpu_cuda_optical_depth": torch.allclose(
            depth_z, expected_depth_tensor, atol=2.0e-5, rtol=0.0
        ),
        "cpu_cuda_hit_range": torch.allclose(
            hit_range, expected_range_tensor, atol=2.0e-5, rtol=0.0
        ),
        "miss_depth_and_range_zero": torch.equal(
            depth_z[~hit], torch.zeros_like(depth_z[~hit])
        )
        and torch.equal(hit_range[~hit], torch.zeros_like(hit_range[~hit])),
        "hit_normals_unit": torch.allclose(
            torch.linalg.vector_norm(normal[hit], dim=-1),
            torch.ones_like(depth_z[hit]),
            atol=2.0e-5,
            rtol=0.0,
        ),
        "body_attached_wrist_pose": torch.allclose(
            origins[1, 3], torch.tensor([2.0, 0.0, 0.0], device=device), atol=2.0e-6
        )
        and torch.allclose(
            directions[1, 3],
            torch.nn.functional.normalize(
                torch.tensor([1.0, -0.25, 0.25], device=device), dim=0
            ),
            atol=2.0e-6,
        ),
        "deterministic_replay": all(
            torch.equal(left, right) for left, right in zip(result, replay)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "CUDA depth-camera smoke failed: " + json.dumps(checks, sort_keys=True)
        )
    return {
        "schema_version": "box3d.depth-camera-smoke/v1",
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "extension": "factory_box3d_cuda_v14",
        "worlds": len(states),
        "cameras": len(camera_rows),
        "rays_per_world": len(pixel_camera),
        "ray_hits": int(hit.sum().item()),
        "checks": checks,
        "maximum_depth_error_m": float(
            (depth_z - expected_depth_tensor).abs().max().item()
        ),
        "maximum_range_error_m": float(
            (hit_range - expected_range_tensor).abs().max().item()
        ),
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
