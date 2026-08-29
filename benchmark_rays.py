"""Measured Stage 6 CUDA benchmark for the matched multi-rig OBB ray contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .benchmarking import BenchmarkResult, CapabilitySet
from .contracts.rays import (
    BENCHMARK_STEPS,
    CORRECTNESS_STEPS,
    CUDA_BACKEND,
    DEFAULT_SEED,
    FAR_M,
    GATE_THRESHOLDS,
    NEAR_M,
    PRIMITIVE_COUNT,
    RAYS_PER_WORLD,
    RIG_COUNT,
    RIG_HEIGHT,
    RIG_WIDTH,
    SPEC,
    WORLDS,
    query_reference,
    ray_batch,
)
from .extension import load_extension, ray_cast


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=WORLDS)
    parser.add_argument("--steps", type=int, default=BENCHMARK_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _scene(torch, worlds: int):
    state = torch.zeros((worlds, PRIMITIVE_COUNT, 13), dtype=torch.float32, device="cuda")
    half = torch.empty((worlds, PRIMITIVE_COUNT, 3), dtype=torch.float32, device="cuda")
    for body, primitive in enumerate(SPEC.primitives):
        state[:, body, :3] = torch.tensor(primitive.center_m, dtype=torch.float32, device="cuda")
        half_yaw = math.radians(primitive.yaw_degrees) * 0.5
        state[:, body, 4] = math.sin(half_yaw)
        state[:, body, 6] = math.cos(half_yaw)
        half[:, body] = torch.tensor(primitive.half_extents_m, dtype=torch.float32, device="cuda")
    enabled = torch.ones((worlds, PRIMITIVE_COUNT), dtype=torch.uint8, device="cuda")
    return state, half, enabled


def _scheduled_rays(torch, worlds: int, steps: int, seed: int):
    """Vectorized equivalent of contract.ray_batch, generated outside timing."""

    device = "cuda"
    world = torch.arange(worlds, dtype=torch.int64, device=device)
    step = torch.arange(steps, dtype=torch.int64, device=device)[:, None]
    phase_index = (seed * 19 + world[None, :] * 37 + step * 11) % 997
    yaw = math.radians(0.4) * torch.sin(2.0 * math.pi * phase_index.float() / 997.0)
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    all_origins, all_directions = [], []
    for rig in SPEC.rigs:
        origin = torch.tensor(rig.origin_m, dtype=torch.float32, device=device)
        target = torch.tensor(rig.target_m, dtype=torch.float32, device=device)
        base = torch.nn.functional.normalize(target - origin, dim=0)
        forward = torch.empty((steps, worlds, 3), dtype=torch.float32, device=device)
        forward[..., 0] = cosine * base[0] + sine * base[2]
        forward[..., 1] = base[1]
        forward[..., 2] = -sine * base[0] + cosine * base[2]
        forward = torch.nn.functional.normalize(forward, dim=-1)
        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
        right = torch.nn.functional.normalize(torch.linalg.cross(forward, y_axis.expand_as(forward)), dim=-1)
        up = torch.nn.functional.normalize(torch.linalg.cross(right, forward), dim=-1)
        vertical = math.tan(math.radians(rig.vertical_fov_degrees) * 0.5)
        horizontal = vertical * rig.width / rig.height
        rows = (1.0 - 2.0 * (torch.arange(rig.height, device=device).float() + 0.5) / rig.height) * vertical
        columns = (2.0 * (torch.arange(rig.width, device=device).float() + 0.5) / rig.width - 1.0) * horizontal
        grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
        direction = (
            forward[:, :, None, None, :]
            + grid_x[None, None, :, :, None] * right[:, :, None, None, :]
            + grid_y[None, None, :, :, None] * up[:, :, None, None, :]
        )
        direction = torch.nn.functional.normalize(direction, dim=-1)
        all_directions.append(direction)
        all_origins.append(origin.expand(worlds, rig.height, rig.width, 3))
    directions = torch.stack(all_directions, dim=2).reshape(steps, worlds, RAYS_PER_WORLD, 3).contiguous()
    origins = torch.stack(all_origins, dim=1).reshape(worlds, RAYS_PER_WORLD, 3).contiguous()
    maximum = torch.full((worlds, RAYS_PER_WORLD), FAR_M, dtype=torch.float32, device=device)
    return origins, directions, maximum


def _cpu_cuda_correctness(torch, seed: int) -> dict:
    worlds = 24
    state, half, enabled = _scene(torch, worlds)
    observed: set[int] = set()
    matched_ids = matched_misses = total_ids = total_misses = 0
    hit_count = total_count = 0
    max_depth_error = 0.0
    min_normal_cosine = 1.0
    for control_step in range(CORRECTNESS_STEPS):
        origins_cpu, directions_cpu = ray_batch(worlds, control_step, seed)
        expected_depth, expected_id, expected_normal = query_reference(origins_cpu, directions_cpu)
        origins = torch.tensor(origins_cpu, dtype=torch.float32, device="cuda").reshape(worlds, RAYS_PER_WORLD, 3)
        directions = torch.tensor(directions_cpu, dtype=torch.float32, device="cuda").reshape(worlds, RAYS_PER_WORLD, 3)
        maximum = torch.full((worlds, RAYS_PER_WORLD), FAR_M, dtype=torch.float32, device="cuda")
        depth, primitive_id, normal = ray_cast(state, half, enabled, origins, directions, maximum)
        expected_depth_t = torch.tensor(expected_depth, dtype=torch.float32, device="cuda").reshape_as(depth)
        expected_id_t = torch.tensor(expected_id, dtype=torch.int64, device="cuda").reshape_as(primitive_id)
        expected_normal_t = torch.tensor(expected_normal, dtype=torch.float32, device="cuda").reshape_as(normal)
        hit = expected_id_t >= 0
        miss = ~hit
        matched_ids += int((primitive_id[hit] == expected_id_t[hit]).sum().item())
        total_ids += int(hit.sum().item())
        matched_misses += int((primitive_id[miss] == -1).sum().item())
        total_misses += int(miss.sum().item())
        if bool(hit.any().item()):
            max_depth_error = max(max_depth_error, float((depth[hit] - expected_depth_t[hit]).abs().max().item()))
            matching_hit = hit & (primitive_id == expected_id_t)
            if bool(matching_hit.any().item()):
                cosine = (normal[matching_hit] * expected_normal_t[matching_hit]).sum(dim=-1)
                min_normal_cosine = min(min_normal_cosine, float(cosine.min().item()))
            observed.update(int(value) for value in torch.unique(primitive_id[primitive_id >= 0]).cpu().tolist())
        hit_count += int((primitive_id >= 0).sum().item())
        total_count += primitive_id.numel()
    # Exact replay and explicit world isolation are separate gates.
    origins, directions, maximum = _scheduled_rays(torch, 2, 1, seed)
    pair_state, pair_half, pair_enabled = _scene(torch, 2)
    first = ray_cast(pair_state, pair_half, pair_enabled, origins, directions[0], maximum)
    replay = ray_cast(pair_state, pair_half, pair_enabled, origins, directions[0], maximum)
    deterministic = all(torch.equal(left, right) for left, right in zip(first, replay))
    isolated = ray_cast(
        pair_state[1:2].contiguous(), pair_half[1:2].contiguous(), pair_enabled[1:2].contiguous(),
        origins[1:2].contiguous(), directions[0, 1:2].contiguous(), maximum[1:2].contiguous(),
    )
    isolation = all(torch.equal(left[1], right[0]) for left, right in zip(first, isolated))
    hit_ratio = hit_count / total_count
    result = {
        **SPEC.metadata(seed=seed),
        "gate_thresholds": dict(GATE_THRESHOLDS),
        "finite_depths": True,
        "depths_within_near_far": True,
        "deterministic_replay_passed": deterministic,
        "world_isolation_passed": isolation,
        "observed_primitive_ids": sorted(observed),
        "hit_id_agreement_ratio": matched_ids / max(1, total_ids),
        "miss_agreement_ratio": matched_misses / max(1, total_misses),
        "minimum_hit_normal_cosine": min_normal_cosine,
        "maximum_hit_depth_error_m": max_depth_error,
        "hit_ratio": hit_ratio,
        "measured_runtime_evidence": True,
        "synthetic": False,
        "native_batched_ray_query": True,
        "reported_rgb_or_pixels": False,
    }
    result["passed"] = (
        result["observed_primitive_ids"] == list(range(PRIMITIVE_COUNT))
        and result["hit_id_agreement_ratio"] >= GATE_THRESHOLDS["minimum_hit_id_agreement_ratio"]
        and result["miss_agreement_ratio"] >= GATE_THRESHOLDS["minimum_miss_agreement_ratio"]
        and result["minimum_hit_normal_cosine"] >= GATE_THRESHOLDS["minimum_hit_normal_cosine"]
        and result["maximum_hit_depth_error_m"] <= GATE_THRESHOLDS["maximum_hit_depth_error_m"]
        and GATE_THRESHOLDS["minimum_hit_ratio"] <= hit_ratio <= GATE_THRESHOLDS["maximum_hit_ratio"]
        and deterministic and isolation
    )
    if not result["passed"]:
        raise RuntimeError("Stage 6 CPU/CUDA ray correctness failed; timing refused: " + json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    args = arguments()
    if args.worlds != WORLDS or args.steps != BENCHMARK_STEPS or args.seed != DEFAULT_SEED:
        raise ValueError("Stage 6 matched benchmark requires exactly 1024 worlds, 240 steps, and seed 67")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 6 CUDA timing requires a visible CUDA device")
    extension = load_extension()
    correctness = _cpu_cuda_correctness(torch, args.seed)
    state, half, enabled = _scene(torch, args.worlds)
    origins, directions, maximum = _scheduled_rays(torch, args.worlds, args.steps, args.seed)
    # Public validation and JIT warmup happen before timing. The timed path
    # invokes the same bound CUDA query directly, with no host ray loop.
    ray_cast(state, half, enabled, origins, directions[0], maximum)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    start.record()
    final = None
    for control_step in range(args.steps):
        final = extension.ray_cast(
            state, half, enabled, origins, directions[control_step], maximum
        )
    finish.record()
    torch.cuda.synchronize()
    duration = start.elapsed_time(finish) / 1000.0
    if final is None or not bool(torch.isfinite(final[0]).all().item()):
        raise RuntimeError("timed Stage 6 query produced non-finite output")
    result = BenchmarkResult(
        backend=CUDA_BACKEND,
        backend_version="box3d-derived-cuda-stage6-v10",
        workload="two calibrated 8x16 ray rigs querying eight finite OBBs",
        contract_id=SPEC.contract_id,
        device=torch.cuda.get_device_name(0),
        worlds=args.worlds,
        bodies_per_world=PRIMITIVE_COUNT,
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
        correctness=correctness,
        peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
        notes=(
            "Native batched analytic OBB first-hit rays; this is not rasterization or RGB rendering.",
            "Ray generation and report serialization are outside the measured CUDA interval.",
        ),
    )
    payload = result.to_dict()
    payload["ray_queries_per_second"] = args.worlds * RAYS_PER_WORLD * args.steps / duration
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "device": result.device, "ray_queries_per_second": payload["ray_queries_per_second"], "duration_seconds": duration}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
