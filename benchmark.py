from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmarking import BenchmarkResult, CapabilitySet, write_result

from .extension import load_extension
from .reference import SphereWorldConfig, assert_valid_state, make_drop_state, step_reference


CONTRACT_ID = "box3d.fixed-sphere-world/v0"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=4096)
    parser.add_argument("--bodies-per-world", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _tensorize(torch, nested, device):
    return torch.tensor(nested, dtype=torch.float32, device=device)


def main() -> int:
    args = arguments()
    if min(args.worlds, args.bodies_per_world, args.steps) <= 0:
        raise ValueError("worlds, bodies and steps must be positive")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required")
    module = load_extension()
    device = torch.device("cuda")
    config = SphereWorldConfig()

    # Correctness is checked on contact, freefall and rotation before timing.
    oracle_state, oracle_mass, oracle_radius = make_drop_state(2, min(4, args.bodies_per_world), seed=args.seed)
    oracle_state[0][0][1] = oracle_radius[0][0] - 0.01
    oracle_state[0][0][8] = -0.5
    oracle_state[0][0][10:13] = [0.3, -0.2, 0.1]
    expected = step_reference(oracle_state, oracle_mass, oracle_radius, config, steps=4)
    assert_valid_state(expected, oracle_radius)
    actual = _tensorize(torch, oracle_state, device)
    mass_tensor = _tensorize(torch, oracle_mass, device)
    radius_tensor = _tensorize(torch, oracle_radius, device)
    for _ in range(4):
        actual = module.step(
            actual,
            mass_tensor,
            radius_tensor,
            config.dt,
            config.substeps,
            config.gravity_y,
            config.restitution,
            config.friction,
        )
    torch.cuda.synchronize()
    expected_tensor = _tensorize(torch, expected, device)
    maximum_error = float((actual - expected_tensor).abs().max().item())
    finite = bool(torch.isfinite(actual).all().item())
    minimum_clearance = float((actual[:, :, 1] - radius_tensor).min().item())
    quaternion_norm_error = float(
        (torch.linalg.norm(actual[:, :, 3:7], dim=-1) - 1.0).abs().max().item()
    )
    correctness = {
        "passed": finite and maximum_error < 2.0e-4 and minimum_clearance >= -2.1e-4 and quaternion_norm_error < 2.0e-5,
        "finite": finite,
        "cpu_reference_maximum_absolute_error": maximum_error,
        "minimum_ground_clearance_m": minimum_clearance,
        "maximum_quaternion_norm_error": quaternion_norm_error,
    }
    if not correctness["passed"]:
        raise RuntimeError(f"CUDA port correctness gate failed: {json.dumps(correctness)}")

    source_state, source_mass, source_radius = make_drop_state(
        args.worlds, args.bodies_per_world, seed=args.seed
    )
    state = _tensorize(torch, source_state, device)
    inverse_mass = _tensorize(torch, source_mass, device)
    radius = _tensorize(torch, source_radius, device)
    call = lambda value: module.step(
        value,
        inverse_mass,
        radius,
        config.dt,
        config.substeps,
        config.gravity_y,
        config.restitution,
        config.friction,
    )
    for _ in range(args.warmup):
        state = call(state)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.steps):
        state = call(state)
    finish.record()
    torch.cuda.synchronize()
    duration = start.elapsed_time(finish) / 1000.0
    timed_finite = bool(torch.isfinite(state).all().item())
    if not timed_finite:
        raise RuntimeError("timed state contains non-finite values")

    result = BenchmarkResult(
        backend="box3d_cuda_stage0",
        backend_version="upstream-30c67b5+factory-v0",
        workload="fixed sphere worlds: integration + plane + pair contacts",
        contract_id=CONTRACT_ID,
        device=torch.cuda.get_device_name(device),
        worlds=args.worlds,
        bodies_per_world=args.bodies_per_world,
        steps=args.steps,
        duration_seconds=duration,
        capabilities=CapabilitySet(
            rigid_body_integration=True,
            static_plane_contacts=True,
            dynamic_contacts=True,
            articulated_joints=False,
            continuous_collision=False,
            ray_queries=False,
            camera_rendering=False,
            robot_manipulation=False,
        ),
        correctness={**correctness, "timed_state_finite": timed_finite},
        peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
        notes=(
            "One CUDA thread owns one fixed-small world; pair detection is O(bodies^2).",
            "This is not full Box3D and is not yet a robotics simulator.",
        ),
    )
    write_result(args.output, result)
    print(json.dumps(result.to_dict(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
