from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmarking import BenchmarkResult, CapabilitySet, write_result

from .extension import load_extension, obb_step
from .obb_reference import (
    CONTRACT_ID,
    OrientedBoxConfig,
    assert_valid_oriented_boxes,
    make_oriented_box_state,
    step_oriented_box_reference,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=4096)
    parser.add_argument("--bodies-per-world", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def _tensorize(torch, value):
    return torch.tensor(value, dtype=torch.float32, device="cuda")


def _minimum_final_clearance(torch, state, half_extents):
    signs = torch.tensor(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=state.dtype,
        device=state.device,
    )
    local = half_extents.unsqueeze(2) * signs.view(1, 1, 8, 3)
    qv = state[:, :, None, 3:6]
    qw = state[:, :, None, 6:7]
    twice = 2.0 * torch.cross(qv.expand_as(local), local, dim=-1)
    rotated = local + qw * twice + torch.cross(qv.expand_as(local), twice, dim=-1)
    return (state[:, :, None, 1] + rotated[..., 1]).min()


def _run(torch, state, inverse_mass, half_extents, inverse_inertia, config, steps):
    touched = torch.zeros(state.shape[:2], dtype=torch.bool, device=state.device)
    maximum_angular_speed = torch.linalg.norm(state[:, :, 10:13], dim=-1)
    minimum_pre_correction = torch.full(
        (state.shape[0],), float("inf"), dtype=state.dtype, device=state.device
    )
    for _ in range(steps):
        state, contacts, clearance = obb_step(
            state, inverse_mass, half_extents, inverse_inertia, config
        )
        touched |= contacts.bool()
        maximum_angular_speed = torch.maximum(
            maximum_angular_speed, torch.linalg.norm(state[:, :, 10:13], dim=-1)
        )
        minimum_pre_correction = torch.minimum(minimum_pre_correction, clearance)
    return state, touched, maximum_angular_speed, minimum_pre_correction


def main() -> int:
    args = arguments()
    if min(args.worlds, args.bodies_per_world, args.steps) <= 0:
        raise ValueError("worlds, bodies and steps must be positive")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required")
    load_extension()
    config = OrientedBoxConfig()

    # Direct CPU/GPU parity across an off-center impact, including angular state.
    source, mass, half, inertia = make_oriented_box_state(1, 2, seed=args.seed)
    source[0][0][1] = 0.045
    source[0][0][7:10] = [0.0, -1.0, 0.0]
    source[0][0][10:13] = [0.0, 0.0, 0.0]
    expected, expected_contacts, _ = step_oriented_box_reference(
        source, mass, half, inertia, config, steps=8
    )
    assert_valid_oriented_boxes(expected, half)
    actual, actual_contacts, _, _ = _run(
        torch,
        _tensorize(torch, source),
        _tensorize(torch, mass),
        _tensorize(torch, half),
        _tensorize(torch, inertia),
        config,
        8,
    )
    torch.cuda.synchronize()
    expected_tensor = _tensorize(torch, expected)
    maximum_error = float((actual - expected_tensor).abs().max().item())
    contact_parity = actual_contacts.cpu().tolist() == expected_contacts
    if maximum_error >= 3.0e-3 or not contact_parity:
        raise RuntimeError(
            f"oriented-box CPU/CUDA parity failed: error={maximum_error}, contacts={contact_parity}"
        )

    source, mass, half, inertia = make_oriented_box_state(
        args.worlds, args.bodies_per_world, seed=args.seed
    )
    state = _tensorize(torch, source)
    inverse_mass = _tensorize(torch, mass)
    half_extents = _tensorize(torch, half)
    inverse_inertia = _tensorize(torch, inertia)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    start.record()
    state, touched, maximum_angular_speed, minimum_pre_correction = _run(
        torch, state, inverse_mass, half_extents, inverse_inertia, config, args.steps
    )
    finish.record()
    torch.cuda.synchronize()
    duration = start.elapsed_time(finish) / 1000.0
    finite = bool(torch.isfinite(state).all().item())
    quaternion_error = float(
        (torch.linalg.norm(state[:, :, 3:7], dim=-1) - 1.0).abs().max().item()
    )
    final_clearance = float(_minimum_final_clearance(torch, state, half_extents).item())
    all_contacted = bool(touched.all().item())
    angular_response = bool((maximum_angular_speed > 0.2).all().item())
    correctness = {
        "passed": finite and quaternion_error < 2.0e-5 and final_clearance >= -2.5e-3 and all_contacted and angular_response,
        "finite": finite,
        "cpu_reference_maximum_absolute_error": maximum_error,
        "cpu_contact_parity": contact_parity,
        "maximum_quaternion_norm_error": quaternion_error,
        "minimum_final_corner_clearance_m": final_clearance,
        "minimum_pre_correction_corner_clearance_m": float(minimum_pre_correction.min().item()),
        "all_boxes_contacted": all_contacted,
        "all_boxes_generated_angular_response": angular_response,
        "minimum_peak_angular_speed_rad_s": float(maximum_angular_speed.min().item()),
    }
    if not correctness["passed"]:
        raise RuntimeError(f"oriented-box correctness gate failed: {json.dumps(correctness)}")

    result = BenchmarkResult(
        backend="box3d_cuda_stage2",
        backend_version="upstream-30c67b5+factory-v2",
        workload="independent tumbling oriented boxes with angular plane contact",
        contract_id=CONTRACT_ID,
        device=torch.cuda.get_device_name(),
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
        correctness=correctness,
        peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
        notes=(
            "Each world contains separated oriented boxes; box/box collision is not part of this contract.",
            "Plane impulses include world-space inverse inertia, contact-point angular velocity, restitution and two-axis friction.",
            "One CUDA thread owns one world and loops over its fixed body set.",
        ),
    )
    write_result(args.output, result)
    print(json.dumps(result.to_dict(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
