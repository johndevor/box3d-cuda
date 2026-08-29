from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .benchmarking import BenchmarkResult, CapabilitySet, write_result

from .extension import gripper_step, load_extension
from .gripper_reference import (
    CONTRACT_ID,
    GripperWorldConfig,
    finger_velocity,
    make_gripper_state,
    run_gripper_reference,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=4096)
    return parser.parse_args()


def _initial_tensors(torch, worlds: int, config: GripperWorldConfig):
    cube, fingers = make_gripper_state(worlds, config)
    return (
        torch.tensor(cube, dtype=torch.float32, device="cuda"),
        torch.tensor(fingers, dtype=torch.float32, device="cuda"),
    )


def _simulate(torch, cube, fingers, velocities, config: GripperWorldConfig):
    touched = torch.zeros(cube.shape[0], dtype=torch.bool, device=cube.device)
    bilateral = torch.zeros_like(touched)
    maximum_height = cube[:, 1].clone()
    release_height = cube[:, 1].clone()
    minimum_clearance = cube[:, 1].min() - config.cube_half_extent[1]
    for step in range(config.total_steps):
        cube, fingers, contacts = gripper_step(cube, fingers, velocities[step], config)
        active = contacts.bool()
        touched |= active.any(dim=1)
        bilateral |= active.all(dim=1)
        maximum_height = torch.maximum(maximum_height, cube[:, 1])
        minimum_clearance = torch.minimum(
            minimum_clearance, cube[:, 1].min() - config.cube_half_extent[1]
        )
        if step == config.release_step - 1:
            release_height = cube[:, 1].clone()
    lifted = maximum_height > config.cube_half_extent[1] + 0.06
    fell = cube[:, 1] < release_height - 0.04
    gates = {
        "finite": bool(torch.isfinite(cube).all().item()),
        "touched": bool(touched.all().item()),
        "bilateral_contact": bool(bilateral.all().item()),
        "lifted": bool(lifted.all().item()),
        "fell_after_release": bool(fell.all().item()),
        "minimum_maximum_height_m": float(maximum_height.min().item()),
        "minimum_release_height_m": float(release_height.min().item()),
        "maximum_final_height_m": float(cube[:, 1].max().item()),
        "minimum_ground_clearance_m": float(minimum_clearance.item()),
    }
    gates["passed"] = all(
        gates[name]
        for name in ("finite", "touched", "bilateral_contact", "lifted", "fell_after_release")
    ) and gates["minimum_ground_clearance_m"] >= -2.1e-4
    return cube, fingers, gates


def main() -> int:
    args = arguments()
    if args.worlds <= 0:
        raise ValueError("worlds must be positive")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required")
    load_extension()
    config = GripperWorldConfig()
    velocities = torch.tensor(
        [finger_velocity(step, config) for step in range(config.total_steps)],
        dtype=torch.float32,
        device="cuda",
    )

    oracle = run_gripper_reference(1, config)
    if not oracle["passed"]:
        raise RuntimeError(f"CPU gripper oracle failed: {oracle}")
    check_cube, check_fingers = _initial_tensors(torch, 1, config)
    check_cube, check_fingers, check = _simulate(
        torch, check_cube, check_fingers, velocities, config
    )
    expected_cube = torch.tensor(oracle["cube_state"], dtype=torch.float32, device="cuda")
    expected_fingers = torch.tensor(oracle["finger_positions"], dtype=torch.float32, device="cuda")
    check["cpu_reference_cube_maximum_absolute_error"] = float(
        (check_cube - expected_cube).abs().max().item()
    )
    check["cpu_reference_finger_maximum_absolute_error"] = float(
        (check_fingers - expected_fingers).abs().max().item()
    )
    check["passed"] = bool(check["passed"]) and check[
        "cpu_reference_cube_maximum_absolute_error"
    ] < 2.0e-3 and check["cpu_reference_finger_maximum_absolute_error"] < 2.0e-5
    if not check["passed"]:
        raise RuntimeError(f"CUDA gripper correctness gate failed: {json.dumps(check)}")

    # Discriminating negative control: with identical geometry and motion but
    # mu=0, side contacts may squeeze the cube but must not carry it upward.
    zero_friction = replace(config, friction=0.0)
    zero_velocities = torch.tensor(
        [finger_velocity(step, zero_friction) for step in range(zero_friction.total_steps)],
        dtype=torch.float32,
        device="cuda",
    )
    zero_cube, zero_fingers = _initial_tensors(torch, 1, zero_friction)
    _, _, zero_control = _simulate(
        torch, zero_cube, zero_fingers, zero_velocities, zero_friction
    )
    friction_control_passed = not zero_control["lifted"] and zero_control[
        "minimum_maximum_height_m"
    ] < 0.04
    if not friction_control_passed:
        raise RuntimeError(
            f"zero-friction negative control unexpectedly lifted: {json.dumps(zero_control)}"
        )

    # Run one complete close/lift/open/fall episode. The CUDA event encloses the
    # identical control-step sequence used by the ManiSkill workload.
    cube, fingers = _initial_tensors(torch, args.worlds, config)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    start.record()
    cube, fingers, timed = _simulate(torch, cube, fingers, velocities, config)
    finish.record()
    torch.cuda.synchronize()
    duration = start.elapsed_time(finish) / 1000.0
    if not timed["passed"]:
        raise RuntimeError(f"timed CUDA gripper correctness gate failed: {json.dumps(timed)}")

    correctness = {
        **timed,
        "cpu_oracle_passed": True,
        "cpu_reference_cube_maximum_absolute_error": check[
            "cpu_reference_cube_maximum_absolute_error"
        ],
        "cpu_reference_finger_maximum_absolute_error": check[
            "cpu_reference_finger_maximum_absolute_error"
        ],
        "no_attachment_state": True,
        "zero_friction_negative_control_passed": friction_control_passed,
        "zero_friction_maximum_height_m": zero_control["minimum_maximum_height_m"],
    }
    result = BenchmarkResult(
        backend="box3d_cuda_stage1",
        backend_version="upstream-30c67b5+factory-v1",
        workload="scripted parallel-jaw box contact, lift, release, and fall",
        contract_id=CONTRACT_ID,
        device=torch.cuda.get_device_name(),
        worlds=args.worlds,
        bodies_per_world=3,
        steps=config.total_steps,
        duration_seconds=duration,
        capabilities=CapabilitySet(
            rigid_body_integration=True,
            static_plane_contacts=True,
            dynamic_contacts=True,
            articulated_joints=False,
            continuous_collision=False,
            ray_queries=False,
            camera_rendering=False,
            robot_manipulation=True,
        ),
        correctness=correctness,
        peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
        notes=(
            "The cube is dynamic and the two fingers are kinematic AABBs.",
            "The lift is generated only by Coulomb contact friction; opening the fingers releases the cube.",
            "This stage does not yet support oriented boxes, angular contact response, joints, CCD, or broad phase.",
        ),
    )
    write_result(args.output, result)
    print(json.dumps(result.to_dict(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
