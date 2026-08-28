"""Correctness-gated CPU/CUDA benchmark for oriented box-pair SAT."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from factory_os.benchmarks import BenchmarkResult, CapabilitySet, write_result

from .extension import _validate_sat_pair_indices, load_extension, sat_step
from .sat_reference import (
    CONTRACT_ID,
    SATConfig,
    assert_valid_sat_boxes,
    make_sat_box_state,
    step_sat_reference,
)


ORACLE_STEPS = 12
CPU_CUDA_STATE_TOLERANCE = 5.0e-3
CPU_CUDA_PENETRATION_TOLERANCE_M = 5.0e-3
MAX_FINAL_PENETRATION_M = 5.0e-3
ANGULAR_RESPONSE_THRESHOLD_RAD_S = 5.0e-2


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def _angular_speed(body) -> float:
    return math.sqrt(sum(value * value for value in body[10:13]))


def run_cpu_correctness_gate(seed: int = 23):
    """Run the independent oracle gate before CUDA compilation or timing.

    The returned nested arrays are the exact parity input/expected output.  A
    caller cannot reach GPU timing through this benchmark when this gate fails.
    """

    source, inverse_mass, half_extents, inverse_inertia, pair_indices = make_sat_box_state(
        2, seed=seed
    )
    config = SATConfig()
    started = time.perf_counter()
    expected, expected_contacts, expected_penetration = step_sat_reference(
        source,
        inverse_mass,
        half_extents,
        inverse_inertia,
        pair_indices,
        config,
        steps=ORACLE_STEPS,
    )
    cpu_duration = time.perf_counter() - started
    assert_valid_sat_boxes(
        expected,
        inverse_mass,
        half_extents,
        inverse_inertia,
        pair_indices,
        max_penetration=MAX_FINAL_PENETRATION_M,
    )
    all_pairs_contacted = all(all(world) for world in expected_contacts)
    quaternion_error = max(
        abs(math.sqrt(sum(value * value for value in body[3:7])) - 1.0)
        for world in expected
        for body in world
    )
    pair_peak_angular_speed = [
        min(
            max(_angular_speed(expected[world][body]) for body in pair)
            for world in range(len(expected))
        )
        for pair in pair_indices
    ]
    rotated_edge_response = all(
        pair_peak_angular_speed[index] > ANGULAR_RESPONSE_THRESHOLD_RAD_S
        for index in (1, 2)
    )
    maximum_penetration = max(max(world) for world in expected_penetration)
    gate = {
        "passed": (
            all_pairs_contacted
            and quaternion_error < 2.0e-5
            and rotated_edge_response
            and maximum_penetration <= MAX_FINAL_PENETRATION_M
        ),
        "cpu_reference_duration_seconds": cpu_duration,
        "cpu_all_pairs_contacted": all_pairs_contacted,
        "cpu_maximum_quaternion_norm_error": quaternion_error,
        "cpu_pair_peak_angular_speed_rad_s": pair_peak_angular_speed,
        "cpu_rotated_and_edge_angular_response": rotated_edge_response,
        "cpu_maximum_final_sat_penetration_m": maximum_penetration,
    }
    if not gate["passed"]:
        raise RuntimeError(
            "CPU SAT oracle correctness failed; CUDA compilation and timing refused: "
            + json.dumps(gate, sort_keys=True, allow_nan=False)
        )
    return (
        source,
        inverse_mass,
        half_extents,
        inverse_inertia,
        pair_indices,
        expected,
        expected_contacts,
        expected_penetration,
        config,
        gate,
    )


def _float_tensor(torch, value):
    return torch.tensor(value, dtype=torch.float32, device="cuda")


def _pair_tensor(torch, value):
    return torch.tensor(value, dtype=torch.int64, device="cuda")


def _run(torch, state, inverse_mass, half_extents, inverse_inertia, pair_indices, config, steps):
    touched = torch.zeros(
        (state.shape[0], pair_indices.shape[0]), dtype=torch.bool, device=state.device
    )
    peak_angular = torch.linalg.norm(state[:, :, 10:13], dim=-1)
    penetration = torch.zeros_like(touched, dtype=state.dtype)
    for _ in range(steps):
        state, contacts, penetration = sat_step(
            state,
            inverse_mass,
            half_extents,
            inverse_inertia,
            pair_indices,
            config,
        )
        touched |= contacts.bool()
        peak_angular = torch.maximum(
            peak_angular, torch.linalg.norm(state[:, :, 10:13], dim=-1)
        )
    return state, touched, penetration, peak_angular


def main() -> int:
    args = arguments()
    if isinstance(args.seed, bool) or args.seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if args.worlds <= 0 or args.steps <= 0 or args.warmup < 0:
        raise ValueError("worlds and steps must be positive; warmup cannot be negative")

    # Deliberately precedes torch/CUDA setup and every GPU timing event.
    (
        oracle_source,
        oracle_mass,
        oracle_half,
        oracle_inertia,
        oracle_pairs,
        expected,
        expected_contacts,
        expected_penetration,
        config,
        cpu_gate,
    ) = run_cpu_correctness_gate(args.seed)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "CPU SAT correctness passed, but CUDA timing requires a CUDA-enabled PyTorch installation"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CPU SAT correctness passed, but CUDA timing requires a visible CUDA device"
        )
    load_extension()
    state = _float_tensor(torch, oracle_source)
    inverse_mass = _float_tensor(torch, oracle_mass)
    half_extents = _float_tensor(torch, oracle_half)
    inverse_inertia = _float_tensor(torch, oracle_inertia)
    pair_indices = _pair_tensor(torch, oracle_pairs)
    actual, actual_contacts, actual_penetration, _ = _run(
        torch,
        state,
        inverse_mass,
        half_extents,
        inverse_inertia,
        pair_indices,
        config,
        ORACLE_STEPS,
    )
    torch.cuda.synchronize()
    expected_tensor = _float_tensor(torch, expected)
    expected_penetration_tensor = _float_tensor(torch, expected_penetration)
    maximum_error = float((actual - expected_tensor).abs().max().item())
    penetration_error = float(
        (actual_penetration - expected_penetration_tensor).abs().max().item()
    )
    contact_parity = actual_contacts.cpu().tolist() == expected_contacts
    cuda_parity = (
        bool(torch.isfinite(actual).all().item())
        and maximum_error <= CPU_CUDA_STATE_TOLERANCE
        and penetration_error <= CPU_CUDA_PENETRATION_TOLERANCE_M
        and contact_parity
    )
    if not cuda_parity:
        raise RuntimeError(
            "CPU/CUDA SAT parity failed; timing refused: "
            + json.dumps(
                {
                    "state_maximum_absolute_error": maximum_error,
                    "penetration_maximum_absolute_error_m": penetration_error,
                    "contact_parity": contact_parity,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )

    # Warm the compiled path on the small oracle workload, then create a fresh
    # state so warmup collisions cannot satisfy the timed contact gate.
    warm_state = _float_tensor(torch, oracle_source)
    for _ in range(args.warmup):
        warm_state, _, _ = sat_step(
            warm_state,
            inverse_mass,
            half_extents,
            inverse_inertia,
            pair_indices,
            config,
        )
    torch.cuda.synchronize()

    source, mass, half, inertia, pairs = make_sat_box_state(args.worlds, seed=args.seed)
    state = _float_tensor(torch, source)
    inverse_mass = _float_tensor(torch, mass)
    half_extents = _float_tensor(torch, half)
    inverse_inertia = _float_tensor(torch, inertia)
    pair_indices = _pair_tensor(torch, pairs)
    # Pair-table host validation is cached by tensor version. Perform its one
    # synchronization before the start event so validation is never counted as
    # kernel time.
    _validate_sat_pair_indices(pair_indices, int(state.shape[1]))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    start.record()
    state, touched, penetration, peak_angular = _run(
        torch,
        state,
        inverse_mass,
        half_extents,
        inverse_inertia,
        pair_indices,
        config,
        args.steps,
    )
    finish.record()
    torch.cuda.synchronize()
    duration = start.elapsed_time(finish) / 1000.0

    finite = bool(torch.isfinite(state).all().item())
    quaternion_error = float(
        (torch.linalg.norm(state[:, :, 3:7], dim=-1) - 1.0).abs().max().item()
    )
    all_pairs_contacted = bool(touched.all().item())
    maximum_penetration = float(penetration.max().item())
    pair_peak_angular = []
    for pair in pairs:
        indices = torch.tensor(pair, dtype=torch.int64, device=state.device)
        pair_peak_angular.append(float(peak_angular.index_select(1, indices).max(dim=1).values.min().item()))
    rotated_edge_response = all(
        pair_peak_angular[index] > ANGULAR_RESPONSE_THRESHOLD_RAD_S
        for index in (1, 2)
    )
    timed_gate = (
        finite
        and quaternion_error < 2.0e-5
        and all_pairs_contacted
        and rotated_edge_response
        and maximum_penetration <= MAX_FINAL_PENETRATION_M
    )
    correctness = {
        **cpu_gate,
        "passed": cpu_gate["passed"] and cuda_parity and timed_gate,
        "cpu_cuda_state_maximum_absolute_error": maximum_error,
        "cpu_cuda_penetration_maximum_absolute_error_m": penetration_error,
        "cpu_cuda_contact_parity": contact_parity,
        "timed_state_finite": finite,
        "timed_all_pairs_contacted": all_pairs_contacted,
        "timed_maximum_quaternion_norm_error": quaternion_error,
        "timed_maximum_final_sat_penetration_m": maximum_penetration,
        "timed_pair_peak_angular_speed_rad_s": pair_peak_angular,
        "timed_rotated_and_edge_angular_response": rotated_edge_response,
        "sat_axis_candidates": 15,
        "degenerate_cross_axes_skipped": True,
        "scenario_seed": args.seed,
        "control_hz": 120,
        "physics_substeps": config.substeps,
        "pair_order": [list(pair) for pair in pairs],
    }
    if not correctness["passed"]:
        raise RuntimeError(
            "timed SAT correctness gate failed: "
            + json.dumps(correctness, sort_keys=True, allow_nan=False)
        )

    result = BenchmarkResult(
        backend="box3d_cuda_stage3",
        backend_version="upstream-30c67b5+factory-v3",
        workload="three explicit OBB pairs per fixed-small world: face, rotated-face, edge",
        contract_id=CONTRACT_ID,
        device=torch.cuda.get_device_name(),
        worlds=args.worlds,
        bodies_per_world=6,
        steps=args.steps,
        duration_seconds=duration,
        capabilities=CapabilitySet(
            rigid_body_integration=True,
            static_plane_contacts=False,
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
            "Every listed OBB pair evaluates 3+3 face axes and 9 edge cross axes; degenerate cross axes are skipped by epsilon.",
            "Normal and friction impulses include contact-point velocity and local-diagonal inverse inertia transformed to world space.",
            "One CUDA thread owns one world and loops over three explicit pairs; this remains a fixed-small-world narrow phase.",
            "CPU oracle correctness and CPU/CUDA parity complete before GPU timing begins.",
            "This benchmark does not establish broad phase, stacking, CCD, joints, robot manipulation, or general Box3D parity.",
        ),
    )
    write_result(args.output, result)
    print(json.dumps(result.to_dict(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
