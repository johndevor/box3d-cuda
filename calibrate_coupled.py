"""Bounded GPU sweep for Stage-7 solver parameters against a PhysX trace."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .contracts.coupling import PARITY_THRESHOLDS

from .benchmark_coupled import _config, _run, make_workload
from .extension import load_extension


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--iterations-only",
        action="store_true",
        help="sweep only velocity iterations to test maximal-coordinate convergence",
    )
    return parser.parse_args()


def _reference_tensors(torch, report: dict, device) -> list[dict]:
    samples = report["parity_trace"]["samples"]
    if report.get("backend") != "maniskill_physx_cuda" or len(samples) != 61:
        raise RuntimeError("calibration requires the measured Stage-7 PhysX trace")
    fields = (
        "joint_positions_rad", "joint_velocities_rad_s", "drive_efforts_nm",
        "body_positions_m", "body_quaternions_xyzw", "body_linear_velocities_mps",
        "pair_contact", "pair_contact_impulse_magnitude_ns",
    )
    return [
        {
            field: torch.as_tensor(sample[field], device=device)
            for field in fields
        }
        for sample in samples
    ]


def _metrics(torch, reference: list[dict], candidate: list[dict]) -> dict[str, float]:
    maximums = {
        "maximum_joint_position_error_rad": 0.0,
        "maximum_joint_velocity_error_rad_s": 0.0,
        "maximum_body_position_error_m": 0.0,
        "maximum_body_orientation_error_rad": 0.0,
        "maximum_body_velocity_error_mps": 0.0,
        "maximum_drive_effort_error_nm": 0.0,
        "maximum_pair_contact_impulse_magnitude_error_ns": 0.0,
    }
    matches = 0
    contact_samples = 0
    for expected, observed in zip(reference, candidate):
        current = {
            "joint_positions_rad": torch.as_tensor(observed["joint_positions_rad"], device=expected["joint_positions_rad"].device),
            "joint_velocities_rad_s": torch.as_tensor(observed["joint_velocities_rad_s"], device=expected["joint_positions_rad"].device),
            "drive_efforts_nm": torch.as_tensor(observed["drive_efforts_nm"], device=expected["joint_positions_rad"].device),
            "body_positions_m": torch.as_tensor(observed["body_positions_m"], device=expected["joint_positions_rad"].device),
            "body_quaternions_xyzw": torch.as_tensor(observed["body_quaternions_xyzw"], device=expected["joint_positions_rad"].device),
            "body_linear_velocities_mps": torch.as_tensor(observed["body_linear_velocities_mps"], device=expected["joint_positions_rad"].device),
            "pair_contact": torch.as_tensor(observed["pair_contact"], device=expected["joint_positions_rad"].device),
            "pair_contact_impulse_magnitude_ns": torch.as_tensor(observed["pair_contact_impulse_magnitude_ns"], device=expected["joint_positions_rad"].device),
        }
        maximums["maximum_joint_position_error_rad"] = max(
            maximums["maximum_joint_position_error_rad"],
            float((expected["joint_positions_rad"] - current["joint_positions_rad"]).abs().max().item()),
        )
        maximums["maximum_joint_velocity_error_rad_s"] = max(
            maximums["maximum_joint_velocity_error_rad_s"],
            float((expected["joint_velocities_rad_s"] - current["joint_velocities_rad_s"]).abs().max().item()),
        )
        maximums["maximum_drive_effort_error_nm"] = max(
            maximums["maximum_drive_effort_error_nm"],
            float((expected["drive_efforts_nm"] - current["drive_efforts_nm"]).abs().max().item()),
        )
        maximums["maximum_body_position_error_m"] = max(
            maximums["maximum_body_position_error_m"],
            float(torch.linalg.vector_norm(expected["body_positions_m"] - current["body_positions_m"], dim=-1).max().item()),
        )
        maximums["maximum_body_velocity_error_mps"] = max(
            maximums["maximum_body_velocity_error_mps"],
            float(torch.linalg.vector_norm(expected["body_linear_velocities_mps"] - current["body_linear_velocities_mps"], dim=-1).max().item()),
        )
        first = expected["body_quaternions_xyzw"]
        second = current["body_quaternions_xyzw"]
        cosine = torch.sum(first * second, dim=-1).abs() / (
            torch.linalg.vector_norm(first, dim=-1) * torch.linalg.vector_norm(second, dim=-1)
        ).clamp_min(1.0e-12)
        maximums["maximum_body_orientation_error_rad"] = max(
            maximums["maximum_body_orientation_error_rad"],
            float((2.0 * torch.acos(cosine.clamp(0.0, 1.0))).max().item()),
        )
        maximums["maximum_pair_contact_impulse_magnitude_error_ns"] = max(
            maximums["maximum_pair_contact_impulse_magnitude_error_ns"],
            float((expected["pair_contact_impulse_magnitude_ns"] - current["pair_contact_impulse_magnitude_ns"]).abs().max().item()),
        )
        matches += int((expected["pair_contact"] == current["pair_contact"]).sum().item())
        contact_samples += expected["pair_contact"].numel()
    expected_displacement = reference[-1]["body_positions_m"][:, 4, 0] - reference[0]["body_positions_m"][:, 4, 0]
    observed_first = torch.as_tensor(candidate[0]["body_positions_m"], device=expected_displacement.device)
    observed_last = torch.as_tensor(candidate[-1]["body_positions_m"], device=expected_displacement.device)
    maximums["maximum_payload_displacement_error_m"] = float(
        (expected_displacement - (observed_last[:, 4, 0] - observed_first[:, 4, 0])).abs().max().item()
    )
    maximums["contact_state_agreement_ratio"] = matches / contact_samples
    ratios = [
        maximums[key] / threshold
        for key, threshold in PARITY_THRESHOLDS.items()
        if key != "minimum_contact_state_agreement_ratio"
    ]
    contact_deficit = max(
        0.0,
        (PARITY_THRESHOLDS["minimum_contact_state_agreement_ratio"] - maximums["contact_state_agreement_ratio"])
        / (1.0 - PARITY_THRESHOLDS["minimum_contact_state_agreement_ratio"]),
    )
    ratios.append(contact_deficit)
    maximums["maximum_threshold_ratio"] = max(ratios)
    maximums["root_mean_square_threshold_ratio"] = math.sqrt(sum(value * value for value in ratios) / len(ratios))
    maximums["passed"] = maximums["maximum_threshold_ratio"] <= 1.0
    return maximums


def main() -> int:
    args = arguments()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for coupled calibration")
    report = json.loads(args.reference.read_text())
    load_extension()
    device = torch.device("cuda")
    reference = _reference_tensors(torch, report, device)
    base = _config()
    candidates = []
    if args.iterations_only:
        for solver_iterations in (4, 8, 12, 16, 24, 32, 48, 64):
            candidates.append((
                replace(
                    base,
                    joints=replace(
                        base.joints, solver_iterations=solver_iterations
                    ),
                    contacts=replace(
                        base.contacts, solver_iterations=solver_iterations
                    ),
                ),
                {"solver_iterations": solver_iterations},
            ))
    else:
        for warm_start_factor in (0.0, 0.2, 0.5, 0.8, 1.0):
            for position_correction in (0.2, 0.4, 0.6, 0.8, 1.0):
                for angular_damping in (0.0, 0.02, 0.1):
                    candidates.append((
                        replace(
                            base,
                            joints=replace(
                                base.joints,
                                warm_start_factor=warm_start_factor,
                                position_correction=position_correction,
                            ),
                            contacts=replace(
                                base.contacts,
                                position_correction=position_correction,
                                angular_damping=angular_damping,
                            ),
                        ),
                        {
                            "warm_start_factor": warm_start_factor,
                            "position_correction": position_correction,
                            "angular_damping": angular_damping,
                        },
                    ))

    rows = []
    for config, parameters in candidates:
        bundle = make_workload(64, device)
        _, diagnostics = _run(
            bundle, config, diagnostics=True, capture_parity=True
        )
        metrics = _metrics(torch, reference, diagnostics["parity_trace"])
        rows.append({
            "parameters": parameters,
            "physical_gates": {
                "maximum_penetration_m": float(diagnostics["max_penetration"].item()),
                "maximum_joint_anchor_error_m": float(diagnostics["max_anchor"].item()),
            },
            "parity": metrics,
        })
        print(json.dumps({
            "completed": len(rows),
            "parameters": parameters,
            "score": metrics["maximum_threshold_ratio"],
        }), flush=True)
    rows.sort(key=lambda row: (
        row["parity"]["maximum_threshold_ratio"],
        row["parity"]["root_mean_square_threshold_ratio"],
    ))
    result = {
        "schema_version": "factory-os.stage7-coupled-calibration/v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(device),
        "candidate_count": len(rows),
        "sweep": "solver_iterations" if args.iterations_only else "stabilization",
        "reference_backend": report["backend"],
        "reference_contract_id": report["contract_id"],
        "best": rows[0],
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"best": rows[0], "candidate_count": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
