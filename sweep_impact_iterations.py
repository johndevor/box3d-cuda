"""Bounded, untimed Stage-7 contact-response sweep on CUDA.

This diagnostic varies only the shared joint/contact iteration count.  It
replays the pre-registered 313-step impact window against a measured PhysX
trace and emits compact localization metrics.  It cannot accept a speedup or
change the production solver by itself.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from factory_os.coupling.contract import (
    CUDA_BACKEND,
    PARITY_THRESHOLDS,
    PHYSICS_SUBSTEPS,
    PHYSX_BACKEND,
    SPEC,
)
from factory_os.coupling.impact import (
    IMPACT_SAMPLED_WORLDS,
    IMPACT_STEPS,
    diagnose_impact_reports,
    validated_impact_trace,
)

from .benchmark_coupled import _config, _pair_signed_separations, _step, _targets, make_workload
from .extension import load_extension


ITERATION_CANDIDATES = (4, 8, 12, 16, 24, 32)


def _norm(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))


def _maximum_absolute(first: Sequence[float], second: Sequence[float]) -> float:
    return max(abs(left - right) for left, right in zip(first, second))


def _overall_errors(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    physx = validated_impact_trace(reference)
    cuda = validated_impact_trace(candidate)
    errors = {
        "joint_position_rad": 0.0,
        "joint_velocity_rad_s": 0.0,
        "drive_effort_nm": 0.0,
        "link2_position_m": 0.0,
        "link2_velocity_mps": 0.0,
        "payload_position_m": 0.0,
        "payload_velocity_mps": 0.0,
    }
    matches = 0
    samples = 0
    for expected, observed in zip(physx, cuda):
        for world in range(IMPACT_SAMPLED_WORLDS):
            errors["joint_position_rad"] = max(
                errors["joint_position_rad"],
                _maximum_absolute(expected["joint_positions_rad"][world], observed["joint_positions_rad"][world]),
            )
            errors["joint_velocity_rad_s"] = max(
                errors["joint_velocity_rad_s"],
                _maximum_absolute(expected["joint_velocities_rad_s"][world], observed["joint_velocities_rad_s"][world]),
            )
            errors["drive_effort_nm"] = max(
                errors["drive_effort_nm"],
                _maximum_absolute(expected["drive_efforts_nm"][world], observed["drive_efforts_nm"][world]),
            )
            for name, field in (
                ("link2_position_m", "link2_position_m"),
                ("link2_velocity_mps", "link2_linear_velocity_mps"),
                ("payload_position_m", "payload_position_m"),
                ("payload_velocity_mps", "payload_linear_velocity_mps"),
            ):
                errors[name] = max(
                    errors[name], _norm(expected[field][world], observed[field][world])
                )
            matches += int(
                expected["link2_payload_contact"][world]
                == observed["link2_payload_contact"][world]
            )
            samples += 1
    errors["link2_payload_contact_agreement_ratio"] = matches / samples
    return errors


def _capture_candidate(torch, device, solver_iterations: int) -> tuple[dict[str, Any], dict[str, float]]:
    base = _config()
    config = replace(
        base,
        joints=replace(base.joints, solver_iterations=solver_iterations),
        contacts=replace(base.contacts, solver_iterations=solver_iterations),
    )
    bundle = make_workload(IMPACT_SAMPLED_WORLDS, device)
    pair_index = SPEC.contact_pair_roles.index("link2_payload")
    parents = torch.tensor(SPEC.joint_parent_body_indices, dtype=torch.int64, device=device)
    children = torch.tensor(SPEC.joint_child_body_indices, dtype=torch.int64, device=device)
    maximum_anchor = 0.0
    maximum_penetration = 0.0
    samples: list[dict[str, Any]] = []
    with torch.no_grad():
        for step in IMPACT_STEPS:
            result = _step(bundle, _targets(bundle["initial_q"], step), config)
            maximum_anchor = max(maximum_anchor, float(result[2].max().item()))
            maximum_penetration = max(maximum_penetration, float(result[9].max().item()))
            joint_velocity = bundle["state"][:, children, 12] - bundle["state"][:, parents, 12]
            samples.append({
                "control_step": step,
                "joint_positions_rad": result[1].detach().cpu().tolist(),
                "joint_velocities_rad_s": joint_velocity.detach().cpu().tolist(),
                "drive_efforts_nm": (result[5] * 120).detach().cpu().tolist(),
                "payload_position_m": bundle["state"][:, 4, :3].detach().cpu().tolist(),
                "payload_linear_velocity_mps": bundle["state"][:, 4, 7:10].detach().cpu().tolist(),
                "link2_position_m": bundle["state"][:, 3, :3].detach().cpu().tolist(),
                "link2_linear_velocity_mps": bundle["state"][:, 3, 7:10].detach().cpu().tolist(),
                "link2_payload_contact": (
                    _pair_signed_separations(bundle)[:, pair_index] <= SPEC.contact_slop_m
                ).detach().cpu().tolist(),
                "link2_payload_impulse_magnitude_ns": result[13][:, pair_index].detach().cpu().tolist(),
            })
    torch.cuda.synchronize(device)
    report = {
        "backend": CUDA_BACKEND,
        "impact_trace": {
            "sampled_worlds": IMPACT_SAMPLED_WORLDS,
            "sampled_world_indices": list(range(IMPACT_SAMPLED_WORLDS)),
            "sampled_steps": list(IMPACT_STEPS),
            "pair_role": "link2_payload",
            "samples": samples,
        },
    }
    return report, {
        "maximum_joint_anchor_error_m_in_window": maximum_anchor,
        "maximum_penetration_m_in_window": maximum_penetration,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the impact iteration sweep requires CUDA")
    reference = json.loads(args.reference.read_text())
    if reference.get("backend") != PHYSX_BACKEND:
        raise RuntimeError("the impact iteration sweep requires a measured PhysX reference")
    validated_impact_trace(reference)
    load_extension()
    device = torch.device("cuda")
    candidates = []
    for solver_iterations in ITERATION_CANDIDATES:
        report, physical = _capture_candidate(torch, device, solver_iterations)
        diagnosis = diagnose_impact_reports([reference, report])
        errors = _overall_errors(reference, report)
        candidates.append({
            "solver_iterations": solver_iterations,
            "impact_diagnosis": diagnosis,
            "overall_impact_window_errors": errors,
            "physical_window_gates": physical,
        })
        print(json.dumps({
            "solver_iterations": solver_iterations,
            "maximum_impulse_relative_error": diagnosis["maximum_cumulative_link2_payload_impulse_relative_error"],
            "maximum_window_end_payload_position_error_m": diagnosis["maximum_window_end_payload_position_error_m"],
            "maximum_joint_position_error_rad": errors["joint_position_rad"],
        }), flush=True)
    result = {
        "schema_version": "factory-os.stage7-impact-iteration-sweep/v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "authority": "diagnostic_only_no_speedup_or_solver_change_is_accepted",
        "device": torch.cuda.get_device_name(device),
        "fixed_parameters": {
            "physics_substeps": PHYSICS_SUBSTEPS,
            "warm_start_factor": _config().joints.warm_start_factor,
            "position_correction": _config().contacts.position_correction,
        },
        "strict_thresholds": dict(PARITY_THRESHOLDS),
        "iteration_candidates": list(ITERATION_CANDIDATES),
        "candidates": candidates,
        "accepted_solver_iterations": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
