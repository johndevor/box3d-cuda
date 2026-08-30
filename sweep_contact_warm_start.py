"""Bounded, untimed Stage-7 contact warm-start sweep on CUDA.

This diagnostic varies only the fraction of the persistent contact impulse
cache replayed at the start of each substep. It replays the pre-registered
313-step impact window against a measured PhysX trace and emits compact
localization metrics. It cannot accept a speedup or production solver change.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .benchmark_coupled import (
    _config,
    _drive_effort_proxy,
    _pair_signed_separations,
    _step,
    _targets,
    make_workload,
)
from .contracts.coupling import (
    CUDA_BACKEND,
    PARITY_THRESHOLDS,
    PHYSICS_SUBSTEPS,
    PHYSX_BACKEND,
    SPEC,
)
from .contracts.impact import (
    IMPACT_SAMPLED_WORLDS,
    IMPACT_STEPS,
    diagnose_impact_reports,
    validated_impact_trace,
)
from .extension import load_extension


CONTACT_WARM_START_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)


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
                _maximum_absolute(
                    expected["joint_positions_rad"][world],
                    observed["joint_positions_rad"][world],
                ),
            )
            errors["joint_velocity_rad_s"] = max(
                errors["joint_velocity_rad_s"],
                _maximum_absolute(
                    expected["joint_velocities_rad_s"][world],
                    observed["joint_velocities_rad_s"][world],
                ),
            )
            errors["drive_effort_nm"] = max(
                errors["drive_effort_nm"],
                _maximum_absolute(
                    expected["drive_efforts_nm"][world],
                    observed["drive_efforts_nm"][world],
                ),
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


def _capture_candidate(torch, device, factor: float) -> tuple[dict[str, Any], dict[str, float]]:
    config = _config()
    bundle = make_workload(IMPACT_SAMPLED_WORLDS, device)
    pair_index = SPEC.contact_pair_roles.index("link2_payload")
    maximum_anchor = 0.0
    maximum_penetration = 0.0
    samples: list[dict[str, Any]] = []
    with torch.no_grad():
        for step in IMPACT_STEPS:
            target = _targets(bundle["initial_q"], step)
            result = _step(
                bundle,
                target,
                config,
                articulation_projection=True,
                contact_warm_start_factor=factor,
            )
            maximum_anchor = max(maximum_anchor, float(result[2].max().item()))
            maximum_penetration = max(maximum_penetration, float(result[9].max().item()))
            effort_proxy, joint_velocity = _drive_effort_proxy(bundle, result[1], target)
            samples.append({
                "control_step": step,
                "joint_positions_rad": result[1].detach().cpu().tolist(),
                "joint_velocities_rad_s": joint_velocity.detach().cpu().tolist(),
                "drive_efforts_nm": effort_proxy.detach().cpu().tolist(),
                "payload_position_m": bundle["state"][:, 4, :3].detach().cpu().tolist(),
                "payload_linear_velocity_mps": bundle["state"][:, 4, 7:10].detach().cpu().tolist(),
                "link2_position_m": bundle["state"][:, 3, :3].detach().cpu().tolist(),
                "link2_linear_velocity_mps": bundle["state"][:, 3, 7:10].detach().cpu().tolist(),
                "link2_payload_contact": (
                    _pair_signed_separations(bundle)[:, pair_index]
                    <= SPEC.contact_slop_m
                ).detach().cpu().tolist(),
                "link2_payload_impulse_magnitude_ns": result[13][
                    :, pair_index
                ].detach().cpu().tolist(),
            })
    torch.cuda.synchronize(device)
    return {
        "backend": CUDA_BACKEND,
        "impact_trace": {
            "sampled_worlds": IMPACT_SAMPLED_WORLDS,
            "sampled_world_indices": list(range(IMPACT_SAMPLED_WORLDS)),
            "sampled_steps": list(IMPACT_STEPS),
            "pair_role": "link2_payload",
            "samples": samples,
        },
    }, {
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
        raise RuntimeError("the contact warm-start sweep requires CUDA")
    reference = json.loads(args.reference.read_text())
    if reference.get("backend") != PHYSX_BACKEND:
        raise RuntimeError("the contact warm-start sweep requires a measured PhysX reference")
    validated_impact_trace(reference)
    load_extension()
    device = torch.device("cuda")
    candidates = []
    for factor in CONTACT_WARM_START_CANDIDATES:
        report, physical = _capture_candidate(torch, device, factor)
        diagnosis = diagnose_impact_reports([reference, report])
        errors = _overall_errors(reference, report)
        candidates.append({
            "contact_warm_start_factor": factor,
            "impact_diagnosis": diagnosis,
            "overall_impact_window_errors": errors,
            "physical_window_gates": physical,
        })
        print(json.dumps({
            "contact_warm_start_factor": factor,
            "maximum_impulse_relative_error": diagnosis[
                "maximum_cumulative_link2_payload_impulse_relative_error"
            ],
            "maximum_window_end_payload_position_error_m": diagnosis[
                "maximum_window_end_payload_position_error_m"
            ],
            "maximum_joint_position_error_rad": errors["joint_position_rad"],
        }), flush=True)
    result = {
        "schema_version": "factory-os.stage7-contact-warm-start-sweep/v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "authority": "diagnostic_only_no_speedup_or_solver_change_is_accepted",
        "device": torch.cuda.get_device_name(device),
        "fixed_parameters": {
            "physics_substeps": PHYSICS_SUBSTEPS,
            "joint_warm_start_factor": _config().joints.warm_start_factor,
            "solver_iterations": _config().joints.solver_iterations,
            "position_correction": _config().contacts.position_correction,
            "articulation_projection": True,
            "contact_generation_distance_m": _config().contacts.contact_generation_distance,
        },
        "strict_thresholds": dict(PARITY_THRESHOLDS),
        "contact_warm_start_candidates": list(CONTACT_WARM_START_CANDIDATES),
        "candidates": candidates,
        "accepted_contact_warm_start_factor": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
