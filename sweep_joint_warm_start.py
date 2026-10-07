"""Bounded, untimed Stage-7 joint warm-start sweep on CUDA.

This diagnostic varies only the persistent joint-row cache replay factor. It
uses the measured PhysX impact window and cannot accept a speedup or production
solver change.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .benchmark_coupled import _config
from .contracts.coupling import PARITY_THRESHOLDS, PHYSICS_SUBSTEPS, PHYSX_BACKEND
from .contracts.impact import diagnose_impact_reports, validated_impact_trace
from .extension import load_extension
from .sweep_contact_warm_start import _capture_candidate, _overall_errors


JOINT_WARM_START_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 0.8, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the joint warm-start sweep requires CUDA")
    reference = json.loads(args.reference.read_text())
    if reference.get("backend") != PHYSX_BACKEND:
        raise RuntimeError("the joint warm-start sweep requires measured PhysX")
    validated_impact_trace(reference)
    load_extension()
    device = torch.device("cuda")
    candidates = []
    for factor in JOINT_WARM_START_CANDIDATES:
        report, physical = _capture_candidate(
            torch, device, joint_factor=factor
        )
        diagnosis = diagnose_impact_reports([reference, report])
        errors = _overall_errors(reference, report)
        candidates.append({
            "joint_warm_start_factor": factor,
            "impact_diagnosis": diagnosis,
            "overall_impact_window_errors": errors,
            "physical_window_gates": physical,
        })
        print(json.dumps({
            "joint_warm_start_factor": factor,
            "maximum_joint_position_error_rad": errors["joint_position_rad"],
            "maximum_joint_velocity_error_rad_s": errors["joint_velocity_rad_s"],
            "maximum_drive_effort_error_nm": errors["drive_effort_nm"],
            "maximum_impulse_relative_error": diagnosis[
                "maximum_cumulative_link2_payload_impulse_relative_error"
            ],
        }), flush=True)
    result = {
        "schema_version": "factory-os.stage7-joint-warm-start-sweep/v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "authority": "diagnostic_only_no_speedup_or_solver_change_is_accepted",
        "device": torch.cuda.get_device_name(device),
        "fixed_parameters": {
            "physics_substeps": PHYSICS_SUBSTEPS,
            "contact_warm_start_factor": 1.0,
            "solver_iterations": _config().joints.solver_iterations,
            "articulation_projection": True,
        },
        "strict_thresholds": dict(PARITY_THRESHOLDS),
        "joint_warm_start_candidates": list(JOINT_WARM_START_CANDIDATES),
        "candidates": candidates,
        "accepted_joint_warm_start_factor": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
