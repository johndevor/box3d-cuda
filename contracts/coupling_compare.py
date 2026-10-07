"""Pure Stage-7 CUDA versus PhysX sampled-output comparison.

No simulator is imported here.  This module consumes two already measured
reports, validates their bounded parity traces, computes matched physical
errors, and only then delegates speedup acceptance to the contract validator.
"""

from __future__ import annotations

import math
import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .coupling import (
    BENCHMARK_STEPS,
    BODY_COUNT,
    CONTRACT_ID,
    CUDA_BACKEND,
    JOINT_COUNT,
    MAX_QUATERNION_NORM_ERROR,
    PAIR_COUNT,
    PARITY_THRESHOLDS,
    PHYSX_BACKEND,
    validate_coupling_contract_speedup,
    validate_coupling_report,
)


SAMPLED_STEPS = tuple(range(0, BENCHMARK_STEPS, 12)) + (BENCHMARK_STEPS - 1,)
SAMPLED_WORLDS = 64
SAMPLED_WORLD_INDICES = tuple(range(SAMPLED_WORLDS))

TRACE_FIELDS = (
    "joint_positions_rad",
    "joint_velocities_rad_s",
    "joint_targets_rad",
    "drive_efforts_nm",
    "body_positions_m",
    "body_quaternions_xyzw",
    "body_linear_velocities_mps",
    "body_angular_velocities_rad_s",
    "pair_contact",
    "pair_contact_impulse_magnitude_ns",
)


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"parity trace requires numeric {path}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"parity trace requires finite {path}")
    return result


def _sequence(value: Any, length: int, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise RuntimeError(f"parity trace requires {path} shape length {length}")
    return value


def _numeric_matrix(value: Any, rows: int, columns: int, path: str) -> list[list[float]]:
    outer = _sequence(value, rows, path)
    return [
        [_finite_number(item, f"{path}[{row}][{column}]") for column, item in enumerate(_sequence(values, columns, f"{path}[{row}]"))]
        for row, values in enumerate(outer)
    ]


def _numeric_tensor3(
    value: Any, first: int, second: int, third: int, path: str
) -> list[list[list[float]]]:
    outer = _sequence(value, first, path)
    return [
        _numeric_matrix(values, second, third, f"{path}[{index}]")
        for index, values in enumerate(outer)
    ]


def _bool_matrix(value: Any, rows: int, columns: int, path: str) -> list[list[bool]]:
    outer = _sequence(value, rows, path)
    result: list[list[bool]] = []
    for row, values in enumerate(outer):
        current = _sequence(values, columns, f"{path}[{row}]")
        if any(type(item) is not bool for item in current):
            raise RuntimeError(f"parity trace requires boolean {path}[{row}]")
        result.append(list(current))
    return result


def _validated_trace(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    trace = report.get("parity_trace")
    if not isinstance(trace, Mapping):
        raise RuntimeError("measured coupling report is missing parity_trace")
    if trace.get("sampled_steps") != list(SAMPLED_STEPS):
        raise RuntimeError("parity_trace sampled_steps must be range(0,720,12)+[719]")
    if trace.get("sampled_worlds") != SAMPLED_WORLDS:
        raise RuntimeError("parity_trace sampled_worlds must equal 64")
    if trace.get("sampled_world_indices") != list(SAMPLED_WORLD_INDICES):
        raise RuntimeError("parity_trace must sample deterministic world indices 0 through 63")
    samples = _sequence(trace.get("samples"), len(SAMPLED_STEPS), "samples")
    validated: list[dict[str, Any]] = []
    for sample_index, (expected_step, sample) in enumerate(zip(SAMPLED_STEPS, samples)):
        if not isinstance(sample, Mapping) or sample.get("control_step") != expected_step:
            raise RuntimeError(f"parity trace sample {sample_index} has wrong control_step")
        missing = [field for field in TRACE_FIELDS if field not in sample]
        if missing:
            raise RuntimeError(f"parity trace sample {sample_index} missing {missing[0]}")
        parsed = {
            "control_step": expected_step,
            "joint_positions_rad": _numeric_matrix(
                sample["joint_positions_rad"], SAMPLED_WORLDS, JOINT_COUNT,
                f"samples[{sample_index}].joint_positions_rad",
            ),
            "joint_velocities_rad_s": _numeric_matrix(
                sample["joint_velocities_rad_s"], SAMPLED_WORLDS, JOINT_COUNT,
                f"samples[{sample_index}].joint_velocities_rad_s",
            ),
            "joint_targets_rad": _numeric_matrix(
                sample["joint_targets_rad"], SAMPLED_WORLDS, JOINT_COUNT,
                f"samples[{sample_index}].joint_targets_rad",
            ),
            "drive_efforts_nm": _numeric_matrix(
                sample["drive_efforts_nm"], SAMPLED_WORLDS, JOINT_COUNT,
                f"samples[{sample_index}].drive_efforts_nm",
            ),
            "body_positions_m": _numeric_tensor3(
                sample["body_positions_m"], SAMPLED_WORLDS, BODY_COUNT, 3,
                f"samples[{sample_index}].body_positions_m",
            ),
            "body_quaternions_xyzw": _numeric_tensor3(
                sample["body_quaternions_xyzw"], SAMPLED_WORLDS, BODY_COUNT, 4,
                f"samples[{sample_index}].body_quaternions_xyzw",
            ),
            "body_linear_velocities_mps": _numeric_tensor3(
                sample["body_linear_velocities_mps"], SAMPLED_WORLDS, BODY_COUNT, 3,
                f"samples[{sample_index}].body_linear_velocities_mps",
            ),
            "body_angular_velocities_rad_s": _numeric_tensor3(
                sample["body_angular_velocities_rad_s"], SAMPLED_WORLDS, BODY_COUNT, 3,
                f"samples[{sample_index}].body_angular_velocities_rad_s",
            ),
            "pair_contact": _bool_matrix(
                sample["pair_contact"], SAMPLED_WORLDS, PAIR_COUNT,
                f"samples[{sample_index}].pair_contact",
            ),
            "pair_contact_impulse_magnitude_ns": _numeric_matrix(
                sample["pair_contact_impulse_magnitude_ns"], SAMPLED_WORLDS, PAIR_COUNT,
                f"samples[{sample_index}].pair_contact_impulse_magnitude_ns",
            ),
        }
        for world in range(SAMPLED_WORLDS):
            for body in range(BODY_COUNT):
                quaternion = parsed["body_quaternions_xyzw"][world][body]
                norm = math.sqrt(sum(component * component for component in quaternion))
                if abs(norm - 1.0) > MAX_QUATERNION_NORM_ERROR:
                    raise RuntimeError(
                        f"parity trace quaternion norm exceeds contract at sample {sample_index}, world {world}, body {body}"
                    )
        validated.append(parsed)
    return validated


def _maximum_absolute(first: Sequence[float], second: Sequence[float]) -> float:
    return max(abs(a - b) for a, b in zip(first, second))


def _vector_error(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _quaternion_angular_error(first: Sequence[float], second: Sequence[float]) -> float:
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    cosine = abs(sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm))
    return 2.0 * math.acos(max(0.0, min(1.0, cosine)))


def compare_coupling_reports(
    reports: Sequence[Mapping[str, Any]], *, validate: bool = True
) -> dict[str, Any]:
    """Validate two measured reports, compare traces, and gate one speedup."""

    if len(reports) != 2 or {item.get("backend") for item in reports} != {PHYSX_BACKEND, CUDA_BACKEND}:
        raise RuntimeError("coupling comparison requires exactly one PhysX and one CUDA report")
    by_backend = {item["backend"]: item for item in reports}
    validate_coupling_report(by_backend[PHYSX_BACKEND], backend=PHYSX_BACKEND)
    validate_coupling_report(by_backend[CUDA_BACKEND], backend=CUDA_BACKEND)
    physx = _validated_trace(by_backend[PHYSX_BACKEND])
    cuda = _validated_trace(by_backend[CUDA_BACKEND])

    maximum_joint_position_error = 0.0
    maximum_joint_velocity_error = 0.0
    maximum_joint_target_error = 0.0
    maximum_drive_effort_error = 0.0
    maximum_body_position_error = 0.0
    maximum_body_orientation_error = 0.0
    maximum_body_velocity_error = 0.0
    maximum_body_angular_velocity_error = 0.0
    maximum_pair_impulse_error = 0.0
    matching_contacts = 0
    contact_samples = 0
    for physx_sample, cuda_sample in zip(physx, cuda):
        for world in range(SAMPLED_WORLDS):
            maximum_joint_position_error = max(
                maximum_joint_position_error,
                _maximum_absolute(
                    physx_sample["joint_positions_rad"][world],
                    cuda_sample["joint_positions_rad"][world],
                ),
            )
            maximum_joint_velocity_error = max(
                maximum_joint_velocity_error,
                _maximum_absolute(
                    physx_sample["joint_velocities_rad_s"][world],
                    cuda_sample["joint_velocities_rad_s"][world],
                ),
            )
            maximum_joint_target_error = max(
                maximum_joint_target_error,
                _maximum_absolute(
                    physx_sample["joint_targets_rad"][world],
                    cuda_sample["joint_targets_rad"][world],
                ),
            )
            maximum_drive_effort_error = max(
                maximum_drive_effort_error,
                _maximum_absolute(
                    physx_sample["drive_efforts_nm"][world],
                    cuda_sample["drive_efforts_nm"][world],
                ),
            )
            for body in range(BODY_COUNT):
                maximum_body_position_error = max(
                    maximum_body_position_error,
                    _vector_error(
                        physx_sample["body_positions_m"][world][body],
                        cuda_sample["body_positions_m"][world][body],
                    ),
                )
                maximum_body_orientation_error = max(
                    maximum_body_orientation_error,
                    _quaternion_angular_error(
                        physx_sample["body_quaternions_xyzw"][world][body],
                        cuda_sample["body_quaternions_xyzw"][world][body],
                    ),
                )
                maximum_body_velocity_error = max(
                    maximum_body_velocity_error,
                    _vector_error(
                        physx_sample["body_linear_velocities_mps"][world][body],
                        cuda_sample["body_linear_velocities_mps"][world][body],
                    ),
                )
                maximum_body_angular_velocity_error = max(
                    maximum_body_angular_velocity_error,
                    _vector_error(
                        physx_sample["body_angular_velocities_rad_s"][world][body],
                        cuda_sample["body_angular_velocities_rad_s"][world][body],
                    ),
                )
            for pair in range(PAIR_COUNT):
                contact_samples += 1
                matching_contacts += int(
                    physx_sample["pair_contact"][world][pair]
                    == cuda_sample["pair_contact"][world][pair]
                )
                maximum_pair_impulse_error = max(
                    maximum_pair_impulse_error,
                    abs(
                        physx_sample["pair_contact_impulse_magnitude_ns"][world][pair]
                        - cuda_sample["pair_contact_impulse_magnitude_ns"][world][pair]
                    ),
                )

    maximum_payload_displacement_error = 0.0
    for world in range(SAMPLED_WORLDS):
        physx_displacement = physx[-1]["body_positions_m"][world][4][0] - physx[0]["body_positions_m"][world][4][0]
        cuda_displacement = cuda[-1]["body_positions_m"][world][4][0] - cuda[0]["body_positions_m"][world][4][0]
        maximum_payload_displacement_error = max(
            maximum_payload_displacement_error,
            abs(physx_displacement - cuda_displacement),
        )

    parity = {
        "passed": False,
        "measured": True,
        "thresholds": dict(PARITY_THRESHOLDS),
        "sampled_steps": list(SAMPLED_STEPS),
        "sampled_worlds": SAMPLED_WORLDS,
        "contact_state_agreement_ratio": matching_contacts / contact_samples,
        "maximum_joint_position_error_rad": maximum_joint_position_error,
        "maximum_joint_velocity_error_rad_s": maximum_joint_velocity_error,
        "maximum_joint_target_error_rad": maximum_joint_target_error,
        "maximum_body_position_error_m": maximum_body_position_error,
        "maximum_body_orientation_error_rad": maximum_body_orientation_error,
        "maximum_body_velocity_error_mps": maximum_body_velocity_error,
        "maximum_body_angular_velocity_error_rad_s": maximum_body_angular_velocity_error,
        "maximum_drive_effort_error_nm": maximum_drive_effort_error,
        "maximum_pair_contact_impulse_magnitude_error_ns": maximum_pair_impulse_error,
        "maximum_payload_displacement_error_m": maximum_payload_displacement_error,
    }
    parity["passed"] = (
        parity["contact_state_agreement_ratio"]
        >= PARITY_THRESHOLDS["minimum_contact_state_agreement_ratio"]
        and all(
            parity[key] <= threshold
            for key, threshold in PARITY_THRESHOLDS.items()
            if key != "minimum_contact_state_agreement_ratio"
        )
    )
    physx_rate = _finite_number(by_backend[PHYSX_BACKEND].get("world_steps_per_second"), "PhysX world_steps_per_second")
    cuda_rate = _finite_number(by_backend[CUDA_BACKEND].get("world_steps_per_second"), "CUDA world_steps_per_second")
    if physx_rate <= 0.0 or cuda_rate <= 0.0:
        raise RuntimeError("coupling world-step rates must be positive")
    comparison = {
        "speedups": [{
            "contract_id": CONTRACT_ID,
            "baseline": PHYSX_BACKEND,
            "candidate": CUDA_BACKEND,
            "world_step_speedup": cuda_rate / physx_rate,
            "output_parity": parity,
        }]
    }
    if validate:
        validate_coupling_contract_speedup(reports, comparison)
    return comparison


__all__ = [
    "SAMPLED_STEPS", "SAMPLED_WORLDS", "SAMPLED_WORLD_INDICES",
    "TRACE_FIELDS", "compare_coupling_reports",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("physx", type=Path)
    parser.add_argument("cuda", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(args.physx.read_text()), json.loads(args.cuda.read_text())]
    result = compare_coupling_reports(reports, validate=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    try:
        validate_coupling_contract_speedup(reports, result)
    except RuntimeError as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
