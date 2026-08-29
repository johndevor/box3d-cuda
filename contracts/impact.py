"""Dense, untimed impact diagnostics for the Stage-7 matched workload.

The strict sampled-output contract in :mod:`box3d_cuda.contracts.coupling_compare`
remains the only speedup gate.  This module adds a dense trace around the
first link-2/payload collision so a discontinuous impact can be distinguished
from a sustained solver-state disagreement without weakening that gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .coupling import CUDA_BACKEND, JOINT_COUNT, PARITY_THRESHOLDS, PHYSX_BACKEND


IMPACT_START_STEP = 0
IMPACT_END_STEP = 312
IMPACT_STEPS = tuple(range(IMPACT_START_STEP, IMPACT_END_STEP + 1))
IMPACT_SAMPLED_WORLDS = 64
MAX_PHASE_SHIFT_STEPS = 3

IMPACT_FIELDS = (
    "joint_positions_rad",
    "joint_velocities_rad_s",
    "drive_efforts_nm",
    "payload_position_m",
    "payload_linear_velocity_mps",
    "link2_position_m",
    "link2_linear_velocity_mps",
    "link2_payload_contact",
    "link2_payload_impulse_magnitude_ns",
)


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"impact trace requires numeric {path}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"impact trace requires finite {path}")
    return result


def _sequence(value: Any, length: int, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise RuntimeError(f"impact trace requires {path} shape length {length}")
    return value


def _matrix(value: Any, columns: int, path: str) -> list[list[float]]:
    rows = _sequence(value, IMPACT_SAMPLED_WORLDS, path)
    return [
        [_number(item, f"{path}[{row}][{column}]") for column, item in enumerate(_sequence(values, columns, f"{path}[{row}]"))]
        for row, values in enumerate(rows)
    ]


def _vector(value: Any, path: str) -> list[float]:
    return [
        _number(item, f"{path}[{index}]")
        for index, item in enumerate(_sequence(value, IMPACT_SAMPLED_WORLDS, path))
    ]


def _booleans(value: Any, path: str) -> list[bool]:
    values = _sequence(value, IMPACT_SAMPLED_WORLDS, path)
    if any(type(item) is not bool for item in values):
        raise RuntimeError(f"impact trace requires boolean {path}")
    return list(values)


def validated_impact_trace(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    trace = report.get("impact_trace")
    if not isinstance(trace, Mapping):
        raise RuntimeError("measured coupling report is missing impact_trace")
    if trace.get("sampled_steps") != list(IMPACT_STEPS):
        raise RuntimeError("impact_trace sampled_steps do not match the pre-registered dense window")
    if trace.get("sampled_worlds") != IMPACT_SAMPLED_WORLDS:
        raise RuntimeError("impact_trace sampled_worlds must equal 64")
    if trace.get("sampled_world_indices") != list(range(IMPACT_SAMPLED_WORLDS)):
        raise RuntimeError("impact_trace must sample deterministic world indices 0 through 63")
    if trace.get("pair_role") != "link2_payload":
        raise RuntimeError("impact_trace must diagnose the link2_payload pair")
    samples = _sequence(trace.get("samples"), len(IMPACT_STEPS), "impact samples")
    parsed: list[dict[str, Any]] = []
    for sample_index, (step, sample) in enumerate(zip(IMPACT_STEPS, samples)):
        if not isinstance(sample, Mapping) or sample.get("control_step") != step:
            raise RuntimeError(f"impact trace sample {sample_index} has wrong control_step")
        missing = [field for field in IMPACT_FIELDS if field not in sample]
        if missing:
            raise RuntimeError(f"impact trace sample {sample_index} missing {missing[0]}")
        parsed.append({
            "control_step": step,
            "joint_positions_rad": _matrix(sample["joint_positions_rad"], JOINT_COUNT, f"samples[{sample_index}].joint_positions_rad"),
            "joint_velocities_rad_s": _matrix(sample["joint_velocities_rad_s"], JOINT_COUNT, f"samples[{sample_index}].joint_velocities_rad_s"),
            "drive_efforts_nm": _matrix(sample["drive_efforts_nm"], JOINT_COUNT, f"samples[{sample_index}].drive_efforts_nm"),
            "payload_position_m": _matrix(sample["payload_position_m"], 3, f"samples[{sample_index}].payload_position_m"),
            "payload_linear_velocity_mps": _matrix(sample["payload_linear_velocity_mps"], 3, f"samples[{sample_index}].payload_linear_velocity_mps"),
            "link2_position_m": _matrix(sample["link2_position_m"], 3, f"samples[{sample_index}].link2_position_m"),
            "link2_linear_velocity_mps": _matrix(sample["link2_linear_velocity_mps"], 3, f"samples[{sample_index}].link2_linear_velocity_mps"),
            "link2_payload_contact": _booleans(sample["link2_payload_contact"], f"samples[{sample_index}].link2_payload_contact"),
            "link2_payload_impulse_magnitude_ns": _vector(sample["link2_payload_impulse_magnitude_ns"], f"samples[{sample_index}].link2_payload_impulse_magnitude_ns"),
        })
    return parsed


def _norm(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _maximum_absolute(first: Sequence[float], second: Sequence[float]) -> float:
    return max(abs(a - b) for a, b in zip(first, second))


def _onset(trace: Sequence[Mapping[str, Any]], world: int) -> int | None:
    for sample in trace:
        if sample["link2_payload_contact"][world]:
            return int(sample["control_step"])
    return None


def _phase_cost(
    physx: Sequence[Mapping[str, Any]], cuda: Sequence[Mapping[str, Any]], world: int, shift: int
) -> float:
    total = 0.0
    count = 0
    for index, physx_sample in enumerate(physx):
        cuda_index = index + shift
        if not 0 <= cuda_index < len(cuda):
            continue
        cuda_sample = cuda[cuda_index]
        for left, right in zip(
            physx_sample["joint_positions_rad"][world],
            cuda_sample["joint_positions_rad"][world],
        ):
            total += (left - right) ** 2
            count += 1
        total += _norm(
            physx_sample["payload_position_m"][world],
            cuda_sample["payload_position_m"][world],
        ) ** 2
        count += 1
    return total / max(1, count)


def diagnose_impact_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return descriptive impact-phase evidence without accepting a speedup."""

    if len(reports) != 2 or {item.get("backend") for item in reports} != {PHYSX_BACKEND, CUDA_BACKEND}:
        raise RuntimeError("impact diagnosis requires exactly one PhysX and one CUDA report")
    by_backend = {item["backend"]: item for item in reports}
    physx = validated_impact_trace(by_backend[PHYSX_BACKEND])
    cuda = validated_impact_trace(by_backend[CUDA_BACKEND])

    onset_errors: list[int] = []
    signed_onset_deltas: list[int] = []
    missing_onsets: list[int] = []
    left_censored_onsets: list[int] = []
    phase_shifts: list[int] = []
    maximum_cumulative_impulse_error = 0.0
    maximum_cumulative_impulse_relative_error = 0.0
    maximum_window_end_payload_position_error = 0.0
    maximum_window_end_payload_velocity_error = 0.0
    precontact_maximum_errors = {
        "joint_position_rad": 0.0,
        "joint_velocity_rad_s": 0.0,
        "drive_effort_nm": 0.0,
        "link2_position_m": 0.0,
        "link2_velocity_mps": 0.0,
    }
    threshold_specs = {
        "joint_position_rad": ("maximum_joint_position_error_rad", "joint_positions_rad", "absolute"),
        "joint_velocity_rad_s": ("maximum_joint_velocity_error_rad_s", "joint_velocities_rad_s", "absolute"),
        "drive_effort_nm": ("maximum_drive_effort_error_nm", "drive_efforts_nm", "absolute"),
        "link2_position_m": ("maximum_body_position_error_m", "link2_position_m", "vector"),
        "link2_velocity_mps": ("maximum_body_velocity_error_mps", "link2_linear_velocity_mps", "vector"),
    }
    first_threshold_crossings: dict[str, int | None] = {key: None for key in threshold_specs}
    first_observed_contact_step: int | None = None
    for world in range(IMPACT_SAMPLED_WORLDS):
        physx_onset = _onset(physx, world)
        cuda_onset = _onset(cuda, world)
        if physx[0]["link2_payload_contact"][world] or cuda[0]["link2_payload_contact"][world]:
            left_censored_onsets.append(world)
        elif physx_onset is None or cuda_onset is None:
            missing_onsets.append(world)
        else:
            signed_delta = cuda_onset - physx_onset
            signed_onset_deltas.append(signed_delta)
            onset_errors.append(abs(signed_delta))
            world_first_contact = min(physx_onset, cuda_onset)
            first_observed_contact_step = (
                world_first_contact
                if first_observed_contact_step is None
                else min(first_observed_contact_step, world_first_contact)
            )
            for physx_sample, cuda_sample in zip(physx, cuda):
                if int(physx_sample["control_step"]) >= world_first_contact:
                    break
                precontact_maximum_errors["joint_position_rad"] = max(
                    precontact_maximum_errors["joint_position_rad"],
                    _maximum_absolute(
                        physx_sample["joint_positions_rad"][world],
                        cuda_sample["joint_positions_rad"][world],
                    ),
                )
                precontact_maximum_errors["joint_velocity_rad_s"] = max(
                    precontact_maximum_errors["joint_velocity_rad_s"],
                    _maximum_absolute(
                        physx_sample["joint_velocities_rad_s"][world],
                        cuda_sample["joint_velocities_rad_s"][world],
                    ),
                )
                precontact_maximum_errors["drive_effort_nm"] = max(
                    precontact_maximum_errors["drive_effort_nm"],
                    _maximum_absolute(
                        physx_sample["drive_efforts_nm"][world],
                        cuda_sample["drive_efforts_nm"][world],
                    ),
                )
                precontact_maximum_errors["link2_position_m"] = max(
                    precontact_maximum_errors["link2_position_m"],
                    _norm(
                        physx_sample["link2_position_m"][world],
                        cuda_sample["link2_position_m"][world],
                    ),
                )
                precontact_maximum_errors["link2_velocity_mps"] = max(
                    precontact_maximum_errors["link2_velocity_mps"],
                    _norm(
                        physx_sample["link2_linear_velocity_mps"][world],
                        cuda_sample["link2_linear_velocity_mps"][world],
                    ),
                )
        best_shift = min(
            range(-MAX_PHASE_SHIFT_STEPS, MAX_PHASE_SHIFT_STEPS + 1),
            key=lambda shift: (_phase_cost(physx, cuda, world, shift), abs(shift)),
        )
        phase_shifts.append(best_shift)
        physx_impulse = sum(sample["link2_payload_impulse_magnitude_ns"][world] for sample in physx)
        cuda_impulse = sum(sample["link2_payload_impulse_magnitude_ns"][world] for sample in cuda)
        impulse_error = abs(cuda_impulse - physx_impulse)
        maximum_cumulative_impulse_error = max(maximum_cumulative_impulse_error, impulse_error)
        maximum_cumulative_impulse_relative_error = max(
            maximum_cumulative_impulse_relative_error,
            impulse_error / max(0.05, abs(physx_impulse)),
        )
        maximum_window_end_payload_position_error = max(
            maximum_window_end_payload_position_error,
            _norm(physx[-1]["payload_position_m"][world], cuda[-1]["payload_position_m"][world]),
        )
        maximum_window_end_payload_velocity_error = max(
            maximum_window_end_payload_velocity_error,
            _norm(
                physx[-1]["payload_linear_velocity_mps"][world],
                cuda[-1]["payload_linear_velocity_mps"][world],
            ),
        )
    shift_histogram = {
        str(shift): phase_shifts.count(shift)
        for shift in range(-MAX_PHASE_SHIFT_STEPS, MAX_PHASE_SHIFT_STEPS + 1)
    }
    onset_delta_histogram = {
        str(delta): signed_onset_deltas.count(delta)
        for delta in range(-max(onset_errors, default=0), max(onset_errors, default=0) + 1)
    }
    for physx_sample, cuda_sample in zip(physx, cuda):
        step = int(physx_sample["control_step"])
        for name, (threshold_name, field, reduction) in threshold_specs.items():
            if first_threshold_crossings[name] is not None:
                continue
            crossed = False
            for world in range(IMPACT_SAMPLED_WORLDS):
                if reduction == "absolute":
                    error = _maximum_absolute(physx_sample[field][world], cuda_sample[field][world])
                else:
                    error = _norm(physx_sample[field][world], cuda_sample[field][world])
                if error > PARITY_THRESHOLDS[threshold_name]:
                    crossed = True
                    break
            if crossed:
                first_threshold_crossings[name] = step
    return {
        "schema_version": "factory-os.stage7-impact-diagnosis/v1",
        "authority": "descriptive_only_strict_output_parity_remains_authoritative",
        "window_control_steps": [IMPACT_START_STEP, IMPACT_END_STEP],
        "sampled_worlds": IMPACT_SAMPLED_WORLDS,
        "maximum_contact_onset_error_control_steps": max(onset_errors, default=None),
        "mean_contact_onset_error_control_steps": sum(onset_errors) / len(onset_errors) if onset_errors else None,
        "cuda_contact_onset_minus_physx_histogram_control_steps": onset_delta_histogram,
        "first_observed_contact_control_step": first_observed_contact_step,
        "worlds_missing_contact_onset": missing_onsets,
        "worlds_with_left_censored_contact_onset": left_censored_onsets,
        "best_phase_shift_histogram_control_steps": shift_histogram,
        "precontact_maximum_errors": precontact_maximum_errors,
        "first_parity_threshold_crossing_control_steps": first_threshold_crossings,
        "maximum_cumulative_link2_payload_impulse_error_ns": maximum_cumulative_impulse_error,
        "maximum_cumulative_link2_payload_impulse_relative_error": maximum_cumulative_impulse_relative_error,
        "maximum_window_end_payload_position_error_m": maximum_window_end_payload_position_error,
        "maximum_window_end_payload_velocity_error_mps": maximum_window_end_payload_velocity_error,
    }


__all__ = [
    "IMPACT_END_STEP",
    "IMPACT_FIELDS",
    "IMPACT_SAMPLED_WORLDS",
    "IMPACT_START_STEP",
    "IMPACT_STEPS",
    "MAX_PHASE_SHIFT_STEPS",
    "diagnose_impact_reports",
    "validated_impact_trace",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("physx", type=Path)
    parser.add_argument("cuda", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose_impact_reports([
        json.loads(args.physx.read_text()),
        json.loads(args.cuda.read_text()),
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
