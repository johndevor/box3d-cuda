"""GPU correctness/throughput micro for reduced-articulation contact response."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .articulation_response_reference import (
    ARTICULATION_RESPONSE_FIELDS,
    planar_two_link_contact_response,
)
from .extension import articulation_response, load_extension


CONTRACT_ID = "box3d.articulation-response-micro/v1"
DEFAULT_WORLDS = 1_048_576
DEFAULT_REPEATS = 100
ORACLE_WORLDS = 64
MAXIMUM_ORACLE_ABSOLUTE_ERROR = 2.0e-5


def _inputs(worlds, device):
    import torch

    lane = torch.arange(worlds, dtype=torch.float32, device=device)
    q1 = 0.55 + 0.4 * torch.remainder(lane * 47.0, 997.0) / 996.0
    q2 = -1.35 + 0.4 * torch.remainder(lane * 29.0, 991.0) / 990.0
    base = torch.tensor((-0.8, 0.14), dtype=torch.float32, device=device).expand(worlds, -1).clone()
    direction1 = torch.stack((torch.cos(q1), torch.sin(q1)), dim=1)
    absolute2 = q1 + q2
    direction2 = torch.stack((torch.cos(absolute2), torch.sin(absolute2)), dim=1)
    center1 = base + 0.35 * direction1
    second = center1 + 0.35 * direction1
    center2 = second + 0.30 * direction2
    contact = center2 + 0.30 * direction2
    centers = torch.stack((center1, center2), dim=1)
    normal = torch.zeros((worlds, 2), dtype=torch.float32, device=device)
    normal[:, 0] = 1.0
    properties = torch.tensor(
        (2.0, 1.5, 0.08406666666666666, 0.0468, 1.0, -1.0, 0.0),
        dtype=torch.float32,
        device=device,
    ).expand(worlds, -1).clone()
    return base, second, centers, contact, normal, properties


def benchmark(*, worlds: int, repeats: int) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("articulation-response benchmark requires CUDA")
    load_extension()
    device = torch.device("cuda")
    inputs = _inputs(worlds, device)
    warm = articulation_response(*(item[:1] for item in inputs))
    torch.cuda.synchronize()
    first = articulation_response(*inputs)
    torch.cuda.synchronize()
    second = articulation_response(*inputs)
    torch.cuda.synchronize()
    deterministic = bool(torch.equal(first, second))

    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        articulation_response(*inputs)
    finish.record()
    torch.cuda.synchronize()
    duration_seconds = start.elapsed_time(finish) / 1000.0

    checked = min(worlds, ORACLE_WORLDS)
    host_inputs = [item[:checked].detach().cpu().tolist() for item in inputs]
    oracle = []
    for index in range(checked):
        value = planar_two_link_contact_response(
            base_joint_xy=host_inputs[0][index],
            second_joint_xy=host_inputs[1][index],
            link1_center_xy=host_inputs[2][index][0],
            link2_center_xy=host_inputs[2][index][1],
            contact_point_xy=host_inputs[3][index],
            normal_xy=host_inputs[4][index],
            link1_mass=host_inputs[5][index][0],
            link2_mass=host_inputs[5][index][1],
            link1_inertia_z=host_inputs[5][index][2],
            link2_inertia_z=host_inputs[5][index][3],
            other_inverse_effective_mass=host_inputs[5][index][4],
            relative_normal_velocity=host_inputs[5][index][5],
            restitution=host_inputs[5][index][6],
        )
        oracle.append(value.packed())
    oracle_tensor = torch.tensor(oracle, dtype=torch.float32, device=device)
    maximum_error = float(torch.max(torch.abs(first[:checked] - oracle_tensor)).item())
    finite = bool(torch.isfinite(first).all().item())
    passed = deterministic and finite and maximum_error <= MAXIMUM_ORACLE_ABSOLUTE_ERROR
    return {
        "schema_version": "box3d.articulation-response-benchmark/v1",
        "contract_id": CONTRACT_ID,
        "backend": "box3d_cuda_articulation_response_micro",
        "device": torch.cuda.get_device_name(device),
        "worlds": worlds,
        "repeats": repeats,
        "evaluations": worlds * repeats,
        "duration_seconds": duration_seconds,
        "world_responses_per_second": worlds * repeats / duration_seconds,
        "response_fields": list(ARTICULATION_RESPONSE_FIELDS),
        "correctness": {
            "passed": passed,
            "finite": finite,
            "deterministic_replay_passed": deterministic,
            "oracle_worlds": checked,
            "maximum_oracle_absolute_error": maximum_error,
            "maximum_oracle_absolute_error_threshold": MAXIMUM_ORACLE_ABSOLUTE_ERROR,
            "production_solver_modified": False,
            "micro_scope": "fixed-base planar two-link frictionless normal impact",
        },
        "sample": dict(zip(ARTICULATION_RESPONSE_FIELDS, map(float, warm[0].tolist()))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=DEFAULT_WORLDS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.worlds <= DEFAULT_WORLDS or not 1 <= args.repeats <= 10_000:
        raise ValueError("worlds or repeats are out of bounds")
    report = benchmark(worlds=args.worlds, repeats=args.repeats)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["correctness"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
