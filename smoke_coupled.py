"""Small measured CUDA correctness gate for the Stage-7 coupled solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .coupled_reference import (
    MAXIMUM_CONTACT_PENETRATION_M,
    MINIMUM_PUSH_DISPLACEMENT_M,
    MINIMUM_RETRACTED_SEPARATION_M,
    PAYLOAD_BODY_INDEX,
    PUSHER_BODY_INDEX,
    CoupledConfig,
    make_coupled_push_state,
)
from .extension import coupled_step, load_extension


def _tensor_bundle(worlds: int, device):
    import torch

    state, inverse_mass, half, inertia, topology, pairs, joint_cache, ids, impulses = (
        make_coupled_push_state(worlds)
    )
    return {
        "state": torch.tensor(state, dtype=torch.float32, device=device),
        "inverse_mass": torch.tensor(inverse_mass, dtype=torch.float32, device=device),
        "half": torch.tensor(half, dtype=torch.float32, device=device),
        "inertia": torch.tensor(inertia, dtype=torch.float32, device=device),
        "joint_indices": torch.tensor(topology.joint_indices, dtype=torch.int64, device=device),
        "joint_types": torch.tensor(topology.joint_types, dtype=torch.int64, device=device),
        "parent_anchor": torch.tensor(topology.parent_anchor_local, dtype=torch.float32, device=device),
        "child_anchor": torch.tensor(topology.child_anchor_local, dtype=torch.float32, device=device),
        "axis": torch.tensor(topology.axis_parent, dtype=torch.float32, device=device),
        "reference": torch.tensor(topology.reference_quaternion_parent_to_child, dtype=torch.float32, device=device),
        "lower": torch.tensor(topology.lower_limit, dtype=torch.float32, device=device),
        "upper": torch.tensor(topology.upper_limit, dtype=torch.float32, device=device),
        "damping": torch.tensor(topology.damping, dtype=torch.float32, device=device),
        "motor_enabled": torch.tensor(topology.motor_enabled, dtype=torch.uint8, device=device),
        "stiffness": torch.zeros((1,), dtype=torch.float32, device=device),
        "pairs": torch.tensor(pairs, dtype=torch.int64, device=device),
        "joint_cache": torch.tensor(joint_cache, dtype=torch.float32, device=device),
        "ids": torch.tensor(ids, dtype=torch.int64, device=device),
        "impulses": torch.tensor(impulses, dtype=torch.float32, device=device),
    }


def _advance(bundle, velocity: float, effort: float, steps: int):
    import torch

    worlds = bundle["state"].shape[0]
    target = torch.full((worlds, 1), velocity, dtype=torch.float32, device=bundle["state"].device)
    maximum_effort = torch.full((worlds, 1), effort, dtype=torch.float32, device=bundle["state"].device)
    output = None
    peak_normal_impulse = 0.0
    peak_penetration = 0.0
    contact_observed = False
    for _ in range(steps):
        output = coupled_step(
            bundle["state"], bundle["inverse_mass"], bundle["half"], bundle["inertia"],
            bundle["joint_indices"], bundle["joint_types"], bundle["parent_anchor"],
            bundle["child_anchor"], bundle["axis"], bundle["reference"], bundle["lower"],
            bundle["upper"], bundle["damping"], bundle["motor_enabled"], target,
            maximum_effort, bundle["pairs"], bundle["ids"], bundle["impulses"],
            CoupledConfig(), stiffness=bundle["stiffness"],
            joint_warm_start_cache=bundle["joint_cache"],
        )
        bundle["state"], bundle["joint_cache"], bundle["ids"], bundle["impulses"] = (
            output[0], output[7], output[10], output[11]
        )
        peak_normal_impulse = max(peak_normal_impulse, float(output[13].max().item()))
        peak_penetration = max(peak_penetration, float(output[9].max().item()))
        contact_observed = contact_observed or bool(torch.any(output[8] != 0).item())
    assert output is not None
    bundle["trajectory_peak_normal_impulse"] = peak_normal_impulse
    bundle["trajectory_peak_penetration"] = peak_penetration
    bundle["trajectory_contact_observed"] = contact_observed
    return output


def run(output_path: Path) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Stage-7 smoke requires CUDA")
    load_extension()
    device = torch.device("cuda")
    push = _tensor_bundle(2, device)
    initial_payload = push["state"][:, PAYLOAD_BODY_INDEX, 0].clone()
    pushed = _advance(push, 1.0, 100.0, 120)
    pushed_payload = push["state"][:, PAYLOAD_BODY_INDEX, 0].clone()
    displacement = pushed_payload - initial_payload
    peak_contact_impulse = push["trajectory_peak_normal_impulse"]
    peak_penetration = push["trajectory_peak_penetration"]
    contact_observed = push["trajectory_contact_observed"]
    released = _advance(push, -1.0, 100.0, 120)
    separation = push["state"][:, PAYLOAD_BODY_INDEX, 0] - push["state"][:, PUSHER_BODY_INDEX, 0] - 0.4

    zero = _tensor_bundle(1, device)
    zero_initial = zero["state"].clone()
    zero_result = _advance(zero, 1.0, 0.0, 120)

    replay_a = _tensor_bundle(2, device)
    replay_b = _tensor_bundle(2, device)
    isolated = _tensor_bundle(1, device)
    replay_out_a = _advance(replay_a, 1.0, 100.0, 80)
    replay_out_b = _advance(replay_b, 1.0, 100.0, 80)
    isolated_out = _advance(isolated, 1.0, 100.0, 80)
    replay_equal = all(torch.equal(left, right) for left, right in zip(replay_out_a, replay_out_b))
    isolation_equal = all(
        torch.equal(batch[0], solo[0])
        for batch, solo in zip(replay_out_a, isolated_out)
        if batch.ndim > 0 and batch.shape[0] == 2 and solo.shape[0] == 1
    )

    checks = {
        "finite_state": bool(torch.isfinite(push["state"]).all().item()),
        "normalized_quaternions": bool(
            torch.all(torch.abs(torch.linalg.vector_norm(push["state"][..., 3:7], dim=-1) - 1.0) <= 2.0e-5).item()
        ),
        "physical_push": bool(torch.all(displacement >= MINIMUM_PUSH_DISPLACEMENT_M).item()),
        "contact_observed": contact_observed,
        "real_contact_impulse": peak_contact_impulse > 0.0,
        "bounded_penetration": peak_penetration <= MAXIMUM_CONTACT_PENETRATION_M,
        "released_without_attachment": bool(torch.all(separation >= MINIMUM_RETRACTED_SEPARATION_M).item()),
        "released_contact_cache_empty": bool(torch.count_nonzero(released[10]).item() == 0 and torch.count_nonzero(released[11]).item() == 0),
        "zero_effort_payload_static": bool(torch.equal(zero["state"][:, PAYLOAD_BODY_INDEX], zero_initial[:, PAYLOAD_BODY_INDEX])),
        "zero_effort_motor_impulse": bool(torch.count_nonzero(zero_result[5]).item() == 0),
        "deterministic_replay": replay_equal,
        "world_isolation": isolation_equal,
    }
    result = {
        "schema_version": "box3d.coupled-smoke/v1",
        "status": "passed" if all(checks.values()) else "failed",
        "backend": "box3d_cuda_stage7",
        "device": torch.cuda.get_device_name(device),
        "checks": checks,
        "metrics": {
            "minimum_payload_displacement_m": float(displacement.min().item()),
            "maximum_penetration_m": peak_penetration,
            "maximum_normal_impulse_ns": peak_contact_impulse,
            "minimum_release_separation_m": float(separation.min().item()),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
