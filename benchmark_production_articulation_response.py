"""Exercise the production coupled solver on the accepted PhysX response micro."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .articulation_response_reference import planar_two_link_contact_response
from .benchmark_coupled import _quaternion_z, _rotate_z, make_workload
from .contracts.coupling import CONTROL_HZ, PHYSICS_SUBSTEPS, SPEC
from .coupled_reference import CoupledConfig
from .extension import coupled_step, load_extension
from .joint_reference import JointConfig
from .sat_reference import SATConfig


def _config() -> CoupledConfig:
    return CoupledConfig(
        joints=JointConfig(
            dt=1.0 / CONTROL_HZ,
            substeps=PHYSICS_SUBSTEPS,
            gravity_y=0.0,
            solver_iterations=SPEC.solver_iterations,
        ),
        contacts=SATConfig(
            dt=1.0 / CONTROL_HZ,
            substeps=PHYSICS_SUBSTEPS,
            gravity_y=0.0,
            restitution=0.0,
            friction=0.0,
            position_slop=SPEC.contact_slop_m,
            solver_iterations=SPEC.solver_iterations,
        ),
    )


def _seed(bundle):
    import torch

    worlds = bundle["state"].shape[0]
    device = bundle["state"].device
    q1 = torch.full((worlds,), 0.75, dtype=torch.float32, device=device)
    q2 = torch.full((worlds,), -1.15, dtype=torch.float32, device=device)
    state = torch.zeros_like(bundle["state"])
    state[..., 6] = 1.0
    state[:, 0, :3] = torch.tensor((0.0, -0.05, 0.0), device=device)
    state[:, 1, :3] = torch.tensor(SPEC.arm_base_center_m, device=device)
    state[:, 2, :3] = state[:, 1, :3] + _rotate_z(q1, (0.35, 0.0))
    second_joint = state[:, 2, :3] + _rotate_z(q1, (0.35, 0.0))
    state[:, 3, :3] = second_joint + _rotate_z(q1 + q2, (0.30, 0.0))
    state[:, 2, 3:7] = _quaternion_z(q1)
    state[:, 3, 3:7] = _quaternion_z(q1 + q2)
    direction2 = torch.stack((torch.cos(q1 + q2), torch.sin(q1 + q2)), dim=1)
    tangent2 = torch.stack((-torch.sin(q1 + q2), torch.cos(q1 + q2)), dim=1)
    support = 0.30 * torch.sign(direction2[:, :1]) * direction2
    support += 0.06 * torch.sign(tangent2[:, :1]) * tangent2
    contact_xy = state[:, 3, :2] + support
    state[:, 4, 0] = contact_xy[:, 0] + 0.14 - 5.0e-4
    state[:, 4, 1] = state[:, 3, 1]
    state[:, 4, 7] = -1.0
    bundle["state"] = state
    bundle["joint_cache"].zero_()
    bundle["contact_feature_ids"].zero_()
    bundle["contact_impulse_cache"].zero_()
    return second_joint[:, :2], contact_xy


def _run(bundle, *, articulation_projection: bool):
    import torch

    worlds = bundle["state"].shape[0]
    controls = torch.zeros((worlds, 2), dtype=torch.float32, device=bundle["state"].device)
    return coupled_step(
        bundle["state"], bundle["inverse_mass"], bundle["half"], bundle["inverse_inertia"],
        bundle["joint_indices"], bundle["joint_types"], bundle["parent_anchor"], bundle["child_anchor"],
        bundle["axis"], bundle["reference"], bundle["lower"], bundle["upper"], bundle["damping"],
        bundle["motor_enabled"], controls, controls, bundle["pairs"], bundle["contact_feature_ids"],
        bundle["contact_impulse_cache"], _config(), motor_target_position=controls,
        stiffness=torch.zeros_like(bundle["stiffness"]), joint_warm_start_cache=bundle["joint_cache"],
        articulation_projection=articulation_projection,
    )


def benchmark(output_path: Path, *, worlds: int = 64) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("production articulation-response benchmark requires CUDA")
    if worlds <= 0 or worlds > 1_048_576:
        raise ValueError("worlds must be in [1,1048576]")
    load_extension()
    device = torch.device("cuda")
    template = make_workload(worlds, device)
    second_joint, contact_xy = _seed(template)
    payload_offset = contact_xy - template["state"][:, 4, :2]
    payload_inverse_effective_mass = (
        SPEC.body_inverse_masses_per_kg[4]
        + payload_offset[:, 1].square() / SPEC.body_inertia_diagonal_kg_m2[4][2]
    )
    expected = planar_two_link_contact_response(
        base_joint_xy=template["state"][0, 1, :2].tolist(),
        second_joint_xy=second_joint[0].tolist(),
        link1_center_xy=template["state"][0, 2, :2].tolist(),
        link2_center_xy=template["state"][0, 3, :2].tolist(),
        contact_point_xy=contact_xy[0].tolist(),
        normal_xy=(1.0, 0.0),
        link1_mass=SPEC.body_masses_kg[2],
        link2_mass=SPEC.body_masses_kg[3],
        link1_inertia_z=SPEC.body_inertia_diagonal_kg_m2[2][2],
        link2_inertia_z=SPEC.body_inertia_diagonal_kg_m2[3][2],
        other_inverse_effective_mass=float(payload_inverse_effective_mass[0].item()),
        relative_normal_velocity=-1.0,
        restitution=0.0,
    )
    modes = {}
    for name, enabled in (("maximal", False), ("projected", True)):
        bundle = make_workload(worlds, device)
        _seed(bundle)
        first = _run(bundle, articulation_projection=enabled)
        second_bundle = make_workload(worlds, device)
        _seed(second_bundle)
        second = _run(second_bundle, articulation_projection=enabled)
        torch.cuda.synchronize()
        state = first[0]
        measured_impulse = SPEC.body_masses_kg[4] * (state[:, 4, 7] + 1.0)
        parents = torch.tensor(SPEC.joint_parent_body_indices, device=device)
        children = torch.tensor(SPEC.joint_child_body_indices, device=device)
        measured_qvel = state[:, children, 12] - state[:, parents, 12]
        expected_qvel = torch.tensor(
            expected.articulated_joint_velocity_delta, dtype=torch.float32, device=device
        ).expand(worlds, -1)
        pair_index = SPEC.contact_pair_roles.index("link2_payload")
        deterministic = all(torch.equal(left, right) for left, right in zip(first, second))
        modes[name] = {
            "articulation_projection": enabled,
            "deterministic": deterministic,
            "finite": bool(torch.isfinite(state).all().item()),
            "contact_worlds": int(torch.count_nonzero(first[8][:, pair_index]).item()),
            "mean_payload_momentum_impulse_ns": float(measured_impulse.mean().item()),
            "maximum_payload_momentum_impulse_error_ns": float(
                torch.abs(measured_impulse - expected.articulated_normal_impulse).max().item()
            ),
            "maximum_joint_velocity_delta_error_rad_s": float(
                torch.abs(measured_qvel - expected_qvel).max().item()
            ),
            "mean_reported_pair_impulse_ns": float(first[13][:, pair_index].mean().item()),
        }
    result = {
        "schema_version": "box3d-cuda.production-articulation-response/v1",
        "contract_id": "box3d.articulation-response-micro/v1",
        "device": torch.cuda.get_device_name(device),
        "worlds": worlds,
        "expected_articulated_normal_impulse_ns": expected.articulated_normal_impulse,
        "expected_joint_velocity_delta_rad_s": list(expected.articulated_joint_velocity_delta),
        "payload_inverse_effective_mass_per_kg": float(payload_inverse_effective_mass[0].item()),
        "modes": modes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=64)
    args = parser.parse_args()
    result = benchmark(args.output, worlds=args.worlds)
    print(json.dumps(result, sort_keys=True))
    return 0 if all(mode["finite"] and mode["deterministic"] for mode in result["modes"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
