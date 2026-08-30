"""Contact-free CUDA micro for Stage-7 gravity and drive decomposition.

This diagnostic holds topology, inertias, timestep, substeps, limits, and the
projected two-revolute solver fixed while evaluating gravity-only, drive-only,
and combined response.  It does not change or promote production behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark_coupled import (
    _drive_effort_proxy,
    _step,
    make_workload,
)
from .contracts.coupling import CONTROL_HZ, PHYSICS_SUBSTEPS, SPEC
from .coupled_reference import CoupledConfig
from .extension import load_extension
from .joint_reference import JointConfig
from .sat_reference import SATConfig


CONTRACT_ID = "box3d.joint-dynamics-decomposition/v1"
CONTROL_STEPS = 8
TARGET_OFFSET_RAD = (0.08, -0.06)
MODES = {
    "gravity_only": {"gravity_y": SPEC.gravity_xyz_mps2[1], "drives": False},
    "drive_only": {"gravity_y": 0.0, "drives": True},
    "combined": {"gravity_y": SPEC.gravity_xyz_mps2[1], "drives": True},
}


def _config(gravity_y: float) -> CoupledConfig:
    return CoupledConfig(
        joints=JointConfig(
            dt=1.0 / CONTROL_HZ,
            substeps=PHYSICS_SUBSTEPS,
            gravity_y=gravity_y,
            solver_iterations=SPEC.solver_iterations,
        ),
        contacts=SATConfig(
            dt=1.0 / CONTROL_HZ,
            substeps=PHYSICS_SUBSTEPS,
            gravity_y=gravity_y,
            restitution=0.0,
            friction=0.0,
            position_slop=SPEC.contact_slop_m,
            contact_generation_distance=0.0,
            solver_iterations=SPEC.solver_iterations,
        ),
    )


def _contact_free_bundle(worlds: int, device, *, drives: bool):
    import torch

    bundle = make_workload(worlds, device)
    # Keep the required contact ABI shape but move the payload far outside the
    # finite floor and arm. The diagnostic fails if any impulse is observed.
    bundle["state"][:, 4, :3] = torch.tensor(
        (20.0, 20.0, 20.0), dtype=torch.float32, device=device
    )
    bundle["initial_state"] = bundle["state"].clone()
    bundle["pairs"] = torch.tensor(((0, 4),), dtype=torch.int64, device=device)
    bundle["contact_feature_ids"] = torch.zeros(
        (worlds, 1, 4), dtype=torch.int64, device=device
    )
    bundle["contact_impulse_cache"] = torch.zeros(
        (worlds, 1, 4, 3), dtype=torch.float32, device=device
    )
    if not drives:
        bundle["motor_enabled"].zero_()
        bundle["stiffness"].zero_()
        bundle["damping"].zero_()
    return bundle


def _run_mode(worlds: int, device, *, gravity_y: float, drives: bool) -> dict:
    import torch

    bundle = _contact_free_bundle(worlds, device, drives=drives)
    target_offset = torch.tensor(
        TARGET_OFFSET_RAD, dtype=torch.float32, device=device
    ).expand(worlds, -1)
    target = bundle["initial_q"] + target_offset if drives else bundle["initial_q"]
    samples = []
    contact_observed = False
    result = None
    for control_step in range(CONTROL_STEPS):
        result = _step(
            bundle,
            target,
            _config(gravity_y),
            articulation_projection=True,
        )
        effort, joint_velocity = _drive_effort_proxy(
            bundle, result[1], target
        )
        contact_observed |= bool(torch.any(result[8]).item())
        contact_observed |= bool(torch.any(result[13] > 0).item())
        samples.append({
            "control_step": control_step,
            "joint_positions_rad": result[1][0].detach().cpu().tolist(),
            "joint_velocities_rad_s": joint_velocity[0].detach().cpu().tolist(),
            "drive_efforts_nm": effort[0].detach().cpu().tolist(),
            "joint_targets_rad": target[0].detach().cpu().tolist(),
            "link_positions_m": bundle["state"][0, 2:4, :3].detach().cpu().tolist(),
            "link_linear_velocities_mps": bundle["state"][0, 2:4, 7:10].detach().cpu().tolist(),
            "link_angular_velocities_rad_s": bundle["state"][0, 2:4, 10:13].detach().cpu().tolist(),
        })
    assert result is not None
    replica_error = max(
        float((result[1] - result[1][0:1]).abs().max().item()),
        float((bundle["state"] - bundle["state"][0:1]).abs().max().item()),
    )
    return {
        "gravity_y_mps2": gravity_y,
        "drives_enabled": drives,
        "contact_observed": contact_observed,
        "maximum_replica_error": replica_error,
        "finite": bool(torch.isfinite(bundle["state"]).all().item()),
        "samples": samples,
        "final_state": bundle["state"].clone(),
        "final_joint_cache": bundle["joint_cache"].clone(),
    }


def benchmark(output_path: Path, *, worlds: int = 64) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("joint-dynamics decomposition requires CUDA")
    if worlds <= 0 or worlds > 4096:
        raise ValueError("worlds must be in [1,4096]")
    load_extension()
    device = torch.device("cuda")
    modes = {}
    for name, configuration in MODES.items():
        first = _run_mode(worlds, device, **configuration)
        second = _run_mode(worlds, device, **configuration)
        deterministic = torch.equal(first.pop("final_state"), second.pop("final_state"))
        deterministic &= torch.equal(
            first.pop("final_joint_cache"), second.pop("final_joint_cache")
        )
        deterministic &= first["samples"] == second["samples"]
        second.pop("samples")
        modes[name] = {**first, "deterministic": deterministic}
    result = {
        "schema_version": "box3d-cuda.joint-dynamics-decomposition/v1",
        "contract_id": CONTRACT_ID,
        "backend": "box3d_cuda_stage7",
        "device": torch.cuda.get_device_name(device),
        "worlds": worlds,
        "control_steps": CONTROL_STEPS,
        "physics_substeps": PHYSICS_SUBSTEPS,
        "control_hz": CONTROL_HZ,
        "target_offset_rad": list(TARGET_OFFSET_RAD),
        "articulation_projection": True,
        "diagnostic_only": True,
        "accepted_solver_change": None,
        "modes": modes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=64)
    args = parser.parse_args()
    result = benchmark(args.output, worlds=args.worlds)
    print(json.dumps(result, sort_keys=True))
    passed = all(
        mode["finite"]
        and mode["deterministic"]
        and not mode["contact_observed"]
        and mode["maximum_replica_error"] == 0.0
        for mode in result["modes"].values()
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
