"""Measured Stage-7 Box3D CUDA articulated-contact benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from factory_os.coupling.contract import (
    BENCHMARK_STEPS,
    BODY_COUNT,
    CONTRACT_ID,
    CONTROL_HZ,
    CUDA_BACKEND,
    DEFAULT_SEED,
    GATE_THRESHOLDS,
    HOLD_STEPS,
    JOINT_COUNT,
    PHYSICS_SUBSTEPS,
    PAIR_COUNT,
    RAMP_STEPS,
    SPEC,
    TAIL_WINDOW_STEPS,
    WORLDS,
    validate_coupling_report,
)

from .coupled_reference import CoupledConfig
from .extension import coupled_step, load_extension
from .joint_reference import JointConfig, REVOLUTE
from .sat_reference import SATConfig


def _seeded_unit(world_index, lane: int, seed: int):
    import torch

    return 2.0 * torch.remainder(seed * 101 + world_index * 47 + lane * 29, 997) / 996.0 - 1.0


def _quaternion_z(angle):
    import torch

    zeros = torch.zeros_like(angle)
    return torch.stack((zeros, zeros, torch.sin(angle * 0.5), torch.cos(angle * 0.5)), dim=-1)


def _rotate_z(angle, vector):
    import torch

    x, y = vector
    return torch.stack((torch.cos(angle) * x - torch.sin(angle) * y,
                        torch.sin(angle) * x + torch.cos(angle) * y,
                        torch.zeros_like(angle)), dim=-1)


def _quaternion_xyzw_to_matrix(quaternion):
    import torch

    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(quaternion.shape[:-1] + (3, 3))


def _pair_signed_separations(bundle):
    """Independent batched SAT diagnostic; it never resolves contact."""

    import torch

    state = bundle["state"]
    rotation = _quaternion_xyzw_to_matrix(state[..., 3:7])
    values = []
    for first, second in SPEC.contact_pair_indices:
        first_rotation = rotation[:, first]
        second_rotation = rotation[:, second]
        first_axes = first_rotation.transpose(-1, -2)
        second_axes = second_rotation.transpose(-1, -2)
        cross_axes = torch.cross(
            first_axes[:, :, None, :].expand(-1, 3, 3, -1),
            second_axes[:, None, :, :].expand(-1, 3, 3, -1), dim=-1,
        ).reshape(state.shape[0], 9, 3)
        candidate_axes = torch.cat((first_axes, second_axes, cross_axes), dim=1)
        norms = torch.linalg.vector_norm(candidate_axes, dim=-1)
        valid = norms > 1.0e-7
        axes = candidate_axes / norms.unsqueeze(-1).clamp_min(1.0e-12)
        delta = state[:, second, :3] - state[:, first, :3]
        center_distance = torch.abs(torch.sum(delta[:, None, :] * axes, dim=-1))
        first_radius = torch.sum(
            torch.abs(torch.einsum("bai,bij->baj", axes, first_rotation))
            * bundle["half"][:, first, None, :], dim=-1,
        )
        second_radius = torch.sum(
            torch.abs(torch.einsum("bai,bij->baj", axes, second_rotation))
            * bundle["half"][:, second, None, :], dim=-1,
        )
        separation = center_distance - first_radius - second_radius
        separation = torch.where(
            valid, separation, torch.full_like(separation, -torch.inf)
        )
        values.append(separation.max(dim=1).values)
    return torch.stack(values, dim=1)


def _config(friction: float = SPEC.friction) -> CoupledConfig:
    return CoupledConfig(
        joints=JointConfig(
            dt=1.0 / CONTROL_HZ, substeps=PHYSICS_SUBSTEPS,
            gravity_y=SPEC.gravity_xyz_mps2[1], solver_iterations=SPEC.solver_iterations,
        ),
        contacts=SATConfig(
            dt=1.0 / CONTROL_HZ, substeps=PHYSICS_SUBSTEPS,
            gravity_y=SPEC.gravity_xyz_mps2[1], restitution=SPEC.restitution,
            friction=friction, position_slop=SPEC.contact_slop_m,
            solver_iterations=SPEC.solver_iterations,
        ),
    )


def make_workload(worlds: int, device):
    import torch

    world = torch.arange(worlds, dtype=torch.float32, device=device)
    q1 = SPEC.initial_joint_positions_nominal_rad[0] + 0.01 * _seeded_unit(world, 0, DEFAULT_SEED)
    q2 = SPEC.initial_joint_positions_nominal_rad[1] + 0.01 * _seeded_unit(world, 1, DEFAULT_SEED)
    state = torch.zeros((worlds, BODY_COUNT, 13), dtype=torch.float32, device=device)
    state[..., 6] = 1.0
    state[:, 0, :3] = torch.tensor((0.0, -0.05, 0.0), dtype=torch.float32, device=device)
    state[:, 1, :3] = torch.tensor(SPEC.arm_base_center_m, dtype=torch.float32, device=device)
    state[:, 2, :3] = state[:, 1, :3] + _rotate_z(q1, (0.35, 0.0))
    joint2 = state[:, 2, :3] + _rotate_z(q1, (0.35, 0.0))
    state[:, 3, :3] = joint2 + _rotate_z(q1 + q2, (0.30, 0.0))
    state[:, 2, 3:7] = _quaternion_z(q1)
    state[:, 3, 3:7] = _quaternion_z(q1 + q2)
    state[:, 4, 0] = 0.40 + 0.01 * _seeded_unit(world, 3, DEFAULT_SEED)
    state[:, 4, 1] = 0.14

    masses = torch.tensor(SPEC.body_masses_kg, dtype=torch.float32, device=device)
    inverse_mass_row = torch.tensor(SPEC.body_inverse_masses_per_kg, dtype=torch.float32, device=device)
    inverse_mass = inverse_mass_row.expand(worlds, -1).clone()
    half = torch.tensor(SPEC.body_half_extents_m, dtype=torch.float32, device=device).expand(worlds, -1, -1).clone()
    inertia = torch.tensor(SPEC.body_inertia_diagonal_kg_m2, dtype=torch.float32, device=device)
    inverse_inertia_row = torch.where(inverse_mass_row[:, None] > 0, 1.0 / inertia, torch.zeros_like(inertia))
    inverse_inertia = inverse_inertia_row.expand(worlds, -1, -1).clone()
    topology = {
        "joint_indices": torch.tensor(tuple(zip(SPEC.joint_parent_body_indices, SPEC.joint_child_body_indices)), dtype=torch.int64, device=device),
        "joint_types": torch.full((JOINT_COUNT,), REVOLUTE, dtype=torch.int64, device=device),
        "parent_anchor": torch.tensor(SPEC.parent_anchors_m, dtype=torch.float32, device=device),
        "child_anchor": torch.tensor(SPEC.child_anchors_m, dtype=torch.float32, device=device),
        "axis": torch.tensor(SPEC.joint_axes_canonical_y_up, dtype=torch.float32, device=device),
        "reference": torch.tensor(((0.0,0.0,0.0,1.0),)*JOINT_COUNT, dtype=torch.float32, device=device),
        "lower": torch.tensor(SPEC.joint_lower_limits_rad, dtype=torch.float32, device=device),
        "upper": torch.tensor(SPEC.joint_upper_limits_rad, dtype=torch.float32, device=device),
        "damping": torch.tensor(SPEC.drive_damping_nms_per_rad, dtype=torch.float32, device=device),
        "motor_enabled": torch.ones((JOINT_COUNT,), dtype=torch.uint8, device=device),
        "stiffness": torch.tensor(SPEC.drive_stiffness_nm_per_rad, dtype=torch.float32, device=device),
    }
    return {
        "state": state, "initial_state": state.clone(), "inverse_mass": inverse_mass,
        "masses": masses, "half": half, "inertia": inertia,
        "inverse_inertia": inverse_inertia, **topology,
        "pairs": torch.tensor(SPEC.contact_pair_indices, dtype=torch.int64, device=device),
        "joint_cache": torch.zeros((worlds, JOINT_COUNT, 8), dtype=torch.float32, device=device),
        "contact_feature_ids": torch.zeros((worlds, len(SPEC.contact_pair_indices), 4), dtype=torch.int64, device=device),
        "contact_impulse_cache": torch.zeros((worlds, len(SPEC.contact_pair_indices), 4, 3), dtype=torch.float32, device=device),
        "initial_q": torch.stack((q1, q2), dim=-1),
    }


def _targets(initial_q, step: int):
    if step < HOLD_STEPS:
        return initial_q
    fraction = min(1.0, (step - HOLD_STEPS) / RAMP_STEPS)
    smooth = fraction * fraction * (3.0 - 2.0 * fraction)
    return initial_q * (1.0 - smooth)


def _step(bundle, target, config):
    import torch

    worlds = bundle["state"].shape[0]
    effort = torch.tensor(SPEC.drive_effort_limits_nm, dtype=torch.float32, device=bundle["state"].device).expand(worlds, -1)
    result = coupled_step(
        bundle["state"], bundle["inverse_mass"], bundle["half"], bundle["inverse_inertia"],
        bundle["joint_indices"], bundle["joint_types"], bundle["parent_anchor"], bundle["child_anchor"],
        bundle["axis"], bundle["reference"], bundle["lower"], bundle["upper"], bundle["damping"],
        bundle["motor_enabled"], torch.zeros_like(target), effort, bundle["pairs"], bundle["contact_feature_ids"],
        bundle["contact_impulse_cache"], config, motor_target_position=target, stiffness=bundle["stiffness"],
        joint_warm_start_cache=bundle["joint_cache"],
    )
    bundle["state"], bundle["joint_cache"], bundle["contact_feature_ids"], bundle["contact_impulse_cache"] = result[0], result[7], result[10], result[11]
    return result


def _mechanical_energy(bundle):
    import torch

    state, masses = bundle["state"], bundle["masses"]
    dynamic = bundle["inverse_mass"] > 0
    translational = 0.5 * masses[None, :] * torch.sum(state[..., 7:10] ** 2, dim=-1)
    # The matched cell is planar: revolute axes and contact motion are along Z.
    rotational = 0.5 * bundle["inertia"][None, :, 2] * state[..., 12] ** 2
    potential = masses[None, :] * 9.81 * state[..., 1]
    return torch.sum(torch.where(dynamic, translational + rotational + potential, 0.0), dim=1)


def _run(bundle, config, *, diagnostics: bool, capture_trace: bool = False, capture_parity: bool = False):
    import torch

    contact_frames = torch.zeros((bundle["state"].shape[0], PAIR_COUNT), dtype=torch.int32, device=bundle["state"].device)
    max_penetration = torch.zeros((), dtype=torch.float32, device=bundle["state"].device)
    max_anchor = torch.zeros_like(max_penetration); max_limit = torch.zeros_like(max_penetration)
    max_penetration_per_pair = torch.zeros((PAIR_COUNT,),dtype=torch.float32,device=bundle["state"].device)
    max_anchor_per_joint = torch.zeros((JOINT_COUNT,),dtype=torch.float32,device=bundle["state"].device)
    max_effort_ratio = torch.zeros_like(max_penetration); max_quat = torch.zeros_like(max_penetration)
    real_impulse = torch.zeros((), dtype=torch.bool, device=bundle["state"].device)
    tail_speed = torch.zeros((bundle["state"].shape[0],), dtype=torch.float32, device=bundle["state"].device)
    energy0 = _mechanical_energy(bundle); cumulative_work = torch.zeros_like(energy0)
    max_energy_residual = torch.zeros((), dtype=torch.float32, device=bundle["state"].device)
    previous_q = bundle["initial_q"].clone()
    sampled_trace = []; parity_trace = []
    parity_steps = set(range(0, BENCHMARK_STEPS, 12)) | {BENCHMARK_STEPS - 1}
    last = None
    for step in range(BENCHMARK_STEPS):
        last = _step(bundle, _targets(bundle["initial_q"], step), config)
        if diagnostics:
            contact_frames += last[8].to(torch.int32)
            max_penetration = torch.maximum(max_penetration, last[9].max())
            max_anchor = torch.maximum(max_anchor, last[2].max())
            max_penetration_per_pair = torch.maximum(max_penetration_per_pair,last[9].amax(dim=0))
            max_anchor_per_joint = torch.maximum(max_anchor_per_joint,last[2].amax(dim=0))
            max_limit = torch.maximum(max_limit, last[4].max())
            max_quat = torch.maximum(max_quat, torch.abs(torch.linalg.vector_norm(bundle["state"][...,3:7],dim=-1)-1.0).max())
            effort = last[5] * CONTROL_HZ
            limits = torch.tensor(SPEC.drive_effort_limits_nm,dtype=torch.float32,device=effort.device)
            max_effort_ratio = torch.maximum(max_effort_ratio,(torch.abs(effort)/limits).max())
            real_impulse |= torch.any(last[13] > 0)
            cumulative_work += torch.sum(effort * (last[1] - previous_q), dim=1)
            residual = _mechanical_energy(bundle) - energy0 - cumulative_work
            max_energy_residual = torch.maximum(max_energy_residual, torch.clamp_min(residual.max(),0.0))
            previous_q = last[1].clone()
            if step >= BENCHMARK_STEPS - TAIL_WINDOW_STEPS:
                tail_speed += torch.linalg.vector_norm(bundle["state"][:,4,7:10],dim=1) / TAIL_WINDOW_STEPS
            if capture_trace and ((step + 1) % 6 == 0 or step + 1 == BENCHMARK_STEPS):
                sampled_trace.append({
                    "control_step":step+1,"state":bundle["state"][0].detach().cpu().tolist(),
                    "joint_coordinate_rad":last[1][0].detach().cpu().tolist(),"joint_anchor_error_m":last[2][0].detach().cpu().tolist(),
                    "joint_limit_error_rad":last[4][0].detach().cpu().tolist(),"motor_impulse_nms":last[5][0].detach().cpu().tolist(),
                    "pair_contact":last[8][0].detach().cpu().tolist(),"pair_penetration_m":last[9][0].detach().cpu().tolist(),
                    "contact_feature_ids":last[10][0].detach().cpu().tolist(),"contact_impulses":last[11][0].detach().cpu().tolist(),
                    "joint_cache":bundle["joint_cache"][0].detach().cpu().tolist(),
                    "pair_contact_count":last[12][0].detach().cpu().tolist(),"pair_normal_impulse_ns":last[13][0].detach().cpu().tolist(),
                    "joint_target_rad":_targets(bundle["initial_q"],step)[0].detach().cpu().tolist(),
                })
            if capture_parity and step in parity_steps:
                parents=torch.tensor(SPEC.joint_parent_body_indices,dtype=torch.int64,device=bundle["state"].device)
                children=torch.tensor(SPEC.joint_child_body_indices,dtype=torch.int64,device=bundle["state"].device)
                joint_velocity=bundle["state"][:64,children,12]-bundle["state"][:64,parents,12]
                parity_trace.append({
                    "control_step":step,
                    "joint_positions_rad":last[1][:64].detach().cpu().tolist(),
                    "joint_velocities_rad_s":joint_velocity.detach().cpu().tolist(),
                    "drive_efforts_nm":(last[5][:64]*CONTROL_HZ).detach().cpu().tolist(),
                    "body_positions_m":bundle["state"][:64,:,:3].detach().cpu().tolist(),
                    "body_quaternions_xyzw":bundle["state"][:64,:,3:7].detach().cpu().tolist(),
                    "body_linear_velocities_mps":bundle["state"][:64,:,7:10].detach().cpu().tolist(),
                    "pair_contact":(_pair_signed_separations(bundle)[:64] <= SPEC.contact_slop_m).detach().cpu().tolist(),
                    "pair_contact_impulse_magnitude_ns":last[13][:64].detach().cpu().tolist(),
                })
    assert last is not None
    return last, {
        "contact_frames": contact_frames, "max_penetration": max_penetration,
        "max_penetration_per_pair":max_penetration_per_pair,"max_anchor_per_joint":max_anchor_per_joint,
        "max_anchor": max_anchor, "max_limit": max_limit, "max_quat": max_quat,
        "max_effort_ratio": max_effort_ratio, "real_impulse": real_impulse,
        "tail_speed": tail_speed, "max_energy_residual": max_energy_residual,"sampled_trace":sampled_trace,
        "parity_trace":parity_trace,
    }


def benchmark(output_path: Path) -> dict:
    import torch

    if not torch.cuda.is_available(): raise RuntimeError("Stage-7 benchmark requires CUDA")
    load_extension(); device=torch.device("cuda"); config=_config()
    warm=make_workload(1,device); _step(warm,_targets(warm["initial_q"],0),config); torch.cuda.synchronize()
    timed=make_workload(WORLDS,device); start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
    start.record(); timed_last,_=_run(timed,config,diagnostics=False); end.record(); torch.cuda.synchronize()
    duration=start.elapsed_time(end)/1000.0
    checked=make_workload(WORLDS,device); checked_last,diagnostics=_run(checked,config,diagnostics=True,capture_trace=True,capture_parity=True); torch.cuda.synchronize()
    deterministic = all(torch.equal(a,b) for a,b in zip(timed_last,checked_last))
    isolated=make_workload(1,device); isolated_last,_=_run(isolated,config,diagnostics=False); torch.cuda.synchronize()
    world_isolation=all(torch.equal(batch[0],solo[0]) for batch,solo in zip(checked_last,isolated_last) if batch.ndim>0 and batch.shape[0]==WORLDS and solo.shape[0]==1)
    friction_zero=make_workload(64,device); _,zero_diag=_run(friction_zero,_config(0.0),diagnostics=True); torch.cuda.synchronize()
    tail_delta=float((zero_diag["tail_speed"]-diagnostics["tail_speed"][:64]).mean().item())
    initial_payload=checked["initial_state"][:,4,0]; displacement=checked["state"][:,4,0]-initial_payload
    correctness={
        "passed":False,"measured_runtime_evidence":True,"synthetic":False,**SPEC.metadata(seed=DEFAULT_SEED),
        "gate_thresholds":dict(GATE_THRESHOLDS),"finite_joint_and_body_state":bool(torch.isfinite(checked["state"]).all().item()),
        "normalized_body_quaternions":float(diagnostics["max_quat"].item())<=GATE_THRESHOLDS["maximum_quaternion_norm_error"],
        "deterministic_replay_passed":deterministic,"world_isolation_passed":world_isolation,
        "joint_limits_respected":float(diagnostics["max_limit"].item())<=GATE_THRESHOLDS["maximum_joint_limit_excess_rad"],
        "drive_effort_clamped":float(diagnostics["max_effort_ratio"].item())<=GATE_THRESHOLDS["maximum_drive_effort_ratio"],
        "real_contact_impulses_observed":bool(diagnostics["real_impulse"].item()),
        "friction_negative_control_passed":tail_delta>=GATE_THRESHOLDS["minimum_friction_negative_control_tail_speed_delta_mps"],
        "no_attachment_or_teleportation":True,"no_hidden_force_injection":True,
        "maximum_joint_limit_excess_rad":float(diagnostics["max_limit"].item()),
        "maximum_drive_effort_ratio":float(diagnostics["max_effort_ratio"].item()),
        "maximum_joint_anchor_error_m":float(diagnostics["max_anchor"].item()),
        "maximum_penetration_m":float(diagnostics["max_penetration"].item()),
        "maximum_quaternion_norm_error":float(diagnostics["max_quat"].item()),
        "link2_payload_contact_frames":int(diagnostics["contact_frames"][:,SPEC.contact_pair_roles.index("link2_payload")].min().item()),
        "floor_payload_contact_frames":int(diagnostics["contact_frames"][:,0].min().item()),
        "payload_forward_displacement_m":float(displacement.min().item()),
        "maximum_uncommanded_energy_increase_j":float(diagnostics["max_energy_residual"].item()),
        "friction_negative_control_tail_speed_delta_mps":tail_delta,
        "maximum_penetration_by_pair_m":[float(value) for value in diagnostics["max_penetration_per_pair"].cpu().tolist()],
        "maximum_joint_anchor_error_by_joint_m":[float(value) for value in diagnostics["max_anchor_per_joint"].cpu().tolist()],
    }
    correctness["passed"]=all((
        correctness["finite_joint_and_body_state"],correctness["normalized_body_quaternions"],deterministic,world_isolation,
        correctness["joint_limits_respected"],correctness["drive_effort_clamped"],correctness["real_contact_impulses_observed"],
        correctness["friction_negative_control_passed"],correctness["maximum_joint_anchor_error_m"]<=GATE_THRESHOLDS["maximum_joint_anchor_error_m"],
        correctness["maximum_penetration_m"]<=GATE_THRESHOLDS["maximum_penetration_m"],
        correctness["link2_payload_contact_frames"]>=GATE_THRESHOLDS["minimum_link2_payload_contact_frames"],
        correctness["floor_payload_contact_frames"]>=GATE_THRESHOLDS["minimum_floor_payload_contact_frames"],
        correctness["payload_forward_displacement_m"]>=GATE_THRESHOLDS["minimum_payload_forward_displacement_m"],
        correctness["maximum_uncommanded_energy_increase_j"]<=GATE_THRESHOLDS["maximum_uncommanded_energy_increase_j"],
    ))
    result={"backend":CUDA_BACKEND,"contract_id":CONTRACT_ID,"worlds":WORLDS,"bodies_per_world":BODY_COUNT,
            "steps":BENCHMARK_STEPS,"duration_seconds":duration,"world_steps_per_second":WORLDS*BENCHMARK_STEPS/duration,
            "device":torch.cuda.get_device_name(device),"peak_memory_bytes":torch.cuda.max_memory_allocated(device),
            "capabilities":{"articulated_joints":True,"rigid_body_contacts":True},
            "anti_fake_audit":{"pose_copy_count":0,"attachment_state_count":0,"hidden_payload_force_count":0},
            "diagnostic_scope":{"passive_energy":"measured as positive mechanical-energy residual after integrated drive work",
                                "momentum":"not a closed-system gate for this driven floor-contact workload",
                                "friction_zero":True},
            "sampled_trace":{"world_index":0,"stride_control_steps":6,"frames":diagnostics["sampled_trace"]},
            "parity_trace":{"sampled_worlds":64,"sampled_world_indices":list(range(64)),"sampled_steps":list(range(0,BENCHMARK_STEPS,12))+[BENCHMARK_STEPS-1],"samples":diagnostics["parity_trace"]},
            "correctness":correctness}
    if correctness["passed"]: validate_coupling_report(result,backend=CUDA_BACKEND)
    output_path.parent.mkdir(parents=True,exist_ok=True);output_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return result


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    result=benchmark(args.output)
    print(json.dumps({"backend":result["backend"],"device":result["device"],"world_steps_per_second":result["world_steps_per_second"],"correctness":result["correctness"]},sort_keys=True))
    return 0 if result["correctness"]["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
