"""Bounded CPU/CUDA gate for the pinned approximate KR240 joint model.

This validates collision-free maximal-coordinate joint dynamics. It does not
validate manufacturer dynamics, meshes, contact, manipulation, or full
multi-turn ranges beyond the solver's principal-angle branch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from .benchmarking import BenchmarkResult, CapabilitySet, write_result
from .extension import joint_step, load_extension
from .industrial_joint_import import compile_industrial_joint_world, load_industrial_joint_model
from .joint_reference import JointConfig, step_joint_reference


CONTRACT_ID = "box3d.kr240-joint-runtime/v1"
CONTROL_HZ = 120
SUBSTEPS = 2
STEPS = 360
HOLD_STEPS = 60
RAMP_STEPS = 60
TAIL_STEPS = 120
DEFAULT_SEED = 67
AMPLITUDES = (0.20, 0.12, 0.18, 0.22, 0.16, 0.25)
FREQUENCIES = (0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
STIFFNESS = (30000.0, 30000.0, 22500.0, 7500.0, 5000.0, 3000.0)
PARITY_WORLD_IDS = (0, 1, 17, 63)
PARITY_STEPS = 120
TRACE_STEPS = tuple(range(0, PARITY_STEPS, 12)) + (PARITY_STEPS - 1,)
REPRESENTABLE_LIMIT_JOINTS = (1, 2, 4)
UNVALIDATED_FULL_RANGE_JOINTS = ("joint_a1", "joint_a4", "joint_a6")
STATE_TOLERANCE = 2.0e-2
DIAGNOSTIC_TOLERANCE = 3.0e-3
CACHE_RELATIVE_TOLERANCE = 3.0e-3
MOTOR_IMPULSE_TOLERANCE = 1.0e-2
QUATERNION_ANGLE_TOLERANCE = 1.0e-2


def target_positions(step: int, world: int, seed: int = DEFAULT_SEED) -> list[float]:
    """Deterministic hold/ramp/sine policy used by both backends."""
    if step < HOLD_STEPS:
        return [0.0] * 6
    elapsed = step - HOLD_STEPS
    ramp = min(1.0, elapsed / RAMP_STEPS)
    result = []
    for joint, (amplitude, frequency) in enumerate(zip(AMPLITUDES, FREQUENCIES)):
        phase = 2.0 * math.pi * ((seed * 17 + world * 31 + joint * 13) % 997) / 997.0
        result.append(ramp * amplitude * math.sin(2.0 * math.pi * frequency * elapsed / CONTROL_HZ + phase))
    return result


def _rotate(q, vector):
    qx, qy, qz, qw = q
    vx, vy, vz = vector
    tx, ty, tz = 2.0 * (qy * vz - qz * vy), 2.0 * (qz * vx - qx * vz), 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _projected_velocities(state, topology):
    result = []
    for (parent, child), axis_local in zip(topology.joint_indices, topology.axis_parent):
        axis = _rotate(state[parent][3:7], axis_local)
        relative = [state[child][10 + index] - state[parent][10 + index] for index in range(3)]
        result.append(sum(axis[index] * relative[index] for index in range(3)))
    return result


def _cpu_inputs(compiled, worlds):
    return (
        [[list(body) for body in compiled.state_y_up] for _ in range(worlds)],
        [list(compiled.inverse_mass) for _ in range(worlds)],
        [[list(row) for row in compiled.inverse_inertia_local] for _ in range(worlds)],
    )


def run_cpu_correctness(model, *, seed: int = DEFAULT_SEED, steps: int = STEPS) -> dict[str, object]:
    compiled = compile_industrial_joint_world(model)
    state, inverse_mass, inverse_inertia = _cpu_inputs(compiled, 1)
    config = JointConfig(gravity_y=-9.81, substeps=SUBSTEPS, solver_iterations=12)
    cache = None
    traces = []
    max_anchor = max_angular = max_limit = max_effort_ratio = max_velocity_ratio = 0.0
    max_quaternion_error = 0.0
    started = time.perf_counter()
    for step in range(steps):
        target = target_positions(step, 0, seed)
        result = step_joint_reference(
            state, inverse_mass, inverse_inertia, compiled.topology,
            [[0.0] * 6], [list(compiled.maximum_effort_nm)], config,
            motor_target_position=[target], stiffness=STIFFNESS, warm_start_cache=cache,
        )
        state, cache = result.state, result.warm_start_cache
        velocity = _projected_velocities(state[0], compiled.topology)
        traces.append((list(result.coordinate[0]), target))
        max_anchor = max(max_anchor, max(result.linear_error_m[0]))
        max_angular = max(max_angular, max(result.angular_error_rad[0]))
        max_limit = max(max_limit, max(result.limit_error[0]))
        max_effort_ratio = max(max_effort_ratio, *(abs(value) / (limit * config.dt) for value, limit in zip(result.motor_impulse[0], compiled.maximum_effort_nm)))
        max_velocity_ratio = max(max_velocity_ratio, *(abs(value) / limit for value, limit in zip(velocity, compiled.maximum_velocity_rad_s)))
        max_quaternion_error = max(max_quaternion_error, *(abs(math.sqrt(sum(value * value for value in body[3:7])) - 1.0) for body in state[0]))
    motion = [max(row[0][joint] for row in traces) - min(row[0][joint] for row in traces) for joint in range(6)]
    tail = traces[-TAIL_STEPS:]
    tail_rms = [math.sqrt(sum((row[0][joint] - row[1][joint]) ** 2 for row in tail) / len(tail)) for joint in range(6)]
    finite = all(math.isfinite(value) for body in state[0] for value in body)
    passed = (
        finite and max_quaternion_error <= 2.0e-5 and max_anchor <= 1.0e-3
        and max_angular <= 1.0e-3 and max_limit <= 2.0e-3
        and max_effort_ratio <= 1.000001 and max_velocity_ratio <= 1.000001
        and min(motion) >= 0.05 and max(tail_rms) <= 0.25
    )
    evidence = {
        "passed": passed, "finite": finite, "joint_motion_range_rad": motion,
        "minimum_joint_motion_range_rad": min(motion),
        "per_joint_tail_rms_tracking_error_rad": tail_rms,
        "maximum_tail_rms_tracking_error_rad": max(tail_rms),
        "maximum_joint_anchor_error_m": max_anchor,
        "maximum_locked_angular_error_rad": max_angular,
        "maximum_joint_limit_excess_rad": max_limit,
        "maximum_command_effort_ratio": max_effort_ratio,
        "maximum_projected_joint_speed_ratio": max_velocity_ratio,
        "maximum_link_quaternion_norm_error": max_quaternion_error,
        "cpu_gate_duration_seconds": time.perf_counter() - started,
    }
    if not passed:
        raise RuntimeError("industrial CPU joint gate failed: " + json.dumps(evidence, sort_keys=True))
    return evidence


def _inputs(torch, compiled, worlds: int):
    ft = lambda value: torch.tensor(value, dtype=torch.float32, device="cuda")
    it = lambda value: torch.tensor(value, dtype=torch.int64, device="cuda")
    topology = compiled.topology
    return {
        "state": ft([[list(body) for body in compiled.state_y_up]]).repeat(worlds, 1, 1),
        "inverse_mass": ft([list(compiled.inverse_mass)]).repeat(worlds, 1),
        "inverse_inertia": ft([[list(value) for value in compiled.inverse_inertia_local]]).repeat(worlds, 1, 1),
        "joint_indices": it(topology.joint_indices), "joint_types": it(topology.joint_types),
        "parent_anchor": ft(topology.parent_anchor_local), "child_anchor": ft(topology.child_anchor_local),
        "axis": ft(topology.axis_parent), "reference": ft(topology.reference_quaternion_parent_to_child),
        "lower": ft(topology.lower_limit), "upper": ft(topology.upper_limit), "damping": ft(topology.damping),
        "enabled": torch.ones(6, dtype=torch.uint8, device="cuda"),
        "target_velocity": torch.zeros((worlds, 6), dtype=torch.float32, device="cuda"),
        "effort": ft([list(compiled.maximum_effort_nm)]).repeat(worlds, 1),
        "stiffness": ft(STIFFNESS), "cache": torch.zeros((worlds, 6, 8), dtype=torch.float32, device="cuda"),
    }


def _step(inputs, target, config):
    result = joint_step(
        inputs["state"], inputs["inverse_mass"], inputs["inverse_inertia"], inputs["joint_indices"], inputs["joint_types"],
        inputs["parent_anchor"], inputs["child_anchor"], inputs["axis"], inputs["reference"], inputs["lower"], inputs["upper"],
        inputs["damping"], inputs["enabled"], inputs["target_velocity"], inputs["effort"], config,
        motor_target_position=target, stiffness=inputs["stiffness"], warm_start_cache=inputs["cache"],
    )
    inputs["state"], inputs["cache"] = result[0], result[7]
    return result


def _targets_tensor(torch, world_ids, step: int, seed: int):
    worlds = world_ids.to(dtype=torch.int64)[:, None]
    joints = torch.arange(6, dtype=torch.int64, device="cuda")[None, :]
    if step < HOLD_STEPS:
        return torch.zeros((world_ids.shape[0], 6), dtype=torch.float32, device="cuda")
    elapsed = step - HOLD_STEPS
    phase = 2.0 * math.pi * ((seed * 17 + worlds * 31 + joints * 13) % 997).float() / 997.0
    frequency = torch.tensor(FREQUENCIES, dtype=torch.float32, device="cuda")[None, :]
    amplitude = torch.tensor(AMPLITUDES, dtype=torch.float32, device="cuda")[None, :]
    return min(1.0, elapsed / RAMP_STEPS) * amplitude * torch.sin(2.0 * math.pi * frequency * elapsed / CONTROL_HZ + phase)


def _quaternion_angle_error(torch, actual, expected):
    dot = (actual * expected).sum(dim=-1).abs().clamp(max=1.0)
    return float((2.0 * torch.acos(dot)).max().item())


def _cpu_cuda_parity(torch, compiled, config, seed):
    world_ids = list(PARITY_WORLD_IDS)
    cpu_state, cpu_mass, cpu_inertia = _cpu_inputs(compiled, len(world_ids))
    cpu_cache = None
    gpu = _inputs(torch, compiled, len(world_ids))
    tensor_ids = torch.tensor(world_ids, dtype=torch.int64, device="cuda")
    maxima = {
        "state": 0.0,
        "diagnostic": 0.0,
        "cache_absolute": 0.0,
        "cache_relative": 0.0,
        "motor": 0.0,
        "quaternion_angle": 0.0,
    }
    trace = []
    for step in range(PARITY_STEPS):
        target_values = [target_positions(step, world, seed) for world in world_ids]
        cpu = step_joint_reference(
            cpu_state, cpu_mass, cpu_inertia, compiled.topology, [[0.0] * 6 for _ in world_ids],
            [list(compiled.maximum_effort_nm) for _ in world_ids], config,
            motor_target_position=target_values, stiffness=STIFFNESS, warm_start_cache=cpu_cache,
        )
        cpu_state, cpu_cache = cpu.state, cpu.warm_start_cache
        gpu_result = _step(gpu, _targets_tensor(torch, tensor_ids, step, seed), config)
        if step in TRACE_STEPS:
            expected_state = torch.tensor(cpu.state, dtype=torch.float32, device="cuda")
            maxima["state"] = max(maxima["state"], float((gpu_result[0] - expected_state).abs().max().item()))
            maxima["quaternion_angle"] = max(maxima["quaternion_angle"], _quaternion_angle_error(torch, gpu_result[0][:, :, 3:7], expected_state[:, :, 3:7]))
            for index, values in ((1, cpu.coordinate), (2, cpu.linear_error_m), (3, cpu.angular_error_rad), (4, cpu.limit_error)):
                maxima["diagnostic"] = max(maxima["diagnostic"], float((gpu_result[index] - torch.tensor(values, dtype=torch.float32, device="cuda")).abs().max().item()))
            expected_cache = torch.tensor(
                cpu.warm_start_cache, dtype=torch.float32, device="cuda"
            )
            cache_error = (gpu_result[7] - expected_cache).abs()
            # Cache rows are impulses, so their magnitudes scale with this
            # 1,120 kg arm. Preserve the raw error, but gate the dimensionless
            # symmetric relative error instead of a scale-dependent N*s bound.
            cache_scale = torch.maximum(
                gpu_result[7].abs(), expected_cache.abs()
            ).clamp_min(1.0)
            maxima["cache_absolute"] = max(
                maxima["cache_absolute"], float(cache_error.max().item())
            )
            maxima["cache_relative"] = max(
                maxima["cache_relative"],
                float((cache_error / cache_scale).max().item()),
            )
            motor_error = (gpu_result[5] - torch.tensor(cpu.motor_impulse, dtype=torch.float32, device="cuda")).abs()
            normalizer = torch.tensor(compiled.maximum_effort_nm, dtype=torch.float32, device="cuda")[None, :] * config.dt
            maxima["motor"] = max(maxima["motor"], float((motor_error / normalizer).max().item()))
            trace.append({"step": step, "world_ids": world_ids, "cpu_coordinate_rad": cpu.coordinate, "cuda_coordinate_rad": gpu_result[1].detach().cpu().tolist()})
    passed = (
        maxima["state"] <= STATE_TOLERANCE and maxima["diagnostic"] <= DIAGNOSTIC_TOLERANCE
        and maxima["cache_relative"] <= CACHE_RELATIVE_TOLERANCE
        and maxima["motor"] <= MOTOR_IMPULSE_TOLERANCE
        and maxima["quaternion_angle"] <= QUATERNION_ANGLE_TOLERANCE
    )
    if not passed:
        raise RuntimeError("industrial CPU/CUDA parity failed: " + json.dumps(maxima, sort_keys=True))
    return maxima, trace


def _gpu_projected_velocities(torch, state, inputs):
    parents = inputs["joint_indices"][:, 0]; children = inputs["joint_indices"][:, 1]
    q = state[:, parents, 3:7]
    axis = inputs["axis"][None, :, :].expand(state.shape[0], -1, -1)
    qv = q[:, :, :3]; twice = 2.0 * torch.cross(qv, axis, dim=-1)
    world_axis = axis + q[:, :, 3:4] * twice + torch.cross(qv, twice, dim=-1)
    relative = state[:, children, 10:13] - state[:, parents, 10:13]
    return (world_axis * relative).sum(dim=-1)


def _gpu_correctness(torch, compiled, config, worlds, seed):
    world_ids = torch.arange(worlds, dtype=torch.int64, device="cuda")
    inputs = _inputs(torch, compiled, worlds)
    q_min = torch.full((worlds, 6), float("inf"), dtype=torch.float32, device="cuda")
    q_max = torch.full((worlds, 6), float("-inf"), dtype=torch.float32, device="cuda")
    tail_squared = torch.zeros((worlds, 6), dtype=torch.float32, device="cuda")
    max_anchor = max_angular = max_limit = max_effort = max_speed = 0.0
    trace = []
    for step in range(STEPS):
        target = _targets_tensor(torch, world_ids, step, seed)
        result = _step(inputs, target, config)
        q_min = torch.minimum(q_min, result[1]); q_max = torch.maximum(q_max, result[1])
        if step >= STEPS - TAIL_STEPS:
            tail_squared += (result[1] - target) ** 2
        max_anchor = max(max_anchor, float(result[2].max().item()))
        max_angular = max(max_angular, float(result[3].max().item()))
        max_limit = max(max_limit, float(result[4].max().item()))
        max_effort = max(max_effort, float((result[5].abs() / (inputs["effort"] * config.dt)).max().item()))
        speed_limits = torch.tensor(compiled.maximum_velocity_rad_s, dtype=torch.float32, device="cuda")[None, :]
        max_speed = max(max_speed, float((_gpu_projected_velocities(torch, result[0], inputs).abs() / speed_limits).max().item()))
        if step in TRACE_STEPS:
            trace.append({"step": step, "world0_coordinate_rad": result[1][0].detach().cpu().tolist(), "world0_target_rad": target[0].detach().cpu().tolist()})
    quaternion_error = float((torch.linalg.vector_norm(result[0][:, :, 3:7], dim=-1) - 1.0).abs().max().item())
    metrics = {
        "finite_joint_state_and_cache": bool(torch.isfinite(result[0]).all().item()) and bool(torch.isfinite(result[7]).all().item()),
        "minimum_joint_motion_range_rad": float((q_max - q_min).min().item()),
        "maximum_tail_rms_tracking_error_rad": float(torch.sqrt(tail_squared / TAIL_STEPS).max().item()),
        "maximum_joint_anchor_error_m": max_anchor, "maximum_locked_angular_error_rad": max_angular,
        "maximum_joint_limit_excess_rad": max_limit, "maximum_command_effort_ratio": max_effort,
        "maximum_projected_joint_speed_ratio": max_speed, "maximum_link_quaternion_norm_error": quaternion_error,
    }
    passed = (
        metrics["finite_joint_state_and_cache"] and quaternion_error <= 2.0e-5 and max_anchor <= 1.0e-3
        and max_angular <= 1.0e-3 and max_limit <= 2.0e-3 and max_effort <= 1.000001
        and max_speed <= 1.000001 and metrics["minimum_joint_motion_range_rad"] >= 0.05
        and metrics["maximum_tail_rms_tracking_error_rad"] <= 0.25
    )
    if not passed:
        raise RuntimeError("industrial CUDA correctness gate failed: " + json.dumps(metrics, sort_keys=True))
    return inputs, result, metrics, trace


def _limit_targets(step, compiled):
    rows = [[0.0] * 6 for _ in range(6)]
    for case, joint in enumerate(REPRESENTABLE_LIMIT_JOINTS):
        for side in range(2):
            row = 2 * case + side
            if step < 180:
                progress = min(1.0, step / 120.0)
                smooth = progress * progress * (3.0 - 2.0 * progress)
                boundary_target = (
                    compiled.topology.lower_limit[joint] - 0.02
                    if side == 0
                    else compiled.topology.upper_limit[joint] + 0.02
                )
                rows[row][joint] = smooth * boundary_target
            else:
                # Zero is strictly inside A2/A3/A5 and gives both sides the
                # same unambiguous inward recovery command under gravity.
                rows[row][joint] = 0.0
    return rows


def _limit_probe(torch, compiled, config):
    cpu_state, cpu_mass, cpu_inertia = _cpu_inputs(compiled, 6)
    cpu_cache = None; gpu = _inputs(torch, compiled, 6)
    cpu_ever = [[False] * 6 for _ in range(6)]
    gpu_ever = torch.zeros((6, 6), dtype=torch.bool, device="cuda")
    active_mismatches = 0; observation_mismatches = 0
    max_cpu_excess = max_gpu_excess = 0.0
    boundary_cpu = boundary_gpu = None
    for step in range(240):
        target_values = _limit_targets(step, compiled)
        cpu = step_joint_reference(
            cpu_state, cpu_mass, cpu_inertia, compiled.topology, [[0.0] * 6 for _ in range(6)],
            [list(compiled.maximum_effort_nm) for _ in range(6)], config,
            motor_target_position=target_values, stiffness=STIFFNESS, warm_start_cache=cpu_cache,
        )
        cpu_state, cpu_cache = cpu.state, cpu.warm_start_cache
        gpu_result = _step(gpu, torch.tensor(target_values, dtype=torch.float32, device="cuda"), config)
        cpu_active = torch.tensor(cpu.limit_active, dtype=torch.bool, device="cuda"); gpu_active = gpu_result[6].to(dtype=torch.bool)
        active_mismatches += int((cpu_active != gpu_active).sum().item()); gpu_ever |= gpu_active
        # Threshold crossing can differ by one FP32/FP64 step. Gate stable
        # pushed/recovery observations and the accumulated activation pattern;
        # retain every transition mismatch as evidence rather than hiding it.
        if step in (150, 179, 239):
            observation_mismatches += int((cpu_active != gpu_active).sum().item())
        for world in range(6):
            for joint in range(6): cpu_ever[world][joint] |= cpu.limit_active[world][joint]
        max_cpu_excess = max(max_cpu_excess, max(max(row) for row in cpu.limit_error)); max_gpu_excess = max(max_gpu_excess, float(gpu_result[4].max().item()))
        if step == 179:
            boundary_cpu = [row[:] for row in cpu.coordinate]; boundary_gpu = gpu_result[1].detach().cpu().tolist()
    final_cpu = cpu.coordinate; final_gpu = gpu_result[1].detach().cpu().tolist()
    observed = True; recovered = True; ever_pattern_match = True
    for case, joint in enumerate(REPRESENTABLE_LIMIT_JOINTS):
        for side in range(2):
            world = 2 * case + side
            observed &= cpu_ever[world][joint] and bool(gpu_ever[world, joint].item())
            ever_pattern_match &= cpu_ever[world][joint] == bool(gpu_ever[world, joint].item())
            recovered &= final_cpu[world][joint] > boundary_cpu[world][joint] if side == 0 else final_cpu[world][joint] < boundary_cpu[world][joint]
            recovered &= final_gpu[world][joint] > boundary_gpu[world][joint] if side == 0 else final_gpu[world][joint] < boundary_gpu[world][joint]
    evidence = {
        "passed": bool(observed and ever_pattern_match and recovered and observation_mismatches == 0 and max_cpu_excess <= 2.0e-3 and max_gpu_excess <= 2.0e-3),
        "probed_joint_names": ["joint_a2", "joint_a3", "joint_a5"],
        "lower_and_upper_limit_activation_observed": bool(observed), "inward_recovery_observed": bool(recovered),
        "cpu_cuda_limit_ever_pattern_match": bool(ever_pattern_match),
        "cpu_cuda_limit_active_transition_mismatch_count": active_mismatches,
        "cpu_cuda_limit_active_observation_mismatch_count": observation_mismatches,
        "limit_active_observation_steps": [150, 179, 239],
        "maximum_cpu_limit_excess_rad": max_cpu_excess, "maximum_cuda_limit_excess_rad": max_gpu_excess,
    }
    if not evidence["passed"]:
        raise RuntimeError("industrial representable-limit gate failed: " + json.dumps(evidence, sort_keys=True))
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True); parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--srdf", type=Path, required=True); parser.add_argument("--srdf-sha256", required=True)
    parser.add_argument("--asset-id", required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=64); parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.worlds != 64 or args.steps != STEPS or args.seed != DEFAULT_SEED or any(
        len(value) != 64 for value in (args.source_sha256, args.manifest_sha256, args.srdf_sha256)
    ):
        raise ValueError("KR240 contract requires worlds=64, steps=360, seed=67, and three SHA-256 values")
    actual_hash = hashlib.sha256(args.urdf.read_bytes()).hexdigest()
    if actual_hash != args.source_sha256: raise RuntimeError("industrial URDF SHA-256 mismatch")
    actual_manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if actual_manifest_hash != args.manifest_sha256: raise RuntimeError("industrial manifest SHA-256 mismatch")
    actual_srdf_hash = hashlib.sha256(args.srdf.read_bytes()).hexdigest()
    if actual_srdf_hash != args.srdf_sha256: raise RuntimeError("industrial SRDF SHA-256 mismatch")
    model = load_industrial_joint_model(args.urdf, asset_id=args.asset_id, source_urdf_sha256=actual_hash)
    compiled = compile_industrial_joint_world(model); cpu = run_cpu_correctness(model, seed=args.seed)
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("industrial CPU gate passed; CUDA PyTorch is required") from error
    if not torch.cuda.is_available(): raise RuntimeError("industrial CPU gate passed; visible CUDA device is required")
    load_extension(); config = JointConfig(gravity_y=-9.81, substeps=SUBSTEPS, solver_iterations=12)
    parity, parity_trace = _cpu_cuda_parity(torch, compiled, config, args.seed)
    limit_probe = _limit_probe(torch, compiled, config)
    _, correctness_result, cuda_metrics, measured_trace = _gpu_correctness(torch, compiled, config, args.worlds, args.seed)
    _, replay_result, _, _ = _gpu_correctness(torch, compiled, config, args.worlds, args.seed)
    replay_error = max(float((correctness_result[0] - replay_result[0]).abs().max().item()), float((correctness_result[7] - replay_result[7]).abs().max().item()))
    if replay_error != 0.0: raise RuntimeError(f"CUDA deterministic state/cache replay failed: {replay_error}")
    isolated = _inputs(torch, compiled, 2); separate = _inputs(torch, compiled, 1)
    isolated["state"][0, 6, 12] = 2.0; isolated["cache"][0, 0, 0] = 0.5
    zero_two = torch.zeros((2, 6), dtype=torch.float32, device="cuda")
    for _ in range(8): _step(isolated, zero_two, config); _step(separate, zero_two[1:2], config)
    isolation_error = max(float((isolated["state"][1] - separate["state"][0]).abs().max().item()), float((isolated["cache"][1] - separate["cache"][0]).abs().max().item()))
    if isolation_error != 0.0: raise RuntimeError(f"CUDA world isolation failed: {isolation_error}")
    warm = _inputs(torch, compiled, 1); zero_id = torch.zeros(1, dtype=torch.int64, device="cuda")
    for step in range(4): _step(warm, _targets_tensor(torch, zero_id, step, args.seed), config)
    timed = _inputs(torch, compiled, args.worlds); world_ids = torch.arange(args.worlds, dtype=torch.int64, device="cuda")
    target_frames = [_targets_tensor(torch, world_ids, step, args.seed) for step in range(STEPS)]
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = torch.cuda.Event(enable_timing=True); finish = torch.cuda.Event(enable_timing=True); start.record()
    for target in target_frames: _step(timed, target, config)
    finish.record(); torch.cuda.synchronize(); duration = start.elapsed_time(finish) / 1000.0
    correctness = {
        **{key: value for key, value in cpu.items() if key != "passed"}, "passed": True,
        "measured_runtime_evidence": True, "synthetic": False, "asset_id": model.asset_id,
        "source_urdf_sha256": actual_hash, "calibration_class": model.calibration_class,
        "source_manifest_sha256": actual_manifest_hash, "source_srdf_sha256": actual_srdf_hash,
        "manufacturer_dynamics": False, "joint_friction_applied": False,
        "collision_geometry_applied": False, "self_collision_applied": False,
        "full_urdf_limit_range_validated": False, "unvalidated_full_range_joint_names": list(UNVALIDATED_FULL_RANGE_JOINTS),
        "cpu_cuda_trace_world_ids": list(PARITY_WORLD_IDS), "cpu_cuda_trace_sample_steps": list(TRACE_STEPS),
        "cpu_cuda_state_maximum_absolute_error": parity["state"],
        "cpu_cuda_diagnostic_maximum_absolute_error": parity["diagnostic"],
        "cpu_cuda_warm_start_cache_maximum_absolute_error": parity["cache_absolute"],
        "cpu_cuda_warm_start_cache_maximum_relative_error": parity["cache_relative"],
        "cpu_cuda_normalized_motor_impulse_error": parity["motor"],
        "cpu_cuda_quaternion_angle_error_rad": parity["quaternion_angle"],
        "cuda_physical_gates": cuda_metrics, "representable_limit_probe": limit_probe,
        "deterministic_replay_passed": replay_error == 0.0, "deterministic_replay_maximum_absolute_error": replay_error,
        "world_isolation_passed": isolation_error == 0.0, "world_isolation_maximum_absolute_error": isolation_error,
        "measured_trace": measured_trace, "parity_trace": parity_trace,
        "runtime_scope": "pinned approximate inertial topology, COM-aware joints, gravity-loaded collision-free motor response, and representable A2/A3/A5 limits",
    }
    report = BenchmarkResult(
        backend="box3d_cuda_kr240", backend_version="box3d-cuda-industrial-v1",
        workload="pinned approximate KR240 seven-body six-revolute-joint collision-free dynamics",
        contract_id=CONTRACT_ID, device=torch.cuda.get_device_name(), worlds=args.worlds,
        bodies_per_world=7, steps=STEPS, duration_seconds=duration,
        capabilities=CapabilitySet(rigid_body_integration=True, static_plane_contacts=False, dynamic_contacts=False, articulated_joints=True, continuous_collision=False, ray_queries=False, camera_rendering=False, robot_manipulation=False),
        correctness=correctness, peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
        notes=(
            "This is a bounded engineering approximation, not manufacturer dynamics or a safety claim.",
            "Meshes, contact, self-collision, payloads, manipulation, joint friction, and controller calibration are outside this gate.",
            "Principal-angle coordinates do not validate the full A1/A4/A6 multi-turn URDF ranges.",
            "No ManiSkill/PhysX parity or speedup claim is made by this result.",
        ),
    )
    write_result(args.output, report); print(json.dumps(report.to_dict(), sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
