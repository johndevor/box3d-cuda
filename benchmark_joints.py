"""Fail-closed CUDA benchmark for the matched six-revolute-joint chain."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from factory_os.benchmarks import BenchmarkResult, CapabilitySet, write_result
from factory_os.joints.contract import (
    BENCHMARK_STEPS, BODY_COUNT, CONTROL_HZ, CUDA_BACKEND, DEFAULT_SEED,
    GATE_THRESHOLDS, HOLD_STEPS, JOINT_COUNT, PHYSICS_SUBSTEPS, SPEC,
    TAIL_WINDOW_STEPS, target_positions_rad,
)
from .extension import joint_step, load_extension
from .joint_reference import JointConfig, JointTopology, REVOLUTE, step_joint_reference


PARITY_STEPS = 30
STATE_TOLERANCE = 2.0e-2
DIAGNOSTIC_TOLERANCE = 3.0e-3
CACHE_TOLERANCE = 3.0e-3


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=BENCHMARK_STEPS)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _topology() -> JointTopology:
    return JointTopology(
        joint_indices=tuple(zip(SPEC.joint_parent_body_indices, SPEC.joint_child_body_indices)),
        joint_types=(REVOLUTE,) * JOINT_COUNT,
        parent_anchor_local=SPEC.parent_anchors_m,
        child_anchor_local=SPEC.child_anchors_m,
        axis_parent=SPEC.joint_axes_canonical_y_up,
        reference_quaternion_parent_to_child=((0.0, 0.0, 0.0, 1.0),) * JOINT_COUNT,
        lower_limit=SPEC.joint_lower_limits_rad,
        upper_limit=SPEC.joint_upper_limits_rad,
        damping=SPEC.motor_damping_nms_per_rad,
        motor_enabled=(True,) * JOINT_COUNT,
        collision_enabled=(False,) * JOINT_COUNT,
    )


def _initial(worlds: int):
    positions = [(0.0, 0.0, 0.0)]
    positions.extend((0.12 + 0.24 * index, 0.0, 0.0) for index in range(JOINT_COUNT))
    one_state = [[*position, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for position in positions]
    inverse_mass = [0.0] + [1.0 / value for value in SPEC.link_masses_kg]
    inverse_inertia = []
    for body_index, diagonal in enumerate(SPEC.body_inertia_diagonal_kg_m2):
        inverse_inertia.append([0.0, 0.0, 0.0] if body_index == 0 else [1.0 / value for value in diagonal])
    return (
        [[body.copy() for body in one_state] for _ in range(worlds)],
        [inverse_mass.copy() for _ in range(worlds)],
        [[row.copy() for row in inverse_inertia] for _ in range(worlds)],
    )


def _joint_velocity(state):
    return [state[child][12] - state[parent][12] for parent, child in zip(SPEC.joint_parent_body_indices, SPEC.joint_child_body_indices)]


def _metrics(trace, final):
    tail = trace[-TAIL_WINDOW_STEPS:]
    squared = []
    for row in tail:
        squared.extend((value - target) ** 2 for value, target in zip(row["q"], row["target"]))
    limit_excess = max(max(row["limit_error"]) for row in trace)
    effort_ratio = max(max(row["effort_ratio"]) for row in trace)
    hold_drift = max(abs(value) for row in trace[:HOLD_STEPS] for value in row["q"])
    max_velocity = max(abs(value) for row in trace for value in row["qdot"])
    quaternion_error = max(abs(1.0 - math.sqrt(sum(value * value for value in body[3:7]))) for body in final.state[0])
    result = {
        "maximum_joint_limit_excess_rad": limit_excess,
        "maximum_command_effort_ratio": effort_ratio,
        "maximum_initial_hold_drift_rad": hold_drift,
        "tail_rms_tracking_error_rad": math.sqrt(sum(squared) / max(1, len(squared))),
        "maximum_absolute_joint_velocity_rad_s": max_velocity,
        "maximum_link_quaternion_norm_error": quaternion_error,
        "maximum_joint_anchor_error_m": max(final.linear_error_m[0]),
        "minimum_joint_motion_range_rad": min(
            max(row["q"][joint] for row in trace) - min(row["q"][joint] for row in trace)
            for joint in range(JOINT_COUNT)
        ),
    }
    result["passed"] = result["minimum_joint_motion_range_rad"] >= 0.02 and all(
        result["tail_rms_tracking_error_rad" if key == "maximum_tail_rms_tracking_error_rad" else key] <= threshold
        for key, threshold in GATE_THRESHOLDS.items()
    )
    return result


def run_cpu_correctness(seed=DEFAULT_SEED):
    state, inverse_mass, inverse_inertia = _initial(1)
    topology = _topology()
    config = JointConfig(gravity_y=0.0, substeps=PHYSICS_SUBSTEPS)
    trace = []
    final = None
    warm_start_cache = None
    started = time.perf_counter()
    for step in range(BENCHMARK_STEPS):
        target = [list(target_positions_rad(step, 0, seed))]
        final = step_joint_reference(
            state, inverse_mass, inverse_inertia, topology, [[0.0] * JOINT_COUNT],
            [list(SPEC.motor_effort_limits_nm)], config, steps=1,
            motor_target_position=target, stiffness=SPEC.motor_stiffness_nm_per_rad,
            warm_start_cache=warm_start_cache,
        )
        state = final.state
        warm_start_cache = final.warm_start_cache
        trace.append({
            "q": final.coordinate[0], "qdot": _joint_velocity(state[0]), "target": target[0],
            "limit_error": final.limit_error[0],
            "effort_ratio": [abs(value) / (limit * config.dt) for value, limit in zip(final.motor_impulse[0], SPEC.motor_effort_limits_nm)],
        })
    metrics = _metrics(trace, final)
    metrics["cpu_gate_duration_seconds"] = time.perf_counter() - started
    if not metrics["passed"]:
        raise RuntimeError("CPU articulated-chain correctness failed; CUDA timing refused: " + json.dumps(metrics, sort_keys=True))
    return metrics


def _tensor_inputs(torch, worlds):
    state, inverse_mass, inverse_inertia = _initial(worlds)
    topology = _topology()
    device = "cuda"
    ft = lambda value: torch.tensor(value, dtype=torch.float32, device=device)
    it = lambda value: torch.tensor(value, dtype=torch.int64, device=device)
    return {
        "state": ft(state), "inverse_mass": ft(inverse_mass), "inverse_inertia": ft(inverse_inertia),
        "joint_indices": it(topology.joint_indices), "joint_types": it(topology.joint_types),
        "parent_anchor_local": ft(topology.parent_anchor_local), "child_anchor_local": ft(topology.child_anchor_local),
        "axis_parent": ft(topology.axis_parent), "reference_quaternion_parent_to_child": ft(topology.reference_quaternion_parent_to_child),
        "lower_limit": ft(topology.lower_limit), "upper_limit": ft(topology.upper_limit),
        "damping": ft(topology.damping), "motor_enabled": torch.ones(JOINT_COUNT, dtype=torch.uint8, device=device),
        "motor_target_velocity": torch.zeros((worlds, JOINT_COUNT), dtype=torch.float32, device=device),
        "maximum_effort": ft([SPEC.motor_effort_limits_nm] * worlds), "stiffness": ft(SPEC.motor_stiffness_nm_per_rad),
        "warm_start_cache": torch.zeros((worlds, JOINT_COUNT, 8), dtype=torch.float32, device=device),
    }


def _targets(torch, worlds, step, seed):
    if step < HOLD_STEPS:
        return torch.zeros((worlds, JOINT_COUNT), dtype=torch.float32, device="cuda")
    elapsed = step - HOLD_STEPS
    ramp = min(1.0, elapsed / 60.0)
    world = torch.arange(worlds, dtype=torch.int64, device="cuda")[:, None]
    joint = torch.arange(JOINT_COUNT, dtype=torch.int64, device="cuda")[None, :]
    phase = 2.0 * math.pi * ((seed * 17 + world * 31 + joint * 13) % 997).float() / 997.0
    frequency = torch.tensor(SPEC.target_frequencies_hz, dtype=torch.float32, device="cuda")[None, :]
    amplitude = torch.tensor([.25 * min(-low, high) for low, high in zip(SPEC.joint_lower_limits_rad, SPEC.joint_upper_limits_rad)], dtype=torch.float32, device="cuda")[None, :]
    return ramp * amplitude * torch.sin(2.0 * math.pi * frequency * (elapsed / CONTROL_HZ) + phase)


def _step(inputs, target, config):
    result = joint_step(
        inputs["state"], inputs["inverse_mass"], inputs["inverse_inertia"],
        inputs["joint_indices"], inputs["joint_types"], inputs["parent_anchor_local"],
        inputs["child_anchor_local"], inputs["axis_parent"], inputs["reference_quaternion_parent_to_child"],
        inputs["lower_limit"], inputs["upper_limit"], inputs["damping"], inputs["motor_enabled"],
        inputs["motor_target_velocity"], inputs["maximum_effort"], config,
        motor_target_position=target, stiffness=inputs["stiffness"],
        warm_start_cache=inputs["warm_start_cache"],
    )
    inputs["state"] = result[0]
    inputs["warm_start_cache"] = result[7]
    return result


def main():
    args = arguments()
    if args.worlds <= 0 or args.steps != BENCHMARK_STEPS or args.warmup < 0 or args.seed < 0:
        raise ValueError("joint benchmark requires positive worlds, nonnegative seed/warmup, and exactly 720 steps")
    cpu = run_cpu_correctness(args.seed)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CPU joint gate passed, but CUDA timing requires CUDA-enabled PyTorch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CPU joint gate passed, but CUDA timing requires a visible CUDA device")
    load_extension()
    config = JointConfig(gravity_y=0.0, substeps=PHYSICS_SUBSTEPS)
    # Exact deterministic short-prefix CPU/CUDA parity.
    cpu_state, cpu_mass, cpu_inertia = _initial(1)
    topology = _topology()
    gpu = _tensor_inputs(torch, 1)
    cpu_final = None
    gpu_final = None
    cpu_warm_start_cache = None
    for step in range(PARITY_STEPS):
        target_values = [list(target_positions_rad(step, 0, args.seed))]
        cpu_final = step_joint_reference(cpu_state, cpu_mass, cpu_inertia, topology, [[0.0] * JOINT_COUNT], [list(SPEC.motor_effort_limits_nm)], config, steps=1, motor_target_position=target_values, stiffness=SPEC.motor_stiffness_nm_per_rad, warm_start_cache=cpu_warm_start_cache)
        cpu_state = cpu_final.state
        cpu_warm_start_cache = cpu_final.warm_start_cache
        gpu_final = _step(gpu, torch.tensor(target_values, dtype=torch.float32, device="cuda"), config)
    torch.cuda.synchronize()
    expected = torch.tensor(cpu_final.state, dtype=torch.float32, device="cuda")
    state_error = float((gpu_final[0] - expected).abs().max().item())
    diagnostic_error = max(
        float((gpu_final[index] - torch.tensor(values, dtype=torch.float32, device="cuda")).abs().max().item())
        for index, values in ((1, cpu_final.coordinate), (2, cpu_final.linear_error_m), (3, cpu_final.angular_error_rad), (4, cpu_final.limit_error))
    )
    expected_cache = torch.tensor(cpu_final.warm_start_cache, dtype=torch.float32, device="cuda")
    cache_error = float((gpu_final[7] - expected_cache).abs().max().item())
    parity = (
        state_error <= STATE_TOLERANCE
        and diagnostic_error <= DIAGNOSTIC_TOLERANCE
        and cache_error <= CACHE_TOLERANCE
    )
    if not parity:
        raise RuntimeError("CPU/CUDA joint parity failed; timing refused: " + json.dumps({"state_error": state_error, "diagnostic_error": diagnostic_error, "cache_error": cache_error}, sort_keys=True))
    # World isolation: perturb only lane zero and prove lane one is unchanged.
    isolated = _tensor_inputs(torch, 2)
    isolated["state"][0, 6, 12] = 2.0
    isolated["warm_start_cache"][0, 0, 0] = 0.5
    separate = _tensor_inputs(torch, 1)
    same_target = torch.zeros((2, JOINT_COUNT), dtype=torch.float32, device="cuda")
    for _ in range(8):
        _step(isolated, same_target, config)
        _step(separate, same_target[1:2], config)
    isolation_error = max(
        float((isolated["state"][1] - separate["state"][0]).abs().max().item()),
        float((isolated["warm_start_cache"][1] - separate["warm_start_cache"][0]).abs().max().item()),
    )
    if isolation_error != 0.0:
        raise RuntimeError(f"CUDA state/cache world isolation failed: {isolation_error}")
    # Low-iteration adversarial proof: explicit cache reuse must not increase
    # fixed-anchor drift or inject unbounded kinetic energy.
    stress_config = JointConfig(
        gravity_y=-9.81, substeps=PHYSICS_SUBSTEPS,
        solver_iterations=1, position_correction=0.0,
    )
    warm_stress = _tensor_inputs(torch, 1)
    cold_stress = _tensor_inputs(torch, 1)
    stress_target = torch.zeros((1, JOINT_COUNT), dtype=torch.float32, device="cuda")
    for _ in range(30):
        warm_result = _step(warm_stress, stress_target, stress_config)
        cold_stress["warm_start_cache"].zero_()
        cold_result = _step(cold_stress, stress_target, stress_config)
    warm_drift = float(warm_result[2].max().item())
    cold_drift = float(cold_result[2].max().item())
    warm_energy = float((warm_result[0][:, :, 7:13] ** 2).sum().item())
    cold_energy = float((cold_result[0][:, :, 7:13] ** 2).sum().item())
    cache_finite = bool(torch.isfinite(warm_result[7]).all().item())
    warm_start_passed = (
        cache_finite
        and warm_drift <= cold_drift + 1.0e-7
        and warm_energy <= cold_energy * 1.05 + 1.0e-6
    )
    if not warm_start_passed:
        raise RuntimeError("CUDA warm-start utility gate failed: " + json.dumps({"warm_drift": warm_drift, "cold_drift": cold_drift, "warm_energy": warm_energy, "cold_energy": cold_energy, "cache_finite": cache_finite}, sort_keys=True))
    # Warm compilation is outside the timed state.
    warm = _tensor_inputs(torch, 1)
    for step in range(args.warmup):
        _step(warm, _targets(torch, 1, step, args.seed), config)
    timed = _tensor_inputs(torch, args.worlds)
    targets = [_targets(torch, args.worlds, step, args.seed) for step in range(args.steps)]
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True); finish = torch.cuda.Event(enable_timing=True)
    start.record()
    result = None
    for target in targets:
        result = _step(timed, target, config)
    finish.record(); torch.cuda.synchronize()
    duration = start.elapsed_time(finish) / 1000.0
    replay = _tensor_inputs(torch, args.worlds)
    for target in targets:
        replay_result = _step(replay, target, config)
    torch.cuda.synchronize()
    replay_error = max(
        float((replay_result[0] - result[0]).abs().max().item()),
        float((replay_result[7] - result[7]).abs().max().item()),
    )
    if replay_error != 0.0:
        raise RuntimeError(f"CUDA deterministic state/cache replay failed: {replay_error}")
    finite = bool(torch.isfinite(result[0]).all().item())
    quaternion_norm = torch.linalg.vector_norm(result[0][:, :, 3:7], dim=-1)
    final_quaternion_error = float((quaternion_norm - 1.0).abs().max().item())
    final_anchor_error = float(result[2].max().item())
    final_limit_error = float(result[4].max().item())
    timed_gate = finite and final_quaternion_error <= GATE_THRESHOLDS["maximum_link_quaternion_norm_error"] and final_anchor_error <= GATE_THRESHOLDS["maximum_joint_anchor_error_m"] and final_limit_error <= GATE_THRESHOLDS["maximum_joint_limit_excess_rad"]
    if not timed_gate:
        raise RuntimeError("timed joint safety gate failed: " + json.dumps({"finite": finite, "quaternion_error": final_quaternion_error, "anchor_error": final_anchor_error, "limit_error": final_limit_error}, sort_keys=True))
    correctness = {
        **SPEC.metadata(seed=args.seed), **{key: value for key, value in cpu.items() if key != "passed"},
        "passed": True, "measured_runtime_evidence": True, "synthetic": False,
        "finite_joint_state": finite, "deterministic_replay_passed": replay_error == 0.0,
        "world_isolation_passed": isolation_error == 0.0, "gate_thresholds": GATE_THRESHOLDS,
        "cpu_cuda_state_maximum_absolute_error": state_error,
        "cpu_cuda_diagnostic_maximum_absolute_error": diagnostic_error,
        "cpu_cuda_warm_start_cache_maximum_absolute_error": cache_error,
        "world_isolation_maximum_absolute_error": isolation_error,
        "deterministic_replay_maximum_absolute_error": replay_error,
        "warm_start_utility_passed": warm_start_passed,
        "warm_start_final_anchor_drift_m": warm_drift,
        "cold_start_final_anchor_drift_m": cold_drift,
        "warm_start_final_velocity_energy_proxy": warm_energy,
        "cold_start_final_velocity_energy_proxy": cold_energy,
        "warm_start_cache_finite": cache_finite,
        "timed_final_quaternion_norm_error": final_quaternion_error,
        "timed_final_anchor_error_m": final_anchor_error,
        "timed_final_limit_error_rad": final_limit_error,
    }
    report = BenchmarkResult(
        backend=CUDA_BACKEND, backend_version="upstream-30c67b5+factory-v5",
        workload="fixed-base six-link revolute chain with bounded PD drives",
        contract_id=SPEC.contract_id, device=torch.cuda.get_device_name(), worlds=args.worlds,
        bodies_per_world=BODY_COUNT, steps=args.steps, duration_seconds=duration,
        capabilities=CapabilitySet(rigid_body_integration=True, static_plane_contacts=False,
            dynamic_contacts=False, articulated_joints=True, continuous_collision=False,
            ray_queries=False, camera_rendering=False, robot_manipulation=False),
        correctness=correctness, peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
        notes=("The matched performance contract contains six revolute joints and no collision shapes.",
               "Generic fixed and prismatic rows are covered by CPU/CUDA micro-contracts, not by this speedup.",
               "This does not establish KR240 mesh import, contact, self-collision, or manipulation parity."),
    )
    write_result(args.output, report)
    print(json.dumps(report.to_dict(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
