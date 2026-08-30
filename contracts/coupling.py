"""Exact Stage-7 articulated-joint plus contact comparison contract.

The workload is a fixed-base two-link arm physically pushing a free box across
a finite floor.  Motor drives are the only post-reset actuation.  Attachment,
pose writes, teleportation, and direct payload forces are forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


CONTRACT_ID = "box3d.joint-contact-pusher/v1"
CUDA_BACKEND = "box3d_cuda_stage7"
PHYSX_BACKEND = "maniskill_physx_cuda"
DEFAULT_SEED = 79
WORLDS = 4096
BODY_COUNT = 5
JOINT_COUNT = 2
PAIR_COUNT = 5
CONTROL_HZ = 120
PHYSICS_SUBSTEPS = 2
BENCHMARK_STEPS = 720
HOLD_STEPS = 60
RAMP_STEPS = 240
TAIL_WINDOW_STEPS = 120
TARGET_SCHEDULE_ID = "seeded-smoothstep-extension/v1"
CUDA_ANGULAR_DAMPING = 0.02
CONTROL_STEP_IMPULSE_DEFINITION = "norm of vector-summed native pair impulses across both physics substeps"
DRIVE_EFFORT_TRACE_DEFINITION = "final-state clamped PD effort from the shared target, stiffness, damping, and limits"

MAX_JOINT_LIMIT_EXCESS_RAD = 0.002
MAX_DRIVE_EFFORT_RATIO = 1.0 + 1.0e-6
MAX_JOINT_ANCHOR_ERROR_M = 0.003
MAX_PENETRATION_M = 0.005
MAX_QUATERNION_NORM_ERROR = 2.0e-5
MIN_LINK2_PAYLOAD_CONTACT_FRAMES = 3
MIN_FLOOR_PAYLOAD_CONTACT_FRAMES = 480
MIN_PAYLOAD_FORWARD_DISPLACEMENT_M = 0.10
MAX_UNCOMMANDED_ENERGY_INCREASE_J = 0.10
MIN_FRICTION_NEGATIVE_CONTROL_TAIL_SPEED_DELTA_MPS = 0.02

MIN_CONTACT_STATE_AGREEMENT_RATIO = 0.99
MAX_JOINT_POSITION_ERROR_RAD = 0.015
MAX_JOINT_VELOCITY_ERROR_RAD_S = 0.15
MAX_BODY_POSITION_ERROR_M = 0.008
MAX_BODY_ORIENTATION_ERROR_RAD = 0.02
MAX_BODY_VELOCITY_ERROR_MPS = 0.10
MAX_BODY_ANGULAR_VELOCITY_ERROR_RAD_S = 0.15
MAX_DRIVE_EFFORT_ERROR_NM = 0.20
MAX_JOINT_TARGET_ERROR_RAD = 1.0e-7
MAX_PAIR_CONTACT_IMPULSE_MAGNITUDE_ERROR_NS = 0.08
MAX_PAYLOAD_DISPLACEMENT_ERROR_M = 0.01


def _box_inertia(mass: float, half: Sequence[float]) -> tuple[float, float, float]:
    hx, hy, hz = half
    return (
        mass * (hy * hy + hz * hz) / 3.0,
        mass * (hx * hx + hz * hz) / 3.0,
        mass * (hx * hx + hy * hy) / 3.0,
    )


@dataclass(frozen=True)
class JointContactPusherSpec:
    contract_id: str = CONTRACT_ID
    body_names: tuple[str, ...] = ("floor", "fixed_base", "link1", "link2", "payload")
    body_half_extents_m: tuple[tuple[float, float, float], ...] = (
        (3.0, 0.05, 1.0),
        (0.10, 0.14, 0.12),
        (0.35, 0.06, 0.10),
        (0.30, 0.06, 0.10),
        (0.14, 0.14, 0.14),
    )
    body_masses_kg: tuple[float, ...] = (1.0, 1.0, 2.0, 1.5, 1.0)
    arm_base_center_m: tuple[float, float, float] = (-0.80, 0.14, 0.0)
    fixed_body_indices: tuple[int, ...] = (0, 1)
    joint_parent_body_indices: tuple[int, ...] = (1, 2)
    joint_child_body_indices: tuple[int, ...] = (2, 3)
    joint_axes_canonical_y_up: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
    )
    joint_lower_limits_rad: tuple[float, ...] = (-1.20, -1.50)
    joint_upper_limits_rad: tuple[float, ...] = (1.20, 1.50)
    parent_anchors_m: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (0.35, 0.0, 0.0),
    )
    child_anchors_m: tuple[tuple[float, float, float], ...] = (
        (-0.35, 0.0, 0.0),
        (-0.30, 0.0, 0.0),
    )
    initial_joint_positions_nominal_rad: tuple[float, ...] = (0.75, -1.15)
    final_joint_targets_rad: tuple[float, ...] = (0.0, 0.0)
    drive_stiffness_nm_per_rad: tuple[float, ...] = (35.0, 25.0)
    drive_damping_nms_per_rad: tuple[float, ...] = (4.0, 3.0)
    drive_effort_limits_nm: tuple[float, ...] = (18.0, 12.0)
    contact_pair_indices: tuple[tuple[int, int], ...] = (
        (0, 4), (0, 2), (0, 3), (2, 4), (3, 4)
    )
    contact_pair_roles: tuple[str, ...] = (
        "floor_payload",
        "floor_link1",
        "floor_link2",
        "link1_payload",
        "link2_payload",
    )
    gravity_xyz_mps2: tuple[float, float, float] = (0.0, -9.81, 0.0)
    friction: float = 0.60
    restitution: float = 0.0
    # PhysX defines this per shape; two authored shapes therefore generate a
    # pair candidate at twice this distance.
    contact_generation_offset_m: float = 0.0015
    contact_rest_offset_m: float = 0.0
    contact_slop_m: float = 0.001
    solver_iterations: int = 8

    @property
    def pair_contact_generation_distance_m(self) -> float:
        return 2.0 * self.contact_generation_offset_m

    def __post_init__(self) -> None:
        if len(self.body_names) != BODY_COUNT or len(self.body_half_extents_m) != BODY_COUNT:
            raise ValueError("Stage-7 requires exactly five bodies")
        if self.joint_parent_body_indices != (1, 2) or self.joint_child_body_indices != (2, 3):
            raise ValueError("Stage-7 serial joint topology is fixed")
        if len(self.contact_pair_indices) != PAIR_COUNT:
            raise ValueError("Stage-7 requires the five explicit contact pairs")
        if any(low >= high for low, high in zip(self.joint_lower_limits_rad, self.joint_upper_limits_rad)):
            raise ValueError("joint limits must be ordered")

    @property
    def body_inverse_masses_per_kg(self) -> tuple[float, ...]:
        return tuple(
            0.0 if index in self.fixed_body_indices else 1.0 / mass
            for index, mass in enumerate(self.body_masses_kg)
        )

    @property
    def body_inertia_diagonal_kg_m2(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            _box_inertia(mass, half)
            for mass, half in zip(self.body_masses_kg, self.body_half_extents_m)
        )

    def metadata(self, *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
        return {
            "scenario_seed": seed,
            "control_hz": CONTROL_HZ,
            "physics_substeps": PHYSICS_SUBSTEPS,
            "benchmark_steps": BENCHMARK_STEPS,
            "body_names": list(self.body_names),
            "body_half_extents_m": [list(value) for value in self.body_half_extents_m],
            "body_masses_kg": list(self.body_masses_kg),
            "arm_base_center_m": list(self.arm_base_center_m),
            "body_inverse_masses_per_kg": list(self.body_inverse_masses_per_kg),
            "body_inertia_diagonal_kg_m2": [list(value) for value in self.body_inertia_diagonal_kg_m2],
            "fixed_body_indices": list(self.fixed_body_indices),
            "joint_types": ["revolute", "revolute"],
            "joint_parent_body_indices": list(self.joint_parent_body_indices),
            "joint_child_body_indices": list(self.joint_child_body_indices),
            "joint_axes_canonical_y_up": [list(value) for value in self.joint_axes_canonical_y_up],
            "joint_lower_limits_rad": list(self.joint_lower_limits_rad),
            "joint_upper_limits_rad": list(self.joint_upper_limits_rad),
            "parent_anchors_m": [list(value) for value in self.parent_anchors_m],
            "child_anchors_m": [list(value) for value in self.child_anchors_m],
            "initial_joint_positions_nominal_rad": list(self.initial_joint_positions_nominal_rad),
            "final_joint_targets_rad": list(self.final_joint_targets_rad),
            "drive_stiffness_nm_per_rad": list(self.drive_stiffness_nm_per_rad),
            "drive_damping_nms_per_rad": list(self.drive_damping_nms_per_rad),
            "drive_effort_limits_nm": list(self.drive_effort_limits_nm),
            "gravity_xyz_mps2_canonical_y_up": list(self.gravity_xyz_mps2),
            "friction": self.friction,
            "restitution": self.restitution,
            "contact_generation_offset_m": self.contact_generation_offset_m,
            "pair_contact_generation_distance_m": self.pair_contact_generation_distance_m,
            "contact_rest_offset_m": self.contact_rest_offset_m,
            "contact_slop_m": self.contact_slop_m,
            "solver_iterations": self.solver_iterations,
            "solver_warm_start": True,
            "contact_pair_order": [list(pair) for pair in self.contact_pair_indices],
            "contact_pair_roles": list(self.contact_pair_roles),
            "collision_filter_allowed_pairs": [list(pair) for pair in self.contact_pair_indices],
            "adjacent_link_collision_enabled": False,
            "self_collision_enabled": False,
            "target_schedule_id": TARGET_SCHEDULE_ID,
            "hold_steps": HOLD_STEPS,
            "ramp_steps": RAMP_STEPS,
            "tail_window_steps": TAIL_WINDOW_STEPS,
            "actuation_policy": "joint_motor_drives_only",
            "payload_attachment": False,
            "post_reset_pose_writes": False,
            "direct_payload_force": False,
            "pair_contact_definition": "SAT signed separation <= contact_slop_m at the end of the control step",
            "pair_contact_impulse_definition": CONTROL_STEP_IMPULSE_DEFINITION,
            "drive_effort_trace_definition": DRIVE_EFFORT_TRACE_DEFINITION,
            "output_layouts": {
                "joint_positions_rad": ["world", "joint"],
                "joint_velocities_rad_s": ["world", "joint"],
                "drive_efforts_nm": ["world", "joint"],
                "body_poses_p_wxyz": ["world", "body", "position_xyz+quaternion_wxyz"],
                "body_linear_velocities_mps": ["world", "body", "xyz"],
                "body_angular_velocities_rad_s": ["world", "body", "xyz"],
                "pair_contact": ["world", "contact_pair"],
                "pair_contact_impulse_magnitude_ns": ["world", "contact_pair"],
            },
            "contact_cache_is_matched_output": False,
            "contact_cache_note": "PhysX internal warm-start cache is not publicly observable; physical impulses are matched instead",
            "friction_negative_control": {
                "friction": 0.0,
                "restitution": self.restitution,
                "steps": BENCHMARK_STEPS,
                "timed": False,
                "comparison": "payload_tail_speed_delta_mps",
            },
            "energy_diagnostic": "maximum_positive_delta_mechanical_energy_minus_integrated_drive_work",
        }


SPEC = JointContactPusherSpec()

GATE_THRESHOLDS = {
    "maximum_joint_limit_excess_rad": MAX_JOINT_LIMIT_EXCESS_RAD,
    "maximum_drive_effort_ratio": MAX_DRIVE_EFFORT_RATIO,
    "maximum_joint_anchor_error_m": MAX_JOINT_ANCHOR_ERROR_M,
    "maximum_penetration_m": MAX_PENETRATION_M,
    "maximum_quaternion_norm_error": MAX_QUATERNION_NORM_ERROR,
    "minimum_link2_payload_contact_frames": MIN_LINK2_PAYLOAD_CONTACT_FRAMES,
    "minimum_floor_payload_contact_frames": MIN_FLOOR_PAYLOAD_CONTACT_FRAMES,
    "minimum_payload_forward_displacement_m": MIN_PAYLOAD_FORWARD_DISPLACEMENT_M,
    "maximum_uncommanded_energy_increase_j": MAX_UNCOMMANDED_ENERGY_INCREASE_J,
    "minimum_friction_negative_control_tail_speed_delta_mps": MIN_FRICTION_NEGATIVE_CONTROL_TAIL_SPEED_DELTA_MPS,
}

PARITY_THRESHOLDS = {
    "minimum_contact_state_agreement_ratio": MIN_CONTACT_STATE_AGREEMENT_RATIO,
    "maximum_joint_position_error_rad": MAX_JOINT_POSITION_ERROR_RAD,
    "maximum_joint_velocity_error_rad_s": MAX_JOINT_VELOCITY_ERROR_RAD_S,
    "maximum_body_position_error_m": MAX_BODY_POSITION_ERROR_M,
    "maximum_body_orientation_error_rad": MAX_BODY_ORIENTATION_ERROR_RAD,
    "maximum_body_velocity_error_mps": MAX_BODY_VELOCITY_ERROR_MPS,
    "maximum_body_angular_velocity_error_rad_s": MAX_BODY_ANGULAR_VELOCITY_ERROR_RAD_S,
    "maximum_drive_effort_error_nm": MAX_DRIVE_EFFORT_ERROR_NM,
    "maximum_joint_target_error_rad": MAX_JOINT_TARGET_ERROR_RAD,
    "maximum_pair_contact_impulse_magnitude_error_ns": MAX_PAIR_CONTACT_IMPULSE_MAGNITUDE_ERROR_NS,
    "maximum_payload_displacement_error_m": MAX_PAYLOAD_DISPLACEMENT_ERROR_M,
}

CUDA_SOLVER_CONFIGURATION = {
    "type": "box3d_projected_pgs",
    "unified_velocity_iterations": SPEC.solver_iterations,
    "warm_start_factor": 0.8,
    "split_position_repair_iterations": 8,
    "two_revolute_articulation_projection": True,
    "angular_damping": CUDA_ANGULAR_DAMPING,
    "sleep_enabled": False,
    "contact_generation_offset_m": SPEC.contact_generation_offset_m,
    "pair_contact_generation_distance_m": SPEC.pair_contact_generation_distance_m,
    "contact_rest_offset_m": SPEC.contact_rest_offset_m,
    "position_repair_slop_m": SPEC.contact_rest_offset_m,
}

PHYSX_PGS_SOLVER_CONFIGURATION = {
    "type": "physx_pgs",
    "position_iterations": SPEC.solver_iterations,
    "velocity_iterations": 2,
    "enhanced_determinism": True,
    "angular_damping": CUDA_ANGULAR_DAMPING,
    "sleep_enabled": False,
    "contact_generation_offset_m": SPEC.contact_generation_offset_m,
    "pair_contact_generation_distance_m": SPEC.pair_contact_generation_distance_m,
    "contact_rest_offset_m": SPEC.contact_rest_offset_m,
}


def _seeded_unit(world_index: int, lane: int, seed: int) -> float:
    if world_index < 0 or seed < 0:
        raise ValueError("world and seed must be nonnegative")
    return 2.0 * ((seed * 101 + world_index * 47 + lane * 29) % 997) / 996.0 - 1.0


def initial_joint_positions_rad(world_index: int, seed: int = DEFAULT_SEED) -> tuple[float, float]:
    return tuple(
        nominal + 0.01 * _seeded_unit(world_index, joint, seed)
        for joint, nominal in enumerate(SPEC.initial_joint_positions_nominal_rad)
    )  # type: ignore[return-value]


def initial_payload_center_m(world_index: int, seed: int = DEFAULT_SEED) -> tuple[float, float, float]:
    return (0.40 + 0.01 * _seeded_unit(world_index, 3, seed), 0.14, 0.0)


def target_positions_rad(
    control_step: int, world_index: int, seed: int = DEFAULT_SEED
) -> tuple[float, float]:
    if control_step < 0:
        raise ValueError("control step must be nonnegative")
    initial = initial_joint_positions_rad(world_index, seed)
    scale = target_scale(control_step)
    return tuple(start * scale for start in initial)  # type: ignore[return-value]


def target_scale(control_step: int) -> float:
    """Shared scalar schedule so both GPU harnesses issue bit-identical targets."""

    if control_step < 0:
        raise ValueError("control step must be nonnegative")
    if control_step < HOLD_STEPS:
        return 1.0
    fraction = min(1.0, (control_step - HOLD_STEPS) / RAMP_STEPS)
    smooth = fraction * fraction * (3.0 - 2.0 * fraction)
    return 1.0 - smooth


def target_batch_rad(worlds: int, control_step: int, seed: int = DEFAULT_SEED) -> list[list[float]]:
    if worlds <= 0:
        raise ValueError("world count must be positive")
    return [list(target_positions_rad(control_step, world, seed)) for world in range(worlds)]


def _numeric(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"coupling evidence missing finite numeric result: {name}")
    return float(value)


def validate_coupling_report(report: Mapping[str, Any], *, backend: str) -> None:
    if report.get("backend") != backend or report.get("contract_id") != CONTRACT_ID:
        raise RuntimeError("coupling report backend or contract mismatch")
    if report.get("worlds") != WORLDS or report.get("bodies_per_world") != BODY_COUNT or report.get("steps") != BENCHMARK_STEPS:
        raise RuntimeError("coupling report must use 4096 worlds, five bodies, and 720 control steps")
    capabilities = report.get("capabilities", {})
    if capabilities.get("articulated_joints") is not True or capabilities.get("rigid_body_contacts") is not True:
        raise RuntimeError("coupling report must measure both joints and contacts")
    correctness = report.get("correctness", {})
    if correctness.get("passed") is not True or correctness.get("measured_runtime_evidence") is not True:
        raise RuntimeError("coupling correctness requires measured passing runtime evidence")
    if correctness.get("synthetic") is not False:
        raise RuntimeError("synthetic coupling fixtures cannot establish parity")
    for key, expected in SPEC.metadata().items():
        if correctness.get(key) != expected:
            raise RuntimeError(f"coupling report requires exact {key}")
    if correctness.get("gate_thresholds") != GATE_THRESHOLDS:
        raise RuntimeError("coupling gate thresholds differ from the fixed contract")
    expected_solver = (
        PHYSX_PGS_SOLVER_CONFIGURATION
        if backend == PHYSX_BACKEND
        else CUDA_SOLVER_CONFIGURATION
    )
    if report.get("solver_configuration") != expected_solver:
        raise RuntimeError("coupling report solver configuration differs from the fixed backend profile")
    for key in (
        "finite_joint_and_body_state", "normalized_body_quaternions",
        "deterministic_replay_passed", "world_isolation_passed",
        "joint_limits_respected", "drive_effort_clamped",
        "real_contact_impulses_observed", "friction_negative_control_passed",
        "no_attachment_or_teleportation", "no_hidden_force_injection",
    ):
        if correctness.get(key) is not True:
            raise RuntimeError(f"coupling evidence failed {key}")
    maximums = {
        "maximum_joint_limit_excess_rad": MAX_JOINT_LIMIT_EXCESS_RAD,
        "maximum_drive_effort_ratio": MAX_DRIVE_EFFORT_RATIO,
        "maximum_joint_anchor_error_m": MAX_JOINT_ANCHOR_ERROR_M,
        "maximum_penetration_m": MAX_PENETRATION_M,
        "maximum_quaternion_norm_error": MAX_QUATERNION_NORM_ERROR,
        "maximum_uncommanded_energy_increase_j": MAX_UNCOMMANDED_ENERGY_INCREASE_J,
    }
    minimums = {
        "link2_payload_contact_frames": MIN_LINK2_PAYLOAD_CONTACT_FRAMES,
        "floor_payload_contact_frames": MIN_FLOOR_PAYLOAD_CONTACT_FRAMES,
        "payload_forward_displacement_m": MIN_PAYLOAD_FORWARD_DISPLACEMENT_M,
        "friction_negative_control_tail_speed_delta_mps": MIN_FRICTION_NEGATIVE_CONTROL_TAIL_SPEED_DELTA_MPS,
    }
    for key, bound in maximums.items():
        if _numeric(correctness.get(key), key) > bound:
            raise RuntimeError(f"coupling evidence exceeded {key}")
    for key, bound in minimums.items():
        if _numeric(correctness.get(key), key) < bound:
            raise RuntimeError(f"coupling evidence fell below {key}")
    if _numeric(report.get("world_steps_per_second"), "world_steps_per_second") <= 0.0:
        raise RuntimeError("coupling timing must be positive")


def validate_coupling_contract_speedup(
    reports: Sequence[Mapping[str, Any]], comparison: Mapping[str, Any]
) -> Mapping[str, Any]:
    matches = [item for item in reports if item.get("contract_id") == CONTRACT_ID]
    if len(matches) != 2 or {item.get("backend") for item in matches} != {PHYSX_BACKEND, CUDA_BACKEND}:
        raise RuntimeError("coupling speedup requires exactly one measured report per backend")
    by_backend = {item["backend"]: item for item in matches}
    validate_coupling_report(by_backend[PHYSX_BACKEND], backend=PHYSX_BACKEND)
    validate_coupling_report(by_backend[CUDA_BACKEND], backend=CUDA_BACKEND)
    rows = [item for item in comparison.get("speedups", ()) if item.get("contract_id") == CONTRACT_ID]
    if len(rows) != 1:
        raise RuntimeError("coupling comparison requires exactly one matched speedup row")
    row = rows[0]
    if row.get("baseline") != PHYSX_BACKEND or row.get("candidate") != CUDA_BACKEND:
        raise RuntimeError("coupling speedup direction must be PhysX to Box3D CUDA")
    parity = row.get("output_parity", {})
    if parity.get("measured") is not True:
        raise RuntimeError("coupling timing requires measured output parity")
    if parity.get("thresholds") != PARITY_THRESHOLDS:
        raise RuntimeError("coupling output parity thresholds differ from the contract")
    if parity.get("sampled_steps") != list(range(0, BENCHMARK_STEPS, 12)) + [BENCHMARK_STEPS - 1]:
        raise RuntimeError("coupling parity must use the fixed sampled control steps")
    if parity.get("sampled_worlds") != 64:
        raise RuntimeError("coupling parity must sample exactly 64 deterministic worlds")
    if _numeric(parity.get("contact_state_agreement_ratio"), "contact_state_agreement_ratio") < MIN_CONTACT_STATE_AGREEMENT_RATIO:
        raise RuntimeError("coupling contact-state parity failed")
    maximums = {
        "maximum_joint_position_error_rad": MAX_JOINT_POSITION_ERROR_RAD,
        "maximum_joint_velocity_error_rad_s": MAX_JOINT_VELOCITY_ERROR_RAD_S,
        "maximum_body_position_error_m": MAX_BODY_POSITION_ERROR_M,
        "maximum_body_orientation_error_rad": MAX_BODY_ORIENTATION_ERROR_RAD,
        "maximum_body_velocity_error_mps": MAX_BODY_VELOCITY_ERROR_MPS,
        "maximum_body_angular_velocity_error_rad_s": MAX_BODY_ANGULAR_VELOCITY_ERROR_RAD_S,
        "maximum_drive_effort_error_nm": MAX_DRIVE_EFFORT_ERROR_NM,
        "maximum_joint_target_error_rad": MAX_JOINT_TARGET_ERROR_RAD,
        "maximum_pair_contact_impulse_magnitude_error_ns": MAX_PAIR_CONTACT_IMPULSE_MAGNITUDE_ERROR_NS,
        "maximum_payload_displacement_error_m": MAX_PAYLOAD_DISPLACEMENT_ERROR_M,
    }
    for key, maximum in maximums.items():
        if _numeric(parity.get(key), key) > maximum:
            raise RuntimeError(f"coupling output parity exceeded {key}")
    if parity.get("passed") is not True:
        raise RuntimeError("coupling timing is forbidden until measured output parity passes")
    _numeric(row.get("world_step_speedup"), "world_step_speedup")
    return row


__all__ = [name for name in globals() if name.isupper()] + [
    "JointContactPusherSpec", "SPEC", "initial_joint_positions_rad",
    "initial_payload_center_m", "target_positions_rad", "target_batch_rad", "target_scale",
    "validate_coupling_report", "validate_coupling_contract_speedup",
]
