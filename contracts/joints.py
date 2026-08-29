"""Backend-neutral Stage-5 articulated-chain comparison contract.

The first matched workload is intentionally smaller than the KR240 asset. It
isolates reduced-coordinate articulation, limits, and motor drives without
contacts, meshes, or robot-controller policy code obscuring parity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


CONTRACT_ID = "box3d.articulated-chain/v1"
CUDA_BACKEND = "box3d_cuda_stage5"
PHYSX_BACKEND = "maniskill_physx_cuda"
DEFAULT_SEED = 53
WORLDS = 4096
BODY_COUNT = 7
JOINT_COUNT = 6
CONTROL_HZ = 120
PHYSICS_SUBSTEPS = 2
BENCHMARK_STEPS = 720
HOLD_STEPS = 60
TARGET_RAMP_STEPS = 60
TAIL_WINDOW_STEPS = 120
TARGET_SCHEDULE_ID = "seeded-sine-pd/v1"

MAX_JOINT_LIMIT_EXCESS_RAD = 0.002
MAX_COMMAND_EFFORT_RATIO = 1.0 + 1.0e-6
MAX_INITIAL_HOLD_DRIFT_RAD = 2.0e-4
MAX_TAIL_RMS_TRACKING_ERROR_RAD = 0.20
MAX_ABSOLUTE_JOINT_VELOCITY_RAD_S = 8.0
MAX_LINK_QUATERNION_NORM_ERROR = 2.0e-5
MAX_JOINT_ANCHOR_ERROR_M = 0.003


def _box_inertia(mass: float, half: Sequence[float]) -> tuple[float, float, float]:
    hx, hy, hz = half
    return (
        mass * (hy * hy + hz * hz) / 3.0,
        mass * (hx * hx + hz * hz) / 3.0,
        mass * (hx * hx + hy * hy) / 3.0,
    )


@dataclass(frozen=True)
class JointChainSpec:
    contract_id: str = CONTRACT_ID
    base_half_extents_m: tuple[float, float, float] = (0.05, 0.05, 0.05)
    link_half_extents_m: tuple[tuple[float, float, float], ...] = ((0.12, 0.10, 0.10),) * JOINT_COUNT
    link_masses_kg: tuple[float, ...] = (3.0, 2.5, 2.0, 1.5, 1.0, 0.75)
    joint_parent_body_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    joint_child_body_indices: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    joint_types: tuple[str, ...] = ("revolute",) * JOINT_COUNT
    joint_axes_canonical_y_up: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 1.0),) * JOINT_COUNT
    joint_lower_limits_rad: tuple[float, ...] = (-1.20, -1.00, -1.10, -0.90, -1.30, -0.80)
    joint_upper_limits_rad: tuple[float, ...] = (1.20, 1.00, 1.10, 0.90, 1.30, 0.80)
    motor_stiffness_nm_per_rad: tuple[float, ...] = (2.0, 1.8, 1.6, 1.4, 1.2, 1.0)
    motor_damping_nms_per_rad: tuple[float, ...] = (0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
    motor_effort_limits_nm: tuple[float, ...] = (3.0, 2.5, 2.0, 1.6, 1.2, 1.0)
    target_frequencies_hz: tuple[float, ...] = (0.05, 0.06, 0.07, 0.08, 0.09, 0.10)

    def __post_init__(self) -> None:
        fields = (
            self.link_half_extents_m, self.link_masses_kg,
            self.joint_parent_body_indices, self.joint_child_body_indices,
            self.joint_types, self.joint_axes_canonical_y_up,
            self.joint_lower_limits_rad, self.joint_upper_limits_rad,
            self.motor_stiffness_nm_per_rad, self.motor_damping_nms_per_rad,
            self.motor_effort_limits_nm, self.target_frequencies_hz,
        )
        if any(len(values) != JOINT_COUNT for values in fields):
            raise ValueError("articulated-chain fields must contain six joints/links")
        if tuple(self.joint_parent_body_indices) != tuple(range(JOINT_COUNT)):
            raise ValueError("joint parents must form the fixed serial topology")
        if tuple(self.joint_child_body_indices) != tuple(range(1, BODY_COUNT)):
            raise ValueError("joint children must form the fixed serial topology")
        if any(kind != "revolute" for kind in self.joint_types):
            raise ValueError("v1 matched workload supports revolute joints only")
        if any(low >= high for low, high in zip(self.joint_lower_limits_rad, self.joint_upper_limits_rad)):
            raise ValueError("joint limits must be ordered")
        if any(value <= 0.0 for value in self.link_masses_kg + self.motor_effort_limits_nm):
            raise ValueError("dynamic masses and effort limits must be positive")

    @property
    def body_masses_kg(self) -> tuple[float, ...]:
        # The base's nominal inertia is explicit even though its fixed flag
        # makes its solver inverse mass exactly zero in both backends.
        return (1.0,) + self.link_masses_kg

    @property
    def body_inverse_masses_per_kg(self) -> tuple[float, ...]:
        return (0.0,) + tuple(1.0 / mass for mass in self.link_masses_kg)

    @property
    def body_inertia_diagonal_kg_m2(self) -> tuple[tuple[float, float, float], ...]:
        return ((1.0, 1.0, 1.0),) + tuple(
            _box_inertia(mass, half)
            for mass, half in zip(self.link_masses_kg, self.link_half_extents_m)
        )

    @property
    def parent_anchors_m(self) -> tuple[tuple[float, float, float], ...]:
        return ((0.0, 0.0, 0.0),) + tuple((0.12, 0.0, 0.0) for _ in range(JOINT_COUNT - 1))

    @property
    def child_anchors_m(self) -> tuple[tuple[float, float, float], ...]:
        return tuple((-0.12, 0.0, 0.0) for _ in range(JOINT_COUNT))

    def metadata(self, *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
        return {
            "scenario_seed": seed,
            "control_hz": CONTROL_HZ,
            "physics_substeps": PHYSICS_SUBSTEPS,
            "gravity_xyz_mps2_canonical_y_up": [0.0, 0.0, 0.0],
            "joint_types": list(self.joint_types),
            "joint_parent_body_indices": list(self.joint_parent_body_indices),
            "joint_child_body_indices": list(self.joint_child_body_indices),
            "joint_axes_canonical_y_up": [list(value) for value in self.joint_axes_canonical_y_up],
            "joint_lower_limits_rad": list(self.joint_lower_limits_rad),
            "joint_upper_limits_rad": list(self.joint_upper_limits_rad),
            "parent_anchors_m": [list(value) for value in self.parent_anchors_m],
            "child_anchors_m": [list(value) for value in self.child_anchors_m],
            "base_half_extents_m": list(self.base_half_extents_m),
            "body_masses_kg": list(self.body_masses_kg),
            "body_inverse_masses_per_kg": list(self.body_inverse_masses_per_kg),
            "body_inertia_diagonal_kg_m2": [list(value) for value in self.body_inertia_diagonal_kg_m2],
            "fixed_body_indices": [0],
            "link_half_extents_m": [list(value) for value in self.link_half_extents_m],
            "motor_stiffness_nm_per_rad": list(self.motor_stiffness_nm_per_rad),
            "motor_damping_nms_per_rad": list(self.motor_damping_nms_per_rad),
            "motor_effort_limits_nm": list(self.motor_effort_limits_nm),
            "target_frequencies_hz": list(self.target_frequencies_hz),
            "target_schedule_id": TARGET_SCHEDULE_ID,
            "hold_steps": HOLD_STEPS,
            "target_ramp_steps": TARGET_RAMP_STEPS,
            "tail_window_steps": TAIL_WINDOW_STEPS,
            "collision_policy": "no_collision_shapes",
            "self_collision_enabled": False,
            "solver_warm_start": True,
        }


SPEC = JointChainSpec()


GATE_THRESHOLDS = {
    "maximum_joint_limit_excess_rad": MAX_JOINT_LIMIT_EXCESS_RAD,
    "maximum_command_effort_ratio": MAX_COMMAND_EFFORT_RATIO,
    "maximum_initial_hold_drift_rad": MAX_INITIAL_HOLD_DRIFT_RAD,
    "maximum_tail_rms_tracking_error_rad": MAX_TAIL_RMS_TRACKING_ERROR_RAD,
    "maximum_absolute_joint_velocity_rad_s": MAX_ABSOLUTE_JOINT_VELOCITY_RAD_S,
    "maximum_link_quaternion_norm_error": MAX_LINK_QUATERNION_NORM_ERROR,
    "maximum_joint_anchor_error_m": MAX_JOINT_ANCHOR_ERROR_M,
}


def phase_radians(world_index: int, joint_index: int, seed: int = DEFAULT_SEED) -> float:
    if world_index < 0 or not 0 <= joint_index < JOINT_COUNT or seed < 0:
        raise ValueError("world/joint/seed indices must be in range")
    return 2.0 * math.pi * ((seed * 17 + world_index * 31 + joint_index * 13) % 997) / 997.0


def target_positions_rad(control_step: int, world_index: int, seed: int = DEFAULT_SEED) -> tuple[float, ...]:
    if control_step < 0:
        raise ValueError("control_step cannot be negative")
    if control_step < HOLD_STEPS:
        return (0.0,) * JOINT_COUNT
    elapsed_steps = control_step - HOLD_STEPS
    ramp = min(1.0, elapsed_steps / TARGET_RAMP_STEPS)
    elapsed_seconds = elapsed_steps / CONTROL_HZ
    result = []
    for joint in range(JOINT_COUNT):
        amplitude = 0.25 * min(-SPEC.joint_lower_limits_rad[joint], SPEC.joint_upper_limits_rad[joint])
        result.append(
            ramp * amplitude * math.sin(
                2.0 * math.pi * SPEC.target_frequencies_hz[joint] * elapsed_seconds
                + phase_radians(world_index, joint, seed)
            )
        )
    return tuple(result)


def target_batch_rad(worlds: int, control_step: int, seed: int = DEFAULT_SEED) -> list[list[float]]:
    if worlds <= 0:
        raise ValueError("worlds must be positive")
    return [list(target_positions_rad(control_step, world, seed)) for world in range(worlds)]


def _numeric(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"joint report missing finite numeric correctness result: {name}")
    return float(value)


def validate_joint_report(report: Mapping[str, Any], *, backend: str) -> None:
    if report.get("backend") != backend or report.get("contract_id") != CONTRACT_ID:
        raise RuntimeError("joint report backend or contract_id mismatch")
    if report.get("worlds") != WORLDS or report.get("bodies_per_world") != BODY_COUNT or report.get("steps") != BENCHMARK_STEPS:
        raise RuntimeError("joint report must use 4096 worlds, seven bodies, and 720 control steps")
    if report.get("capabilities", {}).get("articulated_joints") is not True:
        raise RuntimeError("joint report must measure articulated joints")
    correctness = report.get("correctness", {})
    if correctness.get("passed") is not True or correctness.get("measured_runtime_evidence") is not True:
        raise RuntimeError("joint correctness must pass with measured runtime evidence")
    if correctness.get("synthetic") is not False:
        raise RuntimeError("synthetic fixtures cannot count as measured joint parity")
    for key, expected in SPEC.metadata().items():
        if correctness.get(key) != expected:
            raise RuntimeError(f"joint report requires exact {key}")
    if correctness.get("gate_thresholds") != GATE_THRESHOLDS:
        raise RuntimeError("joint report gate thresholds differ from the fixed contract")
    for key in ("finite_joint_state", "deterministic_replay_passed", "world_isolation_passed"):
        if correctness.get(key) is not True:
            raise RuntimeError(f"joint report failed {key}")
    bounded = {
        "maximum_joint_limit_excess_rad": MAX_JOINT_LIMIT_EXCESS_RAD,
        "maximum_command_effort_ratio": MAX_COMMAND_EFFORT_RATIO,
        "maximum_initial_hold_drift_rad": MAX_INITIAL_HOLD_DRIFT_RAD,
        "tail_rms_tracking_error_rad": MAX_TAIL_RMS_TRACKING_ERROR_RAD,
        "maximum_absolute_joint_velocity_rad_s": MAX_ABSOLUTE_JOINT_VELOCITY_RAD_S,
        "maximum_link_quaternion_norm_error": MAX_LINK_QUATERNION_NORM_ERROR,
        "maximum_joint_anchor_error_m": MAX_JOINT_ANCHOR_ERROR_M,
    }
    for key, maximum in bounded.items():
        if _numeric(correctness.get(key), key) > maximum:
            raise RuntimeError(f"joint report exceeded {key}")


def validate_joint_contract_speedup(
    reports: Sequence[Mapping[str, Any]], comparison: Mapping[str, Any]
) -> Mapping[str, Any]:
    matches = {row.get("backend"): row for row in reports if row.get("contract_id") == CONTRACT_ID}
    if set(matches) != {PHYSX_BACKEND, CUDA_BACKEND}:
        raise RuntimeError("joint speedup requires exactly measured PhysX and Box3D CUDA reports")
    validate_joint_report(matches[PHYSX_BACKEND], backend=PHYSX_BACKEND)
    validate_joint_report(matches[CUDA_BACKEND], backend=CUDA_BACKEND)
    rows = [row for row in comparison.get("speedups", ()) if row.get("contract_id") == CONTRACT_ID]
    if len(rows) != 1:
        raise RuntimeError("joint comparison must contain exactly one matched speedup")
    row = rows[0]
    if row.get("baseline") != PHYSX_BACKEND or row.get("candidate") != CUDA_BACKEND:
        raise RuntimeError("joint speedup direction must be PhysX baseline to Box3D CUDA candidate")
    _numeric(row.get("world_step_speedup"), "world_step_speedup")
    return row


__all__ = [name for name in globals() if name.isupper()] + [
    "JointChainSpec", "SPEC", "phase_radians", "target_positions_rad",
    "target_batch_rad", "validate_joint_report", "validate_joint_contract_speedup",
]
