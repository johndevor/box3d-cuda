"""Dependency-free Stage-5 CPU oracle for fixed-topology rigid-body joints.

The public data contract uses only scalar values and rectangular Python lists
that map directly to FP32 CUDA tensors. This module is a deterministic CPU
reference, not a CUDA implementation or a production reduced-coordinate
articulation solver. It uses bounded maximal-coordinate sequential impulses.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

from .reference import STATE_WIDTH, _integrate_quaternion
from .sat_reference import RigidBox


CONTRACT_ID = "box3d.articulated-joints/v1"
FIXED = 0
REVOLUTE = 1
PRISMATIC = 2
JOINT_TYPE_NAMES = ("fixed", "revolute", "prismatic")

Vec3 = List[float]
Quaternion = List[float]


@dataclass(frozen=True)
class JointConfig:
    dt: float = 1.0 / 120.0
    substeps: int = 2
    gravity_y: float = -9.81
    solver_iterations: int = 12
    position_correction: float = 0.8
    position_slop: float = 1.0e-5
    angular_slop: float = 1.0e-5
    maximum_linear_repair_m: float = 0.1
    maximum_angular_repair_rad: float = 0.2
    warm_start_factor: float = 0.8

    def __post_init__(self) -> None:
        numeric = (
            self.dt,
            self.gravity_y,
            self.position_correction,
            self.position_slop,
            self.angular_slop,
            self.maximum_linear_repair_m,
            self.maximum_angular_repair_rad,
            self.warm_start_factor,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("joint config must contain finite values")
        if self.dt <= 0.0 or self.substeps <= 0 or self.solver_iterations <= 0:
            raise ValueError("dt, substeps, and solver_iterations must be positive")
        if not 0.0 <= self.position_correction <= 1.0:
            raise ValueError("position_correction must be in [0, 1]")
        if not 0.0 <= self.warm_start_factor <= 1.0:
            raise ValueError("warm_start_factor must be in [0, 1]")
        if min(
            self.position_slop,
            self.angular_slop,
            self.maximum_linear_repair_m,
            self.maximum_angular_repair_rad,
        ) < 0.0:
            raise ValueError("joint slop/repair bounds cannot be negative")


@dataclass(frozen=True)
class JointTopology:
    """World-shared fixed topology; each tuple is a CUDA-compatible joint row."""

    joint_indices: Tuple[Tuple[int, int], ...]
    joint_types: Tuple[int, ...]
    parent_anchor_local: Tuple[Tuple[float, float, float], ...]
    child_anchor_local: Tuple[Tuple[float, float, float], ...]
    axis_parent: Tuple[Tuple[float, float, float], ...]
    reference_quaternion_parent_to_child: Tuple[Tuple[float, float, float, float], ...]
    lower_limit: Tuple[float, ...]
    upper_limit: Tuple[float, ...]
    damping: Tuple[float, ...]
    motor_enabled: Tuple[bool, ...]
    collision_enabled: Tuple[bool, ...]

    def __post_init__(self) -> None:
        joint_count = len(self.joint_indices)
        fields = (
            self.joint_types,
            self.parent_anchor_local,
            self.child_anchor_local,
            self.axis_parent,
            self.reference_quaternion_parent_to_child,
            self.lower_limit,
            self.upper_limit,
            self.damping,
            self.motor_enabled,
            self.collision_enabled,
        )
        if joint_count <= 0 or any(len(field) != joint_count for field in fields):
            raise ValueError("all topology rows must have the same positive joint count")
        seen = set()
        for index in range(joint_count):
            pair = self.joint_indices[index]
            if len(pair) != 2 or pair[0] == pair[1] or any(
                isinstance(body, bool) or not isinstance(body, int) or body < 0 for body in pair
            ):
                raise ValueError("joint body indices must be distinct non-negative integers")
            if pair in seen:
                raise ValueError("duplicate directed joint body pair")
            seen.add(pair)
            if self.joint_types[index] not in (FIXED, REVOLUTE, PRISMATIC):
                raise ValueError("unsupported joint type")
            for name, row, length in (
                ("parent anchor", self.parent_anchor_local[index], 3),
                ("child anchor", self.child_anchor_local[index], 3),
                ("axis", self.axis_parent[index], 3),
                ("reference quaternion", self.reference_quaternion_parent_to_child[index], 4),
            ):
                if len(row) != length or not all(math.isfinite(float(value)) for value in row):
                    raise ValueError("{} row must contain {} finite values".format(name, length))
            axis_length = math.sqrt(sum(value * value for value in self.axis_parent[index]))
            if abs(axis_length - 1.0) > 2.0e-5:
                raise ValueError("joint axes must be normalized")
            quaternion_length = math.sqrt(
                sum(value * value for value in self.reference_quaternion_parent_to_child[index])
            )
            if abs(quaternion_length - 1.0) > 2.0e-5:
                raise ValueError("reference quaternions must be normalized")
            lower, upper = self.lower_limit[index], self.upper_limit[index]
            if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
                raise ValueError("joint limits must be finite and ordered")
            if not math.isfinite(self.damping[index]) or self.damping[index] < 0.0:
                raise ValueError("joint damping cannot be negative")
            if not isinstance(self.motor_enabled[index], bool) or not isinstance(
                self.collision_enabled[index], bool
            ):
                raise ValueError("motor/collision flags must be booleans")


@dataclass(frozen=True)
class JointStepResult:
    state: List[List[List[float]]]
    coordinate: List[List[float]]
    linear_error_m: List[List[float]]
    angular_error_rad: List[List[float]]
    limit_error: List[List[float]]
    motor_impulse: List[List[float]]
    limit_active: List[List[bool]]
    warm_start_cache: List[List[List[float]]]


def _add(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return [a[index] + b[index] for index in range(3)]


def _sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return [a[index] - b[index] for index in range(3)]


def _scale(vector: Sequence[float], scalar: float) -> Vec3:
    return [value * scalar for value in vector]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(a[index] * b[index] for index in range(3))


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Sequence[float]) -> Vec3:
    length = _norm(vector)
    if length <= 1.0e-12:
        raise ValueError("cannot normalize degenerate vector")
    return _scale(vector, 1.0 / length)


def _quaternion_multiply(a: Sequence[float], b: Sequence[float]) -> Quaternion:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _quaternion_conjugate(q: Sequence[float]) -> Quaternion:
    return [-q[0], -q[1], -q[2], q[3]]


def _quaternion_normalize(q: Sequence[float]) -> Quaternion:
    length = math.sqrt(sum(value * value for value in q))
    if length <= 1.0e-12:
        raise ValueError("degenerate quaternion")
    return [value / length for value in q]


def _rotate(q: Sequence[float], vector: Sequence[float]) -> Vec3:
    qv = q[:3]
    twice = _scale(_cross(qv, vector), 2.0)
    return _add(vector, _add(_scale(twice, q[3]), _cross(qv, twice)))


def _rotation_vector(q: Sequence[float]) -> Vec3:
    normalized = _quaternion_normalize(q)
    if normalized[3] < 0.0:
        normalized = _scale(normalized, -1.0)
    sine = _norm(normalized[:3])
    if sine <= 1.0e-12:
        return _scale(normalized[:3], 2.0)
    angle = 2.0 * math.atan2(sine, max(0.0, normalized[3]))
    return _scale(normalized[:3], angle / sine)


def _delta_quaternion(rotation_vector: Sequence[float]) -> Quaternion:
    angle = _norm(rotation_vector)
    if angle <= 1.0e-12:
        return _quaternion_normalize(
            [0.5 * rotation_vector[0], 0.5 * rotation_vector[1], 0.5 * rotation_vector[2], 1.0]
        )
    scale = math.sin(0.5 * angle) / angle
    return [
        rotation_vector[0] * scale,
        rotation_vector[1] * scale,
        rotation_vector[2] * scale,
        math.cos(0.5 * angle),
    ]


def _inverse_inertia_world(box: RigidBox, vector: Sequence[float]) -> Vec3:
    local = _rotate(_quaternion_conjugate(box.quaternion), vector)
    local = [local[index] * box.inverse_inertia_local[index] for index in range(3)]
    return _rotate(box.quaternion, local)


def _point_velocity(box: RigidBox, offset: Sequence[float]) -> Vec3:
    return _add(box.linear_velocity, _cross(box.angular_velocity, offset))


def _apply_impulse(box: RigidBox, offset: Sequence[float], impulse: Sequence[float]) -> None:
    if box.inverse_mass == 0.0:
        return
    box.linear_velocity = _add(box.linear_velocity, _scale(impulse, box.inverse_mass))
    box.angular_velocity = _add(
        box.angular_velocity, _inverse_inertia_world(box, _cross(offset, impulse))
    )


def _apply_angular_impulse(box: RigidBox, impulse: Sequence[float]) -> None:
    if box.inverse_mass == 0.0:
        return
    box.angular_velocity = _add(box.angular_velocity, _inverse_inertia_world(box, impulse))


def _linear_effective_mass(
    parent: RigidBox,
    child: RigidBox,
    parent_offset: Sequence[float],
    child_offset: Sequence[float],
    direction: Sequence[float],
) -> float:
    parent_term = _inverse_inertia_world(parent, _cross(parent_offset, direction))
    child_term = _inverse_inertia_world(child, _cross(child_offset, direction))
    return (
        parent.inverse_mass
        + child.inverse_mass
        + _dot(_cross(parent_term, parent_offset), direction)
        + _dot(_cross(child_term, child_offset), direction)
    )


def _angular_effective_mass(parent: RigidBox, child: RigidBox, direction: Sequence[float]) -> float:
    return _dot(_inverse_inertia_world(parent, direction), direction) + _dot(
        _inverse_inertia_world(child, direction), direction
    )


def _tangent_basis(axis: Sequence[float]) -> Tuple[Vec3, Vec3]:
    cardinal = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    seed = min(cardinal, key=lambda candidate: (abs(_dot(axis, candidate)), cardinal.index(candidate)))
    first = _normalize(_cross(seed, axis))
    return first, _normalize(_cross(axis, first))


def _joint_geometry(parent: RigidBox, child: RigidBox, topology: JointTopology, joint: int):
    parent_offset = _rotate(parent.quaternion, topology.parent_anchor_local[joint])
    child_offset = _rotate(child.quaternion, topology.child_anchor_local[joint])
    parent_anchor = _add(parent.position, parent_offset)
    child_anchor = _add(child.position, child_offset)
    axis = _normalize(_rotate(parent.quaternion, topology.axis_parent[joint]))
    relative = _quaternion_multiply(_quaternion_conjugate(parent.quaternion), child.quaternion)
    delta = _quaternion_multiply(
        relative, _quaternion_conjugate(topology.reference_quaternion_parent_to_child[joint])
    )
    rotation_local = _rotation_vector(delta)
    rotation_world = _rotate(parent.quaternion, rotation_local)
    joint_type = topology.joint_types[joint]
    if joint_type == REVOLUTE:
        coordinate = _dot(rotation_local, topology.axis_parent[joint])
        angular_error = _sub(rotation_world, _scale(axis, _dot(rotation_world, axis)))
        linear_error = _sub(child_anchor, parent_anchor)
    elif joint_type == PRISMATIC:
        separation = _sub(child_anchor, parent_anchor)
        coordinate = _dot(separation, axis)
        linear_error = _sub(separation, _scale(axis, coordinate))
        angular_error = rotation_world
    else:
        coordinate = 0.0
        linear_error = _sub(child_anchor, parent_anchor)
        angular_error = rotation_world
    return parent_offset, child_offset, axis, coordinate, linear_error, angular_error


def collision_filter_pairs(topology: JointTopology) -> Tuple[Tuple[int, int], ...]:
    """Return normalized connected-link pairs excluded from collision."""

    return tuple(
        sorted(
            {
                (min(parent, child), max(parent, child))
                for (parent, child), enabled in zip(
                    topology.joint_indices, topology.collision_enabled
                )
                if not enabled
            }
        )
    )


def _validate_layout(
    state,
    inverse_mass,
    inverse_inertia,
    topology: JointTopology,
    motor_target_velocity,
    maximum_effort,
) -> None:
    if not state or not (len(state) == len(inverse_mass) == len(inverse_inertia)):
        raise ValueError("joint world layouts must have equal positive world counts")
    worlds, bodies, joints = len(state), len(state[0]), len(topology.joint_indices)
    if bodies < 2:
        raise ValueError("joint worlds require at least two bodies")
    if len(motor_target_velocity) != worlds or len(maximum_effort) != worlds:
        raise ValueError("motor controls must have one row per world")
    for world in range(worlds):
        if not (
            len(state[world]) == len(inverse_mass[world]) == len(inverse_inertia[world]) == bodies
        ):
            raise ValueError("body counts differ between joint world arrays")
        if len(motor_target_velocity[world]) != joints or len(maximum_effort[world]) != joints:
            raise ValueError("motor controls must have shape [worlds,joints]")
        for body in range(bodies):
            values = state[world][body]
            if len(values) != STATE_WIDTH or not all(math.isfinite(float(value)) for value in values):
                raise ValueError("body state must have 13 finite values")
            quaternion_norm = math.sqrt(sum(float(value) ** 2 for value in values[3:7]))
            if abs(quaternion_norm - 1.0) > 2.0e-5:
                raise ValueError("body quaternions must be normalized")
            if (
                not math.isfinite(float(inverse_mass[world][body]))
                or inverse_mass[world][body] < 0.0
                or len(inverse_inertia[world][body]) != 3
                or any(
                    not math.isfinite(float(value)) or value < 0.0
                    for value in inverse_inertia[world][body]
                )
            ):
                raise ValueError("inverse mass/inertia must be finite and non-negative")
            if inverse_mass[world][body] == 0.0 and any(
                value != 0.0 for value in inverse_inertia[world][body]
            ):
                raise ValueError("static bodies must have zero inverse inertia")
        for joint in range(joints):
            if not (
                math.isfinite(float(motor_target_velocity[world][joint]))
                and math.isfinite(float(maximum_effort[world][joint]))
                and maximum_effort[world][joint] >= 0.0
            ):
                raise ValueError("motor targets/efforts must be finite and effort non-negative")
    if any(max(pair) >= bodies for pair in topology.joint_indices):
        raise ValueError("joint topology references a missing body")


def _integrate(box: RigidBox, h: float, gravity_y: float) -> None:
    if box.inverse_mass == 0.0:
        return
    box.linear_velocity[1] += gravity_y * h
    box.position = _add(box.position, _scale(box.linear_velocity, h))
    state = box.to_state()
    _integrate_quaternion(state, h)
    box.quaternion = state[3:7]


def _solve_linear_rows(
    parent: RigidBox,
    child: RigidBox,
    parent_offset: Sequence[float],
    child_offset: Sequence[float],
    directions: Sequence[Sequence[float]],
    accumulated: List[float],
) -> None:
    for row_index, direction in enumerate(directions):
        relative = _sub(
            _point_velocity(child, child_offset), _point_velocity(parent, parent_offset)
        )
        denominator = _linear_effective_mass(
            parent, child, parent_offset, child_offset, direction
        )
        if denominator <= 1.0e-12:
            continue
        delta = -_dot(relative, direction) / denominator
        accumulated[row_index] += delta
        impulse = _scale(direction, delta)
        _apply_impulse(parent, parent_offset, _scale(impulse, -1.0))
        _apply_impulse(child, child_offset, impulse)


def _solve_angular_rows(
    parent: RigidBox,
    child: RigidBox,
    directions: Sequence[Sequence[float]],
    accumulated: List[float],
) -> None:
    for row_index, direction in enumerate(directions):
        speed = _dot(_sub(child.angular_velocity, parent.angular_velocity), direction)
        denominator = _angular_effective_mass(parent, child, direction)
        if denominator <= 1.0e-12:
            continue
        delta = -speed / denominator
        accumulated[row_index] += delta
        impulse = _scale(direction, delta)
        _apply_angular_impulse(parent, _scale(impulse, -1.0))
        _apply_angular_impulse(child, impulse)


def _solve_actuator(
    parent: RigidBox,
    child: RigidBox,
    topology: JointTopology,
    joint: int,
    axis: Sequence[float],
    parent_offset: Sequence[float],
    child_offset: Sequence[float],
    target: float,
    effort: float,
    accumulated: float,
    h: float,
    coordinate: float,
    target_position: float,
    stiffness: float,
) -> float:
    if effort <= 0.0 or topology.joint_types[joint] == FIXED:
        return accumulated
    if topology.joint_types[joint] == REVOLUTE:
        speed = _dot(_sub(child.angular_velocity, parent.angular_velocity), axis)
        denominator = _angular_effective_mass(parent, child, axis)
    else:
        relative = _sub(
            _point_velocity(child, child_offset), _point_velocity(parent, parent_offset)
        )
        speed = _dot(relative, axis)
        denominator = _linear_effective_mass(
            parent, child, parent_offset, child_offset, axis
        )
    if denominator <= 1.0e-12:
        return accumulated
    limit = effort * h
    if stiffness > 0.0:
        desired = (
            stiffness * (target_position - coordinate)
            - topology.damping[joint] * speed
        ) * h
        updated = max(-limit, min(limit, desired))
    else:
        delta = -topology.damping[joint] * speed * h / denominator
        if topology.motor_enabled[joint]:
            delta += (target - speed) / denominator
        updated = max(-limit, min(limit, accumulated + delta))
    applied = updated - accumulated
    impulse = _scale(axis, applied)
    if topology.joint_types[joint] == REVOLUTE:
        _apply_angular_impulse(parent, _scale(impulse, -1.0))
        _apply_angular_impulse(child, impulse)
    else:
        _apply_impulse(parent, parent_offset, _scale(impulse, -1.0))
        _apply_impulse(child, child_offset, impulse)
    return updated


def _solve_limit(
    parent: RigidBox,
    child: RigidBox,
    topology: JointTopology,
    joint: int,
    axis: Sequence[float],
    coordinate: float,
    parent_offset: Sequence[float],
    child_offset: Sequence[float],
    h: float,
    accumulated: float,
) -> Tuple[float, bool]:
    lower, upper = topology.lower_limit[joint], topology.upper_limit[joint]
    if coordinate < lower:
        direction, violation = 1.0, lower - coordinate
    elif coordinate > upper:
        direction, violation = -1.0, coordinate - upper
    else:
        return 0.0, False
    if topology.joint_types[joint] == REVOLUTE:
        speed = _dot(_sub(child.angular_velocity, parent.angular_velocity), axis)
        denominator = _angular_effective_mass(parent, child, axis)
    elif topology.joint_types[joint] == PRISMATIC:
        relative = _sub(
            _point_velocity(child, child_offset), _point_velocity(parent, parent_offset)
        )
        speed = _dot(relative, axis)
        denominator = _linear_effective_mass(
            parent, child, parent_offset, child_offset, axis
        )
    else:
        return 0.0, False
    target_speed = direction * min(2.0, violation * 0.2 / h)
    delta = (target_speed - speed) / max(denominator, 1.0e-12)
    updated = direction * max(0.0, direction * (accumulated + delta))
    applied = updated - accumulated
    impulse = _scale(axis, applied)
    if topology.joint_types[joint] == REVOLUTE:
        _apply_angular_impulse(parent, _scale(impulse, -1.0))
        _apply_angular_impulse(child, impulse)
    else:
        _apply_impulse(parent, parent_offset, _scale(impulse, -1.0))
        _apply_impulse(child, child_offset, impulse)
    return updated, True


def _apply_cached_rows(
    parent: RigidBox,
    child: RigidBox,
    topology: JointTopology,
    joint: int,
    parent_offset: Sequence[float],
    child_offset: Sequence[float],
    axis: Sequence[float],
    linear_directions: Sequence[Sequence[float]],
    angular_directions: Sequence[Sequence[float]],
    cache: Sequence[float],
) -> None:
    """Apply one world's explicit per-row cache before iterative solving."""

    for row_index, direction in enumerate(linear_directions):
        impulse = _scale(direction, cache[row_index])
        _apply_impulse(parent, parent_offset, _scale(impulse, -1.0))
        _apply_impulse(child, child_offset, impulse)
    for row_index, direction in enumerate(angular_directions):
        impulse = _scale(direction, cache[3 + row_index])
        _apply_angular_impulse(parent, _scale(impulse, -1.0))
        _apply_angular_impulse(child, impulse)
    if topology.joint_types[joint] != FIXED:
        for slot in (6, 7):
            impulse = _scale(axis, cache[slot])
            if topology.joint_types[joint] == REVOLUTE:
                _apply_angular_impulse(parent, _scale(impulse, -1.0))
                _apply_angular_impulse(child, impulse)
            else:
                _apply_impulse(parent, parent_offset, _scale(impulse, -1.0))
                _apply_impulse(child, child_offset, impulse)


def _apply_orientation(box: RigidBox, rotation_world: Sequence[float]) -> None:
    if box.inverse_mass == 0.0:
        return
    box.quaternion = _quaternion_normalize(
        _quaternion_multiply(_delta_quaternion(rotation_world), box.quaternion)
    )


def _repair_angular_pair(
    parent: RigidBox,
    child: RigidBox,
    error_world: Sequence[float],
    config: JointConfig,
) -> None:
    magnitude = _norm(error_world)
    if magnitude <= config.angular_slop:
        return
    direction = _scale(error_world, 1.0 / magnitude)
    parent_weight = _dot(_inverse_inertia_world(parent, direction), direction)
    child_weight = _dot(_inverse_inertia_world(child, direction), direction)
    total = parent_weight + child_weight
    if total <= 1.0e-12:
        return
    correction = min(
        config.maximum_angular_repair_rad,
        (magnitude - config.angular_slop) * config.position_correction,
    )
    _apply_orientation(parent, _scale(direction, correction * parent_weight / total))
    _apply_orientation(child, _scale(direction, -correction * child_weight / total))


def _repair_joint(
    parent: RigidBox,
    child: RigidBox,
    topology: JointTopology,
    joint: int,
    config: JointConfig,
) -> None:
    _, _, axis, coordinate, linear_error, angular_error = _joint_geometry(
        parent, child, topology, joint
    )
    length = _norm(linear_error)
    inverse_sum = parent.inverse_mass + child.inverse_mass
    if length > config.position_slop and inverse_sum > 0.0:
        correction = min(
            config.maximum_linear_repair_m,
            (length - config.position_slop) * config.position_correction,
        )
        direction = _scale(linear_error, correction / length)
        parent.position = _add(
            parent.position, _scale(direction, parent.inverse_mass / inverse_sum)
        )
        child.position = _sub(
            child.position, _scale(direction, child.inverse_mass / inverse_sum)
        )
    _repair_angular_pair(parent, child, angular_error, config)
    joint_type = topology.joint_types[joint]
    lower, upper = topology.lower_limit[joint], topology.upper_limit[joint]
    if joint_type == PRISMATIC and (coordinate < lower or coordinate > upper):
        target = lower if coordinate < lower else upper
        excess = coordinate - target
        inverse_sum = parent.inverse_mass + child.inverse_mass
        if inverse_sum > 0.0:
            correction = max(-config.maximum_linear_repair_m, min(config.maximum_linear_repair_m, excess * config.position_correction))
            parent.position = _add(
                parent.position, _scale(axis, correction * parent.inverse_mass / inverse_sum)
            )
            child.position = _sub(
                child.position, _scale(axis, correction * child.inverse_mass / inverse_sum)
            )
    elif joint_type == REVOLUTE and (coordinate < lower or coordinate > upper):
        target = lower if coordinate < lower else upper
        _repair_angular_pair(parent, child, _scale(axis, coordinate - target), config)


def step_joint_reference(
    state: Sequence[Sequence[Sequence[float]]],
    inverse_mass: Sequence[Sequence[float]],
    inverse_inertia: Sequence[Sequence[Sequence[float]]],
    topology: JointTopology,
    motor_target_velocity: Sequence[Sequence[float]],
    maximum_effort: Sequence[Sequence[float]],
    config: JointConfig = JointConfig(),
    *,
    steps: int = 1,
    motor_target_position: Sequence[Sequence[float]] | None = None,
    stiffness: Sequence[float] | None = None,
    warm_start_cache: Sequence[Sequence[Sequence[float]]] | None = None,
) -> JointStepResult:
    """Advance batched fixed-topology worlds with bounded joint impulses."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    _validate_layout(
        state,
        inverse_mass,
        inverse_inertia,
        topology,
        motor_target_velocity,
        maximum_effort,
    )
    joint_count = len(topology.joint_indices)
    if motor_target_position is None:
        motor_target_position = [[0.0] * joint_count for _ in state]
    if stiffness is None:
        stiffness = [0.0] * joint_count
    if len(motor_target_position) != len(state) or any(
        len(row) != joint_count or any(not math.isfinite(float(value)) for value in row)
        for row in motor_target_position
    ):
        raise ValueError("motor_target_position must have shape [worlds,joints]")
    if len(stiffness) != joint_count or any(
        not math.isfinite(float(value)) or value < 0.0 for value in stiffness
    ):
        raise ValueError("stiffness must contain one finite non-negative value per joint")
    if warm_start_cache is None:
        warm_start_cache = [
            [[0.0] * 8 for _ in range(joint_count)] for _ in state
        ]
    if len(warm_start_cache) != len(state) or any(
        len(world) != joint_count
        or any(
            len(row) != 8
            or any(not math.isfinite(float(value)) for value in row)
            for row in world
        )
        for world in warm_start_cache
    ):
        raise ValueError("warm_start_cache must have shape [worlds,joints,8]")
    worlds = [
        [
            RigidBox.from_state(
                body,
                inverse_mass[world][index],
                (0.5, 0.5, 0.5),
                inverse_inertia[world][index],
            )
            for index, body in enumerate(state[world])
        ]
        for world in range(len(state))
    ]
    motor_total = [[0.0] * joint_count for _ in worlds]
    limit_active = [[False] * joint_count for _ in worlds]
    row_cache = [
        [[float(value) for value in row] for row in world]
        for world in warm_start_cache
    ]
    h = config.dt / config.substeps
    for _ in range(steps):
        for _ in range(config.substeps):
            for world in worlds:
                for body in world:
                    _integrate(body, h, config.gravity_y)
            row_lambda = [
                [
                    [value * config.warm_start_factor for value in row_cache[world][joint]]
                    for joint in range(joint_count)
                ]
                for world in range(len(worlds))
            ]
            for world_index, world in enumerate(worlds):
                for joint, (parent_index, child_index) in enumerate(topology.joint_indices):
                    parent, child = world[parent_index], world[child_index]
                    parent_offset, child_offset, axis, coordinate, _, _ = _joint_geometry(
                        parent, child, topology, joint
                    )
                    if topology.joint_types[joint] == PRISMATIC:
                        linear_directions = _tangent_basis(axis)
                        angular_directions = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
                    elif topology.joint_types[joint] == REVOLUTE:
                        linear_directions = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
                        angular_directions = _tangent_basis(axis)
                    else:
                        linear_directions = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
                        angular_directions = linear_directions
                    # A stale unilateral limit row must not pull a joint when
                    # the current coordinate is already inside its limits.
                    if topology.lower_limit[joint] <= coordinate <= topology.upper_limit[joint]:
                        row_lambda[world_index][joint][7] = 0.0
                    _apply_cached_rows(
                        parent, child, topology, joint, parent_offset, child_offset,
                        axis, linear_directions, angular_directions,
                        row_lambda[world_index][joint],
                    )
            for _ in range(config.solver_iterations):
                for world_index, world in enumerate(worlds):
                    for joint, (parent_index, child_index) in enumerate(topology.joint_indices):
                        parent, child = world[parent_index], world[child_index]
                        parent_offset, child_offset, axis, coordinate, _, _ = _joint_geometry(
                            parent, child, topology, joint
                        )
                        if topology.joint_types[joint] == PRISMATIC:
                            linear_directions = _tangent_basis(axis)
                            angular_directions = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
                        elif topology.joint_types[joint] == REVOLUTE:
                            linear_directions = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
                            angular_directions = _tangent_basis(axis)
                        else:
                            linear_directions = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
                            angular_directions = linear_directions
                        linear_accumulated = row_lambda[world_index][joint][0:len(linear_directions)]
                        _solve_linear_rows(
                            parent,
                            child,
                            parent_offset,
                            child_offset,
                            linear_directions,
                            linear_accumulated,
                        )
                        for row_index, value in enumerate(linear_accumulated):
                            row_lambda[world_index][joint][row_index] = value
                        angular_accumulated = row_lambda[world_index][joint][3:3 + len(angular_directions)]
                        _solve_angular_rows(parent, child, angular_directions, angular_accumulated)
                        # Slice mutations are not reflected in the cache, so
                        # write the accumulated bilateral rows back explicitly.
                        for row_index, value in enumerate(angular_accumulated):
                            row_lambda[world_index][joint][3 + row_index] = value
                        before = row_lambda[world_index][joint][6]
                        row_lambda[world_index][joint][6] = _solve_actuator(
                            parent,
                            child,
                            topology,
                            joint,
                            axis,
                            parent_offset,
                            child_offset,
                            motor_target_velocity[world_index][joint],
                            maximum_effort[world_index][joint],
                            before,
                            h,
                            coordinate,
                            motor_target_position[world_index][joint],
                            stiffness[joint],
                        )
                        updated_limit, active = _solve_limit(
                            parent,
                            child,
                            topology,
                            joint,
                            axis,
                            coordinate,
                            parent_offset,
                            child_offset,
                            h,
                            row_lambda[world_index][joint][7],
                        )
                        row_lambda[world_index][joint][7] = updated_limit
                        if active:
                            limit_active[world_index][joint] = True
            for world_index in range(len(worlds)):
                for joint in range(joint_count):
                    motor_total[world_index][joint] += row_lambda[world_index][joint][6]
            row_cache = row_lambda
            for world in worlds:
                for joint, (parent_index, child_index) in enumerate(topology.joint_indices):
                    _repair_joint(world[parent_index], world[child_index], topology, joint, config)

    coordinate = [[0.0] * joint_count for _ in worlds]
    linear_error = [[0.0] * joint_count for _ in worlds]
    angular_error = [[0.0] * joint_count for _ in worlds]
    limit_error = [[0.0] * joint_count for _ in worlds]
    for world_index, world in enumerate(worlds):
        for joint, (parent_index, child_index) in enumerate(topology.joint_indices):
            _, _, _, value, linear, angular = _joint_geometry(
                world[parent_index], world[child_index], topology, joint
            )
            coordinate[world_index][joint] = value
            linear_error[world_index][joint] = _norm(linear)
            angular_error[world_index][joint] = _norm(angular)
            limit_error[world_index][joint] = max(
                0.0,
                topology.lower_limit[joint] - value,
                value - topology.upper_limit[joint],
            )
    output = [[body.to_state() for body in world] for world in worlds]
    return JointStepResult(
        output,
        coordinate,
        linear_error,
        angular_error,
        limit_error,
        motor_total,
        limit_active,
        row_cache,
    )


def assert_valid_joint_result(result: JointStepResult, *, maximum_error: float = 0.02) -> None:
    for world_index, world in enumerate(result.state):
        for body_index, body in enumerate(world):
            if len(body) != STATE_WIDTH or not all(math.isfinite(float(value)) for value in body):
                raise AssertionError("malformed body {}:{}".format(world_index, body_index))
            norm = math.sqrt(sum(value * value for value in body[3:7]))
            if abs(norm - 1.0) > 2.0e-5:
                raise AssertionError("non-unit quaternion {}:{}".format(world_index, body_index))
        errors = result.linear_error_m[world_index] + result.angular_error_rad[world_index]
        if any(not math.isfinite(value) or value > maximum_error for value in errors):
            raise AssertionError("joint constraint error exceeded in world {}".format(world_index))


__all__ = [
    "CONTRACT_ID",
    "FIXED",
    "REVOLUTE",
    "PRISMATIC",
    "JOINT_TYPE_NAMES",
    "JointConfig",
    "JointTopology",
    "JointStepResult",
    "assert_valid_joint_result",
    "collision_filter_pairs",
    "step_joint_reference",
]
