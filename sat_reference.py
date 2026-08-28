"""Dependency-free CPU oracle for oriented box/box collision and response.

The narrow phase evaluates the 15 separating-axis candidates for two OBBs:
three face normals from each box and all nine edge cross products. Degenerate
cross products are skipped using ``sat_epsilon`` so nearly parallel boxes do
not amplify floating-point noise.

Response is intentionally a compact one-point reference solver. It applies a
normal impulse at an approximate support midpoint, a clamped Coulomb-friction
impulse, and split inverse-mass positional repair. Rotational effective mass
uses local diagonal inverse inertia transformed into world space. This is a
deterministic correctness oracle for a bounded CUDA slice, not a full contact
manifold, broad phase, CCD system, or production stacking solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, List, Optional, Sequence, Tuple

from .reference import STATE_WIDTH, _integrate_quaternion


CONTRACT_ID = "box3d.obb-pair-collision/v1"
PAIR_INDICES: Tuple[Tuple[int, int], ...] = ((0, 1), (2, 3), (4, 5))
DEFAULT_SEED = 23
ANGULAR_RESPONSE_THRESHOLD_RAD_S = 0.05
MAX_POSITION_REPAIR_M = 0.2

Vec3 = List[float]


@dataclass(frozen=True)
class SATConfig:
    dt: float = 1.0 / 120.0
    substeps: int = 2
    gravity_y: float = -9.81
    restitution: float = 0.05
    friction: float = 0.6
    position_slop: float = 1.0e-4
    position_correction: float = 0.8
    angular_damping: float = 0.02
    solver_iterations: int = 6
    sat_epsilon: float = 1.0e-7

    def __post_init__(self) -> None:
        numeric = (
            self.dt,
            self.gravity_y,
            self.restitution,
            self.friction,
            self.position_slop,
            self.position_correction,
            self.angular_damping,
            self.sat_epsilon,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("SAT configuration values must be finite")
        if self.dt <= 0.0 or self.substeps <= 0 or self.solver_iterations <= 0:
            raise ValueError("dt, substeps, and solver_iterations must be positive")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must be in [0, 1]")
        if self.friction < 0.0 or self.position_slop < 0.0 or self.angular_damping < 0.0:
            raise ValueError("friction, slop, and damping cannot be negative")
        if not 0.0 <= self.position_correction <= 1.0:
            raise ValueError("position_correction must be in [0, 1]")
        if self.sat_epsilon <= 0.0:
            raise ValueError("sat_epsilon must be positive")


@dataclass
class RigidBox:
    position: Vec3
    quaternion: List[float]  # xyzw
    linear_velocity: Vec3
    angular_velocity: Vec3
    half_extents: Vec3
    inverse_mass: float
    inverse_inertia_local: Vec3

    def __post_init__(self) -> None:
        self.position = _vector(self.position, 3, "position")
        self.quaternion = _vector(self.quaternion, 4, "quaternion")
        self.linear_velocity = _vector(self.linear_velocity, 3, "linear_velocity")
        self.angular_velocity = _vector(self.angular_velocity, 3, "angular_velocity")
        self.half_extents = _vector(self.half_extents, 3, "half_extents")
        self.inverse_inertia_local = _vector(
            self.inverse_inertia_local, 3, "inverse_inertia_local"
        )
        self.inverse_mass = _finite(self.inverse_mass, "inverse_mass")
        if self.inverse_mass < 0.0:
            raise ValueError("inverse_mass cannot be negative")
        if any(value <= 0.0 for value in self.half_extents):
            raise ValueError("half extents must be positive")
        if any(value < 0.0 for value in self.inverse_inertia_local):
            raise ValueError("inverse inertia cannot be negative")
        norm = math.sqrt(sum(value * value for value in self.quaternion))
        if norm <= 1.0e-15:
            raise ValueError("quaternion norm is zero")
        self.quaternion = [value / norm for value in self.quaternion]
        if self.inverse_mass == 0.0:
            # Static bodies cannot gain angular velocity through an accidentally
            # nonzero inertia input.
            self.inverse_inertia_local = [0.0, 0.0, 0.0]

    @classmethod
    def from_state(
        cls,
        state: Sequence[float],
        inverse_mass: float,
        half_extents: Sequence[float],
        inverse_inertia_local: Sequence[float],
    ) -> "RigidBox":
        if len(state) != STATE_WIDTH:
            raise ValueError("body state must contain 13 values")
        return cls(
            list(state[0:3]),
            list(state[3:7]),
            list(state[7:10]),
            list(state[10:13]),
            list(half_extents),
            inverse_mass,
            list(inverse_inertia_local),
        )

    def to_state(self) -> List[float]:
        return [
            *self.position,
            *self.quaternion,
            *self.linear_velocity,
            *self.angular_velocity,
        ]

    def copy(self) -> "RigidBox":
        return RigidBox.from_state(
            self.to_state(), self.inverse_mass, self.half_extents, self.inverse_inertia_local
        )


@dataclass(frozen=True)
class SATContact:
    normal: Tuple[float, float, float]  # points from A toward B
    depth: float
    point: Tuple[float, float, float]
    axis_index: int
    axis_kind: str
    axes_considered: int
    axes_tested: int


@dataclass(frozen=True)
class SATQuery:
    colliding: bool
    contact: Optional[SATContact]
    separation: float
    separating_axis: Optional[Tuple[float, float, float]]
    axes_considered: int = 15
    axes_tested: int = 0


@dataclass(frozen=True)
class ImpulseResult:
    normal_impulse: float
    friction_impulse: Tuple[float, float, float]
    position_repair_m: float
    relative_normal_speed_before: float
    relative_normal_speed_after: float


def _finite(value: float, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError("{} must be finite".format(field_name))
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be finite".format(field_name)) from exc
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(field_name))
    return result


def _vector(values: Sequence[float], length: int, field_name: str) -> List[float]:
    if len(values) != length:
        raise ValueError("{} must contain {} values".format(field_name, length))
    return [_finite(value, "{}[]".format(field_name)) for value in values]


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


def _length_squared(vector: Sequence[float]) -> float:
    return _dot(vector, vector)


def _normalize(vector: Sequence[float], epsilon: float) -> Optional[Vec3]:
    norm_squared = _length_squared(vector)
    if norm_squared <= epsilon * epsilon:
        return None
    inverse = 1.0 / math.sqrt(norm_squared)
    return _scale(vector, inverse)


def _rotate(quaternion: Sequence[float], vector: Sequence[float]) -> Vec3:
    qv = quaternion[:3]
    twice = _scale(_cross(qv, vector), 2.0)
    return [
        vector[index] + quaternion[3] * twice[index] + _cross(qv, twice)[index]
        for index in range(3)
    ]


def _inverse_rotate(quaternion: Sequence[float], vector: Sequence[float]) -> Vec3:
    return _rotate((-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]), vector)


def _axes(box: RigidBox) -> Tuple[Vec3, Vec3, Vec3]:
    return (
        _rotate(box.quaternion, (1.0, 0.0, 0.0)),
        _rotate(box.quaternion, (0.0, 1.0, 0.0)),
        _rotate(box.quaternion, (0.0, 0.0, 1.0)),
    )


def _projection_radius(box: RigidBox, axes: Sequence[Sequence[float]], axis: Sequence[float]) -> float:
    return sum(
        box.half_extents[index] * abs(_dot(axes[index], axis))
        for index in range(3)
    )


def _support_midpoint(
    box_a: RigidBox,
    box_b: RigidBox,
    axes_a: Sequence[Sequence[float]],
    axes_b: Sequence[Sequence[float]],
    normal: Sequence[float],
    epsilon: float,
) -> Vec3:
    def support(box: RigidBox, axes: Sequence[Sequence[float]], direction: Sequence[float]) -> Vec3:
        point = list(box.position)
        for index, axis in enumerate(axes):
            alignment = _dot(axis, direction)
            sign = 1.0 if alignment > epsilon else (-1.0 if alignment < -epsilon else 0.0)
            point = _add(point, _scale(axis, sign * box.half_extents[index]))
        return point

    point_a = support(box_a, axes_a, normal)
    point_b = support(box_b, axes_b, _scale(normal, -1.0))
    return _scale(_add(point_a, point_b), 0.5)


def sat_query(box_a: RigidBox, box_b: RigidBox, sat_epsilon: float = 1.0e-7) -> SATQuery:
    """Evaluate the complete OBB SAT candidate set.

    The returned separation is positive for separated boxes and non-positive
    for overlap/touching. Exactly parallel edge cross-products are still
    counted among the 15 candidates but skipped as numerically degenerate.
    """

    epsilon = _finite(sat_epsilon, "sat_epsilon")
    if epsilon <= 0.0:
        raise ValueError("sat_epsilon must be positive")
    axes_a = _axes(box_a)
    axes_b = _axes(box_b)
    candidates: List[Tuple[Vec3, str]] = []
    candidates.extend((axis, "face_a") for axis in axes_a)
    candidates.extend((axis, "face_b") for axis in axes_b)
    candidates.extend((_cross(axis_a, axis_b), "edge") for axis_a in axes_a for axis_b in axes_b)
    if len(candidates) != 15:
        raise AssertionError("OBB SAT must contain exactly 15 axis candidates")

    center_delta = _sub(box_b.position, box_a.position)
    maximum_separation = -math.inf
    separating_axis: Optional[Vec3] = None
    minimum_depth = math.inf
    minimum_axis: Optional[Vec3] = None
    minimum_index = -1
    minimum_kind = ""
    tested = 0
    for index, (candidate, kind) in enumerate(candidates):
        axis = _normalize(candidate, epsilon)
        if axis is None:
            continue
        tested += 1
        signed_distance = _dot(center_delta, axis)
        radius = _projection_radius(box_a, axes_a, axis) + _projection_radius(box_b, axes_b, axis)
        separation = abs(signed_distance) - radius
        oriented = axis if signed_distance >= 0.0 else _scale(axis, -1.0)
        if separation > maximum_separation:
            maximum_separation = separation
            separating_axis = oriented
        depth = -separation
        if depth < minimum_depth:
            minimum_depth = depth
            minimum_axis = oriented
            minimum_index = index
            minimum_kind = kind

    if tested < 6:
        raise AssertionError("face axes must never be degenerate")
    if maximum_separation > epsilon:
        return SATQuery(
            False,
            None,
            maximum_separation,
            tuple(separating_axis) if separating_axis is not None else None,
            15,
            tested,
        )
    if minimum_axis is None:
        raise AssertionError("SAT did not select a minimum-translation axis")
    depth = max(0.0, minimum_depth)
    point = _support_midpoint(box_a, box_b, axes_a, axes_b, minimum_axis, epsilon)
    contact = SATContact(
        tuple(minimum_axis),
        depth,
        tuple(point),
        minimum_index,
        minimum_kind,
        15,
        tested,
    )
    return SATQuery(True, contact, -depth, None, 15, tested)


def obb_contact(
    box_a: RigidBox, box_b: RigidBox, sat_epsilon: float = 1.0e-7
) -> Optional[SATContact]:
    return sat_query(box_a, box_b, sat_epsilon).contact


def _inverse_inertia_world(box: RigidBox, vector: Sequence[float]) -> Vec3:
    local = _inverse_rotate(box.quaternion, vector)
    scaled = [local[index] * box.inverse_inertia_local[index] for index in range(3)]
    return _rotate(box.quaternion, scaled)


def _point_velocity(box: RigidBox, offset: Sequence[float]) -> Vec3:
    return _add(box.linear_velocity, _cross(box.angular_velocity, offset))


def _effective_mass(
    box_a: RigidBox,
    box_b: RigidBox,
    offset_a: Sequence[float],
    offset_b: Sequence[float],
    direction: Sequence[float],
) -> float:
    angular_a = _inverse_inertia_world(box_a, _cross(offset_a, direction))
    angular_b = _inverse_inertia_world(box_b, _cross(offset_b, direction))
    rotational_a = _dot(_cross(angular_a, offset_a), direction)
    rotational_b = _dot(_cross(angular_b, offset_b), direction)
    return box_a.inverse_mass + box_b.inverse_mass + rotational_a + rotational_b


def _apply_impulse(box: RigidBox, offset: Sequence[float], impulse: Sequence[float]) -> None:
    if box.inverse_mass == 0.0:
        return
    box.linear_velocity = _add(box.linear_velocity, _scale(impulse, box.inverse_mass))
    angular_delta = _inverse_inertia_world(box, _cross(offset, impulse))
    box.angular_velocity = _add(box.angular_velocity, angular_delta)


def solve_contact(
    box_a: RigidBox,
    box_b: RigidBox,
    contact: SATContact,
    config: SATConfig = SATConfig(),
) -> ImpulseResult:
    """Apply one normal/friction impulse and bounded positional repair."""

    normal = list(contact.normal)
    point = list(contact.point)
    offset_a = _sub(point, box_a.position)
    offset_b = _sub(point, box_b.position)
    relative = _sub(_point_velocity(box_b, offset_b), _point_velocity(box_a, offset_a))
    speed_before = _dot(relative, normal)
    normal_impulse = 0.0
    friction_impulse = [0.0, 0.0, 0.0]
    denominator = _effective_mass(box_a, box_b, offset_a, offset_b, normal)
    if speed_before < 0.0 and denominator > config.sat_epsilon:
        normal_impulse = -(1.0 + config.restitution) * speed_before / denominator
        impulse = _scale(normal, normal_impulse)
        _apply_impulse(box_a, offset_a, _scale(impulse, -1.0))
        _apply_impulse(box_b, offset_b, impulse)

        relative = _sub(_point_velocity(box_b, offset_b), _point_velocity(box_a, offset_a))
        tangent = _sub(relative, _scale(normal, _dot(relative, normal)))
        tangent_direction = _normalize(tangent, config.sat_epsilon)
        if tangent_direction is not None:
            tangent_mass = _effective_mass(
                box_a, box_b, offset_a, offset_b, tangent_direction
            )
            if tangent_mass > config.sat_epsilon:
                unconstrained = -_dot(relative, tangent_direction) / tangent_mass
                maximum = config.friction * normal_impulse
                magnitude = max(-maximum, min(maximum, unconstrained))
                friction_impulse = _scale(tangent_direction, magnitude)
                _apply_impulse(box_a, offset_a, _scale(friction_impulse, -1.0))
                _apply_impulse(box_b, offset_b, friction_impulse)

    inverse_mass_sum = box_a.inverse_mass + box_b.inverse_mass
    repair = 0.0
    if inverse_mass_sum > 0.0:
        repair = min(
            MAX_POSITION_REPAIR_M,
            max(0.0, contact.depth - config.position_slop) * config.position_correction,
        )
        correction = _scale(normal, repair / inverse_mass_sum)
        box_a.position = _sub(box_a.position, _scale(correction, box_a.inverse_mass))
        box_b.position = _add(box_b.position, _scale(correction, box_b.inverse_mass))

    offset_a_after = _sub(point, box_a.position)
    offset_b_after = _sub(point, box_b.position)
    relative_after = _sub(
        _point_velocity(box_b, offset_b_after), _point_velocity(box_a, offset_a_after)
    )
    return ImpulseResult(
        normal_impulse,
        tuple(friction_impulse),
        repair,
        speed_before,
        _dot(relative_after, normal),
    )


def _axis_angle(axis: Sequence[float], angle: float) -> List[float]:
    direction = _normalize(axis, 1.0e-15)
    if direction is None:
        raise ValueError("axis cannot be zero")
    sine = math.sin(angle * 0.5)
    return [direction[0] * sine, direction[1] * sine, direction[2] * sine, math.cos(angle * 0.5)]


def _box_inverse_inertia(mass: float, half_extents: Sequence[float]) -> Vec3:
    if mass <= 0.0:
        return [0.0, 0.0, 0.0]
    hx, hy, hz = half_extents
    # Ixx = m / 3 * (hy^2 + hz^2) for full side lengths 2h.
    return [
        3.0 / (mass * (hy * hy + hz * hz)),
        3.0 / (mass * (hx * hx + hz * hz)),
        3.0 / (mass * (hx * hx + hy * hy)),
    ]


def make_sat_box_state(
    worlds: int,
    *,
    seed: int = DEFAULT_SEED,
) -> Tuple[
    List[List[List[float]]],
    List[List[float]],
    List[List[List[float]]],
    List[List[List[float]]],
    Tuple[Tuple[int, int], ...],
]:
    """Create three deterministic adversarial contact pairs per world.

    Pair 0 is axis-aligned face/face, pair 1 is a rotated offset face contact,
    and pair 2 is a skew edge contact. Worlds receive only tiny deterministic
    position perturbations, small enough to preserve their contact class.
    """

    if isinstance(worlds, bool) or not isinstance(worlds, int) or worlds <= 0:
        raise ValueError("worlds must be a positive integer")
    rng = random.Random(seed)
    state: List[List[List[float]]] = []
    inverse_mass: List[List[float]] = []
    half_extents: List[List[List[float]]] = []
    inverse_inertia: List[List[List[float]]] = []
    for world in range(worlds):
        jitter = rng.uniform(-2.0e-4, 2.0e-4)
        face_half = [0.30, 0.24, 0.20]
        rotated_half = [0.28, 0.20, 0.18]
        edge_half = [0.34, 0.10, 0.10]
        rotated_q = _axis_angle((0.0, 0.0, 1.0), math.radians(27.0))
        rotated_normal = _rotate(rotated_q, (1.0, 0.0, 0.0))
        rotated_tangent = _rotate(rotated_q, (0.0, 1.0, 0.0))
        edge_q_a = _axis_angle((0.3, 1.0, 0.2), math.radians(38.0))
        edge_q_b = _axis_angle((1.0, -0.25, 0.7), math.radians(52.0))
        edge_normal = [-0.3332379544674118, 0.9229660570604666, -0.19257757713874876]

        positions = [
            [-2.0, 0.85, 0.0],
            [-1.37 + jitter, 0.85, 0.0],
            [-0.15, 0.85, 0.0],
            _add(
                [-0.15, 0.85, 0.0],
                _add(_scale(rotated_normal, 0.590 + jitter), _scale(rotated_tangent, 0.11)),
            ),
            [1.20, 0.85, 0.0],
            [
                1.4166549203951353 + (0.052 + jitter) * edge_normal[0],
                1.201088713483212 + (0.052 + jitter) * edge_normal[1],
                -0.0124771768707924 + (0.052 + jitter) * edge_normal[2],
            ],
        ]
        quaternions = [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            rotated_q,
            rotated_q,
            edge_q_a,
            edge_q_b,
        ]
        half = [face_half, face_half, rotated_half, rotated_half, edge_half, edge_half]
        velocities = [
            [0.35, 0.0, 0.0],
            [-0.35, 0.0, 0.0],
            _scale(rotated_normal, 0.30),
            _scale(rotated_normal, -0.30),
            _scale(edge_normal, 0.22),
            _scale(edge_normal, -0.22),
        ]
        world_state = [
            [*positions[index], *quaternions[index], *velocities[index], 0.0, 0.0, 0.0]
            for index in range(6)
        ]
        masses = [1.5, 2.0, 1.2, 1.8, 1.0, 1.4]
        state.append(world_state)
        inverse_mass.append([1.0 / mass for mass in masses])
        half_extents.append([item.copy() for item in half])
        inverse_inertia.append(
            [_box_inverse_inertia(mass, half[index]) for index, mass in enumerate(masses)]
        )
    return state, inverse_mass, half_extents, inverse_inertia, PAIR_INDICES


def _validate_layout(
    state: Sequence[Sequence[Sequence[float]]],
    inverse_mass: Sequence[Sequence[float]],
    half_extents: Sequence[Sequence[Sequence[float]]],
    inverse_inertia: Sequence[Sequence[Sequence[float]]],
    pair_indices: Sequence[Sequence[int]],
) -> None:
    if not state:
        raise ValueError("state must contain at least one world")
    if not (len(state) == len(inverse_mass) == len(half_extents) == len(inverse_inertia)):
        raise ValueError("world counts do not match")
    body_count = len(state[0])
    if body_count <= 0:
        raise ValueError("worlds must contain bodies")
    for world_index in range(len(state)):
        if not (
            len(state[world_index])
            == len(inverse_mass[world_index])
            == len(half_extents[world_index])
            == len(inverse_inertia[world_index])
            == body_count
        ):
            raise ValueError("body counts do not match in world {}".format(world_index))
    seen = set()
    for pair in pair_indices:
        if len(pair) != 2:
            raise ValueError("pair indices must contain two body indices")
        a, b = pair
        if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int) or not isinstance(b, int):
            raise ValueError("pair indices must be integers")
        if a == b or not 0 <= a < body_count or not 0 <= b < body_count:
            raise ValueError("invalid body pair {}".format(tuple(pair)))
        canonical = tuple(sorted((a, b)))
        if canonical in seen:
            raise ValueError("duplicate body pair {}".format(canonical))
        seen.add(canonical)


def _integrate_body(box: RigidBox, h: float, gravity_y: float, angular_damping: float) -> None:
    if box.inverse_mass == 0.0:
        return
    box.linear_velocity[1] += gravity_y * h
    box.position = _add(box.position, _scale(box.linear_velocity, h))
    damping = max(0.0, 1.0 - angular_damping * h)
    box.angular_velocity = _scale(box.angular_velocity, damping)
    temporary = box.to_state()
    _integrate_quaternion(temporary, h)
    box.quaternion = temporary[3:7]


def step_sat_reference(
    state: Sequence[Sequence[Sequence[float]]],
    inverse_mass: Sequence[Sequence[float]],
    half_extents: Sequence[Sequence[Sequence[float]]],
    inverse_inertia: Sequence[Sequence[Sequence[float]]],
    pair_indices: Sequence[Sequence[int]] = PAIR_INDICES,
    config: SATConfig = SATConfig(),
    *,
    steps: int = 1,
) -> Tuple[List[List[List[float]]], List[List[bool]], List[List[float]]]:
    """Integrate and solve the explicitly supplied box pairs.

    ``contacts`` is true if a pair touched/overlapped at any solver visit.
    ``penetration`` is the final SAT minimum-translation depth, or zero when
    separated. This function performs no broad phase and resolves no unlisted
    pairs.
    """

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    _validate_layout(state, inverse_mass, half_extents, inverse_inertia, pair_indices)
    worlds: List[List[RigidBox]] = []
    for world_index, world in enumerate(state):
        worlds.append(
            [
                RigidBox.from_state(
                    body,
                    inverse_mass[world_index][body_index],
                    half_extents[world_index][body_index],
                    inverse_inertia[world_index][body_index],
                )
                for body_index, body in enumerate(world)
            ]
        )
    contacts = [[False for _ in pair_indices] for _ in worlds]
    h = config.dt / config.substeps
    for _ in range(steps):
        for _ in range(config.substeps):
            for world in worlds:
                for body in world:
                    _integrate_body(body, h, config.gravity_y, config.angular_damping)
            for _ in range(config.solver_iterations):
                for world_index, world in enumerate(worlds):
                    for pair_index, (a, b) in enumerate(pair_indices):
                        query = sat_query(world[a], world[b], config.sat_epsilon)
                        if query.contact is None:
                            continue
                        contacts[world_index][pair_index] = True
                        solve_contact(world[a], world[b], query.contact, config)

    penetration = [[0.0 for _ in pair_indices] for _ in worlds]
    for world_index, world in enumerate(worlds):
        for pair_index, (a, b) in enumerate(pair_indices):
            query = sat_query(world[a], world[b], config.sat_epsilon)
            if query.contact is not None:
                penetration[world_index][pair_index] = query.contact.depth
    output = [[body.to_state() for body in world] for world in worlds]
    return output, contacts, penetration


def assert_valid_sat_boxes(
    state: Sequence[Sequence[Sequence[float]]],
    inverse_mass: Sequence[Sequence[float]],
    half_extents: Sequence[Sequence[Sequence[float]]],
    inverse_inertia: Sequence[Sequence[Sequence[float]]],
    pair_indices: Sequence[Sequence[int]] = PAIR_INDICES,
    *,
    max_penetration: float = 0.005,
) -> None:
    _validate_layout(state, inverse_mass, half_extents, inverse_inertia, pair_indices)
    maximum = _finite(max_penetration, "max_penetration")
    if maximum < 0.0:
        raise ValueError("max_penetration cannot be negative")
    for world_index, world in enumerate(state):
        boxes = []
        for body_index, body in enumerate(world):
            if len(body) != STATE_WIDTH or not all(math.isfinite(value) for value in body):
                raise AssertionError("malformed body {}:{}".format(world_index, body_index))
            norm = math.sqrt(sum(value * value for value in body[3:7]))
            if abs(norm - 1.0) > 2.0e-5:
                raise AssertionError("non-unit quaternion {}:{}".format(world_index, body_index))
            boxes.append(
                RigidBox.from_state(
                    body,
                    inverse_mass[world_index][body_index],
                    half_extents[world_index][body_index],
                    inverse_inertia[world_index][body_index],
                )
            )
        for pair_index, (a, b) in enumerate(pair_indices):
            query = sat_query(boxes[a], boxes[b])
            depth = query.contact.depth if query.contact is not None else 0.0
            if depth > maximum:
                raise AssertionError(
                    "pair penetration {}:{} is {:.9f} m".format(
                        world_index, pair_index, depth
                    )
                )


__all__ = [
    "CONTRACT_ID",
    "DEFAULT_SEED",
    "ANGULAR_RESPONSE_THRESHOLD_RAD_S",
    "MAX_POSITION_REPAIR_M",
    "PAIR_INDICES",
    "ImpulseResult",
    "RigidBox",
    "SATConfig",
    "SATContact",
    "SATQuery",
    "assert_valid_sat_boxes",
    "make_sat_box_state",
    "obb_contact",
    "sat_query",
    "solve_contact",
    "step_sat_reference",
]
