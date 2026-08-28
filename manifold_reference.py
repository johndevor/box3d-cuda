"""CPU reference for persistent multi-point OBB contact manifolds.

Face contacts use deterministic incident-face clipping against the four side
planes of the reference face and retain at most four points. Edge-axis SAT
contacts use a deterministic one-point support fallback. Stable integer feature
IDs describe topology rather than quantized world positions.

The public cache layout is deliberately CUDA-friendly:

* feature IDs: ``[world, pair, 4]`` signed 64-bit-compatible integers; zero is empty
* impulses: ``[world, pair, 4, 3]`` normal, tangent-1, tangent-2 accumulators

This is a bounded deterministic oracle. It is not a GPU parity claim or a full
replacement for a production manifold/contact graph solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from .reference import STATE_WIDTH, _integrate_quaternion
from .sat_reference import (
    MAX_POSITION_REPAIR_M,
    RigidBox,
    SATConfig,
    SATContact,
)


CONTRACT_ID = "box3d.obb-manifold/v1"
DEFAULT_SEED = 41
STACK_PAIR_INDICES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
)
PAIR_ROLES = (
    "floor_to_stack_1",
    "stack_1_to_2",
    "stack_2_to_3",
    "stack_3_to_4",
    "floor_to_slider",
)
STACK_PAIR_COUNT = 4
SLIDER_PAIR_INDEX = 4
INITIAL_SLIDER_SPEED_MPS = 1.5
BENCHMARK_STEPS = 720
TAIL_WINDOW_STEPS = 120
MAX_STACK_HEIGHT_ERROR_M = 0.01
MIN_STACK_CENTER_GAP_M = 0.48
MAX_FINAL_PENETRATION_M = 0.005
MAX_TAIL_POSITION_JITTER_M = 0.003
MAX_TAIL_LINEAR_SPEED_MPS = 0.05
MAX_TAIL_ANGULAR_SPEED_RAD_S = 0.15
MIN_PERSISTENT_CONTACT_FRAMES = 120
MAX_SLIDER_FINAL_SPEED_RATIO = 0.05
# A discrete iterative solver can show sub-millimetre-per-second stick/slip
# ripple near rest. Keep this below 0.14% of the 1.5 m/s initial speed while
# separately requiring at least 95% total slowdown and no sustained increase.
MAX_SLIDER_SPEED_INCREASE_MPS = 0.002
MAX_MANIFOLD_POINTS = 4
SAT_AXIS_TIE_EPSILON_M = 1.0e-6
_FACE_FEATURE_TAG = 1 << 60
_EDGE_FEATURE_TAG = 2 << 60

Vec3 = List[float]


@dataclass
class ManifoldPoint:
    feature_id: int
    position: Vec3
    depth: float
    normal_impulse: float = 0.0
    tangent_impulse_1: float = 0.0
    tangent_impulse_2: float = 0.0


@dataclass
class ContactManifold:
    normal: Vec3
    tangent_1: Vec3
    tangent_2: Vec3
    points: List[ManifoldPoint]
    kind: str
    sat_axis_index: int


@dataclass
class _ClipVertex:
    position: Vec3
    source_mask: int
    clip_mask: int = 0


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


def _normalize(vector: Sequence[float], epsilon: float) -> Vec3:
    length = math.sqrt(_dot(vector, vector))
    if length <= epsilon:
        raise ValueError("cannot normalize a degenerate vector")
    return [value / length for value in vector]


def _rotate(quaternion: Sequence[float], vector: Sequence[float]) -> Vec3:
    qv = quaternion[:3]
    twice = _scale(_cross(qv, vector), 2.0)
    second = _cross(qv, twice)
    return [vector[index] + quaternion[3] * twice[index] + second[index] for index in range(3)]


def _inverse_rotate(quaternion: Sequence[float], vector: Sequence[float]) -> Vec3:
    return _rotate((-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]), vector)


def _box_axes(box: RigidBox) -> Tuple[Vec3, Vec3, Vec3]:
    return (
        _rotate(box.quaternion, (1.0, 0.0, 0.0)),
        _rotate(box.quaternion, (0.0, 1.0, 0.0)),
        _rotate(box.quaternion, (0.0, 0.0, 1.0)),
    )


def tangent_basis(normal: Sequence[float], epsilon: float = 1.0e-12) -> Tuple[Vec3, Vec3]:
    """Return a deterministic orthonormal 2-D basis for the tangent plane."""

    cardinal = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    seed = min(cardinal, key=lambda axis: (abs(_dot(axis, normal)), cardinal.index(axis)))
    tangent_1 = _normalize(_cross(seed, normal), epsilon)
    tangent_2 = _normalize(_cross(normal, tangent_1), epsilon)
    return tangent_1, tangent_2


def _clip_polygon(
    polygon: List[_ClipVertex],
    origin: Sequence[float],
    axis: Sequence[float],
    bound: float,
    plane_bit: int,
    epsilon: float,
) -> List[_ClipVertex]:
    if not polygon:
        return []

    def distance(vertex: _ClipVertex) -> float:
        return _dot(_sub(vertex.position, origin), axis) - bound

    result: List[_ClipVertex] = []
    previous = polygon[-1]
    previous_distance = distance(previous)
    previous_inside = previous_distance <= epsilon
    for current in polygon:
        current_distance = distance(current)
        current_inside = current_distance <= epsilon
        if current_inside != previous_inside:
            denominator = previous_distance - current_distance
            fraction = 0.5 if abs(denominator) <= epsilon else previous_distance / denominator
            fraction = max(0.0, min(1.0, fraction))
            point = _add(
                previous.position,
                _scale(_sub(current.position, previous.position), fraction),
            )
            result.append(
                _ClipVertex(
                    point,
                    previous.source_mask | current.source_mask,
                    previous.clip_mask | current.clip_mask | plane_bit,
                )
            )
        if current_inside:
            result.append(current)
        previous = current
        previous_distance = current_distance
        previous_inside = current_inside
    return result


def _vertex_bit(signs: Sequence[int]) -> int:
    index = (1 if signs[0] > 0 else 0) | (2 if signs[1] > 0 else 0) | (4 if signs[2] > 0 else 0)
    return 1 << index


def _face_feature_id(
    pair_index: int,
    reference_is_b: bool,
    reference_axis: int,
    reference_sign: int,
    incident_axis: int,
    incident_sign: int,
    source_mask: int,
    clip_mask: int,
) -> int:
    return (
        _FACE_FEATURE_TAG
        | ((pair_index & 0xFF) << 48)
        | ((1 if reference_is_b else 0) << 47)
        | ((reference_axis & 0x3) << 45)
        | ((1 if reference_sign > 0 else 0) << 44)
        | ((incident_axis & 0x3) << 42)
        | ((1 if incident_sign > 0 else 0) << 41)
        | ((source_mask & 0xFF) << 8)
        | (clip_mask & 0xFF)
    )


def _edge_feature_id(pair_index: int, contact: SATContact) -> int:
    edge_index = max(0, contact.axis_index - 6)
    axis_a, axis_b = divmod(edge_index, 3)
    normal_bits = sum((1 if component >= 0.0 else 0) << index for index, component in enumerate(contact.normal))
    return (
        _EDGE_FEATURE_TAG
        | ((pair_index & 0xFF) << 48)
        | ((axis_a & 0x3) << 44)
        | ((axis_b & 0x3) << 42)
        | normal_bits
        | 1
    )


def _manifold_sat_contact(
    box_a: RigidBox,
    box_b: RigidBox,
    sat_epsilon: float,
) -> Optional[SATContact]:
    """Stage-4 SAT with stable topology selection for redundant axes.

    Near-parallel boxes often have face and edge-cross candidates describing
    the same separating direction. Their depths differ at float round-off
    scale, so selecting the raw minimum makes feature topology dtype-dependent.
    Within the declared metric tie band, face axes win over edge axes and then
    the lower candidate index wins. Meaningfully shallower axes still win.
    """

    axes_a = _box_axes(box_a)
    axes_b = _box_axes(box_b)
    candidates = (
        [(axis, "face_a") for axis in axes_a]
        + [(axis, "face_b") for axis in axes_b]
        + [(_cross(axis_a, axis_b), "edge") for axis_a in axes_a for axis_b in axes_b]
    )
    center_delta = _sub(box_b.position, box_a.position)
    maximum_separation = -math.inf
    records: List[Tuple[float, Vec3, int, str]] = []
    tested = 0
    for index, (candidate, kind) in enumerate(candidates):
        length_squared = _dot(candidate, candidate)
        if length_squared <= sat_epsilon * sat_epsilon:
            continue
        tested += 1
        inverse_length = 1.0 / math.sqrt(length_squared)
        axis = _scale(candidate, inverse_length)
        signed_distance = _dot(center_delta, axis)
        radius = sum(
            box_a.half_extents[k] * abs(_dot(axes_a[k], axis))
            + box_b.half_extents[k] * abs(_dot(axes_b[k], axis))
            for k in range(3)
        )
        separation = abs(signed_distance) - radius
        maximum_separation = max(maximum_separation, separation)
        depth = -separation
        oriented = axis if signed_distance >= 0.0 else _scale(axis, -1.0)
        records.append((depth, oriented, index, kind))
    if maximum_separation > sat_epsilon:
        return None
    if not records or tested < 6:
        raise AssertionError("manifold SAT did not select a valid face axis")
    minimum_depth = min(record[0] for record in records)
    eligible = [
        record for record in records
        if record[0] <= minimum_depth + SAT_AXIS_TIE_EPSILON_M
    ]
    _, normal, axis_index, axis_kind = min(
        eligible,
        key=lambda record: (1 if record[3] == "edge" else 0, record[2]),
    )

    def support(box: RigidBox, axes: Sequence[Sequence[float]], direction: Sequence[float]) -> Vec3:
        point = list(box.position)
        for index, axis in enumerate(axes):
            alignment = _dot(axis, direction)
            sign = 1.0 if alignment > sat_epsilon else (-1.0 if alignment < -sat_epsilon else 0.0)
            point = _add(point, _scale(axis, sign * box.half_extents[index]))
        return point

    point_a = support(box_a, axes_a, normal)
    point_b = support(box_b, axes_b, _scale(normal, -1.0))
    midpoint = _scale(_add(point_a, point_b), 0.5)
    return SATContact(
        tuple(normal),
        max(0.0, minimum_depth),
        tuple(midpoint),
        axis_index,
        axis_kind,
        15,
        tested,
    )


def _reduce_points(points: List[ManifoldPoint], tangent_1: Sequence[float], tangent_2: Sequence[float]) -> List[ManifoldPoint]:
    if len(points) <= MAX_MANIFOLD_POINTS:
        return sorted(points, key=lambda point: point.feature_id)
    # Preserve the patch extent using deterministic extrema in both tangent
    # directions, then fill any duplicate slots by deepest feature order.
    choices: List[ManifoldPoint] = []
    for tangent, sign in ((tangent_1, -1), (tangent_1, 1), (tangent_2, -1), (tangent_2, 1)):
        selected = min(
            points,
            key=lambda point: (
                sign * _dot(point.position, tangent),
                -point.depth,
                point.feature_id,
            ),
        )
        if all(selected.feature_id != existing.feature_id for existing in choices):
            choices.append(selected)
    for point in sorted(points, key=lambda item: (-item.depth, item.feature_id)):
        if len(choices) >= MAX_MANIFOLD_POINTS:
            break
        if all(point.feature_id != existing.feature_id for existing in choices):
            choices.append(point)
    return sorted(choices[:MAX_MANIFOLD_POINTS], key=lambda point: point.feature_id)


def build_manifold(
    box_a: RigidBox,
    box_b: RigidBox,
    pair_index: int = 0,
    sat_epsilon: float = 1.0e-7,
) -> Optional[ContactManifold]:
    contact = _manifold_sat_contact(box_a, box_b, sat_epsilon)
    if contact is None:
        return None
    normal = list(contact.normal)
    tangent_1, tangent_2 = tangent_basis(normal)
    if contact.axis_kind == "edge":
        return ContactManifold(
            normal,
            tangent_1,
            tangent_2,
            [
                ManifoldPoint(
                    _edge_feature_id(pair_index, contact),
                    list(contact.point),
                    contact.depth,
                )
            ],
            "edge",
            contact.axis_index,
        )

    reference_is_b = contact.axis_kind == "face_b"
    reference = box_b if reference_is_b else box_a
    incident = box_a if reference_is_b else box_b
    reference_outward = _scale(normal, -1.0) if reference_is_b else normal
    reference_axes = _box_axes(reference)
    incident_axes = _box_axes(incident)
    reference_axis = contact.axis_index - 3 if reference_is_b else contact.axis_index
    reference_sign = 1 if _dot(reference_axes[reference_axis], reference_outward) >= 0.0 else -1
    reference_center = _add(
        reference.position,
        _scale(reference_axes[reference_axis], reference_sign * reference.half_extents[reference_axis]),
    )
    side_indices = [index for index in range(3) if index != reference_axis]

    incident_axis = max(range(3), key=lambda index: abs(_dot(incident_axes[index], reference_outward)))
    incident_sign = -1 if _dot(incident_axes[incident_axis], reference_outward) >= 0.0 else 1
    incident_center = _add(
        incident.position,
        _scale(incident_axes[incident_axis], incident_sign * incident.half_extents[incident_axis]),
    )
    incident_sides = [index for index in range(3) if index != incident_axis]
    polygon: List[_ClipVertex] = []
    for first, second in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        signs = [0, 0, 0]
        signs[incident_axis] = incident_sign
        signs[incident_sides[0]] = first
        signs[incident_sides[1]] = second
        point = _add(
            incident_center,
            _add(
                _scale(
                    incident_axes[incident_sides[0]],
                    first * incident.half_extents[incident_sides[0]],
                ),
                _scale(
                    incident_axes[incident_sides[1]],
                    second * incident.half_extents[incident_sides[1]],
                ),
            ),
        )
        polygon.append(_ClipVertex(point, _vertex_bit(signs)))

    plane = 0
    for side_index in side_indices:
        axis = reference_axes[side_index]
        extent = reference.half_extents[side_index]
        polygon = _clip_polygon(
            polygon, reference_center, axis, extent, 1 << plane, sat_epsilon
        )
        plane += 1
        polygon = _clip_polygon(
            polygon, reference_center, _scale(axis, -1.0), extent, 1 << plane, sat_epsilon
        )
        plane += 1

    points: List[ManifoldPoint] = []
    for vertex in polygon:
        plane_distance = _dot(_sub(vertex.position, reference_center), reference_outward)
        if plane_distance > sat_epsilon:
            continue
        depth = max(0.0, -plane_distance)
        midpoint = _sub(vertex.position, _scale(reference_outward, plane_distance * 0.5))
        feature_id = _face_feature_id(
            pair_index,
            reference_is_b,
            reference_axis,
            reference_sign,
            incident_axis,
            incident_sign,
            vertex.source_mask,
            vertex.clip_mask,
        )
        points.append(ManifoldPoint(feature_id, midpoint, depth))

    if not points:
        # Rare numerical/topological fallback remains explicit and one-point.
        return ContactManifold(
            normal,
            tangent_1,
            tangent_2,
            [ManifoldPoint(_edge_feature_id(pair_index, contact), list(contact.point), contact.depth)],
            "support_fallback",
            contact.axis_index,
        )
    return ContactManifold(
        normal,
        tangent_1,
        tangent_2,
        _reduce_points(points, tangent_1, tangent_2),
        "face",
        contact.axis_index,
    )


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
    return (
        box_a.inverse_mass
        + box_b.inverse_mass
        + _dot(_cross(angular_a, offset_a), direction)
        + _dot(_cross(angular_b, offset_b), direction)
    )


def _apply_impulse(box: RigidBox, offset: Sequence[float], impulse: Sequence[float]) -> None:
    if box.inverse_mass == 0.0:
        return
    box.linear_velocity = _add(box.linear_velocity, _scale(impulse, box.inverse_mass))
    box.angular_velocity = _add(
        box.angular_velocity,
        _inverse_inertia_world(box, _cross(offset, impulse)),
    )


def _apply_pair_impulse(
    box_a: RigidBox,
    box_b: RigidBox,
    point: ManifoldPoint,
    impulse: Sequence[float],
) -> None:
    offset_a = _sub(point.position, box_a.position)
    offset_b = _sub(point.position, box_b.position)
    _apply_impulse(box_a, offset_a, _scale(impulse, -1.0))
    _apply_impulse(box_b, offset_b, impulse)


def _warm_start(box_a: RigidBox, box_b: RigidBox, manifold: ContactManifold) -> None:
    for point in manifold.points:
        impulse = _add(
            _scale(manifold.normal, point.normal_impulse),
            _add(
                _scale(manifold.tangent_1, point.tangent_impulse_1),
                _scale(manifold.tangent_2, point.tangent_impulse_2),
            ),
        )
        _apply_pair_impulse(box_a, box_b, point, impulse)


def _solve_normal_point(
    box_a: RigidBox,
    box_b: RigidBox,
    manifold: ContactManifold,
    point: ManifoldPoint,
    config: SATConfig,
    h: float,
) -> None:
    offset_a = _sub(point.position, box_a.position)
    offset_b = _sub(point.position, box_b.position)
    relative = _sub(_point_velocity(box_b, offset_b), _point_velocity(box_a, offset_a))
    normal_speed = _dot(relative, manifold.normal)
    denominator = _effective_mass(
        box_a, box_b, offset_a, offset_b, manifold.normal
    )
    if denominator > config.sat_epsilon:
        bounce = -config.restitution * normal_speed if normal_speed < -0.5 else 0.0
        # Penetration drift is handled by the bounded split correction below,
        # not by injecting Baumgarte velocity. Keeping the velocity solve free
        # of penetration energy materially reduces resting-stack jitter.
        delta = (bounce - normal_speed) / denominator
        previous = point.normal_impulse
        point.normal_impulse = max(0.0, previous + delta)
        _apply_pair_impulse(
            box_a,
            box_b,
            point,
            _scale(manifold.normal, point.normal_impulse - previous),
        )


def _solve_friction_point(
    box_a: RigidBox,
    box_b: RigidBox,
    manifold: ContactManifold,
    point: ManifoldPoint,
    config: SATConfig,
) -> None:
    offset_a = _sub(point.position, box_a.position)
    offset_b = _sub(point.position, box_b.position)
    relative = _sub(_point_velocity(box_b, offset_b), _point_velocity(box_a, offset_a))
    tangent_denominators = (
        _effective_mass(box_a, box_b, offset_a, offset_b, manifold.tangent_1),
        _effective_mass(box_a, box_b, offset_a, offset_b, manifold.tangent_2),
    )
    proposed = [point.tangent_impulse_1, point.tangent_impulse_2]
    for index, tangent in enumerate((manifold.tangent_1, manifold.tangent_2)):
        if tangent_denominators[index] > config.sat_epsilon:
            proposed[index] += -_dot(relative, tangent) / tangent_denominators[index]
    limit = config.friction * point.normal_impulse
    magnitude = math.hypot(proposed[0], proposed[1])
    if magnitude > limit and magnitude > config.sat_epsilon:
        proposed = [value * limit / magnitude for value in proposed]
    delta_tangent = _add(
        _scale(manifold.tangent_1, proposed[0] - point.tangent_impulse_1),
        _scale(manifold.tangent_2, proposed[1] - point.tangent_impulse_2),
    )
    point.tangent_impulse_1, point.tangent_impulse_2 = proposed
    _apply_pair_impulse(box_a, box_b, point, delta_tangent)


def _repair_positions(
    box_a: RigidBox,
    box_b: RigidBox,
    manifold: ContactManifold,
    config: SATConfig,
) -> float:
    inverse_sum = box_a.inverse_mass + box_b.inverse_mass
    if inverse_sum <= 0.0:
        return 0.0
    maximum_depth = max(point.depth for point in manifold.points)
    repair = min(
        MAX_POSITION_REPAIR_M,
        max(0.0, maximum_depth - config.position_slop) * config.position_correction,
    )
    correction = _scale(manifold.normal, repair / inverse_sum)
    box_a.position = _sub(box_a.position, _scale(correction, box_a.inverse_mass))
    box_b.position = _add(box_b.position, _scale(correction, box_b.inverse_mass))
    return repair


def _validate_cache(
    worlds: int,
    pairs: int,
    feature_ids: Sequence[Sequence[Sequence[int]]],
    impulses: Sequence[Sequence[Sequence[Sequence[float]]]],
) -> None:
    if len(feature_ids) != worlds or len(impulses) != worlds:
        raise ValueError("cache world count mismatch")
    for world in range(worlds):
        if len(feature_ids[world]) != pairs or len(impulses[world]) != pairs:
            raise ValueError("cache pair count mismatch")
        for pair in range(pairs):
            if len(feature_ids[world][pair]) != MAX_MANIFOLD_POINTS:
                raise ValueError("feature cache must have four slots")
            if len(impulses[world][pair]) != MAX_MANIFOLD_POINTS:
                raise ValueError("impulse cache must have four slots")
            for slot in range(MAX_MANIFOLD_POINTS):
                feature = feature_ids[world][pair][slot]
                values = impulses[world][pair][slot]
                if isinstance(feature, bool) or not isinstance(feature, int) or feature < 0:
                    raise ValueError("cache feature IDs must be non-negative integers")
                if len(values) != 3 or not all(math.isfinite(float(value)) for value in values):
                    raise ValueError("cache impulse entries must contain three finite values")


def empty_manifold_cache(
    worlds: int, pairs: int
) -> Tuple[List[List[List[int]]], List[List[List[List[float]]]]]:
    return (
        [[[0 for _ in range(MAX_MANIFOLD_POINTS)] for _ in range(pairs)] for _ in range(worlds)],
        [
            [
                [[0.0, 0.0, 0.0] for _ in range(MAX_MANIFOLD_POINTS)]
                for _ in range(pairs)
            ]
            for _ in range(worlds)
        ],
    )


def _seed_from_cache(
    manifold: ContactManifold,
    feature_ids: Sequence[int],
    impulses: Sequence[Sequence[float]],
) -> None:
    cached = {
        feature_ids[index]: impulses[index]
        for index in range(MAX_MANIFOLD_POINTS)
        if feature_ids[index] != 0
    }
    for point in manifold.points:
        values = cached.get(point.feature_id)
        if values is not None:
            point.normal_impulse = max(0.0, float(values[0]))
            point.tangent_impulse_1 = float(values[1])
            point.tangent_impulse_2 = float(values[2])


def _write_cache(
    manifold: Optional[ContactManifold],
) -> Tuple[List[int], List[List[float]]]:
    identifiers = [0] * MAX_MANIFOLD_POINTS
    impulses = [[0.0, 0.0, 0.0] for _ in range(MAX_MANIFOLD_POINTS)]
    if manifold is not None:
        for slot, point in enumerate(sorted(manifold.points, key=lambda item: item.feature_id)):
            identifiers[slot] = point.feature_id
            impulses[slot] = [
                point.normal_impulse,
                point.tangent_impulse_1,
                point.tangent_impulse_2,
            ]
    return identifiers, impulses


def _box_inverse_inertia(mass: float, half: Sequence[float]) -> Vec3:
    if mass <= 0.0:
        return [0.0, 0.0, 0.0]
    hx, hy, hz = half
    return [
        3.0 / (mass * (hy * hy + hz * hz)),
        3.0 / (mass * (hx * hx + hz * hz)),
        3.0 / (mass * (hx * hx + hy * hy)),
    ]


def make_manifold_stack_state(
    worlds: int,
    *,
    seed: int = DEFAULT_SEED,
) -> Tuple[
    List[List[List[float]]],
    List[List[float]],
    List[List[List[float]]],
    List[List[List[float]]],
    Tuple[Tuple[int, int], ...],
    List[List[List[int]]],
    List[List[List[List[float]]]],
]:
    """Create a static floor, a four-box stack, and an independent slider."""

    if isinstance(worlds, bool) or not isinstance(worlds, int) or worlds <= 0:
        raise ValueError("worlds must be a positive integer")
    rng = random.Random(seed)
    all_state = []
    all_mass = []
    all_half = []
    all_inertia = []
    for world in range(worlds):
        jitter = rng.uniform(-1.0e-5, 1.0e-5)
        floor_half = [4.0, 0.25, 2.0]
        cube_half = [0.25, 0.25, 0.25]
        state = [[0.0, -0.25, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        for level in range(4):
            state.append(
                [
                    -1.0 + jitter * (level + 1),
                    0.2502 + 0.5001 * level,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
        state.append(
            [2.0, 0.2502, 0.0, 0.0, 0.0, 0.0, 1.0, INITIAL_SLIDER_SPEED_MPS, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        masses = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        half = [floor_half] + [cube_half.copy() for _ in range(5)]
        all_state.append(state)
        all_mass.append([0.0] + [1.0] * 5)
        all_half.append(half)
        all_inertia.append(
            [[0.0, 0.0, 0.0]]
            + [_box_inverse_inertia(mass, cube_half) for mass in masses[1:]]
        )
    cache_ids, cache_impulses = empty_manifold_cache(worlds, len(STACK_PAIR_INDICES))
    return (
        all_state,
        all_mass,
        all_half,
        all_inertia,
        STACK_PAIR_INDICES,
        cache_ids,
        cache_impulses,
    )


def _validate_layout(
    state: Sequence[Sequence[Sequence[float]]],
    inverse_mass: Sequence[Sequence[float]],
    half_extents: Sequence[Sequence[Sequence[float]]],
    inverse_inertia: Sequence[Sequence[Sequence[float]]],
    pair_indices: Sequence[Sequence[int]],
) -> None:
    if not state or not pair_indices:
        raise ValueError("state and pair_indices cannot be empty")
    if not (len(state) == len(inverse_mass) == len(half_extents) == len(inverse_inertia)):
        raise ValueError("world counts do not match")
    bodies = len(state[0])
    for world in range(len(state)):
        if not (
            len(state[world])
            == len(inverse_mass[world])
            == len(half_extents[world])
            == len(inverse_inertia[world])
            == bodies
        ):
            raise ValueError("body counts do not match")
    for pair in pair_indices:
        if len(pair) != 2 or pair[0] == pair[1] or not all(
            isinstance(index, int) and not isinstance(index, bool) and 0 <= index < bodies
            for index in pair
        ):
            raise ValueError("invalid pair indices")


def _integrate(box: RigidBox, h: float, config: SATConfig) -> None:
    if box.inverse_mass == 0.0:
        return
    box.linear_velocity[1] += config.gravity_y * h
    box.position = _add(box.position, _scale(box.linear_velocity, h))
    damping = max(0.0, 1.0 - config.angular_damping * h)
    box.angular_velocity = _scale(box.angular_velocity, damping)
    body = box.to_state()
    _integrate_quaternion(body, h)
    box.quaternion = body[3:7]


def step_manifold_reference(
    state: Sequence[Sequence[Sequence[float]]],
    inverse_mass: Sequence[Sequence[float]],
    half_extents: Sequence[Sequence[Sequence[float]]],
    inverse_inertia: Sequence[Sequence[Sequence[float]]],
    pair_indices: Sequence[Sequence[int]],
    cache_feature_ids: Sequence[Sequence[Sequence[int]]],
    cache_impulses: Sequence[Sequence[Sequence[Sequence[float]]]],
    config: SATConfig = SATConfig(),
    *,
    steps: int = 1,
    warm_start: bool = True,
) -> Tuple[
    List[List[List[float]]],
    List[List[bool]],
    List[List[float]],
    List[List[List[int]]],
    List[List[List[List[float]]]],
    List[List[int]],
]:
    """Advance fixed box pairs and return state, contact and cache tensors.

    ``warm_start=False`` deliberately ignores incoming cached impulses while
    still returning a newly populated cache. It is the clean cold-solver
    baseline used by parity/convergence gates; it does not change topology or
    integration semantics.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be positive")
    if not isinstance(warm_start, bool):
        raise ValueError("warm_start must be a boolean")
    _validate_layout(state, inverse_mass, half_extents, inverse_inertia, pair_indices)
    _validate_cache(len(state), len(pair_indices), cache_feature_ids, cache_impulses)
    worlds = [
        [
            RigidBox.from_state(
                body,
                inverse_mass[world][index],
                half_extents[world][index],
                inverse_inertia[world][index],
            )
            for index, body in enumerate(state[world])
        ]
        for world in range(len(state))
    ]
    ids = [[list(pair) for pair in world] for world in cache_feature_ids]
    impulses = [
        [[list(slot) for slot in pair] for pair in world]
        for world in cache_impulses
    ]
    contacts = [[False for _ in pair_indices] for _ in worlds]
    h = config.dt / config.substeps

    for control_step in range(steps):
        if control_step:
            # Repeated public calls reconstruct RigidBox and normalize its
            # quaternion in __post_init__. Mirror that call boundary so a
            # batched steps=N call follows the identical arithmetic path.
            for world in worlds:
                for body in world:
                    length = math.sqrt(sum(value * value for value in body.quaternion))
                    body.quaternion = [value / length for value in body.quaternion]
        for _ in range(config.substeps):
            for world in worlds:
                for body in world:
                    _integrate(body, h, config)

            manifolds: List[List[Optional[ContactManifold]]] = []
            for world_index, world in enumerate(worlds):
                world_manifolds = []
                for pair_index, (a, b) in enumerate(pair_indices):
                    manifold = build_manifold(world[a], world[b], pair_index, config.sat_epsilon)
                    if manifold is not None:
                        contacts[world_index][pair_index] = True
                        if warm_start:
                            _seed_from_cache(
                                manifold,
                                ids[world_index][pair_index],
                                impulses[world_index][pair_index],
                            )
                            _warm_start(world[a], world[b], manifold)
                    world_manifolds.append(manifold)
                manifolds.append(world_manifolds)

            for _ in range(config.solver_iterations):
                for world_index, world in enumerate(worlds):
                    for pair_index, (a, b) in enumerate(pair_indices):
                        manifold = manifolds[world_index][pair_index]
                        if manifold is None:
                            continue
                        # Solve the complete normal patch before friction. This
                        # avoids turning transient corner imbalance into a
                        # lateral friction torque on otherwise symmetric faces.
                        for point in manifold.points:
                            _solve_normal_point(world[a], world[b], manifold, point, config, h)
                        for point in manifold.points:
                            _solve_friction_point(world[a], world[b], manifold, point, config)

            for world_index, world in enumerate(worlds):
                for pair_index, (a, b) in enumerate(pair_indices):
                    manifold = manifolds[world_index][pair_index]
                    if manifold is not None:
                        _repair_positions(world[a], world[b], manifold, config)
                    ids[world_index][pair_index], impulses[world_index][pair_index] = _write_cache(manifold)

        # CUDA manifold_step returns after one control step and refreshes its
        # final topology at that boundary. Refresh here on every outer step so
        # ``steps=N`` is semantically identical to N calls with ``steps=1``.
        penetration = [[0.0 for _ in pair_indices] for _ in worlds]
        contact_count = [[0 for _ in pair_indices] for _ in worlds]
        for world_index, world in enumerate(worlds):
            for pair_index, (a, b) in enumerate(pair_indices):
                final = build_manifold(world[a], world[b], pair_index, config.sat_epsilon)
                if final is None:
                    ids[world_index][pair_index], impulses[world_index][pair_index] = _write_cache(None)
                    continue
                cached = {
                    ids[world_index][pair_index][slot]: impulses[world_index][pair_index][slot]
                    for slot in range(MAX_MANIFOLD_POINTS)
                    if ids[world_index][pair_index][slot] != 0
                }
                for point in final.points:
                    values = cached.get(point.feature_id)
                    if values is not None:
                        point.normal_impulse, point.tangent_impulse_1, point.tangent_impulse_2 = values
                ids[world_index][pair_index], impulses[world_index][pair_index] = _write_cache(final)
                contact_count[world_index][pair_index] = len(final.points)
                penetration[world_index][pair_index] = max(point.depth for point in final.points)

    output = [[body.to_state() for body in world] for world in worlds]
    return output, contacts, penetration, ids, impulses, contact_count


def assert_valid_manifold_state(
    state: Sequence[Sequence[Sequence[float]]],
    penetration: Sequence[Sequence[float]],
    *,
    max_penetration: float = 0.005,
) -> None:
    if len(state) != len(penetration):
        raise AssertionError("state/penetration world count mismatch")
    for world_index, world in enumerate(state):
        for body_index, body in enumerate(world):
            if len(body) != STATE_WIDTH or not all(math.isfinite(float(value)) for value in body):
                raise AssertionError("malformed body {}:{}".format(world_index, body_index))
            quaternion_norm = math.sqrt(sum(float(value) ** 2 for value in body[3:7]))
            if abs(quaternion_norm - 1.0) > 2.0e-5:
                raise AssertionError("non-unit quaternion {}:{}".format(world_index, body_index))
        if any(not math.isfinite(float(value)) or value < 0 or value > max_penetration for value in penetration[world_index]):
            raise AssertionError("penetration bound exceeded in world {}".format(world_index))


__all__ = [
    "CONTRACT_ID",
    "DEFAULT_SEED",
    "MAX_MANIFOLD_POINTS",
    "SAT_AXIS_TIE_EPSILON_M",
    "PAIR_ROLES",
    "STACK_PAIR_COUNT",
    "STACK_PAIR_INDICES",
    "SLIDER_PAIR_INDEX",
    "INITIAL_SLIDER_SPEED_MPS",
    "BENCHMARK_STEPS",
    "TAIL_WINDOW_STEPS",
    "MAX_STACK_HEIGHT_ERROR_M",
    "MIN_STACK_CENTER_GAP_M",
    "MAX_FINAL_PENETRATION_M",
    "MAX_TAIL_POSITION_JITTER_M",
    "MAX_TAIL_LINEAR_SPEED_MPS",
    "MAX_TAIL_ANGULAR_SPEED_RAD_S",
    "MIN_PERSISTENT_CONTACT_FRAMES",
    "MAX_SLIDER_FINAL_SPEED_RATIO",
    "MAX_SLIDER_SPEED_INCREASE_MPS",
    "ContactManifold",
    "ManifoldPoint",
    "assert_valid_manifold_state",
    "build_manifold",
    "empty_manifold_cache",
    "make_manifold_stack_state",
    "step_manifold_reference",
    "tangent_basis",
]
