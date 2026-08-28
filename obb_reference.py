"""CPU oracle for oriented boxes contacting a static plane.

This stage adds the pieces that make the port genuinely three-dimensional:
world-space inertia, rotated vertices, contact-point velocity, angular impulse,
and two-axis Coulomb friction. It intentionally excludes box/box collision so
the new behavior remains small enough to compare instruction-for-instruction
with the CUDA kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .reference import STATE_WIDTH, _integrate_quaternion


CONTRACT_ID = "box3d.oriented-box-plane/v0"


@dataclass(frozen=True)
class OrientedBoxConfig:
    dt: float = 1.0 / 120.0
    substeps: int = 2
    gravity_y: float = -9.81
    restitution: float = 0.05
    friction: float = 0.6
    position_slop: float = 1.0e-4
    angular_damping: float = 0.02
    solver_iterations: int = 2

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.substeps <= 0 or self.solver_iterations <= 0:
            raise ValueError("dt, substeps and solver_iterations must be positive")
        if self.friction < 0.0 or not 0.0 <= self.restitution <= 1.0:
            raise ValueError("invalid material properties")


def _axis_angle(axis: tuple[float, float, float], angle: float) -> list[float]:
    length = math.sqrt(sum(value * value for value in axis))
    sine = math.sin(angle * 0.5) / length
    return [axis[0] * sine, axis[1] * sine, axis[2] * sine, math.cos(angle * 0.5)]


def make_oriented_box_state(
    worlds: int,
    bodies_per_world: int = 8,
    *,
    seed: int = 17,
) -> tuple[list[list[list[float]]], list[list[float]], list[list[list[float]]], list[list[list[float]]]]:
    if worlds <= 0 or bodies_per_world <= 0:
        raise ValueError("world and body counts must be positive")
    rng = random.Random(seed)
    states: list[list[list[float]]] = []
    inverse_mass: list[list[float]] = []
    half_extents: list[list[list[float]]] = []
    inverse_inertia: list[list[list[float]]] = []
    for world in range(worlds):
        world_states: list[list[float]] = []
        world_mass: list[float] = []
        world_half: list[list[float]] = []
        world_inertia: list[list[float]] = []
        for body in range(bodies_per_world):
            half = [
                0.035 + 0.004 * (body % 3),
                0.025 + 0.003 * ((body + 1) % 3),
                0.03 + 0.002 * ((body + 2) % 3),
            ]
            quaternion = _axis_angle(
                (0.7 + 0.1 * (body % 2), 0.35 + 0.07 * body, 1.0),
                0.32 + 0.11 * (body % 5),
            )
            x = (body % 4 - 1.5) * 0.19 + rng.uniform(-0.002, 0.002)
            z = (body // 4 - 0.5) * 0.20 + rng.uniform(-0.002, 0.002)
            y = 0.18 + 0.085 * body + 0.001 * world
            world_states.append(
                [x, y, z, *quaternion, 0.01 * ((body % 3) - 1), -0.05, 0.0, 0.0, 0.0, 0.0]
            )
            # Unit mass. For full side lengths 2h, Ixx = m/3 * (hy^2 + hz^2).
            world_inertia.append(
                [
                    3.0 / (half[1] ** 2 + half[2] ** 2),
                    3.0 / (half[0] ** 2 + half[2] ** 2),
                    3.0 / (half[0] ** 2 + half[1] ** 2),
                ]
            )
            world_mass.append(1.0)
            world_half.append(half)
        states.append(world_states)
        inverse_mass.append(world_mass)
        half_extents.append(world_half)
        inverse_inertia.append(world_inertia)
    return states, inverse_mass, half_extents, inverse_inertia


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(a[index] * b[index] for index in range(3))


def _rotate(q: list[float], vector: list[float]) -> list[float]:
    qv = q[:3]
    twice = [2.0 * value for value in _cross(qv, vector)]
    cross_again = _cross(qv, twice)
    return [
        vector[index] + q[3] * twice[index] + cross_again[index]
        for index in range(3)
    ]


def _inverse_rotate(q: list[float], vector: list[float]) -> list[float]:
    return _rotate([-q[0], -q[1], -q[2], q[3]], vector)


def _inverse_inertia_world(q: list[float], local_diagonal: list[float], vector: list[float]) -> list[float]:
    local = _inverse_rotate(q, vector)
    scaled = [local[index] * local_diagonal[index] for index in range(3)]
    return _rotate(q, scaled)


def _corners(q: list[float], half: list[float]) -> list[list[float]]:
    return [
        _rotate(q, [sx * half[0], sy * half[1], sz * half[2]])
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]


def minimum_corner_clearance(body: list[float], half: list[float]) -> float:
    return min(body[1] + corner[1] for corner in _corners(body[3:7], half))


def _apply_impulse(body: list[float], q: list[float], inv_mass: float, inv_inertia: list[float], r: list[float], impulse: list[float]) -> None:
    for axis in range(3):
        body[7 + axis] += impulse[axis] * inv_mass
    angular = _inverse_inertia_world(q, inv_inertia, _cross(r, impulse))
    for axis in range(3):
        body[10 + axis] += angular[axis]


def _effective_mass(q: list[float], inv_mass: float, inv_inertia: list[float], r: list[float], direction: list[float]) -> float:
    angular = _inverse_inertia_world(q, inv_inertia, _cross(r, direction))
    return inv_mass + _dot(_cross(angular, r), direction)


def _solve_plane(body: list[float], inv_mass: float, half: list[float], inv_inertia: list[float], config: OrientedBoxConfig) -> tuple[bool, float]:
    q = body[3:7]
    corners = _corners(q, half)
    minimum = min(body[1] + corner[1] for corner in corners)
    if minimum >= 0.0:
        return False, minimum
    body[1] += max(0.0, -minimum - config.position_slop)
    normal = [0.0, 1.0, 0.0]
    touched = False
    # Re-evaluate all initially penetrating vertices for a compact manifold.
    active = [corner for corner in corners if body[1] + corner[1] <= config.position_slop * 2.0]
    for _ in range(config.solver_iterations):
        for r in active:
            point_velocity = [
                body[7 + axis] + _cross(body[10:13], r)[axis]
                for axis in range(3)
            ]
            normal_speed = point_velocity[1]
            if normal_speed >= 0.0:
                continue
            denominator = _effective_mass(q, inv_mass, inv_inertia, r, normal)
            normal_impulse = -(1.0 + config.restitution) * normal_speed / denominator
            _apply_impulse(
                body, q, inv_mass, inv_inertia, r,
                [0.0, normal_impulse, 0.0],
            )
            touched = True
            point_velocity = [
                body[7 + axis] + _cross(body[10:13], r)[axis]
                for axis in range(3)
            ]
            tangent = [point_velocity[0], 0.0, point_velocity[2]]
            tangent_speed = math.hypot(tangent[0], tangent[2])
            if tangent_speed <= 1.0e-12:
                continue
            direction = [tangent[0] / tangent_speed, 0.0, tangent[2] / tangent_speed]
            tangent_mass = _effective_mass(q, inv_mass, inv_inertia, r, direction)
            tangent_impulse = min(tangent_speed / tangent_mass, config.friction * normal_impulse)
            _apply_impulse(
                body, q, inv_mass, inv_inertia, r,
                [-direction[0] * tangent_impulse, 0.0, -direction[2] * tangent_impulse],
            )
    return touched, minimum


def step_oriented_box_reference(
    state: list[list[list[float]]],
    inverse_mass: list[list[float]],
    half_extents: list[list[list[float]]],
    inverse_inertia: list[list[list[float]]],
    config: OrientedBoxConfig = OrientedBoxConfig(),
    *,
    steps: int = 1,
) -> tuple[list[list[list[float]]], list[list[bool]], float]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    output = [[body.copy() for body in world] for world in state]
    contacts = [[False for _ in world] for world in state]
    minimum_clearance = math.inf
    h = config.dt / config.substeps
    damping = max(0.0, 1.0 - config.angular_damping * h)
    for _ in range(steps):
        for _ in range(config.substeps):
            for world_index, world in enumerate(output):
                for body_index, body in enumerate(world):
                    inv_mass = inverse_mass[world_index][body_index]
                    if inv_mass == 0.0:
                        continue
                    body[8] += config.gravity_y * h
                    for axis in range(3):
                        body[axis] += body[7 + axis] * h
                        body[10 + axis] *= damping
                    _integrate_quaternion(body, h)
                    touched, clearance = _solve_plane(
                        body,
                        inv_mass,
                        half_extents[world_index][body_index],
                        inverse_inertia[world_index][body_index],
                        config,
                    )
                    contacts[world_index][body_index] |= touched
                    minimum_clearance = min(minimum_clearance, clearance)
    return output, contacts, minimum_clearance


def assert_valid_oriented_boxes(state: list[list[list[float]]], half_extents: list[list[list[float]]]) -> None:
    for world_index, world in enumerate(state):
        for body_index, body in enumerate(world):
            if len(body) != STATE_WIDTH or not all(math.isfinite(value) for value in body):
                raise AssertionError(f"malformed body {world_index}:{body_index}")
            norm = math.sqrt(sum(value * value for value in body[3:7]))
            if abs(norm - 1.0) > 2.0e-5:
                raise AssertionError(f"non-unit quaternion {world_index}:{body_index}")
            clearance = minimum_corner_clearance(body, half_extents[world_index][body_index])
            if clearance < -2.5e-3:
                raise AssertionError(f"plane penetration {world_index}:{body_index}: {clearance}")
