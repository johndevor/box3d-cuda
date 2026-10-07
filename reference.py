"""Dependency-free oracle for the first CUDA port slice.

This is intentionally small. It validates the exact state layout and contact
semantics used by the GPU kernel without pretending to be full Box3D.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence


STATE_WIDTH = 13


@dataclass(frozen=True)
class SphereWorldConfig:
    dt: float = 1.0 / 120.0
    substeps: int = 2
    gravity_y: float = -9.81
    restitution: float = 0.1
    friction: float = 0.6
    position_slop: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.dt <= 0 or self.substeps <= 0:
            raise ValueError("dt and substeps must be positive")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must be in [0, 1]")
        if self.friction < 0.0:
            raise ValueError("friction cannot be negative")


def make_drop_state(
    worlds: int,
    bodies_per_world: int,
    *,
    seed: int = 7,
) -> tuple[list[list[list[float]]], list[list[float]], list[list[float]]]:
    """Create deterministic separated spheres above a y=0 plane."""
    if worlds <= 0 or bodies_per_world <= 0:
        raise ValueError("world and body counts must be positive")
    rng = random.Random(seed)
    state: list[list[list[float]]] = []
    inverse_mass: list[list[float]] = []
    radius: list[list[float]] = []
    for world in range(worlds):
        world_state: list[list[float]] = []
        world_mass: list[float] = []
        world_radius: list[float] = []
        for body in range(bodies_per_world):
            r = 0.04 + 0.002 * (body % 3)
            x = (body % 4 - 1.5) * 0.14 + rng.uniform(-0.002, 0.002)
            z = (body // 4 - 0.5) * 0.14 + rng.uniform(-0.002, 0.002)
            y = 0.25 + 0.11 * body + 0.003 * world
            # p xyz, q xyzw, v xyz, angular velocity xyz
            world_state.append([x, y, z, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            world_mass.append(1.0)
            world_radius.append(r)
        state.append(world_state)
        inverse_mass.append(world_mass)
        radius.append(world_radius)
    return state, inverse_mass, radius


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _integrate_quaternion(body: list[float], h: float) -> None:
    # CUDA spelling of Box3D b3IntegrateRotation: q2 = normalize(q + .5*w*q).
    qx, qy, qz, qw = body[3:7]
    wx, wy, wz = body[10:13]
    dx, dy, dz = wx * h, wy * h, wz * h
    rx = 0.5 * (dx * qw + dy * qz - dz * qy)
    ry = 0.5 * (-dx * qz + dy * qw + dz * qx)
    rz = 0.5 * (dx * qy - dy * qx + dz * qw)
    rw = -0.5 * (dx * qx + dy * qy + dz * qz)
    q = (qx + rx, qy + ry, qz + rz, qw + rw)
    length = math.sqrt(sum(value * value for value in q))
    body[3:7] = [value / length for value in q]


def step_reference(
    state: list[list[list[float]]],
    inverse_mass: list[list[float]],
    radius: list[list[float]],
    config: SphereWorldConfig = SphereWorldConfig(),
    *,
    steps: int = 1,
) -> list[list[list[float]]]:
    """Semi-implicit integration plus physical sphere contacts.

    Contact support in this first slice is a static plane and sphere/sphere.
    It has no broad phase, joints, CCD, sleeping, or mesh collision yet.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    output = [[body.copy() for body in world] for world in state]
    h = config.dt / config.substeps
    for _ in range(steps):
        for _ in range(config.substeps):
            for world_index, world in enumerate(output):
                masses = inverse_mass[world_index]
                radii = radius[world_index]
                for index, body in enumerate(world):
                    if masses[index] == 0.0:
                        continue
                    body[8] += config.gravity_y * h
                    body[0] += body[7] * h
                    body[1] += body[8] * h
                    body[2] += body[9] * h
                    _integrate_quaternion(body, h)

                # Static y=0 plane. Correct position then apply normal/friction impulses.
                for index, body in enumerate(world):
                    if masses[index] == 0.0:
                        continue
                    penetration = radii[index] - body[1]
                    if penetration > 0.0:
                        body[1] += max(0.0, penetration - config.position_slop)
                        if body[8] < 0.0:
                            normal_delta = -(1.0 + config.restitution) * body[8]
                            body[8] += normal_delta
                            tangent_speed = math.hypot(body[7], body[9])
                            if tangent_speed > 0.0:
                                reduction = min(tangent_speed, config.friction * normal_delta)
                                scale = (tangent_speed - reduction) / tangent_speed
                                body[7] *= scale
                                body[9] *= scale

                # Fixed-small-world narrow phase: exact sphere pair tests, no fake grabs.
                for a in range(len(world)):
                    for b in range(a + 1, len(world)):
                        body_a, body_b = world[a], world[b]
                        dx = body_b[0] - body_a[0]
                        dy = body_b[1] - body_a[1]
                        dz = body_b[2] - body_a[2]
                        distance2 = dx * dx + dy * dy + dz * dz
                        target = radii[a] + radii[b]
                        if distance2 >= target * target or distance2 <= 1.0e-16:
                            continue
                        distance = math.sqrt(distance2)
                        normal = (dx / distance, dy / distance, dz / distance)
                        relative = (
                            body_b[7] - body_a[7],
                            body_b[8] - body_a[8],
                            body_b[9] - body_a[9],
                        )
                        inv_sum = masses[a] + masses[b]
                        if inv_sum == 0.0:
                            continue
                        correction = max(0.0, target - distance - config.position_slop) / inv_sum
                        for axis in range(3):
                            body_a[axis] -= normal[axis] * correction * masses[a]
                            body_b[axis] += normal[axis] * correction * masses[b]
                        normal_speed = _dot(relative, normal)
                        if normal_speed >= 0.0:
                            continue
                        impulse = -(1.0 + config.restitution) * normal_speed / inv_sum
                        for axis in range(3):
                            body_a[7 + axis] -= normal[axis] * impulse * masses[a]
                            body_b[7 + axis] += normal[axis] * impulse * masses[b]
    return output


def assert_valid_state(state: list[list[list[float]]], radius: list[list[float]]) -> None:
    for world_index, world in enumerate(state):
        for body_index, body in enumerate(world):
            if len(body) != STATE_WIDTH or not all(math.isfinite(value) for value in body):
                raise AssertionError(f"non-finite or malformed body {world_index}:{body_index}")
            norm = math.sqrt(sum(value * value for value in body[3:7]))
            if abs(norm - 1.0) > 2.0e-5:
                raise AssertionError(f"non-unit quaternion at {world_index}:{body_index}")
            if body[1] < radius[world_index][body_index] - 2.0e-4:
                raise AssertionError(f"ground penetration at {world_index}:{body_index}")
