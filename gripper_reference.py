"""CPU oracle and shared motion contract for the first manipulation port.

The workload is deliberately small but physically meaningful: two kinematic
box fingers close on a dynamic box, lift it using contact friction, open, and
then let gravity return the box to the floor.  There is no grasp flag, weld,
constraint, or pose-copy path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


CONTRACT_ID = "parallel-jaw-box-lift/v0"
STATE_WIDTH = 13


@dataclass(frozen=True)
class GripperWorldConfig:
    dt: float = 1.0 / 120.0
    substeps: int = 2
    gravity_y: float = -9.81
    restitution: float = 0.0
    friction: float = 1.0
    position_slop: float = 1.0e-4
    position_correction: float = 0.15
    cube_half_extent: tuple[float, float, float] = (0.03, 0.03, 0.03)
    finger_half_extent: tuple[float, float, float] = (0.012, 0.055, 0.04)
    settle_steps: int = 20
    close_steps: int = 64
    lift_steps: int = 120
    open_steps: int = 40
    fall_steps: int = 100
    close_speed: float = 0.10
    squeeze_speed: float = 0.004
    lift_speed: float = 0.12
    open_speed: float = 0.16

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.substeps <= 0:
            raise ValueError("dt and substeps must be positive")
        if self.friction < 0.0:
            raise ValueError("friction cannot be negative")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must be in [0, 1]")
        if min(self.phase_steps) <= 0:
            raise ValueError("all scripted phases must contain steps")

    @property
    def phase_steps(self) -> tuple[int, ...]:
        return (
            self.settle_steps,
            self.close_steps,
            self.lift_steps,
            self.open_steps,
            self.fall_steps,
        )

    @property
    def total_steps(self) -> int:
        return sum(self.phase_steps)

    @property
    def release_step(self) -> int:
        # Release starts when the fingers begin moving apart.
        return self.settle_steps + self.close_steps + self.lift_steps


def make_gripper_state(
    worlds: int,
    config: GripperWorldConfig = GripperWorldConfig(),
) -> tuple[list[list[float]], list[list[list[float]]]]:
    if worlds <= 0:
        raise ValueError("worlds must be positive")
    cube_y = config.cube_half_extent[1]
    finger_y = cube_y + 0.035
    cube = [
        [0.0, cube_y, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for _ in range(worlds)
    ]
    fingers = [[[-0.09, finger_y, 0.0], [0.09, finger_y, 0.0]] for _ in range(worlds)]
    return cube, fingers


def finger_velocity(step: int, config: GripperWorldConfig = GripperWorldConfig()) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return left/right kinematic velocities for one control step."""
    if not 0 <= step < config.total_steps:
        raise ValueError(f"step must be in [0, {config.total_steps})")
    settle_end = config.settle_steps
    close_end = settle_end + config.close_steps
    lift_end = close_end + config.lift_steps
    open_end = lift_end + config.open_steps
    if step < settle_end:
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    if step < close_end:
        return ((config.close_speed, 0.0, 0.0), (-config.close_speed, 0.0, 0.0))
    if step < lift_end:
        return (
            (config.squeeze_speed, config.lift_speed, 0.0),
            (-config.squeeze_speed, config.lift_speed, 0.0),
        )
    if step < open_end:
        return ((-config.open_speed, 0.0, 0.0), (config.open_speed, 0.0, 0.0))
    return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def _solve_floor(body: list[float], config: GripperWorldConfig) -> None:
    penetration = config.cube_half_extent[1] - body[1]
    if penetration <= 0.0:
        return
    body[1] += max(0.0, penetration - config.position_slop)
    if body[8] >= 0.0:
        return
    normal_delta = -(1.0 + config.restitution) * body[8]
    body[8] += normal_delta
    tangent_speed = math.hypot(body[7], body[9])
    if tangent_speed > 0.0:
        reduction = min(tangent_speed, config.friction * normal_delta)
        scale = (tangent_speed - reduction) / tangent_speed
        body[7] *= scale
        body[9] *= scale


def _finger_contact(
    body: list[float],
    finger_position: list[float],
    config: GripperWorldConfig,
) -> tuple[list[float], float] | None:
    delta = [body[axis] - finger_position[axis] for axis in range(3)]
    overlap = [
        config.cube_half_extent[axis] + config.finger_half_extent[axis] - abs(delta[axis])
        for axis in range(3)
    ]
    if min(overlap) <= 0.0:
        return None
    axis = min(range(3), key=overlap.__getitem__)
    normal = [0.0, 0.0, 0.0]
    normal[axis] = 1.0 if delta[axis] >= 0.0 else -1.0
    return normal, overlap[axis]


def _solve_fingers(
    body: list[float],
    finger_world: list[list[float]],
    velocities: tuple[tuple[float, float, float], tuple[float, float, float]],
    config: GripperWorldConfig,
    h: float,
) -> list[bool]:
    manifold = [
        _finger_contact(body, finger_world[index], config)
        for index in range(2)
    ]
    active = [contact is not None for contact in manifold]
    if not any(active):
        return active
    total_normal_impulse = 0.0
    position_delta = [0.0, 0.0, 0.0]
    velocity_delta = [0.0, 0.0, 0.0]
    contact_velocities: list[tuple[float, float, float]] = []
    # Evaluate every point from the same incoming velocity. That is equivalent
    # to a tiny contact manifold and avoids left/right ordering artifacts.
    incoming_velocity = body[7:10]
    for finger_index, contact in enumerate(manifold):
        if contact is None:
            continue
        normal, penetration = contact
        finger_v = velocities[finger_index]
        contact_velocities.append(finger_v)
        normal_speed = sum(
            (incoming_velocity[axis] - finger_v[axis]) * normal[axis]
            for axis in range(3)
        )
        bias_speed = config.position_correction * max(
            0.0, penetration - config.position_slop
        ) / h
        normal_delta = max(0.0, -normal_speed + bias_speed)
        total_normal_impulse += normal_delta
        correction = config.position_correction * max(
            0.0, penetration - config.position_slop
        )
        for axis in range(3):
            velocity_delta[axis] += normal[axis] * normal_delta
            position_delta[axis] += normal[axis] * correction
    for axis in range(3):
        body[7 + axis] += velocity_delta[axis]
        body[axis] += position_delta[axis]
    # Coulomb friction for the complete manifold. It moves toward the average
    # surface velocity and is bounded by mu times the summed normal impulse.
    target = [
        sum(velocity[axis] for velocity in contact_velocities) / len(contact_velocities)
        for axis in range(3)
    ]
    tangent = [body[7 + axis] - target[axis] for axis in range(3)]
    for contact in manifold:
        if contact is None:
            continue
        normal, _ = contact
        normal_component = sum(tangent[axis] * normal[axis] for axis in range(3))
        for axis in range(3):
            tangent[axis] -= normal[axis] * normal_component
    tangent_speed = math.sqrt(sum(value * value for value in tangent))
    if tangent_speed > 0.0:
        friction_delta = min(tangent_speed, config.friction * total_normal_impulse)
        for axis in range(3):
            body[7 + axis] -= tangent[axis] * friction_delta / tangent_speed
    return active


def step_gripper_reference(
    cube_state: list[list[float]],
    finger_positions: list[list[list[float]]],
    velocities: tuple[tuple[float, float, float], tuple[float, float, float]],
    config: GripperWorldConfig = GripperWorldConfig(),
) -> tuple[list[list[float]], list[list[list[float]]], list[list[bool]]]:
    if len(cube_state) != len(finger_positions):
        raise ValueError("cube and finger world counts must match")
    cubes = [body.copy() for body in cube_state]
    fingers = [[finger.copy() for finger in world] for world in finger_positions]
    contacts = [[False, False] for _ in cubes]
    h = config.dt / config.substeps
    for _ in range(config.substeps):
        for world_index, body in enumerate(cubes):
            for finger_index in range(2):
                for axis in range(3):
                    fingers[world_index][finger_index][axis] += velocities[finger_index][axis] * h
            body[8] += config.gravity_y * h
            body[0] += body[7] * h
            body[1] += body[8] * h
            body[2] += body[9] * h
            _solve_floor(body, config)
            active = _solve_fingers(body, fingers[world_index], velocities, config, h)
            for finger_index in range(2):
                contacts[world_index][finger_index] |= active[finger_index]
            _solve_floor(body, config)
    return cubes, fingers, contacts


def run_gripper_reference(
    worlds: int = 1,
    config: GripperWorldConfig = GripperWorldConfig(),
) -> dict[str, object]:
    cube, fingers = make_gripper_state(worlds, config)
    touched = [False] * worlds
    bilateral = [False] * worlds
    maximum_height = [body[1] for body in cube]
    height_at_release = [body[1] for body in cube]
    for step in range(config.total_steps):
        cube, fingers, contacts = step_gripper_reference(
            cube, fingers, finger_velocity(step, config), config
        )
        for world in range(worlds):
            touched[world] |= contacts[world][0] or contacts[world][1]
            bilateral[world] |= contacts[world][0] and contacts[world][1]
            maximum_height[world] = max(maximum_height[world], cube[world][1])
            if step == config.release_step - 1:
                height_at_release[world] = cube[world][1]
    final_height = [body[1] for body in cube]
    lifted = [height > config.cube_half_extent[1] + 0.06 for height in maximum_height]
    fell_after_release = [
        final_height[world] < height_at_release[world] - 0.04
        for world in range(worlds)
    ]
    finite = all(math.isfinite(value) for body in cube for value in body)
    passed = finite and all(touched) and all(bilateral) and all(lifted) and all(fell_after_release)
    return {
        "passed": passed,
        "finite": finite,
        "touched": all(touched),
        "bilateral_contact": all(bilateral),
        "lifted": all(lifted),
        "fell_after_release": all(fell_after_release),
        "minimum_maximum_height_m": min(maximum_height),
        "maximum_final_height_m": max(final_height),
        "minimum_release_height_m": min(height_at_release),
        "cube_state": cube,
        "finger_positions": fingers,
    }
