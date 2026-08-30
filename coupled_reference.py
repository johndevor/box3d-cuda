"""Stage-7 CPU oracle coupling joint rows and persistent OBB contacts.

Each physics substep integrates once, warm-starts contact manifolds once, then
solves joint rows and contact rows inside the same iteration loop. This is a
small maximal-coordinate proof contract, not a CUDA/performance, robot-arm,
broad-phase, CCD, or general contact-graph claim.

Solver order per iteration is fixed:

1. joint linear and constrained-angular rows;
2. joint velocity motor/damping and unilateral limit rows;
3. all contact-manifold normal rows;
4. all two-axis friction rows.

After velocity iterations, bounded contact split repair runs before bounded
joint repair. Cache layouts are explicit: Stage-5 joint rows are
``[W,J,8]`` (three linear, three angular, motor, unilateral limit), and
Stage-4 contacts are feature IDs ``[W,P,4]`` plus normal/tangent impulses
``[W,P,4,3]``. This is a horizontal push proof; the payload is never attached
or pose-copied.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import List, Sequence, Tuple

from .joint_reference import (
    PRISMATIC,
    REVOLUTE,
    JointConfig,
    JointTopology,
    _apply_cached_rows,
    _integrate as _integrate_joint_body,
    _joint_geometry,
    _repair_joint,
    _solve_actuator,
    _solve_angular_rows,
    _solve_limit,
    _solve_linear_rows,
    _tangent_basis,
    collision_filter_pairs,
)
from .manifold_reference import (
    MAX_MANIFOLD_POINTS,
    _repair_positions,
    _seed_from_cache,
    _solve_friction_point,
    _solve_normal_point,
    _warm_start,
    _write_cache,
    build_manifold,
    build_speculative_manifold,
    empty_manifold_cache,
)
from .sat_reference import RigidBox, SATConfig, sat_query


CONTRACT_ID = "box3d.coupled-articulation-contact/v1"
DEFAULT_SEED = 79
WORKLOAD_KIND = "horizontal_prismatic_push"
JOINT_CACHE_WIDTH = 8
PUSHER_BODY_INDEX = 1
PAYLOAD_BODY_INDEX = 2
CONTACT_PAIR_INDICES: Tuple[Tuple[int, int], ...] = ((PUSHER_BODY_INDEX, PAYLOAD_BODY_INDEX),)
MINIMUM_PUSH_DISPLACEMENT_M = 0.20
MINIMUM_RETRACTED_SEPARATION_M = 0.05
MAXIMUM_JOINT_LINEAR_ERROR_M = 0.002
MAXIMUM_JOINT_ANGULAR_ERROR_RAD = 0.01
MAXIMUM_CONTACT_PENETRATION_M = 0.005
POSITION_REPAIR_ITERATIONS = 8


@dataclass(frozen=True)
class CoupledConfig:
    joints: JointConfig = JointConfig(gravity_y=0.0)
    contacts: SATConfig = SATConfig(gravity_y=0.0, solver_iterations=12)

    def __post_init__(self) -> None:
        shared = (
            (self.joints.dt, self.contacts.dt, "dt"),
            (self.joints.substeps, self.contacts.substeps, "substeps"),
            (self.joints.gravity_y, self.contacts.gravity_y, "gravity"),
            (
                self.joints.solver_iterations,
                self.contacts.solver_iterations,
                "solver_iterations",
            ),
        )
        for joint_value, contact_value, name in shared:
            if joint_value != contact_value:
                raise ValueError("joint/contact {} must match in a coupled step".format(name))


@dataclass(frozen=True)
class CoupledStepResult:
    state: List[List[List[float]]]
    joint_coordinate: List[List[float]]
    joint_linear_error_m: List[List[float]]
    joint_angular_error_rad: List[List[float]]
    joint_limit_error: List[List[float]]
    motor_impulse: List[List[float]]
    joint_limit_active: List[List[bool]]
    pair_contacted: List[List[bool]]
    pair_contact_substeps: List[List[int]]
    pair_contact_count: List[List[int]]
    pair_penetration_m: List[List[float]]
    peak_pair_penetration_m: List[List[float]]
    pair_separation_m: List[List[float]]
    pair_normal_impulse: List[List[float]]
    peak_pair_normal_impulse: List[List[float]]
    joint_warm_start_cache: List[List[List[float]]]
    cache_feature_ids: List[List[List[int]]]
    cache_impulses: List[List[List[List[float]]]]


def _box_inverse_inertia(mass: float, half: Sequence[float]) -> List[float]:
    if mass <= 0.0:
        return [0.0, 0.0, 0.0]
    hx, hy, hz = half
    return [
        3.0 / (mass * (hy * hy + hz * hz)),
        3.0 / (mass * (hx * hx + hz * hz)),
        3.0 / (mass * (hx * hx + hy * hy)),
    ]


def make_coupled_push_state(
    worlds: int, *, seed: int = DEFAULT_SEED
):
    """Create a static base, driven prismatic pusher, and free payload box."""

    if isinstance(worlds, bool) or not isinstance(worlds, int) or worlds <= 0:
        raise ValueError("worlds must be a positive integer")
    rng = random.Random(seed)
    state, inverse_mass, half_extents, inverse_inertia = [], [], [], []
    pusher_half = [0.20, 0.20, 0.20]
    payload_half = [0.20, 0.20, 0.20]
    for _ in range(worlds):
        payload_jitter = rng.uniform(-2.0e-4, 2.0e-4)
        state.append(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.65 + payload_jitter, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        inverse_mass.append([0.0, 0.2, 1.0])
        half_extents.append([[0.1, 0.1, 0.1], pusher_half.copy(), payload_half.copy()])
        inverse_inertia.append(
            [
                [0.0, 0.0, 0.0],
                _box_inverse_inertia(5.0, pusher_half),
                _box_inverse_inertia(1.0, payload_half),
            ]
        )
    topology = JointTopology(
        ((0, PUSHER_BODY_INDEX),),
        (PRISMATIC,),
        ((0.0, 0.0, 0.0),),
        ((0.0, 0.0, 0.0),),
        ((1.0, 0.0, 0.0),),
        ((0.0, 0.0, 0.0, 1.0),),
        (0.0,),
        (1.0,),
        (0.1,),
        (True,),
        (False,),
    )
    joint_cache = [
        [[0.0] * JOINT_CACHE_WIDTH for _ in topology.joint_indices]
        for _ in range(worlds)
    ]
    cache_ids, cache_impulses = empty_manifold_cache(worlds, len(CONTACT_PAIR_INDICES))
    return (
        state,
        inverse_mass,
        half_extents,
        inverse_inertia,
        topology,
        CONTACT_PAIR_INDICES,
        joint_cache,
        cache_ids,
        cache_impulses,
    )


def _validate_layout(
    state,
    inverse_mass,
    half_extents,
    inverse_inertia,
    topology,
    contact_pairs,
    joint_cache,
    cache_ids,
    cache_impulses,
    motor_target_velocity,
    maximum_effort,
) -> None:
    worlds = len(state)
    if worlds <= 0 or not (
        len(inverse_mass) == len(half_extents) == len(inverse_inertia) == worlds
    ):
        raise ValueError("coupled body arrays must share a positive world count")
    bodies, joints, pairs = len(state[0]), len(topology.joint_indices), len(contact_pairs)
    if bodies < 3 or joints <= 0 or pairs <= 0:
        raise ValueError("coupled contract requires bodies, joints, and contact pairs")
    if len(joint_cache) != worlds:
        raise ValueError("coupled joint-cache world count mismatch")
    if len(cache_ids) != worlds or len(cache_impulses) != worlds:
        raise ValueError("coupled cache world count mismatch")
    if len(motor_target_velocity) != worlds or len(maximum_effort) != worlds:
        raise ValueError("coupled control world count mismatch")
    filtered = set(collision_filter_pairs(topology))
    for pair in contact_pairs:
        if len(pair) != 2 or pair[0] == pair[1] or min(pair) < 0 or max(pair) >= bodies:
            raise ValueError("invalid coupled contact pair")
        if (min(pair), max(pair)) in filtered:
            raise ValueError("contact pairs cannot include collision-filtered connected links")
    for world in range(worlds):
        if not (
            len(state[world])
            == len(inverse_mass[world])
            == len(half_extents[world])
            == len(inverse_inertia[world])
            == bodies
        ):
            raise ValueError("coupled body rows must be rectangular")
        if len(cache_ids[world]) != pairs or len(cache_impulses[world]) != pairs:
            raise ValueError("coupled cache pair count mismatch")
        if len(joint_cache[world]) != joints or any(
            len(row) != JOINT_CACHE_WIDTH
            or any(not math.isfinite(float(value)) for value in row)
            for row in joint_cache[world]
        ):
            raise ValueError("coupled joint cache must have shape [W,J,8]")
        if len(motor_target_velocity[world]) != joints or len(maximum_effort[world]) != joints:
            raise ValueError("coupled controls must have shape [W,J]")
        for body in range(bodies):
            values = state[world][body]
            if len(values) != 13 or not all(math.isfinite(float(value)) for value in values):
                raise ValueError("coupled body state must contain 13 finite values")
            if len(half_extents[world][body]) != 3 or any(
                not math.isfinite(float(value)) or value <= 0.0
                for value in half_extents[world][body]
            ):
                raise ValueError("coupled box half extents must be finite and positive")
            if len(inverse_inertia[world][body]) != 3:
                raise ValueError("coupled inverse inertia rows must contain three values")
        for pair in range(pairs):
            if len(cache_ids[world][pair]) != MAX_MANIFOLD_POINTS or len(
                cache_impulses[world][pair]
            ) != MAX_MANIFOLD_POINTS:
                raise ValueError("coupled contact cache must use four slots")
            if any(
                len(slot) != 3
                or any(not math.isfinite(float(value)) for value in slot)
                for slot in cache_impulses[world][pair]
            ):
                raise ValueError("coupled contact impulses must use [normal,t1,t2]")


def _joint_directions(joint_type: int, axis: Sequence[float]):
    cardinal = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    if joint_type == PRISMATIC:
        return _tangent_basis(axis), cardinal
    if joint_type == REVOLUTE:
        return cardinal, _tangent_basis(axis)
    return cardinal, cardinal


def _repair_contact_with_articulation_shock(
    left: RigidBox,
    right: RigidBox,
    left_index: int,
    right_index: int,
    articulated_bodies: set[int],
    manifold,
    config: SATConfig,
) -> None:
    """Position-only shock propagation; velocity impulses remain mass-correct."""

    left_articulated = left_index in articulated_bodies
    right_articulated = right_index in articulated_bodies
    # Match the CUDA shock-propagation rule: an articulated body remains
    # movable against fixed geometry, but is held while separating two dynamic
    # bodies so position repair cannot bypass the joint solver.
    left_weight = 0.0 if left_articulated and right.inverse_mass > 0.0 else left.inverse_mass
    right_weight = 0.0 if right_articulated and left.inverse_mass > 0.0 else right.inverse_mass
    total = left_weight + right_weight
    if total <= 0.0:
        return
    depth = max(point.depth for point in manifold.points)
    correction = min(0.2, max(0.0, depth - config.position_slop) * config.position_correction) / total
    for axis in range(3):
        left.position[axis] -= manifold.normal[axis] * correction * left_weight
        right.position[axis] += manifold.normal[axis] * correction * right_weight


def step_coupled_reference(
    state: Sequence[Sequence[Sequence[float]]],
    inverse_mass: Sequence[Sequence[float]],
    half_extents: Sequence[Sequence[Sequence[float]]],
    inverse_inertia: Sequence[Sequence[Sequence[float]]],
    topology: JointTopology,
    contact_pairs: Sequence[Sequence[int]],
    joint_warm_start_cache: Sequence[Sequence[Sequence[float]]],
    cache_feature_ids: Sequence[Sequence[Sequence[int]]],
    cache_impulses: Sequence[Sequence[Sequence[Sequence[float]]]],
    motor_target_velocity: Sequence[Sequence[float]],
    maximum_effort: Sequence[Sequence[float]],
    config: CoupledConfig = CoupledConfig(),
    *,
    steps: int = 1,
    warm_start: bool = True,
    constraint_first_integration: bool = False,
    contact_warm_start_factor: float = 1.0,
) -> CoupledStepResult:
    """Advance joints and contacts together without attachment or pose copying."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if not isinstance(constraint_first_integration, bool):
        raise TypeError("constraint_first_integration must be bool")
    if isinstance(contact_warm_start_factor, bool) or not isinstance(
        contact_warm_start_factor, (int, float)
    ):
        raise TypeError("contact_warm_start_factor must be a real number")
    contact_warm_start_factor = float(contact_warm_start_factor)
    if not math.isfinite(contact_warm_start_factor) or not (
        0.0 <= contact_warm_start_factor <= 1.0
    ):
        raise ValueError("contact_warm_start_factor must be finite and in [0,1]")
    _validate_layout(
        state,
        inverse_mass,
        half_extents,
        inverse_inertia,
        topology,
        contact_pairs,
        joint_warm_start_cache,
        cache_feature_ids,
        cache_impulses,
        motor_target_velocity,
        maximum_effort,
    )
    worlds = [
        [
            RigidBox.from_state(
                body,
                inverse_mass[world][body_index],
                half_extents[world][body_index],
                inverse_inertia[world][body_index],
            )
            for body_index, body in enumerate(state[world])
        ]
        for world in range(len(state))
    ]
    ids = [[list(pair) for pair in world] for world in cache_feature_ids]
    impulses = [
        [[list(slot) for slot in pair] for pair in world] for world in cache_impulses
    ]
    pair_contacted = [[False] * len(contact_pairs) for _ in worlds]
    pair_contact_substeps = [[0] * len(contact_pairs) for _ in worlds]
    peak_penetration = [[0.0] * len(contact_pairs) for _ in worlds]
    peak_normal_impulse = [[0.0] * len(contact_pairs) for _ in worlds]
    limit_active = [[False] * len(topology.joint_indices) for _ in worlds]
    motor_total = [[0.0] * len(topology.joint_indices) for _ in worlds]
    joint_cache = [
        [[float(value) for value in row] for row in world]
        for world in joint_warm_start_cache
    ]
    h = config.joints.dt / config.joints.substeps
    articulated_bodies = {
        body_index
        for pair in topology.joint_indices
        for body_index in pair
    }

    for _ in range(steps):
        for _ in range(config.joints.substeps):
            for world in worlds:
                for body in world:
                    if constraint_first_integration:
                        if body.inverse_mass != 0.0:
                            body.linear_velocity[1] += config.joints.gravity_y * h
                    else:
                        _integrate_joint_body(body, h, config.joints.gravity_y)
            manifolds = []
            for world_index, world in enumerate(worlds):
                world_manifolds = []
                for pair_index, (a, b) in enumerate(contact_pairs):
                    true_manifold = build_manifold(
                        world[a], world[b], pair_index, config.contacts.sat_epsilon
                    )
                    manifold = build_speculative_manifold(
                        world[a],
                        world[b],
                        pair_index,
                        config.contacts.contact_generation_distance,
                        config.contacts.sat_epsilon,
                    )
                    if true_manifold is not None:
                        pair_contacted[world_index][pair_index] = True
                        pair_contact_substeps[world_index][pair_index] += 1
                        peak_penetration[world_index][pair_index] = max(
                            peak_penetration[world_index][pair_index],
                            max(point.depth for point in true_manifold.points),
                        )
                    if manifold is not None:
                        _seed_from_cache(
                            manifold,
                            ids[world_index][pair_index],
                            impulses[world_index][pair_index],
                        )
                        if warm_start and contact_warm_start_factor != 1.0:
                            for point in manifold.points:
                                point.normal_impulse *= contact_warm_start_factor
                                point.tangent_impulse_1 *= contact_warm_start_factor
                                point.tangent_impulse_2 *= contact_warm_start_factor
                    world_manifolds.append(manifold)
                manifolds.append(world_manifolds)

            row_lambda = [
                [
                    [
                        value * config.joints.warm_start_factor if warm_start else 0.0
                        for value in joint_cache[world][joint]
                    ]
                    for joint in range(len(topology.joint_indices))
                ]
                for world in range(len(worlds))
            ]
            # Warm start in the same order as the iterative rows: joints, then contacts.
            for world_index, world in enumerate(worlds):
                for joint, (parent_index, child_index) in enumerate(topology.joint_indices):
                    parent, child = world[parent_index], world[child_index]
                    parent_offset, child_offset, axis, coordinate, _, _ = _joint_geometry(
                        parent, child, topology, joint
                    )
                    linear_directions, angular_directions = _joint_directions(
                        topology.joint_types[joint], axis
                    )
                    if topology.lower_limit[joint] <= coordinate <= topology.upper_limit[joint]:
                        row_lambda[world_index][joint][7] = 0.0
                    _apply_cached_rows(
                        parent,
                        child,
                        topology,
                        joint,
                        parent_offset,
                        child_offset,
                        axis,
                        linear_directions,
                        angular_directions,
                        row_lambda[world_index][joint],
                    )
                if warm_start:
                    for pair_index, (a, b) in enumerate(contact_pairs):
                        manifold = manifolds[world_index][pair_index]
                        if manifold is not None:
                            _warm_start(world[a], world[b], manifold)

            for _ in range(config.joints.solver_iterations):
                for world_index, world in enumerate(worlds):
                    for joint, (parent_index, child_index) in enumerate(
                        topology.joint_indices
                    ):
                        parent, child = world[parent_index], world[child_index]
                        parent_offset, child_offset, axis, coordinate, _, _ = _joint_geometry(
                            parent, child, topology, joint
                        )
                        linear_directions, angular_directions = _joint_directions(
                            topology.joint_types[joint], axis
                        )
                        linear_accumulated = row_lambda[world_index][joint][
                            0 : len(linear_directions)
                        ]
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
                        angular_accumulated = row_lambda[world_index][joint][
                            3 : 3 + len(angular_directions)
                        ]
                        _solve_angular_rows(
                            parent, child, angular_directions, angular_accumulated
                        )
                        for row_index, value in enumerate(angular_accumulated):
                            row_lambda[world_index][joint][3 + row_index] = value
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
                            row_lambda[world_index][joint][6],
                            h,
                            coordinate,
                            0.0,
                            0.0,
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
                for world_index, world in enumerate(worlds):
                    for pair_index, (a, b) in enumerate(contact_pairs):
                        manifold = manifolds[world_index][pair_index]
                        if manifold is None:
                            continue
                        for point in manifold.points:
                            _solve_normal_point(
                                world[a], world[b], manifold, point, config.contacts, h
                            )
                        for point in manifold.points:
                            _solve_friction_point(
                                world[a], world[b], manifold, point, config.contacts
                            )
                        if any(point.depth >= 0.0 for point in manifold.points):
                            peak_normal_impulse[world_index][pair_index] = max(
                                peak_normal_impulse[world_index][pair_index],
                                sum(point.normal_impulse for point in manifold.points),
                            )
            if constraint_first_integration:
                for world in worlds:
                    for body in world:
                        _integrate_joint_body(body, h, 0.0)
            for world_index in range(len(worlds)):
                for joint in range(len(topology.joint_indices)):
                    motor_total[world_index][joint] += row_lambda[world_index][joint][6]
            joint_cache = row_lambda
            for world_index, world in enumerate(worlds):
                for pair_index, (a, b) in enumerate(contact_pairs):
                    manifold = manifolds[world_index][pair_index]
                    ids[world_index][pair_index], impulses[world_index][pair_index] = _write_cache(
                        manifold
                    )
                for _repair_iteration in range(POSITION_REPAIR_ITERATIONS):
                    for pair_index in range(len(contact_pairs) - 1, -1, -1):
                        a, b = contact_pairs[pair_index]
                        repair_manifold = build_manifold(
                            world[a], world[b], pair_index, config.contacts.sat_epsilon
                        )
                        if repair_manifold is not None:
                            _repair_contact_with_articulation_shock(
                                world[a], world[b], a, b, articulated_bodies,
                                repair_manifold, config.contacts,
                            )
                    for joint, (parent_index, child_index) in enumerate(topology.joint_indices):
                        _repair_joint(
                            world[parent_index],
                            world[child_index],
                            topology,
                            joint,
                            config.joints,
                        )

    pair_count = [[0] * len(contact_pairs) for _ in worlds]
    penetration = [[0.0] * len(contact_pairs) for _ in worlds]
    separation = [[0.0] * len(contact_pairs) for _ in worlds]
    normal_impulse = [[0.0] * len(contact_pairs) for _ in worlds]
    for world_index, world in enumerate(worlds):
        for pair_index, (a, b) in enumerate(contact_pairs):
            final = build_manifold(world[a], world[b], pair_index, config.contacts.sat_epsilon)
            final_cache_manifold = (
                final
                if final is not None
                else build_speculative_manifold(
                    world[a],
                    world[b],
                    pair_index,
                    config.contacts.contact_generation_distance,
                    config.contacts.sat_epsilon,
                )
            )
            query = sat_query(world[a], world[b], config.contacts.sat_epsilon)
            separation[world_index][pair_index] = query.separation
            if final is None:
                if final_cache_manifold is not None:
                    _seed_from_cache(
                        final_cache_manifold,
                        ids[world_index][pair_index],
                        impulses[world_index][pair_index],
                    )
                ids[world_index][pair_index], impulses[world_index][pair_index] = _write_cache(
                    final_cache_manifold
                )
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
            pair_count[world_index][pair_index] = len(final.points)
            penetration[world_index][pair_index] = max(point.depth for point in final.points)
            normal_impulse[world_index][pair_index] = sum(
                point.normal_impulse for point in final.points
            )

    joint_coordinate = [[0.0] * len(topology.joint_indices) for _ in worlds]
    joint_linear = [[0.0] * len(topology.joint_indices) for _ in worlds]
    joint_angular = [[0.0] * len(topology.joint_indices) for _ in worlds]
    joint_limit_error = [[0.0] * len(topology.joint_indices) for _ in worlds]
    for world_index, world in enumerate(worlds):
        for joint, (parent_index, child_index) in enumerate(topology.joint_indices):
            _, _, _, coordinate, linear, angular = _joint_geometry(
                world[parent_index], world[child_index], topology, joint
            )
            joint_coordinate[world_index][joint] = coordinate
            joint_linear[world_index][joint] = math.sqrt(sum(value * value for value in linear))
            joint_angular[world_index][joint] = math.sqrt(sum(value * value for value in angular))
            joint_limit_error[world_index][joint] = max(
                0.0,
                topology.lower_limit[joint] - coordinate,
                coordinate - topology.upper_limit[joint],
            )
    output = [[body.to_state() for body in world] for world in worlds]
    return CoupledStepResult(
        output,
        joint_coordinate,
        joint_linear,
        joint_angular,
        joint_limit_error,
        motor_total,
        limit_active,
        pair_contacted,
        pair_contact_substeps,
        pair_count,
        penetration,
        peak_penetration,
        separation,
        normal_impulse,
        peak_normal_impulse,
        joint_cache,
        ids,
        impulses,
    )


def assert_valid_coupled_result(result: CoupledStepResult) -> None:
    for world_index, world in enumerate(result.state):
        for body_index, body in enumerate(world):
            if len(body) != 13 or not all(math.isfinite(float(value)) for value in body):
                raise AssertionError("malformed coupled body {}:{}".format(world_index, body_index))
            quaternion_norm = math.sqrt(sum(value * value for value in body[3:7]))
            if abs(quaternion_norm - 1.0) > 2.0e-5:
                raise AssertionError("non-unit coupled quaternion")
        if max(result.joint_linear_error_m[world_index]) > MAXIMUM_JOINT_LINEAR_ERROR_M:
            raise AssertionError("coupled joint linear error exceeded")
        if max(result.joint_angular_error_rad[world_index]) > MAXIMUM_JOINT_ANGULAR_ERROR_RAD:
            raise AssertionError("coupled joint angular error exceeded")
        if max(result.pair_penetration_m[world_index]) > MAXIMUM_CONTACT_PENETRATION_M:
            raise AssertionError("coupled contact penetration exceeded")


__all__ = [
    "CONTRACT_ID",
    "DEFAULT_SEED",
    "WORKLOAD_KIND",
    "JOINT_CACHE_WIDTH",
    "PUSHER_BODY_INDEX",
    "PAYLOAD_BODY_INDEX",
    "CONTACT_PAIR_INDICES",
    "MINIMUM_PUSH_DISPLACEMENT_M",
    "MINIMUM_RETRACTED_SEPARATION_M",
    "MAXIMUM_JOINT_LINEAR_ERROR_M",
    "MAXIMUM_JOINT_ANGULAR_ERROR_RAD",
    "MAXIMUM_CONTACT_PENETRATION_M",
    "POSITION_REPAIR_ITERATIONS",
    "CoupledConfig",
    "CoupledStepResult",
    "make_coupled_push_state",
    "step_coupled_reference",
    "assert_valid_coupled_result",
]
