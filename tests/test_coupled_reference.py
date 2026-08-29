"""Executable Stage-7 gates for coupled articulation/contact CPU semantics."""

import math

import pytest

from box3d_cuda.coupled_reference import (
    CONTACT_PAIR_INDICES,
    CONTRACT_ID,
    JOINT_CACHE_WIDTH,
    MAXIMUM_CONTACT_PENETRATION_M,
    MINIMUM_PUSH_DISPLACEMENT_M,
    MINIMUM_RETRACTED_SEPARATION_M,
    PAYLOAD_BODY_INDEX,
    PUSHER_BODY_INDEX,
    WORKLOAD_KIND,
    CoupledConfig,
    assert_valid_coupled_result,
    make_coupled_push_state,
    step_coupled_reference,
)
from box3d_cuda.joint_reference import JointConfig
from box3d_cuda.sat_reference import SATConfig


def _step(bundle, target, effort, steps, *, warm_start=True):
    state, inverse_mass, half, inertia, topology, pairs, joint_cache, ids, impulses = bundle
    result = step_coupled_reference(
        state,
        inverse_mass,
        half,
        inertia,
        topology,
        pairs,
        joint_cache,
        ids,
        impulses,
        target,
        effort,
        steps=steps,
        warm_start=warm_start,
    )
    next_bundle = (
        result.state,
        inverse_mass,
        half,
        inertia,
        topology,
        pairs,
        result.joint_warm_start_cache,
        result.cache_feature_ids,
        result.cache_impulses,
    )
    return result, next_bundle


def test_contract_layout_and_initial_separation():
    bundle = make_coupled_push_state(2)
    state, _, _, _, topology, pairs, joint_cache, ids, impulses = bundle
    assert CONTRACT_ID == "box3d.coupled-articulation-contact/v1"
    assert WORKLOAD_KIND == "horizontal_prismatic_push"
    assert pairs == CONTACT_PAIR_INDICES == ((1, 2),)
    assert topology.joint_indices == ((0, 1),)
    assert len(joint_cache) == 2 and len(joint_cache[0][0]) == JOINT_CACHE_WIDTH
    assert len(ids[0][0]) == 4 and len(impulses[0][0]) == 4
    assert all(len(slot) == 3 for slot in impulses[0][0])
    for world in state:
        assert world[PAYLOAD_BODY_INDEX][0] - world[PUSHER_BODY_INDEX][0] > 0.64


def test_driven_link_physically_pushes_then_releases_payload():
    bundle = make_coupled_push_state(1)
    initial_payload_x = bundle[0][0][PAYLOAD_BODY_INDEX][0]
    pushed, bundle = _step(bundle, [[1.0]], [[100.0]], 120)

    assert_valid_coupled_result(pushed)
    assert pushed.pair_contacted == [[True]]
    assert pushed.pair_contact_substeps[0][0] > 100
    assert pushed.pair_contact_count[0][0] == 4
    assert pushed.state[0][PAYLOAD_BODY_INDEX][0] - initial_payload_x >= MINIMUM_PUSH_DISPLACEMENT_M
    assert pushed.peak_pair_penetration_m[0][0] <= MAXIMUM_CONTACT_PENETRATION_M
    assert pushed.peak_pair_normal_impulse[0][0] > 0.0
    assert sum(feature != 0 for feature in pushed.cache_feature_ids[0][0]) == 4
    assert any(abs(value) > 0.0 for value in pushed.joint_warm_start_cache[0][0])

    payload_x_before_retract = pushed.state[0][PAYLOAD_BODY_INDEX][0]
    released, _ = _step(bundle, [[-1.0]], [[100.0]], 120)
    assert_valid_coupled_result(released)
    assert released.pair_contact_count == [[0]]
    assert released.pair_separation_m[0][0] >= MINIMUM_RETRACTED_SEPARATION_M
    assert released.state[0][PAYLOAD_BODY_INDEX][0] > payload_x_before_retract
    assert released.state[0][PUSHER_BODY_INDEX][0] < payload_x_before_retract
    assert released.cache_feature_ids[0][0] == [0, 0, 0, 0]
    assert released.cache_impulses[0][0] == [[0.0, 0.0, 0.0]] * 4


def test_zero_effort_counterfactual_cannot_move_payload():
    bundle = make_coupled_push_state(1)
    initial = bundle[0][0][PAYLOAD_BODY_INDEX]
    result, _ = _step(bundle, [[1.0]], [[0.0]], 120)
    assert result.pair_contacted == [[False]]
    assert result.pair_contact_substeps == [[0]]
    assert result.state[0][PAYLOAD_BODY_INDEX] == initial
    assert result.motor_impulse == [[0.0]]


def test_effort_bound_quaternions_and_joint_limits():
    steps, effort = 120, 25.0
    result, _ = _step(make_coupled_push_state(1), [[2.0]], [[effort]], steps)
    # Each step contains substeps whose impulses sum to no more than effort*dt.
    assert abs(result.motor_impulse[0][0]) <= effort * CoupledConfig().joints.dt * steps + 1e-10
    assert result.joint_limit_error[0][0] <= 1e-8
    assert 0.0 <= result.joint_coordinate[0][0] <= 1.0 + 1e-8
    for body in result.state[0]:
        assert all(math.isfinite(value) for value in body)
        assert math.isclose(sum(value * value for value in body[3:7]), 1.0, abs_tol=2e-5)


def test_deterministic_replay_and_batched_world_isolation():
    first, _ = _step(make_coupled_push_state(2), [[1.0], [0.0]], [[100.0], [0.0]], 80)
    replay, _ = _step(make_coupled_push_state(2), [[1.0], [0.0]], [[100.0], [0.0]], 80)
    solo, _ = _step(make_coupled_push_state(1), [[1.0]], [[100.0]], 80)
    assert first == replay
    assert first.state[0] == solo.state[0]
    assert first.cache_feature_ids[0] == solo.cache_feature_ids[0]
    assert first.state[1][PAYLOAD_BODY_INDEX][7:13] == [0.0] * 6


def test_warm_start_toggle_is_explicit_and_bounded():
    pushed, bundle = _step(make_coupled_push_state(1), [[1.0]], [[100.0]], 70)
    warm, _ = _step(bundle, [[1.0]], [[100.0]], 1, warm_start=True)
    cold, _ = _step(bundle, [[1.0]], [[100.0]], 1, warm_start=False)
    assert pushed.pair_contacted == [[True]]
    assert_valid_coupled_result(warm)
    assert_valid_coupled_result(cold)
    assert warm.peak_pair_penetration_m[0][0] <= MAXIMUM_CONTACT_PENETRATION_M
    assert cold.peak_pair_penetration_m[0][0] <= MAXIMUM_CONTACT_PENETRATION_M


def test_fails_closed_for_mismatched_solver_contract_and_filtered_pair():
    with pytest.raises(ValueError, match="must match"):
        CoupledConfig(contacts=SATConfig(dt=1.0 / 60.0, gravity_y=0.0))

    bundle = list(make_coupled_push_state(1))
    bundle[5] = ((0, 1),)
    with pytest.raises(ValueError, match="collision-filtered"):
        _step(tuple(bundle), [[1.0]], [[100.0]], 1)


def test_fails_closed_for_malformed_joint_or_contact_cache():
    bundle = list(make_coupled_push_state(1))
    bundle[6] = [[[0.0] * (JOINT_CACHE_WIDTH - 1)]]
    with pytest.raises(ValueError, match="joint cache"):
        _step(tuple(bundle), [[1.0]], [[100.0]], 1)

    bundle = list(make_coupled_push_state(1))
    bundle[8][0][0][0] = [0.0, 0.0]
    with pytest.raises(ValueError, match="contact impulses"):
        _step(tuple(bundle), [[1.0]], [[100.0]], 1)


def test_fails_closed_for_nonmatching_iteration_and_gravity_settings():
    joints = JointConfig(gravity_y=0.0, solver_iterations=8)
    with pytest.raises(ValueError, match="solver_iterations"):
        CoupledConfig(joints=joints, contacts=SATConfig(gravity_y=0.0))
    with pytest.raises(ValueError, match="gravity"):
        CoupledConfig(contacts=SATConfig(gravity_y=-9.81))
