from __future__ import annotations

import copy
import math
import unittest

from box3d_cuda.manifold_reference import (
    BENCHMARK_STEPS,
    CONTRACT_ID,
    DEFAULT_SEED,
    INITIAL_SLIDER_SPEED_MPS,
    MAX_FINAL_PENETRATION_M,
    MAX_MANIFOLD_POINTS,
    MAX_SLIDER_FINAL_SPEED_RATIO,
    MAX_SLIDER_SPEED_INCREASE_MPS,
    MAX_STACK_HEIGHT_ERROR_M,
    MAX_TAIL_ANGULAR_SPEED_RAD_S,
    MAX_TAIL_LINEAR_SPEED_MPS,
    MAX_TAIL_POSITION_JITTER_M,
    MIN_PERSISTENT_CONTACT_FRAMES,
    MIN_STACK_CENTER_GAP_M,
    SLIDER_PAIR_INDEX,
    STACK_PAIR_COUNT,
    STACK_PAIR_INDICES,
    TAIL_WINDOW_STEPS,
    assert_valid_manifold_state,
    build_manifold,
    empty_manifold_cache,
    make_manifold_stack_state,
    step_manifold_reference,
)
from box3d_cuda.sat_reference import RigidBox, SATConfig, make_sat_box_state, sat_query


IDENTITY = [0.0, 0.0, 0.0, 1.0]


def _box(
    position,
    *,
    half=(0.5, 0.5, 0.5),
    velocity=(0.0, 0.0, 0.0),
    angular=(0.0, 0.0, 0.0),
    inverse_mass=1.0,
):
    return RigidBox(
        list(position),
        IDENTITY.copy(),
        list(velocity),
        list(angular),
        list(half),
        inverse_mass,
        [6.0, 6.0, 6.0] if inverse_mass else [0.0, 0.0, 0.0],
    )


def _speed(values):
    return math.sqrt(sum(value * value for value in values))


def _kinetic_energy(state):
    # Benchmark dynamic boxes are unit mass. This deliberately excludes
    # rotational energy: it is a conservative lower bound for energy growth.
    return sum(0.5 * sum(value * value for value in body[7:10]) for body in state[0][1:])


def _angular_momentum(state, inverse_mass, inverse_inertia_scalar=6.0):
    total = [0.0, 0.0, 0.0]
    for body, body_inverse_mass in zip(state[0], inverse_mass[0]):
        mass = 1.0 / body_inverse_mass
        momentum = [mass * value for value in body[7:10]]
        position = body[:3]
        orbital = [
            position[1] * momentum[2] - position[2] * momentum[1],
            position[2] * momentum[0] - position[0] * momentum[2],
            position[0] * momentum[1] - position[1] * momentum[0],
        ]
        for axis in range(3):
            total[axis] += orbital[axis] + body[10 + axis] / inverse_inertia_scalar
    return total


class ManifoldGeometryTests(unittest.TestCase):
    def test_contract_layout_is_stable(self):
        self.assertEqual(CONTRACT_ID, "box3d.obb-manifold/v1")
        self.assertEqual(DEFAULT_SEED, 41)
        self.assertEqual(
            STACK_PAIR_INDICES,
            ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5)),
        )
        state, mass, half, inertia, pairs, ids, impulses = make_manifold_stack_state(2)
        self.assertEqual((len(state), len(state[0]), len(state[0][0])), (2, 6, 13))
        self.assertEqual(pairs, STACK_PAIR_INDICES)
        self.assertEqual((len(ids), len(ids[0]), len(ids[0][0])), (2, 5, 4))
        self.assertEqual(
            (len(impulses), len(impulses[0]), len(impulses[0][0]), len(impulses[0][0][0])),
            (2, 5, 4, 3),
        )
        self.assertEqual(len(mass[0]), len(half[0]))
        self.assertEqual(len(mass[0]), len(inertia[0]))

    def test_face_clipping_returns_four_stable_feature_points(self):
        a = _box((0.0, 0.0, 0.0), inverse_mass=0.0)
        b = _box((0.0, 0.9, 0.0))
        first = build_manifold(a, b, pair_index=7)
        second = build_manifold(a, b, pair_index=7)

        self.assertEqual(first.kind, "face")
        self.assertEqual(len(first.points), MAX_MANIFOLD_POINTS)
        self.assertEqual(
            [point.feature_id for point in first.points],
            [point.feature_id for point in second.points],
        )
        self.assertEqual(len({point.feature_id for point in first.points}), 4)
        self.assertTrue(all(0 < point.feature_id < 2**63 for point in first.points))
        self.assertTrue(all(abs(point.depth - 0.1) < 1.0e-12 for point in first.points))
        self.assertAlmostEqual(sum(x * x for x in first.tangent_1), 1.0, places=12)
        self.assertAlmostEqual(sum(x * x for x in first.tangent_2), 1.0, places=12)
        self.assertAlmostEqual(sum(x * y for x, y in zip(first.tangent_1, first.tangent_2)), 0.0, places=12)
        self.assertAlmostEqual(sum(x * y for x, y in zip(first.normal, first.tangent_1)), 0.0, places=12)

    def test_edge_axis_uses_deterministic_one_point_fallback(self):
        state, mass, half, inertia, pairs = make_sat_box_state(1)
        a_index, b_index = pairs[2]
        a = RigidBox.from_state(state[0][a_index], mass[0][a_index], half[0][a_index], inertia[0][a_index])
        b = RigidBox.from_state(state[0][b_index], mass[0][b_index], half[0][b_index], inertia[0][b_index])
        separated = sat_query(a, b)
        b.position = [
            b.position[index]
            - separated.separating_axis[index] * (separated.separation + 0.012)
            for index in range(3)
        ]
        first = build_manifold(a, b, pair_index=2)
        second = build_manifold(a, b, pair_index=2)
        self.assertEqual(first.kind, "edge")
        self.assertEqual(len(first.points), 1)
        self.assertEqual(first.points[0].feature_id, second.points[0].feature_id)
        self.assertAlmostEqual(first.points[0].depth, 0.012, delta=2.0e-6)


class PersistentSolverTests(unittest.TestCase):
    def test_cache_persists_and_invalidates_on_separation(self):
        state, mass, half, inertia, pairs, ids, impulses = make_manifold_stack_state(1)
        state, _, _, ids, impulses, counts = step_manifold_reference(
            state, mass, half, inertia, pairs, ids, impulses, steps=20
        )
        prior_ids = copy.deepcopy(ids)
        self.assertTrue(all(count > 0 for count in counts[0]))
        self.assertTrue(any(slot[0] > 0.0 for pair in impulses[0] for slot in pair))

        state, _, _, ids, impulses, counts = step_manifold_reference(
            state, mass, half, inertia, pairs, ids, impulses, steps=1
        )
        persistent_pairs = 0
        for before, after in zip(prior_ids[0], ids[0]):
            if (set(before) - {0}) & (set(after) - {0}):
                persistent_pairs += 1
        # A changing SAT axis is allowed to invalidate one topological feature;
        # the other patches must actually reuse their stable feature IDs.
        self.assertGreaterEqual(persistent_pairs, 4)

        state[0][5][1] = 2.0
        state, _, _, ids, impulses, counts = step_manifold_reference(
            state,
            mass,
            half,
            inertia,
            pairs,
            ids,
            impulses,
            SATConfig(gravity_y=0.0),
            steps=1,
        )
        self.assertEqual(counts[0][SLIDER_PAIR_INDEX], 0)
        self.assertEqual(ids[0][SLIDER_PAIR_INDEX], [0, 0, 0, 0])
        self.assertEqual(impulses[0][SLIDER_PAIR_INDEX], [[0.0, 0.0, 0.0]] * 4)

    def test_four_box_stack_settles_with_low_tail_jitter(self):
        state, mass, half, inertia, pairs, ids, impulses = make_manifold_stack_state(1)
        state, _, _, ids, impulses, _ = step_manifold_reference(
            state,
            mass,
            half,
            inertia,
            pairs,
            ids,
            impulses,
            steps=BENCHMARK_STEPS - TAIL_WINDOW_STEPS,
        )
        histories = [[] for _ in range(STACK_PAIR_COUNT)]
        peak_linear = 0.0
        peak_angular = 0.0
        persistent_frames = [0] * len(pairs)
        penetration = None
        counts = None
        for _ in range(TAIL_WINDOW_STEPS):
            state, _, penetration, ids, impulses, counts = step_manifold_reference(
                state, mass, half, inertia, pairs, ids, impulses, steps=1
            )
            for index in range(STACK_PAIR_COUNT):
                body = state[0][index + 1]
                histories[index].append(body[1])
                peak_linear = max(peak_linear, _speed(body[7:10]))
                peak_angular = max(peak_angular, _speed(body[10:13]))
            for pair_index, count in enumerate(counts[0]):
                persistent_frames[pair_index] += int(count > 0)

        expected_heights = [0.25 + 0.5 * index for index in range(STACK_PAIR_COUNT)]
        height_error = max(abs(state[0][index + 1][1] - expected) for index, expected in enumerate(expected_heights))
        gaps = [state[0][index + 2][1] - state[0][index + 1][1] for index in range(3)]
        jitter = max(max(history) - min(history) for history in histories)

        self.assertLessEqual(height_error, MAX_STACK_HEIGHT_ERROR_M)
        self.assertGreaterEqual(min(gaps), MIN_STACK_CENTER_GAP_M)
        self.assertLessEqual(max(penetration[0]), MAX_FINAL_PENETRATION_M)
        self.assertLessEqual(jitter, MAX_TAIL_POSITION_JITTER_M)
        self.assertLessEqual(peak_linear, MAX_TAIL_LINEAR_SPEED_MPS)
        self.assertLessEqual(peak_angular, MAX_TAIL_ANGULAR_SPEED_RAD_S)
        self.assertTrue(all(value >= MIN_PERSISTENT_CONTACT_FRAMES for value in persistent_frames))
        self.assertTrue(all(count >= 2 for count in counts[0]))
        assert_valid_manifold_state(state, penetration, max_penetration=MAX_FINAL_PENETRATION_M)

    def test_warm_start_converges_better_without_adding_energy(self):
        warm = make_manifold_stack_state(1)
        cold = copy.deepcopy(warm)
        warm_result = step_manifold_reference(*warm, steps=120, warm_start=True)
        cold_result = step_manifold_reference(*cold, steps=120, warm_start=False)
        warm_state, _, warm_penetration, _, _, warm_counts = warm_result
        cold_state, _, cold_penetration, _, _, _ = cold_result

        warm_error = max(abs(warm_state[0][index][1] - (0.25 + 0.5 * (index - 1))) for index in range(1, 5))
        cold_error = max(abs(cold_state[0][index][1] - (0.25 + 0.5 * (index - 1))) for index in range(1, 5))
        self.assertLess(warm_error, cold_error)
        self.assertLess(max(warm_penetration[0]), max(cold_penetration[0]))
        self.assertLess(_kinetic_energy(warm_state), _kinetic_energy(cold_state))
        self.assertTrue(all(count >= 2 for count in warm_counts[0]))

    def test_slider_friction_slowdown_is_monotonic(self):
        state, mass, half, inertia, pairs, ids, impulses = make_manifold_stack_state(1)
        speeds = [abs(state[0][5][7])]
        maximum_tangent_impulse = 0.0
        for _ in range(80):
            state, _, _, ids, impulses, _ = step_manifold_reference(
                state, mass, half, inertia, pairs, ids, impulses, steps=1
            )
            speeds.append(abs(state[0][5][7]))
            maximum_tangent_impulse = max(
                maximum_tangent_impulse,
                *(abs(value) for slot in impulses[0][SLIDER_PAIR_INDEX] for value in slot[1:]),
            )
        self.assertAlmostEqual(speeds[0], INITIAL_SLIDER_SPEED_MPS)
        self.assertTrue(all(after <= before + MAX_SLIDER_SPEED_INCREASE_MPS for before, after in zip(speeds, speeds[1:])))
        self.assertLessEqual(speeds[-1] / speeds[0], MAX_SLIDER_FINAL_SPEED_RATIO)
        self.assertGreater(maximum_tangent_impulse, 0.0)

        no_friction = make_manifold_stack_state(1)
        result = step_manifold_reference(
            *no_friction,
            SATConfig(friction=0.0),
            steps=40,
        )
        self.assertAlmostEqual(result[0][0][5][7], INITIAL_SLIDER_SPEED_MPS, places=10)

    def test_replay_is_bit_deterministic(self):
        first = step_manifold_reference(*make_manifold_stack_state(2), steps=60)
        second = step_manifold_reference(*make_manifold_stack_state(2), steps=60)
        self.assertEqual(first, second)

    def test_pair_impulses_conserve_momentum_and_quaternions_stay_finite(self):
        state = [[
            [-0.45, 0.0, 0.0, *IDENTITY, 1.0, 0.35, 0.0, 0.0, 0.0, 0.0],
            [0.45, 0.0, 0.0, *IDENTITY, -0.5, -0.15, 0.0, 0.0, 0.0, 0.0],
        ]]
        inverse_mass = [[0.5, 1.0 / 3.0]]
        half = [[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]
        inertia = [[[6.0, 6.0, 6.0], [6.0, 6.0, 6.0]]]
        pairs = ((0, 1),)
        ids, impulses = empty_manifold_cache(1, 1)
        before = [
            state[0][0][7 + axis] / inverse_mass[0][0]
            + state[0][1][7 + axis] / inverse_mass[0][1]
            for axis in range(3)
        ]
        angular_before = _angular_momentum(state, inverse_mass)
        state, _, penetration, _, _, _ = step_manifold_reference(
            state,
            inverse_mass,
            half,
            inertia,
            pairs,
            ids,
            impulses,
            SATConfig(gravity_y=0.0, position_correction=0.0, angular_damping=0.0),
            steps=1,
        )
        after = [
            state[0][0][7 + axis] / inverse_mass[0][0]
            + state[0][1][7 + axis] / inverse_mass[0][1]
            for axis in range(3)
        ]
        for initial, final in zip(before, after):
            self.assertAlmostEqual(initial, final, places=10)
        angular_after = _angular_momentum(state, inverse_mass)
        for initial, final in zip(angular_before, angular_after):
            self.assertAlmostEqual(initial, final, places=10)
        for body in state[0]:
            self.assertTrue(all(math.isfinite(value) for value in body))
            self.assertAlmostEqual(sum(value * value for value in body[3:7]), 1.0, places=10)
        self.assertTrue(all(value >= 0.0 for value in penetration[0]))

    def test_split_position_repair_is_bounded_for_deep_overlap(self):
        state = [[
            [0.0, 0.0, 0.0, *IDENTITY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, *IDENTITY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]]
        inverse_mass = [[1.0, 1.0]]
        half = [[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]
        inertia = [[[6.0, 6.0, 6.0], [6.0, 6.0, 6.0]]]
        ids, impulses = empty_manifold_cache(1, 1)
        result = step_manifold_reference(
            state,
            inverse_mass,
            half,
            inertia,
            ((0, 1),),
            ids,
            impulses,
            SATConfig(gravity_y=0.0, position_correction=1.0, solver_iterations=1),
            steps=1,
        )
        # Two substeps, each capped at 0.2 m total pair separation.
        self.assertLessEqual(abs(result[0][0][0][0]), 0.2000001)
        self.assertLessEqual(abs(result[0][0][1][0]), 0.2000001)


if __name__ == "__main__":
    unittest.main()
