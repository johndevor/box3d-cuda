from __future__ import annotations

import math
import unittest

from box3d_cuda.sat_reference import (
    CONTRACT_ID,
    PAIR_INDICES,
    RigidBox,
    SATConfig,
    SATContact,
    assert_valid_sat_boxes,
    make_sat_box_state,
    sat_query,
    solve_contact,
    step_sat_reference,
)


IDENTITY = [0.0, 0.0, 0.0, 1.0]


def axis_angle(axis, angle):
    norm = math.sqrt(sum(value * value for value in axis))
    sine = math.sin(angle / 2) / norm
    return [axis[0] * sine, axis[1] * sine, axis[2] * sine, math.cos(angle / 2)]


def box(
    position,
    half=(0.5, 0.5, 0.5),
    quaternion=IDENTITY,
    velocity=(0.0, 0.0, 0.0),
    angular=(0.0, 0.0, 0.0),
    inverse_mass=1.0,
    inverse_inertia=(6.0, 6.0, 6.0),
):
    return RigidBox(
        list(position),
        list(quaternion),
        list(velocity),
        list(angular),
        list(half),
        inverse_mass,
        list(inverse_inertia),
    )


def linear_momentum(body):
    if body.inverse_mass == 0:
        return [0.0, 0.0, 0.0]
    mass = 1.0 / body.inverse_mass
    return [mass * velocity for velocity in body.linear_velocity]


def angular_momentum_identity_orientation(body):
    """World-origin angular momentum for identity-oriented test bodies."""
    if body.inverse_mass == 0:
        return [0.0, 0.0, 0.0]
    momentum = linear_momentum(body)
    orbital = [
        body.position[1] * momentum[2] - body.position[2] * momentum[1],
        body.position[2] * momentum[0] - body.position[0] * momentum[2],
        body.position[0] * momentum[1] - body.position[1] * momentum[0],
    ]
    spin = [
        body.angular_velocity[index] / body.inverse_inertia_local[index]
        for index in range(3)
    ]
    return [orbital[index] + spin[index] for index in range(3)]


class SeparatingAxisTests(unittest.TestCase):
    def test_contract_id_and_pair_layout_are_stable(self):
        self.assertEqual(CONTRACT_ID, "box3d.obb-pair-collision/v1")
        self.assertEqual(PAIR_INDICES, ((0, 1), (2, 3), (4, 5)))

    def test_separated_boxes_report_positive_gap(self):
        result = sat_query(box((0, 0, 0)), box((1.25, 0, 0)))
        self.assertFalse(result.colliding)
        self.assertIsNone(result.contact)
        self.assertAlmostEqual(result.separation, 0.25, places=12)
        self.assertEqual(result.axes_considered, 15)
        self.assertEqual(result.axes_tested, 12)  # three parallel edge axes are degenerate

    def test_face_face_depth_point_and_normal(self):
        result = sat_query(box((0, 0, 0)), box((0.9, 0, 0)))
        self.assertTrue(result.colliding)
        contact = result.contact
        self.assertEqual(contact.axis_kind, "face_a")
        self.assertEqual(contact.normal, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(contact.depth, 0.1, places=12)
        self.assertAlmostEqual(contact.point[0], 0.45, places=12)

        reverse = sat_query(box((0.9, 0, 0)), box((0, 0, 0))).contact
        for forward, backward in zip(contact.normal, reverse.normal):
            self.assertAlmostEqual(forward, -backward, places=12)

    def test_rotated_face_contact_uses_rotated_normal(self):
        quaternion = axis_angle((0, 0, 1), math.radians(31))
        expected = [math.cos(math.radians(31)), math.sin(math.radians(31)), 0.0]
        a = box((0, 0, 0), half=(0.4, 0.2, 0.3), quaternion=quaternion)
        b = box([0.77 * value for value in expected], half=(0.4, 0.2, 0.3), quaternion=quaternion)
        contact = sat_query(a, b).contact
        self.assertEqual(contact.axis_kind, "face_a")
        self.assertAlmostEqual(contact.depth, 0.03, places=10)
        for actual, wanted in zip(contact.normal, expected):
            self.assertAlmostEqual(actual, wanted, places=10)

    def test_adversarial_edge_edge_axis_is_not_lost(self):
        state, mass, half, inertia, pairs = make_sat_box_state(1, seed=23)
        a_index, b_index = pairs[2]
        a = RigidBox.from_state(state[0][a_index], mass[0][a_index], half[0][a_index], inertia[0][a_index])
        b = RigidBox.from_state(state[0][b_index], mass[0][b_index], half[0][b_index], inertia[0][b_index])
        separated = sat_query(a, b)
        self.assertFalse(separated.colliding)
        self.assertAlmostEqual(separated.separation, 0.03, delta=0.001)
        b.position = [
            b.position[index]
            - separated.separating_axis[index] * (separated.separation + 0.012)
            for index in range(3)
        ]
        contact = sat_query(a, b).contact
        self.assertEqual(contact.axis_kind, "edge")
        self.assertGreaterEqual(contact.axis_index, 6)
        self.assertAlmostEqual(contact.depth, 0.012, delta=2.0e-6)
        self.assertEqual(contact.axes_tested, 15)

    def test_near_parallel_and_grazing_are_robust(self):
        epsilon = 1.0e-7
        almost_parallel = axis_angle((0, 1, 0), 1.0e-10)
        touching = sat_query(
            box((0, 0, 0)),
            box((1.0 + 0.5 * epsilon, 0, 0), quaternion=almost_parallel),
            epsilon,
        )
        self.assertTrue(touching.colliding)
        self.assertEqual(touching.contact.depth, 0.0)
        self.assertGreaterEqual(touching.axes_tested, 12)
        self.assertLessEqual(touching.axes_tested, 15)
        self.assertTrue(all(math.isfinite(value) for value in touching.contact.normal))

        separated = sat_query(box((0, 0, 0)), box((1.0 + 2.0 * epsilon, 0, 0)), epsilon)
        self.assertFalse(separated.colliding)
        self.assertGreater(separated.separation, epsilon)


class PhysicalResponseTests(unittest.TestCase):
    def test_impulse_conserves_linear_momentum_and_repairs_penetration(self):
        a = box((-0.45, 0, 0), velocity=(1.0, 0.35, 0), inverse_mass=0.5)
        b = box((0.45, 0, 0), velocity=(-0.5, -0.15, 0), inverse_mass=1.0 / 3.0)
        before = [left + right for left, right in zip(linear_momentum(a), linear_momentum(b))]
        contact = sat_query(a, b).contact
        relative_tangent_before = abs(b.linear_velocity[1] - a.linear_velocity[1])
        result = solve_contact(
            a,
            b,
            contact,
            SATConfig(
                gravity_y=0,
                restitution=0.2,
                friction=0.7,
                position_correction=1.0,
            ),
        )
        after = [left + right for left, right in zip(linear_momentum(a), linear_momentum(b))]
        for initial, final in zip(before, after):
            self.assertAlmostEqual(initial, final, places=11)
        self.assertGreater(result.normal_impulse, 0)
        self.assertLess(abs(b.linear_velocity[1] - a.linear_velocity[1]), relative_tangent_before)
        repaired = sat_query(a, b).contact
        self.assertLessEqual(repaired.depth, 1.01e-4)
        self.assertGreaterEqual(result.relative_normal_speed_after, -1.0e-10)

    def test_single_contact_impulses_conserve_total_angular_momentum(self):
        a = box(
            (-0.45, 0.10, 0), velocity=(1.0, 0.4, 0.2),
            inverse_mass=0.5, inverse_inertia=(1, 2, 3),
        )
        b = box(
            (0.45, -0.05, 0), velocity=(-0.5, -0.1, -0.3),
            inverse_mass=1.0 / 3.0, inverse_inertia=(2, 4, 6),
        )
        before = [
            left + right
            for left, right in zip(
                angular_momentum_identity_orientation(a),
                angular_momentum_identity_orientation(b),
            )
        ]
        solve_contact(
            a,
            b,
            sat_query(a, b).contact,
            SATConfig(
                gravity_y=0,
                restitution=0.2,
                friction=0.7,
                position_correction=0,
            ),
        )
        after = [
            left + right
            for left, right in zip(
                angular_momentum_identity_orientation(a),
                angular_momentum_identity_orientation(b),
            )
        ]
        for initial, final in zip(before, after):
            self.assertAlmostEqual(initial, final, places=11)

    def test_world_space_anisotropic_inertia_changes_angular_response(self):
        contact = SATContact(
            normal=(1.0, 0.0, 0.0),
            depth=0.01,
            point=(0.0, 0.4, 0.0),
            axis_index=0,
            axis_kind="face_a",
            axes_considered=15,
            axes_tested=15,
        )
        static = box((1, 0, 0), inverse_mass=0, inverse_inertia=(0, 0, 0))
        identity = box((0, 0, 0), velocity=(1, 0, 0), inverse_inertia=(1, 2, 12))
        rotated = box(
            (0, 0, 0),
            quaternion=axis_angle((1, 0, 0), math.pi / 2),
            velocity=(1, 0, 0),
            inverse_inertia=(1, 2, 12),
        )
        solve_contact(identity, static.copy(), contact, SATConfig(gravity_y=0, friction=0))
        solve_contact(rotated, static.copy(), contact, SATConfig(gravity_y=0, friction=0))
        self.assertNotAlmostEqual(
            math.sqrt(sum(value * value for value in identity.angular_velocity)),
            math.sqrt(sum(value * value for value in rotated.angular_velocity)),
            places=6,
        )

    def test_deterministic_contract_pairs_start_separated_then_physically_contact(self):
        state, mass, half, inertia, pairs = make_sat_box_state(4, seed=31)
        for world_index in range(4):
            for a_index, b_index in pairs:
                a = RigidBox.from_state(
                    state[world_index][a_index], mass[world_index][a_index],
                    half[world_index][a_index], inertia[world_index][a_index],
                )
                b = RigidBox.from_state(
                    state[world_index][b_index], mass[world_index][b_index],
                    half[world_index][b_index], inertia[world_index][b_index],
                )
                query = sat_query(a, b)
                self.assertFalse(query.colliding)
                self.assertGreater(query.separation, 0.025)

        # The matched CPU/CUDA contract uses SATConfig defaults, including
        # gravity. Both bodies in each free-space pair receive the same gravity,
        # so relative collision timing remains unchanged.
        config = SATConfig()
        first = step_sat_reference(state, mass, half, inertia, pairs, config, steps=60)
        second = step_sat_reference(state, mass, half, inertia, pairs, config, steps=60)
        self.assertEqual(first, second)
        output, contacts, penetration = first
        self.assertTrue(all(all(world) for world in contacts))
        self.assertLessEqual(max(max(world) for world in penetration), 0.005)
        self.assertTrue(
            all(
                sum(value * value for value in output[world][body][10:13]) > 1.0e-4
                for world in range(4)
                for body in (2, 3, 4, 5)
            )
        )
        assert_valid_sat_boxes(output, mass, half, inertia, pairs)

    def test_stacked_falling_boxes_remain_finite_and_within_penetration_bound(self):
        half = [[[2.0, 0.25, 2.0], [0.25, 0.25, 0.25], [0.25, 0.25, 0.25]]]
        state = [[
            [0, -0.25, 0, *IDENTITY, 0, 0, 0, 0, 0, 0],
            [0, 0.75, 0, *IDENTITY, 0, 0, 0, 0, 0, 0],
            [0, 1.45, 0, *IDENTITY, 0, 0, 0, 0, 0, 0],
        ]]
        inverse_mass = [[0.0, 1.0, 1.0]]
        inverse_inertia = [[[0, 0, 0], [24, 24, 24], [24, 24, 24]]]
        pairs = ((0, 1), (1, 2))
        output, contacts, penetration = step_sat_reference(
            state,
            inverse_mass,
            half,
            inverse_inertia,
            pairs,
            SATConfig(restitution=0, friction=0.7, solver_iterations=10),
            steps=480,
        )
        self.assertEqual(contacts, [[True, True]])
        self.assertLessEqual(max(penetration[0]), 0.005)
        self.assertAlmostEqual(output[0][1][1], 0.25, delta=0.002)
        self.assertAlmostEqual(output[0][2][1], 0.75, delta=0.003)
        self.assertLess(abs(output[0][2][8]), 0.01)
        assert_valid_sat_boxes(
            output, inverse_mass, half, inverse_inertia, pairs, max_penetration=0.005
        )

    def test_high_spin_quaternions_remain_finite_and_normalized(self):
        state, mass, half, inertia, pairs = make_sat_box_state(1)
        for index, body in enumerate(state[0]):
            body[10:13] = [7.0 + index, -5.0 + 0.2 * index, 4.0]
        output, _, _ = step_sat_reference(
            state,
            mass,
            half,
            inertia,
            pairs,
            SATConfig(gravity_y=0, angular_damping=0.1),
            steps=600,
        )
        for body in output[0]:
            self.assertTrue(all(math.isfinite(value) for value in body))
            norm = math.sqrt(sum(value * value for value in body[3:7]))
            self.assertAlmostEqual(norm, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
