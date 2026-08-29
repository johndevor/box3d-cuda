from __future__ import annotations

import copy
import math
import unittest

from box3d_cuda.joint_reference import (
    CONTRACT_ID,
    FIXED,
    JOINT_TYPE_NAMES,
    PRISMATIC,
    REVOLUTE,
    JointConfig,
    JointTopology,
    assert_valid_joint_result,
    collision_filter_pairs,
    step_joint_reference,
)


IDENTITY = (0.0, 0.0, 0.0, 1.0)


def body(position, quaternion=IDENTITY, velocity=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0)):
    return [*position, *quaternion, *velocity, *angular]


def one_joint_topology(
    joint_type,
    *,
    parent_anchor=(0.0, 0.0, 0.0),
    child_anchor=(0.0, 0.0, 0.0),
    axis=(0.0, 0.0, 1.0),
    limits=(-10.0, 10.0),
    damping=0.0,
    motor_enabled=False,
    collision_enabled=False,
):
    return JointTopology(
        ((0, 1),),
        (joint_type,),
        (parent_anchor,),
        (child_anchor,),
        (axis,),
        (IDENTITY,),
        (limits[0],),
        (limits[1],),
        (damping,),
        (motor_enabled,),
        (collision_enabled,),
    )


def static_root_layout(child_state):
    return (
        [[body((0.0, 0.0, 0.0)), child_state]],
        [[0.0, 1.0]],
        [[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]],
    )


class JointContractTests(unittest.TestCase):
    def test_contract_types_and_collision_filter_are_stable(self):
        topology = JointTopology(
            ((0, 1), (1, 2), (2, 3)),
            (FIXED, REVOLUTE, PRISMATIC),
            ((0.0, 0.0, 0.0),) * 3,
            ((0.0, 0.0, 0.0),) * 3,
            ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
            (IDENTITY,) * 3,
            (-1.0, -2.0, -0.5),
            (1.0, 2.0, 0.5),
            (0.0, 0.1, 0.2),
            (False, True, True),
            (False, True, False),
        )
        self.assertEqual(CONTRACT_ID, "box3d.articulated-joints/v1")
        self.assertEqual(JOINT_TYPE_NAMES, ("fixed", "revolute", "prismatic"))
        self.assertEqual(collision_filter_pairs(topology), ((0, 1), (2, 3)))

    def test_topology_validation_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "normalized"):
            one_joint_topology(REVOLUTE, axis=(0.0, 0.0, 2.0))
        with self.assertRaisesRegex(ValueError, "ordered"):
            one_joint_topology(PRISMATIC, limits=(1.0, -1.0))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            one_joint_topology(99)

    def test_batched_world_layout_and_replay_are_deterministic(self):
        child = body((0.0, -0.5, 0.0))
        state, inverse_mass, inverse_inertia = static_root_layout(child)
        state = [copy.deepcopy(state[0]), copy.deepcopy(state[0])]
        inverse_mass = [inverse_mass[0].copy(), inverse_mass[0].copy()]
        inverse_inertia = [copy.deepcopy(inverse_inertia[0]), copy.deepcopy(inverse_inertia[0])]
        topology = one_joint_topology(
            REVOLUTE, child_anchor=(0.0, 0.5, 0.0), motor_enabled=True
        )
        args = (state, inverse_mass, inverse_inertia, topology, [[0.7], [0.7]], [[5.0], [5.0]])
        first = step_joint_reference(*copy.deepcopy(args), JointConfig(gravity_y=0.0), steps=40)
        second = step_joint_reference(*copy.deepcopy(args), JointConfig(gravity_y=0.0), steps=40)
        self.assertEqual(first, second)
        self.assertEqual(first.state[0], first.state[1])
        self.assertEqual((len(first.state), len(first.state[0]), len(first.state[0][0])), (2, 2, 13))
        self.assertEqual((len(first.coordinate), len(first.coordinate[0])), (2, 1))


class JointPhysicsTests(unittest.TestCase):
    def test_fixed_joint_rejects_linear_and_angular_motion(self):
        state, inverse_mass, inverse_inertia = static_root_layout(
            body((1.0, 0.0, 0.0), velocity=(0.0, 1.0, 0.0), angular=(0.5, 0.2, 0.1))
        )
        result = step_joint_reference(
            state,
            inverse_mass,
            inverse_inertia,
            one_joint_topology(FIXED, parent_anchor=(1.0, 0.0, 0.0)),
            [[0.0]],
            [[0.0]],
            JointConfig(gravity_y=0.0),
            steps=240,
        )
        self.assertAlmostEqual(result.state[0][1][0], 1.0, places=8)
        self.assertLessEqual(result.linear_error_m[0][0], 1.1e-5)
        self.assertLessEqual(result.angular_error_rad[0][0], 1.1e-5)
        self.assertLess(math.sqrt(sum(value * value for value in result.state[0][1][7:13])), 1.0e-10)
        assert_valid_joint_result(result, maximum_error=2.0e-5)

    def test_revolute_velocity_motor_matches_single_dof_solution(self):
        state, inverse_mass, inverse_inertia = static_root_layout(body((0.0, -0.5, 0.0)))
        result = step_joint_reference(
            state,
            inverse_mass,
            inverse_inertia,
            one_joint_topology(
                REVOLUTE,
                child_anchor=(0.0, 0.5, 0.0),
                limits=(-2.0, 2.0),
                motor_enabled=True,
            ),
            [[1.0]],
            [[10.0]],
            JointConfig(gravity_y=0.0),
            steps=120,
        )
        self.assertAlmostEqual(result.state[0][1][12], 1.0, places=8)
        self.assertAlmostEqual(result.coordinate[0][0], 0.9604, delta=0.002)
        self.assertLess(result.linear_error_m[0][0], 2.0e-5)
        self.assertLess(result.angular_error_rad[0][0], 1.0e-10)
        self.assertFalse(result.limit_active[0][0])

    def test_actuator_impulse_respects_effort_times_timestep(self):
        state, inverse_mass, inverse_inertia = static_root_layout(body((0.0, 0.0, 0.0)))
        effort = 2.0
        config = JointConfig(gravity_y=0.0)
        result = step_joint_reference(
            state,
            inverse_mass,
            inverse_inertia,
            one_joint_topology(REVOLUTE, motor_enabled=True),
            [[100.0]],
            [[effort]],
            config,
            steps=1,
        )
        self.assertLessEqual(abs(result.motor_impulse[0][0]), effort * config.dt + 1.0e-12)
        self.assertAlmostEqual(abs(result.motor_impulse[0][0]), effort * config.dt, places=12)

    def test_position_motor_uses_bounded_pd_effort(self):
        state, inverse_mass, inverse_inertia = static_root_layout(body((0.0, 0.0, 0.0)))
        config = JointConfig(gravity_y=0.0)
        result = step_joint_reference(
            state,
            inverse_mass,
            inverse_inertia,
            one_joint_topology(REVOLUTE, damping=2.0, motor_enabled=True),
            [[0.0]],
            [[3.0]],
            config,
            steps=30,
            motor_target_position=[[0.5]],
            stiffness=[20.0],
        )
        self.assertGreater(result.coordinate[0][0], 0.0)
        self.assertLess(result.coordinate[0][0], 0.6)
        self.assertLessEqual(abs(result.motor_impulse[0][0]), 3.0 * config.dt * 30 + 1.0e-9)

    def test_passive_revolute_damping_removes_free_axis_speed(self):
        state, inverse_mass, inverse_inertia = static_root_layout(
            body((0.0, 0.0, 0.0), angular=(0.0, 0.0, 2.0))
        )
        undamped = step_joint_reference(
            copy.deepcopy(state),
            inverse_mass,
            inverse_inertia,
            one_joint_topology(REVOLUTE, damping=0.0),
            [[0.0]],
            [[10.0]],
            JointConfig(gravity_y=0.0),
            steps=60,
        )
        damped = step_joint_reference(
            copy.deepcopy(state),
            inverse_mass,
            inverse_inertia,
            one_joint_topology(REVOLUTE, damping=4.0),
            [[0.0]],
            [[10.0]],
            JointConfig(gravity_y=0.0),
            steps=60,
        )
        self.assertAlmostEqual(undamped.state[0][1][12], 2.0, places=10)
        self.assertLess(abs(damped.state[0][1][12]), 1.0e-6)
        self.assertLess(abs(damped.coordinate[0][0]), abs(undamped.coordinate[0][0]))

    def test_prismatic_motor_hits_limit_without_forbidden_motion(self):
        state, inverse_mass, inverse_inertia = static_root_layout(body((0.0, 0.0, 0.0)))
        result = step_joint_reference(
            state,
            inverse_mass,
            inverse_inertia,
            one_joint_topology(
                PRISMATIC,
                axis=(1.0, 0.0, 0.0),
                limits=(0.0, 0.25),
                motor_enabled=True,
            ),
            [[1.0]],
            [[20.0]],
            JointConfig(gravity_y=0.0),
            steps=120,
        )
        self.assertTrue(result.limit_active[0][0])
        self.assertAlmostEqual(result.coordinate[0][0], 0.25, delta=7.0e-5)
        self.assertLessEqual(result.limit_error[0][0], 7.0e-5)
        self.assertLess(result.linear_error_m[0][0], 1.0e-10)
        self.assertLess(result.angular_error_rad[0][0], 1.0e-10)
        self.assertAlmostEqual(result.state[0][1][1], 0.0, places=12)

    def test_revolute_motor_cannot_drive_through_angular_limit(self):
        state, inverse_mass, inverse_inertia = static_root_layout(body((0.0, -0.5, 0.0)))
        result = step_joint_reference(
            state,
            inverse_mass,
            inverse_inertia,
            one_joint_topology(
                REVOLUTE,
                child_anchor=(0.0, 0.5, 0.0),
                limits=(-0.3, 0.3),
                motor_enabled=True,
            ),
            [[2.0]],
            [[20.0]],
            JointConfig(gravity_y=0.0),
            steps=120,
        )
        self.assertTrue(result.limit_active[0][0])
        self.assertAlmostEqual(result.coordinate[0][0], 0.3, delta=1.0e-5)
        self.assertLessEqual(result.limit_error[0][0], 1.0e-5)
        self.assertLess(result.linear_error_m[0][0], 5.0e-5)

    def test_two_link_pendulum_keeps_both_hinges_connected(self):
        angle = 0.4
        quaternion = (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0))

        def rotate(vector):
            return (
                math.cos(angle) * vector[0] - math.sin(angle) * vector[1],
                math.sin(angle) * vector[0] + math.cos(angle) * vector[1],
                vector[2],
            )

        first_position = tuple(-value for value in rotate((0.0, 0.5, 0.0)))
        second_position = tuple(
            first_position[index] + rotate((0.0, -1.0, 0.0))[index]
            for index in range(3)
        )
        state = [[
            body((0.0, 0.0, 0.0)),
            body(first_position, quaternion),
            body(second_position, quaternion),
        ]]
        inverse_mass = [[0.0, 1.0, 1.0]]
        inverse_inertia = [[[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]]
        topology = JointTopology(
            ((0, 1), (1, 2)),
            (REVOLUTE, REVOLUTE),
            ((0.0, 0.0, 0.0), (0.0, -0.5, 0.0)),
            ((0.0, 0.5, 0.0), (0.0, 0.5, 0.0)),
            ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
            (IDENTITY, IDENTITY),
            (-3.0, -3.0),
            (3.0, 3.0),
            (0.05, 0.05),
            (False, False),
            (False, False),
        )
        result = step_joint_reference(
            state,
            inverse_mass,
            inverse_inertia,
            topology,
            [[0.0, 0.0]],
            [[20.0, 20.0]],
            steps=600,
        )
        self.assertTrue(all(error < 5.0e-4 for error in result.linear_error_m[0]))
        self.assertTrue(all(error < 1.0e-10 for error in result.angular_error_rad[0]))
        self.assertGreater(abs(result.coordinate[0][0] - angle), 0.1)
        self.assertTrue(all(error == 0.0 for error in result.limit_error[0]))
        assert_valid_joint_result(result, maximum_error=5.0e-4)


if __name__ == "__main__":
    unittest.main()
