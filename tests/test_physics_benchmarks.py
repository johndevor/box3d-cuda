from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from box3d_cuda.reference import SphereWorldConfig, assert_valid_state, make_drop_state, step_reference
from box3d_cuda.obb_reference import (
    OrientedBoxConfig,
    assert_valid_oriented_boxes,
    make_oriented_box_state,
    step_oriented_box_reference,
)
from box3d_cuda.gripper_reference import (
    GripperWorldConfig,
    finger_velocity,
    make_gripper_state,
    run_gripper_reference,
    step_gripper_reference,
)
from box3d_cuda.benchmarking.compare import compare, markdown
from box3d_cuda.benchmarking.contract import BenchmarkResult, CapabilitySet, load_result, write_result


CAPABILITIES = CapabilitySet(True, True, True, False, False, False, False, False)


class Box3DCudaReferenceTests(unittest.TestCase):
    def test_drop_world_is_deterministic_and_never_crosses_ground(self):
        state, mass, radius = make_drop_state(3, 6, seed=11)
        first = step_reference(state, mass, radius, SphereWorldConfig(), steps=240)
        second = step_reference(state, mass, radius, SphereWorldConfig(), steps=240)

        self.assertEqual(first, second)
        assert_valid_state(first, radius)

    def test_contact_uses_impulse_instead_of_attachment(self):
        state, mass, radius = make_drop_state(1, 1)
        state[0][0][1] = radius[0][0] - 0.01
        state[0][0][8] = -1.0

        result = step_reference(state, mass, radius, SphereWorldConfig(restitution=0.5))

        self.assertGreater(result[0][0][8], 0.0)
        self.assertGreaterEqual(result[0][0][1], radius[0][0] - 2.0e-4)

    def test_pair_contact_separates_overlapping_spheres(self):
        state, mass, radius = make_drop_state(1, 2)
        state[0][0][0:3] = [0.0, 0.2, 0.0]
        state[0][1][0:3] = [0.04, 0.2, 0.0]
        target = radius[0][0] + radius[0][1]

        result = step_reference(state, mass, radius)
        separation = result[0][1][0] - result[0][0][0]

        self.assertGreaterEqual(separation, target - 2.0e-4)

    def test_parallel_jaw_grasp_lifts_and_physically_releases_cube(self):
        result = run_gripper_reference(worlds=3)

        self.assertTrue(result["passed"], result)
        self.assertTrue(result["touched"])
        self.assertTrue(result["bilateral_contact"])
        self.assertTrue(result["lifted"])
        self.assertTrue(result["fell_after_release"])

    def test_open_fingers_do_not_drag_cube_without_contact(self):
        config = GripperWorldConfig()
        cube, fingers = make_gripper_state(1, config)
        cube[0][1] = 0.2
        before = cube[0][0]

        cube, _, contacts = step_gripper_reference(
            cube,
            fingers,
            finger_velocity(config.release_step, config),
            config,
        )

        self.assertEqual(contacts, [[False, False]])
        self.assertEqual(cube[0][0], before)

    def test_parallel_jaw_lift_fails_without_friction(self):
        result = run_gripper_reference(worlds=1, config=GripperWorldConfig(friction=0.0))

        self.assertTrue(result["touched"])
        self.assertTrue(result["bilateral_contact"])
        self.assertFalse(result["lifted"])
        self.assertLess(result["minimum_maximum_height_m"], 0.04)


class OrientedBoxReferenceTests(unittest.TestCase):
    def test_tumbling_boxes_remain_finite_and_above_plane(self):
        state, mass, half, inertia = make_oriented_box_state(2, 6)

        result, contacts, _ = step_oriented_box_reference(
            state, mass, half, inertia, OrientedBoxConfig(), steps=500
        )

        assert_valid_oriented_boxes(result, half)
        self.assertTrue(all(all(world) for world in contacts))

    def test_off_center_plane_contact_generates_angular_velocity(self):
        state, mass, half, inertia = make_oriented_box_state(1, 1)
        state[0][0][1] = 0.045
        state[0][0][7:10] = [0.0, -1.0, 0.0]
        state[0][0][10:13] = [0.0, 0.0, 0.0]

        result, contacts, _ = step_oriented_box_reference(
            state, mass, half, inertia, steps=8
        )

        self.assertTrue(contacts[0][0])
        self.assertGreater(sum(value * value for value in result[0][0][10:13]), 0.01)


class PhysicsComparisonTests(unittest.TestCase):
    def result(self, backend: str, contract: str, duration: float) -> BenchmarkResult:
        return BenchmarkResult(
            backend=backend,
            backend_version="test",
            workload="test",
            contract_id=contract,
            device="test-gpu",
            worlds=100,
            bodies_per_world=4,
            steps=10,
            duration_seconds=duration,
            capabilities=CAPABILITIES,
            correctness={"passed": True},
        )

    def test_speedup_requires_same_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "a.json", Path(directory) / "b.json"]
            write_result(paths[0], self.result("a", "one/v1", 1.0))
            write_result(paths[1], self.result("b", "two/v1", 0.5))

            report = compare(load_result(path) for path in paths)

        self.assertEqual(report["speedups"], [])
        self.assertIn("invalid", report["parity_warning"])
        self.assertIn("No shared", markdown(report))

    def test_matching_contract_produces_measured_speedup(self):
        payloads = [self.result("baseline", "same/v1", 2.0).to_dict(), self.result("candidate", "same/v1", 0.5).to_dict()]

        report = compare(payloads)

        self.assertEqual(report["speedups"][0]["world_step_speedup"], 4.0)

    def test_failed_correctness_cannot_be_written_as_performance(self):
        with self.assertRaisesRegex(ValueError, "correctness"):
            BenchmarkResult(
                backend="broken", backend_version="test", workload="test", contract_id="test/v1",
                device="test", worlds=1, bodies_per_world=1, steps=1, duration_seconds=1.0,
                capabilities=CAPABILITIES, correctness={"passed": False},
            )


if __name__ == "__main__":
    unittest.main()
