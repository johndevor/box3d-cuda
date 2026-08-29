from __future__ import annotations

import math
from pathlib import Path
import unittest

from box3d_cuda.ray_reference import (
    depth_images_from_hits,
    make_camera_rays,
    query_rays,
)
from box3d_cuda.smoke_camera import _fixture


ROOT = Path(__file__).resolve().parents[1]
CUDA_PATH = ROOT / "csrc" / "camera.cu"
BINDINGS_PATH = ROOT / "csrc" / "bindings.cpp"
EXTENSION_PATH = ROOT / "extension.py"
BENCHMARK_PATH = ROOT / "benchmark_depth_camera.py"


class CameraCudaSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cuda = CUDA_PATH.read_text()
        cls.bindings = BINDINGS_PATH.read_text()
        cls.extension = EXTENSION_PATH.read_text()

    def test_camera_entry_points_are_bound_end_to_end(self):
        for token in ("box3d_camera_rays_cuda", "box3d_camera_depth_cuda"):
            self.assertIn(token, self.cuda)
            self.assertIn(token, self.bindings)
        self.assertIn('module.def("camera_rays"', self.bindings)
        self.assertIn('module.def("camera_depth"', self.bindings)
        self.assertIn("def camera_rays(", self.extension)
        self.assertIn("def depth_camera_query(", self.extension)

    def test_one_cuda_lane_owns_one_world_ray_without_host_pixel_loop(self):
        self.assertIn("total = worlds * rays", self.cuda)
        self.assertIn("world = flat_ray / rays", self.cuda)
        self.assertIn("ray = flat_ray - world * rays", self.cuda)
        self.assertIn("pixel_camera[ray]", self.cuda)
        wrapper = self.extension.split("def depth_camera_query(", 1)[1]
        self.assertNotIn("for pixel", wrapper)
        self.assertNotIn("for world", wrapper)

    def test_body_attached_pose_uses_parent_transform_and_xyzw_composition(self):
        self.assertIn("rotate3(body + 3, local_position", self.cuda)
        self.assertIn("multiply_quaternion_xyzw", self.cuda)
        self.assertIn("body[axis] + rotated_position[axis]", self.cuda)
        self.assertIn("parent < 0", self.cuda)

    def test_calibrated_pixel_centers_and_optical_depth_are_explicit(self):
        self.assertIn("+ 0.5f - camera_intrinsics[2]", self.cuda)
        self.assertIn("+ 0.5f - camera_intrinsics[3]", self.cuda)
        self.assertIn("forward_cosine[flat_ray] = inverse_length", self.cuda)
        self.assertIn("distance[index] * forward_cosine[index]", self.cuda)
        self.assertIn("body_index[index] < 0", self.cuda)
        self.assertIn("depth_z[index] = 0.0f", self.cuda)


class CameraFixtureOracleTests(unittest.TestCase):
    def test_multi_camera_wrist_fixture_has_expected_cpu_geometry(self):
        states, half_extents, rig = _fixture()
        rays = make_camera_rays(rig, states)
        self.assertEqual(rays.frame_offsets, (0, 3, 7))
        self.assertEqual(rays.camera_index, [0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(rays.origins_m[0][0], [0.0, 3.0, 0.0])
        for actual, expected in zip(rays.origins_m[1][3], (2.0, 0.0, 0.0)):
            self.assertAlmostEqual(actual, expected, places=12)
        expected_direction = (1.0, -0.25, 0.25)
        length = math.sqrt(sum(value * value for value in expected_direction))
        for actual, expected in zip(
            rays.directions[1][3],
            (value / length for value in expected_direction),
        ):
            self.assertAlmostEqual(actual, expected, places=12)

        hits = query_rays(
            rays.origins_m,
            rays.directions,
            rays.maximum_distance_m,
            states,
            half_extents,
            [[], []],
            [[], []],
        )
        images = depth_images_from_hits(rig, rays, hits)
        self.assertEqual([image.camera_id for image in images[0]], ["scene", "wrist"])
        self.assertTrue(any(value > 0.0 for row in images[0][1].depth_z_m for value in row))
        self.assertTrue(any(value == 0.0 for row in images[1][0].depth_z_m for value in row))


class CameraBenchmarkContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = BENCHMARK_PATH.read_text()

    def test_timing_includes_camera_rays_query_and_depth_on_cuda(self):
        launch = self.source.split("def _launch(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("extension.camera_rays(", launch)
        self.assertIn("extension.ray_cast(", launch)
        self.assertIn("extension.camera_depth(", launch)
        self.assertNotIn("for pixel", launch)
        self.assertIn("torch.cuda.Event", self.source)

    def test_correctness_gates_precede_timing_and_claims_are_bounded(self):
        self.assertLess(
            self.source.index("run_camera_correctness()"),
            self.source.index("start.record()"),
        )
        self.assertIn("deterministic_replay_passed", self.source)
        self.assertIn("reported_rgb_or_raster_pixels", self.source)
        self.assertIn("No matched PhysX/ManiSkill speedup is claimed", self.source)


if __name__ == "__main__":
    unittest.main()
