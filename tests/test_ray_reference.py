from __future__ import annotations

import math
import unittest

from box3d_cuda.ray_reference import (
    BOX,
    CONTRACT_ID,
    MISS,
    PLANE,
    CameraRig,
    PinholeCamera,
    depth_images_from_hits,
    make_camera_rays,
    query_rays,
)


IDENTITY = (0.0, 0.0, 0.0, 1.0)


def body(position, quaternion=IDENTITY):
    return [*position, *quaternion, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def query(
    origins,
    directions,
    maximum,
    boxes,
    half,
    plane_normals=None,
    plane_offsets=None,
):
    return query_rays(
        [origins],
        [directions],
        [maximum],
        [boxes],
        [half],
        [plane_normals or []],
        [plane_offsets or []],
    )


class RayIntersectionTests(unittest.TestCase):
    def test_contract_and_axis_aligned_box_hit(self):
        result = query(
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0]],
            [10.0],
            [body((0.0, 0.0, 5.0))],
            [[1.0, 1.0, 1.0]],
        )
        self.assertEqual(CONTRACT_ID, "box3d.rays-depth/v1")
        self.assertEqual(result.geometry_kind, [[BOX]])
        self.assertEqual(result.body_index, [[0]])
        self.assertEqual(result.geometry_index, [[0]])
        self.assertAlmostEqual(result.distance_m[0][0], 4.0, places=12)
        self.assertEqual(result.normal[0][0], [0.0, 0.0, -1.0])
        self.assertEqual(result.hit_position_m[0][0], [0.0, 0.0, 4.0])

    def test_rotated_box_and_inside_exit_have_analytical_distances(self):
        angle = math.pi / 4.0
        quaternion = (0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0))
        rotated = query(
            [[0.0, 0.0, -3.0]],
            [[0.0, 0.0, 1.0]],
            [10.0],
            [body((0.0, 0.0, 0.0), quaternion)],
            [[1.0, 0.5, 0.25]],
        )
        self.assertAlmostEqual(rotated.distance_m[0][0], 3.0 - 0.25 / math.cos(angle), places=10)
        expected_normal = [-math.sin(angle), 0.0, -math.cos(angle)]
        for actual, expected in zip(rotated.normal[0][0], expected_normal):
            self.assertAlmostEqual(actual, expected, places=10)

        inside = query(
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [5.0],
            [body((0.0, 0.0, 0.0))],
            [[1.0, 2.0, 3.0]],
        )
        self.assertAlmostEqual(inside.distance_m[0][0], 1.0, places=12)
        self.assertEqual(inside.normal[0][0], [1.0, 0.0, 0.0])

    def test_plane_occludes_box_and_equal_distance_tie_is_deterministic(self):
        occluded = query(
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0]],
            [10.0],
            [body((0.0, 0.0, 5.0))],
            [[1.0, 1.0, 1.0]],
            [[0.0, 0.0, -1.0]],
            [-2.0],
        )
        self.assertEqual(occluded.geometry_kind, [[PLANE]])
        self.assertEqual(occluded.body_index, [[-1]])
        self.assertEqual(occluded.geometry_index, [[0]])
        self.assertAlmostEqual(occluded.distance_m[0][0], 2.0)
        self.assertEqual(occluded.normal[0][0], [0.0, 0.0, -1.0])

        tied = query(
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0]],
            [10.0],
            [body((0.0, 0.0, 5.0))],
            [[1.0, 1.0, 1.0]],
            [[0.0, 0.0, -1.0]],
            [-4.0],
        )
        self.assertEqual(tied.geometry_kind, [[BOX]])

    def test_miss_uses_maximum_distance_and_endpoint_sentinel(self):
        result = query(
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [3.5],
            [body((0.0, 0.0, 5.0))],
            [[1.0, 1.0, 1.0]],
        )
        self.assertEqual(result.geometry_kind, [[MISS]])
        self.assertEqual(result.body_index, [[-1]])
        self.assertEqual(result.geometry_index, [[-1]])
        self.assertEqual(result.normal[0][0], [0.0, 0.0, 0.0])
        self.assertEqual(result.distance_m[0][0], 3.5)
        self.assertEqual(result.hit_position_m[0][0], [3.5, 0.0, 0.0])

    def test_batched_worlds_are_isolated_and_rectangular(self):
        result = query_rays(
            [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]],
            [[20.0], [20.0]],
            [[body((0.0, 0.0, 5.0))], [body((0.0, 0.0, 8.0))]],
            [[(1.0, 1.0, 1.0)], [(1.0, 1.0, 1.0)]],
            [[], []],
            [[], []],
        )
        self.assertEqual(result.distance_m, [[4.0], [7.0]])
        self.assertEqual(result.body_index, [[0], [0]])

    def test_query_validation_rejects_non_unit_directions(self):
        with self.assertRaisesRegex(ValueError, "normalized"):
            query(
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 2.0]],
                [10.0],
                [body((0.0, 0.0, 5.0))],
                [[1.0, 1.0, 1.0]],
            )

    def test_grazing_box_and_parallel_plane_boundaries_are_explicit(self):
        grazing = query(
            [[1.0, 0.0, 0.0], [1.0 + 2.0e-7, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            [10.0, 10.0],
            [body((0.0, 0.0, 5.0))],
            [[1.0, 1.0, 1.0]],
        )
        self.assertEqual(grazing.geometry_kind[0], [BOX, MISS])
        self.assertAlmostEqual(grazing.distance_m[0][0], 4.0)

        plane_only = query(
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            [2.0, 1.0],
            [],
            [],
            [[0.0, 1.0, 0.0]],
            [0.0],
        )
        self.assertEqual(plane_only.geometry_kind[0], [MISS, PLANE])
        self.assertEqual(plane_only.distance_m[0], [2.0, 1.0])


class CameraTests(unittest.TestCase):
    def test_calibrated_center_and_corner_rays(self):
        camera = PinholeCamera("scene", 3, 1, 1.0, 1.0, 1.5, 0.5, 10.0)
        rays = make_camera_rays(CameraRig((camera,)), [[body((0.0, 0.0, 0.0))]])
        self.assertEqual(rays.frame_offsets, (0, 3))
        self.assertEqual(rays.pixel_xy, [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(rays.directions[0][1], [0.0, 0.0, 1.0])
        root_half = 1.0 / math.sqrt(2.0)
        for actual, expected in zip(rays.directions[0][0], (-root_half, 0.0, root_half)):
            self.assertAlmostEqual(actual, expected, places=15)
        for actual, expected in zip(rays.directions[0][2], (root_half, 0.0, root_half)):
            self.assertAlmostEqual(actual, expected, places=15)

    def test_body_attached_camera_pose_rotates_origin_and_forward(self):
        angle = math.pi / 2.0
        parent_quaternion = (0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0))
        camera = PinholeCamera(
            "wrist",
            1,
            1,
            1.0,
            1.0,
            0.5,
            0.5,
            10.0,
            parent_body=0,
            position_parent_m=(0.0, 0.0, 1.0),
        )
        rays = make_camera_rays(
            CameraRig((camera,)), [[body((1.0, 2.0, 3.0), parent_quaternion)]]
        )
        for actual, expected in zip(rays.origins_m[0][0], (2.0, 2.0, 3.0)):
            self.assertAlmostEqual(actual, expected, places=12)
        for actual, expected in zip(rays.directions[0][0], (1.0, 0.0, 0.0)):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_seeded_jitter_replays_and_camera_append_is_composable(self):
        first = PinholeCamera("wrist", 2, 2, 2.0, 2.0, 1.0, 1.0, 10.0)
        second = PinholeCamera("scene", 1, 1, 1.0, 1.0, 0.5, 0.5, 12.0)
        state = [[body((0.0, 0.0, 0.0))]]
        original = make_camera_rays(CameraRig((first,)), state, seed=73, pixel_jitter=0.4)
        replay = make_camera_rays(CameraRig((first,)), state, seed=73, pixel_jitter=0.4)
        changed = make_camera_rays(CameraRig((first,)), state, seed=74, pixel_jitter=0.4)
        extended = make_camera_rays(
            CameraRig((first, second)), state, seed=73, pixel_jitter=0.4
        )
        self.assertEqual(original, replay)
        self.assertNotEqual(original.directions, changed.directions)
        self.assertEqual(extended.directions[0][:4], original.directions[0])
        self.assertEqual(extended.forward_cosine[0][:4], original.forward_cosine[0])
        self.assertEqual(extended.frame_offsets, (0, 4, 5))

    def test_multi_camera_depth_reconstruction_and_occlusion(self):
        scene = PinholeCamera("scene", 3, 1, 1.0, 1.0, 1.5, 0.5, 20.0)
        inspection = PinholeCamera(
            "inspection", 1, 1, 1.0, 1.0, 0.5, 0.5, 20.0,
            position_parent_m=(0.0, 0.0, 1.0),
        )
        rig = CameraRig((scene, inspection))
        state = [[body((0.0, 0.0, 0.0))]]
        rays = make_camera_rays(rig, state)
        boxes = [[body((0.0, 0.0, 5.0)), body((0.0, 0.0, 8.0))]]
        half = [[(10.0, 10.0, 0.5), (10.0, 10.0, 0.5)]]
        hits = query_rays(
            rays.origins_m,
            rays.directions,
            rays.maximum_distance_m,
            boxes,
            half,
            [[]],
            [[]],
        )
        images = depth_images_from_hits(rig, rays, hits)
        self.assertEqual([image.camera_id for image in images[0]], ["scene", "inspection"])
        self.assertEqual(images[0][0].body_index, [[0, 0, 0]])
        for depth in images[0][0].depth_z_m[0]:
            self.assertAlmostEqual(depth, 4.5, places=10)
        self.assertAlmostEqual(images[0][1].depth_z_m[0][0], 3.5, places=10)
        self.assertGreater(images[0][0].range_m[0][0], images[0][0].depth_z_m[0][0])

    def test_camera_validation_rejects_duplicates_and_bad_intrinsics(self):
        camera = PinholeCamera("same", 1, 1, 1.0, 1.0, 0.5, 0.5, 10.0)
        with self.assertRaisesRegex(ValueError, "unique"):
            CameraRig((camera, camera))
        with self.assertRaisesRegex(ValueError, "positive"):
            PinholeCamera("bad", 1, 1, 0.0, 1.0, 0.5, 0.5, 10.0)


if __name__ == "__main__":
    unittest.main()
