"""Dependency-free Stage-6 oracle for batched rays and pinhole depth cameras.

This module is an inspectable scalar CPU reference. It provides no CUDA,
rendering, mesh, texture, RGB, lens-distortion, rolling-shutter, or noise-model
claim. Camera rays and hit tensors are rectangular and intentionally map to a
future FP32 CUDA contract without changing physics state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import List, Sequence, Tuple


CONTRACT_ID = "box3d.rays-depth/v1"
MISS = 0
BOX = 1
PLANE = 2
MISS_BODY_INDEX = -1
MISS_GEOMETRY_INDEX = -1
RAY_EPSILON = 1.0e-7

Vec3 = List[float]


@dataclass(frozen=True)
class RayQueryResult:
    distance_m: List[List[float]]
    body_index: List[List[int]]
    geometry_kind: List[List[int]]
    geometry_index: List[List[int]]
    normal: List[List[Vec3]]
    hit_position_m: List[List[Vec3]]


@dataclass(frozen=True)
class PinholeCamera:
    """Calibrated camera whose +X/+Y/+Z axes mean right/down/forward."""

    camera_id: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    maximum_distance_m: float
    parent_body: int = -1
    position_parent_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion_parent_from_camera_xyzw: Tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        1.0,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id or len(self.camera_id) > 128:
            raise ValueError("camera_id must be a non-empty string of at most 128 characters")
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or not 1 <= self.width <= 4096
            or not 1 <= self.height <= 4096
        ):
            raise ValueError("camera dimensions must be integers in [1, 4096]")
        numeric = (
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.maximum_distance_m,
            *self.position_parent_m,
            *self.quaternion_parent_from_camera_xyzw,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("camera calibration/pose values must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0 or self.maximum_distance_m <= 0.0:
            raise ValueError("focal lengths and maximum distance must be positive")
        if len(self.position_parent_m) != 3 or len(self.quaternion_parent_from_camera_xyzw) != 4:
            raise ValueError("camera pose must contain xyz and xyzw rows")
        quaternion_norm = math.sqrt(
            sum(value * value for value in self.quaternion_parent_from_camera_xyzw)
        )
        if abs(quaternion_norm - 1.0) > 2.0e-5:
            raise ValueError("camera quaternion must be normalized")
        if isinstance(self.parent_body, bool) or not isinstance(self.parent_body, int) or self.parent_body < -1:
            raise ValueError("parent_body must be -1 or a non-negative integer")


@dataclass(frozen=True)
class CameraRig:
    cameras: Tuple[PinholeCamera, ...]

    def __post_init__(self) -> None:
        if not self.cameras:
            raise ValueError("camera rig must contain at least one camera")
        identifiers = [camera.camera_id for camera in self.cameras]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("camera IDs must be unique")


@dataclass(frozen=True)
class CameraRayBatch:
    origins_m: List[List[Vec3]]
    directions: List[List[Vec3]]
    maximum_distance_m: List[List[float]]
    forward_cosine: List[List[float]]
    camera_index: List[int]
    pixel_xy: List[Tuple[int, int]]
    frame_offsets: Tuple[int, ...]


@dataclass(frozen=True)
class DepthImage:
    camera_id: str
    width: int
    height: int
    depth_z_m: List[List[float]]
    range_m: List[List[float]]
    body_index: List[List[int]]


def _add(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return [a[index] + b[index] for index in range(3)]


def _sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return [a[index] - b[index] for index in range(3)]


def _scale(vector: Sequence[float], scalar: float) -> Vec3:
    return [value * scalar for value in vector]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(a[index] * b[index] for index in range(3))


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Sequence[float]) -> Vec3:
    length = _norm(vector)
    if length <= 1.0e-12:
        raise ValueError("direction/normal cannot be degenerate")
    return _scale(vector, 1.0 / length)


def _rotate(quaternion: Sequence[float], vector: Sequence[float]) -> Vec3:
    qv = quaternion[:3]
    twice = _scale(_cross(qv, vector), 2.0)
    return _add(vector, _add(_scale(twice, quaternion[3]), _cross(qv, twice)))


def _inverse_rotate(quaternion: Sequence[float], vector: Sequence[float]) -> Vec3:
    return _rotate((-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]), vector)


def _quaternion_multiply(a: Sequence[float], b: Sequence[float]) -> List[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _finite_vector(row, length: int, name: str) -> None:
    if len(row) != length or not all(math.isfinite(float(value)) for value in row):
        raise ValueError("{} must contain {} finite values".format(name, length))


def _validate_query_layout(
    origins,
    directions,
    maximum_distance,
    box_state,
    box_half_extents,
    plane_normals,
    plane_offsets,
) -> Tuple[int, int, int, int]:
    worlds = len(origins)
    if worlds <= 0 or not all(
        len(value) == worlds
        for value in (
            directions,
            maximum_distance,
            box_state,
            box_half_extents,
            plane_normals,
            plane_offsets,
        )
    ):
        raise ValueError("all ray/geometry arrays must share a positive world count")
    rays = len(origins[0])
    boxes = len(box_state[0])
    planes = len(plane_normals[0])
    if rays <= 0 or boxes + planes <= 0:
        raise ValueError("queries require at least one ray and one geometry")
    for world in range(worlds):
        if not (
            len(origins[world])
            == len(directions[world])
            == len(maximum_distance[world])
            == rays
        ):
            raise ValueError("ray counts must be rectangular")
        if len(box_state[world]) != boxes or len(box_half_extents[world]) != boxes:
            raise ValueError("box counts must be rectangular")
        if len(plane_normals[world]) != planes or len(plane_offsets[world]) != planes:
            raise ValueError("plane counts must be rectangular")
        for ray in range(rays):
            _finite_vector(origins[world][ray], 3, "ray origin")
            _finite_vector(directions[world][ray], 3, "ray direction")
            if abs(_norm(directions[world][ray]) - 1.0) > 2.0e-5:
                raise ValueError("ray directions must be normalized")
            distance = maximum_distance[world][ray]
            if not math.isfinite(float(distance)) or distance <= 0.0:
                raise ValueError("ray maximum distances must be finite and positive")
        for box in range(boxes):
            _finite_vector(box_state[world][box], 13, "box state")
            _finite_vector(box_half_extents[world][box], 3, "box half extent")
            if any(value <= 0.0 for value in box_half_extents[world][box]):
                raise ValueError("box half extents must be positive")
            quaternion_norm = math.sqrt(
                sum(float(value) ** 2 for value in box_state[world][box][3:7])
            )
            if abs(quaternion_norm - 1.0) > 2.0e-5:
                raise ValueError("box quaternions must be normalized")
        for plane in range(planes):
            _finite_vector(plane_normals[world][plane], 3, "plane normal")
            if abs(_norm(plane_normals[world][plane]) - 1.0) > 2.0e-5:
                raise ValueError("plane normals must be normalized")
            if not math.isfinite(float(plane_offsets[world][plane])):
                raise ValueError("plane offsets must be finite")
    return worlds, rays, boxes, planes


def _ray_box(
    origin: Sequence[float],
    direction: Sequence[float],
    state: Sequence[float],
    half_extents: Sequence[float],
    maximum_distance: float,
    epsilon: float,
):
    local_origin = _inverse_rotate(state[3:7], _sub(origin, state[:3]))
    local_direction = _inverse_rotate(state[3:7], direction)
    near, far = -math.inf, math.inf
    near_normal = [0.0, 0.0, 0.0]
    far_normal = [0.0, 0.0, 0.0]
    for axis in range(3):
        component = local_direction[axis]
        if abs(component) <= epsilon:
            if local_origin[axis] < -half_extents[axis] - epsilon or local_origin[axis] > half_extents[axis] + epsilon:
                return None
            continue
        lower = (-half_extents[axis] - local_origin[axis]) / component
        upper = (half_extents[axis] - local_origin[axis]) / component
        lower_normal = [0.0, 0.0, 0.0]
        upper_normal = [0.0, 0.0, 0.0]
        lower_normal[axis] = -1.0
        upper_normal[axis] = 1.0
        if lower > upper:
            lower, upper = upper, lower
            lower_normal, upper_normal = upper_normal, lower_normal
        if lower > near:
            near, near_normal = lower, lower_normal
        if upper < far:
            far, far_normal = upper, upper_normal
        if near > far + epsilon:
            return None
    if far < -epsilon:
        return None
    distance, local_normal = (near, near_normal) if near >= 0.0 else (max(0.0, far), far_normal)
    if distance > maximum_distance + epsilon:
        return None
    normal = _normalize(_rotate(state[3:7], local_normal))
    return max(0.0, distance), normal


def _ray_plane(
    origin: Sequence[float],
    direction: Sequence[float],
    normal: Sequence[float],
    offset: float,
    maximum_distance: float,
    epsilon: float,
):
    denominator = _dot(normal, direction)
    if abs(denominator) <= epsilon:
        return None
    distance = (offset - _dot(normal, origin)) / denominator
    if distance < -epsilon or distance > maximum_distance + epsilon:
        return None
    facing_normal = list(normal) if denominator < 0.0 else _scale(normal, -1.0)
    return max(0.0, distance), facing_normal


def query_rays(
    origins_m: Sequence[Sequence[Sequence[float]]],
    directions: Sequence[Sequence[Sequence[float]]],
    maximum_distance_m: Sequence[Sequence[float]],
    box_state: Sequence[Sequence[Sequence[float]]],
    box_half_extents_m: Sequence[Sequence[Sequence[float]]],
    plane_normals: Sequence[Sequence[Sequence[float]]],
    plane_offsets_m: Sequence[Sequence[float]],
    *,
    epsilon: float = RAY_EPSILON,
) -> RayQueryResult:
    """Return the deterministic nearest box or double-sided plane hit per ray."""

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("ray epsilon must be finite and positive")
    worlds, rays, boxes, planes = _validate_query_layout(
        origins_m,
        directions,
        maximum_distance_m,
        box_state,
        box_half_extents_m,
        plane_normals,
        plane_offsets_m,
    )
    distances, bodies, kinds, indices, normals, positions = [], [], [], [], [], []
    for world in range(worlds):
        world_distance, world_body, world_kind, world_index, world_normal, world_position = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for ray in range(rays):
            origin, direction = origins_m[world][ray], directions[world][ray]
            best_distance = float(maximum_distance_m[world][ray])
            best_body, best_kind, best_index = MISS_BODY_INDEX, MISS, MISS_GEOMETRY_INDEX
            best_normal = [0.0, 0.0, 0.0]
            best_key = boxes + planes + 1
            for box in range(boxes):
                hit = _ray_box(
                    origin,
                    direction,
                    box_state[world][box],
                    box_half_extents_m[world][box],
                    best_distance,
                    epsilon,
                )
                if hit is None:
                    continue
                distance, normal = hit
                if distance < best_distance - epsilon or (
                    abs(distance - best_distance) <= epsilon and box < best_key
                ):
                    best_distance, best_body, best_kind, best_index = distance, box, BOX, box
                    best_normal, best_key = normal, box
            for plane in range(planes):
                hit = _ray_plane(
                    origin,
                    direction,
                    plane_normals[world][plane],
                    plane_offsets_m[world][plane],
                    best_distance,
                    epsilon,
                )
                key = boxes + plane
                if hit is None:
                    continue
                distance, normal = hit
                if distance < best_distance - epsilon or (
                    abs(distance - best_distance) <= epsilon and key < best_key
                ):
                    best_distance = distance
                    best_body, best_kind, best_index = MISS_BODY_INDEX, PLANE, plane
                    best_normal, best_key = normal, key
            world_distance.append(best_distance)
            world_body.append(best_body)
            world_kind.append(best_kind)
            world_index.append(best_index)
            world_normal.append(best_normal)
            world_position.append(_add(origin, _scale(direction, best_distance)))
        distances.append(world_distance)
        bodies.append(world_body)
        kinds.append(world_kind)
        indices.append(world_index)
        normals.append(world_normal)
        positions.append(world_position)
    return RayQueryResult(distances, bodies, kinds, indices, normals, positions)


def _camera_seed(seed: int, world: int, camera_id: str) -> int:
    value = 2166136261
    for byte in camera_id.encode("utf-8"):
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return (int(seed) * 0x9E3779B1 + world * 0x85EBCA77 + value) & 0xFFFFFFFFFFFFFFFF


def make_camera_rays(
    rig: CameraRig,
    body_state: Sequence[Sequence[Sequence[float]]],
    *,
    seed: int = 61,
    pixel_jitter: float = 0.0,
) -> CameraRayBatch:
    """Generate deterministic calibrated rays for every camera and world."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("camera ray seed must be a non-negative integer")
    if not math.isfinite(pixel_jitter) or not 0.0 <= pixel_jitter <= 0.5:
        raise ValueError("pixel_jitter must be in [0, 0.5]")
    if not body_state:
        raise ValueError("body_state must contain at least one world")
    body_count = len(body_state[0])
    for world in body_state:
        if len(world) != body_count:
            raise ValueError("camera body state must be rectangular")
        for body in world:
            _finite_vector(body, 13, "camera parent body state")
            quaternion_norm = math.sqrt(sum(float(value) ** 2 for value in body[3:7]))
            if abs(quaternion_norm - 1.0) > 2.0e-5:
                raise ValueError("camera parent body quaternions must be normalized")
    if any(camera.parent_body >= body_count for camera in rig.cameras):
        raise ValueError("camera rig references a missing parent body")
    camera_index: List[int] = []
    pixel_xy: List[Tuple[int, int]] = []
    offsets = [0]
    for index, camera in enumerate(rig.cameras):
        for y in range(camera.height):
            for x in range(camera.width):
                camera_index.append(index)
                pixel_xy.append((x, y))
        offsets.append(len(camera_index))
    all_origins, all_directions, all_maximum, all_forward = [], [], [], []
    for world_index, world in enumerate(body_state):
        origins, directions, maximum, forward = [], [], [], []
        for camera in rig.cameras:
            if camera.parent_body == -1:
                position = list(camera.position_parent_m)
                quaternion = list(camera.quaternion_parent_from_camera_xyzw)
            else:
                parent = world[camera.parent_body]
                position = _add(
                    parent[:3], _rotate(parent[3:7], camera.position_parent_m)
                )
                quaternion = _quaternion_multiply(
                    parent[3:7], camera.quaternion_parent_from_camera_xyzw
                )
            rng = random.Random(_camera_seed(seed, world_index, camera.camera_id))
            for y in range(camera.height):
                for x in range(camera.width):
                    jitter_x = rng.uniform(-pixel_jitter, pixel_jitter) if pixel_jitter else 0.0
                    jitter_y = rng.uniform(-pixel_jitter, pixel_jitter) if pixel_jitter else 0.0
                    local = _normalize(
                        (
                            (x + 0.5 + jitter_x - camera.cx) / camera.fx,
                            (y + 0.5 + jitter_y - camera.cy) / camera.fy,
                            1.0,
                        )
                    )
                    origins.append(position.copy())
                    directions.append(_normalize(_rotate(quaternion, local)))
                    maximum.append(camera.maximum_distance_m)
                    forward.append(local[2])
        all_origins.append(origins)
        all_directions.append(directions)
        all_maximum.append(maximum)
        all_forward.append(forward)
    return CameraRayBatch(
        all_origins,
        all_directions,
        all_maximum,
        all_forward,
        camera_index,
        pixel_xy,
        tuple(offsets),
    )


def depth_images_from_hits(
    rig: CameraRig, rays: CameraRayBatch, hits: RayQueryResult
) -> List[List[DepthImage]]:
    """Map flat hit rows to range and optical-axis depth images; misses are zero."""

    worlds = len(rays.origins_m)
    total_rays = rays.frame_offsets[-1]
    if not (
        len(hits.distance_m)
        == len(hits.body_index)
        == len(hits.geometry_kind)
        == worlds
    ):
        raise ValueError("hit/ray world count mismatch")
    output: List[List[DepthImage]] = []
    for world in range(worlds):
        if len(hits.distance_m[world]) != total_rays:
            raise ValueError("hit/ray count mismatch")
        images = []
        for camera_index, camera in enumerate(rig.cameras):
            start, end = rays.frame_offsets[camera_index : camera_index + 2]
            depth_values, range_values, body_values = [], [], []
            for row_start in range(start, end, camera.width):
                depth_row, range_row, body_row = [], [], []
                for ray in range(row_start, row_start + camera.width):
                    if hits.geometry_kind[world][ray] == MISS:
                        depth_row.append(0.0)
                        range_row.append(0.0)
                    else:
                        distance = hits.distance_m[world][ray]
                        depth_row.append(distance * rays.forward_cosine[world][ray])
                        range_row.append(distance)
                    body_row.append(hits.body_index[world][ray])
                depth_values.append(depth_row)
                range_values.append(range_row)
                body_values.append(body_row)
            images.append(
                DepthImage(
                    camera.camera_id,
                    camera.width,
                    camera.height,
                    depth_values,
                    range_values,
                    body_values,
                )
            )
        output.append(images)
    return output


__all__ = [
    "CONTRACT_ID",
    "MISS",
    "BOX",
    "PLANE",
    "MISS_BODY_INDEX",
    "MISS_GEOMETRY_INDEX",
    "RAY_EPSILON",
    "RayQueryResult",
    "PinholeCamera",
    "CameraRig",
    "CameraRayBatch",
    "DepthImage",
    "query_rays",
    "make_camera_rays",
    "depth_images_from_hits",
]
