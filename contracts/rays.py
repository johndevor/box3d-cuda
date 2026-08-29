"""Backend-neutral Stage-6 native batched-ray comparison contract.

This contract deliberately measures ray-query distance and primitive identity,
not a raster camera, RGB, or pixels.  Ray construction is performed outside
the timed region; both backends time one native batched first-hit query over
the same normalized rays and fixed collision scene.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


CONTRACT_ID = "box3d.batched-ray-depth/v1"
CUDA_BACKEND = "box3d_cuda_stage6"
PHYSX_BACKEND = "maniskill_physx_cuda"
DEFAULT_SEED = 67
WORLDS = 1024
PRIMITIVE_COUNT = 8
RIG_COUNT = 2
RIG_HEIGHT = 8
RIG_WIDTH = 16
RAYS_PER_RIG = RIG_HEIGHT * RIG_WIDTH
RAYS_PER_WORLD = RIG_COUNT * RAYS_PER_RIG
BENCHMARK_STEPS = 240
CORRECTNESS_STEPS = 16
NEAR_M = 0.05
FAR_M = 8.0
MISS_DEPTH_M = FAR_M
MISS_PRIMITIVE_ID = -1
QUERY_SCHEDULE_ID = "seeded-rig-yaw/v1"

MIN_HIT_ID_AGREEMENT_RATIO = 0.995
MIN_MISS_AGREEMENT_RATIO = 0.999
MAX_HIT_DEPTH_ERROR_M = 0.002
MIN_HIT_NORMAL_COSINE = 0.999
MIN_HIT_RATIO = 0.10
MAX_HIT_RATIO = 0.90


def _normalize(value: Sequence[float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(float(component) ** 2 for component in value))
    if not math.isfinite(length) or length <= 1.0e-12:
        raise ValueError("ray direction or camera basis is degenerate")
    return tuple(float(component) / length for component in value)  # type: ignore[return-value]


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(x) - float(y) for x, y in zip(a, b))  # type: ignore[return-value]


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@dataclass(frozen=True)
class RayRig:
    name: str
    origin_m: tuple[float, float, float]
    target_m: tuple[float, float, float]
    vertical_fov_degrees: float = 55.0
    height: int = RIG_HEIGHT
    width: int = RIG_WIDTH


@dataclass(frozen=True)
class ScenePrimitive:
    primitive_id: int
    name: str
    center_m: tuple[float, float, float]
    half_extents_m: tuple[float, float, float]
    yaw_degrees: float = 0.0


@dataclass(frozen=True)
class RayDepthSpec:
    contract_id: str = CONTRACT_ID
    rigs: tuple[RayRig, ...] = (
        RayRig("front_oblique", (0.0, 1.2, -2.0), (0.0, 0.5, 2.5)),
        RayRig("side_oblique", (2.5, 1.4, 0.0), (0.0, 0.5, 3.0)),
    )
    primitives: tuple[ScenePrimitive, ...] = (
        ScenePrimitive(0, "ground_slab", (0.0, -0.10, 2.0), (6.0, 0.10, 6.0)),
        ScenePrimitive(1, "box_center", (0.0, 0.5, 2.0), (0.5, 0.5, 0.5)),
        ScenePrimitive(2, "box_left", (-1.60, 0.35, 1.00), (0.38, 0.35, 0.45), -18.0),
        ScenePrimitive(3, "box_right", (1.3, 0.75, 4.0), (0.4, 0.75, 0.4), 24.0),
        ScenePrimitive(4, "pillar_left", (-0.75, 0.55, 1.35), (0.28, 0.55, 0.28), 12.0),
        ScenePrimitive(5, "low_right", (1.0, 0.28, 2.6), (0.42, 0.28, 0.32), -31.0),
        ScenePrimitive(6, "far_center", (-0.25, 0.85, 4.7), (0.65, 0.85, 0.22), 8.0),
        ScenePrimitive(7, "thin_panel", (1.75, 0.60, 2.1), (0.18, 0.60, 0.75), 38.0),
    )

    def __post_init__(self) -> None:
        if len(self.rigs) != RIG_COUNT or any((rig.height, rig.width) != (RIG_HEIGHT, RIG_WIDTH) for rig in self.rigs):
            raise ValueError("v1 requires two ordered 8x16 ray rigs")
        if tuple(item.primitive_id for item in self.primitives) != tuple(range(PRIMITIVE_COUNT)):
            raise ValueError("primitive IDs must be stable, dense, and ordered")
        if any(any(value <= 0.0 for value in item.half_extents_m) for item in self.primitives):
            raise ValueError("all Stage-6 OBB half extents must be positive")

    def metadata(self, *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
        return {
            "scenario_seed": seed,
            "query_schedule_id": QUERY_SCHEDULE_ID,
            "query_steps": BENCHMARK_STEPS,
            "correctness_steps": CORRECTNESS_STEPS,
            "canonical_basis": "right-handed-y-up",
            "physx_basis_transform_xyz": ["x", "-z", "y"],
            "query_kind": "first-hit-ray-distance",
            "distance_semantics": "euclidean-ray-length-m",
            "directions_normalized": True,
            "near_m": NEAR_M,
            "far_m": FAR_M,
            "miss_depth_m": MISS_DEPTH_M,
            "miss_primitive_id": MISS_PRIMITIVE_ID,
            "ray_input_layout": ["world", "rig", "row", "column", "xyz"],
            "depth_output_layout": ["world", "rig", "row", "column"],
            "hit_id_output_layout": ["world", "rig", "row", "column"],
            "hit_normal_output_layout": ["world", "rig", "row", "column", "xyz"],
            "hit_normal_basis": "right-handed-y-up",
            "ray_batch_shape": [WORLDS, RIG_COUNT, RIG_HEIGHT, RIG_WIDTH, 3],
            "depth_batch_shape": [WORLDS, RIG_COUNT, RIG_HEIGHT, RIG_WIDTH],
            "rig_order": [rig.name for rig in self.rigs],
            "rig_specs": [
                {
                    "name": rig.name,
                    "origin_m": list(rig.origin_m),
                    "target_m": list(rig.target_m),
                    "vertical_fov_degrees": rig.vertical_fov_degrees,
                    "height": rig.height,
                    "width": rig.width,
                }
                for rig in self.rigs
            ],
            "primitive_order": [item.name for item in self.primitives],
            "primitive_specs": [
                {
                    "primitive_id": item.primitive_id,
                    "name": item.name,
                    "shape": "obb",
                    "center_m": list(item.center_m),
                    "half_extents_m": list(item.half_extents_m),
                    "yaw_degrees_canonical_y_up": item.yaw_degrees,
                }
                for item in self.primitives
            ],
            "timing_scope": "one-native-batched-first-hit-query",
            "timing_excludes": ["ray_generation", "host_device_transfer", "report_serialization"],
            "timing_includes": ["native_scene_query", "device_synchronization"],
            "camera_rendering": False,
            "reported_rgb_or_pixels": False,
        }


SPEC = RayDepthSpec()

GATE_THRESHOLDS = {
    "minimum_hit_id_agreement_ratio": MIN_HIT_ID_AGREEMENT_RATIO,
    "minimum_miss_agreement_ratio": MIN_MISS_AGREEMENT_RATIO,
    "maximum_hit_depth_error_m": MAX_HIT_DEPTH_ERROR_M,
    "minimum_hit_normal_cosine": MIN_HIT_NORMAL_COSINE,
    "minimum_hit_ratio": MIN_HIT_RATIO,
    "maximum_hit_ratio": MAX_HIT_RATIO,
}


def rig_yaw_radians(control_step: int, world_index: int, seed: int = DEFAULT_SEED) -> float:
    if control_step < 0 or world_index < 0 or seed < 0:
        raise ValueError("step, world, and seed must be nonnegative")
    phase = 2.0 * math.pi * ((seed * 19 + world_index * 37 + control_step * 11) % 997) / 997.0
    return math.radians(0.4) * math.sin(phase)


def _yaw_y(value: Sequence[float], angle: float) -> tuple[float, float, float]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        cosine * value[0] + sine * value[2],
        value[1],
        -sine * value[0] + cosine * value[2],
    )


def rays_for_rig(
    rig: RayRig, control_step: int, world_index: int, seed: int = DEFAULT_SEED
) -> tuple[list[list[tuple[float, float, float]]], list[list[tuple[float, float, float]]]]:
    """Generate one normalized rig in canonical Y-up, outside timed work."""

    forward = _normalize(_sub(rig.target_m, rig.origin_m))
    forward = _normalize(_yaw_y(forward, rig_yaw_radians(control_step, world_index, seed)))
    right = _normalize(_cross(forward, (0.0, 1.0, 0.0)))
    up = _normalize(_cross(right, forward))
    vertical = math.tan(math.radians(rig.vertical_fov_degrees) * 0.5)
    horizontal = vertical * rig.width / rig.height
    origin_rows: list[list[tuple[float, float, float]]] = []
    direction_rows: list[list[tuple[float, float, float]]] = []
    for row in range(rig.height):
        origin_row = []
        direction_row = []
        y = (1.0 - 2.0 * (row + 0.5) / rig.height) * vertical
        for column in range(rig.width):
            x = (2.0 * (column + 0.5) / rig.width - 1.0) * horizontal
            direction = _normalize(tuple(forward[i] + x * right[i] + y * up[i] for i in range(3)))
            origin_row.append(rig.origin_m)
            direction_row.append(direction)
        origin_rows.append(origin_row)
        direction_rows.append(direction_row)
    return origin_rows, direction_rows


def ray_batch(
    worlds: int, control_step: int, seed: int = DEFAULT_SEED
) -> tuple[list[Any], list[Any]]:
    if worlds <= 0:
        raise ValueError("worlds must be positive")
    origins, directions = [], []
    for world in range(worlds):
        world_origins, world_directions = [], []
        for rig in SPEC.rigs:
            rig_origins, rig_directions = rays_for_rig(rig, control_step, world, seed)
            world_origins.append(rig_origins)
            world_directions.append(rig_directions)
        origins.append(world_origins)
        directions.append(world_directions)
    return origins, directions


def _aabb_distance_and_normal(
    origin: Sequence[float], direction: Sequence[float], center: Sequence[float], half: Sequence[float]
) -> tuple[float, tuple[float, float, float]]:
    minimum, maximum = -math.inf, math.inf
    enter_normal, exit_normal = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    for axis in range(3):
        low, high = center[axis] - half[axis], center[axis] + half[axis]
        if abs(direction[axis]) <= 1.0e-12:
            if origin[axis] < low or origin[axis] > high:
                return math.inf, (0.0, 0.0, 0.0)
            continue
        first, second = (low - origin[axis]) / direction[axis], (high - origin[axis]) / direction[axis]
        first_normal = [0.0, 0.0, 0.0]
        second_normal = [0.0, 0.0, 0.0]
        first_normal[axis], second_normal[axis] = -1.0, 1.0
        if first > second:
            first, second = second, first
            first_normal, second_normal = second_normal, first_normal
        if first > minimum:
            minimum, enter_normal = first, tuple(first_normal)
        if second < maximum:
            maximum, exit_normal = second, tuple(second_normal)
        if maximum < minimum:
            return math.inf, (0.0, 0.0, 0.0)
    return (minimum, enter_normal) if minimum >= 0.0 else (maximum, exit_normal)


def _obb_distance_and_normal(
    origin: Sequence[float], direction: Sequence[float], item: ScenePrimitive
) -> tuple[float, tuple[float, float, float]]:
    angle = math.radians(item.yaw_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    offset = _sub(origin, item.center_m)
    # Inverse canonical Y-axis rotation into the box's local frame.
    local_origin = (
        cosine * offset[0] - sine * offset[2],
        offset[1],
        sine * offset[0] + cosine * offset[2],
    )
    local_direction = (
        cosine * direction[0] - sine * direction[2],
        direction[1],
        sine * direction[0] + cosine * direction[2],
    )
    distance, local_normal = _aabb_distance_and_normal(
        local_origin, local_direction, (0.0, 0.0, 0.0), item.half_extents_m
    )
    normal = (
        cosine * local_normal[0] + sine * local_normal[2],
        local_normal[1],
        -sine * local_normal[0] + cosine * local_normal[2],
    )
    return distance, normal


def query_ray_reference(
    origin: Sequence[float], direction: Sequence[float]
) -> tuple[float, int, tuple[float, float, float]]:
    direction = _normalize(direction)
    best_distance, best_id, best_normal = FAR_M, MISS_PRIMITIVE_ID, (0.0, 0.0, 0.0)
    for item in SPEC.primitives:
        distance, normal = _obb_distance_and_normal(origin, direction, item)
        if NEAR_M <= distance <= FAR_M and distance < best_distance:
            best_distance, best_id, best_normal = distance, item.primitive_id, normal
    return best_distance, best_id, best_normal


def query_reference(
    origins: Sequence[Any], directions: Sequence[Any]
) -> tuple[list[Any], list[Any], list[Any]]:
    """Small CPU oracle preserving the canonical [W,R,H,V] layout."""

    if len(origins) != len(directions) or not origins:
        raise ValueError("origin and direction world batches must match and be nonempty")
    all_depths, all_ids, all_normals = [], [], []
    for world_origins, world_directions in zip(origins, directions):
        if len(world_origins) != RIG_COUNT or len(world_directions) != RIG_COUNT:
            raise ValueError("ray world must contain the two ordered rigs")
        world_depths, world_ids, world_normals = [], [], []
        for rig_origins, rig_directions in zip(world_origins, world_directions):
            if len(rig_origins) != RIG_HEIGHT or len(rig_directions) != RIG_HEIGHT:
                raise ValueError("ray rig height differs from the contract")
            rig_depths, rig_ids, rig_normals = [], [], []
            for origin_row, direction_row in zip(rig_origins, rig_directions):
                if len(origin_row) != RIG_WIDTH or len(direction_row) != RIG_WIDTH:
                    raise ValueError("ray rig width differs from the contract")
                results = [query_ray_reference(origin, direction) for origin, direction in zip(origin_row, direction_row)]
                rig_depths.append([result[0] for result in results])
                rig_ids.append([result[1] for result in results])
                rig_normals.append([result[2] for result in results])
            world_depths.append(rig_depths)
            world_ids.append(rig_ids)
            world_normals.append(rig_normals)
        all_depths.append(world_depths)
        all_ids.append(world_ids)
        all_normals.append(world_normals)
    return all_depths, all_ids, all_normals


def canonical_y_up_to_physx_z_up(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError("basis transform requires xyz")
    return (float(value[0]), -float(value[2]), float(value[1]))


def _numeric(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"ray report missing finite numeric correctness result: {name}")
    return float(value)


def validate_ray_report(report: Mapping[str, Any], *, backend: str) -> None:
    if report.get("backend") != backend or report.get("contract_id") != CONTRACT_ID:
        raise RuntimeError("ray report backend or contract_id mismatch")
    if report.get("worlds") != WORLDS or report.get("bodies_per_world") != PRIMITIVE_COUNT or report.get("steps") != BENCHMARK_STEPS:
        raise RuntimeError("ray report must use 1024 worlds, eight OBBs, and 240 query steps")
    capabilities = report.get("capabilities", {})
    if capabilities.get("ray_queries") is not True or capabilities.get("camera_rendering") is not False:
        raise RuntimeError("ray report must measure ray queries and must not imply camera rendering")
    correctness = report.get("correctness", {})
    if correctness.get("passed") is not True or correctness.get("measured_runtime_evidence") is not True:
        raise RuntimeError("ray correctness must pass with measured runtime evidence")
    if correctness.get("synthetic") is not False or correctness.get("native_batched_ray_query") is not True:
        raise RuntimeError("synthetic or non-native ray work cannot count as measured parity")
    for key, expected in SPEC.metadata().items():
        if correctness.get(key) != expected:
            raise RuntimeError(f"ray report requires exact {key}")
    if correctness.get("gate_thresholds") != GATE_THRESHOLDS:
        raise RuntimeError("ray report gate thresholds differ from the fixed contract")
    for key in (
        "finite_depths", "depths_within_near_far", "deterministic_replay_passed",
        "world_isolation_passed",
    ):
        if correctness.get(key) is not True:
            raise RuntimeError(f"ray report failed {key}")
    if correctness.get("reported_rgb_or_pixels") is not False:
        raise RuntimeError("ray evidence cannot claim RGB or pixels")
    if correctness.get("observed_primitive_ids") != list(range(PRIMITIVE_COUNT)):
        raise RuntimeError("ray evidence must observe every fixed scene primitive")
    minimums = {
        "hit_id_agreement_ratio": MIN_HIT_ID_AGREEMENT_RATIO,
        "miss_agreement_ratio": MIN_MISS_AGREEMENT_RATIO,
        "minimum_hit_normal_cosine": MIN_HIT_NORMAL_COSINE,
    }
    for key, minimum in minimums.items():
        if _numeric(correctness.get(key), key) < minimum:
            raise RuntimeError(f"ray report fell below {key}")
    if _numeric(correctness.get("maximum_hit_depth_error_m"), "maximum_hit_depth_error_m") > MAX_HIT_DEPTH_ERROR_M:
        raise RuntimeError("ray report exceeded maximum_hit_depth_error_m")
    hit_ratio = _numeric(correctness.get("hit_ratio"), "hit_ratio")
    if not MIN_HIT_RATIO <= hit_ratio <= MAX_HIT_RATIO:
        raise RuntimeError("ray report hit_ratio is outside the fixed scene coverage band")


def validate_ray_contract_speedup(
    reports: Sequence[Mapping[str, Any]], comparison: Mapping[str, Any]
) -> Mapping[str, Any]:
    matches = [row for row in reports if row.get("contract_id") == CONTRACT_ID]
    if len(matches) != 2 or {row.get("backend") for row in matches} != {PHYSX_BACKEND, CUDA_BACKEND}:
        raise RuntimeError("ray speedup requires exactly measured PhysX and Box3D CUDA reports")
    by_backend = {row["backend"]: row for row in matches}
    validate_ray_report(by_backend[PHYSX_BACKEND], backend=PHYSX_BACKEND)
    validate_ray_report(by_backend[CUDA_BACKEND], backend=CUDA_BACKEND)
    rows = [row for row in comparison.get("speedups", ()) if row.get("contract_id") == CONTRACT_ID]
    if len(rows) != 1:
        raise RuntimeError("ray comparison must contain exactly one matched speedup")
    row = rows[0]
    if row.get("baseline") != PHYSX_BACKEND or row.get("candidate") != CUDA_BACKEND:
        raise RuntimeError("ray speedup direction must be PhysX baseline to Box3D CUDA candidate")
    _numeric(row.get("ray_queries_per_second_speedup"), "ray_queries_per_second_speedup")
    return row


__all__ = [name for name in globals() if name.isupper()] + [
    "RayRig", "ScenePrimitive", "RayDepthSpec", "SPEC", "rig_yaw_radians",
    "rays_for_rig", "ray_batch", "query_ray_reference", "query_reference",
    "canonical_y_up_to_physx_z_up", "validate_ray_report",
    "validate_ray_contract_speedup",
]
