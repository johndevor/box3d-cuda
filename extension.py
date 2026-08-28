"""Lazy build/load boundary for the CUDA implementation."""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path


_SAT_PAIR_VALIDATION_CACHE: dict[int, tuple] = {}
_RAY_VALIDATION_CACHE: dict[int, tuple] = {}
_RAY_GEOMETRY_VALIDATION_CACHE: dict[int, tuple] = {}


@lru_cache(maxsize=1)
def load_extension():
    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("Box3D CUDA requires a visible CUDA device")
    root = Path(__file__).resolve().parent
    return load(
        name="factory_box3d_cuda_v10",
        sources=[
            str(root / "csrc" / "bindings.cpp"),
            str(root / "csrc" / "step.cu"),
            str(root / "csrc" / "gripper.cu"),
            str(root / "csrc" / "obb.cu"),
            str(root / "csrc" / "sat.cu"),
            str(root / "csrc" / "manifold.cu"),
            str(root / "csrc" / "joint.cu"),
            str(root / "csrc" / "ray.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=False,
    )


def _validate_ray_values(ray_directions, maximum_distance) -> None:
    """Validate static ray packets once per tensor mutation version.

    The cached GPU reductions keep validation out of repeated timed launches
    while still failing closed after an in-place edit.
    """

    import torch

    key = id(ray_directions)
    versions = (
        int(getattr(ray_directions, "_version", -1)),
        int(getattr(maximum_distance, "_version", -1)),
        id(maximum_distance),
    )
    cached = _RAY_VALIDATION_CACHE.get(key)
    if cached is not None and cached[0] is ray_directions and cached[1] is maximum_distance and cached[2:] == versions[:2]:
        return
    lengths = torch.linalg.vector_norm(ray_directions, dim=-1)
    if not bool(torch.isfinite(ray_directions).all().item()):
        raise ValueError("ray directions must be finite")
    if not bool(torch.isfinite(maximum_distance).all().item()):
        raise ValueError("maximum_distance must be finite")
    if not bool(torch.all(torch.abs(lengths - 1.0) <= 1.0e-4).item()):
        raise ValueError("ray directions must be unit length within 1e-4")
    if not bool(torch.all(maximum_distance > 0.0).item()):
        raise ValueError("maximum_distance must be positive")
    if len(_RAY_VALIDATION_CACHE) >= 128:
        _RAY_VALIDATION_CACHE.clear()
    _RAY_VALIDATION_CACHE[key] = (
        ray_directions, maximum_distance, versions[0], versions[1]
    )


def _validate_ray_geometry(state, half_extents) -> None:
    """Fail closed on pose/extents, amortized for immutable geometry packets."""

    import torch

    key = id(state)
    versions = (
        int(getattr(state, "_version", -1)),
        int(getattr(half_extents, "_version", -1)),
    )
    cached = _RAY_GEOMETRY_VALIDATION_CACHE.get(key)
    if cached is not None and cached[0] is state and cached[1] is half_extents and cached[2:] == versions:
        return
    if not bool(torch.isfinite(state[..., :7]).all().item()):
        raise ValueError("ray body poses must be finite")
    quaternion_length = torch.linalg.vector_norm(state[..., 3:7], dim=-1)
    if not bool(torch.all(torch.abs(quaternion_length - 1.0) <= 1.0e-4).item()):
        raise ValueError("ray body quaternions must be unit length within 1e-4")
    if not bool(torch.isfinite(half_extents).all().item()) or not bool(torch.all(half_extents > 0.0).item()):
        raise ValueError("half_extents must be finite and positive")
    if len(_RAY_GEOMETRY_VALIDATION_CACHE) >= 128:
        _RAY_GEOMETRY_VALIDATION_CACHE.clear()
    _RAY_GEOMETRY_VALIDATION_CACHE[key] = (state, half_extents, *versions)


def ray_cast(state, half_extents, body_enabled, ray_origins, ray_directions, maximum_distance):
    """Return nearest OBB hit distance, body index, and world-space normal."""

    import torch

    tensors = (state, half_extents, body_enabled, ray_origins, ray_directions, maximum_distance)
    if not all(isinstance(item, torch.Tensor) for item in tensors):
        raise TypeError("ray inputs must all be torch tensors")
    if not all(item.is_cuda for item in tensors):
        raise ValueError("ray inputs must all be CUDA tensors")
    if len({item.device for item in tensors}) != 1:
        raise ValueError("ray inputs must share one CUDA device")
    if any(item.dtype != torch.float32 for item in (state, half_extents, ray_origins, ray_directions, maximum_distance)):
        raise ValueError("ray state, geometry, and query tensors must be float32")
    if body_enabled.dtype != torch.uint8:
        raise ValueError("body_enabled must be uint8")
    if state.ndim != 3 or state.shape[2] != 13:
        raise ValueError("ray state must have shape [worlds,bodies,13]")
    worlds, bodies = state.shape[:2]
    if worlds <= 0 or not 1 <= bodies <= 32:
        raise ValueError("ray worlds require 1..32 bodies")
    if tuple(half_extents.shape) != (worlds, bodies, 3):
        raise ValueError("half_extents must have shape [worlds,bodies,3]")
    if tuple(body_enabled.shape) != (worlds, bodies):
        raise ValueError("body_enabled must have shape [worlds,bodies]")
    if ray_origins.ndim != 3 or ray_origins.shape[0] != worlds or ray_origins.shape[2] != 3:
        raise ValueError("ray_origins must have shape [worlds,rays,3]")
    rays = ray_origins.shape[1]
    if not 1 <= rays <= 262144 or worlds * rays > 1048576:
        raise ValueError("ray batch must contain 1..1,048,576 total rays and at most 262,144 rays/world")
    if tuple(ray_directions.shape) != (worlds, rays, 3):
        raise ValueError("ray_directions must have shape [worlds,rays,3]")
    if tuple(maximum_distance.shape) != (worlds, rays):
        raise ValueError("maximum_distance must have shape [worlds,rays]")
    _validate_ray_geometry(state, half_extents)
    if not bool(torch.isfinite(ray_origins).all().item()):
        raise ValueError("ray origins must be finite")
    _validate_ray_values(ray_directions, maximum_distance)
    return load_extension().ray_cast(
        state, half_extents, body_enabled, ray_origins, ray_directions, maximum_distance
    )


def step(state, inverse_mass, radius, *, dt, substeps, gravity_y, restitution, friction):
    return load_extension().step(
        state,
        inverse_mass,
        radius,
        float(dt),
        int(substeps),
        float(gravity_y),
        float(restitution),
        float(friction),
    )


def gripper_step(cube_state, finger_positions, finger_velocity, config):
    return load_extension().gripper_step(
        cube_state,
        finger_positions,
        finger_velocity,
        float(config.dt),
        int(config.substeps),
        float(config.gravity_y),
        float(config.restitution),
        float(config.friction),
        float(config.position_slop),
        float(config.position_correction),
        *map(float, config.cube_half_extent),
        *map(float, config.finger_half_extent),
    )


def obb_step(state, inverse_mass, half_extents, inverse_inertia, config):
    return load_extension().obb_step(
        state,
        inverse_mass,
        half_extents,
        inverse_inertia,
        float(config.dt),
        int(config.substeps),
        float(config.gravity_y),
        float(config.restitution),
        float(config.friction),
        float(config.position_slop),
        float(config.angular_damping),
        int(config.solver_iterations),
    )


def _validate_sat_config(config) -> None:
    required = (
        "dt", "substeps", "gravity_y", "restitution", "friction",
        "position_slop", "position_correction", "angular_damping",
        "solver_iterations", "sat_epsilon",
    )
    missing = [name for name in required if not hasattr(config, name)]
    if missing:
        raise TypeError(f"SAT config is missing fields: {', '.join(missing)}")
    numeric = (
        "dt", "gravity_y", "restitution", "friction", "position_slop",
        "position_correction", "angular_damping", "sat_epsilon",
    )
    for name in numeric:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"SAT config {name} must be a finite number")
    for name in ("substeps", "solver_iterations"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"SAT config {name} must be an integer")
    if not 0.0 < float(config.dt) <= 1.0:
        raise ValueError("SAT config dt must be in (0,1]")
    if not 1 <= config.substeps <= 64 or not 1 <= config.solver_iterations <= 64:
        raise ValueError("SAT substeps and solver_iterations must be in [1,64]")
    if abs(float(config.gravity_y)) > 1000.0:
        raise ValueError("SAT gravity_y must be bounded by 1000 m/s^2")
    if not 0.0 <= float(config.restitution) <= 1.0:
        raise ValueError("SAT restitution must be in [0,1]")
    if not 0.0 <= float(config.friction) <= 10.0:
        raise ValueError("SAT friction must be in [0,10]")
    if not 0.0 <= float(config.position_slop) <= 0.1:
        raise ValueError("SAT position_slop must be in [0,0.1]")
    if not 0.0 <= float(config.position_correction) <= 1.0:
        raise ValueError("SAT position_correction must be in [0,1]")
    if not 0.0 <= float(config.angular_damping) <= 100.0:
        raise ValueError("SAT angular_damping must be in [0,100]")
    if not 0.0 < float(config.sat_epsilon) <= 0.01:
        raise ValueError("SAT sat_epsilon must be in (0,0.01]")


def _validate_sat_pair_indices(pair_indices, bodies: int) -> None:
    """Validate small static pair tables once per tensor version.

    Copying three pairs to the host on every step would serialize the timed
    CUDA workload.  Torch's mutation version is part of the cache key, so an
    in-place edit is revalidated before the next launch.
    """

    identity = id(pair_indices)
    version = int(getattr(pair_indices, "_version", -1))
    cached = _SAT_PAIR_VALIDATION_CACHE.get(identity)
    if cached is not None and cached[0] is pair_indices and cached[1:] == (version, bodies):
        return
    values = pair_indices.detach().cpu().tolist()
    if any(len(pair) != 2 for pair in values):
        raise ValueError("pair_indices must contain index pairs")
    normalized = [tuple(int(index) for index in pair) for pair in values]
    if any(left == right or left < 0 or right < 0 or left >= bodies or right >= bodies
           for left, right in normalized):
        raise ValueError("pair_indices must reference two distinct in-range bodies")
    canonical = [tuple(sorted(pair)) for pair in normalized]
    if len(canonical) != len(set(canonical)):
        raise ValueError("pair_indices must not contain duplicate undirected pairs")
    if len(_SAT_PAIR_VALIDATION_CACHE) >= 128:
        _SAT_PAIR_VALIDATION_CACHE.clear()
    # Retaining the small pair tensor prevents object/data-pointer reuse from
    # accidentally inheriting a prior validation decision.
    _SAT_PAIR_VALIDATION_CACHE[identity] = (pair_indices, version, bodies)


def _validate_joint_config(config) -> None:
    required = (
        "dt", "substeps", "gravity_y", "solver_iterations",
        "position_correction", "position_slop", "angular_slop",
        "maximum_linear_repair_m", "maximum_angular_repair_rad",
        "warm_start_factor",
    )
    missing = [name for name in required if not hasattr(config, name)]
    if missing:
        raise TypeError(f"joint config is missing fields: {', '.join(missing)}")
    numeric = [name for name in required if name not in ("substeps", "solver_iterations")]
    if any(isinstance(getattr(config, name), bool) or not isinstance(getattr(config, name), (int, float)) or not math.isfinite(float(getattr(config, name))) for name in numeric):
        raise ValueError("joint config numeric fields must be finite")
    if not isinstance(config.substeps, int) or not isinstance(config.solver_iterations, int):
        raise ValueError("joint substeps and solver_iterations must be integers")
    if not 0.0 < float(config.dt) <= 1.0 or not 1 <= config.substeps <= 64 or not 1 <= config.solver_iterations <= 64:
        raise ValueError("joint dt/substeps/solver_iterations are out of bounds")
    if not 0.0 <= float(config.position_correction) <= 1.0:
        raise ValueError("joint position_correction must be in [0,1]")
    if any(float(getattr(config, name)) < 0.0 for name in ("position_slop", "angular_slop", "maximum_linear_repair_m", "maximum_angular_repair_rad")):
        raise ValueError("joint slop and repair bounds must be non-negative")
    if not 0.0 <= float(config.warm_start_factor) <= 1.0:
        raise ValueError("joint warm_start_factor must be in [0,1]")


def sat_step(state, inverse_mass, half_extents, inverse_inertia, pair_indices, config):
    """Advance fixed-small worlds with 15-axis OBB pair contact."""

    import torch

    tensors = (state, inverse_mass, half_extents, inverse_inertia, pair_indices)
    if not all(isinstance(item, torch.Tensor) for item in tensors):
        raise TypeError("SAT inputs must be torch tensors")
    if not all(item.is_cuda for item in tensors):
        raise ValueError("SAT inputs must all be CUDA tensors")
    if len({item.device for item in tensors}) != 1:
        raise ValueError("SAT inputs must be on the same CUDA device")
    if state.dtype != torch.float32 or inverse_mass.dtype != torch.float32 or \
            half_extents.dtype != torch.float32 or inverse_inertia.dtype != torch.float32:
        raise ValueError("SAT state and material tensors must be float32")
    if pair_indices.dtype != torch.int64:
        raise ValueError("SAT pair_indices must be int64")
    if state.ndim != 3 or state.shape[2] != 13:
        raise ValueError("SAT state must have shape [worlds,bodies,13]")
    worlds, bodies = state.shape[:2]
    if worlds <= 0 or not 2 <= bodies <= 32:
        raise ValueError("SAT requires positive worlds and 2..32 bodies per world")
    if tuple(inverse_mass.shape) != (worlds, bodies):
        raise ValueError("SAT inverse_mass shape mismatch")
    if tuple(half_extents.shape) != (worlds, bodies, 3) or inverse_inertia.shape != half_extents.shape:
        raise ValueError("SAT half_extents and inverse_inertia must have shape [worlds,bodies,3]")
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2 or not 1 <= pair_indices.shape[0] <= 64:
        raise ValueError("SAT pair_indices must have shape [pairs,2] with 1..64 pairs")
    _validate_sat_config(config)
    _validate_sat_pair_indices(pair_indices, bodies)
    return load_extension().sat_step(
        state,
        inverse_mass,
        half_extents,
        inverse_inertia,
        pair_indices,
        float(config.dt),
        int(config.substeps),
        float(config.gravity_y),
        float(config.restitution),
        float(config.friction),
        float(config.position_slop),
        float(config.position_correction),
        float(config.angular_damping),
        int(config.solver_iterations),
        float(config.sat_epsilon),
    )


def manifold_step(
    state,
    inverse_mass,
    half_extents,
    inverse_inertia,
    pair_indices,
    cache_feature_ids,
    cache_impulses,
    config,
):
    """Advance fixed pairs with persistent clipped manifolds and warm starting."""

    import torch

    tensors = (
        state, inverse_mass, half_extents, inverse_inertia, pair_indices,
        cache_feature_ids, cache_impulses,
    )
    if not all(isinstance(item, torch.Tensor) for item in tensors):
        raise TypeError("manifold inputs must be torch tensors")
    if not all(item.is_cuda for item in tensors):
        raise ValueError("manifold inputs must all be CUDA tensors")
    if len({item.device for item in tensors}) != 1:
        raise ValueError("manifold inputs must be on the same CUDA device")
    float_tensors = (state, inverse_mass, half_extents, inverse_inertia, cache_impulses)
    if any(item.dtype != torch.float32 for item in float_tensors):
        raise ValueError("manifold state, material, and impulse tensors must be float32")
    if pair_indices.dtype != torch.int64 or cache_feature_ids.dtype != torch.int64:
        raise ValueError("manifold pair indices and feature IDs must be int64")
    if state.ndim != 3 or state.shape[2] != 13:
        raise ValueError("manifold state must have shape [worlds,bodies,13]")
    worlds, bodies = state.shape[:2]
    if worlds <= 0 or not 2 <= bodies <= 32:
        raise ValueError("manifold requires positive worlds and 2..32 bodies per world")
    if tuple(inverse_mass.shape) != (worlds, bodies):
        raise ValueError("manifold inverse_mass shape mismatch")
    if tuple(half_extents.shape) != (worlds, bodies, 3) or inverse_inertia.shape != half_extents.shape:
        raise ValueError("manifold half_extents and inverse_inertia must have shape [worlds,bodies,3]")
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2 or not 1 <= pair_indices.shape[0] <= 16:
        raise ValueError("manifold pair_indices must have shape [1..16,2]")
    pairs = pair_indices.shape[0]
    if tuple(cache_feature_ids.shape) != (worlds, pairs, 4):
        raise ValueError("cache_feature_ids must have shape [worlds,pairs,4]")
    if tuple(cache_impulses.shape) != (worlds, pairs, 4, 3):
        raise ValueError("cache_impulses must have shape [worlds,pairs,4,3]")
    _validate_sat_config(config)
    _validate_sat_pair_indices(pair_indices, bodies)
    return load_extension().manifold_step(
        state,
        inverse_mass,
        half_extents,
        inverse_inertia,
        pair_indices,
        cache_feature_ids,
        cache_impulses,
        float(config.dt),
        int(config.substeps),
        float(config.gravity_y),
        float(config.restitution),
        float(config.friction),
        float(config.position_slop),
        float(config.position_correction),
        float(config.angular_damping),
        int(config.solver_iterations),
        float(config.sat_epsilon),
    )


def joint_step(
    state,
    inverse_mass,
    inverse_inertia,
    joint_indices,
    joint_types,
    parent_anchor_local,
    child_anchor_local,
    axis_parent,
    reference_quaternion_parent_to_child,
    lower_limit,
    upper_limit,
    damping,
    motor_enabled,
    motor_target_velocity,
    maximum_effort,
    config,
    *,
    motor_target_position=None,
    stiffness=None,
    warm_start_cache=None,
):
    """Advance fixed-small articulated worlds with explicit joint tensors."""

    import torch

    if motor_target_position is None:
        motor_target_position = torch.zeros_like(motor_target_velocity)
    if stiffness is None:
        stiffness = torch.zeros_like(lower_limit)
    if warm_start_cache is None:
        warm_start_cache = torch.zeros(
            (state.shape[0], joint_indices.shape[0], 8),
            dtype=torch.float32,
            device=state.device,
        )
    tensors = (
        state, inverse_mass, inverse_inertia, joint_indices, joint_types,
        parent_anchor_local, child_anchor_local, axis_parent,
        reference_quaternion_parent_to_child, lower_limit, upper_limit,
        damping, motor_enabled, motor_target_velocity, motor_target_position,
        stiffness, maximum_effort, warm_start_cache,
    )
    if not all(isinstance(item, torch.Tensor) for item in tensors):
        raise TypeError("joint inputs must all be torch tensors")
    if not all(item.is_cuda for item in tensors):
        raise ValueError("joint inputs must all be CUDA tensors")
    if len({item.device for item in tensors}) != 1:
        raise ValueError("joint inputs must be on the same CUDA device")
    float_tensors = (
        state, inverse_mass, inverse_inertia, parent_anchor_local,
        child_anchor_local, axis_parent, reference_quaternion_parent_to_child,
        lower_limit, upper_limit, damping, motor_target_velocity,
        motor_target_position, stiffness, maximum_effort, warm_start_cache,
    )
    if any(item.dtype != torch.float32 for item in float_tensors):
        raise ValueError("joint state, topology, and control tensors must be float32")
    if joint_indices.dtype != torch.int64 or joint_types.dtype != torch.int64:
        raise ValueError("joint indices and types must be int64")
    if motor_enabled.dtype != torch.uint8:
        raise ValueError("motor_enabled must be uint8")
    if state.ndim != 3 or state.shape[2] != 13:
        raise ValueError("joint state must have shape [worlds,bodies,13]")
    worlds, bodies = state.shape[:2]
    if worlds <= 0 or not 2 <= bodies <= 32:
        raise ValueError("joint worlds require 2..32 bodies")
    if tuple(inverse_mass.shape) != (worlds, bodies):
        raise ValueError("joint inverse_mass shape mismatch")
    if tuple(inverse_inertia.shape) != (worlds, bodies, 3):
        raise ValueError("joint inverse_inertia must have shape [worlds,bodies,3]")
    if joint_indices.ndim != 2 or joint_indices.shape[1] != 2 or not 1 <= joint_indices.shape[0] <= 16:
        raise ValueError("joint_indices must have shape [1..16,2]")
    joints = joint_indices.shape[0]
    vector_rows = (parent_anchor_local, child_anchor_local, axis_parent)
    if any(tuple(item.shape) != (joints, 3) for item in vector_rows):
        raise ValueError("joint anchors and axes must have shape [joints,3]")
    if tuple(reference_quaternion_parent_to_child.shape) != (joints, 4):
        raise ValueError("joint reference quaternions must have shape [joints,4]")
    if any(tuple(item.shape) != (joints,) for item in (joint_types, lower_limit, upper_limit, damping, motor_enabled)):
        raise ValueError("joint scalar topology arrays must have shape [joints]")
    if tuple(motor_target_velocity.shape) != (worlds, joints) or tuple(motor_target_position.shape) != (worlds, joints) or tuple(maximum_effort.shape) != (worlds, joints):
        raise ValueError("joint controls must have shape [worlds,joints]")
    if tuple(stiffness.shape) != (joints,):
        raise ValueError("joint stiffness must have shape [joints]")
    if tuple(warm_start_cache.shape) != (worlds, joints, 8):
        raise ValueError("joint warm_start_cache must have shape [worlds,joints,8]")
    _validate_joint_config(config)
    _validate_sat_pair_indices(joint_indices, bodies)
    return load_extension().joint_step(
        state, inverse_mass, inverse_inertia, joint_indices, joint_types,
        parent_anchor_local, child_anchor_local, axis_parent,
        reference_quaternion_parent_to_child, lower_limit, upper_limit,
        damping, motor_enabled, motor_target_velocity, motor_target_position,
        stiffness, maximum_effort, warm_start_cache,
        float(config.warm_start_factor),
        float(config.dt), int(config.substeps), float(config.gravity_y),
        int(config.solver_iterations), float(config.position_correction),
        float(config.position_slop), float(config.angular_slop),
        float(config.maximum_linear_repair_m),
        float(config.maximum_angular_repair_rad),
    )
