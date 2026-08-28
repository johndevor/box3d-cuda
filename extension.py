"""Lazy build/load boundary for the CUDA implementation."""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path


_SAT_PAIR_VALIDATION_CACHE: dict[int, tuple] = {}


@lru_cache(maxsize=1)
def load_extension():
    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("Box3D CUDA requires a visible CUDA device")
    root = Path(__file__).resolve().parent
    return load(
        name="factory_box3d_cuda_v7",
        sources=[
            str(root / "csrc" / "bindings.cpp"),
            str(root / "csrc" / "step.cu"),
            str(root / "csrc" / "gripper.cu"),
            str(root / "csrc" / "obb.cu"),
            str(root / "csrc" / "sat.cu"),
            str(root / "csrc" / "manifold.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=False,
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
