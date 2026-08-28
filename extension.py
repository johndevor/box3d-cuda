"""Lazy build/load boundary for the CUDA implementation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_extension():
    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("Box3D CUDA requires a visible CUDA device")
    root = Path(__file__).resolve().parent
    return load(
        name="factory_box3d_cuda_v1",
        sources=[
            str(root / "csrc" / "bindings.cpp"),
            str(root / "csrc" / "step.cu"),
            str(root / "csrc" / "gripper.cu"),
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
