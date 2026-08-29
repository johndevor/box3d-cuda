"""Deterministic visualization exports for the accepted Stage 0-2 CPU oracles.

These traces are deliberately presentation artifacts.  They replay the exact
dependency-free reference functions that gate the CUDA kernels, but they are
not live GPU execution or additional benchmark evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .gripper_reference import (
    CONTRACT_ID as GRIPPER_CONTRACT,
    GripperWorldConfig,
    finger_velocity,
    make_gripper_state,
    step_gripper_reference,
)
from .obb_reference import (
    CONTRACT_ID as OBB_CONTRACT,
    OrientedBoxConfig,
    make_oriented_box_state,
    minimum_corner_clearance,
    step_oriented_box_reference,
)
from .reference import SphereWorldConfig, make_drop_state, step_reference


SPHERE_CONTRACT = "box3d.fixed-sphere-world/v0"
SCHEMA_VERSION = "box3d.cpu-oracle-visualization/v1"


def _body(body: list[float]) -> dict[str, list[float]]:
    return {
        "position_m": body[0:3],
        "quaternion_xyzw": body[3:7],
        "linear_velocity_mps": body[7:10],
        "angular_velocity_rad_s": body[10:13],
    }


def sphere_replay(*, steps: int = 500, sample_every: int = 4) -> dict[str, Any]:
    config = SphereWorldConfig()
    state, inverse_mass, radii = make_drop_state(1, 8, seed=7)
    frames = [{"step": 0, "bodies": [_body(body) for body in state[0]], "contact_active": [False] * 8}]
    contact_ever = [False] * 8
    for step in range(1, steps + 1):
        state = step_reference(state, inverse_mass, radii, config)
        active = [body[1] <= radii[0][index] + 2.5e-4 for index, body in enumerate(state[0])]
        contact_ever = [before or now for before, now in zip(contact_ever, active, strict=True)]
        if step % sample_every == 0 or step == steps:
            frames.append({"step": step, "bodies": [_body(body) for body in state[0]], "contact_active": active})
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": 0,
        "contract_id": SPHERE_CONTRACT,
        "control_hz": round(1.0 / config.dt),
        "physics_substeps": config.substeps,
        "sample_every_control_steps": sample_every,
        "radii_m": radii[0],
        "contact_ever": contact_ever,
        "frames": frames,
        "honest_boundary": "Offline deterministic CPU-oracle replay; not live CUDA or PhysX output.",
    }


def _gripper_phase(step: int, config: GripperWorldConfig) -> str:
    boundaries = [
        config.settle_steps,
        config.settle_steps + config.close_steps,
        config.settle_steps + config.close_steps + config.lift_steps,
        config.settle_steps + config.close_steps + config.lift_steps + config.open_steps,
    ]
    return ("settle" if step < boundaries[0] else "close" if step < boundaries[1]
            else "lift" if step < boundaries[2] else "release" if step < boundaries[3] else "fall")


def gripper_replay(*, sample_every: int = 2) -> dict[str, Any]:
    config = GripperWorldConfig()
    cube, fingers = make_gripper_state(1, config)
    frames = [{
        "step": 0,
        "phase": "settle",
        "cube": _body(cube[0]),
        "finger_positions_m": fingers[0],
        "contact_active": [False, False],
    }]
    contact_ever = [False, False]
    maximum_height = cube[0][1]
    for step in range(config.total_steps):
        cube, fingers, contacts = step_gripper_reference(
            cube, fingers, finger_velocity(step, config), config,
        )
        contact_ever = [before or now for before, now in zip(contact_ever, contacts[0], strict=True)]
        maximum_height = max(maximum_height, cube[0][1])
        completed = step + 1
        if completed % sample_every == 0 or completed == config.total_steps:
            frames.append({
                "step": completed,
                "phase": _gripper_phase(step, config),
                "cube": _body(cube[0]),
                "finger_positions_m": fingers[0],
                "contact_active": contacts[0],
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": 1,
        "contract_id": GRIPPER_CONTRACT,
        "control_hz": round(1.0 / config.dt),
        "physics_substeps": config.substeps,
        "sample_every_control_steps": sample_every,
        "cube_half_extents_m": list(config.cube_half_extent),
        "finger_half_extents_m": list(config.finger_half_extent),
        "contact_ever": contact_ever,
        "maximum_cube_height_m": maximum_height,
        "frames": frames,
        "honest_boundary": "Offline deterministic CPU-oracle replay of friction-only grasping; no weld, attachment, or pose copy.",
    }


def obb_plane_replay(*, steps: int = 500, sample_every: int = 4) -> dict[str, Any]:
    config = OrientedBoxConfig()
    state, inverse_mass, half_extents, inverse_inertia = make_oriented_box_state(1, 8, seed=17)
    frames = [{"step": 0, "bodies": [_body(body) for body in state[0]], "contact_active": [False] * 8}]
    contact_ever = [False] * 8
    minimum_clearance = min(minimum_corner_clearance(body, half_extents[0][index]) for index, body in enumerate(state[0]))
    for step in range(1, steps + 1):
        state, contacts, clearance = step_oriented_box_reference(
            state, inverse_mass, half_extents, inverse_inertia, config,
        )
        contact_ever = [before or now for before, now in zip(contact_ever, contacts[0], strict=True)]
        minimum_clearance = min(minimum_clearance, clearance)
        if step % sample_every == 0 or step == steps:
            frames.append({"step": step, "bodies": [_body(body) for body in state[0]], "contact_active": contacts[0]})
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": 2,
        "contract_id": OBB_CONTRACT,
        "control_hz": round(1.0 / config.dt),
        "physics_substeps": config.substeps,
        "sample_every_control_steps": sample_every,
        "half_extents_m": half_extents[0],
        "contact_ever": contact_ever,
        "minimum_transient_corner_clearance_m": minimum_clearance,
        "frames": frames,
        "honest_boundary": "Offline deterministic CPU-oracle replay of oriented boxes against a plane; box-box collision is excluded.",
    }


def write_replays(output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    payloads = [sphere_replay(), gripper_replay(), obb_plane_replay()]
    paths = []
    for payload in payloads:
        path = output / f"physics-stage{payload['stage']}-replay.json"
        path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for path in write_replays(args.output):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
