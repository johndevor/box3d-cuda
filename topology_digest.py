"""Canonical topology identity for the proposed resident-scene ABI v2.

The byte stream deliberately mirrors World's independently implemented Rust
encoder.  Mutable episode state is not part of this digest; snapshots carry it.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterable, Sequence


TOPOLOGY_DOMAIN = b"world.box3d-cuda.native-topology/v2\0"


class CanonicalTopologyEncoder:
    """Small, dependency-free little-endian SHA-256 stream encoder."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256()

    def bytes(self, values: bytes) -> None:
        self.u64(len(values))
        self._hash.update(values)

    def boolean(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("canonical boolean must be bool")
        self._hash.update(bytes((int(value),)))

    def u32(self, value: int) -> None:
        if not 0 <= value <= 0xFFFF_FFFF:
            raise ValueError("canonical u32 is out of range")
        self._hash.update(struct.pack("<I", value))

    def u64(self, value: int) -> None:
        if not 0 <= value <= 0xFFFF_FFFF_FFFF_FFFF:
            raise ValueError("canonical u64 is out of range")
        self._hash.update(struct.pack("<Q", value))

    def f32(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError("canonical f32 must be finite")
        # Both signs of zero have the one accepted encoding. Packing first also
        # performs the same IEEE-754 binary32 rounding as the native contract.
        bits = 0 if value == 0.0 else struct.unpack("<I", struct.pack("<f", value))[0]
        self.u32(bits)

    def u32s(self, values: Iterable[int]) -> None:
        packed = tuple(values)
        self.u64(len(packed))
        for value in packed:
            self.u32(value)

    def f32s(self, values: Iterable[float]) -> None:
        packed = tuple(values)
        self.u64(len(packed))
        for value in packed:
            self.f32(value)

    def digest(self) -> bytes:
        return self._hash.digest()


def canonical_topology_sha256(
    *,
    abi_version: int,
    draft_revision: int,
    environments: int,
    bodies: int,
    joints: int,
    contact_pairs: int,
    substeps: int,
    solver_iterations: int,
    material_binding: int,
    uses_environment_gravity: bool,
    dt: float,
    solver_parameters: Sequence[float],
    body_caller_ids: Sequence[int],
    body_motion: Sequence[int],
    joint_caller_ids: Sequence[int],
    joint_body_indices: Sequence[int],
    joint_types: Sequence[int],
    joint_parent_anchor: Sequence[float],
    joint_child_anchor: Sequence[float],
    joint_axis_parent: Sequence[float],
    joint_reference_xyzw: Sequence[float],
    joint_lower_limit: Sequence[float],
    joint_upper_limit: Sequence[float],
    joint_damping: Sequence[float],
    joint_stiffness: Sequence[float],
    joint_control_mode: Sequence[int],
    contact_pair_caller_ids: Sequence[int],
    contact_body_indices: Sequence[int],
) -> bytes:
    """Return the canonical 32-byte topology SHA-256.

    ``solver_parameters`` is the nine registration floats in header order:
    warm start, contact slop, position correction, angular damping, SAT
    epsilon, joint position slop, joint angular slop, maximum linear repair,
    and maximum angular repair.
    """

    if len(solver_parameters) != 9:
        raise ValueError("solver_parameters must have nine scalars")

    digest = CanonicalTopologyEncoder()
    digest.bytes(TOPOLOGY_DOMAIN)
    for value in (
        abi_version,
        draft_revision,
        environments,
        bodies,
        joints,
        contact_pairs,
        substeps,
        solver_iterations,
        material_binding,
    ):
        digest.u32(value)
    digest.boolean(uses_environment_gravity)
    digest.f32(dt)
    for value in solver_parameters:
        digest.f32(value)

    digest.u32s(body_caller_ids)
    digest.u32s(body_motion)
    digest.u32s(joint_caller_ids)
    digest.u32s(joint_body_indices)
    digest.u32s(joint_types)
    digest.f32s(joint_parent_anchor)
    digest.f32s(joint_child_anchor)
    digest.f32s(joint_axis_parent)
    digest.f32s(joint_reference_xyzw)
    digest.f32s(joint_lower_limit)
    digest.f32s(joint_upper_limit)
    digest.f32s(joint_damping)
    digest.f32s(joint_stiffness)
    digest.u32s(joint_control_mode)
    digest.u32s(contact_pair_caller_ids)
    digest.u32s(contact_body_indices)
    return digest.digest()
