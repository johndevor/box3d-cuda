"""Pinned KR240-class joint metadata for the CUDA articulated solver.

This importer deliberately exposes only the dynamics fields needed to build a
fixed-topology joint tensor.  It first runs the existing hash/provenance asset
resolver, then parses the verified URDF.  Successful parsing is not a claim
that the vendor meshes, contacts, or controller are runtime validated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Tuple
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class LinkDynamics:
    name: str
    mass_kg: float
    center_of_mass_m_z_up: Vec3
    inertia_diagonal_kg_m2: Vec3


@dataclass(frozen=True)
class JointDynamics:
    name: str
    kind: str
    parent_index: int
    child_index: int
    origin_m_z_up: Vec3
    origin_rpy_rad: Vec3
    axis_z_up: Vec3
    lower: float
    upper: float
    effort: float
    velocity: float
    damping: float
    friction: float


@dataclass(frozen=True)
class IndustrialJointModel:
    asset_id: str
    calibration_class: str
    coordinate_system: str
    links: Tuple[LinkDynamics, ...]
    joints: Tuple[JointDynamics, ...]
    collision_filter_pairs: Tuple[Tuple[int, int], ...]
    source_urdf_sha256: str
    runtime_validated: bool = False
    manufacturer_dynamics: bool = False


def _vec(text: str | None, *, default: Vec3 = (0.0, 0.0, 0.0)) -> Vec3:
    if text is None:
        return default
    values = tuple(float(value) for value in text.split())
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("URDF vectors must contain three finite numbers")
    return values  # type: ignore[return-value]


def _positive(value: str | None, label: str) -> float:
    parsed = float(value or "nan")
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return parsed


def load_industrial_joint_model(
    urdf_path: Path,
    *,
    asset_id: str,
    source_urdf_sha256: str,
    calibration_class: str = "bounded_engineering_approximation",
) -> IndustrialJointModel:
    """Parse the bounded seven-link/six-revolute URDF contract.

    Hash verification and asset resolution belong to the caller. This keeps
    the engine package independent of Factory OS while preserving the exact
    parser and dynamics behavior used by the original integration.
    """

    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    link_nodes = root.findall("link")
    links = []
    for node in link_nodes:
        inertial = node.find("inertial")
        if inertial is None:
            # The display-only flange has no independent dynamic state.
            continue
        mass_node = inertial.find("mass")
        inertia = inertial.find("inertia")
        if mass_node is None or inertia is None:
            raise ValueError(f"link {node.attrib['name']} has incomplete dynamics")
        diagonal = tuple(_positive(inertia.attrib.get(key), f"{node.attrib['name']} {key}") for key in ("ixx", "iyy", "izz"))
        inertial_origin = inertial.find("origin")
        links.append(LinkDynamics(
            name=node.attrib["name"],
            mass_kg=_positive(mass_node.attrib.get("value"), f"{node.attrib['name']} mass"),
            center_of_mass_m_z_up=_vec(None if inertial_origin is None else inertial_origin.attrib.get("xyz")),
            inertia_diagonal_kg_m2=diagonal,  # type: ignore[arg-type]
        ))
    dynamic_names = [link.name for link in links]
    dynamic_index = {name: index for index, name in enumerate(dynamic_names)}
    joints = []
    for node in root.findall("joint"):
        if node.attrib.get("type") != "revolute":
            continue
        parent = node.find("parent")
        child = node.find("child")
        origin = node.find("origin")
        axis = node.find("axis")
        limit = node.find("limit")
        dynamics = node.find("dynamics")
        if None in (parent, child, limit):
            raise ValueError(f"joint {node.attrib.get('name')} is incomplete")
        parent_name = parent.attrib["link"]  # type: ignore[union-attr]
        child_name = child.attrib["link"]  # type: ignore[union-attr]
        if parent_name not in dynamic_index or child_name not in dynamic_index:
            raise ValueError("moving joints must connect dynamic-model links")
        joints.append(JointDynamics(
            name=node.attrib["name"],
            kind="revolute",
            parent_index=dynamic_index[parent_name],
            child_index=dynamic_index[child_name],
            origin_m_z_up=_vec(None if origin is None else origin.attrib.get("xyz")),
            origin_rpy_rad=_vec(None if origin is None else origin.attrib.get("rpy")),
            axis_z_up=_vec(None if axis is None else axis.attrib.get("xyz"), default=(1.0, 0.0, 0.0)),
            lower=float(limit.attrib["lower"]),  # type: ignore[union-attr]
            upper=float(limit.attrib["upper"]),  # type: ignore[union-attr]
            effort=_positive(limit.attrib.get("effort"), "joint effort"),  # type: ignore[union-attr]
            velocity=_positive(limit.attrib.get("velocity"), "joint velocity"),  # type: ignore[union-attr]
            damping=max(0.0, float("0" if dynamics is None else dynamics.attrib.get("damping", "0"))),
            friction=max(0.0, float("0" if dynamics is None else dynamics.attrib.get("friction", "0"))),
        ))
    if len(links) != 7 or len(joints) != 6:
        raise ValueError("KR240 CUDA import requires seven inertial links and six revolute joints")
    if any(joint.lower >= joint.upper for joint in joints):
        raise ValueError("joint lower limits must be below upper limits")
    # Directly connected links are excluded from self-contact; every other
    # pair remains available to the collision pipeline.
    filtered = tuple(tuple(sorted((joint.parent_index, joint.child_index))) for joint in joints)
    return IndustrialJointModel(
        asset_id=asset_id,
        calibration_class=calibration_class,
        coordinate_system="URDF_Z_UP_RIGHT_HANDED",
        links=tuple(links),
        joints=tuple(joints),
        collision_filter_pairs=filtered,
        source_urdf_sha256=source_urdf_sha256,
    )


def load_kr240_joint_model(asset_root: Path | None = None) -> IndustrialJointModel:
    """Compatibility adapter for a Factory OS verified KR240 asset.

    Standalone consumers should resolve and verify their own URDF, then call
    :func:`load_industrial_joint_model`. Factory OS remains an optional,
    lazily imported adapter rather than an engine dependency.
    """

    try:
        from factory_os.environments.asset_resolver import (
            resolve_local_kr240_physics_assets,
        )
    except ImportError as error:
        raise RuntimeError(
            "Factory OS is not installed; use load_industrial_joint_model "
            "with an independently verified URDF"
        ) from error
    bindings = resolve_local_kr240_physics_assets(asset_root)
    binding = bindings["robot.floor"]
    parsed_uri = urlparse(binding.uri)
    return load_industrial_joint_model(
        Path(unquote(parsed_uri.path)),
        asset_id="generated.kr240r2900.physics-approx-v1",
        source_urdf_sha256=binding.sha256,
    )
