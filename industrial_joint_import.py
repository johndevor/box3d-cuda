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


@dataclass(frozen=True)
class IndustrialJointWorld:
    """One zero-coordinate industrial arm compiled for the joint solver."""

    state_y_up: Tuple[Tuple[float, ...], ...]
    inverse_mass: Tuple[float, ...]
    inverse_inertia_local: Tuple[Vec3, ...]
    topology: object
    maximum_effort_nm: Tuple[float, ...]
    maximum_velocity_rad_s: Tuple[float, ...]
    link_frame_positions_y_up: Tuple[Vec3, ...]
    link_frame_quaternions_xyzw_y_up: Tuple[Tuple[float, float, float, float], ...]


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


def _nonnegative(value: str | None, label: str) -> float:
    parsed = float("0" if value is None else value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be non-negative and finite")
    return parsed


def _qmul(a, b):
    return (
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    )


def _qrotate(q, v):
    qv = (q[0], q[1], q[2])
    cross = (
        qv[1] * v[2] - qv[2] * v[1],
        qv[2] * v[0] - qv[0] * v[2],
        qv[0] * v[1] - qv[1] * v[0],
    )
    twice = tuple(2.0 * value for value in cross)
    again = (
        qv[1] * twice[2] - qv[2] * twice[1],
        qv[2] * twice[0] - qv[0] * twice[2],
        qv[0] * twice[1] - qv[1] * twice[0],
    )
    return tuple(v[index] + q[3] * twice[index] + again[index] for index in range(3))


def _rpy_quaternion(rpy: Vec3):
    roll, pitch, yaw = rpy
    qx = (math.sin(roll / 2.0), 0.0, 0.0, math.cos(roll / 2.0))
    qy = (0.0, math.sin(pitch / 2.0), 0.0, math.cos(pitch / 2.0))
    qz = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    return _qmul(_qmul(qz, qy), qx)


def compile_industrial_joint_world(model: IndustrialJointModel) -> IndustrialJointWorld:
    """Compile URDF zero pose into maximal-coordinate solver tensors.

    URDF link-local coordinates remain body-local coordinates. World poses are
    rotated from URDF z-up to the engine's canonical y-up frame by -90 degrees
    about +x. Body positions are inertial centers of mass, so both local anchor
    offsets account for each link's inertial origin.
    """

    from .joint_reference import JointTopology, REVOLUTE

    if len(model.links) != 7 or len(model.joints) != 6:
        raise ValueError("industrial joint world requires seven links and six joints")
    basis = (-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    link_position = [(0.0, 0.0, 0.0)] * len(model.links)
    link_quaternion = [(0.0, 0.0, 0.0, 1.0)] * len(model.links)
    for joint in model.joints:
        parent_p = link_position[joint.parent_index]
        parent_q = link_quaternion[joint.parent_index]
        origin_q = _rpy_quaternion(joint.origin_rpy_rad)
        translated = _qrotate(parent_q, joint.origin_m_z_up)
        link_position[joint.child_index] = tuple(parent_p[index] + translated[index] for index in range(3))
        link_quaternion[joint.child_index] = _qmul(parent_q, origin_q)
    world_position = [tuple(_qrotate(basis, position)) for position in link_position]
    world_quaternion = [_qmul(basis, quaternion) for quaternion in link_quaternion]
    state = []
    for link, frame_p, frame_q in zip(model.links, world_position, world_quaternion):
        center = _qrotate(frame_q, link.center_of_mass_m_z_up)
        position = tuple(frame_p[index] + center[index] for index in range(3))
        state.append((*position, *frame_q, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    parent_anchors = []
    child_anchors = []
    axes = []
    references = []
    for joint in model.joints:
        parent_com = model.links[joint.parent_index].center_of_mass_m_z_up
        child_com = model.links[joint.child_index].center_of_mass_m_z_up
        parent_anchors.append(tuple(joint.origin_m_z_up[index] - parent_com[index] for index in range(3)))
        child_anchors.append(tuple(-value for value in child_com))
        origin_q = _rpy_quaternion(joint.origin_rpy_rad)
        axis = _qrotate(origin_q, joint.axis_z_up)
        length = math.sqrt(sum(value * value for value in axis))
        if length <= 1.0e-12:
            raise ValueError(f"joint {joint.name} axis has zero length")
        axes.append(tuple(value / length for value in axis))
        references.append(origin_q)
    topology = JointTopology(
        joint_indices=tuple((joint.parent_index, joint.child_index) for joint in model.joints),
        joint_types=(REVOLUTE,) * len(model.joints),
        parent_anchor_local=tuple(parent_anchors),
        child_anchor_local=tuple(child_anchors),
        axis_parent=tuple(axes),
        reference_quaternion_parent_to_child=tuple(references),
        lower_limit=tuple(joint.lower for joint in model.joints),
        upper_limit=tuple(joint.upper for joint in model.joints),
        damping=tuple(joint.damping for joint in model.joints),
        motor_enabled=(True,) * len(model.joints),
        collision_enabled=(False,) * len(model.joints),
    )
    return IndustrialJointWorld(
        state_y_up=tuple(tuple(value for value in body) for body in state),
        inverse_mass=(0.0,) + tuple(1.0 / link.mass_kg for link in model.links[1:]),
        inverse_inertia_local=((0.0, 0.0, 0.0),) + tuple(
            tuple(1.0 / value for value in link.inertia_diagonal_kg_m2)
            for link in model.links[1:]
        ),
        topology=topology,
        maximum_effort_nm=tuple(joint.effort for joint in model.joints),
        maximum_velocity_rad_s=tuple(joint.velocity for joint in model.joints),
        link_frame_positions_y_up=tuple(world_position),
        link_frame_quaternions_xyzw_y_up=tuple(world_quaternion),
    )


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
        products = tuple(
            float(inertia.attrib.get(key, "0")) for key in ("ixy", "ixz", "iyz")
        )
        if not all(math.isfinite(value) and abs(value) <= 1.0e-12 for value in products):
            raise ValueError(
                f"link {node.attrib['name']} has unsupported off-diagonal inertia"
            )
        inertial_origin = inertial.find("origin")
        inertial_rpy = _vec(None if inertial_origin is None else inertial_origin.attrib.get("rpy"))
        if any(abs(value) > 1.0e-12 for value in inertial_rpy):
            raise ValueError(
                f"link {node.attrib['name']} has unsupported inertial-frame rotation"
            )
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
            damping=_nonnegative(None if dynamics is None else dynamics.attrib.get("damping"), "joint damping"),
            friction=_nonnegative(None if dynamics is None else dynamics.attrib.get("friction"), "joint friction"),
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
