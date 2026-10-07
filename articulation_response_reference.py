"""Dependency-free reduced-articulation contact-response micro oracle.

This module does not change the production Stage-7 solver.  It isolates the
specific structural question left by the matched PhysX comparison: how a
contact impulse applied to the end link of a fixed-base two-link planar arm is
distributed through the articulated mass matrix instead of treating that link
as a free rigid body.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, Tuple


Vec2 = Tuple[float, float]
Mat2 = Tuple[Tuple[float, float], Tuple[float, float]]
ARTICULATION_RESPONSE_WIDTH = 9
ARTICULATION_RESPONSE_FIELDS = (
    "articulated_inverse_effective_mass",
    "free_link_inverse_effective_mass",
    "other_inverse_effective_mass",
    "articulated_normal_impulse",
    "free_link_normal_impulse",
    "joint_velocity_delta_0",
    "joint_velocity_delta_1",
    "mass_matrix_determinant",
    "impulse_scale_vs_free_link",
)


def _vec2(value: Sequence[float], name: str) -> Vec2:
    if len(value) != 2:
        raise ValueError(f"{name} must contain two values")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _cross_z_point(point: Vec2, origin: Vec2) -> Vec2:
    """Return ``z_axis cross (point - origin)``."""

    return (-(point[1] - origin[1]), point[0] - origin[0])


def _dot(left: Vec2, right: Vec2) -> float:
    return left[0] * right[0] + left[1] * right[1]


@dataclass(frozen=True)
class PlanarTwoLinkContactResponse:
    mass_matrix: Mat2
    contact_jacobian: Vec2
    articulated_inverse_effective_mass: float
    free_link_inverse_effective_mass: float
    other_inverse_effective_mass: float
    articulated_normal_impulse: float
    free_link_normal_impulse: float
    articulated_joint_velocity_delta: Vec2

    def packed(self) -> Tuple[float, ...]:
        determinant = (
            self.mass_matrix[0][0] * self.mass_matrix[1][1]
            - self.mass_matrix[0][1] * self.mass_matrix[1][0]
        )
        return (
            self.articulated_inverse_effective_mass,
            self.free_link_inverse_effective_mass,
            self.other_inverse_effective_mass,
            self.articulated_normal_impulse,
            self.free_link_normal_impulse,
            *self.articulated_joint_velocity_delta,
            determinant,
            self.impulse_scale_vs_free_link,
        )

    @property
    def impulse_scale_vs_free_link(self) -> float:
        if self.free_link_normal_impulse == 0.0:
            return 1.0
        return self.articulated_normal_impulse / self.free_link_normal_impulse


def planar_two_link_contact_response(
    *,
    base_joint_xy: Sequence[float],
    second_joint_xy: Sequence[float],
    link1_center_xy: Sequence[float],
    link2_center_xy: Sequence[float],
    contact_point_xy: Sequence[float],
    normal_xy: Sequence[float],
    link1_mass: float,
    link2_mass: float,
    link1_inertia_z: float,
    link2_inertia_z: float,
    other_inverse_effective_mass: float,
    relative_normal_velocity: float,
    restitution: float = 0.0,
) -> PlanarTwoLinkContactResponse:
    """Solve one frictionless normal-impact micro in generalized coordinates.

    ``normal_xy`` points from the articulated link toward the other body.
    ``relative_normal_velocity`` is ``(v_other - v_link) dot normal`` and is
    therefore negative for an approaching contact.  The other body's inverse
    effective mass already includes any rotational term at its contact point.
    """

    base = _vec2(base_joint_xy, "base_joint_xy")
    second = _vec2(second_joint_xy, "second_joint_xy")
    center1 = _vec2(link1_center_xy, "link1_center_xy")
    center2 = _vec2(link2_center_xy, "link2_center_xy")
    contact = _vec2(contact_point_xy, "contact_point_xy")
    normal = _vec2(normal_xy, "normal_xy")
    normal_length = math.hypot(*normal)
    if abs(normal_length - 1.0) > 2.0e-5:
        raise ValueError("normal_xy must be normalized")

    mass1 = _positive(link1_mass, "link1_mass")
    mass2 = _positive(link2_mass, "link2_mass")
    inertia1 = _positive(link1_inertia_z, "link1_inertia_z")
    inertia2 = _positive(link2_inertia_z, "link2_inertia_z")
    other_inverse = float(other_inverse_effective_mass)
    speed = float(relative_normal_velocity)
    restitution = float(restitution)
    if not math.isfinite(other_inverse) or other_inverse < 0.0:
        raise ValueError("other_inverse_effective_mass must be finite and non-negative")
    if not math.isfinite(speed):
        raise ValueError("relative_normal_velocity must be finite")
    if not math.isfinite(restitution) or not 0.0 <= restitution <= 1.0:
        raise ValueError("restitution must be in [0, 1]")

    link1_jacobian = _cross_z_point(center1, base)
    link2_first_jacobian = _cross_z_point(center2, base)
    link2_second_jacobian = _cross_z_point(center2, second)
    m00 = (
        mass1 * _dot(link1_jacobian, link1_jacobian)
        + inertia1
        + mass2 * _dot(link2_first_jacobian, link2_first_jacobian)
        + inertia2
    )
    m01 = mass2 * _dot(link2_first_jacobian, link2_second_jacobian) + inertia2
    m11 = mass2 * _dot(link2_second_jacobian, link2_second_jacobian) + inertia2
    determinant = m00 * m11 - m01 * m01
    if determinant <= 1.0e-12:
        raise ValueError("articulated mass matrix is singular")

    contact_first_jacobian = _cross_z_point(contact, base)
    contact_second_jacobian = _cross_z_point(contact, second)
    jacobian = (
        _dot(normal, contact_first_jacobian),
        _dot(normal, contact_second_jacobian),
    )
    inverse_times_jacobian = (
        (m11 * jacobian[0] - m01 * jacobian[1]) / determinant,
        (-m01 * jacobian[0] + m00 * jacobian[1]) / determinant,
    )
    articulated_inverse = max(0.0, _dot(jacobian, inverse_times_jacobian))

    contact_offset = (contact[0] - center2[0], contact[1] - center2[1])
    angular_jacobian = contact_offset[0] * normal[1] - contact_offset[1] * normal[0]
    free_inverse = 1.0 / mass2 + angular_jacobian * angular_jacobian / inertia2

    numerator = max(0.0, -(1.0 + restitution) * speed)
    articulated_denominator = articulated_inverse + other_inverse
    free_denominator = free_inverse + other_inverse
    articulated_impulse = (
        numerator / articulated_denominator if articulated_denominator > 1.0e-12 else 0.0
    )
    free_impulse = numerator / free_denominator if free_denominator > 1.0e-12 else 0.0
    # The impulse on the articulated side is opposite the outward normal.
    joint_velocity_delta = (
        -inverse_times_jacobian[0] * articulated_impulse,
        -inverse_times_jacobian[1] * articulated_impulse,
    )

    return PlanarTwoLinkContactResponse(
        mass_matrix=((m00, m01), (m01, m11)),
        contact_jacobian=jacobian,
        articulated_inverse_effective_mass=articulated_inverse,
        free_link_inverse_effective_mass=free_inverse,
        other_inverse_effective_mass=other_inverse,
        articulated_normal_impulse=articulated_impulse,
        free_link_normal_impulse=free_impulse,
        articulated_joint_velocity_delta=joint_velocity_delta,
    )


@dataclass(frozen=True)
class PlanarTwoLinkPositionProjection:
    correction_distance: float
    pseudo_impulse: float
    articulated_joint_position_delta: Vec2
    articulated_point_displacement: float
    other_point_displacement: float


def planar_two_link_position_projection(
    *,
    penetration: float,
    position_slop: float,
    position_correction: float,
    **response_arguments,
) -> PlanarTwoLinkPositionProjection:
    """Project penetration through the same reduced contact Jacobian."""

    depth = float(penetration)
    slop = float(position_slop)
    factor = float(position_correction)
    if not all(math.isfinite(value) for value in (depth, slop, factor)):
        raise ValueError("position projection values must be finite")
    if depth < 0.0 or slop < 0.0 or not 0.0 <= factor <= 1.0:
        raise ValueError(
            "position projection requires nonnegative depth/slop and correction in [0,1]"
        )
    correction = min(0.2, max(0.0, depth - slop) * factor)
    response = planar_two_link_contact_response(
        relative_normal_velocity=-correction,
        restitution=0.0,
        **response_arguments,
    )
    articulated_displacement = (
        response.articulated_inverse_effective_mass
        * response.articulated_normal_impulse
    )
    other_displacement = (
        response.other_inverse_effective_mass * response.articulated_normal_impulse
    )
    return PlanarTwoLinkPositionProjection(
        correction_distance=correction,
        pseudo_impulse=response.articulated_normal_impulse,
        articulated_joint_position_delta=response.articulated_joint_velocity_delta,
        articulated_point_displacement=articulated_displacement,
        other_point_displacement=other_displacement,
    )


__all__ = [
    "ARTICULATION_RESPONSE_FIELDS",
    "ARTICULATION_RESPONSE_WIDTH",
    "PlanarTwoLinkContactResponse",
    "PlanarTwoLinkPositionProjection",
    "planar_two_link_contact_response",
    "planar_two_link_position_projection",
]
