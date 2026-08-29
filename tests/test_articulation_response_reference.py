import math

import pytest

from box3d_cuda.articulation_response_reference import (
    ARTICULATION_RESPONSE_FIELDS,
    ARTICULATION_RESPONSE_WIDTH,
    planar_two_link_contact_response,
)


def _benchmark_geometry():
    q1, q2 = 0.75, -1.15
    base = (-0.8, 0.14)
    center1 = (base[0] + 0.35 * math.cos(q1), base[1] + 0.35 * math.sin(q1))
    second = (center1[0] + 0.35 * math.cos(q1), center1[1] + 0.35 * math.sin(q1))
    center2 = (
        second[0] + 0.30 * math.cos(q1 + q2),
        second[1] + 0.30 * math.sin(q1 + q2),
    )
    normal = (1.0, 0.0)
    contact = (center2[0] + 0.30 * math.cos(q1 + q2), center2[1] + 0.30 * math.sin(q1 + q2))
    return base, second, center1, center2, contact, normal


def test_bent_fixed_base_arm_has_different_contact_effective_mass_than_free_link():
    base, second, center1, center2, contact, normal = _benchmark_geometry()
    response = planar_two_link_contact_response(
        base_joint_xy=base,
        second_joint_xy=second,
        link1_center_xy=center1,
        link2_center_xy=center2,
        contact_point_xy=contact,
        normal_xy=normal,
        link1_mass=2.0,
        link2_mass=1.5,
        link1_inertia_z=0.08406666666666666,
        link2_inertia_z=0.0468,
        other_inverse_effective_mass=1.0,
        relative_normal_velocity=-1.0,
    )

    assert response.articulated_inverse_effective_mass > 0.0
    assert response.articulated_inverse_effective_mass < response.free_link_inverse_effective_mass
    assert response.articulated_normal_impulse > response.free_link_normal_impulse
    assert response.impulse_scale_vs_free_link > 1.0
    assert response.mass_matrix[0][1] == response.mass_matrix[1][0]
    assert all(math.isfinite(value) for value in response.articulated_joint_velocity_delta)
    assert len(response.packed()) == ARTICULATION_RESPONSE_WIDTH == len(ARTICULATION_RESPONSE_FIELDS)


def test_separating_contact_produces_no_impulse():
    base, second, center1, center2, contact, normal = _benchmark_geometry()
    response = planar_two_link_contact_response(
        base_joint_xy=base,
        second_joint_xy=second,
        link1_center_xy=center1,
        link2_center_xy=center2,
        contact_point_xy=contact,
        normal_xy=normal,
        link1_mass=2.0,
        link2_mass=1.5,
        link1_inertia_z=0.08406666666666666,
        link2_inertia_z=0.0468,
        other_inverse_effective_mass=1.0,
        relative_normal_velocity=0.25,
    )

    assert response.articulated_normal_impulse == 0.0
    assert response.free_link_normal_impulse == 0.0
    assert response.articulated_joint_velocity_delta == (-0.0, -0.0)


@pytest.mark.parametrize(
    ("field", "value"),
    (("normal_xy", (2.0, 0.0)), ("link1_mass", 0.0), ("restitution", 1.1)),
)
def test_invalid_inputs_fail_closed(field, value):
    base, second, center1, center2, contact, normal = _benchmark_geometry()
    arguments = dict(
        base_joint_xy=base,
        second_joint_xy=second,
        link1_center_xy=center1,
        link2_center_xy=center2,
        contact_point_xy=contact,
        normal_xy=normal,
        link1_mass=2.0,
        link2_mass=1.5,
        link1_inertia_z=0.08406666666666666,
        link2_inertia_z=0.0468,
        other_inverse_effective_mass=1.0,
        relative_normal_velocity=-1.0,
        restitution=0.0,
    )
    arguments[field] = value

    with pytest.raises(ValueError):
        planar_two_link_contact_response(**arguments)
