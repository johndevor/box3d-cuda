import pytest
from dataclasses import replace

from box3d_cuda.articulation_oracle import PlanarTwoLink
from box3d_cuda.contracts.coupling import SPEC


INITIAL_Q = (0.7400602698326111, -1.1593574285507202)


def test_stage7_two_link_mass_matrix_and_gravity_acceleration_are_frozen():
    oracle = PlanarTwoLink.stage7()
    matrix = oracle.mass_matrix(INITIAL_Q)
    acceleration = oracle.acceleration(INITIAL_Q, (0.0, 0.0))
    assert matrix[0][0] == pytest.approx(1.4978216584742798)
    assert matrix[0][1] == pytest.approx(0.3077774959038067)
    assert matrix[1][1] == pytest.approx(0.1818)
    assert acceleration == pytest.approx((-10.11796792395, -5.049573462525227))


def test_stage7_two_substep_gravity_golden_is_deterministic():
    oracle = PlanarTwoLink.stage7()
    expected = (
        (0.7395332198841659, -1.1596202347656062),
        (-0.08433378794372962, -0.04203360214547393),
    )
    first = oracle.step(INITIAL_Q, (0.0, 0.0))
    second = oracle.step(INITIAL_Q, (0.0, 0.0))
    assert first[0] == pytest.approx(expected[0])
    assert first[1] == pytest.approx(expected[1])
    assert second == first


def test_stage7_two_substep_drive_only_golden_is_deterministic():
    oracle = replace(PlanarTwoLink.stage7(), gravity_mps2=0.0)
    target = (INITIAL_Q[0] + 0.08, INITIAL_Q[1] - 0.06)
    expected = (
        (0.7403364503656445, -1.1602330402693655),
        (0.043506675742280565, -0.13720874330654673),
    )
    kwargs = {
        "stiffness_nm_per_rad": SPEC.drive_stiffness_nm_per_rad,
        "damping_nms_per_rad": SPEC.drive_damping_nms_per_rad,
        "effort_limit_nm": SPEC.drive_effort_limits_nm,
    }
    first = oracle.step_pd(INITIAL_Q, (0.0, 0.0), target, **kwargs)
    second = oracle.step_pd(INITIAL_Q, (0.0, 0.0), target, **kwargs)
    assert first[0] == pytest.approx(expected[0])
    assert first[1] == pytest.approx(expected[1])
    assert second == first


def test_stage7_pd_effort_limit_is_explicit_and_symmetric():
    torque = PlanarTwoLink.pd_torque(
        (0.0, 0.0),
        (0.0, 0.0),
        (100.0, -100.0),
        SPEC.drive_stiffness_nm_per_rad,
        SPEC.drive_damping_nms_per_rad,
        SPEC.drive_effort_limits_nm,
    )
    assert torque == SPEC.drive_effort_limits_nm[:1] + (-SPEC.drive_effort_limits_nm[1],)


@pytest.mark.parametrize(
    "field,value",
    [
        ("stiffness_nm_per_rad", (-1.0, 1.0)),
        ("damping_nms_per_rad", (1.0, -1.0)),
        ("effort_limit_nm", (1.0, -1.0)),
    ],
)
def test_stage7_pd_oracle_rejects_negative_controller_parameters(field, value):
    arguments = {
        "stiffness_nm_per_rad": (1.0, 1.0),
        "damping_nms_per_rad": (1.0, 1.0),
        "effort_limit_nm": (1.0, 1.0),
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        PlanarTwoLink.pd_torque(
            (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), **arguments
        )


@pytest.mark.parametrize(
    "q,qdot,torque",
    [((0.0,), (0.0, 0.0), (0.0, 0.0)), ((0.0, 0.0), (0.0,), (0.0, 0.0)), ((0.0, 0.0), (0.0, 0.0), (0.0,))],
)
def test_oracle_rejects_malformed_vectors(q, qdot, torque):
    with pytest.raises((ValueError, IndexError)):
        PlanarTwoLink.stage7().step(q, qdot, torque_nm=torque)
