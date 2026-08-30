import pytest

from box3d_cuda.articulation_oracle import PlanarTwoLink


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


@pytest.mark.parametrize(
    "q,qdot,torque",
    [((0.0,), (0.0, 0.0), (0.0, 0.0)), ((0.0, 0.0), (0.0,), (0.0, 0.0)), ((0.0, 0.0), (0.0, 0.0), (0.0,))],
)
def test_oracle_rejects_malformed_vectors(q, qdot, torque):
    with pytest.raises((ValueError, IndexError)):
        PlanarTwoLink.stage7().step(q, qdot, torque_nm=torque)
