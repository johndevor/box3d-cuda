from pathlib import Path

from box3d_cuda.coupled_reference import POSITION_REPAIR_ITERATIONS


ROOT = Path(__file__).resolve().parents[1]


def test_position_repair_iteration_count_matches_cuda_source() -> None:
    source = (ROOT / "csrc" / "coupled.cu").read_text()

    assert f"POSITION_REPAIR_ITERATIONS = {POSITION_REPAIR_ITERATIONS}" in source
    assert "repair_iteration<POSITION_REPAIR_ITERATIONS" in source


def test_articulation_shock_rule_matches_cuda_source() -> None:
    source = (ROOT / "csrc" / "coupled.cu").read_text()

    assert "left_articulated&&right_inverse_mass>0.0f" in source
    assert "right_articulated&&left_inverse_mass>0.0f" in source
