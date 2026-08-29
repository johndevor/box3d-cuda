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


def test_articulation_projection_is_explicit_and_opt_in() -> None:
    extension = (ROOT / "extension.py").read_text()
    benchmark = (ROOT / "benchmark_coupled.py").read_text()
    bindings = (ROOT / "csrc" / "bindings.cpp").read_text()
    cuda = (ROOT / "csrc" / "coupled.cu").read_text()

    assert "articulation_projection=False" in extension
    assert "requires exactly two revolute joints" in extension
    assert 'parser.add_argument("--articulation-projection",action="store_true")' in benchmark
    assert "bool articulation_projection" in bindings
    assert "build_two_revolute_articulation" in cuda
    assert "if(articulation_projection)solve_projected_normal" in cuda
    assert "if(articulation_projection)solve_projected_motors" in cuda


def test_coupled_kernel_has_a_torch_independent_native_boundary() -> None:
    source = (ROOT / "csrc" / "coupled.cu").read_text()

    assert source.count("#ifndef BOX3D_CUDA_NATIVE_KERNELS_ONLY") == 2
    assert "__global__ void coupled_kernel" in source
