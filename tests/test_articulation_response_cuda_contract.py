from pathlib import Path

from box3d_cuda.articulation_response_reference import ARTICULATION_RESPONSE_WIDTH


ROOT = Path(__file__).resolve().parents[1]


def test_cuda_micro_is_isolated_from_production_coupled_solver() -> None:
    coupled = (ROOT / "csrc" / "coupled.cu").read_text()
    micro = (ROOT / "csrc" / "articulation_response.cu").read_text()

    assert "articulation_response(" not in coupled
    assert "not called by the production coupled solver" in micro


def test_cuda_response_width_matches_cpu_contract() -> None:
    source = (ROOT / "csrc" / "articulation_response.cu").read_text()

    assert f"RESPONSE_WIDTH = {ARTICULATION_RESPONSE_WIDTH}" in source
    assert "articulation_response_kernel" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK" in source


def test_benchmark_requires_oracle_parity_and_determinism() -> None:
    source = (ROOT / "benchmark_articulation_response.py").read_text()

    assert "maximum_error <= MAXIMUM_ORACLE_ABSOLUTE_ERROR" in source
    assert "torch.equal(first, second)" in source
    assert '"production_solver_modified": False' in source
