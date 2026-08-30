from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_decomposition_is_bounded_contact_free_and_diagnostic_only() -> None:
    source = (ROOT / "benchmark_joint_dynamics_decomposition.py").read_text()

    assert 'CONTRACT_ID = "box3d.joint-dynamics-decomposition/v1"' in source
    assert "CONTROL_STEPS = 8" in source
    assert '"gravity_only"' in source
    assert '"drive_only"' in source
    assert '"combined"' in source
    assert "articulation_projection=True" in source
    assert '"contact_observed"' in source
    assert '"diagnostic_only": True' in source
    assert '"accepted_solver_change": None' in source


def test_decomposition_does_not_change_native_abi_v2() -> None:
    native = (ROOT / "proposals" / "box3d_cuda_v2.h").read_text()
    assert "BOX3D_CUDA_ABI_V2_DRAFT_REVISION 3u" in native
