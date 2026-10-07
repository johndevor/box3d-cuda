from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "proposals" / "box3d_cuda_machine_coupling_v1.h"


def test_machine_coupling_header_is_additive_and_installed() -> None:
    base = (ROOT / "proposals" / "box3d_cuda_v2.h").read_text()
    extension = HEADER.read_text()
    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert "BOX3D_CUDA_CAP_V2_EXTERNAL_WRENCH_STEP" not in base
    assert "BOX3D_CUDA_CAP_V2_JOINT_VELOCITY_OUTPUT" not in base
    assert "UINT64_C(1) << 18" in extension
    assert "UINT64_C(1) << 19" in extension
    assert "box3d_cuda_scene_step_wrench_v1" in extension
    assert "proposals/box3d_cuda_machine_coupling_v1.h" in cmake
    query_smoke = (ROOT / "examples" / "installed_v2_query_smoke.c").read_text()
    assert "box3d_cuda_machine_coupling_v1.h" in query_smoke
    assert "BOX3D_CUDA_CAP_V2_EXTERNAL_WRENCH_STEP" in query_smoke
    assert "BOX3D_CUDA_CAP_V2_JOINT_VELOCITY_OUTPUT" in query_smoke


def test_machine_coupling_source_advertises_and_implements_both_capabilities() -> None:
    native = (ROOT / "csrc" / "native_scene_v2.cu").read_text()
    coupled = (ROOT / "csrc" / "coupled.cu").read_text()
    for term in (
        "BOX3D_CUDA_CAP_V2_EXTERNAL_WRENCH_STEP",
        "BOX3D_CUDA_CAP_V2_JOINT_VELOCITY_OUTPUT",
        "box3d_cuda_scene_step_wrench_v1",
        "gather_joint_velocity",
        "external_force_xyz",
        "external_torque_xyz",
    ):
        assert term in native
    assert "force[k]*inverse_mass[flat]*h" in coupled
    assert "coupled_joint::iworld" in coupled
    assert 'implementation_version_minor = 5' in native
    smoke = (ROOT / "examples" / "native_machine_coupling_v1_smoke.cu").read_text()
    assert "expected_dynamic[] = {0.25f, 0.5f, 0.75f, 0.5f, 2.0f, 4.5f}" in smoke
    assert "!near(q[0], 0.0f) || !near(qdot[0], 3.0f)" in smoke
    assert "!near(q[1], 2.0f) || !near(qdot[1], 4.0f)" in smoke


def test_machine_coupling_header_compiles_with_frozen_layout(tmp_path: Path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        return
    source = tmp_path / "machine_coupling.c"
    source.write_text(
        '#include "box3d_cuda_machine_coupling_v1.h"\n'
        "_Static_assert(sizeof(box3d_cuda_scene_step_desc_v2) == 160u, \"r3\");\n"
        "_Static_assert(sizeof(box3d_cuda_scene_wrench_step_desc_v1) == 184u, \"extension\");\n"
        "int main(void) { return 0; }\n",
        encoding="utf-8",
    )
    subprocess.run(
        [compiler, "-std=c11", "-Werror", "-I", str(HEADER.parent), "-c", str(source)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
