from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "proposals" / "box3d_cuda_v2.h"


def test_v2_proposal_is_not_an_installed_or_implemented_abi() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text()
    native = (ROOT / "csrc" / "native_api.cu").read_text()

    assert HEADER.is_file()
    assert "proposals" not in cmake
    assert "box3d_cuda_scene_register_v2" not in native


def test_v2_proposal_records_world_integration_semantics() -> None:
    source = HEADER.read_text()

    for term in (
        "BOX3D_CUDA_JOINT_FIXED_V2 = 0",
        "BOX3D_CUDA_JOINT_REVOLUTE_V2 = 1",
        "BOX3D_CUDA_JOINT_PRISMATIC_V2 = 2",
        "const int64_t* contact_feature_ids",
        "const float* state;",
        "float* contact_impulse_cache",
        "BOX3D_CUDA_CAP_V2_KINEMATIC_BODIES",
        "BOX3D_CUDA_CAP_V2_PER_PAIR_MATERIAL",
        "BOX3D_CUDA_CAP_V2_LINEAR_OBB_RAYS",
    ):
        assert term in source


def test_v2_proposal_compiles_as_c11(tmp_path: Path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        return
    source = tmp_path / "proposal.c"
    source.write_text(
        '#include "box3d_cuda_v2.h"\n'
        "_Static_assert(BOX3D_CUDA_STATE_WIDTH_V2 == 13u, \"state width\");\n"
        "_Static_assert(BOX3D_CUDA_JOINT_CACHE_WIDTH_V2 == 8u, \"cache width\");\n"
        "_Static_assert(sizeof(box3d_cuda_scene_handle_v2) == 8u, \"handle\");\n"
        "int main(void) { return BOX3D_CUDA_JOINT_PRISMATIC_V2 != 2; }\n",
        encoding="utf-8",
    )
    subprocess.run(
        [compiler, "-std=c11", "-Werror", "-I", str(HEADER.parent), "-c", str(source)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
