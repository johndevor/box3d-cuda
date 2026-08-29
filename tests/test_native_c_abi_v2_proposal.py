from pathlib import Path
import shutil
import subprocess

import pytest

from box3d_cuda.topology_digest import (
    CanonicalTopologyEncoder,
    canonical_topology_sha256,
)


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
        "BOX3D_CUDA_ABI_V2_DRAFT_REVISION 3u",
        "BOX3D_CUDA_CAP_V2_PARTIAL_ENVIRONMENT_RESTORE",
        "uint8_t topology_sha256[BOX3D_CUDA_TOPOLOGY_HASH_BYTES_V2]",
        "uint8_t* contact_ever",
        "uint32_t* contact_count",
        "fixed joints require DISABLED",
        "box3d_cuda_scene_capture_desc_v2",
        "box3d_cuda_scene_restore_desc_v2",
        "const float* inverse_mass",
        "float* material_friction",
        "float* gravity_xyz",
        "const float* gravity_xyz",
        "const uint8_t* environment_mask",
    ):
        assert term in source


def _world_two_body_digest(draft_revision: int) -> bytes:
    return canonical_topology_sha256(
        abi_version=2 << 16,
        draft_revision=draft_revision,
        environments=8,
        bodies=2,
        joints=0,
        contact_pairs=1,
        substeps=2,
        solver_iterations=12,
        material_binding=0,
        uses_environment_gravity=False,
        dt=1.0 / 60.0,
        solver_parameters=(
            0.8,
            1.0e-4,
            0.8,
            0.02,
            1.0e-7,
            1.0e-5,
            1.0e-5,
            0.1,
            0.2,
        ),
        body_caller_ids=(0, 1),
        body_motion=(0, 2),
        joint_caller_ids=(),
        joint_body_indices=(),
        joint_types=(),
        joint_parent_anchor=(),
        joint_child_anchor=(),
        joint_axis_parent=(),
        joint_reference_xyzw=(),
        joint_lower_limit=(),
        joint_upper_limit=(),
        joint_damping=(),
        joint_stiffness=(),
        joint_control_mode=(),
        contact_pair_caller_ids=(0,),
        contact_body_indices=(0, 1),
    )


def test_canonical_topology_sha256_matches_world_cross_repo_golden() -> None:
    assert _world_two_body_digest(1).hex() == (
        "06ab2f776dbf99c4dbf1a72220e6e56947fd24c1baeaa2d8068e837decaae0c4"
    )
    assert _world_two_body_digest(2).hex() == (
        "d664dfee5af110dc61e9ab8c3cf8568fdca1c23d501260e2c47d96f15271b34c"
    )
    assert _world_two_body_digest(3).hex() == (
        "a972d5b13f43183306b9fe4f5b27d22f4e3c9ee518d4edd998f36d8109e7dca4"
    )


def test_canonical_topology_rejects_nonfinite_and_normalizes_negative_zero() -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_topology_sha256(
            abi_version=2 << 16,
            draft_revision=2,
            environments=1,
            bodies=0,
            joints=0,
            contact_pairs=0,
            substeps=1,
            solver_iterations=1,
            material_binding=0,
            uses_environment_gravity=False,
            dt=float("nan"),
            solver_parameters=(0.0,) * 9,
            body_caller_ids=(),
            body_motion=(),
            joint_caller_ids=(),
            joint_body_indices=(),
            joint_types=(),
            joint_parent_anchor=(),
            joint_child_anchor=(),
            joint_axis_parent=(),
            joint_reference_xyzw=(),
            joint_lower_limit=(),
            joint_upper_limit=(),
            joint_damping=(),
            joint_stiffness=(),
            joint_control_mode=(),
            contact_pair_caller_ids=(),
            contact_body_indices=(),
        )

    positive = CanonicalTopologyEncoder()
    positive.f32(0.0)
    negative = CanonicalTopologyEncoder()
    negative.f32(-0.0)
    assert positive.digest() == negative.digest()


def test_v2_proposal_compiles_as_c11(tmp_path: Path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        return
    source = tmp_path / "proposal.c"
    source.write_text(
        "#include <stddef.h>\n"
        '#include "box3d_cuda_v2.h"\n'
        "_Static_assert(BOX3D_CUDA_STATE_WIDTH_V2 == 13u, \"state width\");\n"
        "_Static_assert(BOX3D_CUDA_JOINT_CACHE_WIDTH_V2 == 8u, \"cache width\");\n"
        "_Static_assert(sizeof(box3d_cuda_scene_handle_v2) == 8u, \"handle\");\n"
        "_Static_assert(sizeof(box3d_cuda_api_info_v2) == 128u, \"api layout\");\n"
        "_Static_assert(sizeof(box3d_cuda_scene_register_desc_v2) == 336u, "
        '"register layout");\n'
        "_Static_assert(sizeof(box3d_cuda_scene_info_v2) == 88u, \"info layout\");\n"
        "_Static_assert(sizeof(box3d_cuda_scene_step_desc_v2) == 160u, \"step layout\");\n"
        "_Static_assert(sizeof(box3d_cuda_scene_capture_desc_v2) == 136u, "
        '"capture layout");\n'
        "_Static_assert(sizeof(box3d_cuda_scene_restore_desc_v2) == 144u, "
        '"restore layout");\n'
        "_Static_assert(sizeof(box3d_cuda_ray_query_desc_v2) == 80u, \"ray layout\");\n"
        "_Static_assert(offsetof(box3d_cuda_scene_step_desc_v2, contact_ever) "
        '== 120u, "contact ever offset");\n'
        "_Static_assert(offsetof(box3d_cuda_scene_capture_desc_v2, stream) "
        '== 128u, "capture stream offset");\n'
        "_Static_assert(offsetof(box3d_cuda_scene_restore_desc_v2, environment_mask) "
        '== 128u, "restore mask offset");\n'
        "_Static_assert(offsetof(box3d_cuda_scene_restore_desc_v2, stream) "
        '== 136u, "restore stream offset");\n'
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
