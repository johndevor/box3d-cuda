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


def test_impact_iteration_sweep_uses_production_observables_and_projection() -> None:
    source = (ROOT / "sweep_impact_iterations.py").read_text()

    assert "articulation_projection=True" in source
    assert "_drive_effort_proxy(bundle, result[1], target)" in source
    assert '"drive_efforts_nm": effort_proxy' in source
    assert '"contact_generation_distance_m"' in source


def test_projected_path_solves_velocity_before_pose_integration() -> None:
    source = (ROOT / "csrc" / "coupled.cu").read_text()

    gravity = source.index("value[7]+=gravity_x*h")
    solve = source.index("solve_joint_rows(", gravity)
    projected_pose = source.index("if(articulation_projection)for(int body", solve)
    repair = source.index("for(int repair_iteration=0", projected_pose)
    assert gravity < solve < projected_pose < repair
    assert "if(!articulation_projection)" in source[gravity:solve]


def test_projected_path_distributes_split_repair_through_articulation() -> None:
    source = (ROOT / "csrc" / "coupled.cu").read_text()

    assert "repair_contact_with_articulation_projection" in source
    assert "apply_projected_pair_pose_impulse" in source
    assert "apply_articulation_pose_delta" in source
    assert "if(articulation_projection)repair_contact_with_articulation_projection" in source
    assert "else repair_contact_with_articulation_shock" in source


def test_stage7_repair_targets_rest_offset_not_contact_classification_slop() -> None:
    benchmark = (ROOT / "benchmark_coupled.py").read_text()
    contract = (ROOT / "contracts" / "coupling.py").read_text()

    assert "position_slop=SPEC.contact_rest_offset_m" in benchmark
    assert '"position_repair_slop_m": SPEC.contact_rest_offset_m' in contract


def test_speculative_generation_is_global_but_native_v2_remains_frozen() -> None:
    cuda = (ROOT / "csrc" / "coupled.cu").read_text()
    native = (ROOT / "csrc" / "native_scene_v2.cu").read_text()

    assert "coupled_contact::speculative_manifold" in cuda
    assert "contact_actual[MAX_CONTACT_PAIRS]" in cuda
    assert "if(contact_actual[pair])contact_ever" in cuda
    assert "contact_generation_distance" in cuda
    assert "contact_generation_distance */ 0.0f" in native


def test_speculative_generation_is_global_but_native_v2_remains_frozen() -> None:
    cuda = (ROOT / "csrc" / "coupled.cu").read_text()
    native = (ROOT / "csrc" / "native_scene_v2.cu").read_text()

    assert "coupled_contact::speculative_manifold" in cuda
    assert "contact_actual[MAX_CONTACT_PAIRS]" in cuda
    assert "if(contact_actual[pair])contact_ever" in cuda
    assert "contact_generation_distance" in cuda
    assert "contact_generation_distance */ 0.0f" in native


def test_coupled_kernel_has_a_torch_independent_native_boundary() -> None:
    source = (ROOT / "csrc" / "coupled.cu").read_text()
    manifold = (ROOT / "csrc" / "manifold.cu").read_text()
    joint = (ROOT / "csrc" / "joint.cu").read_text()

    assert source.count("#ifndef BOX3D_CUDA_NATIVE_KERNELS_ONLY") == 2
    assert "__global__ void coupled_kernel" in source
    assert manifold.index("#ifndef BOX3D_DEVICE_HELPERS_ONLY") < manifold.index("#include <torch/extension.h>")
    assert joint.index("#ifndef BOX3D_DEVICE_HELPERS_ONLY") < joint.index("#include <torch/extension.h>")
