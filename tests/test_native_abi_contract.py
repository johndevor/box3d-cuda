from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
HEADER = (ROOT / "include" / "box3d_cuda" / "box3d_cuda.h").read_text()
NATIVE = (ROOT / "csrc" / "native_api.cu").read_text()
STEP = (ROOT / "csrc" / "step.cu").read_text()


def test_c_header_is_versioned_and_does_not_depend_on_torch():
    assert "BOX3D_CUDA_ABI_VERSION_MAJOR 1u" in HEADER
    assert "box3d_cuda_sphere_step_v1" in HEADER
    assert "struct_size" in HEADER
    assert "abi_version" in HEADER
    assert "torch" not in HEADER.lower()


def test_native_entry_point_reuses_exact_stage0_kernel():
    assert '#include "step.cu"' in NATIVE
    assert "step_worlds<<<" in NATIVE
    assert "BOX3D_CUDA_NATIVE_KERNELS_ONLY" in STEP
    assert len(re.findall(r"__global__ void step_worlds", STEP)) == 1


def test_native_translation_unit_compile_checks_coupled_kernel_without_torch():
    assert "#define BOX3D_CUDA_NATIVE_KERNELS_ONLY 1" in NATIVE
    assert '#include "coupled.cu"' in NATIVE


def test_native_descriptor_is_device_pointer_and_stream_based():
    for field in ("float* state", "const float* inverse_mass", "const float* radius", "void* stream"):
        assert field in HEADER
