// SPDX-License-Identifier: MIT
// Versioned C ABI over the exact Stage-0 kernel used by the Python binding.
#include "box3d_cuda/box3d_cuda.h"

#include <cuda_runtime.h>

#define BOX3D_CUDA_NATIVE_KERNELS_ONLY 1
#include "step.cu"

extern "C" uint32_t box3d_cuda_get_abi_version(void) {
  return BOX3D_CUDA_ABI_VERSION;
}

extern "C" uint64_t box3d_cuda_get_capabilities_v1(void) {
  return BOX3D_CUDA_CAPABILITY_SPHERE_STEP |
         BOX3D_CUDA_CAPABILITY_STATIC_PLANE_CONTACTS |
         BOX3D_CUDA_CAPABILITY_SPHERE_PAIR_CONTACTS;
}

extern "C" const char* box3d_cuda_status_string_v1(box3d_cuda_status_v1 status) {
  switch (status) {
    case BOX3D_CUDA_STATUS_SUCCESS: return "success";
    case BOX3D_CUDA_STATUS_INVALID_ARGUMENT: return "invalid argument";
    case BOX3D_CUDA_STATUS_ABI_MISMATCH: return "ABI mismatch";
    case BOX3D_CUDA_STATUS_CUDA_ERROR: return "CUDA error";
    default: return "unknown status";
  }
}

extern "C" box3d_cuda_status_v1 box3d_cuda_sphere_step_v1(
    const box3d_cuda_sphere_step_desc_v1* descriptor) {
  if (descriptor == nullptr ||
      descriptor->struct_size != sizeof(box3d_cuda_sphere_step_desc_v1) ||
      descriptor->abi_version != BOX3D_CUDA_ABI_VERSION) {
    return descriptor != nullptr && descriptor->abi_version != BOX3D_CUDA_ABI_VERSION
        ? BOX3D_CUDA_STATUS_ABI_MISMATCH
        : BOX3D_CUDA_STATUS_INVALID_ARGUMENT;
  }
  if (descriptor->worlds == 0 || descriptor->bodies == 0 ||
      descriptor->substeps == 0 || descriptor->dt <= 0.0f ||
      descriptor->restitution < 0.0f || descriptor->friction < 0.0f ||
      descriptor->state == nullptr || descriptor->inverse_mass == nullptr ||
      descriptor->radius == nullptr) {
    return BOX3D_CUDA_STATUS_INVALID_ARGUMENT;
  }
  if (descriptor->device_ordinal >= 0 &&
      cudaSetDevice(descriptor->device_ordinal) != cudaSuccess) {
    return BOX3D_CUDA_STATUS_CUDA_ERROR;
  }
  constexpr int threads = 128;
  const int blocks = (static_cast<int>(descriptor->worlds) + threads - 1) / threads;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(descriptor->stream);
  step_worlds<<<blocks, threads, 0, stream>>>(
      descriptor->state, descriptor->inverse_mass, descriptor->radius,
      static_cast<int>(descriptor->worlds), static_cast<int>(descriptor->bodies),
      descriptor->dt, static_cast<int>(descriptor->substeps),
      descriptor->gravity_y, descriptor->restitution, descriptor->friction);
  return cudaPeekAtLastError() == cudaSuccess
      ? BOX3D_CUDA_STATUS_SUCCESS
      : BOX3D_CUDA_STATUS_CUDA_ERROR;
}
