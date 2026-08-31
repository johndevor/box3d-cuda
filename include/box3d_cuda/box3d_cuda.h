#ifndef BOX3D_CUDA_BOX3D_CUDA_H_
#define BOX3D_CUDA_BOX3D_CUDA_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(BOX3D_CUDA_BUILD_SHARED)
#    define BOX3D_CUDA_API __declspec(dllexport)
#  else
#    define BOX3D_CUDA_API __declspec(dllimport)
#  endif
#else
#  define BOX3D_CUDA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define BOX3D_CUDA_ABI_VERSION_MAJOR 1u
#define BOX3D_CUDA_ABI_VERSION_MINOR 0u
#define BOX3D_CUDA_ABI_VERSION ((BOX3D_CUDA_ABI_VERSION_MAJOR << 16u) | BOX3D_CUDA_ABI_VERSION_MINOR)
#define BOX3D_CUDA_STATE_WIDTH_V1 13u

typedef enum box3d_cuda_status_v1 {
  BOX3D_CUDA_STATUS_SUCCESS = 0,
  BOX3D_CUDA_STATUS_INVALID_ARGUMENT = 1,
  BOX3D_CUDA_STATUS_ABI_MISMATCH = 2,
  BOX3D_CUDA_STATUS_CUDA_ERROR = 3
} box3d_cuda_status_v1;

typedef enum box3d_cuda_capability_v1 {
  BOX3D_CUDA_CAPABILITY_SPHERE_STEP = UINT64_C(1) << 0,
  BOX3D_CUDA_CAPABILITY_STATIC_PLANE_CONTACTS = UINT64_C(1) << 1,
  BOX3D_CUDA_CAPABILITY_SPHERE_PAIR_CONTACTS = UINT64_C(1) << 2
} box3d_cuda_capability_v1;

/* All pointers are CUDA device pointers. state is updated in place. The
 * stream is a cudaStream_t represented as void*; NULL selects the default
 * stream. Arrays use [worlds, bodies, 13], [worlds, bodies], and
 * [worlds, bodies] contiguous row-major layouts respectively. */
typedef struct box3d_cuda_sphere_step_desc_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  int32_t device_ordinal;
  uint32_t worlds;
  uint32_t bodies;
  uint32_t substeps;
  float dt;
  float gravity_y;
  float restitution;
  float friction;
  float* state;
  const float* inverse_mass;
  const float* radius;
  void* stream;
} box3d_cuda_sphere_step_desc_v1;

BOX3D_CUDA_API uint32_t box3d_cuda_get_abi_version(void);
BOX3D_CUDA_API uint64_t box3d_cuda_get_capabilities_v1(void);
BOX3D_CUDA_API const char* box3d_cuda_status_string_v1(box3d_cuda_status_v1 status);
BOX3D_CUDA_API box3d_cuda_status_v1 box3d_cuda_sphere_step_v1(
    const box3d_cuda_sphere_step_desc_v1* descriptor);

#ifdef __cplusplus
}
#endif

#endif
