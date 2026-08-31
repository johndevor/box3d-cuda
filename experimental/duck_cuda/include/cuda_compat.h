// SPDX-License-Identifier: MIT
// Host-only shim so duck_cuda_kernel.h compiles unchanged with plain
// clang++ (the serial parity build). The real CUDA driver (duck_cuda.cu)
// never includes this file; nvcc defines __CUDACC__ and DW_HD expands to
// __host__ __device__ there instead.
#ifndef DUCK_CUDA_COMPAT_H
#define DUCK_CUDA_COMPAT_H
#if defined(__CUDACC__)
#define DW_HD __host__ __device__
#else
#define DW_HD
#define __device__
#define __host__
#define __forceinline__ inline
#endif
#endif
