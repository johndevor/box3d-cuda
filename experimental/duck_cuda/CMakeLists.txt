cmake_minimum_required(VERSION 3.24)
project(duck_cuda LANGUAGES CXX)

# Serial parity build (always available; the local test vehicle).
add_library(duck_cuda_serial SHARED src/duck_cuda_serial.cpp)
target_include_directories(duck_cuda_serial PUBLIC include)
target_compile_features(duck_cuda_serial PUBLIC cxx_std_17)
target_compile_options(duck_cuda_serial PRIVATE
    -O2 -Wall -Wextra -ffp-contract=off)

# Real CUDA build (remote only; no nvcc on the development Mac).
#   -DDUCK_CUDA=ON            enable the CUDA target
#   -DDUCK_CUDA_FAST=ON       fmad on (throughput variant); OFF = parity build
option(DUCK_CUDA "Build the CUDA library (requires nvcc)" OFF)
option(DUCK_CUDA_FAST "Enable fmad (fast variant; parity builds keep it off)" OFF)
if(DUCK_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # sm_120: RTX 5090 (Blackwell); sm_89: RTX 4090 fallback.
  set(CMAKE_CUDA_ARCHITECTURES "89;120")
  add_library(duck_cuda SHARED src/duck_cuda.cu)
  target_include_directories(duck_cuda PUBLIC include)
  target_compile_options(duck_cuda PRIVATE -O3)
  if(DUCK_CUDA_FAST)
    target_compile_options(duck_cuda PRIVATE --fmad=true)
  else()
    # Parity variant: no fused multiply-add so device arithmetic matches the
    # serial clang++ build (-ffp-contract=off) term for term.
    target_compile_options(duck_cuda PRIVATE --fmad=false)
  endif()
endif()
