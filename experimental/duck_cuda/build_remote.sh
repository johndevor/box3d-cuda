#!/usr/bin/env bash
# Remote (CUDA machine) build for the batched duck lane. Requires CUDA 12.8+
# for sm_120 (RTX 5090); the same commands work on a 4090 via the sm_89
# fallback baked into every binary. Not runnable on the development Mac.
#
# Produces:
#   build-remote/libduck_cuda.so          parity build (fmad off)
#   build-remote/libduck_cuda_fast.so     throughput build (fmad on)
#   build-remote/libduck_cuda_serial.so   host serial reference
#
# Point walk/env/cuda_lane.py at the result:
#   DUCK_CUDA_LIBRARY=$PWD/build-remote/libduck_cuda.so python ...
#
# The CUDA library is WARP-PER-ENV (32 lanes cooperate on one env; best at
# E=8192-16384). Add -DDW_WARP_LANES=1 to the nvcc lines (or
# -DDUCK_CUDA_THREAD_PER_ENV=ON to cmake) for the legacy thread-per-env
# layout. After any rebuild run tests/remote_gpu_parity.py -- its windowed
# tolerance gates + bitwise on-device determinism are the cross-build
# contract for the lane-cooperative reductions.
set -euo pipefail
cd "$(dirname "$0")"

# --- Option A: cmake ---------------------------------------------------------
cmake -B build-remote -DCMAKE_BUILD_TYPE=Release -DDUCK_CUDA=ON
cmake --build build-remote -j

# --- Option B: raw nvcc (equivalent, no cmake needed) -------------------------
mkdir -p build-remote
nvcc -std=c++17 -O3 --fmad=false \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_120,code=sm_120 \
    -Iinclude -Xcompiler -fPIC -shared \
    src/duck_cuda.cu -o build-remote/libduck_cuda.so
nvcc -std=c++17 -O3 --fmad=true \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_120,code=sm_120 \
    -Iinclude -Xcompiler -fPIC -shared \
    src/duck_cuda.cu -o build-remote/libduck_cuda_fast.so
g++ -std=c++17 -O2 -Wall -Wextra -ffp-contract=off \
    -Iinclude -fPIC -shared \
    src/duck_cuda_serial.cpp -o build-remote/libduck_cuda_serial.so

echo "built: $(ls build-remote)"
