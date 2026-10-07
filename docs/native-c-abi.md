# Native C ABI

The public ABI starts at `BOX3D_CUDA_ABI_VERSION` 1.0 and is declared in
`include/box3d_cuda/box3d_cuda.h`. It has no PyTorch dependency. Build the
shared library and smoke executable with:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/box3d_cuda_native_smoke
```

ABI v1 deliberately exposes only the accepted Stage-0 sphere/infinite-plane
step. It consumes contiguous CUDA device arrays and launches asynchronously on
the supplied CUDA stream. The caller owns allocation, lifetime, stream
synchronization, and device selection. `state` is updated in place.

The descriptor contains its byte size and ABI version. Callers must populate
both. New optional entry points can be added without changing existing
symbols; an incompatible struct or semantic change requires a new suffixed
entry point and ABI major version.

The Python API remains the compatibility binding for the broader Stage 0-7
surface. Its lazy PyTorch extension and tensor validation are not part of the
C ABI.
