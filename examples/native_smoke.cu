#include "box3d_cuda/box3d_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
  constexpr uint32_t worlds = 2, bodies = 1;
  std::vector<float> host_state(worlds * bodies * BOX3D_CUDA_STATE_WIDTH_V1, 0.0f);
  std::vector<float> host_mass(worlds * bodies, 1.0f);
  std::vector<float> host_radius(worlds * bodies, 0.1f);
  for (uint32_t world = 0; world < worlds; ++world) {
    host_state[world * BOX3D_CUDA_STATE_WIDTH_V1 + 1] = 0.5f + 0.1f * world;
    host_state[world * BOX3D_CUDA_STATE_WIDTH_V1 + 6] = 1.0f;
  }
  float *state = nullptr, *mass = nullptr, *radius = nullptr;
  cudaMalloc(&state, host_state.size() * sizeof(float));
  cudaMalloc(&mass, host_mass.size() * sizeof(float));
  cudaMalloc(&radius, host_radius.size() * sizeof(float));
  cudaMemcpy(state, host_state.data(), host_state.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(mass, host_mass.data(), host_mass.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(radius, host_radius.data(), host_radius.size() * sizeof(float), cudaMemcpyHostToDevice);

  box3d_cuda_sphere_step_desc_v1 descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = BOX3D_CUDA_ABI_VERSION;
  descriptor.device_ordinal = -1;
  descriptor.worlds = worlds;
  descriptor.bodies = bodies;
  descriptor.substeps = 2;
  descriptor.dt = 1.0f / 120.0f;
  descriptor.gravity_y = -9.81f;
  descriptor.restitution = 0.1f;
  descriptor.friction = 0.6f;
  descriptor.state = state;
  descriptor.inverse_mass = mass;
  descriptor.radius = radius;
  for (int step = 0; step < 240; ++step) {
    if (box3d_cuda_sphere_step_v1(&descriptor) != BOX3D_CUDA_STATUS_SUCCESS) return 2;
  }
  cudaDeviceSynchronize();
  cudaMemcpy(host_state.data(), state, host_state.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(state); cudaFree(mass); cudaFree(radius);
  for (uint32_t world = 0; world < worlds; ++world) {
    const float y = host_state[world * BOX3D_CUDA_STATE_WIDTH_V1 + 1];
    if (!std::isfinite(y) || y < 0.0998f) return 3;
  }
  std::printf("box3d_cuda native ABI v%u.%u smoke passed\n",
              BOX3D_CUDA_ABI_VERSION_MAJOR, BOX3D_CUDA_ABI_VERSION_MINOR);
  return 0;
}
