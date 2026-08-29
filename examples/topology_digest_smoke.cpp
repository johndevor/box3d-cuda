#include "../csrc/topology_sha256.h"

#include <cstdio>
#include <cstring>

int main() {
  {
    box3d_cuda_native::Sha256 sha;
    static constexpr char abc[] = "abc";
    static constexpr uint8_t abc_expected[32] = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
    };
    uint8_t abc_actual[32] = {};
    sha.update(abc, 3);
    sha.finish(abc_actual);
    if (std::memcmp(abc_actual, abc_expected, sizeof(abc_expected)) != 0) {
      std::fprintf(stderr, "native SHA-256 primitive failed known vector\n");
      return 3;
    }
  }
  const uint32_t body_ids[] = {0, 1};
  const uint32_t body_motion[] = {BOX3D_CUDA_BODY_FIXED_V2,
                                  BOX3D_CUDA_BODY_DYNAMIC_V2};
  const uint32_t pair_ids[] = {0};
  const uint32_t pair_bodies[] = {0, 1};
  box3d_cuda_scene_register_desc_v2 descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = BOX3D_CUDA_ABI_VERSION_V2;
  descriptor.environments = 8;
  descriptor.bodies = 2;
  descriptor.contact_pairs = 1;
  descriptor.substeps = 2;
  descriptor.solver_iterations = 12;
  descriptor.material_binding = BOX3D_CUDA_MATERIAL_GLOBAL_V2;
  descriptor.dt = 1.0f / 60.0f;
  descriptor.warm_start_factor = 0.8f;
  descriptor.contact_slop = 1.0e-4f;
  descriptor.position_correction = 0.8f;
  descriptor.angular_damping = 0.02f;
  descriptor.sat_epsilon = 1.0e-7f;
  descriptor.joint_position_slop = 1.0e-5f;
  descriptor.joint_angular_slop = 1.0e-5f;
  descriptor.maximum_linear_repair = 0.1f;
  descriptor.maximum_angular_repair = 0.2f;
  descriptor.body_caller_ids = body_ids;
  descriptor.body_motion = body_motion;
  descriptor.contact_pair_caller_ids = pair_ids;
  descriptor.contact_body_indices = pair_bodies;

  static constexpr uint8_t expected[32] = {
      0xa9, 0x72, 0xd5, 0xb1, 0x3f, 0x43, 0x18, 0x33,
      0x06, 0xb9, 0xfe, 0x4f, 0x5b, 0x27, 0xd2, 0x2f,
      0x4e, 0x3c, 0x9e, 0xe5, 0x18, 0xd4, 0xed, 0xd9,
      0x98, 0xf3, 0x6d, 0x81, 0x09, 0xe7, 0xdc, 0xa4,
  };
  uint8_t actual[32] = {};
  if (!box3d_cuda_native::compute_topology_sha256(descriptor, actual)) {
    std::fprintf(stderr, "topology encoder rejected the golden descriptor\n");
    return 1;
  }
  if (std::memcmp(actual, expected, sizeof(expected)) != 0) {
    std::fprintf(stderr, "native topology digest does not match World golden: ");
    for (uint8_t byte : actual) std::fprintf(stderr, "%02x", byte);
    std::fputc('\n', stderr);
    return 2;
  }
  std::puts("Box3D CUDA ABI-v2 r3 topology golden passed");
  return 0;
}
