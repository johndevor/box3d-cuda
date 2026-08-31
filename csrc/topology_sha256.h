#ifndef BOX3D_CUDA_TOPOLOGY_SHA256_H_
#define BOX3D_CUDA_TOPOLOGY_SHA256_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "../proposals/box3d_cuda_v2.h"

namespace box3d_cuda_native {

class Sha256 final {
 public:
  Sha256() = default;

  void update(const void* source, size_t byte_count) {
    const auto* bytes = static_cast<const uint8_t*>(source);
    total_bytes_ += byte_count;
    while (byte_count != 0) {
      const size_t available = sizeof(block_) - block_size_;
      const size_t copied = byte_count < available ? byte_count : available;
      std::memcpy(block_ + block_size_, bytes, copied);
      block_size_ += copied;
      bytes += copied;
      byte_count -= copied;
      if (block_size_ == sizeof(block_)) {
        transform(block_);
        block_size_ = 0;
      }
    }
  }

  void finish(uint8_t output[32]) {
    const uint64_t message_bits = total_bytes_ * UINT64_C(8);
    block_[block_size_++] = 0x80;
    if (block_size_ > 56) {
      while (block_size_ < sizeof(block_)) block_[block_size_++] = 0;
      transform(block_);
      block_size_ = 0;
    }
    while (block_size_ < 56) block_[block_size_++] = 0;
    for (int shift = 56; shift >= 0; shift -= 8) {
      block_[block_size_++] = static_cast<uint8_t>(message_bits >> shift);
    }
    transform(block_);
    for (size_t index = 0; index < 8; ++index) {
      output[index * 4 + 0] = static_cast<uint8_t>(state_[index] >> 24);
      output[index * 4 + 1] = static_cast<uint8_t>(state_[index] >> 16);
      output[index * 4 + 2] = static_cast<uint8_t>(state_[index] >> 8);
      output[index * 4 + 3] = static_cast<uint8_t>(state_[index]);
    }
  }

 private:
  static uint32_t rotate_right(uint32_t value, uint32_t amount) {
    return (value >> amount) | (value << (32 - amount));
  }

  void transform(const uint8_t block[64]) {
    static constexpr uint32_t constants[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };
    uint32_t words[64] = {};
    for (size_t index = 0; index < 16; ++index) {
      words[index] = (static_cast<uint32_t>(block[index * 4 + 0]) << 24) |
                     (static_cast<uint32_t>(block[index * 4 + 1]) << 16) |
                     (static_cast<uint32_t>(block[index * 4 + 2]) << 8) |
                     static_cast<uint32_t>(block[index * 4 + 3]);
    }
    for (size_t index = 16; index < 64; ++index) {
      const uint32_t s0 = rotate_right(words[index - 15], 7) ^
                          rotate_right(words[index - 15], 18) ^
                          (words[index - 15] >> 3);
      const uint32_t s1 = rotate_right(words[index - 2], 17) ^
                          rotate_right(words[index - 2], 19) ^
                          (words[index - 2] >> 10);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    uint32_t a = state_[0];
    uint32_t b = state_[1];
    uint32_t c = state_[2];
    uint32_t d = state_[3];
    uint32_t e = state_[4];
    uint32_t f = state_[5];
    uint32_t g = state_[6];
    uint32_t h = state_[7];
    for (size_t index = 0; index < 64; ++index) {
      const uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                            rotate_right(e, 25);
      const uint32_t choose = (e & f) ^ ((~e) & g);
      const uint32_t temporary1 = h + sum1 + choose + constants[index] + words[index];
      const uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                            rotate_right(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temporary2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  uint32_t state_[8] = {
      0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
  };
  uint8_t block_[64] = {};
  size_t block_size_ = 0;
  uint64_t total_bytes_ = 0;
};

class CanonicalTopologyWriter final {
 public:
  void bytes(const uint8_t* values, uint64_t count) {
    u64(count);
    hash_.update(values, static_cast<size_t>(count));
  }

  void boolean(bool value) {
    const uint8_t encoded = value ? 1u : 0u;
    hash_.update(&encoded, sizeof(encoded));
  }

  void u32(uint32_t value) {
    const uint8_t encoded[4] = {
        static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8),
        static_cast<uint8_t>(value >> 16), static_cast<uint8_t>(value >> 24)};
    hash_.update(encoded, sizeof(encoded));
  }

  void u64(uint64_t value) {
    uint8_t encoded[8];
    for (size_t index = 0; index < sizeof(encoded); ++index) {
      encoded[index] = static_cast<uint8_t>(value >> (index * 8));
    }
    hash_.update(encoded, sizeof(encoded));
  }

  bool f32(float value) {
    if (!std::isfinite(value)) return false;
    uint32_t bits = 0;
    if (value != 0.0f) std::memcpy(&bits, &value, sizeof(bits));
    u32(bits);
    return true;
  }

  void u32s(const uint32_t* values, uint64_t count) {
    u64(count);
    for (uint64_t index = 0; index < count; ++index) u32(values[index]);
  }

  bool f32s(const float* values, uint64_t count) {
    u64(count);
    for (uint64_t index = 0; index < count; ++index) {
      if (!f32(values[index])) return false;
    }
    return true;
  }

  void finish(uint8_t output[32]) { hash_.finish(output); }

 private:
  Sha256 hash_;
};

inline bool compute_topology_sha256(
    const box3d_cuda_scene_register_desc_v2& descriptor,
    uint8_t output[BOX3D_CUDA_TOPOLOGY_HASH_BYTES_V2]) {
  static constexpr uint8_t domain[] =
      "world.box3d-cuda.native-topology/v2\0";
  CanonicalTopologyWriter writer;
  writer.bytes(domain, sizeof(domain) - 1);
  writer.u32(descriptor.abi_version);
  writer.u32(BOX3D_CUDA_ABI_V2_DRAFT_REVISION);
  writer.u32(descriptor.environments);
  writer.u32(descriptor.bodies);
  writer.u32(descriptor.joints);
  writer.u32(descriptor.contact_pairs);
  writer.u32(descriptor.substeps);
  writer.u32(descriptor.solver_iterations);
  writer.u32(descriptor.material_binding);
  writer.boolean(descriptor.environment_gravity_xyz != nullptr);
  if (!writer.f32(descriptor.dt)) return false;
  const float solver_parameters[9] = {
      descriptor.warm_start_factor,
      descriptor.contact_slop,
      descriptor.position_correction,
      descriptor.angular_damping,
      descriptor.sat_epsilon,
      descriptor.joint_position_slop,
      descriptor.joint_angular_slop,
      descriptor.maximum_linear_repair,
      descriptor.maximum_angular_repair,
  };
  for (float value : solver_parameters) {
    if (!writer.f32(value)) return false;
  }

  writer.u32s(descriptor.body_caller_ids, descriptor.bodies);
  writer.u32s(descriptor.body_motion, descriptor.bodies);
  writer.u32s(descriptor.joint_caller_ids, descriptor.joints);
  writer.u32s(descriptor.joint_body_indices,
              static_cast<uint64_t>(descriptor.joints) * 2);
  writer.u32s(descriptor.joint_types, descriptor.joints);
  if (!writer.f32s(descriptor.joint_parent_anchor,
                   static_cast<uint64_t>(descriptor.joints) * 3) ||
      !writer.f32s(descriptor.joint_child_anchor,
                   static_cast<uint64_t>(descriptor.joints) * 3) ||
      !writer.f32s(descriptor.joint_axis_parent,
                   static_cast<uint64_t>(descriptor.joints) * 3) ||
      !writer.f32s(descriptor.joint_reference_xyzw,
                   static_cast<uint64_t>(descriptor.joints) * 4) ||
      !writer.f32s(descriptor.joint_lower_limit, descriptor.joints) ||
      !writer.f32s(descriptor.joint_upper_limit, descriptor.joints) ||
      !writer.f32s(descriptor.joint_damping, descriptor.joints) ||
      !writer.f32s(descriptor.joint_stiffness, descriptor.joints)) {
    return false;
  }
  writer.u32s(descriptor.joint_control_mode, descriptor.joints);
  writer.u32s(descriptor.contact_pair_caller_ids, descriptor.contact_pairs);
  writer.u32s(descriptor.contact_body_indices,
              static_cast<uint64_t>(descriptor.contact_pairs) * 2);
  writer.finish(output);
  return true;
}

}  // namespace box3d_cuda_native

#endif
