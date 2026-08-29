#include <box3d_cuda/box3d_cuda_v2.h>

#include <stdint.h>
#include <stdio.h>

int main(void) {
  const uint64_t expected = BOX3D_CUDA_CAP_V2_ORIENTED_BOXES |
                            BOX3D_CUDA_CAP_V2_EXPLICIT_CONTACT_PAIRS |
                            BOX3D_CUDA_CAP_V2_FIXED_JOINTS |
                            BOX3D_CUDA_CAP_V2_REVOLUTE_JOINTS |
                            BOX3D_CUDA_CAP_V2_PRISMATIC_JOINTS |
                            BOX3D_CUDA_CAP_V2_PERSISTENT_CONTACTS |
                            BOX3D_CUDA_CAP_V2_RESIDENT_STATE |
                            BOX3D_CUDA_CAP_V2_DETERMINISTIC_SNAPSHOT |
                            BOX3D_CUDA_CAP_V2_ASYNC_CALLER_STREAM |
                            BOX3D_CUDA_CAP_V2_GLOBAL_MATERIAL |
                            BOX3D_CUDA_CAP_V2_PARTIAL_ENVIRONMENT_RESTORE;
  box3d_cuda_api_info_v2 info = {0};
  info.struct_size = sizeof(info);
  info.abi_version = BOX3D_CUDA_ABI_VERSION_V2;
  if (box3d_cuda_get_abi_version_v2() != BOX3D_CUDA_ABI_VERSION_V2 ||
      box3d_cuda_query_api_v2(&info) != BOX3D_CUDA_STATUS_V2_SUCCESS ||
      info.draft_revision != BOX3D_CUDA_ABI_V2_DRAFT_REVISION ||
      info.capabilities != expected) {
    fputs("installed ABI-v2 query failed\n", stderr);
    return 1;
  }
  printf("Box3D CUDA installed ABI-v2 r%u mask 0x%llx passed\n",
         info.draft_revision, (unsigned long long)info.capabilities);
  return 0;
}
