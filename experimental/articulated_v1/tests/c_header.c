#include "articulated_v1.h"
#include "box3d_cuda/experimental_joint_v1.h"
#include "box3d_cuda_machine_coupling_v1.h"
#include <stddef.h>
_Static_assert(AV1_ABI != BOX3D_JOINT_V1_ABI, "separate articulated ABI");
_Static_assert(AV1_ABI != BOX3D_CUDA_ABI_VERSION_V2, "no default r3 activation");
_Static_assert(sizeof(av1_body)==32,"body ABI");
_Static_assert(sizeof(av1_hinge)==152,"hinge ABI");
_Static_assert(sizeof(av1_model)==104,"model ABI");
_Static_assert(sizeof(av1_snapshot)==56,"snapshot ABI");
_Static_assert(sizeof(box3d_joint_v1_params)==136,"frozen joint ABI");
_Static_assert(sizeof(box3d_cuda_scene_register_desc_v2)==336,"frozen r3 ABI");
_Static_assert(sizeof(box3d_cuda_scene_wrench_step_desc_v1)==184,"frozen machine ABI");
int main(void){return 0;}
