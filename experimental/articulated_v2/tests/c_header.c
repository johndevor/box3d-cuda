#include "articulated_v2.h"
#include "box3d_cuda/experimental_joint_v1.h"
#include "box3d_cuda_machine_coupling_v1.h"
_Static_assert(AV2_ABI!=AV1_ABI&&AV2_ABI!=BOX3D_JOINT_V1_ABI&&AV2_ABI!=BOX3D_CUDA_ABI_VERSION_V2,"separate ABI");
_Static_assert(sizeof(av2_limit)==88,"limit ABI");
_Static_assert(sizeof(av2_registration)==56,"registration ABI");
_Static_assert(sizeof(av2_step)==56,"step ABI");
_Static_assert(sizeof(av2_pre_view)==216,"PRE ABI");
_Static_assert(sizeof(av2_solution)==32,"solution ABI");
_Static_assert(sizeof(av2_snapshot)==64,"snapshot ABI");
_Static_assert(sizeof(av2_state_view)==80,"view ABI");
_Static_assert(sizeof(av1_model)==104&&sizeof(av1_snapshot)==56,"frozen AV1 ABI");
_Static_assert(sizeof(box3d_joint_v1_params)==136,"frozen joint ABI");
_Static_assert(sizeof(box3d_cuda_scene_register_desc_v2)==336,"frozen r3 ABI");
_Static_assert(sizeof(box3d_cuda_scene_wrench_step_desc_v1)==184,"frozen machine ABI");
int main(void){return 0;}
