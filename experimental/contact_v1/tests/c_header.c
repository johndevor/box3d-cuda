// SPDX-License-Identifier: MIT
#include "contact_v1.h"
_Static_assert(sizeof(bcv1_body)==68,"body f32 layout");
_Static_assert(sizeof(bcv1_shape)==416,"shape layout");
_Static_assert(sizeof(bcv1_point)==40,"point layout LP64");
_Static_assert(sizeof(bcv1_manifold)==200,"manifold layout LP64");
_Static_assert(sizeof(bcv1_registration)==56,"registration layout LP64");
int main(void) { return BCV1_VERSION!=1; }
