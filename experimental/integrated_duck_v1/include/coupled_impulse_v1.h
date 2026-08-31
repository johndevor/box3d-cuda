// SPDX-License-Identifier: MIT
#ifndef BOX3D_COUPLED_IMPULSE_V1_H
#define BOX3D_COUPLED_IMPULSE_V1_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
enum { CIV1_OK=0,CIV1_INVALID=1,CIV1_NUMERIC=2,CIV1_NO_CONVERGENCE=3,CIV1_ALLOCATION=4 };
typedef struct civ1_contact { uint32_t first_row; double friction; } civ1_contact;
typedef struct civ1_problem {
 uint32_t dofs,rows,contacts,max_iterations; // N<=32, R<=384, C<=64; max<=16384
 double impulse_tolerance; // positive <=1e-5; scalar correction / scaled tangent KKT
 const double *mass,*smooth_velocity; // [N,N] SPD including armature, [N]
 const double *jacobian,*target,*regularizer,*lower,*upper,*warm; // [R,N], [R] each
 const civ1_contact* contact; // [C], strictly increasing nonoverlapping triples n,t1,t2
} civ1_problem;
typedef struct civ1_result {
 double *velocity,*impulse; // caller [N], [R]; staged, untouched on failure
 uint32_t iterations;
 double joint_residual,normal_residual,tangent_residual,momentum_residual;
} civ1_result;
/* Full generalized non-associated Coulomb solve. Each scalar joint row solves
 * G*v-target+R*lambda against its fixed interval. Contact n: lambda>=0;
 * tangents minimize their conditional 2D quadratic in disk ||t||<=mu*lambda_n.
 * No cone-radius derivative is added to the normal equation (no dilatancy).
 * Contact rows must have R=0, normal bounds [0,+inf], tangent bounds [-inf,+inf].
 * K=G*M^-1*G^T is allowed PSD/redundant; only M is Cholesky-factored.
 * Scalar residuals are stable projected impulse corrections; tangent residuals
 * certify conditional KKT using the MIN eigenvalue of its response block,
 * plus disk feasibility/complementarity, avoiding cancellation of impulses.
 * Row velocities retain
 * authored linear/angular units, no cross-unit dot-product convergence norm.
 * Caller supplies valid host capacities, nonoverlapping writable arrays and
 * serialization. No scene mutation,
 * pose integration, position clipping, external dependencies, or CUDA claims.
 */
int civ1_solve(const civ1_problem*,civ1_result*);
#ifdef __cplusplus
}
#endif
#endif
