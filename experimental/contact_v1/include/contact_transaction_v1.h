// SPDX-License-Identifier: MIT
#ifndef BOX3D_CONTACT_TRANSACTION_V1_H
#define BOX3D_CONTACT_TRANSACTION_V1_H
#include "contact_v1.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Additive experimental host transaction; only in the combined candidate.
 * No body integration or solve occurs in prepare. bodies are the complete
 * articulated POST state; cache is the solved PRE manifold/impulses.
 * Callers serialize BOTH owners for prepare/validate/commit/reset/capture.
 * A stale generation or wrong owner rejects; commit swaps preallocated state
 * and cannot allocate/fail after successful validation under that lock.
 * Destroy stages before scene. Old r3/machine and sealed contact unchanged.
 */
typedef struct bcx1_stage bcx1_stage;
enum { BCX1_STALE=6 };
bcv1_status bcx1_prepare_solved(const bcv1_scene*, const bcv1_body* post_bodies,
 const bcv1_manifold* solved_pre_cache, double dt, bcx1_stage**);
bcv1_status bcx1_prepare_restore(const bcv1_scene*, const bcv1_snapshot*,
 const uint8_t* mask, bcx1_stage**);
bcv1_status bcx1_stage_read(const bcx1_stage*, bcv1_body*, bcv1_manifold*, double*);
bcv1_status bcx1_stage_query(const bcx1_stage*, bcv1_manifold*);
int bcx1_validate_commit(const bcv1_scene*, const bcx1_stage*);
int bcx1_commit(bcv1_scene*, bcx1_stage*);
void bcx1_stage_destroy(bcx1_stage*);
#ifdef __cplusplus
}
#endif
#endif
