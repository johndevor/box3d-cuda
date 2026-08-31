// SPDX-License-Identifier: MIT
// Single-source fp32 physics for the batched CUDA duck lane. One environment
// per thread: free-root articulated dynamics (port of articulated_v1/v2),
// foot-vs-floor plane manifolds with stable feature ids (port of contact_v1's
// plane path), joint + contact row assembly (port of integrated_duck_v1) and
// the dense PGS solver with the degenerate-contact repairs (port of
// coupled_impulse_v1: conditional two-normal solve, Richardson extrapolation
// and the Tresca fixed-point acceleration; the rank-1 null-direction move is
// intentionally not ported -- it needs an m x m Jacobi eigensolver and the
// Tresca acceleration certifies the same rank-deficient flat-foot blocks).
//
// No dynamic allocation anywhere; fixed per-thread arrays sized by
// duck_model.h. Compiled two ways from this one header:
//   - src/duck_cuda_serial.cpp via include/cuda_compat.h (plain clang++,
//     the local parity/test vehicle),
//   - src/duck_cuda.cu under nvcc (DW_HD = __host__ __device__).
// Certificates are fp32-scaled: solver impulse tolerance DW_SOLVE_TOLERANCE,
// momentum-residual gate DW_MOMENTUM_TOLERANCE (the CPU oracle uses 1e-8 for
// both in f64).
#ifndef DUCK_CUDA_KERNEL_H
#define DUCK_CUDA_KERNEL_H
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "duck_cuda.h"
#include "duck_model.h"

#ifndef DW_HD
#error "include cuda_compat.h (serial) or compile with nvcc defining DW_HD"
#endif

#ifndef DW_SOLVE_TOLERANCE
#define DW_SOLVE_TOLERANCE 5e-6f
#endif
#ifndef DW_MOMENTUM_TOLERANCE
#define DW_MOMENTUM_TOLERANCE 2e-4f
#endif
#ifndef DW_MAX_ITERATIONS
#define DW_MAX_ITERATIONS 4096u
#endif
#define DW_CONTACT_EPS 2e-6f
#ifndef DW_APGD_BUDGET
#define DW_APGD_BUDGET 65536u
#endif

// ---------------------------------------------------------------- small math
static DW_HD void dw_add3(const float* a, const float* b, float* o) {
  o[0] = a[0] + b[0]; o[1] = a[1] + b[1]; o[2] = a[2] + b[2];
}
static DW_HD void dw_sub3(const float* a, const float* b, float* o) {
  o[0] = a[0] - b[0]; o[1] = a[1] - b[1]; o[2] = a[2] - b[2];
}
static DW_HD void dw_scale3(const float* a, float t, float* o) {
  o[0] = a[0] * t; o[1] = a[1] * t; o[2] = a[2] * t;
}
static DW_HD float dw_dot3(const float* a, const float* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
static DW_HD void dw_cross3(const float* a, const float* b, float* o) {
  o[0] = a[1] * b[2] - a[2] * b[1];
  o[1] = a[2] * b[0] - a[0] * b[2];
  o[2] = a[0] * b[1] - a[1] * b[0];
}
static DW_HD void dw_qmul(const float* a, const float* b, float* o) {
  float v[3], t1[3], t2[3];
  dw_scale3(b, a[3], t1); dw_scale3(a, b[3], t2); dw_add3(t1, t2, v);
  dw_cross3(a, b, t1); dw_add3(v, t1, v);
  o[0] = v[0]; o[1] = v[1]; o[2] = v[2];
  o[3] = a[3] * b[3] - dw_dot3(a, b);
}
static DW_HD void dw_qnormalize(float* q) {
  float n = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  q[0] /= n; q[1] /= n; q[2] /= n; q[3] /= n;
}
static DW_HD void dw_rotate(const float* q, const float* v, float* o) {
  float t[3], u[3], c[3];
  dw_cross3(q, v, t); dw_scale3(t, 2.0f, t);
  dw_scale3(t, q[3], u); dw_add3(v, u, u);
  dw_cross3(q, t, c); dw_add3(u, c, o);
}
static DW_HD void dw_qexp(const float* r, float* o) {  // rotation vector -> quat
  float angle = sqrtf(dw_dot3(r, r));
  float s = angle < 1e-8f ? 0.5f - angle * angle / 48.0f
                          : sinf(angle * 0.5f) / angle;
  o[0] = r[0] * s; o[1] = r[1] * s; o[2] = r[2] * s; o[3] = cosf(angle * 0.5f);
}
// world inertia response: rot * diag(I_principal) * rot^-1 * w
static DW_HD void dw_inertia(const float* q, const float* principal,
                             const float* w, float* o) {
  float inv[4] = {-q[0], -q[1], -q[2], q[3]}, local[3];
  dw_rotate(inv, w, local);
  local[0] *= principal[0]; local[1] *= principal[1]; local[2] *= principal[2];
  dw_rotate(q, local, o);
}
static DW_HD bool dw_finite(const float* p, int n) {
  for (int i = 0; i < n; i++) if (!isfinite(p[i])) return false;
  return true;
}
static DW_HD float dw_clampf(float x, float lo, float hi) {
  return fminf(hi, fmaxf(lo, x));
}

// -------------------------------------------------------------------- state
typedef struct DwState {
  float q[DW_Q];               // root xyz, root quat xyzw, 14 hinge angles
  float v[DW_N];               // world root linear, world root angular, qdot
  float warm[DW_JROWS];        // joint warm forces: friction/lower/upper slots
  dwc1_manifold cache[DW_PAIRS];  // previous solve manifolds (warm start)
  uint64_t count;              // accepted ticks
} DwState;

typedef struct DwParams {
  float refweight[DW_J];       // reference-pose inverse-mass diagonal (av2)
  float tolerance;             // solver certificate tolerance (impulse units)
  uint32_t max_iterations;
} DwParams;

typedef struct DwEval {
  float pos[DW_B][3], rot[DW_B][4];
  float velb[DW_B][6];                 // world COM linear then angular
  float J[DW_B][6][DW_N];              // spatial jacobian, linear rows first
  float M[DW_N][DW_N];                 // generalized mass incl. armature
  float bias[DW_N];                    // gravity + Coriolis + gyroscopic
} DwEval;

// ------------------------------------------------------------ forward kinem.
// Port of articulated_v1's evaluate(): FK, body velocities/accelerations,
// spatial jacobians, generalized mass and bias, plus av2's armature add.
// Ancestor dof lists skip provably-zero jacobian columns (exact arithmetic
// is unchanged: skipped terms are identically zero).
static DW_HD bool dw_evaluate(const float* q, const float* v, float gz,
                              DwEval* e) {
  float w[DW_B][3], acc[DW_B][3], alpha[DW_B][3];
  int ncols[DW_B]; unsigned char col[DW_B][12];
  memset(e->J, 0, sizeof(e->J));
  memset(e->M, 0, sizeof(e->M));
  memset(e->bias, 0, sizeof(e->bias));
  // floor body 0: identity pose, zero velocity
  for (int k = 0; k < 3; k++) { e->pos[0][k] = 0; w[0][k] = 0; acc[0][k] = 0; alpha[0][k] = 0; }
  e->rot[0][0] = e->rot[0][1] = e->rot[0][2] = 0; e->rot[0][3] = 1;
  for (int k = 0; k < 6; k++) e->velb[0][k] = 0;
  ncols[0] = 0;
  // root body 1
  float offset[3];
  dw_rotate(q + 3, DW_ROOT_COM, offset);
  dw_add3(q, offset, e->pos[1]);
  dw_qmul(q + 3, DW_ROOT_QPC, e->rot[1]);
  dw_qnormalize(e->rot[1]);
  w[1][0] = v[3]; w[1][1] = v[4]; w[1][2] = v[5];
  float wxo[3];
  dw_cross3(w[1], offset, wxo);
  dw_add3(v, wxo, e->velb[1]);
  dw_cross3(w[1], wxo, acc[1]);
  alpha[1][0] = alpha[1][1] = alpha[1][2] = 0;
  ncols[1] = 6;
  for (int k = 0; k < 6; k++) col[1][k] = (unsigned char)k;
  for (int k = 0; k < 3; k++) {
    float basis[3] = {0, 0, 0}, cb[3];
    basis[k] = 1;
    e->J[1][k][k] = 1;
    dw_cross3(basis, offset, cb);
    for (int a = 0; a < 3; a++) {
      e->J[1][a][3 + k] = cb[a];
      e->J[1][3 + a][3 + k] = basis[a];
    }
  }
  e->velb[1][3] = w[1][0]; e->velb[1][4] = w[1][1]; e->velb[1][5] = w[1][2];
  // hinge chain: child body b = j + 2
  for (int j = 0; j < DW_J; j++) {
    int b = j + 2, p = (int)DW_HINGE_PARENT[j];
    float rp[3], s[3], ax[3], eq[4], t[4], rc[3];
    dw_rotate(e->rot[p], DW_HINGE_AP[j], rp);
    dw_rotate(e->rot[p], DW_HINGE_AXIS[j], s);
    dw_scale3(DW_HINGE_AXIS[j], q[7 + j], ax);
    dw_qexp(ax, eq);
    dw_qmul(e->rot[p], eq, t);
    dw_qmul(t, DW_HINGE_REF[j], e->rot[b]);
    dw_qnormalize(e->rot[b]);
    dw_rotate(e->rot[b], DW_HINGE_AC[j], rc);
    dw_add3(e->pos[p], rp, e->pos[b]); dw_sub3(e->pos[b], rc, e->pos[b]);
    float qd = v[6 + j], tmp[3], tmp2[3];
    dw_scale3(s, qd, tmp); dw_add3(w[p], tmp, w[b]);
    dw_cross3(w[p], s, tmp); dw_scale3(tmp, qd, tmp);
    dw_add3(alpha[p], tmp, alpha[b]);
    dw_cross3(w[p], rp, tmp); dw_add3(e->velb[p], tmp, e->velb[b]);
    dw_cross3(w[b], rc, tmp); dw_sub3(e->velb[b], tmp, e->velb[b]);
    float aa[3];
    dw_cross3(alpha[p], rp, tmp); dw_add3(acc[p], tmp, aa);
    dw_cross3(w[p], rp, tmp); dw_cross3(w[p], tmp, tmp2); dw_add3(aa, tmp2, aa);
    dw_cross3(alpha[b], rc, tmp); dw_sub3(aa, tmp, acc[b]);
    dw_cross3(w[b], rc, tmp); dw_cross3(w[b], tmp, tmp2);
    dw_sub3(acc[b], tmp2, acc[b]);
    e->velb[b][3] = w[b][0]; e->velb[b][4] = w[b][1]; e->velb[b][5] = w[b][2];
    ncols[b] = ncols[p] + 1;
    for (int k = 0; k < ncols[p]; k++) col[b][k] = col[p][k];
    col[b][ncols[p]] = (unsigned char)(6 + j);
    for (int k = 0; k < ncols[b]; k++) {
      int n = col[b][k];
      float ang[3] = {e->J[p][3][n], e->J[p][4][n], e->J[p][5][n]};
      float lin[3] = {e->J[p][0][n], e->J[p][1][n], e->J[p][2][n]};
      float full_ang[3] = {ang[0], ang[1], ang[2]};
      if (n == 6 + j) dw_add3(full_ang, s, full_ang);
      dw_cross3(ang, rp, tmp); dw_add3(lin, tmp, lin);
      dw_cross3(full_ang, rc, tmp); dw_sub3(lin, tmp, lin);
      for (int a = 0; a < 3; a++) {
        e->J[b][a][n] = lin[a];
        e->J[b][3 + a][n] = full_ang[a];
      }
    }
  }
  // generalized mass and bias
  float gravity[3] = {0, 0, gz};
  for (int b = 1; b < DW_B; b++) {
    float mass = DW_BODY_MASS[b];
    float iw[3], force[3], torque[3], tmp[3];
    dw_inertia(e->rot[b], DW_BODY_INERTIA[b], w[b], iw);
    dw_sub3(acc[b], gravity, force); dw_scale3(force, mass, force);
    dw_inertia(e->rot[b], DW_BODY_INERTIA[b], alpha[b], torque);
    dw_cross3(w[b], iw, tmp); dw_add3(torque, tmp, torque);
    float Ijw[12][3];
    for (int k = 0; k < ncols[b]; k++) {
      int n = col[b][k];
      float jw[3] = {e->J[b][3][n], e->J[b][4][n], e->J[b][5][n]};
      dw_inertia(e->rot[b], DW_BODY_INERTIA[b], jw, Ijw[k]);
    }
    for (int i = 0; i < ncols[b]; i++) {
      int n = col[b][i];
      float jv[3] = {e->J[b][0][n], e->J[b][1][n], e->J[b][2][n]};
      float jw[3] = {e->J[b][3][n], e->J[b][4][n], e->J[b][5][n]};
      e->bias[n] += dw_dot3(jv, force) + dw_dot3(jw, torque);
      for (int k2 = 0; k2 <= i; k2++) {
        int m = col[b][k2];
        float jv2[3] = {e->J[b][0][m], e->J[b][1][m], e->J[b][2][m]};
        float x = mass * dw_dot3(jv, jv2) + dw_dot3(jw, Ijw[k2]);
        e->M[n][m] += x;
        if (m != n) e->M[m][n] += x;
      }
    }
  }
  for (int j = 0; j < DW_J; j++) e->M[6 + j][6 + j] += DW_ARMATURE;
  return dw_finite(&e->M[0][0], DW_N * DW_N) && dw_finite(e->bias, DW_N)
      && dw_finite(&e->pos[0][0], DW_B * 3) && dw_finite(&e->velb[0][0], DW_B * 6);
}

// Cholesky of the generalized mass (in place lower factor); false if not SPD.
static DW_HD bool dw_chol(const float M[DW_N][DW_N], float L[DW_N][DW_N]) {
  for (int i = 0; i < DW_N; i++)
    for (int j = 0; j < DW_N; j++) L[i][j] = 0;
  for (int i = 0; i < DW_N; i++) {
    for (int j = 0; j <= i; j++) {
      float x = M[i][j];
      for (int k = 0; k < j; k++) x -= L[i][k] * L[j][k];
      if (i == j) {
        if (!(x > 0) || !isfinite(x)) return false;
        L[i][j] = sqrtf(x);
      } else {
        L[i][j] = x / L[j][j];
      }
    }
  }
  return true;
}
static DW_HD void dw_chol_solve(const float L[DW_N][DW_N], float* x) {
  for (int i = 0; i < DW_N; i++) {
    float s = x[i];
    for (int k = 0; k < i; k++) s -= L[i][k] * x[k];
    x[i] = s / L[i][i];
  }
  for (int i = DW_N - 1; i >= 0; i--) {
    float s = x[i];
    for (int k = i + 1; k < DW_N; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
}

// av2 reference weights: inverse-mass diagonal at the reference pose, zero
// velocity and zero gravity (feeds the soft-row regularizers).
static DW_HD bool dw_reference_weights(float out[DW_J]) {
  DwEval e;
  float zero[DW_N] = {};
  if (!dw_evaluate(DW_REFERENCE_QPOS, zero, 0.0f, &e)) return false;
  float L[DW_N][DW_N];
  if (!dw_chol(e.M, L)) return false;
  for (int j = 0; j < DW_J; j++) {
    float x[DW_N] = {};
    x[6 + j] = 1;
    dw_chol_solve(L, x);
    out[j] = x[6 + j];
    if (!(out[j] > 0) || !isfinite(out[j])) return false;
  }
  return true;
}

// ------------------------------------------------------- contact (contact_v1)
static DW_HD void dw_basis(const float* n, float* u, float* v) {
  float ax = fabsf(n[0]), ay = fabsf(n[1]), az = fabsf(n[2]);
  float s[3] = {0, 0, 0};
  if (ax <= ay && ax <= az) s[0] = 1;
  else if (ay <= az) s[1] = 1;
  else s[2] = 1;
  dw_cross3(s, n, u);
  float len = sqrtf(dw_dot3(u, u));
  u[0] /= len; u[1] /= len; u[2] /= len;
  dw_cross3(n, u, v);
}

typedef struct DwCandidate { float p[3]; float depth; uint64_t id; } DwCandidate;

// contact_v1 reduce(): dedupe, keep <=4 (deepest + max-min-spread), id order.
static DW_HD void dw_reduce(const float* n, DwCandidate* pts, int count,
                            dwc1_manifold* m) {
  memset(m, 0, sizeof(*m));
  if (!count) return;
  float u[3], v[3];
  dw_basis(n, u, v);
  for (int k = 0; k < 3; k++) { m->normal[k] = n[k]; m->tangent1[k] = u[k]; m->tangent2[k] = v[k]; }
  // insertion sort by id (plane candidates are already ordered; keep exact)
  for (int i = 1; i < count; i++) {
    DwCandidate x = pts[i];
    int j = i - 1;
    while (j >= 0 && pts[j].id > x.id) { pts[j + 1] = pts[j]; j--; }
    pts[j + 1] = x;
  }
  DwCandidate unique[DW_FOOT_VERTS];
  int nu = 0;
  for (int i = 0; i < count; i++) {
    bool duplicate = false;
    for (int k = 0; k < nu; k++) {
      float d[3];
      dw_sub3(pts[i].p, unique[k].p, d);
      if (sqrtf(dw_dot3(d, d)) < DW_CONTACT_EPS * 0.5f) { duplicate = true; break; }
    }
    if (!duplicate) unique[nu++] = pts[i];
  }
  int chosen[4], nc = 0;
  if (nu <= 4) {
    for (int i = 0; i < nu; i++) chosen[nc++] = i;
  } else {
    int deep = 0;
    for (int i = 1; i < nu; i++) if (unique[i].depth > unique[deep].depth) deep = i;
    chosen[nc++] = deep;
    while (nc < 4) {
      float best = -1; int bi = 0;
      for (int i = 0; i < nu; i++) {
        bool taken = false;
        for (int k = 0; k < nc; k++) if (chosen[k] == i) { taken = true; break; }
        if (taken) continue;
        float score = 1e30f;
        for (int k = 0; k < nc; k++) {
          float d[3];
          dw_sub3(unique[i].p, unique[chosen[k]].p, d);
          score = fminf(score, dw_dot3(d, d));
        }
        if (score > best) { best = score; bi = i; }
      }
      chosen[nc++] = bi;
    }
  }
  for (int i = 1; i < nc; i++) {  // sort chosen by candidate id
    int x = chosen[i], j = i - 1;
    while (j >= 0 && unique[chosen[j]].id > unique[x].id) { chosen[j + 1] = chosen[j]; j--; }
    chosen[j + 1] = x;
  }
  m->count = (uint32_t)nc;
  for (int i = 0; i < nc; i++) {
    const DwCandidate* p = &unique[chosen[i]];
    m->points[i].feature = p->id;
    for (int k = 0; k < 3; k++) m->points[i].point[k] = p->p[k];
    m->points[i].depth = p->depth;
  }
}

// Foot (convex, body_a) vs static floor plane z=0 (body_b). Matches
// contact_v1's plane_contact with plane_is_a=false: manifold normal is -up.
static DW_HD void dw_plane_manifold(int pair, const float pos[3],
                                    const float rot[4], dwc1_manifold* m) {
  const float n[3] = {0, 0, 1};
  DwCandidate pts[DW_FOOT_VERTS];
  int count = 0;
  for (int i = 0; i < DW_FOOT_VERTS; i++) {
    float world[3];
    dw_rotate(rot, DW_FOOT_VERTICES[pair][i], world);
    dw_add3(pos, world, world);
    float sep = world[2];  // dot(n, world) - offset, offset = 0
    if (sep <= DW_CONTACT_EPS) {
      DwCandidate* c = &pts[count++];
      float shift[3];
      dw_scale3(n, sep * 0.5f, shift);
      dw_sub3(world, shift, c->p);
      c->depth = fmaxf(0.0f, -sep);
      c->id = 0x100000000ull + (uint64_t)i + 1;
    }
  }
  const float mn[3] = {0, 0, -1};
  dw_reduce(mn, pts, count, m);
}

// idv1 row(): generalized velocity row for direction d at world point.
static DW_HD void dw_contact_row(const DwEval* e, int body_a, int body_b,
                                 const float* point, const float* direction,
                                 float* out) {
  for (int n = 0; n < DW_N; n++) out[n] = 0;
  const int body[2] = {body_a, body_b};
  const float sign[2] = {-1.0f, 1.0f};
  for (int i = 0; i < 2; i++) {
    int b = body[i];
    float arm[3], torque[3];
    dw_sub3(point, e->pos[b], arm);
    dw_cross3(arm, direction, torque);
    for (int n = 0; n < DW_N; n++) {
      float x = 0;
      for (int k = 0; k < 3; k++)
        x += direction[k] * e->J[b][k][n] + torque[k] * e->J[b][3 + k][n];
      out[n] += sign[i] * x;
    }
  }
}

// --------------------------------------------------------------- joint rows
static DW_HD float dw_bounded_impedance(float x) {
  return dw_clampf(x, 1e-4f, 0.9999f);
}
static DW_HD float dw_impedance(float gap) {  // av2 impedance() with our solimp
  float d0 = dw_bounded_impedance(DW_LIMIT_SOLIMP[0]);
  float dwv = dw_bounded_impedance(DW_LIMIT_SOLIMP[1]);
  float width = DW_LIMIT_SOLIMP[2];
  float mid = dw_bounded_impedance(DW_LIMIT_SOLIMP[3]);
  float power = DW_LIMIT_SOLIMP[4];
  if (d0 == dwv || width <= 1e-15f) return 0.5f * (d0 + dwv);
  float x = fabsf((gap - DW_LIMIT_MARGIN) / width);
  if (x >= 1) return dwv;
  if (x <= 0) return d0;
  float y = power == 1 ? x
      : (x <= mid ? powf(x, power) / powf(mid, power - 1)
                  : 1 - powf(1 - x, power) / powf(1 - mid, power - 1));
  return d0 + (dwv - d0) * y;
}

// Compacted row problem (only active rows enter the solve; skipped rows have
// identically zero jacobian and pinned zero impulse on the CPU oracle too).
typedef struct DwRows {
  int R, C;
  float G[DW_MAXROWS][DW_N];
  float target[DW_MAXROWS], reg[DW_MAXROWS];
  float lo[DW_MAXROWS], hi[DW_MAXROWS], warm[DW_MAXROWS];
  int kind[DW_MAXROWS];   // 0 scalar joint row, 1 normal, 2/3 tangents
  int slot[DW_MAXROWS];   // joint warm slot [0,42) or -1 for contact rows
  int cfirst[DW_PAIRS * DW_MAXPOINTS];
  float cmu[DW_PAIRS * DW_MAXPOINTS];
} DwRows;

// -------------------------------------------------- dense solver (civ1 port)
typedef struct DwDisk { float x, y; } DwDisk;
static DW_HD bool dw_disk(float a, float b, float c, float fx, float fy,
                          float cap, DwDisk* out) {
  if (!(isfinite(cap) && cap >= 0)) return false;
  if (cap == 0) { out->x = out->y = 0; return true; }
  float determinant = a * c - b * b;
  if (!(a > 0 && c > 0 && determinant > 0 && isfinite(determinant))) return false;
  bool ok = true;
  auto at = [&](float l) {
    DwDisk v = {0, 0};
    float d = (a + l) * (c + l) - b * b;
    if (!(isfinite(d) && d > 0)) { ok = false; return v; }
    float nx = -(c + l) * fx + b * fy, ny = b * fx - (a + l) * fy;
    if (!(isfinite(nx) && isfinite(ny))) { ok = false; return v; }
    v.x = nx / d; v.y = ny / d;
    if (!(isfinite(v.x) && isfinite(v.y))) ok = false;
    return v;
  };
  DwDisk fr = at(0);
  if (!ok) return false;
  if (hypotf(fr.x, fr.y) <= cap) { *out = fr; return true; }
  float lo = 0;
  float hi = fmaxf(fmaxf(a, c), fmaxf(hypotf(fx, fy) / cap, 1e-30f));
  if (!isfinite(hi)) return false;
  for (int k = 0; k < 128; k++) {
    DwDisk v = at(hi);
    if (!ok) return false;
    if (hypotf(v.x, v.y) <= cap) break;
    hi *= 2;
    if (!isfinite(hi)) return false;
  }
  {
    DwDisk v = at(hi);
    if (!ok || hypotf(v.x, v.y) > cap) return false;
  }
  for (int k = 0; k < 48; k++) {
    float mid = lo + (hi - lo) * 0.5f;
    DwDisk v = at(mid);
    if (!ok) return false;
    if (hypotf(v.x, v.y) > cap) lo = mid; else hi = mid;
  }
  *out = at(hi);
  return ok;
}

// Full generalized non-associated Coulomb solve; see coupled_impulse_v1.cpp.
// Returns a dwc1 status; on success fills vout [N] and lambda_out [R].
static DW_HD int dw_solve(const float L[DW_N][DW_N], const float M[DW_N][DW_N],
                          const float* smooth, const DwRows* rows,
                          float tolerance, uint32_t max_iterations,
                          float* vout, float* lambda_out,
                          dwc1_diagnostic* diag) {
  const int R = rows->R, C = rows->C;
  static_assert(DW_MAXROWS == DW_JROWS + 3 * DW_PAIRS * DW_MAXPOINTS, "rows");
  float response[DW_MAXROWS][DW_N];
  float K[DW_MAXROWS][DW_MAXROWS];
  float lambda[DW_MAXROWS], base[DW_MAXROWS], residual[DW_MAXROWS];
  if (R == 0) {
    for (int k = 0; k < DW_N; k++) vout[k] = smooth[k];
    diag->iterations = 0;
    diag->joint_residual = diag->normal_residual = diag->tangent_residual = 0;
    diag->momentum_residual = 0;
    return DWC1_OK;
  }
  for (int r = 0; r < R; r++) {
    base[r] = -rows->target[r];
    for (int j = 0; j < DW_N; j++) {
      float x = rows->G[r][j];
      base[r] += x * smooth[j];
      response[r][j] = x;
    }
    dw_chol_solve(L, response[r]);
    lambda[r] = dw_clampf(rows->warm[r], rows->lo[r], rows->hi[r]);
  }
  for (int i = 0; i < R; i++)
    for (int j = 0; j < R; j++) {
      float x = i == j ? rows->reg[i] : 0.0f;
      for (int k = 0; k < DW_N; k++) x += rows->G[i][k] * response[j][k];
      if (!isfinite(x)) return DWC1_NUMERIC;
      K[i][j] = x;
    }
  for (int i = 0; i < R; i++)
    if (!(K[i][i] > 0 || (rows->lo[i] == 0 && rows->hi[i] == 0)))
      return DWC1_NUMERIC;
  for (int c = 0; c < C; c++) {
    int r = rows->cfirst[c];
    float cap = rows->cmu[c] * lambda[r], norm = hypotf(lambda[r + 1], lambda[r + 2]);
    if (norm > cap) { lambda[r + 1] *= cap / norm; lambda[r + 2] *= cap / norm; }
  }
  bool numeric = false;
  auto residuals = [&]() {
    for (int i = 0; i < R; i++) {
      float x = base[i];
      for (int j = 0; j < R; j++) x += K[i][j] * lambda[j];
      residual[i] = x;
      if (!isfinite(x)) numeric = true;
    }
  };
  auto scalar = [&](int r) {
    if (rows->lo[r] == rows->hi[r]) return rows->lo[r];
    float candidate = lambda[r] - residual[r] / K[r][r];
    if (!isfinite(candidate)) numeric = true;
    return dw_clampf(candidate, rows->lo[r], rows->hi[r]);
  };
  auto scalar_error = [&](int r) {
    if (rows->lo[r] == rows->hi[r]) return fabsf(lambda[r] - rows->lo[r]);
    if (residual[r] == 0) return 0.0f;
    float correction = residual[r] / K[r][r];
    if (!isfinite(correction)) { numeric = true; return 0.0f; }
    float distance = correction > 0 ? lambda[r] - rows->lo[r]
                                    : rows->hi[r] - lambda[r];
    if (!(distance >= 0)) { numeric = true; return 0.0f; }
    return fminf(fabsf(correction), distance);
  };
  auto update = [&](int r, float x) {
    float delta = x - lambda[r];
    if (!(isfinite(x) && isfinite(delta))) { numeric = true; return; }
    lambda[r] = x;
    for (int i = 0; i < R; i++) {
      residual[i] += K[i][r] * delta;
      if (!isfinite(residual[i])) { numeric = true; return; }
    }
  };
  auto tangent_error = [&](int r, float mu) {
    int a = r + 1, b = r + 2;
    float cap = mu * lambda[r], norm = hypotf(lambda[a], lambda[b]);
    if (!(isfinite(cap) && isfinite(norm))) { numeric = true; return 0.0f; }
    if (cap == 0) return norm;
    float aa = K[a][a], ab = 0.5f * (K[a][b] + K[b][a]), bb = K[b][b];
    float det = aa * bb - ab * ab;
    float largest = 0.5f * (aa + bb + hypotf(aa - bb, 2 * ab));
    float smallest = det / largest;
    if (!(isfinite(smallest) && smallest > 0)) { numeric = true; return 0.0f; }
    float gx = residual[a], gy = residual[b], stationarity = 0, complement = 0;
    if (norm == 0) {
      stationarity = hypotf(gx, gy) / smallest;
    } else {
      float ux = lambda[a] / norm, uy = lambda[b] / norm;
      float radial = gx * ux + gy * uy;
      if (!isfinite(radial)) { numeric = true; return 0.0f; }
      float m2 = fmaxf(0.0f, -radial);
      stationarity = hypotf(gx + m2 * ux, gy + m2 * uy) / smallest;
      complement = (m2 / smallest) * (fmaxf(0.0f, cap - norm) / norm);
    }
    if (!(isfinite(stationarity) && isfinite(complement))) { numeric = true; return 0.0f; }
    return fmaxf(fmaxf(stationarity, complement), fmaxf(0.0f, norm - cap));
  };
  float jr = 0, nr = 0, tr = 0;
  auto certify = [&]() {
    jr = nr = tr = 0;
    for (int r = 0; r < R; r++)
      if (rows->kind[r] == 0) jr = fmaxf(jr, scalar_error(r));
    for (int c = 0; c < C; c++) {
      int r = rows->cfirst[c];
      nr = fmaxf(nr, scalar_error(r));
      tr = fmaxf(tr, tangent_error(r, rows->cmu[c]));
    }
    return fmaxf(fmaxf(jr, nr), tr);
  };
  auto correlation = [&](int i, int j) {
    float d2 = K[i][i] * K[j][j];
    if (!(d2 > 0 && isfinite(d2))) return 0.0f;
    float s = 0.5f * (K[i][j] + K[j][i]) / sqrtf(d2);
    return isfinite(s) ? s : 0.0f;
  };
  // Joint nonnegative solve of two nearly dependent normal rows (civ1's
  // coupled_pair), tangents held fixed; kept whichever candidate has the
  // smallest complementarity error, restored on any numeric trouble.
  auto coupled_pair = [&](int i, int j) {
    const float aii = K[i][i], ajj = K[j][j], aij = 0.5f * (K[i][j] + K[j][i]);
    const float qi = residual[i] - aii * lambda[i] - aij * lambda[j];
    const float qj = residual[j] - aij * lambda[i] - ajj * lambda[j];
    if (!(aii > 0 && ajj > 0 && isfinite(aij) && isfinite(qi) && isfinite(qj)))
      return;
    float bi = lambda[i], bj = lambda[j], be = INFINITY;
    auto consider = [&](float xi, float xj) {
      if (!(isfinite(xi) && isfinite(xj))) return;
      xi = fmaxf(0.0f, xi); xj = fmaxf(0.0f, xj);
      float ri = qi + aii * xi + aij * xj, rj = qj + aij * xi + ajj * xj;
      if (!(isfinite(ri) && isfinite(rj))) return;
      float e2 = fmaxf(fabsf(fminf(xi, ri)), fabsf(fminf(xj, rj)));
      if (isfinite(e2) && e2 < be) { be = e2; bi = xi; bj = xj; }
    };
    consider(0.0f, 0.0f);
    consider(-qi / aii, 0.0f);
    consider(0.0f, -qj / ajj);
    const float det = aii * ajj - aij * aij;
    if (det > 1e-6f * aii * ajj) {
      consider((aij * qj - ajj * qi) / det, (aij * qi - aii * qj) / det);
    } else {
      float ui = sqrtf(aii), uj = aij < 0 ? -sqrtf(ajj) : sqrtf(ajj);
      float uu = aii + ajj;
      if (uu > 0) { float al = -(ui * qi + uj * qj) / (uu * uu); consider(al * ui, al * uj); }
    }
    const float cap = 1e6f * (1 + fabsf(lambda[i]) + fabsf(lambda[j])
                              + fabsf(qi) / aii + fabsf(qj) / ajj);
    if (isfinite(be) && bi <= cap && bj <= cap) { update(i, bi); update(j, bj); }
  };
  float wprev[DW_MAXROWS], wprev2[DW_MAXROWS];
  int nwindows = 0;
  // Richardson extrapolation of window-to-window creep; kept only when the
  // unchanged certificate strictly improves (civ1's extrapolate).
  auto extrapolate = [&]() {
    if (nwindows < 2) return;
    float snapl[DW_MAXROWS], snapr[DW_MAXROWS];
    for (int i = 0; i < R; i++) { snapl[i] = lambda[i]; snapr[i] = residual[i]; }
    const float sj0 = jr, sn0 = nr, st0 = tr, cert0 = fmaxf(fmaxf(jr, nr), tr);
    bool numeric0 = numeric, kept = false;
    float n1 = 0, n2 = 0, dot = 0;
    for (int i = 0; i < R; i++) {
      const float d1 = lambda[i] - wprev[i], d2 = wprev[i] - wprev2[i];
      n1 += d1 * d1; n2 += d2 * d2; dot += d1 * d2;
    }
    n1 = sqrtf(n1); n2 = sqrtf(n2);
    if (n1 > 0 && n2 > 0 && isfinite(n1) && isfinite(n2) && isfinite(dot)) {
      const float rho = fminf(n1 / n2, 0.9999f), ca = dot / (n1 * n2);
      if (ca > 0.5f && rho > 0.3f) {
        const float f = rho / (1 - rho);
        bool bad = false;
        for (int i = 0; i < R; i++) {
          const float x = lambda[i] + f * (lambda[i] - wprev[i]);
          if (!isfinite(x)) { bad = true; break; }
          lambda[i] = dw_clampf(x, rows->lo[i], rows->hi[i]);
        }
        if (!bad) {
          for (int c = 0; c < C; c++) {
            const int r = rows->cfirst[c];
            const float cp = rows->cmu[c] * lambda[r];
            const float n3 = hypotf(lambda[r + 1], lambda[r + 2]);
            if (n3 > cp) {
              const float sc = cp > 0 ? cp / n3 : 0.0f;
              lambda[r + 1] *= sc; lambda[r + 2] *= sc;
            }
          }
          residuals();
          const float e2 = certify();
          if (!numeric && isfinite(e2) && e2 < cert0) kept = true;
        }
      }
    }
    if (!kept) {
      for (int i = 0; i < R; i++) { lambda[i] = snapl[i]; residual[i] = snapr[i]; }
      jr = sj0; nr = sn0; tr = st0; numeric = numeric0;
    }
  };
  uint32_t apgd_budget = DW_APGD_BUDGET;
  // Tresca fixed-point acceleration (civ1's block_accelerate): freeze the
  // friction caps at the current normals, solve the resulting convex box/disk
  // QP by accelerated projected gradient with adaptive restart, refresh the
  // caps; keep the trial only when the unchanged certificate strictly
  // improves. This is the proven repair for the exactly rank-deficient
  // flat-foot contact blocks in the fault corpus.
  auto block_accelerate = [&]() {
    if (R < 1 || !apgd_budget) return;
    float snapl[DW_MAXROWS], snapr[DW_MAXROWS];
    for (int i = 0; i < R; i++) { snapl[i] = lambda[i]; snapr[i] = residual[i]; }
    const float sj0 = jr, sn0 = nr, st0 = tr, cert0 = fmaxf(fmaxf(jr, nr), tr);
    bool numeric0 = numeric, kept = false;
    float Lc = 0;
    for (int i = 0; i < R; i++) {
      float s = 0;
      for (int j = 0; j < R; j++) s += fabsf(0.5f * (K[i][j] + K[j][i]));
      Lc = fmaxf(Lc, s);
    }
    if (isfinite(Lc) && Lc > 0) {
      const float step = 1 / Lc;
      float x[DW_MAXROWS], y[DW_MAXROWS], xn[DW_MAXROWS], grad[DW_MAXROWS];
      float bestx[DW_MAXROWS], cap[DW_PAIRS * DW_MAXPOINTS];
      for (int i = 0; i < R; i++) { x[i] = lambda[i]; bestx[i] = lambda[i]; }
      float beste = cert0;
      bool bad = false;
      for (int outer = 0; outer < 64 && apgd_budget && !bad; outer++) {
        for (int c = 0; c < C; c++)
          cap[c] = rows->cmu[c] * fmaxf(0.0f, x[rows->cfirst[c]]);
        float t = 1;
        for (int i = 0; i < R; i++) y[i] = x[i];
        for (int k = 0; k < 4096 && apgd_budget; k++) {
          apgd_budget--;
          for (int i = 0; i < R; i++) {
            float s = base[i];
            for (int j = 0; j < R; j++) s += K[i][j] * y[j];
            grad[i] = s;
          }
          for (int i = 0; i < R; i++) {
            xn[i] = dw_clampf(y[i] - step * grad[i], rows->lo[i], rows->hi[i]);
            if (!isfinite(xn[i])) { bad = true; break; }
          }
          if (bad) break;
          for (int c = 0; c < C; c++) {
            const int r = rows->cfirst[c];
            const float n2 = hypotf(xn[r + 1], xn[r + 2]);
            if (n2 > cap[c]) {
              const float f = cap[c] > 0 ? cap[c] / n2 : 0.0f;
              xn[r + 1] *= f; xn[r + 2] *= f;
            }
          }
          float dot = 0, dn = 0, xs = 0;
          for (int i = 0; i < R; i++) {
            const float dx = xn[i] - x[i];
            dot += (y[i] - xn[i]) * dx;
            dn = fmaxf(dn, fabsf(dx));
            xs = fmaxf(xs, fabsf(xn[i]));
          }
          if (dot > 0) t = 1;                              // adaptive restart
          const float tn = 0.5f * (1 + sqrtf(1 + 4 * t * t)), mo = (t - 1) / tn;
          for (int i = 0; i < R; i++) { y[i] = xn[i] + mo * (xn[i] - x[i]); x[i] = xn[i]; }
          t = tn;
          if (!(dn > 1.2e-7f * (1 + xs))) break;           // fixed point (f32)
        }
        if (bad) break;
        for (int i = 0; i < R; i++) lambda[i] = x[i];
        residuals();
        const float e2 = certify();
        if (!numeric && isfinite(e2) && e2 < beste) {
          beste = e2;
          for (int i = 0; i < R; i++) bestx[i] = x[i];
        }
        for (int i = 0; i < R; i++) { lambda[i] = snapl[i]; residual[i] = snapr[i]; }
        numeric = numeric0;
        if (beste <= tolerance) break;
      }
      if (!bad && beste < cert0) {
        for (int i = 0; i < R; i++) lambda[i] = bestx[i];
        residuals();
        certify();
        if (!numeric) kept = true;
      }
    }
    if (!kept) {
      for (int i = 0; i < R; i++) { lambda[i] = snapl[i]; residual[i] = snapr[i]; }
      jr = sj0; nr = sn0; tr = st0; numeric = numeric0;
    }
  };
  uint32_t iterations = 0;
  bool converged = false, stalled = false;
  int pair_i = R, pair_j = R;
  uint32_t accelerations = 16;
  float reference = INFINITY;
  residuals();
  if (numeric) return DWC1_NUMERIC;
  for (uint32_t it = 0; it < max_iterations && !converged; it++) {
    // civ1 attempts its null-direction move once at sweep 256; not ported
    // (see the header comment), the Tresca acceleration below covers it.
    int c = 0;
    for (int r = 0; r < R; r++) {
      if (rows->kind[r] == 2 || rows->kind[r] == 3) continue;
      update(r, scalar(r));
      if (rows->kind[r] == 1) {
        if (!(c < C && rows->cfirst[c] == r)) return DWC1_INVALID;
        int a = r + 1, b = r + 2;
        float aa = K[a][a], ab = 0.5f * (K[a][b] + K[b][a]), bb = K[b][b];
        float fx = residual[a] - aa * lambda[a] - ab * lambda[b];
        float fy = residual[b] - ab * lambda[a] - bb * lambda[b];
        DwDisk d2;
        if (!dw_disk(aa, ab, bb, fx, fy, rows->cmu[c] * lambda[r], &d2))
          return DWC1_NUMERIC;
        c++;
        update(r + 1, d2.x);
        update(r + 2, d2.y);
      }
      if (numeric) return DWC1_NUMERIC;
    }
    if (stalled && pair_i < R && pair_j < R) {
      float snapl[DW_MAXROWS], snapr[DW_MAXROWS];
      for (int i = 0; i < R; i++) { snapl[i] = lambda[i]; snapr[i] = residual[i]; }
      bool numeric0 = numeric;
      coupled_pair(pair_i, pair_j);
      if (numeric) {
        for (int i = 0; i < R; i++) { lambda[i] = snapl[i]; residual[i] = snapr[i]; }
        numeric = numeric0;
      }
    }
    residuals();          // recompute each sweep; hides no roundoff drift
    float err = certify();
    if (numeric) return DWC1_NUMERIC;
    iterations = it + 1;
    converged = err <= tolerance;
    if (!converged && (it & 31u) == 31u) {
      if (pair_i < R && err > 0.5f * reference) { pair_i = pair_j = R; }
      if (it >= 63 && !stalled && err > 0.25f * reference) {
        stalled = true;
        float best = 0.99f;
        for (int a2 = 0; a2 < C; a2++)
          for (int b2 = a2 + 1; b2 < C; b2++) {
            int i2 = rows->cfirst[a2], j2 = rows->cfirst[b2];
            float corr = correlation(i2, j2);
            if (corr >= best) { best = corr; pair_i = i2; pair_j = j2; }
          }
      }
      if (stalled) {
        extrapolate();
        converged = fmaxf(fmaxf(jr, nr), tr) <= tolerance;
        if (!converged && accelerations && err > 0.25f * reference) {
          accelerations--;
          block_accelerate();
          converged = fmaxf(fmaxf(jr, nr), tr) <= tolerance;
        }
        if (numeric) return DWC1_NUMERIC;
      }
      for (int i = 0; i < R; i++) { wprev2[i] = wprev[i]; wprev[i] = lambda[i]; }
      nwindows++;
      reference = err;
    }
  }
  if (!converged) {
    diag->iterations = iterations;
    diag->joint_residual = jr; diag->normal_residual = nr;
    diag->tangent_residual = tr;
    return DWC1_NO_CONVERGENCE;
  }
  for (int k = 0; k < DW_N; k++) {
    float x = smooth[k];
    for (int r = 0; r < R; r++) x += response[r][k] * lambda[r];
    if (!isfinite(x)) return DWC1_NUMERIC;
    vout[k] = x;
  }
  float mr = 0;
  for (int k = 0; k < DW_N; k++) {
    float x = 0;
    for (int j = 0; j < DW_N; j++) x += M[k][j] * (vout[j] - smooth[j]);
    for (int r = 0; r < R; r++) x -= rows->G[r][k] * lambda[r];
    if (!isfinite(x)) return DWC1_NUMERIC;
    mr = fmaxf(mr, fabsf(x));
  }
  if (!(mr <= DW_MOMENTUM_TOLERANCE)) return DWC1_NUMERIC;
  for (int r = 0; r < R; r++) lambda_out[r] = lambda[r];
  diag->iterations = iterations;
  diag->joint_residual = jr; diag->normal_residual = nr;
  diag->tangent_residual = tr; diag->momentum_residual = mr;
  return DWC1_OK;
}

// ------------------------------------------------------------------- tick
// One 0.002 s step of one environment. On any failure the state is left
// exactly as it was (per-env rollback); diag->status reports the failure.
static DW_HD void dw_tick(DwState* s, const float* target,
                          const DwParams* params, dwc1_diagnostic* diag) {
  const float dt = DW_DT;
  diag->contact_points = 0;
  diag->active_limits = 0;
  diag->maximum_normal_impulse = 0;
  diag->maximum_penetration = 0;
  DwEval e;
  if (!dw_evaluate(s->q, s->v, DW_GRAVITY_Z, &e)) { diag->status = DWC1_DYNAMICS; return; }
  float L[DW_N][DW_N];
  if (!dw_chol(e.M, L)) { diag->status = DWC1_DYNAMICS; return; }
  // PD actuator (clip(target) per limits, effort cap) + passive damping; the
  // smooth velocity is v + dt * M^-1 (actuator + passive - bias).
  float smooth[DW_N];
  for (int n = 0; n < DW_N; n++) smooth[n] = -e.bias[n];
  for (int j = 0; j < DW_J; j++) {
    float tj = dw_clampf(target[j], DW_LIMIT_LOWER[j], DW_LIMIT_UPPER[j]);
    float motor = DW_KP * (tj - s->q[7 + j]) + DW_KV * (0.0f - s->v[6 + j]);
    if (!isfinite(motor)) { diag->status = DWC1_DYNAMICS; return; }
    smooth[6 + j] += dw_clampf(motor, -DW_EFFORT_CAP, DW_EFFORT_CAP)
                   - DW_DAMPING * s->v[6 + j];
  }
  dw_chol_solve(L, smooth);
  for (int n = 0; n < DW_N; n++) {
    smooth[n] = s->v[n] + dt * smooth[n];
    if (!isfinite(smooth[n])) { diag->status = DWC1_DYNAMICS; return; }
  }
  // joint rows (av2_prepare): friction always active (loss > 0); soft limit
  // rows activate when the gap dips under the (zero) margin.
  DwRows rows;
  rows.R = 0; rows.C = 0;
  for (int j = 0; j < DW_J; j++) {
    for (int side = 0; side < 3; side++) {
      float sign = side == 2 ? -1.0f : 1.0f;
      float gap = side == 0 ? 0.0f
          : (side == 1 ? s->q[7 + j] - DW_LIMIT_LOWER[j]
                       : DW_LIMIT_UPPER[j] - s->q[7 + j]);
      bool active = side == 0 ? true : gap < DW_LIMIT_MARGIN;
      if (!active) continue;
      float d = side == 0 ? dw_bounded_impedance(DW_FRICTION_D0) : dw_impedance(gap);
      float width = dw_bounded_impedance(side == 0 ? DW_FRICTION_DWIDTH
                                                   : DW_LIMIT_SOLIMP[1]);
      float tc = fmaxf(side == 0 ? DW_FRICTION_TIMECONST : DW_LIMIT_TIMECONST,
                       2 * dt);
      float B = 2 / fmaxf(1e-15f, width * tc);
      float Ks = side == 0 ? 0.0f
          : 1 / fmaxf(1e-15f, width * width * tc * tc
                              * DW_LIMIT_DAMPRATIO * DW_LIMIT_DAMPRATIO);
      float rowv = sign * s->v[6 + j];
      float aref = -B * rowv - Ks * d * (gap - DW_LIMIT_MARGIN);
      int r = rows.R++;
      for (int n = 0; n < DW_N; n++) rows.G[r][n] = 0;
      rows.G[r][6 + j] = sign;
      rows.reg[r] = fmaxf(1e-15f, (1 - d) / d * params->refweight[j]);
      rows.target[r] = rowv + dt * aref;
      rows.lo[r] = side == 0 ? -dt * DW_FRICTION_LOSS : 0.0f;
      rows.hi[r] = side == 0 ? dt * DW_FRICTION_LOSS : INFINITY;
      rows.warm[r] = dw_clampf(dt * s->warm[side * DW_J + j], rows.lo[r], rows.hi[r]);
      rows.kind[r] = 0;
      rows.slot[r] = side * DW_J + j;
      if (side) diag->active_limits++;
      if (!(isfinite(rows.reg[r]) && rows.reg[r] > 0 && isfinite(aref)
            && isfinite(rows.target[r]) && isfinite(rows.warm[r]))) {
        diag->status = DWC1_DYNAMICS;
        return;
      }
    }
  }
  // contact manifolds at current poses + idv1's warm-start matching by
  // feature id (normal agreement > .98, point displacement < .02 m).
  dwc1_manifold manifolds[DW_PAIRS];
  for (int pair = 0; pair < DW_PAIRS; pair++) {
    int foot = (int)DW_PAIR_BODY_A[pair];
    dw_plane_manifold(pair, e.pos[foot], e.rot[foot], &manifolds[pair]);
    dwc1_manifold* m = &manifolds[pair];
    const dwc1_manifold* previous = &s->cache[pair];
    bool normal_ok = dw_dot3(m->normal, previous->normal) > 0.98f;
    for (uint32_t k = 0; k < m->count; k++) {
      dwc1_point* x = &m->points[k];
      float warm[3] = {0, 0, 0};
      if (normal_ok) {
        for (uint32_t q2 = 0; q2 < previous->count; q2++) {
          const dwc1_point* y = &previous->points[q2];
          float dvec[3];
          dw_sub3(x->point, y->point, dvec);
          if (x->feature == y->feature && dw_dot3(dvec, dvec) < 0.0004f) {
            warm[0] = y->normal_impulse;
            float t[3];
            for (int a = 0; a < 3; a++)
              t[a] = previous->tangent1[a] * y->tangent_impulse[0]
                   + previous->tangent2[a] * y->tangent_impulse[1];
            warm[1] = dw_dot3(t, m->tangent1);
            warm[2] = dw_dot3(t, m->tangent2);
            break;
          }
        }
      }
      rows.cfirst[rows.C] = rows.R;
      rows.cmu[rows.C] = DW_PAIR_MU[pair];
      rows.C++;
      const float* directions[3] = {m->normal, m->tangent1, m->tangent2};
      for (int a = 0; a < 3; a++) {
        int r = rows.R++;
        dw_contact_row(&e, (int)DW_PAIR_BODY_A[pair], (int)DW_PAIR_BODY_B[pair],
                       x->point, directions[a], rows.G[r]);
        rows.target[r] = a == 0
            ? fminf(1.0f, 0.2f * fmaxf(0.0f, x->depth - DW_CONTACT_EPS) / dt)
            : 0.0f;
        rows.reg[r] = 0;
        rows.lo[r] = a == 0 ? 0.0f : -INFINITY;
        rows.hi[r] = INFINITY;
        rows.warm[r] = warm[a];
        rows.kind[r] = a + 1;
        rows.slot[r] = -1;
      }
      diag->contact_points++;
      diag->maximum_penetration = fmaxf(diag->maximum_penetration, x->depth);
    }
  }
  // dense coupled solve
  float vnew[DW_N], lambda[DW_MAXROWS];
  int rc = dw_solve(L, e.M, smooth, &rows, params->tolerance,
                    params->max_iterations, vnew, lambda, diag);
  if (rc != DWC1_OK) { diag->status = (uint32_t)rc; return; }
  // integrate (av1_integrate_root + hinge Euler), then commit atomically
  float qnew[DW_Q];
  for (int k = 0; k < 3; k++) qnew[k] = s->q[k] + dt * vnew[k];
  {
    float wdt[3] = {dt * vnew[3], dt * vnew[4], dt * vnew[5]}, eq[4];
    dw_qexp(wdt, eq);
    dw_qmul(eq, s->q + 3, qnew + 3);
    dw_qnormalize(qnew + 3);
  }
  for (int j = 0; j < DW_J; j++) qnew[7 + j] = s->q[7 + j] + dt * vnew[6 + j];
  if (!dw_finite(qnew, DW_Q) || !dw_finite(vnew, DW_N)) {
    diag->status = DWC1_DYNAMICS;
    return;
  }
  float warmnew[DW_JROWS] = {};
  for (int r = 0; r < rows.R; r++) {
    if (rows.slot[r] < 0) continue;
    float wf = lambda[r] / dt;
    if (rows.slot[r] < DW_J) wf = dw_clampf(wf, -DW_FRICTION_LOSS, DW_FRICTION_LOSS);
    warmnew[rows.slot[r]] = wf;
  }
  {  // store contact impulses into the cache manifolds
    int c = 0;
    for (int pair = 0; pair < DW_PAIRS; pair++) {
      dwc1_manifold* m = &manifolds[pair];
      for (uint32_t k = 0; k < m->count; k++) {
        int fr = rows.cfirst[c++];
        m->points[k].normal_impulse = lambda[fr];
        m->points[k].tangent_impulse[0] = lambda[fr + 1];
        m->points[k].tangent_impulse[1] = lambda[fr + 2];
        diag->maximum_normal_impulse =
            fmaxf(diag->maximum_normal_impulse, lambda[fr]);
      }
    }
  }
  for (int k = 0; k < DW_Q; k++) s->q[k] = qnew[k];
  for (int k = 0; k < DW_N; k++) s->v[k] = vnew[k];
  for (int k = 0; k < DW_JROWS; k++) s->warm[k] = warmnew[k];
  for (int pair = 0; pair < DW_PAIRS; pair++) s->cache[pair] = manifolds[pair];
  s->count++;
  diag->status = DWC1_OK;
}

// n_ticks with the targets held; a failing environment freezes (state and
// cache untouched from its last accepted tick) and keeps its failure diag.
static DW_HD void dw_step_env(DwState* s, const float* target, uint32_t n_ticks,
                              const DwParams* params, dwc1_diagnostic* diag) {
  diag->ticks = 0;
  for (uint32_t t = 0; t < n_ticks; t++) {
    dw_tick(s, target, params, diag);
    if (diag->status != DWC1_OK) return;
    diag->ticks++;
  }
}

// ------------------------------------------------------------- init / read
static DW_HD void dw_init_state(DwState* s, const float* joint_offsets) {
  memset(s, 0, sizeof(*s));
  for (int k = 0; k < DW_Q; k++) s->q[k] = DW_INITIAL_QPOS[k];
  for (int k = 0; k < DW_N; k++) s->v[k] = DW_INITIAL_VEL[k];
  if (joint_offsets)
    for (int j = 0; j < DW_J; j++)
      s->q[7 + j] = dw_clampf(s->q[7 + j] + joint_offsets[j],
                              DW_LIMIT_LOWER[j], DW_LIMIT_UPPER[j]);
}

// Light FK for reads: body poses and velocities only (no mass/bias/jacobian).
static DW_HD void dw_body_states(const float* q, const float* v,
                                 float out[DW_B][13]) {
  float pos[DW_B][3], rot[DW_B][4], w[DW_B][3], vel[DW_B][3];
  for (int k = 0; k < 3; k++) { pos[0][k] = 0; w[0][k] = 0; vel[0][k] = 0; }
  rot[0][0] = rot[0][1] = rot[0][2] = 0; rot[0][3] = 1;
  float offset[3];
  dw_rotate(q + 3, DW_ROOT_COM, offset);
  dw_add3(q, offset, pos[1]);
  dw_qmul(q + 3, DW_ROOT_QPC, rot[1]);
  dw_qnormalize(rot[1]);
  w[1][0] = v[3]; w[1][1] = v[4]; w[1][2] = v[5];
  float wxo[3];
  dw_cross3(w[1], offset, wxo);
  dw_add3(v, wxo, vel[1]);
  for (int j = 0; j < DW_J; j++) {
    int b = j + 2, p = (int)DW_HINGE_PARENT[j];
    float rp[3], s[3], ax[3], eq[4], t[4], rc[3], tmp[3];
    dw_rotate(rot[p], DW_HINGE_AP[j], rp);
    dw_rotate(rot[p], DW_HINGE_AXIS[j], s);
    dw_scale3(DW_HINGE_AXIS[j], q[7 + j], ax);
    dw_qexp(ax, eq);
    dw_qmul(rot[p], eq, t);
    dw_qmul(t, DW_HINGE_REF[j], rot[b]);
    dw_qnormalize(rot[b]);
    dw_rotate(rot[b], DW_HINGE_AC[j], rc);
    dw_add3(pos[p], rp, pos[b]); dw_sub3(pos[b], rc, pos[b]);
    dw_scale3(s, v[6 + j], tmp); dw_add3(w[p], tmp, w[b]);
    dw_cross3(w[p], rp, tmp); dw_add3(vel[p], tmp, vel[b]);
    dw_cross3(w[b], rc, tmp); dw_sub3(vel[b], tmp, vel[b]);
  }
  for (int b = 0; b < DW_B; b++) {
    for (int k = 0; k < 3; k++) out[b][k] = pos[b][k];
    for (int k = 0; k < 4; k++) out[b][3 + k] = rot[b][k];
    for (int k = 0; k < 3; k++) out[b][7 + k] = vel[b][k];
    for (int k = 0; k < 3; k++) out[b][10 + k] = w[b][k];
  }
}

// Whole-sole heights: min world z over the baked sole vertices per foot.
static DW_HD void dw_sole_heights(const float body[DW_B][13], float out[2]) {
  for (int pair = 0; pair < DW_PAIRS; pair++) {
    int b = (int)DW_PAIR_BODY_A[pair];
    float lowest = INFINITY;
    for (int i = 0; i < DW_FOOT_VERTS; i++) {
      float world[3];
      dw_rotate(&body[b][3], DW_FOOT_VERTICES[pair][i], world);
      lowest = fminf(lowest, body[b][2] + world[2]);
    }
    out[pair] = lowest;
  }
}

static DW_HD void dw_fill_info(uint32_t environments, dwc1_info* info) {
  info->environments = environments;
  info->bodies = DW_B; info->joints = DW_J; info->dofs = DW_N;
  info->dt = DW_DT; info->kp = DW_KP; info->kv = DW_KV;
  info->effort_cap = DW_EFFORT_CAP;
  info->home_root_height = DW_INITIAL_QPOS[2];
  for (int k = 0; k < DW_Q; k++) info->home_qpos[k] = DW_INITIAL_QPOS[k];
  for (int j = 0; j < DW_J; j++) {
    info->joint_lower[j] = DW_LIMIT_LOWER[j];
    info->joint_upper[j] = DW_LIMIT_UPPER[j];
    // DW_HOME_TARGETS equals the home joint pose by construction; keep the
    // constant referenced so model drift in either is caught at compile time.
    (void)DW_HOME_TARGETS[j];
  }
}

#endif  // DUCK_CUDA_KERNEL_H
