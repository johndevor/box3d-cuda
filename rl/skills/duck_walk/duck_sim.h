// SPDX-License-Identifier: MIT
// Batched Open Duck Mini v2 worlds for World2's open-duck-walk task: one thread (CUDA) or one loop iteration (CPU) per
// environment. Maximal-coordinate rigid bodies (one per rigid group of the World2 model), revolute joints and foot-floor
// contacts solved by sequential impulses with Baumgarte stabilisation and warm starting, substepped; the STS3215 servo
// model and the IMU signal chain are World2's (operations/servo-bus.mjs bamStep, operations/sensors imu) step for step at
// World2's 1/120 s physics step. Frames as World2: Y up, gravity -Y.
//
// The same source builds for the CPU (debugging on a laptop, parity checks) and for CUDA (training).
#pragma once
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

namespace duck {

constexpr int NB = 15;          // bodies
constexpr int NJ = 14;          // servo joints
constexpr int NF = 2;           // feet
constexpr int NC = 8;           // corners per foot box
constexpr int SW = 13;          // body state: p3 q4(xyzw) v3 w3 (centre of mass, principal frame)
constexpr int NSERVO = 8;       // servo state: theta_m omega_m target_fw goal engaged vt q_prev pending_goal
constexpr int NSP = 14;         // servo params: vin kt R gain max_pwm armature fb fv backlash kg cg gmax maxvel cmd_delay_steps
constexpr int NIMU = 32;        // imu state (see imu_step)
constexpr int QDELAY = 4;       // imu delay queue length (steps)

struct Model {                  // shared by all environments (device pointers)
  const float *pa, *ca, *axis, *ref, *lo, *hi;      // [NJ*3] [NJ*3] [NJ*3] [NJ*4] [NJ] [NJ]
  const int *jp, *jc;                               // [NJ]
  const float *foot_c, *foot_h, *foot_q;            // [NF*3] body-local centre, half extents, rotation (xyzw)
  const int *foot_b;                                // [NF]
  const float *imu_p, *imu_q;                       // body-local mount (trunk)
  int imu_b;
  float dt; int substeps, iterations;
  float baumgarte, slop, contact_beta, contact_slop, max_bias;
};

struct Env {                    // per-environment arrays (device pointers), indexed by e
  float *state;                 // [E,NB,SW]
  const float *inv_mass, *inv_inertia;   // [E,NB] [E,NB,3]
  float *servo;                 // [E,NJ,NSERVO]
  const float *sp;              // [E,NJ,NSP]
  float *jl;                    // [E,NJ,8] joint warm-start impulses: 3 lin, 2 ang, motor, limit, spare
  float *cl;                    // [E,NF,NC,3] contact warm-start impulses: normal, t1, t2
  const float *mu;              // [E] friction coefficient (foot-floor)
  const float *damp;            // [E,2] linear, angular body damping
  float *imu;                   // [E,NIMU]
  const float *imu_par;         // [E,8]: gyro sigma_in, accel sigma_in, gyro alpha, accel alpha, delay steps, fusion tau, -, -
  const float *imu_bias;        // [E,6]
  uint32_t *rng;                // [E]
  const float *goal_new;        // [E,NJ] goals written this policy tick (rad, servo sense, from the standing pose)
  const int *goal_delay;        // [E] outer steps after which the new goals apply (World2: 1 = the first servo step)
  const float *push;            // [E,6]: impulse (N·s) on the trunk centre of mass, applied at the first substep; + angular
  const uint8_t *torque_on;     // [E]
  float *out;                   // [E,NOUT] per-call outputs (see OUT_*)
};
constexpr int OUT_CONTACT = 0;     // [2] foot contact (any corner loaded during the last outer step)
constexpr int OUT_FORCE = 2;       // [2] mean normal force over the call (N)
constexpr int OUT_SLIP = 4;        // [2] mean tangential speed of loaded corners (m/s)
constexpr int OUT_POWER = 6;       // [1] mean electrical power proxy sum |tau_m * omega_m|
constexpr int OUT_SOLE = 7;        // [2] lowest sole corner height (m) at the end
constexpr int OUT_TAU = 9;         // [NJ] motor torque at the end
constexpr int NOUT = 9 + NJ;

// ------------------------------------------------------------------ small vector algebra
HD float dot3(const float *a, const float *b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
HD void cross3(const float *a, const float *b, float *o) { o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0]; }
HD void qmul(const float *a, const float *b, float *o) {
  o[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1]; o[1] = a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0];
  o[2] = a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3]; o[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2];
}
HD void qnorm(float *q) { float l = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]; if (l < 1e-20f) { q[0] = q[1] = q[2] = 0; q[3] = 1; return; } l = 1.0f / sqrtf(l); for (int k = 0; k < 4; k++) q[k] *= l; }
HD void rot(const float *q, const float *v, float *o) { float t[3], u[3]; cross3(q, v, t); for (int k = 0; k < 3; k++) t[k] *= 2; cross3(q, t, u); for (int k = 0; k < 3; k++) o[k] = v[k] + q[3] * t[k] + u[k]; }
HD void rotinv(const float *q, const float *v, float *o) { float c[4] = {-q[0], -q[1], -q[2], q[3]}; rot(c, v, o); }
// world inverse inertia applied to v: R diag(d) R^T v
HD void iworld(const float *q, const float *d, const float *v, float *o) { float l[3]; rotinv(q, v, l); for (int k = 0; k < 3; k++) l[k] *= d[k]; rot(q, l, o); }
HD void integrate_q(float *b, float h) { float q[4] = {b[3], b[4], b[5], b[6]}, r[4] = {b[10] * h * .5f, b[11] * h * .5f, b[12] * h * .5f, 0}, o[4]; qmul(r, q, o); for (int k = 0; k < 4; k++) q[k] += o[k]; qnorm(q); for (int k = 0; k < 4; k++) b[3 + k] = q[k]; }
HD void rotvec(const float *q0, float *o) { float q[4] = {q0[0], q0[1], q0[2], q0[3]}; qnorm(q); if (q[3] < 0) for (int k = 0; k < 4; k++) q[k] = -q[k]; float s = sqrtf(dot3(q, q)); if (s < 1e-12f) { for (int k = 0; k < 3; k++) o[k] = 2 * q[k]; return; } float a = 2 * atan2f(s, q[3]); for (int k = 0; k < 3; k++) o[k] = q[k] * a / s; }
HD void basis(const float *a, float *t1, float *t2) { float e[3] = {0, 0, 0}; e[fabsf(a[0]) < .9f ? 0 : 1] = 1; cross3(e, a, t1); float l = sqrtf(dot3(t1, t1)); for (int k = 0; k < 3; k++) t1[k] /= l; cross3(a, t1, t2); }
HD float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

// body helpers (b: state row, im: inverse mass, ii: inverse principal inertia)
HD void pointvel(const float *b, const float *r, float *o) { float x[3]; cross3(b + 10, r, x); for (int k = 0; k < 3; k++) o[k] = b[7 + k] + x[k]; }
HD void apply_imp(float *b, float im, const float *ii, const float *r, const float *j) { for (int k = 0; k < 3; k++) b[7 + k] += j[k] * im; float x[3], y[3]; cross3(r, j, x); iworld(b + 3, ii, x, y); for (int k = 0; k < 3; k++) b[10 + k] += y[k]; }
HD void apply_aimp(float *b, const float *ii, const float *j) { float y[3]; iworld(b + 3, ii, j, y); for (int k = 0; k < 3; k++) b[10 + k] += y[k]; }
HD float lin_k(const float *b, float im, const float *ii, const float *r, const float *d) { float x[3], y[3], z[3]; cross3(r, d, x); iworld(b + 3, ii, x, y); cross3(y, r, z); return im + dot3(z, d); }
HD float ang_k(const float *b, const float *ii, const float *d) { float y[3]; iworld(b + 3, ii, d, y); return dot3(y, d); }

// ------------------------------------------------------------------ random numbers (per environment, xorshift32)
HD float urand(uint32_t &s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return (s >> 8) * (1.0f / 16777216.0f); }
HD float nrand(uint32_t &s) { float u1 = fmaxf(urand(s), 1e-7f), u2 = urand(s); return sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2); }

// ------------------------------------------------------------------ joint geometry
struct JG { float rp[3], rc[3], axis[3], coord, sep[3], ang[3]; };
HD JG joint_geometry(const float *p, const float *c, const Model &M, int j) {
  JG g; rot(p + 3, M.pa + 3 * j, g.rp); rot(c + 3, M.ca + 3 * j, g.rc); rot(p + 3, M.axis + 3 * j, g.axis);
  for (int k = 0; k < 3; k++) g.sep[k] = (c[k] + g.rc[k]) - (p[k] + g.rp[k]);
  float qp[4] = {-p[3], -p[4], -p[5], p[6]}, qr[4], qrefc[4] = {-M.ref[4 * j], -M.ref[4 * j + 1], -M.ref[4 * j + 2], M.ref[4 * j + 3]}, qd[4], rl[3], rw[3];
  qmul(qp, c + 3, qr); qmul(qr, qrefc, qd); rotvec(qd, rl); g.coord = dot3(rl, M.axis + 3 * j); rot(p + 3, rl, rw);
  float a = dot3(rw, g.axis); for (int k = 0; k < 3; k++) g.ang[k] = rw[k] - g.axis[k] * a;
  return g;
}

// ------------------------------------------------------------------ the servo (World2 operations/servo-bus.mjs bamStep)
// sv: theta_m omega_m target_fw goal engaged vt q_prev pending; P: vin kt R gain max_pwm armature fb fv backlash kg cg gmax maxvel
HD void servo_step(float *sv, const float *P, float q, bool on, float dt, float *tau_m_out, float &power) {
  const float w = (q - sv[6]) / dt; sv[6] = q;
  float duty = 0, tau_m = 0;
  if (on) {
    const float v = P[12];
    sv[2] = fmaxf(sv[2] - v * dt, fminf(sv[2] + v * dt, sv[3]));
    const float s = 6.28318530718f / 4096.f, qe = rintf(q / s) * s;
    duty = clampf(P[3] * (sv[2] - qe), -P[4], P[4]);
    tau_m = P[1] / P[2] * (P[0] * duty - P[1] * sv[1]);
  } else sv[2] = q;
  const float hb = P[8] * .5f, sgn = sv[4];
  const float tau_g = sgn != 0 ? clampf(P[10] * (sv[5] - w), -P[11], P[11]) : 0.f;
  const float tau = tau_m - tau_g - P[7] * sv[1], wtry = sv[1] + dt * tau / P[5], c = P[6] * dt / P[5];
  sv[1] = fabsf(wtry) <= c ? 0.f : wtry - (wtry > 0 ? c : -c);
  sv[0] += dt * sv[1];
  sv[7] = sv[4] != 0 ? (sv[5] != 0 ? sv[5] : 1e-9f) : 0.f;   // (the motor command of the previous step: P[13] > 0 applies it one step late)
  const float d2 = sv[0] - q;
  sv[4] = fabsf(d2) > hb ? (d2 > 0 ? 1.f : -1.f) : 0.f;
  sv[5] = sv[4] != 0 ? sv[1] + P[9] / P[10] * (d2 - sv[4] * hb) : 0.f;
  *tau_m_out = tau_m; power += fabsf(tau_m * sv[1]);
}

// ------------------------------------------------------------------ the IMU (World2 operations/sensors imu + signal.mjs)
// imu state: [0:3] filtered gyro y, [3:6] filtered accel y, [6:9] previous sensor-point velocity, [9] has_prev,
// [10:13] ghat, [13] ghat_init, [14] head of queue, [15] (unused), [16:16+QDELAY*... ] no: queue below
// queue: QDELAY entries of 6 floats at imu + 32? -> stored in a separate region: imu[NIMU] holds only 16 floats + queue 16? (see layout)
// Layout (NIMU = 32 + 6*QDELAY is too big for one row?) -> we keep QDELAY = 2 entries: [16:22] out(t-1), [22:28] out(t-2); [28:31] last output gravity
HD void imu_step(float *S, const float *par, const float *bias, uint32_t &rng, const float *body, const Model &M, float dt) {
  // truth: gyro (sensor frame), specific force from the sensor point's velocity difference
  float qs[4], r[3], vp[3], wx[3];
  qmul(body + 3, M.imu_q, qs);
  // sensor point relative to the centre of mass: mount is body-local (principal frame)
  rot(body + 3, M.imu_p, r); cross3(body + 10, r, wx); for (int k = 0; k < 3; k++) vp[k] = body[7 + k] + wx[k];
  float ap[3] = {0, 0, 0};
  if (S[9] > 0) for (int k = 0; k < 3; k++) ap[k] = (vp[k] - S[6 + k]) / dt;
  for (int k = 0; k < 3; k++) S[6 + k] = vp[k]; S[9] = 1;
  float sf[3] = {ap[0], ap[1] + 9.81f, ap[2]}, gt[3], at[3];
  rotinv(qs, body + 10, gt); rotinv(qs, sf, at);
  // signal chain: bias + white noise (scaled to the filtered sigma) -> first-order low-pass
  const float sg = par[0], sa = par[1], ag = par[2], aa = par[3];
  if (S[15] == 0) {   // first sample: a draw of the filtered reading
    for (int k = 0; k < 3; k++) { S[k] = gt[k] + bias[k] + par[6] * nrand(rng); S[3 + k] = at[k] + bias[3 + k] + par[7] * nrand(rng); }
    S[15] = 1;
  } else for (int k = 0; k < 3; k++) {
    S[k] += ag * (gt[k] + bias[k] + sg * nrand(rng) - S[k]);
    S[3 + k] += aa * (at[k] + bias[3 + k] + sa * nrand(rng) - S[3 + k]);
  }
  // transport delay: outputs t-0, t-1, t-2 (par[4] selects)
  for (int k = 0; k < 6; k++) { S[22 + k] = S[16 + k]; S[16 + k] = S[k]; }   // [16:22] = now, [22:28] = one step back
  const int dsteps = (int)par[4];
  float out[6];
  for (int k = 0; k < 6; k++) out[k] = dsteps <= 0 ? S[k] : (dsteps == 1 ? S[22 + k] : S[22 + k]);   // (2+: as 1, queue length)
  // onboard fusion on the delayed outputs
  float n = sqrtf(dot3(out + 3, out + 3));
  if (S[13] == 0) { for (int k = 0; k < 3; k++) S[10 + k] = n > 1e-9f ? -out[3 + k] / n : (k == 1 ? -1.f : 0.f); S[13] = 1; }
  else {
    float cx[3], g[3]; cross3(S + 10, out, cx);
    for (int k = 0; k < 3; k++) g[k] = S[10 + k] + cx[k] * dt;
    if (n > 1e-9f && fabsf(n - 9.81f) < 3.f) { float kf = fminf(1.f, dt / par[5]); for (int k = 0; k < 3; k++) g[k] += kf * (-out[3 + k] / n - g[k]); }
    float m = sqrtf(dot3(g, g)); if (m > 1e-12f) for (int k = 0; k < 3; k++) S[10 + k] = g[k] / m;
  }
  for (int k = 0; k < 3; k++) S[28 + k] = out[k];   // the gyro reading the program sees
}

// ------------------------------------------------------------------ one environment, one policy call (n_outer steps)
HD void step_env(int e, const Model &M, Env &E, int n_outer) {
  float *st = E.state + (size_t)e * NB * SW;
  const float *im = E.inv_mass + (size_t)e * NB, *ii = E.inv_inertia + (size_t)e * NB * 3;
  float *sv = E.servo + (size_t)e * NJ * NSERVO, *jl = E.jl + (size_t)e * NJ * 8, *cl = E.cl + (size_t)e * NF * NC * 3;
  const float *sp = E.sp + (size_t)e * NJ * NSP, mu = E.mu[e], dl = E.damp[2 * e], da = E.damp[2 * e + 1];
  float *out = E.out + (size_t)e * NOUT;
  uint32_t rng = E.rng[e];
  const bool on = E.torque_on[e] != 0;
  for (int k = 0; k < NOUT; k++) out[k] = 0;
  const float dt = M.dt, h = dt / M.substeps;
  float force_acc[2] = {0, 0}, slip_acc[2] = {0, 0}, power = 0; int contact_last[2] = {0, 0};
  for (int outer = 0; outer < n_outer; outer++) {
    for (int k = 0; k < 2; k++) contact_last[k] = 0;
    // ---- the rigid-body step (World2: one Rapier step of 1/120 s with the motors as the servo step left them)
    for (int ss = 0; ss < M.substeps; ss++) {
      // external forces: gravity, the push (first substep of the call), damping
      for (int b = 0; b < NB; b++) {
        float *x = st + b * SW;
        x[8] += -9.81f * h;
        const float fl = 1.f / (1.f + h * dl), fa = 1.f / (1.f + h * da);
        for (int k = 0; k < 3; k++) { x[7 + k] *= fl; x[10 + k] *= fa; }
      }
      if (outer == 0 && ss == 0) { float *t = st; for (int k = 0; k < 3; k++) t[7 + k] += E.push[6 * e + k] * im[0]; float aj[3] = {E.push[6 * e + 3], E.push[6 * e + 4], E.push[6 * e + 5]}; apply_aimp(t, ii, aj); }
      // contact candidates: the corners of each sole box under a small margin
      float cpt[NF][NC][3]; int cact[NF][NC];
      for (int f = 0; f < NF; f++) {
        const float *b = st + M.foot_b[f] * SW;
        for (int c = 0; c < NC; c++) {
          float lc[3] = {M.foot_h[3 * f] * ((c & 1) ? 1.f : -1.f), M.foot_h[3 * f + 1] * ((c & 2) ? 1.f : -1.f), M.foot_h[3 * f + 2] * ((c & 4) ? 1.f : -1.f)}, l2[3], wl[3];
          rot(M.foot_q + 4 * f, lc, l2); for (int k = 0; k < 3; k++) l2[k] += M.foot_c[3 * f + k];
          rot(b + 3, l2, wl); for (int k = 0; k < 3; k++) cpt[f][c][k] = wl[k];   // relative to the body's centre of mass
          const float y = b[1] + wl[1];
          cact[f][c] = y < 0.002f;
          if (!cact[f][c]) for (int k = 0; k < 3; k++) cl[(f * NC + c) * 3 + k] = 0;
        }
      }
      // warm start: joints and contacts
      for (int j = 0; j < NJ; j++) {
        float *p = st + M.jp[j] * SW, *c = st + M.jc[j] * SW; const float *iip = ii + 3 * M.jp[j], *iic = ii + 3 * M.jc[j]; const float imp = im[M.jp[j]], imc = im[M.jc[j]];
        JG g = joint_geometry(p, c, M, j); float t1[3], t2[3]; basis(g.axis, t1, t2); float *L = jl + 8 * j;
        for (int r = 0; r < 3; r++) { float J[3] = {0, 0, 0}, N[3]; J[r] = L[r] * 0.9f; for (int k = 0; k < 3; k++) N[k] = -J[k]; apply_imp(p, imp, iip, g.rp, N); apply_imp(c, imc, iic, g.rc, J); L[r] *= 0.9f; }
        for (int r = 0; r < 2; r++) { const float *d = r ? t2 : t1; float J[3], N[3]; for (int k = 0; k < 3; k++) { J[k] = d[k] * L[3 + r] * 0.9f; N[k] = -J[k]; } apply_aimp(p, iip, N); apply_aimp(c, iic, J); L[3 + r] *= 0.9f; }
        L[5] = 0; L[6] = 0;   // motor and limit restart each substep (motor implicit damper; limit re-detected)
      }
      for (int f = 0; f < NF; f++) for (int c = 0; c < NC; c++) if (cact[f][c]) {
        float *b = st + M.foot_b[f] * SW; const float *iib = ii + 3 * M.foot_b[f]; float *L = cl + (f * NC + c) * 3;
        float J[3] = {L[1] * 0.8f, L[0] * 0.8f, L[2] * 0.8f}; L[0] *= .8f; L[1] *= .8f; L[2] *= .8f; apply_imp(b, im[M.foot_b[f]], iib, cpt[f][c], J);
      }
      // iterations
      for (int it = 0; it < M.iterations; it++) {
        for (int j = 0; j < NJ; j++) {
          float *p = st + M.jp[j] * SW, *c = st + M.jc[j] * SW; const float *iip = ii + 3 * M.jp[j], *iic = ii + 3 * M.jc[j]; const float imp = im[M.jp[j]], imc = im[M.jc[j]];
          JG g = joint_geometry(p, c, M, j); float t1[3], t2[3]; basis(g.axis, t1, t2); float *L = jl + 8 * j;
          // point constraint (3 world axes), Baumgarte
          for (int r = 0; r < 3; r++) {
            float d[3] = {0, 0, 0}; d[r] = 1; float vp[3], vc[3]; pointvel(p, g.rp, vp); pointvel(c, g.rc, vc);
            const float rel = vc[r] - vp[r], k = lin_k(p, imp, iip, g.rp, d) + lin_k(c, imc, iic, g.rc, d);
            const float bias = clampf(M.baumgarte / h * g.sep[r], -M.max_bias, M.max_bias);
            const float dl2 = -(rel + bias) / k; L[r] += dl2; float J[3] = {0, 0, 0}, N[3]; J[r] = dl2; for (int q = 0; q < 3; q++) N[q] = -J[q];
            apply_imp(p, imp, iip, g.rp, N); apply_imp(c, imc, iic, g.rc, J);
          }
          // the two angular rows perpendicular to the axis
          for (int r = 0; r < 2; r++) {
            const float *d = r ? t2 : t1; float wr[3]; for (int k = 0; k < 3; k++) wr[k] = c[10 + k] - p[10 + k];
            const float rel = dot3(wr, d), k = ang_k(p, iip, d) + ang_k(c, iic, d), bias = clampf(M.baumgarte / h * dot3(g.ang, d), -M.max_bias * 20, M.max_bias * 20);
            const float dl2 = -(rel + bias) / k; L[3 + r] += dl2; float J[3], N[3]; for (int q = 0; q < 3; q++) { J[q] = d[q] * dl2; N[q] = -J[q]; } apply_aimp(p, iip, N); apply_aimp(c, iic, J);
          }
          float wr[3]; for (int k = 0; k < 3; k++) wr[k] = c[10 + k] - p[10 + k];
          const float speed = dot3(wr, g.axis), ka = ang_k(p, iip, g.axis) + ang_k(c, iic, g.axis);
          // the servo's gear as a force-based velocity motor (World2 bamStep 9): damping cg toward vt, |impulse| <= gmax*h
          const float *P = sp + NSP * j; const float *s = sv + NSERVO * j;
          const bool late = P[13] > 0; const float m_on = late ? s[7] : s[4], m_vt = late ? s[7] : s[5];
          if (m_on != 0) {
            const float hc = h * P[10], old = L[5];
            float dlm = (hc * (m_vt - speed) - old) / (1.f + hc * ka);
            const float nw = clampf(old + dlm, -P[11] * h, P[11] * h); dlm = nw - old; L[5] = nw;
            float J[3], N[3]; for (int q = 0; q < 3; q++) { J[q] = g.axis[q] * dlm; N[q] = -J[q]; } apply_aimp(p, iip, N); apply_aimp(c, iic, J);
          }
          // end stops
          const float lo = M.lo[j], hi = M.hi[j];
          if (g.coord < lo || g.coord > hi) {
            const float sgn = g.coord < lo ? 1.f : -1.f, viol = g.coord < lo ? lo - g.coord : g.coord - hi, old = L[6];
            float wr2[3]; for (int k = 0; k < 3; k++) wr2[k] = c[10 + k] - p[10 + k];
            const float sp2 = dot3(wr2, g.axis);
            float dlm = (sgn * fminf(2.f, viol * M.baumgarte / h) - sp2) / ka;
            const float nw = sgn * fmaxf(0.f, sgn * (old + dlm)); dlm = nw - old; L[6] = nw;
            float J[3], N[3]; for (int q = 0; q < 3; q++) { J[q] = g.axis[q] * dlm; N[q] = -J[q]; } apply_aimp(p, iip, N); apply_aimp(c, iic, J);
          }
        }
        // contacts: normal then friction (Coulomb, the normal impulse's cone per corner)
        for (int f = 0; f < NF; f++) for (int c = 0; c < NC; c++) if (cact[f][c]) {
          const int bi = M.foot_b[f]; float *b = st + bi * SW; const float *iib = ii + 3 * bi; const float *r = cpt[f][c]; float *L = cl + (f * NC + c) * 3;
          float v[3]; pointvel(b, r, v);
          const float depth = -(b[1] + r[1]);
          const float n[3] = {0, 1, 0}, kn = lin_k(b, im[bi], iib, r, n);
          const float bias = depth > M.contact_slop ? -fminf(M.contact_beta / h * (depth - M.contact_slop), M.max_bias) : 0.f;
          float dn = -(v[1] + bias) / kn; const float on2 = fmaxf(0.f, L[0] + dn); dn = on2 - L[0]; L[0] = on2;
          float J[3] = {0, dn, 0}; apply_imp(b, im[bi], iib, r, J);
          pointvel(b, r, v);
          for (int t = 0; t < 2; t++) {
            float d[3] = {t == 0 ? 1.f : 0.f, 0, t == 1 ? 1.f : 0.f}; const float kt = lin_k(b, im[bi], iib, r, d);
            float dt2 = -v[t == 0 ? 0 : 2] / kt; const float lim = mu * L[0], nt = clampf(L[1 + t] + dt2, -lim, lim); dt2 = nt - L[1 + t]; L[1 + t] = nt;
            float Jt[3] = {d[0] * dt2, 0, d[2] * dt2}; apply_imp(b, im[bi], iib, r, Jt);
            pointvel(b, r, v);
          }
        }
      }
      // contact bookkeeping (normal force, slip) and integration
      for (int f = 0; f < NF; f++) for (int c = 0; c < NC; c++) if (cact[f][c]) {
        const float *L = cl + (f * NC + c) * 3; if (L[0] > 0) { contact_last[f] = 1; force_acc[f] += L[0] / h; float v[3]; pointvel(st + M.foot_b[f] * SW, cpt[f][c], v); slip_acc[f] += sqrtf(v[0] * v[0] + v[2] * v[2]) * L[0] / h; }
      }
      for (int b = 0; b < NB; b++) { float *x = st + b * SW; for (int k = 0; k < 3; k++) x[k] += x[7 + k] * h; integrate_q(x, h); }
    }
    // ---- the servo step (after the physics step, as World2's step hook)
    for (int j = 0; j < NJ; j++) {
      float *s = sv + NSERVO * j; const float *P = sp + NSP * j;
      // goals written at the policy tick apply after goal_delay outer steps (World2: write time + command delay <= time)
      if (outer + 1 == E.goal_delay[e]) s[3] = E.goal_new[e * NJ + j];
      const float *p = st + M.jp[j] * SW, *c = st + M.jc[j] * SW; JG g = joint_geometry(p, c, M, j);
      float tau; servo_step(s, P, g.coord, on, dt, &tau, power); out[OUT_TAU + j] = tau;
    }
    // ---- the IMU (World2's sensing hook, after the servo step)
    imu_step(E.imu + (size_t)e * NIMU, E.imu_par + 8 * e, E.imu_bias + 6 * e, rng, st + M.imu_b * SW, M, dt);
  }
  const float steps = (float)(n_outer * M.substeps);
  for (int f = 0; f < NF; f++) {
    out[OUT_CONTACT + f] = (float)contact_last[f]; out[OUT_FORCE + f] = force_acc[f] / steps; out[OUT_SLIP + f] = force_acc[f] > 1e-6f ? slip_acc[f] / force_acc[f] : 0.f;
    // lowest corner height of the sole now
    const float *b = st + M.foot_b[f] * SW; float lo = 1e9f;
    for (int c = 0; c < NC; c++) { float lc[3] = {M.foot_h[3 * f] * ((c & 1) ? 1.f : -1.f), M.foot_h[3 * f + 1] * ((c & 2) ? 1.f : -1.f), M.foot_h[3 * f + 2] * ((c & 4) ? 1.f : -1.f)}, l2[3], wl[3]; rot(M.foot_q + 4 * f, lc, l2); for (int k = 0; k < 3; k++) l2[k] += M.foot_c[3 * f + k]; rot(b + 3, l2, wl); lo = fminf(lo, b[1] + wl[1]); }
    out[OUT_SOLE + f] = lo;
  }
  out[OUT_POWER] = power / n_outer;
  E.rng[e] = rng;
}

}  // namespace duck
