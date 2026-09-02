"""Reward v5 for the fixed-base arm reach task (walk/env/arm_reach.py).

HISTORY (each version's failure is a measured fact, not a guess):
  v1  gaussian at sigma = 0.10 * reach + linear far-field + acquisition
      bonus + small regularizers. First KR240 GPU leg (runs/gpu/20260902-
      142436-arm-reach-kr240, 126M steps): SHUFFLE -- reward -0.25 -> +0.9,
      judge 0/12, min tip distance 1.44 cm but never a 14-boundary hold,
      joint speeds to 1.81x the URDF limit on a6. sigma = 35 cm saturates
      far above the judge's 2 cm scale (no gradient where it matters).
  v2  + tight gaussian at the acquisition radius rho, + hold bonus,
      boundary speed excess above 1.0x. 8.3M-step CPU run (runs/arm-
      local-v2): 0 acquisitions; the actor hovered at a median 7.9 cm
      where v2 was FLAT, and 66 % of TICKS exceeded the speed limit while
      boundary speeds stayed <= 1.03x -- the overshoot is intra-step (a
      stiff PD answering a full-slew target step peaks at ~1.8x the
      commanded average), invisible to a boundary-only qd penalty.
  v3  + mid gaussian at 4 rho, + COMMANDED-speed penalty (weight 0.5) on
      the slew-clamped target increment (which both paths see exactly),
      no alive term: existing cost a noisy policy up to -3/step, so at
      update ~350 (runs/arm-local-v3) the population learned to crash
      into the floor proxy at step 20 -- the suicide pathology.
  v4  commanded-speed 0.1, alive 0.5, proxy 10. 8.3M-step CPU run
      (runs/arm-local-v4): still 0 acquisitions (median 4.5 cm, 7.8 % of
      steps inside 2 cm, max hold 8, wrist std still ~1.0 pre-tanh, 66 %
      speed-violating ticks). STOCHASTIC rollouts of that actor put the
      tip inside 2 cm on 4.4 % of steps with a max hold of 6 and 1.84 cm
      of exploration jitter per step: a 14-boundary hold is never
      SAMPLED, and between 2 and 5 cm the reward was still flat (tight
      gaussian 0.006 at 4.5 cm, mid gaussian 0.73 -> 0.94).
  v5  geometric gaussian ladder (rho, 2, 4, 8 rho) + per-joint commanded-
      speed weights (0.25 on the wrist). 7.1M-step CPU run (runs/arm-
      local-v5): violating ticks 2356 -> ~2150 (falling), but the same
      plateau -- median 4.8 cm, max hold 8, 1.7 cm/step stochastic
      jitter, 0 acquisitions: the hold bonus needs 14 consecutive
      in-radius boundaries, which the training-time policy never
      SAMPLES, so it teaches nothing until the std has already fallen.
      BUT the same reward at 128 envs for 1200 s (runs/arm-local-v5-long)
      starts acquiring at ~20M env-steps (acq/ep 0.02-0.12 and rising,
      violating ticks falling): a BUDGET floor, not a shaping wall. A v6
      experiment adding a dense "inside and still" term (W 2.0 at 2 rho,
      scaled by 1 - mean u^2) was tried and REVERTED: 6.9M steps, 0
      acquisitions and violating ticks rising 2400 -> 2600.

v5 (shipped): constant gradient in LOG distance from 16 cm down to 1 cm,
and a commanded-speed term heavy enough on the tip-blind wrist (a6 is a
pure tool roll -- nothing but this term ever quiets its exploration noise,
the source of the 1.8x speed peaks). Per policy step (0.02 s), d = |tip -
target|, R = reach (variant), rho = the judge's acquisition radius
(ACQ_RADIUS_M[variant], 2 cm / 1.5 cm), h = consecutive in-radius policy
boundaries INCLUDING this one (the env's hold counter after its update; 0
when outside), H = ACQ_HOLD_STEPS = 14, u_j = (target_j(t) - target_j(t-1))
/ MAX_TARGET_INCREMENT_j the slew-clamped commanded speed as a fraction of
the URDF limit (|u_j| <= 1; 0 for done envs):
  + W_ALIVE                                         living pays: with the
                                                    far-field gaussian at
                                                    reset distance (~0.2
                                                    at 43 cm) and the
                                                    worst-case command
                                                    penalty (-1.05) a
                                                    live env still nets
                                                    > 0, so a proxy crash
                                                    forfeits reward
                                                    instead of escaping a
                                                    penalty (v3 lesson)
  + W_DIST * exp(-(d / (DIST_SIGMA_FRAC * R))^2)   far-field shaping (v1)
  + W_SCALE * sum_k exp(-(d / (DIST_SCALE_RADII[k] * rho))^2)
        geometric LADDER of gaussians at sigma = rho, 2 rho, 4 rho, 8 rho
        (2/4/8/16 cm on the KR240): each octave of approach earns about
        one more unit, so the gradient is ~constant in log d all the way
        from 16 cm to 1 cm -- the flat 2-8 cm shelf that stalled v2/v4 is
        gone -- and the curvature at every scale is what pushes the
        exploration std down (a 1.8 cm jitter costs real reward at 4 cm)
  - W_LIN  * d / R                                  far-field gradient (v1)
  + W_HOLD * min(h, H) / H         (h >= 1)         dense per-boundary hold
                                                    bonus growing with the
                                                    consecutive count: a
                                                    full 14-boundary hold
                                                    earns W_HOLD*7.5, so
                                                    holding is the attractor
  + W_ACQUIRE            on the step a target is acquired (env switches)
  - sum_j W_COMMAND_SPEED_J[j] * u_j^2
        the judge's speed clause is violated by the PD's response to
        full-slew commands (measured peak/average 1.8x on the wrist), so
        commanding full speed IS the violation: 0.1 on a1-a3 (a purposeful
        transfer at the IK baseline's 0.8x on two joints costs 0.13/step),
        0.25 on the wrist a4-a6 (bang-bang full-slew wrist jitter costs
        0.75/step -- 60x what v1's action-rate term charged it); all six
        saturating costs 1.05/step < W_ALIVE + far-field
  - sum_j W_ACTION_RATE_J[j] * (a_j - a_prev_j)^2   0.05 on a1-a3, 0.20 on
                                                    the wrist a4-a6 (jitter)
  - W_TORQUE * mean_j (tau_j / cap_j)^2             PD torque estimate
  - W_SPEED * sum_j max(0, |qd_j| / vlim_j - SPEED_FRAC)^2
        boundary quadratic excess above SPEED_FRAC = 1.0 x the URDF limit
        (the judge's clause 4 boundary). SIZING: the maximum dense reward
        per step is W_ALIVE + W_DIST + 4 * W_SCALE + W_HOLD = 7.0 (on
        target, full hold); the observed 1.8x excess on ONE joint costs
        W_SPEED * (1.8 - 1.0)^2 = 11.0 * 0.64 = 7.04 > 7.0, so a speed
        violation of that size at a boundary can never be bought with
        distance reward (1.5x costs 2.75; 1.1x costs 0.11)
  - W_PROXY              on a proxy-violation step (the env terminates):
                         10 = about 10 steps of the alive term, on top of
                         the forfeited remainder of the episode
Every constant here is ALSO emitted into the generated arm headers (DW_RW_*
block, experimental/duck_cuda/tools/generate_model_arm.py) and the kernel's
DW_ENV_KIND_REACH policy layer ports this function operation for operation
(arm/tests/test_arm_device_policy.py pins bit-exact parity), so a change
here fails the header-drift test until regenerated.
"""
from __future__ import annotations

import numpy as np

W_ALIVE = 1.0
W_DIST = 1.0
DIST_SIGMA_FRAC = 0.10
W_SCALE = 1.0
DIST_SCALE_RADII = (1.0, 2.0, 4.0, 8.0)
W_LIN = 0.5
W_HOLD = 1.0
W_ACQUIRE = 10.0
W_COMMAND_SPEED_J = (0.1, 0.1, 0.1, 0.25, 0.25, 0.25)
W_ACTION_RATE_J = (0.05, 0.05, 0.05, 0.2, 0.2, 0.2)
W_TORQUE = 0.05
W_SPEED = 11.0
SPEED_FRAC = 1.0
W_PROXY = 10.0
VERSION = 5

CONSTANT_NAMES = ("W_ALIVE", "W_DIST", "DIST_SIGMA_FRAC", "W_SCALE", "W_LIN",
                  "W_HOLD", "W_ACQUIRE", "W_TORQUE", "W_SPEED", "SPEED_FRAC",
                  "W_PROXY")
TABLE_NAMES = ("DIST_SCALE_RADII", "W_COMMAND_SPEED_J", "W_ACTION_RATE_J")


def reward(cur: dict, action: np.ndarray, prev_action: np.ndarray,
           reach: float, effort_cap: np.ndarray, velocity_limit: np.ndarray,
           acq_radius: float, hold_steps: int) -> np.ndarray:
    """[E] f64 reward. cur: {'tip_dist' [E], 'hold' [E] int (consecutive
    in-radius boundaries incl. this one, 0 outside), 'acquired' [E] bool,
    'command_speed' [E,J] (slew-clamped target increment / max increment,
    0 for done envs), 'torque' [E,J], 'qd' [E,J], 'proxy_violation' [E]
    bool}."""
    d = np.asarray(cur["tip_dist"], float)
    sigma = DIST_SIGMA_FRAC * float(reach)
    r = W_DIST * np.exp(-(d / sigma) ** 2) - W_LIN * d / float(reach)
    r = r + W_ALIVE
    for k in DIST_SCALE_RADII:              # geometric ladder, in table order
        r = r + W_SCALE * np.exp(-(d / (float(k) * float(acq_radius))) ** 2)
    u = np.asarray(cur["command_speed"], float)
    hold = np.minimum(np.asarray(cur["hold"], np.int64), int(hold_steps))
    r = r + W_HOLD * hold / float(hold_steps)
    r = r + W_ACQUIRE * np.asarray(cur["acquired"], bool)
    r = r - (np.asarray(W_COMMAND_SPEED_J, float) * (u * u)).sum(1)
    da = np.asarray(action, float) - np.asarray(prev_action, float)
    r = r - (np.asarray(W_ACTION_RATE_J, float) * (da * da)).sum(1)
    tq = np.asarray(cur["torque"], float) / np.asarray(effort_cap, float)
    r = r - W_TORQUE * (tq * tq).mean(1)
    over = np.maximum(0.0, np.abs(np.asarray(cur["qd"], float))
                      / np.asarray(velocity_limit, float) - SPEED_FRAC)
    r = r - W_SPEED * (over * over).sum(1)
    r = r - W_PROXY * np.asarray(cur["proxy_violation"], bool)
    return r
