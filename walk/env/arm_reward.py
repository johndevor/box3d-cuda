"""Reward v1 for the fixed-base arm reach task (walk/env/arm_reach.py).

Dense tip-distance shaping + acquisition bonus + action-rate + torque
regularizers (the spec), plus a speed regularizer that mirrors the frozen
judge's joint-speed clause and a proxy penalty that mirrors its
self-collision / floor clause. Every constant here is ALSO emitted into the
generated arm headers (DW_ARM_RW_*, experimental/duck_cuda/tools/
generate_model_arm.py) so a python-side change fails the header-drift test
until regenerated -- the same reward <-> header pin the duck and humanoid
carry, even though the arm's obs/reward run python-side today (the kernel's
device policy layer cannot express the arm's obs; arm/FEASIBILITY.md
section 6 states the exact DW_ENV_* extension it would need).

Per policy step (0.02 s), with d = |tip - target|, reach R (variant):
  + W_DIST * exp(-(d / (DIST_SIGMA_FRAC * R))^2)     near-field shaping
  - W_LIN  * d / R                                    far-field gradient
  + W_ACQUIRE            on the step a target is acquired (env switches)
  - W_ACTION_RATE * |a - a_prev|^2
  - W_TORQUE * mean_j (tau_j / cap_j)^2               PD torque estimate
  - W_SPEED * sum_j max(0, |qd_j| / vlim_j - SPEED_FRAC)^2
  - W_PROXY              on a proxy-violation step (the env terminates)
Magnitudes: hovering on target earns ~+1/step (400 steps -> ~400/episode)
plus 5 x W_ACQUIRE for the judge's five targets; the regularizers are
sized to stay below ~0.1/step under normal motion (torque and speed are
normalized to the per-joint caps so both variants see the same scale --
one policy can span the family).
"""
from __future__ import annotations

import numpy as np

W_DIST = 1.0
DIST_SIGMA_FRAC = 0.10
W_LIN = 0.5
W_ACQUIRE = 10.0
W_ACTION_RATE = 0.05
W_TORQUE = 0.05
W_SPEED = 0.5
SPEED_FRAC = 0.9
W_PROXY = 5.0

CONSTANT_NAMES = ("W_DIST", "DIST_SIGMA_FRAC", "W_LIN", "W_ACQUIRE",
                  "W_ACTION_RATE", "W_TORQUE", "W_SPEED", "SPEED_FRAC",
                  "W_PROXY")


def reward(cur: dict, action: np.ndarray, prev_action: np.ndarray,
           reach: float, effort_cap: np.ndarray, velocity_limit: np.ndarray
           ) -> np.ndarray:
    """[E] f64 reward. cur: {'tip_dist' [E], 'acquired' [E] bool,
    'torque' [E,J], 'qd' [E,J], 'proxy_violation' [E] bool}."""
    d = np.asarray(cur["tip_dist"], float)
    sigma = DIST_SIGMA_FRAC * float(reach)
    r = W_DIST * np.exp(-(d / sigma) ** 2) - W_LIN * d / float(reach)
    r = r + W_ACQUIRE * np.asarray(cur["acquired"], bool)
    da = np.asarray(action, float) - np.asarray(prev_action, float)
    r = r - W_ACTION_RATE * (da * da).sum(1)
    tq = np.asarray(cur["torque"], float) / np.asarray(effort_cap, float)
    r = r - W_TORQUE * (tq * tq).mean(1)
    over = np.maximum(0.0, np.abs(np.asarray(cur["qd"], float))
                      / np.asarray(velocity_limit, float) - SPEED_FRAC)
    r = r - W_SPEED * (over * over).sum(1)
    r = r - W_PROXY * np.asarray(cur["proxy_violation"], bool)
    return r
