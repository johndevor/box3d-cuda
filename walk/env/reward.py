"""Reward for velocity-commanded flat-floor duck walking.

Called once per policy step (dt = 0.02 s). All terms are vectorised over E
envs. Prior runs proved that survival-only rewards produce a lunge with feet
never leaving the ground, so explicit gait-shaping terms (air time, foot
clearance, double-support penalty, footfall alternation) are mandatory.

State dicts (numpy arrays over E envs):
    root_lin_vel [E,3] world-frame base linear velocity (m/s)
    root_ang_vel [E,3] world-frame base angular velocity (rad/s)
    foot_contact [E,2] bool (left, right) foot-vs-floor contact flags
    sole_height  [E,2] min world z over each foot's sole vertices (m)
    action       [E,14] the action that produced this state
    torque       [E,14] per-joint PD torque estimate (N m), current state only

`command` is the commanded forward (world +x) velocity per env, [E] m/s.
"""
from __future__ import annotations

import numpy as np

CONTROL_DT = 0.02

# ---- weights (one-line rationale each) -----------------------------------
W_TRACK = 1.0            # primary objective: match commanded forward speed.
TRACK_SIGMA_SQ = 0.01    # exp(-err^2/0.01): ~0.05 m/s error still scores 0.78.
W_ALIVE = 0.5            # small survival bonus so early falls are dominated.
W_LATERAL = 0.5          # penalise sideways drift (vy^2) + yaw spin (wz^2).
W_ACTION_RATE = 0.01     # discourage bang-bang actions; keeps slew headroom.
W_TORQUE = 2e-4          # mild effort cost; cap is 3.23 Nm so sum(tau^2)<=146.
W_AIR_TIME = 1.0         # per qualified touchdown: swings must last 0.1-0.4 s.
AIR_TIME_MIN = 0.10      # below this a touchdown is a shuffle, not a step.
AIR_TIME_MAX = 0.40      # above this the duck is hopping/ballistic, not walking.
W_CLEARANCE = 0.1        # per swing foot per step whose whole sole clears >=10 mm.
CLEARANCE_M = 0.010      # matches the strict evaluator's sole-clearance bound.
W_DOUBLE_SUPPORT = 0.5   # penalty per step once both feet stay grounded too long.
DOUBLE_SUPPORT_GRACE = 0.25  # s of continuous double support tolerated at |cmd|>0.
W_ALTERNATE = 0.5        # extra bonus when a qualified touchdown switches feet.


class GaitTracker:
    """Per-env stateful bookkeeping for the gait-shaping terms.

    The environment owns one tracker; `reset(mask)` must be called whenever
    the corresponding envs are reset so no gait credit crosses episodes.
    """

    def __init__(self, environments: int):
        self.E = int(environments)
        self.air_time = np.zeros((self.E, 2))        # s since liftoff per foot
        self.double_support = np.zeros(self.E)       # s of continuous double support
        self.last_foot = np.full(self.E, -1, np.int64)  # last qualified footfall

    def reset(self, mask: np.ndarray | None = None) -> None:
        m = np.ones(self.E, bool) if mask is None else np.asarray(mask, bool)
        self.air_time[m] = 0.0
        self.double_support[m] = 0.0
        self.last_foot[m] = -1


def reward(prev_state: dict, state: dict, action: np.ndarray, command: np.ndarray,
           tracker: GaitTracker, dt: float = CONTROL_DT) -> np.ndarray:
    """Per-env float32 reward; updates `tracker` in place."""
    a = np.asarray(action, dtype=np.float64)
    cmd = np.asarray(command, dtype=np.float64)
    vx = state["root_lin_vel"][:, 0]
    vy = state["root_lin_vel"][:, 1]
    wz = state["root_ang_vel"][:, 2]
    contact = np.asarray(state["foot_contact"], bool)
    prev_contact = np.asarray(prev_state["foot_contact"], bool)
    sole = state["sole_height"]

    # 1. forward-velocity tracking
    r = W_TRACK * np.exp(-np.square(vx - cmd) / TRACK_SIGMA_SQ)
    # 2. alive bonus
    r += W_ALIVE
    # 3. lateral / yaw velocity penalty
    r -= W_LATERAL * (np.square(vy) + np.square(wz))
    # 4. action-rate penalty
    r -= W_ACTION_RATE * np.square(a - np.asarray(prev_state["action"])).sum(1)
    # 5. torque penalty
    r -= W_TORQUE * np.square(np.asarray(state["torque"])).sum(1)

    # 6. per-foot air-time bonus on touchdown, and 9. alternation bonus
    touchdown = ~prev_contact & contact                       # [E,2]
    qualified = touchdown & (tracker.air_time >= AIR_TIME_MIN) \
        & (tracker.air_time <= AIR_TIME_MAX)
    r += W_AIR_TIME * qualified.sum(1)
    for foot in (0, 1):
        hit = qualified[:, foot]
        r += W_ALTERNATE * (hit & (tracker.last_foot == 1 - foot))
        tracker.last_foot[hit] = foot
    # air-time accounting: grows while airborne, clears while in contact
    tracker.air_time = np.where(contact, 0.0, tracker.air_time + dt)

    # 7. foot-clearance bonus: swing foot whose whole sole clears >= 10 mm
    r += W_CLEARANCE * ((~contact) & (sole >= CLEARANCE_M)).sum(1)

    # 8. double-support penalty beyond the duty grace while commanded to move
    both = contact.all(1)
    tracker.double_support = np.where(both, tracker.double_support + dt, 0.0)
    r -= W_DOUBLE_SUPPORT * ((np.abs(cmd) > 0)
                             & (tracker.double_support > DOUBLE_SUPPORT_GRACE))
    return r.astype(np.float32)
