"""Behavior-cloning dataset: the reference gait rolled CLOSED over real obs.

For a spread of (seed, command) configs -- phase0 varies per (seed, env,
episode) through the env's own counter-based stream, plus small joint-offset
state noise -- replay reference-driven actions through FlatFloorHumanoidEnv
on the CPU lane and record (obs[52], action[12]) at every LIVE policy step.

The action label is the demonstrator policy's output at this obs:

    a = clip((ref_q(bin(phase + LEAD*2*pi*hz*CONTROL_DT)) - HOME)
             / ACTION_SCALE  +  ankle_balance_assist, -1, 1)

computed FROM THE OBSERVATION alone -- phase/command/gravity/rate
channels at their J-derived offsets; no privileged state leaks into the
labels. Two measured
refinements over the raw next-phase reference lookup (which produced a
non-stepping clone -- the open-loop demonstrator tipped before its first
alternation):

  LEAD = 2 policy steps: the PD + slew chain lags targets by ~2 steps;
  leading the phase keeps the PHYSICAL joints near the reference at its
  own phase (demonstrator alternation improved measurably).

  balance assists (gains swept across commands x seeds for max survival
  and alternating lifts; all channel indices J-derived): sagittal --
  pitch_assist = -ANKLE_KP*grav_x on both ankles plus HIP_PITCH_SHARE of
  it on both hip pitches (ankles alone saturate their 140 N*m cap once
  the lean grows; the hips share the recovery); lateral (H1) --
  roll_assist = ROLL_KP*grav_lat + ROLL_KD*roll_rate on both hip rolls,
  the channel H0 lacked. Feedback in the labels is the point of closing
  the loop over real obs -- the clone learns reference-tracking AND the
  reflexes. The demonstrator still falls in ~1 s (open-loop-ish); the
  dataset intentionally contains that honest instability. PPO owns
  survival.

The knee plateau (0.6 rad) saturates at a = 1.0 (target 0.5 = the action
box edge): labels and rollout stay mutually consistent because the SAME
clipped actions drive the recording env, and the imitation reward's
shortfall from the unreachable 0.1 rad is < 2% of the bonus at
IMIT_SIGMA. The env's slew (0.16 rad/step) shapes targets identically at
recording and deployment time.

Obs are REAL-LANE observations (gravity/velocity/contact channels with the
genuine early instability of open-ish-loop stepping), not synthetic FK
obs. Episodes record until termination or `steps`; done envs stop
contributing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "humanoid") not in sys.path:
    sys.path.insert(0, str(ROOT / "humanoid"))

from walk.env import humanoid_flat as hf  # noqa: E402
from walk.env import humanoid_reward as hr  # noqa: E402

import h1_lowering as lowering  # noqa: E402  (ACTIVE lowering: H1)
import h1_family  # noqa: E402  (family variants: their own REF + lane)

REF = np.asarray(hr.REF_GAIT, dtype=np.float64)          # [64, J]
BINS = int(hr.REF_BINS)
LEAD_STEPS = 2            # PD + slew lag compensation (module docstring)
# balance-assist gains (swept for max alternating lifts / survival):
ANKLE_KP = 3.0            # ankle pitch vs gravity-x
ANKLE_KD = 0.0            # ankle pitch vs body-frame pitch rate
HIP_PITCH_SHARE = 1.2     # hips carry 1.2x the ankle pitch assist
ROLL_KP = 0.0             # v3.2: OFF. The reference's overdrive/taper roll
ROLL_KD = 0.0             # owns the weight transfer and kp_roll 500 holds
#                           it; the crude lateral assist (v2-era 5.0/-0.3,
#                           swept when the reference could not transfer)
#                           now actively degrades executed swings --
#                           measured 23 mm vs 45 mm peak clearance.
# J-derived obs channel indices (walk/env/humanoid_flat.py layout)
_T = 3 * hf.ACT
IDX_GRAV_X = _T           # forward gravity component (pitch lean)
IDX_GRAV_LAT = _T + 2     # lateral gravity component (roll lean)
IDX_ROLL_RATE = _T + 3    # body-frame omega x
IDX_PITCH_RATE = _T + 5   # body-frame omega z (sagittal axes are local z)
IDX_CMD = _T + 9
IDX_SIN, IDX_COS = hf.OBS - 2, hf.OBS - 1
_JN = list(lowering.JOINT_NAMES)
ANKLES = (_JN.index("left_ankle"), _JN.index("right_ankle"))
HIPS = (_JN.index("left_hip"), _JN.index("right_hip"))
ROLLS = ((_JN.index("left_hip_roll"), _JN.index("right_hip_roll"))
         if "left_hip_roll" in _JN else ())


def reference_table(variant: str | None = None) -> np.ndarray:
    """[64, J] reference for a family member (base: the module REF)."""
    if h1_family.is_base(variant):
        return REF
    return hr.load_reference(h1_family.reference_gait_path(variant))


def reference_actions(obs: np.ndarray, ref: np.ndarray | None = None
                      ) -> np.ndarray:
    """[E, J] demonstrator actions from the observations alone:
    lead-compensated reference tracking + ankle-pitch and (H1) hip-roll
    balance assists. `ref` = the member's table (default: H1 REF)."""
    table = REF if ref is None else np.asarray(ref, np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    phase = np.arctan2(obs[:, IDX_SIN], obs[:, IDX_COS])   # [-pi, pi]
    hz = hf.PHASE_HZ_BASE + hf.PHASE_HZ_PER_MPS * obs[:, IDX_CMD]
    frac = np.mod(phase / (2.0 * np.pi)
                  + LEAD_STEPS * hz * hf.CONTROL_DT, 1.0)
    bins = (frac * BINS).astype(int) % BINS
    a = (table[bins] - hf.HOME) / hf.ACTION_SCALE
    pitch_assist = -(ANKLE_KP * obs[:, IDX_GRAV_X]
                     + ANKLE_KD * obs[:, IDX_PITCH_RATE])
    for j in ANKLES:
        a[:, j] += pitch_assist
    for j in HIPS:
        a[:, j] += HIP_PITCH_SHARE * pitch_assist
    roll_assist = (ROLL_KP * obs[:, IDX_GRAV_LAT]
                   + ROLL_KD * obs[:, IDX_ROLL_RATE])
    for j in ROLLS:
        a[:, j] += roll_assist
    return np.clip(a, -1.0, 1.0)


def rollout_pairs(env, steps: int, ref: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Roll reference actions closed-loop; (obs [N,52], act [N,12]) pairs
    from live steps only (a step's pair is recorded iff the env was live
    when the action was applied)."""
    obs = env.reset()
    live = np.ones(env.E, bool)
    xs, ys = [], []
    for _ in range(steps):
        a = reference_actions(obs, ref)
        if live.any():
            xs.append(obs[live].astype(np.float32))
            ys.append(a[live].astype(np.float32))
        obs, _, done, _ = env.step(a)
        live &= ~np.asarray(done, bool)
        if not live.any():
            break
    return (np.concatenate(xs) if xs else np.zeros((0, hf.OBS), np.float32),
            np.concatenate(ys) if ys else np.zeros((0, hf.ACT), np.float32))


def default_lane_factory(E, offsets, variant: str | None = None):
    """fp32 CPU-serial humanoid lane: the SAME physics the GPU legs train
    on (obs distribution match for the BC init) and ~500x faster than the
    f64 oracle lane on contact-heavy stepping. Pass lane_factory=None to
    build_dataset for this default; pass an explicit factory (e.g. the f64
    NativeHumanoidLane) to override. `variant` selects the family member's
    build (base: unchanged call)."""
    from walk.env.humanoid_cuda_lane import CudaHumanoidLane  # noqa: PLC0415
    if h1_family.is_base(variant):
        return CudaHumanoidLane(E, joint_offsets=offsets)
    return CudaHumanoidLane(E, joint_offsets=offsets, variant=variant)


def build_dataset(seeds=(11, 22, 33, 44), commands=hf.COMMANDS_MPS,
                  envs_per_config: int = 4, steps: int = 60,
                  perturbation_rad: float = 0.02,
                  library_path=None, lane_factory=None,
                  variant: str | None = None) -> dict:
    """(obs, act) pairs across a (seed x command) spread.

    Per config: `envs_per_config` envs (each with its OWN counter-drawn
    phase0 and +-perturbation_rad joint-offset noise), pinned to `command`,
    rolled `steps` policy steps (~ up to steps*0.02 s; 3 cycles at cmd 0.75
    is 1.2 s = 60 steps). Returns {"obs", "act", "meta"}. `variant`: family
    member (its lane, its reference table; None = H1.1, unchanged).
    """
    ref = reference_table(variant)
    if lane_factory is None and library_path is None:
        lane_factory = (default_lane_factory if h1_family.is_base(variant)
                        else (lambda E, off: default_lane_factory(E, off, variant)))
    all_obs, all_act, per_config = [], [], {}
    for seed in seeds:
        env = hf.FlatFloorHumanoidEnv(
            environments=envs_per_config, seed=int(seed),
            perturbation_rad=perturbation_rad,
            library_path=library_path, lane_factory=lane_factory,
            variant=variant)
        try:
            for cmd in commands:
                env.reset()
                env.set_command(cmd)
                # set_command after reset pins every env to cmd while each
                # keeps its own episode-drawn phase0/noise
                obs, act = rollout_pairs(env, steps, ref)
                per_config[f"seed{seed}-cmd{cmd:.2f}"] = int(len(obs))
                all_obs.append(obs)
                all_act.append(act)
        finally:
            env.close()
    obs = np.concatenate(all_obs)
    act = np.concatenate(all_act)
    return {"obs": obs, "act": act,
            "meta": {"pairs": int(len(obs)), "per_config": per_config,
                     "seeds": list(map(int, seeds)),
                     "commands": [float(c) for c in commands],
                     "envs_per_config": envs_per_config, "steps": steps,
                     "perturbation_rad": perturbation_rad,
                     "variant": h1_family.canonical(variant)}}
