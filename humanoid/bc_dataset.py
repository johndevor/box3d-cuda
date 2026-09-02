"""Behavior-cloning dataset: the reference gait rolled CLOSED over real obs.

For a spread of (seed, command) configs -- phase0 varies per (seed, env,
episode) through the env's own counter-based stream, plus small joint-offset
state noise -- replay reference-driven actions through FlatFloorHumanoidEnv
on the CPU lane and record (obs[52], action[12]) at every LIVE policy step.

The action label is the demonstrator policy's output at this obs:

    a = clip((ref_q(bin(phase + LEAD*2*pi*hz*CONTROL_DT)) - HOME)
             / ACTION_SCALE  +  ankle_balance_assist, -1, 1)

computed FROM THE OBSERVATION alone -- phase channels (obs[50:52]),
command (obs[45]), gravity-x (obs[36]) and body-frame pitch rate
(obs[41]); no privileged state leaks into the labels. Two measured
refinements over the raw next-phase reference lookup (which produced a
non-stepping clone -- the open-loop demonstrator tipped before its first
alternation):

  LEAD = 2 policy steps: the PD + slew chain lags targets by ~2 steps;
  leading the phase keeps the PHYSICAL joints near the reference at its
  own phase (demonstrator alternation improved measurably).

  ankle_balance_assist g = -(2.0*obs[36] + 0.1*obs[41]) added to BOTH
  ankle actions: minimal sagittal balance feedback (gains swept across
  commands x seeds for max alternating lifts). Feedback in the labels is
  the point of closing the loop over real obs -- the clone learns
  reference-tracking AND the reflex. NOTE the morphology has zero roll
  authority (all 12 axes are sagittal) and single support is laterally
  statically unstable by ~1 cm, so EVERY open-loop-ish demonstrator falls
  within ~1 s; the dataset intentionally contains that honest instability.
  PPO owns survival.

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

REF = np.asarray(hr.REF_GAIT, dtype=np.float64)          # [64, 12]
BINS = int(hr.REF_BINS)
LEAD_STEPS = 2            # PD + slew lag compensation (module docstring)
ANKLE_KP = 2.0            # ankle assist vs gravity-x (obs[36]); swept
ANKLE_KD = 0.1            # ankle assist vs body pitch rate (obs[41]); swept
LEFT_ANKLE, RIGHT_ANKLE = 4, 7


def reference_actions(obs: np.ndarray) -> np.ndarray:
    """[E, 12] demonstrator actions from the observations alone:
    lead-compensated reference tracking + ankle balance assist."""
    obs = np.asarray(obs, dtype=np.float64)
    sin_p, cos_p = obs[:, 50], obs[:, 51]
    cmd = obs[:, 45]
    phase = np.arctan2(sin_p, cos_p)                     # [-pi, pi]
    hz = hf.PHASE_HZ_BASE + hf.PHASE_HZ_PER_MPS * cmd
    frac = np.mod(phase / (2.0 * np.pi)
                  + LEAD_STEPS * hz * hf.CONTROL_DT, 1.0)
    bins = (frac * BINS).astype(int) % BINS
    a = (REF[bins] - hf.HOME) / hf.ACTION_SCALE
    assist = -(ANKLE_KP * obs[:, 36] + ANKLE_KD * obs[:, 41])
    a[:, LEFT_ANKLE] += assist
    a[:, RIGHT_ANKLE] += assist
    return np.clip(a, -1.0, 1.0)


def rollout_pairs(env, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Roll reference actions closed-loop; (obs [N,52], act [N,12]) pairs
    from live steps only (a step's pair is recorded iff the env was live
    when the action was applied)."""
    obs = env.reset()
    live = np.ones(env.E, bool)
    xs, ys = [], []
    for _ in range(steps):
        a = reference_actions(obs)
        if live.any():
            xs.append(obs[live].astype(np.float32))
            ys.append(a[live].astype(np.float32))
        obs, _, done, _ = env.step(a)
        live &= ~np.asarray(done, bool)
        if not live.any():
            break
    return (np.concatenate(xs) if xs else np.zeros((0, hf.OBS), np.float32),
            np.concatenate(ys) if ys else np.zeros((0, hf.ACT), np.float32))


def default_lane_factory(E, offsets):
    """fp32 CPU-serial humanoid lane: the SAME physics the GPU legs train
    on (obs distribution match for the BC init) and ~500x faster than the
    f64 oracle lane on contact-heavy stepping. Pass lane_factory=None to
    build_dataset for this default; pass an explicit factory (e.g. the f64
    NativeHumanoidLane) to override."""
    from walk.env.humanoid_cuda_lane import CudaHumanoidLane  # noqa: PLC0415
    return CudaHumanoidLane(E, joint_offsets=offsets)


def build_dataset(seeds=(11, 22, 33, 44), commands=hf.COMMANDS_MPS,
                  envs_per_config: int = 4, steps: int = 60,
                  perturbation_rad: float = 0.02,
                  library_path=None, lane_factory=None) -> dict:
    """(obs, act) pairs across a (seed x command) spread.

    Per config: `envs_per_config` envs (each with its OWN counter-drawn
    phase0 and +-perturbation_rad joint-offset noise), pinned to `command`,
    rolled `steps` policy steps (~ up to steps*0.02 s; 3 cycles at cmd 0.75
    is 1.2 s = 60 steps). Returns {"obs", "act", "meta"}.
    """
    if lane_factory is None and library_path is None:
        lane_factory = default_lane_factory
    all_obs, all_act, per_config = [], [], {}
    for seed in seeds:
        env = hf.FlatFloorHumanoidEnv(
            environments=envs_per_config, seed=int(seed),
            perturbation_rad=perturbation_rad,
            library_path=library_path, lane_factory=lane_factory)
        try:
            for cmd in commands:
                env.reset()
                env.set_command(cmd)
                # set_command after reset pins every env to cmd while each
                # keeps its own episode-drawn phase0/noise
                obs, act = rollout_pairs(env, steps)
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
                     "perturbation_rad": perturbation_rad}}
