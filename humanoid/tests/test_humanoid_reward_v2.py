"""Reward v2 anti-attractor pins (lunge-and-slide must strictly lose).

Run: .venv/bin/python -B humanoid/tests/test_humanoid_reward_v2.py

Scores AUTHORED reward-state trajectories through the real
walk/env/humanoid_reward.reward() (same code the header pins feed the
kernel) and pins the property that broke leg-1:

  - "perfect stand+lean" (both feet planted for 8 s, the 0.054 m/s slide
    creep the judge measured on the leg-1 actor): mean per-step reward
    STRICTLY NEGATIVE at every command (v1 paid it +0.28..+0.45);
  - "crude in-phase alternating stepper" (0.6 s cycle, 0.2 s swings,
    qualified-step-shaped but tracking only HALF the command): mean
    per-step reward strictly positive and the stand->step gap decisive
    (> 2.0 per step);
  - the same clean synthetic gait the EVALUATOR passes scores clearly
    higher under v2 than the stand-lean trace;
  - the v2 weight values themselves are pinned (any further change must
    edit this test deliberately);
  - preflight-style sanity: random actions on the real env still produce
    varying rewards (no flat-objective regression);
  - termination: leaning past the frozen judge's 30 deg is now fatal in
    the env (MAX_TILT 28 deg < judge 30 deg; header cos pinned).
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "humanoid"))

from walk.env import humanoid_flat as hf  # noqa: E402
from walk.env import humanoid_reward as hr  # noqa: E402

DT = hr.CONTROL_DT              # 0.02 s policy step
STEPS = 400                     # 8 s episode


def _state(E, contact, vx, foot_x, sole, joint_q=None, phase=0.0,
           contact_ticks=None):
    return {
        "root_lin_vel": np.tile([vx, 0.0, 0.0], (E, 1)),
        "root_ang_vel": np.zeros((E, 3)),
        "foot_contact": np.asarray(contact, bool).reshape(E, 2),
        "sole_height": np.asarray(sole, float).reshape(E, 2),
        "action": np.zeros((E, 12)),
        "torque": np.zeros((E, 12)),
        "foot_x": np.asarray(foot_x, float).reshape(E, 2),
        "joint_q": np.zeros((E, 12)) if joint_q is None else joint_q,
        "phase": np.full(E, float(phase)),
        "contact_ticks": (np.full((E, 2), hr.TICKS_FULL, int)
                          if contact_ticks is None
                          else np.asarray(contact_ticks).reshape(E, 2)),
    }


def stand_lean_total(cmd: float, creep_mps: float = 0.054) -> np.ndarray:
    """Per-step rewards for a perfect 8 s stand+lean (leg-1's strategy)."""
    tracker = hr.GaitTracker(1)
    phase_hz = hf.PHASE_HZ_BASE + hf.PHASE_HZ_PER_MPS * cmd
    prev = _state(1, [True, True], creep_mps, [0.0, 0.0], [0.0, 0.0])
    out = []
    for t in range(STEPS):
        phase = 2.0 * math.pi * phase_hz * t * DT
        cur = _state(1, [True, True], creep_mps, [0.0, 0.0], [0.0, 0.0],
                     phase=phase)
        out.append(float(hr.reward(prev, cur, np.zeros((1, 12)),
                                   np.array([cmd]), tracker)[0]))
        prev = cur
    return np.asarray(out)


def crude_stepper_total(cmd: float) -> np.ndarray:
    """Per-step rewards for a crude in-phase alternating stepper.

    0.6 s cycle (30 policy steps), 0.12 s swings (6 steps each), gait
    locked to its own clock (idealized alignment -- phase0 is free per
    episode, so a stepping policy CAN align; the pin measures the aligned
    payoff): left stance on sin >= 0 (phase cycle first half, touchdown at
    k=0), right on the second half (touchdown at k=15); feet advance 0.3 m
    per own swing, root tracks only HALF the command (crude), sole clears
    40 mm mid-swing.
    """
    tracker = hr.GaitTracker(1)
    cycle, swing = 30, 6                        # policy steps
    foot_x = np.array([0.0, 0.0])
    prev = _state(1, [True, True], 0.5 * cmd, foot_x, [0.0, 0.0])
    out = []
    for t in range(STEPS):
        k = t % cycle
        left_air = 24 <= k < 24 + swing         # ends exactly at k=0 (td)
        right_air = 9 <= k < 9 + swing          # ends at k=15 (td)
        contact = [not left_air, not right_air]
        sole = [0.04 if left_air else 0.0, 0.04 if right_air else 0.0]
        for f, air in enumerate((left_air, right_air)):
            if air:
                foot_x[f] += 0.3 / swing        # 0.3 m per swing (> 0.15)
        phase = 2.0 * math.pi * (k + 0.5) / cycle   # gait-locked clock
        ticks = [0 if left_air else hr.TICKS_FULL,
                 0 if right_air else hr.TICKS_FULL]
        # v2.1: the crude stepper also imitates -- joints track the
        # reference table at its own phase (full W_IMIT bonus)
        bin_ = int(((k + 0.5) / cycle % 1.0) * hr.REF_BINS) % hr.REF_BINS
        jq = np.asarray(hr.REF_GAIT)[bin_][None, :].copy()
        cur = _state(1, contact, 0.5 * cmd, foot_x.copy(), sole,
                     joint_q=jq, phase=phase, contact_ticks=ticks)
        out.append(float(hr.reward(prev, cur, np.zeros((1, 12)),
                                   np.array([cmd]), tracker)[0]))
        prev = cur
    return np.asarray(out)


class AntiAttractorPins(unittest.TestCase):
    def test_v2_weight_values_pinned(self):
        self.assertEqual(hr.TRACK_SIGMA_SQ, 0.09)
        self.assertEqual(hr.W_AIR_TIME, 3.0)
        self.assertEqual(hr.W_DOUBLE_SUPPORT, 1.5)
        self.assertEqual(hr.W_PHASE, 1.0)
        # unchanged-from-v1 spot checks (shape frozen to the kernel's)
        self.assertEqual(hr.W_ALIVE, 0.5)
        self.assertEqual(hr.PLACEMENT_MIN_M, 0.15)
        # v2.1: imitation live from the synthetic reference table
        self.assertEqual(hr.W_IMIT, 0.5)
        self.assertEqual(np.asarray(hr.REF_GAIT).shape, (64, 12))

    def test_stand_lean_strictly_negative_at_every_command(self):
        # v2.1: the stander leaks a mean +0.24 from imitation (the reference
        # passes near HOME twice per cycle) yet stays STRICTLY negative
        # every settled step (measured -0.65 / -0.76 / -0.76 mean).
        for cmd in hf.COMMANDS_MPS:
            r = stand_lean_total(cmd)
            settled = r[int(0.5 / DT):]         # past EMA + grace transients
            self.assertLess(float(settled.mean()), -0.60, cmd)
            self.assertLess(float(settled.max()), -0.35, cmd)
            self.assertLess(float(r.sum()), 0.0, cmd)  # whole episode loses

    def test_crude_stepper_beats_standing_decisively(self):
        # v2.1: the imitating stepper collects the full +0.5 W_IMIT vs the
        # stander's +0.24 leak -> the v2 gap WIDENS (measured >= 2.8/step).
        for cmd in hf.COMMANDS_MPS:
            step = crude_stepper_total(cmd)
            stand = stand_lean_total(cmd)
            gap = float(step.mean() - stand.mean())
            self.assertGreater(float(step.mean()), 1.5, cmd)
            self.assertGreater(gap, 2.5, (cmd, step.mean(), stand.mean()))
        print("stand vs step (mean/step): " + ", ".join(
            f"cmd {c:.2f}: {stand_lean_total(c).mean():+.2f} -> "
            f"{crude_stepper_total(c).mean():+.2f}" for c in hf.COMMANDS_MPS),
            file=sys.stderr)

    def test_evaluator_clean_gait_outscores_stand_lean(self):
        """The trajectory the FROZEN judge passes must also be the reward's
        preference -- ties the two artifacts together."""
        from humanoid.tests.test_humanoid_gait_eval import synthetic_trace
        from walk.eval.humanoid_gait import evaluate_episode
        trace = synthetic_trace(cmd=0.5)        # judge-passing gait
        self.assertTrue(evaluate_episode(trace)["passed"])
        # score it through the reward at policy cadence (every 10th tick)
        ticks = trace["ticks"]
        contact = np.asarray(ticks["contact"], bool)[9::10]
        sole = np.asarray(ticks["sole_height"], float)[9::10]
        foot_pos = np.asarray(ticks["foot_pos"], float)[9::10]
        n = len(contact)
        tick_contact = np.asarray(ticks["contact"], bool).reshape(-1, 10, 2)
        ticks_full = tick_contact.sum(axis=1)
        tracker = hr.GaitTracker(1)
        cmd = 0.5
        # gait-locked idealized phase for the trace's 0.6 s cycle: left
        # touchdown at cycle time 0.3 s must open the sin>=0 window
        # (phase0 is a free per-episode offset, so alignment is reachable)
        prev = _state(1, contact[0], cmd, foot_pos[0, :, 0], sole[0])
        rewards = []
        for t in range(1, n):
            tt = (t + 1) * 10 * 0.002           # trace time of this step
            phase = 2.0 * math.pi * ((tt - 0.3) % 0.6) / 0.6 + 0.1
            cur = _state(1, contact[t], cmd, foot_pos[t, :, 0], sole[t],
                         phase=phase, contact_ticks=ticks_full[t])
            rewards.append(float(hr.reward(prev, cur, np.zeros((1, 12)),
                                           np.array([cmd]), tracker)[0]))
            prev = cur
        clean = float(np.mean(rewards))
        stand = float(stand_lean_total(cmd).mean())
        self.assertGreater(clean, 1.0)
        self.assertGreater(clean - stand, 2.0)
        print(f"judge-passing gait vs stand+lean (cmd 0.5): "
              f"{clean:+.2f} vs {stand:+.2f} per step", file=sys.stderr)

    def test_random_action_reward_still_varies(self):
        env = hf.FlatFloorHumanoidEnv(environments=2, seed=99)
        try:
            rng = np.random.default_rng(0xF11)
            rews = []
            for _ in range(10):
                _, r, done, _ = env.step(
                    rng.uniform(-1.0, 1.0, (2, 12)).astype(np.float32))
                rews.extend(np.asarray(r)[~np.asarray(done)])
                if np.asarray(done).all():
                    env.reset()
            rews = np.asarray(rews)
            self.assertGreaterEqual(rews.size, 10)
            self.assertGreater(float(rews.std()), 1e-7)
        finally:
            env.close()

    def test_termination_tighter_than_judge(self):
        from walk.eval import humanoid_gait
        self.assertEqual(hf.MAX_TILT_RAD, math.radians(28.0))
        self.assertLess(math.degrees(hf.MAX_TILT_RAD),
                        humanoid_gait.TILT_MAX_DEG)
        # header pins carry the same boundary to the kernel
        text = (ROOT / "humanoid" / "include" / "duck_model.h").read_text()
        self.assertIn(f"#define DW_ENV_MAX_TILT_RAD {hf.MAX_TILT_RAD!r}",
                      text)
        self.assertIn("#define DW_ENV_COS_MAX_TILT "
                      f"{math.cos(hf.MAX_TILT_RAD)!r}", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
