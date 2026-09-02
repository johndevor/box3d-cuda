"""FlatFloorHumanoidEnv gates: obs layout, header pins, reward v1, termination.

Run: .venv/bin/python -B humanoid/tests/test_humanoid_env.py
"""
from __future__ import annotations

import math
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "humanoid"))

from walk.env import humanoid_flat as hf  # noqa: E402
from walk.env import humanoid_reward as hr  # noqa: E402
from walk.env import reward as duck_reward  # noqa: E402
from walk.env.humanoid_flat import FlatFloorHumanoidEnv  # noqa: E402
import h0_lowering as h0  # noqa: E402

HEADER = ROOT / "humanoid" / "include" / "duck_model.h"


def _macro(text: str, name: str) -> str:
    m = re.search(rf"#define {name} ([^ /\n]+)", text)
    assert m, name
    return m.group(1)


class ObsLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = FlatFloorHumanoidEnv(environments=2, seed=5)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_dims_and_class_attributes(self):
        # walk/train reads OBS/ACT off the factory class (run.py:236-237)
        self.assertEqual(FlatFloorHumanoidEnv.OBS, 58)   # H1: 3*14+16
        self.assertEqual(FlatFloorHumanoidEnv.ACT, 14)
        self.assertEqual(hf.OBS, 3 * hf.ACT + 16)

    def test_reset_obs_layout(self):
        obs = self.env.reset()
        A, T = hf.ACT, 3 * hf.ACT
        self.assertEqual(obs.shape, (2, hf.OBS))
        self.assertEqual(obs.dtype, np.float32)
        # home pose: q-HOME, qdot, prev action all zero
        np.testing.assert_array_equal(obs[:, 0:T], 0.0)
        # gravity in body frame: authored y-up-local body frame -> (0,-1,0)
        np.testing.assert_allclose(obs[:, T:T + 3], [[0.0, -1.0, 0.0]] * 2,
                                   atol=1e-6)
        # root velocities zero at reset
        np.testing.assert_array_equal(obs[:, T + 3:T + 9], 0.0)
        # command slot mirrors the per-env command
        np.testing.assert_allclose(obs[:, T + 9], self.env.command, atol=0)
        # reserved zeros (duck obs[52:54] convention)
        np.testing.assert_array_equal(obs[:, T + 10:T + 12], 0.0)
        # phase clock is a unit vector matching phase0 (t=0)
        np.testing.assert_allclose(np.hypot(obs[:, -2], obs[:, -1]), 1.0,
                                   atol=1e-6)
        np.testing.assert_allclose(obs[:, -2], np.sin(self.env._phase0),
                                   atol=1e-6)

    def test_step_obs_slots_move_coherently(self):
        self.env.reset()
        a = np.full((2, hf.ACT), 0.1, np.float32)
        obs, r, done, info = self.env.step(a)
        self.assertFalse(done.any())
        # prev-action slot echoes the clipped action
        np.testing.assert_allclose(obs[:, 2 * hf.ACT:3 * hf.ACT], 0.1, atol=1e-6)
        # joints moved toward the +0.1*scale targets
        self.assertGreater(float(obs[:, 0:hf.ACT].mean()), 0.0)
        # both feet still planted after one gentle step
        np.testing.assert_array_equal(obs[:, 3 * hf.ACT + 12:3 * hf.ACT + 14], 1.0)
        # phase advanced by 2*pi*hz*dt from phase0
        hz = hf.PHASE_HZ_BASE + hf.PHASE_HZ_PER_MPS * self.env.command
        expect = self.env._phase0 + 2.0 * math.pi * hz * hf.CONTROL_DT
        np.testing.assert_allclose(obs[:, -2], np.sin(expect), atol=1e-6)

    def test_action_slew_and_limit_clip(self):
        self.env.reset()
        a = np.ones((2, hf.ACT), np.float32)      # request HOME + 0.5
        self.env.step(a)
        # one step can move targets at most MAX_TARGET_INCREMENT = 0.16
        np.testing.assert_allclose(
            self.env._targets, np.minimum(0.5, hf.MAX_TARGET_INCREMENT),
            atol=1e-12)
        for _ in range(3):
            self.env.step(a)
        # after 4 steps: min(0.5, 4*0.16=0.64) = 0.5, then joint-limit clip
        lim = self.env._lane.joint_limits
        expect = np.tile(np.clip(np.minimum(0.5, 0.64),
                                 lim[:, 0], lim[:, 1]), (2, 1))
        np.testing.assert_allclose(self.env._effective, expect, atol=1e-12)


class HeaderPinTests(unittest.TestCase):
    """Exact-repr bit-parity: env/reward module constants vs the header."""

    @classmethod
    def setUpClass(cls):
        cls.text = HEADER.read_text()

    def test_header_pins_bit_parity(self):
        pins = {
            "DW_PHASE_HZ_BASE": hf.PHASE_HZ_BASE,
            "DW_PHASE_HZ_PER_MPS": hf.PHASE_HZ_PER_MPS,
            "DW_ENV_CONTROL_DT": hf.CONTROL_DT,
            "DW_ENV_ACTION_SCALE": hf.ACTION_SCALE,
            "DW_ENV_MAX_TARGET_INCREMENT": hf.MAX_TARGET_INCREMENT,
            "DW_ENV_QDOT_OBS_SCALE": hf.QDOT_OBS_SCALE,
            "DW_ENV_MIN_HEIGHT_FRACTION": hf.MIN_HEIGHT_FRACTION,
            "DW_ENV_MAX_TILT_RAD": hf.MAX_TILT_RAD,
            "DW_IMIT_W": hr.W_IMIT,
            "DW_IMIT_SIGMA_SQ": hr.IMIT_SIGMA_SQ,
        }
        for name in ["W_TRACK", "TRACK_SIGMA_SQ", "TRACK_EMA_S", "W_ALIVE",
                     "W_LATERAL", "W_ACTION_RATE", "W_TORQUE", "W_AIR_TIME",
                     "AIR_TIME_MIN", "AIR_TIME_MAX", "PLACEMENT_MIN_M",
                     "OPP_SUPPORT_FRAC", "W_CHATTER", "CHATTER_MAX_S",
                     "W_FLICKER", "STANCE_MIN_S", "W_CLEARANCE",
                     "CLEARANCE_M", "W_DOUBLE_SUPPORT",
                     "DOUBLE_SUPPORT_GRACE", "W_ALTERNATE", "W_SAME_FOOT",
                     "W_PHASE"]:
            pins[f"DW_RW_{name}"] = getattr(hr, name)
        for macro, value in pins.items():
            self.assertEqual(_macro(self.text, macro), repr(float(value)),
                             macro)
        self.assertEqual(_macro(self.text, "DW_ENV_OBS"), str(hf.OBS))
        self.assertEqual(_macro(self.text, "DW_ENV_ACT"), str(hf.ACT))
        self.assertEqual(_macro(self.text, "DW_ENV_TICKS_PER_STEP"), "10u")
        self.assertEqual(_macro(self.text, "DW_ENV_HORIZON_STEPS"),
                         f"{hf.HORIZON_STEPS}u")
        self.assertEqual(_macro(self.text, "DW_RW_TICKS_FULL"),
                         f"{hr.TICKS_FULL}u")
        self.assertIn("DW_ENV_COMMANDS_MPS[3] = {"
                      + ",".join(repr(float(c)) for c in hf.COMMANDS_MPS)
                      + "}", self.text)

    def test_ref_gait_live_and_pinned_in_header(self):
        # v2.1: the imitation hook is LIVE from the synthetic reference
        self.assertIsNotNone(hr.REF_GAIT)
        self.assertEqual(np.asarray(hr.REF_GAIT).shape, (64, hf.ACT))
        self.assertEqual(hr.W_IMIT, 0.5)
        block = self.text[self.text.index("DW_REF_GAIT"):]
        block = block[:block.index(";")]
        # bin 0's exact f64 row is embedded verbatim (full bit-parity is
        # pinned by test_reference_gait.test_header_carries_identical_table)
        row0 = "{" + ",".join(repr(float(x))
                              for x in np.asarray(hr.REF_GAIT)[0]) + "}"
        self.assertIn(row0, block)
        self.assertNotEqual(block.count("{" + ",".join(["0.0"] * hf.ACT) + "}"),
                            64, "header still carries the zero placeholder")


class RewardTests(unittest.TestCase):
    def _state(self, E=2, contact=True, vx=0.0):
        return {
            "root_lin_vel": np.tile([vx, 0.0, 0.0], (E, 1)),
            "root_ang_vel": np.zeros((E, 3)),
            "foot_contact": np.full((E, 2), bool(contact)),
            "sole_height": np.zeros((E, 2)),
            "action": np.zeros((E, hf.ACT)),
            "torque": np.zeros((E, hf.ACT)),
            "foot_x": np.zeros((E, 2)),
            "joint_q": np.zeros((E, hf.ACT)),
            "phase": np.zeros(E),
        }

    def test_standing_reward_matches_closed_form(self):
        tracker = hr.GaitTracker(2)
        prev = self._state()
        state = self._state()
        cmd = np.array([0.75, 0.5])
        r = None
        for _ in range(20):   # exceed the 0.25 s double-support grace
            r = hr.reward(prev, state, np.zeros((2, hf.ACT)), cmd, tracker)
        # v_avg -> 0: track = exp(-cmd^2/sigma^2); alive; phase term nets 0
        # (left matches sin(0)>=0 stance, right mismatches); double-support
        # penalty active; v2.1: imitation leak at bin 0 (phase 0, joints 0)
        imit = hr.W_IMIT * np.exp(
            -np.mean(np.square(np.asarray(hr.REF_GAIT)[0]))
            / hr.IMIT_SIGMA_SQ)
        expect = (hr.W_TRACK * np.exp(-cmd ** 2 / hr.TRACK_SIGMA_SQ)
                  + hr.W_ALIVE - hr.W_DOUBLE_SUPPORT + imit)
        np.testing.assert_allclose(r, expect, atol=1e-5)

    def test_tracker_is_ducks(self):
        self.assertIs(hr.GaitTracker, duck_reward.GaitTracker)

    def test_qualified_step_bonus_fires(self):
        tracker = hr.GaitTracker(1)
        cmd = np.array([0.75])
        prev = self._state(E=1)
        # build stance credit on both feet (full-contact steps)
        s = self._state(E=1)
        s["contact_ticks"] = np.full((1, 2), hr.TICKS_FULL)
        for _ in range(10):
            hr.reward(prev, s, np.zeros((1, hf.ACT)), cmd, tracker)
        # left foot lifts off...
        air = self._state(E=1)
        air["foot_contact"] = np.array([[False, True]])
        air["contact_ticks"] = np.array([[0, hr.TICKS_FULL]])
        prev_c = s
        for _ in range(int(0.2 / hr.CONTROL_DT)):   # 0.2 s swing, in window
            r_air = hr.reward(prev_c, air, np.zeros((1, hf.ACT)), cmd, tracker)
            prev_c = air
        # ...and touches down 0.2 m forward at left-stance phase (sin>=0)
        down = self._state(E=1)
        down["foot_x"] = np.array([[0.2, 0.0]])
        down["contact_ticks"] = np.full((1, 2), hr.TICKS_FULL)
        down["phase"] = np.array([0.5])              # sin > 0: left window
        r = hr.reward(prev_c, down, np.zeros((1, hf.ACT)), cmd, tracker)
        r_base = hr.reward(down, down, np.zeros((1, hf.ACT)), cmd,
                           hr.GaitTracker(1))
        # the touchdown step must contain the qualified-step bonus
        self.assertGreater(float(r[0]), float(r_air[0]) + hr.W_AIR_TIME - 1.0)
        self.assertEqual(int(tracker.last_foot[0]), 0)


class TerminationTests(unittest.TestCase):
    def test_termination_thresholds(self):
        env = FlatFloorHumanoidEnv(environments=1, seed=9)
        try:
            self.assertAlmostEqual(env._min_height, 0.7 * 1.15, places=9)
            # v2: termination just inside the FROZEN judge's 30 deg, so a
            # judge-failing lean is never a survivable strategy
            self.assertEqual(hf.MAX_TILT_RAD, math.radians(28.0))
            from walk.eval import humanoid_gait  # noqa: PLC0415
            self.assertLess(math.degrees(hf.MAX_TILT_RAD),
                            humanoid_gait.TILT_MAX_DEG)
            # done stays terminal until reset (no auto-reset), duck contract
            env._done[:] = True
            obs, r, done, info = env.step(np.zeros((1, hf.ACT)))
            self.assertTrue(done.all())
            self.assertEqual(float(r[0]), 0.0)
        finally:
            env.close()

    def test_env_var_clock_override_is_read_at_import(self):
        # documented mechanism only: values are module-level like flat.py's
        self.assertEqual(hf.PHASE_HZ_BASE, 0.0)
        self.assertEqual(hf.PHASE_HZ_PER_MPS, 3.33)


if __name__ == "__main__":
    unittest.main(verbosity=2)
