"""Tests for walk/eval/mujoco_xval.py (cross-simulator validation harness).

Run: .venv/bin/python -B -m unittest walk.eval.tests.test_mujoco_xval -v

(a) home hold: the model's native position servos at zero-action targets keep
    the duck upright for 2 s — validates the actuation mapping;
(b) reset observation parity vs FlatFloorDuckEnv — catches joint-order and
    frame mistakes (loose tolerance; random env slots excluded);
(c) an 8 s zero-policy trace round-trips through the strict evaluator without
    rejection — validates the duckgridwalk.episode/1 schema emission.

Budget: < 60 s total (MuJoCo runs ~100k steps/s; the native-env parity check
reuses the pinned prebuilt dylib and is skipped if none exists).
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from walk.env.flat import HOME, MIN_HEIGHT_FRACTION
from walk.eval import mujoco_xval
from walk.eval.gait import evaluate_episode
from walk.eval.mujoco_xval import MujocoDuckLane, observe, run_episode

ROOT = Path(__file__).resolve().parents[3]

try:
    import mujoco  # noqa: F401
    _MUJOCO_MISSING = None
except ImportError as exc:                          # pragma: no cover
    _MUJOCO_MISSING = str(exc)

_LANE = None


def _lane() -> MujocoDuckLane:
    global _LANE
    if _LANE is None:
        _LANE = MujocoDuckLane()
    return _LANE


@unittest.skipIf(_MUJOCO_MISSING, "mujoco not installed")
class TestModelAndActuation(unittest.TestCase):

    def test_model_loads_with_pinned_contract(self):
        lane = _lane()
        # Loading already enforced dt, joint mapping, servo gains, home key.
        self.assertEqual(lane.model.opt.timestep, 0.002)
        self.assertEqual(lane.model.nu, 14)
        np.testing.assert_allclose(
            lane.model.key_qpos[lane.key_id][lane.qadr], HOME, atol=0)

    def test_home_hold_two_seconds_stays_upright(self):
        """Zero-action effective targets on the native servos: tilt < 5 deg."""
        lane = _lane()
        lane.reset()
        # Floor-clear reset must match the native lane's pinned root height.
        self.assertLess(abs(lane.home_root_height - 0.16788827542191784),
                        1e-12)
        effective = np.clip(HOME, lane.joint_limits[:, 0],
                            lane.joint_limits[:, 1])
        state = None
        for _ in range(100):                   # 100 policy steps = 2 s
            state = lane.policy_step(effective)
        tilt_deg = math.degrees(mujoco_xval._tilt_rad(state.q[0, 3:7]))
        self.assertLess(tilt_deg, 5.0, f"home hold tilted {tilt_deg:.2f} deg")
        self.assertGreater(state.q[0, 2],
                           MIN_HEIGHT_FRACTION * lane.home_root_height)
        self.assertTrue(bool(state.finite()[0]))
        self.assertTrue(all(state.foot_contact[0]),
                        "both feet should be planted at home")


@unittest.skipIf(_MUJOCO_MISSING, "mujoco not installed")
class TestResetObsParity(unittest.TestCase):

    def test_reset_obs_matches_flat_env(self):
        """MuJoCo reset obs ~= FlatFloorDuckEnv reset obs (order/frames).

        Slots 51:54 (random per-episode command) and 56:58 (random phase0)
        are excluded; everything else must agree loosely.
        """
        lib = ROOT / "build" / "libintegrated_duck-pinned-97c3d37.dylib"
        if not lib.is_file():
            candidates = sorted((ROOT / "build").glob("libintegrated_duck-*"))
            if not candidates:
                raise unittest.SkipTest(
                    "no prebuilt native duck dylib; parity check needs one")
            lib = candidates[0]
        from walk.env.flat import FlatFloorDuckEnv
        env = FlatFloorDuckEnv(environments=1, seed=4242,
                               perturbation_rad=0.0, library_path=lib)
        try:
            env_obs = env.reset()[0]
        finally:
            env.close()

        lane = _lane()
        lane.reset()
        mj_obs = observe(lane.tick_state(), np.zeros((1, 14)), command=0.0,
                         t_steps=0)[0]

        # Joint-order check: q-offset slots are HOME-relative, must be ~0.
        self.assertLess(np.abs(mj_obs[0:14]).max(), 0.05)
        # Frame check: gravity-in-body-frame is (0, 0, -1) when upright.
        self.assertLess(mj_obs[44], -0.9)
        self.assertLess(abs(mj_obs[42]), 0.1)
        self.assertLess(abs(mj_obs[43]), 0.1)
        # Loose elementwise parity on the non-random slots.
        keep = np.r_[0:51, 54:56]
        diff = np.abs(mj_obs[keep] - env_obs[keep])
        self.assertLess(
            float(diff.max()), 0.05,
            f"reset obs mismatch: max |diff| {diff.max():.4f} at slot "
            f"{keep[int(diff.argmax())]}")


@unittest.skipIf(_MUJOCO_MISSING, "mujoco not installed")
class TestTraceSchemaRoundTrip(unittest.TestCase):

    def test_zero_policy_trace_roundtrips_through_evaluator(self):
        """8 s standing trace: accepted (not rejected) by the evaluator."""
        trace = run_episode(_lane(), lambda obs: np.zeros((1, 14)),
                            command=0.15, seconds=8.0, seed=0)
        self.assertEqual(trace["schema"], "duckgridwalk.episode/1")
        self.assertEqual(len(trace["ticks"]["time_s"]), 4000)
        self.assertTrue(trace["truncated_at_horizon"])
        self.assertFalse(trace["terminated"])
        result = evaluate_episode(trace)
        self.assertIs(result["rejected"], False)
        self.assertTrue(
            result["criteria"]["single_episode_no_reset_or_failure"]["pass"])
        # Standing still is a valid trace but not a walk.
        self.assertTrue(result["criteria"]["tilt_within_30_degrees"]["pass"])
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
