"""ArmReachEnv gates: obs/act contract, reachable targets, acquisition,
termination, and env-level lane parity (oracle vs fp32 serial kernel).

Run: .venv/bin/python -B arm/tests/test_arm_env.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "arm"))

from walk.env import arm_reach as ar  # noqa: E402
from walk.env import arm_reward as rw  # noqa: E402
from walk.env.arm_cuda_lane import CudaArmLane  # noqa: E402
from walk.eval import arm_reach_judge as judge  # noqa: E402
from walk.eval.arm_acceptance import ScriptedIKPolicy  # noqa: E402
import arm_lowering as al  # noqa: E402


def _serial(variant):
    return lambda E, off: CudaArmLane(E, variant=variant, joint_offsets=off)


class ContractTests(unittest.TestCase):
    def test_obs_layout_and_dims(self):
        for v in ("kr240", "lite"):
            env = ar.ArmReachEnv(environments=3, seed=1, variant=v,
                                 lane_factory=_serial(v))
            try:
                self.assertEqual((env.OBS, env.ACT), (27, 6))
                obs = env.reset()
                self.assertEqual(obs.shape, (3, 27))
                self.assertEqual(obs.dtype, np.float32)
                s = env.spec
                np.testing.assert_allclose(obs[:, 0:6], np.tile(s.home_q, (3, 1)),
                                           atol=1e-6)
                np.testing.assert_allclose(obs[:, 6:12], 0.0, atol=1e-6)
                tip = al.home_tip(s)
                np.testing.assert_allclose(obs[:, 15:18], np.tile(tip, (3, 1)),
                                           atol=1e-5)
                np.testing.assert_allclose(obs[:, 18:21], obs[:, 12:15] - obs[:, 15:18],
                                           atol=1e-6)
                np.testing.assert_array_equal(obs[:, 21:27], 0.0)
                # targets are inside the drawn tier ball and proxy-clear
                r = np.array([judge.tier_radius(v, t) for t in env.tier])
                self.assertTrue((np.linalg.norm(obs[:, 12:15] - tip, axis=1) <= r).all())
                o, rew, done, info = env.step(np.zeros((3, 6), np.float32))
                self.assertEqual((o.shape, rew.shape, done.shape), ((3, 27), (3,), (3,)))
                self.assertEqual(rew.dtype, np.float32)
                self.assertIn("solver_iterations", info)
                np.testing.assert_array_equal(o[:, 21:27], 0.0)   # prev action
            finally:
                env.close()

    def test_action_maps_to_limit_scaled_targets_with_slew(self):
        env = ar.ArmReachEnv(environments=2, seed=2, variant="kr240",
                             lane_factory=_serial("kr240"))
        try:
            lim = env.joint_limits
            a = np.array([[1.0] * 6, [-1.0] * 6], np.float32)
            t0 = env._targets.copy()
            env.step(a)
            inc = env.max_target_increment
            np.testing.assert_allclose(env._targets[0], np.minimum(t0[0] + inc, lim[:, 1]))
            np.testing.assert_allclose(env._targets[1], np.maximum(t0[1] - inc, lim[:, 0]))
            np.testing.assert_allclose(inc, al.velocity_limits(env.spec) * ar.CONTROL_DT)
            for _ in range(2):                         # slew accumulates while live
                _, _, done, _ = env.step(a)
            self.assertFalse(done.any())
            np.testing.assert_allclose(env._targets[0], np.minimum(t0[0] + 3 * inc, lim[:, 1]))
            np.testing.assert_allclose(env._targets[1], np.maximum(t0[1] - 3 * inc, lim[:, 0]))
        finally:
            env.close()

    def test_target_sampler_reachable_by_construction_and_seeded(self):
        for v in ("kr240", "lite"):
            s = al.spec(v)
            center = al.home_tip(s)
            for tier in judge.TIERS:
                rng = np.random.default_rng(100 + tier)
                pts = np.array([ar.sample_target(s, rng, tier) for _ in range(300)])
                d = np.linalg.norm(pts - center, axis=1)
                self.assertTrue((d <= judge.tier_radius(v, tier) + 1e-12).all())
                self.assertGreater(d.max(), 0.8 * judge.tier_radius(v, tier))
                self.assertTrue((pts[:, 2] >= judge.FLOOR_MARGIN_FRAC * al.reach(s)).all())
                a = np.array([ar.sample_target(s, np.random.default_rng(7), tier)
                              for _ in range(3)])
                b = np.array([ar.sample_target(s, np.random.default_rng(7), tier)
                              for _ in range(3)])
                np.testing.assert_array_equal(a, b)

    def test_reset_is_deterministic_per_seed_and_mask(self):
        e1 = ar.ArmReachEnv(environments=4, seed=9, variant="lite",
                            lane_factory=_serial("lite"))
        e2 = ar.ArmReachEnv(environments=4, seed=9, variant="lite",
                            lane_factory=_serial("lite"))
        try:
            np.testing.assert_array_equal(e1.reset(), e2.reset())
            np.testing.assert_array_equal(e1.target, e2.target)
            a = np.full((4, 6), 0.3, np.float32)
            for _ in range(5):
                o1, r1, d1, _ = e1.step(a)
                o2, r2, d2, _ = e2.step(a)
                np.testing.assert_array_equal(o1, o2)
                np.testing.assert_array_equal(r1, r2)
            t_before = e1.target.copy()
            o = e1.reset(mask=[1, 0, 0, 0])
            np.testing.assert_allclose(o[0, 0:6], al.LITE.home_q, atol=1e-6)
            np.testing.assert_array_equal(e1.target[1:], t_before[1:])
            self.assertFalse(np.array_equal(e1.target[0], t_before[0]))
        finally:
            e1.close()
            e2.close()


class TaskTests(unittest.TestCase):
    def test_scripted_ik_acquires_targets_and_earns_bonus(self):
        """The scripted IK baseline acquires >= 2 tier-0 targets within 4 s
        on the oracle lane and the acquisition bonus is paid exactly once
        per switch (task/reward/acquisition bookkeeping consistency)."""
        for v in ("kr240", "lite"):
            env = ar.ArmReachEnv(environments=1, seed=4242, variant=v, tier=0)
            try:
                pol = ScriptedIKPolicy(v)
                obs = env.reset()
                bonuses = 0
                for t in range(200):
                    obs, r, done, info = env.step(pol(obs))
                    self.assertFalse(done.any(), (v, t, info))
                    if info["acquired"][0]:
                        bonuses += 1
                        self.assertGreater(float(r[0]), rw.W_ACQUIRE - 1.0)
                self.assertGreaterEqual(int(info["acquired_total"][0]), 2, v)
                self.assertEqual(bonuses, int(info["acquired_total"][0]))
                self.assertEqual(int(info["target_index"][0]), bonuses)
                print(f"{v} scripted IK: {bonuses} tier-0 targets in 4 s, "
                      f"final dist {info['tip_dist'][0]:.3f} m", file=sys.stderr)
            finally:
                env.close()

    def test_proxy_violation_terminates(self):
        """Driving the tip below the floor margin (a2 down, a3 straight)
        terminates with the proxy flag and a W_PROXY-sized penalty."""
        env = ar.ArmReachEnv(environments=1, seed=3, variant="kr240",
                             lane_factory=_serial("kr240"))
        try:
            env.reset()
            lim = env.joint_limits
            q_goal = np.array([0.0, lim[1, 1], 0.0, 0.0, 0.0, 0.0])   # a2 at +0.61
            a = 2.0 * (q_goal - lim[:, 0]) / (lim[:, 1] - lim[:, 0]) - 1.0
            a = np.clip(a, -1, 1)[None].astype(np.float32)
            crashed = False
            for t in range(300):
                obs, r, done, info = env.step(a)
                if done[0]:
                    crashed = bool(info["proxy_violation"][0])
                    self.assertLess(float(r[0]), -rw.W_PROXY + 1.5)
                    break
            self.assertTrue(crashed, "expected a floor-proxy termination")
            # done envs are frozen: reward 0, obs constant until reset
            o2, r2, d2, _ = env.step(a)
            self.assertTrue(d2[0])
            self.assertEqual(float(r2[0]), 0.0)
        finally:
            env.close()

    def test_env_parity_oracle_vs_serial_kernel(self):
        """Same seed + same actions on both lanes: obs within 1e-3 (rad/m),
        reward within 1e-2, identical done flags and target sequences."""
        for v in ("kr240", "lite"):
            eo = ar.ArmReachEnv(environments=2, seed=77, variant=v)
            es = ar.ArmReachEnv(environments=2, seed=77, variant=v,
                                lane_factory=_serial(v))
            try:
                oo, os_ = eo.reset(), es.reset()
                np.testing.assert_array_equal(eo.target, es.target)
                worst_o = worst_r = 0.0
                rng = np.random.default_rng(5)
                pol = ScriptedIKPolicy(v)
                for t in range(60):
                    a = pol(oo) if t % 2 else rng.uniform(-0.3, 0.3, (2, 6))
                    a = np.asarray(a, np.float32)
                    oo, ro, do, _ = eo.step(a)
                    os_, rs, ds, _ = es.step(a)
                    np.testing.assert_array_equal(do, ds)
                    worst_o = max(worst_o, float(np.abs(oo - os_).max()))
                    worst_r = max(worst_r, float(np.abs(ro - rs).max()))
                self.assertLess(worst_o, 1e-3, v)
                self.assertLess(worst_r, 1e-2, v)
                print(f"{v} env parity obs={worst_o:.2e} reward={worst_r:.2e}",
                      file=sys.stderr)
            finally:
                eo.close()
                es.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
