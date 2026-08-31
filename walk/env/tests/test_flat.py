"""FlatFloorDuckEnv behaviour tests over the real native lane.

Run: .venv/bin/python -B -m unittest discover -s walk
"""
import math
import unittest

import numpy as np

from walk.env.contract import ACT, OBS, SolverFault
from walk.env.flat import CONTROL_DT, HORIZON_STEPS, FlatFloorDuckEnv


class TestHomeHold(unittest.TestCase):
    def test_zero_action_holds_home_for_two_seconds(self):
        """Reproduces the known-good home-hold: E=2, 100 policy steps."""
        env = FlatFloorDuckEnv(environments=2, seed=0)
        try:
            action = np.zeros((2, ACT), np.float32)
            contact_by_step5 = None
            for step in range(100):
                obs, reward, done, info = env.step(action)
                self.assertEqual(obs.shape, (2, OBS))
                self.assertFalse(done.any(), f"terminated at step {step}")
                if step == 4:
                    contact_by_step5 = info["foot_contact"].copy()
            tilt_deg = np.degrees(info["tilt_rad"])
            self.assertLess(float(tilt_deg.max()), 2.0)
            self.assertIsNotNone(contact_by_step5)
            self.assertTrue(contact_by_step5.all(),
                            "both feet must be in contact by policy step 5")
            self.assertTrue(np.isfinite(obs).all() and np.isfinite(reward).all())
        finally:
            env.close()


class TestDeterminism(unittest.TestCase):
    def test_same_seed_identical_observations(self):
        def rollout():
            env = FlatFloorDuckEnv(environments=2, seed=11)
            try:
                rng = np.random.default_rng(99)
                out = [env.reset(seed=11)]
                for _ in range(50):
                    action = rng.uniform(-0.3, 0.3, (2, ACT)).astype(np.float32)
                    obs, reward, done, _ = env.step(action)
                    out.append(obs.copy())
                    out.append(reward.copy())
                return out
            finally:
                env.close()

        a, b = rollout(), rollout()
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


class TestMaskedReset(unittest.TestCase):
    def test_masked_reset_of_env0_leaves_env1_untouched(self):
        actions = np.random.default_rng(5).uniform(
            -0.2, 0.2, (20, 2, ACT)).astype(np.float32)

        def run(reset_at_10: bool):
            env = FlatFloorDuckEnv(environments=2, seed=3)
            try:
                trail = []
                for i in range(20):
                    if reset_at_10 and i == 10:
                        obs = env.reset(mask=np.array([True, False]))
                        # env1 rows of the reset observation must match live state
                        trail.append(("reset-obs", obs[1].copy()))
                    obs, reward, done, info = env.step(actions[i])
                    trail.append((obs[1].copy(), reward[1], done[1],
                                  info["foot_contact"][1].copy()))
                return trail
            finally:
                env.close()

        plain, masked = run(False), run(True)
        pi = iter(plain)
        for row in masked:
            if isinstance(row[0], str):
                continue  # the extra reset observation row
            ref = next(pi)
            np.testing.assert_array_equal(row[0], ref[0])
            self.assertEqual(row[1], ref[1])
            self.assertEqual(row[2], ref[2])
            np.testing.assert_array_equal(row[3], ref[3])

    def test_masked_reset_restarts_env0_episode_clock(self):
        env = FlatFloorDuckEnv(environments=2, seed=3)
        try:
            action = np.zeros((2, ACT), np.float32)
            for _ in range(7):
                env.step(action)
            obs = env.reset(mask=np.array([True, False]))
            # phase clock: env0 back at t=0 -> sin=0, cos=1; env1 keeps running
            self.assertAlmostEqual(float(obs[0, 56]), 0.0, places=6)
            self.assertAlmostEqual(float(obs[0, 57]), 1.0, places=6)
            from walk.env.flat import PHASE_HZ_PER_MPS
            phase1 = 2 * math.pi * PHASE_HZ_PER_MPS * float(env._command[1]) \
                * 7 * CONTROL_DT
            self.assertAlmostEqual(float(obs[1, 56]), math.sin(phase1), places=5)
            _, _, done, info = env.step(action)
            self.assertFalse(done.any())
            np.testing.assert_array_equal(
                info["episode_time"], np.float32([CONTROL_DT, 8 * CONTROL_DT]))
        finally:
            env.close()


class TestTermination(unittest.TestCase):
    def test_horizon_and_no_auto_reset(self):
        env = FlatFloorDuckEnv(environments=1, seed=0)
        try:
            env._t[:] = HORIZON_STEPS - 1  # fast-forward the episode clock
            action = np.zeros((1, ACT), np.float32)
            _, _, done, _ = env.step(action)
            self.assertTrue(done.all(), "8 s horizon must terminate")
            _, reward, done, _ = env.step(action)
            self.assertTrue(done.all(), "done must stay terminal without reset")
            self.assertEqual(float(reward[0]), 0.0)
            obs = env.reset(mask=np.array([True]))
            _, _, done, _ = env.step(action)
            self.assertFalse(done.any())
            self.assertEqual(obs.shape, (1, OBS))
        finally:
            env.close()


class TestSolverFaultPath(unittest.TestCase):
    def test_repaired_solver_survives_perturbed_reset(self):
        """The workstream-A civ1 repair landed: the reset perturbation that
        formerly stalled during foot settling (seed 1) must now hold 60
        zero-action steps without any fault at the strict tolerance."""
        env = FlatFloorDuckEnv(environments=2, seed=1, perturbation_rad=0.02)
        try:
            action = np.zeros((2, ACT), np.float32)
            for _ in range(60):
                _, _, done, _ = env.step(action)
            self.assertFalse(done.any())
        finally:
            env.close()

    def test_injected_faults_are_persisted_and_raised(self):
        """Fault handling pinned independently of solver quality: a lane whose
        tick reports a nonzero rc must persist the failing env's state and
        raise SolverFault (never return post-fault observations)."""
        from walk.env import native_lane

        class FaultingLane(native_lane.NativeDuckLane):
            calls = 0

            def tick(self, targets):
                type(self).calls += 1
                if type(self).calls > 5:
                    diags = [dict(environment=e, phase=3, native_status=3,
                                  iterations=4096, contact_points=2,
                                  active_limits=0, joint_residual=1e-6,
                                  normal_residual=1e-6, tangent_residual=1e-6,
                                  momentum_residual=1e-9,
                                  maximum_normal_impulse=1.0,
                                  maximum_penetration=1e-4)
                             for e in range(self.E)]
                    return 5, diags
                return super().tick(targets)

        env = FlatFloorDuckEnv(
            environments=2, seed=1, perturbation_rad=0.0,
            lane_factory=lambda E, offsets: FaultingLane(E, joint_offsets=offsets))
        try:
            action = np.zeros((2, ACT), np.float32)
            with self.assertRaises(SolverFault) as caught:
                for _ in range(60):
                    env.step(action)
        finally:
            env.close()
        fault = caught.exception
        import json
        from pathlib import Path
        path = Path(fault.saved_problem_path)
        self.assertTrue(path.is_file())
        payload = json.loads(path.read_text())
        self.assertEqual(payload["environment"], fault.env_index)
        for key in ["diagnostics", "state", "effective_targets", "action"]:
            self.assertIn(key, payload)
        self.assertEqual(len(payload["state"]["qpos"]), 21)
        self.assertTrue(any(d["native_status"] != 0
                            for d in payload["diagnostics"]))


if __name__ == "__main__":
    unittest.main()
