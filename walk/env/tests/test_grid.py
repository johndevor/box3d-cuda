"""CubeGridDuckEnv behaviour tests over the real dwv1 cube-grid lane.

Run: .venv/bin/python -B -m unittest discover -s walk/env/tests
"""
import json
import unittest
from pathlib import Path

import numpy as np

from walk.env import grid_lane
from walk.env.contract import ACT, OBS, SolverFault
from walk.env.flat import FlatFloorDuckEnv
from walk.env.grid import CubeGridDuckEnv

# Flat flush 8x8 STATIC grid: zero jitter, spacing == cube_size (cubes flush
# at one height). This is grid_lane.DEFAULT_GRID; spelled out for clarity.
FLUSH_GRID = dict(nx=8, nz=8, cube_size=0.06, spacing=0.06, height_jitter=0.0,
                  dynamic=False)

# Terrain seeds for the 8 mm jitter cases (test d). Terrain seed 41 is the
# pinned known-good stepped grid that converges at the civ1 impulse tolerance
# 1e-8 (the dwv1 README documents static grids at 1e-8; its own jittered-grid
# gate uses seed 41). Terrain seed 7 reproduces the documented workstream-A
# civ1 stall ("duck resting across stepped cube tops", CIV1_NO_CONVERGENCE
# just above 1e-8) and therefore needs the README's interim 1e-6 tolerance.
JITTER_M = 0.008
GOOD_TERRAIN_SEED = 41
STALLING_TERRAIN_SEED = 7


class TestGridHold(unittest.TestCase):
    def test_zero_action_holds_on_flush_static_grid(self):
        """(a) 100-step home hold on the flush 8x8 static grid."""
        env = CubeGridDuckEnv(environments=2, seed=0, grid=FLUSH_GRID)
        try:
            self.assertEqual(env._lane.impulse_tolerance,
                             grid_lane.STATIC_IMPULSE_TOLERANCE)
            action = np.zeros((2, ACT), np.float32)
            contact_by_step10 = None
            for step in range(100):
                obs, reward, done, info = env.step(action)
                self.assertEqual(obs.shape, (2, OBS))
                self.assertFalse(done.any(), f"terminated at step {step}")
                if step == 9:
                    contact_by_step10 = info["foot_contact"].copy()
            tilt_deg = np.degrees(info["tilt_rad"])
            self.assertLess(float(tilt_deg.max()), 2.0)
            self.assertIsNotNone(contact_by_step10)
            self.assertTrue(contact_by_step10.all(),
                            "both feet must be in contact by policy step 10")
            self.assertTrue(np.isfinite(obs).all() and np.isfinite(reward).all())
            # Settled on cube tops: sole height above the SUPPORTING surface
            # (cube top, not the floor) must be ~0, not ~cube_size.
            sole = env._lane.read().sole_height
            self.assertLess(float(np.abs(sole).max()), 0.002,
                            "sole height must be measured above the cube tops")
        finally:
            env.close()


class TestGridDeterminism(unittest.TestCase):
    def test_same_seed_identical_observations(self):
        """(b) 50 steps, same seed twice: bit-exact obs and rewards."""
        def rollout():
            env = CubeGridDuckEnv(environments=2, seed=11, grid=FLUSH_GRID)
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


class TestGridMaskedReset(unittest.TestCase):
    def test_masked_reset_of_env0_leaves_env1_untouched(self):
        """(c) resetting env0 mid-episode must not perturb env1's stream."""
        actions = np.random.default_rng(5).uniform(
            -0.2, 0.2, (20, 2, ACT)).astype(np.float32)

        def run(reset_at_10: bool):
            env = CubeGridDuckEnv(environments=2, seed=3, grid=FLUSH_GRID)
            try:
                trail = []
                for i in range(20):
                    if reset_at_10 and i == 10:
                        obs = env.reset(mask=np.array([True, False]))
                        # env1 rows of the reset observation match live state
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


class TestJitteredGrid(unittest.TestCase):
    def test_8mm_jitter_holds_50_steps_at_static_default_1e8(self):
        """(d) 8 mm stepped static grid holds at the documented 1e-8 default."""
        env = CubeGridDuckEnv(
            environments=1, seed=0,
            grid=dict(FLUSH_GRID, height_jitter=JITTER_M, seed=GOOD_TERRAIN_SEED))
        try:
            self.assertEqual(env._lane.impulse_tolerance, 1e-8,
                             "static grids default to the pinned 1e-8")
            self.assertEqual(env._lane.jtol, 1e-8)
            action = np.zeros((1, ACT), np.float32)
            for step in range(50):
                obs, reward, done, info = env.step(action)
                self.assertFalse(done.any(), f"terminated at step {step}")
            self.assertTrue(np.isfinite(obs).all())
            self.assertTrue(info["foot_contact"].all())
        finally:
            env.close()

    def test_formerly_stalling_terrain_now_holds_at_1e8_and_at_1e6(self):
        """(d) the stepped-cube-top configuration that stalled civ1 before the
        workstream-A2 repair (terrain seed pinned below) must now hold 50
        zero-action steps at the STRICT 1e-8 tolerance, and still hold at the
        README's formerly-required interim 1e-6 constructor tolerance."""
        stall = dict(FLUSH_GRID, height_jitter=JITTER_M, seed=STALLING_TERRAIN_SEED)
        env = CubeGridDuckEnv(environments=1, seed=0, grid=stall)
        try:
            action = np.zeros((1, ACT), np.float32)
            for step in range(50):
                obs, reward, done, info = env.step(action)
                self.assertFalse(done.any(), f"terminated at step {step}")
            self.assertTrue(np.isfinite(obs).all())
        finally:
            env.close()

        env = CubeGridDuckEnv(environments=1, seed=0, grid=stall,
                              impulse_tolerance=1e-6)
        try:
            self.assertEqual(env._lane.jtol, 1e-6, "jtol tracks the tolerance")
            action = np.zeros((1, ACT), np.float32)
            for step in range(50):
                obs, reward, done, info = env.step(action)
                self.assertFalse(done.any(), f"terminated at step {step}")
            self.assertTrue(np.isfinite(obs).all())
        finally:
            env.close()

    def test_dynamic_grid_defaults_to_1e6(self):
        """Dynamic grids default to the README's interim 1e-6 (construction
        only; dynamic stepping is dwv1 milestone-2 territory)."""
        env = CubeGridDuckEnv(environments=1, seed=0,
                              grid=dict(nx=2, nz=2, dynamic=True))
        try:
            self.assertEqual(env._lane.impulse_tolerance,
                             grid_lane.DYNAMIC_IMPULSE_TOLERANCE)
            self.assertEqual(env._lane.jtol, 1e-6)
        finally:
            env.close()


class TestObsParity(unittest.TestCase):
    def test_first_observation_matches_flat_in_shared_slots(self):
        """(e) zero-jitter flush grid at cube-top height vs the flat floor.

        Both lanes decode the same pinned floor-clear reset frame; the grid
        lane only lifts the root z by cube_size + 1.5 mm gap, and absolute
        root height is not part of the observation. Joint q/qdot, gravity
        and command slots are therefore expected to agree; we assert
        approximately (atol 1e-6) rather than bitwise because the two lanes
        build their float32 observations from independently round-tripped
        native reads. Exact equality is NOT expected for sole-height-derived
        quantities later in the episode: at reset the grid feet sit 1.5 mm
        (the construction gap above the tallest cube top) higher above their
        support than flat's pinned floor clearance — but sole height is not
        observed, so the first obs may agree everywhere."""
        flat = FlatFloorDuckEnv(environments=1, seed=7)
        try:
            obs_flat = flat.reset()
        finally:
            flat.close()
        env = CubeGridDuckEnv(environments=1, seed=7, grid=FLUSH_GRID)
        try:
            obs_grid = env.reset()
            # feet start at (approximately) the same relative support height
            sole = env._lane.read().sole_height
            self.assertLess(float(np.abs(sole).max()), 0.005)
        finally:
            env.close()
        for name, sl in [("joint q", slice(0, 14)), ("joint qdot", slice(14, 28)),
                         ("gravity", slice(42, 45)), ("command", slice(51, 54))]:
            np.testing.assert_allclose(
                obs_grid[:, sl], obs_flat[:, sl], rtol=0, atol=1e-6,
                err_msg=f"{name} slots must match the flat-floor observation")
        # same episode-command draw (same _episode_rng stream) => exact
        np.testing.assert_array_equal(obs_grid[:, 51], obs_flat[:, 51])


if __name__ == "__main__":
    unittest.main()
