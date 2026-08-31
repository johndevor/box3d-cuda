"""Milestone 2 gates: DYNAMIC cubes (dwv1) — drop/sleep, friction, stacking,
islands, duck-cube coupling.

Coupled duck+cube islands currently need civ1 impulse tolerance 1e-6 (with
matching av2 jtol) on some resting ticks: the redundant-normal-row stall that
workstream A owns. Momentum residual stays <=1e-8 (checked every step).

Run: .venv/bin/python -B experimental/duck_world_v1/tests/test_dynamic_cubes.py
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'walk/env'))
import world as w  # noqa: E402

np = w.np
HOME = np.array(w.HOME)
SOLVE = dict(tolerance=1e-6, jtol=1e-6)


class DynamicCubeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = w.library(w.build())

    def scene(self, environments=1, lift=0., **kw):
        spec = dict(nx=5, nz=5, cube_size=.05, spacing=.06, base_height=.02,
                    height_jitter=0., dynamic=1, cube_mass=.1, friction=.8,
                    origin_x=1., seed=11)
        spec.update(kw)
        s, cm = w.duck_grid_scene(self.lib, environments, w.grid_spec(**spec), lift=lift)
        self.addCleanup(s.close)
        return s

    def run_until(self, s, ticks, stop=None):
        target = np.tile(HOME, (s.E, 1))
        for t in range(ticks):
            rc, d = s.step(target=target, **SOLVE)
            self.assertEqual(rc, 0, (t, d[0]))
            for x in d:
                self.assertLessEqual(x['momentum_residual'], 1e-8, (t, x))
            if stop and stop(d):
                return t, d
        return ticks-1, d

    def test_drop_settle_sleep(self):
        """5x5 cubes dropped 2cm settle on the floor and all fall asleep."""
        s = self.scene()
        t, d = self.run_until(s, 750, stop=lambda d: d[0]['awake_cubes'] == 0)
        self.assertEqual(d[0]['awake_cubes'], 0, 'all 25 cubes asleep')
        self.assertLess(t, 750)
        x = s.read()
        np.testing.assert_allclose(x.cube_pose[0, :, 2], .025, atol=1e-3)
        self.assertTrue((x.cube_awake[0] == 0).all())
        self.assertTrue((x.cube_velocity[0] == 0).all(), 'sleeping cubes are still')

    def test_push_slides_and_stops(self):
        """A pushed cube slides roughly v^2/(2 mu g) and re-sleeps."""
        s = self.scene(nx=1, nz=1, base_height=0., friction=.5)
        self.run_until(s, 80, stop=lambda d: d[0]['awake_cubes'] == 0)
        start = s.read().cube_pose[0, 0].copy()
        self.assertEqual(s.override_cube(0, 0, start, [.5, 0, 0, 0, 0, 0]), 0)
        t, d = self.run_until(s, 600, stop=lambda d: d[0]['awake_cubes'] == 0)
        self.assertEqual(d[0]['awake_cubes'], 0, 'cube stopped and slept')
        slid = s.read().cube_pose[0, 0, 0]-start[0]
        expected = .5**2/(2*.5*9.81)
        self.assertGreater(slid, .5*expected)
        self.assertLess(slid, 1.5*expected)

    def test_stack_stays_stable(self):
        """Two stacked cubes hold position for 2 s and sleep."""
        s = self.scene(nx=2, nz=1, spacing=.2, base_height=0.)
        self.run_until(s, 80, stop=lambda d: d[0]['awake_cubes'] == 0)
        base = s.read().cube_pose[0, 0].copy()
        top = base.copy(); top[2] += .051
        self.assertEqual(s.override_cube(0, 1, top, [0.]*6), 0)
        self.run_until(s, 1000)  # 2 s
        x = s.read()
        np.testing.assert_allclose(x.cube_pose[0, 0, 2], .025, atol=1e-3)
        np.testing.assert_allclose(x.cube_pose[0, 1, 2], .075, atol=2e-3)
        np.testing.assert_allclose(x.cube_pose[0, 1, :2], base[:2], atol=2e-3)
        self.assertTrue((x.cube_awake[0] == 0).all(), 'stack asleep')

    def test_island_partition(self):
        """Pair + two singletons: 3 cube islands, merged only through contact."""
        s = self.scene(nx=2, nz=2, spacing=.3, base_height=0., cube_size=.05)
        self.run_until(s, 80, stop=lambda d: d[0]['awake_cubes'] == 0)
        x = s.read()
        poses = x.cube_pose[0].copy()
        # butt cube1 against cube0 (2mm overlap) and wake everything
        poses[1] = poses[0]; poses[1][0] += .048
        self.assertEqual(s.override_cube(0, 1, poses[1], [0.]*6), 0)
        for c in (0, 2, 3):
            self.assertEqual(s.override_cube(0, c, poses[c], [0.]*6), 0)
        target = np.tile(HOME, (1, 1))
        rc, d = s.step(target=target, **SOLVE)
        self.assertEqual(rc, 0, d[0])
        self.assertEqual(d[0]['awake_cubes'], 4)
        # duck island + {0,1} + {2} + {3}
        self.assertEqual(d[0]['islands'], 4)
        self.assertEqual(d[0]['duck_island_cubes'], 0)
        self.assertGreaterEqual(d[0]['max_island_dofs'], 12, 'pair island is 2 cubes')

    def test_duck_on_dynamic_cubes_joins_island(self):
        """Duck standing on dynamic cubes couples them into its island."""
        s = self.scene(nx=2, nz=2, cube_size=.1, spacing=.12, base_height=0.,
                       cube_mass=.5, friction=1., origin_x=-.03, seed=5, lift=None)
        target = np.tile(HOME, (1, 1))
        seen = 0
        for t in range(1000):
            rc, d = s.step(target=target, **SOLVE)
            self.assertEqual(rc, 0, (t, d[0]))
            self.assertLessEqual(d[0]['momentum_residual'], 1e-8)
            seen = max(seen, d[0]['duck_island_cubes'])
        self.assertGreaterEqual(seen, 1, 'cube joined the duck island')
        x = s.read()
        q = x.q[0, 3:7]
        tilt = np.degrees(np.arccos(min(1., abs(1-2*(q[0]**2+q[1]**2)))))
        self.assertLess(tilt, 5., 'duck upright on dynamic cubes')
        np.testing.assert_allclose(x.cube_pose[0, :, 2], .05, atol=3e-3)

    def test_override_rejected_for_static(self):
        s = self.scene(dynamic=0)
        rc = s.override_cube(0, 0, [1, 0, 0.03, 0, 0, 0, 1], [0.]*6)
        self.assertNotEqual(rc, 0)


if __name__ == '__main__':
    unittest.main()
