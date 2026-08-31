"""Milestone 1 gates: duck on a STATIC cube grid (dwv1).

Run: .venv/bin/python -B experimental/duck_world_v1/tests/test_static_grid.py
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'walk/env'))
import world as w  # noqa: E402

np = w.np
HOME = np.array(w.HOME)


def tilt_degrees(q_xyzw):
    return np.degrees(np.arccos(min(1., abs(1-2*(q_xyzw[0]**2+q_xyzw[1]**2)))))


class StaticGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = w.library(w.build())

    def scene(self, environments=1, **kw):
        spec = dict(nx=8, nz=8, cube_size=.06, spacing=.06, base_height=0.,
                    height_jitter=0., friction=.8, seed=3)
        spec.update(kw)
        s, cm = w.duck_grid_scene(self.lib, environments, w.grid_spec(**spec), gap=.0015)
        self.addCleanup(s.close)
        return s

    def hold(self, s, ticks, **kw):
        target = np.tile(HOME, (s.E, 1))
        diagnostics = []
        for t in range(ticks):
            rc, d = s.step(target=target, **kw)
            self.assertEqual(rc, 0, (t, d[0]))
            for x in d:
                self.assertLessEqual(x['momentum_residual'], 1e-8, (t, x))
            diagnostics.append(d)
        return diagnostics

    def test_rest_and_upright_hold(self):
        """Duck at HOME 1.5mm above the grid rests within 1s, upright 2s more."""
        s = self.scene()
        self.hold(s, 500)
        x = s.read()
        self.assertTrue(all(x.foot[0]), 'both feet in contact after 1s')
        self.assertLess(np.abs(x.v[0, :6]).max(), .01, 'root at rest after 1s')
        diagnostics = self.hold(s, 1000)
        x = s.read()
        self.assertLess(tilt_degrees(x.q[0, 3:7]), 2., 'upright after 2s hold')
        self.assertTrue(all(x.foot[0]))
        last = diagnostics[-1][0]
        self.assertEqual(last['islands'], 1)
        self.assertEqual(last['awake_cubes'], 0)
        self.assertGreater(last['contact_points'], 0)
        worst = max(d[0]['maximum_penetration'] for d in diagnostics)
        self.assertLess(worst, .002, 'penetration bounded (repair term only)')

    def test_jittered_grid_contact(self):
        """Feet contact seed-jittered cube tops; heights are deterministic."""
        s = self.scene(spacing=.062, height_jitter=.005, seed=41)
        x = s.read()
        heights = x.cube_pose[0, :, 2]
        self.assertGreater(heights.max()-heights.min(), 1e-4, 'jitter applied')
        s2 = self.scene(spacing=.062, height_jitter=.005, seed=42)
        self.assertFalse(np.array_equal(heights, s2.read().cube_pose[0, :, 2]),
                         'different seed, different heights')
        diagnostics = self.hold(s, 600)
        self.assertGreater(diagnostics[-1][0]['contact_points'], 0)
        contacts = s.query(0)
        self.assertTrue(any(c.kind_a == 0 and c.kind_b == 1 for c in contacts),
                        'foot-cube manifolds present')

    def test_deterministic_trajectory(self):
        """Same seed twice: bit-identical state at every checkpoint."""
        streams = []
        for _ in range(2):
            s = self.scene(height_jitter=.004, seed=7)
            checkpoints = []
            target = np.tile(HOME, (1, 1))
            for t in range(400):
                rc, d = s.step(target=target)
                self.assertEqual(rc, 0, (t, d[0]))
                if t % 50 == 49:
                    checkpoints.append(s.read().bytes())
            streams.append(checkpoints)
        self.assertEqual(streams[0], streams[1])

    def test_masked_restore_and_reset(self):
        """Masked restore is bit-exact for selected envs, no-op for others."""
        s = self.scene(environments=2)
        target = np.tile(HOME, (2, 1))
        target[1, 0] += .05  # desynchronize the two environments
        for t in range(60):
            rc, d = s.step(target=target)
            self.assertEqual(rc, 0, (t, d))
        snapshot = s.capture()
        at_capture = s.read()
        for t in range(60):
            rc, d = s.step(target=target)
            self.assertEqual(rc, 0, (t, d))
        before = s.read()
        self.assertNotEqual(before.bytes(0), at_capture.bytes(0))
        self.assertEqual(s.reset(mask=[1, 0], snapshot=snapshot), 0)
        after = s.read()
        self.assertEqual(after.bytes(0), at_capture.bytes(0), 'env0 restored exactly')
        self.assertEqual(after.bytes(1), before.bytes(1), 'env1 untouched')
        initial = self.scene(environments=2).read()
        self.assertEqual(s.reset(mask=[0, 1]), 0)
        after = s.read()
        self.assertEqual(after.bytes(0), at_capture.bytes(0), 'env0 still restored')
        self.assertEqual(after.bytes(1), initial.bytes(1), 'env1 reset to initial')

    def test_failed_step_leaves_state_unchanged(self):
        """A rejected step (invalid solver tolerance) is fully transactional."""
        s = self.scene()
        self.hold(s, 50)
        before = s.read().bytes()
        rc, d = s.step(target=np.tile(HOME, (1, 1)), tolerance=1.)  # civ1 rejects >1e-5
        self.assertNotEqual(rc, 0)
        self.assertEqual(d[0]['phase'], 3, 'rejected in the solve phase')
        self.assertEqual(s.read().bytes(), before, 'no state change on failure')
        rc, d = s.step(target=np.tile(HOME, (1, 1)))
        self.assertEqual(rc, 0, d[0])


if __name__ == '__main__':
    unittest.main()
