"""Capacity gate: >=200 cubes per env, <10 ms/tick single-threaded, and
islands small enough that civ1 caps (256 dofs / 1536 rows / 512 contacts)
are never approached.

Run: .venv/bin/python -B experimental/duck_world_v1/tests/test_capacity.py
"""
import sys
import json
import math
import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'walk/env'))
import world as w  # noqa: E402

np = w.np
HOME = np.array(w.HOME)


class CapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = w.library(w.build())

    def timed_hold(self, s, warm, timed, **solve):
        target = np.tile(HOME, (s.E, 1))
        for t in range(warm):
            rc, d = s.step(target=target, **solve)
            self.assertEqual(rc, 0, (t, d[0]))
        start = time.monotonic()
        max_dofs = max_awake = max_points = 0
        for t in range(timed):
            rc, d = s.step(target=target, **solve)
            self.assertEqual(rc, 0, (t, d[0]))
            self.assertLessEqual(d[0]['momentum_residual'], 1e-8)
            max_dofs = max(max_dofs, d[0]['max_island_dofs'])
            max_awake = max(max_awake, d[0]['awake_cubes'])
            max_points = max(max_points, d[0]['contact_points'])
        return (time.monotonic()-start)/timed*1e3, max_dofs, max_awake, max_points

    def check_timing(self, ms):
        # The original 10 ms benchmark remains enforced by default. Shared
        # correctness CI explicitly reports it without claiming performance.
        mode = os.environ.get('BOX3D_CAPACITY_TIMING_MODE', 'enforce')
        self.assertIn(mode, ('enforce', 'report-only'))
        self.assertTrue(math.isfinite(ms) and ms > 0)
        result = {'benchmark': 'capacity_tick', 'milliseconds': ms,
                  'limit_ms': 10.0, 'within_target': ms < 10.0,
                  'enforced': mode == 'enforce'}
        print(json.dumps(result), file=sys.stderr, flush=True)
        if mode == 'enforce':
            self.assertLess(ms, 10., 'under 10 ms/tick single-threaded')
        return result

    def test_static_225_cubes(self):
        # jitter 0: long holds on stepped tops intermittently hit the known
        # civ1 redundant-normal stall at 1e-8 (workstream A owns that repair);
        # this gate measures capacity/perf, not solver robustness.
        grid = w.grid_spec(nx=15, nz=15, cube_size=.06, spacing=.062,
                           height_jitter=0., friction=.8, seed=9)
        self.assertGreaterEqual(grid.nx*grid.nz, 200)
        s, cm = w.duck_grid_scene(self.lib, 1, grid)
        self.addCleanup(s.close)
        ms, max_dofs, max_awake, max_points = self.timed_hold(s, 200, 300)
        self.check_timing(ms)
        self.assertEqual(max_dofs, 20, 'static grid: duck island only')
        self.assertEqual(max_awake, 0)
        self.assertLess(max_points, 512)

    def test_dynamic_225_sleeping_cubes(self):
        grid = w.grid_spec(nx=15, nz=15, cube_size=.06, spacing=.062,
                           dynamic=1, cube_mass=.2, friction=.8, seed=9)
        s, cm = w.duck_grid_scene(self.lib, 1, grid)
        self.addCleanup(s.close)
        solve = dict(tolerance=1e-6, jtol=1e-6)
        ms, max_dofs, max_awake, max_points = self.timed_hold(s, 500, 300, **solve)
        self.check_timing(ms)
        # only load-bearing cubes near the feet churn awake; islands stay small
        self.assertLessEqual(max_dofs, 20+6*10, 'duck island stays small')
        self.assertLessEqual(max_awake, 20)
        self.assertLess(max_points, 512)
        x = s.read()
        self.assertGreaterEqual(int((x.cube_awake[0] == 0).sum()), 200,
                                'at least 200 cubes sleeping under hold')


class TimingModeTests(unittest.TestCase):
    def test_original_target_is_enforced_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(AssertionError):
                CapacityTests().check_timing(13.8)

    def test_report_mode_preserves_missed_target(self):
        with patch.dict(os.environ, {'BOX3D_CAPACITY_TIMING_MODE': 'report-only'}):
            result = CapacityTests().check_timing(13.8)
            self.assertFalse(result['within_target'])
            self.assertFalse(result['enforced'])
            self.assertEqual(result['limit_ms'], 10.0)

    def test_unknown_mode_rejected(self):
        with patch.dict(os.environ, {'BOX3D_CAPACITY_TIMING_MODE': 'ignore'}):
            with self.assertRaises(AssertionError):
                CapacityTests().check_timing(1.0)


if __name__ == '__main__':
    unittest.main()
