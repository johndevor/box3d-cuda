"""Capture round-trip against the real flat-floor env (short episode).

Run: .venv/bin/python -B -m unittest discover -s walk
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from walk.env.flat import FlatFloorDuckEnv
from walk.eval import capture, gait


class TestCaptureRoundTrip(unittest.TestCase):
    def test_zero_policy_capture_records_per_tick_trace(self):
        env = FlatFloorDuckEnv(environments=1, seed=0)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                traces = capture.capture_episodes(
                    env, lambda obs: np.zeros((1, 14), np.float32),
                    command=0.15, seconds=1.0, out_dir=tmp, seed=0)
                files = sorted(Path(tmp).glob("*.json"))
                self.assertEqual(len(files), 1)
                reloaded = json.loads(files[0].read_text())
        finally:
            env.close()
        self.assertEqual(len(traces), 1)
        trace = traces[0]
        self.assertEqual(trace["schema"], gait.SCHEMA)
        self.assertEqual(trace["command_mps"], 0.15)
        ticks = trace["ticks"]
        self.assertEqual(len(ticks["time_s"]), 500)  # 1 s at 2 ms per tick
        for key in ["base_pos", "base_quat_xyzw", "tilt_deg", "foot_pos",
                    "sole_height", "contact"]:
            self.assertEqual(len(ticks[key]), 500)
        self.assertFalse(trace["terminated"])
        self.assertEqual(trace["resets"], 0)
        # standing duck: settled, upright, both feet in contact at the end
        self.assertEqual(ticks["contact"][-1], [True, True])
        self.assertLess(max(ticks["tilt_deg"]), 2.0)
        self.assertAlmostEqual(ticks["time_s"][-1], 1.0, places=9)
        self.assertEqual(reloaded, trace)  # JSON round-trip is lossless
        # the strict evaluator must reject a 1 s prefix outright
        result = gait.evaluate_episode(trace)
        self.assertTrue(result["rejected"])
        self.assertIn("shorter than 8 s", result["criteria"]
                      ["single_episode_no_reset_or_failure"]["detail"])


if __name__ == "__main__":
    unittest.main()
