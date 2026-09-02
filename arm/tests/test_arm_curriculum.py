"""arm-reach tech-tree ladder: loads for the arm only, knobs map onto the
reach semantics of set_gate_termination, and the controller advances on
the reach mapping of gate_proxy_* (acquisitions / violating ticks).

Run: .venv/bin/python -B arm/tests/test_arm_curriculum.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "arm"))

from walk.env.arm_cuda_lane import CudaArmLane  # noqa: E402
from walk.eval import arm_reach_judge as judge  # noqa: E402
from walk.train import curriculum_controller as cc  # noqa: E402
from walk.train import gpu_train as gt  # noqa: E402


class ArmLadderTests(unittest.TestCase):
    def test_ladder_loads_for_arm_only_and_is_monotone(self):
        stages = cc.load_ladder("arm-reach", "arm")
        self.assertEqual([s.name for s in stages],
                         ["free", "violations_cap_2000", "violations_cap_800",
                          "violations_cap_200", "deadline_cap_40",
                          "judge_tight", "full"])
        self.assertIsNone(stages[-1].advance_metric)
        thresholds = [s.advance_threshold for s in stages[:-1]]
        self.assertEqual(thresholds, sorted(thresholds))
        self.assertLess(thresholds[-1], judge.N_TARGETS + 1)
        caps = [s.max_alternation_violations for s in stages[1:]]
        self.assertEqual(caps, [2000, 800, 200, 40, 1, 1])   # loose -> tight
        self.assertEqual(caps, sorted(caps, reverse=True))
        # the first cap sits ABOVE the measured noisy-policy baseline
        # (~2300 violating ticks/episode, runs/arm-local-v5-long)
        self.assertGreaterEqual(caps[0], 2000)
        self.assertEqual(stages[-2].advance_threshold, 4.0)
        self.assertEqual(stages[-1].max_alternation_violations, 1,
                         "judge-tight = zero violating ticks tolerated")
        self.assertTrue(all(s.commands is None for s in stages),
                        "the arm has no command channel")
        for robot in ("duck", "humanoid"):
            with self.assertRaises(SystemExit):
                cc.load_ladder("arm-reach", robot)
        with self.assertRaises(SystemExit):
            cc.load_ladder("humanoid-walk", "arm")

    def test_controller_advances_on_reach_gate_proxy_rows(self):
        ctl = cc.CurriculumController(cc.load_ladder("arm-reach", "arm"),
                                      quiet=True)
        row = {"update": 0, "gate_proxy_ep_qualified_l": 0.0,
               "gate_proxy_ep_qualified_r": 0.0,
               "gate_proxy_ep_alt_violations": 2400.0, "ep_len_mean": 400.0}
        for u in range(20):                       # shuffle: never advances
            self.assertIsNone(ctl.observe({**row, "update": u}))
        self.assertEqual(ctl.stage.name, "free")
        row["gate_proxy_ep_qualified_l"] = 0.08   # acquisitions appear (measured scale)
        # the trailing-8 median crosses 1.0 once 5 of the last 8 rows carry
        # acquisitions (the window is cumulative until a transition)
        events = [ctl.observe({**row, "update": 100 + u}) for u in range(8)]
        fired = [e for e in events if e is not None]
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0]["direction"], "advance")
        self.assertEqual(ctl.stage.name, "violations_cap_2000")
        # violating ticks do not gate stage 2's advance (the cap knob does
        # the enforcing); acquisitions must keep rising: median 0.25 needed
        row["gate_proxy_ep_qualified_l"] = 0.12
        for u in range(12):
            self.assertIsNone(ctl.observe({**row, "update": 200 + u}))
        self.assertEqual(ctl.stage.name, "violations_cap_2000")

    def test_apply_pushes_reach_knobs_to_the_lane(self):
        cfg = gt.GpuTrainConfig(robot="arm", variant="kr240", envs=2,
                                lane_env=True, out="/tmp/unused")
        env = gt.LanePolicyEnv(cfg)
        try:
            stages = cc.load_ladder("arm-reach", "arm")
            ctl = cc.CurriculumController(stages, quiet=True)
            ctl.index = len(stages) - 1               # 'full'
            ctl.apply(env)
            self.assertIsNone(env.command_override)
            # zero violating ticks tolerated: a flail dies fast with the
            # violating-tick reason (3) or the 2 s deadline (2)
            env.reset(seed=1)
            rng = np.random.default_rng(0)
            died = None
            for t in range(120):
                _o, _r, d, _i = env.step(rng.uniform(-1, 1, (2, 6)))
                if d.any():
                    died = t
                    break
            self.assertIsNotNone(died)
            self.assertLess(died, 100, "2 s deadline = step 100 at most")
            reasons = set(env._lane.gate_proxy()["termination_reason"].tolist())
            self.assertTrue(reasons & {2, 3}, reasons)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
