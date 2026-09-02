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
        # every non-terminal stage advances on episode-length-normalised
        # evidence (acquisitions per live second), ramping toward the
        # judge's 5 acquisitions in 8 s = 0.625 acq/s
        self.assertTrue(all(s.advance_metric == "acq_per_live_s"
                            for s in stages[:-1]))
        thresholds = [s.advance_threshold for s in stages[:-1]]
        self.assertEqual(thresholds, sorted(thresholds))
        self.assertLess(thresholds[-1], judge.N_TARGETS / judge.EPISODE_SECONDS)
        # every cap stage ALSO advances once its own median violating ticks
        # are inside the judge's tolerance
        for st in stages[1:5]:
            self.assertEqual((st.advance_metric2, st.advance_below2),
                             ("alt_violations", True))
            self.assertLessEqual(st.advance_threshold2, 5.0)
        caps = [s.max_alternation_violations for s in stages[1:]]
        self.assertEqual(caps, [2000, 800, 200, 40, 1, 1])   # loose -> tight
        self.assertEqual(caps, sorted(caps, reverse=True))
        # the first cap sits ABOVE the measured noisy-policy baseline
        # (~2300 violating ticks/episode, runs/arm-local-v5-long)
        self.assertGreaterEqual(caps[0], 2000)
        self.assertEqual(stages[-2].advance_threshold, 0.5)   # 4 acq per 8 s
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
        row["gate_proxy_ep_qualified_l"] = 0.08   # 0.08 per 8 s = 0.01 acq/s
        # the trailing-8 median crosses once 5 of the last 8 rows carry
        # acquisitions (the window is cumulative until a transition)
        events = [ctl.observe({**row, "update": 100 + u}) for u in range(8)]
        fired = [e for e in events if e is not None]
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0]["direction"], "advance")
        self.assertEqual(ctl.stage.name, "violations_cap_2000")
        # normalisation: 0.12 acquisitions over an 8 s episode (0.015 acq/s)
        # does not clear 0.03 acq/s ...
        row["gate_proxy_ep_qualified_l"] = 0.12
        for u in range(12):
            self.assertIsNone(ctl.observe({**row, "update": 200 + u}))
        self.assertEqual(ctl.stage.name, "violations_cap_2000")
        # ... but the SAME 0.12 over 3 s episodes (0.04 acq/s) does: a cap
        # that shortens episodes no longer suppresses its own advance
        row["ep_len_mean"] = 150.0
        events = [ctl.observe({**row, "update": 300 + u}) for u in range(8)]
        self.assertTrue(any(e for e in events))
        self.assertEqual(ctl.stage.name, "violations_cap_800")
        # OR-path: violating ticks already inside the judge's tolerance
        row.update({"gate_proxy_ep_qualified_l": 0.0,
                    "gate_proxy_ep_alt_violations": 3.0, "ep_len_mean": 400.0})
        events = [ctl.observe({**row, "update": 400 + u}) for u in range(8)]
        self.assertTrue(any(e for e in events))
        self.assertEqual(ctl.stage.name, "violations_cap_200")
        self.assertIn("alt_violations", [e for e in events if e][0]["reason"])

    def _replay(self, path):
        import json
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        train = [r for r in rows if r.get("kind") == "train" and "reward_mean" in r]
        ctl = cc.CurriculumController(cc.load_ladder("arm-reach", "arm"), quiet=True)
        events = []
        for r in train:
            ev = ctl.observe(r)
            if ev:
                events.append((ev["update"], ev["direction"], ev["name"]))
        return events, ctl.stage.name

    def test_offline_replay_of_recorded_runs_advances_past_deadline_cap_40(self):
        """The raw acq/episode rule never left deadline_cap_40 in either
        recorded run (GPU: 153 updates at median 0.41/episode; local:
        2267 updates); the normalised rule must, within each run."""
        runs = {"gpu": ROOT / "runs/gpu/20260902-160543-arm-reach-kr240/artifacts/train/gpu-train-out/metrics.jsonl",
                "local": ROOT / "runs/arm-local-delta/metrics.jsonl"}
        seen = 0
        for name, path in runs.items():
            if not path.is_file():
                continue
            seen += 1
            events, final = self._replay(path)
            names = [n for _u, _d, n in events]
            self.assertIn("judge_tight", names,
                          f"{name}: never advanced past deadline_cap_40: {events}")
            u_adv = next(u for u, _d, n in events if n == "judge_tight")
            print(f"{name} offline replay: {events} -> final stage {final} "
                  f"(judge_tight at update {u_adv})", file=sys.stderr)
        if not seen:
            self.skipTest("no recorded metrics.jsonl present")

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
