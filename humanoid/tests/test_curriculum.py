"""Tech-tree curriculum controller gates: ladder, transitions, smoke.

Run: .venv/bin/python -B humanoid/tests/test_curriculum.py

Unit gates (synthetic metric streams -- no physics):
- the humanoid-walk ladder is structurally sound (monotone enforcement,
  judge-anchored knobs, terminal 'full');
- advances exactly when the stage's median-over-window predicate is met
  AND the full dwell has elapsed (a single spike cannot skip a node);
- de-escalates ONE stage after collapse_k consecutive collapsed updates,
  re-advances after recovery, never de-escalates below the first stage;
- never advances past 'full' no matter the metrics;
- guards: duck has no ladder; unknown specs rejected; JSON ladders load
  and are robot-checked.

Integration: gpu_train CPU smoke with --curriculum humanoid-walk on the
serial lane (2 envs, few updates): initial stage applied, per-row
curriculum_stage present, knobs callable, zero crash; plus the config
guards (--curriculum without --lane-env, duck + humanoid-walk).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "humanoid"))

from walk.train import curriculum_controller as cc  # noqa: E402
from walk.train import gpu_train as gt  # noqa: E402


def row(update, qualified=0.0, alt=0.0, ep_len=200.0):
    """A synthetic gpu_train metrics row (only the keys the controller uses)."""
    return {"update": update,
            "gate_proxy_ep_qualified_l": qualified / 2.0,
            "gate_proxy_ep_qualified_r": qualified / 2.0,
            "gate_proxy_ep_alt_violations": alt,
            "ep_len_mean": ep_len}


class LadderTests(unittest.TestCase):
    def test_humanoid_walk_ladder_structure(self):
        stages = cc.load_ladder("humanoid-walk", "humanoid")
        names = [s.name for s in stages]
        self.assertEqual(names, ["free", "swings_appear", "deadline_loose",
                                 "deadline_judge", "alternation_cap", "full"])
        # first two stages enforce nothing
        for s in stages[:2]:
            self.assertEqual((s.first_deadline_ticks,
                              s.max_alternation_violations), (0, 0))
        # deadline anchored to the FROZEN judge: loose = 2x, judge = exact
        from walk.eval import humanoid_gait
        judge_ticks = int(round(humanoid_gait.FIRST_STEP_S / 0.002))
        self.assertEqual(stages[2].first_deadline_ticks, 2 * judge_ticks)
        self.assertEqual(stages[3].first_deadline_ticks, judge_ticks)
        self.assertEqual(stages[5].first_deadline_ticks, judge_ticks)
        # alternation tightens 0 -> 3 -> 1, never loosens
        self.assertEqual([s.max_alternation_violations for s in stages],
                         [0, 0, 0, 0, 3, 1])
        # advance thresholds climb to the judge's MIN_FOOTFALLS
        self.assertEqual([s.advance_threshold for s in stages[:-1]],
                         [0.5, 2.0, 3.0, 4.0, 6.0])
        self.assertEqual(stages[4].advance_threshold,
                         float(humanoid_gait.MIN_FOOTFALLS))
        self.assertIsNone(stages[5].advance_metric)      # terminal
        # judge scores all three commands: enforced stages carry all three
        for s in stages[3:]:
            self.assertEqual(s.commands, (0.50, 0.75, 1.00))

    def test_guards(self):
        with self.assertRaises(SystemExit):
            cc.load_ladder("humanoid-walk", "duck")      # no duck ladder
        with self.assertRaises(SystemExit):
            cc.load_ladder("no-such-ladder", "humanoid")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ladder.json"
            path.write_text(json.dumps({
                "robot": "humanoid",
                "stages": [{"name": "only", "commands": [0.75],
                            "advance_metric": None}]}))
            stages = cc.load_ladder(str(path), "humanoid")
            self.assertEqual(stages[0].name, "only")
            self.assertEqual(stages[0].commands, (0.75,))
            with self.assertRaises(SystemExit):
                cc.load_ladder(str(path), "duck")        # robot mismatch


class TransitionTests(unittest.TestCase):
    def controller(self):
        return cc.CurriculumController(
            cc.load_ladder("humanoid-walk", "humanoid"), quiet=True)

    def test_advances_exactly_on_median_after_full_dwell(self):
        c = self.controller()
        K = c.stage.advance_window
        # K-1 qualifying updates: dwell not complete, no event
        for u in range(K - 1):
            self.assertIsNone(c.observe(row(u, qualified=1.0)))
        ev = c.observe(row(K - 1, qualified=1.0))
        self.assertIsNotNone(ev)
        self.assertEqual((ev["direction"], ev["name"], ev["from"]),
                         ("advance", "swings_appear", "free"))
        self.assertEqual(ev["update"], K - 1)

    def test_single_spike_cannot_skip(self):
        c = self.controller()
        K = c.stage.advance_window
        # median gate: one huge spike among below-threshold rows never advances
        for u in range(3 * K):
            q = 100.0 if u % K == 0 else 0.0
            self.assertIsNone(c.observe(row(u, qualified=q)), u)
        self.assertEqual(c.stage.name, "free")

    def test_window_resets_on_transition(self):
        c = self.controller()
        K = c.stage.advance_window
        for u in range(K):
            c.observe(row(u, qualified=1.0))             # -> swings_appear
        self.assertEqual(c.stage.name, "swings_appear")
        # old window must not leak: even with qualifying metrics, the new
        # stage needs its own full dwell
        for u in range(K, 2 * K - 1):
            self.assertIsNone(c.observe(row(u, qualified=5.0)))
        ev = c.observe(row(2 * K - 1, qualified=5.0))
        self.assertEqual(ev["name"], "deadline_loose")

    def test_never_advances_past_full(self):
        c = self.controller()
        u = 0
        for _ in range(len(c.stages) + 3):               # drive to terminal
            for _ in range(c.stage.advance_window):
                c.observe(row(u, qualified=100.0))
                u += 1
        self.assertEqual(c.stage.name, "full")
        for _ in range(3 * c.stage.advance_window):
            self.assertIsNone(c.observe(row(u, qualified=1000.0)))
            u += 1
        self.assertEqual(c.stage.name, "full")

    def test_deescalates_on_collapse_and_recovers(self):
        c = self.controller()
        u = 0
        for _ in range(2):                               # -> deadline_loose
            for _ in range(c.stage.advance_window):
                c.observe(row(u, qualified=10.0))
                u += 1
        self.assertEqual(c.stage.name, "deadline_loose")
        k = c.stage.collapse_k
        floor = c.stage.collapse_ep_len
        for i in range(k - 1):                           # not yet consecutive
            self.assertIsNone(c.observe(row(u, ep_len=floor - 1)))
            u += 1
        ev = c.observe(row(u, ep_len=floor - 1))
        u += 1
        self.assertIsNotNone(ev)
        self.assertEqual((ev["direction"], ev["name"], ev["from"]),
                         ("de-escalate", "swings_appear", "deadline_loose"))
        # a healthy update in the middle breaks the consecutive count
        c2 = self.controller()
        c2.index = 2
        for i in range(k - 1):
            c2.observe(row(i, ep_len=1.0))
        self.assertIsNone(c2.observe(row(k, ep_len=200.0)))
        for i in range(k - 1):
            self.assertIsNone(c2.observe(row(k + 1 + i, ep_len=1.0)))
        # recovery: after falling back, qualifying metrics re-advance
        for _ in range(c.stage.advance_window):
            ev2 = c.observe(row(u, qualified=10.0))
            u += 1
        self.assertIsNotNone(ev2)
        self.assertEqual(ev2["name"], "deadline_loose")

    def test_never_below_first_stage(self):
        c = self.controller()
        for u in range(4 * c.stage.collapse_k):
            self.assertIsNone(c.observe(row(u, ep_len=0.0)))
        self.assertEqual(c.index, 0)

    def test_ep_len_none_means_nobody_died(self):
        c = self.controller()
        c.index = 3                                      # enforced stage
        line = row(0)
        line["ep_len_mean"] = None                       # no completed episode
        for u in range(3 * c.stage.collapse_k):
            self.assertIsNone(c.observe(dict(line, update=u)))
        self.assertEqual(c.index, 3)


class GpuTrainIntegration(unittest.TestCase):
    def test_config_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = dict(envs=1, horizon=2, updates=1, device="cpu",
                        out=str(Path(tmp) / "x"), preflight_steps=0,
                        checkpoint_every=1000, torch_threads=1, quiet=True)
            with self.assertRaises(SystemExit):          # duck has no ladder
                gt.train(gt.GpuTrainConfig(robot="duck", lane_env=True,
                                           curriculum="humanoid-walk", **base))
            with self.assertRaises(SystemExit):          # needs --lane-env
                gt.train(gt.GpuTrainConfig(robot="humanoid", lane_env=False,
                                           curriculum="humanoid-walk", **base))

    def test_cpu_smoke_with_ladder_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = gt.GpuTrainConfig(
                robot="humanoid", lane_env=True, curriculum="humanoid-walk",
                envs=2, horizon=8, updates=3, seed=20260901, device="cpu",
                out=str(Path(tmp) / "smoke"), preflight_steps=0,
                checkpoint_every=1000, torch_threads=1, quiet=True)
            metrics = gt.train(cfg)
            rows = [m for m in metrics if m.get("kind") == "train"]
            self.assertEqual(len(rows), 3)
            for r in rows:
                self.assertEqual(r["faults"], 0, r)
                self.assertIn("curriculum_stage", r)
                self.assertTrue(np.isfinite(r["reward_mean"]), r)
            # BC-less fresh policy stays in the free stage over 3 updates
            self.assertEqual(rows[0]["curriculum_stage"], "free")
            # command override active: free pins every episode to 0.75
            lines = (Path(cfg.out) / "metrics.jsonl").read_text().splitlines()
            self.assertTrue(any('"curriculum_stage"' in ln for ln in lines))
            print("curriculum smoke:",
                  {k: rows[-1].get(k) for k in
                   ("update", "curriculum_stage", "reward_mean", "faults",
                    "gate_proxy_ep_qualified_l")}, file=sys.stderr)

    def test_command_override_reaches_lane(self):
        cfg = gt.GpuTrainConfig(robot="humanoid", lane_env=True, envs=2,
                                seed=7, perturbation=0.0)
        env = gt.LanePolicyEnv(cfg)
        try:
            env.command_override = (0.75,)
            obs = env.reset()
            T = 3 * env.ACT
            np.testing.assert_allclose(obs[:, T + 9], 0.75, atol=0)
            env.command_override = None                  # env's own draw back
            obs = env.reset()
            self.assertTrue(set(np.round(obs[:, T + 9], 2))
                            <= {0.50, 0.75, 1.00})
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
