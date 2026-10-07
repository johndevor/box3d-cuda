"""Tests for the cube-terrain acceptance judge (walk/eval/grid_acceptance.py).

Run: .venv/bin/python -B -m unittest walk.eval.tests.test_grid_acceptance
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from walk.env.grid_lane import resolve_grid
from walk.eval import grid_acceptance as ga


class TestStageSpecs(unittest.TestCase):
    def test_ladder_stages(self):
        self.assertEqual(sorted(ga.STAGES), ["flush", "rough4", "rough8"])
        for name, spec in ga.STAGES.items():
            self.assertEqual((spec["nx"], spec["nz"]), (45, 10), name)
            self.assertEqual(spec["cube_size"], 0.06, name)
            self.assertEqual(spec["spacing"], 0.06, name)   # flush pitch
            self.assertFalse(spec["dynamic"], name)
            self.assertNotIn("seed", spec,
                             "terrain seed must follow the env seed")
            self.assertLessEqual(spec["nx"] * spec["nz"], 1024)  # dwv1 cap
        self.assertEqual(ga.STAGES["flush"]["height_jitter"], 0.0)
        self.assertEqual(ga.STAGES["rough4"]["height_jitter"], 0.004)
        self.assertEqual(ga.STAGES["rough8"]["height_jitter"], 0.008)

    def test_stage_covers_strict_travel_envelope(self):
        """dwv1 places cube ix at origin_x + (ix - (nx-1)/2) * pitch, so the
        terrain (cube edges included) must cover the worst-case strict
        translation bound ahead of the duck's start at x = 0 — 0.20 m/s x
        8 s x 150% = 2.4 m — plus margin behind and laterally."""
        for name, spec in ga.STAGES.items():
            pitch = spec["spacing"]
            half = spec["cube_size"] / 2
            x_lo = spec["origin_x"] - (spec["nx"] - 1) / 2 * pitch - half
            x_hi = spec["origin_x"] + (spec["nx"] - 1) / 2 * pitch + half
            y_lo = spec["origin_y"] - (spec["nz"] - 1) / 2 * pitch - half
            y_hi = spec["origin_y"] + (spec["nz"] - 1) / 2 * pitch + half
            eps = 1e-9
            self.assertLessEqual(x_lo, -0.30 + eps, name)    # margin behind
            self.assertGreaterEqual(x_hi, 2.40 - eps, name)  # 0.20 * 8 * 1.5
            self.assertLessEqual(y_lo, -0.30 + eps, name)    # lateral margin
            self.assertGreaterEqual(y_hi, 0.30 - eps, name)

    def test_terrain_seed_follows_env_seed(self):
        for stage in ga.STAGES:
            for seed in ga.SEEDS:
                spec = resolve_grid(ga.stage_grid(stage), default_seed=seed)
                self.assertEqual(spec["seed"], seed, (stage, seed))

    def test_stage_grid_returns_copy_and_rejects_unknown(self):
        spec = ga.stage_grid("flush")
        spec["height_jitter"] = 999.0
        self.assertEqual(ga.STAGES["flush"]["height_jitter"], 0.0)
        with self.assertRaises(ValueError):
            ga.stage_grid("lava")

    def test_seeds_and_commands_match_flat_acceptance(self):
        from walk.eval import acceptance
        self.assertEqual(tuple(ga.SEEDS), tuple(acceptance.SEEDS))
        self.assertEqual(tuple(ga.COMMANDS), tuple(acceptance.COMMANDS))


def _fake_seed_result(seed, commands, passed):
    episodes, details = {}, {}
    for cmd in commands:
        key = f"seed{seed}-cmd{cmd:.2f}"
        episodes[key] = {"passed": passed, "qualified": 6, "left": 3,
                         "right": 3, "failed_criteria": [],
                         "duration_s": 8.0, "base_x_travel_m": 1.0,
                         "terminated": False, "solver_fault": False}
        details[key] = {"passed": passed, "criteria": {}}
    return {"grid": {"nx": 8, "seed": seed}, "episodes": episodes,
            "details": details}


class TestJudgePlumbing(unittest.TestCase):
    def test_aggregation_all_pass(self):
        with mock.patch.object(ga, "judge_seed",
                               side_effect=lambda a, st, s, c, *rest:
                               _fake_seed_result(s, c, True)) as m:
            rec = ga.judge("actor.pt", "rough8", seeds=(1, 2),
                           commands=(0.10, 0.20), jobs=1, verbose=False)
        self.assertEqual(m.call_count, 2)
        self.assertTrue(rec["accepted"])
        self.assertEqual(rec["schema"], "duckgridwalk.grid-acceptance/1")
        self.assertEqual(rec["stage"], "rough8")
        self.assertEqual(rec["stage_grid"], ga.STAGES["rough8"])
        self.assertEqual(sorted(rec["episodes"]),
                         ["seed1-cmd0.10", "seed1-cmd0.20",
                          "seed2-cmd0.10", "seed2-cmd0.20"])
        self.assertEqual(sorted(rec["resolved_grids_by_seed"]), ["1", "2"])

    def test_aggregation_one_failure_rejects(self):
        def fake(actor, stage, seed, commands, *rest):
            return _fake_seed_result(seed, commands, seed != 2)
        with mock.patch.object(ga, "judge_seed", side_effect=fake):
            rec = ga.judge("actor.pt", "flush", seeds=(1, 2, 3),
                           commands=(0.15,), jobs=1, verbose=False)
        self.assertFalse(rec["accepted"])
        self.assertFalse(rec["episodes"]["seed2-cmd0.15"]["passed"])
        self.assertTrue(rec["episodes"]["seed1-cmd0.15"]["passed"])


class TestJudgeTinyReal(unittest.TestCase):
    """Short real capture through the whole pipeline: CubeGridDuckEnv ->
    capture_episodes -> gait.evaluate_episode. 0.4 s episodes never pass the
    8 s integrity criterion; the point is exercising the real seams."""

    def test_short_real_episode_produces_structured_result(self):
        import torch
        from walk.train.ppo import Actor
        with tempfile.TemporaryDirectory() as td:
            actor_path = Path(td) / "tiny.pt"
            torch.save({"arch": "ff",
                        "state_dict": Actor(58, 14).state_dict()}, actor_path)
            res = ga.judge_seed(str(actor_path), "flush", seed=0,
                                commands=(0.15,), seconds=0.4)
        self.assertEqual(res["grid"]["seed"], 0)     # terrain seed = env seed
        self.assertEqual(res["grid"]["nx"], 45)
        self.assertEqual(res["grid"]["origin_x"], 1.05)
        rec = res["episodes"]["seed0-cmd0.15"]
        self.assertFalse(rec["passed"])
        for key in ("qualified", "left", "right", "failed_criteria",
                    "duration_s", "base_x_travel_m", "terminated",
                    "solver_fault"):
            self.assertIn(key, rec)
        detail = res["details"]["seed0-cmd0.15"]
        self.assertEqual(detail["schema"], "duckgridwalk.gait_eval/1")
        self.assertIn("single_episode_no_reset_or_failure", detail["criteria"])
        # JSON-serialisable with the judge's writer settings
        self.assertTrue(json.dumps(res, default=ga._json_default))

    def test_json_default_handles_numpy_scalars(self):
        """gait.py criteria can contain np.bool_ / np.float64 (surfaced by
        the first rough4 baseline with qualified footfalls)."""
        blob = {"a": np.bool_(True), "b": np.float64(1.5), "c": np.int64(3)}
        self.assertEqual(json.loads(json.dumps(blob, default=ga._json_default)),
                         {"a": True, "b": 1.5, "c": 3})
        with self.assertRaises(TypeError):
            json.dumps({"x": object()}, default=ga._json_default)


class TestObsParitySmoke(unittest.TestCase):
    """The accepted actor was trained on flat.py's obs semantics; the flush
    grid must present the same first observation in the shared slots
    (mirrors walk/env/tests/test_grid.py::TestObsParity)."""

    def test_flush_first_obs_matches_flat(self):
        from walk.env.flat import FlatFloorDuckEnv
        from walk.env.grid import CubeGridDuckEnv
        flat = FlatFloorDuckEnv(environments=1, seed=7)
        try:
            obs_flat = flat.reset()
        finally:
            flat.close()
        env = CubeGridDuckEnv(environments=1, seed=7,
                              grid=ga.stage_grid("flush"))
        try:
            obs_grid = env.reset()
        finally:
            env.close()
        for name, sl in [("joint q", slice(0, 14)),
                         ("joint qdot", slice(14, 28)),
                         ("gravity", slice(42, 45)),
                         ("command", slice(51, 54))]:
            np.testing.assert_allclose(
                obs_grid[:, sl], obs_flat[:, sl], rtol=0, atol=1e-6,
                err_msg=f"{name} obs slots must match flat")
        np.testing.assert_array_equal(obs_grid[:, 51], obs_flat[:, 51])


if __name__ == "__main__":
    unittest.main()
