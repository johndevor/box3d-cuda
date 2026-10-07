"""End-to-end CPU PPO smoke on the humanoid env (Phase 2 bring-up gate).

Run: .venv/bin/python -B humanoid/tests/test_humanoid_train_smoke.py

Gate (coordinator spec): a short CPU training run -- 2 envs, tiny horizon,
2 PPO updates -- through the UNMODIFIED walk/train/run.py machinery
(VecEnv worker + learner; OBS=52/ACT=12 read off the env class) completes
end-to-end with ZERO solver faults and zero poisoned envs, producing
finite, non-constant rewards.

preflight_steps=0: run.py's preflight drives uniform +-1 actions and, per
the flat.py/duck contract, done envs keep ticking physics with frozen
targets -- for a FALLEN humanoid that means grinding degenerate pile
states at up to 16384 solver iterations per 2 ms tick, which is wall-time
prohibitive for a smoke (measured). VecEnv training rollouts don't have
the issue (workers reset done envs immediately), so the training loop
itself is the smoke. The reward-variance signal preflight would have
checked is asserted here directly from the rollout metrics instead.
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

from walk.train.run import TrainConfig, train  # noqa: E402


class HumanoidTrainSmoke(unittest.TestCase):
    def test_two_updates_two_envs_no_faults(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = TrainConfig(
                env="walk.env.humanoid_flat:FlatFloorHumanoidEnv",
                env_kwargs={},
                workers=1,
                total_envs=2,
                horizon=8,
                updates=2,
                seed=20260901,
                out=str(Path(tmp) / "smoke"),
                checkpoint_every=1000,
                eval_every=0,
                preflight_steps=0,        # see module docstring
                torch_threads=1,
                quiet=True,
            )
            metrics = train(cfg)
            rows = [m for m in metrics if m.get("kind") == "train"]
            self.assertEqual(len(rows), 2, metrics)
            for row in rows:
                self.assertEqual(row["faults"], 0, row)
                self.assertEqual(row["poisoned_envs"], 0, row)
                self.assertIsNotNone(row["reward_mean"], row)
                self.assertTrue(np.isfinite(row["reward_mean"]), row)
                self.assertTrue(np.isfinite(row["reward_std"]), row)
                self.assertGreater(row["batch_transitions"], 0, row)
                self.assertIsNotNone(row["pi_loss"], row)
            # reward carries signal (non-constant across states)
            self.assertGreater(max(r["reward_std"] for r in rows), 1e-7)
            # faults.jsonl stayed empty
            faults = (Path(cfg.out) / "faults.jsonl").read_text().strip()
            self.assertEqual(faults, "", faults)
            # config round-trips (run dir is usable for resume/forensics)
            saved = json.loads((Path(cfg.out) / "config.json").read_text())
            self.assertEqual(saved["env"],
                             "walk.env.humanoid_flat:FlatFloorHumanoidEnv")
            print("smoke:",
                  {k: rows[-1][k] for k in ["update", "reward_mean",
                                            "reward_std", "faults",
                                            "episodes", "steps_per_s"]},
                  file=sys.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
