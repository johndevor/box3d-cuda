"""Learning + determinism tests for the PPO trainer on StubEnv."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from walk.train.ppo import PPOConfig
from walk.train.run import TrainConfig, train

LOSS_KEYS = ("pi_loss", "v_loss", "entropy", "approx_kl", "reward_mean")


def make_cfg(out: str, **over) -> TrainConfig:
    cfg = TrainConfig(
        env="walk.env.contract:StubEnv",
        env_kwargs={},
        workers=2,
        total_envs=16,
        horizon=64,
        updates=5,
        seed=123,
        out=out,
        checkpoint_every=1000,
        eval_every=0,
        preflight_steps=50,
        torch_threads=2,
        quiet=True,
        ppo=PPOConfig(),
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def train_lines(metrics):
    return [m for m in metrics if m["kind"] == "train"]


class TestLearning(unittest.TestCase):
    def test_reward_increases_over_30_updates(self):
        with tempfile.TemporaryDirectory() as td:
            metrics = train_lines(train(make_cfg(td, updates=30)))
        self.assertEqual(len(metrics), 30)
        first = sum(m["reward_mean"] for m in metrics[:3]) / 3
        last = sum(m["reward_mean"] for m in metrics[-3:]) / 3
        self.assertGreater(
            last, first + 1e-4,
            f"PPO failed to improve StubEnv reward: first3={first:.5f} last3={last:.5f}",
        )


class TestDeterminism(unittest.TestCase):
    def test_same_seed_identical_first_5_updates(self):
        with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
            m1 = train_lines(train(make_cfg(ta, updates=5)))
            m2 = train_lines(train(make_cfg(tb, updates=5)))
        self.assertEqual(len(m1), 5)
        for a, b in zip(m1, m2):
            for k in LOSS_KEYS:
                self.assertEqual(a[k], b[k], f"update {a['update']} field {k}: {a[k]} != {b[k]}")

    def test_preflight_aborts_on_flat_reward(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = make_cfg(
                td,
                env="walk.train.tests.test_train:FlatRewardEnv",
                workers=1,
                total_envs=4,
                updates=1,
            )
            with self.assertRaises(SystemExit) as ctx:
                train(cfg)
            self.assertIn("PREFLIGHT FAILED", str(ctx.exception))


from walk.env.contract import StubEnv  # noqa: E402
import numpy as np  # noqa: E402


class FlatRewardEnv(StubEnv):
    """Reward is a constant: the preflight must refuse to train on this."""

    def step(self, action):
        obs, rew, done, info = super().step(action)
        return obs, np.full_like(rew, -1.0), done, info


if __name__ == "__main__":
    unittest.main()
