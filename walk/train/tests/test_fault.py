"""SolverFault handling: poisoned rollouts, fault logging, training continues."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from walk.env.contract import SolverFault, StubEnv
from walk.train.ppo import PPOConfig, make_nets, ppo_update
from walk.train.run import collect_rollout, make_batch, train
from walk.train.tests.test_train import make_cfg, train_lines
from walk.train.vec import VecEnv

MARKER_CHANNEL = 54  # foot-contact slot, unused by StubEnv


class FaultyStubEnv(StubEnv):
    """StubEnv that raises SolverFault once, on a chosen worker, mid-rollout.

    worker_id is injected by VecEnv via the "$WORKER" kwarg substitution.
    Observations carry worker_id + 1 in MARKER_CHANNEL so tests can prove
    which shard each training transition came from.
    """

    def __init__(self, environments=8, seed=0, worker_id=0, fault_on_worker=1,
                 fault_at_step=10, **kw):
        self._wid = int(worker_id)
        self._fault_on = int(fault_on_worker)
        self._fault_at = int(fault_at_step)
        self._step_calls = 0
        self._faulted = False
        super().__init__(environments=environments, seed=seed, **kw)

    def _obs(self):
        out = super()._obs()
        out[:, MARKER_CHANNEL] = float(self._wid + 1)
        return out

    def step(self, action):
        self._step_calls += 1
        if (not self._faulted) and self._wid == self._fault_on and self._step_calls == self._fault_at:
            self._faulted = True
            path = os.path.join(tempfile.gettempdir(), f"dgw_fault_{os.getpid()}_{self._wid}.json")
            with open(path, "w") as f:
                json.dump({"env_index": 2, "step_calls": self._step_calls, "worker": self._wid}, f)
            raise SolverFault(2, path, "synthetic fault for tests")
        return super().step(action)


SPEC = "walk.train.tests.test_fault:FaultyStubEnv"
KW = {"worker_id": "$WORKER", "fault_on_worker": 1, "fault_at_step": 10}


class TestFaultPoisoning(unittest.TestCase):
    def test_poisoned_shard_absent_from_batch(self):
        vec = VecEnv(SPEC, KW, workers=2, total_envs=16, seed=7)
        try:
            torch.manual_seed(0)
            actor, critic = make_nets(vec.obs_dim, vec.act_dim, PPOConfig())
            gen = torch.Generator()
            gen.manual_seed(0)
            obs = vec.reset(7)
            horizon = 32

            ro = collect_rollout(vec, actor, critic, obs, horizon, gen)
            # worker 1's shard (envs 8..15) is poisoned, worker 0's is not
            self.assertTrue(ro.poisoned[8:].all())
            self.assertFalse(ro.poisoned[:8].any())
            self.assertEqual(len(ro.faults), 1)
            self.assertEqual(ro.faults[0].worker, 1)
            self.assertEqual(ro.faults[0].env_index, 8 + 2)
            self.assertTrue(Path(ro.faults[0].saved_problem_path).is_file())

            batch = make_batch(ro, critic, 0.99, 0.95)
            self.assertEqual(batch["obs"].shape[0], horizon * 8)
            markers = batch["obs"][:, MARKER_CHANNEL]
            self.assertTrue(bool((markers == 1.0).all()),
                            "batch contains transitions from the faulted shard")

            # training continues: update on the partial batch, then a clean window
            opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
            stats = ppo_update(actor, critic, opt, batch, PPOConfig(), gen)
            self.assertTrue(np.isfinite(stats["pi_loss"]))

            ro2 = collect_rollout(vec, actor, critic, ro.next_obs_np, horizon, gen)
            self.assertFalse(ro2.poisoned.any())
            self.assertEqual(len(ro2.faults), 0)
            batch2 = make_batch(ro2, critic, 0.99, 0.95)
            self.assertEqual(batch2["obs"].shape[0], horizon * 16)
            self.assertEqual(sorted(set(batch2["obs"][:, MARKER_CHANNEL].tolist())), [1.0, 2.0])
        finally:
            vec.close()

    def test_full_training_loop_logs_fault_and_continues(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = make_cfg(
                td, env=SPEC, env_kwargs=dict(KW), updates=2, horizon=32,
                preflight_steps=0,  # eval instance would be worker_id=0, fine either way
            )
            metrics = train_lines(train(cfg))
            m1, m2 = metrics
            self.assertEqual(m1["faults"], 1)
            self.assertEqual(m1["faults_total"], 1)
            self.assertEqual(m1["poisoned_envs"], 8)
            self.assertEqual(m1["batch_transitions"], 32 * 8)
            self.assertIsNotNone(m1["pi_loss"])
            self.assertEqual(m2["faults"], 0)
            self.assertEqual(m2["poisoned_envs"], 0)
            self.assertEqual(m2["batch_transitions"], 32 * 16)

            fault_lines = [json.loads(l) for l in (Path(td) / "faults.jsonl").read_text().splitlines()]
            self.assertEqual(len(fault_lines), 1)
            self.assertEqual(fault_lines[0]["update"], 1)
            self.assertEqual(fault_lines[0]["worker"], 1)
            self.assertTrue(fault_lines[0]["saved_problem_path"])
            self.assertTrue(fault_lines[0]["copied_to"] and Path(fault_lines[0]["copied_to"]).is_file())


if __name__ == "__main__":
    unittest.main()
