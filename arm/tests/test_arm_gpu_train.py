"""gpu_train robot-switch gates for the arm: adapter + CPU PPO smoke.

Run: .venv/bin/python -B arm/tests/test_arm_gpu_train.py

- robot_classes("arm", variant) returns OBS/ACT 27/6 with the variant bound
  into CudaArmLane / ArmReachEnv (default variant kr240; bad variant ->
  SystemExit); the duck row is untouched (identity pins);
- --lane-env builds a LanePolicyEnv over CudaArmLane (kernel ABI v8 reach
  device policy path) with OBS/ACT 27/6 and the gate_proxy hook the
  curriculum machinery expects;
- gpu_train.train(robot="arm") runs 8 envs x 2 updates on cpu end to end
  through the UNMODIFIED trainer loop (preflight included), zero faults,
  finite non-constant rewards, and actor_final.pt round-trips into a
  27 -> ... -> 6 feed-forward actor -- on BOTH the python-side path and
  the --lane-env device path.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "arm"))

import torch  # noqa: E402

from walk.env import arm_reach as ar  # noqa: E402
from walk.env import contract  # noqa: E402
from walk.env.arm_cuda_lane import CudaArmLane  # noqa: E402
from walk.env.cuda_lane import CudaDuckLane  # noqa: E402
from walk.env.flat import FlatFloorDuckEnv  # noqa: E402
from walk.train import gpu_train as gt  # noqa: E402
from walk.train import ppo  # noqa: E402


class RobotSwitchTests(unittest.TestCase):
    def test_duck_identity_untouched(self):
        obs, act, lane_cls, env_cls = gt.robot_classes("duck")
        self.assertEqual((obs, act), (contract.OBS, contract.ACT))
        self.assertIs(lane_cls, CudaDuckLane)
        self.assertIs(env_cls, FlatFloorDuckEnv)

    def test_robot_classes_arm(self):
        for variant, expect in ((None, "kr240"), ("kr240", "kr240"), ("lite", "lite")):
            obs, act, lane_cls, env_cls = gt.robot_classes("arm", variant)
            self.assertEqual((obs, act), (27, 6))
            self.assertEqual(lane_cls.func, CudaArmLane)
            self.assertEqual(lane_cls.keywords, {"variant": expect})
            self.assertEqual(env_cls.func, ar.ArmReachEnv)
            self.assertEqual(env_cls.keywords, {"variant": expect})
        with self.assertRaises(SystemExit):
            gt.robot_classes("arm", "ur10")

    def test_lane_env_builds_arm_device_path(self):
        cfg = gt.GpuTrainConfig(robot="arm", variant="lite", envs=2,
                                lane_env=True, out="/tmp/unused")
        env = gt.LanePolicyEnv(cfg)
        try:
            self.assertEqual((env.OBS, env.ACT), (27, 6))
            self.assertIsInstance(env._lane, CudaArmLane)
            self.assertEqual(env._lane.variant, "lite")
            self.assertTrue(env._lane.fast_termination)
            self.assertTrue(hasattr(env._lane, "gate_proxy"))
            self.assertTrue(hasattr(env._lane, "set_gate_termination"))
            obs = env.reset(seed=3)
            self.assertEqual(obs.shape, (2, 27))
            o, r, d, info = env.step(np.zeros((2, 6), np.float32))
            self.assertEqual((o.shape, r.shape, d.shape), ((2, 27), (2,), (2,)))
            self.assertEqual(info["faults"], 0)
        finally:
            env.close()

    def test_make_env_binds_variant(self):
        cfg = gt.GpuTrainConfig(robot="arm", variant="lite", envs=2,
                                out="/tmp/unused")
        env = gt.make_env(cfg)
        try:
            self.assertIsInstance(env, ar.ArmReachEnv)
            self.assertEqual(env.variant, "lite")
            self.assertIsInstance(env._lane, CudaArmLane)
            self.assertEqual(env._lane.variant, "lite")
        finally:
            env.close()


class ArmTrainSmoke(unittest.TestCase):
    def test_two_updates_eight_envs_cpu(self):
        self._smoke(lane_env=False)

    def test_two_updates_eight_envs_cpu_lane_env(self):
        rows = self._smoke(lane_env=True)
        # the device path publishes the gate_proxy_* metrics the chain /
        # curriculum machinery reads (reach mapping: acquired / violations)
        for row in rows:
            self.assertIn("gate_proxy_ep_qualified_l", row)
            self.assertIn("gate_proxy_ep_alt_violations", row)

    def _smoke(self, lane_env: bool):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = gt.GpuTrainConfig(
                robot="arm", variant="kr240", envs=8, horizon=8, updates=2,
                seed=20260902, device="cpu", out=str(Path(tmp) / "smoke"),
                preflight_steps=20, checkpoint_every=1000, torch_threads=1,
                quiet=True, lane_env=lane_env)
            metrics = gt.train(cfg)
            rows = [m for m in metrics if m.get("kind") == "train"]
            self.assertEqual(len(rows), 2, metrics)
            for row in rows:
                self.assertEqual(row["faults"], 0, row)
                self.assertTrue(np.isfinite(row["reward_mean"]), row)
                self.assertTrue(np.isfinite(row["pi_loss"]), row)
                self.assertTrue(np.isfinite(row["v_loss"]), row)
            self.assertGreater(max(r["reward_std"] for r in rows), 1e-7)
            raw = torch.load(Path(cfg.out) / "actor_final.pt", map_location="cpu",
                             weights_only=False)
            arch, sd = ppo.unpack_actor_file(raw)
            self.assertEqual(arch, "ff")
            actor = ppo.Actor(27, 6)
            actor.load_state_dict(sd)
            out = actor.deterministic(torch.zeros(3, 27))
            self.assertEqual(tuple(out.shape), (3, 6))
            self.assertEqual((Path(cfg.out) / "faults.jsonl").read_text().strip(), "")
            print(f"arm smoke lane_env={lane_env}:",
                  {k: rows[-1][k] for k in ["update", "reward_mean",
                                            "reward_std", "faults",
                                            "steps_per_s"]}, file=sys.stderr)
            return rows


if __name__ == "__main__":
    unittest.main(verbosity=2)
