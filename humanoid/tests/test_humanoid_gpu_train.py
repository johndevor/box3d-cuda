"""gpu_train robot-switch gates: duck identity pins, humanoid adapter, smoke.

Run: .venv/bin/python -B humanoid/tests/test_humanoid_gpu_train.py

Gates:
- robot_classes("duck") returns exactly the module-level contract dims and
  the original classes (the structural half of the byte-identity guarantee;
  the behavioral half -- identical trajectory + actor fingerprints on a
  2-env lane-env smoke -- was proven at change time and holds because the
  duck branch executes the same constructor calls and RNG streams);
- LanePolicyEnv(robot=humanoid) exposes OBS/ACT 52/12 over CudaHumanoidLane;
- FlatFloorHumanoidEnv over the fp32 humanoid lane and the in-kernel policy
  path (CudaHumanoidLane.step_policy) agree obs/reward/done side by side
  (the robot-generic kernel's contract; duck gate tolerances 1e-4/1e-3);
- gpu_train.train(robot="humanoid", lane_env=True) runs 2 envs x 2 updates
  zero-fault with finite losses, and actor_final.pt round-trips through
  ppo.unpack_actor_file into a 52->...->12 feed-forward actor;
- --randomization (shared dwc1 DR contract since ABI v7) is accepted on the
  device policy path (--lane-env) and rejected without it; --accept-every is
  accepted for --robot humanoid (the batched frozen-judge probe, gated in
  humanoid/tests/test_humanoid_accept_probe.py) and passes --validate-only.
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

import torch  # noqa: E402

from walk.env import contract  # noqa: E402
from walk.env import humanoid_flat as hf  # noqa: E402
from walk.env.cuda_lane import CudaDuckLane  # noqa: E402
from walk.env.flat import FlatFloorDuckEnv  # noqa: E402
from walk.env.humanoid_cuda_lane import CudaHumanoidLane  # noqa: E402
from walk.train import gpu_train as gt  # noqa: E402
from walk.train import ppo  # noqa: E402


class RobotSwitchTests(unittest.TestCase):
    def test_robot_classes_duck_identity(self):
        obs, act, lane_cls, env_cls = gt.robot_classes("duck")
        self.assertIs(obs, gt.OBS)
        self.assertIs(act, gt.ACT)
        self.assertEqual((obs, act), (contract.OBS, contract.ACT))
        self.assertIs(lane_cls, CudaDuckLane)
        self.assertIs(env_cls, FlatFloorDuckEnv)
        # legacy class-level defaults stay the duck's
        self.assertEqual((gt.LanePolicyEnv.OBS, gt.LanePolicyEnv.ACT),
                         (58, 14))

    def test_robot_classes_humanoid(self):
        obs, act, lane_cls, env_cls = gt.robot_classes("humanoid")
        self.assertEqual((obs, act), (58, 14))  # H1
        self.assertIs(lane_cls, CudaHumanoidLane)
        self.assertIs(env_cls, hf.FlatFloorHumanoidEnv)
        with self.assertRaises(SystemExit):
            gt.robot_classes("emu")

    def test_lane_policy_env_humanoid_dims(self):
        cfg = gt.GpuTrainConfig(robot="humanoid", envs=1, lane_env=True,
                                perturbation=0.0)
        env = gt.LanePolicyEnv(cfg)
        try:
            self.assertEqual((env.OBS, env.ACT), (58, 14))
            obs = env.reset(seed=7)
            self.assertEqual(obs.shape, (1, 58))
            obs, rew, done, info = env.step(np.zeros((1, 14), np.float32))
            self.assertEqual(obs.shape, (1, 58))
            self.assertEqual(info["faults"], 0)
            self.assertFalse(done.any())
        finally:
            env.close()

    def test_randomization_without_lane_env_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = gt.GpuTrainConfig(robot="humanoid", envs=1, updates=1,
                                    horizon=2, out=str(Path(tmp) / "x"),
                                    preflight_steps=0,
                                    randomization={"r_mass": 0.1},
                                    lane_env=False)
            with self.assertRaises(SystemExit):
                gt.train(cfg)

    def test_accept_every_validates_for_humanoid(self):
        """--accept-every is no longer duck-only: the launcher preflight
        (--validate-only) must accept the humanoid specs' flag set."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = gt.GpuTrainConfig(robot="humanoid", variant="h1_stocky",
                                    envs=1, updates=1, horizon=2,
                                    out=str(Path(tmp) / "x"), lane_env=True,
                                    curriculum="humanoid-walk",
                                    rsi_fraction=0.5, accept_every=8,
                                    preflight_steps=0, validate_only=True)
            self.assertEqual(gt.train(cfg), [])

    def test_randomized_humanoid_lane_env_trains(self):
        """--randomization (incl. the v7 gravity scale) on --robot humanoid
        --lane-env: 2 envs x 2 updates run clean with finite losses."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = gt.GpuTrainConfig(
                robot="humanoid", envs=2, lane_env=True, horizon=8,
                updates=2, seed=3, device="cpu", out=str(Path(tmp) / "dr"),
                randomization={"r_mass": 0.15, "r_friction": 0.3,
                               "r_kp": 0.15, "r_damping": 0.3,
                               "max_latency_steps": 2, "r_gravity": 0.5095},
                preflight_steps=0, checkpoint_every=1000, torch_threads=1,
                quiet=True)
            rows = [m for m in gt.train(cfg) if m.get("kind") == "train"]
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["faults"], 0, row)
                self.assertTrue(np.isfinite(row["pi_loss"]), row)
            saved = json.loads((Path(cfg.out) / "config.json").read_text())
            self.assertEqual(saved["randomization"]["r_gravity"], 0.5095)


class EnvVsKernelPolicyParity(unittest.TestCase):
    def test_env_and_lane_policy_agree(self):
        """FlatFloorHumanoidEnv over the fp32 lane vs dwc1_step_policy,
        same seed stream, same actions: duck-suite gate tolerances."""
        E, S, steps = 2, 31, 40
        env = hf.FlatFloorHumanoidEnv(
            environments=E, seed=S,
            lane_factory=lambda n, off: CudaHumanoidLane(n, joint_offsets=off))
        lane = CudaHumanoidLane(E)
        try:
            # env.__init__ already consumed episode 1 of the counter stream;
            # read its post-reset obs without another reset, and put the
            # lane on the same episode with ONE reset_policy(seed).
            obs_env = env.set_command(env.command)
            obs_lane = lane.reset_policy(seed=S)
            np.testing.assert_allclose(obs_lane, obs_env, atol=1e-6,
                                       err_msg="reset obs")
            # Gentle actions: the fp32 solve (5e-6 tolerance, 4096 sweeps)
            # still has less stall headroom than the f64 oracle at 16384 --
            # sigma 0.3 flail can fault it where f64 survives (both sides
            # here run the SAME fp32 physics, but the python env raises
            # SolverFault while the in-kernel path freezes+finishes, so a
            # fault ends the comparison rather than proving divergence).
            rng = np.random.default_rng(99)
            worst_obs = worst_rew = 0.0
            for t in range(steps):
                a = np.clip(rng.normal(0.0, 0.1, (E, hf.ACT)),
                            -0.2, 0.2).astype(np.float32)
                oe, re_, de, _ = env.step(a)
                ol, rl, dl, diag = lane.step_policy(a)
                self.assertTrue((diag["status"] == 0).all(), (t, diag))
                np.testing.assert_array_equal(dl, de, err_msg=f"done t={t}")
                worst_obs = max(worst_obs, float(np.abs(ol - oe).max()))
                worst_rew = max(worst_rew, float(np.abs(rl - re_).max()))
                if de.all():
                    break
            self.assertLess(worst_obs, 1e-4, "obs parity gate")
            self.assertLess(worst_rew, 1e-3, "reward parity gate")
            print(f"env-vs-kernel policy parity: obs<={worst_obs:.2e} "
                  f"reward<={worst_rew:.2e} over {t + 1} steps",
                  file=sys.stderr)
        finally:
            env.close()
            lane.close()


class HumanoidGpuTrainSmoke(unittest.TestCase):
    def test_lane_env_smoke_and_actor_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = gt.GpuTrainConfig(
                robot="humanoid", envs=2, lane_env=True, horizon=8,
                updates=2, seed=20260901, device="cpu",
                out=str(Path(tmp) / "smoke"), preflight_steps=0,
                checkpoint_every=1000, torch_threads=1, quiet=True)
            metrics = gt.train(cfg)
            rows = [m for m in metrics if m.get("kind") == "train"]
            self.assertEqual(len(rows), 2, metrics)
            for row in rows:
                self.assertEqual(row["faults"], 0, row)
                for k in ("reward_mean", "pi_loss", "v_loss", "approx_kl"):
                    self.assertTrue(np.isfinite(row[k]), (k, row))
            # actor arch 52 -> (256, 256) -> 12, loadable via unpack_actor_file
            arch, sd = ppo.unpack_actor_file(torch.load(
                Path(cfg.out) / "actor_final.pt", map_location="cpu",
                weights_only=False))
            self.assertEqual(arch, "ff")
            actor, _ = ppo.make_nets(58, 14, ppo.PPOConfig())
            actor.load_state_dict(sd)          # raises on any shape mismatch
            shapes = [tuple(v.shape) for v in sd.values()]
            self.assertIn((256, 58), shapes)   # input layer consumes 58 (H1)
            self.assertIn((14, 256), shapes)   # mu head emits 14 (H1)
            with torch.no_grad():
                mu = actor.deterministic(torch.zeros(1, 58))
            self.assertEqual(tuple(mu.shape), (1, 14))
            saved = json.loads((Path(cfg.out) / "config.json").read_text())
            self.assertEqual(saved["robot"], "humanoid")
            print("gpu_train humanoid smoke:",
                  {k: rows[-1][k] for k in ["update", "reward_mean",
                                            "faults"]}, file=sys.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
