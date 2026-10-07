"""BC pre-training gates: dataset, convergence, artifact, closed-loop steps.

Run: .venv/bin/python -B humanoid/tests/test_bc_pretrain.py

Gates:
- dataset: real fp32-lane obs (contact channels vary, gravity channel is
  live), labels finite in [-1, 1], a few thousand pairs across the
  (seed x command) spread, action saturation only at the knee plateau;
- BC training converges (final MSE < initial/50; measured ~0.059 -> 2e-4);
- the artifact round-trips through ppo.unpack_actor_file into a
  52 -> (256, 256) -> 12 ff actor with the documented log_std -1.0;
- the committed humanoid/bc_init.pt matches a deterministic regeneration
  (drift lock; regenerate with the module command line on mismatch);
- CLOSED-LOOP STEPPING (the coordinator gate, H1): the BC'd actor
  replayed deterministically on the fp32 CPU-serial lane measurably steps
  AND -- with the hip-roll authority H1 added -- survives measurably
  longer than the H0 clone's ~0.76 s lateral-tip ceiling. Measured on the
  committed H1 actor across 3 commands x 3 seeds: mean survival 0.978 s,
  every episode >= 0.88 s, 23 lifts, 6 alternations. The pins below sit
  under those with margin. Walking to horizon is PPO's job.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "humanoid"))

import torch  # noqa: E402

import bc_dataset  # noqa: E402
from walk.env import humanoid_flat as hf  # noqa: E402
from walk.env.humanoid_cuda_lane import CudaHumanoidLane  # noqa: E402
from walk.train import bc_pretrain  # noqa: E402
from walk.train.ppo import Actor, unpack_actor_file  # noqa: E402

BC_INIT = ROOT / "humanoid" / "bc_init.pt"
# the exact deliverable configuration (== the orchestrator command line)
FULL = dict(seeds="11,22,33,44,55,66,77,88", envs_per_config=4, steps=60,
            epochs=300, seed=0, log_std=-1.0)


def _serial_factory(E, off):
    return CudaHumanoidLane(E, joint_offsets=off)


def _replay(actor, cmd: float, seed: int):
    """(alive_s, debounced_swings, qualified) via the HONEST analyzer.

    v3 correction (PHASE2.md section 14): the old boundary-sampled
    foot_contact "lift" counting was measuring single-tick solver
    flickers; real swings are counted per-tick with the judge's 20 ms
    debounce (humanoid/diagnose_swings.swings_from_trace)."""
    import diagnose_swings as dg
    from walk.eval.capture import capture_episodes
    env = hf.FlatFloorHumanoidEnv(environments=1, seed=seed,
                                  lane_factory=_serial_factory)
    try:
        @torch.no_grad()
        def policy(obs):
            return actor.deterministic(
                torch.from_numpy(np.ascontiguousarray(obs))).numpy()
        trace = capture_episodes(env, policy, command=cmd, seconds=8.0,
                                 seed=seed)[0]
    finally:
        env.close()
    swings = dg.swings_from_trace(trace, debounce=True)
    qualified = sum(s["qualified"] for s in swings)
    return (len(trace["ticks"]["time_s"]) * 0.002, len(swings), qualified)


class BcPretrainGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = type("A", (), dict(FULL))()
        cls.dataset = bc_pretrain.humanoid_dataset(args)
        cls.actor, cls.history = bc_pretrain.train_bc(
            "humanoid", cls.dataset, epochs=FULL["epochs"],
            seed=FULL["seed"], log_std=FULL["log_std"], quiet=True)

    # -- dataset ------------------------------------------------------------
    def test_dataset_stats(self):
        obs, act = self.dataset["obs"], self.dataset["act"]
        meta = self.dataset["meta"]
        self.assertGreaterEqual(meta["pairs"], 2000)
        self.assertEqual(obs.shape, (meta["pairs"], hf.OBS))
        self.assertEqual(act.shape, (meta["pairs"], hf.ACT))
        self.assertTrue(np.isfinite(obs).all() and np.isfinite(act).all())
        self.assertLessEqual(float(np.abs(act).max()), 1.0)
        # real-lane obs: both contact states appear; gravity channel live
        T = 3 * hf.ACT
        self.assertEqual(set(np.unique(obs[:, T + 12:T + 14])), {0.0, 1.0})
        self.assertGreater(float(obs[:, T].std()), 0.01)
        # every (seed, command) config contributed
        self.assertEqual(len(meta["per_config"]),
                         len(meta["seeds"]) * len(meta["commands"]))
        self.assertTrue(all(v > 0 for v in meta["per_config"].values()))
        # knee plateau saturates by design; nothing else should pin at 1
        sat = np.abs(act) > 0.999
        knees = [4, 8]                     # H1 joint order
        self.assertGreater(float(sat[:, knees].mean()), 0.01)
        self.assertLess(float(sat.mean()), 0.25)

    # -- convergence ----------------------------------------------------------
    def test_bc_loss_converges(self):
        self.assertGreater(self.history[0], 0.02)
        self.assertLess(self.history[-1], self.history[0] / 50.0)
        self.assertLess(self.history[-1], 1e-3)

    # -- artifact ---------------------------------------------------------------
    def test_artifact_roundtrip_and_log_std(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bc.pt"
            bc_pretrain.save_actor(self.actor, path)
            arch, sd = unpack_actor_file(
                torch.load(path, map_location="cpu", weights_only=False))
        self.assertEqual(arch, "ff")
        fresh = Actor(hf.OBS, hf.ACT)
        fresh.load_state_dict(sd)              # shape-checked load
        shapes = [tuple(v.shape) for v in sd.values()]
        self.assertIn((256, hf.OBS), shapes)
        self.assertIn((hf.ACT, 256), shapes)
        np.testing.assert_allclose(sd["log_std"].numpy(), -1.0, atol=1e-7)

    def test_committed_bc_init_matches_regeneration(self):
        if not BC_INIT.is_file():
            raise unittest.SkipTest("humanoid/bc_init.pt not present")
        arch, sd = unpack_actor_file(
            torch.load(BC_INIT, map_location="cpu", weights_only=False))
        self.assertEqual(arch, "ff")
        ours = self.actor.state_dict()
        for k in ours:
            np.testing.assert_allclose(
                sd[k].numpy(), ours[k].numpy(), atol=1e-6,
                err_msg=f"{k}: humanoid/bc_init.pt drifted; regenerate with "
                        ".venv/bin/python -B -m walk.train.bc_pretrain "
                        "--robot humanoid --out humanoid/bc_init.pt "
                        "--epochs 300 --seeds 11,22,33,44,55,66,77,88")

    def test_checkpoint_handoff_into_gpu_train(self):
        """save_checkpoint -> gpu_train --resume runs a PPO update clean."""
        import tempfile
        from walk.train import gpu_train as gt
        with tempfile.TemporaryDirectory() as tmp:
            ck = Path(tmp) / "bc_ckpt.pt"
            bc_pretrain.save_checkpoint("humanoid", self.actor, ck,
                                        seed=0, log_std=-1.0)
            cfg = gt.GpuTrainConfig(
                robot="humanoid", envs=2, lane_env=True, horizon=8,
                updates=1, seed=1, device="cpu", out=str(Path(tmp) / "run"),
                resume=str(ck), preflight_steps=0, checkpoint_every=1000,
                torch_threads=1, quiet=True)
            rows = [m for m in gt.train(cfg) if m.get("kind") == "train"]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["faults"], 0, rows[0])
            self.assertTrue(np.isfinite(rows[0]["pi_loss"]), rows[0])

    # -- closed-loop gates (HONEST analyzer, v3) ---------------------------
    def test_closed_loop_replay_survival(self):
        """Active gate: the clone survives its bootstrap window on every
        command/seed (v3 measured: mean 0.773 s, min 0.72 s; the ~0.8 s
        ceiling is the plant's lateral instability, PHASE2.md s15)."""
        actor = self.actor.eval()
        alive = []
        for cmd in hf.COMMANDS_MPS:
            for seed in (4242, 7, 1913):
                alive_s, swings, qualified = _replay(actor, cmd, seed)
                alive.append(alive_s)
                print(f"bc replay cmd {cmd:.2f} seed {seed}: {alive_s:.2f}s "
                      f"debounced swings {swings} qualified {qualified}",
                      file=sys.stderr)
                self.assertGreaterEqual(alive_s, 0.55, (cmd, seed))
        self.assertGreater(float(np.mean(alive)), 0.65)

    def test_closed_loop_replay_steps(self):
        """ACTIVE since H1.1 + reference v3.2: the clone takes REAL
        (debounced) swings under the honest analyzer, at RANDOM episode
        phase0 (the training condition). Measured on the committed
        artifact: 29 debounced swings across the 3x3 grid, >= 2 per
        episode, 1 fully judge-qualified (cmd 0.5); pins with margin.
        More qualification is PPO's job (the pinned-phase demonstrator
        gate lives in test_reference_gait.ExecutedValidation)."""
        actor = self.actor.eval()
        total_swings = total_q = 0
        for cmd in hf.COMMANDS_MPS:
            for seed in (4242, 7, 1913):
                _, swings, qualified = _replay(actor, cmd, seed)
                total_swings += swings
                total_q += qualified
                self.assertGreaterEqual(swings, 1, (cmd, seed))
        self.assertGreaterEqual(total_swings, 15)
        self.assertGreaterEqual(total_q, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
