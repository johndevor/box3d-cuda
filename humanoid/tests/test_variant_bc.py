"""BC seed gates per humanoid FAMILY variant (twin of test_bc_pretrain.py).

Run: .venv/bin/python -B humanoid/tests/test_variant_bc.py

Per variant (humanoid/h1_family.py, non-base members):
- the dataset rolls on the VARIANT's fp32 lane with the VARIANT's reference
  table (meta.variant, real-lane obs, labels in [-1, 1]);
- BC converges (final MSE < initial/50);
- the committed humanoid/variants/<name>/bc_init.pt matches a deterministic
  regeneration with the deliverable configuration (drift lock), and the
  committed bc_init_ckpt.pt carries config.variant so gpu_train --resume
  refuses a cross-variant resume;
- closed-loop replay of the clone on the variant lane (random episode
  phase0, the training condition) survives its bootstrap window and takes
  real (debounced) swings, the honest analyzer.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "humanoid"))

import torch  # noqa: E402

import h1_family as fam  # noqa: E402
from walk.env import humanoid_flat as hf  # noqa: E402
from walk.env.humanoid_cuda_lane import CudaHumanoidLane  # noqa: E402
from walk.train import bc_pretrain  # noqa: E402
from walk.train import gpu_train as gt  # noqa: E402
from walk.train.ppo import unpack_actor_file  # noqa: E402

VARIANTS = tuple(v for v in fam.variant_names() if v != "h1")
FULL = dict(seeds="11,22,33,44,55,66,77,88", envs_per_config=4, steps=60,
            epochs=300, seed=0, log_std=-1.0)


def _replay(actor, variant: str, cmd: float, seed: int):
    import diagnose_swings as dg
    from walk.eval.capture import capture_episodes
    env = hf.FlatFloorHumanoidEnv(
        environments=1, seed=seed, variant=variant,
        lane_factory=lambda E, off: CudaHumanoidLane(
            E, joint_offsets=off, variant=variant))
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
    return (len(trace["ticks"]["time_s"]) * 0.002, len(swings),
            sum(s["qualified"] for s in swings))


class VariantBcGates(unittest.TestCase):
    _cache: dict = {}

    @classmethod
    def _trained(cls, v):
        if v not in cls._cache:
            args = type("A", (), dict(FULL, variant=v))()
            dataset = bc_pretrain.humanoid_dataset(args)
            actor, history = bc_pretrain.train_bc(
                "humanoid", dataset, epochs=FULL["epochs"], seed=FULL["seed"],
                log_std=FULL["log_std"], quiet=True, variant=v)
            cls._cache[v] = (dataset, actor.eval(), history)
        return cls._cache[v]

    def _dataset_and_convergence(self, v):
        dataset, _, history = self._trained(v)
        obs, act, meta = dataset["obs"], dataset["act"], dataset["meta"]
        self.assertEqual(meta["variant"], v)
        self.assertGreaterEqual(meta["pairs"], 2000)
        self.assertEqual(obs.shape, (meta["pairs"], hf.OBS))
        self.assertEqual(act.shape, (meta["pairs"], hf.ACT))
        self.assertTrue(np.isfinite(obs).all() and np.isfinite(act).all())
        self.assertLessEqual(float(np.abs(act).max()), 1.0)
        T = 3 * hf.ACT
        self.assertEqual(set(np.unique(obs[:, T + 12:T + 14])), {0.0, 1.0})
        self.assertGreater(history[0], 0.02)
        self.assertLess(history[-1], history[0] / 50.0)

    def _committed_artifacts(self, v):
        _, actor, _ = self._trained(v)
        path, ck = fam.bc_init_path(v), fam.bc_ckpt_path(v)
        self.assertTrue(path.is_file(), f"{path} missing: regenerate with "
                        f"walk.train.bc_pretrain --robot humanoid --variant {v}")
        arch, sd = unpack_actor_file(
            torch.load(path, map_location="cpu", weights_only=False))
        self.assertEqual(arch, "ff")
        ours = actor.state_dict()
        for k in ours:
            np.testing.assert_allclose(
                sd[k].numpy(), ours[k].numpy(), atol=1e-6,
                err_msg=f"{v} {k}: bc_init.pt drifted; regenerate with "
                        ".venv/bin/python -B -m walk.train.bc_pretrain --robot "
                        f"humanoid --variant {v} --out {path} --checkpoint-out "
                        f"{ck} --epochs 300 --seeds {FULL['seeds']}")
        raw = torch.load(ck, map_location="cpu", weights_only=False)
        self.assertEqual(raw["config"]["variant"], v)
        self.assertEqual(raw["config"]["robot"], "humanoid")
        # cross-variant resume is refused by gpu_train
        with tempfile.TemporaryDirectory() as tmp:
            other = [w for w in VARIANTS if w != v][0]
            cfg = gt.GpuTrainConfig(robot="humanoid", variant=other, envs=1,
                                    lane_env=True, updates=1, horizon=2,
                                    out=str(Path(tmp) / "x"), resume=str(ck),
                                    preflight_steps=0, quiet=True)
            with self.assertRaises(SystemExit):
                gt.train(cfg)

    def _closed_loop(self, v):
        _, actor, _ = self._trained(v)
        alive, total_swings = [], 0
        for cmd in hf.COMMANDS_MPS:
            for seed in (4242, 7):
                alive_s, swings, qualified = _replay(actor, v, cmd, seed)
                alive.append(alive_s)
                total_swings += swings
                print(f"[{v}] bc replay cmd {cmd:.2f} seed {seed}: {alive_s:.2f}s "
                      f"debounced swings {swings} qualified {qualified}",
                      file=sys.stderr)
                self.assertGreaterEqual(alive_s, 0.55, (v, cmd, seed))
        self.assertGreater(float(np.mean(alive)), 0.65, v)
        self.assertGreaterEqual(total_swings, 6, v)


for _v in VARIANTS:
    def _mk(name, v=_v):
        return lambda self: getattr(self, name)(v)
    for _n in ("_dataset_and_convergence", "_committed_artifacts", "_closed_loop"):
        setattr(VariantBcGates, f"test{_n}_{_v}", _mk(_n))


if __name__ == "__main__":
    unittest.main(verbosity=2)
