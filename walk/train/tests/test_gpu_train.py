"""Local (cpu + serial lane) tests for the single-process GPU trainer.

Runs walk.train.gpu_train end to end against the SERIAL dwc1 dylib built by
walk.env.cuda_lane.build_library() — the same wrapper the CUDA .so uses on
the GPU host — so everything except the device is exercised locally.
"""
from __future__ import annotations

import json
import math
import tempfile
import time
import unittest
from pathlib import Path

import torch

from walk.env.cuda_lane import build_library
from walk.train import gpu_train

ENVS = 8
HORIZON = 16
SEED = 123


def argv(out: str, lib: str, **over) -> list[str]:
    base = {
        "envs": ENVS, "horizon": HORIZON, "updates": 3, "seed": SEED,
        "device": "cpu", "library": lib, "out": out,
        "preflight-steps": 30, "checkpoint-every": 2,
    }
    base.update(over)
    args = []
    for k, v in base.items():
        if v is None:
            continue
        args += [f"--{k}", str(v)]
    return args + ["--quiet"]


def read_metrics(out: Path) -> list[dict]:
    path = out / "metrics.jsonl"
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


class TestGpuTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = str(build_library())

    def test_three_updates_cpu_serial(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            metrics = gpu_train.main(argv(td, self.lib))
            train_lines = [m for m in metrics if "skipped" not in m]
            self.assertEqual(len(train_lines), 3)
            # metrics.jsonl written and parseable, rewards finite
            lines = read_metrics(out)
            self.assertEqual(len([m for m in lines if m["kind"] == "train"]), 3)
            for m in train_lines:
                self.assertTrue(math.isfinite(m["reward_mean"]),
                                f"non-finite reward at update {m['update']}: {m}")
                self.assertTrue(math.isfinite(m["pi_loss"]))
                self.assertTrue(math.isfinite(m["v_loss"]))
                self.assertGreater(m["steps_per_s"], 0.0)
                self.assertEqual(m["faults"], 0)
            self.assertEqual(train_lines[-1]["env_steps"], 3 * ENVS * HORIZON)
            # checkpoints: every 2 updates + latest at exit
            self.assertTrue((out / "ckpt_000002.pt").is_file())
            self.assertTrue((out / "latest.pt").is_file())
            ck = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(ck["update"], 3)
            # actor_final.pt: plain cpu state_dict loadable into a fresh Actor
            self.assertTrue((out / "actor_final.pt").is_file())
            sd = torch.load(out / "actor_final.pt", map_location="cpu", weights_only=True)
            from walk.train.ppo import Actor
            actor = Actor(58, 14)
            actor.load_state_dict(sd)  # raises on any mismatch
            for v in sd.values():
                self.assertEqual(v.device.type, "cpu")

    def test_resume_runs_two_more_updates(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            gpu_train.main(argv(td, self.lib, updates=3))
            metrics = gpu_train.main(
                argv(td, self.lib, updates=5, resume=str(out / "latest.pt")))
            self.assertEqual([m["update"] for m in metrics], [4, 5])
            lines = [m for m in read_metrics(out) if m["kind"] == "train"]
            self.assertEqual([m["update"] for m in lines], [1, 2, 3, 4, 5])
            # env_steps carried across the resume
            self.assertEqual(lines[-1]["env_steps"], 5 * ENVS * HORIZON)
            ck = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(ck["update"], 5)

    def test_max_wall_s_exits_cleanly_with_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            t0 = time.perf_counter()
            metrics = gpu_train.main(
                argv(td, self.lib, updates=100000, **{"max-wall-s": 1}))
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 60.0, "wall-clock stop did not engage")
            self.assertLess(len(metrics), 100000)
            self.assertTrue((out / "latest.pt").is_file())
            self.assertTrue((out / "actor_final.pt").is_file())
            ck = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(ck["update"], len([m for m in metrics
                                                if "skipped" not in m]))


if __name__ == "__main__":
    unittest.main()
