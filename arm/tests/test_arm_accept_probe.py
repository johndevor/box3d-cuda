"""--accept-every for --robot arm: probe == CPU harness, no training
perturbation, acceptance artifacts + self-stop.

Run: .venv/bin/python -B arm/tests/test_arm_accept_probe.py

Gates:
- EQUALITY (the frozen judge's verdict): walk.eval.arm_acceptance
  .run_batched_probe (12 cells as ONE E=12 batch on CudaArmLane, the
  in-run probe's path) and run_acceptance (12 sequential E=1 episodes)
  agree cell for cell -- passed, failed criteria, acquisition times and the
  full criteria detail -- for the scripted IK baseline (kr240), and the
  per-tick traces are BITWISE equal for every cell;
- gpu_train.run_robot_probe on a trained GPU arm actor (runs/gpu, the
  spec's acceptance job judged it 0/12) equals the harness cell for cell;
- RESTORATION PROOF: an arm --lane-env --curriculum arm-reach run probing
  every 2 updates writes the same train metrics stream as the run without
  probes (pinned seeds), with 12-cell accept rows;
- a passing probe writes <out>/accepted/ (schema training_acceptance/2,
  robot arm) and prints ARM REACH ACCEPTED, stopping training.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "arm"))

import torch  # noqa: E402

from walk.eval import arm_acceptance as aa  # noqa: E402
from walk.train import gpu_train as gt  # noqa: E402

TRAINED_ACTOR = (ROOT / "runs/gpu/20260902-142436-arm-reach-kr240/artifacts"
                 / "train/gpu-train-out/actor_final.pt")


def _same_cells(tc, probe_eps: dict, harness_eps: dict):
    tc.assertEqual(sorted(probe_eps), sorted(harness_eps))
    for cell, h in harness_eps.items():
        p = probe_eps[cell]
        for key in ("passed", "failed_criteria", "acquisition_times_s",
                    "criteria"):
            tc.assertEqual(p[key], h[key], f"{cell}: {key}")


class ProbeEqualsHarness(unittest.TestCase):
    def test_ik_baseline_kr240(self):
        factory = lambda: aa.ScriptedIKPolicy("kr240")  # noqa: E731
        probe = aa.run_batched_probe(
            "kr240", lambda n, seed: aa.make_env("kr240", seed, "serial", None,
                                                 environments=n),
            factory, quiet=True)
        harness = aa.run_acceptance("kr240", factory, quiet=True)
        self.assertEqual(probe["cells_total"], 12)
        _same_cells(self, probe["episodes"], harness["episodes"])
        self.assertEqual(probe["accepted"], harness["accepted"])
        self.assertEqual(len(probe["failed_cells"]),
                         12 - sum(v["passed"] for v in harness["episodes"].values()))
        print(f"arm probe == harness (ik): {probe['cells_passed']}/12 "
              f"failed={probe['failed_cells']}", file=sys.stderr)

    def test_ik_baseline_traces_bitwise_equal(self):
        cells = aa.protocol_cells()
        env = aa.make_env("kr240", 4242, "serial", None, environments=len(cells))
        try:
            batched = aa.capture_arm_episodes(
                aa.BatchedCellArmEnv(env, cells), aa.ScriptedIKPolicy("kr240"),
                tier=[t for _, t in cells], seconds=8.0, seed=4242)
        finally:
            env.close()
        for seed in aa.SEEDS:
            env = aa.make_env("kr240", seed, "serial", None)
            try:
                for tier in aa.TIERS:
                    ref = aa.capture_arm_episodes(
                        env, aa.ScriptedIKPolicy("kr240"), tier=tier,
                        seconds=8.0, seed=seed)[0]
                    got = batched[cells.index((seed, tier))]
                    self.assertEqual(got["tier"], ref["tier"])
                    self.assertEqual(got["terminated"], ref["terminated"])
                    for key, col in ref["ticks"].items():
                        self.assertTrue(
                            np.array_equal(np.asarray(col),
                                           np.asarray(got["ticks"][key])),
                            f"seed {seed} tier {tier}: column {key} differs")
            finally:
                env.close()

    def test_gpu_train_probe_on_trained_actor_equals_harness(self):
        if not TRAINED_ACTOR.is_file():
            self.skipTest(f"{TRAINED_ACTOR} not present")
        arch, actor = aa.load_actor(str(TRAINED_ACTOR))
        cfg = gt.GpuTrainConfig(robot="arm", variant="kr240", policy=arch,
                                device="cpu", out="/tmp/unused",
                                torch_threads=1)
        probe = gt.run_robot_probe(cfg, actor, torch.device("cpu"))
        harness = aa.run_acceptance(
            "kr240", lambda: aa.make_actor_policy(arch, actor), quiet=True)
        self.assertEqual(probe["cells_total"], 12)
        _same_cells(self, probe["episodes"], harness["episodes"])
        self.assertEqual(probe["confirmed"], harness["accepted"])
        self.assertEqual(probe["protocol"]["tiers"], [0, 1, 2])
        self.assertEqual(probe["protocol"]["variant"], "kr240")
        print(f"arm probe == harness (trained actor): "
              f"{probe['cells_passed']}/12 probe_wall={probe['probe_wall_s']:.1f}s",
              file=sys.stderr)


class RestorationProof(unittest.TestCase):
    KEYS = ("reward_mean", "reward_std", "ep_return_mean", "ep_len_mean",
            "episodes", "pi_loss", "v_loss", "entropy", "approx_kl",
            "clip_frac", "env_steps", "gate_proxy_ep_qualified_l",
            "gate_proxy_ep_alt_violations", "curriculum_stage")

    def _run(self, out: Path, accept_every: int):
        cfg = gt.GpuTrainConfig(
            robot="arm", variant="lite", envs=2, lane_env=True, horizon=8,
            updates=4, seed=20260902, device="cpu", out=str(out),
            preflight_steps=0, checkpoint_every=1000, torch_threads=1,
            quiet=True, curriculum="arm-reach", accept_every=accept_every)
        return gt.train(cfg)

    def test_metrics_stream_identical_with_and_without_probes(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = self._run(Path(tmp) / "plain", 0)
            probed = self._run(Path(tmp) / "probed", 2)
            train_a = [m for m in plain if m.get("kind") == "train"]
            train_b = [m for m in probed if m.get("kind") == "train"]
            self.assertEqual([m["update"] for m in train_b], [1, 2, 3, 4])
            for a, b in zip(train_a, train_b):
                for k in self.KEYS:
                    self.assertEqual(a.get(k), b.get(k),
                                     f"update {a['update']} field {k}")
            accepts = [m for m in probed if m.get("kind") == "accept"]
            self.assertEqual([m["update"] for m in accepts], [2, 4])
            for m in accepts:
                self.assertEqual(m["robot"], "arm")
                self.assertEqual(m["variant"], "lite")
                self.assertEqual(m["cells_total"], 12)
                self.assertGreater(m["probe_wall_s"], 0.0)
            print(f"arm restoration proof: probes cost "
                  f"{[round(m['probe_wall_s'], 2) for m in accepts]} s",
                  file=sys.stderr)


class AcceptStopsTraining(unittest.TestCase):
    def test_passing_probe_writes_artifacts_and_stops(self):
        def fake_probe(cfg, actor, device):
            cells = {f"seed{s}-tier{t}": {"passed": True, "failed_criteria": [],
                                          "acquisition_times_s": [1.0] * 5,
                                          "criteria": {}}
                     for s in aa.SEEDS for t in aa.TIERS}
            return {"stage1_passed": True, "confirmed": True,
                    "episodes": cells, "confirmation": {},
                    "cells_passed": 12, "cells_total": 12, "failed_cells": [],
                    "protocol": {"seeds": list(aa.SEEDS),
                                 "tiers": list(aa.TIERS),
                                 "variant": "kr240", "seconds": 8.0},
                    "probe_wall_s": 0.25}

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(gt, "run_robot_probe", fake_probe):
            out = Path(tmp) / "run"
            cfg = gt.GpuTrainConfig(
                robot="arm", variant="kr240", envs=2, lane_env=True,
                horizon=8, updates=5, seed=3, device="cpu", out=str(out),
                preflight_steps=0, checkpoint_every=1000, torch_threads=1,
                quiet=False, accept_every=1)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                metrics = gt.train(cfg)
            text = buf.getvalue()
            self.assertEqual(len([m for m in metrics if m["kind"] == "train"]), 1)
            self.assertIn("[accept u1] stage1=True (12/12 episodes) "
                          "confirmed=True", text)
            self.assertIn("ARM REACH ACCEPTED at update 1 after", text)
            acc = json.loads((out / "accepted" / "acceptance.json").read_text())
            self.assertEqual(acc["schema"], "duckgridwalk.training_acceptance/2")
            self.assertEqual(acc["robot"], "arm")
            self.assertEqual(acc["variant"], "kr240")
            self.assertEqual(acc["protocol"]["tiers"], [0, 1, 2])
            self.assertTrue((out / "accepted" / "actor_accepted.pt").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
