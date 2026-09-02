"""--accept-every for --robot humanoid: probe == CPU harness, no training
perturbation, acceptance artifacts + self-stop.

Run: .venv/bin/python -B humanoid/tests/test_humanoid_accept_probe.py

Gates:
- EQUALITY (the frozen judge's verdict): gpu_train.run_robot_probe (the
  in-run probe: 12 cells as ONE E=12 batch on the training lane class, a
  fresh CudaHumanoidLane, DR/gates/RSI off) and walk.eval.humanoid_acceptance
  .run_acceptance (the CPU harness: 12 sequential E=1 episodes) agree cell
  for cell -- passed, failed criteria, qualified footfalls AND the full
  criteria detail (displacement, tilt, footfall sequence floats) -- on
    * evidence/humanoid-accepted-20260902/actor-humanoid-accepted.pt (12/12)
    * evidence/humanoid-stocky-20260902/actor-stocky-11of12.pt, --variant
      h1_stocky (11/12, the same failing cell);
- the batched capture's per-tick traces are BITWISE equal to the harness's
  for every cell of one seed (the row-wise policy + pinned harness phase0);
- RESTORATION PROOF: a humanoid --lane-env training run with --curriculum,
  --rsi-fraction and --randomization that probes every 2 updates writes the
  SAME train metrics stream as the identical run without probes (the probe
  builds its own lane; the trainer's knobs, counters and RNG streams are
  untouched), and its accept rows carry the 12-cell record;
- a passing probe writes <out>/accepted/{actor_accepted.pt, acceptance.json}
  (schema training_acceptance/2, robot humanoid), prints HUMANOID WALKING
  ACCEPTED and stops training (exit path identical to the duck's);
- the probe env is the fresh default-knob instance the harness builds.
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
sys.path.insert(0, str(ROOT / "humanoid"))

import torch  # noqa: E402

from walk.eval import humanoid_acceptance as ha  # noqa: E402
from walk.eval.capture import capture_episodes  # noqa: E402
from walk.train import gpu_train as gt  # noqa: E402

ACCEPTED_ACTOR = ROOT / "evidence/humanoid-accepted-20260902/actor-humanoid-accepted.pt"
STOCKY_ACTOR = ROOT / "evidence/humanoid-stocky-20260902/actor-stocky-11of12.pt"
STOCKY_FAILING_CELL = "seed90210-cmd0.50"       # measured with the CPU harness


def _cfg(variant=None, **over):
    base = dict(robot="humanoid", variant=variant, policy="ff", device="cpu",
                library=None, out="/tmp/unused", torch_threads=1)
    base.update(over)
    return gt.GpuTrainConfig(**base)


class ProbeEqualsHarness(unittest.TestCase):
    def _compare(self, actor_path: Path, variant, expect_pass: int,
                 expect_failed: list):
        arch, actor = ha.load_actor(str(actor_path))
        probe = gt.run_robot_probe(_cfg(variant), actor, torch.device("cpu"))
        harness = ha.run_acceptance(str(actor_path), quiet=True,
                                    variant=variant)
        self.assertEqual(probe["cells_total"], 12)
        self.assertEqual(probe["cells_passed"], expect_pass)
        self.assertEqual(probe["failed_cells"], expect_failed)
        self.assertEqual(probe["confirmed"], expect_pass == 12)
        self.assertEqual(probe["stage1_passed"], expect_pass == 12)
        self.assertEqual(sorted(probe["episodes"]), sorted(harness["episodes"]))
        for cell, h in harness["episodes"].items():
            p = probe["episodes"][cell]
            for key in ("passed", "failed_criteria", "qualified", "left",
                        "right", "swings_examined", "criteria"):
                self.assertEqual(p[key], h[key], f"{cell}: {key}")
        self.assertEqual(harness["accepted"], probe["confirmed"])
        print(f"probe == harness: {actor_path.name} variant={variant or 'h1'}"
              f" {probe['cells_passed']}/12 failed={probe['failed_cells']}"
              f" probe_wall={probe['probe_wall_s']:.1f}s", file=sys.stderr)

    def test_accepted_actor_12_of_12_both(self):
        self._compare(ACCEPTED_ACTOR, None, 12, [])

    def test_stocky_actor_11_of_12_same_failing_cell(self):
        self._compare(STOCKY_ACTOR, "h1_stocky", 11, [STOCKY_FAILING_CELL])

    def test_traces_bitwise_equal_one_seed(self):
        """Batched E=12 capture vs the harness's E=1 capture, seed 90210
        (the stocky actor's marginal seed): every tick column identical."""
        arch, actor = ha.load_actor(str(STOCKY_ACTOR))
        cells = ha.protocol_cells()
        env = ha.make_env(4242, "serial", None, "h1_stocky",
                          environments=len(cells))
        try:
            wrapped = ha.BatchedCellEnv(
                env, [c for _, c in cells],
                [ha.harness_phase0(s, ha.COMMANDS.index(c)) for s, c in cells])
            batched = capture_episodes(
                wrapped, ha.make_policy(arch, actor, rowwise=True),
                command=None, seconds=8.0, seed=4242)
        finally:
            env.close()
        env = ha.make_env(90210, "serial", None, "h1_stocky")
        try:
            for cmd in ha.COMMANDS:
                ref = capture_episodes(env, ha.make_policy(arch, actor),
                                       command=cmd, seconds=8.0, seed=90210)[0]
                got = batched[cells.index((90210, cmd))]
                self.assertEqual(got["command_mps"], ref["command_mps"])
                self.assertEqual(got["terminated"], ref["terminated"])
                for key, col in ref["ticks"].items():
                    self.assertTrue(
                        np.array_equal(np.asarray(col),
                                       np.asarray(got["ticks"][key])),
                        f"cmd {cmd}: tick column {key} differs")
        finally:
            env.close()


class ProbeEnvIsFreshDefault(unittest.TestCase):
    def test_probe_env_knobs(self):
        env = gt._make_robot_probe_env(_cfg("h1_stocky"), 12, 4242)
        try:
            self.assertEqual(env.E, 12)
            self.assertEqual(env.variant, "h1_stocky")
            self.assertEqual(env._lane.variant, "h1_stocky")
            self.assertIsNone(env._lane.randomization)
            self.assertFalse(env._lane.fast_termination)
            self.assertFalse(env._rz_on)
        finally:
            env.close()


class RestorationProof(unittest.TestCase):
    """The probe must not perturb training: metrics stream identical with
    and without probes when seeds are pinned (lane-env + curriculum + RSI +
    DR -- every knob the probe turns off in ITS OWN lane)."""

    KEYS = ("reward_mean", "reward_std", "ep_return_mean", "ep_len_mean",
            "episodes", "pi_loss", "v_loss", "entropy", "approx_kl",
            "clip_frac", "env_steps", "gate_proxy_ep_qualified_l",
            "gate_proxy_ep_qualified_r", "gate_proxy_ep_alt_violations",
            "curriculum_stage")

    def _run(self, out: Path, accept_every: int):
        cfg = gt.GpuTrainConfig(
            robot="humanoid", envs=2, lane_env=True, horizon=8, updates=4,
            seed=20260902, device="cpu", out=str(out), preflight_steps=0,
            checkpoint_every=1000, torch_threads=1, quiet=True,
            curriculum="humanoid-walk", rsi_fraction=0.5,
            randomization={"r_mass": 0.15, "r_friction": 0.3, "r_kp": 0.15,
                           "r_damping": 0.3, "max_latency_steps": 2,
                           "r_gravity": 0.5095},
            accept_every=accept_every)
        return gt.train(cfg)

    def test_metrics_stream_identical_with_and_without_probes(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = self._run(Path(tmp) / "plain", 0)
            probed = self._run(Path(tmp) / "probed", 2)
            train_a = [m for m in plain if m.get("kind") == "train"]
            train_b = [m for m in probed if m.get("kind") == "train"]
            self.assertEqual([m["update"] for m in train_a], [1, 2, 3, 4])
            self.assertEqual([m["update"] for m in train_b], [1, 2, 3, 4])
            for a, b in zip(train_a, train_b):
                for k in self.KEYS:
                    self.assertEqual(a.get(k), b.get(k),
                                     f"update {a['update']} field {k}: "
                                     f"{a.get(k)!r} != {b.get(k)!r}")
            accepts = [m for m in probed if m.get("kind") == "accept"]
            self.assertEqual([m["update"] for m in accepts], [2, 4])
            for m in accepts:
                self.assertEqual(m["robot"], "humanoid")
                self.assertEqual(m["cells_total"], 12)
                self.assertFalse(m["accepted"])       # a fresh net cannot walk
                self.assertGreater(m["probe_wall_s"], 0.0)
                self.assertEqual(len(m["episodes"]), 12)
                self.assertEqual(len(m["failed_cells"]), 12 - m["cells_passed"])
            self.assertFalse((Path(tmp) / "probed" / "accepted").exists())
            self.assertFalse([m for m in plain if m.get("kind") == "accept"])
            ck = torch.load(Path(tmp) / "probed" / "latest.pt",
                            map_location="cpu", weights_only=False)
            self.assertGreater(ck["probe_wall_s"], 0.0)
            # probe wall is booked separately from training wall
            self.assertLess(ck["train_wall_s"], ck["train_wall_s"]
                            + ck["probe_wall_s"])
            print(f"restoration proof: 4 updates identical; probes cost "
                  f"{[round(m['probe_wall_s'], 2) for m in accepts]} s",
                  file=sys.stderr)


class AcceptStopsTraining(unittest.TestCase):
    def test_passing_probe_writes_artifacts_and_stops(self):
        def fake_probe(cfg, actor, device):
            cells = {f"seed{s}-cmd{c:.2f}": {"passed": True, "qualified": 12,
                                              "left": 6, "right": 6,
                                              "failed_criteria": [],
                                              "criteria": {},
                                              "swings_examined": 12}
                     for s in ha.SEEDS for c in ha.COMMANDS}
            return {"stage1_passed": True, "confirmed": True,
                    "episodes": cells, "confirmation": {},
                    "cells_passed": 12, "cells_total": 12, "failed_cells": [],
                    "protocol": {"seeds": list(ha.SEEDS),
                                 "commands": list(ha.COMMANDS),
                                 "variant": "h1_stocky", "seconds": 8.0},
                    "probe_wall_s": 0.5}

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(gt, "run_robot_probe", fake_probe):
            out = Path(tmp) / "run"
            cfg = gt.GpuTrainConfig(
                robot="humanoid", variant="h1_stocky", envs=2, lane_env=True,
                horizon=8, updates=5, seed=3, device="cpu", out=str(out),
                preflight_steps=0, checkpoint_every=1000, torch_threads=1,
                quiet=False, accept_every=1)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                metrics = gt.train(cfg)
            text = buf.getvalue()
            self.assertEqual(len([m for m in metrics if m["kind"] == "train"]), 1)
            self.assertEqual(len([m for m in metrics if m["kind"] == "accept"]), 1)
            self.assertIn("[accept u1] stage1=True (12/12 episodes) "
                          "confirmed=True", text)
            self.assertIn("HUMANOID WALKING ACCEPTED at update 1 after", text)
            self.assertNotIn("\nWALKING ACCEPTED", text)
            acc = json.loads((out / "accepted" / "acceptance.json").read_text())
            self.assertEqual(acc["schema"], "duckgridwalk.training_acceptance/2")
            self.assertTrue(acc["accepted"])
            self.assertEqual(acc["robot"], "humanoid")
            self.assertEqual(acc["variant"], "h1_stocky")
            self.assertEqual(acc["update"], 1)
            self.assertEqual(acc["cells_passed"], 12)
            self.assertEqual(acc["protocol"]["commands"], [0.5, 0.75, 1.0])
            self.assertEqual(acc["probe_wall_last_s"], 0.5)
            from walk.train import ppo  # noqa: PLC0415
            arch, sd = ppo.unpack_actor_file(torch.load(
                out / "accepted" / "actor_accepted.pt", map_location="cpu",
                weights_only=True))
            self.assertEqual(arch, "ff")
            ppo.Actor(58, 14).load_state_dict(sd)
            ck = torch.load(out / "latest.pt", map_location="cpu",
                            weights_only=False)
            self.assertEqual(ck["update"], 1)
            self.assertEqual(ck["config"]["variant"], "h1_stocky")


class CudaCandidateProtocol(unittest.TestCase):
    """Non-authority (cuda) lane: two CONSECUTIVE 12/12 probes nominate a
    candidate; training continues; candidates are bounded. Simulated by
    forcing _probe_lane_is_authority False on the cpu serial lane and
    scripting the probe verdicts."""

    @staticmethod
    def _probe(passed: bool):
        cells = {f"seed{s}-cmd{c:.2f}": {"passed": passed, "qualified": 12,
                                          "left": 6, "right": 6,
                                          "failed_criteria": [] if passed
                                          else ["footfalls_alternate"],
                                          "criteria": {}, "swings_examined": 12}
                 for s in ha.SEEDS for c in ha.COMMANDS}
        n = 12 if passed else 0
        return {"stage1_passed": passed, "confirmed": passed,
                "episodes": cells, "confirmation": {},
                "cells_passed": n, "cells_total": 12,
                "failed_cells": [] if passed else sorted(cells),
                "protocol": {"seeds": list(ha.SEEDS), "commands": list(ha.COMMANDS),
                             "variant": "h1_stocky", "seconds": 8.0},
                "probe_wall_s": 0.1}

    def _run(self, verdicts, updates, max_candidates=None):
        it = iter(verdicts)
        fake = lambda cfg, actor, device: self._probe(next(it))  # noqa: E731
        patches = [mock.patch.object(gt, "run_robot_probe", fake),
                   mock.patch.object(gt, "_probe_lane_is_authority",
                                     lambda cfg, device: False)]
        if max_candidates is not None:
            patches.append(mock.patch.object(gt, "MAX_CANDIDATES", max_candidates))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            cfg = gt.GpuTrainConfig(
                robot="humanoid", variant="h1_stocky", envs=2, lane_env=True,
                horizon=8, updates=updates, seed=3, device="cpu", out=str(out),
                preflight_steps=0, checkpoint_every=1000, torch_threads=1,
                quiet=False, accept_every=1)
            buf = io.StringIO()
            with contextlib.ExitStack() as st:
                for pt in patches:
                    st.enter_context(pt)
                with contextlib.redirect_stdout(buf):
                    metrics = gt.train(cfg)
            acc = out / "accepted"
            files = sorted(p.name for p in acc.glob("*")) if acc.exists() else []
            jsons = {p.name: json.loads(p.read_text())
                     for p in acc.glob("*.json")} if acc.exists() else {}
            return metrics, buf.getvalue(), files, jsons

    def test_two_consecutive_passes_nominate_and_training_continues(self):
        # pass, pass(candidate), fail, pass, pass(candidate), pass(candidate)
        metrics, text, files, jsons = self._run(
            [True, True, False, True, True, True], updates=6)
        train = [m for m in metrics if m["kind"] == "train"]
        accepts = [m for m in metrics if m["kind"] == "accept"]
        self.assertEqual(len(train), 6)                    # never self-stopped
        self.assertEqual([m["consecutive_passes"] for m in accepts],
                         [1, 2, 0, 1, 2, 3])
        self.assertTrue(all(m["lane_is_authority"] is False for m in accepts))
        self.assertIn("[accept u1] on-device 12/12 #1: need 2 consecutive", text)
        self.assertIn("HUMANOID WALKING ACCEPTED-CANDIDATE at update 2", text)
        self.assertIn("HUMANOID WALKING ACCEPTED-CANDIDATE at update 5", text)
        self.assertIn("HUMANOID WALKING ACCEPTED-CANDIDATE at update 6", text)
        self.assertNotIn("HUMANOID WALKING ACCEPTED at update", text)
        self.assertIn("candidates=3", text)
        self.assertEqual(files, ["acceptance.json", "actor_accepted.pt",
                                 "candidate_000002.json", "candidate_000002.pt",
                                 "candidate_000005.json", "candidate_000005.pt",
                                 "candidate_000006.json", "candidate_000006.pt"])
        self.assertEqual(jsons["acceptance.json"]["update"], 2)  # first candidate
        self.assertFalse(jsons["acceptance.json"]["lane_is_authority"])
        self.assertEqual(jsons["acceptance.json"]["consecutive_passes"], 2)
        self.assertEqual(jsons["candidate_000006.json"]["consecutive_passes"], 3)
        self.assertIsNone(jsons["candidate_000006.json"]["cpu_confirmed"])
        self.assertEqual([m.get("candidate", "").split("/")[-1] for m in accepts
                          if m.get("candidate")],
                         ["candidate_000002.pt", "candidate_000005.pt",
                          "candidate_000006.pt"])

    def test_candidates_are_bounded_to_newest(self):
        metrics, text, files, jsons = self._run([True] * 5, updates=5,
                                                max_candidates=2)
        # candidates at u2..u5; only the newest 2 kept
        self.assertEqual([f for f in files if f.startswith("candidate_")],
                         ["candidate_000004.json", "candidate_000004.pt",
                          "candidate_000005.json", "candidate_000005.pt"])
        self.assertIn("actor_accepted.pt", files)          # first candidate kept

    def test_single_pass_on_authority_lane_still_self_stops(self):
        fake = lambda cfg, actor, device: self._probe(True)  # noqa: E731
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(gt, "run_robot_probe", fake):
            cfg = gt.GpuTrainConfig(
                robot="humanoid", variant="h1_stocky", envs=2, lane_env=True,
                horizon=8, updates=5, seed=3, device="cpu",
                out=str(Path(tmp) / "run"), preflight_steps=0,
                checkpoint_every=1000, torch_threads=1, quiet=True,
                accept_every=1)
            metrics = gt.train(cfg)
            accepts = [m for m in metrics if m["kind"] == "accept"]
            self.assertEqual(len([m for m in metrics if m["kind"] == "train"]), 1)
            self.assertTrue(accepts[0]["lane_is_authority"])
            self.assertTrue(accepts[0]["accepted"])
            self.assertTrue(gt._probe_lane_is_authority(cfg, torch.device("cpu")))
            self.assertFalse(gt._probe_lane_is_authority(cfg, torch.device("cuda")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
