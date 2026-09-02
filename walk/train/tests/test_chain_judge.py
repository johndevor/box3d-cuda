"""gpu/chain.py: acceptance stop for every robot's ACCEPTED line / accepted
artifacts, robot inference from the spec, the local CPU-harness judge
plumbing and the best-by-judge-cells bookkeeping.

Run: .venv/bin/python -B -m unittest walk.train.tests.test_chain_judge
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("gpu_chain", ROOT / "gpu/chain.py")
chain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chain)

HUMANOID_LOG = """[u 120] sps=   146987 rew=+3.0096 pi=+0.0005 v=624.5926 kl=0.0066 faults=0
[accept u120] stage1=False (11/12 episodes) confirmed=False probe_wall=3.2s failed=['seed90210-cmd0.50']
[u 128] sps=   143240 rew=+2.9935 pi=+0.0021 v=624.7112 kl=0.0102 faults=0
[accept u128] stage1=True (12/12 episodes) confirmed=True probe_wall=3.1s
HUMANOID WALKING ACCEPTED at update 128 after 1234.5 s (31.0 s probing)
"""
DUCK_LOG = """[accept u6245] stage1=True (12/12 episodes) confirmed=True probe_wall=40.1s
WALKING ACCEPTED at update 6245 after 27957.7 s (401.2 s probing)
"""
ARM_LOG = """[accept u16] stage1=False (9/12 episodes) confirmed=False probe_wall=1.9s failed=['seed4242-tier2', 'seed7-tier2', 'seed90210-tier2']
"""


def _run_dir(tmp: Path, log: str, accepted: bool = False,
             metrics_rows: list | None = None) -> Path:
    run = tmp / "20260902-160000-spec"
    (run / "logs").mkdir(parents=True)
    (run / "logs" / "train.log").write_text(log)
    out = run / "artifacts" / "train" / "gpu-train-out"
    out.mkdir(parents=True)
    (out / "actor_final.pt").write_bytes(b"actor-bytes")
    (out / "latest.pt").write_bytes(b"ckpt-bytes")
    if metrics_rows is not None:
        (out / "metrics.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in metrics_rows))
    if accepted:
        (out / "accepted").mkdir()
        (out / "accepted" / "acceptance.json").write_text('{"accepted": true}')
        (out / "accepted" / "actor_accepted.pt").write_bytes(b"acc")
    return run


class LegScoreTests(unittest.TestCase):
    def test_humanoid_probe_lines_and_accepted_line(self):
        with tempfile.TemporaryDirectory() as td:
            run = _run_dir(Path(td), HUMANOID_LOG)
            best, probes, confirmed = chain.leg_score(run)
            self.assertEqual((best, probes, confirmed), (12.0, 2, True))

    def test_duck_line_still_matches(self):
        with tempfile.TemporaryDirectory() as td:
            best, probes, confirmed = chain.leg_score(_run_dir(Path(td), DUCK_LOG))
            self.assertEqual((best, probes, confirmed), (12.0, 1, True))

    def test_arm_failed_tail_does_not_break_regex(self):
        with tempfile.TemporaryDirectory() as td:
            best, probes, confirmed = chain.leg_score(_run_dir(Path(td), ARM_LOG))
            self.assertEqual((best, probes, confirmed), (9.0, 1, False))

    def test_accepted_artifacts_alone_stop_the_chain(self):
        with tempfile.TemporaryDirectory() as td:
            run = _run_dir(Path(td), "", accepted=True)
            self.assertIsNotNone(chain.accepted_dir(run))
            best, probes, confirmed = chain.leg_score(run)
            self.assertEqual(probes, 0)
            self.assertTrue(confirmed)

    def test_proxy_fallback_without_probes(self):
        rows = [{"kind": "train", "update": i, "gate_proxy_ep_qualified_l": 2.0,
                 "gate_proxy_ep_qualified_r": 3.0, "ep_len_mean": 100.0}
                for i in range(5)]
        with tempfile.TemporaryDirectory() as td:
            run = _run_dir(Path(td), "", metrics_rows=rows)
            self.assertEqual(chain.leg_score(run), (5.0, 0, False))
            self.assertIsNone(chain.accepted_dir(run))


class RobotInferenceTests(unittest.TestCase):
    def test_infer_from_specs(self):
        cases = {
            "gpu/specs/humanoid-tree-stocky-continue.json": ("humanoid", "h1_stocky"),
            "gpu/specs/humanoid-tree-tall-continue.json": ("humanoid", "h1_tall"),
            "gpu/specs/humanoid-tree-continue.json": ("humanoid", None),
            "gpu/specs/humanoid-generalist.json": ("humanoid", None),
            "gpu/specs/arm-reach-kr240.json": ("arm", "kr240"),
            "gpu/specs/arm-reach-lite.json": ("arm", "lite"),
            "gpu/specs/continue-ff-short.json": ("duck", None),
        }
        for rel, want in cases.items():
            cmd = chain.spec_train_command(ROOT / rel)
            self.assertIn("walk.train.gpu_train", cmd, rel)
            self.assertEqual(chain.infer_robot(cmd), want, rel)

    def test_specs_probe_and_collect_accepted_artifacts(self):
        expect_n = {"humanoid-tree-stocky-continue": 8,
                    "humanoid-tree-tall-continue": 8,
                    "humanoid-tree-continue": 8, "humanoid-generalist": 8,
                    "arm-reach-kr240": 16, "arm-reach-lite": 16}
        for name, n in expect_n.items():
            spec = json.loads((ROOT / f"gpu/specs/{name}.json").read_text())
            train = next(j for j in spec["jobs"] if j["name"] == "train")
            self.assertIn(f"--accept-every {n} ", train["command"], name)
            self.assertIn("gpu-train-out/accepted/acceptance.json",
                          train["artifacts"], name)
            self.assertIn("gpu-train-out/accepted/actor_accepted.pt",
                          train["artifacts"], name)

    def test_judge_command_per_robot(self):
        actor, out = Path("/a/actor_final.pt"), Path("/o")
        h = chain.judge_command("humanoid", "h1_stocky", actor, out)
        self.assertEqual(h[2:4], ["-m", "walk.eval.humanoid_acceptance"])
        self.assertEqual(h[-2:], ["--variant", "h1_stocky"])
        base = chain.judge_command("humanoid", None, actor, out)
        self.assertNotIn("--variant", base)
        arm = chain.judge_command("arm", None, actor, out)
        self.assertEqual(arm[2:4], ["-m", "walk.eval.arm_acceptance"])
        self.assertEqual(arm[-2:], ["--variant", "kr240"])
        duck = chain.judge_command("duck", None, actor, out)
        self.assertEqual(duck[2:4], ["-m", "walk.eval.acceptance"])
        with self.assertRaises(ValueError):
            chain.judge_command("emu", None, actor, out)


class JudgeAndBestTests(unittest.TestCase):
    RECORD = {"accepted": False, "episodes": {
        "seed4242-cmd0.50": {"passed": True}, "seed4242-cmd0.75": {"passed": True},
        "seed90210-cmd0.50": {"passed": False, "failed_criteria": ["footfalls_alternate"]}}}

    def test_summarize(self):
        s = chain.summarize_acceptance(self.RECORD)
        self.assertEqual(s, {"cells_passed": 2, "cells_total": 3,
                             "failed_cells": ["seed90210-cmd0.50"],
                             "accepted": False})

    def test_judge_actor_runs_harness_and_parses_record(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "judge"
            seen = {}

            def fake_run(cmd, cwd, capture_output, text, timeout):
                seen["cmd"] = cmd
                Path(cmd[cmd.index("--out") + 1]).mkdir(parents=True, exist_ok=True)
                (Path(cmd[cmd.index("--out") + 1]) / "acceptance.json").write_text(
                    json.dumps(self.RECORD))
                return mock.Mock(returncode=1, stdout="2/3 episodes pass\n", stderr="")

            with mock.patch.object(chain.subprocess, "run", fake_run):
                judged = chain.judge_actor("humanoid", "h1_stocky",
                                           Path("/x/actor_final.pt"), out)
            self.assertEqual(seen["cmd"][3], "walk.eval.humanoid_acceptance")
            self.assertEqual(judged["cells_passed"], 2)
            self.assertEqual(judged["failed_cells"], ["seed90210-cmd0.50"])
            self.assertTrue((out / "judge.log").is_file())

    def test_judge_actor_without_record_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(chain.subprocess, "run",
                                   lambda *a, **k: mock.Mock(returncode=2, stdout="", stderr="boom")):
                self.assertIsNone(chain.judge_actor("arm", "lite", Path("/x.pt"),
                                                    Path(td) / "j"))

    def test_update_best_keeps_max_cells_across_calls(self):
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td) / "runs"
            a1, a2, a3 = (Path(td) / f"a{i}.pt" for i in range(3))
            for i, p in enumerate((a1, a2, a3)):
                p.write_bytes(bytes([i]))
            j = lambda n: {"cells_passed": n, "cells_total": 12,  # noqa: E731
                           "failed_cells": [], "accepted": n == 12}
            self.assertTrue(chain.update_best("spec", j(10), a1, Path("/r1"), 1, runs))
            self.assertFalse(chain.update_best("spec", j(10), a2, Path("/r2"), 2, runs))
            self.assertTrue(chain.update_best("spec", j(11), a3, Path("/r3"), 3, runs))
            best_pt, best_json = chain.best_paths("spec", runs)
            self.assertEqual(best_pt.read_bytes(), bytes([2]))
            rec = json.loads(best_json.read_text())
            self.assertEqual((rec["cells_passed"], rec["leg"]), (11, 3))
            # a worse later leg never overwrites
            self.assertFalse(chain.update_best("spec", j(9), a1, Path("/r4"), 4, runs))
            self.assertEqual(best_pt.read_bytes(), bytes([2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
