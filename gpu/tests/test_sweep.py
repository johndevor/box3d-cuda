"""Unit tests for gpu/sweep.py with the launcher / generator / eval
subprocesses replaced by an in-memory fake runner (no provider calls, no GPU,
no doppler, no DAYTONA_API_KEY needed).

Run:  .venv/bin/python -m unittest discover -s gpu/tests -v
"""

import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sweep  # noqa: E402


def make_episode(cmd, passed, feet):
    """Minimal walk/eval/gait.evaluate_episode-shaped result."""
    return {"schema": "duckgridwalk.gait_eval/1", "rejected": False,
            "command_mps": cmd, "passed": passed,
            "criteria": {}, "footfalls": [{"foot": f} for f in feet]}


DEFAULT_EPISODES = [
    make_episode(0.10, True, ["left", "right", "left", "right", "left", "right"]),
    make_episode(0.15, True, ["right", "left", "right", "left", "right", "left"]),
    make_episode(0.20, False, ["left", "left", "right"]),
]


class FakeRunner:
    """Stands in for subprocess.run for all three sweep subprocesses."""

    def __init__(self, sweep_mod, launch_exits=(0,), episodes=None,
                 generator_exit=0, eval_exit=0, wrong_header=False):
        self.sweep = sweep_mod
        self.launch_exits = list(launch_exits)
        self.episodes = DEFAULT_EPISODES if episodes is None else episodes
        self.generator_exit = generator_exit
        self.eval_exit = eval_exit
        self.wrong_header = wrong_header
        self.calls = []          # (kind, cmd, kwargs)
        self._stamp = 0

    def __call__(self, cmd, **kw):
        cmd_s = [str(c) for c in cmd]
        if any("generate_model.py" in c for c in cmd_s):
            self.calls.append(("generate", cmd_s, kw))
            if self.generator_exit:
                return SimpleNamespace(returncode=self.generator_exit)
            out = Path(cmd_s[cmd_s.index("--output") + 1])
            env = kw["env"]
            base = "9999.0" if self.wrong_header else env[sweep.ENV_BASE]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                "// fake generated header\n"
                f"#define DW_PHASE_HZ_BASE {base}\n"
                f"#define DW_PHASE_HZ_PER_MPS {env[sweep.ENV_PER_MPS]}\n")
            return SimpleNamespace(returncode=0)
        if any("run_daytona.py" in c for c in cmd_s):
            self.calls.append(("launch", cmd_s, kw))
            rc = self.launch_exits.pop(0) if self.launch_exits else 0
            if "--dry-run" not in cmd_s and rc in (0, 2):
                # exit 2 = job failed: run dir exists but (here) no artifacts
                self._stamp += 1
                run_dir = (self.sweep.RUNS_GPU_DIR /
                           f"20260831-{self._stamp:06d}-sweep-train")
                run_dir.mkdir(parents=True)
                if rc == 0:
                    actor = run_dir / self.sweep.ACTOR_REL
                    actor.parent.mkdir(parents=True, exist_ok=True)
                    actor.write_bytes(b"fake-actor")
                    (run_dir / self.sweep.LATEST_REL).write_bytes(b"fake-latest")
            return SimpleNamespace(returncode=rc)
        if "-c" in cmd_s:
            self.calls.append(("eval", cmd_s, kw))
            if self.eval_exit:
                return SimpleNamespace(returncode=self.eval_exit)
            out = Path(cmd_s[cmd_s.index("-c") + 4])   # -c SCRIPT actor lib OUT
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(self.episodes))
            return SimpleNamespace(returncode=0)
        raise AssertionError(f"unexpected command: {cmd_s}")

    def of(self, kind):
        return [c for c in self.calls if c[0] == kind]


class SweepTestCase(unittest.TestCase):
    """Base: redirect every repo-relative path sweep.py writes to a tmp dir."""

    def setUp(self):
        self._stack = ExitStack()
        self.tmp = Path(self._stack.enter_context(tempfile.TemporaryDirectory()))
        self._stack.enter_context(mock.patch.object(sweep, "REPO_ROOT", self.tmp))
        self._stack.enter_context(
            mock.patch.object(sweep, "RUNS_GPU_DIR", self.tmp / "runs" / "gpu"))
        self.warmstart = self.tmp / "warmstart-sweep.pt"
        self.warmstart.write_bytes(b"fake-checkpoint")
        self.out = self.tmp / "sweep-out"

    def tearDown(self):
        self._stack.close()

    def run_main(self, runner, configs="[[0.8,6],[1.2,4]]", extra=()):
        argv = ["--configs", configs, "--warmstart", str(self.warmstart),
                "--out", str(self.out), *extra]
        logs = []
        code = sweep.main(argv, runner=runner, log=logs.append)
        return code, logs


# --------------------------------------------------------------------------
# Config parsing / spec validation
# --------------------------------------------------------------------------

class TestParseConfigs(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(sweep.parse_configs("[[0,10],[0.8,6],[1.2,4]]"),
                         [(0.0, 10.0), (0.8, 6.0), (1.2, 4.0)])

    def test_invalid_json(self):
        with self.assertRaises(sweep.SweepError):
            sweep.parse_configs("not json")

    def test_not_pairs(self):
        for bad in ("[]", "[[1]]", "[[1,2,3]]", '[["a",2]]', "[[true,2]]", "[1,2]"):
            with self.assertRaises(sweep.SweepError):
                sweep.parse_configs(bad)

    def test_degenerate_clock_rejected(self):
        with self.assertRaises(sweep.SweepError):
            sweep.parse_configs("[[0,0]]")
        with self.assertRaises(sweep.SweepError):
            sweep.parse_configs("[[-1,10]]")


class TestValidateSpec(unittest.TestCase):
    def test_real_spec_is_valid(self):
        raw = sweep.validate_spec()  # the committed gpu/specs/sweep-train.json
        self.assertEqual(raw["name"], "sweep-train")
        self.assertIn(sweep.STAGE_HEADER_REL, raw["upload_extra"])
        self.assertIn(sweep.STAGE_WARMSTART_REL, raw["upload_extra"])
        self.assertIn(sweep.REMOTE_HEADER_DEST, raw["jobs"][0]["command"])
        self.assertEqual(raw["resources"]["ttl_minutes"], 25)
        train = [j for j in raw["jobs"] if j["name"] == "train"][0]
        self.assertIn("--max-wall-s 780", train["command"])
        self.assertIn("--lane-env", train["command"])
        self.assertIn(f"--resume {sweep.STAGE_WARMSTART_REL}", train["command"])

    def test_missing_header_injection_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "spec.json"
            bad.write_text(json.dumps({
                "name": "x",
                "upload_extra": [sweep.STAGE_HEADER_REL, sweep.STAGE_WARMSTART_REL],
                "jobs": [{"name": "build", "command": "make"}],
            }))
            with self.assertRaises(sweep.SweepError):
                sweep.validate_spec(bad)


# --------------------------------------------------------------------------
# Header regeneration
# --------------------------------------------------------------------------

class TestRegenerateHeader(SweepTestCase):
    def test_called_with_right_env_and_interpreter(self):
        runner = FakeRunner(sweep)
        out = self.tmp / "runs" / "sweep-tmp" / "duck_model.h"
        sweep.regenerate_header(0.8, 6.0, out, runner=runner, log=lambda *_: None)
        (kind, cmd, kw), = runner.calls
        self.assertEqual(kind, "generate")
        self.assertEqual(cmd[0], str(sweep.VENV_PY))
        self.assertEqual(cmd[1], "-B")
        self.assertIn("generate_model.py", cmd[2])
        self.assertEqual(cmd[cmd.index("--output") + 1], str(out))
        self.assertEqual(kw["env"][sweep.ENV_BASE], "0.8")
        self.assertEqual(kw["env"][sweep.ENV_PER_MPS], "6.0")
        text = out.read_text()
        self.assertIn("#define DW_PHASE_HZ_BASE 0.8", text)
        self.assertIn("#define DW_PHASE_HZ_PER_MPS 6.0", text)

    def test_generator_failure_raises(self):
        runner = FakeRunner(sweep, generator_exit=7)
        with self.assertRaises(sweep.SweepError):
            sweep.regenerate_header(0.8, 6.0, self.tmp / "h.h",
                                    runner=runner, log=lambda *_: None)

    def test_baked_value_mismatch_raises(self):
        runner = FakeRunner(sweep, wrong_header=True)
        with self.assertRaises(sweep.SweepError):
            sweep.regenerate_header(0.8, 6.0, self.tmp / "h.h",
                                    runner=runner, log=lambda *_: None)


# --------------------------------------------------------------------------
# Metrics and ranking
# --------------------------------------------------------------------------

class TestMetrics(unittest.TestCase):
    def test_longest_alternating(self):
        la = sweep.longest_alternating
        self.assertEqual(la([]), 0)
        self.assertEqual(la(["left"]), 1)
        self.assertEqual(la(["left", "left"]), 1)
        self.assertEqual(la(["left", "right", "left", "left", "right"]), 3)
        self.assertEqual(la(["left", "right"] * 4), 8)

    def test_summarize_eval(self):
        m = sweep.summarize_eval(DEFAULT_EPISODES)
        self.assertEqual(m["commands_passed"], 2)
        self.assertEqual(m["qualified_left"], 3 + 3 + 2)
        self.assertEqual(m["qualified_right"], 3 + 3 + 1)
        self.assertEqual(m["total_qualified"], 15)
        self.assertEqual(m["alt_sum"], 6 + 6 + 2)
        self.assertTrue(m["per_command"]["0.10"]["passed"])
        self.assertFalse(m["per_command"]["0.20"]["passed"])
        self.assertEqual(m["per_command"]["0.20"]["longest_alternating"], 2)

    def test_rank_results(self):
        def rec(tag, status, passed=0, alt=0):
            metrics = ({"commands_passed": passed, "alt_sum": alt}
                       if status == "ok" else None)
            return {"tag": tag, "status": status, "metrics": metrics}
        records = [rec("worst", "ok", 1, 4), rec("failed", "leg_failed"),
                   rec("best", "ok", 3, 10), rec("tiebreak", "ok", 3, 18)]
        ranked = sweep.rank_results(records)
        self.assertEqual([r["tag"] for r in ranked],
                         ["tiebreak", "best", "worst", "failed"])


# --------------------------------------------------------------------------
# End-to-end main() with the fake runner
# --------------------------------------------------------------------------

class TestMainFlow(SweepTestCase):
    def test_config_iteration_and_success(self):
        runner = FakeRunner(sweep, launch_exits=[0, 0])
        code, logs = self.run_main(runner)
        self.assertEqual(code, 0)
        # one header regeneration per config, with the config's env vars
        gens = runner.of("generate")
        self.assertEqual(len(gens), 2)
        self.assertEqual([g[2]["env"][sweep.ENV_BASE] for g in gens],
                         ["0.8", "1.2"])
        self.assertEqual([g[2]["env"][sweep.ENV_PER_MPS] for g in gens],
                         ["6.0", "4.0"])
        # one launch per config: same interpreter, inherited environment
        launches = runner.of("launch")
        self.assertEqual(len(launches), 2)
        for _, cmd, kw in launches:
            self.assertEqual(cmd[0], sys.executable)
            self.assertEqual(cmd[1], "-B")
            self.assertIn(str(sweep.SPEC_PATH), cmd)
            self.assertNotIn("env", kw)   # inherited env, never rebuilt/printed
        # eval ran per config with the config env
        evals = runner.of("eval")
        self.assertEqual(len(evals), 2)
        self.assertEqual(evals[0][2]["env"][sweep.ENV_BASE], "0.8")
        self.assertEqual(evals[1][2]["env"][sweep.ENV_PER_MPS], "4.0")
        # results jsonl: one line per config, both ok
        lines = [json.loads(x) for x in
                 (self.out / "sweep-results.jsonl").read_text().splitlines()]
        self.assertEqual([tuple(r["config"]) for r in lines],
                         [(0.8, 6.0), (1.2, 4.0)])
        self.assertEqual([r["status"] for r in lines], ["ok", "ok"])
        self.assertEqual(lines[0]["metrics"]["commands_passed"], 2)
        # warmstart was staged at the fixed spec-relative path
        self.assertEqual((self.tmp / sweep.STAGE_WARMSTART_REL).read_bytes(),
                         b"fake-checkpoint")

    def test_one_config_fails_sweep_continues_exit0(self):
        # first leg: spot capacity (exit 3, no run dir); second: fine
        runner = FakeRunner(sweep, launch_exits=[3, 0])
        code, logs = self.run_main(runner)
        self.assertEqual(code, 0)
        lines = [json.loads(x) for x in
                 (self.out / "sweep-results.jsonl").read_text().splitlines()]
        self.assertEqual([r["status"] for r in lines], ["leg_failed", "ok"])
        self.assertEqual(lines[0]["legs"][0]["launch_exit"], 3)
        self.assertEqual(len(runner.of("eval")), 1)  # failed config never evaled

    def test_job_failure_exit2_recorded_and_continues(self):
        runner = FakeRunner(sweep, launch_exits=[2, 0])
        code, _ = self.run_main(runner)
        self.assertEqual(code, 0)
        lines = [json.loads(x) for x in
                 (self.out / "sweep-results.jsonl").read_text().splitlines()]
        self.assertEqual(lines[0]["status"], "leg_failed")
        self.assertEqual(lines[0]["legs"][0]["launch_exit"], 2)
        self.assertIn("error", lines[0]["legs"][0])

    def test_all_configs_fail_exit1(self):
        runner = FakeRunner(sweep, launch_exits=[3, 3])
        code, _ = self.run_main(runner)
        self.assertEqual(code, 1)

    def test_missing_warmstart_is_usage_error(self):
        runner = FakeRunner(sweep)
        code = sweep.main(["--configs", "[[0,10]]",
                           "--warmstart", str(self.tmp / "nope.pt"),
                           "--out", str(self.out)],
                          runner=runner, log=lambda *_: None)
        self.assertEqual(code, 1)
        self.assertEqual(runner.calls, [])

    def test_legs_chain_resume_from_latest(self):
        runner = FakeRunner(sweep, launch_exits=[0, 0])
        code, _ = self.run_main(runner, configs="[[0.8,6]]",
                                extra=["--legs-per-config", "2"])
        self.assertEqual(code, 0)
        rec = json.loads((self.out / "sweep-results.jsonl").read_text())
        self.assertEqual(len(rec["legs"]), 2)
        self.assertEqual(rec["legs"][0]["resume"], str(self.warmstart))
        self.assertTrue(rec["legs"][1]["resume"].endswith(str(sweep.LATEST_REL)))
        self.assertEqual(len(runner.of("eval")), 1)  # eval only the final actor

    def test_dry_run_skips_launch_and_eval(self):
        runner = FakeRunner(sweep)
        code, _ = self.run_main(runner, extra=["--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(len(runner.of("generate")), 2)
        launches = runner.of("launch")
        self.assertEqual(len(launches), 2)
        for _, cmd, _kw in launches:
            self.assertIn("--dry-run", cmd)
        self.assertEqual(runner.of("eval"), [])
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["mode"], "dry-run")
        self.assertEqual([r["status"] for r in summary["ranked"]],
                         ["dry_run_ok", "dry_run_ok"])


class TestSummaryFormat(SweepTestCase):
    def test_summary_and_ranking(self):
        # config 1 passes 2 commands, config 2 passes 3 -> config 2 ranks first
        strong = [make_episode(c, True, ["left", "right"] * 4)
                  for c in (0.10, 0.15, 0.20)]

        class TwoEvalRunner(FakeRunner):
            def __call__(self, cmd, **kw):
                if "-c" in [str(c) for c in cmd] and len(self.of("eval")) == 1:
                    self.episodes = strong
                return super().__call__(cmd, **kw)

        runner = TwoEvalRunner(sweep, launch_exits=[0, 0])
        code, _ = self.run_main(runner)
        self.assertEqual(code, 0)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["schema"], "duckgridwalk.sweep_summary/1")
        self.assertEqual(summary["configs_total"], 2)
        self.assertEqual(summary["configs_completed"], 2)
        self.assertEqual(summary["rank_key"], ["commands_passed", "alt_sum"])
        ranked = summary["ranked"]
        self.assertEqual([tuple(r["config"]) for r in ranked],
                         [(1.2, 4.0), (0.8, 6.0)])   # 3 passes beats 2
        self.assertEqual(ranked[0]["metrics"]["commands_passed"], 3)
        self.assertEqual(ranked[0]["metrics"]["alt_sum"], 24)
        for rec in ranked:
            for key in ("config", "tag", "status", "legs", "metrics", "at"):
                self.assertIn(key, rec)
        self.assertTrue(Path(summary["results_jsonl"]).is_file())


if __name__ == "__main__":
    unittest.main()
