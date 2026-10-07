"""Unit tests for gpu/sweep.py with the launcher / generator / eval
subprocesses replaced by an in-memory fake runner (no provider calls, no GPU,
no doppler, no DAYTONA_API_KEY needed), plus tests for run_daytona.py's
--label-suffix flag (fake provider, no network).

Run:  .venv/bin/python -m unittest discover -s gpu/tests -v
"""

import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))        # gpu/tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # gpu

import run_daytona as rd  # noqa: E402
import sweep  # noqa: E402
from test_launcher import FAKE_SECRET, FakeProvider, make_git_repo  # noqa: E402


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
    """Stands in for subprocess.run for all three sweep subprocesses.
    Thread-safe: --parallel N drives it from N worker threads."""

    def __init__(self, sweep_mod, launch_exits=(0,), episodes=None,
                 generator_exit=0, eval_exit=0, wrong_header=False,
                 exits_by_suffix=None):
        self.sweep = sweep_mod
        self.launch_exits = list(launch_exits)
        self.exits_by_suffix = dict(exits_by_suffix or {})
        self.episodes = DEFAULT_EPISODES if episodes is None else episodes
        self.generator_exit = generator_exit
        self.eval_exit = eval_exit
        self.wrong_header = wrong_header
        self.calls = []          # (kind, cmd, kwargs)
        self._stamp = 0
        self._lock = threading.Lock()

    def _arg(self, cmd_s, flag, default=None):
        return cmd_s[cmd_s.index(flag) + 1] if flag in cmd_s else default

    def __call__(self, cmd, **kw):
        cmd_s = [str(c) for c in cmd]
        if any("generate_model.py" in c for c in cmd_s):
            with self._lock:
                self.calls.append(("generate", cmd_s, kw))
            if self.generator_exit:
                return SimpleNamespace(returncode=self.generator_exit)
            out = Path(self._arg(cmd_s, "--output"))
            env = kw["env"]
            base = "9999.0" if self.wrong_header else env[sweep.ENV_BASE]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                "// fake generated header\n"
                f"#define DW_PHASE_HZ_BASE {base}\n"
                f"#define DW_PHASE_HZ_PER_MPS {env[sweep.ENV_PER_MPS]}\n")
            return SimpleNamespace(returncode=0)
        if any("run_daytona.py" in c for c in cmd_s):
            spec_path = Path(self._arg(cmd_s, "--spec"))
            spec_name = json.loads(spec_path.read_text())["name"]
            suffix = self._arg(cmd_s, "--label-suffix")
            runs_dir = Path(self._arg(cmd_s, "--runs-dir",
                                      self.sweep.RUNS_GPU_DIR))
            with self._lock:
                self.calls.append(("launch", cmd_s, kw))
                if suffix in self.exits_by_suffix:
                    rc = self.exits_by_suffix[suffix]
                elif self.launch_exits:
                    rc = self.launch_exits.pop(0)
                else:
                    rc = 0
                self._stamp += 1
                stamp = self._stamp
            if "--dry-run" not in cmd_s and rc in (0, 2):
                # exit 2 = job failed: run dir exists but (here) no artifacts
                run_dir = runs_dir / f"20260831-{stamp:06d}-{spec_name}"
                run_dir.mkdir(parents=True)
                if rc == 0:
                    actor = run_dir / self.sweep.ACTOR_REL
                    actor.parent.mkdir(parents=True, exist_ok=True)
                    actor.write_bytes(b"fake-actor")
                    (run_dir / self.sweep.LATEST_REL).write_bytes(b"fake-latest")
            return SimpleNamespace(returncode=rc)
        if "-c" in cmd_s:
            with self._lock:
                self.calls.append(("eval", cmd_s, kw))
            if self.eval_exit:
                return SimpleNamespace(returncode=self.eval_exit)
            out = Path(cmd_s[cmd_s.index("-c") + 4])   # -c SCRIPT actor lib OUT
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(self.episodes))
            return SimpleNamespace(returncode=0)
        raise AssertionError(f"unexpected command: {cmd_s}")

    def of(self, kind):
        with self._lock:
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

    def run_param_main(self, runner, param_configs, warmstart=False, extra=()):
        argv = ["--param-configs", param_configs, "--out", str(self.out), *extra]
        if warmstart:
            argv += ["--warmstart", str(self.warmstart)]
        logs = []
        code = sweep.main(argv, runner=runner, log=logs.append)
        return code, logs

    def results(self):
        return [json.loads(x) for x in
                (self.out / "sweep-results.jsonl").read_text().splitlines()]


# --------------------------------------------------------------------------
# Config parsing / spec validation / spec derivation
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
        build = [j for j in raw["jobs"] if j["name"] == "build"][0]
        self.assertIn("SWEEP_FAST_BUILD", build["command"])
        self.assertIn("libduck_cuda_fast.so", build["command"])
        self.assertIn("./build_remote.sh", build["command"])  # default path
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


class TestDeriveSpec(unittest.TestCase):
    def setUp(self):
        self.base = sweep.validate_spec()

    def job(self, raw, name):
        return [j for j in raw["jobs"] if j["name"] == name][0]

    def test_sequential_default_is_identity_content(self):
        raw = sweep.derive_spec(self.base)
        self.assertEqual(raw, self.base)   # same name, paths, wall-s, build

    def test_per_config_paths_and_name(self):
        raw = sweep.derive_spec(self.base, cfg_index=2)
        self.assertEqual(raw["name"], "sweep-train-cfg2")
        self.assertEqual(raw["upload_extra"],
                         ["runs/sweep-tmp-2/duck_model.h",
                          "runs/sweep-tmp-2/warmstart.pt"])
        self.assertIn("runs/sweep-tmp-2/duck_model.h",
                      self.job(raw, "inject-header")["command"])
        self.assertIn("--resume runs/sweep-tmp-2/warmstart.pt",
                      self.job(raw, "train")["command"])
        self.assertNotIn("runs/sweep-tmp/", json.dumps(raw))

    def test_train_wall_s_flows_into_command_and_timeout(self):
        raw = sweep.derive_spec(self.base, train_wall_s=300)
        train = self.job(raw, "train")
        self.assertIn("--max-wall-s 300", train["command"])
        self.assertNotIn("--max-wall-s 780", train["command"])
        self.assertEqual(train["timeout_s"], 300 + sweep.TRAIN_TIMEOUT_SLACK_S)

    def test_train_wall_s_bounds(self):
        for bad in (10, 4000):
            with self.assertRaises(sweep.SweepError):
                sweep.derive_spec(self.base, train_wall_s=bad)

    def test_fast_build_prefix(self):
        raw = sweep.derive_spec(self.base, fast_build=True)
        build = self.job(raw, "build")
        self.assertTrue(build["command"].startswith("export SWEEP_FAST_BUILD=1; "))
        # default derivation leaves the build command untouched
        self.assertNotIn("export SWEEP_FAST_BUILD",
                         self.job(sweep.derive_spec(self.base), "build")["command"])


class TestParseParamConfigs(unittest.TestCase):
    def test_valid(self):
        got = sweep.parse_param_configs(
            '[{"gamma":0.995},{"gamma":0.99,"lr":1.5e-4},'
            '{"horizon":128,"envs":2048,"policy":"gru"}]')
        self.assertEqual(got, [{"gamma": 0.995}, {"gamma": 0.99, "lr": 1.5e-4},
                               {"horizon": 128, "envs": 2048, "policy": "gru"}])

    def test_unknown_key_rejected_with_allowlist(self):
        with self.assertRaises(sweep.SweepError) as ctx:
            sweep.parse_param_configs('[{"batchsize":64}]')
        self.assertIn("gamma", str(ctx.exception))   # message names the allowlist

    def test_int_keys_enforced(self):
        for bad in ('[{"horizon":64.5}]', '[{"envs":"many"}]',
                    '[{"horizon":0}]', '[{"envs":true}]'):
            with self.assertRaises(sweep.SweepError):
                sweep.parse_param_configs(bad)

    def test_policy_values(self):
        self.assertEqual(sweep.parse_param_configs('[{"policy":"ff"}]'),
                         [{"policy": "ff"}])
        with self.assertRaises(sweep.SweepError):
            sweep.parse_param_configs('[{"policy":"lstm"}]')

    def test_shape_errors(self):
        for bad in ("nope", "[]", "[{}]", "[[1,2]]", '[{"gamma":-0.5}]',
                    '[{"lr":true}]'):
            with self.assertRaises(sweep.SweepError):
                sweep.parse_param_configs(bad)

    def test_param_flags_and_tag(self):
        params = {"gamma": 0.995, "lr": 3e-4, "envs": 2048, "policy": "ff"}
        self.assertEqual(sweep.param_flags(params),
                         "--gamma 0.995 --lr 0.0003 --envs 2048 --policy ff")
        self.assertEqual(sweep.param_tag(1, {"gamma": 0.99, "gae_lambda": 0.95}),
                         "p1-gamma0.99-gae_lambda0.95")


class TestDeriveParamSpec(unittest.TestCase):
    def setUp(self):
        self.base = sweep.validate_spec()

    def job(self, raw, name):
        matches = [j for j in raw["jobs"] if j["name"] == name]
        return matches[0] if matches else None

    def derive(self, **kw):
        kw.setdefault("params", {"gamma": 0.995, "lr": 3e-4})
        kw.setdefault("keep_header", False)
        kw.setdefault("keep_warmstart", False)
        kw.setdefault("name_tag", "pcfg")
        return sweep.derive_spec(self.base, **kw)

    def test_flags_appended_before_redirection_tail(self):
        cmd = self.job(self.derive(), "train")["command"]
        self.assertIn("--gamma 0.995 --lr 0.0003", cmd)
        self.assertLess(cmd.index("--gamma 0.995"), cmd.index(" 2>&1"))
        self.assertTrue(cmd.rstrip().endswith("| tee gpu-train.log"))

    def test_fresh_init_strips_resume_and_warmstart_upload(self):
        raw = self.derive()
        self.assertNotIn("--resume", self.job(raw, "train")["command"])
        self.assertEqual(raw["upload_extra"], [])

    def test_header_job_and_upload_removed(self):
        raw = self.derive()
        self.assertIsNone(self.job(raw, "inject-header"))
        self.assertNotIn("duck_model.h", json.dumps(raw))

    def test_warmstart_kept_when_requested(self):
        raw = self.derive(keep_warmstart=True)
        self.assertIn("--resume runs/sweep-tmp/warmstart.pt",
                      self.job(raw, "train")["command"])
        self.assertEqual(raw["upload_extra"], ["runs/sweep-tmp/warmstart.pt"])

    def test_parallel_naming_and_paths(self):
        raw = self.derive(cfg_index=2, keep_warmstart=True)
        self.assertEqual(raw["name"], "sweep-train-pcfg2")
        self.assertEqual(raw["upload_extra"], ["runs/sweep-tmp-2/warmstart.pt"])
        self.assertIn("--resume runs/sweep-tmp-2/warmstart.pt",
                      self.job(raw, "train")["command"])


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
# End-to-end main() with the fake runner (sequential)
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
        # one launch per config: same interpreter, inherited environment,
        # derived spec under <out>/specs, no label suffix when sequential
        launches = runner.of("launch")
        self.assertEqual(len(launches), 2)
        for _, cmd, kw in launches:
            self.assertEqual(cmd[0], sys.executable)
            self.assertEqual(cmd[1], "-B")
            spec_arg = Path(cmd[cmd.index("--spec") + 1])
            self.assertEqual(spec_arg.parent, self.out / "specs")
            self.assertEqual(json.loads(spec_arg.read_text())["name"],
                             "sweep-train")
            self.assertNotIn("--label-suffix", cmd)
            self.assertNotIn("env", kw)   # inherited env, never rebuilt/printed
        # eval ran per config with the config env
        evals = runner.of("eval")
        self.assertEqual(len(evals), 2)
        self.assertEqual(evals[0][2]["env"][sweep.ENV_BASE], "0.8")
        self.assertEqual(evals[1][2]["env"][sweep.ENV_PER_MPS], "4.0")
        # results jsonl: one line per config, both ok
        lines = self.results()
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
        lines = self.results()
        self.assertEqual([r["status"] for r in lines], ["leg_failed", "ok"])
        self.assertEqual(lines[0]["legs"][0]["launch_exit"], 3)
        self.assertEqual(len(runner.of("eval")), 1)  # failed config never evaled

    def test_job_failure_exit2_recorded_and_continues(self):
        runner = FakeRunner(sweep, launch_exits=[2, 0])
        code, _ = self.run_main(runner)
        self.assertEqual(code, 0)
        lines = self.results()
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
        rec = self.results()[0]
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

    def test_train_wall_s_flows_into_written_spec(self):
        runner = FakeRunner(sweep, launch_exits=[0])
        code, _ = self.run_main(runner, configs="[[0.8,6]]",
                                extra=["--train-wall-s", "300"])
        self.assertEqual(code, 0)
        _, cmd, _ = runner.of("launch")[0]
        raw = json.loads(Path(cmd[cmd.index("--spec") + 1]).read_text())
        train = [j for j in raw["jobs"] if j["name"] == "train"][0]
        self.assertIn("--max-wall-s 300", train["command"])
        self.assertEqual(train["timeout_s"], 300 + sweep.TRAIN_TIMEOUT_SLACK_S)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["train_wall_s"], 300)

    def test_fast_build_flows_into_written_spec(self):
        runner = FakeRunner(sweep, launch_exits=[0])
        code, _ = self.run_main(runner, configs="[[0.8,6]]",
                                extra=["--fast-build"])
        self.assertEqual(code, 0)
        _, cmd, _ = runner.of("launch")[0]
        raw = json.loads(Path(cmd[cmd.index("--spec") + 1]).read_text())
        build = [j for j in raw["jobs"] if j["name"] == "build"][0]
        self.assertTrue(build["command"].startswith("export SWEEP_FAST_BUILD=1; "))

    def test_runs_dir_passthrough(self):
        runner = FakeRunner(sweep, launch_exits=[0])
        code, _ = self.run_main(runner, configs="[[0.8,6]]",
                                extra=["--runs-dir", "runs/scratch/gpu"])
        self.assertEqual(code, 0)
        _, cmd, _ = runner.of("launch")[0]
        self.assertEqual(cmd[cmd.index("--runs-dir") + 1],
                         str(self.tmp / "runs/scratch/gpu"))
        self.assertEqual(self.results()[0]["status"], "ok")


# --------------------------------------------------------------------------
# Parallel mode
# --------------------------------------------------------------------------

class TestParallel(SweepTestCase):
    CONFIGS = "[[0,10],[0.8,6],[1.2,4]]"

    def test_three_configs_complete_concurrently(self):
        runner = FakeRunner(sweep)
        code, _ = self.run_main(runner, configs=self.CONFIGS,
                                extra=["--parallel", "3"])
        self.assertEqual(code, 0)
        launches = runner.of("launch")
        self.assertEqual(len(launches), 3)
        # unique label suffixes, one per config
        suffixes = {c[1][c[1].index("--label-suffix") + 1] for c in launches}
        self.assertEqual(suffixes, {"cfg0", "cfg1", "cfg2"})
        # unique derived specs with per-config names and staging paths
        spec_names, stage_dirs = set(), set()
        for _, cmd, _kw in launches:
            raw = json.loads(Path(cmd[cmd.index("--spec") + 1]).read_text())
            spec_names.add(raw["name"])
            stage_dirs.update(Path(x).parent.name for x in raw["upload_extra"])
        self.assertEqual(spec_names,
                         {"sweep-train-cfg0", "sweep-train-cfg1", "sweep-train-cfg2"})
        self.assertEqual(stage_dirs,
                         {"sweep-tmp-0", "sweep-tmp-1", "sweep-tmp-2"})
        # per-config header staged in per-config dirs, baked per config
        for i, base in enumerate(("0.0", "0.8", "1.2")):
            text = (self.tmp / f"runs/sweep-tmp-{i}/duck_model.h").read_text()
            self.assertIn(f"#define DW_PHASE_HZ_BASE {base}", text)
            self.assertTrue(
                (self.tmp / f"runs/sweep-tmp-{i}/warmstart.pt").is_file())
        # all three completed; jsonl has 3 rows (completion order), summary ok
        lines = self.results()
        self.assertEqual(len(lines), 3)
        self.assertEqual({r["status"] for r in lines}, {"ok"})
        self.assertEqual({r["label_suffix"] for r in lines},
                         {"cfg0", "cfg1", "cfg2"})
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["parallel"], 3)
        self.assertEqual(summary["configs_completed"], 3)

    def test_one_parallel_config_failing_does_not_sink_others(self):
        runner = FakeRunner(sweep, exits_by_suffix={"cfg1": 3})
        code, _ = self.run_main(runner, configs=self.CONFIGS,
                                extra=["--parallel", "3"])
        self.assertEqual(code, 0)
        by_suffix = {r["label_suffix"]: r for r in self.results()}
        self.assertEqual(by_suffix["cfg1"]["status"], "leg_failed")
        self.assertEqual(by_suffix["cfg1"]["legs"][0]["launch_exit"], 3)
        self.assertEqual(by_suffix["cfg0"]["status"], "ok")
        self.assertEqual(by_suffix["cfg2"]["status"], "ok")
        self.assertEqual(len(runner.of("eval")), 2)

    def test_parallel_dry_run(self):
        runner = FakeRunner(sweep)
        code, _ = self.run_main(runner, configs=self.CONFIGS,
                                extra=["--parallel", "3", "--dry-run"])
        self.assertEqual(code, 0)
        for _, cmd, _kw in runner.of("launch"):
            self.assertIn("--dry-run", cmd)
            self.assertIn("--label-suffix", cmd)
        self.assertEqual(runner.of("eval"), [])

    def test_parallel_must_be_positive(self):
        runner = FakeRunner(sweep)
        code, _ = self.run_main(runner, extra=["--parallel", "0"])
        self.assertEqual(code, 1)
        self.assertEqual(runner.calls, [])


class TestParamSweep(SweepTestCase):
    PCONFIGS = '[{"gamma":0.995},{"gamma":0.99,"lr":1.5e-4}]'

    def test_combined_configs_and_param_configs_rejected(self):
        runner = FakeRunner(sweep)
        code, logs = self.run_main(
            runner, extra=["--param-configs", self.PCONFIGS])
        self.assertEqual(code, 1)
        self.assertTrue(any("mutually exclusive" in line for line in logs))
        self.assertEqual(runner.calls, [])

    def test_neither_config_flag_rejected(self):
        code = sweep.main(["--out", str(self.out)],
                          runner=FakeRunner(sweep), log=lambda *_: None)
        self.assertEqual(code, 1)

    def test_param_sweep_fresh_init_default_clock_eval(self):
        runner = FakeRunner(sweep)
        code, _ = self.run_param_main(runner, self.PCONFIGS)
        self.assertEqual(code, 0)
        # no header regeneration in param mode
        self.assertEqual(runner.of("generate"), [])
        # derived specs: params appended, no resume, no uploads, no header job
        launches = runner.of("launch")
        self.assertEqual(len(launches), 2)
        raws = []
        for _, cmd, _kw in launches:
            raws.append(json.loads(Path(cmd[cmd.index("--spec") + 1]).read_text()))
        train0 = [j for j in raws[0]["jobs"] if j["name"] == "train"][0]
        train1 = [j for j in raws[1]["jobs"] if j["name"] == "train"][0]
        self.assertIn("--gamma 0.995", train0["command"])
        self.assertIn("--gamma 0.99 --lr 0.00015", train1["command"])
        for raw, train in ((raws[0], train0), (raws[1], train1)):
            self.assertNotIn("--resume", train["command"])
            self.assertEqual(raw["upload_extra"], [])
            self.assertNotIn("inject-header", [j["name"] for j in raw["jobs"]])
        # nothing staged (fresh init, no header)
        self.assertFalse((self.tmp / "runs" / "sweep-tmp").exists())
        # eval ran under the DEFAULT clock: no DUCK overrides in its env
        evals = runner.of("eval")
        self.assertEqual(len(evals), 2)
        for _, _cmd, kw in evals:
            self.assertNotIn(sweep.ENV_BASE, kw["env"])
            self.assertNotIn(sweep.ENV_PER_MPS, kw["env"])
        # records carry the params dict as the config field
        lines = self.results()
        self.assertEqual(lines[0]["config"], {"gamma": 0.995})
        self.assertEqual(lines[1]["config"], {"gamma": 0.99, "lr": 1.5e-4})
        self.assertEqual([r["status"] for r in lines], ["ok", "ok"])
        self.assertEqual(lines[0]["legs"][0]["resume"], None)

    def test_param_sweep_honors_warmstart(self):
        runner = FakeRunner(sweep)
        code, _ = self.run_param_main(runner, '[{"gamma":0.995}]', warmstart=True)
        self.assertEqual(code, 0)
        _, cmd, _ = runner.of("launch")[0]
        raw = json.loads(Path(cmd[cmd.index("--spec") + 1]).read_text())
        train = [j for j in raw["jobs"] if j["name"] == "train"][0]
        self.assertIn("--resume runs/sweep-tmp/warmstart.pt", train["command"])
        self.assertEqual(raw["upload_extra"], ["runs/sweep-tmp/warmstart.pt"])
        self.assertEqual((self.tmp / "runs/sweep-tmp/warmstart.pt").read_bytes(),
                         b"fake-checkpoint")

    def test_fresh_init_multi_leg_rejected(self):
        runner = FakeRunner(sweep)
        code, logs = self.run_param_main(runner, '[{"gamma":0.995}]',
                                         extra=["--legs-per-config", "2"])
        self.assertEqual(code, 1)
        self.assertTrue(any("--warmstart" in line for line in logs))
        self.assertEqual(runner.calls, [])

    def test_parallel_param_sweep_uses_pcfg_labels(self):
        runner = FakeRunner(sweep)
        code, _ = self.run_param_main(
            runner, '[{"gamma":0.995},{"gamma":0.99},{"gamma":0.97}]',
            extra=["--parallel", "3"])
        self.assertEqual(code, 0)
        launches = runner.of("launch")
        suffixes = {c[1][c[1].index("--label-suffix") + 1] for c in launches}
        self.assertEqual(suffixes, {"pcfg0", "pcfg1", "pcfg2"})
        names = set()
        for _, cmd, _kw in launches:
            names.add(json.loads(Path(cmd[cmd.index("--spec") + 1])
                                 .read_text())["name"])
        self.assertEqual(names, {"sweep-train-pcfg0", "sweep-train-pcfg1",
                                 "sweep-train-pcfg2"})
        self.assertEqual({r["status"] for r in self.results()}, {"ok"})

    def test_one_param_config_failing_continues(self):
        runner = FakeRunner(sweep, launch_exits=[3, 0])
        code, _ = self.run_param_main(runner, self.PCONFIGS)
        self.assertEqual(code, 0)
        lines = self.results()
        self.assertEqual([r["status"] for r in lines], ["leg_failed", "ok"])


class TestGruEvalScript(unittest.TestCase):
    """Run the inline eval script in-process with the env/capture/gait modules
    mocked and a REAL RecurrentActor checkpoint: proves the unpack_actor_file
    + gru hidden-state policy path end to end (no native lib, no sandbox)."""

    def test_gru_actor_eval_path(self):
        import importlib
        import numpy as np
        torch = importlib.import_module("torch")
        ppo = importlib.import_module("walk.train.ppo")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            actor = ppo.RecurrentActor(58, 14)
            actor_path = d / "actor_final.pt"
            torch.save({"arch": "gru", "state_dict": actor.state_dict()},
                       actor_path)
            policies, steps = [], []

            class FakeEnv:
                def __init__(self, environments=1, seed=0, library_path=None):
                    self.E = environments
                def close(self):
                    pass

            def fake_capture(env, policy, command=None, seconds=8.0, seed=None):
                policies.append(policy)
                for _ in range(3):  # hidden state must carry within an episode
                    a = policy(np.zeros((1, 58), dtype=np.float32))
                    steps.append(a.shape)
                return [{"command_mps": command}]

            def fake_eval(trace):
                return {"command_mps": trace["command_mps"], "passed": True,
                        "footfalls": []}

            mods = {
                "walk.env.contract": SimpleNamespace(OBS=58, ACT=14),
                "walk.env.flat": SimpleNamespace(FlatFloorDuckEnv=FakeEnv),
                "walk.eval.capture": SimpleNamespace(capture_episodes=fake_capture),
                "walk.eval.gait": SimpleNamespace(evaluate_episode=fake_eval),
            }
            out_json = d / "eval.json"
            argv = ["eval", str(actor_path), "libfake.dylib", str(out_json),
                    "4242", "0.1,0.15,0.2", "8.0"]
            with mock.patch.dict(sys.modules, mods), \
                    mock.patch.object(sys, "argv", argv):
                exec(compile(sweep.EVAL_SCRIPT, "<eval-script>", "exec"),
                     {"__name__": "__main__"})
            episodes = json.loads(out_json.read_text())
            self.assertEqual([e["command_mps"] for e in episodes],
                             [0.1, 0.15, 0.2])
            self.assertEqual(steps, [(1, 14)] * 9)
            # fresh policy (fresh gru hidden state) per commanded episode
            self.assertEqual(len(set(map(id, policies))), 3)

    def test_legacy_ff_state_dict_still_loads(self):
        import importlib
        import numpy as np
        torch = importlib.import_module("torch")
        ppo = importlib.import_module("walk.train.ppo")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            actor_path = d / "actor_final.pt"
            torch.save(ppo.Actor(58, 14).state_dict(), actor_path)  # legacy
            outputs = []

            class FakeEnv:
                def __init__(self, environments=1, seed=0, library_path=None):
                    self.E = environments
                def close(self):
                    pass

            def fake_capture(env, policy, command=None, seconds=8.0, seed=None):
                outputs.append(policy(np.zeros((1, 58), dtype=np.float32)).shape)
                return [{"command_mps": command}]

            mods = {
                "walk.env.contract": SimpleNamespace(OBS=58, ACT=14),
                "walk.env.flat": SimpleNamespace(FlatFloorDuckEnv=FakeEnv),
                "walk.eval.capture": SimpleNamespace(capture_episodes=fake_capture),
                "walk.eval.gait": SimpleNamespace(
                    evaluate_episode=lambda t: {"command_mps": t["command_mps"]}),
            }
            out_json = d / "eval.json"
            argv = ["eval", str(actor_path), "lib", str(out_json),
                    "4242", "0.1", "8.0"]
            with mock.patch.dict(sys.modules, mods), \
                    mock.patch.object(sys, "argv", argv):
                exec(compile(sweep.EVAL_SCRIPT, "<eval-script>", "exec"),
                     {"__name__": "__main__"})
            self.assertEqual(outputs, [(1, 14)])
            self.assertEqual(len(json.loads(out_json.read_text())), 1)


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


# --------------------------------------------------------------------------
# run_daytona.py --label-suffix
# --------------------------------------------------------------------------

class TestLauncherLabelSuffix(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.logs = []

    def tearDown(self):
        self._tmp.cleanup()

    def log(self, *a):
        self.logs.append(" ".join(str(x) for x in a))

    def write_spec(self):
        spec_path = self.tmp / "spec.json"
        spec_path.write_text(json.dumps({
            "name": "t",
            "jobs": [{"name": "j", "command": "echo hi", "timeout_s": 5}],
        }))
        return spec_path

    def test_label_with_suffix(self):
        self.assertEqual(rd.label_with_suffix(None),
                         {"launcher": "duck-grid-walk-gpu"})
        self.assertEqual(rd.label_with_suffix("cfg2"),
                         {"launcher": "duck-grid-walk-gpu-cfg2"})
        self.assertIsNot(rd.label_with_suffix(None), rd.LAUNCHER_LABEL)

    def test_invalid_suffix_rejected_before_anything_runs(self):
        code = rd.main(["run", "--spec", str(self.write_spec()),
                        "--label-suffix", "bad slug!", "--dry-run"],
                       provider_factory=lambda: self.fail("factory called"),
                       repo_root=self.tmp, env={}, log=self.log)
        self.assertEqual(code, rd.EXIT_ERROR)
        self.assertTrue(any("--label-suffix" in line for line in self.logs))

    def _run_main(self, extra_args=()):
        repo = make_git_repo(self.tmp / "repo")
        fake = FakeProvider()
        env = dict(os.environ, DAYTONA_API_KEY=FAKE_SECRET)
        code = rd.main(
            ["run", "--spec", str(self.write_spec()),
             "--runs-dir", str(self.tmp / "runs"), *extra_args],
            provider_factory=lambda: fake, repo_root=repo, env=env, log=self.log)
        self.assertEqual(code, 0)
        manifest_path, = (self.tmp / "runs").rglob("manifest.json")
        return json.loads(manifest_path.read_text())

    def test_manifest_records_suffixed_label(self):
        m = self._run_main(["--label-suffix", "cfg2"])
        self.assertEqual(m["label"], {"launcher": "duck-grid-walk-gpu-cfg2"})

    def test_default_label_unchanged(self):
        m = self._run_main()
        self.assertEqual(m["label"], {"launcher": "duck-grid-walk-gpu"})


if __name__ == "__main__":
    unittest.main()
