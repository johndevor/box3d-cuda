"""Unit tests for detached (session-based) execution of long jobs and for
best-effort artifact salvage. No network, no daytona SDK, no real key.

Run:  .venv/bin/python -m unittest discover -s gpu/tests -v
"""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_daytona as rd  # noqa: E402
from test_launcher import FakeProvider, make_git_repo, make_spec  # noqa: E402

FAKE_SECRET = "dtn_FAKESECRET_abcdef1234567890"


class DetachedFakeProvider(FakeProvider):
    """FakeProvider plus the detached session-command surface.

    poll_script: sequence consumed one item per poll — an int (exit code),
    None (still running), or an Exception instance (raised, simulating a
    transient proxy failure). When exhausted, polls return 0.
    """

    def __init__(self, *args, poll_script=(None, 0), logs_output="detached out\n",
                 logs_exc_count=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.poll_script = list(poll_script)
        self.logs_output = logs_output
        self.logs_exc_count = logs_exc_count
        self.detached_commands = []
        self.poll_calls = 0
        self.log_fetches = 0

    def exec_detached_start(self, handle, command):
        self.detached_commands.append(command)
        return ("sess", f"cmd-{len(self.detached_commands)}")

    def exec_detached_poll(self, handle, ref):
        self.poll_calls += 1
        item = self.poll_script.pop(0) if self.poll_script else 0
        if isinstance(item, Exception):
            raise item
        return item

    def exec_detached_logs(self, handle, ref):
        self.log_fetches += 1
        if self.logs_exc_count > 0:
            self.logs_exc_count -= 1
            raise ConnectionError("proxy.app.daytona.io read timeout")
        return self.logs_output


class DetachedTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.repo = make_git_repo(self.tmp / "repo")
        self.run_dir = self.tmp / "run"
        self.logs = []
        self.sleeps = []

    def tearDown(self):
        self._tmp.cleanup()

    def log(self, *a):
        self.logs.append(" ".join(str(x) for x in a))

    def logged(self, needle):
        return any(needle in line for line in self.logs)

    def run_spec(self, spec, provider, deadline=None):
        return rd.execute_run(
            spec, self.repo, self.run_dir, provider,
            rd.Redactor([FAKE_SECRET]), deadline or rd.Deadline(),
            log=self.log, sleep_fn=self.sleeps.append,
        )

    def manifest(self):
        return json.loads((self.run_dir / "manifest.json").read_text())

    @staticmethod
    def long_job(name="train", timeout_s=1200, artifacts=(), **extra):
        j = {"name": name, "command": f"run-{name}", "timeout_s": timeout_s,
             "artifacts": list(artifacts)}
        j.update(extra)
        return j


class TestDetachedSelection(unittest.TestCase):
    def test_auto_detach_above_threshold(self):
        spec = make_spec(jobs=[{"name": "j", "command": "x",
                                "timeout_s": rd.DETACH_THRESHOLD_S + 1}])
        self.assertTrue(rd.job_is_detached(spec.jobs[0]))

    def test_no_auto_detach_at_threshold(self):
        spec = make_spec(jobs=[{"name": "j", "command": "x",
                                "timeout_s": rd.DETACH_THRESHOLD_S}])
        self.assertFalse(rd.job_is_detached(spec.jobs[0]))

    def test_explicit_true_wins_for_short_job(self):
        spec = make_spec(jobs=[{"name": "j", "command": "x", "timeout_s": 30,
                                "detached": True}])
        self.assertTrue(rd.job_is_detached(spec.jobs[0]))

    def test_explicit_false_wins_for_long_job(self):
        spec = make_spec(jobs=[{"name": "j", "command": "x", "timeout_s": 3000,
                                "detached": False}])
        self.assertFalse(rd.job_is_detached(spec.jobs[0]))

    def test_non_bool_detached_rejected(self):
        with self.assertRaises(rd.SpecError):
            make_spec(jobs=[{"name": "j", "command": "x", "timeout_s": 30,
                             "detached": "yes"}])

    def test_existing_train_specs_auto_detach_train_job(self):
        specs_dir = Path(__file__).resolve().parent.parent / "specs"
        for name in ("fresh-ff.json", "fresh-gru.json", "train-duck-gpu.json",
                     "sweep-train.json"):
            spec = rd.load_spec(specs_dir / name)
            train = next(j for j in spec.jobs if j.name == "train")
            build = next(j for j in spec.jobs if j.name == "build")
            self.assertTrue(rd.job_is_detached(train), name)
            self.assertFalse(rd.job_is_detached(build), name)


class TestDetachedFlow(DetachedTestCase):
    def test_detached_job_runs_via_session_not_exec(self):
        provider = DetachedFakeProvider(poll_script=[None, None, 0])
        spec = make_spec(jobs=[self.long_job()])
        code = self.run_spec(spec, provider)
        self.assertEqual(code, 0)
        self.assertEqual(len(provider.detached_commands), 1)
        self.assertIn("run-train", provider.detached_commands[0])
        self.assertIn(rd.REMOTE_WORKDIR, provider.detached_commands[0])
        # only the payload-unpack command went through blocking exec
        self.assertEqual(len(provider.commands), 1)
        self.assertIn("tar -xf", provider.commands[0])
        self.assertEqual(provider.poll_calls, 3)
        log_text = (self.run_dir / "logs" / "train.log").read_text()
        self.assertEqual(log_text, "detached out\n")

    def test_short_job_still_uses_blocking_exec(self):
        provider = DetachedFakeProvider()
        spec = make_spec(jobs=[{"name": "quick", "command": "echo hi",
                                "timeout_s": 60, "artifacts": []}])
        code = self.run_spec(spec, provider)
        self.assertEqual(code, 0)
        self.assertEqual(provider.detached_commands, [])
        self.assertTrue(any("echo hi" in c for c in provider.commands))

    def test_exit_code_propagation_and_stop_on_error(self):
        provider = DetachedFakeProvider(poll_script=[None, 7])
        spec = make_spec(jobs=[self.long_job("train"),
                               {"name": "after", "command": "echo x",
                                "timeout_s": 10, "artifacts": []}])
        code = self.run_spec(spec, provider)
        self.assertEqual(code, rd.EXIT_JOB_FAILED)
        m = self.manifest()
        self.assertEqual(m["jobs"][0]["exit_code"], 7)
        self.assertTrue(m["jobs"][1].get("skipped"))
        self.assertTrue(provider.deleted)

    def test_failed_detached_job_still_downloads_artifacts(self):
        payload = b"checkpoint-bytes"
        provider = DetachedFakeProvider(
            poll_script=[1],
            artifacts={f"{rd.REMOTE_WORKDIR}/out/latest.pt": payload})
        spec = make_spec(jobs=[self.long_job(artifacts=["out/latest.pt"])])
        code = self.run_spec(spec, provider)
        self.assertEqual(code, rd.EXIT_JOB_FAILED)
        dest = self.run_dir / "artifacts" / "train" / "out" / "latest.pt"
        self.assertEqual(dest.read_bytes(), payload)
        art = self.manifest()["jobs"][0]["artifacts"]["out/latest.pt"]
        self.assertEqual(art["sha256"], hashlib.sha256(payload).hexdigest())

    def test_poll_retries_transient_errors_then_succeeds(self):
        provider = DetachedFakeProvider(poll_script=[
            None,
            ConnectionError("read timeout 1"),
            ConnectionError("read timeout 2"),
            ConnectionError("read timeout 3"),
            None,
            0,
        ])
        spec = make_spec(jobs=[self.long_job()])
        code = self.run_spec(spec, provider)
        self.assertEqual(code, 0)
        self.assertEqual(provider.poll_calls, 6)
        self.assertTrue(self.logged("poll failed (1/5)"))
        self.assertTrue(self.logged("retrying"))

    def test_five_consecutive_poll_failures_abort_but_salvage_and_delete(self):
        payload = b"salvaged-checkpoint"
        provider = DetachedFakeProvider(
            poll_script=[ConnectionError(f"timeout {i}") for i in range(9)],
            artifacts={f"{rd.REMOTE_WORKDIR}/out/latest.pt": payload})
        spec = make_spec(jobs=[self.long_job(artifacts=["out/latest.pt"])])
        code = self.run_spec(spec, provider)
        self.assertEqual(code, rd.EXIT_ERROR)
        self.assertEqual(provider.poll_calls, rd.MAX_POLL_FAILURES)
        self.assertTrue(provider.deleted)
        m = self.manifest()
        art = m["salvaged_artifacts"]["train"]["out/latest.pt"]
        self.assertEqual(art["status"], "ok")
        dest = self.run_dir / "artifacts" / "train" / "out" / "latest.pt"
        self.assertEqual(dest.read_bytes(), payload)
        self.assertTrue(self.logged("salvaged artifacts"))

    def test_intermittent_failures_reset_consecutive_counter(self):
        script = []
        for _ in range(3):
            script += [ConnectionError("t1"), ConnectionError("t2"), None]
        script.append(0)
        provider = DetachedFakeProvider(poll_script=script)
        spec = make_spec(jobs=[self.long_job()])
        code = self.run_spec(spec, provider)
        self.assertEqual(code, 0)  # never 5 consecutive, so never fatal

    def test_log_fetch_failures_are_best_effort(self):
        provider = DetachedFakeProvider(poll_script=[0], logs_exc_count=99)
        spec = make_spec(jobs=[self.long_job()])
        code = self.run_spec(spec, provider)
        self.assertEqual(code, 0)
        self.assertEqual(provider.log_fetches, rd.MAX_POLL_FAILURES)
        self.assertEqual((self.run_dir / "logs" / "train.log").read_text(), "")
        self.assertTrue(self.logged("log fetch failed"))

    def test_job_timeout_enforced_with_exit_124(self):
        t = [0.0]

        def clock():
            t[0] += 100.0
            return t[0]

        provider = DetachedFakeProvider(poll_script=[None] * 100)
        spec = make_spec(jobs=[self.long_job(timeout_s=700)])
        code = self.run_spec(spec, provider,
                             deadline=rd.Deadline(rd.WALL_CLOCK_CAP_S, clock=clock))
        self.assertEqual(code, rd.EXIT_JOB_FAILED)
        m = self.manifest()
        self.assertEqual(m["jobs"][0]["exit_code"], rd.JOB_TIMEOUT_EXIT_CODE)
        self.assertTrue(self.logged("exceeded its 700s timeout"))
        self.assertTrue(provider.deleted)

    def test_poll_error_messages_are_redacted(self):
        provider = DetachedFakeProvider(poll_script=[
            ConnectionError(f"auth {FAKE_SECRET} rejected"), None, 0])
        spec = make_spec(jobs=[self.long_job()])
        code = self.run_spec(spec, provider)
        self.assertEqual(code, 0)
        self.assertFalse(any(FAKE_SECRET in line for line in self.logs))
        self.assertTrue(self.logged("[REDACTED]"))


class TestSalvage(DetachedTestCase):
    def two_jobs_spec(self):
        return make_spec(jobs=[
            {"name": "a", "command": "run-a", "timeout_s": 10,
             "artifacts": ["a-out.txt"]},
            {"name": "b", "command": "run-b", "timeout_s": 10,
             "artifacts": ["b-out.txt"]},
        ])

    def test_sync_exec_exception_still_salvages_artifacts(self):
        payload = b"partial-a"
        provider = DetachedFakeProvider(
            exec_exc=rd.LauncherError("stream broke"),
            artifacts={f"{rd.REMOTE_WORKDIR}/a-out.txt": payload})
        code = self.run_spec(self.two_jobs_spec(), provider)
        self.assertEqual(code, rd.EXIT_ERROR)
        m = self.manifest()
        sal = m["salvaged_artifacts"]
        self.assertEqual(sal["a"]["a-out.txt"]["status"], "ok")
        self.assertEqual(sal["b"]["b-out.txt"]["status"], "missing")
        dest = self.run_dir / "artifacts" / "a" / "a-out.txt"
        self.assertEqual(dest.read_bytes(), payload)
        self.assertTrue(provider.deleted)

    def test_skipped_jobs_get_salvage_attempt(self):
        provider = DetachedFakeProvider(
            exec_results={"run-a": (5, "boom\n")},
            artifacts={f"{rd.REMOTE_WORKDIR}/b-out.txt": b"early-b"})
        code = self.run_spec(self.two_jobs_spec(), provider)
        self.assertEqual(code, rd.EXIT_JOB_FAILED)
        m = self.manifest()
        # job a failed but ran: artifacts collected normally, not salvaged
        self.assertNotIn("a", m.get("salvaged_artifacts", {}))
        self.assertEqual(m["salvaged_artifacts"]["b"]["b-out.txt"]["status"], "ok")

    def test_no_salvage_key_when_everything_collected(self):
        provider = DetachedFakeProvider()
        code = self.run_spec(self.two_jobs_spec(), provider)
        self.assertEqual(code, 0)
        self.assertNotIn("salvaged_artifacts", self.manifest())

    def test_salvage_download_error_marked_not_fatal(self):
        class ExplodingDownloads(DetachedFakeProvider):
            def download(self, handle, remote_path):
                raise ConnectionError(f"boom {FAKE_SECRET}")

        provider = ExplodingDownloads(exec_exc=rd.LauncherError("stream broke"))
        code = self.run_spec(self.two_jobs_spec(), provider)
        self.assertEqual(code, rd.EXIT_ERROR)
        m = self.manifest()
        entry = m["salvaged_artifacts"]["a"]["a-out.txt"]
        self.assertEqual(entry["status"], "error")
        self.assertNotIn(FAKE_SECRET, json.dumps(m))
        self.assertTrue(provider.deleted)  # salvage failure never blocks delete


if __name__ == "__main__":
    unittest.main()
