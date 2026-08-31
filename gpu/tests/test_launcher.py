"""Unit tests for gpu/run_daytona.py with a fake provider (no network,
no daytona SDK, no DAYTONA_API_KEY needed).

Run:  .venv/bin/python -m unittest discover -s gpu/tests -v
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_daytona as rd  # noqa: E402

FAKE_SECRET = "dtn_FAKESECRET_abcdef1234567890"


def make_spec(jobs=None, **overrides):
    raw = {
        "name": "t",
        "tar_globs": [],
        "upload_extra": [],
        "jobs": jobs or [
            {"name": "j1", "command": "echo one", "timeout_s": 10, "artifacts": []},
        ],
        "resources": {},
    }
    raw.update(overrides)
    return rd.parse_spec(raw)


def make_git_repo(root):
    root = Path(root)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "tracked.txt").write_text("tracked\n")
    (root / "sub").mkdir()
    (root / "sub" / "inner.py").write_text("print('hi')\n")
    (root / "untracked.bin").write_text("NOT IN TAR\n")
    env = dict(os.environ,
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    subprocess.run(["git", "-C", str(root), "add", "tracked.txt", "sub/inner.py"],
                   check=True, env=env)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "init"],
                   check=True, env=env)
    # extra file that is NOT git-tracked but listed in upload_extra
    (root / "extra.json").write_text("{}\n")
    return root


class FakeProvider:
    """In-memory stand-in for DaytonaProvider. Records every call."""

    def __init__(self, exec_results=None, artifacts=None, existing_ids=(),
                 create_exc=None, exec_exc=None, delete_exc=None,
                 gone_after_delete=True):
        self.exec_results = dict(exec_results or {})   # substring -> (code, output)
        self.artifacts = dict(artifacts or {})         # remote path -> bytes|None
        self.existing_ids = list(existing_ids)
        self.create_exc = create_exc
        self.exec_exc = exec_exc
        self.delete_exc = delete_exc
        self.gone_after_delete = gone_after_delete
        self.created = False
        self.deleted = False
        self.uploads = []
        self.commands = []

    def list_launcher_sandbox_ids(self):
        return list(self.existing_ids)

    def create(self, spec, gpu_types, timeout_s):
        if self.create_exc:
            raise self.create_exc
        self.created = True
        self.gpu_types = list(gpu_types)
        return rd.SandboxHandle(sandbox_id="sb-fake-1")

    def upload(self, handle, local_path, remote_path):
        self.uploads.append((str(local_path), remote_path))

    def exec(self, handle, command, timeout_s):
        self.commands.append(command)
        if self.exec_exc and "tar -xf" not in command:
            raise self.exec_exc
        for needle, (code, out) in self.exec_results.items():
            if needle in command:
                return rd.ExecResult(exit_code=code, output=out)
        return rd.ExecResult(exit_code=0, output="ok\n")

    def download(self, handle, remote_path):
        return self.artifacts.get(remote_path)

    def delete(self, handle):
        if self.delete_exc:
            raise self.delete_exc
        self.deleted = True

    def sandbox_gone(self, sandbox_id):
        return self.gone_after_delete


class LauncherTestCase(unittest.TestCase):
    def setUp(self):
        # Fake-provider retries must not consume real provider polling time.
        sleep_patch = patch.object(rd.time, "sleep", return_value=None)
        self.mock_sleep = sleep_patch.start()
        self.addCleanup(sleep_patch.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.repo = make_git_repo(self.tmp / "repo")
        self.run_dir = self.tmp / "run"
        self.logs = []

    def tearDown(self):
        self._tmp.cleanup()

    def log(self, *a):
        self.logs.append(" ".join(str(x) for x in a))

    def run_spec(self, spec, provider, redactor=None, deadline=None):
        return rd.execute_run(
            spec, self.repo, self.run_dir, provider,
            redactor or rd.Redactor([FAKE_SECRET]),
            deadline or rd.Deadline(),
            log=self.log,
        )

    def manifest(self):
        return json.loads((self.run_dir / "manifest.json").read_text())


class TestSpecValidation(unittest.TestCase):
    def test_valid_spec_parses(self):
        spec = make_spec()
        self.assertEqual(spec.name, "t")
        self.assertEqual(spec.resources.gpu_types, ["RTX-5090"])
        self.assertEqual(spec.resources.ttl_minutes, 25)

    def test_rejects_missing_name(self):
        with self.assertRaises(rd.SpecError):
            rd.parse_spec({"jobs": [{"name": "a", "command": "x", "timeout_s": 5}]})

    def test_rejects_empty_jobs(self):
        with self.assertRaises(rd.SpecError):
            rd.parse_spec({"name": "x", "jobs": []})

    def test_rejects_bad_timeout(self):
        for bad in (0, -5, "10", rd.WALL_CLOCK_CAP_S + 1):
            with self.assertRaises(rd.SpecError):
                make_spec(jobs=[{"name": "j", "command": "x", "timeout_s": bad}])

    def test_rejects_absolute_or_dotdot_artifacts(self):
        for bad in ("/etc/passwd", "../escape"):
            with self.assertRaises(rd.SpecError):
                make_spec(jobs=[{"name": "j", "command": "x", "timeout_s": 5,
                                 "artifacts": [bad]}])

    def test_rejects_unknown_gpu_type(self):
        with self.assertRaises(rd.SpecError):
            make_spec(resources={"gpu_types": ["A100"]})

    def test_rejects_unsafe_names(self):
        with self.assertRaises(rd.SpecError):
            make_spec(name="../evil")
        with self.assertRaises(rd.SpecError):
            make_spec(jobs=[{"name": "a/b", "command": "x", "timeout_s": 5}])

    def test_rejects_excessive_ttl(self):
        with self.assertRaises(rd.SpecError):
            make_spec(resources={"ttl_minutes": 999})

    def test_real_compile_spec_is_valid(self):
        spec = rd.load_spec(Path(__file__).resolve().parent.parent
                            / "specs" / "compile-duck-cuda.json")
        self.assertEqual(spec.name, "compile-duck-cuda")
        self.assertEqual([j.name for j in spec.jobs],
                         ["gpu-info", "build"])
        self.assertTrue(all(not job.continue_on_error for job in spec.jobs))


class TestTarBuild(LauncherTestCase):
    def test_tar_contains_exactly_git_tracked_files(self):
        spec = make_spec()
        tar_path = self.tmp / "p.tar"
        sha = rd.build_payload_tar(self.repo, spec, tar_path)
        self.assertRegex(sha, r"^[0-9a-f]{40}$")
        names = sorted(rd.tar_member_names(tar_path))
        self.assertEqual(names, ["sub/inner.py", "tracked.txt"])
        self.assertNotIn("untracked.bin", names)

    def test_tar_globs_filter(self):
        spec = make_spec(tar_globs=["sub"])
        tar_path = self.tmp / "p.tar"
        rd.build_payload_tar(self.repo, spec, tar_path)
        self.assertEqual(rd.tar_member_names(tar_path), ["sub/inner.py"])

    def test_upload_extra_appended(self):
        spec = make_spec(upload_extra=["extra.json"])
        tar_path = self.tmp / "p.tar"
        rd.build_payload_tar(self.repo, spec, tar_path)
        self.assertIn("extra.json", rd.tar_member_names(tar_path))

    def test_upload_extra_missing_file_rejected(self):
        spec = make_spec(upload_extra=["nope.txt"])
        with self.assertRaises(rd.SpecError):
            rd.build_payload_tar(self.repo, spec, self.tmp / "p.tar")


class TestJobSequencing(LauncherTestCase):
    def jobs3(self, coe_second=False):
        return [
            {"name": "a", "command": "run-a", "timeout_s": 10, "artifacts": []},
            {"name": "b", "command": "run-b", "timeout_s": 10, "artifacts": [],
             "continue_on_error": coe_second},
            {"name": "c", "command": "run-c", "timeout_s": 10, "artifacts": []},
        ]

    def test_all_jobs_run_in_order_on_success(self):
        provider = FakeProvider()
        code = self.run_spec(make_spec(jobs=self.jobs3()), provider)
        self.assertEqual(code, 0)
        job_cmds = [c for c in provider.commands if "run-" in c]
        self.assertEqual([c.split("(")[1][:5] for c in job_cmds],
                         ["run-a", "run-b", "run-c"])
        m = self.manifest()
        self.assertEqual([j["exit_code"] for j in m["jobs"]], [0, 0, 0])

    def test_stop_on_first_failure(self):
        provider = FakeProvider(exec_results={"run-b": (7, "boom\n")})
        code = self.run_spec(make_spec(jobs=self.jobs3()), provider)
        self.assertEqual(code, rd.EXIT_JOB_FAILED)
        self.assertFalse(any("run-c" in c for c in provider.commands))
        m = self.manifest()
        self.assertEqual(m["jobs"][1]["exit_code"], 7)
        self.assertTrue(m["jobs"][2].get("skipped"))
        self.assertTrue(provider.deleted)

    def test_continue_on_error(self):
        provider = FakeProvider(exec_results={"run-b": (7, "boom\n")})
        code = self.run_spec(make_spec(jobs=self.jobs3(coe_second=True)), provider)
        self.assertEqual(code, rd.EXIT_JOB_FAILED)  # still nonzero overall
        self.assertTrue(any("run-c" in c for c in provider.commands))

    def test_concurrency_guard(self):
        provider = FakeProvider(existing_ids=["sb-old"])
        code = self.run_spec(make_spec(), provider)
        self.assertEqual(code, rd.EXIT_ERROR)
        self.assertFalse(provider.created)


class TestArtifacts(LauncherTestCase):
    def test_artifact_download_paths_and_hashes(self):
        payload = b"artifact-bytes"
        provider = FakeProvider(
            artifacts={f"{rd.REMOTE_WORKDIR}/out/log.txt": payload})
        spec = make_spec(jobs=[{"name": "j1", "command": "echo hi",
                                "timeout_s": 10, "artifacts": ["out/log.txt"]}])
        code = self.run_spec(spec, provider)
        self.assertEqual(code, 0)
        dest = self.run_dir / "artifacts" / "j1" / "out" / "log.txt"
        self.assertEqual(dest.read_bytes(), payload)
        m = self.manifest()
        art = m["jobs"][0]["artifacts"]["out/log.txt"]
        self.assertEqual(art["status"], "ok")
        import hashlib
        self.assertEqual(art["sha256"], hashlib.sha256(payload).hexdigest())

    def test_missing_artifact_recorded_not_fatal(self):
        provider = FakeProvider()
        spec = make_spec(jobs=[{"name": "j1", "command": "echo hi",
                                "timeout_s": 10, "artifacts": ["gone.txt"]}])
        code = self.run_spec(spec, provider)
        self.assertEqual(code, 0)
        m = self.manifest()
        self.assertEqual(m["jobs"][0]["artifacts"]["gone.txt"]["status"], "missing")


class TestRedaction(LauncherTestCase):
    def test_secret_in_job_output_never_reaches_disk(self):
        provider = FakeProvider(
            exec_results={"leaky": (0, f"token is {FAKE_SECRET} ok\n")})
        spec = make_spec(jobs=[{"name": "leak", "command": "leaky",
                                "timeout_s": 10, "artifacts": []}])
        code = self.run_spec(spec, provider)
        self.assertEqual(code, 0)
        for f in self.run_dir.rglob("*"):
            if f.is_file() and f.suffix != ".tar":
                content = f.read_text()
                self.assertNotIn(FAKE_SECRET, content, f"secret leaked into {f}")
        log_text = (self.run_dir / "logs" / "leak.log").read_text()
        self.assertIn("[REDACTED]", log_text)

    def test_key_pattern_redacted_even_without_known_value(self):
        r = rd.Redactor([])  # no known secret value
        self.assertNotIn("dtn_", r.redact("x dtn_abcdefgh12345 y"))

    def test_exception_messages_redacted(self):
        provider = FakeProvider(
            create_exc=rd.LauncherError(f"auth failed with {FAKE_SECRET}"))
        code = self.run_spec(make_spec(), provider)
        self.assertEqual(code, rd.EXIT_ERROR)
        self.assertFalse(any(FAKE_SECRET in line for line in self.logs))


class TestDeletion(LauncherTestCase):
    def test_deletion_attempted_when_job_exec_raises(self):
        provider = FakeProvider(exec_exc=rd.LauncherError("transport died"))
        code = self.run_spec(make_spec(), provider)
        self.assertEqual(code, rd.EXIT_ERROR)
        self.assertTrue(provider.deleted)
        receipt = json.loads((self.run_dir / "deletion_receipt.json").read_text())
        self.assertTrue(receipt["delete_attempted"])
        self.assertTrue(receipt["verified_gone"])

    def test_deletion_verification_failure_is_nonzero_and_loud(self):
        provider = FakeProvider(gone_after_delete=False)
        code = self.run_spec(make_spec(), provider)
        self.assertEqual(code, rd.EXIT_DELETE_UNVERIFIED)
        receipt = json.loads((self.run_dir / "deletion_receipt.json").read_text())
        self.assertFalse(receipt["verified_gone"])
        self.assertTrue(any("NOT VERIFIED" in line for line in self.logs))

    def test_delete_error_still_verifies_and_writes_receipt(self):
        provider = FakeProvider(delete_exc=RuntimeError("api 500"),
                                gone_after_delete=False)
        code = self.run_spec(make_spec(), provider)
        self.assertEqual(code, rd.EXIT_DELETE_UNVERIFIED)
        receipt = json.loads((self.run_dir / "deletion_receipt.json").read_text())
        self.assertIn("api 500", receipt["delete_error"])

    def test_no_delete_when_sandbox_never_created(self):
        provider = FakeProvider(create_exc=rd.SpotUnavailableError("no spot"))
        code = self.run_spec(make_spec(), provider)
        self.assertEqual(code, rd.EXIT_SPOT_UNAVAILABLE)
        self.assertFalse(provider.deleted)
        receipt = json.loads((self.run_dir / "deletion_receipt.json").read_text())
        self.assertFalse(receipt["delete_attempted"])


class TestSpotAndBudget(LauncherTestCase):
    def test_spot_unavailable_typed_exit_no_retry(self):
        provider = FakeProvider(create_exc=rd.SpotUnavailableError("capacity"))
        code = self.run_spec(make_spec(), provider)
        self.assertEqual(code, rd.EXIT_SPOT_UNAVAILABLE)
        self.assertFalse(provider.created)
        m = self.manifest()
        self.assertEqual(m["exit_code"], rd.EXIT_SPOT_UNAVAILABLE)

    def test_budget_exceeded_aborts_but_deletes(self):
        t = [0.0]

        def clock():
            t[0] += 400.0  # each tick burns 400s of the 1500s budget
            return t[0]

        provider = FakeProvider()
        jobs = [{"name": f"j{i}", "command": f"run-{i}", "timeout_s": 100,
                 "artifacts": []} for i in range(9)]
        code = self.run_spec(make_spec(jobs=jobs), provider,
                             deadline=rd.Deadline(rd.WALL_CLOCK_CAP_S, clock=clock))
        self.assertEqual(code, rd.EXIT_BUDGET_EXCEEDED)
        self.assertTrue(provider.deleted)

    def test_gpu_type_fallback_appended(self):
        provider = FakeProvider()
        code = rd.execute_run(make_spec(), self.repo, self.run_dir, provider,
                              rd.Redactor([]), rd.Deadline(),
                              extra_gpu_types=["RTX-4090"], log=self.log)
        self.assertEqual(code, 0)
        self.assertEqual(provider.gpu_types, ["RTX-5090", "RTX-4090"])


class TestDryRun(LauncherTestCase):
    def test_dry_run_without_api_key_makes_zero_provider_calls(self):
        spec_path = self.tmp / "spec.json"
        spec_path.write_text(json.dumps({
            "name": "dry",
            "jobs": [{"name": "j", "command": "echo hi", "timeout_s": 5}],
        }))

        def forbidden_factory():
            raise AssertionError("provider factory called during --dry-run")

        env = {k: v for k, v in os.environ.items() if k != "DAYTONA_API_KEY"}
        code = rd.main(
            ["run", "--spec", str(spec_path), "--dry-run",
             "--runs-dir", str(self.tmp / "runs")],
            provider_factory=forbidden_factory,
            repo_root=self.repo, env=env, log=self.log,
        )
        self.assertEqual(code, 0)
        self.assertTrue(any("No provider calls" in line for line in self.logs))
        tars = list((self.tmp / "runs").rglob("payload.tar"))
        self.assertEqual(len(tars), 1)

    def test_real_run_without_api_key_refuses_before_provider(self):
        spec_path = self.tmp / "spec.json"
        spec_path.write_text(json.dumps({
            "name": "nokey",
            "jobs": [{"name": "j", "command": "echo hi", "timeout_s": 5}],
        }))

        def forbidden_factory():
            raise AssertionError("provider factory called without API key")

        env = {k: v for k, v in os.environ.items() if k != "DAYTONA_API_KEY"}
        code = rd.main(
            ["run", "--spec", str(spec_path),
             "--runs-dir", str(self.tmp / "runs")],
            provider_factory=forbidden_factory,
            repo_root=self.repo, env=env, log=self.log,
        )
        self.assertEqual(code, rd.EXIT_ERROR)
        self.assertTrue(any("doppler run" in line for line in self.logs))


if __name__ == "__main__":
    unittest.main()
