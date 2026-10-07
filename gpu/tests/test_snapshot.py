"""Unit tests for the baked-snapshot fast path (bake mode, image manifest,
use_snapshot run wiring). No network, no daytona SDK, no real key.

Run:  .venv/bin/python -m unittest discover -s gpu/tests -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_daytona as rd  # noqa: E402
from test_launcher import FakeProvider, make_git_repo, make_spec  # noqa: E402

FAKE_KEY = "dtn_FAKEKEY_abcdef1234567890"
SPECS_DIR = Path(__file__).resolve().parent.parent / "specs"


class SnapshotFakeProvider(FakeProvider):
    """FakeProvider plus the snapshot surface used by bake/use_snapshot."""

    def __init__(self, *args, snapshot_id="snap-id-1", snapshot_lookup_ok=True,
                 from_snapshot_exc=None, activate_exc=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshot_id = snapshot_id
        self.snapshot_lookup_ok = snapshot_lookup_ok
        self.from_snapshot_exc = from_snapshot_exc
        self.activate_exc = activate_exc
        self.snapshots_created = []
        self.from_snapshot_calls = []
        self.activated = []
        self.create_timeouts = []
        self.deleted_by_id = []

    def create(self, spec, gpu_types, timeout_s):
        self.create_timeouts.append(("image", timeout_s))
        return super().create(spec, gpu_types, timeout_s)

    def create_from_snapshot(self, spec, snapshot_name, timeout_s):
        self.create_timeouts.append(("snapshot", timeout_s))
        if self.from_snapshot_exc:
            raise self.from_snapshot_exc
        self.from_snapshot_calls.append(snapshot_name)
        return rd.SandboxHandle(sandbox_id="sb-snap-1")

    def create_snapshot(self, handle, name, timeout_s):
        self.snapshots_created.append(name)

    def snapshot_exists(self, name):
        if self.snapshot_lookup_ok and name in self.snapshots_created:
            return self.snapshot_id
        return None

    def activate_snapshot(self, name):
        if self.activate_exc:
            raise self.activate_exc
        self.activated.append(name)
        return "active"

    def delete_by_id(self, sandbox_id):
        self.deleted_by_id.append(sandbox_id)


class OrphanFakeProvider(SnapshotFakeProvider):
    """Simulates a create that fails AFTER the sandbox exists server-side:
    the concurrency guard sees nothing, but post-failure listing finds the
    orphan (labels were attached at create)."""

    def __init__(self, *args, orphan_ids=("sb-orphan-1",), **kwargs):
        super().__init__(*args, **kwargs)
        self.orphan_ids = list(orphan_ids)
        self.list_calls = 0

    def list_launcher_sandbox_ids(self):
        self.list_calls += 1
        if self.list_calls == 1:
            return []  # concurrency guard: clean before create
        return list(self.orphan_ids)


class SnapshotTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.repo = make_git_repo(self.tmp / "repo")
        self.runs = self.tmp / "runs"
        self.logs = []

    def tearDown(self):
        self._tmp.cleanup()

    def log(self, *a):
        self.logs.append(" ".join(str(x) for x in a))

    def logged(self, needle):
        return any(needle in line for line in self.logs)

    # -- helpers ----------------------------------------------------------

    def write_spec(self, raw, name="spec.json"):
        p = self.tmp / name
        p.write_text(json.dumps(raw))
        return p

    def base_raw_spec(self, **overrides):
        raw = {
            "name": "snaptest",
            "jobs": [{"name": "j1", "command": "echo hi", "timeout_s": 10,
                      "artifacts": []}],
        }
        raw.update(overrides)
        return raw

    def call_main(self, argv, provider, with_key=True):
        env = {k: v for k, v in os.environ.items() if k != "DAYTONA_API_KEY"}
        if with_key:
            env["DAYTONA_API_KEY"] = FAKE_KEY
        if argv[0] != "warm":
            argv = argv + ["--runs-dir", str(self.runs)]
        return rd.main(argv, provider_factory=lambda: provider,
                       repo_root=self.repo, env=env, log=self.log)

    def image_manifest(self):
        p = rd.image_manifest_path(self.repo)
        return json.loads(p.read_text()) if p.is_file() else None

    def run_manifest(self):
        files = list(self.runs.rglob("manifest.json"))
        self.assertEqual(len(files), 1)
        return json.loads(files[0].read_text())


class TestUseSnapshotParsing(unittest.TestCase):
    def test_default_false(self):
        self.assertFalse(make_spec().use_snapshot)

    def test_true_parses(self):
        self.assertTrue(make_spec(use_snapshot=True).use_snapshot)

    def test_non_bool_rejected(self):
        with self.assertRaises(rd.SpecError):
            make_spec(use_snapshot="yes")

    def test_bake_image_spec_on_disk_is_valid(self):
        spec = rd.load_spec(SPECS_DIR / "bake-image.json")
        self.assertEqual([j.name for j in spec.jobs],
                         ["gpu-info", "deps", "prebuild"])
        self.assertFalse(spec.use_snapshot)  # bake always starts from image
        self.assertIn("prebuilt-sha.txt", spec.jobs[2].artifacts)
        self.assertIn(rd.REMOTE_PREBUILT_DIR, spec.jobs[2].command)

    def test_train_specs_have_guarded_builds(self):
        for name in ("fresh-ff.json", "fresh-gru.json",
                     "train-duck-gpu.json", "sweep-train.json"):
            spec = rd.load_spec(SPECS_DIR / name)
            self.assertTrue(spec.use_snapshot, name)
            build = next(j for j in spec.jobs if j.name == "build")
            self.assertIn("prebuilt cache HIT", build.command, name)
            self.assertIn("prebuilt cache MISS", build.command, name)
            self.assertIn(rd.REMOTE_PREBUILT_DIR, build.command, name)

    def test_sweep_train_guard_keeps_fast_build_branch(self):
        spec = rd.load_spec(SPECS_DIR / "sweep-train.json")
        build = next(j for j in spec.jobs if j.name == "build")
        self.assertIn("SWEEP_FAST_BUILD", build.command)


class TestImageManifestIO(SnapshotTestCase):
    def test_missing_file_returns_none(self):
        self.assertIsNone(rd.load_image_manifest(self.tmp / "nope.json"))

    def test_garbage_returns_none(self):
        p = self.tmp / "bad.json"
        p.write_text("{not json")
        self.assertIsNone(rd.load_image_manifest(p))

    def test_manifest_without_snapshot_returns_none(self):
        p = self.tmp / "empty.json"
        p.write_text(json.dumps({"source_sha": "abc"}))
        self.assertIsNone(rd.load_image_manifest(p))

    def test_roundtrip(self):
        p = self.tmp / "ok.json"
        p.write_text(json.dumps({"snapshot": "snap-x", "source_sha": "abc"}))
        self.assertEqual(rd.load_image_manifest(p)["snapshot"], "snap-x")


class TestBakeFlow(SnapshotTestCase):
    def bake_spec_path(self):
        return self.write_spec(self.base_raw_spec(
            name="bake-t",
            jobs=[{"name": "prebuild", "command": "fake-build",
                   "timeout_s": 30, "artifacts": ["prebuilt-sha.txt"]}],
        ))

    def test_bake_success_snapshots_and_writes_image_manifest(self):
        provider = SnapshotFakeProvider(
            artifacts={f"{rd.REMOTE_WORKDIR}/prebuilt-sha.txt": b"cafe1234\n"})
        code = self.call_main(["bake", "--spec", str(self.bake_spec_path())],
                              provider)
        self.assertEqual(code, 0)
        self.assertEqual(len(provider.snapshots_created), 1)
        snap_name = provider.snapshots_created[0]
        self.assertTrue(snap_name.startswith(rd.SNAPSHOT_NAME_PREFIX + "-"))
        im = self.image_manifest()
        self.assertEqual(im["snapshot"], snap_name)
        self.assertEqual(im["snapshot_id"], "snap-id-1")
        self.assertEqual(im["source_sha"], "cafe1234")
        self.assertEqual(im["prebuilt_dir"], rd.REMOTE_PREBUILT_DIR)
        self.assertRegex(im["commit"], r"^[0-9a-f]{40}$")
        self.assertTrue(provider.deleted)
        rm = self.run_manifest()
        self.assertEqual(rm["image_manifest"]["snapshot"], snap_name)
        self.assertTrue(rm["deletion"]["verified_gone"])

    def test_bake_job_failure_means_no_snapshot_no_manifest(self):
        provider = SnapshotFakeProvider(exec_results={"fake-build": (3, "boom\n")})
        code = self.call_main(["bake", "--spec", str(self.bake_spec_path())],
                              provider)
        self.assertEqual(code, rd.EXIT_JOB_FAILED)
        self.assertEqual(provider.snapshots_created, [])
        self.assertIsNone(self.image_manifest())
        self.assertTrue(provider.deleted)

    def test_bake_snapshot_verification_failure_is_error_and_deletes(self):
        provider = SnapshotFakeProvider(
            snapshot_lookup_ok=False,
            artifacts={f"{rd.REMOTE_WORKDIR}/prebuilt-sha.txt": b"cafe1234"})
        code = self.call_main(["bake", "--spec", str(self.bake_spec_path())],
                              provider)
        self.assertEqual(code, rd.EXIT_ERROR)
        self.assertIsNone(self.image_manifest())
        self.assertTrue(provider.deleted)
        self.assertTrue(self.logged("not found after creation"))

    def test_bake_without_sha_artifact_records_null_sha(self):
        spec_path = self.write_spec(self.base_raw_spec(name="bake-nosha"))
        provider = SnapshotFakeProvider()
        code = self.call_main(["bake", "--spec", str(spec_path)], provider)
        self.assertEqual(code, 0)
        self.assertIsNone(self.image_manifest()["source_sha"])

    def test_bake_dry_run_without_key_makes_zero_provider_calls(self):
        def forbidden():
            raise AssertionError("provider factory called during bake --dry-run")

        env = {k: v for k, v in os.environ.items() if k != "DAYTONA_API_KEY"}
        code = rd.main(
            ["bake", "--spec", str(self.bake_spec_path()), "--dry-run",
             "--runs-dir", str(self.runs)],
            provider_factory=forbidden, repo_root=self.repo, env=env,
            log=self.log,
        )
        self.assertEqual(code, 0)
        self.assertTrue(self.logged("bake would snapshot as"))
        self.assertIsNone(self.image_manifest())

    def test_bake_secret_never_reaches_image_manifest(self):
        provider = SnapshotFakeProvider(snapshot_id=f"id-{FAKE_KEY}-x")
        spec_path = self.write_spec(self.base_raw_spec(name="bake-redact"))
        code = self.call_main(["bake", "--spec", str(spec_path)], provider)
        self.assertEqual(code, 0)
        text = rd.image_manifest_path(self.repo).read_text()
        self.assertNotIn(FAKE_KEY, text)
        self.assertIn("[REDACTED]", text)


class TestRunWithSnapshot(SnapshotTestCase):
    def seed_image_manifest(self, snapshot="duck-gpu-prebuilt-20260831-000000",
                            source_sha="cafe1234"):
        p = rd.image_manifest_path(self.repo)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"snapshot": snapshot, "source_sha": source_sha}))
        return snapshot

    def snap_spec_path(self, use_snapshot=True):
        return self.write_spec(self.base_raw_spec(use_snapshot=use_snapshot))

    def test_run_uses_snapshot_when_spec_opts_in_and_manifest_exists(self):
        snap = self.seed_image_manifest()
        provider = SnapshotFakeProvider()
        code = self.call_main(["run", "--spec", str(self.snap_spec_path())],
                              provider)
        self.assertEqual(code, 0)
        self.assertEqual(provider.from_snapshot_calls, [snap])
        self.assertFalse(provider.created)  # image-based create never called
        self.assertEqual(self.run_manifest()["snapshot_used"], snap)
        self.assertTrue(provider.deleted)

    def test_run_falls_back_to_image_without_manifest(self):
        provider = SnapshotFakeProvider()
        code = self.call_main(["run", "--spec", str(self.snap_spec_path())],
                              provider)
        self.assertEqual(code, 0)
        self.assertEqual(provider.from_snapshot_calls, [])
        self.assertTrue(provider.created)
        self.assertIsNone(self.run_manifest()["snapshot_used"])
        self.assertTrue(self.logged("falling back to the base image"))

    def test_no_snapshot_flag_forces_base_image(self):
        self.seed_image_manifest()
        provider = SnapshotFakeProvider()
        code = self.call_main(
            ["run", "--spec", str(self.snap_spec_path()), "--no-snapshot"],
            provider)
        self.assertEqual(code, 0)
        self.assertEqual(provider.from_snapshot_calls, [])
        self.assertTrue(provider.created)

    def test_spec_without_use_snapshot_ignores_manifest(self):
        self.seed_image_manifest()
        provider = SnapshotFakeProvider()
        code = self.call_main(
            ["run", "--spec", str(self.snap_spec_path(use_snapshot=False))],
            provider)
        self.assertEqual(code, 0)
        self.assertEqual(provider.from_snapshot_calls, [])
        self.assertTrue(provider.created)

    def test_spot_unavailable_from_snapshot_create_is_typed_exit(self):
        self.seed_image_manifest()
        provider = SnapshotFakeProvider(
            from_snapshot_exc=rd.SpotUnavailableError("capacity"))
        code = self.call_main(["run", "--spec", str(self.snap_spec_path())],
                              provider)
        self.assertEqual(code, rd.EXIT_SPOT_UNAVAILABLE)
        self.assertFalse(provider.deleted)  # nothing was created

    def test_snapshot_create_uses_longer_timeout_than_image_create(self):
        self.seed_image_manifest()
        provider = SnapshotFakeProvider()
        code = self.call_main(["run", "--spec", str(self.snap_spec_path())],
                              provider)
        self.assertEqual(code, 0)
        kind, timeout = provider.create_timeouts[0]
        self.assertEqual(kind, "snapshot")
        self.assertEqual(timeout, rd.SNAPSHOT_CREATE_TIMEOUT_S)

        provider2 = SnapshotFakeProvider()
        code = self.call_main(
            ["run", "--spec", str(self.snap_spec_path(use_snapshot=False))],
            provider2)
        self.assertEqual(code, 0)
        kind, timeout = provider2.create_timeouts[0]
        self.assertEqual(kind, "image")
        self.assertEqual(timeout, rd.CREATE_TIMEOUT_S)

    def test_run_dry_run_reports_snapshot_plan_without_key(self):
        snap = self.seed_image_manifest()

        def forbidden():
            raise AssertionError("provider factory called during --dry-run")

        env = {k: v for k, v in os.environ.items() if k != "DAYTONA_API_KEY"}
        code = rd.main(
            ["run", "--spec", str(self.snap_spec_path()), "--dry-run",
             "--runs-dir", str(self.runs)],
            provider_factory=forbidden, repo_root=self.repo, env=env,
            log=self.log,
        )
        self.assertEqual(code, 0)
        self.assertTrue(self.logged(f"create from snapshot {snap}"))


class TestOrphanCleanup(SnapshotTestCase):
    """A create that raises after the sandbox exists server-side must not
    leave an orphan blocking the concurrency guard."""

    def execute(self, provider, spec=None):
        spec = spec or rd.parse_spec(self.base_raw_spec())
        return rd.execute_run(
            spec, self.repo, self.tmp / "rundir", provider,
            rd.Redactor([FAKE_KEY]), rd.Deadline(),
            log=self.log, sleep_fn=lambda s: None,
        )

    def direct_manifest(self):
        return json.loads((self.tmp / "rundir" / "manifest.json").read_text())

    def test_create_timeout_orphan_is_deleted_and_verified(self):
        provider = OrphanFakeProvider(
            create_exc=rd.LauncherError("create timed out in PULLING_SNAPSHOT"))
        code = self.execute(provider)
        self.assertEqual(code, rd.EXIT_ERROR)  # original error still surfaces
        self.assertEqual(provider.deleted_by_id, ["sb-orphan-1"])
        oc = self.direct_manifest()["orphan_cleanup"]
        self.assertEqual(oc["ids"], ["sb-orphan-1"])
        self.assertTrue(oc["verified_gone"])
        self.assertTrue(self.logged("deleting orphan"))
        self.assertTrue(self.logged("verified gone"))

    def test_orphan_cleanup_unverified_forces_exit_4_and_loud_message(self):
        provider = OrphanFakeProvider(
            create_exc=rd.LauncherError("create timed out"),
            gone_after_delete=False)
        code = self.execute(provider)
        self.assertEqual(code, rd.EXIT_DELETE_UNVERIFIED)
        oc = self.direct_manifest()["orphan_cleanup"]
        self.assertFalse(oc["verified_gone"])
        self.assertTrue(self.logged("ORPHAN SANDBOX CLEANUP NOT VERIFIED"))
        self.assertTrue(self.logged("launcher=duck-grid-walk-gpu"))

    def test_spot_failure_with_no_orphan_keeps_exit_3(self):
        provider = OrphanFakeProvider(
            create_exc=rd.SpotUnavailableError("no spot capacity"),
            orphan_ids=())
        code = self.execute(provider)
        self.assertEqual(code, rd.EXIT_SPOT_UNAVAILABLE)
        self.assertEqual(provider.deleted_by_id, [])
        oc = self.direct_manifest()["orphan_cleanup"]
        self.assertEqual(oc["ids"], [])
        self.assertTrue(oc["verified_gone"])

    def test_orphan_cleanup_also_covers_from_snapshot_create(self):
        provider = OrphanFakeProvider(
            from_snapshot_exc=rd.LauncherError("timeout pulling snapshot"))
        spec = rd.parse_spec(self.base_raw_spec(use_snapshot=True))
        code = rd.execute_run(
            spec, self.repo, self.tmp / "rundir", provider,
            rd.Redactor([FAKE_KEY]), rd.Deadline(),
            log=self.log, sleep_fn=lambda s: None,
            snapshot_name="duck-gpu-prebuilt-x",
        )
        self.assertEqual(code, rd.EXIT_ERROR)
        self.assertEqual(provider.deleted_by_id, ["sb-orphan-1"])

    def test_delete_by_id_error_marks_unverified_not_crash(self):
        class BadDelete(OrphanFakeProvider):
            def delete_by_id(self, sandbox_id):
                raise ConnectionError(f"api 500 {FAKE_KEY}")

        provider = BadDelete(create_exc=rd.LauncherError("boom"),
                             gone_after_delete=False)
        code = self.execute(provider)
        self.assertEqual(code, rd.EXIT_DELETE_UNVERIFIED)
        oc = self.direct_manifest()["orphan_cleanup"]
        self.assertFalse(oc["verified_gone"])
        self.assertTrue(any("delete sb-orphan-1" in e for e in oc["errors"]))
        self.assertNotIn(FAKE_KEY, json.dumps(oc))

    def test_guard_message_suggests_label_cleanup(self):
        provider = SnapshotFakeProvider(existing_ids=["sb-old"])
        code = self.execute(provider)
        self.assertEqual(code, rd.EXIT_ERROR)
        self.assertTrue(self.logged("label launcher=duck-grid-walk-gpu"))
        self.assertTrue(self.logged("Manual cleanup"))


class TestSnapshotActivation(SnapshotTestCase):
    def bake_spec_path(self):
        return self.write_spec(self.base_raw_spec(name="bake-act"))

    def test_bake_activates_snapshot_and_records_state(self):
        provider = SnapshotFakeProvider()
        code = self.call_main(["bake", "--spec", str(self.bake_spec_path())],
                              provider)
        self.assertEqual(code, 0)
        self.assertEqual(provider.activated, provider.snapshots_created)
        self.assertEqual(self.image_manifest()["activated_state"], "active")

    def test_bake_activation_failure_is_not_fatal(self):
        provider = SnapshotFakeProvider(
            activate_exc=rd.LauncherError("activation 500"))
        code = self.call_main(["bake", "--spec", str(self.bake_spec_path())],
                              provider)
        self.assertEqual(code, 0)  # bake still succeeds
        im = self.image_manifest()
        self.assertIsNone(im["activated_state"])
        self.assertTrue(self.logged("activation failed"))
        self.assertTrue(self.logged("warm"))

    def test_warm_activates_manifest_snapshot(self):
        snap = "duck-gpu-prebuilt-20260901-000000"
        p = rd.image_manifest_path(self.repo)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"snapshot": snap}))
        provider = SnapshotFakeProvider()
        code = self.call_main(["warm"], provider)
        self.assertEqual(code, 0)
        self.assertEqual(provider.activated, [snap])
        self.assertTrue(self.logged("state: active"))

    def test_warm_without_manifest_errors(self):
        provider = SnapshotFakeProvider()
        code = self.call_main(["warm"], provider)
        self.assertEqual(code, rd.EXIT_ERROR)
        self.assertEqual(provider.activated, [])
        self.assertTrue(self.logged("run `bake` first"))

    def test_warm_without_key_refuses_before_provider(self):
        p = rd.image_manifest_path(self.repo)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"snapshot": "snap-x"}))

        def forbidden():
            raise AssertionError("provider factory called without key")

        env = {k: v for k, v in os.environ.items() if k != "DAYTONA_API_KEY"}
        code = rd.main(["warm"], provider_factory=forbidden,
                       repo_root=self.repo, env=env, log=self.log)
        self.assertEqual(code, rd.EXIT_ERROR)
        self.assertTrue(self.logged("doppler run"))


if __name__ == "__main__":
    unittest.main()
