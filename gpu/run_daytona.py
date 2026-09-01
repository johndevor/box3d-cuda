#!/Users/john/.cache/box3d-cuda-host-runtime-0.207.0/bin/python -B
"""Minimal, bounded, safe Daytona launcher for staged GPU jobs.

Runs a job spec (compile / parity / bench / train) on an ephemeral spot GPU
sandbox, downloads artifacts, and always deletes the sandbox (verified).

Two modes:
  run  — execute a spec. If the spec sets "use_snapshot": true and
         gpu/image-manifest.json records a baked snapshot, the sandbox is
         created from that snapshot (deps + prebuilt CUDA libs already in
         place), skipping the ~2-3 min provision/nvcc/pip overhead.
  bake — execute a bake spec (deps + prebuild into /opt/duck/prebuilt/<sha>),
         snapshot the sandbox, verify the snapshot exists, record it in
         gpu/image-manifest.json, then delete the sandbox as usual.

Documented invocation (the API key must come from Doppler, never argv):

    doppler run --project hallway --config dev --only-secrets DAYTONA_API_KEY \
        --no-fallback -- \
        /Users/john/.cache/box3d-cuda-host-runtime-0.207.0/bin/python -B \
        gpu/run_daytona.py run --spec gpu/specs/<name>.json [--dry-run]

Secrets discipline (enforced here):
  * The key is only ever read from the environment by the Daytona SDK itself.
  * It is never printed, never logged, never passed on argv, never exported
    into the sandbox workload environment (remote exec gets env=None).
  * Anything matching the key value or the Daytona key pattern is redacted
    from all captured output before it is written to disk.

Budget policy (hard):
  * One sandbox at a time (checked via launcher label before create).
  * Total host wall clock cap: 3300 s per invocation.
  * Per-job remote timeouts come from the spec, clipped to remaining budget.
  * Deletion is always attempted in a finally block and VERIFIED by listing
    sandboxes afterward; a deletion receipt is written into the run dir.
    Unverified deletion => loud message + exit code 4.

Exit codes:
  0 success | 1 usage/spec/launcher error | 2 job failed | 3 spot capacity
  unavailable (no retry) | 4 sandbox deletion could not be verified |
  5 host wall-clock budget exceeded.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HOST_RUNTIME = "/Users/john/.cache/box3d-cuda-host-runtime-0.207.0/bin/python"
DOPPLER_INVOCATION = (
    "doppler run --project hallway --config dev --only-secrets DAYTONA_API_KEY "
    f"--no-fallback -- {HOST_RUNTIME} -B gpu/run_daytona.py run "
    "--spec gpu/specs/<name>.json [--dry-run]"
)

WALL_CLOCK_CAP_S = 3300  # fits one 40-min training leg + provision/build/transfer
LAUNCHER_LABEL = {"launcher": "duck-grid-walk-gpu"}
# --label-suffix scopes the one-sandbox-at-a-time guard and the deletion
# verification to an exact label value (e.g. duck-grid-walk-gpu-cfg2), so a
# sweep driver may run N configs in N concurrent sandboxes, one per suffix.
LABEL_SUFFIX_RE = re.compile(r"[A-Za-z0-9._\-]{1,32}")


def label_with_suffix(suffix=None):
    """The launcher label dict, optionally with '-<suffix>' appended."""
    label = dict(LAUNCHER_LABEL)
    if suffix:
        label["launcher"] = f"{LAUNCHER_LABEL['launcher']}-{suffix}"
    return label
REMOTE_WORKDIR = "/tmp/duckwork"
REMOTE_TAR = "/tmp/duck-payload.tar"
# Baked-snapshot support: `bake` provisions from the base image, preinstalls
# deps + prebuilds the CUDA libs under this remote prefix, snapshots the
# sandbox and records it here; `run` uses the snapshot when the spec sets
# "use_snapshot": true and the manifest exists.
REMOTE_PREBUILT_DIR = "/opt/duck/prebuilt"
IMAGE_MANIFEST_NAME = "image-manifest.json"
SNAPSHOT_NAME_PREFIX = "duck-gpu-prebuilt"
DEFAULT_IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
KNOWN_GPU_TYPES = ("RTX-5090", "RTX-4090", "H100", "H200", "RTX-PRO-6000")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_JOB_FAILED = 2
EXIT_SPOT_UNAVAILABLE = 3
EXIT_DELETE_UNVERIFIED = 4
EXIT_BUDGET_EXCEEDED = 5


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class LauncherError(Exception):
    """Generic launcher failure (exit 1)."""


class SpecError(LauncherError):
    """The job spec is invalid (exit 1)."""


class SpotUnavailableError(LauncherError):
    """Spot GPU capacity unavailable. Clean typed failure; never retried."""


class ConcurrencyError(LauncherError):
    """Another launcher sandbox already exists; one sandbox at a time."""


class BudgetExceededError(LauncherError):
    """Host wall clock cap (1500 s) exhausted."""


class DeletionUnverifiedError(LauncherError):
    """Sandbox deletion could not be verified by listing."""


# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------

_KEY_PATTERN = re.compile(r"dtn_[A-Za-z0-9_\-]{8,}")


class Redactor:
    """Removes secret values from text before anything touches disk."""

    def __init__(self, secret_values=()):
        self._secrets = [s for s in secret_values if s and len(s) >= 6]

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        return cls([env.get("DAYTONA_API_KEY", "")])

    def redact(self, text):
        if not isinstance(text, str):
            return text
        for s in self._secrets:
            text = text.replace(s, "[REDACTED]")
        return _KEY_PATTERN.sub("[REDACTED]", text)


# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------

@dataclasses.dataclass
class JobSpec:
    name: str
    command: str
    timeout_s: int
    artifacts: list
    continue_on_error: bool = False


@dataclasses.dataclass
class ResourceSpec:
    cpu: int = 4
    memory_gib: int = 16
    disk_gib: int = 20
    gpu: int = 1
    gpu_types: list = dataclasses.field(default_factory=lambda: ["RTX-5090"])
    spot: bool = True
    ttl_minutes: int = 25
    image: str = DEFAULT_IMAGE


@dataclasses.dataclass
class Spec:
    name: str
    tar_globs: list
    upload_extra: list
    jobs: list           # list[JobSpec]
    resources: ResourceSpec
    use_snapshot: bool = False  # create from the baked snapshot in
                                # gpu/image-manifest.json when available


def _require(cond, msg):
    if not cond:
        raise SpecError(msg)


def load_spec(path):
    p = Path(path)
    _require(p.is_file(), f"spec file not found: {path}")
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise SpecError(f"spec is not valid JSON: {e}") from e
    return parse_spec(raw)


def parse_spec(raw):
    _require(isinstance(raw, dict), "spec must be a JSON object")
    _require(isinstance(raw.get("name"), str) and raw["name"], "spec.name must be a non-empty string")
    _require(re.fullmatch(r"[A-Za-z0-9._\-]+", raw["name"]), "spec.name must be filesystem-safe")
    tar_globs = raw.get("tar_globs", [])
    _require(isinstance(tar_globs, list) and all(isinstance(g, str) for g in tar_globs),
             "spec.tar_globs must be a list of strings")
    upload_extra = raw.get("upload_extra", [])
    _require(isinstance(upload_extra, list) and all(isinstance(g, str) for g in upload_extra),
             "spec.upload_extra must be a list of strings")
    jobs_raw = raw.get("jobs")
    _require(isinstance(jobs_raw, list) and jobs_raw, "spec.jobs must be a non-empty list")
    jobs = []
    for i, j in enumerate(jobs_raw):
        _require(isinstance(j, dict), f"jobs[{i}] must be an object")
        _require(isinstance(j.get("name"), str) and j["name"], f"jobs[{i}].name must be a non-empty string")
        _require(re.fullmatch(r"[A-Za-z0-9._\-]+", j["name"]), f"jobs[{i}].name must be filesystem-safe")
        _require(isinstance(j.get("command"), str) and j["command"], f"jobs[{i}].command must be a non-empty string")
        _require(isinstance(j.get("timeout_s"), int) and 0 < j["timeout_s"] <= WALL_CLOCK_CAP_S,
                 f"jobs[{i}].timeout_s must be an int in (0, {WALL_CLOCK_CAP_S}]")
        arts = j.get("artifacts", [])
        _require(isinstance(arts, list) and all(isinstance(a, str) for a in arts),
                 f"jobs[{i}].artifacts must be a list of strings")
        _require(all(not a.startswith("/") and ".." not in a.split("/") for a in arts),
                 f"jobs[{i}].artifacts must be workdir-relative paths without '..'")
        coe = j.get("continue_on_error", False)
        _require(isinstance(coe, bool), f"jobs[{i}].continue_on_error must be a boolean")
        jobs.append(JobSpec(j["name"], j["command"], j["timeout_s"], list(arts), coe))
    names = [j.name for j in jobs]
    _require(len(names) == len(set(names)), "job names must be unique")

    r = raw.get("resources", {})
    _require(isinstance(r, dict), "spec.resources must be an object")
    res = ResourceSpec(
        cpu=r.get("cpu", 4),
        memory_gib=r.get("memory_gib", 16),
        disk_gib=r.get("disk_gib", 20),
        gpu=r.get("gpu", 1),
        gpu_types=list(r.get("gpu_types", ["RTX-5090"])),
        spot=r.get("spot", True),
        ttl_minutes=r.get("ttl_minutes", 25),
        image=r.get("image", DEFAULT_IMAGE),
    )
    for f in ("cpu", "memory_gib", "disk_gib", "gpu", "ttl_minutes"):
        _require(isinstance(getattr(res, f), int) and getattr(res, f) > 0,
                 f"resources.{f} must be a positive int")
    _require(res.ttl_minutes <= 60, "resources.ttl_minutes must be <= 60 (bounded budget)")
    _require(isinstance(res.spot, bool), "resources.spot must be a boolean")
    _require(isinstance(res.image, str) and res.image, "resources.image must be a non-empty string")
    _require(res.gpu_types and all(g in KNOWN_GPU_TYPES for g in res.gpu_types),
             f"resources.gpu_types entries must be one of {KNOWN_GPU_TYPES}")
    use_snapshot = raw.get("use_snapshot", False)
    _require(isinstance(use_snapshot, bool), "spec.use_snapshot must be a boolean")
    return Spec(raw["name"], tar_globs, upload_extra, jobs, res, use_snapshot)


# --------------------------------------------------------------------------
# Payload tar (git archive HEAD + upload_extra)
# --------------------------------------------------------------------------

def build_payload_tar(repo_root, spec, out_path):
    """Build a plain tar from `git archive HEAD -- <tar_globs>` plus
    upload_extra files. Returns the commit SHA. No secrets go anywhere near
    this archive."""
    repo_root = Path(repo_root)
    sha = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    cmd = ["git", "-C", str(repo_root), "archive", "--format=tar", "HEAD"]
    if spec.tar_globs:
        cmd += ["--"] + list(spec.tar_globs)
    proc = subprocess.run(cmd, check=False, capture_output=True)
    if proc.returncode != 0:
        raise LauncherError(
            "git archive failed (bad tar_globs?): "
            + proc.stderr.decode("utf-8", "replace").strip()
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(proc.stdout)
    if spec.upload_extra:
        with tarfile.open(out_path, "a") as tf:
            existing = set(tf.getnames())
            for rel in spec.upload_extra:
                _require(not rel.startswith("/") and ".." not in rel.split("/"),
                         f"upload_extra path must be repo-relative without '..': {rel}")
                src = repo_root / rel
                _require(src.is_file(), f"upload_extra file not found: {rel}")
                if rel not in existing:
                    tf.add(src, arcname=rel)
    return sha


def tar_member_names(tar_path):
    with tarfile.open(tar_path, "r") as tf:
        return [m.name for m in tf.getmembers() if m.isfile()]


# --------------------------------------------------------------------------
# Baked-image manifest (gpu/image-manifest.json)
# --------------------------------------------------------------------------

def image_manifest_path(repo_root):
    return Path(repo_root) / "gpu" / IMAGE_MANIFEST_NAME


def load_image_manifest(path):
    """Return the baked-image manifest dict, or None if absent/unreadable."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or not data.get("snapshot"):
        return None
    return data


# --------------------------------------------------------------------------
# Provider abstraction (real transport lives ONLY here; tests inject a fake)
# --------------------------------------------------------------------------

@dataclasses.dataclass
class SandboxHandle:
    sandbox_id: str
    raw: object = None


@dataclasses.dataclass
class ExecResult:
    exit_code: int
    output: str


class DaytonaProvider:
    """Real transport against daytona SDK 0.207.0. Imports daytona lazily so
    --dry-run and tests never need the SDK or the key. The API key is read
    from the environment by the SDK itself; this class never touches it."""

    def __init__(self, label=None):
        if not os.environ.get("DAYTONA_API_KEY"):
            raise LauncherError(
                "DAYTONA_API_KEY is not set. Run via doppler:\n  " + DOPPLER_INVOCATION
            )
        import daytona  # noqa: F401  (pinned 0.207.0 in the host runtime)
        self._daytona = daytona
        self._client = daytona.Daytona()  # reads DAYTONA_API_KEY from env
        # Exact label value this launcher instance owns: the concurrency guard
        # and the deletion verification are scoped to it (default unchanged).
        self._label = dict(label or LAUNCHER_LABEL)

    def _list_labeled(self):
        """Sandboxes carrying exactly this launcher's label value."""
        d = self._daytona
        try:
            query = d.ListSandboxesQuery(labels=self._label)
            return list(self._client.list(query))
        except TypeError:
            return [b for b in self._client.list()
                    if (getattr(b, "labels", None) or {}).get("launcher")
                    == self._label["launcher"]]

    def list_launcher_sandbox_ids(self):
        boxes = self._list_labeled()
        out = []
        for b in boxes:
            state = str(getattr(b, "state", "")).lower()
            if "destroy" not in state and "delet" not in state:
                out.append(b.id)
        return out

    def create(self, spec, gpu_types, timeout_s):
        d = self._daytona
        gpu_enum = [getattr(d.GpuType, g.replace("-", "_")) for g in gpu_types]
        params = d.CreateSandboxFromImageParams(
            image=spec.resources.image,
            resources=d.Resources(
                cpu=spec.resources.cpu,
                memory=spec.resources.memory_gib,
                disk=spec.resources.disk_gib,
                gpu=spec.resources.gpu,
                gpu_type=gpu_enum,
            ),
            spot=spec.resources.spot,
            ttl_minutes=spec.resources.ttl_minutes,
            # provider requires GPU sandboxes to be ephemeral: 0 = delete on stop;
            # ttl_minutes remains the hard wall bound
            auto_delete_interval=0,
            labels=dict(self._label, spec=spec.name),
        )
        return self._create(params, timeout_s)

    def create_from_snapshot(self, spec, snapshot_name, timeout_s):
        """Create a sandbox from a baked snapshot. Resources/GPU type come
        from the snapshot itself (the SDK's from-snapshot params carry no
        resources); spot/ttl/labels still apply."""
        d = self._daytona
        params = d.CreateSandboxFromSnapshotParams(
            snapshot=snapshot_name,
            spot=spec.resources.spot,
            ttl_minutes=spec.resources.ttl_minutes,
            auto_delete_interval=0,
            labels=dict(self._label, spec=spec.name),
        )
        return self._create(params, timeout_s)

    def _create(self, params, timeout_s):
        d = self._daytona
        try:
            sb = self._client.create(params, timeout=timeout_s)
        except d.DaytonaError as e:
            msg = str(e).lower()
            if any(k in msg for k in ("spot", "capacity", "no available", "insufficient", "unavailable", "quota")):
                raise SpotUnavailableError(f"spot GPU capacity unavailable: {e}") from e
            raise LauncherError(f"sandbox creation failed: {e}") from e
        return SandboxHandle(sandbox_id=sb.id, raw=sb)

    def create_snapshot(self, handle, name, timeout_s):
        """Snapshot the sandbox filesystem into a reusable named snapshot."""
        d = self._daytona
        try:
            handle.raw.create_snapshot(name, timeout=timeout_s)
        except d.DaytonaError as e:
            raise LauncherError(f"snapshot creation failed: {e}") from e

    def snapshot_exists(self, name):
        """Return the snapshot id if the named snapshot exists, else None."""
        d = self._daytona
        try:
            snap = self._client.snapshot.get(name)
        except d.DaytonaNotFoundError:
            return None
        except d.DaytonaError as e:
            raise LauncherError(f"snapshot lookup failed: {e}") from e
        return getattr(snap, "id", None) or name

    def upload(self, handle, local_path, remote_path):
        handle.raw.fs.upload_file(str(local_path), remote_path)

    def exec(self, handle, command, timeout_s):
        # env=None on purpose: the workload must never see host secrets.
        resp = handle.raw.process.exec(command, env=None, timeout=int(timeout_s))
        return ExecResult(exit_code=int(resp.exit_code), output=resp.result or "")

    def download(self, handle, remote_path):
        try:
            return handle.raw.fs.download_file(remote_path)
        except self._daytona.DaytonaError:
            return None

    def delete(self, handle):
        self._client.delete(handle.raw, wait=True)

    def sandbox_gone(self, sandbox_id):
        for b in self._list_labeled():
            if b.id == sandbox_id:
                state = str(getattr(b, "state", "")).lower()
                return "destroy" in state or "delet" in state
        return True


# --------------------------------------------------------------------------
# Deadline
# --------------------------------------------------------------------------

class Deadline:
    def __init__(self, cap_s=WALL_CLOCK_CAP_S, clock=time.monotonic):
        self._clock = clock
        self._end = clock() + cap_s

    def remaining(self):
        return self._end - self._clock()

    def check(self, what):
        if self.remaining() <= 0:
            raise BudgetExceededError(
                f"host wall clock cap of {WALL_CLOCK_CAP_S}s exhausted before: {what}"
            )

    def clip(self, timeout_s):
        return max(1, min(int(timeout_s), int(self.remaining())))


# --------------------------------------------------------------------------
# Run flow
# --------------------------------------------------------------------------

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_text(path, text, redactor):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(redactor.redact(text))


def _write_json(path, obj, redactor):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(redactor.redact(json.dumps(obj, indent=2, sort_keys=True)) + "\n")


def execute_run(spec, repo_root, run_dir, provider, redactor, deadline,
                extra_gpu_types=(), log=print, label=None,
                snapshot_name=None, post_success_hook=None):
    """Full lifecycle: tar -> create -> upload -> jobs -> artifacts ->
    manifest -> delete (finally, verified). Returns process exit code.

    snapshot_name: create the sandbox from this baked snapshot instead of
    the spec's base image (resources then come from the snapshot).
    post_success_hook(handle, run_dir, manifest): runs after all jobs
    succeed, before deletion — used by `bake` to snapshot the sandbox."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tar_path = run_dir / "payload.tar"
    commit = build_payload_tar(repo_root, spec, tar_path)

    gpu_types = list(spec.resources.gpu_types)
    for g in extra_gpu_types:
        if g not in gpu_types:
            gpu_types.append(g)

    manifest = {
        "spec": dataclasses.asdict(spec),
        "commit": commit,
        "label": dict(label or LAUNCHER_LABEL),
        "gpu_types": gpu_types,
        "snapshot_used": snapshot_name,
        "sandbox_id": None,
        "payload_tar_sha256": _sha256(tar_path),
        "jobs": [],
        "deletion": None,
        "exit_code": None,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    handle = None
    exit_code = EXIT_OK
    try:
        deadline.check("sandbox creation")
        existing = provider.list_launcher_sandbox_ids()
        if existing:
            raise ConcurrencyError(
                f"another launcher sandbox already exists ({existing}); "
                "one sandbox at a time — delete it first."
            )
        if snapshot_name:
            log(f"[launcher] creating sandbox from snapshot {snapshot_name} "
                f"(spot={spec.resources.spot}, ttl={spec.resources.ttl_minutes}m)")
            handle = provider.create_from_snapshot(
                spec, snapshot_name, timeout_s=deadline.clip(300))
        else:
            log(f"[launcher] creating sandbox (gpu_types={gpu_types}, "
                f"image={spec.resources.image}, spot={spec.resources.spot}, "
                f"ttl={spec.resources.ttl_minutes}m)")
            handle = provider.create(spec, gpu_types, timeout_s=deadline.clip(300))
        manifest["sandbox_id"] = handle.sandbox_id
        log(f"[launcher] sandbox created: {handle.sandbox_id}")

        deadline.check("payload upload")
        provider.upload(handle, tar_path, REMOTE_TAR)
        unpack = provider.exec(
            handle,
            # rm -rf first: a baked snapshot may carry the bake's stale
            # workdir; runs must start from exactly the uploaded payload
            f"rm -rf {REMOTE_WORKDIR} && mkdir -p {REMOTE_WORKDIR} && "
            f"tar -xf {REMOTE_TAR} -C {REMOTE_WORKDIR}",
            timeout_s=deadline.clip(120),
        )
        if unpack.exit_code != 0:
            raise LauncherError(f"payload unpack failed: {redactor.redact(unpack.output)[:2000]}")

        stop = False
        for job in spec.jobs:
            if stop:
                manifest["jobs"].append({"name": job.name, "skipped": True})
                continue
            deadline.check(f"job {job.name}")
            t0 = time.monotonic()
            log(f"[launcher] job {job.name}: {job.command}")
            result = provider.exec(
                handle,
                f"cd {REMOTE_WORKDIR} && ({job.command})",
                timeout_s=deadline.clip(job.timeout_s),
            )
            wall = round(time.monotonic() - t0, 2)
            log_file = run_dir / "logs" / f"{job.name}.log"
            _write_text(log_file, result.output, redactor)

            arts = {}
            for rel in job.artifacts:
                data = provider.download(handle, f"{REMOTE_WORKDIR}/{rel}")
                if data is None:
                    arts[rel] = {"status": "missing"}
                    continue
                dest = run_dir / "artifacts" / job.name / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                arts[rel] = {
                    "status": "ok",
                    "path": str(dest),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            manifest["jobs"].append({
                "name": job.name,
                "exit_code": result.exit_code,
                "wall_time_s": wall,
                "log": str(log_file),
                "artifacts": arts,
            })
            if result.exit_code != 0:
                log(f"[launcher] job {job.name} FAILED (exit {result.exit_code})")
                exit_code = EXIT_JOB_FAILED
                if not job.continue_on_error:
                    stop = True
            else:
                log(f"[launcher] job {job.name} ok ({wall}s)")
        if exit_code == EXIT_OK and post_success_hook is not None:
            post_success_hook(handle, run_dir, manifest)
    except SpotUnavailableError as e:
        log(f"[launcher] SPOT UNAVAILABLE: {redactor.redact(str(e))} (no retry, by policy)")
        exit_code = EXIT_SPOT_UNAVAILABLE
    except BudgetExceededError as e:
        log(f"[launcher] BUDGET EXCEEDED: {redactor.redact(str(e))}")
        exit_code = EXIT_BUDGET_EXCEEDED
    except LauncherError as e:
        log(f"[launcher] ERROR: {redactor.redact(str(e))}")
        exit_code = EXIT_ERROR
    finally:
        receipt = {
            "sandbox_id": handle.sandbox_id if handle else None,
            "delete_attempted": False,
            "delete_error": None,
            "verified_gone": None,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if handle is not None:
            receipt["delete_attempted"] = True
            try:
                provider.delete(handle)
            except Exception as e:  # still verify below
                receipt["delete_error"] = redactor.redact(str(e))
            try:
                # the listing is eventually-consistent: poll up to 30 s before
                # declaring the deletion unverified
                gone = False
                for _ in range(10):
                    gone = bool(provider.sandbox_gone(handle.sandbox_id))
                    if gone:
                        break
                    time.sleep(3)
                receipt["verified_gone"] = gone
            except Exception as e:
                receipt["delete_error"] = (receipt["delete_error"] or "") + \
                    f" | verification listing failed: {redactor.redact(str(e))}"
                receipt["verified_gone"] = False
            if not receipt["verified_gone"]:
                log("=" * 70)
                log(f"[launcher] !!! SANDBOX DELETION NOT VERIFIED: {handle.sandbox_id}")
                log("[launcher] !!! It may still be running and BILLING. Delete it")
                log("[launcher] !!! manually in the Daytona dashboard NOW.")
                log("=" * 70)
                exit_code = EXIT_DELETE_UNVERIFIED
            else:
                log(f"[launcher] sandbox {handle.sandbox_id} deleted and verified gone")
        manifest["deletion"] = receipt
        manifest["exit_code"] = exit_code
        _write_json(run_dir / "deletion_receipt.json", receipt, redactor)
        _write_json(run_dir / "manifest.json", manifest, redactor)
        log(f"[launcher] run dir: {run_dir}")
    return exit_code


# --------------------------------------------------------------------------
# Bake (image snapshot with deps + prebuilt CUDA libs)
# --------------------------------------------------------------------------

def make_bake_hook(spec, provider, redactor, deadline, manifest_path,
                   snapshot_name, log=print):
    """post_success_hook for `bake`: snapshot the sandbox, verify the
    snapshot exists, and record it (plus the prebuilt source sha reported by
    the bake job's prebuilt-sha.txt artifact) in gpu/image-manifest.json."""

    def hook(handle, run_dir, manifest):
        deadline.check("snapshot creation")
        log(f"[bake] snapshotting sandbox {handle.sandbox_id} as {snapshot_name}")
        provider.create_snapshot(handle, snapshot_name, timeout_s=deadline.clip(900))
        snapshot_id = provider.snapshot_exists(snapshot_name)
        if not snapshot_id:
            raise LauncherError(
                f"snapshot {snapshot_name} not found after creation; "
                "NOT recording it in the image manifest"
            )
        source_sha = None
        for sha_file in sorted(Path(run_dir).glob("artifacts/*/prebuilt-sha.txt")):
            source_sha = sha_file.read_text().strip() or None
        image_manifest = {
            "snapshot": snapshot_name,
            "snapshot_id": snapshot_id,
            "source_sha": source_sha,
            "commit": manifest.get("commit"),
            "base_image": spec.resources.image,
            "bake_spec": spec.name,
            "prebuilt_dir": REMOTE_PREBUILT_DIR,
            "baked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        _write_json(Path(manifest_path), image_manifest, redactor)
        manifest["image_manifest"] = image_manifest
        log(f"[bake] snapshot verified ({snapshot_id}); wrote {manifest_path}")
        if source_sha:
            log(f"[bake] prebuilt source sha: {source_sha}")
        else:
            log("[bake] warning: no prebuilt-sha.txt artifact found; "
                "source_sha recorded as null (build guards will always rebuild)")

    return hook


# --------------------------------------------------------------------------
# Dry run
# --------------------------------------------------------------------------

def dry_run(spec, repo_root, run_dir, log=print):
    """Validate the spec and build the tar. ZERO provider calls; works with
    no DAYTONA_API_KEY in the environment."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tar_path = run_dir / "payload.tar"
    commit = build_payload_tar(repo_root, spec, tar_path)
    names = tar_member_names(tar_path)
    log(f"[dry-run] spec: {spec.name}")
    log(f"[dry-run] commit: {commit}")
    log(f"[dry-run] payload: {tar_path} ({len(names)} files, "
        f"sha256={_sha256(tar_path)[:16]}...)")
    log(f"[dry-run] image: {spec.resources.image}")
    log(f"[dry-run] resources: cpu={spec.resources.cpu} mem={spec.resources.memory_gib}GiB "
        f"disk={spec.resources.disk_gib}GiB gpu={spec.resources.gpu} "
        f"types={spec.resources.gpu_types} spot={spec.resources.spot} "
        f"ttl={spec.resources.ttl_minutes}m")
    for j in spec.jobs:
        log(f"[dry-run] job {j.name}: timeout={j.timeout_s}s "
            f"continue_on_error={j.continue_on_error} artifacts={j.artifacts}")
        log(f"[dry-run]   $ {j.command}")
    log("[dry-run] plan OK. No provider calls were made.")
    return EXIT_OK


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _build_parser():
    ap = argparse.ArgumentParser(
        prog="run_daytona.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Documented invocation:\n  " + DOPPLER_INVOCATION,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd, help_text in (
        ("run", "run a job spec on an ephemeral GPU sandbox"),
        ("bake", "run a bake spec, snapshot the sandbox, and record the "
                 "snapshot in gpu/image-manifest.json"),
    ):
        p = sub.add_parser(
            cmd, help=help_text,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="Documented invocation:\n  " + DOPPLER_INVOCATION,
        )
        p.add_argument("--spec", required=True, help="path to spec JSON (gpu/specs/<name>.json)")
        p.add_argument("--dry-run", action="store_true",
                       help="validate spec + build tar + print plan; zero provider calls")
        p.add_argument("--gpu-type", action="append", default=[], choices=list(KNOWN_GPU_TYPES),
                       help="additional GPU type fallback (e.g. RTX-4090); may repeat")
        p.add_argument("--runs-dir", default=None,
                       help="base dir for run outputs (default: <repo>/runs/gpu)")
        p.add_argument("--label-suffix", default=None,
                       help="append '-<slug>' to the launcher label value and scope "
                            "the concurrency check + deletion verification to that "
                            "exact label (enables N concurrent sweep sandboxes)")
        if cmd == "run":
            p.add_argument("--no-snapshot", action="store_true",
                           help="ignore gpu/image-manifest.json and provision from "
                                "the base image even if the spec sets use_snapshot")
    return ap


def main(argv=None, provider_factory=None, repo_root=None, env=None, log=print):
    args = _build_parser().parse_args(argv)
    env = os.environ if env is None else env
    repo_root = Path(repo_root or Path(__file__).resolve().parent.parent)

    try:
        spec = load_spec(args.spec)
    except SpecError as e:
        log(f"[launcher] spec error: {e}")
        return EXIT_ERROR

    if args.label_suffix is not None and not LABEL_SUFFIX_RE.fullmatch(args.label_suffix):
        log("[launcher] --label-suffix must match [A-Za-z0-9._-]{1,32}")
        return EXIT_ERROR
    label = label_with_suffix(args.label_suffix)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    runs_base = Path(args.runs_dir) if args.runs_dir else repo_root / "runs" / "gpu"
    run_dir = runs_base / f"{stamp}-{spec.name}"
    manifest_path = image_manifest_path(repo_root)

    # snapshot selection (run only; bake always provisions from the base image)
    snapshot_name = None
    if args.cmd == "run" and spec.use_snapshot and not getattr(args, "no_snapshot", False):
        im = load_image_manifest(manifest_path)
        if im:
            snapshot_name = im["snapshot"]
            log(f"[launcher] spec requests snapshot; using {snapshot_name} "
                f"(baked source sha: {im.get('source_sha')})")
        else:
            log(f"[launcher] spec requests snapshot but {manifest_path} is "
                "missing/empty; falling back to the base image (run `bake` "
                "to create one)")

    if args.dry_run:
        try:
            code = dry_run(spec, repo_root, run_dir, log=log)
            if args.cmd == "bake":
                log(f"[dry-run] bake would snapshot as {SNAPSHOT_NAME_PREFIX}-{stamp} "
                    f"and write {manifest_path}")
            elif snapshot_name:
                log(f"[dry-run] run would create from snapshot {snapshot_name}")
            return code
        except LauncherError as e:
            log(f"[dry-run] error: {e}")
            return EXIT_ERROR

    if not env.get("DAYTONA_API_KEY"):
        log("[launcher] DAYTONA_API_KEY is not set. Run via doppler:")
        log("  " + DOPPLER_INVOCATION)
        return EXIT_ERROR

    redactor = Redactor.from_env(env)
    deadline = Deadline(WALL_CLOCK_CAP_S)
    try:
        # test factories take no args; the real provider is scoped to the label
        provider = provider_factory() if provider_factory else DaytonaProvider(label=label)
    except LauncherError as e:
        log(f"[launcher] {redactor.redact(str(e))}")
        return EXIT_ERROR

    post_success_hook = None
    if args.cmd == "bake":
        post_success_hook = make_bake_hook(
            spec, provider, redactor, deadline, manifest_path,
            snapshot_name=f"{SNAPSHOT_NAME_PREFIX}-{stamp}", log=log,
        )
    return execute_run(
        spec, repo_root, run_dir, provider, redactor, deadline,
        extra_gpu_types=args.gpu_type, log=log, label=label,
        snapshot_name=snapshot_name, post_success_hook=post_success_hook,
    )


if __name__ == "__main__":
    sys.exit(main())
