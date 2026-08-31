#!/Users/john/.cache/box3d-cuda-host-runtime-0.207.0/bin/python -B
"""Minimal, bounded, safe Daytona launcher for staged GPU jobs.

Runs a job spec (compile / parity / bench / train) on an ephemeral spot GPU
sandbox, downloads artifacts, and always deletes the sandbox (verified).

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
  * Total host wall clock cap: 1500 s per invocation.
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

WALL_CLOCK_CAP_S = 1500
LAUNCHER_LABEL = {"launcher": "duck-grid-walk-gpu"}
REMOTE_WORKDIR = "/tmp/duckwork"
REMOTE_TAR = "/tmp/duck-payload.tar"
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
    return Spec(raw["name"], tar_globs, upload_extra, jobs, res)


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

    def __init__(self):
        if not os.environ.get("DAYTONA_API_KEY"):
            raise LauncherError(
                "DAYTONA_API_KEY is not set. Run via doppler:\n  " + DOPPLER_INVOCATION
            )
        import daytona  # noqa: F401  (pinned 0.207.0 in the host runtime)
        self._daytona = daytona
        self._client = daytona.Daytona()  # reads DAYTONA_API_KEY from env

    def list_launcher_sandbox_ids(self):
        d = self._daytona
        try:
            query = d.ListSandboxesQuery(labels=LAUNCHER_LABEL)
            boxes = list(self._client.list(query))
        except TypeError:
            boxes = [b for b in self._client.list()
                     if (getattr(b, "labels", None) or {}).get("launcher") == LAUNCHER_LABEL["launcher"]]
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
            auto_delete_interval=spec.resources.ttl_minutes,
            labels=dict(LAUNCHER_LABEL, spec=spec.name),
        )
        try:
            sb = self._client.create(params, timeout=timeout_s)
        except d.DaytonaError as e:
            msg = str(e).lower()
            if any(k in msg for k in ("spot", "capacity", "no available", "insufficient", "unavailable", "quota")):
                raise SpotUnavailableError(f"spot GPU capacity unavailable: {e}") from e
            raise LauncherError(f"sandbox creation failed: {e}") from e
        return SandboxHandle(sandbox_id=sb.id, raw=sb)

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
        for b in self._client.list():
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
                extra_gpu_types=(), log=print):
    """Full lifecycle: tar -> create -> upload -> jobs -> artifacts ->
    manifest -> delete (finally, verified). Returns process exit code."""
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
        "gpu_types": gpu_types,
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
        log(f"[launcher] creating spot sandbox (gpu_types={gpu_types}, "
            f"image={spec.resources.image}, ttl={spec.resources.ttl_minutes}m)")
        handle = provider.create(spec, gpu_types, timeout_s=deadline.clip(300))
        manifest["sandbox_id"] = handle.sandbox_id
        log(f"[launcher] sandbox created: {handle.sandbox_id}")

        deadline.check("payload upload")
        provider.upload(handle, tar_path, REMOTE_TAR)
        unpack = provider.exec(
            handle,
            f"mkdir -p {REMOTE_WORKDIR} && tar -xf {REMOTE_TAR} -C {REMOTE_WORKDIR}",
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
                receipt["verified_gone"] = bool(provider.sandbox_gone(handle.sandbox_id))
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
    run = sub.add_parser(
        "run", help="run a job spec on an ephemeral spot GPU sandbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Documented invocation:\n  " + DOPPLER_INVOCATION,
    )
    run.add_argument("--spec", required=True, help="path to spec JSON (gpu/specs/<name>.json)")
    run.add_argument("--dry-run", action="store_true",
                     help="validate spec + build tar + print plan; zero provider calls")
    run.add_argument("--gpu-type", action="append", default=[], choices=list(KNOWN_GPU_TYPES),
                     help="additional GPU type fallback (e.g. RTX-4090); may repeat")
    run.add_argument("--runs-dir", default=None,
                     help="base dir for run outputs (default: <repo>/runs/gpu)")
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

    stamp = time.strftime("%Y%m%d-%H%M%S")
    runs_base = Path(args.runs_dir) if args.runs_dir else repo_root / "runs" / "gpu"
    run_dir = runs_base / f"{stamp}-{spec.name}"

    if args.dry_run:
        try:
            return dry_run(spec, repo_root, run_dir, log=log)
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
        provider = (provider_factory or DaytonaProvider)()
    except LauncherError as e:
        log(f"[launcher] {redactor.redact(str(e))}")
        return EXIT_ERROR
    return execute_run(
        spec, repo_root, run_dir, provider, redactor, deadline,
        extra_gpu_types=args.gpu_type, log=log,
    )


if __name__ == "__main__":
    sys.exit(main())
