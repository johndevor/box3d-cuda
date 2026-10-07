# gpu/ — bounded Daytona GPU launcher

Runs staged GPU jobs (compile / parity / bench / train) from a JSON job spec
on an **ephemeral spot RTX 5090 sandbox**, downloads artifacts, and always
deletes the sandbox with verification.

## Invocation

The API key lives in Doppler (project `hallway`, config `dev`, secret
`DAYTONA_API_KEY`). The launcher reads it from the environment only — never
pass it on argv, never export it into the sandbox. The exact command:

```sh
doppler run --project hallway --config dev --only-secrets DAYTONA_API_KEY --no-fallback -- \
  /Users/john/.cache/box3d-cuda-host-runtime-0.207.0/bin/python -B \
  gpu/run_daytona.py run --spec gpu/specs/compile-duck-cuda.json
```

Dry run (validates spec, builds the payload tar, prints the plan, **zero
provider calls**, works without the key):

```sh
/Users/john/.cache/box3d-cuda-host-runtime-0.207.0/bin/python -B \
  gpu/run_daytona.py run --spec gpu/specs/compile-duck-cuda.json --dry-run
```

Optional GPU fallback: `--gpu-type RTX-4090` appends a fallback type after
the spec's `gpu_types` (default `["RTX-5090"]`). `--label-suffix <slug>`
scopes the one-sandbox guard to `duck-grid-walk-gpu-<slug>` so a sweep can
run several configs concurrently, one suffix each.

## Baked snapshot fast path (skip provision/nvcc/pip overhead)

Fresh runs pay ~2-3 min for image pull + nvcc build + `pip install numpy`.
The `bake` mode eliminates that by snapshotting a fully prepared sandbox:

```sh
doppler run --project hallway --config dev --only-secrets DAYTONA_API_KEY --no-fallback -- \
  /Users/john/.cache/box3d-cuda-host-runtime-0.207.0/bin/python -B \
  gpu/run_daytona.py bake --spec gpu/specs/bake-image.json
```

What bake does:
1. Provisions from the base image and runs the bake spec's jobs: capture
   `nvidia-smi`, `pip install numpy` for `python3`, build
   `experimental/duck_cuda` from git HEAD, and store the built
   `libduck_cuda*.so` under `/opt/duck/prebuilt/<source-sha>/` (the sha is a
   content hash of the `experimental/duck_cuda` tree, reported back via the
   `prebuilt-sha.txt` artifact).
2. Snapshots the sandbox (`Sandbox.create_snapshot`), verifies the snapshot
   exists by name, and records `{snapshot, snapshot_id, source_sha, commit,
   base_image, baked_at}` in `gpu/image-manifest.json`.
3. Deletes the sandbox with the usual verified-deletion policy.

Using it: a spec that sets `"use_snapshot": true` (all train specs do) is
created from the manifest's snapshot instead of the base image whenever
`gpu/image-manifest.json` exists. `--no-snapshot` forces the base image.
Resources/GPU type are then inherited from the snapshot (the SDK's
from-snapshot params carry no resource overrides); spot/TTL/labels still
come from the spec.

The train specs' `build` jobs are guarded: they recompute the source hash of
the uploaded `experimental/duck_cuda` tree and, if
`/opt/duck/prebuilt/<sha>/libduck_cuda*.so` exists, copy the cached libs and
**skip nvcc** ("prebuilt cache HIT"). A mismatch is normal — it just means
sources changed since the bake — and triggers a regular rebuild ("prebuilt
cache MISS"). Likewise numpy: the train commands' `import numpy || pip
install numpy` guard becomes a no-op on a baked snapshot.

Refresh workflow: re-run the bake whenever `experimental/duck_cuda` or the
base image changes materially (a stale snapshot still works — builds just
fall back to nvcc). The bake overwrites `gpu/image-manifest.json`; old
snapshots can be pruned in the Daytona dashboard. Deleting
`gpu/image-manifest.json` disables the fast path entirely.

Cold pulls and `warm`: idle snapshots transition to `inactive`
(SnapshotState) and the next from-snapshot create pays a slow
`PULLING_SNAPSHOT` (it once exceeded the create timeout). Mitigations:
from-snapshot creation uses a 600 s create timeout (vs 300 s from image),
bake ends with a best-effort `snapshot.activate()` (recorded as
`activated_state` in the manifest), and you can re-pin on demand before a
run session:

```sh
doppler run --project hallway --config dev --only-secrets DAYTONA_API_KEY --no-fallback -- \
  /Users/john/.cache/box3d-cuda-host-runtime-0.207.0/bin/python -B \
  gpu/run_daytona.py warm
```

Orphan protection: if `Daytona.create` itself raises (e.g. timeout while
the sandbox is still provisioning/pulling), the sandbox can exist
server-side without the launcher ever holding a handle. On ANY create
exception the launcher now lists sandboxes by its exact label, deletes
whatever it finds, and verifies — recorded as `orphan_cleanup` in
`manifest.json`. If that verification fails, exit code 4 with a loud
message. Manual cleanup is always: delete sandboxes labeled
`launcher=duck-grid-walk-gpu[-<suffix>]` in the Daytona dashboard.

## Budget policy (hard limits)

- **One sandbox at a time.** Before creating, the launcher lists sandboxes
  carrying the `launcher=duck-grid-walk-gpu` label and refuses to start if
  any exist.
- **3300 s host wall-clock cap** per invocation. Per-job remote timeouts come
  from the spec (`timeout_s`) and are clipped to the remaining budget. When
  the budget is exhausted, remaining jobs are aborted (exit 5) but deletion
  still runs.
- **TTL 25 min** on the sandbox itself (`ttl_minutes` + `auto_delete_interval`)
  as a server-side safety net in case the launcher dies.
- **Deletion is always attempted in a `finally` block**, then VERIFIED by
  listing sandboxes. A `deletion_receipt.json` is written into the run dir.
  If deletion cannot be verified, the launcher exits 4 with a loud message —
  go delete the sandbox in the Daytona dashboard immediately.

## Detached execution for long jobs (reliability)

A single long-lived `process.exec` stream is fragile: a transient proxy read
timeout (`DaytonaConnectionTimeoutError`) once killed a 40-min train leg.
Jobs with `timeout_s > 600` — or any job that sets `"detached": true`
(`"detached": false` opts a long job out) — therefore run detached:

1. The command is started server-side via the SDK session API
   (`process.create_session` + `execute_session_command(run_async=True)`),
   so the workload survives any client/proxy hiccup. No nohup/pid files
   needed — the session tracks the exit code.
2. The launcher polls every ~15 s with short calls
   (`get_session_command(...).exit_code`, `None` while running). Transient
   poll failures are RETRIED; only 5 *consecutive* failures abort the leg.
   The counter resets on any successful poll.
3. On completion the full log is fetched via `get_session_command_logs`
   (also retried, best effort) and written redacted as usual.
4. The job's `timeout_s` is still enforced (exit code 124 on expiry), and
   the 3300 s host budget still bounds everything.

Artifact salvage: artifacts are downloaded per job even when the job fails,
and a best-effort salvage pass runs before deletion for any job whose
artifacts were never collected (exec raised, polling gave up, job skipped).
Salvaged files land in the normal `artifacts/<job>/` layout and are recorded
under `salvaged_artifacts` in `manifest.json` with per-file
`ok`/`missing`/`error` status — a broken stream can no longer lose a
training leg's checkpoints.

## Spot-failure behavior

Spot capacity for RTX-5090 can be unavailable. Creation errors whose message
indicates capacity/spot problems are surfaced as a typed
`SpotUnavailableError` → clean **exit code 3**, no retry loop, no sandbox
left behind. Just re-run later or add `--gpu-type RTX-4090`.

## Exit codes

| code | meaning |
|------|---------|
| 0 | success |
| 1 | usage / spec / launcher error (incl. missing key, concurrency guard) |
| 2 | a job exited nonzero |
| 3 | spot capacity unavailable (typed, no retry) |
| 4 | sandbox deletion could not be verified — act immediately |
| 5 | 3300 s host wall-clock budget exceeded |

## Secrets discipline (enforced in code)

- The key is read from the environment by the Daytona SDK itself
  (`Daytona()` env fallback); the launcher never holds, prints, or logs it.
- Remote `process.exec` is called with `env=None` — the workload never sees
  host secrets.
- Everything written to disk (job logs, manifest, receipts, error messages)
  passes through a redactor that removes the key value and anything matching
  the `dtn_...` key pattern.
- No `doppler secrets download`; use `doppler run --only-secrets` as above.

## Spec format (`gpu/specs/<name>.json`)

```jsonc
{
  "name": "my-spec",                  // filesystem-safe
  "tar_globs": ["csrc", "experimental"], // git pathspecs for `git archive HEAD`;
                                          // [] = whole tracked tree
  "upload_extra": ["local/file.bin"], // optional repo-relative files appended
                                      // to the tar (may be untracked)
  "jobs": [
    {
      "name": "build",                // filesystem-safe, unique
      "command": "cmake ... && ...",  // runs via `cd /tmp/duckwork && (cmd)`
      "timeout_s": 600,               // remote timeout, <= 1500
      "artifacts": ["build.log"],     // workdir-relative, downloaded per job
      "continue_on_error": false,     // default false: stop on first failure
      "detached": true                // optional; default: auto-detach when
                                      // timeout_s > 600 (see above)
    }
  ],
  "use_snapshot": false,              // true = create from the baked snapshot
                                      // in gpu/image-manifest.json when present
  "resources": {                      // all optional; defaults shown
    "cpu": 4, "memory_gib": 16, "disk_gib": 20, "gpu": 1,
    "gpu_types": ["RTX-5090"], "spot": true, "ttl_minutes": 25,
    "image": "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
  }
}
```

To add a spec: drop the JSON in `gpu/specs/`, check it with `--dry-run`
(prints the file list and plan), then run via the doppler invocation above.
Job semantics: jobs run sequentially in the payload workdir; the first
nonzero exit stops the run unless that job sets `continue_on_error`;
artifacts are downloaded even for failed jobs.

Outputs land in `runs/gpu/<timestamp>-<name>/`:
`payload.tar`, `logs/<job>.log` (redacted), `artifacts/<job>/<path>`,
`manifest.json` (spec, commit SHA, sandbox id, per-job exit codes / wall
times / artifact SHA256s, deletion receipt), `deletion_receipt.json`.

## Daytona SDK 0.207.0 surface actually used

Verified against the pinned host runtime
`/Users/john/.cache/box3d-cuda-host-runtime-0.207.0/bin/python`
(`import daytona` works there without the key):

- `daytona.Daytona()` — reads `DAYTONA_API_KEY` (+ optional
  `DAYTONA_API_URL`, `DAYTONA_TARGET`) from the environment.
- `Daytona.create(params, timeout=...) -> Sandbox` with
  `CreateSandboxFromImageParams(image=..., resources=Resources(cpu, memory,
  disk, gpu, gpu_type=[GpuType.RTX_5090, ...]), spot=True, ttl_minutes=25,
  auto_delete_interval=25, labels={...})`.
- `Daytona.list(ListSandboxesQuery(labels={...})) -> Iterator[Sandbox]` —
  used for the concurrency guard and deletion verification.
- `Daytona.delete(sandbox, wait=True)` (also `Sandbox.delete()`).
- `Sandbox.fs.upload_file(src, dst)` / `Sandbox.fs.download_file(path) ->
  bytes` (plus `upload_files`, `download_files` for batches).
- `Sandbox.process.exec(command, cwd=None, env=None, timeout=None) ->
  ExecuteResponse` with fields `exit_code`, `result` (combined output),
  `artifacts` (`ExecutionArtifacts` with `stdout` / charts) — used for
  short jobs only.
- Sessions (detached long jobs): `process.create_session(session_id)`;
  `process.execute_session_command(session_id,
  SessionExecuteRequest(command=..., run_async=True), timeout=...) ->
  SessionExecuteResponse` (`cmd_id`); `process.get_session_command(
  session_id, cmd_id) -> Command` (`exit_code` is `None` while running);
  `process.get_session_command_logs(session_id, cmd_id) ->
  SessionCommandLogsResponse` (`output`/`stdout`/`stderr`).
- Snapshots: `Sandbox.create_snapshot(name, timeout=...)` captures the
  sandbox filesystem into a named snapshot;
  `CreateSandboxFromSnapshotParams(snapshot=<name>, spot=..., ttl_minutes=...,
  labels=...)` creates from it (no `resources` field — cpu/mem/disk/gpu/
  gpu_type are carried by the snapshot, confirmed by the `Snapshot` model's
  fields); `Daytona.snapshot` is a `SnapshotService` with
  `get(name) -> Snapshot`, `list(...)`, `delete(snapshot)`,
  `create(CreateSnapshotParams, ...)`, and
  `activate(snapshot_or_name) -> Snapshot` (re-pins an `inactive` snapshot;
  `SnapshotState` values include `active`, `inactive`, `pulling`,
  `snapshotting`, `error`, `removing`). `activate` takes no region
  argument, so a single call suffices — bake calls it automatically and
  the `warm` subcommand re-runs it on demand.
- Typed errors: `daytona.DaytonaError` base, plus `DaytonaTimeoutError`,
  `DaytonaNotFoundError`, `DaytonaRateLimitError`, etc.
- `GpuType` is a str enum: `RTX_5090 = "RTX-5090"`, `RTX_4090`, `H100`,
  `H200`, `RTX_PRO_6000`.

## Tests

No network, no SDK, no key — a fake provider is injected:

```sh
.venv/bin/python -m unittest discover -s gpu/tests -v
```

## First real spec: `compile-duck-cuda`

Uploads the whole tracked tree and runs: `nvidia-smi` + `nvcc --version`
capture → build of `experimental/duck_cuda` (`build_remote.sh` if present,
else CMake configure+build with `CMAKE_CUDA_ARCHITECTURES=120`, else a
documented placeholder echo since the engine has not landed yet) → smoke
run of any `test_*` / `*_test` binary found (continue_on_error, so a missing
test binary never fails the run). Adjust the `build` / `smoke-test` commands
in the spec once the CUDA engine lands.
