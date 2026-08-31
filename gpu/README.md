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
the spec's `gpu_types` (default `["RTX-5090"]`).

## Budget policy (hard limits)

- **One sandbox at a time.** Before creating, the launcher lists sandboxes
  carrying the `launcher=duck-grid-walk-gpu` label and refuses to start if
  any exist.
- **1500 s host wall-clock cap** per invocation. Per-job remote timeouts come
  from the spec (`timeout_s`) and are clipped to the remaining budget. When
  the budget is exhausted, remaining jobs are aborted (exit 5) but deletion
  still runs.
- **TTL 25 min** on the sandbox itself (`ttl_minutes` + `auto_delete_interval`)
  as a server-side safety net in case the launcher dies.
- **Deletion is always attempted in a `finally` block**, then VERIFIED by
  listing sandboxes. A `deletion_receipt.json` is written into the run dir.
  If deletion cannot be verified, the launcher exits 4 with a loud message —
  go delete the sandbox in the Daytona dashboard immediately.

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
| 5 | 1500 s host wall-clock budget exceeded |

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
      "continue_on_error": false      // default false: stop on first failure
    }
  ],
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
  `artifacts` (`ExecutionArtifacts` with `stdout` / charts).
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
