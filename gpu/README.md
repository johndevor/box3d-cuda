# gpu/ — bounded Daytona GPU launcher

Runs staged GPU jobs (compile / parity / bench / train) from a JSON job spec
on an **ephemeral GPU sandbox**, downloads artifacts, and always
deletes the sandbox with verification.

Remote execution is opt-in and paid. The checked-in specs currently request
one RTX 5090 with `spot: false`; the ResourceSpec default is spot. Read the
selected JSON before launching. Neither unit tests nor CPU verification
allocate remote resources.

## Invocation

Supply `DAYTONA_API_KEY` through your secret manager. The launcher reads it
from the environment only — never
pass it on argv, never export it into the sandbox. The exact command:

```sh
doppler run --project YOUR_PROJECT --config YOUR_CONFIG --only-secrets DAYTONA_API_KEY --no-fallback -- \
  python -B \
  gpu/run_daytona.py run --spec gpu/specs/compile-duck-cuda.json
```

Dry run (validates spec, builds the payload tar, prints the plan, **zero
provider calls**, works without the key):

```sh
python -B \
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
- **TTL 25 min** on the sandbox itself (`ttl_minutes`); ephemeral deletion on
  stop is selected with `auto_delete_interval=0`. These are separate controls.
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

Real transport requires installed Daytona **0.207.0**, checked before client
construction. Dry-run and fake-provider tests do not need Daytona installed:

- `daytona.Daytona()` — reads `DAYTONA_API_KEY` (+ optional
  `DAYTONA_API_URL`, `DAYTONA_TARGET`) from the environment.
- `Daytona.create(params, timeout=...) -> Sandbox` with
  `CreateSandboxFromImageParams(image=..., resources=Resources(cpu, memory,
  disk, gpu, gpu_type=[GpuType.RTX_5090, ...]), spot=True, ttl_minutes=25,
  auto_delete_interval=0, labels={...})`.
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

Uploads the tracked tree and runs only `nvidia-smi` + `nvcc --version`, then
the explicit `experimental/duck_cuda/build_remote.sh` build. Missing sources
or tools fail; there is no placeholder success or executable discovery.
No smoke binary is executed by this compile spec. Functional parity/bench
and training use separate specs and need separate evidence.
