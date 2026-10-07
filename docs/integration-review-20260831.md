# Duck/grid integration review — 2026-08-31

GitHub had no open or historical pull request at review time. The incoming
work was branch `duck-grid-walk`, with an independent root history. Integration
preserves both histories rather than copying over or force-pushing `main`.

- Main baseline: `91516339dbc3a3da0b2a2302fcb397fa4d1408f1`.
- Reviewed code: `c44f4ac224f03cf594e169519a514e2b9d7fdf8c`.
- Subsequent metrics-only commit: `5eb06a452274e6be306be43d24d75c0f675f72d4`.
- Existing native r3/machine-v1 implementation, headers and CMake are unchanged.
  New experimental interfaces remain separate and are not a native repin.

## Corrections made

1. Device-policy solver faults now persist diagnostics and raise before PPO
   consumes the transition. Two offline tests cover artifact creation and
   rejection of rollout/episode samples. A partial repeat is honestly marked
   as not exactly replayable.
2. Pinned model/reference data and hardware notices are bundled. Physics
   helpers no longer depend on a personal checkout. CAD exporters require an
   explicit external asset root; the walking exporter rejects dynamic grids
   rather than rendering moving cubes at a fixed pose.
3. Compile-only provider specs no longer pass through placeholders or run
   smoke binaries. Unsupported SDK versions fail before client construction.
   Fake-provider tests bypass real polling delays; production waits are unchanged.
4. Added saved solver-failure regressions, a bounded CPU verifier and a
   provider-free Linux CI workflow including tiny CPU trainer tests.
5. Historical logs, throughput semantics and experimental capacity limits are
   documented separately from accepted behavior. Machine-specific gate
   binaries are retained in incoming history, not shipped at the merged tip.

## Local verification

CPython 3.11.15, NumPy 2.2.6, PyTorch 2.9.1, Apple clang 21, macOS arm64.

| Gate | Result |
| --- | --- |
| Native combined build, strict C header and sanitizer | 14 jobs pass; 104 Python test executions within those jobs |
| Existing/root contracts | 181 pass |
| Contact repair | 6 pass |
| Saved nonconvergence problems | 16 pass, including 14 inherited base tests |
| Static/dynamic grid and capacities | 13 pass |
| Serial CUDA-source parity | 9 pass, 1 skip (external fault corpus absent) |
| Provider mocks/admission | 33 pass |
| Environments, gait, fault quarantine and CPU trainer | 48 pass |

Counts are per suite, not deduplicated across repeated tests. No CUDA device,
paid provider, new GPU training, or independent walking evaluation ran here.

Reproduction: `python -B scripts/verify_duck_cpu.py --output /absolute/fresh/dir`.
The full local run is retained at `gates-integration-r6/result.json`, SHA-256
`c8709e762e5b863e29db5766659443059e758106153911a3edfa9bd11ad14b7c`.
After the final historical-helper path fixes, all 14 native jobs were rerun:
`gates-final-native/result.json`, SHA-256
`85141c4c9f38b5e5200f2897144786f351f670102677b0c009672190e8da99bf`.
Exporter/home-hold syntax checks also pass. Ignored local gate directories
retain raw commands/logs; public CI uploads fresh verification artifacts.

Earlier local attempts remain rejected evidence. An imported new regression
initially demanded 1e-12 disk feasibility instead of the solver's published
1e-8 certificate. A trial projection fix broke a moving-cube test at tick
916 and was removed after comparison with the untouched incoming solver.
The new regression now checks the existing certificate. The final native
solver source is byte-identical to the incoming branch: no solver algorithm
or physical tolerance was changed by this integration. Other preliminary
failures were stale compile-spec assertions and real waits in fake tests.

## Linux CI timing distinction

The first clean Linux CI run (`33437081190`) passed native/root/repair/saved
gates and the grid physics assertions, but rejected the dynamic-capacity
timing assertion: 13.7871 ms versus the unchanged 10 ms target. It is retained
as failed performance evidence, not a physics failure or a performance pass.
CI now explicitly uses correctness-only verification: all physical assertions
remain, timing is reported with its original pass/fail value, and the result
states that timing was not enforced. Default local verification still enforces
10 ms. Three offline tests prove default rejection, honest missed-target
reporting and rejection of unknown modes. No solver or workload was changed.
The same-head push run (`33437076290`) passed the full strict suite, including
timing. Both results are preserved; one cannot assume a shared runner meets a
fixed wall-clock target. Local delta verification passes all five capacity and
timing-mode tests with the original benchmark enforced.

## Security and local work preservation

Gitleaks 8.30.1 found no secrets in the staged integration. Incoming history
had five generic-key detections: all were source-file SHA-256 entries in
historical gate manifests, reviewed as false positives; no broad suppression
was added. No credentials or local trained-policy pickle was imported.

The original checkout's untracked voxel contracts, oracles, benchmarks,
tests and evidence, plus separate terrain/robot experimental worktrees, were
left intact. Unfinished experiments are not implicitly accepted or published
by this integration. Native GPU admission, the 200k-active-body target and
independent learned walking remain separate future gates.
