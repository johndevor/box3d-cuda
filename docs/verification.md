# Extraction verification

The extraction intentionally changed repository boundaries and imports, not
physics behavior. Verification on 2026-08-29 used standalone commit `96f884b`
through a Factory OS consumer bundle pinned to that exact revision.

- Standalone CPU/source suite: 83 passed.
- Factory OS unchanged suite: 295 passed, 2 expected no-local-CUDA skips, and
  163 subtests passed.
- Python 3.12 wheel build and isolated wheel import/oracle smoke: passed.
- Public C header compiled as C11 without CUDA or PyTorch headers: passed.
- `daytona-box3d-extraction-20260829-r1` on RTX 5090: native C ABI CMake build
  and device smoke passed; all existing Stage 0-5 CUDA/ManiSkill benchmarks,
  correctness gates, and exact-contract comparisons passed.
- `daytona-box3d-extraction-stage6-20260829-r1` on RTX 5090: ray smoke and full
  fixed Stage-6 benchmark passed (`box3d.batched-ray-depth/v1`).
- `daytona-box3d-extraction-stage7-20260829-r1` on RTX 5090: coupled smoke and
  full bounded Stage-7 correctness benchmark passed
  (`box3d.joint-contact-pusher/v1`).

Stage 7's previously rejected strict instantaneous PhysX output-parity claim
remains rejected. This extraction does not weaken that gate or turn the
correctness benchmark into a speedup claim.
