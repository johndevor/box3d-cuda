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

## Post-extraction calibrated depth cameras

Standalone commit `ba306ee` added a CUDA camera compiler above the existing OBB
ray kernel. `daytona-stage6-depth-camera-20260829-r60` rebuilt the complete
PyTorch extension on an RTX 5090, reran the original nearest-hit ray smoke, and
then compared two heterogeneous calibrated scene/wrist cameras against the
scalar CPU oracle. World-space origins and directions, hit IDs, range,
optical-axis depth, body-attached pose composition, miss-zero semantics, unit
normals, and deterministic replay all passed. Maximum depth error was
`4.7684e-7 m`; maximum range error was `9.5367e-7 m`. The capped ephemeral
sandbox was deleted and absence verified after both result files downloaded.

This proves analytic OBB depth cameras, not rasterization, RGB, materials,
textures, lens distortion, rolling shutter, noise, or matched PhysX throughput.

## Post-extraction articulation-response diagnosis

The standalone response oracle and isolated CUDA kernel were added without
changing the production coupled solver. On RTX 5090,
`daytona-stage7-articulation-response-20260829-r22` evaluated 104,857,600
two-link responses in 0.056205 seconds (1.8656 billion responses/second), with
maximum CUDA-versus-CPU error `3.5763e-7`.

The matched PhysX CUDA challenger initially exposed a measurement error:
end-of-step pair-force queries can be zero after a transient impact has already
separated the bodies. Payload momentum change is therefore the authoritative
impulse measure in the zero-gravity, zero-friction, zero-drive micro.

After including both sides' rotational contact Jacobians,
`daytona-stage7-physx-articulation-response-20260829-r36` matched the same
reduced-coordinate oracle across all 64 worlds:

- maximum normal-impulse error: `0.0003842 N s`;
- maximum joint-velocity-delta error: `0.0035944 rad/s`;
- maximum payload-velocity error: `0.0003842 m/s`;
- seeded penetration: approximately `0.5 mm`;
- finite outputs and verified ephemeral-sandbox deletion: passed.

This validates articulation-projected contact response as the next production
integration target. It does not by itself accept the full Stage-7 trajectory
parity gate.
