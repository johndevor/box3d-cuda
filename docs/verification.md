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
  (`box3d.joint-contact-pusher/v2`).

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

`daytona-stage6-depth-camera-benchmark-20260829-r61` then measured the complete
analytic camera path at 1,024 worlds × two 16×16 cameras × 240 frames. The
125,829,120 output pixels completed in `0.007688352 s`, or 16.366 billion depth
pixels/s and 63.930 million camera frames/s. The timed interval includes CUDA
camera pose/ray compilation, linear OBB ray casting, and optical-axis depth
conversion. Timed outputs were finite and bounded, misses were exact zero,
directions were normalized within `1.7881e-7`, replay was bit-exact, and the
hit/miss population was nontrivial. Peak Torch allocation was 116,944,896
bytes. The original ray smoke and 62.9-million-ray Stage-6 benchmark also
passed in the same clean RTX run; all four downloaded artifacts and sandbox
deletion were verified.

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

## Native machine coupling

Standalone commit `36a0591` adds the separately named, capability-gated
machine-coupling extension without changing ABI-v2 r3. The bounded run
`daytona-native-machine-coupling-20260829-r1` compiled the shared library with
CUDA 12.8 on an RTX 5090 and passed the topology, native-v1, resident-r3, and
installed-header smokes. It also executed all four agreed force/torque and
signed joint-velocity goldens; the installed query reported the exact current
mask `0xe07ff`.

The run bundle contained 100 files with SHA-256
`545567c988867a0f20382aac7af8d8b01bbaf2c282094b42c0510200c2b98c3f`.
The redacted Factory OS evidence record has SHA-256
`928a42e7b26fc96de2cc28b2fae5a06e70d57a17a9ae98325e607de274037328`.
The ephemeral sandbox was deleted on the first attempt and verified absent.

## Parallel vision PPO baseline

Standalone commit `cc39245` introduced a dependency-free learner/environment
layout and GAE oracle plus a GPU-resident execution benchmark. Run
`daytona-parallel-trainer-20260829-r1` used an RTX 5090 for eight independent
learners with 512 environments each, a 32-step horizon, two PPO updates, two
PPO epochs, and two calibrated 8x8 analytic depth/instance cameras per world.

The timed rollout processed 262,144 physical world-actions at 914,284/s and
33,554,432 depth pixels at 117,028,383/s. PPO optimization took 0.299887 s;
rollout took 0.286720 s; peak Torch CUDA allocation was 478,583,296 bytes. All
ten execution gates passed: finite observations/rewards/advantages/losses,
bit-exact camera replay, bounded instance IDs, nontrivial hits/misses, changed
physics state, finite per-learner returns, and nonzero parameter updates for
all eight learners. All 4,096 worlds recorded contact in the final rollout.

The result artifact SHA-256 is
`6742191fe46aeb589d5843c89509a52e6ad176b6ac2bf17872951d88025ab301`;
the redacted provider evidence SHA-256 is
`35f752445fff55ddd8274c3b6d7b04dcdba9b00af2f9702dc080fb2141f357e4`.
The ephemeral sandbox was deleted and verified absent. This accepts the
parallel trainer's execution and throughput only. It does not accept a
learning curve, asynchronous partial resets, RGB, or raster rendering.

Run `daytona-parallel-trainer-async-20260829-r2` exercised async-v2 on an RTX
5090: eight independent seeds, 512 environments each, 32 steps per update, and
eight updates. All 13 execution gates passed. Every learner completed 781--784
episodes per update; 6,261 worlds reset independently per update; selected and
unselected state/cache slices were exact. The timed rollouts processed 1.509M
world-actions/s and 193.2M analytic depth pixels/s with 482,697,728 bytes peak
Torch allocation. The result SHA-256 is
`a95a1ce4cc99c4d98a93dbebface3e2c5ea2cb4d45626d3ad7fbb7b2f4611e6b`.

Learning remains rejected: four of eight curves improved, below the required
six. This is accepted async reset/execution evidence, not a learned-policy
claim. A fixed-action reward-sensitivity probe was added afterward and remains
to be exercised on CUDA. The full local suite remains green at 163 tests plus
18 subtests.
