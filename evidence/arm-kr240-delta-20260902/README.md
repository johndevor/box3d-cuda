# arm-kr240-delta-20260902 — KR240 reach, delta action contract (best actor so far)

Actor: `actor-arm-kr240-delta.pt` — byte copy of
`runs/gpu/20260902-160543-arm-reach-kr240/artifacts/train/gpu-train-out/actor_final.pt`
(sha256 8baafb3f9af07f38b2d3647329276fb111ef011ce4cd6776fc0e235452bea328). Feed-forward 27 → 256 → 256 → 6, 74,508 parameters.
Trained at commit 3fbbae5 (arm DELTA action contract: target_j += a_j · v_max_j · 0.02 s,
a = 0 holds) for one 15-minute RTX 5090 leg at E = 16384 (metrics in the run dir).

Judge (frozen, walk/eval/arm_reach_judge.py): 5 seeded reachable targets in 8 s, acquired
in order with a 2 cm radius held 14 consecutive policy steps; URDF joint limits and joint
speeds respected; no floor / base-column proxy violation. Acceptance = 12/12 over
seeds (4242, 7, 1913, 90210) × tiers (0, 1, 2).

Result of the CPU-serial harness on this actor
(`runs/gpu/20260902-160543-arm-reach-kr240/artifacts/acceptance/acceptance-out/acceptance.json`,
reproduced bit-for-bit by `scripts/build_dashboard.py`):

- **1/12 judge-clean** (seed 90210, tier 0: 5/5 acquired at 0.62 / 1.52 / 2.16 / 3.14 / 4.00 s).
- **8/12 cells acquire all 5 targets in order** but fail only the joint-speed clause
  (max speed ratio 1.02–1.05 of the URDF limit, mostly joint a3).
- 3 cells (tier 1–2) end early on the floor / base-column proxy; 1 cell acquires 3 of 5.
- 0 % speed-violating ticks during training vs 59–66 % under the absolute contract;
  first acquisition at 0.55 M env-steps vs ~20 M (36×).

Status: candidate, NOT accepted. The lite (0.5× Froude-scaled) variant has a spec
(`gpu/specs/arm-reach-lite.json`) but no trained actor.

Reproduce: `.venv/bin/python -B -m walk.eval.arm_acceptance --variant kr240 --actor evidence/arm-kr240-delta-20260902/actor-arm-kr240-delta.pt`
