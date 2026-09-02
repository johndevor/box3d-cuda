#!/usr/bin/env python3
"""Build the static results dashboard: dashboard/index.html (+ data/).

One command, idempotent, CPU only:

    .venv/bin/python -B scripts/build_dashboard.py            # build (uses clip cache)
    .venv/bin/python -B scripts/build_dashboard.py --force    # re-record every clip
    .venv/bin/python -B scripts/build_dashboard.py --skip-record   # HTML/manifest only

What it does
  1. Registry of every trained policy we have (evidence/**, best arm run),
     grouped by robot family; copies the arm actor into
     evidence/arm-kr240-delta-20260902/ with a README if missing.
  2. Re-judges every humanoid / arm actor with the frozen CPU harnesses
     (walk/eval/humanoid_acceptance.py, walk/eval/arm_acceptance.py) on the
     serial lane -- seconds each -- and cross-checks against the committed
     acceptance JSONs where they exist. The duck verdict is read from
     evidence/walking-accepted-20260901/acceptance.json.
  3. Records PRE-RECORDED replay clips by rolling the real actors out on the
     serial lanes (the exact acceptance-cell episodes: same seed / episode
     stream as the harness) and dumping body poses per 20 ms policy step,
     plus per-clip frozen-judge verdicts and contact flags. Generalist
     clips also under pinned Earth gravity and the 'earth_light_soft' combo
     (FlatFloorHumanoidEnv.pin_randomization).
  4. Summary strip + onboarding-cost table from runs/gpu/** metrics and a
     read-only `git log`.
  5. Writes dashboard/data/*.js (clips as classic scripts so the page works
     from file:// with no server), dashboard/data/manifest.js and
     dashboard/index.html (viewer code in dashboard/app.js / style.css).

Clips are cached by actor sha256 + clip parameters (dashboard/data/.cache.json).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "humanoid"))
sys.path.insert(0, str(ROOT / "arm"))
sys.path.insert(0, str(ROOT / "scripts"))

DASH = ROOT / "dashboard"
DATA = DASH / "data"
CACHE = DATA / ".cache.json"
DUCK_LIB = ROOT / "build" / "libintegrated_duck-pinned-97c3d37.dylib"
OLD_REPO = Path("/Users/john/Code/box3d-cuda-voxel-gate-c1")
DUCK_VIEW_JSON = OLD_REPO / "evidence/open-duck-zero-hold-view-v1/open-duck-zero-hold-view.json"
DUCK_XML = OLD_REPO / "evidence/open-duck-zero-hold-cpu-v1/model/open_duck_mini_v2.xml"
DUCK_CAD_PY = OLD_REPO / "scripts/export_open_duck_recorded_view.py"

ARM_RUN = ROOT / "runs/gpu/20260902-160543-arm-reach-kr240"
ARM_ACTOR_SRC = ARM_RUN / "artifacts/train/gpu-train-out/actor_final.pt"
ARM_EVIDENCE = ROOT / "evidence/arm-kr240-delta-20260902"

G_AUTHORED = 20.0
DR_RANGES = {"r_mass": 0.15, "r_friction": 0.3, "r_kp": 0.15, "r_damping": 0.3,
             "max_latency_steps": 2, "r_gravity": 0.5095}
DR_CONFIGS = {
    "nominal": {},
    "gravity_9.81": {"gravity_scale": 9.81 / G_AUTHORED},
    "earth_light_soft": {"gravity_scale": 9.81 / G_AUTHORED, "mass_scale": 0.85,
                         "kp_scale": 0.85, "friction_scale": 0.7,
                         "latency_steps": 1},
}
DR_LABELS = {"nominal": "nominal (authored 20 m/s²)",
             "gravity_9.81": "Earth gravity 9.81 m/s²",
             "earth_light_soft": "earth_light_soft (g 9.81, mass ×0.85, kp ×0.85, μ ×0.7, latency 1)"}

HUMANOID_CMDS = (0.5, 0.75, 1.0)
DUCK_CMDS = (0.10, 0.15, 0.20)
SEED = 4242


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def read_clip_js(path: Path, cid: str) -> dict:
    """Inverse of the clip file layout: '//meta:{...}\\nDGW.data("clip:ID",{...});\\n'."""
    body = path.read_text().split("\n", 1)[1]
    prefix = 'DGW.data("clip:' + cid + '",'
    assert body.startswith(prefix) and body.endswith(");\n"), path
    return json.loads(body[len(prefix):-3])


def js_data(name: str, payload) -> str:
    return f"DGW.data({json.dumps(name)},{json.dumps(payload, separators=(',', ':'))});\n"


# ============================================================ registry
# Each card: a trained policy. `clips` lists what the recorder must produce.
CARDS = [
    {
        "id": "duck-walking-v1", "family": "duck", "rig": "duck",
        "robot": "Open Duck plain-14", "variant": "flat floor, 14 hinges, DR + latency",
        "title": "Duck walker v1", "date": "2026-09-01",
        "actor": "evidence/walking-accepted-20260901/actor-walking-v1.pt",
        "evidence": "evidence/walking-accepted-20260901/",
        "judge": "walk/eval/gait.py (frozen; 20 ms debounce amendment)",
        "status": "accepted",
        "verdict_sub": "4-seed audit · in-run 2-seed protocol passed (accepted)",
        "verdict_note": "In-run 2-seed protocol passed at update 6245 (6/6 probes + 11 s confirmations); the stricter 4-seed audit is 10/12 (seed 1913 alternation at 0.10 / 0.20).",
        "lineage": "fresh-ff (CPU warm start) → continue-ff → continue-ff-short: one lineage over 52 ephemeral RTX 5090 legs (~13 min each), 1.64 B env-steps, 7.77 h accumulated GPU wall clock, DR + command latency on from the start.",
        "metrics": [("Commands", "0.10 / 0.15 / 0.20 m/s"), ("Qualified footfalls (s4242)", "25 / 38 / 51"),
                    ("Env-steps", "1.64 B"), ("GPU wall", "7.77 h, 52 legs"), ("Params", "84,508 (58→256→256→14)")],
        "clips": [{"kind": "duck", "command": c} for c in DUCK_CMDS],
        "actor_arch": "ff",
    },
    {
        "id": "humanoid-h11-accepted", "family": "humanoid", "rig": "h1", "hvariant": None,
        "robot": "Humanoid H1.1", "variant": "base body, per-joint gain tables",
        "title": "H1.1 walker — accepted", "date": "2026-09-02",
        "actor": "evidence/humanoid-accepted-20260902/actor-humanoid-accepted.pt",
        "evidence": "evidence/humanoid-accepted-20260902/",
        "judge": "walk/eval/humanoid_gait.py (frozen)",
        "status": "accepted",
        "lineage": "H1.1 gains → BC clone of authored reference gait v3.2 → tech-tree curriculum → RSI 0.5 → chained 10-min legs at E = 16384; ~11 legs from first-ever qualified swing to 12/12 in one session (commit dfa35ff).",
        "metrics": [("Commands", "0.50 / 0.75 / 1.00 m/s"), ("Params", "84,508 FF")],
        "clips": [{"kind": "humanoid", "command": c} for c in HUMANOID_CMDS],
    },
    {
        "id": "humanoid-walks-075", "family": "humanoid", "rig": "h1", "hvariant": None,
        "robot": "Humanoid H1.1", "variant": "milestone checkpoint (0.75 m/s)",
        "title": "H1.1 first walker (0.75 m/s milestone)", "date": "2026-09-02",
        "actor": "evidence/humanoid-walking-075-20260902/actor-humanoid-walks-075.pt",
        "evidence": "evidence/humanoid-walking-075-20260902/",
        "judge": "walk/eval/humanoid_gait.py (frozen)",
        "status": "milestone",
        "verdict_note": "All 4 seeds pass at cmd 0.75 (18–19 qualified, L/R symmetric). Gaps: 1.00 walks but translation < 60 %; 0.50 falls / alternation break.",
        "lineage": "BC (v3.2 ref, H1.1 gains) → tech-tree curriculum → RSI 0.5 → 6 chained 10-min legs at E = 16384 (runs/gpu/20260902-020511-humanoid-tree-continue).",
        "metrics": [("Milestone", "4/4 seeds at 0.75 m/s")],
        "clips": [{"kind": "humanoid", "command": c} for c in HUMANOID_CMDS],
    },
    {
        "id": "humanoid-generalist-gru-r2", "family": "generalist", "rig": "h1", "hvariant": None,
        "robot": "Humanoid H1.1", "variant": "generalist: residual GRU + domain randomization",
        "title": "Generalist GRU r2 (DR incl. gravity)", "date": "2026-09-02",
        "actor": "evidence/humanoid-generalist-20260902/actor-generalist-gru-r2.pt",
        "evidence": "evidence/humanoid-generalist-20260902/",
        "judge": "walk/eval/humanoid_gait.py (frozen) + humanoid/dr_brittleness.py",
        "status": "candidate",
        "verdict_note": "Commit 2564549 reports nominal 12/12 held; the CPU authority harness re-run at build time scores 10/12 (seed 1913 @ 1.00 and seed 90210 @ 0.50 fail alternation) — the same marginal-cell divergence seen on the stocky probe. Brittleness (seeds 4242/7): 77/90 cells across 15 pinned dynamics vs 63/90 for the FF specialist; Earth gravity 4/6 → earth_light_soft 6/6.",
        "lineage": "Accepted FF specialist warm-started exactly into a residual GRU (trunk = specialist, GRU head zeroed) → PPO under per-env DR: mass ±15 %, friction ±30 %, kp ±15 %, latency 0–2 steps, one-sided gravity 9.81..20 m/s²; r1 (3 legs) → r2 (6 legs total, plateau stop).",
        "metrics": [("Brittleness", "77/90 (FF 63/90)"), ("Params", "165,866 GRU-128 + FF trunk")],
        "brittleness_md": "evidence/humanoid-generalist-20260902/brittleness-gru-r2.md",
        "clips": [{"kind": "humanoid", "command": c, "dr": cfg}
                  for cfg in ("nominal", "gravity_9.81", "earth_light_soft")
                  for c in HUMANOID_CMDS],
    },
    {
        "id": "humanoid-stocky-accepted", "family": "variants", "rig": "h1_stocky", "hvariant": "h1_stocky",
        "robot": "Humanoid H1-STOCKY", "variant": "+20 % mass, −6 % legs, +3 cm soles",
        "title": "H1-STOCKY — accepted", "date": "2026-09-02",
        "actor": "evidence/humanoid-stocky-accepted-20260902/actor-stocky-accepted.pt",
        "evidence": "evidence/humanoid-stocky-accepted-20260902/",
        "judge": "walk/eval/humanoid_gait.py (frozen), CPU authority harness",
        "status": "accepted",
        "verdict_note": "12/12 on the CPU authority; all 6 checkpoints of the final 3-leg chain pass 12/12. Second never-seen humanoid body walking judge-clean.",
        "lineage": "h1_stocky BC seed → tech tree → RSI → 16 ten-minute legs; in-run probe (2 consecutive on-device 12/12) nominated the checkpoint, CPU harness confirmed.",
        "metrics": [("GPU legs", "16 × 10 min")],
        "acceptance_json": "evidence/humanoid-stocky-accepted-20260902/acceptance-cpu-authority.json",
        "clips": [{"kind": "humanoid", "command": c} for c in HUMANOID_CMDS],
    },
    {
        "id": "humanoid-stocky-candidate", "family": "variants", "rig": "h1_stocky", "hvariant": "h1_stocky",
        "robot": "Humanoid H1-STOCKY", "variant": "on-device probe candidate",
        "title": "H1-STOCKY — probe candidate", "date": "2026-09-02",
        "actor": "evidence/humanoid-stocky-candidate-20260902/actor-stocky-accepted.pt",
        "evidence": "evidence/humanoid-stocky-candidate-20260902/",
        "judge": "walk/eval/humanoid_gait.py (frozen)",
        "status": "candidate",
        "verdict_note": "On-device (CUDA fp32) in-run probe reported 12/12; CPU authority gives 10/12 (seeds 4242 & 7 at 1.00 fail alternation). Marginal-cell CUDA-vs-serial divergence; protocol hardened afterwards.",
        "lineage": "Same stocky chain, earlier checkpoint; recorded as a candidate, not an acceptance.",
        "metrics": [],
        "acceptance_json": "evidence/humanoid-stocky-candidate-20260902/cpu-harness/acceptance.json",
        "clips": [{"kind": "humanoid", "command": c} for c in HUMANOID_CMDS],
    },
    {
        "id": "humanoid-tall-6of12", "family": "variants", "rig": "h1_tall", "hvariant": "h1_tall",
        "robot": "Humanoid H1-TALL", "variant": "+12 % legs, +5 % torso, density-preserved mass",
        "title": "H1-TALL — best so far", "date": "2026-09-02",
        "actor": "evidence/humanoid-tall-20260902/actor-tall-6of12.pt",
        "evidence": "evidence/humanoid-tall-20260902/",
        "judge": "walk/eval/humanoid_gait.py (frozen)",
        "status": "partial",
        "verdict_note": "6/12 after 10 legs: 4/4 at 0.75 m/s; remaining failures are alternation dropouts at 1.0 / 0.5 traced to swing clearance (41 mm median vs 30 mm bar) → CLEARANCE_M raised to 0.045 for variants; chain continued.",
        "lineage": "h1_tall BC seed → tech tree → RSI → chained 10-min legs (humanoid-tree-tall-continue).",
        "metrics": [],
        "acceptance_json": "evidence/humanoid-tall-20260902/acceptance-6of12.json",
        "clips": [{"kind": "humanoid", "command": c} for c in HUMANOID_CMDS],
    },
    {
        "id": "arm-kr240-delta", "family": "arm", "rig": "kr240",
        "robot": "KUKA KR240 (pinned URDF)", "variant": "6-axis fixed base, delta action contract",
        "title": "KR240 reach — delta contract", "date": "2026-09-02",
        "actor": "evidence/arm-kr240-delta-20260902/actor-arm-kr240-delta.pt",
        "evidence": "evidence/arm-kr240-delta-20260902/",
        "judge": "walk/eval/arm_reach_judge.py (frozen: 5 targets / 8 s, 3 tiers, URDF speed & limit clauses, floor/column proxies)",
        "status": "partial",
        "verdict_note": "1/12 judge-clean (seed 90210 tier 0); 8/12 cells acquire all 5 targets in order (tier 0 in ~4 s) but exceed the URDF joint-speed clause by ≤ 4 %; 3 cells hit the floor/column proxy. Delta contract: first acquisition at 0.55 M steps vs ~20 M under absolute targets.",
        "lineage": "Arm category on the robot-generic stack (f00c4c6) → device policy path ABI v8 → reward v5 → DELTA action contract (3fbbae5); this actor is leg runs/gpu/20260902-160543-arm-reach-kr240 (15 min, E = 16384). Lite variant: spec exists, no trained actor yet.",
        "metrics": [("Reach", "3.46 m"), ("Moving mass", "690 kg"), ("Acq. radius", "2 cm, 14-step hold")],
        "acceptance_json": "runs/gpu/20260902-160543-arm-reach-kr240/artifacts/acceptance/acceptance-out/acceptance.json",
        "clips": [{"kind": "arm", "seed": s, "tier": t} for s in (4242, 90210) for t in (0, 1, 2)],
    },
]

FAMILIES = [
    ("duck", "Duck", "Open Duck plain-14 biped on the flat floor — the first accepted walker."),
    ("humanoid", "Humanoid H1.1", "68 kg, 14 hinges, y-up authored body; the accepted base member."),
    ("generalist", "Humanoid generalist", "GRU + domain randomization (mass, friction, kp, latency, gravity Earth..2g)."),
    ("variants", "Humanoid variants", "Parametric family members H1-TALL and H1-STOCKY, never-seen bodies, same recipe."),
    ("arm", "Arm KR240", "Fixed-base 6-axis KUKA KR240 reach task (lite variant: no actor yet)."),
]

# ============================================================ arm evidence dir
ARM_README = """# arm-kr240-delta-20260902 — KR240 reach, delta action contract (best actor so far)

Actor: `actor-arm-kr240-delta.pt` — byte copy of
`runs/gpu/20260902-160543-arm-reach-kr240/artifacts/train/gpu-train-out/actor_final.pt`
(sha256 {sha}). Feed-forward 27 → 256 → 256 → 6, 74,508 parameters.
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
"""


def ensure_arm_evidence() -> None:
    dst = ARM_EVIDENCE / "actor-arm-kr240-delta.pt"
    if not ARM_ACTOR_SRC.exists():
        log(f"WARN arm actor source missing: {ARM_ACTOR_SRC}")
        return
    ARM_EVIDENCE.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or sha256(dst) != sha256(ARM_ACTOR_SRC):
        shutil.copy2(ARM_ACTOR_SRC, dst)
        log(f"copied arm actor -> {rel(dst)}")
    readme = ARM_EVIDENCE / "README.md"
    text = ARM_README.format(sha=sha256(dst))
    if not readme.exists() or readme.read_text() != text:
        readme.write_text(text)
        log(f"wrote {rel(readme)}")
    src_acc = Path(CARDS[-1]["acceptance_json"])
    if (ROOT / src_acc).exists():
        dst_acc = ARM_EVIDENCE / "acceptance-1of12.json"
        if not dst_acc.exists():
            shutil.copy2(ROOT / src_acc, dst_acc)
            log(f"copied {rel(dst_acc)}")


# ============================================================ recorders
def _cache_load() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _cache_save(c: dict) -> None:
    CACHE.write_text(json.dumps(c, indent=1, sort_keys=True))


def humanoid_env(seed: int, variant: str | None, dr: str | None):
    from walk.env import humanoid_flat as hf
    from walk.env.humanoid_cuda_lane import CudaHumanoidLane
    vk = {} if variant in (None, "", "h1") else {"variant": variant}
    if dr:
        env = hf.FlatFloorHumanoidEnv(
            environments=1, seed=seed, perturbation_rad=0.0,
            randomization=dict(DR_RANGES),
            lane_factory=lambda E, off: CudaHumanoidLane(
                E, joint_offsets=off, randomization=dict(DR_RANGES), **vk), **vk)
        env.pin_randomization(**DR_CONFIGS[dr])
        return env
    return hf.FlatFloorHumanoidEnv(
        environments=1, seed=seed, perturbation_rad=0.0,
        lane_factory=lambda E, off: CudaHumanoidLane(E, joint_offsets=off, **vk), **vk)


def record_humanoid_cells(actor_path: Path, variant: str | None, dr: str | None,
                          commands=HUMANOID_CMDS, seed: int = SEED,
                          seconds: float = 8.0) -> list[dict]:
    """Harness-protocol episodes (one env per seed, commands in order) with
    per-step body poses + contact flags and the frozen judge's verdict."""
    from walk.eval.capture import _append_tick, _new_trace
    from walk.eval.humanoid_acceptance import load_actor, make_policy
    from walk.eval.humanoid_gait import evaluate_episode
    from walk.env.contract import SolverFault
    arch, actor = load_actor(str(actor_path))
    env = humanoid_env(seed, variant, dr)
    clips = []
    try:
        for cmd in commands:
            policy = make_policy(arch, actor)
            obs = env.reset(seed=seed)
            obs = env.set_command(cmd)
            trace = _new_trace(cmd, seed, 0)
            frames, contacts, fell_at, fault = [], [], None, False

            def on_tick(state):
                _append_tick(trace, state, 0)

            st = env._lane.read()
            frames.append(np.round(st.body_state[0, :, :7], 4).tolist())
            contacts.append([bool(x) for x in st.foot_contact[0]])
            steps = int(round(seconds / 0.02))
            try:
                for step in range(steps):
                    a = np.asarray(policy(obs), np.float32)
                    obs, _r, done, _info = env.step(a, on_tick=on_tick)
                    st = env._lane.read()
                    frames.append(np.round(st.body_state[0, :, :7], 4).tolist())
                    contacts.append([bool(x) for x in st.foot_contact[0]])
                    if done[0]:
                        if step < steps - 1:
                            trace["terminated"] = True
                            fell_at = round((step + 1) * 0.02, 2)
                        else:
                            trace["truncated_at_horizon"] = True
                        break
            except SolverFault as f:
                trace["solver_fault"] = True
                trace["terminated"] = True
                trace["fault_path"] = f.saved_problem_path
                fault = True
            if not trace["terminated"] and not trace["truncated_at_horizon"]:
                trace["truncated_at_horizon"] = True
            r = evaluate_episode(trace)
            q = [f for f in r.get("footfalls", []) if f.get("qualified")]
            fails = [k for k, v in r["criteria"].items() if str(v.get("pass")) != "True"]
            pos = trace["ticks"]["base_pos"]
            clips.append({
                "schema": "duckgridwalk.dashboard-humanoid-clip/1",
                "robot": "humanoid", "variant": variant or "h1", "arch": arch,
                "actor": rel(actor_path), "seed": seed, "command": cmd,
                "dr": dr or "nominal", "dr_pins": DR_CONFIGS.get(dr or "nominal", {}),
                "dt": 0.02, "seconds": seconds, "fell_at": fell_at,
                "solver_fault": fault,
                "physics": "CudaHumanoidLane serial fp32 (training physics), deterministic",
                "frames": frames, "contacts": contacts,
                "footfalls": [{"foot": f["foot"], "t": round(f["touchdown_rel_s"], 3),
                               "lift": round(f["liftoff_rel_s"], 3)} for f in q],
                "verdict": {"passed": bool(r["passed"]), "qualified": len(q),
                            "left": sum(1 for f in q if f["foot"] == "left"),
                            "right": sum(1 for f in q if f["foot"] == "right"),
                            "failed_criteria": fails,
                            "distance_m": round(pos[-1][0] - pos[0][0], 3) if pos else None,
                            "alive_s": round(len(pos) * 0.002, 2)},
            })
    finally:
        env.close()
    return clips


def _load_duck_cad():
    if not (DUCK_CAD_PY.exists() and DUCK_VIEW_JSON.exists() and DUCK_XML.exists()):
        return None
    if str(OLD_REPO) not in sys.path:
        sys.path.insert(0, str(OLD_REPO))
    spec = importlib.util.spec_from_file_location("duck_cad_export", DUCK_CAD_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def duck_rig() -> dict | None:
    """Duck CAD rig (compact): per body name, per geometry flat triangles."""
    cad = _load_duck_cad()
    if cad is None:
        return None
    base = json.loads(DUCK_VIEW_JSON.read_text())
    geoms = []
    for g in base["geometry"]:
        if g["mesh"] == "foot_bottom_tpu" and not g.get("collision"):
            continue
        flat = [round(v, 5) for tri in g["triangles"] for p in tri for v in p]
        geoms.append({"body": g["body"], "rgba": g["rgba"], "name": g["name"], "v": flat})
    names = [b["name"] for b in base["bodies"]]
    feet = [i for i, n in enumerate(names) if "foot" in n]
    return {"kind": "mesh", "bodies": names, "geometry": geoms, "feet": feet,
            "asset_notice": base.get("asset_notice", "")}


def record_duck_cells(actor_path: Path, commands=DUCK_CMDS, seed: int = SEED,
                      seconds: float = 8.0) -> list[dict]:
    import torch
    from walk.env.flat import FlatFloorDuckEnv
    from walk.eval.capture import _append_tick, _new_trace
    from walk.eval.gait import evaluate_episode
    from walk.train.ppo import Actor, RecurrentActor, unpack_actor_file
    from walk.env.contract import SolverFault
    cad = _load_duck_cad()
    fk_bodies = None
    if cad is not None:
        fk_bodies, _, _ = cad.load_model(DUCK_XML)
    raw = torch.load(actor_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "actor" in raw:
        raw = raw["actor"]
    arch, sd = unpack_actor_file(raw)
    actor = (RecurrentActor(58, 14) if arch == "gru" else Actor(58, 14))
    actor.load_state_dict(sd)
    actor.eval()
    lib = str(DUCK_LIB) if DUCK_LIB.exists() else None
    env = FlatFloorDuckEnv(environments=1, seed=seed, perturbation_rad=0.0, library_path=lib)
    clips = []
    try:
        for cmd in commands:
            h = {"h": None}

            @torch.no_grad()
            def policy(obs):
                o = torch.from_numpy(np.ascontiguousarray(obs))
                if arch == "gru":
                    if h["h"] is None:
                        h["h"] = actor.initial_state(1)
                    a, h["h"] = actor.deterministic(o, h["h"])
                    return a.numpy()
                return actor.deterministic(o).numpy()

            obs = env.reset(seed=seed)
            obs = env.set_command(cmd)
            trace = _new_trace(cmd, seed, 0)
            ticks = []

            def on_tick(state):
                _append_tick(trace, state, 0)
                ticks.append((state.q[0].copy(), state.foot_contact[0].copy(),
                              state.body_state[0, :, :7].copy()))

            fell_at, fault = None, False
            steps = int(round(seconds / 0.02))
            try:
                for step in range(steps):
                    obs, _r, done, _ = env.step(policy(obs), on_tick=on_tick)
                    if done[0]:
                        if step < steps - 1:
                            trace["terminated"] = True
                            fell_at = round((step + 1) * 0.02, 2)
                        else:
                            trace["truncated_at_horizon"] = True
                        break
            except SolverFault as f:
                trace["solver_fault"] = True
                trace["terminated"] = True
                fault = True
            if not trace["terminated"] and not trace["truncated_at_horizon"]:
                trace["truncated_at_horizon"] = True
            r = evaluate_episode(trace)
            q = [f for f in r.get("footfalls", []) if f.get("qualified")]
            fails = [k for k, v in r["criteria"].items() if str(v.get("pass")) != "True"]
            frames, contacts = [], []
            kept = ticks[9::10]                      # one frame per policy step
            for qv, contact, body in kept:
                if fk_bodies is not None:
                    base_pose = [float(qv[0]), float(qv[1]), float(qv[2]),
                                 float(qv[6]), float(qv[3]), float(qv[4]), float(qv[5])]
                    poses = cad.forward_kinematics(fk_bodies, {
                        "base_pose": base_pose, "joint_q": [float(x) for x in qv[7:21]]})
                    # emit p(3) + q_xyzw(4) per body, matching the other rigs
                    frames.append([[round(p[0][0], 4), round(p[0][1], 4), round(p[0][2], 4),
                                    round(p[1][1], 4), round(p[1][2], 4), round(p[1][3], 4),
                                    round(p[1][0], 4)] for p in poses])
                else:
                    frames.append(np.round(body, 4).tolist())
                contacts.append([bool(contact[0]), bool(contact[1])])
            if fk_bodies is not None and frames:
                drift = float(np.linalg.norm(np.array(frames[0][1][:3]) - kept[0][0][:3]))
                if drift > 0.06:
                    raise SystemExit(f"duck FK/root mismatch {drift:.3f} m - convention error")
            pos = trace["ticks"]["base_pos"]
            clips.append({
                "schema": "duckgridwalk.dashboard-duck-clip/1",
                "robot": "duck", "variant": "plain14", "arch": arch,
                "actor": rel(actor_path), "seed": seed, "command": cmd,
                "dr": "nominal", "dt": 0.02, "seconds": seconds, "fell_at": fell_at,
                "solver_fault": fault, "cad": fk_bodies is not None,
                "physics": "Box3D integrated duck CPU lane (idv1/civ1, f64 oracle), deterministic",
                "frames": frames, "contacts": contacts,
                "footfalls": [{"foot": f["foot"], "t": round(f["touchdown_rel_s"], 3),
                               "lift": round(f["liftoff_rel_s"], 3)} for f in q],
                "verdict": {"passed": bool(r["passed"]), "qualified": len(q),
                            "left": sum(1 for f in q if f["foot"] == "left"),
                            "right": sum(1 for f in q if f["foot"] == "right"),
                            "failed_criteria": fails,
                            "distance_m": round(pos[-1][0] - pos[0][0], 3) if pos else None,
                            "alive_s": round(len(pos) * 0.002, 2)},
            })
    finally:
        env.close()
    return clips


def humanoid_rig(variant: str | None) -> dict:
    import h1_family
    lw = h1_family.load_lowering(variant)
    return {"kind": "boxes",
            "bodies": [{"name": b[0], "half": [round(float(x), 4) for x in b[2]],
                        "mass": b[3]} for b in lw.BODIES],
            "feet": list(lw.FOOT_BODIES), "pelvis": 1,
            "home_root_height": None}


# ============================================================ verdicts
def judge_humanoid(actor: Path, variant: str | None) -> dict:
    from walk.eval.humanoid_acceptance import run_acceptance
    return run_acceptance(str(actor), lane="serial", quiet=True, variant=variant)


def judge_arm(actor: Path, variant: str = "kr240") -> dict:
    from walk.eval.arm_acceptance import load_actor, make_actor_policy, run_acceptance
    arch, act = load_actor(str(actor))
    return run_acceptance(variant, lambda: make_actor_policy(arch, act), lane="serial",
                          quiet=True)


def cell_summary(acc: dict) -> dict:
    eps = acc.get("episodes", {})
    n = len(eps)
    npass = sum(1 for v in eps.values() if v.get("passed"))
    cells = []
    for k, v in eps.items():
        cells.append({"cell": k, "passed": bool(v.get("passed")),
                      "qualified": v.get("qualified"),
                      "left": v.get("left"), "right": v.get("right"),
                      "acquired": (sum(1 for t in v.get("acquisition_times_s", []) if t is not None)
                                   if "acquisition_times_s" in v else None),
                      "failed": v.get("failed_criteria", [])})
    return {"passed": npass, "total": n, "cells": cells, "accepted": bool(acc.get("accepted"))}


def parse_brittleness(md: Path) -> dict | None:
    if not md.exists():
        return None
    rows = []
    for line in md.read_text().splitlines():
        if not line.startswith("|") or line.startswith("|---") or "config" in line.split("|")[1]:
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 3:
            continue
        p, t = cols[2].split("/")
        rows.append({"config": cols[0], "pins": cols[1], "passed": int(p), "total": int(t),
                     "cells": cols[3:]})
    return {"rows": rows, "passed": sum(r["passed"] for r in rows),
            "total": sum(r["total"] for r in rows)}


# ============================================================ GPU legs / cost
SPEC_FAMILY = [
    (r"^(train-duck-gpu|continue-ff|continue-ff-short|fresh-ff|fresh-gru|sweep-train.*)$", "duck"),
    (r"^humanoid-(bringup|train-ff|train-v21|iter|iter-continue|continue-ff|minimal|tree|tree-continue)$", "humanoid"),
    (r"^humanoid-generalist$", "generalist"),
    (r"^humanoid-tree-tall.*$", "tall"),
    (r"^humanoid-tree-stocky.*$", "stocky"),
    (r"^arm-.*$", "arm"),
    (r"^(parity-bench-duck-cuda|compile-duck-cuda|bake-image)$", "infra"),
]


def gpu_legs() -> dict:
    fam = {}
    for d in sorted((ROOT / "runs/gpu").glob("*")):
        if not d.is_dir():
            continue
        spec = d.name[16:]
        f = next((n for pat, n in SPEC_FAMILY if re.match(pat, spec)), "other")
        a = fam.setdefault(f, {"launches": 0, "trained": 0, "wall_s": 0.0, "env_steps": 0,
                               "accepted_legs": 0})
        a["launches"] += 1
        m = d / "artifacts/train/gpu-train-out/metrics.jsonl"
        if m.exists():
            last = None
            with open(m) as fh:
                for line in fh:
                    if '"kind": "train"' in line:
                        last = line
            if last:
                j = json.loads(last)
                a["trained"] += 1
                a["wall_s"] += float(j.get("wall_total_s") or 0.0)
                a["env_steps"] += int(j.get("env_steps") or 0)
        if (d / "artifacts/train/gpu-train-out/accepted/acceptance.json").exists():
            a["accepted_legs"] += 1
    return fam


def git_log() -> list[dict]:
    try:
        out = subprocess.run(["git", "log", "--format=%h|%at|%s"], cwd=ROOT,
                             capture_output=True, text=True, timeout=20, check=True).stdout
    except Exception as e:  # noqa: BLE001
        log(f"WARN git log unavailable ({e}); onboarding table without commit data")
        return []
    rows = []
    for line in out.splitlines():
        h, t, s = line.split("|", 2)
        rows.append({"hash": h, "t": int(t), "subject": s})
    return rows


ONBOARDING = [
    # family key, label, first commit, result commit, result label, subject regex
    ("duck", "Duck (Open Duck plain-14)", "ec88e34", "a851d2d", "ACCEPTED (in-run protocol)",
     r"duck|walk|gait|reward|kernel|cuda|ppo|train|accept|launcher|sandbox|daytona|probe|clock|flicker|footfall|chatter|debounce|spec|solver|contact|engine"),
    ("humanoid", "Humanoid H1.1", "4d7ba13", "dfa35ff", "ACCEPTED 12/12",
     r"humanoid|h0|h1|bc |tech-tree|tree|reference|clock|swing|gait|judge|curriculum|rsi|chain"),
    ("generalist", "Humanoid generalist (GRU + DR)", "6d4e168", "2564549", "10/12 on CPU authority (commit: 12/12); brittleness 77/90",
     r"generalist|gru|brittle|randomiz|gravity"),
    ("tall", "H1-TALL", "a86e1dc", "9b87220", "6/12 (chain continuing)",
     r"tall|family|variant"),
    ("stocky", "H1-STOCKY", "a86e1dc", "c3b3e41", "ACCEPTED 12/12",
     r"stocky|family|variant|probe"),
    ("arm", "Arm KR240 reach", "f00c4c6", "3fbbae5", "1/12 (8/12 acquire 5/5), not accepted",
     r"\barm\b|kr240|reach|delta"),
]


def onboarding_table(legs: dict, commits: list[dict]) -> list[dict]:
    by_hash = {c["hash"]: c for c in commits}
    rows = []
    for key, label, first, last, result, pat in ONBOARDING:
        c0, c1 = by_hash.get(first), by_hash.get(last)
        hours = ((c1["t"] - c0["t"]) / 3600.0) if (c0 and c1) else None
        n_commits = None
        if c0 and c1:
            n_commits = sum(1 for c in commits if c0["t"] <= c["t"] <= c1["t"]
                            and re.search(pat, c["subject"], re.I))
        lg = legs.get(key, {})
        rows.append({"family": key, "label": label, "first_commit": first,
                     "result_commit": last, "result": result,
                     "wall_hours": None if hours is None else round(hours, 1),
                     "commits": n_commits,
                     "gpu_legs": lg.get("trained", 0), "gpu_launches": lg.get("launches", 0),
                     "gpu_hours": round(lg.get("wall_s", 0.0) / 3600.0, 2),
                     "env_steps_b": round(lg.get("env_steps", 0) / 1e9, 2),
                     "first_date": time.strftime("%m-%d %H:%M", time.localtime(c0["t"])) if c0 else None,
                     "result_date": time.strftime("%m-%d %H:%M", time.localtime(c1["t"])) if c1 else None})
    return rows


# ============================================================ HTML
INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>duck-grid-walk — trained robots</title>
<link rel="stylesheet" href="style.css">
<script>window.DGW={_d:{},data(n,p){this._d[n]=p;(this._w[n]||[]).forEach(f=>f(p));delete this._w[n];},_w:{}};</script>
<script src="vendor/three.min.js"></script>
<script src="vendor/OrbitControls.js"></script>
<script src="data/manifest.js"></script>
</head><body>
<div id="app"></div>
<script src="app.js"></script>
</body></html>
"""


# ============================================================ main
def build(force: bool, skip_record: bool, only: set[str] | None) -> int:
    t_start = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    ensure_arm_evidence()
    cache = _cache_load()
    generated = []
    problems = []

    # --- rigs
    rigs = {}
    for v in ("h1", "h1_tall", "h1_stocky"):
        rigs[v] = humanoid_rig(None if v == "h1" else v)
    from export_arm_view import arm_geometry
    rigs["kr240"] = {"kind": "arm", **arm_geometry("kr240")}
    drig = duck_rig()
    if drig is None:
        problems.append("duck CAD rig unavailable (box3d-cuda-voxel-gate-c1 evidence missing); "
                        "duck clips fall back to lane body boxes")
        rigs["duck"] = {"kind": "boxes-generic", "feet": [], "bodies": []}
    else:
        p = DATA / "rig-duck.js"
        p.write_text(js_data("rig:duck", drig))
        generated.append(p)
        rigs["duck"] = {"kind": "mesh-external", "file": "data/rig-duck.js",
                        "bodies": drig["bodies"], "feet": drig["feet"],
                        "asset_notice": drig["asset_notice"]}

    # --- cards: verdicts + clips
    cards_out = []
    for card in CARDS:
        if only and card["id"] not in only:
            continue
        actor = ROOT / card["actor"]
        c = {k: v for k, v in card.items() if k not in ("clips",)}
        c["actor"] = rel(actor)
        if not actor.exists():
            problems.append(f"{card['id']}: actor missing {rel(actor)}")
            c["verdict"] = {"label": "actor missing", "passed": 0, "total": 0}
            c["clips"] = []
            cards_out.append(c)
            continue
        asha = sha256(actor)
        c["actor_sha256"] = asha
        c["actor_bytes"] = actor.stat().st_size

        # verdict (frozen judge, recomputed on the serial lane; cached by sha)
        vkey = f"verdict:{card['id']}:{asha}"
        if vkey in cache and not force:
            verdict = cache[vkey]
        else:
            t0 = time.time()
            if card["family"] in ("humanoid", "generalist", "variants"):
                acc = judge_humanoid(actor, card.get("hvariant"))
                verdict = cell_summary(acc)
                verdict["source"] = "recomputed: walk/eval/humanoid_acceptance.py (serial lane)"
            elif card["family"] == "arm":
                acc = judge_arm(actor)
                verdict = cell_summary(acc)
                verdict["source"] = "recomputed: walk/eval/arm_acceptance.py (serial lane)"
            else:
                acc = json.loads((ROOT / "evidence/walking-accepted-20260901/acceptance.json").read_text())
                verdict = cell_summary(acc)
                verdict["source"] = "evidence/walking-accepted-20260901/acceptance.json (4-seed audit)"
            verdict["wall_s"] = round(time.time() - t0, 1)
            cache[vkey] = verdict
            _cache_save(cache)
            log(f"judged {card['id']}: {verdict['passed']}/{verdict['total']} in {verdict['wall_s']}s")
        # cross-check with the committed acceptance json, if any
        aj = card.get("acceptance_json")
        if aj and (ROOT / aj).exists():
            committed = cell_summary(json.loads((ROOT / aj).read_text()))
            verdict["committed"] = {"path": aj, "passed": committed["passed"], "total": committed["total"],
                                    "matches": committed["passed"] == verdict["passed"]
                                    and all(a["passed"] == b["passed"] for a, b in
                                            zip(committed["cells"], verdict["cells"]))}
            if not verdict["committed"]["matches"]:
                problems.append(f"{card['id']}: recomputed verdict differs from {aj}")
        verdict["label"] = f"{verdict['passed']}/{verdict['total']}"
        c["verdict"] = verdict
        if card.get("brittleness_md"):
            c["brittleness"] = parse_brittleness(ROOT / card["brittleness_md"])

        # clips
        clips_meta = []
        groups: dict[tuple, list] = {}
        for spec in card["clips"]:
            gk = (spec["kind"], spec.get("dr"), spec.get("seed"))
            groups.setdefault(gk, []).append(spec)
        for (kind, dr, seed), specs in groups.items():
            ids = []
            for spec in specs:
                if kind == "arm":
                    cid = f"{card['id']}--seed{seed}-tier{spec['tier']}"
                else:
                    cid = f"{card['id']}--{dr or 'nominal'}-cmd{spec['command']:.2f}"
                ids.append(cid)
            ckey = f"clip:{ids[0]}:{asha}:{len(specs)}"
            files = [DATA / f"{cid}.js" for cid in ids]
            if not force and not skip_record and all(f.exists() for f in files) and cache.get(ckey) == asha:
                for cid, f in zip(ids, files):
                    clips_meta.append(json.loads(f.read_text().split("\n", 1)[0].split("//meta:")[1]))
                continue
            if skip_record:
                for cid, f in zip(ids, files):
                    if f.exists():
                        clips_meta.append(json.loads(f.read_text().split("\n", 1)[0].split("//meta:")[1]))
                continue
            t0 = time.time()
            try:
                if kind == "humanoid":
                    clips = record_humanoid_cells(actor, card.get("hvariant"), dr,
                                                  commands=[s["command"] for s in specs])
                elif kind == "duck":
                    clips = record_duck_cells(actor, commands=[s["command"] for s in specs])
                else:
                    from export_arm_view import record_arm_cells
                    clips = record_arm_cells(str(actor), seed=seed, tiers=[s["tier"] for s in specs])
            except Exception as e:  # noqa: BLE001
                problems.append(f"{card['id']} {kind} {dr} {seed}: recording failed: {e!r}")
                continue
            for cid, f, clip in zip(ids, files, clips):
                clip["id"] = cid
                clip["card"] = card["id"]
                clip["rig"] = card["rig"]
                meta = {k: v for k, v in clip.items()
                        if k not in ("frames", "contacts", "targets", "target_index", "hold_steps",
                                     "footfalls", "acquired_steps")}
                meta["file"] = f"data/{f.name}"
                meta["n_frames"] = len(clip["frames"])
                if kind == "arm":
                    meta["label"] = f"seed {seed} · tier {clip['tier']} (r {clip['tier_radius_m']} m)"
                elif kind == "humanoid":
                    meta["label"] = f"{clip['command']:.2f} m/s" + (f" · {dr}" if dr and dr != "nominal" else "")
                else:
                    meta["label"] = f"{clip['command']:.2f} m/s"
                f.write_text("//meta:" + json.dumps(meta) + "\n" + js_data(f"clip:{cid}", clip))
                meta["bytes"] = f.stat().st_size
                clips_meta.append(meta)
                generated.append(f)
            cache[ckey] = asha
            _cache_save(cache)
            log(f"recorded {card['id']} {kind} {dr or ''} seed {seed}: {len(clips)} clips in {time.time() - t0:.1f}s")
        for m in clips_meta:
            if "bytes" not in m and (DATA / Path(m["file"]).name).exists():
                m["bytes"] = (DATA / Path(m["file"]).name).stat().st_size
        c["clips"] = clips_meta
        cards_out.append(c)

    # --- sanity: z-up, floor at the bottom
    for c in cards_out:
        for m in c["clips"]:
            f = DATA / Path(m["file"]).name
            if not f.exists():
                continue
            clip = read_clip_js(f, m["id"])
            if c["family"] == "arm":
                fr0 = np.asarray(clip["frames"][0]["bodies"], float)
                zs = fr0[2:, 2]
                assert (zs > 0).all(), f"{m['id']}: arm link below floor"
                continue
            fr0 = np.asarray(clip["frames"][0], float)
            if c["family"] == "duck":
                assert fr0[1, 2] > 0.1, f"{m['id']}: duck trunk z {fr0[1, 2]}"
            else:
                pelvis = fr0[1, 2]
                feet = fr0[list(rigs[c["rig"]]["feet"]), 2]
                assert pelvis > 0.8 and (feet > 0).all() and pelvis > feet.max() + 0.5, \
                    f"{m['id']}: z-up violated pelvis {pelvis} feet {feet}"

    # --- summary
    legs = gpu_legs()
    commits = git_log()
    onboarding = onboarding_table(legs, commits)
    robots = ["Open Duck plain-14", "Humanoid H1.1", "Humanoid H1-TALL", "Humanoid H1-STOCKY", "KUKA KR240"]
    accepted = [c["id"] for c in cards_out if c.get("status") == "accepted"]
    total_launches = sum(v["launches"] for v in legs.values())
    total_trained = sum(v["trained"] for v in legs.values())
    total_gpu_h = sum(v["wall_s"] for v in legs.values()) / 3600.0
    total_steps = sum(v["env_steps"] for v in legs.values())
    summary = {"robots": robots, "robots_onboarded": len(robots),
               "accepted_policies": len(accepted), "accepted_ids": accepted,
               "policies_total": len(cards_out),
               "gpu_launches": total_launches, "gpu_training_legs": total_trained,
               "gpu_training_hours": round(total_gpu_h, 1),
               "env_steps_b": round(total_steps / 1e9, 1),
               "legs_by_family": legs, "onboarding": onboarding,
               "commits_total": len(commits),
               "clips_total": sum(len(c["clips"]) for c in cards_out),
               "clip_bytes": sum(m.get("bytes", 0) for c in cards_out for m in c["clips"])}

    manifest = {"schema": "duckgridwalk.dashboard-manifest/1",
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "families": [{"id": k, "title": t, "blurb": b} for k, t, b in FAMILIES],
                "cards": cards_out, "rigs": rigs, "summary": summary,
                "dr_labels": DR_LABELS, "problems": problems}
    mp = DATA / "manifest.js"
    mp.write_text(js_data("manifest", manifest))
    generated.append(mp)
    ip = DASH / "index.html"
    ip.write_text(INDEX_HTML)
    generated.append(ip)

    log("")
    log(f"dashboard: {rel(ip)}  ({len(cards_out)} cards, {summary['clips_total']} clips, "
        f"{summary['clip_bytes'] / 1e6:.1f} MB of clip data)")
    for c in cards_out:
        log(f"  [{c['status']:9s}] {c['title']:42s} {c['verdict']['label']:26s} clips={len(c['clips'])}")
    log(f"generated {len(generated)} files under {rel(DASH)} in {time.time() - t_start:.0f}s")
    if problems:
        log("PROBLEMS:")
        for p in problems:
            log("  - " + p)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-record every clip / verdict")
    ap.add_argument("--skip-record", action="store_true", help="manifest + HTML only")
    ap.add_argument("--only", default=None, help="comma-separated card ids")
    a = ap.parse_args()
    return build(a.force, a.skip_record, set(a.only.split(",")) if a.only else None)


if __name__ == "__main__":
    raise SystemExit(main())
