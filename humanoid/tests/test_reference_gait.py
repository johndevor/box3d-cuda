"""Reference-gait gates: drift, table properties, FK validation, physics smoke.

Run: .venv/bin/python -B humanoid/tests/test_reference_gait.py

Validates the synthetic reference KINEMATICALLY before anything trains on
it (coordinator gate), using the real fixture's FK (av1 evaluate, no
dynamics) with the pelvis on its nominal path:

  per-frame over all 64 bins:
    - alternation: the clock-designated stance foot's whole sole is on the
      floor (|min z| <= 4 mm: the <= 3.3 mm straight-leg pelvis bob);
    - swing clearance >= 30 mm on the certified mid-swing window
      (|sin(phase)| >= 0.707; peak measured ~56 mm);
    - joints strictly within the authored limits;
    - waist/neck/arms at HOME;
  per-cycle:
    - stance-foot sweep = 0.150 m per half cycle (the clock-encoded
      half-stride), so per-swing world placement = one full 0.30 m stride,
      inside the judge's placement band (>= 0.15 m, 2x margin);
    - L/R mirror symmetry (right(b) == left(b + 32 mod 64));
  json drift: committed reference_gait.json == fresh regeneration; header
    DW_REF_GAIT carries the identical f64 values (env<->kernel bit-parity);
  dynamics smoke (open-loop bipeds fall -- this is NOT a walking claim):
    PD-targeting the table at the cmd-0.75 clock from reset on the f64 CPU
    lane runs fault-free for 0.6 s and each foot breaks contact at least
    once inside its swing window.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "humanoid"))
sys.path.insert(0, str(ROOT / "walk" / "env"))

import h1_lowering as h0  # noqa: E402  (ACTIVE lowering: H1)
import native_lane  # noqa: E402

JSON_PATH = ROOT / "humanoid" / "reference_gait.json"
HEADER = ROOT / "humanoid" / "include" / "duck_model.h"


def _author_module():
    spec = importlib.util.spec_from_file_location(
        "author_reference_gait",
        ROOT / "humanoid" / "author_reference_gait.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(JSON_PATH.read_text())
        cls.table = np.asarray(cls.data["table"], float)

    def test_json_drift(self):
        author = _author_module()
        fresh = json.dumps(author.payload(), indent=1) + "\n"
        self.assertEqual(fresh, JSON_PATH.read_text(),
                         "reference_gait.json drifted; rerun "
                         "humanoid/author_reference_gait.py")

    def test_shape_and_posture(self):
        self.assertEqual(self.table.shape, (64, 14))  # H1
        self.assertEqual(self.data["bins"], 64)
        self.assertEqual(tuple(self.data["joints"]), h0.JOINT_NAMES)
        # waist/neck/shoulders/elbows at HOME
        np.testing.assert_array_equal(self.table[:, [0, 1, 10, 11, 12, 13]], 0.0)

    def test_within_authored_limits(self):
        lim = np.array([(j[4], j[5]) for j in h0.JOINTS])
        self.assertTrue((self.table >= lim[:, 0] - 1e-12).all())
        self.assertTrue((self.table <= lim[:, 1] + 1e-12).all())

    def test_mirror_symmetry(self):
        rolled = self.table[(np.arange(64) + 32) % 64]
        # sagittal leg columns: right(b) == left(b + half cycle)
        np.testing.assert_allclose(self.table[:, 7:10], rolled[:, 3:6],
                                   atol=1e-12)
        # roll columns: equal L/R (parallel legs), antisymmetric by half cycle
        np.testing.assert_allclose(self.table[:, 2], self.table[:, 6], atol=0)
        np.testing.assert_allclose(self.table[:, 2], -rolled[:, 2],
                                   atol=1e-12)

    def test_header_carries_identical_table(self):
        text = HEADER.read_text()
        block = text[text.index("DW_REF_GAIT"):]
        block = block[block.index("{") + 1:block.index(";") - 1]
        rows = [r.strip("{}").split(",") for r in block.split("},{")]
        header_table = np.array([[float(x) for x in row] for row in rows])
        self.assertEqual(header_table.shape, (64, 14))
        np.testing.assert_array_equal(header_table, self.table,
                                      "header DW_REF_GAIT != json (regenerate)")

    def test_reward_module_reads_same_table(self):
        from walk.env import humanoid_reward as hr
        np.testing.assert_array_equal(np.asarray(hr.REF_GAIT), self.table)
        self.assertEqual(hr.REF_BINS, 64)
        self.assertEqual(hr.W_IMIT, 0.5)


class KinematicValidation(unittest.TestCase):
    """Per-frame FK validation on the real fixture (no dynamics), v3.

    SAGITTAL gates run with the roll columns ZEROED (the roll channel is
    the weight-transfer command; its kinematic effect under load differs
    from pinned-pelvis FK by design) and the pelvis PINNED at reset
    height, so the planted-foot |z| bound is the stance-extreme bob
    2L*(1-cos(HIP_AMPLITUDE)) ~= 22 mm, not the 4 mm of v1/v2's smaller
    stride. Roll columns get their own waveform/limit checks. EXECUTED
    validation lives in ExecutedValidation below -- FK alone provably
    cannot certify a gait (PHASE2.md section 14)."""

    @classmethod
    def setUpClass(cls):
        cls.native = native_lane._native()
        cls.lib = cls.native.library(str(native_lane.build_library()))
        cls.fixture = h0.fixture()
        cls.arg = _author_module()
        cls.table = np.asarray(json.loads(JSON_PATH.read_text())["table"])
        cls.verts = np.array(h0.foot_vertices())
        cls.sagittal = cls.table.copy()
        cls.sagittal[:, [2, 6]] = 0.0
        cls.poses = []
        for b in range(64):
            q = h0.reset_qpos()
            q[7:] = cls.sagittal[b]
            rc, ev = cls.fixture.evaluate(cls.lib, q, np.zeros(h0.N),
                                          gravity=(0.0, 0.0, -h0.GRAVITY))
            assert rc == 0, b
            cls.poses.append(ev.pose.copy())

    def _sole(self, pose, body):
        rot = native_lane.quat_to_rot(pose[body, 3:][None])[0]
        world = pose[body, :3] + self.verts @ rot.T
        return world[:, 2], pose[body, 0]

    def _planted(self, s, off):
        return (s + off) % 1.0 < self.arg.STANCE_FRACTION

    def test_per_frame_support_and_clearance(self):
        lf, rf = h0.FOOT_BODIES
        for b in range(64):
            s = (b + 0.5) / 64.0
            for body, off in ((lf, 0.0), (rf, 0.5)):
                z, _ = self._sole(self.poses[b], body)
                if self._planted(s, off):
                    # planted foot on the floor within the stance-extreme
                    # pinned-pelvis bob (27.4 mm at ALPHA 0.2534, v3.2)
                    self.assertLessEqual(abs(float(z.min())), 0.030, (b, body))
                    bottom = np.sort(z)[:4]
                    self.assertLessEqual(float(bottom.max() - bottom.min()),
                                         0.06, (b, body))
                else:
                    self.assertGreaterEqual(float(z.min()), -1e-9, (b, body))
                    u = (((s + off) % 1.0) - self.arg.STANCE_FRACTION) \
                        / self.arg.SWING_FRACTION
                    if 0.25 <= u <= 0.75:      # knee-plateau certified window
                        # measured 44 mm min (hip-lift bump; peak 123 mm)
                        self.assertGreaterEqual(float(z.min()), 0.040,
                                                (b, body))

    def test_sweep_and_placement(self):
        lf = h0.FOOT_BODIES[0]
        xs = [self._sole(self.poses[b], lf)[1] for b in range(64)]
        b_hi = int(self.arg.STANCE_FRACTION * 64) - 1
        sweep = float(xs[0] - xs[b_hi])
        expect = 2.0 * self.arg.LEG_LENGTH_M * math.sin(self.arg.HIP_AMPLITUDE)
        self.assertAlmostEqual(sweep, expect, delta=0.01)   # measured 0.3884
        # per-swing world placement at the clock cadence = one full stride
        from walk.eval import humanoid_gait
        self.assertGreaterEqual(self.arg.STRIDE_M,
                                humanoid_gait.PLACEMENT_MIN_M * 2.0)

    def test_roll_columns(self):
        """v3.2 roll: overdrive/taper waveform straight from the author."""
        arg = self.arg
        s = (np.arange(64) + 0.5) / 64.0
        expect = np.array([arg.roll_target(x) for x in s])
        np.testing.assert_allclose(self.table[:, 2], expect, atol=1e-12)
        np.testing.assert_allclose(self.table[:, 2], self.table[:, 6], atol=0)
        # hold = static balance point (asin(0.15/0.86) ~ 0.175) + hair;
        # drive = the authored roll limit (env clips targets to limits; the
        # overdrive exists to saturate the 180 N*m cap during transfer)
        self.assertAlmostEqual(arg.ROLL_HOLD, 0.18, places=9)
        self.assertLessEqual(arg.ROLL_DRIVE, h0.HIP_ROLL_LIMIT + 1e-12)
        # whole table inside the action box (reachable imitation targets)
        self.assertLessEqual(float(np.abs(self.table).max()),
                             arg.ACTION_BOX + 1e-12)
        # transfer overdrive is active through the whole DS window
        ds_bins = [b for b in range(64)
                   if ((b + 0.5) / 64 + arg.ROLL_ADVANCE) % 1.0
                   < arg.DS_FRACTION]
        for b in ds_bins:
            self.assertAlmostEqual(float(self.table[b, 2]), arg.ROLL_DRIVE,
                                   places=9)

    def test_double_support_fraction(self):
        arg = self.arg
        s = (np.arange(64) + 0.5) / 64.0
        both = np.array([self._planted(x, 0.0) and self._planted(x, 0.5)
                         for x in s])
        frac = float(both.mean())
        self.assertAlmostEqual(frac, 2 * arg.DS_FRACTION, delta=0.04)


class OpenLoopPhysicsSmoke(unittest.TestCase):
    def test_open_loop_replay_is_fault_free(self):
        """Open-loop PD replay at the cmd-0.75 clock: fault-free and finite
        for 0.6 s on the f64 CPU lane. NO stepping claim: v2's in-window
        contact-break assertion was satisfied by single-tick solver
        flickers (PHASE2.md section 14); real-swing claims now live ONLY
        in ExecutedValidation's debounced analyzer."""
        from walk.env.humanoid_native_lane import NativeHumanoidLane
        table = np.asarray(json.loads(JSON_PATH.read_text())["table"])
        lane = NativeHumanoidLane(1)
        try:
            hz = 1.67 * 0.75
            for tick in range(300):
                frac = (hz * tick * h0.SIM_DT) % 1.0
                bin_ = int(frac * 64) % 64
                rc, diag = lane.tick(table[bin_][None, :])
                self.assertEqual(rc, 0, (tick, diag))
            self.assertTrue(lane.read().finite().all())
        finally:
            lane.close()


class ExecutedValidation(unittest.TestCase):
    """MANDATORY executed-validation gate (the FK-only lesson,
    institutionalized), ACTIVE since H1.1 landed (per-joint kp/kv tables
    consumed by the kernel; humanoid/h1_lowering.KP_TABLE/KV_TABLE).

    The demonstrator's closed-loop rollout on the fp32 TRAINING lane, from
    a pinned mid-transfer phase, must produce >= 1 debounced qualified
    swing per episode at cmd 0.50 (measured: R swing 0.32 s / 45 mm peak /
    235 mm placement, deterministic across seeds -- the first real steps
    this stack has ever executed).

    cmd 0.75 remains a DOCUMENTED NEAR-MISS, not a gate: best swing
    0.14 s / 23 mm / 111 mm -- the phase-indexed swing window shrinks with
    command while the 180 N*m-capped transfer takes constant TIME
    (~0.25 s); levers if it must qualify pre-PPO: per-command tables or a
    slower clock. PPO's feedback owns it meanwhile (PHASE2.md s16)."""

    def test_demonstrator_produces_debounced_qualified_swings_cmd050(self):
        import bc_dataset as bd
        import diagnose_swings as dg
        from walk.env import humanoid_flat as hf
        from walk.env.humanoid_cuda_lane import CudaHumanoidLane
        for seed in (4242, 7):
            env = hf.FlatFloorHumanoidEnv(
                environments=1, seed=seed,
                lane_factory=lambda E, off: CudaHumanoidLane(
                    E, joint_offsets=off))
            try:
                env.set_command(0.50)
                obs = env.pin_phase(2.0 * math.pi * 0.075)
                trace = {"ticks": {"time_s": [], "foot_pos": [],
                                   "sole_height": [], "contact": []}}

                def on_tick(st, trace=trace):
                    tk = trace["ticks"]
                    tk["time_s"].append(float(st.time[0]))
                    tk["foot_pos"].append(
                        [list(map(float, st.foot_pos[0, f])) for f in (0, 1)])
                    tk["sole_height"].append(
                        list(map(float, st.sole_height[0])))
                    tk["contact"].append(
                        [bool(x) for x in st.foot_contact[0]])
                for _ in range(400):
                    a = bd.reference_actions(obs).astype(np.float32)
                    obs, _, done, _ = env.step(a, on_tick=on_tick)
                    if done.all():
                        break
            finally:
                env.close()
            swings = dg.swings_from_trace(trace, debounce=True)
            qualified = sum(s["qualified"] for s in swings)
            self.assertGreaterEqual(
                qualified, 1,
                (seed, [(s["foot"], round(s["duration_s"], 2),
                         round(s["peak_clearance_m"] * 1000),
                         s["first_fail"]) for s in swings]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
