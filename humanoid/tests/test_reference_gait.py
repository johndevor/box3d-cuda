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
    """Per-frame FK validation on the real fixture (no dynamics)."""

    @classmethod
    def setUpClass(cls):
        cls.native = native_lane._native()
        cls.lib = cls.native.library(str(native_lane.build_library()))
        cls.fixture = h0.fixture()
        cls.table = np.asarray(json.loads(JSON_PATH.read_text())["table"])
        cls.verts = np.array(h0.foot_vertices())
        # SAGITTAL validation uses the roll-zeroed table: the roll columns
        # are LOAD COMPENSATION (0.25 rad commands what sags to the desired
        # ~0.1 rad lean under the single-support gravity moment); their
        # pure-FK pose over-leans by design, so stance flatness/clearance
        # are the sagittal columns' contract and the roll columns get their
        # own amplitude/timing checks (test_roll_columns).
        cls.sagittal = cls.table.copy()
        cls.sagittal[:, [2, 6]] = 0.0
        # nominal pelvis path: constant height (straight-leg bob <= 3.3 mm
        # is measured on the FEET below, not compensated here)
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

    def test_per_frame_alternation_clearance_and_flatness(self):
        for b in range(64):
            phase = 2.0 * math.pi * (b + 0.5) / 64.0
            left, right = h0.FOOT_BODIES
            stance, swing = (left, right) if math.sin(phase) >= 0 else (right, left)
            z_st, _ = self._sole(self.poses[b], stance)
            z_sw, _ = self._sole(self.poses[b], swing)
            # stance foot: whole sole on the floor within the pelvis bob;
            # flatness over the BOTTOM face (lowest 4 of the 8 box corners)
            self.assertLessEqual(abs(float(z_st.min())), 0.004, b)
            bottom = np.sort(z_st)[:4]
            self.assertLessEqual(float(bottom.max() - bottom.min()), 0.02, b)
            # swing foot never below the floor
            self.assertGreaterEqual(float(z_sw.min()), -1e-9, b)
            # certified mid-swing window: whole-sole clearance >= 30 mm
            if abs(math.sin(phase)) >= math.sqrt(0.5):
                self.assertGreaterEqual(float(z_sw.min()), 0.030, b)

    def test_peak_clearance_and_sweep(self):
        clear = []
        left_x = []
        for b in range(64):
            phase = 2.0 * math.pi * (b + 0.5) / 64.0
            z_l, x_l = self._sole(self.poses[b], h0.FOOT_BODIES[0])
            left_x.append(x_l)
            if math.sin(phase) < 0:                 # left swing bins
                clear.append(float(z_l.min()))
        self.assertGreaterEqual(max(clear), 0.050)  # measured ~56 mm peak
        # stance-foot sweep over the stance half-cycle = half-stride 0.15 m
        sweep = left_x[0] - left_x[31]              # bin 0 (td) -> bin 31 (lo)
        self.assertAlmostEqual(float(sweep), 0.15, delta=0.005)
        # -> per-swing world placement at the clock cadence = full stride
        # 0.30 m: inside the judge's band (>= PLACEMENT_MIN_M with 2x margin)
        from walk.eval import humanoid_gait
        self.assertGreaterEqual(2.0 * float(sweep),
                                humanoid_gait.PLACEMENT_MIN_M * 1.9)

    def test_roll_columns(self):
        """H1 roll channel: amplitude, timing and limit checks."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "arg", ROOT / "humanoid" / "author_reference_gait.py")
        arg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arg)
        p = 2.0 * np.pi * (np.arange(64) + 0.5) / 64.0
        expect = arg.ROLL_AMPLITUDE * np.sin(p + arg.ROLL_PHASE_ADVANCE)
        np.testing.assert_allclose(self.table[:, 2], expect, atol=1e-12)
        # inside the authored roll limit with headroom
        self.assertLessEqual(float(np.abs(self.table[:, 2]).max()),
                             h0.HIP_ROLL_LIMIT - 0.1)
        # action-box reachable (targets = 0.25 <= ACTION_SCALE 0.5)
        self.assertLessEqual(float(np.abs(self.table[:, 2]).max()) / 0.5, 1.0)
        # lean is TOWARD the stance side and pre-established at liftoff:
        # positive (pelvis -> left foot at -y) through most of left stance
        left_stance = np.sin(p) >= 0
        self.assertGreater(float(self.table[left_stance, 2].mean()), 0.1)
        self.assertGreater(float(self.table[0, 2]), 0.05)   # already leaning


class OpenLoopPhysicsSmoke(unittest.TestCase):
    def test_open_loop_replay_lifts_alternating_feet(self):
        """Open-loop PD replay at the cmd-0.75 clock: fault-free 0.6 s, each
        foot airborne at least once in its own swing window. Falling later
        is expected (open-loop biped); NOT a stability or walking claim."""
        from walk.env.humanoid_native_lane import NativeHumanoidLane
        table = np.asarray(json.loads(JSON_PATH.read_text())["table"])
        lane = NativeHumanoidLane(1)
        try:
            hz = 3.33 * 0.75                    # env clock, cmd 0.75
            lf, rf = h0.FOOT_BODIES
            air_in_window = {lf: False, rf: False}
            for tick in range(300):             # 0.6 s = 1.5 cycles
                t = tick * h0.SIM_DT
                frac = (hz * t) % 1.0
                bin_ = int(frac * 64) % 64
                rc, diag = lane.tick(table[bin_][None, :])
                self.assertEqual(rc, 0, (tick, diag))
                x = lane.read()
                self.assertTrue(x.finite().all(), tick)
                phase = 2.0 * math.pi * frac
                if math.sin(phase) < 0 and not x.foot_contact[0, 0]:
                    air_in_window[lf] = True    # left airborne in left swing
                if math.sin(phase) >= 0 and not x.foot_contact[0, 1]:
                    air_in_window[rf] = True    # right airborne in right swing
            self.assertTrue(air_in_window[lf], "left foot never lifted")
            self.assertTrue(air_in_window[rf], "right foot never lifted")
        finally:
            lane.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
