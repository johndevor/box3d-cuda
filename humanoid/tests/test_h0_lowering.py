"""H0 lowering fixture gates: sourcing, topology, FK reset, bundle parity.

Run: .venv/bin/python -B humanoid/tests/test_h0_lowering.py

Checks (no time advancement anywhere):
- fixture invariants: B14/J12, child-of-joint-j-is-body-j+2 topology, total
  dynamic mass 68 kg (humanoid.rs:934-955 invariant), per-joint effort
  tiers, limits containing the home pose, box inertia formula;
- av1 FK at the reset qpos reproduces every authored body center (rotated
  y-up -> z-up) and orientation QX90, i.e. joint anchors close exactly;
- the reset is floor-clear-and-touching: both soles at world z == 0 within
  f32 rounding;
- cross-check EVERY registered constant against the materialized H0 bundle
  /Users/john/Code/world/evidence/humanoid-balance-r4-preflight-20260830/
  humanoid-cuda-training-bundle-v2.json (skipped when absent), remembering
  the bundle's per-env pelvis/torso mass randomization;
- cross-check naming/ordering against the frozen H0 contract module
  (box3d-arm-lab humanoid_h0.py, skipped when absent).
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "humanoid"))
sys.path.insert(0, str(ROOT / "walk" / "env"))

import h0_lowering as h0  # noqa: E402
import native_lane  # noqa: E402

BUNDLE = Path("/Users/john/Code/world/evidence/"
              "humanoid-balance-r4-preflight-20260830/"
              "humanoid-cuda-training-bundle-v2.json")
H0_CONTRACT_DIR = Path("/Users/john/Code/box3d-arm-lab/factory_os/"
                       "independent_validation")


class LoweringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = native_lane._native()
        cls.lib = cls.native.library(str(native_lane.build_library()))
        cls.fixture = h0.fixture()

    # -- topology / identity ------------------------------------------------
    def test_dimensions_and_topology(self):
        f = self.fixture
        self.assertEqual((f.J, f.B, f.N), (12, 14, 18))
        self.assertEqual(len(h0.BODY_NAMES), 14)
        self.assertEqual(len(h0.JOINT_NAMES), 12)
        # child of joint j is body j+2: the parent table must only reference
        # earlier bodies (tree order) and the H0 parent map exactly.
        expected_parents = (1, 2, 1, 4, 5, 1, 7, 8, 2, 10, 2, 12)
        for j, hinge in enumerate(f.hinge):
            self.assertEqual(hinge.parent, expected_parents[j], j)
            self.assertLess(hinge.parent, j + 2, "tree order")
            self.assertEqual(hinge.motor_enabled, 1)

    def test_total_dynamic_mass_is_68kg(self):
        total = sum(b[3] for b in h0.BODIES[1:])
        self.assertAlmostEqual(total, h0.TOTAL_DYNAMIC_MASS, places=12)
        self.assertAlmostEqual(
            sum(self.fixture.body[b].mass for b in range(1, 14)), 68.0,
            places=12)
        self.assertEqual(self.fixture.body[0].mass, 0.0)

    def test_effort_tiers_and_gains(self):
        self.assertEqual(h0.EFFORT, (180.0, 70.0, 180.0, 140.0, 140.0, 180.0,
                                     140.0, 140.0, 70.0, 70.0, 70.0, 70.0))
        for hinge in self.fixture.hinge:
            self.assertEqual(hinge.kp, np.float32(90.0))
            self.assertEqual(hinge.kv, np.float32(8.0))
            self.assertEqual(hinge.armature, 0.0)
            self.assertEqual(hinge.damping, 0.0)
            self.assertEqual(hinge.loss, 0.0)

    def test_limits_contain_home(self):
        for (_, _, _, _, lower, upper, _), target in zip(h0.JOINTS,
                                                         h0.HOME_TARGETS):
            self.assertLess(lower, target)
            self.assertGreater(upper, target)

    def test_box_inertia_formula(self):
        # [WS] solid box: I = m*(h_j^2 + h_k^2)/3 on half-extents,
        # independently re-derived by the H0 contract (humanoid_h0.py:284).
        m, half = 1.5, (0.23, 0.07, 0.14)
        self.assertEqual(
            h0.box_inertia(m, half),
            (m * (half[1] ** 2 + half[2] ** 2) / 3.0,
             m * (half[0] ** 2 + half[2] ** 2) / 3.0,
             m * (half[0] ** 2 + half[1] ** 2) / 3.0))

    def test_sim_dt_cadence_pins(self):
        # Phase 2: duck env contract cadence (0.02 s = 10 x 0.002 s); must
        # stay at or below the authored per-substep stability bound 1/240
        # (FEASIBILITY.md section 5).
        self.assertEqual(h0.AUTHORED_DT, 1.0 / 120.0)
        self.assertEqual(h0.AUTHORED_SUBSTEPS, 2)
        self.assertEqual(h0.SIM_DT, 0.002)
        self.assertLessEqual(h0.SIM_DT, h0.AUTHORED_DT / h0.AUTHORED_SUBSTEPS)
        self.assertEqual(h0.CONTROL_DT, 0.02)
        self.assertEqual(h0.TICKS_PER_CONTROL * h0.SIM_DT, h0.CONTROL_DT)

    # -- FK / reset -----------------------------------------------------------
    def test_fk_reset_reproduces_authored_centers(self):
        rc, ev = self.fixture.evaluate(self.lib, h0.reset_qpos(),
                                       np.zeros(18),
                                       gravity=(0.0, 0.0, -h0.GRAVITY))
        self.assertEqual(rc, 0)
        for b, (name, pos, _, _) in enumerate(h0.BODIES):
            if b == 0:
                continue
            np.testing.assert_allclose(ev.pose[b, :3], h0.y_up_to_z_up(pos),
                                       atol=1e-12, err_msg=name)
            np.testing.assert_allclose(np.abs(ev.pose[b, 3:]),
                                       np.abs(h0.QX90), atol=1e-12,
                                       err_msg=name)

    def test_reset_soles_touch_floor_exactly(self):
        rc, ev = self.fixture.evaluate(self.lib, h0.reset_qpos(),
                                       np.zeros(18),
                                       gravity=(0.0, 0.0, -h0.GRAVITY))
        self.assertEqual(rc, 0)
        verts = np.array(h0.foot_vertices())
        for b in h0.FOOT_BODIES:
            rot = native_lane.quat_to_rot(ev.pose[b, 3:][None])[0]
            world_z = ev.pose[b, 2] + (verts @ rot.T)[:, 2]
            self.assertAlmostEqual(float(world_z.min()), 0.0, places=12)
            self.assertAlmostEqual(float(world_z.max()), 0.14, places=12)

    def test_registration_round_trip(self):
        scene, _ = h0.scene(self.lib, 2)
        try:
            x = scene.read()
            self.assertEqual(x.q.shape, (2, 19))
            self.assertEqual(x.v.shape, (2, 18))
            np.testing.assert_array_equal(x.q[0], x.q[1])
            np.testing.assert_array_equal(x.q[0], h0.reset_qpos())
            np.testing.assert_array_equal(x.v[0], np.zeros(18))
            self.assertEqual(x.count[0], 0)
        finally:
            scene.close()

    # -- external cross-checks --------------------------------------------------
    def test_bundle_cross_check(self):
        if not BUNDLE.is_file():
            raise unittest.SkipTest(f"bundle not present: {BUNDLE}")
        bundle = json.loads(BUNDLE.read_text())
        reg = bundle["registration"]
        self.assertEqual((reg["bodies"], reg["joints"], reg["contact_pairs"]),
                         (14, 12, 2))
        self.assertEqual(reg["global_gravity_xyz"], [0.0, -20.0, 0.0])
        self.assertEqual(reg["global_friction"], h0.FRICTION)
        self.assertEqual(reg["global_restitution"], h0.RESTITUTION)
        self.assertEqual(np.float32(reg["dt"]), np.float32(h0.AUTHORED_DT))
        self.assertEqual(reg["substeps"], h0.AUTHORED_SUBSTEPS)
        pairs = [tuple(reg["joint_body_indices"][2 * j:2 * j + 2])
                 for j in range(12)]
        for j, (parent, child) in enumerate(pairs):
            self.assertEqual(parent, h0.JOINTS[j][1], j)
            self.assertEqual(child, j + 2, j)
        for j in range(12):
            np.testing.assert_allclose(
                reg["joint_parent_anchor"][3 * j:3 * j + 3], h0.JOINTS[j][2],
                atol=1e-7, err_msg=f"parent anchor {j}")
            np.testing.assert_allclose(
                reg["joint_child_anchor"][3 * j:3 * j + 3], h0.JOINTS[j][3],
                atol=1e-7, err_msg=f"child anchor {j}")
            np.testing.assert_allclose(
                reg["joint_axis_parent"][3 * j:3 * j + 3], h0.AXIS, atol=0)
            np.testing.assert_allclose(
                reg["joint_reference_xyzw"][4 * j:4 * j + 4],
                h0.REFERENCE_XYZW, atol=0)
            self.assertAlmostEqual(reg["joint_lower_limit"][j],
                                   h0.JOINTS[j][4], places=6)
            self.assertAlmostEqual(reg["joint_upper_limit"][j],
                                   h0.JOINTS[j][5], places=6)
            self.assertEqual(reg["joint_stiffness"][j], h0.KP)
            self.assertEqual(reg["joint_damping"][j], h0.KV)
        self.assertEqual(bundle["control"]["maximum_effort"], list(h0.EFFORT))
        self.assertEqual(set(bundle["control"]["maximum_speed"]),
                         {h0.SPEED_LIMIT})
        self.assertEqual(set(bundle["control"]["maximum_acceleration"]),
                         {h0.ACCELERATION_LIMIT})
        # initial block env 0: positions/quats/velocities + half extents.
        # Masses: pelvis+torso carry per-env DR scales (humanoid.rs:397-408);
        # authored values must sit inside the 0.90-1.10 window; every other
        # body must match exactly.
        init = bundle["initial"]
        for b, (name, pos, half, mass) in enumerate(h0.BODIES):
            state = init["state"][13 * b:13 * b + 13]
            np.testing.assert_allclose(state[:3], pos, atol=1e-7,
                                       err_msg=name)
            self.assertEqual(state[3:7], [0.0, 0.0, 0.0, 1.0], name)
            self.assertEqual(state[7:], [0.0] * 6, name)
            np.testing.assert_allclose(init["half_extents"][3 * b:3 * b + 3],
                                       half, atol=1e-7, err_msg=name)
            if b == 0:
                self.assertEqual(init["inverse_mass"][b], 0.0)
                continue
            got_mass = 1.0 / init["inverse_mass"][b]
            if name in ("pelvis", "torso"):
                self.assertLessEqual(abs(got_mass / mass - 1.0), 0.10 + 1e-6,
                                     name)
                scale = got_mass / mass
            else:
                self.assertAlmostEqual(got_mass, mass, places=5, msg=name)
                scale = 1.0
            expected = np.array(h0.box_inertia(mass, half)) * scale
            got = 1.0 / np.array(init["inverse_inertia"][3 * b:3 * b + 3])
            np.testing.assert_allclose(got, expected, rtol=1e-5, err_msg=name)

    def test_h0_contract_cross_check(self):
        if not (H0_CONTRACT_DIR / "humanoid_h0.py").is_file():
            raise unittest.SkipTest("H0 contract module not present")
        sys.path.insert(0, str(H0_CONTRACT_DIR))
        try:
            import humanoid_h0  # noqa: PLC0415
        finally:
            sys.path.pop(0)
        self.assertEqual(tuple(humanoid_h0.BODY_NAMES), h0.BODY_NAMES)
        # contract joint names use no side prefixes beyond ours
        self.assertEqual(tuple(humanoid_h0.JOINT_NAMES), h0.JOINT_NAMES)
        self.assertEqual(humanoid_h0.CONTACT_PAIR_INDICES, (0, 1))
        self.assertEqual(humanoid_h0.OBSERVATION_BODY_INDICES, (1, 2, 6, 9))
        self.assertEqual(humanoid_h0.OBSERVATION_BODY_INDICES[2:],
                         h0.FOOT_BODIES)
        # the contract's revolute q extractor agrees with our FK convention:
        # pose a test angle on the left knee via av1 FK, then recover it.
        f = self.fixture
        q = h0.reset_qpos()
        q[7 + 3] = 0.4  # left_knee
        rc, ev = f.evaluate(self.lib, q, np.zeros(18),
                            gravity=(0.0, 0.0, -h0.GRAVITY))
        self.assertEqual(rc, 0)
        parent = ev.pose[h0.JOINTS[3][1]]
        child = ev.pose[3 + 2]
        state_p = [*parent[:3], *parent[3:], 0, 0, 0, 0, 0, 0]
        state_c = [*child[:3], *child[3:], 0, 0, 0, 0, 0, 0]
        angle, rate = humanoid_h0.revolute_q_qdot(
            state_p, state_c, h0.AXIS, h0.REFERENCE_XYZW)
        self.assertAlmostEqual(angle, 0.4, places=10)
        self.assertEqual(rate, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
