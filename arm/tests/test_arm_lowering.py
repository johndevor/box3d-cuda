"""Arm lowering gates: URDF fidelity, topology, scaling laws, gain rules.

Run: .venv/bin/python -B arm/tests/test_arm_lowering.py

Gates:
- FK of the KR240 lowering reproduces the pinned KR240 joint-runtime trace's
  link COM world positions (kr240-joints.json world0_body_state, y-up frame
  mapped (x, z, -y)) to < 1 mm at its recorded joint coordinates (the trace
  carries its own controller's ~1e-4 rad sag);
- topology: child(j) = body j+2, kernel parent[0] == floor 0 (structural
  weld), oracle parent[0] == root 1; hinge REF/axis relations hold;
- the lite variant obeys the documented Froude scaling exactly (lengths x
  0.5, masses x 1/8, inertia x 1/32, effort x 1/16, angular rates x sqrt 2);
- fk_batch == fk; gains are float32-exact and satisfy every feasibility row
  (arm/feasibility_check.py); the HOME tip pin.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "arm"))

import arm_lowering as al  # noqa: E402
import feasibility_check as fc  # noqa: E402

KR240_JSON = Path("/Users/john/Code/box3d-arm-lab/factory_os/artifacts/"
                  "physics_comparisons/daytona-kr240-joints-20260829-r70/"
                  "kr240-joints.json")


class LoweringTests(unittest.TestCase):
    def test_fk_matches_pinned_kr240_trace(self):
        if not KR240_JSON.is_file():
            self.skipTest("pinned KR240 trace not available on this machine")
        d = json.loads(KR240_JSON.read_text())
        errs = []
        for row in d["correctness"]["measured_trace"][:6]:
            q = np.array(row["world0_coordinate_rad"])
            f = al.fk(al.KR240, q)
            theirs = np.array(row["world0_body_state"])[1:, :3]   # links 1..6
            ours = np.stack([f.link_pos[:, 0], f.link_pos[:, 2],
                             -f.link_pos[:, 1]], 1)               # -> y-up
            errs.append(float(np.abs(ours - theirs).max()))
        # step 0 (near rest): sub-mm; later rows carry their solver's own
        # per-joint anchor error (<= 0.44 mm each, json
        # maximum_joint_anchor_error_m) accumulated along the chain
        self.assertLess(errs[0], 1e-3, f"FK vs pinned trace @0: {errs[0]:.2e} m")
        self.assertLess(max(errs), 3e-3, f"FK vs pinned trace: {max(errs):.2e} m")
        print(f"fk_vs_kr240_trace step0 {errs[0]:.2e} m, max {max(errs):.2e} m",
              file=sys.stderr)

    def test_topology_and_hinge_convention(self):
        for s in al.VARIANTS.values():
            k = al.hinge_rows(s, kernel=True)
            o = al.hinge_rows(s, kernel=False)
            self.assertEqual([r[0] for r in k], [0, 2, 3, 4, 5, 6])
            self.assertEqual([r[0] for r in o], [1, 2, 3, 4, 5, 6])
            for rk, ro, jt in zip(k, o, s.joints):
                np.testing.assert_allclose(rk[3], ro[3])          # axis
                np.testing.assert_allclose(rk[4], ro[4])          # REF
                np.testing.assert_allclose(rk[2], ro[2])          # AC
                R = al.rpy_to_rot(jt.rpy)
                np.testing.assert_allclose(rk[3], R @ np.asarray(jt.axis),
                                           atol=1e-12)
                np.testing.assert_allclose(al.quat_to_rot(rk[4]), R, atol=1e-12)
            # joint-0 anchors differ ONLY by the base COM (floor vs root frame)
            np.testing.assert_allclose(k[0][1] - o[0][1], s.base.com, atol=1e-12)
        self.assertEqual((al.B, al.J, al.N, al.Q), (8, 6, 12, 13))

    def test_lite_scaling_laws(self):
        a, b = al.KR240, al.LITE
        sL, sm = 0.5, 0.125
        self.assertAlmostEqual(b.base.mass, a.base.mass * sm)
        for la, lb in zip(a.links, b.links):
            self.assertAlmostEqual(lb.mass, la.mass * sm)
            np.testing.assert_allclose(lb.com, np.asarray(la.com) * sL)
            np.testing.assert_allclose(lb.inertia,
                                       np.asarray(la.inertia) * sm * sL * sL)
        for ja, jb in zip(a.joints, b.joints):
            np.testing.assert_allclose(jb.xyz, np.asarray(ja.xyz) * sL)
            self.assertEqual(jb.rpy, ja.rpy)
            self.assertEqual(jb.axis, ja.axis)
            self.assertEqual((jb.lower, jb.upper), (ja.lower, ja.upper))
            self.assertAlmostEqual(jb.effort, ja.effort * sm * sL)
            self.assertAlmostEqual(jb.velocity, ja.velocity / np.sqrt(sL))
        self.assertAlmostEqual(al.reach(b), al.reach(a) * sL, places=9)
        self.assertAlmostEqual(al.moving_mass(a), 690.0)
        self.assertAlmostEqual(al.moving_mass(b), 86.25)
        self.assertAlmostEqual(al.moving_mass(a) + a.base.mass, 1120.0)

    def test_fk_batch_equals_fk_and_home_tip(self):
        rng = np.random.default_rng(11)
        for s in al.VARIANTS.values():
            Q = rng.uniform(-2.5, 2.5, (64, al.J))
            tip, origins = al.fk_batch(s, Q)
            for i, q in enumerate(Q):
                f = al.fk(s, q)
                np.testing.assert_allclose(tip[i], f.tip, atol=1e-12)
                np.testing.assert_allclose(origins[i], f.joint_pos, atol=1e-12)
        np.testing.assert_allclose(al.home_tip(al.KR240),
                                   [2.406, 0.0, 1.683], atol=2e-3)
        np.testing.assert_allclose(al.home_tip(al.LITE),
                                   0.5 * al.home_tip(al.KR240), atol=1e-12)

    def test_gains_and_feasibility_rows(self):
        for v, s in al.VARIANTS.items():
            kp, kv = al.gains(s)
            np.testing.assert_array_equal(kp, kp.astype(np.float32))   # f32-exact
            np.testing.assert_array_equal(kv, kv.astype(np.float32))
            self.assertTrue((kp > 0).all() and (kv > 0).all())
            t = fc.tables(s)
            self.assertEqual(fc.check(t), [], v)
            # every joint holds full horizontal extension unloaded
            self.assertTrue(all(r["hold_ratio"] < 0.6 for r in t["joints"]), v)
            # FINDING pinned: the URDF's bounded A2 effort cannot lift the
            # rated payload at FULL extension (13.8 vs 12 kN*m), but holds it
            # at the home pose
            a2 = t["joints"][1]
            self.assertGreater(a2["hold_plus_payload_ratio"], 1.0)
            self.assertLess(a2["home_hold_plus_payload_ratio"], 0.75)

    def test_certificates_scale_with_effort(self):
        ca, cb = al.certificates(al.KR240), al.certificates(al.LITE)
        self.assertAlmostEqual(ca["reference_impulse"], 12000.0 * al.SIM_DT)
        self.assertAlmostEqual(cb["reference_impulse"], 750.0 * al.SIM_DT)
        # 3-significant-digit rounding of the emitted pin
        self.assertAlmostEqual(ca["solve_tolerance"] / ca["reference_impulse"],
                               al.SOLVE_TOL_REL, delta=1e-2 * al.SOLVE_TOL_REL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
