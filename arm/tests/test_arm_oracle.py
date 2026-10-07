"""CPU-oracle home-hold gate for the fixed-base arm (both variants).

Run: .venv/bin/python -B arm/tests/test_arm_oracle.py

THE GATE (mission spec): zero-action home-hold for 8 simulated seconds
(4000 ticks x 0.002 s) on the f64 idv1 CPU lane under gravity with the
joints holding, with
  - every native step accepted (rc == 0, native_status == 0),
  - momentum residual <= 1e-8 at EVERY step,
  - BASE DRIFT < 1e-6 m (translation) and < 1e-6 rad (rotation) at the
    end -- the virtual weld (arm_lowering.weld_force) is the fixed base,
  - joint sag under gravity within the design bound (SAG_MAX_RAD 0.01 at
    full extension; the home pose is less loaded), joints at rest,
  - zero contact points (P = 0: the arm never touches the floor).
Measured: drift ~1e-19 m / ~1e-17 rad, momentum 0.0, 1 solver sweep/tick.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "arm"))

from walk.env import arm_native_lane as an  # noqa: E402
import arm_lowering as al  # noqa: E402

TICKS = 4000                      # 8.0 s at SIM_DT = 0.002
MOMENTUM_GATE = 1e-8
DRIFT_GATE_M = 1e-6
DRIFT_GATE_RAD = 1e-6


class ArmOracleHomeHold(unittest.TestCase):
    def _hold(self, variant: str):
        lane = an.NativeArmLane(1, variant)
        try:
            s = lane.spec
            home = lane.home_joint_q[None, :]
            x0 = lane.read()
            worst = dict(momentum=0.0, iterations=0, torque=0.0)
            for t in range(TICKS):
                pre = lane.read()
                torque = np.clip(lane.kp * (home - pre.q[:, 7:])
                                 - lane.kv * pre.v[:, 6:],
                                 -lane.effort_cap, lane.effort_cap)
                worst["torque"] = max(worst["torque"],
                                      float((np.abs(torque) / lane.effort_cap).max()))
                rc, diag = lane.tick(home)
                self.assertEqual(rc, 0, (variant, t, diag))
                d = diag[0]
                self.assertEqual(d["native_status"], 0, (variant, t, d))
                self.assertLessEqual(d["momentum_residual"], MOMENTUM_GATE, (t, d))
                self.assertEqual(d["contact_points"], 0, (variant, t))
                worst["momentum"] = max(worst["momentum"], d["momentum_residual"])
                worst["iterations"] = max(worst["iterations"], d["iterations"])
            x = lane.read()
            self.assertTrue(x.finite().all())
            self.assertEqual(int(x.count[0]), TICKS)
            drift = float(np.abs(x.q[0, :3] - x0.q[0, :3]).max())
            rot = float(2.0 * np.abs(x.q[0, 3:6]).max())
            self.assertLess(drift, DRIFT_GATE_M, "base translation drift")
            self.assertLess(rot, DRIFT_GATE_RAD, "base rotation drift")
            self.assertLess(float(np.abs(x.v[0, :6]).max()), 1e-9, "base at rest")
            sag = np.abs(x.q[0, 7:] - np.asarray(s.home_q))
            self.assertLess(float(sag.max()), al.SAG_MAX_RAD, "gravity sag")
            self.assertLess(float(np.abs(x.v[0, 6:]).max()), 1e-6, "joints at rest")
            # the flange stayed where FK says the sagged pose puts it
            tip = al.tip_from_body_state(s, x.body_state)[0]
            np.testing.assert_allclose(tip, al.fk(s, x.q[0, 7:]).tip, atol=1e-6)
            print(f"{variant} home_hold 8s: drift={drift:.1e} m rot={rot:.1e} rad "
                  f"sag<={sag.max():.4f} rad torque/cap<={worst['torque']:.3f} "
                  f"momentum<={worst['momentum']:.1e} iters<={worst['iterations']}",
                  file=sys.stderr)
        finally:
            lane.close()

    def test_kr240_home_hold_8s(self):
        self._hold("kr240")

    def test_lite_home_hold_8s(self):
        self._hold("lite")

    def test_restore_returns_to_reset(self):
        lane = an.NativeArmLane(2, "kr240")
        try:
            before = lane.read()
            tgt = lane.home_joint_q[None, :].repeat(2, 0) + 0.2
            for _ in range(20):
                rc, _ = lane.tick(tgt)
                self.assertEqual(rc, 0)
            lane.restore([1, 0])
            x = lane.read()
            np.testing.assert_array_equal(x.q[0], before.q[0])
            self.assertEqual(int(x.count[0]), 0)
            self.assertEqual(int(x.count[1]), 20)
            self.assertGreater(float(np.abs(x.q[1, 7:] - before.q[1, 7:]).max()), 1e-3)
        finally:
            lane.close()

    def test_weld_cancels_static_gravity_exactly(self):
        """At rest the applied root force must equal the root dofs' gravity
        bias (from av1_evaluate) so the welded root feels zero net
        generalized force -- the reason the drift is ~1e-19, not 1e-6."""
        native = an.native_lane._native()
        lib = native.library(str(an.build_library()))
        for s in al.VARIANTS.values():
            f = al.fixture(s)
            q = al.reset_qpos(s)
            rc, ev = f.evaluate(lib, q, np.zeros(al.N), gravity=al.GRAVITY_VEC)
            self.assertEqual(rc, 0)
            link_pos = ev.pose[2:, :3][None]
            force = al.weld_force(s, q[None], np.zeros((1, al.N)), link_pos)[0]
            # f = applied - bias on the root dofs -> should vanish (relative)
            scale = float(np.abs(ev.bias[:6]).max())
            np.testing.assert_allclose(force[:6], ev.bias[:6], rtol=0, atol=1e-9 * scale)
            np.testing.assert_array_equal(force[6:], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
