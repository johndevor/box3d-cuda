"""CPU-oracle home-hold gates for every humanoid FAMILY variant.

Run: .venv/bin/python -B humanoid/tests/test_variant_oracle.py

The variant twin of test_humanoid_oracle.py (the H1 gate, unchanged),
PARAMETRIZED over humanoid/h1_family.py's non-base members: zero-action
home-hold for 2 simulated seconds (1000 ticks x 0.002 s) on the f64 idv1
CPU lane built from the VARIANT's lowering, with
  - every native step accepted (rc == 0, native_status == 0),
  - momentum residual <= 1e-8 at EVERY step,
  - both feet in contact at every post-step read,
  - final tilt < 5 degrees, height held,
plus the duck home-hold health bounds and restore-to-reset, and the
1/120 instability pin re-derived with the VARIANT's per-joint tables.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "humanoid"))

from walk.env import humanoid_native_lane as hn  # noqa: E402
import h1_family as fam  # noqa: E402

VARIANTS = tuple(v for v in fam.variant_names() if v != "h1")
TICKS = 1000
MOMENTUM_GATE = 1e-8
TILT_GATE_DEG = 5.0
JOINT_LIMIT_VIOLATION_RAD = 0.05
PENETRATION_M = 0.01
MAX_JOINT_SPEED = 250.0
MAX_BASE_LINEAR_SPEED = 20.0
MAX_BASE_ANGULAR_SPEED = 250.0


class VariantOracleHomeHold(unittest.TestCase):
    def _home_hold(self, variant: str):
        lw = fam.load_lowering(variant)
        lane = hn.NativeHumanoidLane(1, variant=variant)
        try:
            self.assertEqual(lane.variant, variant)
            self.assertAlmostEqual(lane.home_root_height,
                                   float(lw.reset_qpos()[2]), places=12)
            np.testing.assert_array_equal(lane.effort_cap, np.array(lw.EFFORT))
            home = lane.home_joint_q[None, :]
            lim = lane.joint_limits
            worst = dict(momentum=0.0, torque=0.0, penetration=0.0, iterations=0)
            for t in range(TICKS):
                pre = lane.read()
                torque = np.clip(
                    lane.kp * (home - pre.q[:, 7:]) - lane.kv * pre.v[:, 6:],
                    -lane.effort_cap, lane.effort_cap)
                worst["torque"] = max(worst["torque"], float(np.abs(torque).max()))
                rc, diag = lane.tick(home)
                self.assertEqual(rc, 0, (variant, t, diag))
                d = diag[0]
                self.assertEqual(d["native_status"], 0, (variant, t, d))
                self.assertLessEqual(d["momentum_residual"], MOMENTUM_GATE,
                                     (variant, t, d))
                worst["momentum"] = max(worst["momentum"], d["momentum_residual"])
                worst["penetration"] = max(worst["penetration"],
                                           d["maximum_penetration"])
                worst["iterations"] = max(worst["iterations"], d["iterations"])
                x = lane.read()
                self.assertTrue(x.finite().all(), (variant, t))
                self.assertTrue(bool(x.foot_contact.all()),
                                (variant, t, x.foot_contact))
                violation = np.maximum(
                    np.maximum(lim[:, 0] - x.q[0, 7:], x.q[0, 7:] - lim[:, 1]),
                    0.0).max()
                self.assertLessEqual(float(violation), JOINT_LIMIT_VIOLATION_RAD)
                self.assertLessEqual(float(np.abs(x.v[0, 6:]).max()),
                                     MAX_JOINT_SPEED)
                self.assertLessEqual(float(np.linalg.norm(x.v[0, :3])),
                                     MAX_BASE_LINEAR_SPEED)
                self.assertLessEqual(float(np.linalg.norm(x.v[0, 3:6])),
                                     MAX_BASE_ANGULAR_SPEED)
            x = lane.read()
            self.assertEqual(int(x.count[0]), TICKS)
            tilt_deg = math.degrees(float(hn.tilt(x.q[0, 3:7][None])[0]))
            self.assertLess(tilt_deg, TILT_GATE_DEG)
            self.assertLessEqual(worst["penetration"], PENETRATION_M)
            self.assertGreater(float(x.q[0, 2]), 0.95 * lane.home_root_height)
            self.assertLess(worst["torque"], 0.01 * min(lw.EFFORT))
            print(f"[{variant}] home_hold 2s: momentum<={worst['momentum']:.2e} "
                  f"tilt={tilt_deg:.2e} deg torque<={worst['torque']:.2e} "
                  f"pen<={worst['penetration']:.2e} m "
                  f"iters<={worst['iterations']}", file=sys.stderr)
        finally:
            lane.close()

    def _restore(self, variant: str):
        lane = hn.NativeHumanoidLane(2, variant=variant)
        try:
            before = lane.read()
            for _ in range(10):
                rc, _ = lane.tick(lane.home_joint_q[None, :].repeat(2, 0))
                self.assertEqual(rc, 0)
            lane.restore([1, 0])
            x = lane.read()
            np.testing.assert_array_equal(x.q[0], before.q[0])
            self.assertEqual(int(x.count[0]), 0)
            self.assertEqual(int(x.count[1]), 10)
        finally:
            lane.close()

    def _dt_pin(self, variant: str):
        lw = fam.load_lowering(variant)
        native = hn.native_lane._native()
        lib = native.library(str(hn.build_library()))
        f = lw.fixture()
        rc, ev = f.evaluate(lib, lw.reset_qpos(), np.zeros(lw.N),
                            gravity=(0.0, 0.0, -lw.GRAVITY))
        self.assertEqual(rc, 0)
        minv = np.linalg.inv(ev.mass)
        n = lw.N
        kp, kd = np.zeros((n, n)), np.zeros((n, n))
        for j in range(lw.J):
            kp[6 + j, 6 + j] = lw.KP_TABLE[j]
            kd[6 + j, 6 + j] = lw.KV_TABLE[j]

        def radius(dt):
            a = np.block([[np.eye(n) - dt * minv @ kd, -dt * minv @ kp],
                          [dt * (np.eye(n) - dt * minv @ kd),
                           np.eye(n) - dt * dt * minv @ kp]])
            return float(np.abs(np.linalg.eigvals(a)).max())
        self.assertGreater(radius(lw.AUTHORED_DT), 1.5, variant)
        self.assertLessEqual(radius(lw.SIM_DT), 1.0 + 1e-9, variant)

    def test_variant_env_refuses_mismatched_lane(self):
        from walk.env import humanoid_flat as hf
        with self.assertRaises(ValueError):
            hf.FlatFloorHumanoidEnv(
                environments=1, variant="h1_tall",
                lane_factory=lambda E, off: hn.NativeHumanoidLane(E))


for _v in VARIANTS:
    def _mk(name, v=_v):
        return lambda self: getattr(self, name)(v)
    setattr(VariantOracleHomeHold, f"test_home_hold_2s_gates_{_v}", _mk("_home_hold"))
    setattr(VariantOracleHomeHold, f"test_restore_returns_to_reset_{_v}", _mk("_restore"))
    setattr(VariantOracleHomeHold, f"test_authored_dt_unstable_pin_{_v}", _mk("_dt_pin"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
