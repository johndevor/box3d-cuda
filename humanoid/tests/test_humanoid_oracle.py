"""CPU-oracle home-hold gate for the H0 humanoid (Phase 1 acceptance).

Run: .venv/bin/python -B humanoid/tests/test_humanoid_oracle.py

THE GATE (mission spec): zero-action home-hold for 2 simulated seconds
(480 ticks x 1/240 s) on the f64 idv1 CPU lane, with
  - every native step accepted (rc == 0, native_status == 0),
  - momentum residual <= 1e-8 at EVERY step,
  - both feet in contact at every post-step read,
  - final tilt < 5 degrees (body +Y vs world +Z),
plus supporting health bounds mirroring the duck's home-hold protocol
(experimental/integrated_duck_v1/run_home_hold.py GATES) and the torque
headroom fact that licenses the fp32 kernel's scalar effort cap.
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
import h0_lowering as h0  # noqa: E402

TICKS = 480                       # 2.0 s at SIM_DT = 1/240
MOMENTUM_GATE = 1e-8
TILT_GATE_DEG = 5.0
# duck home-hold health bounds (run_home_hold.py:21), same physical meaning
JOINT_LIMIT_VIOLATION_RAD = 0.05
PENETRATION_M = 0.01
MAX_JOINT_SPEED = 250.0
MAX_BASE_LINEAR_SPEED = 20.0
MAX_BASE_ANGULAR_SPEED = 250.0


class HumanoidOracleHomeHold(unittest.TestCase):
    def test_home_hold_2s_gates(self):
        lane = hn.NativeHumanoidLane(1)
        try:
            home = lane.home_joint_q[None, :]
            lim = lane.joint_limits
            worst = dict(momentum=0.0, torque=0.0, penetration=0.0,
                         iterations=0)
            for t in range(TICKS):
                pre = lane.read()
                torque = np.clip(
                    lane.kp * (home - pre.q[:, 7:]) - lane.kv * pre.v[:, 6:],
                    -lane.effort_cap, lane.effort_cap)
                worst["torque"] = max(worst["torque"],
                                      float(np.abs(torque).max()))
                rc, diag = lane.tick(home)
                self.assertEqual(rc, 0, (t, diag))
                d = diag[0]
                self.assertEqual(d["native_status"], 0, (t, d))
                self.assertLessEqual(d["momentum_residual"], MOMENTUM_GATE,
                                     (t, d))
                worst["momentum"] = max(worst["momentum"],
                                        d["momentum_residual"])
                worst["penetration"] = max(worst["penetration"],
                                           d["maximum_penetration"])
                worst["iterations"] = max(worst["iterations"],
                                          d["iterations"])
                x = lane.read()
                self.assertTrue(x.finite().all(), t)
                self.assertTrue(bool(x.foot_contact.all()),
                                (t, x.foot_contact))
                violation = np.maximum(
                    np.maximum(lim[:, 0] - x.q[0, 7:],
                               x.q[0, 7:] - lim[:, 1]), 0.0).max()
                self.assertLessEqual(float(violation),
                                     JOINT_LIMIT_VIOLATION_RAD, t)
                self.assertLessEqual(float(np.abs(x.v[0, 6:]).max()),
                                     MAX_JOINT_SPEED, t)
                self.assertLessEqual(float(np.linalg.norm(x.v[0, :3])),
                                     MAX_BASE_LINEAR_SPEED, t)
                self.assertLessEqual(float(np.linalg.norm(x.v[0, 3:6])),
                                     MAX_BASE_ANGULAR_SPEED, t)
            x = lane.read()
            self.assertEqual(int(x.count[0]), TICKS)
            tilt_deg = math.degrees(float(hn.tilt(x.q[0, 3:7][None])[0]))
            self.assertLess(tilt_deg, TILT_GATE_DEG)
            self.assertLessEqual(worst["penetration"], PENETRATION_M)
            # height held (not part of the mission gate; sanity)
            self.assertGreater(float(x.q[0, 2]),
                               0.95 * lane.home_root_height)
            # torque headroom: the fp32 kernel bakes the scalar MIN effort
            # tier (70 N*m); prove the clamp can never bind in this regime
            # so scalar-vs-table is a non-difference for Phase 1 parity.
            self.assertLess(worst["torque"], 0.01 * min(h0.EFFORT))
            print(f"home_hold 2s: momentum<={worst['momentum']:.2e} "
                  f"tilt={tilt_deg:.2e} deg torque<={worst['torque']:.2e} "
                  f"pen<={worst['penetration']:.2e} m "
                  f"iters<={worst['iterations']}", file=sys.stderr)
        finally:
            lane.close()

    def test_restore_returns_to_reset(self):
        lane = hn.NativeHumanoidLane(2)
        try:
            before = lane.read()
            for _ in range(10):
                rc, _ = lane.tick(lane.home_joint_q[None, :].repeat(2, 0))
                self.assertEqual(rc, 0)
            lane.restore([1, 0])       # env 0 back to reset, env 1 running
            x = lane.read()
            np.testing.assert_array_equal(x.q[0], before.q[0])
            self.assertEqual(int(x.count[0]), 0)
            self.assertEqual(int(x.count[1]), 10)
        finally:
            lane.close()

    def test_authored_dt_would_be_unstable_documented(self):
        """Pin the 1/120 instability finding (FEASIBILITY.md section 5).

        The linearized one-tick map of this stack's semi-implicit update at
        the authored engine dt (1/120, PD applied once per tick) has
        spectral radius > 1.5; at the per-substep 1/240 it is <= 1 + 1e-9.
        Uses av1's mass matrix at the reset pose; no stepping.
        """
        native = hn.native_lane._native()
        lib = native.library(str(hn.build_library()))
        f = h0.fixture()
        rc, ev = f.evaluate(lib, h0.reset_qpos(), np.zeros(18),
                            gravity=(0.0, 0.0, -h0.GRAVITY))
        self.assertEqual(rc, 0)
        minv = np.linalg.inv(ev.mass)
        n = 18
        kp = np.zeros((n, n))
        kd = np.zeros((n, n))
        for j in range(12):
            kp[6 + j, 6 + j] = h0.KP
            kd[6 + j, 6 + j] = h0.KV
        def radius(dt):
            a = np.block(
                [[np.eye(n) - dt * minv @ kd, -dt * minv @ kp],
                 [dt * (np.eye(n) - dt * minv @ kd),
                  np.eye(n) - dt * dt * minv @ kp]])
            return float(np.abs(np.linalg.eigvals(a)).max())
        self.assertGreater(radius(h0.AUTHORED_DT), 1.5)
        self.assertLessEqual(radius(h0.SIM_DT), 1.0 + 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
