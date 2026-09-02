"""Parity gates: fp32 serial arm kernel builds vs the f64 CPU oracle.

Run: .venv/bin/python -B arm/tests/test_arm_serial_parity.py

The arm twin of humanoid/tests/test_humanoid_serial_parity.py for the
PURE-REGENERATION builds (one per variant): the unchanged kernel sources
compiled against arm/include/<variant>/duck_model.h (shadowed by include
order; zero kernel edits; duck and humanoid builds untouched).

Gates (both variants unless noted):
 (a) header drift: each committed arm header matches a fresh regeneration
     from arm/arm_lowering.py + the python contract pins; the duck's
     committed header regenerates byte-identically (untouched);
 (b) home-hold 8 s: kernel (structural weld: joint 0 parented on the static
     floor) vs oracle (virtual weld) -- joint |dq| < 1e-4 rad, flange
     < 1 mm, zero contact points, fp32 momentum residual within the
     header's DW_MOMENTUM_TOLERANCE; the kernel's PHANTOM root free-falls
     exactly 1/2 g t^2 (decoupled), while the arm ignores it;
 (c) scripted target sequence 3 s (multi-joint sinusoids inside the
     limits, joint speeds up to the URDF limits): joint < 1e-4 rad, flange
     < 1 mm;
 (d) limit-hitting segment (a3 driven into its upper soft limit):
     bounded divergence, no fault on either lane, the limit row engages on
     both (active_limits > 0), excess < LIMIT_TOL of the judge;
 (e) bit-identical determinism between two scenes in one build;
 (f) dwc1_info dims/tables == lowering; set_state round trip.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "arm"))

from walk.env import arm_cuda_lane as ac  # noqa: E402
from walk.env import arm_native_lane as an  # noqa: E402
import arm_lowering as al  # noqa: E402

TOOLS = ROOT / "experimental" / "duck_cuda" / "tools"
VARIANTS = ("kr240", "lite")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _header_float(variant: str, macro: str) -> float:
    text = ac.header_path(variant).read_text()
    m = re.search(rf"#define {macro} ([0-9.eE+-]+)f?", text)
    assert m, macro
    return float(m.group(1))


def _tips(s, xc, xg):
    return (al.tip_from_body_state(s, xc.body_state),
            al.tip_from_body_state(s, xg.body_state))


class ArmSerialParityTests(unittest.TestCase):
    # -- gate (a): generated headers --------------------------------------
    def test_arm_headers_drift(self):
        gen = _load("generate_model_arm")
        low = gen.load_lowering()
        for v in VARIANTS:
            fresh = gen.emit(low, v)
            committed = ac.header_path(v).read_text()
            self.assertEqual(fresh, committed,
                             f"arm/include/{v}/duck_model.h drifted; regenerate "
                             f"with tools/generate_model_arm.py --variant {v}")
            self.assertIn("DW_HINGE_PARENT[DW_J] = {0,2,3,4,5,6}", committed)

    def test_duck_header_untouched_by_arm_work(self):
        gen = _load("generate_model")
        native, fixture, cm = gen.load_fixture()
        fresh = gen.emit(native, fixture, cm)
        committed = (ROOT / "experimental" / "duck_cuda" / "include"
                     / "duck_model.h").read_text()
        self.assertEqual(fresh, committed, "duck duck_model.h must stay byte-identical")

    # -- gate (b): home-hold parity + phantom decoupling -----------------------
    def _hold(self, v):
        s = al.spec(v)
        cpu, gpu = an.NativeArmLane(1, v), ac.CudaArmLane(1, v)
        try:
            home = cpu.home_joint_q[None, :]
            mom_tol = _header_float(v, "DW_MOMENTUM_TOLERANCE")
            worst_mom = 0.0
            for block in range(8):                 # 8 x 500 ticks = 8 s
                for _ in range(500):
                    rc, d = cpu.tick(home)
                    self.assertEqual(rc, 0, (v, block, d))
                rc, dg = gpu.tick_block(home, 500)
                self.assertEqual(rc, 0, (v, block, dg))
                self.assertEqual(dg[0]["contact_points"], 0)
                worst_mom = max(worst_mom, dg[0]["momentum_residual"])
                self.assertLessEqual(dg[0]["momentum_residual"], mom_tol)
            xc, xg = cpu.read(), gpu.read()
            tipc, tipg = _tips(s, xc, xg)
            dq = float(np.abs(xg.q[0, 7:] - xc.q[0, 7:]).max())
            dtip = float(np.abs(tipg - tipc).max())
            self.assertLess(dq, 1e-4, "home-hold joint parity")
            self.assertLess(dtip, 1e-3, "home-hold flange parity")
            self.assertEqual(int(gpu.contact_points().sum()), 0)
            # phantom root: exact decoupled free fall under the kernel's
            # semi-implicit Euler (v += g dt; z += v dt): z_n = -g dt^2 n(n+1)/2
            n = 4000
            z_expect = -al.GRAVITY * al.SIM_DT ** 2 * n * (n + 1) / 2   # -313.998
            self.assertAlmostEqual(float(xg.q[0, 2]), z_expect, delta=0.01)
            self.assertLess(float(np.abs(xg.q[0, :2]).max()), 1e-4)   # fp32 @ z=-314
            np.testing.assert_allclose(xg.q[0, 3:7], [0, 0, 0, 1], atol=1e-4)
            # the oracle's welded root did not move
            self.assertLess(float(np.abs(xc.q[0, :3]).max()), 1e-6)
            print(f"{v} home_hold_parity dq={dq:.2e} rad dtip={dtip*1e3:.3f} mm "
                  f"phantom z={xg.q[0, 2]:.2f} m mom<={worst_mom:.1e}",
                  file=sys.stderr)
        finally:
            cpu.close()
            gpu.close()

    def test_home_hold_parity_kr240(self):
        self._hold("kr240")

    def test_home_hold_parity_lite(self):
        self._hold("lite")

    # -- gate (c)+(d): scripted sequences ---------------------------------------
    def _scripted(self, v):
        s = al.spec(v)
        cpu, gpu = an.NativeArmLane(1, v), ac.CudaArmLane(1, v)
        try:
            lim = al.joint_limits(s)
            hq = np.asarray(s.home_q)
            amp = np.array([0.6, 0.35, 0.5, 0.8, 0.6, 1.0])
            hz = np.array([0.3, 0.25, 0.4, 0.5, 0.45, 0.6])
            worst_q = worst_tip = 0.0
            max_speed = 0.0
            for step in range(150):                     # 3 s
                tt = step * al.CONTROL_DT
                tgt = hq + amp * np.sin(2 * np.pi * hz * tt + np.arange(6))
                tgt = np.clip(tgt, lim[:, 0], lim[:, 1])[None]
                rc, dc = cpu.tick_block(tgt, 10)
                self.assertEqual(rc, 0, (v, step, dc))
                rc, dg = gpu.tick_block(tgt, 10)
                self.assertEqual(rc, 0, (v, step, dg))
                xc, xg = cpu.read(), gpu.read()
                tipc, tipg = _tips(s, xc, xg)
                worst_q = max(worst_q, float(np.abs(xg.q[0, 7:] - xc.q[0, 7:]).max()))
                worst_tip = max(worst_tip, float(np.abs(tipg - tipc).max()))
                max_speed = max(max_speed, float(np.abs(xc.v[0, 6:]).max()))
            self.assertLess(worst_q, 1e-4, "scripted joint parity")
            self.assertLess(worst_tip, 1e-3, "scripted flange parity")
            self.assertGreater(max_speed, 0.5, "the sequence must actually move")
            # (d) drive a3 into its upper soft limit for 1 s on both lanes
            tgt = hq.copy()
            tgt[2] = lim[2, 1] + 0.5                    # beyond the limit
            limits_engaged = False
            for step in range(50):
                rc, dc = cpu.tick_block(tgt[None], 10)
                self.assertEqual(rc, 0, (v, "limit", step, dc))
                rc, dg = gpu.tick_block(tgt[None], 10)
                self.assertEqual(rc, 0, (v, "limit", step, dg))
                limits_engaged |= dg[0]["active_limits"] > 0
            xc, xg = cpu.read(), gpu.read()
            self.assertTrue(limits_engaged, "kernel limit row never engaged")
            excess_c = float(xc.q[0, 9] - lim[2, 1])
            excess_g = float(xg.q[0, 9] - lim[2, 1])
            self.assertLess(abs(excess_c), 0.01, "oracle soft-limit excess")
            self.assertLess(abs(excess_g), 0.01, "kernel soft-limit excess")
            dq_lim = float(np.abs(xg.q[0, 7:] - xc.q[0, 7:]).max())
            self.assertLess(dq_lim, 5e-3, "limit-hit joint parity")
            print(f"{v} scripted dq={worst_q:.2e} rad dtip={worst_tip*1e3:.3f} mm "
                  f"max|qd|={max_speed:.2f}; limit excess cpu={excess_c:.2e} "
                  f"gpu={excess_g:.2e} dq={dq_lim:.2e}", file=sys.stderr)
        finally:
            cpu.close()
            gpu.close()

    def test_scripted_parity_kr240(self):
        self._scripted("kr240")

    def test_scripted_parity_lite(self):
        self._scripted("lite")

    # -- gate (e): determinism -------------------------------------------------
    def test_two_runs_bit_identical(self):
        a, b = ac.CudaArmLane(2, "kr240"), ac.CudaArmLane(2, "kr240")
        try:
            tgt = np.tile(al.KR240.home_q, (2, 1)) + np.array([[0.3], [-0.3]])
            for _ in range(20):
                self.assertEqual(a.tick_block(tgt, 10)[0], 0)
                self.assertEqual(b.tick_block(tgt, 10)[0], 0)
            sa, sb = a.read(), b.read()
            for name in ["q", "v", "body_state", "count"]:
                self.assertEqual(getattr(sa, name).tobytes(),
                                 getattr(sb, name).tobytes(), name)
        finally:
            a.close()
            b.close()

    # -- gate (f): info / state injection ---------------------------------------
    def test_info_reports_arm_dims_and_tables(self):
        for v in VARIANTS:
            lane = ac.CudaArmLane(3, v)
            try:
                s = al.spec(v)
                self.assertEqual((lane.B, lane.J, lane.P), (8, 6, 2))
                kp, kv = al.gains(s)
                np.testing.assert_array_equal(lane.kp, kp)
                np.testing.assert_array_equal(lane.kv, kv)
                self.assertEqual(lane.kp_scalar, float(np.float32(kp.min())))
                np.testing.assert_array_equal(lane.effort_cap, al.effort(s))
                self.assertEqual(lane.effort_cap_scalar, float(al.effort(s).min()))
                np.testing.assert_allclose(lane.joint_limits, al.joint_limits(s),
                                           atol=1e-6)
                np.testing.assert_allclose(lane.home_joint_q, s.home_q, atol=1e-7)
                self.assertEqual(np.float32(_header_float(v, "DW_GRAVITY_Z")),
                                 np.float32(-9.81))
            finally:
                lane.close()

    def test_set_state_round_trip(self):
        lane = ac.CudaArmLane(1, "lite")
        try:
            q = np.array(al.reset_qpos(al.LITE), np.float32)
            q[7:] += 0.1
            v = np.zeros(al.N, np.float32)
            v[8] = 0.5
            lane.set_state(0, q, v, np.zeros(3 * al.J), count=17)
            x = lane.read()
            np.testing.assert_allclose(x.q[0], q, atol=1e-6)
            np.testing.assert_allclose(x.v[0], v, atol=0)
            self.assertEqual(int(x.count[0]), 17)
            rc, diag = lane.tick(np.asarray(al.LITE.home_q)[None])
            self.assertEqual(rc, 0, diag)
        finally:
            lane.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
