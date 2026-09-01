"""Parity gates: fp32 serial humanoid kernel build vs the f64 CPU oracle.

Run: .venv/bin/python -B humanoid/tests/test_humanoid_serial_parity.py

The humanoid twin of experimental/duck_cuda/tests/test_serial_parity.py's
gate (a), for the PURE-REGENERATION build: the unchanged kernel sources
compiled against the generated humanoid/include/duck_model.h (shadowed by
include order; zero kernel edits, the duck build untouched).

Gates:
 (a) header drift: the committed humanoid header matches a fresh
     regeneration from humanoid/h0_lowering.py, AND the duck's committed
     header + generator are untouched by the humanoid work (the duck
     generator's own drift test is authoritative; here we double-lock that
     generate_model.py regenerates the duck header byte-identically);
 (b) home-hold, 1000 ticks (2 s at 0.002): |root - CPU| < 2 mm, tilt
     difference < 1 deg, both feet in contact on both lanes, fp32 momentum
     residual within the kernel's own DW_MOMENTUM_TOLERANCE (2e-4);
 (c) effort caps never bind on the oracle trajectory (the kernel now
     clamps with the per-joint DW_EFFORT_CAP_TABLE, tiers 180/140/70;
     torque stays far below even the minimum tier here, so this gate is
     exact-model parity);
 (d) bit-identical determinism between two humanoid scenes in one build;
 (e) contact-free dynamics parity: seeded random joint poses dropped from
     +0.5 m, PD toward home, 40 ticks fully in-air on both lanes -- bounded
     divergence (root < 1 mm, joints < 5e-3 rad), no contact, effort caps
     untouched (exercises FK/mass/bias/PD/joint rows without the flat-foot
     contact degeneracy);
 (f) WORKSTREAM-A REPAIR gate (was: known-limitation containment): the
     degenerate flat-foot standing stall is FIXED. Perturbed standing
     targets used to stall the coupled solve on BOTH lanes at tick ~6
     (exactly-coplanar 4-corner box soles -> exactly singular contact
     block whose sweeps/Tresca limit-cycle; humanoid/FEASIBILITY.md
     section 6). civ1 now carries the full repair stack (per-call APGD
     budgets, cap-refresh damping schedule, best-iterate memory,
     last-window certificate polish, load-aware exhaustion ceiling with
     strict joint rows + the untouched absolute 1e-8 momentum gate), and
     the fp32 kernel mirrors the tier/budget/damping economics. The gate
     asserts both lanes tick THROUGH the standing transient (>=120 ticks,
     the old stall window was <20) under a held waist lean; if the f64
     lane's default 4096-iteration budget cannot finish the eventual
     ground-impact solve after the lean topples the robot, that late
     fault must still be a contained clean NO_CONVERGENCE (frozen finite
     state) -- never a stall inside the standing window.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "humanoid"))

from walk.env import humanoid_cuda_lane as hc  # noqa: E402
from walk.env import humanoid_native_lane as hn  # noqa: E402
import h0_lowering as h0  # noqa: E402

TOOLS = ROOT / "experimental" / "duck_cuda" / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tilt_deg(q7: np.ndarray) -> float:
    return math.degrees(float(hn.tilt(np.asarray(q7)[3:7][None])[0]))


class HumanoidSerialParityTests(unittest.TestCase):
    # -- gate (a): generated headers --------------------------------------
    def test_humanoid_header_drift(self):
        gen = _load("generate_model_humanoid")
        fresh = gen.emit(gen.load_lowering())
        committed = (ROOT / "humanoid" / "include" / "duck_model.h").read_text()
        self.assertEqual(fresh, committed,
                         "humanoid duck_model.h drifted; regenerate with "
                         "tools/generate_model_humanoid.py")

    def test_duck_header_untouched_by_humanoid_work(self):
        gen = _load("generate_model")
        native, fixture, cm = gen.load_fixture()
        fresh = gen.emit(native, fixture, cm)
        committed = (ROOT / "experimental" / "duck_cuda" / "include"
                     / "duck_model.h").read_text()
        self.assertEqual(fresh, committed,
                         "duck duck_model.h must stay byte-identical")

    # -- gate (b)+(c): home-hold parity ------------------------------------
    def test_home_hold_1000_ticks_parity(self):
        cpu = hn.NativeHumanoidLane(1)
        gpu = hc.CudaHumanoidLane(1)
        try:
            home = cpu.home_joint_q[None, :]
            worst_torque = 0.0
            for t in range(1000):
                pre = cpu.read()
                torque = np.clip(
                    cpu.kp * (home - pre.q[:, 7:]) - cpu.kv * pre.v[:, 6:],
                    -cpu.effort_cap, cpu.effort_cap)
                worst_torque = max(worst_torque, float(np.abs(torque).max()))
                rc, diag = cpu.tick(home)
                self.assertEqual(rc, 0, (t, diag))
                rc, diag = gpu.tick(home)
                self.assertEqual(rc, 0, (t, diag))
                self.assertLessEqual(diag[0]["momentum_residual"], 2e-4, t)
            qc = cpu.read()
            qg = gpu.read()
            root_mm = 1000.0 * float(np.abs(qg.q[0, :3] - qc.q[0, :3]).max())
            tilt = abs(_tilt_deg(qg.q[0]) - _tilt_deg(qc.q[0]))
            self.assertLess(root_mm, 2.0, "root drift gate (2 mm)")
            self.assertLess(tilt, 1.0, "tilt gate (1 deg)")
            self.assertTrue(bool(qc.foot_contact.all()), "oracle feet")
            self.assertTrue(bool(qg.foot_contact.all()), "serial feet")
            # (c) scalar-cap validity: min authored tier never approached
            self.assertLess(worst_torque, 0.01 * gpu.effort_cap)
            print(f"home_hold_parity root_drift={root_mm:.2e} mm "
                  f"tilt_diff={tilt:.2e} deg torque<={worst_torque:.2e}",
                  file=sys.stderr)
        finally:
            cpu.close()
            gpu.close()

    # -- gate (d): determinism ----------------------------------------------
    def test_two_runs_bit_identical(self):
        a, b = hc.CudaHumanoidLane(2), hc.CudaHumanoidLane(2)
        try:
            home = np.zeros((2, h0.J))
            for _ in range(80):
                self.assertEqual(a.tick(home)[0], 0)
                self.assertEqual(b.tick(home)[0], 0)
            sa, sb = a.read(), b.read()
            for name in ["q", "v", "body_state", "sole_height",
                         "foot_contact", "count"]:
                self.assertEqual(getattr(sa, name).tobytes(),
                                 getattr(sb, name).tobytes(), name)
        finally:
            a.close()
            b.close()

    # -- gate (e): contact-free dynamics parity --------------------------------
    def test_in_air_dynamics_parity(self):
        rng = np.random.default_rng(2026)
        offsets = rng.uniform(-0.3, 0.3, (1, h0.J))
        native = hn.native_lane._native()
        lib = native.library(str(hn.build_library()))
        cpu, _ = h0.scene(lib, 1, joint_offsets=offsets, root_lift=0.5)
        gpu = hc.CudaHumanoidLane(1)
        try:
            q0 = cpu.read().q[0]
            gpu.set_state(0, np.float32(q0), np.zeros(6 + h0.J, np.float32),
                          np.zeros(3 * h0.J, np.float32))
            home = np.zeros((1, h0.J))
            worst_torque = 0.0
            for t in range(40):        # 0.08 s: drop 0.064 m of the 0.5 m
                pre = np.array(cpu.read().q[0])
                prev = np.array(cpu.read().v[0])
                torque = np.abs(h0.KP * (0.0 - pre[7:]) - h0.KV * prev[6:])
                worst_torque = max(worst_torque, float(torque.max()))
                rc, diag = cpu.step(dt=h0.SIM_DT, target=home,
                                    max_iterations=4096, tolerance=1e-8)
                self.assertEqual(rc, 0, (t, diag))
                self.assertEqual(diag[0]["contact_points"], 0, t)
                rc, diag = gpu.tick(home)
                self.assertEqual(rc, 0, (t, diag))
                self.assertEqual(diag[0]["contact_points"], 0, t)
            qc = np.array(cpu.read().q[0])
            xg = gpu.read()
            root_mm = 1000.0 * float(np.abs(xg.q[0, :3] - qc[:3]).max())
            joint = float(np.abs(xg.q[0, 7:] - qc[7:]).max())
            self.assertLess(root_mm, 1.0, "in-air root gate (1 mm)")
            self.assertLess(joint, 5e-3, "in-air joint gate (5e-3 rad)")
            self.assertFalse(bool(xg.foot_contact.any()))
            # effort caps immaterial here too (below even the 70 tier):
            self.assertLess(worst_torque, min(h0.EFFORT))
            print(f"in_air_parity root={root_mm:.2e} mm joint={joint:.2e} "
                  f"rad torque<={worst_torque:.1f}", file=sys.stderr)
        finally:
            cpu.close()
            gpu.close()

    # -- gate (f): flat-foot standing stall is REPAIRED ------------------------
    def test_perturbed_standing_ticks_through_both_lanes(self):
        """Pins the workstream-A repair (was: stall containment).

        Held waist lean on both lanes. Before the repair both stalled at
        tick ~6 inside the standing transient (degenerate coplanar box-sole
        contact block). Now both lanes must clear the standing window
        (>=120 ticks; measured post-repair: fp32 240/240 clean, f64 clean
        through at least the topple). A LATE f64 fault is tolerated only as
        a contained clean NO_CONVERGENCE on the post-topple ground impact
        (that solve is budget-bound at the lane's default 4096 iterations;
        it converges at 16384), never as a standing-window stall.
        """
        target = np.zeros((1, h0.J))
        target[0, 0] = 0.05                    # tiny waist lean, held
        cpu = hn.NativeHumanoidLane(1)
        gpu = hc.CudaHumanoidLane(1)
        try:
            cpu_fault = gpu_fault = None
            for t in range(240):
                rc, diag = cpu.tick(target)
                if rc:
                    cpu_fault = (t, diag[0]["native_status"])
                    break
            for t in range(240):
                rc, diag = gpu.tick(target)
                if rc:
                    gpu_fault = (t, diag[0]["native_status"])
                    break
            # fp32 build ticks through the whole 240-tick lean (measured).
            self.assertIsNone(gpu_fault,
                              "fp32 lane regressed into a standing stall")
            xc, xg = cpu.read(), gpu.read()
            self.assertTrue(xg.finite().all())
            self.assertEqual(int(xg.count[0]), 240)
            if cpu_fault is None:
                self.assertTrue(xc.finite().all())
                self.assertEqual(int(xc.count[0]), 240)
                print("perturbed standing: CPU 240/240, fp32 240/240 clean "
                      "(workstream-A repair)", file=sys.stderr)
            else:
                # Only the late budget-bound ground impact may fault, and
                # only as a detected + contained clean NO_CONVERGENCE.
                self.assertGreaterEqual(cpu_fault[0], 120,
                                        "standing-window stall regressed")
                self.assertEqual(cpu_fault[1], 3,
                                 "clean NO_CONVERGENCE (CPU)")
                self.assertTrue(xc.finite().all())
                self.assertEqual(int(xc.count[0]), cpu_fault[0])
                print(f"perturbed standing: CPU contained post-topple fault "
                      f"tick {cpu_fault[0]} (4096-iteration budget), "
                      f"fp32 240/240 clean", file=sys.stderr)
        finally:
            cpu.close()
            gpu.close()

    # -- info / state injection ----------------------------------------------
    def test_info_reports_humanoid_dims(self):
        lane = hc.CudaHumanoidLane(3)
        try:
            self.assertEqual((lane.B, lane.J, lane.P), (14, 12, 2))
            self.assertAlmostEqual(lane.home_root_height, 1.15, places=6)
            self.assertEqual(lane.kp, 90.0)
            self.assertEqual(lane.kv, 8.0)
            self.assertEqual(lane.effort_cap, 70.0)   # scalar MIN tier
            np.testing.assert_allclose(
                lane.joint_limits,
                np.array([(j[4], j[5]) for j in h0.JOINTS]), atol=1e-6)
            np.testing.assert_array_equal(lane.home_joint_q, np.zeros(12))
        finally:
            lane.close()

    def test_set_state_round_trip(self):
        lane = hc.CudaHumanoidLane(1)
        try:
            q = np.array(h0.reset_qpos(), np.float32)
            q[2] += 0.05
            v = np.zeros(6 + h0.J, np.float32)
            v[0] = 0.25
            lane.set_state(0, q, v, np.zeros(3 * h0.J), count=17)
            x = lane.read()
            np.testing.assert_allclose(x.q[0], q, atol=1e-6)
            np.testing.assert_allclose(x.v[0], v, atol=0)
            self.assertEqual(int(x.count[0]), 17)
            rc, diag = lane.tick(np.zeros((1, h0.J)))
            self.assertEqual(rc, 0, diag)
        finally:
            lane.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
