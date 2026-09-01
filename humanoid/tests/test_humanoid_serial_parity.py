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
 (b) home-hold, 480 ticks (2 s at 1/240): |root - CPU| < 2 mm, tilt
     difference < 1 deg, both feet in contact on both lanes, fp32 momentum
     residual within the kernel's own DW_MOMENTUM_TOLERANCE (2e-4);
 (c) the scalar effort cap (70 = min authored tier) never binds on the
     oracle trajectory, so the kernel's scalar-vs-per-joint-cap limitation
     is inactive for this gate (exact-model parity);
 (d) bit-identical determinism between two humanoid scenes in one build;
 (e) contact-free dynamics parity: seeded random joint poses dropped from
     +0.5 m, PD toward home, 40 ticks fully in-air on both lanes -- bounded
     divergence (root < 1 mm, joints < 5e-3 rad), no contact, effort caps
     untouched (exercises FK/mass/bias/PD/joint rows without the flat-foot
     contact degeneracy);
 (f) KNOWN-LIMITATION containment (NOT a pass gate on the physics): any
     perturbed STANDING targets stall the coupled solve on BOTH lanes --
     the pre-existing documented degenerate flat-foot contact failure
     (walk/env/flat.py:55-61 "workstream A"; duck_cuda_kernel.h header
     notes the un-ported rank-1 null-direction repair), which the
     humanoid's exactly-coplanar 4-corner box feet amplify. The gate
     asserts the fault is DETECTED and CONTAINED identically on both
     lanes: clean NO_CONVERGENCE status, frozen finite state, no NaN.
     Phase 2 walking needs the workstream-A solver repair first (see
     humanoid/FEASIBILITY.md section 6).
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
    def test_home_hold_480_ticks_parity(self):
        cpu = hn.NativeHumanoidLane(1)
        gpu = hc.CudaHumanoidLane(1)
        try:
            home = cpu.home_joint_q[None, :]
            worst_torque = 0.0
            for t in range(480):
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
            for t in range(40):                # 0.167 s: still > 0.17 m up
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
            # per-joint-vs-scalar cap immaterial here too:
            self.assertLess(worst_torque, min(h0.EFFORT))
            print(f"in_air_parity root={root_mm:.2e} mm joint={joint:.2e} "
                  f"rad torque<={worst_torque:.1f}", file=sys.stderr)
        finally:
            cpu.close()
            gpu.close()

    # -- gate (f): documented flat-foot stall is contained ---------------------
    def test_perturbed_standing_stall_contained_both_lanes(self):
        """Pins the KNOWN solver-robustness limitation (workstream A).

        Perturbed standing targets make the coupled solve stall on BOTH the
        f64 oracle and the fp32 build (the documented degenerate flat-foot
        contact failure the duck also has, walk/env/flat.py:55-61); this
        gate asserts detection + containment, and must be UPDATED (expected
        to start passing tick-through) once the workstream-A repair lands.
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
            self.assertIsNotNone(cpu_fault, "oracle no longer stalls: "
                                 "workstream-A repair landed? update gates")
            self.assertIsNotNone(gpu_fault, "serial no longer stalls: "
                                 "workstream-A repair landed? update gates")
            self.assertEqual(cpu_fault[1], 3, "clean NO_CONVERGENCE (CPU)")
            self.assertEqual(gpu_fault[1], 3, "clean NO_CONVERGENCE (fp32)")
            # containment: state frozen at the last accepted tick, finite
            xc, xg = cpu.read(), gpu.read()
            self.assertTrue(xc.finite().all())
            self.assertTrue(xg.finite().all())
            self.assertEqual(int(xc.count[0]), cpu_fault[0])
            self.assertEqual(int(xg.count[0]), gpu_fault[0])
            print(f"stall containment: CPU tick {cpu_fault[0]}, "
                  f"fp32 tick {gpu_fault[0]} (documented workstream-A gap)",
                  file=sys.stderr)
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
