"""fp32 serial kernel build vs f64 CPU oracle, per humanoid FAMILY variant.

Run: .venv/bin/python -B humanoid/tests/test_variant_serial_parity.py

The variant twin of test_humanoid_serial_parity.py (the H1 gates,
unchanged), PARAMETRIZED over humanoid/h1_family.py's non-base members.
Each variant's serial dylib is the UNCHANGED kernel compiled against
humanoid/variants/<name>/include/duck_model.h (shadow header by include
order; zero kernel edits):
 (a) header drift: committed variant header == fresh regeneration (with
     the variant's reference table);
 (b) home-hold 1000 ticks: |root - CPU| < 2 mm, tilt < 1 deg, both feet in
     contact on both lanes, fp32 momentum residual within 2e-4;
 (c) effort caps never bind on the oracle trajectory (min tier margin);
 (d) bit-identical determinism between two variant scenes in one build;
 (e) contact-free dynamics parity (in-air drop, PD to home, 40 ticks);
 (f) perturbed standing ticks through the standing window on both lanes;
 (g) info reports the variant's tables (kp/kv/effort/limits/home height)
     and a base-built library is refused by the variant lane;
 (h) FlatFloorHumanoidEnv(variant) over the fp32 lane == dwc1_step_policy
     obs/reward/done side by side (robot-generic kernel contract).
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
import h1_family as fam  # noqa: E402

TOOLS = ROOT / "experimental" / "duck_cuda" / "tools"
VARIANTS = tuple(v for v in fam.variant_names() if v != "h1")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tilt_deg(q7: np.ndarray) -> float:
    return math.degrees(float(hn.tilt(np.asarray(q7)[3:7][None])[0]))


class VariantSerialParity(unittest.TestCase):
    def _header_drift(self, v):
        gen = _load("generate_model_humanoid")
        self.assertEqual(gen.emit_variant(v), fam.header_path(v).read_text(),
                         f"{v} header drifted; regenerate with "
                         f"generate_model_humanoid.py --lowering {v}")

    def _home_hold_parity(self, v):
        lw = fam.load_lowering(v)
        cpu = hn.NativeHumanoidLane(1, variant=v)
        gpu = hc.CudaHumanoidLane(1, variant=v)
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
                self.assertEqual(rc, 0, (v, t, diag))
                rc, diag = gpu.tick(home)
                self.assertEqual(rc, 0, (v, t, diag))
                self.assertLessEqual(diag[0]["momentum_residual"], 2e-4, (v, t))
            qc, qg = cpu.read(), gpu.read()
            root_mm = 1000.0 * float(np.abs(qg.q[0, :3] - qc.q[0, :3]).max())
            tilt = abs(_tilt_deg(qg.q[0]) - _tilt_deg(qc.q[0]))
            self.assertLess(root_mm, 2.0, (v, "root drift gate (2 mm)"))
            self.assertLess(tilt, 1.0, (v, "tilt gate (1 deg)"))
            self.assertTrue(bool(qc.foot_contact.all()), (v, "oracle feet"))
            self.assertTrue(bool(qg.foot_contact.all()), (v, "serial feet"))
            self.assertLess(worst_torque, 0.01 * min(lw.EFFORT))
            print(f"[{v}] home_hold_parity root_drift={root_mm:.2e} mm "
                  f"tilt_diff={tilt:.2e} deg torque<={worst_torque:.2e}",
                  file=sys.stderr)
        finally:
            cpu.close()
            gpu.close()

    def _determinism(self, v):
        a, b = hc.CudaHumanoidLane(2, variant=v), hc.CudaHumanoidLane(2, variant=v)
        try:
            home = np.zeros((2, a.J))
            for _ in range(80):
                self.assertEqual(a.tick(home)[0], 0)
                self.assertEqual(b.tick(home)[0], 0)
            sa, sb = a.read(), b.read()
            for name in ["q", "v", "body_state", "sole_height", "foot_contact",
                         "count"]:
                self.assertEqual(getattr(sa, name).tobytes(),
                                 getattr(sb, name).tobytes(), (v, name))
        finally:
            a.close()
            b.close()

    def _in_air_parity(self, v):
        lw = fam.load_lowering(v)
        rng = np.random.default_rng(2026)
        offsets = rng.uniform(-0.3, 0.3, (1, lw.J))
        native = hn.native_lane._native()
        lib = native.library(str(hn.build_library()))
        cpu, _ = lw.scene(lib, 1, joint_offsets=offsets, root_lift=0.5)
        gpu = hc.CudaHumanoidLane(1, variant=v)
        try:
            q0 = cpu.read().q[0]
            gpu.set_state(0, np.float32(q0), np.zeros(6 + lw.J, np.float32),
                          np.zeros(3 * lw.J, np.float32))
            home = np.zeros((1, lw.J))
            for t in range(40):
                rc, diag = cpu.step(dt=lw.SIM_DT, target=home,
                                    max_iterations=4096, tolerance=1e-8)
                self.assertEqual(rc, 0, (v, t, diag))
                self.assertEqual(diag[0]["contact_points"], 0, (v, t))
                rc, diag = gpu.tick(home)
                self.assertEqual(rc, 0, (v, t, diag))
                self.assertEqual(diag[0]["contact_points"], 0, (v, t))
            qc = np.array(cpu.read().q[0])
            xg = gpu.read()
            root_mm = 1000.0 * float(np.abs(xg.q[0, :3] - qc[:3]).max())
            joint = float(np.abs(xg.q[0, 7:] - qc[7:]).max())
            self.assertLess(root_mm, 1.0, (v, "in-air root gate (1 mm)"))
            self.assertLess(joint, 5e-3, (v, "in-air joint gate (5e-3 rad)"))
            self.assertFalse(bool(xg.foot_contact.any()))
            print(f"[{v}] in_air_parity root={root_mm:.2e} mm joint={joint:.2e}",
                  file=sys.stderr)
        finally:
            cpu.close()
            gpu.close()

    def _perturbed_standing(self, v):
        lw = fam.load_lowering(v)
        target = np.zeros((1, lw.J))
        target[0, 0] = 0.05
        cpu = hn.NativeHumanoidLane(1, variant=v)
        gpu = hc.CudaHumanoidLane(1, variant=v)
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
            self.assertIsNone(gpu_fault, (v, "fp32 standing stall"))
            xc, xg = cpu.read(), gpu.read()
            self.assertTrue(xg.finite().all())
            self.assertEqual(int(xg.count[0]), 240)
            if cpu_fault is not None:
                self.assertGreaterEqual(cpu_fault[0], 120, (v, "standing stall"))
                self.assertEqual(cpu_fault[1], 3)
            self.assertTrue(xc.finite().all())
        finally:
            cpu.close()
            gpu.close()

    def _info(self, v):
        lw = fam.load_lowering(v)
        lane = hc.CudaHumanoidLane(3, variant=v)
        try:
            self.assertEqual(lane.variant, v)
            self.assertEqual((lane.B, lane.J, lane.P), (16, 14, 2))
            self.assertAlmostEqual(lane.home_root_height,
                                   float(lw.reset_qpos()[2]), places=5)
            np.testing.assert_array_equal(lane.kp, np.array(lw.KP_TABLE))
            np.testing.assert_array_equal(lane.kv, np.array(lw.KV_TABLE))
            np.testing.assert_array_equal(lane.effort_cap, np.array(lw.EFFORT))
            self.assertAlmostEqual(lane.effort_cap_scalar, min(lw.EFFORT),
                                   places=4)
            np.testing.assert_allclose(
                lane.joint_limits,
                np.array([(j[4], j[5]) for j in lw.JOINTS]), atol=1e-6)
            # header dir / library are the variant's, distinct from the base
            self.assertNotEqual(lane.library_path, hc.build_library())
            self.assertEqual(hc.header_dir(v), fam.header_dir(v))
        finally:
            lane.close()
        with self.assertRaises(RuntimeError):
            hc.CudaHumanoidLane(1, variant=v, library_path=hc.build_library())

    def _env_vs_kernel_policy(self, v):
        from walk.env import humanoid_flat as hf
        E, S, steps = 2, 31, 40
        env = hf.FlatFloorHumanoidEnv(
            environments=E, seed=S, variant=v,
            lane_factory=lambda n, off: hc.CudaHumanoidLane(
                n, joint_offsets=off, variant=v))
        lane = hc.CudaHumanoidLane(E, variant=v)
        try:
            self.assertEqual(env.variant, v)
            obs_env = env.set_command(env.command)
            obs_lane = lane.reset_policy(seed=S)
            np.testing.assert_allclose(obs_lane, obs_env, atol=1e-6)
            rng = np.random.default_rng(99)
            worst_obs = worst_rew = 0.0
            for t in range(steps):
                a = np.clip(rng.normal(0.0, 0.1, (E, hf.ACT)),
                            -0.2, 0.2).astype(np.float32)
                oe, re_, de, _ = env.step(a)
                ol, rl, dl, diag = lane.step_policy(a)
                self.assertTrue((diag["status"] == 0).all(), (v, t))
                np.testing.assert_array_equal(dl, de, err_msg=f"{v} done t={t}")
                worst_obs = max(worst_obs, float(np.abs(ol - oe).max()))
                worst_rew = max(worst_rew, float(np.abs(rl - re_).max()))
                if de.all():
                    break
            self.assertLess(worst_obs, 1e-4, (v, "obs parity"))
            self.assertLess(worst_rew, 1e-3, (v, "reward parity"))
            print(f"[{v}] env-vs-kernel policy parity: obs<={worst_obs:.2e} "
                  f"reward<={worst_rew:.2e}", file=sys.stderr)
        finally:
            env.close()
            lane.close()


for _v in VARIANTS:
    def _mk(name, v=_v):
        return lambda self: getattr(self, name)(v)
    for _n in ("_header_drift", "_home_hold_parity", "_determinism",
               "_in_air_parity", "_perturbed_standing", "_info",
               "_env_vs_kernel_policy"):
        setattr(VariantSerialParity, f"test{_n}_{_v}", _mk(_n))


if __name__ == "__main__":
    unittest.main(verbosity=2)
