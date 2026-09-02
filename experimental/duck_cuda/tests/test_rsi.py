"""Reference State Initialization (dwc1_set_rsi) + per-joint gain tables.

Run: .venv/bin/python -B experimental/duck_cuda/tests/test_rsi.py

RSI (DeepMimic-style): with probability = fraction, a policy reset starts
the env ON the DW_REF_GAIT reference cycle at the bin aligned with its
freshly drawn phase offset -- joint q from the table row (limit-clamped),
joint qdot from the table's finite difference at the gait clock rate, root
at the reset pose -- so the imitation reward term is consistent at t = 0.
Default OFF (0.0) keeps every reset bit-identical (covered by the duck
fingerprint protocol as well). The per-joint DW_KP_TABLE / DW_KV_TABLE are
drift-pinned by the generator header tests; the duck's uniform broadcast
is fingerprint-proven bit-identical.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "humanoid"))

from walk.env import flat  # noqa: E402
from walk.env import reward as reward_mod  # noqa: E402
from walk.env.cuda_lane import CudaDuckLane  # noqa: E402

REF = np.asarray(reward_mod.REF_GAIT, np.float64)
BINS = int(reward_mod.REF_BINS)


def _bin_of(phase0: float) -> int:
    frac = math.fmod(phase0 / (2.0 * math.pi), 1.0)
    if frac < 0.0:
        frac += 1.0
    return int(frac * BINS) % BINS


class RsiTests(unittest.TestCase):
    def test_rsi_resets_land_on_the_reference_row(self):
        E = 8
        lane = CudaDuckLane(E)
        try:
            lane.set_rsi(1.0)
            cmd = 0.15
            phases = [2.0 * math.pi * (k / E + 0.013) for k in range(E)]
            lane.reset_policy(commands=[cmd] * E, phase_offsets=phases)
            st = lane.read()
            hz = flat.PHASE_HZ_BASE + flat.PHASE_HZ_PER_MPS * cmd
            lo = lane.joint_limits[:, 0]
            hi = lane.joint_limits[:, 1]
            for e in range(E):
                b = _bin_of(phases[e])
                q_ref = np.clip(REF[b], lo, hi)
                qd_ref = (REF[(b + 1) % BINS] - REF[b]) * BINS * hz
                np.testing.assert_allclose(
                    np.asarray(st.q[e][7:], np.float64), q_ref,
                    atol=1e-6, err_msg=f"env {e} bin {b} joint q")
                np.testing.assert_allclose(
                    np.asarray(st.v[e][6:], np.float64), qd_ref,
                    atol=1e-5, err_msg=f"env {e} bin {b} joint qdot")
                # root untouched: reset height, zero root rates
                self.assertAlmostEqual(float(st.q[e][2]),
                                       float(lane.home_root_height), places=5)
                self.assertEqual(float(np.abs(st.v[e][:6]).max()), 0.0)
            # imitation-term consistency at t=0: the kernel's phase bin at
            # t=0 is _bin_of(phase0) by the same numpy-mirroring formula,
            # which is exactly the row we just verified the pose against.
            print(f"rsi: {E}/{E} resets on-reference "
                  f"(bins {[ _bin_of(p) for p in phases ]})", file=sys.stderr)
        finally:
            lane.close()

    def test_rsi_off_is_todays_reset_and_fraction_is_respected(self):
        E = 64
        a = CudaDuckLane(E)
        b = CudaDuckLane(E)
        try:
            phases = [2.0 * math.pi * ((7 * k) % E) / E for k in range(E)]
            cmds = [0.15] * E
            # OFF (default, never called) vs explicit 0.0: bit-identical
            a.reset_policy(commands=cmds, phase_offsets=phases)
            b.set_rsi(0.0)
            b.reset_policy(commands=cmds, phase_offsets=phases)
            np.testing.assert_array_equal(np.asarray(a.read().q),
                                          np.asarray(b.read().q))
            home = np.asarray(a.read().q)[:, 7:]
            # fraction 0.5: some (not all, not none) envs start on-table,
            # and the scene-level draw stream is deterministic per scene.
            b.set_rsi(0.5)
            b.reset_policy(commands=cmds, phase_offsets=phases)
            q1 = np.asarray(b.read().q)[:, 7:]
            moved1 = int((np.abs(q1 - home).max(axis=1) > 1e-9).sum())
            self.assertGreater(moved1, 8, "fraction 0.5 barely fired")
            self.assertLess(moved1, 56, "fraction 0.5 fired everywhere")
            c = CudaDuckLane(E)
            try:
                c.set_rsi(0.5)
                c.reset_policy(commands=cmds, phase_offsets=phases)
                np.testing.assert_array_equal(
                    q1, np.asarray(c.read().q)[:, 7:],
                    "scene RSI draw stream must be deterministic")
            finally:
                c.close()
            print(f"rsi fraction 0.5: {moved1}/{E} envs on-reference "
                  f"(deterministic)", file=sys.stderr)
        finally:
            a.close()
            b.close()

    def test_rsi_env_still_steps_cleanly(self):
        lane = CudaDuckLane(4)
        try:
            lane.set_rsi(1.0)
            lane.reset_policy(seed=0)
            for _t in range(20):
                _o, _r, done, diag = lane.step_policy(
                    np.zeros((4, 14), np.float32))
                self.assertEqual(int((diag["status"] != 0).sum()), 0)
            self.assertFalse(bool(done.all()),
                             "all RSI-initialized envs died standing still")
        finally:
            lane.close()


class RsiHumanoidSmokeTests(unittest.TestCase):
    def test_h1_rsi_reset_lands_off_home_and_ticks(self):
        from walk.env import humanoid_cuda_lane as hc
        lane = hc.CudaHumanoidLane(2)
        try:
            lane.set_rsi(1.0)
            lane.reset_policy(seed=0, commands=[0.75, 0.75],
                              phase_offsets=[0.7, 3.9])
            q = np.asarray(lane.read().q)[:, 7:]
            # H1 home pose is all-zero joints; a populated reference table
            # must move at least some joints at reset.
            self.assertGreater(float(np.abs(q).max()), 1e-3,
                               "RSI no-op on a populated reference table")
            J = q.shape[1]
            for _t in range(5):
                _o, _r, _d, diag = lane.step_policy(
                    np.zeros((2, J), np.float32))
                self.assertEqual(int((diag["status"] != 0).sum()), 0)
        finally:
            lane.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
