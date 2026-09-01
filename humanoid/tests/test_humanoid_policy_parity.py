"""Device-policy parity: dwc1_step_policy on the HUMANOID header build vs
walk/env/humanoid_flat.py (the python contract), f64 chain.

Run: .venv/bin/python -B humanoid/tests/test_humanoid_policy_parity.py

The humanoid twin of experimental/duck_cuda/tests/test_serial_parity.py::
test_step_policy_matches_python_env: the kernel's device policy layer is
now generated-header generic (DW_ENV_* contract block: OBS 52 / ACT 12,
humanoid termination up-axis, phase clock, action->target chain), so
FlatFloorHumanoidEnv (numpy f64 obs/reward over this same serial fp32 lane
build) and dwc1_step_policy are run side by side over identical physics --
obs/reward/done must agree far inside the duck's own parity gates.

The dwc1 policy ABI is driven directly here (CudaHumanoidLane is the
physics-path wrapper; its Phase-1 docstring predates the generic kernel
edit): dwc1_reset_policy with humanoid_flat's exact counter-based
(seed, env, episode) command/phase0 draws, dwc1_step_policy, dwc1_observe.

NOTE the CudaHumanoidLane wrapper still publishes the SCALAR dwc1_info
effort cap (min tier 70); the kernel and the python env both clamp with the
per-joint tiers (DW_EFFORT_CAP_TABLE / h0_lowering.EFFORT), so the lane
instance handed to the env gets its `effort_cap` overridden to the authored
per-joint array below -- the walk/env owner should lift that field to the
table when the wrapper grows its policy-path methods.
"""
from __future__ import annotations

import ctypes as C
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "humanoid"))

from walk.env import humanoid_cuda_lane as hc  # noqa: E402
from walk.env import humanoid_flat as hf  # noqa: E402
from walk.env.cuda_lane import DIAG_DTYPE, Diagnostic, _fp  # noqa: E402
import h0_lowering as h0  # noqa: E402

E = 4
OBS, ACT = hf.OBS, hf.ACT
DP = C.POINTER(C.c_double)
U8P = C.POINTER(C.c_uint8)


def _tiered_lane(n, offsets):
    lane = hc.CudaHumanoidLane(n, joint_offsets=offsets)
    lane.effort_cap = np.array(h0.EFFORT, np.float64)  # per-joint tiers
    return lane


class _DevPolicy:
    """Raw dwc1 policy ABI over a CudaHumanoidLane build, with
    humanoid_flat's exact per-episode command/phase0 draw recipe."""

    def __init__(self, environments, seed=0):
        self.E = environments
        self.seed = seed
        self.lane = _tiered_lane(environments, None)
        self.episode = np.zeros(environments, np.int64)

    def reset(self, mask=None):
        m = (np.ones(self.E, bool) if mask is None
             else np.asarray(mask, bool).reshape(self.E))
        cmd = np.zeros(self.E, np.float64)
        ph0 = np.zeros(self.E, np.float64)
        for e in np.flatnonzero(m):
            rng = hf._episode_rng(self.seed, int(e), int(self.episode[e]) + 1)
            draw = rng.random()
            cmd[e] = (hf.COMMANDS_MPS[0] if draw < 0.5
                      else hf.COMMANDS_MPS[1] if draw < 0.75
                      else hf.COMMANDS_MPS[2])
            ph0[e] = 2.0 * math.pi * rng.random()
            self.episode[e] += 1
        mc = (C.c_uint8 * self.E)(*[1 if x else 0 for x in m])
        rc = self.lane._lib.dwc1_reset_policy(
            self.lane._h, mc, cmd.ctypes.data_as(DP), ph0.ctypes.data_as(DP))
        if rc:
            raise RuntimeError(f"dwc1_reset_policy status={rc}")
        return self.observe()

    def observe(self):
        obs = np.empty((self.E, OBS), np.float32)
        rc = self.lane._lib.dwc1_observe(self.lane._h, _fp(obs))
        if rc:
            raise RuntimeError(f"dwc1_observe status={rc}")
        return obs

    def step(self, actions, n_ticks=10):
        a = np.ascontiguousarray(actions, np.float32).reshape(self.E, ACT)
        obs = np.empty((self.E, OBS), np.float32)
        reward = np.empty(self.E, np.float32)
        done = np.empty(self.E, np.uint8)
        diag = (Diagnostic * self.E)()
        rc = self.lane._lib.dwc1_step_policy(
            self.lane._h, _fp(a), int(n_ticks), _fp(obs), _fp(reward),
            done.ctypes.data_as(U8P), diag)
        if rc:
            raise RuntimeError(f"dwc1_step_policy status={rc}")
        return (obs, reward, done.astype(bool),
                np.frombuffer(diag, dtype=DIAG_DTYPE).copy())

    def close(self):
        self.lane.close()


class HumanoidPolicyParityTests(unittest.TestCase):
    def setUp(self):
        # SolverFault artifacts raised on purpose by the fault-parity leg
        # below go to a scratch dir, not the training fault corpus.
        import tempfile
        self._fault_dir = hf.FAULT_DIR
        hf.FAULT_DIR = Path(tempfile.mkdtemp(prefix="humanoid_parity_faults"))

    def tearDown(self):
        hf.FAULT_DIR = self._fault_dir

    def test_step_policy_matches_python_env(self):
        env = hf.FlatFloorHumanoidEnv(
            environments=E, seed=0,
            lane_factory=lambda n, off: _tiered_lane(n, off))
        dev = _DevPolicy(E, seed=0)
        try:
            obs_py = env.reset()                 # episode-2 command draws
            dev.reset()                          # episode 1
            obs_dev = dev.reset()                # episode 2
            np.testing.assert_array_equal(obs_py, obs_dev,
                                          "reset observations")
            self.assertTrue(np.array_equal(env.command.astype(np.float32),
                                           obs_dev[:, 45]))
            rng = np.random.default_rng(0)
            worst_obs = worst_rew = 0.0
            resets = fault_pairs = 0
            for t in range(200):
                a = np.clip(rng.normal(0, 0.4, (E, ACT)),
                            -1, 1).astype(np.float32)
                # Flail actions can legitimately exhaust the fp32 solver on a
                # stomp impact (this lane's own iteration budget). The python
                # env RAISES SolverFault while dwc1_step_policy CONTAINS the
                # same fault (env marked done, state frozen, diagnostic set)
                # -- assert exactly that FAULT PARITY, then resync with a
                # full reset on both sides (same episode counters -> same
                # command/phase0 draws keep the streams aligned).
                try:
                    o_py, r_py, d_py, _info = env.step(a)
                except hf.SolverFault as fault:
                    o_dev, r_dev, d_dev, diag = dev.step(a)
                    e = int(fault.env_index)
                    self.assertNotEqual(int(diag["status"][e]), 0,
                                        f"device did not fault @ {t}")
                    self.assertTrue(bool(d_dev[e]),
                                    f"device fault not contained @ {t}")
                    fault_pairs += 1
                    o_py = env.reset()
                    o_dev = dev.reset()
                    self.assertLess(float(np.abs(o_py - o_dev).max()), 1e-4,
                                    f"post-fault reset obs @ {t}")
                    continue
                o_dev, r_dev, d_dev, diag = dev.step(a)
                self.assertEqual((diag["status"] != 0).sum(), 0, t)
                worst_obs = max(worst_obs, float(np.abs(o_py - o_dev).max()))
                worst_rew = max(worst_rew, float(np.abs(r_py - r_dev).max()))
                np.testing.assert_array_equal(d_py, d_dev, f"done @ {t}")
                if d_py.any():                   # trainer-style masked reset
                    resets += 1
                    o_py = env.reset(mask=d_py)
                    o_dev = dev.reset(mask=d_dev)
                    self.assertLess(float(np.abs(o_py - o_dev).max()), 1e-4,
                                    f"post-reset obs @ {t}")
            self.assertGreater(resets + fault_pairs, 0,
                               "action sequence never terminated")
            self.assertLess(worst_obs, 1e-4, "obs parity gate")
            self.assertLess(worst_rew, 1e-3, "reward parity gate")
            print(f"humanoid step_policy_parity obs={worst_obs:.3e} "
                  f"reward={worst_rew:.3e} masked_resets={resets} "
                  f"fault_pairs={fault_pairs}", file=sys.stderr)
        finally:
            env.close()
            dev.close()

    def test_zero_action_hold_100_policy_steps(self):
        """0.0 actions = HOME targets (the duck's 0.0/0.0 hold gate): the
        humanoid must stand for 100 policy steps (2 s) on the device path
        with no terminations and stay in parity with the python env."""
        env = hf.FlatFloorHumanoidEnv(
            environments=E, seed=0,
            lane_factory=lambda n, off: _tiered_lane(n, off))
        dev = _DevPolicy(E, seed=0)
        try:
            obs_py = env.reset()
            dev.reset()
            obs_dev = dev.reset()
            np.testing.assert_array_equal(obs_py, obs_dev)
            zeros = np.zeros((E, ACT), np.float32)
            worst_obs = worst_rew = 0.0
            for t in range(100):
                o_py, r_py, d_py, _ = env.step(zeros)
                o_dev, r_dev, d_dev, diag = dev.step(zeros)
                self.assertEqual((diag["status"] != 0).sum(), 0, t)
                self.assertFalse(d_py.any(), f"python env fell @ {t}")
                self.assertFalse(d_dev.any(), f"device env fell @ {t}")
                worst_obs = max(worst_obs, float(np.abs(o_py - o_dev).max()))
                worst_rew = max(worst_rew, float(np.abs(r_py - r_dev).max()))
            self.assertLess(worst_obs, 1e-4)
            self.assertLess(worst_rew, 1e-3)
            print(f"humanoid zero-action hold obs={worst_obs:.3e} "
                  f"reward={worst_rew:.3e}", file=sys.stderr)
        finally:
            env.close()
            dev.close()


class FastTerminationTests(unittest.TestCase):
    """dwc1_set_fast_termination (training throughput switch, default OFF).

    OFF-by-default correctness is proven by the bit-exact parity tests
    above (they never touch the switch). Here: with the switch ON, an env
    that is already done at block entry does NO physics work -- its state
    is bitwise frozen, reward 0, done latched -- while live envs keep
    stepping normally.
    """

    def test_done_env_skip_freezes_state(self):
        dev = _DevPolicy(2, seed=0)
        lib = dev.lane._lib
        lib.dwc1_set_fast_termination.argtypes = [C.c_void_p, C.c_uint32]
        lib.dwc1_set_fast_termination.restype = C.c_int
        try:
            dev.reset()
            rng = np.random.default_rng(7)
            done = np.zeros(2, bool)
            for t in range(200):
                a = np.clip(rng.normal(0, 0.6, (2, ACT)), -1, 1)
                _, _, done, diag = dev.step(a)
                if done.any():
                    break
            self.assertTrue(done.any(), "no env terminated under flail")
            self.assertEqual(lib.dwc1_set_fast_termination(dev.lane._h, 1), 0)
            q0 = np.array(dev.lane.read().q, np.float64)
            v0 = np.array(dev.lane.read().v, np.float64)
            obs, rew, done2, diag = dev.step(np.full((2, ACT), 0.3))
            q1 = np.array(dev.lane.read().q, np.float64)
            v1 = np.array(dev.lane.read().v, np.float64)
            for e in range(2):
                if done[e]:
                    # frozen bitwise, done latched, reward 0, clean status
                    np.testing.assert_array_equal(q0[e], q1[e])
                    np.testing.assert_array_equal(v0[e], v1[e])
                    self.assertTrue(bool(done2[e]))
                    self.assertEqual(float(rew[e]), 0.0)
                    self.assertEqual(int(diag["status"][e]), 0)
                else:
                    # live envs keep stepping normally
                    self.assertFalse(np.array_equal(q0[e], q1[e]))
        finally:
            dev.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
