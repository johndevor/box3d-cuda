"""Device policy path for the arm (kernel ABI v8, DW_ENV_KIND_REACH).

Run: .venv/bin/python -B arm/tests/test_arm_device_policy.py

The arm twin of humanoid/tests/test_humanoid_policy_parity.py plus the
arm's own judge cross-check:

 (a) PARITY: dwc1_step_policy (CudaArmLane.step_policy / reset_policy) vs
     ArmReachEnv over the SAME serial fp32 build, 420 steps x E=4 (the
     400-step horizon inside, so trainer-style masked resets occur), both
     variants, trainer-style masked resets: obs / reward agree within the
     gates (obs 1e-5, reward 1e-5), done flags and target sequences
     identical. The chain is f64 operation-for-operation like numpy; the
     one place the two can legitimately differ is numpy's 3-term rotation
     products in the tip / link-origin geometry (BLAS association), i.e.
     ULP-level f64 noise -- the measured worst case is printed.
 (b) JUDGE CROSS-CHECK: the frozen arm judge (walk/eval/arm_reach_judge.py)
     evaluates per-tick traces captured from ArmReachEnv over the same
     build under the same actions (the two paths' physics are identical by
     (a)); the device gate-proxy counters must equal the judge's counts
     EXACTLY: acquisitions (count and 1-based step <-> time_s),
     proxy-violating ticks (clause 5's violating_ticks), and the
     limit/speed violating-tick counts recomputed with the judge's own
     clause 3/4 predicates (== 0 iff the clause passes).
 (c) gate_proxy MAPPING + episode snapshot, gate-termination knobs on the
     reach counters, ABI guards (reach entries invalid on a locomotion
     build; set_command / set_rsi invalid on the reach build).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "arm"))

from walk.env import arm_reach as ar  # noqa: E402
from walk.env import cuda_lane  # noqa: E402
from walk.env.arm_cuda_lane import CudaArmLane  # noqa: E402
from walk.eval import arm_reach_judge as judge  # noqa: E402
from walk.eval.arm_acceptance import ScriptedIKPolicy, capture_arm_episodes  # noqa: E402
import arm_lowering as al  # noqa: E402

E = 4
OBS_TOL, REW_TOL = 1e-5, 1e-5


def _serial(variant):
    return lambda n, off: CudaArmLane(n, variant=variant, joint_offsets=off)


class DevicePolicyParityTests(unittest.TestCase):
    def _parity(self, variant, seed=0, steps=420):   # > horizon: masked resets
        env = ar.ArmReachEnv(environments=E, seed=seed, variant=variant,
                             lane_factory=_serial(variant))
        dev = CudaArmLane(E, variant=variant)
        try:
            obs_py = env.reset()                 # episode-2 draws
            dev.reset_policy(seed=seed)          # episode 1
            obs_dev = dev.reset_policy()         # episode 2
            np.testing.assert_array_equal(obs_py, obs_dev, "reset obs")
            np.testing.assert_array_equal(env.tier, dev.tier)
            np.testing.assert_array_equal(env.target,
                                          dev.reach_state()["target"])
            rng = np.random.default_rng(seed)
            pol = ScriptedIKPolicy(variant)
            worst_obs = worst_rew = 0.0
            resets = acquisitions = 0
            exact_obs = exact_rew = True
            for t in range(steps):
                # IK-driven steps acquire targets (exercising the queue);
                # noisy steps exercise slew / speed / torque terms.
                a = pol(obs_py) if t % 3 else rng.uniform(-0.4, 0.4, (E, 6))
                a = np.asarray(a, np.float32)
                o_py, r_py, d_py, info = env.step(a)
                o_dev, r_dev, d_dev, diag = dev.step_policy(a)
                self.assertEqual((diag["status"] != 0).sum(), 0, t)
                worst_obs = max(worst_obs, float(np.abs(o_py - o_dev).max()))
                worst_rew = max(worst_rew, float(np.abs(r_py - r_dev).max()))
                exact_obs &= bool(np.array_equal(o_py, o_dev))
                exact_rew &= bool(np.array_equal(r_py, r_dev))
                np.testing.assert_array_equal(d_py, d_dev, f"done @ {t}")
                rs = dev.reach_state()
                np.testing.assert_array_equal(
                    info["target_index"], rs["target_index"], f"index @ {t}")
                np.testing.assert_array_equal(env.target, rs["target"],
                                              f"target sequence @ {t}")
                acquisitions += int(info["acquired"].sum())
                obs_py = o_py
                if d_py.any():                   # trainer-style masked reset
                    resets += 1
                    obs_py = env.reset(mask=d_py)
                    o_dev = dev.reset_policy(mask=d_dev)
                    np.testing.assert_array_equal(obs_py, o_dev,
                                                  f"post-reset obs @ {t}")
                    np.testing.assert_array_equal(env.target,
                                                  dev.reach_state()["target"])
            self.assertGreater(acquisitions, 0, "no target acquired: the "
                               "queue promotion was never exercised")
            self.assertGreater(resets, 0, "no masked reset exercised")
            self.assertLess(worst_obs, OBS_TOL, "obs parity gate")
            self.assertLess(worst_rew, REW_TOL, "reward parity gate")
            print(f"{variant} device_policy_parity obs={worst_obs:.3e} "
                  f"reward={worst_rew:.3e} bit_exact obs={exact_obs} "
                  f"reward={exact_rew} acquisitions={acquisitions} "
                  f"masked_resets={resets}", file=sys.stderr)
        finally:
            env.close()
            dev.close()

    def test_parity_kr240(self):
        self._parity("kr240")

    def test_parity_lite(self):
        self._parity("lite")

    def test_proxy_crash_terminates_in_parity(self):
        """Driving a2 down (arm_env's proxy test) must terminate BOTH paths
        at the same step with the proxy penalty and reason FELL; the done
        env then freezes (reward 0, obs constant) until reset."""
        env = ar.ArmReachEnv(environments=1, seed=3, variant="kr240",
                             lane_factory=_serial("kr240"))
        dev = CudaArmLane(1, variant="kr240")
        try:
            env.reset()
            dev.reset_policy(seed=3)
            dev.reset_policy()
            lim = env.joint_limits
            q_goal = np.array([0.0, lim[1, 1], 0.0, 0.0, 0.0, 0.0])
            a = np.clip(2.0 * (q_goal - lim[:, 0]) / (lim[:, 1] - lim[:, 0])
                        - 1.0, -1, 1)[None].astype(np.float32)
            crashed_at = None
            for t in range(300):
                o_py, r_py, d_py, info = env.step(a)
                o_dev, r_dev, d_dev, _ = dev.step_policy(a)
                self.assertEqual(bool(d_py[0]), bool(d_dev[0]), t)
                self.assertLess(abs(float(r_py[0]) - float(r_dev[0])), REW_TOL)
                if d_py[0]:
                    crashed_at = t
                    self.assertTrue(bool(info["proxy_violation"][0]))
                    break
            self.assertIsNotNone(crashed_at, "expected a proxy termination")
            gp = dev.gate_proxy()[0]
            self.assertEqual(cuda_lane.TERM_REASONS[int(gp["termination_reason"])],
                             "fell")
            # done envs keep ticking physics with frozen targets on BOTH
            # paths (no auto-reset): reward 0, done latched, obs in parity
            o2_py, r2_py, d2_py, _ = env.step(a)
            o2, r2, d2, _ = dev.step_policy(a)
            self.assertTrue(bool(d2[0]) and bool(d2_py[0]))
            self.assertEqual(float(r2[0]), 0.0)
            self.assertEqual(float(r2_py[0]), 0.0)
            self.assertLess(float(np.abs(o2 - o2_py).max()), OBS_TOL)
            print(f"proxy crash parity: both paths terminated at step "
                  f"{crashed_at}", file=sys.stderr)
        finally:
            env.close()
            dev.close()


def _judge_tick_counts(trace: dict) -> dict:
    """The judge's clause 3/4/5 per-tick predicates and its acquisition
    records on a trace (exactly walk/eval/arm_reach_judge.evaluate_episode's
    arithmetic, exposed per tick for the counter cross-check)."""
    s = al.spec(trace["variant"])
    ticks = trace["ticks"]
    q = np.asarray(ticks["q"], float)
    qd = np.asarray(ticks["qd"], float)
    tip = np.asarray(ticks["tip"], float)
    wrist = np.asarray(ticks["wrist"], float)
    elbow = np.asarray(ticks["elbow"], float)
    lim = al.joint_limits(s)
    excess = np.maximum(np.maximum(lim[:, 0] - q, q - lim[:, 1]), 0.0)
    limit_ticks = int((excess.max(1) > judge.LIMIT_TOL_RAD + 1e-12).sum())
    ratio = np.abs(qd) / al.velocity_limits(s)
    speed_ticks = int((ratio.max(1) > judge.SPEED_TOL_FRAC + 1e-9).sum())
    proxy_ticks = int(judge.proxy_violation(trace["variant"], tip, wrist,
                                            elbow).sum())
    return {"limit": limit_ticks, "speed": speed_ticks, "proxy": proxy_ticks,
            "acq": judge.acquisitions(trace), "n_ticks": len(q)}


class JudgeCrossCheckTests(unittest.TestCase):
    def _cross_check(self, variant, policy_name, tier, seed, seconds):
        # Device path: the counters. Python path over the SAME build: the
        # per-tick trace the frozen judge consumes. Same seed + same actions
        # => identical physics (parity above), so the judge's counts must
        # equal the device counters exactly.
        dev = CudaArmLane(1, variant=variant, tier=tier)
        env = ar.ArmReachEnv(environments=1, seed=seed, variant=variant,
                             tier=tier, lane_factory=_serial(variant))
        try:
            actions = []
            if policy_name == "ik":
                pol = ScriptedIKPolicy(variant)

                def policy(obs):
                    a = np.asarray(pol(obs), np.float32)
                    actions.append(a.copy())
                    return a
            else:
                rng = np.random.default_rng(seed)

                def policy(obs):
                    a = rng.uniform(-1.0, 1.0, (1, 6)).astype(np.float32)
                    actions.append(a.copy())
                    return a
            traces = capture_arm_episodes(env, policy, tier, seconds=seconds,
                                          seed=seed)
            trace = traces[0]
            # replay the recorded actions on the device path: the env was
            # constructed with this seed (episode 1 at construction) and
            # capture's reset(seed=seed) is then its episode 2 -- align.
            dev.reset_policy(seed=seed)
            dev.reset_policy()
            np.testing.assert_array_equal(
                np.asarray(trace["ticks"]["target"][0]),
                dev.reach_state()["target"][0], "first target of the episode")
            steps = 0
            for a in actions:
                _o, _r, done, diag = dev.step_policy(a)
                self.assertEqual(int(diag["status"][0]), 0)
                steps += 1
                if done[0]:
                    break
            rs = dev.reach_state()[0]
            gp = dev.gate_proxy()[0]
            jc = _judge_tick_counts(trace)
            # (1) tick coverage: the trace holds exactly the accepted ticks
            self.assertEqual(jc["n_ticks"], steps * ar.TICKS_PER_STEP)
            # (2) clause 5: violating ticks -- exact
            self.assertEqual(int(rs["proxy_violation_ticks"]), jc["proxy"])
            # (3) clauses 3/4: per-tick predicates -- exact
            self.assertEqual(int(rs["limit_violation_ticks"]), jc["limit"])
            self.assertEqual(int(rs["speed_violation_ticks"]), jc["speed"])
            # (4) clause 2: acquisitions -- count and times exact
            acq = jc["acq"]
            n_judge = sum(1 for a in acq if a["acquired"])
            self.assertEqual(min(int(rs["target_index"]), judge.N_TARGETS),
                             n_judge)
            for k, rec in enumerate(acq):
                step = int(rs["acquire_step"][k])
                self.assertEqual(rec["acquired"], step > 0, k)
                if rec["acquired"]:
                    self.assertAlmostEqual(rec["time_s"],
                                           step * ar.CONTROL_DT, delta=1e-9)
            # (5) the shared gate_proxy mapping
            self.assertEqual(int(gp["qualified_left"]), int(rs["target_index"]))
            self.assertEqual(int(gp["qualified_right"]), 0)
            self.assertEqual(int(gp["alternation_violations"]),
                             jc["limit"] + jc["speed"] + jc["proxy"])
            # the judge's verdict on a non-terminated trace agrees with the
            # counters' implied clauses
            if not trace["terminated"] and seconds >= judge.EPISODE_SECONDS:
                verdict = judge.evaluate_episode(trace)
                c = verdict["criteria"]
                self.assertEqual(c["joint_limits_respected"]["pass"],
                                 jc["limit"] == 0)
                self.assertEqual(c["joint_speed_within_urdf_limits"]["pass"],
                                 jc["speed"] == 0)
                self.assertEqual(
                    c["no_self_collision_or_floor_proxy_violation"]["pass"],
                    jc["proxy"] == 0)
                self.assertEqual(
                    c["no_self_collision_or_floor_proxy_violation"]["detail"]
                    ["violating_ticks"], int(rs["proxy_violation_ticks"]))
            print(f"{variant}/{policy_name} tier{tier}: judge acquisitions="
                  f"{n_judge} device index={int(rs['target_index'])} "
                  f"acq_steps={rs['acquire_step'].tolist()} ticks limit/speed/"
                  f"proxy={jc['limit']}/{jc['speed']}/{jc['proxy']} "
                  f"(device {int(rs['limit_violation_ticks'])}/"
                  f"{int(rs['speed_violation_ticks'])}/"
                  f"{int(rs['proxy_violation_ticks'])}) terminated="
                  f"{trace['terminated']}", file=sys.stderr)
            return n_judge, jc
        finally:
            env.close()
            dev.close()

    def test_ik_baseline_acquisitions_match_judge_kr240(self):
        n, _ = self._cross_check("kr240", "ik", tier=0, seed=4242, seconds=8.0)
        self.assertGreaterEqual(n, 3, "IK baseline must acquire targets")

    def test_ik_baseline_acquisitions_match_judge_lite(self):
        n, _ = self._cross_check("lite", "ik", tier=1, seed=7, seconds=8.0)
        self.assertGreaterEqual(n, 2)

    def test_flail_violations_match_judge(self):
        # random +-1 actions saturate the slew: joint speeds exceed the URDF
        # limits (clause 4) on many ticks, and the arm may hit the proxy.
        for variant, seed in (("kr240", 11), ("lite", 12)):
            _n, jc = self._cross_check(variant, "flail", tier=2, seed=seed,
                                       seconds=8.0)
            self.assertGreater(jc["speed"] + jc["limit"] + jc["proxy"], 0,
                               f"{variant}: flail produced no violation, "
                               "the exactness check is vacuous")


class GateProxyAndAbiTests(unittest.TestCase):
    def test_episode_snapshot_and_gate_termination_knobs(self):
        dev = CudaArmLane(2, variant="kr240", tier=0)
        try:
            pol = ScriptedIKPolicy("kr240")
            obs = dev.reset_policy(seed=4242)
            for _t in range(150):
                obs, _r, done, diag = dev.step_policy(pol(obs))
                self.assertEqual(int((diag["status"] != 0).sum()), 0)
                self.assertFalse(done.any())
            live = dev.reach_state()
            self.assertGreater(int(live["target_index"].min()), 0)
            # reset -> the finished episode lands in the episode_* snapshot
            dev.reset_policy()
            snap = dev.reach_state()
            gp = dev.gate_proxy()
            np.testing.assert_array_equal(snap["episode_acquired"],
                                          live["target_index"])
            np.testing.assert_array_equal(snap["episode_acquire_step"],
                                          live["acquire_step"])
            np.testing.assert_array_equal(gp["episode_qualified_left"],
                                          live["target_index"])
            np.testing.assert_array_equal(
                gp["episode_alternation_violations"],
                live["limit_violation_ticks"] + live["speed_violation_ticks"]
                + live["proxy_violation_ticks"])
            self.assertTrue((snap["target_index"] == 0).all())
            self.assertTrue((snap["hold"] == 0).all())
            self.assertTrue((snap["valid"] == 1).all())
            self.assertTrue((snap["next_valid"] == 1).all())
            self.assertEqual(int(gp["episode_termination_reason"][0]), 0)
            # first-acquisition deadline: a standing arm (zero action holds
            # mid-range... use HOME-holding actions) never acquires -> dies
            # at exactly the deadline tick (300 ticks = step 30, 0-indexed 29)
            dev.set_gate_termination(first_deadline_ticks=300)
            dev.reset_policy()
            lim = dev.joint_limits
            home_a = (2.0 * (dev.home_joint_q - lim[:, 0])
                      / (lim[:, 1] - lim[:, 0]) - 1.0)[None].astype(np.float32)
            home_a = np.repeat(home_a, 2, 0)
            died_at = None
            for t in range(80):
                _o, _r, done, _d = dev.step_policy(home_a)
                if done.all():
                    died_at = t
                    break
            self.assertEqual(died_at, 29)
            gp = dev.gate_proxy()
            self.assertTrue((gp["termination_reason"] == 2).all(),
                            "DWC1_TERM_GATE_DEADLINE")
            # violating-tick cap: flail speed violations reach the cap
            dev.set_gate_termination(first_deadline_ticks=0,
                                     max_alternation_violations=5)
            dev.reset_policy()
            rng = np.random.default_rng(1)
            died = False
            for _t in range(100):
                _o, _r, done, _d = dev.step_policy(
                    rng.uniform(-1, 1, (2, 6)).astype(np.float32))
                if done.any():
                    died = True
                    break
            self.assertTrue(died, "violating-tick cap never fired")
            gp = dev.gate_proxy()
            self.assertIn(3, gp["termination_reason"].tolist(),
                          "DWC1_TERM_ALTERNATION (violating-tick cap)")
            dev.set_gate_termination()      # off again
        finally:
            dev.close()

    def test_starved_queue_terminates_loudly(self):
        """Raw ABI: an acquisition with no queued target ends the env with
        DWC1_TERM_REACH_STARVED instead of repeating the target."""
        import ctypes as C
        dev = CudaArmLane(1, variant="kr240", tier=0)
        try:
            dev.reset_policy(seed=4242)
            # drop the queued target: push active only, leaving next invalid
            rs = dev.reach_state()
            self.assertEqual(int(rs["next_valid"][0]), 1)
            # rebuild the env state with only the active slot valid via a
            # fresh reset (clears both) + active-only push
            mc = (C.c_uint8 * 1)(1)
            tier = np.array([0.0]); key = np.array([1.0])
            self.assertEqual(dev._lib.dwc1_reset_policy(
                dev._h, mc, tier.ctypes.data_as(cuda_lane.DP),
                key.ctypes.data_as(cuda_lane.DP)), 0)
            act = np.ascontiguousarray(rs["target"][0], np.float64)
            self.assertEqual(dev._lib.dwc1_reach_set_targets(
                dev._h, mc, act.ctypes.data_as(cuda_lane.DP), None), 0)
            rs = dev.reach_state()
            self.assertEqual((int(rs["valid"][0]), int(rs["next_valid"][0])), (1, 0))
            dev._index[:] = 0
            dev._refill_queue = lambda: None     # host deliberately silent
            pol = ScriptedIKPolicy("kr240")
            obs = dev.observe()
            reason = None
            for _t in range(300):
                obs, r, done, _d = dev.step_policy(pol(obs))
                if done[0]:
                    reason = int(dev.gate_proxy()[0]["termination_reason"])
                    break
            self.assertEqual(cuda_lane.TERM_REASONS[reason], "reach_starved")
            self.assertEqual(int(dev.reach_state()[0]["starved"]), 1)
            self.assertGreater(float(r[0]), 5.0, "acquisition bonus still paid")
        finally:
            dev.close()

    def test_abi_kind_guards(self):
        import ctypes as C
        from walk.env.cuda_lane import CudaDuckLane
        duck = CudaDuckLane(1)
        arm = CudaArmLane(1, variant="lite")
        try:
            self.assertEqual(int(duck._lib.dwc1_env_kind()),
                             cuda_lane.ENV_KIND_LOCOMOTION)
            self.assertEqual(int(duck._lib.dwc1_obs_width()), cuda_lane.OBS)
            self.assertEqual(int(arm._lib.dwc1_env_kind()), cuda_lane.ENV_KIND_REACH)
            self.assertEqual(int(arm._lib.dwc1_obs_width()), ar.OBS)
            out = (cuda_lane.ReachState * 1)()
            self.assertEqual(duck._lib.dwc1_reach_get(duck._h, out), 1)
            t = np.zeros(3)
            self.assertEqual(duck._lib.dwc1_reach_set_targets(
                duck._h, None, t.ctypes.data_as(cuda_lane.DP), None), 1)
            cmd = np.zeros(1)
            self.assertEqual(arm._lib.dwc1_set_command(
                arm._h, cmd.ctypes.data_as(cuda_lane.DP)), 1)
            self.assertEqual(arm._lib.dwc1_set_rsi(arm._h, C.c_double(0.5)), 1)
            self.assertEqual(arm._lib.dwc1_set_rsi(arm._h, C.c_double(0.0)), 0)
            with self.assertRaises(NotImplementedError):
                CudaArmLane(1, variant="lite", randomization={"r_mass": 0.1})
        finally:
            duck.close()
            arm.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
