"""Humanoid strict-evaluator gates: synthetic trajectories + leg-1 actor.

Run: .venv/bin/python -B humanoid/tests/test_humanoid_gait_eval.py

Synthetic hand-authored traces, each passed/failed for the RIGHT reason:
  - a clean alternating gait (0.6 s cycle, 0.2 s swings, 80 mm clearance,
    0.45 m steps, both feet planted between swings)      -> PASSES;
  - the same gait shuffled flat (20 mm max clearance)    -> fails ONLY the
    clearance clause (footfall counts collapse);
  - a one-leg hop (right foot never swings)              -> fails per-foot
    balance and alternation;
  - standing still                                       -> fails footfall
    counts, translation band and first-step deadline;
  - a 6 ms mid-stance contact dropout is debounced away (no phantom swing);
    a 200 ms real swing is NOT debounced;
  - integrity: terminated / spliced traces are rejected outright.

Integration: the leg-1 GPU actor (runs/gpu/20260901-200306-humanoid-train-ff)
through the acceptance harness on the CPU-serial fp32 lane, one seed x all
three commands: expected NOT accepted (bring-up actor); the test asserts
the harness completes, reports per-criterion verdicts, and prints them.
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

from walk.eval import humanoid_gait as hg  # noqa: E402

LEG1_ACTOR = (ROOT / "runs" / "gpu" / "20260901-200306-humanoid-train-ff"
              / "artifacts" / "train" / "gpu-train-out" / "actor_final.pt")

QX90 = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
DT = 0.002


def synthetic_trace(cmd: float = 0.75, seconds: float = 8.0,
                    clearance_peak: float = 0.08, feet=("left", "right"),
                    swings: bool = True, contact_dropout: tuple | None = None):
    """Hand-authored kinematically consistent episode trace.

    Gait: 0.6 s cycle; left swing on cycle time [0.1, 0.3), right on
    [0.4, 0.6); each swinging foot advances cmd*0.6 m during its swing and
    holds position in stance; sole rises as a sine bump of `clearance_peak`;
    base advances at exactly cmd. `feet` limits which feet ever swing;
    `swings=False` is a stand-still. `contact_dropout=(foot, t0, t1)`
    forces the contact flag False on [t0, t1) without any kinematic swing.
    """
    n = int(round(seconds / DT))
    t = DT * np.arange(1, n + 1)
    cycle = 0.6
    swing_len = 0.2
    windows = {"left": 0.1, "right": 0.4}      # swing start within the cycle
    contact = np.ones((n, 2), bool)
    sole = np.zeros((n, 2))
    foot_x = np.zeros((n, 2))
    step_len = cmd * cycle
    for f, name in enumerate(("left", "right")):
        base_x = 0.0
        if not swings or name not in feet:
            foot_x[:, f] = 0.0
            continue
        for k in range(int(seconds / cycle) + 2):
            s0 = k * cycle + windows[name]
            s1 = s0 + swing_len
            i0, i1 = int(round(s0 / DT)), int(round(s1 / DT))
            if i0 >= n:
                break
            j1 = min(i1, n)
            contact[i0:j1, f] = False
            prog = (np.arange(i0, j1) - i0 + 0.5) / (i1 - i0)
            sole[i0:j1, f] = clearance_peak * np.sin(np.pi * prog)
            foot_x[i0:j1, f] = base_x + step_len * prog
            if i1 < n:
                base_x += step_len
                foot_x[i1:, f] = base_x
    if contact_dropout is not None:
        f, d0, d1 = contact_dropout
        contact[int(round(d0 / DT)):int(round(d1 / DT)), f] = False
    base = np.zeros((n, 3))
    base[:, 0] = cmd * t
    base[:, 2] = 1.15
    return {
        "schema": hg.SCHEMA, "dt": DT, "policy_dt": 0.02,
        "command_mps": float(cmd), "seed": 0, "env_index": 0,
        "resets": 0, "terminated": False, "truncated_at_horizon": True,
        "solver_fault": False,
        "ticks": {
            "time_s": t.tolist(),
            "base_pos": base.tolist(),
            "base_quat_xyzw": [QX90] * n,
            "tilt_deg": [90.0] * n,      # duck-frame junk: must be ignored
            "foot_pos": np.stack(
                [np.stack([foot_x[:, f],
                           np.full(n, 0.15 if f == 0 else -0.15),
                           np.full(n, 0.07)], axis=1)
                 for f in range(2)], axis=1).tolist(),
            "sole_height": sole.tolist(),
            "contact": contact.tolist(),
        },
    }


class SyntheticTrajectoryTests(unittest.TestCase):
    def test_clean_alternating_gait_passes(self):
        r = hg.evaluate_episode(synthetic_trace())
        self.assertFalse(r["rejected"])
        failing = {k: v for k, v in r["criteria"].items() if not v["pass"]}
        self.assertEqual(failing, {}, failing)
        self.assertTrue(r["passed"])
        q = r["footfalls"]
        self.assertGreaterEqual(len(q), hg.MIN_FOOTFALLS)
        self.assertEqual(r["criteria"]["tilt_within_30_degrees"]
                         ["detail"]["max_tilt_deg"], 0.0)  # duck column ignored
        # every qualified swing shows the authored numbers
        self.assertAlmostEqual(q[0]["placement_m"], 0.45, places=6)
        self.assertAlmostEqual(q[0]["swing_s"], 0.2, places=6)

    def test_shuffle_fails_on_clearance_only(self):
        r = hg.evaluate_episode(synthetic_trace(clearance_peak=0.020))
        self.assertFalse(r["passed"])
        self.assertFalse(r["criteria"]["at_least_6_qualified_footfalls"]["pass"])
        # every examined swing must be disqualified BY THE CLEARANCE CLAUSE
        full = hg._qualified_footfalls(synthetic_trace(clearance_peak=0.020))
        qualified, swings = full
        self.assertEqual(qualified, [])
        self.assertGreater(len(swings), 10)
        for s in swings:
            self.assertEqual(s["disqualified_because"],
                             ["whole-sole 30 mm clearance held < 30 ms"], s)

    def test_one_leg_hop_fails_balance_and_alternation(self):
        r = hg.evaluate_episode(synthetic_trace(feet=("left",)))
        self.assertFalse(r["passed"])
        self.assertFalse(r["criteria"]["at_least_3_per_foot"]["pass"])
        self.assertEqual(r["criteria"]["at_least_3_per_foot"]
                         ["detail"]["right"], 0)
        self.assertFalse(r["criteria"]["footfalls_alternate"]["pass"])
        self.assertTrue(all(x == "left" for x in
                            r["criteria"]["footfalls_alternate"]["detail"]))
        # the left-only steps themselves are fine (opposite foot is planted)
        self.assertTrue(r["criteria"]["at_least_6_qualified_footfalls"]["pass"])

    def test_stand_still_fails_footfalls_and_translation(self):
        trace = synthetic_trace(swings=False)
        trace["ticks"]["base_pos"] = (np.zeros((len(trace["ticks"]["time_s"]), 3))
                                      + [0.0, 0.0, 1.15]).tolist()
        r = hg.evaluate_episode(trace)
        self.assertFalse(r["passed"])
        self.assertEqual(r["criteria"]["at_least_6_qualified_footfalls"]
                         ["detail"], 0)
        self.assertFalse(r["criteria"]["translation_60_to_150_percent"]["pass"])
        self.assertAlmostEqual(
            r["criteria"]["translation_60_to_150_percent"]["detail"]["ratio"],
            0.0, places=9)
        self.assertFalse(r["criteria"]["first_step_within_2p5_s"]["pass"])
        self.assertFalse(r["criteria"]["last_step_within_final_1p5_s"]["pass"])

    def test_contact_debounce_fills_6ms_dropout_only(self):
        # 6 ms mid-stance dropout: debounced away, gait still passes clean
        r = hg.evaluate_episode(synthetic_trace(
            contact_dropout=(0, 4.02, 4.026)))
        self.assertTrue(r["passed"],
                        {k: v for k, v in r["criteria"].items()
                         if not v["pass"]})
        # a real 200 ms swing is NOT debounced: the clean trace's swing count
        # survives (26 bracketed swings examined either way)
        self.assertGreaterEqual(r["swings_examined"], 20)

    def test_terminated_trace_rejected(self):
        trace = synthetic_trace()
        trace["terminated"] = True
        r = hg.evaluate_episode(trace)
        self.assertTrue(r["rejected"])
        self.assertFalse(r["passed"])
        self.assertIn("terminated",
                      r["criteria"]["single_episode_no_reset_or_failure"]
                      ["detail"])

    def test_short_trace_rejected(self):
        trace = synthetic_trace(seconds=4.0)
        r = hg.evaluate_episode(trace)
        self.assertTrue(r["rejected"])
        self.assertIn("shorter", r["criteria"]
                      ["single_episode_no_reset_or_failure"]["detail"])

    def test_humanoid_tilt_formula(self):
        self.assertAlmostEqual(float(hg.humanoid_tilt_deg(np.array(QX90))),
                               0.0, places=9)
        self.assertAlmostEqual(
            float(hg.humanoid_tilt_deg(np.array([0.0, 0.0, 0.0, 1.0]))),
            90.0, places=9)

    def test_run_requires_all_three_commands(self):
        good = synthetic_trace(cmd=0.75)
        r = hg.evaluate_run([good, synthetic_trace(cmd=0.5),
                             synthetic_trace(cmd=1.0)])
        self.assertTrue(r["commands_complete"])
        self.assertTrue(r["passed"])
        r2 = hg.evaluate_run([good, good, good])
        self.assertFalse(r2["commands_complete"])
        self.assertFalse(r2["passed"])


class Leg1ActorIntegration(unittest.TestCase):
    def test_leg1_actor_verdict(self):
        """Bring-up actor through the frozen judge (1 seed x 3 commands).

        Expected NOT accepted; the value of this test is the per-criterion
        report (what the reward needs next) and that the harness runs
        end-to-end on the CPU-serial fp32 lane."""
        if not LEG1_ACTOR.is_file():
            raise unittest.SkipTest(f"leg-1 actor not present: {LEG1_ACTOR}")
        from walk.eval.humanoid_acceptance import run_acceptance
        result = run_acceptance(str(LEG1_ACTOR), lane="serial",
                                seeds=(4242,), quiet=True)
        self.assertEqual(len(result["episodes"]), 3)
        for key, ep in result["episodes"].items():
            self.assertIsInstance(ep["failed_criteria"], list, key)
            print(f"leg1 {key}: passed={ep['passed']} q={ep['qualified']} "
                  f"(L{ep['left']}/R{ep['right']}) "
                  f"fails={ep['failed_criteria']}", file=sys.stderr)
        self.assertFalse(result["accepted"],
                         "leg-1 actor unexpectedly ACCEPTED -- verify and "
                         "celebrate, then update this gate")


if __name__ == "__main__":
    unittest.main(verbosity=2)
