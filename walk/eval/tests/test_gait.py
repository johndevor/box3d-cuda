"""Strict gait evaluator unit tests on hand-built synthetic traces.

Run: .venv/bin/python -B -m unittest discover -s walk
"""
import copy
import unittest

import numpy as np

from walk.eval import gait

DT = 0.002
N = int(round(8.0 / DT))  # 4000 ticks


def synthetic_trace(command=0.15, walking=True, speed_scale=1.0):
    """Hand-built 8 s episode.

    walking=True: alternating 0.3 s swings every 0.4 s starting at t=0.5 s
    (left first), whole-sole apex 15 mm, 60 mm forward foot travel per swing,
    stationary stance feet, upright base translating at `command`.
    walking=False: a lunge/shuffle — both feet grounded for all 8 s, soles
    never leave the floor, base still translates (so only the footfall and
    clearance criteria can fail).
    """
    t = DT * np.arange(1, N + 1)
    contact = np.ones((N, 2), bool)
    sole = np.zeros((N, 2))
    foot_x = np.zeros((N, 2))
    base_x = command * speed_scale * t
    if walking:
        step_x = [0.0, 0.0]
        k = 0
        while True:
            start = 0.5 + 0.4 * k
            end = start + 0.3
            if end + gait.SUPPORT_S > 8.0 - DT:
                break
            foot = k % 2  # left, right, left, ...
            i0, i1 = int(round(start / DT)), int(round(end / DT))
            contact[i0:i1, foot] = False
            phase = (np.arange(i1 - i0) + 0.5) / (i1 - i0)
            sole[i0:i1, foot] = 0.015 * np.sin(np.pi * phase)
            foot_x[i0:i1, foot] = step_x[foot] + 0.060 * phase
            step_x[foot] += 0.060
            foot_x[i1:, foot] = step_x[foot]
            k += 1
    foot_pos = np.zeros((N, 2, 3))
    foot_pos[:, :, 0] = foot_x
    foot_pos[:, 0, 1], foot_pos[:, 1, 1] = 0.08, -0.08
    return {
        "schema": gait.SCHEMA, "dt": DT, "policy_dt": 0.02,
        "command_mps": command, "resets": 0, "terminated": False,
        "truncated_at_horizon": True, "solver_fault": False,
        "ticks": {
            "time_s": t.tolist(),
            "base_pos": np.c_[base_x, np.zeros(N), np.full(N, 0.168)].tolist(),
            "base_quat_xyzw": [[0.0, 0.0, 0.0, 1.0]] * N,
            "tilt_deg": [0.0] * N,
            "foot_pos": foot_pos.tolist(),
            "sole_height": sole.tolist(),
            "contact": contact.tolist(),
        },
    }


class TestAlternatingWalkPasses(unittest.TestCase):
    def test_walking_trace_passes_every_criterion(self):
        result = gait.evaluate_episode(synthetic_trace())
        self.assertFalse(result["rejected"])
        for name, c in result["criteria"].items():
            self.assertTrue(c["pass"], f"{name}: {c['detail']}")
        self.assertTrue(result["passed"])
        feet = [f["foot"] for f in result["footfalls"]]
        self.assertGreaterEqual(len(feet), 6)
        self.assertEqual(feet, [("left", "right")[i % 2] for i in range(len(feet))])

    def test_three_command_run_passes(self):
        run = gait.evaluate_run([synthetic_trace(c) for c in (0.10, 0.15, 0.20)])
        self.assertTrue(run["commands_complete"])
        self.assertTrue(run["passed"])

    def test_run_requires_all_three_commands(self):
        run = gait.evaluate_run([synthetic_trace(c) for c in (0.10, 0.15, 0.15)])
        self.assertFalse(run["commands_complete"])
        self.assertFalse(run["passed"])


class TestLungeFails(unittest.TestCase):
    def test_lunge_fails_exactly_on_footfall_and_clearance_criteria(self):
        """Feet never clear 10 mm / never leave the ground: every footfall
        criterion fails; translation, tilt and integrity still pass."""
        result = gait.evaluate_episode(synthetic_trace(walking=False))
        self.assertFalse(result["rejected"])
        self.assertFalse(result["passed"])
        c = result["criteria"]
        must_fail = ["at_least_6_qualified_footfalls", "at_least_3_per_foot",
                     "footfalls_alternate", "first_step_within_2p5_s",
                     "last_step_within_final_1p5_s"]
        must_pass = ["single_episode_no_reset_or_failure",
                     "translation_60_to_150_percent", "tilt_within_30_degrees"]
        for name in must_fail:
            self.assertFalse(c[name]["pass"], name)
        for name in must_pass:
            self.assertTrue(c[name]["pass"], name)
        self.assertEqual(result["footfalls"], [])

    def test_shuffle_without_clearance_disqualifies_on_clearance(self):
        """Steps with real swings but soles never reaching 10 mm must be
        disqualified specifically by the clearance rule."""
        trace = synthetic_trace()
        s = np.asarray(trace["ticks"]["sole_height"])
        trace["ticks"]["sole_height"] = np.minimum(s, 0.008).tolist()
        result = gait.evaluate_episode(trace)
        self.assertFalse(result["passed"])
        self.assertEqual(result["footfalls"], [])
        self.assertFalse(result["criteria"]["at_least_6_qualified_footfalls"]["pass"])


class TestRejectedTraces(unittest.TestCase):
    def test_reset_concatenated_trace_is_rejected(self):
        trace = synthetic_trace()
        trace["resets"] = 1
        result = gait.evaluate_episode(trace)
        self.assertTrue(result["rejected"])
        self.assertFalse(result["passed"])

    def test_spliced_clock_is_rejected(self):
        """Two half-episodes glued together (clock restarts) must reject even
        if the resets counter lies."""
        trace = synthetic_trace()
        t = trace["ticks"]["time_s"]
        trace["ticks"]["time_s"] = t[: N // 2] + t[: N // 2 + N % 2]
        result = gait.evaluate_episode(trace)
        self.assertTrue(result["rejected"])
        self.assertIn("clock", result["criteria"]
                      ["single_episode_no_reset_or_failure"]["detail"])

    def test_terminated_or_faulted_or_short_traces_are_rejected(self):
        for mutate in (
            lambda x: x.update(terminated=True),
            lambda x: x.update(solver_fault=True),
            lambda x: x["ticks"].update(
                {k: v[: N // 2] for k, v in x["ticks"].items()}),
        ):
            trace = copy.deepcopy(synthetic_trace())
            mutate(trace)
            result = gait.evaluate_episode(trace)
            self.assertTrue(result["rejected"])


class TestIndividualQualifiers(unittest.TestCase):
    def make(self):
        return synthetic_trace()

    def test_translation_out_of_band_fails(self):
        slow = gait.evaluate_episode(synthetic_trace(speed_scale=0.5))
        fast = gait.evaluate_episode(synthetic_trace(speed_scale=1.6))
        self.assertFalse(slow["criteria"]["translation_60_to_150_percent"]["pass"])
        self.assertFalse(fast["criteria"]["translation_60_to_150_percent"]["pass"])

    def test_tilt_over_30_degrees_fails(self):
        trace = self.make()
        trace["ticks"]["tilt_deg"][2000] = 31.0
        result = gait.evaluate_episode(trace)
        self.assertFalse(result["criteria"]["tilt_within_30_degrees"]["pass"])

    def test_backward_placement_disqualifies(self):
        trace = self.make()
        fp = np.asarray(trace["ticks"]["foot_pos"])
        fp[:, :, 0] *= 0.0  # feet never move forward
        trace["ticks"]["foot_pos"] = fp.tolist()
        result = gait.evaluate_episode(trace)
        self.assertEqual(result["footfalls"], [])

    def test_opposite_foot_lift_disqualifies(self):
        trace = self.make()
        contact = np.asarray(trace["ticks"]["contact"], bool)
        # lift the right foot during the first 20% of every left swing
        for i0, i1 in gait._runs(~contact[:, 0]):
            contact[i0:i0 + (i1 - i0) // 5, 1] = False
        trace["ticks"]["contact"] = contact.tolist()
        result = gait.evaluate_episode(trace)
        self.assertEqual([f for f in result["footfalls"] if f["foot"] == "left"],
                         [])

    def test_stance_slide_disqualifies(self):
        trace = self.make()
        fp = np.asarray(trace["ticks"]["foot_pos"])
        fp[:, :, 0] += 0.2 * DT * np.arange(N)[:, None]  # feet skate forward
        trace["ticks"]["foot_pos"] = fp.tolist()
        result = gait.evaluate_episode(trace)
        self.assertEqual(result["footfalls"], [])
        qualified, swings = gait._qualified_footfalls(trace)
        self.assertEqual(qualified, [])
        self.assertTrue(any("stance slip beyond 25 mm bound"
                            in s["disqualified_because"] for s in swings))


if __name__ == "__main__":
    unittest.main()
