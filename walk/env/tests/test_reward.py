"""Unit tests for the reward terms and gait tracker (no physics).

Run: .venv/bin/python -B -m unittest discover -s walk
"""
import unittest

import numpy as np

from walk.env import reward as rw


def make_state(E=1, vx=0.0, vy=0.0, wz=0.0, contact=((True, True),),
               sole=((0.0, 0.0),), action=None, torque=None):
    return {
        "root_lin_vel": np.c_[np.full(E, vx), np.full(E, vy), np.zeros(E)],
        "root_ang_vel": np.c_[np.zeros(E), np.zeros(E), np.full(E, wz)],
        "foot_contact": np.array(contact * (E // len(contact)) if len(contact) != E
                                 else contact, bool).reshape(E, 2),
        "sole_height": np.asarray(sole * (E // len(sole)) if len(sole) != E
                                  else sole, float).reshape(E, 2),
        "action": np.zeros((E, 14)) if action is None else np.asarray(action),
        "torque": np.zeros((E, 14)) if torque is None else np.asarray(torque),
    }


class TestSmoothTerms(unittest.TestCase):
    def test_velocity_tracking_and_alive(self):
        # preset the rolling average to steady state so the term arithmetic
        # is exact (the EMA converges to vx during real walking)
        tracker = rw.GaitTracker(1)
        tracker.v_avg[:] = 0.15
        prev, cur = make_state(), make_state(vx=0.15)
        r = rw.reward(prev, cur, np.zeros((1, 14)), np.array([0.15]), tracker)
        self.assertAlmostEqual(float(r[0]), rw.W_TRACK + rw.W_ALIVE, places=5)
        tracker = rw.GaitTracker(1)
        tracker.v_avg[:] = 0.05
        r_off = rw.reward(prev, make_state(vx=0.05), np.zeros((1, 14)),
                          np.array([0.15]), tracker)
        expected = rw.W_TRACK * np.exp(-0.01 / rw.TRACK_SIGMA_SQ) + rw.W_ALIVE
        self.assertAlmostEqual(float(r_off[0]), expected, places=5)

    def test_penalties_reduce_reward(self):
        base = float(rw.reward(make_state(), make_state(vx=0.1),
                               np.zeros((1, 14)), np.array([0.1]),
                               rw.GaitTracker(1))[0])
        lateral = float(rw.reward(make_state(), make_state(vx=0.1, vy=0.3, wz=0.5),
                                  np.zeros((1, 14)), np.array([0.1]),
                                  rw.GaitTracker(1))[0])
        self.assertAlmostEqual(base - lateral,
                               rw.W_LATERAL * (0.09 + 0.25), places=5)
        act = np.full((1, 14), 0.5)
        rate = float(rw.reward(make_state(), make_state(vx=0.1, action=act),
                               act, np.array([0.1]), rw.GaitTracker(1))[0])
        self.assertAlmostEqual(base - rate,
                               rw.W_ACTION_RATE * 14 * 0.25, places=5)
        tau = np.full((1, 14), 2.0)
        torq = float(rw.reward(make_state(), make_state(vx=0.1, torque=tau),
                               np.zeros((1, 14)), np.array([0.1]),
                               rw.GaitTracker(1))[0])
        self.assertAlmostEqual(base - torq, rw.W_TORQUE * 14 * 4.0, places=5)


class TestGaitShaping(unittest.TestCase):
    def steps(self, tracker, sequence, cmd=0.15):
        """sequence: list of (contact_pair, sole_pair); returns rewards."""
        tracker.v_avg[:] = cmd          # steady-state rolling average
        out = []
        prev = make_state(contact=(sequence[0][0],), sole=(sequence[0][1],))
        for contact, sole in sequence[1:]:
            cur = make_state(vx=cmd, contact=(contact,), sole=(sole,))
            out.append(float(rw.reward(prev, cur, np.zeros((1, 14)),
                                       np.array([cmd]), tracker)[0]))
            prev = cur
        return out

    def test_air_time_bonus_and_alternation(self):
        flat = (0.0, 0.0)
        up = (0.02, 0.0)
        # left swing 0.2 s (10 steps) with clearance, then touchdown
        seq = [((True, True), flat)] * 5 + [((False, True), up)] * 10 \
            + [((True, True), flat)]
        tracker = rw.GaitTracker(1)
        rewards = self.steps(tracker, seq)
        base = rewards[-2]  # swing step with clearance bonus
        touchdown = rewards[-1]
        self.assertAlmostEqual(touchdown - (base - rw.W_CLEARANCE),
                               rw.W_AIR_TIME, places=5)
        self.assertEqual(tracker.last_foot[0], 0)
        # an alternating right-foot step now earns air time + alternation
        seq2 = [((True, True), flat)] * 5 + [((True, False), (0.0, 0.02))] * 10 \
            + [((True, True), flat)]
        rewards2 = self.steps(tracker, seq2)
        self.assertAlmostEqual(rewards2[-1] - (rewards2[-2] - rw.W_CLEARANCE),
                               rw.W_AIR_TIME + rw.W_ALTERNATE, places=5)
        self.assertEqual(tracker.last_foot[0], 1)

    def test_too_short_and_too_long_swings_earn_nothing(self):
        flat = (0.0, 0.0)
        for steps in (3, 25):  # 0.06 s and 0.5 s are outside [0.1, 0.4]
            tracker = rw.GaitTracker(1)
            seq = [((True, True), flat)] + [((False, True), flat)] * steps \
                + [((True, True), flat)]
            rewards = self.steps(tracker, seq)
            self.assertLess(rewards[-1] - rewards[-2], rw.W_AIR_TIME / 2)
            self.assertEqual(tracker.last_foot[0], -1)

    def test_double_support_penalty_after_grace(self):
        tracker = rw.GaitTracker(1)
        seq = [((True, True), (0.0, 0.0))] * 20
        rewards = self.steps(tracker, seq)
        grace_steps = int(rw.DOUBLE_SUPPORT_GRACE / rw.CONTROL_DT)
        self.assertAlmostEqual(rewards[grace_steps] - rewards[grace_steps - 2],
                               -rw.W_DOUBLE_SUPPORT, places=5)

    def test_tracker_reset_clears_state(self):
        tracker = rw.GaitTracker(2)
        tracker.air_time[:] = 0.2
        tracker.last_foot[:] = 1
        tracker.double_support[:] = 1.0
        tracker.reset(np.array([True, False]))
        self.assertEqual(tracker.air_time[0].tolist(), [0.0, 0.0])
        self.assertEqual(tracker.last_foot.tolist(), [-1, 1])
        self.assertEqual(tracker.double_support.tolist(), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()


class PhaseLockTest(unittest.TestCase):
    def _state(self, contact, phase):
        E = len(phase)
        return {"root_lin_vel": np.zeros((E, 3)), "root_ang_vel": np.zeros((E, 3)),
                "foot_contact": np.array(contact, bool), "sole_height": np.zeros((E, 2)),
                "action": np.zeros((E, 14)), "torque": np.zeros((E, 14)),
                "phase": np.asarray(phase, float)}

    def test_phase_locked_stance_rewards_alternation(self):
        from walk.env import reward as rm
        tr = rm.GaitTracker(3)
        # env0: left stance during left window (sin>=0) and right up -> both match
        # env1: limp (left always down, right down) during left window -> 1 match
        # env2: exactly wrong (left up, right down in left window) -> 1 match
        phase = [0.1, 0.1, 0.1]  # sin>0: left window
        contact = [[True, False], [True, True], [False, True]]
        prev = self._state(contact, phase)
        r = rm.reward(prev, self._state(contact, phase), np.zeros((3, 14)),
                      np.full(3, 0.15), tr)
        # signed term: each foot flipping match->mismatch swings 2*W_PHASE
        self.assertAlmostEqual(float(r[0] - r[1]), 2 * rm.W_PHASE, places=5)
        # limp keeps one matching foot (left), fully inverted stance matches none
        self.assertAlmostEqual(float(r[1] - r[2]), 2 * rm.W_PHASE, places=5)

    def test_phase_term_absent_without_phase_key(self):
        from walk.env import reward as rm
        tr = rm.GaitTracker(1)
        s = self._state([[True, False]], [0.1]); del s["phase"]
        r_no = rm.reward(s, dict(s), np.zeros((1, 14)), np.full(1, 0.15), rm.GaitTracker(1))
        s2 = self._state([[True, False]], [0.1])
        r_yes = rm.reward(s2, dict(s2), np.zeros((1, 14)), np.full(1, 0.15), tr)
        self.assertAlmostEqual(float(r_yes[0] - r_no[0]), 2 * rm.W_PHASE, places=5)


class SameFootPenaltyTest(unittest.TestCase):
    def test_repeated_foot_qualified_touchdown_is_penalized(self):
        flat = (0.0, 0.0)
        up = (0.02, 0.0)
        swing = [((True, True), flat)] * 5 + [((False, True), up)] * 10 \
            + [((True, True), flat)]
        tracker = rw.GaitTracker(1)
        tracker.v_avg[:] = 0.15
        helper = TestGaitShaping()
        first = helper.steps.__func__(helper, tracker, swing)
        self.assertEqual(tracker.last_foot[0], 0)
        second = helper.steps.__func__(helper, tracker, swing)
        # same left foot again: step bonus still paid, minus the repeat penalty
        self.assertAlmostEqual(first[-1] - second[-1], rw.W_SAME_FOOT, places=5)
