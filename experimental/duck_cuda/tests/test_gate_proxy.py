"""gate_proxy_* judge-shadow counters: ranking validation + semantics.

Run: .venv/bin/python -B experimental/duck_cuda/tests/test_gate_proxy.py

The in-kernel counters (dwc1_gate_proxy_get) approximate the FROZEN
walking judge's core footfall clauses at tick resolution (swing duration,
whole-sole contiguous clearance, forward placement, per-foot counts,
alternation violations) WITHOUT the judge's 20 ms contact-debounce sensor
model or its support/slip clauses. They are metrics for population culling
and monitoring -- never a substitute for the judge. This suite pins the
one property that makes them usable for culling: they RANK actors the way
the judge does.

Measured at pin time (and asserted with margin below):
  - duck walking-accepted actor (runs/actor-walking-v1.pt), cmd 0.15,
    8 s: proxy 19 L + 19 R qualified, 0 alternation violations -- exactly
    the frozen judge's accepted count for this command
    (runs/acceptance-FINAL: qualified 38, left 19, right 19);
  - duck standing (zero actions): 0 everywhere;
  - the humanoid reward-shuffle actor
    (runs/gpu/20260901-211217-humanoid-train-ff, H0-era 52/12, replayed on
    the current H1 lane driving its own 12 joints with the two added hip
    rolls held at home): 0 qualified swings -- the judge likewise scores
    shuffling at zero (no 150 mm placement ever happens).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "humanoid"))

from walk.env.cuda_lane import CudaDuckLane  # noqa: E402

DUCK_ACTOR = ROOT / "runs" / "actor-walking-v1.pt"
SHUFFLE_ACTOR = (ROOT / "runs" / "gpu" / "20260901-211217-humanoid-train-ff"
                 / "artifacts" / "train" / "gpu-train-out" / "actor_final.pt")
# H0 12-joint index -> H1 14-joint index (H1 inserts hip rolls at 2 and 6).
H0_TO_H1 = [0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]


def _load_actor(path, obs_dim, act_dim):
    import torch
    from walk.train.ppo import Actor
    ck = torch.load(path, map_location="cpu", weights_only=False)
    actor = Actor(obs_dim, act_dim)
    actor.load_state_dict(ck["state_dict"])
    actor.eval()
    return torch, actor


class GateProxyRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DUCK_ACTOR.is_file():
            raise unittest.SkipTest(f"missing {DUCK_ACTOR}")
        try:
            import torch  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f"torch unavailable: {e}")

    def test_walking_actor_matches_judge_and_standing_scores_zero(self):
        torch, actor = _load_actor(DUCK_ACTOR, 58, 14)
        lane = CudaDuckLane(1)
        try:
            obs = lane.reset_policy(seed=0, commands=[0.15],
                                    phase_offsets=[0.0])
            with torch.no_grad():
                for _t in range(400):
                    a = actor.deterministic(
                        torch.from_numpy(obs).float()).numpy()
                    obs, _r, done, diag = lane.step_policy(a)
                    self.assertEqual(int(diag["status"][0]), 0)
                    if done.any():
                        break
            walk = lane.gate_proxy()[0]
            # judge reference (acceptance-FINAL, cmd 0.15): 19 L / 19 R.
            self.assertGreaterEqual(int(walk["qualified_left"]), 12)
            self.assertGreaterEqual(int(walk["qualified_right"]), 12)
            self.assertLessEqual(int(walk["alternation_violations"]), 2)
            total = int(walk["qualified_left"]) + int(walk["qualified_right"])
            self.assertGreaterEqual(total, 30, "walking actor undercounted")
            # reset -> the finished episode lands in the episode_* snapshot
            lane.reset_policy(commands=[0.15], phase_offsets=[0.0])
            snap = lane.gate_proxy()[0]
            self.assertEqual(int(snap["episode_qualified_left"]),
                             int(walk["qualified_left"]))
            self.assertEqual(int(snap["episode_qualified_right"]),
                             int(walk["qualified_right"]))
            self.assertEqual(int(snap["qualified_left"]), 0)
            self.assertEqual(int(snap["qualified_right"]), 0)
            # standing control on the SAME lane: zero everywhere
            for _t in range(400):
                _o, _r, done, _d = lane.step_policy(
                    np.zeros((1, 14), np.float32))
                if done.any():
                    break
            stand = lane.gate_proxy()[0]
            self.assertEqual(int(stand["qualified_left"]), 0)
            self.assertEqual(int(stand["qualified_right"]), 0)
            print(f"gate_proxy duck: walking L/R="
                  f"{int(walk['qualified_left'])}/"
                  f"{int(walk['qualified_right'])} "
                  f"alt_viol={int(walk['alternation_violations'])}; "
                  f"standing 0/0 (judge: 19/19)", file=sys.stderr)
        finally:
            lane.close()

    def test_shuffle_actor_scores_zero_like_the_judge(self):
        if not SHUFFLE_ACTOR.is_file():
            raise unittest.SkipTest(f"missing {SHUFFLE_ACTOR}")
        from walk.env import humanoid_cuda_lane as hc
        torch, actor = _load_actor(SHUFFLE_ACTOR, 52, 12)
        m = np.array(H0_TO_H1)
        lane = hc.CudaHumanoidLane(1)
        try:
            obs = lane.reset_policy(seed=0, commands=[0.75],
                                    phase_offsets=[0.0])

            def obs52(o58):
                o = o58[0]
                return np.concatenate([o[0:14][m], o[14:28][m],
                                       o[28:42][m], o[42:58]])[None, :]

            with torch.no_grad():
                for _t in range(400):
                    a12 = actor.deterministic(
                        torch.from_numpy(obs52(obs)).float()).numpy()[0]
                    a14 = np.zeros((1, 14), np.float32)
                    a14[0, m] = a12
                    obs, _r, done, _diag = lane.step_policy(a14)
                    if done.any():
                        break
            gp = lane.gate_proxy()[0]
            total = int(gp["qualified_left"]) + int(gp["qualified_right"])
            self.assertLessEqual(total, 2,
                                 "shuffle actor gamed the gate proxy")
            print(f"gate_proxy shuffle: qualified total={total} "
                  f"(judge scores shuffling 0)", file=sys.stderr)
        finally:
            lane.close()


class GateTerminationTests(unittest.TestCase):
    """dwc1_set_gate_termination: OPT-IN judge-aligned death rules.

    Both knobs default 0 = OFF (bit-identity with knobs off is covered by
    the fingerprint protocol). NOTE: the coordinator's suggested "BC
    stepper survives the deadline" pairing is not realizable with the
    existing captured actors -- the BC-init humanoid actor falls at policy
    step ~42 with ZERO gate-qualified swings when actually simulated (it
    is an imitation warm start, not a walker) -- so the survives-side is
    pinned on the only judge-ACCEPTED stepper that exists, the duck
    walking actor, and the standing duck provides the deterministic
    deadline-death control on the same robot.
    """

    def test_shuffle_actor_dies_at_the_deadline(self):
        if not SHUFFLE_ACTOR.is_file():
            raise unittest.SkipTest(f"missing {SHUFFLE_ACTOR}")
        from walk.env import humanoid_cuda_lane as hc
        torch, actor = _load_actor(SHUFFLE_ACTOR, 52, 12)
        m = np.array(H0_TO_H1)
        lane = hc.CudaHumanoidLane(1)
        try:
            # 0.6 s deadline (300 ticks = 30 policy steps): before the
            # shuffle actor's natural fall (~step 54), so the death must
            # come from the gate rule, not the tilt.
            lane.set_gate_termination(first_deadline_ticks=300)
            obs = lane.reset_policy(seed=0, commands=[0.75],
                                    phase_offsets=[0.0])
            died_at = None
            with torch.no_grad():
                for t in range(80):
                    a12 = actor.deterministic(torch.from_numpy(
                        np.concatenate([obs[0][0:14][m], obs[0][14:28][m],
                                        obs[0][28:42][m], obs[0][42:58]]
                                       )[None, :]).float()).numpy()[0]
                    a14 = np.zeros((1, 14), np.float32)
                    a14[0, m] = a12
                    obs, _r, done, _diag = lane.step_policy(a14)
                    if done.any():
                        died_at = t
                        break
            gp = lane.gate_proxy()[0]
            self.assertEqual(died_at, 29, "deadline fires at tick 300 "
                             "= policy step 30 (0-indexed 29)")
            self.assertEqual(int(gp["termination_reason"]), 2,
                             "DWC1_TERM_GATE_DEADLINE")
            print(f"gate_termination shuffle: died at step {died_at} "
                  f"reason=gate_deadline", file=sys.stderr)
        finally:
            lane.close()

    def test_walking_actor_survives_the_judge_deadline(self):
        torch, actor = _load_actor(DUCK_ACTOR, 58, 14)
        lane = CudaDuckLane(1)
        try:
            # the judge's own first-step clause: 2.5 s (1250 ticks), plus
            # the alternation cap at its tightest (1) -- the accepted
            # walker has zero violations and must sail through both.
            lane.set_gate_termination(first_deadline_ticks=1250,
                                      max_alternation_violations=1)
            obs = lane.reset_policy(seed=0, commands=[0.15],
                                    phase_offsets=[0.0])
            with torch.no_grad():
                for t in range(200):        # 4 s >> 2.5 s deadline
                    a = actor.deterministic(
                        torch.from_numpy(obs).float()).numpy()
                    obs, _r, done, _diag = lane.step_policy(a)
                    self.assertFalse(bool(done.any()),
                                     f"walker killed at step {t}")
            gp = lane.gate_proxy()[0]
            self.assertGreaterEqual(
                int(gp["qualified_left"]) + int(gp["qualified_right"]), 15)
            # deterministic deadline control on the same lane: standing
            # produces no qualified swing and must die AT the deadline.
            lane.reset_policy(commands=[0.15], phase_offsets=[0.0])
            died_at = None
            for t in range(200):
                _o, _r, done, _d = lane.step_policy(
                    np.zeros((1, 14), np.float32))
                if done.any():
                    died_at = t
                    break
            gp = lane.gate_proxy()[0]
            self.assertEqual(died_at, 124, "1250-tick deadline = step 125")
            self.assertEqual(int(gp["termination_reason"]), 2)
            print(f"gate_termination duck: walker survived 200 steps; "
                  f"standing died at step {died_at} reason=gate_deadline",
                  file=sys.stderr)
        finally:
            lane.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
