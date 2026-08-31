"""Parity gates: fp32 serial duck_cuda build vs the f64 CPU oracle lane.

Run: .venv/bin/python -B experimental/duck_cuda/tests/test_serial_parity.py

PARITY CONTRACT (see also include/duck_cuda.h). All state and arithmetic in
the CUDA lane are float32; divergence from the f64 integrated_duck_v1 oracle
grows with simulated time and contact activity. The enforced gates:

 (a) home-hold, 500 ticks (1 s): |root position - CPU| < 2 mm and tilt
     difference < 1 deg (measured headroom is ~4 orders of magnitude);
 (b) seeded random actions (clip 0.5, 10-tick target holds), 300 ticks:
     bounded divergence at every policy-step boundary (root < 5 mm, joints
     < 0.05 rad), no NaN, unit root quaternions, no ground penetration
     beyond 5 mm (whole-sole height >= -5 mm);
 (c) every recorded fault-corpus state (runs/flat-001-crashed/faults) steps
     10 ticks without solver fault or NaN.

Also enforced: the generated duck_model.h matches a fresh regeneration from
the pinned fixtures (drift test), bit-identical determinism between two
scenes within one build, FlatFloorDuckEnv holding home for 100 policy steps
on the serial lane, and the device policy path (dwc1_step_policy: in-kernel
obs + reward + termination) matching FlatFloorDuckEnv side by side for
200 policy steps within obs < 1e-4 / reward < 1e-3 / identical done flags
(measured: bit-identical). The throughput test reports serial ticks/s per
core (approximates per-thread GPU cost).
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DUCK_CUDA = ROOT / "experimental" / "duck_cuda"
FAULTS = ROOT / "runs" / "flat-001-crashed" / "faults"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "walk" / "env"))

from walk.env import cuda_lane  # noqa: E402
from walk.env.cuda_lane import CudaDuckLane  # noqa: E402


def _random_targets(lane, steps: int, seed: int = 1234) -> np.ndarray:
    """Seeded per-policy-step random targets, actions clipped to +-0.5."""
    rng = np.random.default_rng(seed)
    home, lim = lane.home_joint_q, lane.joint_limits
    out = np.zeros((steps, 14))
    for s in range(steps):
        a = np.clip(rng.normal(0.0, 0.5, 14), -0.5, 0.5)
        out[s] = np.clip(home + 0.25 * a, lim[:, 0], lim[:, 1])
    return out


def _tilt_deg(q: np.ndarray) -> float:
    up = 1.0 - 2.0 * (q[3] * q[3] + q[4] * q[4])
    return math.degrees(math.acos(min(1.0, max(-1.0, float(up)))))


class SerialParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import native_lane
        cls.native_lane = native_lane
        cls.cpu_available = Path(native_lane.build_library()).is_file()

    # -- duck_model.h drift -------------------------------------------------
    def test_generated_model_header_matches_fixtures(self):
        spec = importlib.util.spec_from_file_location(
            "generate_model", DUCK_CUDA / "tools" / "generate_model.py")
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        native, fixture, cm = gen.load_fixture()
        fresh = gen.emit(native, fixture, cm)
        committed = (DUCK_CUDA / "include" / "duck_model.h").read_text()
        self.assertEqual(fresh, committed,
                         "duck_model.h drifted from the pinned fixtures; "
                         "regenerate with tools/generate_model.py")

    # -- gate (a): home hold --------------------------------------------------
    def test_home_hold_500_ticks_parity(self):
        cpu = self.native_lane.NativeDuckLane(1)
        gpu = CudaDuckLane(1)
        try:
            home = cpu.home_joint_q[None, :]
            for t in range(500):
                rc, diag = cpu.tick(home)
                self.assertEqual(rc, 0, (t, diag))
                rc, diag = gpu.tick(home)
                self.assertEqual(rc, 0, (t, diag))
            qc = cpu.read().q[0]
            qg = gpu.read().q[0]
            root_mm = 1000.0 * np.abs(qg[:3] - qc[:3]).max()
            tilt = abs(_tilt_deg(qg) - _tilt_deg(qc))
            self.assertLess(root_mm, 2.0, "root drift gate (2 mm)")
            self.assertLess(tilt, 1.0, "tilt gate (1 deg)")
            print(f"home_hold_parity root_drift={root_mm:.2e} mm "
                  f"tilt_diff={tilt:.2e} deg", file=sys.stderr)
        finally:
            cpu.close()
            gpu.close()

    # -- gate (b): random actions ----------------------------------------------
    def test_random_actions_300_ticks_parity(self):
        cpu = self.native_lane.NativeDuckLane(1)
        gpu = CudaDuckLane(1)
        try:
            targets = _random_targets(cpu, 30)
            worst_root = worst_joint = worst_pen = 0.0
            for s in range(30):
                for _ in range(10):
                    rc, diag = cpu.tick(targets[s][None, :])
                    self.assertEqual(rc, 0, (s, diag))
                    rc, diag = gpu.tick(targets[s][None, :])
                    self.assertEqual(rc, 0, (s, diag))
                sc, sg = cpu.read(), gpu.read()
                self.assertTrue(sg.finite().all(), s)
                self.assertAlmostEqual(
                    float(np.sum(np.square(sg.q[0, 3:7]))), 1.0, places=5)
                worst_root = max(worst_root,
                                 float(np.abs(sg.q[0, :3] - sc.q[0, :3]).max()))
                worst_joint = max(worst_joint,
                                  float(np.abs(sg.q[0, 7:] - sc.q[0, 7:]).max()))
                worst_pen = max(worst_pen, -float(sg.sole_height.min()))
            self.assertLess(worst_root, 5e-3, "root divergence gate (5 mm)")
            self.assertLess(worst_joint, 0.05, "joint divergence gate (0.05 rad)")
            self.assertLessEqual(worst_pen, 5e-3, "penetration gate (5 mm)")
            print(f"random_action_parity root={1000*worst_root:.2e} mm "
                  f"joint={worst_joint:.2e} rad pen={1000*worst_pen:.3f} mm",
                  file=sys.stderr)
        finally:
            cpu.close()
            gpu.close()

    # -- gate (c): fault corpus -------------------------------------------------
    def test_fault_corpus_states_step_10_ticks(self):
        if not FAULTS.is_dir():
            raise unittest.SkipTest("fault corpus not present: " + str(FAULTS))
        files = sorted(FAULTS.glob("*.json"))
        self.assertTrue(files, "empty fault corpus")
        lane = CudaDuckLane(1)
        try:
            failures = []
            for f in files:
                a = json.loads(f.read_text())
                st = a["state"]
                lane.set_state(0, st["qpos"], st["velocity"], st["warm_force"],
                               cache=st["pre_contact_cache"][:2],
                               count=st["step_count"])
                ok = True
                for _ in range(10):
                    rc, diag = lane.tick(
                        np.asarray(a["effective_targets"])[None, :])
                    if rc or diag[0]["native_status"]:
                        ok = False
                        break
                state = lane.read()
                if not (ok and state.finite().all()):
                    failures.append((f.name, rc, diag[0]))
            self.assertEqual(failures, [],
                             f"{len(failures)}/{len(files)} corpus states faulted")
            print(f"fault_corpus {len(files)}/{len(files)} states stepped "
                  "10 ticks clean", file=sys.stderr)
        finally:
            lane.close()

    # -- fast path: one dwc1_step call per policy step ---------------------------
    def test_tick_block_bit_identical_to_tick_loop(self):
        # tick_block(t, 10) must produce exactly the state of 10x tick(t):
        # dwc1_step loops ticks inside one call (one kernel launch on device),
        # so the arithmetic per tick is the same code path either way.
        a, b = CudaDuckLane(2), CudaDuckLane(2)
        try:
            targets = _random_targets(a, 12, seed=21)
            for s in range(12):
                batch = np.tile(targets[s], (2, 1))
                rc, diag = a.tick_block(batch, 10)
                self.assertEqual(rc, 0, (s, diag))
                self.assertTrue(all(d["ticks"] == 10 for d in diag))
                for _ in range(10):
                    self.assertEqual(b.tick(batch)[0], 0, s)
            sa, sb = a.read(), b.read()
            for name in ["q", "v", "body_state", "sole_height", "foot_contact",
                         "count"]:
                self.assertEqual(getattr(sa, name).tobytes(),
                                 getattr(sb, name).tobytes(), name)
            for e in range(2):
                da, db = a.state_dump(e), b.state_dump(e)
                # contact_ticks legitimately differ: the block call counted
                # all 10 ticks, the loop lane's counter covers only its most
                # recent 1-tick step call. Everything else is bit-identical.
                da.pop("contact_ticks")
                db.pop("contact_ticks")
                self.assertEqual(json.dumps(da, sort_keys=True),
                                 json.dumps(db, sort_keys=True), e)
        finally:
            a.close()
            b.close()

    def test_contact_ticks_match_per_tick_accumulation(self):
        # read().contact_ticks after tick_block(t, 10) must equal what the
        # slow path accumulates by reading foot_contact after every tick.
        # Covers the settling phase (feet start above the floor: partial
        # counts), steady standing (10/10), and a random-action sequence.
        for seed, label in [(None, "home_hold"), (99, "random_action")]:
            a, b = CudaDuckLane(3), CudaDuckLane(3)
            try:
                if seed is None:
                    targets = np.tile(a.home_joint_q, (20, 1))
                else:
                    targets = _random_targets(a, 20, seed=seed)
                saw_partial = saw_full = False
                for s in range(20):
                    batch = np.tile(targets[s], (3, 1))
                    rc, diag = a.tick_block(batch, 10)
                    self.assertEqual(rc, 0, (label, s, diag))
                    accumulated = np.zeros((3, 2), np.int32)
                    for _ in range(10):
                        self.assertEqual(b.tick(batch)[0], 0, (label, s))
                        accumulated += b.read().foot_contact
                    ticks = a.read().contact_ticks
                    np.testing.assert_array_equal(ticks, accumulated,
                                                  f"{label} step {s}")
                    saw_partial |= bool((ticks % 10 != 0).any())
                    saw_full |= bool((ticks == 10).any())
                self.assertTrue(saw_full, label)
                if label == "home_hold":
                    self.assertTrue(saw_partial,
                                    "settling should yield partial counts")
            finally:
                a.close()
                b.close()

    # -- device policy path vs the python env (obs/reward contract) --------------
    def test_step_policy_matches_python_env(self):
        # walk/env/flat.py + walk/env/reward.py are the contract: run
        # FlatFloorDuckEnv (numpy f64 obs/reward over this same serial lane
        # build) and dwc1_step_policy side by side for 200 deterministic
        # policy steps with a masked reset in the middle. The device policy
        # chain runs in f64 mirroring numpy operation for operation, so the
        # physics trajectories are bit-identical and obs/reward/done should
        # agree far inside the gates (obs < 1e-4, reward < 1e-3, done equal).
        from walk.env.flat import FlatFloorDuckEnv
        E = 4
        env = FlatFloorDuckEnv(
            environments=E, seed=0,
            lane_factory=lambda n, off: CudaDuckLane(n, joint_offsets=off))
        dev = CudaDuckLane(E)
        try:
            obs_py = env.reset()                # episode-2 command draws
            dev.reset_policy(seed=0)            # episode 1
            obs_dev = dev.reset_policy()        # episode 2
            np.testing.assert_array_equal(obs_py, obs_dev, "reset observations")
            self.assertTrue(np.array_equal(env.command.astype(np.float32),
                                           obs_dev[:, 51]))
            rng = np.random.default_rng(0)
            worst_obs = worst_rew = 0.0
            resets = 0
            for t in range(200):
                a = np.clip(rng.normal(0, 0.4, (E, 14)), -1, 1).astype(np.float32)
                o_py, r_py, d_py, _info = env.step(a)
                o_dev, r_dev, d_dev, diag = dev.step_policy(a)
                self.assertEqual((diag["status"] != 0).sum(), 0, t)
                worst_obs = max(worst_obs, float(np.abs(o_py - o_dev).max()))
                worst_rew = max(worst_rew, float(np.abs(r_py - r_dev).max()))
                np.testing.assert_array_equal(d_py, d_dev, f"done flags @ {t}")
                if d_py.any() and resets < 3:   # trainer-style masked reset
                    resets += 1
                    o_py = env.reset(mask=d_py)
                    o_dev = dev.reset_policy(mask=d_dev)
                    self.assertLess(float(np.abs(o_py - o_dev).max()), 1e-4,
                                    f"post-reset obs @ {t}")
            self.assertGreater(resets, 0, "action sequence never terminated")
            self.assertLess(worst_obs, 1e-4, "obs parity gate")
            self.assertLess(worst_rew, 1e-3, "reward parity gate")
            print(f"step_policy_parity obs={worst_obs:.3e} reward={worst_rew:.3e} "
                  f"masked_resets={resets}", file=sys.stderr)
        finally:
            env.close()
            dev.close()

    # -- reward v9: flickered-stance credit reset, genuinely exercised ------------
    def test_step_policy_parity_covers_flickered_stance(self):
        # reward.py v9: stance credit accrues only on FULL-contact steps
        # (contact_ticks == 10); tick-scale flicker inside a stance resets it
        # and can flip a later touchdown's stance_ok qualification gate. The
        # main parity test's trajectories never hit that path, so this one
        # drives a marching pattern + seeded noise that (a) produces many
        # flickered-stance steps and (b) was verified to DISCRIMINATE the
        # rules: a build with the pre-v9 rule (stance += dt on any contact)
        # diverges from the python env by 1.5 (a flipped qualified-step
        # bonus) at step 22 of this exact sequence. Runtime vacuity guards
        # below keep the sequence honest if the physics ever drifts.
        from walk.env.flat import FlatFloorDuckEnv
        E, T, dt = 4, 150, 0.02
        rng = np.random.default_rng(2)
        actions = np.zeros((T, E, 14), np.float32)
        for t in range(T):
            ph = 2 * np.pi * t / 20.0
            base = np.zeros(14)
            base[[2, 3, 4]] = np.array([.9, -.9, .5]) * np.sin(ph)
            base[[11, 12, 13]] = np.array([.9, -.9, .5]) * np.sin(ph + np.pi)
            base[[0, 9]] = .3 * np.sin(ph / 2)
            actions[t] = np.clip(base[None, :] + rng.normal(0, .15, (E, 14)),
                                 -1, 1).astype(np.float32)
        env = FlatFloorDuckEnv(
            environments=E, seed=0,
            lane_factory=lambda n, off: CudaDuckLane(n, joint_offsets=off))
        dev = CudaDuckLane(E)
        try:
            env.reset()
            dev.reset_policy(seed=0)
            dev.reset_policy()
            worst_obs = worst_rew = 0.0
            flickered = 0                       # contact at both boundaries, ct < 10
            gate_divergent_touchdowns = 0       # v6-vs-v9 stance_ok differs
            prevc = np.zeros((E, 2), bool)
            stance_v9 = np.zeros((E, 2))
            stance_v6 = np.zeros((E, 2))
            pre_ok_differs = np.zeros((E, 2), bool)  # at the last liftoff
            for t in range(T):
                o_py, r_py, d_py, _info = env.step(actions[t])
                o_dev, r_dev, d_dev, diag = dev.step_policy(actions[t])
                self.assertEqual((diag["status"] != 0).sum(), 0, t)
                worst_obs = max(worst_obs, float(np.abs(o_py - o_dev).max()))
                worst_rew = max(worst_rew, float(np.abs(r_py - r_dev).max()))
                np.testing.assert_array_equal(d_py, d_dev, f"done flags @ {t}")
                st = dev.read()
                c, ct = st.foot_contact, st.contact_ticks
                flickered += int((prevc & c & (ct < 10)).any())
                liftoff = prevc & ~c
                touchdown = ~prevc & c
                gate_divergent_touchdowns += int((touchdown & pre_ok_differs).any())
                pre_ok_differs = np.where(
                    liftoff, (stance_v6 >= 0.06) != (stance_v9 >= 0.06),
                    pre_ok_differs)
                stance_v9 = np.where(c & (ct >= 10), stance_v9 + dt, 0.0)
                stance_v6 = np.where(c, stance_v6 + dt, 0.0)
                prevc = c.copy()
            # vacuity guards: the v9 path must actually be exercised
            self.assertGreater(flickered, 20, "no flickered-stance steps")
            self.assertGreater(gate_divergent_touchdowns, 0,
                               "no touchdown whose stance_ok gate depends on v9")
            self.assertLess(worst_obs, 1e-4, "obs parity gate")
            self.assertLess(worst_rew, 1e-3, "reward parity gate")
            print(f"flickered_stance_parity obs={worst_obs:.3e} "
                  f"reward={worst_rew:.3e} flickered_steps={flickered} "
                  f"gate_divergent_touchdowns={gate_divergent_touchdowns}",
                  file=sys.stderr)
        finally:
            env.close()
            dev.close()

    # -- determinism -------------------------------------------------------------
    def test_two_runs_bit_identical(self):
        a, b = CudaDuckLane(2), CudaDuckLane(2)
        try:
            targets = _random_targets(a, 10, seed=7)
            for s in range(10):
                batch = np.tile(targets[s], (2, 1))
                for _ in range(10):
                    self.assertEqual(a.tick(batch)[0], 0)
                    self.assertEqual(b.tick(batch)[0], 0)
            sa, sb = a.read(), b.read()
            for name in ["q", "v", "body_state", "sole_height"]:
                self.assertEqual(getattr(sa, name).tobytes(),
                                 getattr(sb, name).tobytes(), name)
            self.assertEqual(json.dumps(a.state_dump(0), sort_keys=True),
                             json.dumps(b.state_dump(0), sort_keys=True))
        finally:
            a.close()
            b.close()

    # -- FlatFloorDuckEnv on the serial lane ---------------------------------------
    def test_flat_env_holds_home_100_policy_steps(self):
        from walk.env.flat import FlatFloorDuckEnv
        env = FlatFloorDuckEnv(
            environments=4, seed=0,
            lane_factory=lambda E, offsets: CudaDuckLane(E, joint_offsets=offsets))
        try:
            obs = env.reset()
            self.assertEqual(obs.shape, (4, 58))
            done = np.zeros(4, bool)
            for t in range(100):
                obs, reward, done, info = env.step(np.zeros((4, 14), np.float32))
                self.assertFalse(done.any(), (t, info))
                self.assertTrue(np.isfinite(obs).all() and np.isfinite(reward).all())
            state = env._lane.read()
            self.assertTrue(
                (state.q[:, 2] >= 0.7 * env._lane.home_root_height).all())
            self.assertTrue(info["foot_contact"].all())
        finally:
            env.close()

    # -- throughput report ------------------------------------------------------------
    def test_serial_throughput_report(self):
        E = 64
        lane = CudaDuckLane(E)
        try:
            home = np.tile(lane.home_joint_q, (E, 1))
            for _ in range(20):     # settle into contact-active standing
                self.assertEqual(lane.tick(home)[0], 0)
            n = 200
            t0 = time.perf_counter()
            for _ in range(n):
                self.assertEqual(lane.tick(home)[0], 0)
            elapsed = time.perf_counter() - t0
            rate = E * n / elapsed
            print(f"serial_throughput={rate:.0f} ticks/s/core (E={E})",
                  file=sys.stderr)
            self.assertGreater(rate, 5000.0)
        finally:
            lane.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
