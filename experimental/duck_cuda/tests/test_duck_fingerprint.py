"""Duck (and humanoid, randomization OFF) trajectory FINGERPRINTS.

Run: .venv/bin/python -B experimental/duck_cuda/tests/test_duck_fingerprint.py
Re-pin (ONLY after an intentional, reviewed physics change):
     .venv/bin/python -B experimental/duck_cuda/tests/test_duck_fingerprint.py --repin

The bit-identity protocol every kernel/ABI change must satisfy: a fixed
seeded driver runs the serial duck build through (a) the raw tick path,
(b) the device policy path with randomization OFF, (c) the device policy
path with the ACCEPTED lineage's duck DR config (r_gravity absent), and
the humanoid build through (d) its policy path with randomization OFF;
every obs/reward/done/state byte stream is SHA-256 hashed and compared to
the committed pin in fixtures/trajectory_fingerprints.json. The pin was
recorded on the ABI-v6 build BEFORE the r_gravity/ABI-v7 change, so a
green run here is the proof that the change is bit-identical on every
path where the new feature is absent or neutral.
"""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "humanoid"))

from walk.env.cuda_lane import CudaDuckLane  # noqa: E402

PIN = Path(__file__).resolve().parent / "fixtures" / "trajectory_fingerprints.json"
# the accepted duck lineage's DR config (gpu/specs/continue-ff-short.json)
DUCK_DR = {"r_mass": 0.1, "r_friction": 0.2, "r_kp": 0.1, "r_damping": 0.2,
           "max_latency_steps": 1}


def _h(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _policy_fingerprint(lane, act_dim: int, steps: int, sigma: float,
                        clip: float, seed: int) -> str:
    rng = np.random.default_rng(seed)
    h = hashlib.sha256()
    for _ in range(steps):
        a = np.clip(rng.normal(0.0, sigma, (lane.E, act_dim)), -clip,
                    clip).astype(np.float32)
        obs, rew, done, diag = lane.step_policy(a)
        h.update(obs.tobytes()); h.update(rew.tobytes())
        h.update(done.astype(np.uint8).tobytes())
        h.update(np.ascontiguousarray(diag["status"]).tobytes())
        if done.any():
            lane.reset_policy(mask=done)
    st = lane.read()
    h.update(st.q.tobytes()); h.update(st.v.tobytes())
    return h.hexdigest()


def compute() -> dict:
    out = {}
    # (a) raw tick path: seeded random targets, 20 x 10-tick blocks
    lane = CudaDuckLane(3)
    try:
        rng = np.random.default_rng(1234)
        home, lim = lane.home_joint_q, lane.joint_limits
        h = hashlib.sha256()
        for _ in range(20):
            a = np.clip(rng.normal(0.0, 0.5, (3, 14)), -0.5, 0.5)
            t = np.clip(home + 0.25 * a, lim[:, 0], lim[:, 1])
            rc, _diag = lane.tick_block(t, 10)
            assert rc == 0
            st = lane.read()
            h.update(st.q.tobytes()); h.update(st.v.tobytes())
            h.update(st.sole_height.tobytes())
            h.update(st.contact_ticks.tobytes())
        out["duck_tick_path"] = h.hexdigest()
    finally:
        lane.close()
    # (b) duck policy path, randomization OFF
    lane = CudaDuckLane(4)
    try:
        lane.reset_policy(seed=5)
        out["duck_policy_off"] = _policy_fingerprint(lane, 14, 80, 0.4, 1.0, 17)
    finally:
        lane.close()
    # (c) duck policy path, ACCEPTED-lineage DR config (no r_gravity)
    lane = CudaDuckLane(4, randomization=dict(DUCK_DR))
    try:
        lane.reset_policy(seed=5)
        out["duck_policy_dr_v6"] = _policy_fingerprint(lane, 14, 80, 0.4, 1.0, 17)
    finally:
        lane.close()
    # (d) humanoid policy path, randomization OFF (gentle actions: no faults)
    from walk.env.humanoid_cuda_lane import CudaHumanoidLane
    lane = CudaHumanoidLane(4)
    try:
        lane.reset_policy(seed=5)
        out["humanoid_policy_off"] = _policy_fingerprint(lane, lane.J, 60, 0.1,
                                                         0.2, 17)
    finally:
        lane.close()
    return out


class FingerprintTests(unittest.TestCase):
    def test_trajectory_fingerprints_pinned(self):
        self.assertTrue(PIN.is_file(), f"missing pin {PIN}")
        want = json.loads(PIN.read_text())["fingerprints"]
        got = compute()
        for k in want:
            self.assertEqual(got[k], want[k], f"fingerprint drift on {k}")
        print("fingerprints:", json.dumps(got, indent=1), file=sys.stderr)


if __name__ == "__main__":
    if "--repin" in sys.argv:
        fp = compute()
        PIN.parent.mkdir(parents=True, exist_ok=True)
        PIN.write_text(json.dumps({"schema": "duckgridwalk.trajectory_fingerprints/1",
                                   "fingerprints": fp}, indent=1) + "\n")
        print(json.dumps(fp, indent=1))
    else:
        unittest.main(verbosity=2)
