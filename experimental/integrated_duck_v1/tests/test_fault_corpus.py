"""Solver-fault corpus gates for civ1 (workstream A); numpy + ctypes only.

Replays the training fault corpus (schema duckgridwalk.solver_fault/1) from
runs/flat-001-crashed/faults. Each artifact is the full pre-step state of the
policy-step tick whose contact solve returned CIV1_NO_CONVERGENCE.

These faults are warm-start dependent: a fresh scene stepped from the same
qpos/velocity mostly converges, because idv1 seeds contact impulses from the
previous tick's manifold cache. The converter below therefore rebuilds the
EXACT failing civ1 problem: the av2 side (mass, smooth velocity, joint rows,
per-row warm impulses) is restored through an av2 snapshot carrying the
artifact's qpos/velocity/joint warm force, and the contact rows replicate
integrated_duck_v1.cpp's lowering bit-for-bit from the artifact's
current_geometry manifolds, including the cache warm-impulse matching
(feature id + point distance + normal agreement, float32 tangent rotation).

Reproduction is confirmed on the CURRENT library: all repairs in
coupled_impulse_v1.cpp are gated behind the stall detector, which cannot fire
before the sweep-63 window, so a 63-sweep budget is provably the pre-repair
ordinary path and must fail on every artifact. The acceptance gate then
requires >=95% of the corpus to converge with valid certificates within the
training budget (4096 sweeps, tolerance 1e-8); failures are listed by name.
"""
import ctypes as C
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FAULTS = ROOT / "runs" / "flat-001-crashed" / "faults"
D = C.POINTER(C.c_double)


class Contact(C.Structure):
    _fields_ = [("first_row", C.c_uint32), ("friction", C.c_double)]


class Problem(C.Structure):
    _fields_ = [("dofs", C.c_uint32), ("rows", C.c_uint32), ("contacts", C.c_uint32),
                ("max_iterations", C.c_uint32), ("impulse_tolerance", C.c_double),
                ("mass", D), ("smooth_velocity", D), ("jacobian", D), ("target", D),
                ("regularizer", D), ("lower", D), ("upper", D), ("warm", D),
                ("contact", C.POINTER(Contact))]


class Result(C.Structure):
    _fields_ = [("velocity", D), ("impulse", D), ("iterations", C.c_uint32),
                ("joint_residual", C.c_double), ("normal_residual", C.c_double),
                ("tangent_residual", C.c_double), ("momentum_residual", C.c_double)]


def reconstruct(artifact, av, av2lib, fixture, mu_pairs, pairs):
    """Rebuild the exact civ1 problem of the artifact's failing tick."""
    st = artifact["state"]
    scene = av.Scene(av2lib, fixture, np.array([st["qpos"]]), np.array([st["velocity"]]),
                     lim=av.limits(fixture), gravity=[[0.0, 0.0, -9.81]])
    try:
        snap = scene.capture()
        snap.q[0] = st["qpos"]
        snap.v[0] = st["velocity"]
        snap.warm[0] = st["warm_force"]
        snap.time[0] = st["time_s"]
        snap.count[0] = st["step_count"]
        rc, stage = scene.restore(snap)
        assert rc == 0 and scene.commit(stage) == 0
        _, pre = scene.pre(dt=artifact["dt"], target=[artifact["effective_targets"]])
        N, J = fixture.N, fixture.J
        rows_g = [pre["G"][0][r].tolist() for r in range(3 * J)]
        target = list(pre["target"][0])
        reg = list(pre["R"][0])
        lower = list(pre["lower"][0])
        upper = list(pre["upper"][0])
        warm = list(pre["warm"][0])
        pose, Js = pre["pose"][0], pre["J"][0]
        contacts = []
        for pair_idx, m in enumerate(st["current_geometry"]):
            mu = float(np.float32(mu_pairs[pair_idx]))
            prev = st["pre_contact_cache"][pair_idx]
            body_a, body_b = pairs[pair_idx]
            normal_ok = float(np.array(m["normal"]) @ np.array(prev["normal"])) > .98
            for x in m["points"][:m["count"]]:
                w = [0.0, 0.0, 0.0]
                if normal_ok:
                    for y in prev["points"][:prev["count"]]:
                        dist = sum((float(a) - float(b)) ** 2
                                   for a, b in zip(x["point"], y["point"]))
                        if x["feature"] == y["feature"] and dist < .0004:
                            t = (np.array(prev["tangent1"], np.float32)
                                 * np.float32(y["tangent_impulse"][0])
                                 + np.array(prev["tangent2"], np.float32)
                                 * np.float32(y["tangent_impulse"][1]))
                            w[0] = float(np.float32(y["normal_impulse"]))
                            w[1] = float(t.astype(np.float64) @ np.array(m["tangent1"]))
                            w[2] = float(t.astype(np.float64) @ np.array(m["tangent2"]))
                            break
                contacts.append((len(target), mu))
                for a, direction in enumerate([m["normal"], m["tangent1"], m["tangent2"]]):
                    out = [0.0] * N
                    for body, sign in ((body_a, -1.0), (body_b, 1.0)):
                        px, py, pz = pose[body][:3]
                        arm = [x["point"][0]-px, x["point"][1]-py, x["point"][2]-pz]
                        torque = [arm[1]*direction[2]-arm[2]*direction[1],
                                  arm[2]*direction[0]-arm[0]*direction[2],
                                  arm[0]*direction[1]-arm[1]*direction[0]]
                        for n in range(N):
                            for k in range(3):
                                out[n] += sign*(direction[k]*Js[body][k][n]
                                                + torque[k]*Js[body][3+k][n])
                    rows_g.append(out)
                    target.append(min(1.0, 0.2*max(0.0, x["depth"]-2e-6)/artifact["dt"])
                                  if a == 0 else 0.0)
                    reg.append(0.0)
                    lower.append(0.0 if a == 0 else -math.inf)
                    upper.append(math.inf)
                    warm.append(w[a])
        return dict(mass=pre["mass"][0].copy(), smooth=pre["smooth"][0].copy(),
                    g=np.array(rows_g), target=np.array(target), reg=np.array(reg),
                    lower=np.array(lower), upper=np.array(upper), warm=np.array(warm),
                    contacts=contacts, N=N)
    finally:
        scene.close()


class FaultCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FAULTS.is_dir():
            raise unittest.SkipTest("fault corpus not present: " + str(FAULTS))
        sys.path.insert(0, str(ROOT / "walk" / "env"))
        import native_lane
        cls.libpath = str(native_lane.build_library())
        cls.native = native_lane._native()
        cls.av = cls.native.av
        cls.lib = C.CDLL(cls.libpath)
        cls.lib.civ1_solve.argtypes = [C.POINTER(Problem), C.POINTER(Result)]
        cls.lib.civ1_solve.restype = C.c_int
        cls.files = sorted(FAULTS.glob("*.json"))
        cls.artifacts = [json.loads(f.read_text()) for f in cls.files]
        assert cls.artifacts, "empty fault corpus"
        for a in cls.artifacts:
            assert a["schema"] == "duckgridwalk.solver_fault/1"
            assert a["dt"] == 0.002 and a["tolerance"] == 1e-8
            assert a["max_iterations"] == 4096
        fixture = cls.av.duck()
        cm = cls.native.contact.Model(cls.native.contact.library(cls.libpath))
        pairs = [(p.body_a, p.body_b) for p in cm.pairs]
        cls.problems = [reconstruct(a, cls.av, cls.av.library(cls.libpath), fixture,
                                    cm.mu, pairs) for a in cls.artifacts]
        cls.fixture, cls.cm, cls.pairs = fixture, cm, pairs

    def civ1(self, prob, iterations, tolerance):
        n, r = prob["N"], len(prob["target"])
        values = [prob["mass"], prob["smooth"], prob["g"], prob["target"], prob["reg"],
                  prob["lower"], prob["upper"], prob["warm"]]
        arrays = [(C.c_double*np.asarray(v).size)(*np.asarray(v, float).ravel())
                  for v in values]
        cp = (Contact*len(prob["contacts"]))(*(Contact(*c) for c in prob["contacts"]))
        p = Problem(n, r, len(prob["contacts"]), iterations, tolerance, *arrays, cp)
        velocity = (C.c_double*n)()
        impulse = (C.c_double*r)()
        res = Result(velocity, impulse, 0, 0., 0., 0., 0.)
        code = self.lib.civ1_solve(C.byref(p), C.byref(res))
        return code, np.array(velocity), np.array(impulse), res

    def test_converter_matches_native_lane_stepping(self):
        # Faithfulness check: for fresh (zero-warm) state the direct civ1
        # reconstruction and a real E=1 idv1_step must agree exactly on
        # status and iteration count.
        lanelib = self.native.library(self.libpath)
        for a in self.artifacts[:3]:
            scene = self.native.Scene(lanelib, self.fixture,
                                      np.array([a["state"]["qpos"]]),
                                      np.array([a["state"]["velocity"]]),
                                      self.cm.shapes, self.cm.pairs,
                                      np.tile(self.cm.mu, (1, 1)))
            rc, diag = scene.step(dt=a["dt"], target=[a["effective_targets"]],
                                  max_iterations=4096, tolerance=1e-8)
            scene.close()
            fresh = dict(a)
            fresh["state"] = {**a["state"],
                              "warm_force": [0.0]*len(a["state"]["warm_force"]),
                              "time_s": 0.0, "step_count": 0}
            prob = reconstruct(fresh, self.av, self.av.library(self.libpath),
                               self.fixture, self.cm.mu, self.pairs)
            code, _, _, res = self.civ1(prob, 4096, 1e-8)
            self.assertEqual(code, diag[0]["native_status"])
            if code == 0:
                self.assertEqual(res.iterations, diag[0]["iterations"])

    def test_faults_reproduce_on_the_pure_ordinary_path(self):
        # All repairs are gated behind the stall detector (sweep-63 window at
        # the earliest): a 63-sweep budget is exactly the pre-repair sweep and
        # must fail on every recorded fault (they all needed > 4096 before).
        for f, prob in zip(self.files, self.problems):
            code, _, _, _ = self.civ1(prob, 63, 1e-8)
            self.assertEqual(code, 3, f.name)

    def test_corpus_converges_with_valid_certificates(self):
        failures, converged = [], 0
        for f, prob in zip(self.files, self.problems):
            code, v, impulse, res = self.civ1(prob, 4096, 1e-8)
            if code != 0:
                failures.append((f.name, code))
                continue
            converged += 1
            self.assertLessEqual(max(res.joint_residual, res.normal_residual,
                                     res.tangent_residual), 1e-8, f.name)
            self.assertLessEqual(res.momentum_residual, 1e-8, f.name)
            self.assertLessEqual(res.iterations, 4096, f.name)
            # independent physical checks in numpy
            momentum = prob["mass"] @ (v - prob["smooth"]) - prob["g"].T @ impulse
            self.assertLessEqual(np.abs(momentum).max(), 2e-9, f.name)
            for (r, mu) in prob["contacts"]:
                self.assertGreaterEqual(impulse[r], -1e-12, f.name)
                self.assertLessEqual(np.hypot(impulse[r+1], impulse[r+2]),
                                     mu*impulse[r] + 1e-8, f.name)
        total = len(self.problems)
        self.assertGreaterEqual(
            converged, math.ceil(.95*total),
            f"corpus convergence {converged}/{total}; failures: {failures}")
        print(f"fault_corpus_converged={converged}/{total} failures={failures}",
              file=sys.stderr)

    def test_repaired_library_survives_fresh_lane_batch_step(self):
        # End-to-end: one batched idv1_step over every artifact state (fresh
        # warm) must complete without any solver fault.
        E = len(self.artifacts)
        lanelib = self.native.library(self.libpath)
        scene = self.native.Scene(lanelib, self.fixture,
                                  np.array([a["state"]["qpos"] for a in self.artifacts]),
                                  np.array([a["state"]["velocity"] for a in self.artifacts]),
                                  self.cm.shapes, self.cm.pairs,
                                  np.tile(self.cm.mu, (E, 1)))
        rc, diag = scene.step(dt=0.002,
                              target=[a["effective_targets"] for a in self.artifacts],
                              max_iterations=4096, tolerance=1e-8)
        scene.close()
        self.assertEqual(rc, 0, [d for d in diag if d["native_status"]])
        for d in diag:
            self.assertEqual(d["native_status"], 0)
            self.assertLessEqual(max(d["joint_residual"], d["normal_residual"],
                                     d["tangent_residual"]), 1e-8)
            self.assertLessEqual(d["momentum_residual"], 1e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
