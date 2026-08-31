"""Degenerate-contact repair gates for civ1 (workstream A); numpy + ctypes only.

Covers the two documented CIV1_NO_CONVERGENCE failure modes of the dense PGS
solver on flat-ground legged-robot contact states:
  (a) two nearly linearly dependent contact NORMAL rows through the response
      matrix (correlation >= 0.994): the scalar per-row sweep contracts too
      slowly and cannot certify within 4096 sweeps; the repaired solver must
      converge via the conditional two-normal joint solve (tangents fixed).
  (b) a rank-5 six-row two-contact block with a verified self-stress
      direction (nullspace of the contact-space operator): friction impulses
      redistribute along the null direction without reducing the residual
      certificate; the repaired solver must converge via the single bounded
      feasible null-direction boundary move at sweep 256.
Also asserts randomized well-posed problems still match an independent pure
python reference solve, so the repair changes nothing off the degenerate path.
"""
import ctypes as C
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
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


def correlated_normal_pair(correlation=0.998):
    """Two contacts whose normal rows correlate through K; tangents decouple."""
    n = 6
    s = math.sqrt(1.0 - correlation * correlation)
    g = np.zeros((6, n))
    g[0, 0] = 1.0                                  # contact 1 normal
    g[1, 2] = 1.0
    g[2, 3] = 1.0
    g[3, 0] = correlation                          # contact 2 normal, correlated
    g[3, 1] = s
    g[4, 4] = 1.0
    g[5, 5] = 1.0
    reference = np.array([100.0, 0.0, 0.0, 50.0, 0.0, 0.0])
    target = g @ g.T @ reference                   # mass = I -> K = G G^T
    lower = np.array([0.0, -math.inf, -math.inf, 0.0, -math.inf, -math.inf])
    return g, target, lower, reference


# Exact instance (found by randomized search against the pre-repair solver):
# six contact rows confined to a five dimensional span -> exactly rank-1
# deficient with a self-stress direction mixing the friction rows of both
# contacts. The pre-repair solver certifies only down to ~1e-5 within 4096
# sweeps on this block; below that it redistributes without converging.
SELF_STRESS_G = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.639326723198958, -0.13916030612762617, -0.7547733294981832,
     0.04704222869268952, 0.0],
    [0.0, 0.23285498548077302, -0.15431833266442727, -0.6070748612535706,
     0.7439250773931326, 0.0],
    [-0.30553248095394553, 1.3111593509190798, 0.13733997897607872,
     -0.030368335255533622, -0.02073209268743956, 0.0],
    [0.0, 0.10354701263540472, -0.23333780228100298, -0.7948419445463721,
     -0.550506829558632, 0.0],
    [0.0, -0.37423190305759063, -0.22515471238603135, -0.08115925598080423,
     0.8959179724689642, 0.0]])
SELF_STRESS_SMOOTH = np.array([
    -2.092013457546283, -2.041964579181469, -0.02486908645926548,
    -0.5004080281686283, -1.5369733417895965, -0.1749203912442473])
SELF_STRESS_MU = 0.8


class NullnormRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        override = os.environ.get("COUPLED_IMPULSE_LIBRARY")
        if override:
            cls.path = Path(override).resolve()
        else:
            cls.directory = tempfile.TemporaryDirectory(prefix="independent-nullnorm-repair-")
            cls.addClassCleanup(cls.directory.cleanup)
            cls.path = Path(cls.directory.name)/"libcoupled_impulse.dylib"
            command = ["clang++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                       "-dynamiclib", "-I", str(ROOT/"experimental/integrated_duck_v1/include"),
                       str(ROOT/"experimental/integrated_duck_v1/src/coupled_impulse_v1.cpp"),
                       "-o", str(cls.path)]
            compiled = subprocess.run(command, capture_output=True, text=True)
            if compiled.returncode:
                raise AssertionError((command, compiled.returncode, compiled.stdout, compiled.stderr))
            print("independent_compile_command=" + repr(command), file=sys.stderr)
        cls.library_sha = hashlib.sha256(cls.path.read_bytes()).hexdigest()
        print("independent_library=" + str(cls.path) + " sha256=" + cls.library_sha, file=sys.stderr)
        cls.lib = C.CDLL(str(cls.path))
        cls.lib.civ1_solve.argtypes = [C.POINTER(Problem), C.POINTER(Result)]
        cls.lib.civ1_solve.restype = C.c_int

    @classmethod
    def tearDownClass(cls):
        assert hashlib.sha256(cls.path.read_bytes()).hexdigest() == cls.library_sha
        assert "mujoco" not in sys.modules

    def problem(self, mass, velocity, jacobian, target=None, regularizer=None, lower=None,
                upper=None, contacts=(), warm=None, iterations=16384, tolerance=1e-12):
        mass = np.asarray(mass, dtype=float)
        n = len(velocity)
        jacobian = np.asarray(jacobian, dtype=float).reshape(-1, n)
        r = len(jacobian)
        values = [mass, velocity, jacobian, np.zeros(r) if target is None else target,
                  np.zeros(r) if regularizer is None else regularizer,
                  np.full(r, -math.inf) if lower is None else lower,
                  np.full(r, math.inf) if upper is None else upper,
                  np.zeros(r) if warm is None else warm]
        arrays = [(C.c_double*np.asarray(v).size)(*np.asarray(v).ravel()) for v in values]
        cp = (Contact*len(contacts))(*(Contact(*c) for c in contacts))
        p = Problem(n, r, len(contacts), iterations, tolerance, *arrays, cp)
        p._keepalive = arrays, cp
        return p

    def solve(self, p, expected=0):
        velocity = (C.c_double*p.dofs)(*([12345.25]*p.dofs))
        impulse = (C.c_double*p.rows)(*([-9876.5]*p.rows))
        result = Result(velocity, impulse, 987, 111., 222., 333., 444.)
        before = bytes(velocity), bytes(impulse), bytes(result)
        code = self.lib.civ1_solve(C.byref(p), C.byref(result))
        self.assertEqual(code, expected)
        if expected:
            self.assertEqual((bytes(velocity), bytes(impulse), bytes(result)), before)
        else:
            self.assertLessEqual(max(result.joint_residual, result.normal_residual,
                                     result.tangent_residual), p.impulse_tolerance)
            self.assertLessEqual(result.momentum_residual, 1e-8)
        return np.array(velocity), np.array(impulse), result

    def check_contact(self, mass, g, smooth, impulse, velocity, first, mu, target=None):
        target = np.zeros(len(g)) if target is None else np.asarray(target)
        residual = g@velocity-target
        jn, jt = impulse[first], impulse[first+1:first+3]
        slip = residual[first+1:first+3]
        self.assertGreaterEqual(jn, -1e-12)
        self.assertGreaterEqual(residual[first], -2e-9)
        self.assertLessEqual(abs(jn*residual[first]), 2e-9)
        self.assertLessEqual(np.linalg.norm(jt), mu*jn+2e-10)
        if np.linalg.norm(slip) > 1e-6 and mu*jn > 1e-10:
            self.assertAlmostEqual(np.linalg.norm(jt), mu*jn, delta=2e-10)
            multiplier = -float(slip@jt)/float(jt@jt)
            self.assertGreater(multiplier, 0.)
            self.assertLessEqual(np.linalg.norm(slip+multiplier*jt), 2e-9)
        np.testing.assert_allclose(mass@(velocity-smooth), g.T@impulse, atol=2e-10, rtol=0)

    def test_original_scalar_sweep_stalls_on_correlated_normal_pair(self):
        # Exact pure-python replica of the pre-repair per-row sweep on the
        # correlated pair. Tangent rows use disjoint dofs and zero targets, so
        # they stay identically zero through every disk solve and the original
        # algorithm reduces to projected scalar Gauss-Seidel on the two normal
        # rows: this replica IS the original iteration for this problem.
        g, target, _, _ = correlated_normal_pair()
        k = (g @ g.T)[np.ix_([0, 3], [0, 3])]
        correlation = k[0, 1] / math.sqrt(k[0, 0] * k[1, 1])
        self.assertGreaterEqual(correlation, 0.994)
        base = -target[[0, 3]]
        lam = np.zeros(2)
        for _ in range(4096):
            for i in range(2):
                lam[i] = max(0.0, lam[i] - (base[i] + k[i] @ lam) / k[i, i])
            residual = base + k @ lam
            error = 0.0
            for i in range(2):
                correction = residual[i] / k[i, i]
                distance = lam[i] if correction > 0 else math.inf
                error = max(error, min(abs(correction), distance))
        # After 4096 full sweeps the convergence certificate is still orders
        # of magnitude above tolerance: the original algorithm cannot pass.
        self.assertGreater(error, 100 * 1e-11)

    def test_two_normal_joint_solve_repairs_correlated_pair(self):
        g, target, lower, reference = correlated_normal_pair()
        contacts = [(0, 0.5), (3, 0.5)]
        smooth = np.zeros(6)
        # The ordinary path alone (repair activates only after sweep 64) must
        # still report honest non-convergence within a 64-sweep budget.
        self.solve(self.problem(np.eye(6), smooth, g, target=target, lower=lower,
                                contacts=contacts, iterations=64, tolerance=1e-11), 3)
        p = self.problem(np.eye(6), smooth, g, target=target, lower=lower,
                         contacts=contacts, iterations=512, tolerance=1e-11)
        v, impulse, result = self.solve(p)
        self.assertLessEqual(result.iterations, 512)
        np.testing.assert_allclose(impulse, reference, atol=1e-6, rtol=0)
        np.testing.assert_allclose(v, smooth + g.T @ impulse, atol=1e-9, rtol=0)
        self.check_contact(np.eye(6), g, smooth, impulse, v, 0, 0.5, target)
        self.check_contact(np.eye(6), g, smooth, impulse, v, 3, 0.5, target)

    def test_rank_deficient_self_stress_block_is_verified_degenerate(self):
        singular = np.linalg.svd(SELF_STRESS_G, compute_uv=False)
        self.assertGreater(singular[4], 0.05)          # exactly rank-1 deficient
        self.assertLess(singular[5], 1e-12)
        u = np.linalg.svd(SELF_STRESS_G)[0]
        d = u[:, 5]                                    # self-stress direction
        # Nullness holds against the original jacobian rows and (mass = I)
        # every response row, not merely against entries of K.
        self.assertLess(np.abs(d @ SELF_STRESS_G).max(), 1e-12)
        block = SELF_STRESS_G @ SELF_STRESS_G.T
        eigen = np.sort(np.abs(np.linalg.eigvalsh(0.5 * (block + block.T))))
        self.assertLess(eigen[0], 1e-12)
        self.assertGreater(eigen[1], 1e-3)
        # The self-stress mixes friction rows of both contacts.
        self.assertGreater(np.linalg.norm(d[[1, 2]]), 0.1)
        self.assertGreater(np.linalg.norm(d[[4, 5]]), 0.1)

    def test_null_direction_boundary_move_repairs_self_stress_block(self):
        lower = np.array([0.0, -math.inf, -math.inf, 0.0, -math.inf, -math.inf])
        contacts = [(0, SELF_STRESS_MU), (3, SELF_STRESS_MU)]
        # The move happens at sweep 256; capping at 256 sweeps keeps the
        # ordinary path only, which must honestly fail on this block.
        self.solve(self.problem(np.eye(6), SELF_STRESS_SMOOTH, SELF_STRESS_G, lower=lower,
                                contacts=contacts, iterations=256, tolerance=1e-11), 3)
        p = self.problem(np.eye(6), SELF_STRESS_SMOOTH, SELF_STRESS_G, lower=lower,
                         contacts=contacts, iterations=4096, tolerance=1e-11)
        v, impulse, result = self.solve(p)
        self.assertGreater(result.iterations, 256)     # converged only via the move
        self.assertLessEqual(result.iterations, 4096)
        for first in (0, 3):
            self.check_contact(np.eye(6), SELF_STRESS_G, SELF_STRESS_SMOOTH,
                               impulse, v, first, SELF_STRESS_MU)
        self.assertGreater(impulse[0], 0.1)
        self.assertGreater(impulse[3], 0.1)

    def test_randomized_well_posed_rows_match_pure_python_reference(self):
        for seed in range(12):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(1000 + seed)
                n = int(rng.integers(3, 9))
                a = rng.normal(size=(n, n))
                mass = a @ a.T + n * np.eye(n)
                g = rng.normal(size=(n - 1, n))
                regularizer = 0.05 + rng.random(n - 1)
                target = rng.normal(size=n - 1)
                smooth = rng.normal(size=n)
                response = np.linalg.solve(mass, g.T)
                reference = np.linalg.solve(g @ response + np.diag(regularizer),
                                            target - g @ smooth)
                p = self.problem(mass, smooth, g, target=target, regularizer=regularizer)
                v, impulse, _ = self.solve(p)
                scale = max(1.0, float(np.abs(reference).max()))
                np.testing.assert_allclose(impulse, reference, atol=1e-9*scale, rtol=0)
                np.testing.assert_allclose(v, smooth + response @ reference,
                                           atol=1e-9*scale, rtol=0)

    def test_randomized_well_posed_contacts_keep_kkt_certificates(self):
        for seed in range(12):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(2000 + seed)
                n = 6
                a = rng.normal(size=(n, n))
                mass = a @ a.T + n * np.eye(n)
                g = np.linalg.qr(rng.normal(size=(n, n)))[0]  # orthonormal rows
                smooth = rng.normal(size=n)
                for first in (0, 3):                          # approaching normals
                    if g[first] @ smooth > -0.2:
                        smooth -= (g[first] @ smooth + 1.0) * g[first]
                mu = float(0.3 + rng.random())
                lower = np.array([0., -math.inf, -math.inf, 0., -math.inf, -math.inf])
                p = self.problem(mass, smooth, g, lower=lower, contacts=[(0, mu), (3, mu)])
                v, impulse, _ = self.solve(p)
                self.check_contact(mass, g, smooth, impulse, v, 0, mu)
                self.check_contact(mass, g, smooth, impulse, v, 3, mu)


if __name__ == "__main__":
    unittest.main(verbosity=2)
