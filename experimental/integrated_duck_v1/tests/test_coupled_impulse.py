"""Independent generalized impulse algebra; no robot, simulator or GPU.

The mass matrix is supplied in full, including authored armature. NumPy dense
solves and analytic KKT conditions are independent of the implementation.
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


class CoupledImpulseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        override = os.environ.get("COUPLED_IMPULSE_LIBRARY")
        if override:
            cls.path = Path(override).resolve()
        else:
            cls.directory = tempfile.TemporaryDirectory(prefix="independent-coupled-impulse-")
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

    def test_joint_force_to_impulse_scaling_with_full_mass_and_armature(self):
        rigid = np.array([[3., .4, -.2], [.4, 2., .3], [-.2, .3, 1.]])
        mass = rigid + np.diag([.07, .11, .19])
        g = np.array([[1., .3, 0.], [-.2, 1., .4]])
        regularizer = np.array([.2, .35])
        vpre, asmooth, aref = np.array([.4, -.3, .2]), np.array([.7, -.2, .1]), np.array([1.2, -.8])
        response = np.linalg.solve(mass, g.T)
        force = np.linalg.solve(g@response + np.diag(regularizer), aref-g@asmooth)
        for h in (.002, .01, .02):
            with self.subTest(h=h):
                smooth = vpre + h*asmooth
                p = self.problem(mass, smooth, g, target=g@vpre+h*aref, regularizer=regularizer)
                v, impulse, _ = self.solve(p)
                np.testing.assert_allclose(impulse, h*force, atol=2e-11, rtol=0)
                np.testing.assert_allclose(v, smooth+response@(h*force), atol=2e-11, rtol=0)
                np.testing.assert_allclose(mass@(v-smooth), g.T@impulse, atol=2e-12, rtol=0)
        # Omitting armature cannot pass this same independent force reference.
        wrong = np.linalg.solve(g@np.linalg.solve(rigid, g.T)+np.diag(regularizer), aref-g@asmooth)
        self.assertGreater(np.linalg.norm(wrong-force), .01)

    def test_offdiagonal_mass_propagates_single_row_to_unconstrained_dofs(self):
        mass = np.array([[2., .5, .1], [.5, 1.5, -.2], [.1, -.2, 1.]])
        g = np.array([[1., 0., .4]])
        smooth = np.array([-1., .3, .2])
        response = np.linalg.solve(mass, g.T)
        expected = -float((g@smooth)[0])/float((g@response)[0, 0])
        v, impulse, _ = self.solve(self.problem(mass, smooth, g))
        self.assertAlmostEqual(impulse[0], expected, delta=1e-12)
        np.testing.assert_allclose(v, smooth+response[:, 0]*expected, atol=1e-12, rtol=0)
        self.assertGreater(abs(v[1]-smooth[1]), .01)

    def check_contact(self, mass, g, smooth, impulse, velocity, first, mu, target=None):
        target = np.zeros(len(g)) if target is None else np.asarray(target)
        residual = g@velocity-target
        jn, jt = impulse[first], impulse[first+1:first+3]
        slip = residual[first+1:first+3]
        self.assertGreaterEqual(jn, -1e-12)
        self.assertGreaterEqual(residual[first], -2e-9)
        self.assertLessEqual(abs(jn*residual[first]), 2e-9)
        self.assertLessEqual(np.linalg.norm(jt), mu*jn+2e-10)
        if np.linalg.norm(slip) > 1e-6 and mu:
            self.assertAlmostEqual(np.linalg.norm(jt), mu*jn, delta=2e-10)
            multiplier = -float(slip@jt)/float(jt@jt)
            self.assertGreater(multiplier, 0.)
            self.assertLessEqual(np.linalg.norm(slip+multiplier*jt), 2e-9)
        np.testing.assert_allclose(mass@(velocity-smooth), g.T@impulse, atol=2e-10, rtol=0)
        return residual

    def test_anisotropic_offcenter_coulomb_kkt_without_normal_dilatancy(self):
        k = np.array([[1., .2, -.1], [.2, 2., .6], [-.1, .6, .8]])
        mass, g, smooth = np.linalg.inv(k), np.eye(3), np.array([-1., 3., 2.])
        for mu in (.6, 1.):
            with self.subTest(mu=mu):
                v, impulse, _ = self.solve(self.problem(mass, smooth, g, lower=[0., -math.inf, -math.inf], contacts=[(0, mu)]))
                residual = self.check_contact(mass, g, smooth, impulse, v, 0, mu)
                self.assertGreater(impulse[0], .1)
                self.assertAlmostEqual(residual[0], 0., delta=2e-9)
                tangent, slip = impulse[1:], residual[1:]
                multiplier = -float(slip@tangent)/float(tangent@tangent)
                self.assertGreater(mu*multiplier*np.linalg.norm(tangent), .01,
                                   "associated cone derivative would alter normal equation")
                conditional_rhs = smooth[1:]+k[1:, 0]*impulse[0]
                wrong = -np.linalg.solve(k[1:, 1:], conditional_rhs)
                wrong *= mu*impulse[0]/np.linalg.norm(wrong)
                wrong_slip = conditional_rhs+k[1:, 1:]@wrong
                wrong_multiplier = -float(wrong_slip@wrong)/float(wrong@wrong)
                self.assertGreater(np.linalg.norm(wrong_slip+wrong_multiplier*wrong), .01)

    def test_simultaneous_unilateral_limit_and_contact(self):
        mass = np.array([[2., .2, .1, 0.], [.2, 1.5, .15, 0.], [.1, .15, 1., .1], [0., 0., .1, 1.]])
        g = np.array([[1., 0., .4, 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        smooth = np.array([-2., -1., 2., 1.])
        regularizer = np.array([.2, 0., 0., 0.])
        p = self.problem(mass, smooth, g, regularizer=regularizer,
                         lower=[0., 0., -math.inf, -math.inf], contacts=[(1, .6)])
        v, impulse, _ = self.solve(p)
        self.assertGreater(impulse[0], .1)
        self.assertAlmostEqual(float(g[0]@v)+regularizer[0]*impulse[0], 0., delta=2e-9)
        self.check_contact(mass, g, smooth, impulse, v, 1, .6)

    def test_redundant_normals_psd_global_matrix_is_supported(self):
        mass, g, smooth = np.diag([2., 3., 4.]), np.vstack([np.eye(3), np.eye(3)]), np.array([-1., 2., 3.])
        v, impulse, _ = self.solve(self.problem(mass, smooth, g,
            lower=[0., -math.inf, -math.inf]*2, contacts=[(0, 0.), (3, 0.)]))
        self.assertAlmostEqual(impulse[0]+impulse[3], 2., delta=1e-12)
        np.testing.assert_allclose(v, [0., 2., 3.], atol=1e-12, rtol=0)
        self.check_contact(mass, g, smooth, impulse, v, 0, 0.)
        self.check_contact(mass, g, smooth, impulse, v, 3, 0.)

    def test_zero_friction_removes_warm_tangential_impulses(self):
        v, impulse, _ = self.solve(self.problem(np.eye(3), [-1., 2., 3.], np.eye(3),
            lower=[0., -math.inf, -math.inf], contacts=[(0, 0.)], warm=[10., 30., -20.]))
        np.testing.assert_array_equal(impulse, [1., 0., 0.])
        np.testing.assert_array_equal(v, [0., 2., 3.])

    def test_zero_rows_preserves_smooth_velocity(self):
        v, impulse, result = self.solve(self.problem([[2., .1], [.1, 1.]], [.2, -.7], []))
        np.testing.assert_array_equal(v, [.2, -.7])
        self.assertEqual(len(impulse), 0)
        self.assertEqual(result.iterations, 0)

    def test_invalid_numeric_inputs_leave_entire_output_untouched(self):
        cases = [("mass-nan", [[math.nan, 0.], [0., 1.]], [0., 0.], [[1., 0.]], 1),
                 ("asymmetric", [[1., .1], [0., 1.]], [0., 0.], [[1., 0.]], 1),
                 ("indefinite", [[1., 2.], [2., 1.]], [0., 0.], [[1., 0.]], 2),
                 ("velocity-inf", np.eye(2), [math.inf, 0.], [[1., 0.]], 1),
                 ("jacobian-nan", np.eye(2), [0., 0.], [[math.nan, 0.]], 1),
                 ("zero-free-row", np.eye(2), [0., 0.], [[0., 0.]], 2)]
        for name, mass, velocity, g, expected in cases:
            with self.subTest(case=name):
                self.solve(self.problem(mass, velocity, g), expected)
        for kwargs in (dict(regularizer=[-.1]), dict(target=[math.nan]), dict(warm=[math.inf]),
                       dict(lower=[2.], upper=[1.]), dict(tolerance=0.), dict(tolerance=1e-4), dict(iterations=0)):
            with self.subTest(case=kwargs):
                self.solve(self.problem(np.eye(2), [0., 0.], [[1., 0.]], **kwargs), 1)

    def test_invalid_contact_layout_and_regularization_are_atomic(self):
        for kwargs in (dict(contacts=[(1, .6)]), dict(contacts=[(0, -.1)]),
                       dict(contacts=[(0, math.nan)]), dict(contacts=[(0, .6), (0, .6)]),
                       dict(contacts=[(0, .6)], regularizer=[.1, 0., 0.]),
                       dict(contacts=[(0, .6)], upper=[math.inf, 1., math.inf])):
            with self.subTest(case=kwargs):
                self.solve(self.problem(np.eye(3), [-1., 2., 3.], np.eye(3),
                                        lower=[0., -math.inf, -math.inf], **kwargs), 1)

    def test_nonconvergence_does_not_publish_partial_iterates(self):
        self.solve(self.problem([[1., .9], [.9, 1.]], [-1., 1.], np.eye(2), iterations=1), 3)

    def test_overflow_in_momentum_verification_is_numeric_and_atomic(self):
        # Analytically finite solution: lambda=2^700, velocity=2^300.
        # But M*delta_v and G^T*lambda both overflow to infinity in binary64.
        # Their difference must not turn NaN into a false zero residual.
        mass, jacobian, target = math.ldexp(1., 1000), math.ldexp(1., 600), math.ldexp(1., 900)
        self.assertTrue(math.isfinite(math.ldexp(1., 700)))
        self.assertTrue(math.isfinite(math.ldexp(1., 300)))
        self.solve(self.problem([[mass]], [0.], [[jacobian]], target=[target]), 2)

    def test_extreme_disk_bracket_overflow_rejects_without_output(self):
        self.solve(self.problem(np.eye(3), [-1e-150, 1e50, 0.], np.eye(3),
                   lower=[0., -math.inf, -math.inf], contacts=[(0, 1.)],
                   tolerance=1e-160, iterations=1), 2)

    def test_large_scalar_impulse_cannot_hide_nonzero_correction(self):
        big = math.ldexp(1., 60)
        # The second row is fixed at lambda=1. Its coupling leaves the first
        # residual exactly 1, while subtracting that correction from 2^60 is
        # swallowed by binary64 rounding. A difference-of-iterates certificate
        # would falsely claim zero error; the direct correction must reject.
        self.assertEqual(big - 1., big)
        self.assertEqual((-big + big) + 1., 1.)
        self.solve(self.problem(np.eye(2), [0., 0.], [[1., 0.], [1., 1.]],
                   target=[big, 0.], lower=[-math.inf, 1.], upper=[math.inf, 1.],
                   warm=[big, 1.], iterations=1), 3)

    def test_minimum_eigenvalue_certificate_exposes_soft_tangent_error(self):
        # A scalar row AFTER the contact adds 5e-7 residual along the soft
        # tangent. Dividing by max eig=1e12 hides it as 5e-19; dividing by min
        # eig=1 exposes the still-required correction. The first sweep is not
        # converged even though every individual block was solved once.
        mass = np.diag([1., 1., 1e-12, 1.])
        g = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                      [0., 0., 1., 0.], [0., 1., 0., 1.]])
        params = dict(target=[0., 0., 0., 1e-6],
                      lower=[0., -math.inf, -math.inf, -math.inf], contacts=[(0, 1.)])
        self.assertLess(5e-7/1e12, 1e-12)
        self.assertGreater(5e-7/1., 1e-12)
        self.solve(self.problem(mass, [-1., 0., 0., 0.], g, iterations=1, **params), 3)
        velocity, impulse, _ = self.solve(self.problem(mass, [-1., 0., 0., 0.], g, **params))
        np.testing.assert_allclose(velocity, [0., 0., 0., 1e-6], atol=2e-12, rtol=0)
        np.testing.assert_allclose(impulse, [1., -1e-6, 0., 1e-6], atol=2e-12, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
