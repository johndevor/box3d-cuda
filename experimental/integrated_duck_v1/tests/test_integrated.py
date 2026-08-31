"""Independent small integrated-owner tests; never step the Duck morphology."""
import ctypes as C
import hashlib
import itertools
import os
from pathlib import Path
import re
import sys
import unittest

DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIRECTORY))
import native as n

np = n.np


def exhaustive(pre, geometry, pairs, environment=0):
    """Independent exhaustive box-QP; zero-friction normals only.

    Uses the declared PRE mass/Jacobian tensors, never the integrated solver.
    Enumerates free/lower/upper assignments and checks full KKT conditions.
    """
    e = environment
    active = np.flatnonzero(pre['active'][e])
    rows = [x.copy() for x in pre['G'][e, active]]
    target = list(pre['target'][e, active]); regularizer = list(pre['R'][e, active])
    lower = list(pre['lower'][e, active]); upper = list(pre['upper'][e, active])
    joint_rows = len(rows)
    for pair, manifold in zip(pairs, geometry):
        for point in manifold.points[:manifold.count]:
            row = np.zeros(pre['v'].shape[1])
            for body, sign in ((pair.body_a, -1.), (pair.body_b, 1.)):
                arm = np.array(point.point)-pre['pose'][e, body, :3]
                wrench = np.r_[np.array(manifold.normal), np.cross(arm, np.array(manifold.normal))]
                row += sign*wrench@pre['J'][e, body]
            rows.append(row)
            target.append(min(1., .2*max(0., point.depth-2e-6)/pre['dt']))
            regularizer.append(0.); lower.append(0.); upper.append(np.inf)
    G = np.array(rows).reshape((-1, pre['v'].shape[1]))
    target, lower, upper = np.array(target), np.array(lower), np.array(upper)
    inverse = np.linalg.inv(pre['mass'][e])
    H = G@inverse@G.T+np.diag(regularizer)
    rhs = G@pre['smooth'][e]-target
    choices = []
    for lo, hi in zip(lower, upper):
        modes = [0]
        if np.isfinite(lo): modes.append(-1)
        if np.isfinite(hi) and hi != lo: modes.append(1)
        choices.append(modes)
    for mode in itertools.product(*choices):
        x = np.zeros(len(rows)); free = [i for i, m in enumerate(mode) if m == 0]
        fixed = [i for i, m in enumerate(mode) if m != 0]
        for i in fixed: x[i] = lower[i] if mode[i] == -1 else upper[i]
        if free:
            try: x[free] = np.linalg.solve(H[np.ix_(free, free)], -rhs[free]-H[np.ix_(free, fixed)]@x[fixed])
            except np.linalg.LinAlgError: continue
        gradient = H@x+rhs
        if np.any(x < lower-1e-10) or np.any(x > upper+1e-10): continue
        if any((m == 0 and abs(gradient[i]) > 1e-9) or (m == -1 and gradient[i] < -1e-9) or
               (m == 1 and gradient[i] > 1e-9) for i, m in enumerate(mode)): continue
        return pre['smooth'][e]+inverse@G.T@x, G, x, joint_rows
    raise AssertionError('independent exhaustive KKT solution not found')


class IntegratedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(os.environ.get('INTEGRATED_DUCK_LIBRARY', '/tmp/libintegrated_duck_candidate_r1.dylib')).resolve()
        cls.digest = hashlib.sha256(cls.path.read_bytes()).hexdigest()
        print('integrated_test_library='+str(cls.path)+' sha256='+cls.digest, file=sys.stderr, flush=True)
        cls.lib = n.library(cls.path)

    @classmethod
    def tearDownClass(cls):
        if hashlib.sha256(cls.path.read_bytes()).hexdigest() != cls.digest:
            raise AssertionError('library changed during synthetic integrated gate')

    def setUp(self): self.owned = []

    def tearDown(self):
        for scene in reversed(self.owned): scene.close()

    def fixture(self, joints=1):
        f = n.av.Fixture(joints)
        if joints:
            f.hinge[0].axis[:] = [0., 1., 0.]
            f.hinge[0].ap[:] = [.4, 0., 0.]
            f.hinge[0].ac[:] = [0., 0., 0.]
            f.hinge[0].armature = .027
        return f

    def scene(self, f, environments=1, contact=True, offcenter=False, joint_q=0., downward=-1.):
        q = np.tile(f.reference, (environments, 1)); q[:, 2] = .1
        if f.J: q[:, 7] = joint_q
        v = np.zeros((environments, f.N)); v[:, 2] = downward
        shapes = n.shapes_for_fixture(f); n.box(shapes[1])
        if offcenter:
            # The total two-link COM is x=.2: patch[.4,.6] lies outside it.
            # Earlier patch[.15,.35] contained it and did not excite rotation.
            for vertex in shapes[1].vertices[:8]: vertex[0] += .5
        pairs = (n.contact.Pair*int(contact))()
        if contact: pairs[0] = n.contact.Pair(91, 0, 1)
        s = n.Scene(self.lib, f, q, v, shapes, pairs, np.zeros((environments, int(contact))), gravity=np.zeros((environments, 3)))
        self.owned.append(s)
        return s

    def av_scene(self, scene):
        a = n.av.Scene(self.lib, scene.f, scene.q.copy(), scene.v.copy(), lim=scene.limits, gravity=scene.gravity.copy())
        self.owned.append(a)
        return a

    def assert_success(self, result):
        rc, diagnostic = result
        self.assertEqual(rc, 0, str(diagnostic))
        for d in diagnostic:
            self.assertEqual(d['phase'], 6)
            for key in ('joint_residual', 'normal_residual', 'tangent_residual', 'momentum_residual'):
                self.assertTrue(np.isfinite(d[key]))
                self.assertLessEqual(d[key], 1e-8, (key, d))
        return diagnostic

    def test_gravity_free_box_impact_matches_analytic_mass_response(self):
        scene = self.scene(self.fixture(0))
        before = scene.read()
        self.assertEqual(before.geometry[0].count, 4)
        # Tighten requested convergence rather than loosen angular acceptance:
        # a .1m lever amplifies a normal-velocity residual into angular error.
        diagnostic = self.assert_success(scene.step(tolerance=1e-10))
        after = scene.read()
        np.testing.assert_allclose(after.v[0], np.zeros(6), atol=2e-8, rtol=0)
        self.assertAlmostEqual(sum(p.normal_impulse for p in after.cache[0].points), 1., delta=2e-6)
        self.assertEqual(after.time[0], .002)
        self.assertEqual(after.count[0], 1)
        np.testing.assert_allclose(after.q[0, :3], before.q[0, :3]+.002*after.v[0, :3], atol=1e-12, rtol=0)
        self.assertEqual(diagnostic[0]['contact_points'], 4)
        self.assertEqual(after.bodies[0].state[2], before.bodies[0].state[2])

    def test_joint_only_matches_av2_independent_exhaustive_solution(self):
        f = self.fixture(1); f.hinge[0].loss = .07; f.hinge[0].damping = .2
        f.hinge[0].kp = 4.; f.hinge[0].cap = 2.; f.hinge[0].motor_enabled = 1
        scene = self.scene(f, contact=False, joint_q=.42, downward=0.)
        independent = self.av_scene(scene)
        target = np.array([[.1]])
        p, pre = independent.pre(target=target)
        velocity, impulses = n.av.solve(pre)
        rc, stage = independent.complete(p, velocity, impulses)
        self.assertEqual(rc, 0)
        self.assertEqual(independent.commit(stage), 0)
        expected = independent.capture()
        self.assert_success(scene.step(target=target))
        actual = scene.read()
        np.testing.assert_allclose(actual.v, expected.v, atol=2e-8, rtol=0)
        np.testing.assert_allclose(actual.q, expected.q, atol=2e-10, rtol=0)
        np.testing.assert_allclose(actual.warm, expected.warm, atol=1e-5, rtol=0)
        np.testing.assert_array_equal(actual.time, expected.time)
        np.testing.assert_array_equal(actual.count, expected.count)

    def test_offcenter_contact_uses_full_mass_and_moves_root_and_joint(self):
        scene = self.scene(self.fixture(1), offcenter=True)
        independent = self.av_scene(scene)
        _, pre = independent.pre()
        geometry = scene.read().geometry
        expected, rows, impulses, _ = exhaustive(pre, geometry, scene.pairs)
        self.assertGreater(abs(pre['mass'][0, 4, 6]), .01, 'fixture must couple root and hinge')
        self.assertGreater(abs(expected[4]), .01)
        self.assertGreater(abs(expected[6]), .01)
        self.assert_success(scene.step(tolerance=1e-10))
        after = scene.read()
        np.testing.assert_allclose(after.v[0], expected, atol=2e-8, rtol=0)
        np.testing.assert_allclose(pre['mass'][0]@(after.v[0]-pre['smooth'][0]), rows.T@impulses, atol=2e-8, rtol=0)
        self.assertGreater(abs(after.bodies[2].state[7])+abs(after.bodies[2].state[9])+abs(after.bodies[2].state[11]), .01)

    def test_simultaneous_soft_limit_and_contact_matches_exhaustive_qp(self):
        f = self.fixture(1); f.hinge[0].loss = .03
        scene = self.scene(f, offcenter=True, joint_q=.42)
        independent = self.av_scene(scene)
        _, pre = independent.pre()
        expected, rows, impulses, joint_rows = exhaustive(pre, scene.read().geometry, scene.pairs)
        self.assertGreater(joint_rows, 0)
        self.assertGreater(np.max(np.abs(impulses[:joint_rows])), 0.)
        self.assertGreater(np.max(impulses[joint_rows:]), 0.)
        diagnostic = self.assert_success(scene.step(tolerance=1e-10))
        self.assertGreater(diagnostic[0]['active_limits'], 0)
        self.assertGreater(diagnostic[0]['contact_points'], 0)
        np.testing.assert_allclose(scene.read().v[0], expected, atol=2e-8, rtol=0)

    def test_second_environment_nan_target_preserves_both_owners(self):
        scene = self.scene(self.fixture(1), environments=2)
        before = scene.read()
        rc, diagnostics = scene.step(target=np.array([[0.], [np.nan]]))
        self.assertNotEqual(rc, 0)
        self.assertEqual(scene.read().bytes(), before.bytes())
        self.assertEqual(len(diagnostics), 2)
        self.assert_success(scene.step())
        self.assertEqual(list(scene.read().count), [1, 1])

    def test_second_environment_solver_failure_after_first_solve_rolls_back(self):
        scene = self.scene(self.fixture(0), environments=2, downward=[0., -1.])
        before = scene.read()
        rc, diagnostics = scene.step(max_iterations=1, tolerance=1e-12)
        self.assertNotEqual(rc, 0)
        self.assertGreater(diagnostics[0]['iterations'], 0, 'first environment must finish its private solve')
        self.assertEqual(diagnostics[0]['native_status'], 0)
        self.assertNotEqual(diagnostics[1]['native_status'], 0)
        self.assertEqual(diagnostics[1]['phase'], 3)
        self.assertEqual(scene.read().bytes(), before.bytes())
        self.assert_success(scene.step())
        self.assertEqual(list(scene.read().count), [1, 1])

    def test_capture_masked_reset_restore_and_deterministic_replay(self):
        scene = self.scene(self.fixture(1), environments=2, offcenter=True)
        initial = scene.read(); snapshot = scene.capture()
        self.assert_success(scene.step())
        final = scene.read()
        self.assertNotEqual(final.bytes(0), initial.bytes(0))
        self.assertGreater(sum(p.normal_impulse for m in final.cache for p in m.points[:m.count]), 0.)
        self.assertEqual(scene.reset([255, 0], snapshot), 0)
        subset = scene.read()
        self.assertEqual(subset.bytes(0), initial.bytes(0))
        self.assertEqual(subset.bytes(1), final.bytes(1))
        self.assertEqual(scene.reset([0, 7]), 0)
        self.assertEqual(scene.read().bytes(), initial.bytes())
        self.assert_success(scene.step())
        self.assertEqual(scene.read().bytes(), final.bytes())
        current = scene.read().bytes()
        self.assertEqual(scene.reset([0, 0], snapshot), 0)
        self.assertEqual(scene.read().bytes(), current)

    def test_cross_owner_snapshot_rejected_even_identical_topology(self):
        a = self.scene(self.fixture(0), environments=2)
        b = self.scene(self.fixture(0), environments=2)
        snapshot = a.capture(); before_a, before_b = a.read().bytes(), b.read().bytes()
        self.assertNotEqual(b.reset([255, 0], snapshot), 0)
        self.assertEqual(a.read().bytes(), before_a)
        self.assertEqual(b.read().bytes(), before_b)

    def test_driver_source_validates_two_owners_under_lock_before_commit(self):
        source = (DIRECTORY/'src/integrated_duck_v1.cpp').read_text()
        executable = re.sub(r'//[^\n]*|/\*.*?\*/', '', source, flags=re.S)
        self.assertNotRegex(executable, r'\bbcv1_step\s*\(')
        helper = source[source.index('void commit('):source.index('void reset(')]
        self.assertLess(helper.index('av2_validate_commit('), helper.index('av2_commit('))
        self.assertLess(helper.index('bcx1_validate_commit('), helper.index('av2_commit('))
        self.assertLess(helper.index('bcx1_validate_commit('), helper.index('bcx1_commit('))
        for name in ('idv1_step', 'idv1_restore', 'idv1_reset'):
            start = source.index('int '+name+'(')
            tail = source[start:]
            self.assertIn('std::lock_guard<std::mutex> lock(s->lock)', tail.split('\nint ', 1)[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
