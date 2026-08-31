"""Independent black-box CPU contact checks through contact_v1.h only.

No implementation helper, simulator, NumPy, GPU or provider is used. Analytic
expectations use authored tetrahedra/cubes and rigid point-impulse mechanics.
Set CONTACT_V1_LIBRARY to test an explicitly selected compiled library.
"""
import ctypes as C
import hashlib
import math
import os
from pathlib import Path
import struct
import sys
import unittest


OK, INVALID, CAPACITY, NUMERIC, TOPOLOGY = 0, 1, 2, 3, 4
F3, F13 = C.c_float * 3, C.c_float * 13


class Body(C.Structure):
    _fields_ = [("state", F13), ("inverse_mass", C.c_float), ("inverse_inertia", F3)]


class Shape(C.Structure):
    _fields_ = [("caller_id", C.c_uint32), ("kind", C.c_uint32),
                ("vertex_count", C.c_uint32), ("fixed", C.c_uint32),
                ("vertices", F3 * 32), ("plane_normal", F3), ("plane_offset", C.c_float)]


class Pair(C.Structure):
    _fields_ = [("caller_id", C.c_uint32), ("body_a", C.c_uint32), ("body_b", C.c_uint32)]


class Point(C.Structure):
    _fields_ = [("feature", C.c_uint64), ("point", F3), ("depth", C.c_float),
                ("normal_impulse", C.c_float), ("tangent_impulse", C.c_float * 2)]


class Manifold(C.Structure):
    _fields_ = [("count", C.c_uint32), ("normal", F3), ("tangent1", F3),
                ("tangent2", F3), ("points", Point * 4)]


class Registration(C.Structure):
    _fields_ = [("version", C.c_uint32), ("environments", C.c_uint32),
                ("bodies", C.c_uint32), ("pairs", C.c_uint32),
                ("shapes", C.POINTER(Shape)), ("contact_pairs", C.POINTER(Pair)),
                ("initial", C.POINTER(Body)), ("gravity_xyz", C.POINTER(C.c_float)),
                ("pair_friction", C.POINTER(C.c_float))]


def body(position=(0., 0., 0.), velocity=(0., 0., 0.), inverse_mass=1.,
         inverse_inertia=(1., 1., 1.), omega=(0., 0., 0.)):
    return Body(F13(*position, 0., 0., 0., 1., *velocity, *omega), inverse_mass, F3(*inverse_inertia))


def plane(caller_id=100):
    result = Shape(caller_id=caller_id, kind=2, fixed=1)
    result.plane_normal[:] = (0., 0., 1.)
    return result


def convex(vertices, caller_id=101):
    result = Shape(caller_id=caller_id, kind=1, vertex_count=len(vertices), fixed=0)
    for i, vertex in enumerate(vertices):
        result.vertices[i][:] = vertex
    return result


def tetra(caller_id=101, offcenter=False):
    # Both authored tetrahedra have the arithmetic vertex centroid at the body
    # COM. Exactly one vertex touches z=0 when the body COM is at z=.75.
    vertices = ((.2, 0., -.75), (-.4, -.5, .25), (.4, -.5, .25), (-.2, 1., .25)) if offcenter else (
        (0., 0., -.75), (-.5, -.5, .25), (.5, -.5, .25), (0., 1., .25))
    return convex(vertices, caller_id)


def cube(caller_id):
    return convex([(x, y, z) for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)], caller_id)


def norm(values):
    return math.sqrt(sum(v * v for v in values))


def defined_bytes(bodies, manifolds, clocks):
    """Compare all ABI-defined scalars, excluding unspecified C struct padding."""
    result = bytes(bodies) + bytes(clocks)
    for manifold in manifolds:
        result += struct.pack("<I", manifold.count)
        for vector in (manifold.normal, manifold.tangent1, manifold.tangent2):
            result += bytes(vector)
        for point in manifold.points:
            result += struct.pack("<Q", point.feature) + bytes(point.point)
            result += struct.pack("<ff", point.depth, point.normal_impulse) + bytes(point.tangent_impulse)
    return result


class IndependentResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library_path = Path(os.environ.get("CONTACT_V1_LIBRARY", "/tmp/box3d-contact-v1-local/libcontact_v1.dylib")).resolve()
        cls.library_hash = hashlib.sha256(cls.library_path.read_bytes()).hexdigest()
        print("independent_contact_library=" + str(cls.library_path) + " sha256=" + cls.library_hash,
              file=sys.stderr, flush=True)
        cls.lib = C.CDLL(str(cls.library_path))
        signatures = {
            "bcv1_create": ([C.POINTER(Registration), C.POINTER(C.c_void_p)], C.c_int),
            "bcv1_destroy": ([C.c_void_p], None),
            "bcv1_read": ([C.c_void_p, C.POINTER(Body), C.POINTER(Manifold), C.POINTER(C.c_double)], C.c_int),
            "bcv1_query": ([C.c_void_p, C.POINTER(Manifold)], C.c_int),
            "bcv1_step": ([C.c_void_p, C.c_float, C.c_uint32], C.c_int),
            "bcv1_capture": ([C.c_void_p, C.POINTER(C.c_void_p)], C.c_int),
            "bcv1_snapshot_destroy": ([C.c_void_p], None),
            "bcv1_restore": ([C.c_void_p, C.c_void_p, C.POINTER(C.c_uint8)], C.c_int),
        }
        for name, (args, result) in signatures.items():
            getattr(cls.lib, name).argtypes = args
            getattr(cls.lib, name).restype = result

    @classmethod
    def tearDownClass(cls):
        if hashlib.sha256(cls.library_path.read_bytes()).hexdigest() != cls.library_hash:
            raise AssertionError("library artifact changed during independent evaluator; result identity ambiguous")
        assert "mujoco" not in sys.modules and "numpy" not in sys.modules

    def register(self, shapes, pairs, bodies, environments=1, gravity=None, friction=None):
        b, p = len(shapes), len(pairs)
        self.assertEqual(len(bodies), environments * b)
        arrays = ((Shape * b)(*shapes), (Pair * p)(*pairs), (Body * len(bodies))(*bodies),
                  (C.c_float * (environments * 3))(*(gravity or [0.] * (environments * 3))),
                  (C.c_float * (environments * p))(*(friction or [0.] * (environments * p))))
        registration = Registration(1, environments, b, p, *arrays)
        handle = C.c_void_p()
        self.assertEqual(self.lib.bcv1_create(C.byref(registration), C.byref(handle)), OK)
        self.assertTrue(handle.value)
        self.addCleanup(self.lib.bcv1_destroy, handle)
        return handle, environments, b, p

    def floor_scene(self, velocity=(0., 0., -2.), offcenter=False, inverse_mass=.5,
                    inverse_inertia=(1., 2., 3.), mu=0.):
        return self.register([plane(), tetra(offcenter=offcenter)], [Pair(10, 0, 1)],
                             [body(inverse_mass=0., inverse_inertia=(0., 0., 0.)),
                              body((0., 0., .75), velocity, inverse_mass, inverse_inertia)], friction=[mu])

    def read(self, scene):
        handle, e, b, p = scene
        bodies, manifolds, clocks = (Body * (e*b))(), (Manifold * (e*p))(), (C.c_double * e)()
        self.assertEqual(self.lib.bcv1_read(handle, bodies, manifolds, clocks), OK)
        return bodies, manifolds, clocks

    def query(self, scene):
        handle, e, _, p = scene
        result = (Manifold * (e*p))()
        self.assertEqual(self.lib.bcv1_query(handle, result), OK)
        return result

    def capture(self, scene):
        snapshot = C.c_void_p()
        self.assertEqual(self.lib.bcv1_capture(scene[0], C.byref(snapshot)), OK)
        self.assertTrue(snapshot.value)
        self.addCleanup(self.lib.bcv1_snapshot_destroy, snapshot)
        return snapshot

    def assert_vector(self, actual, expected, tolerance=2e-6):
        for i, (a, b) in enumerate(zip(actual, expected)):
            self.assertAlmostEqual(a, b, delta=tolerance, msg=f"component {i}: {a} != {b}")

    def test_ctypes_layout_matches_public_fixed_width_fields(self):
        self.assertEqual([C.sizeof(t) for t in (Body, Shape, Pair, Point, Manifold, Registration)],
                         [68, 416, 12, 40, 200, 56])

    def test_unique_tetra_vertex_plane_impact_has_analytic_inelastic_impulse(self):
        scene = self.floor_scene()
        before = self.query(scene)[0]
        self.assertEqual(before.count, 1)
        self.assertEqual(before.points[0].depth, 0.)  # No stabilization/bias term.
        self.assert_vector(before.points[0].point, (0., 0., 0.))
        self.assert_vector(before.normal, (0., 0., 1.))
        self.assertEqual(before.points[0].normal_impulse, 0.)
        self.assertEqual(self.lib.bcv1_step(scene[0], .005, 32), OK)
        states, cache, clocks = self.read(scene)
        self.assertAlmostEqual(cache[0].points[0].normal_impulse, 4., delta=2e-6)
        self.assert_vector(states[1].state[7:10], (0., 0., 0.))
        self.assert_vector(states[1].state[10:13], (0., 0., 0.))
        self.assert_vector(states[1].state[:3], (0., 0., .75))
        self.assertEqual(bytes(states[0]), bytes(body(inverse_mass=0., inverse_inertia=(0., 0., 0.))))
        self.assertAlmostEqual(clocks[0], C.c_float(.005).value, delta=1e-15)
        fresh = self.query(scene)[0]
        self.assertEqual(fresh.points[0].normal_impulse, 0.)

    def test_offcenter_normal_impulse_produces_predicted_torque(self):
        scene = self.floor_scene(offcenter=True)
        initial = self.query(scene)[0]
        self.assertEqual(initial.count, 1)
        self.assertEqual(initial.points[0].depth, 0.)
        self.assertEqual(self.lib.bcv1_step(scene[0], .005, 32), OK)
        states, cache, _ = self.read(scene)
        x = C.c_float(.2).value
        # K = inverse mass + (r cross n)^T I^-1 (r cross n).
        impulse = 2. / (.5 + 2. * x*x)
        self.assertAlmostEqual(cache[0].points[0].normal_impulse, impulse, delta=2e-6)
        self.assert_vector(states[1].state[7:10], (0., 0., -2. + .5*impulse))
        self.assert_vector(states[1].state[10:13], (0., -2.*x*impulse, 0.))
        contact_normal_velocity = states[1].state[9] - x*states[1].state[11]
        self.assertAlmostEqual(contact_normal_velocity, 0., delta=2e-6)

    def test_convex_convex_equal_opposite_impulse_and_momentum(self):
        scene = self.register([cube(21), cube(22)], [Pair(30, 0, 1)],
                              [body((-.5, 0., 0.), (1., 0., 0.), .5),
                               body((.5, 0., 0.), (-1., 0., 0.), .25)])
        initial = self.query(scene)[0]
        self.assertGreater(initial.count, 0)
        self.assert_vector(initial.normal, (1., 0., 0.))
        self.assertTrue(all(point.depth == 0. for point in initial.points[:initial.count]))
        self.assertEqual(self.lib.bcv1_step(scene[0], .005, 128), OK)
        states, cache, _ = self.read(scene)
        impulse = sum(p.normal_impulse for p in cache[0].points[:cache[0].count])
        expected_velocity = (2.*1. + 4.*-1.) / 6.
        self.assertAlmostEqual(impulse, 8./3., delta=2e-5)
        for state in states:
            self.assert_vector(state.state[7:10], (expected_velocity, 0., 0.), 2e-5)
            self.assert_vector(state.state[10:13], (0., 0., 0.), 2e-5)
        for axis in range(3):
            momentum = 2.*states[0].state[7+axis] + 4.*states[1].state[7+axis]
            self.assertAlmostEqual(momentum, -2. if axis == 0 else 0., delta=2e-5)
        self.assertAlmostEqual(2.*(states[0].state[7]-1.), -impulse, delta=2e-5)
        self.assertAlmostEqual(4.*(states[1].state[7]+1.), impulse, delta=2e-5)

    def test_explicit_pair_friction_zero_and_coulomb_disk(self):
        scene = self.register([plane(), tetra(101), tetra(102)], [Pair(10, 0, 1), Pair(11, 0, 2)],
                              [body(inverse_mass=0., inverse_inertia=(0., 0., 0.)),
                               body((-2., 0., .75), (2., 0., -1.)),
                               body((2., 0., .75), (2., 0., -1.))], friction=[0., .25])
        self.assertEqual(self.lib.bcv1_step(scene[0], .005, 64), OK)
        states, caches, _ = self.read(scene)
        self.assertAlmostEqual(states[1].state[7], 2., delta=1e-7)
        for pair, mu in zip(caches, (0., .25)):
            self.assertEqual(pair.count, 1)
            point = pair.points[0]
            self.assertGreater(point.normal_impulse, 0.)
            self.assertLessEqual(norm(point.tangent_impulse), mu*point.normal_impulse + 2e-6)
            if mu == 0.:
                self.assertEqual(norm(point.tangent_impulse), 0.)
            else:
                self.assertAlmostEqual(norm(point.tangent_impulse), mu*point.normal_impulse, delta=2e-6)
        self.assertLess(states[2].state[7], states[1].state[7])
        self.assertAlmostEqual(states[2].state[7], 1.75, delta=2e-6)

    def test_diagonal_sliding_uses_coulomb_disk_not_componentwise_square(self):
        scene = self.floor_scene(velocity=(2., 2., -1.), inverse_mass=1.,
                                 inverse_inertia=(1., 1., 1.), mu=.25)
        self.assertEqual(self.lib.bcv1_step(scene[0], .005, 64), OK)
        states, caches, _ = self.read(scene)
        self.assertEqual(caches[0].count, 1)
        point = caches[0].points[0]
        self.assertAlmostEqual(point.normal_impulse, 1., delta=2e-6)
        self.assertAlmostEqual(norm(point.tangent_impulse), .25, delta=2e-6)
        component = 2. - .25/math.sqrt(2.)
        self.assert_vector(states[1].state[7:10], (component, component, 0.), 2e-6)

    def test_anisotropic_offcenter_friction_satisfies_conditional_kkt(self):
        # The pre-integration contact lever arm is r=(.2,0,-.75), with
        # inverse_mass=.5 and principal inverse inertia=(1,2,3). In world XY,
        # K_t=diag(1.625, 1.1825), and K_xn=.3. The distinct diagonal values
        # discriminate a true constrained solve from radial clipping of K^-1 b.
        # These are impulses; no division by dt or invented force is involved.
        x, z, inv_mass = C.c_float(.2).value, -.75, .5
        kxx, kyy, kxn = inv_mass + 2.*z*z, inv_mass + z*z + 3.*x*x, -2.*z*x
        for authored_mu in (.6, 1.):
            with self.subTest(mu=authored_mu):
                scene = self.floor_scene(velocity=(5., 3., -1.), offcenter=True, mu=authored_mu)
                before = self.query(scene)[0]
                self.assertEqual(before.count, 1)
                self.assertEqual(before.points[0].depth, 0.)
                self.assertEqual(self.lib.bcv1_step(scene[0], .001, 128), OK)
                states, caches, _ = self.read(scene)
                manifold, state = caches[0], states[1].state
                self.assertEqual(manifold.count, 1)
                point = manifold.points[0]
                jx = manifold.tangent1[0]*point.tangent_impulse[0] + manifold.tangent2[0]*point.tangent_impulse[1]
                jy = manifold.tangent1[1]*point.tangent_impulse[0] + manifold.tangent2[1]*point.tangent_impulse[1]
                jn = point.normal_impulse
                cap = C.c_float(authored_mu).value * jn
                tangent_norm = math.hypot(jx, jy)
                self.assertGreater(jn, 0.)
                self.assertAlmostEqual(tangent_norm, cap, delta=2e-6)
                # Cache impulses are PRE-integration; returned state is POST.
                # Reconstruct conditional final slip at the actual solve point
                # from initial velocity plus the public accumulated impulses.
                # The first evaluator draft mixed POST omega with the PRE arm
                # and failed at .000278495/.001754260 m/s for mu .6/1. That was
                # a stage mismatch, not a widened bound or native solver fix.
                expected_linear = (5. + inv_mass*jx, 3. + inv_mass*jy, -1. + inv_mass*jn)
                expected_momentum = (-z*jy, z*jx - x*jn, x*jy)
                solve_omega = (expected_momentum[0], 2.*expected_momentum[1], 3.*expected_momentum[2])
                self.assert_vector(state[7:10], expected_linear, 2e-6)
                slip = (expected_linear[0] + z*solve_omega[1],
                        expected_linear[1] + x*solve_omega[2] - z*solve_omega[0])
                self.assertGreater(math.hypot(*slip), .05)
                multiplier = -(slip[0]*jx + slip[1]*jy)/(tangent_norm*tangent_norm)
                self.assertGreater(multiplier, 0.)
                residual = math.hypot(slip[0] + multiplier*jx, slip[1] + multiplier*jy)
                self.assertLessEqual(residual, 2e-6, "friction impulse must oppose final conditional slip")
                self.assert_vector(slip, (5. + kxn*jn + kxx*jx, 3. + kyy*jy), 2e-6)
                self.assertAlmostEqual(expected_linear[2] - x*solve_omega[1], 0., delta=2e-6)
                # Integration rotates anisotropic inertia: omega is not held
                # constant. Independently require conserved WORLD momentum
                # I_world(q_POST)*omega_POST = accumulated contact torque.
                qnorm = norm(state[3:7])
                qx, qy, qz, qw = [v/qnorm for v in state[3:7]]
                rotation = ((1.-2.*(qy*qy+qz*qz), 2.*(qx*qy-qw*qz), 2.*(qx*qz+qw*qy)),
                            (2.*(qx*qy+qw*qz), 1.-2.*(qx*qx+qz*qz), 2.*(qy*qz-qw*qx)),
                            (2.*(qx*qz-qw*qy), 2.*(qy*qz+qw*qx), 1.-2.*(qx*qx+qy*qy)))
                local_omega = [sum(rotation[i][k]*state[10+i] for i in range(3)) for k in range(3)]
                principal_inertia = (1., .5, 1./3.)
                post_momentum = [sum(rotation[i][k]*principal_inertia[k]*local_omega[k]
                                     for k in range(3)) for i in range(3)]
                self.assert_vector(post_momentum, expected_momentum, 2e-6)
                # A deliberately wrong radial projection remains disk-feasible
                # but must fail this same stationarity condition substantially.
                b = (5. + kxn*jn, 3.)
                unconstrained = (-b[0]/kxx, -b[1]/kyy)
                scale = cap/math.hypot(*unconstrained)
                self.assertLess(scale, 1.)
                wrong = (unconstrained[0]*scale, unconstrained[1]*scale)
                wrong_slip = (b[0] + kxx*wrong[0], b[1] + kyy*wrong[1])
                wrong_multiplier = -(wrong_slip[0]*wrong[0] + wrong_slip[1]*wrong[1])/(cap*cap)
                wrong_residual = math.hypot(wrong_slip[0] + wrong_multiplier*wrong[0],
                                            wrong_slip[1] + wrong_multiplier*wrong[1])
                self.assertGreater(wrong_residual, .05)
                print(f"anisotropic_kkt mu={authored_mu} residual_m_s={residual:.12g} "
                      f"radial_negative_residual_m_s={wrong_residual:.12g} "
                      f"impulse_norm={tangent_norm:.12g} coulomb_cap={cap:.12g}",
                      file=sys.stderr, flush=True)

    def test_invalid_registration_pairs_friction_and_capacity_rejected(self):
        fixed_body = body(inverse_mass=0., inverse_inertia=(0., 0., 0.))
        for label in ("self-pair", "duplicate-pair", "plane-plane", "negative-friction",
                      "nonfinite-friction", "too-many-bodies"):
            shapes, pairs = [plane(), tetra()], [Pair(10, 0, 1)]
            bodies, friction = [fixed_body, body((0., 0., .75))], [0.]
            expected = INVALID
            if label == "self-pair":
                pairs = [Pair(10, 1, 1)]
            elif label == "duplicate-pair":
                pairs = [Pair(10, 0, 1), Pair(11, 0, 1)]
                friction = [0., 0.]
            elif label == "plane-plane":
                shapes, bodies = [plane(), plane(102)], [fixed_body, fixed_body]
            elif label == "negative-friction":
                friction = [-.1]
            elif label == "nonfinite-friction":
                friction = [math.nan]
            else:
                shapes += [tetra(102+i) for i in range(31)]
                bodies += [body((float(i), 0., 10.)) for i in range(31)]
                expected = CAPACITY
            arrays = ((Shape*len(shapes))(*shapes), (Pair*len(pairs))(*pairs),
                      (Body*len(bodies))(*bodies), F3(0., 0., 0.), (C.c_float*len(friction))(*friction))
            descriptor = Registration(1, 1, len(shapes), len(pairs), *arrays)
            handle = C.c_void_p()
            with self.subTest(case=label):
                status = self.lib.bcv1_create(C.byref(descriptor), C.byref(handle))
                if handle.value:
                    self.lib.bcv1_destroy(handle)
                self.assertEqual(status, expected)
                self.assertFalse(handle.value, "rejected registration must not publish a live handle")

    def two_environment_scene(self, caller_id=101):
        floor = body(inverse_mass=0., inverse_inertia=(0., 0., 0.))
        return self.register([plane(), tetra(caller_id)], [Pair(10, 0, 1)],
                             [floor, body((0., 0., .75), (1., 0., -1.)),
                              floor, body((0., 0., .9), (-1., 0., -.5))], environments=2,
                             gravity=[0., 0., -1., 0., 0., -3.], friction=[.25, .75])

    def test_two_environment_mask_restores_exact_state_cache_clock(self):
        scene = self.two_environment_scene()
        initial = self.read(scene)
        snapshot = self.capture(scene)
        for _ in range(3):
            self.assertEqual(self.lib.bcv1_step(scene[0], .005, 32), OK)
        advanced = self.read(scene)
        self.assertNotEqual(defined_bytes(*initial), defined_bytes(*advanced))
        for mask_values in ((255, 0), (0, 2), (1, 127)):
            with self.subTest(mask=mask_values):
                # Restore advanced state before each selected-reset experiment.
                if mask_values == (255, 0):
                    advanced_snapshot = self.capture(scene)
                else:
                    self.assertEqual(self.lib.bcv1_restore(scene[0], advanced_snapshot, None), OK)
                mask = (C.c_uint8 * 2)(*mask_values)
                self.assertEqual(self.lib.bcv1_restore(scene[0], snapshot, mask), OK)
                actual = self.read(scene)
                for env, selected in enumerate(mask_values):
                    expected = initial if selected else advanced
                    def segment(readout):
                        return defined_bytes((Body*2)(*readout[0][env*2:env*2+2]),
                                             (Manifold*1)(readout[1][env]), (C.c_double*1)(readout[2][env]))
                    self.assertEqual(segment(actual), segment(expected))
        before_zero = defined_bytes(*self.read(scene))
        self.assertEqual(self.lib.bcv1_restore(scene[0], advanced_snapshot, (C.c_uint8*2)(0, 0)), OK)
        self.assertEqual(defined_bytes(*self.read(scene)), before_zero)

    def test_restore_replays_environment_parameters_and_warm_cache_exactly(self):
        scene = self.two_environment_scene()
        self.assertEqual(self.lib.bcv1_step(scene[0], .005, 32), OK)
        snapshot = self.capture(scene)
        self.assertEqual(self.lib.bcv1_step(scene[0], .005, 32), OK)
        expected = defined_bytes(*self.read(scene))
        self.assertEqual(self.lib.bcv1_restore(scene[0], snapshot, None), OK)
        self.assertEqual(self.lib.bcv1_step(scene[0], .005, 32), OK)
        self.assertEqual(defined_bytes(*self.read(scene)), expected)

    def test_topology_mismatch_is_atomic_even_for_zero_mask(self):
        destination = self.two_environment_scene()
        foreign = self.two_environment_scene(caller_id=987)
        foreign_snapshot = self.capture(foreign)
        self.assertEqual(self.lib.bcv1_step(destination[0], .005, 32), OK)
        before = defined_bytes(*self.read(destination))
        for mask in (None, (C.c_uint8*2)(255, 0), (C.c_uint8*2)(0, 0)):
            with self.subTest(mask=None if mask is None else list(mask)):
                self.assertEqual(self.lib.bcv1_restore(destination[0], foreign_snapshot, mask), TOPOLOGY)
                self.assertEqual(defined_bytes(*self.read(destination)), before)

    def test_invalid_step_parameters_fail_without_partial_state(self):
        scene = self.two_environment_scene()
        self.assertEqual(self.lib.bcv1_step(scene[0], .005, 32), OK)
        before = defined_bytes(*self.read(scene))
        for dt, iterations in ((0., 32), (-.001, 32), (.0101, 32), (math.nan, 32),
                               (math.inf, 32), (.005, 0), (.005, 129)):
            with self.subTest(dt=dt, iterations=iterations):
                self.assertNotEqual(self.lib.bcv1_step(scene[0], dt, iterations), OK)
                self.assertEqual(defined_bytes(*self.read(scene)), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
