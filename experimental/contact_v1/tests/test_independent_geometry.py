"""Independent double geometry vs host C ABI; no production geometry helpers.

Every reference input is first rounded through its actual ctypes float field.
No simulation calls, dependencies, simulator imports, or GPU execution.
"""
import ctypes as C
import hashlib
import itertools
import math
import os
from pathlib import Path
import sys
import unittest

F3, F13 = C.c_float*3, C.c_float*13


class Body(C.Structure):
    _fields_ = [("state", F13), ("inverse_mass", C.c_float), ("inverse_inertia", F3)]


class Shape(C.Structure):
    _fields_ = [("caller_id", C.c_uint32), ("kind", C.c_uint32), ("vertex_count", C.c_uint32),
                ("fixed", C.c_uint32), ("vertices", F3*32), ("plane_normal", F3), ("plane_offset", C.c_float)]


class Pair(C.Structure):
    _fields_ = [("caller_id", C.c_uint32), ("body_a", C.c_uint32), ("body_b", C.c_uint32)]


class Point(C.Structure):
    _fields_ = [("feature", C.c_uint64), ("point", F3), ("depth", C.c_float),
                ("normal_impulse", C.c_float), ("tangent_impulse", C.c_float*2)]


class Manifold(C.Structure):
    _fields_ = [("count", C.c_uint32), ("normal", F3), ("tangent1", F3), ("tangent2", F3), ("points", Point*4)]


class Registration(C.Structure):
    _fields_ = [("version", C.c_uint32), ("environments", C.c_uint32), ("bodies", C.c_uint32), ("pairs", C.c_uint32),
                ("shapes", C.POINTER(Shape)), ("contact_pairs", C.POINTER(Pair)), ("initial", C.POINTER(Body)),
                ("gravity_xyz", C.POINTER(C.c_float)), ("pair_friction", C.POINTER(C.c_float))]


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def scale(a, x):
    return tuple(v*x for v in a)


def cross(a, b):
    return a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]


def unit(a):
    return scale(a, 1/math.sqrt(dot(a, a)))


def axis_angle(axis, angle):
    return (*scale(unit(axis), math.sin(angle/2)), math.cos(angle/2))


def rotate(q, p):
    q = unit(q)
    u, w = q[:3], q[3]
    return add(add(scale(p, w*w-dot(u, u)), scale(u, 2*dot(u, p))), scale(cross(u, p), 2*w))


def body(p=(0., 0., 0.), q=(0., 0., 0., 1.), fixed=False):
    return Body(F13(*p, *q, 0., 0., 0., 0., 0., 0.), 0. if fixed else 1., F3(*((0.,)*3 if fixed else (1.,)*3)))


def convex(vertices, identity):
    shape = Shape(caller_id=identity, kind=1, vertex_count=len(vertices), fixed=0)
    for i, p in enumerate(vertices):
        shape.vertices[i][:] = p
    return shape


def box(half, identity):
    return convex(list(itertools.product(*[(-h, h) for h in half])), identity)


def plane(n, offset, identity=55):
    return Shape(caller_id=identity, kind=2, fixed=1, plane_normal=F3(*n), plane_offset=offset)


def world_vertices(shape, b):
    return [add(b.state[:3], rotate(b.state[3:7], p)) for p in shape.vertices[:shape.vertex_count]]


def hull(vertices):
    """All supporting planes; true edges share at least two distinct planes."""
    faces = []
    for i, j, k in itertools.combinations(range(len(vertices)), 3):
        n = cross(sub(vertices[j], vertices[i]), sub(vertices[k], vertices[i]))
        if dot(n, n) <= 1e-22:
            continue
        n = unit(n)
        d = dot(n, vertices[i])
        distances = [dot(n, v)-d for v in vertices]
        if max(distances) <= 1e-9:
            pass
        elif min(distances) >= -1e-9:
            n, d = scale(n, -1), -d
        else:
            continue
        if any(math.dist(n, old[0]) <= 1e-8 and abs(d-old[1]) <= 1e-8 for old in faces):
            continue
        members = tuple(i for i, v in enumerate(vertices) if abs(dot(n, v)-d) <= 1e-8)
        faces.append((n, d, members))
    edges = [(i, j) for i, j in itertools.combinations(range(len(vertices)), 2)
             if sum(i in f[2] and j in f[2] for f in faces) >= 2]
    return faces, edges


def sat(a, b):
    fa, ea = hull(a)
    fb, eb = hull(b)
    candidates = [(n, "face_a", None) for n, _, _ in fa]+[(n, "face_b", None) for n, _, _ in fb]
    for ia, ja in ea:
        for ib, jb in eb:
            n = cross(sub(a[ja], a[ia]), sub(b[jb], b[ib]))
            if dot(n, n) > 1e-20:
                candidates.append((unit(n), "edge", ((ia, ja), (ib, jb))))
    rows = []
    for n, kind, edges in candidates:
        aa, bb = [dot(n, v) for v in a], [dot(n, v) for v in b]
        forward, reverse = max(aa)-min(bb), max(bb)-min(aa)
        if forward <= reverse:
            rows.append((forward, n, kind, edges))
        else:
            rows.append((reverse, scale(n, -1), kind, edges))
    return min(rows, key=lambda x: x[0]), rows


def closest_segments(a0, a1, b0, b1):
    """Bounded quadratic minimization over [0,1]^2; no engine math reused."""
    u, v, w = sub(a1, a0), sub(b1, b0), sub(a0, b0)
    aa, bb, cc, dd, ee = dot(u, u), dot(u, v), dot(v, v), dot(u, w), dot(v, w)
    clamp = lambda x: min(1., max(0., x))
    candidates = [(0., clamp(ee/cc)), (1., clamp((ee+bb)/cc)), (clamp(-dd/aa), 0.), (clamp((bb-dd)/aa), 1.)]
    determinant = aa*cc-bb*bb
    if determinant > 1e-24:
        s, t = (bb*ee-cc*dd)/determinant, (aa*ee-bb*dd)/determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append((s, t))
    s, t = min(candidates, key=lambda st: math.dist(add(a0, scale(u, st[0])), add(b0, scale(v, st[1]))))
    return add(a0, scale(u, s)), add(b0, scale(v, t)), s, t


def edge_fixture(angle, penetration):
    a, b = box((1., .08, .05), 10), box((.85, .07, .06), 20)
    qa, qb = (0., 0., 0., 1.), axis_angle((0., 1., 1.), angle)
    edge_a, edge_b = (1., 0., 0.), rotate(qb, (1., 0., 0.))
    n = unit(cross(edge_a, edge_b))
    va, vb = world_vertices(a, body()), world_vertices(b, body(q=qb))
    sa, sb = max(dot(n, v) for v in va), min(dot(n, v) for v in vb)
    support_a = [v for v in va if abs(dot(n, v)-sa) <= 1e-7]
    support_b = [v for v in vb if abs(dot(n, v)-sb) <= 1e-7]
    ca = scale(add(support_a[0], support_a[-1]), .5)
    cb = scale(add(support_b[0], support_b[-1]), .5)
    position_b = add(sub(ca, cb), add((.3, 0., 0.), scale(n, -penetration)))
    return a, b, body(q=qa), body(position_b, qb)


class IndependentGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(os.environ.get("CONTACT_V1_LIBRARY", "/tmp/box3d-contact-v1-local/libcontact_v1.dylib")).resolve()
        cls.digest = hashlib.sha256(cls.path.read_bytes()).hexdigest()
        print("independent_geometry_library="+str(cls.path)+" sha256="+cls.digest, file=sys.stderr, flush=True)
        cls.lib = C.CDLL(str(cls.path))
        for name, args, result in (
            ("bcv1_create", [C.POINTER(Registration), C.POINTER(C.c_void_p)], C.c_int),
            ("bcv1_destroy", [C.c_void_p], None),
            ("bcv1_query", [C.c_void_p, C.POINTER(Manifold)], C.c_int),
        ):
            getattr(cls.lib, name).argtypes, getattr(cls.lib, name).restype = args, result

    @classmethod
    def tearDownClass(cls):
        if hashlib.sha256(cls.path.read_bytes()).hexdigest() != cls.digest:
            raise AssertionError("native library changed during independent geometry gate")

    def query(self, shapes, bodies, reversed_pair=False):
        arrays = (Shape*2)(*shapes), (Pair*1)(Pair(99, 1 if reversed_pair else 0, 0 if reversed_pair else 1)), (Body*2)(*bodies), (C.c_float*3)(0, 0, 0), (C.c_float*1)(0)
        r, h = Registration(1, 1, 2, 1, *arrays), C.c_void_p()
        self.assertEqual(self.lib.bcv1_create(C.byref(r), C.byref(h)), 0, "registration rejected independent finite geometry")
        self.assertTrue(h.value)
        try:
            result, again = Manifold(), Manifold()
            self.assertEqual(self.lib.bcv1_query(h, C.byref(result)), 0)
            self.assertEqual(self.lib.bcv1_query(h, C.byref(again)), 0)
            self.assertEqual(bytes(result), bytes(again), "repeated read-only query nondeterministic")
            self.assertLessEqual(result.count, 4)
            for p in result.points[:result.count]:
                self.assertTrue(all(math.isfinite(x) for x in (*p.point, p.depth, p.normal_impulse, *p.tangent_impulse)))
                self.assertEqual(p.normal_impulse, 0.)
                self.assertEqual(tuple(p.tangent_impulse), (0., 0.))
            return result
        finally:
            self.lib.bcv1_destroy(h)

    def near(self, a, b, tolerance=2e-5):
        self.assertLessEqual(math.dist(a, b), tolerance, f"{tuple(a)} != {tuple(b)}")

    def check_convex(self, shapes, bodies, reversed_pair=False):
        a, b = [world_vertices(s, body_) for s, body_ in zip(shapes, bodies)]
        if reversed_pair:
            a, b = b, a
        best, candidates = sat(a, b)
        result = self.query(shapes, bodies, reversed_pair)
        tolerance = 2e-5*(1+max(math.sqrt(dot(v, v)) for v in a+b))
        if best[0] < -tolerance:
            self.assertEqual(result.count, 0, f"false contact independent SAT gap={best[0]}")
            return result, best
        self.assertGreater(result.count, 0, f"missed independent SAT contact depth={best[0]}")
        self.assertAlmostEqual(math.sqrt(dot(result.normal, result.normal)), 1., delta=2e-6)
        tied = [row for row in candidates if abs(row[0]-best[0]) <= tolerance]
        self.assertLessEqual(min(math.dist(result.normal, row[1]) for row in tied), 2e-4, f"nonminimum SAT normal best={best}, actual={tuple(result.normal)}")
        for point in result.points[:result.count]:
            self.assertGreaterEqual(point.depth, 0.)
            self.assertLessEqual(point.depth, best[0]+tolerance)
            # A and B contact witnesses are the midpoint +/- half-depth normal.
            # Each must lie on its real convex boundary, not a support midpoint
            # outside either hull's feature footprint.
            wa = add(point.point, scale(result.normal, .5*point.depth))
            wb = sub(point.point, scale(result.normal, .5*point.depth))
            for witness, vertices in ((wa, a), (wb, b)):
                faces, _ = hull(vertices)
                distances = [dot(n, witness)-d for n, d, _ in faces]
                self.assertLessEqual(max(distances), tolerance, f"witness outside exact hull: {witness}; distances={distances}")
                self.assertLessEqual(min(abs(x) for x in distances), tolerance, f"witness not on hull boundary: {witness}")
        return result, best

    def test_shifted_rotated_and_rescaled_plane(self):
        tetra = convex(((-.4, -.3, -.8), (.6, -.3, .2), (-.1, .7, .2), (-.1, -.1, .5)), 77)
        qp = axis_angle((1., 2., -.5), .71)
        pbody = body((.3, -.4, .2), qp, fixed=True)
        # Public admission is near-unit, not arbitrary rescaling. The original
        # n=(.3,-.2,1) fixture was out of contract and is retained below as a
        # rejection check. Inside the admitted band BOTH n and d must rescale.
        source_length = math.sqrt(1.13)
        for factor in (1., 1.+1e-5):
            with self.subTest(normal_scale=factor):
                floor = plane(scale((.3, -.2, 1.), factor/source_length), .17*factor/source_length)
                length = math.sqrt(dot(floor.plane_normal, floor.plane_normal))
                n = rotate(pbody.state[3:7], scale(floor.plane_normal, 1/length))
                offset = floor.plane_offset/length+dot(n, pbody.state[:3])
                q = axis_angle((.2, -.5, 1.), .43)
                free = body((-.1, .2, .1), q)
                vertices = world_vertices(tetra, free)
                minimum = min(dot(n, v)-offset for v in vertices)
                free.state[:3] = add(free.state[:3], scale(n, -.037-minimum))
                vertices = world_vertices(tetra, free)
                expected_depth = -min(dot(n, v)-offset for v in vertices)
                for reverse in (False, True):
                    m = self.query((floor, tetra), (pbody, free), reverse)
                    self.assertGreater(m.count, 0)
                    self.near(m.normal, scale(n, -1 if reverse else 1))
                    self.assertAlmostEqual(max(p.depth for p in m.points[:m.count]), expected_depth, delta=2e-6)
                    for point in m.points[:m.count]:
                        plane_witness = add(point.point, scale(n, .5*point.depth))
                        convex_witness = sub(point.point, scale(n, .5*point.depth))
                        self.assertAlmostEqual(dot(n, plane_witness), offset, delta=2e-6)
                        self.assertLessEqual(min(math.dist(convex_witness, v) for v in vertices), 2e-5)

    def test_out_of_contract_arbitrary_plane_scaling_is_rejected(self):
        for factor in (1., 7.25):
            with self.subTest(arbitrary_scale=factor):
                shapes = (Shape*2)(plane(scale((.3, -.2, 1.), factor), .17*factor), box((.5, .5, .5), 77))
                pairs, bodies = (Pair*1)(Pair(99, 0, 1)), (Body*2)(body(fixed=True), body())
                gravity, friction = (C.c_float*3)(0., 0., 0.), (C.c_float*1)(0.)
                r = Registration(1, 1, 2, 1, shapes, pairs, bodies, gravity, friction)
                h = C.c_void_p()
                status = self.lib.bcv1_create(C.byref(r), C.byref(h))
                if h.value:
                    self.lib.bcv1_destroy(h)
                self.assertEqual(status, 1)
                self.assertFalse(h.value)

    def test_asymmetric_tetra_containment_and_reversed_owner(self):
        a = convex(((.1, .2, -.3), (.6, .1, .2), (-.2, .7, .1), (-.3, -.2, .6)), 1)
        b = box((1.8, 1.6, 1.4), 2)
        bodies = body((.21, -.12, .07), axis_angle((1, 2, 3), .37)), body((-.1, .04, .09), axis_angle((1, -.2, .3), .15))
        forward, best = self.check_convex((a, b), bodies)
        reverse, other = self.check_convex((a, b), bodies, True)
        self.assertGreater(best[0], .8, "containment must use signed exit, not interval intersection width")
        self.assertAlmostEqual(best[0], other[0], delta=1e-12)
        self.near(forward.normal, scale(reverse.normal, -1))
        self.assertAlmostEqual(max(p.depth for p in forward.points[:forward.count]), best[0], delta=2e-5)

    def test_skew_and_near_parallel_true_edge_witnesses(self):
        # At .002 rad a 5mm overlap has a face minimum, not an edge minimum.
        # The separately authored 5um case exercises the intended shallow edge.
        for angle, penetration in ((.4, .005), (.002, .000005)):
            with self.subTest(angle_rad=angle, authored_penetration_m=penetration):
                a, b, ba, bb = edge_fixture(angle, penetration)
                result, best = self.check_convex((a, b), (ba, bb))
                self.assertEqual(best[2], "edge", f"fixture does not select edge SAT: {best}")
                self.assertEqual(result.count, 1)
                va, vb = world_vertices(a, ba), world_vertices(b, bb)
                normal = best[1]
                sa, sb = max(dot(normal, v) for v in va), min(dot(normal, v) for v in vb)
                ea = [v for v in va if abs(dot(normal, v)-sa) <= 1e-7]
                eb = [v for v in vb if abs(dot(normal, v)-sb) <= 1e-7]
                self.assertEqual((len(ea), len(eb)), (2, 2))
                wa, wb, s, t = closest_segments(*ea, *eb)
                self.assertGreater(min(s, t, 1-s, 1-t), .05, "must witness edge interiors")
                point = result.points[0]
                self.near(point.point, scale(add(wa, wb), .5), 3e-5)
                wrong_support_midpoint = scale(add(ea[0], eb[0]), .5)
                self.assertGreater(math.dist(wrong_support_midpoint, point.point), .1, "fixture must detect arbitrary support midpoint")

    def test_deterministic_rotated_convex_sat_32_poses(self):
        a = convex(((-.4, -.3, -.2), (.7, -.2, -.1), (-.1, .65, -.1), (-.1, .05, .6)), 11)
        b = box((.31, .27, .23), 12)
        for index in range(32):
            with self.subTest(pose=index):
                qa = axis_angle((1., .3, -.7), index*.137)
                qb = axis_angle((-.2, 1., .4), -.09-index*.173)
                radial = .15 if index%3 else 1.5
                p = (radial*math.cos(index*.7), radial*math.sin(index*.7), .04*math.sin(index))
                self.check_convex((a, b), (body((.02, -.03, .01), qa), body(p, qb)), bool(index%2))

    def test_null_query_does_not_overwrite_output_canary(self):
        output = Manifold()
        C.memset(C.byref(output), 0xA5, C.sizeof(output))
        before = bytes(output)
        self.assertEqual(self.lib.bcv1_query(None, C.byref(output)), 1)
        self.assertEqual(bytes(output), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
