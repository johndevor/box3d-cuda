"""Host-ABI transaction tests: no robot, solver step, simulator, or GPU."""
import ctypes as C
import hashlib
import itertools
import math
import os
from pathlib import Path
import struct
import sys
import unittest

F3 = C.c_float*3


class Body(C.Structure):
    _fields_ = [("state", C.c_float*13), ("inverse_mass", C.c_float), ("inverse_inertia", F3)]


class Shape(C.Structure):
    _fields_ = [("caller_id", C.c_uint32), ("kind", C.c_uint32), ("vertex_count", C.c_uint32), ("fixed", C.c_uint32),
                ("vertices", F3*32), ("plane_normal", F3), ("plane_offset", C.c_float)]


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


def clone(array):
    return type(array).from_buffer_copy(bytes(array))


def manifold_bytes(m):
    # C struct padding is unspecified; compare every declared field instead.
    data = struct.pack("<I", m.count)+bytes(m.normal)+bytes(m.tangent1)+bytes(m.tangent2)
    for p in m.points:
        data += struct.pack("<Q", p.feature)+bytes(p.point)+struct.pack("<fff", p.depth, p.normal_impulse, p.tangent_impulse[0])+struct.pack("<f", p.tangent_impulse[1])
    return data


def environment_bytes(payload, e):
    bodies, cache, clocks = payload
    return bytes(bodies[2*e])+bytes(bodies[2*e+1])+manifold_bytes(cache[e])+struct.pack("<d", clocks[e])


class ContactTransactions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("CONTACT_V1_LIBRARY"):
            raise RuntimeError("set CONTACT_V1_LIBRARY to a freshly built library; use run_local.py")
        cls.path = Path(os.environ["CONTACT_V1_LIBRARY"]).resolve()
        cls.digest = hashlib.sha256(cls.path.read_bytes()).hexdigest()
        print("transaction_library="+str(cls.path)+" sha256="+cls.digest, file=sys.stderr, flush=True)
        cls.lib = C.CDLL(str(cls.path))
        P, V = C.POINTER, C.c_void_p
        signatures = {
            "bcv1_create": ([P(Registration), P(V)], C.c_int), "bcv1_destroy": ([V], None),
            "bcv1_read": ([V, P(Body), P(Manifold), P(C.c_double)], C.c_int),
            "bcv1_query": ([V, P(Manifold)], C.c_int),
            "bcv1_capture": ([V, P(V)], C.c_int), "bcv1_snapshot_destroy": ([V], None),
            "bcv1_restore": ([V, V, P(C.c_uint8)], C.c_int),
            "bcx1_prepare_solved": ([V, P(Body), P(Manifold), C.c_double, P(V)], C.c_int),
            "bcx1_prepare_restore": ([V, V, P(C.c_uint8), P(V)], C.c_int),
            "bcx1_stage_read": ([V, P(Body), P(Manifold), P(C.c_double)], C.c_int),
            "bcx1_stage_query": ([V, P(Manifold)], C.c_int),
            "bcx1_validate_commit": ([V, V], C.c_int), "bcx1_commit": ([V, V], C.c_int),
            "bcx1_stage_destroy": ([V], None),
        }
        for name, (args, result) in signatures.items():
            getattr(cls.lib, name).argtypes, getattr(cls.lib, name).restype = args, result

    @classmethod
    def tearDownClass(cls):
        if hashlib.sha256(cls.path.read_bytes()).hexdigest() != cls.digest:
            raise AssertionError("library changed during transaction test")

    def setUp(self):
        self.scenes, self.stages, self.snapshots = [], [], []

    def tearDown(self):
        # Never leave a stage with a dead owner.
        for stage in reversed(self.stages):
            self.lib.bcx1_stage_destroy(stage)
        for snapshot in reversed(self.snapshots):
            self.lib.bcv1_snapshot_destroy(snapshot)
        for scene in reversed(self.scenes):
            self.lib.bcv1_destroy(scene)

    def scene(self, identity=77):
        floor = Shape(caller_id=10, kind=2, fixed=1, plane_normal=F3(0, 0, 1))
        box = Shape(caller_id=identity, kind=1, vertex_count=8)
        for i, vertex in enumerate(itertools.product((-.5, .5), repeat=3)):
            box.vertices[i][:] = vertex
        shapes, pairs = (Shape*2)(floor, box), (Pair*1)(Pair(88, 0, 1))
        bodies = (Body*4)()
        for e in range(2):
            bodies[2*e].state[6] = bodies[2*e+1].state[6] = 1.
            bodies[2*e+1].state[:3] = (e*2., 0., .5)
            bodies[2*e+1].inverse_mass = 1.+e
            bodies[2*e+1].inverse_inertia[:] = (1.+e, 2.+e, 3.+e)
        gravity, mu = (C.c_float*6)(0, 0, -9, 0, 0, -3), (C.c_float*2)(.6, .8)
        registration, handle = Registration(1, 2, 2, 1, shapes, pairs, bodies, gravity, mu), C.c_void_p()
        self.assertEqual(self.lib.bcv1_create(C.byref(registration), C.byref(handle)), 0)
        self.assertTrue(handle.value)
        self.scenes.append(handle)
        return handle

    def read(self, handle, stage=False):
        result = (Body*4)(), (Manifold*2)(), (C.c_double*2)()
        function = self.lib.bcx1_stage_read if stage else self.lib.bcv1_read
        self.assertEqual(function(handle, *result), 0)
        return result

    def query(self, handle, stage=False):
        result = (Manifold*2)()
        function = self.lib.bcx1_stage_query if stage else self.lib.bcv1_query
        self.assertEqual(function(handle, result), 0)
        return result

    def snapshot(self, scene):
        out = C.c_void_p()
        self.assertEqual(self.lib.bcv1_capture(scene, C.byref(out)), 0)
        self.snapshots.append(out)
        return out

    def packet(self, scene):
        bodies = self.read(scene)[0]
        cache = self.query(scene)
        for e in range(2):
            bodies[2*e+1].state[2] += .2+e*.1
            bodies[2*e+1].state[7] = .25+e
            for point in cache[e].points[:cache[e].count]:
                point.normal_impulse = .5+e
                point.tangent_impulse[:] = (.1, -.05)
        return bodies, cache

    def prepare(self, scene, packet=None, dt=.00200000012345678):
        bodies, cache = packet or self.packet(scene)
        out = C.c_void_p()
        self.assertEqual(self.lib.bcx1_prepare_solved(scene, bodies, cache, dt, C.byref(out)), 0)
        self.stages.append(out)
        return out

    def equal(self, a, b):
        for e in range(2):
            self.assertEqual(environment_bytes(a, e), environment_bytes(b, e), f"environment {e} changed")

    def test_private_prepare_pre_cache_post_query_copy_and_double_clock(self):
        scene = self.scene()
        before = self.read(scene)
        packet = self.packet(scene)
        self.assertEqual([m.count for m in packet[1]], [4, 4])
        dt = .00200000012345678
        self.assertNotEqual(dt, C.c_float(dt).value)
        stage = self.prepare(scene, packet, dt)
        self.equal(self.read(scene), before)
        staged = self.read(stage, True)
        self.assertEqual(bytes(staged[0]), bytes(packet[0]))
        self.assertEqual([manifold_bytes(x) for x in staged[1]], [manifold_bytes(x) for x in packet[1]])
        self.assertEqual(list(staged[2]), [dt, dt])
        self.assertEqual([m.count for m in self.query(stage, True)], [0, 0], "query must use POST geometry, not solved PRE cache")
        packet[0][1].state[0] = 99.
        packet[1][0].points[0].normal_impulse = 99.
        self.equal(self.read(stage, True), staged)
        self.assertEqual(self.lib.bcx1_validate_commit(scene, stage), 0)
        self.assertEqual(self.lib.bcx1_commit(scene, stage), 0)
        self.equal(self.read(scene), staged)

    def test_wrong_pre_cache_second_environment_and_invalid_post_rollback(self):
        scene = self.scene()
        before = self.read(scene)
        mutations = {
            "PRE_count": lambda b, m: setattr(m[1], "count", 3),
            "PRE_point": lambda b, m: m[1].points[0].point.__setitem__(0, 12.),
            "PRE_feature": lambda b, m: setattr(m[1].points[0], "feature", 123456),
            "PRE_depth": lambda b, m: setattr(m[1].points[0], "depth", .1),
            "PRE_normal": lambda b, m: m[1].normal.__setitem__(0, .5),
            "POST_nonfinite": lambda b, m: b[3].state.__setitem__(7, math.nan),
            "POST_mass": lambda b, m: setattr(b[3], "inverse_mass", 99.),
            "POST_fixed": lambda b, m: b[2].state.__setitem__(7, 1.),
            "POST_second_fixed_floor_pose": lambda b, m: b[2].state.__setitem__(2, .125),
            "negative_impulse": lambda b, m: setattr(m[1].points[0], "normal_impulse", -1.),
            "friction_disk": lambda b, m: m[1].points[0].tangent_impulse.__setitem__(0, 99.),
        }
        for name, mutation in mutations.items():
            with self.subTest(mutation=name):
                b, m = self.packet(scene)
                mutation(b, m)
                caller_before = bytes(b), bytes(m)
                out = C.c_void_p(0x12345678)
                self.assertNotEqual(self.lib.bcx1_prepare_solved(scene, b, m, .002, C.byref(out)), 0)
                self.assertEqual(out.value, 0x12345678)
                self.assertEqual((bytes(b), bytes(m)), caller_before)
                self.equal(self.read(scene), before)

    def test_invalid_dt_and_null_inputs_preserve_stage_pointer(self):
        scene = self.scene()
        b, m = self.packet(scene)
        before = self.read(scene)
        for dt in (0., -.001, .01001, math.nan, math.inf):
            with self.subTest(dt=dt):
                out = C.c_void_p(1234)
                self.assertEqual(self.lib.bcx1_prepare_solved(scene, b, m, dt, C.byref(out)), 1)
                self.assertEqual(out.value, 1234)
                self.equal(self.read(scene), before)
        for owner, bodies, cache in ((None, b, m), (scene, None, m), (scene, b, None)):
            out = C.c_void_p(1234)
            self.assertEqual(self.lib.bcx1_prepare_solved(owner, bodies, cache, .002, C.byref(out)), 1)
            self.assertEqual(out.value, 1234)

    def test_stale_wrong_owner_and_double_commit(self):
        a, b = self.scene(), self.scene()
        stage = self.prepare(a)
        other = self.prepare(a)
        before_a, before_b = self.read(a), self.read(b)
        self.assertEqual(self.lib.bcx1_validate_commit(b, stage), 6)
        self.assertEqual(self.lib.bcx1_commit(b, stage), 6)
        self.equal(self.read(a), before_a); self.equal(self.read(b), before_b)
        self.assertEqual(self.lib.bcx1_commit(a, other), 0)
        committed = self.read(a)
        self.assertEqual(self.lib.bcx1_validate_commit(a, stage), 6)
        self.assertEqual(self.lib.bcx1_commit(a, stage), 6)
        self.assertEqual(self.lib.bcx1_validate_commit(a, other), 6)
        self.assertEqual(self.lib.bcx1_commit(a, other), 6)
        self.equal(self.read(a), committed)
        out = (Manifold*2)()
        C.memset(out, 0xA5, C.sizeof(out))
        canary = bytes(out)
        self.assertEqual(self.lib.bcx1_stage_query(other, out), 1)
        self.assertEqual(bytes(out), canary)
        self.assertEqual(self.lib.bcx1_stage_read(other, None, out, None), 1)
        self.assertEqual(bytes(out), canary)

    def test_restore_nonzero_masks_stage_privacy_and_exact_peer(self):
        for mask_values in ((255, 0), (0, 7), (0, 0), (1, 255)):
            with self.subTest(mask=mask_values):
                scene = self.scene()
                initial = self.read(scene)
                snapshot = self.snapshot(scene)
                solved = self.prepare(scene)
                self.assertEqual(self.lib.bcx1_commit(scene, solved), 0)
                final = self.read(scene)
                mask = (C.c_uint8*2)(*mask_values)
                out = C.c_void_p()
                self.assertEqual(self.lib.bcx1_prepare_restore(scene, snapshot, mask, C.byref(out)), 0)
                self.stages.append(out)
                self.equal(self.read(scene), final)
                restored = self.read(out, True)
                for e, selected in enumerate(mask_values):
                    self.assertEqual(environment_bytes(restored, e), environment_bytes(initial if selected else final, e))
                mask[:] = (0, 0)  # prepare copied selection; caller can reuse mask.
                self.assertEqual(self.lib.bcx1_commit(scene, out), 0)
                self.equal(self.read(scene), restored)
                self.assertEqual(self.lib.bcv1_restore(scene, snapshot, None), 0)
                self.equal(self.read(scene), initial)

    def test_restore_wrong_topology_unchanged_pointer_and_scene(self):
        scene, wrong = self.scene(), self.scene(identity=78)
        snapshot = self.snapshot(wrong)
        before = self.read(scene)
        out = C.c_void_p(4321)
        self.assertEqual(self.lib.bcx1_prepare_restore(scene, snapshot, None, C.byref(out)), 4)
        self.assertEqual(out.value, 4321)
        self.equal(self.read(scene), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
