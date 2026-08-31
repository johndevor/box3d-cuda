"""Thin host ctypes binding for the experimental combined CPU lane.

No PyTorch, simulator, rendering, training, or hidden stepping. Uses the sealed
model lowerer and AV2 descriptor definitions; tests can author tiny fixtures.
"""
import ctypes as C
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experimental/articulated_v2/tests'))
import api as av
sys.path.insert(0, str(ROOT/'experimental/contact_v1'))
import model_translation as contact
D = C.c_double; DP = C.POINTER(D); U = C.c_uint32; P = C.c_void_p


class Registration(C.Structure):
    _fields_ = [('articulation', C.POINTER(av.Registration)), ('pairs', U), ('reserved', U),
                ('shapes', C.POINTER(contact.Shape)), ('contact_pairs', C.POINTER(contact.Pair)),
                ('friction', C.POINTER(C.c_float))]


class Diagnostic(C.Structure):
    _fields_ = [(n, U) for n in ['environment', 'phase', 'native_status', 'iterations',
                               'contact_points', 'active_limits']] + [(n, D) for n in [
        'joint_residual', 'normal_residual', 'tangent_residual', 'momentum_residual',
        'maximum_normal_impulse', 'maximum_penetration']]


def library(path=None):
    lib = av.library(path or os.environ['INTEGRATED_DUCK_LIBRARY'])
    specs = {'idv1_create': [C.POINTER(Registration), C.POINTER(P)], 'idv1_destroy': [P],
             'idv1_step': [P, C.POINTER(av.Step), U, D, C.POINTER(Diagnostic)],
             'idv1_read': [P, DP, DP, DP, DP, C.POINTER(C.c_uint64), C.POINTER(contact.Body),
                           C.POINTER(contact.Manifold), C.POINTER(contact.Manifold)],
             'idv1_capture': [P, C.POINTER(P)], 'idv1_snapshot_destroy': [P],
             'idv1_restore': [P, P, C.POINTER(C.c_uint8)], 'idv1_reset': [P, C.POINTER(C.c_uint8)]}
    for name, args in specs.items():
        fn = getattr(lib, name); fn.argtypes = args
        fn.restype = None if name.endswith('destroy') else C.c_int
    return lib


class State:
    def __init__(self, s):
        self.q = np.zeros((s.E, 7+s.J)); self.v = np.zeros((s.E, s.N))
        self.warm = np.zeros((s.E, s.J*3)); self.time = np.zeros(s.E)
        self.count = np.zeros(s.E, dtype='uint64')
        self.bodies = (contact.Body*(s.E*s.B))()
        self.cache = (contact.Manifold*(s.E*s.P))()
        self.geometry = (contact.Manifold*(s.E*s.P))()
        self.E = s.E; self.B = s.B; self.P = s.P

    def bytes(self, environment=None):
        if environment is None:
            return b''.join(getattr(self, n).tobytes() for n in ['q', 'v', 'warm', 'time', 'count']) + bytes(self.bodies) + bytes(self.cache) + bytes(self.geometry)
        e = environment
        return b''.join(getattr(self, n)[e].tobytes() for n in ['q', 'v', 'warm', 'time', 'count']) + b''.join(bytes(x) for x in self.bodies[e*self.B:(e+1)*self.B]) + b''.join(bytes(x) for x in self.cache[e*self.P:(e+1)*self.P]) + b''.join(bytes(x) for x in self.geometry[e*self.P:(e+1)*self.P])


class Scene:
    def __init__(self, lib, fixture, q, v, shapes, pairs, friction, gravity=None, limits=None):
        self.lib = lib; self.f = fixture; self.E = len(q); self.J = fixture.J
        self.N = fixture.N; self.B = fixture.B; self.P = len(pairs); self.snapshots = []
        self.q = av.checked(q, 'd', (self.E, self.J+7), 'q')
        self.v = av.checked(v, 'd', (self.E, self.N), 'v')
        self.gravity = av.checked(gravity if gravity is not None else [[0,0,-9.81]]*self.E, 'd', (self.E, 3), 'gravity')
        self.limits = av.limits(fixture) if limits is None else limits
        self.shapes = shapes; self.pairs = pairs
        self.mu = av.checked(friction, 'f', (self.E, self.P), 'friction')
        a = av.desc(av.Registration); a.environments = self.E; a.model = C.pointer(fixture.model)
        a.limits = self.limits; a.q = av.dp(self.q); a.v = av.dp(self.v); a.gravity = av.dp(self.gravity)
        self.registration = a
        self.desc = Registration(C.pointer(a), self.P, 0, shapes, pairs, self.mu.ctypes.data_as(C.POINTER(C.c_float)))
        self.h = P(); rc = lib.idv1_create(C.byref(self.desc), C.byref(self.h))
        if rc: raise ValueError('idv1_create status='+str(rc))

    def close(self):
        for p in self.snapshots: self.lib.idv1_snapshot_destroy(p)
        self.snapshots.clear()
        if self.h: self.lib.idv1_destroy(self.h); self.h = None

    def read(self):
        x = State(self)
        rc = self.lib.idv1_read(self.h, av.dp(x.q), av.dp(x.v), av.dp(x.warm), av.dp(x.time),
                              x.count.ctypes.data_as(C.POINTER(C.c_uint64)), x.bodies, x.cache, x.geometry)
        if rc: raise RuntimeError('idv1_read status='+str(rc))
        return x

    def step(self, dt=.002, target=None, target_velocity=None, force=None, max_iterations=4096, tolerance=1e-8):
        target = np.zeros((self.E, self.J)) if target is None else av.checked(target, 'd', (self.E, self.J), 'target')
        tv = np.zeros_like(target) if target_velocity is None else av.checked(target_velocity, 'd', target.shape, 'target_velocity')
        d = av.desc(av.Step); d.dt = dt; d.mtol = d.jtol = 1e-8
        d.target = av.dp(target); d.targetv = av.dp(tv)
        if force is not None:
            force = av.checked(force, 'd', (self.E, self.N), 'force'); d.force = av.dp(force)
        diagnostics = (Diagnostic*self.E)()
        rc = self.lib.idv1_step(self.h, C.byref(d), max_iterations, tolerance, diagnostics)
        return rc, [{n: getattr(x, n) for n, _ in Diagnostic._fields_} for x in diagnostics]

    def capture(self):
        p = P(); rc = self.lib.idv1_capture(self.h, C.byref(p))
        if rc: raise RuntimeError('idv1_capture status='+str(rc))
        self.snapshots.append(p); return p

    def reset(self, mask=None, snapshot=None):
        a = None if mask is None else (C.c_uint8*self.E)(*mask)
        return self.lib.idv1_restore(self.h, snapshot, a) if snapshot else self.lib.idv1_reset(self.h, a)


def shapes_for_fixture(f):
    shapes = (contact.Shape*f.B)()
    for b, shape in enumerate(shapes): shape.caller_id=b; shape.fixed=int(b==0)
    shapes[0].kind=2; shapes[0].plane_normal[2]=1
    return shapes


def box(shape, half=(.1,.1,.1)):
    shape.kind=1; shape.vertex_count=8
    for i in range(8): shape.vertices[i][:]=[half[k]*(1 if i&(1<<k) else -1) for k in range(3)]


def duck_scene(lib, environments=1):
    """Load exact accepted floor-clear reset; no model time advancement."""
    f = av.duck(); cm = contact.Model(contact.library(lib._name))
    frame = cm.record['frames'][0]
    q = np.array([*frame['base_pose'][:3], *frame['base_pose'][4:], frame['base_pose'][3], *frame['joint_q']])
    v = np.array(frame['qvel'], dtype='d'); v[3:6] = av.rot(q[3:7])@v[3:6]
    if q[2] != .16788827542191784: raise ValueError('floor-clear reset pin mismatch')
    return Scene(lib, f, np.tile(q,(environments,1)), np.tile(v,(environments,1)), cm.shapes, cm.pairs,
                 np.tile(cm.mu,(environments,1))), cm
