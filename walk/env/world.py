"""ctypes wrapper for duck_world_v1 (dwv1): duck + cube grid + floor per env.

Mirrors experimental/integrated_duck_v1/native.py. Also carries the build
helper that compiles libduck_world with exactly the clang++ flags
run_local.py uses, and duck_grid_scene() which reuses native.py's Open Duck
lowering for the articulation/shape part. No torch, no simulator, no network.
"""
import ctypes as C
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experimental/integrated_duck_v1'))
import native  # noqa: E402  (adds av2 api + contact model_translation paths)
av = native.av
contact = native.contact
D = C.c_double; DP = C.POINTER(D); U = C.c_uint32; P = C.c_void_p; U8 = C.c_uint8

FLAGS = ['-std=c++17', '-Wall', '-Wextra', '-Werror', '-ffp-contract=off', '-O2']
INCLUDES = ['experimental/duck_world_v1/include', 'experimental/integrated_duck_v1/include',
            'experimental/contact_v1/include', 'experimental/articulated_v1/include',
            'experimental/articulated_v2/include', 'include', 'csrc']
UNITS = ['experimental/duck_world_v1/src/duck_world_v1.cpp',
         'experimental/integrated_duck_v1/src/coupled_impulse_v1.cpp',
         'experimental/integrated_duck_v1/src/integrated_duck_v1.cpp',
         'experimental/contact_v1/src/contact_v1.cpp',
         'experimental/articulated_v1/src/articulated_v1.cpp',
         'experimental/articulated_v2/src/articulated_v2.cpp',
         'csrc/experimental_joint_v1.cpp']
HEADERS = ['experimental/duck_world_v1/include/duck_world_v1.h',
           'experimental/duck_world_v1/src/dwv1_geometry.h']


def build(output=None):
    """Compile libduck_world.dylib with run_local.py's flags; cached by source hash."""
    compiler = shutil.which('clang++')
    if not compiler:
        raise RuntimeError('existing clang toolchain required')
    digest = hashlib.sha256()
    for rel in UNITS+HEADERS:
        digest.update((ROOT/rel).read_bytes())
    name = 'libduck_world.dylib' if sys.platform == 'darwin' else 'libduck_world.so'
    out = Path(output) if output else Path(tempfile.gettempdir())/('dwv1-'+digest.hexdigest()[:16])
    out.mkdir(parents=True, exist_ok=True)
    lib = out/name
    if not lib.exists():
        command = [compiler, *FLAGS, '-fPIC', '-shared',
                   *[x for p in INCLUDES for x in ['-I', str(ROOT/p)]],
                   *[str(ROOT/u) for u in UNITS], '-o', str(lib)]
        subprocess.run(command, check=True, cwd=ROOT)
    return lib


class Grid(C.Structure):
    _fields_ = [(n, U) for n in ['nx', 'nz', 'dynamic', 'reserved']] + [(n, D) for n in [
        'cube_size', 'spacing', 'base_height', 'height_jitter', 'origin_x', 'origin_y',
        'cube_mass', 'friction']] + [('seed', C.c_uint64)]


def grid_spec(nx=4, nz=4, cube_size=.06, spacing=0., base_height=0., height_jitter=0.,
              origin_x=0., origin_y=0., dynamic=0, cube_mass=.1, friction=.8, seed=0):
    return Grid(nx, nz, dynamic, 0, cube_size, spacing, base_height, height_jitter,
                origin_x, origin_y, cube_mass, friction, seed)


class Registration(C.Structure):
    _fields_ = [('articulation', C.POINTER(av.Registration)), ('pairs', U), ('reserved', U),
                ('shapes', C.POINTER(contact.Shape)), ('contact_pairs', C.POINTER(contact.Pair)),
                ('friction', C.POINTER(C.c_float)), ('grid', Grid)]


class Diagnostic(C.Structure):
    _fields_ = [(n, U) for n in ['environment', 'phase', 'native_status', 'iterations',
                                 'contact_points', 'active_limits', 'islands', 'awake_cubes',
                                 'duck_island_cubes', 'max_island_dofs', 'reserved0',
                                 'reserved1']] + [(n, D) for n in [
        'joint_residual', 'normal_residual', 'tangent_residual', 'momentum_residual',
        'maximum_normal_impulse', 'maximum_penetration']]


class Contact(C.Structure):
    _fields_ = [(n, U) for n in ['kind_a', 'index_a', 'kind_b', 'index_b']] + [
        ('manifold', contact.Manifold)]


def library(path):
    lib = native.library(path)
    specs = {'dwv1_create': [C.POINTER(Registration), C.POINTER(P)], 'dwv1_destroy': [P],
             'dwv1_step': [P, C.POINTER(av.Step), U, D, C.POINTER(Diagnostic)],
             'dwv1_read': [P, DP, DP, DP, DP, C.POINTER(C.c_uint64), C.POINTER(contact.Body),
                           C.POINTER(contact.Manifold), C.POINTER(contact.Manifold),
                           DP, DP, C.POINTER(U8), C.POINTER(U8)],
             'dwv1_query': [P, U, C.POINTER(Contact), U, C.POINTER(U)],
             'dwv1_override_cube': [P, U, U, DP, DP],
             'dwv1_capture': [P, C.POINTER(P)], 'dwv1_snapshot_destroy': [P],
             'dwv1_restore': [P, P, C.POINTER(U8)], 'dwv1_reset': [P, C.POINTER(U8)]}
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
        self.cube_pose = np.zeros((s.E, s.M, 7)); self.cube_velocity = np.zeros((s.E, s.M, 6))
        self.cube_awake = np.zeros((s.E, s.M), dtype='uint8')
        self.foot = np.zeros((s.E, s.F), dtype='uint8')
        self.E = s.E; self.B = s.B; self.P = s.P; self.M = s.M; self.F = s.F

    def bytes(self, environment=None):
        names = ['q', 'v', 'warm', 'time', 'count', 'cube_pose', 'cube_velocity',
                 'cube_awake', 'foot']
        if environment is None:
            return b''.join(getattr(self, n).tobytes() for n in names) + bytes(self.bodies) \
                + bytes(self.cache) + bytes(self.geometry)
        e = environment
        return b''.join(getattr(self, n)[e].tobytes() for n in names) \
            + b''.join(bytes(x) for x in self.bodies[e*self.B:(e+1)*self.B]) \
            + b''.join(bytes(x) for x in self.cache[e*self.P:(e+1)*self.P]) \
            + b''.join(bytes(x) for x in self.geometry[e*self.P:(e+1)*self.P])


class Scene:
    def __init__(self, lib, fixture, q, v, shapes, pairs, friction, grid, gravity=None,
                 limits=None):
        self.lib = lib; self.f = fixture; self.E = len(q); self.J = fixture.J
        self.N = fixture.N; self.B = fixture.B; self.P = len(pairs); self.snapshots = []
        self.grid = grid; self.M = grid.nx*grid.nz
        self.F = sum(1 for s in shapes if s.kind == 1 and not s.fixed)
        self.q = av.checked(q, 'd', (self.E, self.J+7), 'q')
        self.v = av.checked(v, 'd', (self.E, self.N), 'v')
        self.gravity = av.checked(gravity if gravity is not None else [[0, 0, -9.81]]*self.E,
                                  'd', (self.E, 3), 'gravity')
        self.limits = av.limits(fixture) if limits is None else limits
        self.shapes = shapes; self.pairs = pairs
        self.mu = av.checked(friction, 'f', (self.E, self.P), 'friction')
        a = av.desc(av.Registration); a.environments = self.E; a.model = C.pointer(fixture.model)
        a.limits = self.limits; a.q = av.dp(self.q); a.v = av.dp(self.v); a.gravity = av.dp(self.gravity)
        self.registration = a
        self.desc = Registration(C.pointer(a), self.P, 0, shapes, pairs,
                                 self.mu.ctypes.data_as(C.POINTER(C.c_float)), grid)
        self.h = P(); rc = lib.dwv1_create(C.byref(self.desc), C.byref(self.h))
        if rc: raise ValueError('dwv1_create status='+str(rc))

    def close(self):
        for p in self.snapshots: self.lib.dwv1_snapshot_destroy(p)
        self.snapshots.clear()
        if self.h: self.lib.dwv1_destroy(self.h); self.h = None

    def read(self):
        x = State(self)
        rc = self.lib.dwv1_read(
            self.h, av.dp(x.q), av.dp(x.v), av.dp(x.warm), av.dp(x.time),
            x.count.ctypes.data_as(C.POINTER(C.c_uint64)), x.bodies, x.cache, x.geometry,
            av.dp(x.cube_pose), av.dp(x.cube_velocity),
            x.cube_awake.ctypes.data_as(C.POINTER(U8)), x.foot.ctypes.data_as(C.POINTER(U8)))
        if rc: raise RuntimeError('dwv1_read status='+str(rc))
        return x

    def step(self, dt=.002, target=None, target_velocity=None, force=None,
             max_iterations=16384, tolerance=1e-8, jtol=1e-8):
        # mtol (momentum) stays pinned at 1e-8 per PLAN; jtol (joint KKT
        # verification in av2_complete) may be loosened to match a looser civ1
        # impulse tolerance while workstream A's civ1 stall repair is pending.
        target = np.zeros((self.E, self.J)) if target is None else av.checked(
            target, 'd', (self.E, self.J), 'target')
        tv = np.zeros_like(target) if target_velocity is None else av.checked(
            target_velocity, 'd', target.shape, 'target_velocity')
        d = av.desc(av.Step); d.dt = dt; d.mtol = 1e-8; d.jtol = jtol
        d.target = av.dp(target); d.targetv = av.dp(tv)
        if force is not None:
            force = av.checked(force, 'd', (self.E, self.N), 'force'); d.force = av.dp(force)
        diagnostics = (Diagnostic*self.E)()
        rc = self.lib.dwv1_step(self.h, C.byref(d), max_iterations, tolerance, diagnostics)
        return rc, [{n: getattr(x, n) for n, _ in Diagnostic._fields_} for x in diagnostics]

    def query(self, environment=0):
        count = U()
        rc = self.lib.dwv1_query(self.h, environment, None, 0, C.byref(count))
        if rc: raise RuntimeError('dwv1_query status='+str(rc))
        out = (Contact*max(1, count.value))()
        rc = self.lib.dwv1_query(self.h, environment, out, count.value, C.byref(count))
        if rc: raise RuntimeError('dwv1_query status='+str(rc))
        return list(out[:count.value])

    def override_cube(self, environment, cube, pose, velocity):
        pose = av.checked(pose, 'd', (7,), 'pose')
        velocity = av.checked(velocity, 'd', (6,), 'velocity')
        return self.lib.dwv1_override_cube(self.h, environment, cube, av.dp(pose),
                                           av.dp(velocity))

    def capture(self):
        p = P(); rc = self.lib.dwv1_capture(self.h, C.byref(p))
        if rc: raise RuntimeError('dwv1_capture status='+str(rc))
        self.snapshots.append(p); return p

    def reset(self, mask=None, snapshot=None):
        a = None if mask is None else (U8*self.E)(*mask)
        return self.lib.dwv1_restore(self.h, snapshot, a) if snapshot else \
            self.lib.dwv1_reset(self.h, a)


def duck_grid_scene(lib, environments=1, grid=None, lift=None, gap=.0015):
    """Open Duck (native.py lowering) above/next to a cube grid.

    lift=None raises the pinned floor-clear reset by the tallest possible cube
    top plus gap, so the feet start `gap` above the grid; lift=0 keeps the
    duck on the floor (place the grid elsewhere via origin_x/origin_y).
    """
    grid = grid if grid is not None else grid_spec()
    f = av.duck(); cm = contact.Model(contact.library(lib._name))
    frame = cm.record['frames'][0]
    q = np.array([*frame['base_pose'][:3], *frame['base_pose'][4:], frame['base_pose'][3],
                  *frame['joint_q']])
    v = np.array(frame['qvel'], dtype='d'); v[3:6] = av.rot(q[3:7])@v[3:6]
    if q[2] != .16788827542191784: raise ValueError('floor-clear reset pin mismatch')
    q[2] += (grid.base_height+grid.height_jitter+grid.cube_size+gap) if lift is None else lift
    return Scene(lib, f, np.tile(q, (environments, 1)), np.tile(v, (environments, 1)),
                 cm.shapes, cm.pairs, np.tile(cm.mu, (environments, 1)), grid), cm


HOME = (.002, .053, -.63, 1.368, -.784, 0., 0., 0., 0., -.003, -.065, .635, 1.379, -.796)
