"""Batched fp32 duck lane over the single-source CUDA/serial dwc1 library.

Implements the same duck-typed lane surface as walk/env/native_lane.py's
NativeDuckLane, so FlatFloorDuckEnv(lane_factory=...) runs unchanged on it:

    lane.E, lane.J, lane.home_joint_q, lane.joint_limits, lane.home_root_height
    lane.kp / lane.kv / lane.effort_cap / lane.library_path
    lane.tick(targets)   -> (rc, diagnostics per env)
    lane.read()          -> LaneState
    lane.restore(mask)   -> restore masked envs to the creation state
    lane.state_dump(e)   -> JSON-serialisable full state of env e
    lane.close()

Locally this loads the SERIAL build (libduck_cuda_serial.dylib, plain clang++,
one env per loop iteration) compiled from the identical physics header the
real CUDA driver uses; remotely the same wrapper loads libduck_cuda.so built
by experimental/duck_cuda/build_remote.sh (set DUCK_CUDA_LIBRARY or pass
library_path). The physics parity contract vs the f64 CPU lane is documented
in experimental/duck_cuda/include/duck_cuda.h and enforced by
experimental/duck_cuda/tests/test_serial_parity.py.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .native_lane import LaneState

ROOT = Path(__file__).resolve().parents[2]
DUCK_CUDA = ROOT / "experimental" / "duck_cuda"
BUILD_DIR = ROOT / "build"

_FLAGS = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off"]
_SOURCES = [
    DUCK_CUDA / "src" / "duck_cuda_serial.cpp",
    DUCK_CUDA / "src" / "duck_cuda_kernel.h",
    DUCK_CUDA / "include" / "duck_cuda.h",
    DUCK_CUDA / "include" / "duck_model.h",
    DUCK_CUDA / "include" / "cuda_compat.h",
]

J, B, P, Q, N, JROWS = 14, 16, 2, 21, 20, 42
SIM_DT = 0.002
LEFT_FOOT_BODY = 6
RIGHT_FOOT_BODY = 15
FOOT_BODIES = (LEFT_FOOT_BODY, RIGHT_FOOT_BODY)

F = C.c_float
FP = C.POINTER(F)
U8P = C.POINTER(C.c_uint8)
U64P = C.POINTER(C.c_uint64)
DP = C.POINTER(C.c_double)


class Point(C.Structure):
    _fields_ = [("feature", C.c_uint64), ("point", F * 3), ("depth", F),
                ("normal_impulse", F), ("tangent_impulse", F * 2)]


class Manifold(C.Structure):
    _fields_ = [("count", C.c_uint32), ("normal", F * 3), ("tangent1", F * 3),
                ("tangent2", F * 3), ("points", Point * 4)]


class Diagnostic(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in [
        "environment", "status", "iterations", "contact_points",
        "active_limits", "ticks"]] + [(n, F) for n in [
        "joint_residual", "normal_residual", "tangent_residual",
        "momentum_residual", "maximum_normal_impulse", "maximum_penetration"]]


class Info(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in
                ["environments", "bodies", "joints", "dofs"]] + \
               [(n, F) for n in ["dt", "kp", "kv", "effort_cap",
                                 "home_root_height"]] + \
               [("home_qpos", F * Q), ("joint_lower", F * J),
                ("joint_upper", F * J)]


def _source_digest() -> str:
    h = hashlib.sha256()
    for p in _SOURCES:
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def build_library() -> Path:
    """Build (or reuse the cached) serial dwc1 dylib under build/."""
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out = BUILD_DIR / f"libduck_cuda_serial-{_source_digest()}{suffix}"
    if out.is_file():
        return out
    compiler = shutil.which("clang++")
    if not compiler:
        raise RuntimeError("clang++ toolchain required to build the serial duck lane")
    tmp = out.with_suffix(out.suffix + ".tmp")
    cmd = [compiler, *_FLAGS, "-I", str(DUCK_CUDA / "include"), "-fPIC",
           "-shared", str(_SOURCES[0]), "-o", str(tmp)]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    tmp.replace(out)
    return out


def load_library(path: Path):
    lib = C.CDLL(str(path))
    specs = {
        "dwc1_create": [C.c_uint32, FP, C.POINTER(C.c_void_p)],
        "dwc1_destroy": [C.c_void_p],
        "dwc1_info_get": [C.c_void_p, C.POINTER(Info)],
        "dwc1_step": [C.c_void_p, FP, C.c_uint32, C.POINTER(Diagnostic)],
        "dwc1_read": [C.c_void_p, FP, FP, FP, DP, U64P, FP, U8P, FP,
                      C.POINTER(Manifold)],
        "dwc1_reset": [C.c_void_p, U8P],
        "dwc1_set_state": [C.c_void_p, C.c_uint32, FP, FP, FP,
                           C.POINTER(Manifold), C.c_uint64],
        "dwc1_query": [C.c_void_p, C.POINTER(Manifold)],
    }
    for name, args in specs.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = None if name.endswith("destroy") else C.c_int
    return lib


def _fp(a: np.ndarray) -> FP:
    return a.ctypes.data_as(FP)


def _manifold_json(m: Manifold) -> dict:
    return {"count": int(m.count), "normal": list(m.normal),
            "tangent1": list(m.tangent1), "tangent2": list(m.tangent2),
            "points": [{"feature": int(p.feature), "point": list(p.point),
                        "depth": float(p.depth),
                        "normal_impulse": float(p.normal_impulse),
                        "tangent_impulse": list(p.tangent_impulse)}
                       for p in m.points[:m.count]]}


class CudaDuckLane:
    """E batched fp32 flat-floor ducks over the dwc1 C ABI."""

    def __init__(self, environments: int, joint_offsets: np.ndarray | None = None,
                 library_path: str | Path | None = None):
        path = library_path or os.environ.get("DUCK_CUDA_LIBRARY")
        self.library_path = Path(path) if path else build_library()
        self._lib = load_library(self.library_path)
        self.E = int(environments)
        self.J, self.B, self.P = J, B, P
        offsets = None
        if joint_offsets is not None:
            offsets = np.ascontiguousarray(joint_offsets, dtype=np.float32)
            if offsets.shape != (self.E, J):
                raise ValueError("joint_offsets requires shape [E, J]")
        self._h = C.c_void_p()
        rc = self._lib.dwc1_create(
            self.E, _fp(offsets) if offsets is not None else None,
            C.byref(self._h))
        if rc:
            raise ValueError(f"dwc1_create status={rc}")
        info = Info()
        rc = self._lib.dwc1_info_get(self._h, C.byref(info))
        if rc:
            raise RuntimeError(f"dwc1_info_get status={rc}")
        self.joint_limits = np.array(
            [[lo, hi] for lo, hi in zip(info.joint_lower, info.joint_upper)],
            dtype=np.float64)
        self.home_joint_q = np.array(info.home_qpos[7:], dtype=np.float64)
        self.home_root_height = float(info.home_root_height)
        self.kp = float(info.kp)
        self.kv = float(info.kv)
        self.effort_cap = float(info.effort_cap)

    # -- stepping ----------------------------------------------------------
    def tick(self, targets: np.ndarray):
        """One 0.002 s tick of all E envs; returns (rc, diagnostics dicts)."""
        t = np.ascontiguousarray(targets, dtype=np.float32).reshape(self.E, J)
        diag = (Diagnostic * self.E)()
        rc = self._lib.dwc1_step(self._h, _fp(t), 1, diag)
        out = []
        for d in diag:
            row = {name: getattr(d, name) for name, _ in Diagnostic._fields_}
            row["native_status"] = int(d.status)   # NativeDuckLane key
            row["phase"] = 3 if d.status else 6
            out.append(row)
        return rc, out

    # -- reads --------------------------------------------------------------
    def read(self) -> LaneState:
        q = np.zeros((self.E, Q), np.float32)
        v = np.zeros((self.E, N), np.float32)
        t = np.zeros(self.E, np.float64)
        count = np.zeros(self.E, np.uint64)
        body = np.zeros((self.E, B, 13), np.float32)
        contact = np.zeros((self.E, P), np.uint8)
        sole = np.zeros((self.E, P), np.float32)
        rc = self._lib.dwc1_read(
            self._h, _fp(q), _fp(v), None, t.ctypes.data_as(DP),
            count.ctypes.data_as(U64P), _fp(body),
            contact.ctypes.data_as(U8P), _fp(sole), None)
        if rc:
            raise RuntimeError(f"dwc1_read status={rc}")
        feet = body[:, list(FOOT_BODIES), :].astype(np.float64)
        return LaneState(q=q.astype(np.float64), v=v.astype(np.float64),
                         time=t, count=count,
                         body_state=body.astype(np.float64),
                         foot_contact=contact.astype(bool),
                         foot_pos=feet[:, :, :3].copy(),
                         sole_height=sole.astype(np.float64))

    # -- snapshot / reset ----------------------------------------------------
    def restore(self, mask: np.ndarray | None = None) -> None:
        m = None
        if mask is not None:
            flat = np.asarray(mask).reshape(-1)
            if len(flat) != self.E:
                raise ValueError("mask requires length E")
            m = (C.c_uint8 * self.E)(*[1 if x else 0 for x in flat])
        rc = self._lib.dwc1_reset(self._h, m)
        if rc:
            raise RuntimeError(f"dwc1_reset status={rc}")

    def set_state(self, env: int, qpos, velocity, warm_force,
                  cache: list[dict] | None = None, count: int = 0) -> None:
        """Full single-env state injection (fault-corpus replay)."""
        qf = np.ascontiguousarray(qpos, np.float32).reshape(Q)
        vf = np.ascontiguousarray(velocity, np.float32).reshape(N)
        wf = np.ascontiguousarray(warm_force, np.float32).reshape(JROWS)
        cm = None
        if cache is not None:
            cm = (Manifold * P)()
            for p in range(P):
                d = cache[p]
                cm[p].count = int(d["count"])
                cm[p].normal[:] = d["normal"]
                cm[p].tangent1[:] = d["tangent1"]
                cm[p].tangent2[:] = d["tangent2"]
                for i, pt in enumerate(d["points"][:int(d["count"])]):
                    cm[p].points[i].feature = int(pt["feature"])
                    cm[p].points[i].point[:] = pt["point"]
                    cm[p].points[i].depth = float(pt["depth"])
                    cm[p].points[i].normal_impulse = float(pt["normal_impulse"])
                    cm[p].points[i].tangent_impulse[:] = pt["tangent_impulse"]
        rc = self._lib.dwc1_set_state(self._h, int(env), _fp(qf), _fp(vf),
                                      _fp(wf), cm, int(count))
        if rc:
            raise RuntimeError(f"dwc1_set_state status={rc}")

    # -- forensics ------------------------------------------------------------
    def state_dump(self, env: int) -> dict:
        e = int(env)
        q = np.zeros((self.E, Q), np.float32)
        v = np.zeros((self.E, N), np.float32)
        w = np.zeros((self.E, JROWS), np.float32)
        t = np.zeros(self.E, np.float64)
        count = np.zeros(self.E, np.uint64)
        body = np.zeros((self.E, B, 13), np.float32)
        cache = (Manifold * (self.E * P))()
        rc = self._lib.dwc1_read(self._h, _fp(q), _fp(v), _fp(w),
                                 t.ctypes.data_as(DP),
                                 count.ctypes.data_as(U64P), _fp(body),
                                 None, None, cache)
        if rc:
            raise RuntimeError(f"dwc1_read status={rc}")
        geometry = (Manifold * (self.E * P))()
        rc = self._lib.dwc1_query(self._h, geometry)
        if rc:
            raise RuntimeError(f"dwc1_query status={rc}")
        return {
            "qpos": q[e].astype(float).tolist(),
            "velocity": v[e].astype(float).tolist(),
            "warm_force": w[e].astype(float).tolist(),
            "time_s": float(t[e]), "step_count": int(count[e]),
            "bodies": [[float(x) for x in body[e, b]] for b in range(B)],
            "pre_contact_cache": [_manifold_json(cache[e * P + p])
                                  for p in range(P)],
            "current_geometry": [_manifold_json(geometry[e * P + p])
                                 for p in range(P)],
        }

    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lib.dwc1_destroy(self._h)
            self._h = None
