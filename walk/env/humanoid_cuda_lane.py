"""Batched fp32 H0 humanoid lane over the single-source dwc1 kernel (serial).

The humanoid twin of walk/env/cuda_lane.py's CudaDuckLane, PHYSICS PATH ONLY
(tick / tick_block / read / restore / state_dump / set_state / query): the
kernel's device policy layer is duck-specific and not valid for this model
(see experimental/duck_cuda/tools/generate_model_humanoid.py and
humanoid/FEASIBILITY.md section 1.4), so no step_policy/observe here.

Build: the UNCHANGED kernel sources (src/duck_cuda_serial.cpp ->
src/duck_cuda_kernel.h) compiled with -Ihumanoid/include ahead of
-Iexperimental/duck_cuda/include, so the generated humanoid
humanoid/include/duck_model.h shadows the duck header. Zero kernel edits;
the duck's own build and committed header are untouched.

The f64 CPU oracle is walk/env/humanoid_native_lane.py; parity gates live in
humanoid/tests/test_humanoid_serial_parity.py.

NOTE dwc1_read's `time` output is hardcoded `count * 0.002` in
duck_cuda_serial.cpp (a duck literal, cosmetic only -- physics uses DW_DT);
this lane recomputes time as count * h0.SIM_DT instead of reading it.
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

from . import cuda_lane
from .cuda_lane import (  # reuse the model-size-independent dwc1 ctypes ABI
    Diagnostic, Info, Manifold, load_library, _fp, _manifold_json,
)
from .cuda_lane import CudaLaneState

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "humanoid") not in sys.path:
    sys.path.insert(0, str(ROOT / "humanoid"))
import h0_lowering as h0  # noqa: E402

DUCK_CUDA = ROOT / "experimental" / "duck_cuda"
HUMANOID_INCLUDE = ROOT / "humanoid" / "include"
BUILD_DIR = ROOT / "build"

# Identical flags AND identical fp32 certificates to the duck build
# (DW_SOLVE_TOLERANCE 5e-6 / DW_MOMENTUM_TOLERANCE 2e-4 defaults): the
# home-hold regime converges to the duck absolutes despite the humanoid's
# much larger impulse scale. Perturbed standing targets -- which stalled
# both lanes before the workstream-A solver repair (humanoid/FEASIBILITY.md
# section 6, now resolved) -- tick through clean and are gate-pinned by
# humanoid/tests/test_humanoid_serial_parity.py.
_FLAGS = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-ffp-contract=off"]
_SOURCES = [
    DUCK_CUDA / "src" / "duck_cuda_serial.cpp",
    DUCK_CUDA / "src" / "duck_cuda_kernel.h",
    DUCK_CUDA / "include" / "duck_cuda.h",
    DUCK_CUDA / "include" / "cuda_compat.h",
    HUMANOID_INCLUDE / "duck_model.h",          # the humanoid shadow header
]

J, B, P, Q, N, JROWS = h0.J, h0.B, 2, 7 + h0.J, 6 + h0.J, 3 * h0.J
SIM_DT = h0.SIM_DT
FOOT_BODIES = h0.FOOT_BODIES

FP = cuda_lane.FP
U8P = cuda_lane.U8P
U32P = cuda_lane.U32P
U64P = cuda_lane.U64P
DP = cuda_lane.DP


def _source_digest() -> str:
    h = hashlib.sha256()
    h.update(" ".join(_FLAGS).encode())      # tolerance flags are part of it
    for p in _SOURCES:
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def build_library() -> Path:
    """Build (or reuse the cached) serial HUMANOID dwc1 dylib under build/."""
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out = BUILD_DIR / f"libhumanoid_cuda_serial-{_source_digest()}{suffix}"
    if out.is_file():
        return out
    compiler = shutil.which("clang++")
    if not compiler:
        raise RuntimeError(
            "clang++ toolchain required to build the serial humanoid lane")
    tmp = out.with_suffix(out.suffix + ".tmp")
    cmd = [compiler, *_FLAGS,
           "-I", str(HUMANOID_INCLUDE),        # MUST precede the duck include
           "-I", str(DUCK_CUDA / "include"),
           "-fPIC", "-shared", str(_SOURCES[0]), "-o", str(tmp)]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    tmp.replace(out)
    return out


class CudaHumanoidLane:
    """E batched fp32 flat-floor H0 humanoids over the dwc1 C ABI (serial)."""

    def __init__(self, environments: int,
                 joint_offsets: np.ndarray | None = None,
                 library_path: str | Path | None = None):
        path = library_path or os.environ.get("HUMANOID_CUDA_LIBRARY")
        self.library_path = Path(path) if path else build_library()
        self._lib = load_library(self.library_path)
        abi = int(self._lib.dwc1_abi_version())
        if abi != cuda_lane.ABI_VERSION:
            raise RuntimeError(f"{self.library_path} exports dwc1 ABI v{abi}; "
                               f"expected v{cuda_lane.ABI_VERSION}")
        if not 1 <= int(environments) <= 65536:
            raise ValueError("environments must be in [1, 65536]")
        self.E = int(environments)
        self.J, self.B, self.P = J, B, P
        offsets = None
        if joint_offsets is not None:
            offsets = np.ascontiguousarray(joint_offsets, dtype=np.float32)
            if offsets.shape != (self.E, J):
                raise ValueError("joint_offsets requires shape [E, J]")
        self._h = C.c_void_p()
        rc = self._lib.dwc1_create(
            self.E, _fp(offsets) if offsets is not None else None, None,
            C.byref(self._h))
        if rc:
            raise ValueError(f"dwc1_create status={rc}")
        info = Info()
        rc = self._lib.dwc1_info_get(self._h, C.byref(info))
        if rc:
            raise RuntimeError(f"dwc1_info_get status={rc}")
        if (info.bodies, info.joints, info.dofs) != (B, J, N):
            raise RuntimeError(
                f"{self.library_path} is not a humanoid build: "
                f"B{info.bodies}/J{info.joints}/N{info.dofs}")
        self.joint_limits = np.array(
            [[lo, hi] for lo, hi in zip(info.joint_lower[:J],
                                        info.joint_upper[:J])])
        self.home_joint_q = np.array(info.home_qpos[7:Q], dtype=np.float64)
        self.home_root_height = float(info.home_root_height)
        self.kp = float(info.kp)
        self.kv = float(info.kv)
        self.effort_cap = float(info.effort_cap)  # scalar MIN tier; the
        # authored per-joint tiers live in h0_lowering.EFFORT (see the
        # generator docstring for the Phase 2 kernel edit).

    # -- stepping ----------------------------------------------------------
    def tick_block(self, targets: np.ndarray, n_ticks: int):
        t = np.ascontiguousarray(targets, dtype=np.float32).reshape(self.E, J)
        diag = (Diagnostic * self.E)()
        rc = self._lib.dwc1_step(self._h, _fp(t), int(n_ticks), diag)
        out = []
        for d in diag:
            row = {name: getattr(d, name) for name, _ in Diagnostic._fields_}
            row["native_status"] = int(d.status)
            row["phase"] = 3 if d.status else 6
            out.append(row)
        return rc, out

    def tick(self, targets: np.ndarray):
        return self.tick_block(targets, 1)

    # -- reads --------------------------------------------------------------
    def read(self) -> CudaLaneState:
        qpos = np.empty((self.E, Q), np.float32)
        vel = np.empty((self.E, N), np.float32)
        warm = np.empty((self.E, JROWS), np.float32)
        count = np.empty(self.E, np.uint64)
        body = np.empty((self.E, B, 13), np.float32)
        contact = np.empty((self.E, P), np.uint8)
        sole = np.empty((self.E, P), np.float32)
        ticks = np.empty((self.E, P), np.uint32)
        rc = self._lib.dwc1_read(
            self._h, _fp(qpos), _fp(vel), _fp(warm), None,
            count.ctypes.data_as(U64P), _fp(body),
            contact.ctypes.data_as(U8P), _fp(sole), None,
            ticks.ctypes.data_as(U32P))
        if rc:
            raise RuntimeError(f"dwc1_read status={rc}")
        feet = body[:, list(FOOT_BODIES), :].astype("d")
        return CudaLaneState(
            q=qpos.astype("d"), v=vel.astype("d"),
            time=count.astype("d") * SIM_DT,     # serial `time` is duck-dt
            count=count, body_state=body.astype("d"),
            foot_contact=contact.astype(bool),
            foot_pos=feet[:, :, :3].copy(), sole_height=sole.astype("d"),
            contact_ticks=ticks.astype(np.int32))

    # -- state injection / snapshot ------------------------------------------
    def set_state(self, env: int, qpos, velocity, warm, cache=None, count=0):
        q = np.ascontiguousarray(qpos, np.float32).reshape(Q)
        v = np.ascontiguousarray(velocity, np.float32).reshape(N)
        w = np.ascontiguousarray(warm, np.float32).reshape(JROWS)
        c2 = None
        if cache is not None:
            c2 = (Manifold * P)()
            for p, m in enumerate(cache):
                c2[p].count = int(m["count"])
                c2[p].normal[:] = m["normal"]
                c2[p].tangent1[:] = m["tangent1"]
                c2[p].tangent2[:] = m["tangent2"]
                for k, pt in enumerate(m["points"][:4]):
                    c2[p].points[k].feature = int(pt["feature"])
                    c2[p].points[k].point[:] = pt["point"]
                    c2[p].points[k].depth = float(pt["depth"])
                    c2[p].points[k].normal_impulse = float(pt["normal_impulse"])
                    c2[p].points[k].tangent_impulse[:] = pt["tangent_impulse"]
        rc = self._lib.dwc1_set_state(self._h, int(env), _fp(q), _fp(v),
                                      _fp(w), c2, int(count))
        if rc:
            raise RuntimeError(f"dwc1_set_state status={rc}")

    def restore(self, mask: np.ndarray | None = None) -> None:
        m = None
        if mask is not None:
            arr = np.asarray(mask).reshape(-1)
            if len(arr) != self.E:
                raise ValueError("mask requires length E")
            m = (C.c_uint8 * self.E)(*[int(bool(x)) for x in arr])
        rc = self._lib.dwc1_reset(self._h, m)
        if rc:
            raise RuntimeError(f"dwc1_reset status={rc}")

    # -- forensics ------------------------------------------------------------
    def state_dump(self, env: int) -> dict:
        x = self.read()
        cache = (Manifold * (self.E * P))()
        rc = self._lib.dwc1_read(self._h, None, None, None, None, None, None,
                                 None, None, cache, None)
        if rc:
            raise RuntimeError(f"dwc1_read status={rc}")
        e = int(env)
        return {"qpos": x.q[e].tolist(), "velocity": x.v[e].tolist(),
                "time_s": float(x.time[e]), "step_count": int(x.count[e]),
                "pre_contact_cache": [_manifold_json(cache[e * P + p])
                                      for p in range(P)]}

    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lib.dwc1_destroy(self._h)
            self._h = None
