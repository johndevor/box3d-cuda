"""Batched fp32 fixed-base arm lane over the single-source dwc1 kernel (serial).

The arm twin of walk/env/humanoid_cuda_lane.py, PHYSICS PATH ONLY: the
UNCHANGED kernel sources (src/duck_cuda_serial.cpp -> src/duck_cuda_kernel.h)
compiled with -Iarm/include/<variant> ahead of -Iexperimental/duck_cuda/
include, so the generated arm header (experimental/duck_cuda/tools/
generate_model_arm.py) shadows the duck header. Zero kernel edits; the duck
and humanoid builds are untouched.

FIXED BASE (structural weld): the generated header parents joint 0 on the
STATIC FLOOR body 0 (DW_HINGE_PARENT[0] == 0), which dw_evaluate supports
as-is (body 0 is initialised to identity pose / zero motion / colmask 0 and
parents are read through the table). The kernel's mandatory free root
(body 1) is a decoupled PHANTOM carrying the base_link's mass: it free-falls
under gravity and never interacts with the arm (its 6-dof mass block is
exactly block-diagonal from the arm's, and no joint or contact row has a
root column). Readers must ignore body 1; the arm's world frame is the
floor's (base_link at the origin).

NO device policy path: dwc1_step_policy / dwc1_observe implement the duck /
humanoid 3*J+16 observation and locomotion reward; the arm's 27-dim obs
(target / tip / delta) and reach reward are NOT expressible in the kernel's
DW_ENV_* contract block. step_policy raises; walk/env/arm_reach.py runs the
obs/reward chain python-side over tick_block. arm/FEASIBILITY.md section 6
states the exact kernel extension that would enable a device path.

The f64 CPU oracle is walk/env/arm_native_lane.py; parity gates live in
arm/tests/test_arm_serial_parity.py.
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
from .cuda_lane import Diagnostic, Info, Manifold, load_library, _fp
from .arm_native_lane import ArmLaneState

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "arm") not in sys.path:
    sys.path.insert(0, str(ROOT / "arm"))
import arm_lowering as al  # noqa: E402

DUCK_CUDA = ROOT / "experimental" / "duck_cuda"
ARM_INCLUDE = ROOT / "arm" / "include"
BUILD_DIR = ROOT / "build"

_FLAGS = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-ffp-contract=off"]
_KERNEL_SOURCES = [
    DUCK_CUDA / "src" / "duck_cuda_serial.cpp",
    DUCK_CUDA / "src" / "duck_cuda_kernel.h",
    DUCK_CUDA / "include" / "duck_cuda.h",
    DUCK_CUDA / "include" / "cuda_compat.h",
]

J, B, P, Q, N, JROWS = al.J, al.B, 2, al.Q, al.N, 3 * al.J
SIM_DT = al.SIM_DT
FP, U8P, U32P, U64P = cuda_lane.FP, cuda_lane.U8P, cuda_lane.U32P, cuda_lane.U64P


def header_path(variant: str) -> Path:
    return ARM_INCLUDE / al.spec(variant).variant / "duck_model.h"


def _source_digest(variant: str) -> str:
    h = hashlib.sha256()
    h.update(" ".join(_FLAGS).encode())
    for p in _KERNEL_SOURCES + [header_path(variant)]:
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def build_library(variant: str) -> Path:
    """Build (or reuse the cached) serial ARM dwc1 dylib for `variant`."""
    variant = al.spec(variant).variant
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out = BUILD_DIR / f"libarm_{variant}_cuda_serial-{_source_digest(variant)}{suffix}"
    if out.is_file():
        return out
    compiler = shutil.which("clang++")
    if not compiler:
        raise RuntimeError("clang++ toolchain required to build the serial arm lane")
    tmp = out.with_suffix(out.suffix + ".tmp")
    cmd = [compiler, *_FLAGS,
           "-I", str(header_path(variant).parent),   # MUST precede duck include
           "-I", str(DUCK_CUDA / "include"),
           "-fPIC", "-shared", str(_KERNEL_SOURCES[0]), "-o", str(tmp)]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    tmp.replace(out)
    return out


class CudaArmLane:
    """E batched fp32 fixed-base arms over the dwc1 C ABI (serial build)."""

    def __init__(self, environments: int, variant: str = "kr240",
                 joint_offsets: np.ndarray | None = None,
                 library_path: str | Path | None = None,
                 fast_termination: bool = False):
        self.spec = al.spec(variant)
        self.variant = self.spec.variant
        path = library_path or os.environ.get(f"ARM_{self.variant.upper()}_CUDA_LIBRARY")
        self.library_path = Path(path) if path else build_library(self.variant)
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
                f"{self.library_path} is not an arm build: "
                f"B{info.bodies}/J{info.joints}/N{info.dofs}")
        # fast_termination only matters for the (absent) device policy path;
        # accepted for lane-surface compatibility, forwarded for symmetry.
        self.fast_termination = bool(fast_termination)
        if self.fast_termination:
            self._lib.dwc1_set_fast_termination(self._h, 1)
        self.joint_limits = np.array(
            [[lo, hi] for lo, hi in zip(info.joint_lower[:J],
                                        info.joint_upper[:J])])
        self.home_joint_q = np.array(info.home_qpos[7:Q], dtype=np.float64)
        self.home_root_height = float(info.home_root_height)
        kp, kv = al.gains(self.spec)
        self.kp = kp                        # per-joint tables (== header)
        self.kv = kv
        self.kp_scalar = float(info.kp)
        self.kv_scalar = float(info.kv)
        self.effort_cap = al.effort(self.spec)
        self.effort_cap_scalar = float(info.effort_cap)
        self.velocity_limits = al.velocity_limits(self.spec)

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

    def step_policy(self, *_args, **_kwargs):
        raise NotImplementedError(
            "the dwc1 device policy layer implements the duck/humanoid "
            "3*J+16 locomotion contract; the arm's 27-dim reach obs is "
            "python-side (walk/env/arm_reach.py) -- see arm/FEASIBILITY.md "
            "section 6 for the kernel extension it would need")

    # -- reads --------------------------------------------------------------
    def read(self) -> ArmLaneState:
        qpos = np.empty((self.E, Q), np.float32)
        vel = np.empty((self.E, N), np.float32)
        warm = np.empty((self.E, JROWS), np.float32)
        count = np.empty(self.E, np.uint64)
        body = np.empty((self.E, B, 13), np.float32)
        rc = self._lib.dwc1_read(
            self._h, _fp(qpos), _fp(vel), _fp(warm), None,
            count.ctypes.data_as(U64P), _fp(body), None, None, None, None)
        if rc:
            raise RuntimeError(f"dwc1_read status={rc}")
        return ArmLaneState(q=qpos.astype("d"), v=vel.astype("d"),
                            time=count.astype("d") * SIM_DT, count=count,
                            body_state=body.astype("d"))

    def contact_points(self) -> np.ndarray:
        """[E,2] placeholder-pair manifold counts (must always be 0)."""
        contact = np.empty((self.E, P), np.uint8)
        rc = self._lib.dwc1_read(self._h, None, None, None, None, None, None,
                                 contact.ctypes.data_as(U8P), None, None, None)
        if rc:
            raise RuntimeError(f"dwc1_read status={rc}")
        return contact

    # -- state injection / snapshot ------------------------------------------
    def set_state(self, env: int, qpos, velocity, warm, count=0):
        q = np.ascontiguousarray(qpos, np.float32).reshape(Q)
        v = np.ascontiguousarray(velocity, np.float32).reshape(N)
        w = np.ascontiguousarray(warm, np.float32).reshape(JROWS)
        rc = self._lib.dwc1_set_state(self._h, int(env), _fp(q), _fp(v),
                                      _fp(w), None, int(count))
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
        e = int(env)
        return {"variant": self.variant, "qpos": x.q[e].tolist(),
                "velocity": x.v[e].tolist(), "time_s": float(x.time[e]),
                "step_count": int(x.count[e]),
                "bodies": x.body_state[e].tolist()}

    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lib.dwc1_destroy(self._h)
            self._h = None


# silence the unused-import linter for the re-exported Manifold type
_ = Manifold
