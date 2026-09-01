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

Fast path beyond the documented protocol (native_lane.py documents no
optional fast-path convention, so the names below are the convention):

    lane.tick_block(targets, n_ticks) -> (rc, diagnostics per env)
        ONE dwc1_step call = one device kernel launch covering all n_ticks
        (PD recomputed per tick from current q/qdot, targets held), and
    lane.read().contact_ticks  (CudaLaneState, [E, 2] int32)
        per-foot count of accepted contact ticks during the most recent
        step call -- the per-tick foot-contact accumulation the reward's
        flicker penalty needs, without a device->host read per tick.

    With the env stepping via tick_block(targets, 10) + one read(), a policy
    step costs 1 kernel launch + 1 state readback instead of 10 + 10.

Device policy path (ABI v3): observation + reward + termination in-kernel,
removing the full-state readback entirely for training:

    lane.step_policy(actions, n_ticks=10)
        -> (obs [E,58] f32, reward [E] f32, done [E] bool, diagnostics)
    lane.reset_policy(mask, seed=..., commands=..., phase_offsets=...) -> obs
        (flat.py-exact counter-based per-episode resampling: command from
        {0.10, 0.15, 0.20} m/s, then the v10 gait-phase offset 2*pi*random())
    lane.observe() -> obs;  lane.set_command(commands) -> obs

    walk/env/flat.py and walk/env/reward.py are the contract; the
    in-kernel policy chain runs in f64 mirroring numpy operation for
    operation, so obs/reward/done are bit-identical to FlatFloorDuckEnv
    running over the same lane build (verified by
    experimental/duck_cuda/tests/test_serial_parity.py). Per policy step:
    1 kernel launch, one tiny H2D (actions) and one tiny D2H
    (obs+reward+done+diagnostics = 296 B/env); no state readback.

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
import dataclasses
import hashlib
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .native_lane import LaneState

ABI_VERSION = 5  # must match DWC1_ABI_VERSION in duck_cuda.h

OBS = 58
COMMANDS_MPS = (0.10, 0.15, 0.20)   # flat.py per-episode forward commands


def _episode_rng(seed: int, env: int, episode: int) -> np.random.Generator:
    """flat.py's counter-based RNG, replicated exactly (command sampling)."""
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, int(env), int(episode)])


@dataclasses.dataclass
class CudaLaneState(LaneState):
    """LaneState plus the per-foot contact tick counters of the last step."""
    contact_ticks: np.ndarray = None  # [E, 2] int32, (left, right)

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
U32P = C.POINTER(C.c_uint32)
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


# numpy view of Diagnostic[E] (packed: 6 x u32 + 6 x f32, no padding); the
# training hot path gets a structured array instead of E python dicts.
DIAG_DTYPE = np.dtype([(name, "u4") for name in [
    "environment", "status", "iterations", "contact_points",
    "active_limits", "ticks"]] + [(name, "f4") for name in [
    "joint_residual", "normal_residual", "tangent_residual",
    "momentum_residual", "maximum_normal_impulse", "maximum_penetration"]])
assert DIAG_DTYPE.itemsize == C.sizeof(Diagnostic)


class Info(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in
                ["environments", "bodies", "joints", "dofs"]] + \
               [(n, F) for n in ["dt", "kp", "kv", "effort_cap",
                                 "home_root_height"]] + \
               [("home_qpos", F * Q), ("joint_lower", F * J),
                ("joint_upper", F * J)]


class DeviceInfo(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in [
        "lanes_per_env", "threads_per_block", "min_blocks_per_sm", "sm_count",
        "step_regs_per_thread", "step_local_bytes", "step_blocks_per_sm",
        "policy_regs_per_thread", "policy_local_bytes",
        "policy_blocks_per_sm"]] + \
        [(n, C.c_uint64) for n in ["stack_limit_bytes",
                                   "workspace_bytes_per_env"]] + \
        [(n, C.c_uint32) for n in ["resident_envs_estimate", "reserved"]]


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
                      C.POINTER(Manifold), U32P],
        "dwc1_step_policy": [C.c_void_p, FP, C.c_uint32, FP, FP, U8P,
                             C.POINTER(Diagnostic)],
        "dwc1_observe": [C.c_void_p, FP],
        "dwc1_set_command": [C.c_void_p, DP],
        "dwc1_reset_policy": [C.c_void_p, U8P, DP, DP],
        "dwc1_device_info_get": [C.c_void_p, C.POINTER(DeviceInfo)],
        "dwc1_abi_version": [],
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
        abi = int(self._lib.dwc1_abi_version())
        if abi != ABI_VERSION:
            raise RuntimeError(
                f"{self.library_path} exports dwc1 ABI v{abi}; "
                f"this wrapper requires v{ABI_VERSION} (rebuild the library)")
        # The CUDA build is warp-per-env (32 lanes/env): throughput saturates
        # around E=8192-16384 on an RTX 5090; larger batches only add memory
        # (~sizeof(DwWork) ~ 100 KB workspace per env).
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
        # device policy path bookkeeping (mirrors FlatFloorDuckEnv seeding)
        self._seed = 0
        self._episode = np.zeros(self.E, np.int64)

    # -- stepping ----------------------------------------------------------
    def tick_block(self, targets: np.ndarray, n_ticks: int):
        """n_ticks 0.002 s ticks of all E envs holding `targets`, in ONE
        dwc1_step call (one device kernel launch); (rc, diagnostics dicts).

        Per-foot contact tick counters (read().contact_ticks) are zeroed at
        the start of the call and count this call's accepted contact ticks.
        """
        t = np.ascontiguousarray(targets, dtype=np.float32).reshape(self.E, J)
        diag = (Diagnostic * self.E)()
        rc = self._lib.dwc1_step(self._h, _fp(t), int(n_ticks), diag)
        out = []
        for d in diag:
            row = {name: getattr(d, name) for name, _ in Diagnostic._fields_}
            row["native_status"] = int(d.status)   # NativeDuckLane key
            row["phase"] = 3 if d.status else 6
            out.append(row)
        return rc, out

    def tick(self, targets: np.ndarray):
        """One 0.002 s tick of all E envs; returns (rc, diagnostics dicts)."""
        return self.tick_block(targets, 1)

    # -- device policy path (obs + reward + termination in-kernel) ----------
    def step_policy(self, actions: np.ndarray, n_ticks: int = 10):
        """One full policy step on device: action -> slew-limited targets ->
        n_ticks physics -> reward.py -> termination -> 58-dim observation.

        Returns (obs [E,58] f32, reward [E] f32, done [E] bool, diagnostics)
        where diagnostics is a numpy structured array (DIAG_DTYPE); a nonzero
        `status` entry means that env hit a solver fault, froze at its last
        accepted tick and was marked done (the python env raises SolverFault
        instead). One kernel launch + one small readback per call.
        """
        a = np.ascontiguousarray(actions, dtype=np.float32).reshape(self.E, J)
        obs = np.empty((self.E, OBS), np.float32)
        reward = np.empty(self.E, np.float32)
        done = np.empty(self.E, np.uint8)
        diag = (Diagnostic * self.E)()
        rc = self._lib.dwc1_step_policy(self._h, _fp(a), int(n_ticks),
                                        _fp(obs), _fp(reward),
                                        done.ctypes.data_as(U8P), diag)
        if rc:
            raise RuntimeError(f"dwc1_step_policy status={rc}")
        diagnostics = np.frombuffer(diag, dtype=DIAG_DTYPE).copy()
        return obs, reward, done.astype(bool), diagnostics

    def device_info(self) -> dict:
        """Launch/occupancy telemetry (zeros for GPU-only fields on serial)."""
        info = DeviceInfo()
        rc = self._lib.dwc1_device_info_get(self._h, C.byref(info))
        if rc:
            raise RuntimeError(f"dwc1_device_info_get status={rc}")
        return {name: int(getattr(info, name))
                for name, _ in DeviceInfo._fields_ if name != "reserved"}

    def observe(self) -> np.ndarray:
        """Current 58-dim observations (no stepping); reset()-style read."""
        obs = np.empty((self.E, OBS), np.float32)
        rc = self._lib.dwc1_observe(self._h, _fp(obs))
        if rc:
            raise RuntimeError(f"dwc1_observe status={rc}")
        return obs

    def set_command(self, commands) -> np.ndarray:
        """Override every env's commanded forward velocity (m/s)."""
        c = np.ascontiguousarray(
            np.broadcast_to(np.asarray(commands, np.float64), (self.E,)))
        rc = self._lib.dwc1_set_command(self._h, c.ctypes.data_as(DP))
        if rc:
            raise RuntimeError(f"dwc1_set_command status={rc}")
        return self.observe()

    def reset_policy(self, mask: np.ndarray | None = None,
                     seed: int | None = None, commands=None,
                     phase_offsets=None) -> np.ndarray:
        """Masked policy reset mirroring FlatFloorDuckEnv.reset(): physics and
        tracker state back to creation; per-env command AND per-episode gait
        phase offset resampled with flat.py's exact counter-based (seed, env,
        episode) RNG -- one stream per episode, command drawn first, then
        phase0 = 2*pi*rng.random() (v10). Pass `commands`/`phase_offsets` [E]
        to override the drawn values. Returns fresh observations."""
        if seed is not None:
            self._seed = int(seed)
        m = np.ones(self.E, bool) if mask is None \
            else np.asarray(mask, bool).reshape(self.E)
        cmd = np.zeros(self.E, np.float64)
        ph0 = np.zeros(self.E, np.float64)
        for e in np.flatnonzero(m):
            rng = _episode_rng(self._seed, int(e), int(self._episode[e]) + 1)
            cmd[e] = COMMANDS_MPS[rng.integers(len(COMMANDS_MPS))]
            ph0[e] = 2.0 * math.pi * rng.random()
            self._episode[e] += 1
        if commands is not None:
            cmd[:] = np.broadcast_to(np.asarray(commands, np.float64), (self.E,))
        if phase_offsets is not None:
            ph0[:] = np.broadcast_to(
                np.asarray(phase_offsets, np.float64), (self.E,))
        mc = (C.c_uint8 * self.E)(*[1 if x else 0 for x in m])
        rc = self._lib.dwc1_reset_policy(self._h, mc, cmd.ctypes.data_as(DP),
                                         ph0.ctypes.data_as(DP))
        if rc:
            raise RuntimeError(f"dwc1_reset_policy status={rc}")
        return self.observe()

    # -- reads --------------------------------------------------------------
    def read(self) -> CudaLaneState:
        q = np.zeros((self.E, Q), np.float32)
        v = np.zeros((self.E, N), np.float32)
        t = np.zeros(self.E, np.float64)
        count = np.zeros(self.E, np.uint64)
        body = np.zeros((self.E, B, 13), np.float32)
        contact = np.zeros((self.E, P), np.uint8)
        sole = np.zeros((self.E, P), np.float32)
        ticks = np.zeros((self.E, P), np.uint32)
        rc = self._lib.dwc1_read(
            self._h, _fp(q), _fp(v), None, t.ctypes.data_as(DP),
            count.ctypes.data_as(U64P), _fp(body),
            contact.ctypes.data_as(U8P), _fp(sole), None,
            ticks.ctypes.data_as(U32P))
        if rc:
            raise RuntimeError(f"dwc1_read status={rc}")
        feet = body[:, list(FOOT_BODIES), :].astype(np.float64)
        return CudaLaneState(q=q.astype(np.float64), v=v.astype(np.float64),
                             time=t, count=count,
                             body_state=body.astype(np.float64),
                             foot_contact=contact.astype(bool),
                             foot_pos=feet[:, :, :3].copy(),
                             sole_height=sole.astype(np.float64),
                             contact_ticks=ticks.astype(np.int32))

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
        ticks = np.zeros((self.E, P), np.uint32)
        cache = (Manifold * (self.E * P))()
        rc = self._lib.dwc1_read(self._h, _fp(q), _fp(v), _fp(w),
                                 t.ctypes.data_as(DP),
                                 count.ctypes.data_as(U64P), _fp(body),
                                 None, None, cache,
                                 ticks.ctypes.data_as(U32P))
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
            "contact_ticks": [int(x) for x in ticks[e]],
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
