"""Batched fp32 H0 humanoid lane over the single-source dwc1 kernel (serial).

The humanoid twin of walk/env/cuda_lane.py's CudaDuckLane: physics path
(tick / tick_block / read / restore / state_dump / set_state / query) AND,
since the kernel went robot-generic via the DW_ENV_* contract block, the
device policy path (step_policy / observe / set_command / reset_policy):
obs (52) + humanoid_reward v1 + termination in-kernel, bit-exact vs
FlatFloorHumanoidEnv running over this same lane build. No domain
randomization surface yet (the humanoid env v1 has none); reset_policy
draws command + phase0 with humanoid_flat's exact counter-based stream.

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
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import cuda_lane
from .cuda_lane import (  # reuse the model-size-independent dwc1 ctypes ABI
    DIAG_DTYPE, Diagnostic, Info, Manifold, load_library, _fp, _manifold_json,
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
OBS = 3 * J + 16                       # 52; pinned as DW_ENV_OBS in the header
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
        # dwc1_info's effort_cap is a scalar summary (the MIN tier); the
        # kernel's PD clamp and reward estimate consume the per-joint
        # DW_EFFORT_CAP_TABLE (authored tiers 180/140/70). Expose both.
        self.effort_cap = float(info.effort_cap)
        self.effort_cap_per_joint = np.array(h0.EFFORT, dtype=np.float64)
        # device policy path bookkeeping (mirrors FlatFloorHumanoidEnv seeding)
        self._seed = 0
        self._episode = np.zeros(self.E, np.int64)

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

    # -- device policy path (obs + reward + termination in-kernel) ----------
    def step_policy(self, actions: np.ndarray, n_ticks: int = 10):
        """One full policy step in-kernel: action -> slew-limited targets ->
        n_ticks physics -> humanoid_reward v1 -> termination -> 52-dim obs.

        Returns (obs [E,52] f32, reward [E] f32, done [E] bool, diagnostics
        structured array). A nonzero `status` entry means that env hit a
        solver fault, froze at its last accepted tick and was marked done
        (FlatFloorHumanoidEnv raises SolverFault instead)."""
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

    def observe(self) -> np.ndarray:
        """Current 52-dim observations (no stepping); reset()-style read."""
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
        """Masked policy reset mirroring FlatFloorHumanoidEnv.reset():
        physics and tracker state back to creation; per-env command AND
        per-episode gait phase offset resampled with humanoid_flat's exact
        counter-based (seed, env, episode) stream -- draw = rng.random()
        picks the command with the duck's slowest-oversampling split
        (0.50 if draw < 0.5 else 0.75 if draw < 0.75 else 1.00), then
        phase0 = 2*pi*rng.random(). No randomization draws (the humanoid
        env v1 has no DR surface). Pass `commands`/`phase_offsets` [E] to
        override the drawn values. Returns fresh observations."""
        from .humanoid_flat import COMMANDS_MPS, _episode_rng  # noqa: PLC0415
        if seed is not None:
            self._seed = int(seed)
        m = np.ones(self.E, bool) if mask is None \
            else np.asarray(mask, bool).reshape(self.E)
        cmd = np.zeros(self.E, np.float64)
        ph0 = np.zeros(self.E, np.float64)
        for e in np.flatnonzero(m):
            rng = _episode_rng(self._seed, int(e), int(self._episode[e]) + 1)
            draw = rng.random()
            cmd[e] = (COMMANDS_MPS[0] if draw < 0.5
                      else COMMANDS_MPS[1] if draw < 0.75
                      else COMMANDS_MPS[2])
            ph0[e] = 2.0 * math.pi * rng.random()
            self._episode[e] += 1
        if commands is not None:
            cmd[:] = np.broadcast_to(np.asarray(commands, np.float64),
                                     (self.E,))
        if phase_offsets is not None:
            ph0[:] = np.broadcast_to(np.asarray(phase_offsets, np.float64),
                                     (self.E,))
        mc = (C.c_uint8 * self.E)(*[1 if x else 0 for x in m])
        rc = self._lib.dwc1_reset_policy(self._h, mc, cmd.ctypes.data_as(DP),
                                         ph0.ctypes.data_as(DP))
        if rc:
            raise RuntimeError(f"dwc1_reset_policy status={rc}")
        return self.observe()

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
