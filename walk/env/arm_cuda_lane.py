"""Batched fp32 fixed-base arm lane over the single-source dwc1 kernel (serial).

The arm twin of walk/env/humanoid_cuda_lane.py: the PHYSICS path (tick /
tick_block / read / restore / set_state) AND, since kernel ABI v8, the
DEVICE POLICY PATH (step_policy / observe / reset_policy / gate_proxy /
reach_state): the kernel sources (src/duck_cuda_serial.cpp ->
src/duck_cuda_kernel.h) compiled with -Iarm/include/<variant> ahead of
-Iexperimental/duck_cuda/include, so the generated arm header
(experimental/duck_cuda/tools/generate_model_arm.py, `#define DW_ENV_KIND 1`
= DW_ENV_KIND_REACH) shadows the duck header and selects the kernel's REACH
policy layer. Duck and humanoid builds compile the unchanged LOCOMOTION
layer (fingerprint-pinned).

FIXED BASE (structural weld): the generated header parents joint 0 on the
STATIC FLOOR body 0 (DW_HINGE_PARENT[0] == 0), which dw_evaluate supports
as-is (body 0 is initialised to identity pose / zero motion / colmask 0 and
parents are read through the table). The kernel's mandatory free root
(body 1) is a decoupled PHANTOM carrying the base_link's mass: it free-falls
under gravity and never interacts with the arm (its 6-dof mass block is
exactly block-diagonal from the arm's, and no joint or contact row has a
root column). Readers must ignore body 1; the arm's world frame is the
floor's (base_link at the origin).

DEVICE POLICY PATH (ABI v8, DW_ENV_KIND_REACH). walk/env/arm_reach.py +
arm_reward.py are the contract; the kernel mirrors them in f64 over the
same fp32 physics (obs 27 = [q, 0.25*qd, target, tip, target-tip, prev
action]; limit-scaled slew; the frozen judge's acquisition rule; reward v1;
proxy/non-finite/horizon termination). The TARGET SEQUENCE is host-drawn
here with arm_reach's exact counter-based stream (arm_reach.episode_draw:
per (seed, env, episode) the tier, then targets in acquisition order) and
pushed through dwc1_reach_set_targets: at reset the active AND one queued
target; after every step, envs whose target_index advanced (dwc1_reach_get)
get the following draw queued, so the kernel always holds the next target
at acquisition time and the presented sequence is bit-identical to
ArmReachEnv's (which draws lazily from the same rng). Host cost: one
sample_target (~0.5 ms) per acquisition + two per reset.

    lane.step_policy(actions)  -> (obs [E,27] f32, reward [E] f32, done [E]
                                   bool, diagnostics)
    lane.reset_policy(mask, seed=..., tiers=...) -> obs
    lane.observe() -> obs;  lane.reach_state() -> REACH_STATE_DTYPE array
    lane.gate_proxy() -> the shared dwc1_gate_proxy struct in its REACH
        mapping (qualified_left = targets acquired, alternation_violations
        = judge-clause violating ticks), so gpu_train's gate_proxy_* metrics
        and the curriculum knobs (set_gate_termination: first-acquisition
        deadline / violating-tick cap) work unchanged.

Parity (arm/tests/test_arm_device_policy.py): obs/reward/done vs
ArmReachEnv over this same build; the only arithmetic that is not
operation-for-operation identical is numpy's 3-term rotation products in
the tip / link-origin geometry (BLAS association) -- ULP-level f64 noise.

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
from .cuda_lane import (DIAG_DTYPE, GATE_PROXY_DTYPE, REACH_STATE_DTYPE,
                        Diagnostic, GateProxy, Info, Manifold, ReachState,
                        load_library, _fp)
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
DP = cuda_lane.DP


def _obs_dim() -> int:
    from . import arm_reach  # noqa: PLC0415  (the python contract)
    return int(arm_reach.OBS)


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

    OBS = 27   # arm_reach.OBS; cross-checked against dwc1_obs_width below

    def __init__(self, environments: int, variant: str = "kr240",
                 joint_offsets: np.ndarray | None = None,
                 library_path: str | Path | None = None,
                 fast_termination: bool = False,
                 randomization: dict | None = None,
                 tier: int | None = None):
        self.spec = al.spec(variant)
        self.variant = self.spec.variant
        path = library_path or os.environ.get(f"ARM_{self.variant.upper()}_CUDA_LIBRARY")
        self.library_path = Path(path) if path else build_library(self.variant)
        self._lib = load_library(self.library_path)
        abi = int(self._lib.dwc1_abi_version())
        if abi != cuda_lane.ABI_VERSION:
            raise RuntimeError(f"{self.library_path} exports dwc1 ABI v{abi}; "
                               f"expected v{cuda_lane.ABI_VERSION}")
        kind = int(self._lib.dwc1_env_kind())
        if kind != cuda_lane.ENV_KIND_REACH:
            raise RuntimeError(f"{self.library_path} compiled env kind {kind}; "
                               "the arm lane needs DW_ENV_KIND_REACH (an arm "
                               "header build)")
        self.OBS = _obs_dim()
        if int(self._lib.dwc1_obs_width()) != self.OBS:
            raise RuntimeError(f"{self.library_path} writes "
                               f"{int(self._lib.dwc1_obs_width())}-wide obs; "
                               f"arm_reach.OBS is {self.OBS}")
        if not 1 <= int(environments) <= 65536:
            raise ValueError("environments must be in [1, 65536]")
        if randomization:
            # The kernel's DR (mass/kp/damping/gravity/latency) is model-
            # generic, but ArmReachEnv has no python mirror of it yet, so
            # the parity contract cannot be stated; refuse rather than run
            # an unverified stream.
            raise NotImplementedError(
                "domain randomization is not specified for the arm lane")
        self.randomization = None
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
        # Training lanes freeze crashed envs (no solver work on ticks that
        # teach nothing); parity lanes keep the default OFF so both sides
        # of a comparison run identical physics.
        self.fast_termination = bool(fast_termination)
        if self.fast_termination:
            rc = self._lib.dwc1_set_fast_termination(self._h, 1)
            if rc:
                raise RuntimeError(f"dwc1_set_fast_termination status={rc}")
        # device policy path bookkeeping (mirrors ArmReachEnv seeding)
        self._seed = 0
        self._episode = np.zeros(self.E, np.int64)
        self._tier_pin = None if tier is None else int(tier)
        self._tier = np.zeros(self.E, np.int64)
        self._rng = [None] * self.E
        self._index = np.zeros(self.E, np.int64)   # host shadow of target_index
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

    # -- device policy path (obs + reward + termination in-kernel) ----------
    def step_policy(self, actions: np.ndarray, n_ticks: int = 10):
        """One full policy step in-kernel (DW_ENV_KIND_REACH): action ->
        limit-scaled slew targets -> n_ticks physics -> tip FK ->
        acquisition -> arm_reward v1 -> termination -> 27-dim obs; then the
        host queues the following target for every env that acquired one.

        Returns (obs [E,27] f32, reward [E] f32, done [E] bool, diagnostics
        structured array). A nonzero `status` entry means that env hit a
        solver fault, froze at its last accepted tick and was marked done
        (ArmReachEnv raises SolverFault instead)."""
        a = np.ascontiguousarray(actions, dtype=np.float32).reshape(self.E, J)
        obs = np.empty((self.E, self.OBS), np.float32)
        reward = np.empty(self.E, np.float32)
        done = np.empty(self.E, np.uint8)
        diag = (Diagnostic * self.E)()
        rc = self._lib.dwc1_step_policy(self._h, _fp(a), int(n_ticks),
                                        _fp(obs), _fp(reward),
                                        done.ctypes.data_as(U8P), diag)
        if rc:
            raise RuntimeError(f"dwc1_step_policy status={rc}")
        self._refill_queue()
        diagnostics = np.frombuffer(diag, dtype=DIAG_DTYPE).copy()
        return obs, reward, done.astype(bool), diagnostics

    def _refill_queue(self) -> None:
        """Queue the next host draw for every env whose target_index
        advanced since the last call (exactly arm_reach's lazy draw order,
        one target ahead)."""
        rs = self.reach_state()
        idx = rs["target_index"].astype(np.int64)
        advanced = idx != self._index
        if not advanced.any():
            return
        from .arm_reach import sample_target  # noqa: PLC0415
        nxt = np.zeros((self.E, 3), np.float64)
        for e in np.flatnonzero(advanced):
            nxt[e] = sample_target(self.spec, self._rng[e], int(self._tier[e]))
        self._push_targets(advanced, None, nxt)
        self._index = idx

    def _push_targets(self, mask, active, nxt) -> None:
        mc = (C.c_uint8 * self.E)(*[1 if x else 0 for x in mask])
        act = None if active is None else np.ascontiguousarray(
            active, np.float64).reshape(self.E, 3)
        nx = None if nxt is None else np.ascontiguousarray(
            nxt, np.float64).reshape(self.E, 3)
        rc = self._lib.dwc1_reach_set_targets(
            self._h, mc,
            act.ctypes.data_as(DP) if act is not None else None,
            nx.ctypes.data_as(DP) if nx is not None else None)
        if rc:
            raise RuntimeError(f"dwc1_reach_set_targets status={rc}")

    def reset_policy(self, mask: np.ndarray | None = None,
                     seed: int | None = None, tiers=None) -> np.ndarray:
        """Masked policy reset mirroring ArmReachEnv.reset(): physics and
        policy state back to creation; per env the FROZEN draw order of
        arm_reach.episode_draw from the counter-based (seed, env, episode)
        rng -- tier (or the pinned tier), the first target, and ONE target
        ahead (the queued next) -- pushed via dwc1_reach_set_targets after
        dwc1_reset_policy(tier, episode key). `tiers` [E] overrides the
        drawn tier for this reset only (evaluation). Returns fresh obs."""
        from .arm_reach import episode_draw, sample_target  # noqa: PLC0415
        if seed is not None:
            self._seed = int(seed)
        m = np.ones(self.E, bool) if mask is None \
            else np.asarray(mask, bool).reshape(self.E)
        tier = np.zeros(self.E, np.float64)
        key = np.zeros(self.E, np.float64)
        t0 = np.zeros((self.E, 3), np.float64)
        t1 = np.zeros((self.E, 3), np.float64)
        pins = None if tiers is None else np.broadcast_to(
            np.asarray(tiers, np.int64), (self.E,))
        for e in np.flatnonzero(m):
            pin = self._tier_pin if pins is None else int(pins[e])
            rng, tr, target = episode_draw(
                self.spec, self._seed, int(e), int(self._episode[e]) + 1, pin)
            self._rng[e] = rng
            self._tier[e] = tr
            tier[e] = float(tr)
            t0[e] = target
            t1[e] = sample_target(self.spec, rng, tr)     # one draw ahead
            self._episode[e] += 1
            key[e] = float(self._episode[e])
        mc = (C.c_uint8 * self.E)(*[1 if x else 0 for x in m])
        rc = self._lib.dwc1_reset_policy(self._h, mc, tier.ctypes.data_as(DP),
                                         key.ctypes.data_as(DP))
        if rc:
            raise RuntimeError(f"dwc1_reset_policy status={rc}")
        self._push_targets(m, t0, t1)
        self._index[m] = 0
        return self.observe()

    def pin_tier(self, tier: int | None) -> None:
        """Evaluation hook (ArmReachEnv.pin_tier twin): pin the tier drawn
        at the NEXT resets (None restores per-episode uniform draws)."""
        self._tier_pin = None if tier is None else int(tier)

    @property
    def tier(self) -> np.ndarray:
        return self._tier.copy()

    def observe(self) -> np.ndarray:
        """Current 27-dim observations (no stepping); reset()-style read."""
        obs = np.empty((self.E, self.OBS), np.float32)
        rc = self._lib.dwc1_observe(self._h, _fp(obs))
        if rc:
            raise RuntimeError(f"dwc1_observe status={rc}")
        return obs

    def reach_state(self) -> np.ndarray:
        """dwc1_reach_state per env (REACH_STATE_DTYPE): active/queued
        target, target_index, hold, and the judge-shadow counters
        (acquire_step[5], limit/speed/proxy violating ticks, starved) for
        the current episode plus the last completed episode's snapshot."""
        out = (ReachState * self.E)()
        rc = self._lib.dwc1_reach_get(self._h, out)
        if rc:
            raise RuntimeError(f"dwc1_reach_get status={rc}")
        return np.frombuffer(out, dtype=REACH_STATE_DTYPE).copy()

    def gate_proxy(self) -> np.ndarray:
        """dwc1_gate_proxy in its REACH mapping (GATE_PROXY_DTYPE):
        qualified_left = targets acquired, qualified_right = 0,
        alternation_violations = judge-clause violating ticks; episode_*
        the last completed episode. Metrics only."""
        out = (GateProxy * self.E)()
        rc = self._lib.dwc1_gate_proxy_get(self._h, out)
        if rc:
            raise RuntimeError(f"dwc1_gate_proxy_get status={rc}")
        return np.frombuffer(out, dtype=GATE_PROXY_DTYPE).copy()

    def set_gate_termination(self, first_deadline_ticks: int = 0,
                             max_alternation_violations: int = 0) -> None:
        """OPT-IN judge-aligned death rules (both 0 = off): terminate a live
        env once `first_deadline_ticks` accepted ticks pass without any
        target acquired, or once its judge-clause violating-tick count
        reaches `max_alternation_violations` (the humanoid knob names,
        reach semantics -- see duck_cuda.h)."""
        rc = self._lib.dwc1_set_gate_termination(
            self._h, int(first_deadline_ticks),
            int(max_alternation_violations))
        if rc:
            raise RuntimeError(f"dwc1_set_gate_termination status={rc}")

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
