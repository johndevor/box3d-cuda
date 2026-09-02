"""f64 CPU ORACLE lane for the fixed-base arm (both variants).

The arm twin of walk/env/humanoid_native_lane.py: the same combined idv1
dylib (model-generic; the model arrives via the registration), the arm
fixture from arm/arm_lowering.py. This is the physics oracle for the fp32
kernel build (walk/env/arm_cuda_lane.py).

FIXED BASE = VIRTUAL WELD (arm/FEASIBILITY.md section 1). articulated_v1
rejects a floor-parented hinge (parent < 1), so the oracle's root body 1 IS
the base_link, made WELD_MASS_FACTOR (1e6) x heavier and held by
arm_lowering.weld_force -- exact cancellation of the static gravity
generalized force on the 6 root dofs plus a stiff explicit PD -- applied
through av2_step.applied_force (native.Scene.step(force=...)) at EVERY tick
by this lane. Measured: base drift 1e-19 m / ~1e-9 rad over 8 s (test gate
1e-6 m). The kernel lane welds structurally instead (joint 0 parented to the
static floor body); both are gate-pinned by arm/tests.

No contact pairs are registered (P = 0): the arm never touches the floor in
this lowering; the judge's proxy clause keeps it clear.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "arm") not in sys.path:
    sys.path.insert(0, str(ROOT / "arm"))

from . import native_lane  # noqa: E402
from .native_lane import build_library  # noqa: E402,F401

import arm_lowering as al  # noqa: E402

SIM_DT = al.SIM_DT
MAX_SOLVER_ITERATIONS = 4096
IMPULSE_TOLERANCE = 1e-8


@dataclasses.dataclass
class ArmLaneState:
    """Numpy view of one batched read. body_state rows: floor 0, root/base 1,
    link_1..link_6 = bodies 2..7, each [p(3), q_xyzw(4), v(3), omega(3)]."""
    q: np.ndarray            # [E, 13] f64: root xyz, quat xyzw, 6 joint q
    v: np.ndarray            # [E, 12] f64
    time: np.ndarray         # [E]
    count: np.ndarray        # [E] u64
    body_state: np.ndarray   # [E, 8, 13] f64

    def finite(self) -> np.ndarray:
        return (np.isfinite(self.q).all(1) & np.isfinite(self.v).all(1)
                & np.isfinite(self.body_state).all((1, 2)))


class NativeArmLane:
    """Batched fixed-base arm over the idv1 native lane (f64 oracle)."""

    def __init__(self, environments: int, variant: str = "kr240",
                 joint_offsets: np.ndarray | None = None,
                 library_path: str | Path | None = None):
        self.spec = al.spec(variant)
        self.variant = self.spec.variant
        native = self._native = native_lane._native()
        self.library_path = Path(library_path) if library_path else build_library()
        lib = self._lib = native.library(self.library_path)
        self.E = int(environments)
        self.J, self.B, self.P = al.J, al.B, 0
        self._scene, self._fixture = al.scene(lib, self.spec, self.E,
                                              joint_offsets=joint_offsets)
        self.joint_limits = al.joint_limits(self.spec)
        self.home_joint_q = np.asarray(self.spec.home_q, float)
        self.home_root_height = 0.0
        kp, kv = al.gains(self.spec)
        self.kp = kp                                  # per-joint tables
        self.kv = kv
        self.effort_cap = al.effort(self.spec)
        self.velocity_limits = al.velocity_limits(self.spec)
        self._snapshot = self._scene.capture()

    # -- stepping ---------------------------------------------------------
    def tick(self, targets: np.ndarray):
        """One SIM_DT native step at the given joint targets, with the
        virtual-weld root force computed from the current state."""
        x = self._scene.read()
        body = np.frombuffer(memoryview(x.bodies), dtype=np.float32)
        body = body.reshape(self.E, self.B, 17)[:, :, :13].astype("d")
        force = al.weld_force(self.spec, x.q, x.v,
                              body[:, list(al.LINK_BODIES), :3])
        return self._scene.step(dt=SIM_DT, target=targets, force=force,
                                max_iterations=MAX_SOLVER_ITERATIONS,
                                tolerance=IMPULSE_TOLERANCE)

    def tick_block(self, targets: np.ndarray, n_ticks: int):
        """n_ticks held-target ticks; returns the LAST tick's (rc, diags)
        with per-env max iterations accumulated (the env only reads them)."""
        worst = None
        for _ in range(int(n_ticks)):
            rc, diags = self.tick(targets)
            if worst is None:
                worst = [dict(d) for d in diags]
            else:
                for w, d in zip(worst, diags):
                    w["iterations"] = max(w["iterations"], d["iterations"])
                    w["momentum_residual"] = max(w["momentum_residual"],
                                                 d["momentum_residual"])
                    w["native_status"] = d["native_status"]
            if rc or any(d["native_status"] for d in diags):
                return rc, diags
        return 0, worst

    # -- reads ------------------------------------------------------------
    def read(self) -> ArmLaneState:
        x = self._scene.read()
        body = np.frombuffer(memoryview(x.bodies), dtype=np.float32)
        body = body.reshape(self.E, self.B, 17)[:, :, :13].astype("d")
        return ArmLaneState(q=x.q, v=x.v, time=x.time, count=x.count,
                            body_state=body)

    # -- snapshot / reset ---------------------------------------------------
    def restore(self, mask: np.ndarray | None = None) -> None:
        m = None if mask is None else [int(bool(x))
                                       for x in np.asarray(mask).reshape(-1)]
        if m is not None and len(m) != self.E:
            raise ValueError("mask requires length E")
        rc = self._scene.reset(mask=m, snapshot=self._snapshot)
        if rc:
            raise RuntimeError(f"idv1_restore status={rc}")

    # -- forensics ----------------------------------------------------------
    def state_dump(self, env: int) -> dict:
        x = self._scene.read()
        e = int(env)
        return {"variant": self.variant,
                "qpos": x.q[e].tolist(), "velocity": x.v[e].tolist(),
                "warm_force": x.warm[e].tolist(), "time_s": float(x.time[e]),
                "step_count": int(x.count[e]),
                "bodies": [list(b.state)
                           for b in x.bodies[e * self.B:(e + 1) * self.B]]}

    def close(self) -> None:
        if getattr(self, "_scene", None) is not None:
            self._scene.close()
            self._scene = None
