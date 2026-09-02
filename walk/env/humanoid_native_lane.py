"""Flat-floor native CPU backend for the batched H0 humanoid.

The humanoid twin of :mod:`walk.env.native_lane` (NativeDuckLane): same
combined idv1 dylib (the native lane is model-generic; the model arrives via
the registration), same lane surface, H0 fixture from
:mod:`humanoid.h0_lowering`. This is the f64 physics ORACLE for the fp32
humanoid kernel build (walk/env/humanoid_cuda_lane.py).

Differences from the duck lane, all traceable to the H0 lowering:
- B contact bodies from the ACTIVE lowering (H1: 16; left/right foot at
  h0.FOOT_BODIES = (7, 11)); body 0 = the
  fixed floor plane z = 0. 2 contact pairs (no foot-vs-foot: H0 authors no
  self-collision).
- foot collider = the exact 8 box corners (single-OBB feet), so whole-sole
  height = min world z over 8 vertices.
- tick dt = h0.SIM_DT = 1/240 s (the authored engine's per-substep drive
  cadence; see h0_lowering.py and humanoid/FEASIBILITY.md section 5).
- gravity (0, 0, -20) after the y-up -> z-up lowering rotation.
- `tilt` uses the body +Y axis (the humanoid's authored up axis) against
  world +Z -- NOT the duck's body-Z formula: after the lowering rotation the
  root reset quaternion is QX90, so up = R[2][1] = 2*(qy*qz + qx*qw).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "humanoid") not in sys.path:
    sys.path.insert(0, str(ROOT / "humanoid"))

from . import native_lane
from .native_lane import LaneState, quat_to_rot, build_library  # noqa: F401

import h1_lowering as h0  # noqa: E402  (ACTIVE lowering: H1)

LEFT_FOOT_BODY, RIGHT_FOOT_BODY = h0.FOOT_BODIES
FOOT_BODIES = h0.FOOT_BODIES
FLOOR_PAIRS = (0, 1)
SIM_DT = h0.SIM_DT
MAX_SOLVER_ITERATIONS = 16384         # same certificates as the duck lane;
# 16384 matches the grid lane and lets the degenerate flat-foot repair
# certify post-topple impacts that stay budget-bound at 4096
IMPULSE_TOLERANCE = 1e-8


def tilt(q: np.ndarray) -> np.ndarray:
    """Angle [rad] between the humanoid's up axis (body +Y) and world +Z.

    q: [..., 4] root xyzw. up = (R @ [0,1,0])_z = 2*(qy*qz + qx*qw); equals
    exactly 0 at the reset orientation QX90.
    """
    up = 2.0 * (q[..., 1] * q[..., 2] + q[..., 0] * q[..., 3])
    return np.arccos(np.clip(up, -1.0, 1.0))


class NativeHumanoidLane:
    """Batched flat-floor H0 humanoid scene over the idv1 native lane."""

    def __init__(self, environments: int, joint_offsets: np.ndarray | None = None,
                 library_path: str | Path | None = None):
        native = self._native = native_lane._native()
        self.library_path = Path(library_path) if library_path else build_library()
        lib = self._lib = native.library(self.library_path)
        self.E = int(environments)
        self.J = h0.J
        self.B = h0.B
        self.P = 2
        self._scene, fixture = h0.scene(lib, self.E, joint_offsets=joint_offsets)
        self.joint_limits = np.array([(j[4], j[5]) for j in h0.JOINTS])
        self.home_joint_q = np.array(h0.HOME_TARGETS)
        self.home_root_height = float(h0.reset_qpos()[2])
        self.kp = float(h0.KP)
        self.kv = float(h0.KV)
        self.effort_cap = np.array(h0.EFFORT)     # PER-JOINT (unlike the duck)
        self.foot_vertices = np.array([h0.foot_vertices()] * 2)  # [2, 8, 3]
        self._snapshot = self._scene.capture()    # initial (perturbed) state

    # -- stepping ---------------------------------------------------------
    def tick(self, targets: np.ndarray):
        """One SIM_DT native step of all E envs at the given joint targets."""
        return self._scene.step(dt=SIM_DT, target=targets,
                                max_iterations=MAX_SOLVER_ITERATIONS,
                                tolerance=IMPULSE_TOLERANCE)

    # -- reads ------------------------------------------------------------
    def read(self) -> LaneState:
        x = self._scene.read()
        body = np.frombuffer(memoryview(x.bodies), dtype=np.float32)
        body = body.reshape(self.E, self.B, 17)[:, :, :13].astype("d")
        contact = np.array(
            [[x.cache[e * self.P + p].count > 0 for p in FLOOR_PAIRS]
             for e in range(self.E)], dtype=bool)
        feet = body[:, list(FOOT_BODIES), :]           # [E, 2, 13]
        rot = quat_to_rot(feet[:, :, 3:7])             # [E, 2, 3, 3]
        world = feet[:, :, None, :3] + np.einsum("efij,fvj->efvi", rot,
                                                 self.foot_vertices)
        return LaneState(q=x.q, v=x.v, time=x.time, count=x.count,
                         body_state=body, foot_contact=contact,
                         foot_pos=feet[:, :, :3].copy(),
                         sole_height=world[..., 2].min(axis=2))

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
        return {
            "qpos": x.q[e].tolist(), "velocity": x.v[e].tolist(),
            "warm_force": x.warm[e].tolist(), "time_s": float(x.time[e]),
            "step_count": int(x.count[e]),
            "bodies": [list(b.state)
                       for b in x.bodies[e * self.B:(e + 1) * self.B]],
            "pre_contact_cache": [native_lane._manifold_json(m)
                                  for m in x.cache[e * self.P:(e + 1) * self.P]],
            "current_geometry": [native_lane._manifold_json(m)
                                 for m in x.geometry[e * self.P:(e + 1) * self.P]],
        }

    def close(self) -> None:
        if getattr(self, "_scene", None) is not None:
            self._scene.close()
            self._scene = None
