"""Flat-floor native backend adapter for the batched duck environment.

Every integrated_duck_v1 (idv1) specific detail lives here: building/caching
the combined dylib, replicating ``native.duck_scene`` registration with
per-env joint perturbations, batched ticking, state reads (poses, velocities,
foot-vs-floor contact flags from the solve-cache manifolds, whole-sole
heights from the baked foot collider vertices), masked snapshot restore, and
full-state JSON dumps for solver-fault forensics.

A future cube-grid backend (workstream B, walk/env/world.py) only has to
provide the same duck-typed surface as :class:`NativeDuckLane`:

    lane.E, lane.J, lane.home_joint_q, lane.joint_limits, lane.home_root_height
    lane.tick(targets)   -> (rc, diagnostics per env)
    lane.read()          -> LaneState
    lane.restore(mask)   -> restore masked envs to the initial snapshot
    lane.state_dump(e)   -> JSON-serialisable full state of env e
    lane.close()

Verified against the pinned duck lowering (see PLAN.md / native.py):
- 16 contact bodies; body 6 = left ``foot_assembly``, body 15 = right
  ``foot_assembly_2`` (NOT 11/12); body 0 = fixed floor plane at z = 0.
- 3 contact pairs; pair 0 = left-foot-vs-floor, pair 1 = right-foot-vs-floor,
  pair 2 = foot-vs-foot.
- contact body state = [p_xyz(3), q_xyzw(4), v_world(3), omega_world(3)];
  foot collider = 18 baked convex vertices in the principal-COM body frame,
  so whole-sole height = min over vertices of world z (floor plane z = 0).
"""
from __future__ import annotations

import ctypes as C
import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LANE = ROOT / "experimental" / "integrated_duck_v1"
BUILD_DIR = ROOT / "build"

# Compile exactly the way experimental/integrated_duck_v1/run_local.py does.
_INCLUDES = [ROOT / x for x in [
    "experimental/integrated_duck_v1/include", "experimental/contact_v1/include",
    "experimental/articulated_v1/include", "experimental/articulated_v2/include",
    "include", "csrc"]]
_UNITS = [ROOT / x for x in [
    "experimental/integrated_duck_v1/src/coupled_impulse_v1.cpp",
    "experimental/integrated_duck_v1/src/integrated_duck_v1.cpp",
    "experimental/contact_v1/src/contact_v1.cpp",
    "experimental/articulated_v1/src/articulated_v1.cpp",
    "experimental/articulated_v2/src/articulated_v2.cpp",
    "csrc/experimental_joint_v1.cpp"]]
_FLAGS = ["-std=c++17", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off", "-O2"]

LEFT_FOOT_BODY = 6
RIGHT_FOOT_BODY = 15
FOOT_BODIES = (LEFT_FOOT_BODY, RIGHT_FOOT_BODY)
FLOOR_PAIRS = (0, 1)  # cache manifold pair indices: (left, right) foot vs floor
SIM_DT = 0.002
MAX_SOLVER_ITERATIONS = 4096
IMPULSE_TOLERANCE = 1e-8


def _source_digest() -> str:
    h = hashlib.sha256()
    paths = []
    for base in _INCLUDES + _UNITS:
        paths.extend(sorted(base.rglob("*")) if base.is_dir() else [base])
    for p in paths:
        if p.is_file():
            h.update(str(p.relative_to(ROOT)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def build_library() -> Path:
    """Build (or reuse the cached) combined native dylib under build/."""
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out = BUILD_DIR / f"libintegrated_duck-{_source_digest()}{suffix}"
    if out.is_file():
        return out
    compiler = shutil.which("clang++")
    if not compiler:
        raise RuntimeError("clang++ toolchain required to build the native duck lane")
    flags = list(_FLAGS)
    for p in _INCLUDES:
        flags.extend(["-I", str(p)])
    tmp = out.with_suffix(out.suffix + ".tmp")
    cmd = [compiler, *flags, "-fPIC", "-shared", *map(str, _UNITS), "-o", str(tmp)]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    tmp.replace(out)
    return out


def _native():
    if str(LANE) not in sys.path:
        sys.path.insert(0, str(LANE))
    import native  # noqa: PLC0415  (idv1 ctypes bindings)
    return native


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Batched xyzw quaternion -> body-to-world rotation matrices [..., 3, 3]."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


@dataclasses.dataclass
class LaneState:
    """Numpy view of one batched native read (policy-step or tick boundary)."""
    q: np.ndarray            # [E, 7+J] f64: root xyz, root quat xyzw, joint q
    v: np.ndarray            # [E, 6+J] f64: world lin vel, world ang vel, joint qdot
    time: np.ndarray         # [E] f64 simulated seconds
    count: np.ndarray        # [E] u64 accepted native ticks
    body_state: np.ndarray   # [E, B, 13] f64: p(3), q_xyzw(4), v(3), omega(3)
    foot_contact: np.ndarray  # [E, 2] bool from solve-cache manifolds (L, R)
    foot_pos: np.ndarray     # [E, 2, 3] foot body COM world positions
    sole_height: np.ndarray  # [E, 2] min world z over the 18 sole vertices

    def finite(self) -> np.ndarray:
        """Per-env all-finite flag."""
        return (np.isfinite(self.q).all(1) & np.isfinite(self.v).all(1)
                & np.isfinite(self.body_state).all((1, 2)) & np.isfinite(self.time))


def _manifold_json(m) -> dict:
    return {"count": int(m.count), "normal": list(m.normal), "tangent1": list(m.tangent1),
            "tangent2": list(m.tangent2),
            "points": [{"feature": int(p.feature), "point": list(p.point),
                        "depth": float(p.depth), "normal_impulse": float(p.normal_impulse),
                        "tangent_impulse": list(p.tangent_impulse)}
                       for p in m.points[:m.count]]}


class NativeDuckLane:
    """Batched flat-floor duck scene over the idv1 native lane."""

    def __init__(self, environments: int, joint_offsets: np.ndarray | None = None,
                 library_path: str | Path | None = None):
        native = self._native = _native()
        self.library_path = Path(library_path) if library_path else build_library()
        lib = self._lib = native.library(self.library_path)
        fixture = native.av.duck()
        cm = native.contact.Model(native.contact.library(str(self.library_path)))
        frame = cm.record["frames"][0]
        base = np.array([*frame["base_pose"][:3], *frame["base_pose"][4:],
                         frame["base_pose"][3], *frame["joint_q"]])
        if base[2] != 0.16788827542191784:
            raise ValueError("floor-clear reset pin mismatch")
        v = np.array(frame["qvel"], dtype="d")
        v[3:6] = native.av.rot(base[3:7]) @ v[3:6]

        self.E = int(environments)
        self.J = fixture.J
        self.B = fixture.B
        self.P = len(cm.pairs)
        limits = native.av.limits(fixture)
        self.joint_limits = np.array([[l.lower, l.upper] for l in limits])
        q = np.tile(base, (self.E, 1))
        if joint_offsets is not None:
            off = np.asarray(joint_offsets, dtype="d")
            if off.shape != (self.E, self.J):
                raise ValueError("joint_offsets requires shape [E, J]")
            q[:, 7:] = np.clip(q[:, 7:] + off,
                               self.joint_limits[:, 0], self.joint_limits[:, 1])
        self._scene = native.Scene(lib, fixture, q, np.tile(v, (self.E, 1)),
                                   cm.shapes, cm.pairs, np.tile(cm.mu, (self.E, 1)),
                                   limits=limits)
        self.home_joint_q = base[7:].copy()
        self.home_root_height = float(base[2])
        self.kp = float(fixture.hinge[0].kp)
        self.kv = float(fixture.hinge[0].kv)
        self.effort_cap = float(fixture.hinge[0].cap)
        # 18 baked convex sole vertices per foot, principal-COM body frame.
        self.foot_vertices = np.array(
            [[list(cm.shapes[b].vertices[i]) for i in range(cm.shapes[b].vertex_count)]
             for b in FOOT_BODIES])
        self._snapshot = self._scene.capture()  # initial (perturbed) state

    # -- stepping ---------------------------------------------------------
    def tick(self, targets: np.ndarray):
        """One 0.002 s native step of all E envs at the given joint targets.

        Returns (rc, diagnostics) where diagnostics is a list of E dicts; the
        caller decides fault policy (rc != 0 rolls the failing tick back).
        """
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
        world = feet[:, :, None, :3] + np.einsum("efij,fvj->efvi", rot, self.foot_vertices)
        return LaneState(q=x.q, v=x.v, time=x.time, count=x.count, body_state=body,
                         foot_contact=contact, foot_pos=feet[:, :, :3].copy(),
                         sole_height=world[..., 2].min(axis=2))

    # -- snapshot / reset --------------------------------------------------
    def restore(self, mask: np.ndarray | None = None) -> None:
        m = None if mask is None else [int(bool(x)) for x in np.asarray(mask).reshape(-1)]
        if m is not None and len(m) != self.E:
            raise ValueError("mask requires length E")
        rc = self._scene.reset(mask=m, snapshot=self._snapshot)
        if rc:
            raise RuntimeError(f"idv1_restore status={rc}")

    # -- forensics ---------------------------------------------------------
    def state_dump(self, env: int) -> dict:
        """Full JSON-serialisable state of one env (idv1_read fields)."""
        x = self._scene.read()
        e = int(env)
        return {
            "qpos": x.q[e].tolist(), "velocity": x.v[e].tolist(),
            "warm_force": x.warm[e].tolist(), "time_s": float(x.time[e]),
            "step_count": int(x.count[e]),
            "bodies": [list(b.state) for b in x.bodies[e * self.B:(e + 1) * self.B]],
            "pre_contact_cache": [_manifold_json(m)
                                  for m in x.cache[e * self.P:(e + 1) * self.P]],
            "current_geometry": [_manifold_json(m)
                                 for m in x.geometry[e * self.P:(e + 1) * self.P]],
        }

    def close(self) -> None:
        if getattr(self, "_scene", None) is not None:
            self._scene.close()
            self._scene = None
