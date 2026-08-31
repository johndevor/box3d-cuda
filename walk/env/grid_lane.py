"""Cube-grid native backend adapter (duck_world_v1 / dwv1) for the batched env.

Conforms to the lane protocol documented at the top of walk/env/native_lane.py
(and the README backend-swap section):

    lane.E, lane.J, lane.home_joint_q, lane.joint_limits, lane.home_root_height
    lane.kp, lane.kv, lane.effort_cap
    lane.tick(targets)   -> (rc, diagnostics per env)   # one 0.002 s tick
    lane.read()          -> LaneState (native_lane.LaneState, reused verbatim)
    lane.restore(mask)   -> restore masked envs to the initial snapshot
    lane.state_dump(e)   -> JSON-serialisable full state of env e
    lane.close()

Only grid-specific logic lives here:
- scene construction mirrors ``world.duck_grid_scene`` (same pinned floor-clear
  reset frame, duck lifted by ``base_height + height_jitter + cube_size + gap``
  so the feet start ``gap`` = 1.5 mm above the tallest possible cube top —
  exactly the clearance the flat lane's reset has above the floor), extended
  with per-env joint-q offsets like native_lane.NativeDuckLane;
- foot contact comes from dwv1's per-foot flags (set for foot-vs-floor AND
  foot-vs-cube solved manifolds, see duck_world_v1.cpp), not from the two
  foot-vs-floor cache pairs the flat lane inspects;
- ``sole_height`` is reported ABOVE THE SUPPORTING SURFACE, not absolute z:
  per foot, support = max top of the cubes lying under any of the 18 sole
  vertices (cube tops taken as pose z + cube_size/2 — exact for static grids,
  an axis-aligned approximation for tumbling dynamic cubes), else the floor
  plane z = 0; sole_height = min vertex world z - support. Reward clearance
  terms and the strict evaluator need height-above-support, and on the floor
  (support 0) this reduces bit-exactly to the flat lane's definition.

Solver tolerances (see experimental/duck_world_v1/README.md, "Known
limitations / interim choices"): static grids converge at the pinned civ1
impulse tolerance 1e-8 (default for dynamic=False); coupled dynamic islands
can stall the civ1 sweep just above 1e-8 until the workstream-A repair lands,
so dynamic=True defaults to 1e-6 with av2 joint-KKT jtol matched at 1e-6
(av2_step legally accepts up to 1e-5). The momentum residual stays pinned at
1e-8 inside dwv1 regardless. Both are overridable per constructor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import world
from .native_lane import (FOOT_BODIES, LaneState, _manifold_json, quat_to_rot)

SIM_DT = 0.002
# dwv1's validated sweep budget (world.Scene.step default; every dwv1 gate
# runs with it). The flat lane's 4096 is NOT enough here: the foot-settling
# impact on a flush static grid peaks around ~7.6k civ1 sweeps (6-point
# multi-cube contact block), and capping at 4096 surfaces as a phase-3
# CIV1_NO_CONVERGENCE fault ~0.28 s after reset.
MAX_SOLVER_ITERATIONS = 16384
STATIC_IMPULSE_TOLERANCE = 1e-8         # dwv1 README: static grids hold 1e-8
DYNAMIC_IMPULSE_TOLERANCE = 1e-6        # dwv1 README: coupled dynamic islands
JTOL_MAX = 1e-5                         # av2_step's legal joint-KKT ceiling
FOOT_GAP = 0.0015                       # feet start 1.5 mm above the grid

# Defaults produce a flat flush 8x8 static grid centered under the duck.
DEFAULT_GRID = dict(nx=8, nz=8, cube_size=0.06, spacing=None, base_height=0.0,
                    height_jitter=0.0, origin_x=0.0, origin_y=0.0,
                    dynamic=False, cube_mass=0.1, friction=0.8, seed=None)


def resolve_grid(grid: dict | None, default_seed: int = 0) -> dict:
    """Fill grid defaults: spacing -> cube_size (flush), seed -> env seed."""
    spec = dict(DEFAULT_GRID)
    grid = dict(grid or {})
    unknown = sorted(set(grid) - set(spec))
    if unknown:
        raise ValueError(f"unknown grid keys {unknown}; allowed: {sorted(spec)}")
    spec.update(grid)
    if spec["spacing"] is None:
        spec["spacing"] = float(spec["cube_size"])
    if spec["seed"] is None:
        spec["seed"] = int(default_seed) & 0xFFFFFFFFFFFFFFFF
    spec["dynamic"] = int(bool(spec["dynamic"]))
    return spec


class GridLane:
    """Batched duck-on-cube-grid scene over the dwv1 native lane."""

    def __init__(self, environments: int, grid: dict | None = None,
                 joint_offsets: np.ndarray | None = None,
                 library_path: str | Path | None = None,
                 impulse_tolerance: float | None = None,
                 jtol: float | None = None, default_seed: int = 0):
        self.grid_spec = resolve_grid(grid, default_seed)
        dynamic = bool(self.grid_spec["dynamic"])
        self.impulse_tolerance = float(impulse_tolerance) if impulse_tolerance \
            is not None else (DYNAMIC_IMPULSE_TOLERANCE if dynamic
                              else STATIC_IMPULSE_TOLERANCE)
        # av2 joint-KKT verification tracks the civ1 impulse tolerance
        # (>= the pinned 1e-8 floor, <= av2's legal 1e-5 ceiling).
        self.jtol = float(jtol) if jtol is not None else \
            min(max(self.impulse_tolerance, 1e-8), JTOL_MAX)
        self.library_path = Path(library_path) if library_path else world.build()
        lib = self._lib = world.library(self.library_path)
        fixture = world.av.duck()
        cm = world.contact.Model(world.contact.library(str(self.library_path)))
        frame = cm.record["frames"][0]
        base = np.array([*frame["base_pose"][:3], *frame["base_pose"][4:],
                         frame["base_pose"][3], *frame["joint_q"]])
        if base[2] != 0.16788827542191784:
            raise ValueError("floor-clear reset pin mismatch")
        vel = np.array(frame["qvel"], dtype="d")
        vel[3:6] = world.av.rot(base[3:7]) @ vel[3:6]
        # Lift the pinned floor-clear reset so the feet start FOOT_GAP above
        # the tallest possible cube top (same clearance flat has above z=0).
        g = self.grid_spec
        base[2] += g["base_height"] + g["height_jitter"] + g["cube_size"] + FOOT_GAP

        self.E = int(environments)
        self.J = fixture.J
        self.B = fixture.B
        self.P = len(cm.pairs)
        limits = world.av.limits(fixture)
        self.joint_limits = np.array([[l.lower, l.upper] for l in limits])
        q = np.tile(base, (self.E, 1))
        if joint_offsets is not None:
            off = np.asarray(joint_offsets, dtype="d")
            if off.shape != (self.E, self.J):
                raise ValueError("joint_offsets requires shape [E, J]")
            q[:, 7:] = np.clip(q[:, 7:] + off,
                               self.joint_limits[:, 0], self.joint_limits[:, 1])
        spec = world.grid_spec(
            nx=g["nx"], nz=g["nz"], cube_size=g["cube_size"], spacing=g["spacing"],
            base_height=g["base_height"], height_jitter=g["height_jitter"],
            origin_x=g["origin_x"], origin_y=g["origin_y"], dynamic=g["dynamic"],
            cube_mass=g["cube_mass"], friction=g["friction"], seed=g["seed"])
        self._scene = world.Scene(lib, fixture, q, np.tile(vel, (self.E, 1)),
                                  cm.shapes, cm.pairs,
                                  np.tile(cm.mu, (self.E, 1)), spec, limits=limits)
        self.M = self._scene.M                       # cubes per env
        self.F = self._scene.F                       # feet (convex non-fixed)
        if self.F != 2:
            raise ValueError(f"expected 2 duck feet, dwv1 detected {self.F}")
        self.home_joint_q = base[7:].copy()
        self.home_root_height = float(base[2])       # lifted reset root height
        self.kp = float(fixture.hinge[0].kp)
        self.kv = float(fixture.hinge[0].kv)
        self.effort_cap = float(fixture.hinge[0].cap)
        # 18 baked convex sole vertices per foot, principal-COM body frame.
        self.foot_vertices = np.array(
            [[list(cm.shapes[b].vertices[i]) for i in range(cm.shapes[b].vertex_count)]
             for b in FOOT_BODIES])
        self._snapshot = self._scene.capture()       # initial (perturbed) state

    # -- stepping ---------------------------------------------------------
    def tick(self, targets: np.ndarray):
        """One 0.002 s native step of all E envs at the given joint targets.

        Returns (rc, diagnostics list of E dicts); the caller decides fault
        policy (rc != 0 leaves the failing tick fully rolled back)."""
        return self._scene.step(dt=SIM_DT, target=targets,
                                max_iterations=MAX_SOLVER_ITERATIONS,
                                tolerance=self.impulse_tolerance, jtol=self.jtol)

    # -- reads ------------------------------------------------------------
    def _support_height(self, verts: np.ndarray, cube_pose: np.ndarray) -> np.ndarray:
        """Per-foot supporting-surface height: max cube top under any sole
        vertex, else the floor plane z = 0. verts [E,2,V,3], cube_pose [E,M,7]."""
        half = 0.5 * float(self.grid_spec["cube_size"])
        centers = cube_pose[:, :, :2]                          # [E, M, 2]
        tops = cube_pose[:, :, 2] + half                       # [E, M]
        under = ((np.abs(verts[:, :, :, None, 0] - centers[:, None, None, :, 0]) <= half)
                 & (np.abs(verts[:, :, :, None, 1] - centers[:, None, None, :, 1]) <= half))
        under_any = under.any(axis=2)                          # [E, 2, M]
        best = np.where(under_any, tops[:, None, :], -np.inf).max(axis=2)
        return np.where(np.isfinite(best), best, 0.0)          # floor fallback

    def read(self) -> LaneState:
        x = self._scene.read()
        body = np.frombuffer(memoryview(x.bodies), dtype=np.float32)
        body = body.reshape(self.E, self.B, 17)[:, :, :13].astype("d")
        feet = body[:, list(FOOT_BODIES), :]           # [E, 2, 13]
        rot = quat_to_rot(feet[:, :, 3:7])             # [E, 2, 3, 3]
        verts = feet[:, :, None, :3] + np.einsum("efij,fvj->efvi", rot,
                                                 self.foot_vertices)
        support = self._support_height(verts, x.cube_pose)
        return LaneState(q=x.q, v=x.v, time=x.time, count=x.count, body_state=body,
                         foot_contact=x.foot.astype(bool),   # dwv1 per-foot flags
                         foot_pos=feet[:, :, :3].copy(),
                         sole_height=verts[..., 2].min(axis=2) - support)

    # -- snapshot / reset --------------------------------------------------
    def restore(self, mask: np.ndarray | None = None) -> None:
        m = None if mask is None else [int(bool(x)) for x in np.asarray(mask).reshape(-1)]
        if m is not None and len(m) != self.E:
            raise ValueError("mask requires length E")
        rc = self._scene.reset(mask=m, snapshot=self._snapshot)
        if rc:
            raise RuntimeError(f"dwv1_restore status={rc}")

    # -- forensics ---------------------------------------------------------
    def state_dump(self, env: int) -> dict:
        """Full JSON-serialisable state of one env (dwv1_read fields)."""
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
            "cube_pose": x.cube_pose[e].tolist(),
            "cube_velocity": x.cube_velocity[e].tolist(),
            "cube_awake": x.cube_awake[e].tolist(),
            "foot_contact": x.foot[e].tolist(),
            "grid": dict(self.grid_spec),
        }

    def close(self) -> None:
        if getattr(self, "_scene", None) is not None:
            self._scene.close()
            self._scene = None
