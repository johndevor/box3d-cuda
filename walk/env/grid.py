"""CubeGridDuckEnv: batched Open Duck env on a duck_world_v1 cube grid.

The cube-grid counterpart of FlatFloorDuckEnv. ALL task semantics are reused
from walk/env/flat.py via the lane-swap seam it was built for (README
"Swapping in the cube-grid backend"): action mapping HOME + 0.25 * a, slew
limit 0.1048 rad per policy step, per-2ms-tick PD kp=13.37 / kv=0 / cap=3.23
applied by the lane, 10 ticks per policy step, the OBS=58 layout from
contract.py, the reward from walk/env/reward.py, and the same termination.
Only the backend differs: steps go through duck_world_v1 (walk/env/world.py)
via grid_lane.GridLane instead of the idv1 flat lane, so `sole_height` (and
therefore the reward clearance/air-time terms) is measured above the
SUPPORTING surface — the max cube top under the foot, else the floor.

Termination keeps flat.py's "root below 0.7 x reset height" rule; the reset
root height here includes the grid lift (duck starts standing on cube tops),
so the fall threshold is relative to the cube-top support exactly as flat's
is relative to the floor.

Solver-fault handling is identical to flat.py (persist the failing envs'
full post-rollback state to runs/faults/, raise SolverFault, never return
post-fault observations); the artifact additionally records the grid spec
and the grid lane's actual solver tolerances.

Tolerances (constructor `impulse_tolerance` / `jtol`, default None): per the
duck_world_v1 README interim notes, static grids (dynamic=False) default to
the pinned civ1 impulse tolerance 1e-8; dynamic grids default to 1e-6 with
av2 jtol matched at 1e-6 (av2 accepts up to 1e-5) until the workstream-A
civ1 stall repair lands. dwv1 keeps the momentum residual pinned at 1e-8
internally either way.
"""
from __future__ import annotations

import datetime as _dt
import json

import numpy as np

from . import grid_lane
from .contract import SolverFault
from .flat import FAULT_DIR, SIM_DT, FlatFloorDuckEnv


class CubeGridDuckEnv(FlatFloorDuckEnv):
    """E parallel ducks on cube grids; obs/reward/termination as flat.py."""

    def __init__(self, environments: int = 16, seed: int = 0,
                 grid: dict | None = None, perturbation_rad: float = 0.0,
                 library_path=None, impulse_tolerance: float | None = None,
                 jtol: float | None = None):
        # Stored before super().__init__ because it builds the lane. If
        # grid["seed"] is unset the terrain seed follows the env seed (so
        # reset(seed=...) also re-seeds the height jitter, deterministically).
        self._grid_arg = dict(grid or {})
        self._impulse_tolerance_arg = impulse_tolerance
        self._jtol_arg = jtol
        super().__init__(environments=environments, seed=seed,
                         perturbation_rad=perturbation_rad,
                         library_path=library_path,
                         lane_factory=self._make_grid_lane)

    def _make_grid_lane(self, environments: int, joint_offsets):
        return grid_lane.GridLane(
            environments, grid=self._grid_arg, joint_offsets=joint_offsets,
            library_path=self._library_path,
            impulse_tolerance=self._impulse_tolerance_arg,
            jtol=self._jtol_arg, default_seed=self._seed)

    @property
    def grid(self) -> dict:
        """The fully-resolved grid spec the current lane was built with."""
        return dict(self._lane.grid_spec)

    # ------------------------------------------------------------------
    def _raise_fault(self, rc, diagnostics, bad, tick, action):
        """flat.py's fault handling with the grid lane's actual solver
        parameters (its tolerance may differ from native_lane's constant)
        and the grid spec recorded for forensics."""
        FAULT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        failing = sorted({d["environment"] for d in bad}) or list(range(self.E))
        first_path = None
        for e in failing:
            payload = {
                "schema": "duckgridwalk.solver_fault/1",
                "backend": "duck_world_v1",
                "environment": int(e), "status_rc": int(rc),
                "tick_of_policy_step": int(tick),
                "policy_step": int(self._t[e]),
                "command_mps": float(self._command[e]),
                "action": np.asarray(action)[e].tolist(),
                "effective_targets": self._effective[e].tolist(),
                "dt": SIM_DT,
                "max_iterations": grid_lane.MAX_SOLVER_ITERATIONS,
                "tolerance": self._lane.impulse_tolerance,
                "jtol": self._lane.jtol,
                "grid": dict(self._lane.grid_spec),
                "diagnostics": [d for d in diagnostics if d["environment"] == e],
                "all_diagnostics": diagnostics,
                "state": self._lane.state_dump(e),
            }
            path = FAULT_DIR / f"{stamp}-env{int(e)}.json"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            first_path = first_path or path
        raise SolverFault(int(failing[0]), str(first_path),
                          f"dwv1_step rc={rc} envs={failing}")
