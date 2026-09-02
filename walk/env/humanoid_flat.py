"""FlatFloorHumanoidEnv: batched flat-floor H0 humanoid env (CPU idv1 lane).

The humanoid twin of walk/env/flat.py's FlatFloorDuckEnv, following the
duck's proven recipe step for step (flat.py is untouched and remains the
duck's contract). OBS = 52, ACT = 12; one policy step = 10 native ticks x
0.002 s = 0.02 s (the duck env contract's cadence -- see
humanoid/h0_lowering.py SIM_DT note).

Observation layout (3*J + 16 with J from the ACTIVE lowering -- H1: J=14,
OBS=58; the tail block is the duck's same 16 channels at base T = 3*J):

    [   0:J]  joint q - HOME (rad)                    HOME = zeros (reset)
    [  J:2J]  0.05 * joint qdot                       QDOT_OBS_SCALE
    [ 2J:3J]  previous action (live envs only)
    [T   :T+3]  gravity direction, body frame (-R[2,:]) = (0,-1,0) at reset
    [T+3 :T+6]  root angular velocity, body frame (R^T w)
    [T+6 :T+9]  root linear velocity, body frame (R^T v)
    [T+9]       commanded forward velocity (m/s)
    [T+10:T+12] reserved zeros (duck convention)
    [T+12:T+14] foot contact flags (left, right)
    [T+14:T+16] gait phase clock (sin, cos)

Action semantics (duck recipe with humanoid-sourced numbers):
    requested = HOME + ACTION_SCALE * action, slew-limited per policy step
    by MAX_TARGET_INCREMENT, joint-limit clipped, held for all 10 ticks.
    ACTION_SCALE = 0.5 rad: covers the full waist/neck/ankle authored
    ranges and the swing amplitudes early walking needs (step ~0.25 m at
    leg ~0.77 m -> hip ~0.35 rad, knee ~0.5 rad); the duck's 0.25 covers a
    similar fraction of its ranges. MAX_TARGET_INCREMENT = 8.0 rad/s *
    0.02 s = 0.16 rad: 8.0 is the AUTHORED H0 actuator speed limit
    (humanoid.rs:785), whose host-side shaping role the slew implements
    (humanoid_h0.py::shape_action_targets clamps to +-speed the same way).

Commands: (0.5, 0.75, 1.0) m/s -- duck commands (0.10/0.15/0.20) scaled by
the ~5x leg-length ratio; same slowest-command oversampling.

Gait clock: phase_hz = HUM_PHASE_HZ_BASE + HUM_PHASE_HZ_PER_MPS * command,
sweepable via HUMANOID_PHASE_HZ_* env vars like the duck's. PER_MPS = 3.33
by the duck's own recipe phase_hz = v / (2 * step_length) with the reward's
PLACEMENT_MIN_M = 0.15 m minimum step (the duck's 16.67 encodes exactly its
30 mm placement floor).

Termination: root height < 0.7 * home height (duck MIN_HEIGHT_FRACTION) or
tilt > 45 deg -- tilt is the HUMANOID formula (body +Y vs world +Z,
humanoid_native_lane.tilt), not the duck's body-Z one -- or non-finite
state, or the 400-step (8 s) horizon.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os as _os
import sys
from pathlib import Path

import numpy as np

from . import humanoid_native_lane, humanoid_reward as reward_mod
from .contract import DuckEnvBatch, SolverFault

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "humanoid") not in sys.path:
    sys.path.insert(0, str(ROOT / "humanoid"))
import h1_lowering as h0  # noqa: E402  (ACTIVE lowering: H1)

FAULT_DIR = ROOT / "runs" / "faults"

ACT = h0.J                          # active lowering's joint count (H1: 14)
OBS = 3 * ACT + 16                                 # H1: 58; H0 was 52
_T = 3 * ACT                        # obs tail-block base (J-derived offsets)

HOME = np.zeros(ACT)                # authored reset pose (all joints 0)
CONTROL_DT = h0.CONTROL_DT          # 0.02 s
SIM_DT = h0.SIM_DT                  # 0.002 s
TICKS_PER_STEP = h0.TICKS_PER_CONTROL   # 10
ACTION_SCALE = 0.5                  # rad; rationale in module docstring
MAX_TARGET_INCREMENT = h0.SPEED_LIMIT * CONTROL_DT  # 8 rad/s * 0.02 = 0.16
# Affine gait clock (duck flat.py mechanism, humanoid constants; sweepable).
PHASE_HZ_BASE = float(_os.environ.get("HUMANOID_PHASE_HZ_BASE", "0.0"))
PHASE_HZ_PER_MPS = float(_os.environ.get("HUMANOID_PHASE_HZ_PER_MPS", "3.33"))
COMMANDS_MPS = (0.50, 0.75, 1.00)   # ~5x duck (leg-length ratio)
HORIZON_STEPS = 400                 # 8 s at 0.02 s -- duck-proven horizon
MIN_HEIGHT_FRACTION = 0.7           # of home root height (1.15 m -> 0.805)
# v2 (was 45 deg, the duck recipe): the leg-1 lunge survived 8 s at a
# 32.5-34.7 deg lean -- past the FROZEN judge's 30 deg tilt criterion but
# under the old termination. Leaning past the judge's limit must not be a
# survivable strategy; 28 deg sits just inside 30 with margin for the
# f32-lane vs judge measurement gap. Human walking pitch is <= ~15 deg, so
# no legitimate gait is clipped. (Duck env keeps its own 45.)
MAX_TILT_RAD = math.radians(28.0)
MAX_PERTURBATION_RAD = 0.02         # duck spec bound for reset perturbation
QDOT_OBS_SCALE = 0.05               # duck pinned-candidate scale


def _episode_rng(seed: int, env: int, episode: int) -> np.random.Generator:
    """Counter-based RNG (identical scheme to flat.py's)."""
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, int(env), int(episode)])


class FlatFloorHumanoidEnv(DuckEnvBatch):
    """E parallel flat-floor H0 humanoids; obs layout in the module docstring."""

    OBS = OBS
    ACT = ACT

    def __init__(self, environments: int = 16, seed: int = 0,
                 perturbation_rad: float = 0.0,
                 library_path=None, lane_factory=None):
        if not 0.0 <= float(perturbation_rad) <= MAX_PERTURBATION_RAD:
            raise ValueError(
                f"perturbation_rad must be in [0, {MAX_PERTURBATION_RAD}]")
        self.E = int(environments)
        self._perturbation = float(perturbation_rad)
        self._seed = int(seed)
        self._library_path = library_path
        self._lane_factory = lane_factory or (
            lambda E, offsets: humanoid_native_lane.NativeHumanoidLane(
                E, joint_offsets=offsets, library_path=self._library_path))
        self._build_lane()
        self._tracker = reward_mod.GaitTracker(self.E)
        self._episode = np.zeros(self.E, np.int64)
        self._command = np.zeros(self.E)
        self._t = np.zeros(self.E, np.int64)
        self._done = np.zeros(self.E, bool)
        self._prev_action = np.zeros((self.E, ACT))
        self._phase0 = np.zeros(self.E)
        self._targets = np.tile(HOME, (self.E, 1))
        self._effective = self._clip_limits(self._targets)
        self._applied = self._effective
        self.reset()

    # ------------------------------------------------------------------
    def _build_lane(self) -> None:
        offsets = np.stack([
            _episode_rng(self._seed, e, 0).uniform(
                -self._perturbation, self._perturbation, ACT)
            for e in range(self.E)]) if self._perturbation else None
        self._lane = self._lane_factory(self.E, offsets)
        if self._library_path is None:
            self._library_path = self._lane.library_path
        self._min_height = MIN_HEIGHT_FRACTION * self._lane.home_root_height

    def _clip_limits(self, targets: np.ndarray) -> np.ndarray:
        lim = self._lane.joint_limits
        return np.clip(targets, lim[:, 0], lim[:, 1])

    # ------------------------------------------------------------------
    def reset(self, mask: np.ndarray | None = None,
              seed: int | None = None) -> np.ndarray:
        if seed is not None and int(seed) != self._seed:
            self._lane.close()
            self._seed = int(seed)
            self._build_lane()
            mask = None
        m = (np.ones(self.E, bool) if mask is None
             else np.asarray(mask, bool).reshape(self.E))
        if m.any():
            self._lane.restore(m)
            for e in np.flatnonzero(m):
                rng = _episode_rng(self._seed, int(e),
                                   int(self._episode[e]) + 1)
                # oversample the slowest command (longest balance demands),
                # exactly the duck's split
                draw = rng.random()
                self._command[e] = (COMMANDS_MPS[0] if draw < 0.5
                                    else COMMANDS_MPS[1] if draw < 0.75
                                    else COMMANDS_MPS[2])
                self._phase0[e] = 2.0 * math.pi * rng.random()
                self._episode[e] += 1
            self._t[m] = 0
            self._done[m] = False
            self._prev_action[m] = 0.0
            self._targets[m] = HOME
            self._effective = self._clip_limits(self._targets)
            self._applied = self._effective
            self._tracker.reset(m)
        state = self._lane.read()
        self._prev = self._reward_state(state, self._prev_action,
                                        np.zeros((self.E, ACT)))
        return self._observe(state)

    def set_command(self, commands) -> np.ndarray:
        """Pin per-env commanded forward velocity (m/s); returns fresh obs."""
        self._command[:] = np.broadcast_to(
            np.asarray(commands, dtype=np.float64), (self.E,))
        return self._observe(self._lane.read())

    @property
    def command(self) -> np.ndarray:
        return self._command.copy()

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray, on_tick=None):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(self.E, ACT),
                    -1.0, 1.0)
        live = ~self._done
        requested = HOME + ACTION_SCALE * a
        new_targets = np.clip(requested,
                              self._targets - MAX_TARGET_INCREMENT,
                              self._targets + MAX_TARGET_INCREMENT)
        self._targets = np.where(live[:, None], new_targets, self._targets)
        self._effective = self._clip_limits(self._targets)
        self._applied = self._effective

        iterations = np.zeros(self.E, np.int32)
        if on_tick is None and hasattr(self._lane, "tick_block"):
            rc, diagnostics = self._lane.tick_block(self._applied,
                                                    TICKS_PER_STEP)
            bad = [d for d in diagnostics if d["native_status"] != 0]
            if rc or bad:
                self._raise_fault(rc, diagnostics, bad, 0, a)
            iterations = np.asarray([d["iterations"] for d in diagnostics],
                                    np.int32)
            mid = self._lane.read()
            contact_ticks = np.asarray(mid.contact_ticks, np.int32).copy()
        else:
            contact_ticks = np.zeros((self.E, 2), np.int32)
            for tick in range(TICKS_PER_STEP):
                rc, diagnostics = self._lane.tick(self._applied)
                bad = [d for d in diagnostics if d["native_status"] != 0]
                if rc or bad:
                    self._raise_fault(rc, diagnostics, bad, tick, a)
                iterations = np.maximum(
                    iterations, [d["iterations"] for d in diagnostics])
                mid = self._lane.read()
                contact_ticks += mid.foot_contact
                if on_tick is not None:
                    on_tick(mid)

        state = mid
        finite = state.finite()
        cur = self._reward_state(state, a, self._torque(state))
        cur["contact_ticks"] = contact_ticks.copy()
        r = reward_mod.reward(self._prev, cur, a, self._command,
                              self._tracker, dt=CONTROL_DT)
        self._prev = cur

        self._t[live] += 1
        tilt = humanoid_native_lane.tilt(state.q[:, 3:7])
        fell = ((state.q[:, 2] < self._min_height) | (tilt > MAX_TILT_RAD)
                | ~finite)
        newly_done = live & (fell | (self._t >= HORIZON_STEPS))
        r = np.where(live, r, 0.0).astype(np.float32)
        self._done |= newly_done
        self._prev_action = np.where(live[:, None], a, self._prev_action)

        info = {"solver_iterations": iterations,
                "episode_time": (self._t * CONTROL_DT).astype(np.float32),
                "foot_contact": state.foot_contact.copy(),
                "command": self._command.copy(),
                "tilt_rad": tilt.astype(np.float32)}
        return self._observe(state), r, self._done.copy(), info

    # ------------------------------------------------------------------
    def _torque(self, state) -> np.ndarray:
        """Boundary PD estimate; effort caps are PER-JOINT (H0 tiers)."""
        raw = self._lane.kp * (self._applied - state.q[:, 7:]) \
            - self._lane.kv * state.v[:, 6:]
        return np.clip(raw, -self._lane.effort_cap, self._lane.effort_cap)

    def _phase(self) -> np.ndarray:
        return self._phase0 + 2.0 * math.pi \
            * (PHASE_HZ_BASE + PHASE_HZ_PER_MPS * self._command) \
            * self._t * CONTROL_DT

    def _reward_state(self, state, action, torque):
        return {"root_lin_vel": state.v[:, 0:3].copy(),
                "root_ang_vel": state.v[:, 3:6].copy(),
                "foot_contact": state.foot_contact.copy(),
                "sole_height": state.sole_height.copy(),
                "action": np.asarray(action).copy(),
                "torque": np.asarray(torque).copy(),
                "foot_x": state.foot_pos[:, :, 0].copy(),
                "joint_q": state.q[:, 7:].copy(),
                "phase": self._phase()}

    def _observe(self, state) -> np.ndarray:
        obs = np.zeros((self.E, OBS), np.float32)
        rot = humanoid_native_lane.quat_to_rot(state.q[:, 3:7])
        obs[:, 0:ACT] = state.q[:, 7:] - HOME
        obs[:, ACT:2 * ACT] = QDOT_OBS_SCALE * state.v[:, 6:]
        obs[:, 2 * ACT:_T] = self._prev_action
        obs[:, _T:_T + 3] = -rot[:, 2, :]                    # gravity, body
        obs[:, _T + 3:_T + 6] = np.einsum("eji,ej->ei", rot, state.v[:, 3:6])
        obs[:, _T + 6:_T + 9] = np.einsum("eji,ej->ei", rot, state.v[:, 0:3])
        obs[:, _T + 9] = self._command
        obs[:, _T + 12:_T + 14] = state.foot_contact
        phase = self._phase()
        obs[:, _T + 14] = np.sin(phase)
        obs[:, _T + 15] = np.cos(phase)
        return obs

    # ------------------------------------------------------------------
    def _raise_fault(self, rc, diagnostics, bad, tick, action):
        FAULT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        failing = sorted({d["environment"] for d in bad}) or list(range(self.E))
        first_path = None
        for e in failing:
            payload = {
                "schema": "duckgridwalk.humanoid_solver_fault/1",
                "environment": int(e), "status_rc": int(rc),
                "tick_of_policy_step": int(tick),
                "policy_step": int(self._t[e]),
                "command_mps": float(self._command[e]),
                "action": np.asarray(action)[e].tolist(),
                "effective_targets": self._effective[e].tolist(),
                "dt": SIM_DT,
                "max_iterations": humanoid_native_lane.MAX_SOLVER_ITERATIONS,
                "tolerance": humanoid_native_lane.IMPULSE_TOLERANCE,
                "diagnostics": [d for d in diagnostics
                                if d["environment"] == e],
                "all_diagnostics": diagnostics,
                "state": self._lane.state_dump(e),
            }
            path = FAULT_DIR / f"{stamp}-humanoid-env{int(e)}.json"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            first_path = first_path or path
        raise SolverFault(int(failing[0]), str(first_path),
                          f"idv1_step rc={rc} envs={failing}")

    def close(self) -> None:
        self._lane.close()
