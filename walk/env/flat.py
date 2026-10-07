"""FlatFloorDuckEnv: batched flat-floor Open Duck env over the native idv1 lane.

Implements walk/env/contract.py (OBS = 58, ACT = 14). One policy step =
10 native ticks x 0.002 s. Action semantics per PLAN.md / the plain-14
candidate: requested = HOME + 0.25 * action; targets slew-limited to
0.1048 rad per policy step against the previously stored targets; effective
targets are joint-limit clipped and held for all 10 ticks. The native lane
applies the PD law kp=13.37, kv=0, effort cap 3.23 per tick against the
current q/qdot (identical to run_home_hold.py's actuator computation).

Backend-agnostic by construction: all native specifics live in
walk/env/native_lane.py; pass `lane_factory` to swap in the cube-grid world
(workstream B) behind the same class interface.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path

import numpy as np

from . import native_lane, reward as reward_mod
from .contract import ACT, OBS, DuckEnvBatch, SolverFault

ROOT = Path(__file__).resolve().parents[2]
FAULT_DIR = ROOT / "runs" / "faults"

HOME = np.array([.002, .053, -.63, 1.368, -.784, 0., 0., 0., 0.,
                 -.003, -.065, .635, 1.379, -.796])
CONTROL_DT = 0.02
SIM_DT = 0.002
TICKS_PER_STEP = 10
ACTION_SCALE = 0.25
MAX_TARGET_INCREMENT = 5.24 * CONTROL_DT          # 0.1048 rad per policy step
import os as _os
# Affine gait clock: phase_hz = BASE + PER_MPS * command. Sweepable via env
# vars (read once at import; the kernel bakes the same values through the
# generated header, drift-checked by the duck_cuda test suite).
# History: fixed 2.5 Hz -> per-mps 16.67 (knife-edge: demanded exactly the
# evaluator's 30 mm minimum step) -> 10.0. Base term lets low speeds keep a
# workable cadence without knife-edging step length.
PHASE_HZ_BASE = float(_os.environ.get("DUCK_PHASE_HZ_BASE", "0.0"))
PHASE_HZ_PER_MPS = float(_os.environ.get("DUCK_PHASE_HZ_PER_MPS", "16.67"))
# 16.67: sweep-validated, twice re-confirmed empirically. Both attempts to move
# it off 16.67 on theoretical grounds (10.0 "knife-edge" theory, 12.5
# "constant-step" theory) regressed the trained lineage hard. The clock the
# policies entrain to beats the clock the arithmetic prefers. Do not move
# without a fresh sweep on a fresh lineage.
COMMANDS_MPS = (0.10, 0.15, 0.20)                  # per-episode forward commands
HORIZON_STEPS = 400                                # 8 s at 0.02 s per step
MIN_HEIGHT_FRACTION = 0.7                          # of the HOME root height
MAX_TILT_RAD = math.radians(45.0)
# Spec bound for the deterministic per-env reset perturbation. NOTE: until the
# workstream-A solver robustness repair lands, ANY nonzero joint perturbation
# (even 1e-4 rad) makes civ1 stall (CIV1_NO_CONVERGENCE, native_status=3)
# during the foot-settling impact ~0.23 s in — the documented degenerate
# contact-block failure. The default is therefore 0.0 (bit-exact known-good
# home-hold reset); pass perturbation_rad<=0.02 to enable once A lands.
MAX_PERTURBATION_RAD = 0.02
QDOT_OBS_SCALE = 0.05                              # matches the pinned candidate


def _episode_rng(seed: int, env: int, episode: int) -> np.random.Generator:
    """Counter-based RNG: per-env draws never depend on other envs' resets."""
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, int(env), int(episode)])


def _draw_randomization(rng, cfg):
    """The lane contract's per-episode DR draw stream (cuda_lane.py draws
    3-8), imported lazily so this module keeps its import graph."""
    from .cuda_lane import draw_randomization  # noqa: PLC0415
    return draw_randomization(rng, cfg)


class FlatFloorDuckEnv(DuckEnvBatch):
    """E parallel flat-floor ducks; see walk/env/README.md for the obs layout."""

    def __init__(self, environments: int = 16, seed: int = 0,
                 perturbation_rad: float = 0.0,
                 library_path=None, lane_factory=None,
                 randomization: dict | None = None):
        if not 0.0 <= float(perturbation_rad) <= MAX_PERTURBATION_RAD:
            raise ValueError(f"perturbation_rad must be in [0, {MAX_PERTURBATION_RAD}]")
        self.E = int(environments)
        self._perturbation = float(perturbation_rad)
        self._seed = int(seed)
        self._library_path = library_path
        # Domain randomization per walk/env/cuda_lane.py's normative spec:
        # per-episode mass/friction/kp/damping scales + command latency,
        # applied INSIDE the lane physics; requires a lane with
        # set_randomization (the duck_cuda lanes; the CPU idv1 lane cannot).
        rz = dict(randomization or {})
        self._rz = {k: float(rz.pop(k, 0.0)) for k in
                    ("r_mass", "r_friction", "r_kp", "r_damping")}
        self._rz_latency = int(rz.pop("max_latency_steps", 0))
        # ABI v7: one-sided gravity scale (authored magnitude = maximum);
        # drawn LAST and only when > 0, so pre-v7 configs keep their exact
        # RNG stream (see walk/env/cuda_lane.py contract, draw 8).
        self._rz["r_gravity"] = float(rz.pop("r_gravity", 0.0))
        if rz:
            raise ValueError(f"unknown randomization keys: {sorted(rz)}")
        self._rz_on = any(v > 0 for v in self._rz.values()) or self._rz_latency > 0
        self._lane_factory = lane_factory or (
            lambda E, offsets: native_lane.NativeDuckLane(
                E, joint_offsets=offsets, library_path=self._library_path))
        self._build_lane()
        if self._rz_on and not hasattr(self._lane, "set_randomization"):
            raise ValueError("randomization requires a lane with set_randomization")
        self._rand_scales = np.ones((self.E, 5))   # mass, mu, kp, damping, g
        self._latency = np.zeros(self.E, np.int64)
        self._ring_P = self._rz_latency + 1
        self._ring = np.zeros((self.E, self._ring_P, ACT))
        self._tracker = reward_mod.GaitTracker(self.E)
        self._episode = np.zeros(self.E, np.int64)   # per-env episode counter
        self._command = np.zeros(self.E)
        self._t = np.zeros(self.E, np.int64)         # policy steps this episode
        self._done = np.zeros(self.E, bool)
        self._prev_action = np.zeros((self.E, ACT))
        self._phase0 = np.zeros(self.E)              # per-episode gait-phase offset
        self._targets = np.tile(HOME, (self.E, 1))   # pre-clip slew reference
        self._effective = self._clip_limits(self._targets)
        self.reset()

    # ------------------------------------------------------------------
    def _build_lane(self) -> None:
        offsets = np.stack([
            _episode_rng(self._seed, e, 0).uniform(
                -self._perturbation, self._perturbation, ACT)
            for e in range(self.E)]) if self._perturbation else None
        self._lane = self._lane_factory(self.E, offsets)
        # Pin the resolved dylib for this process: later lane rebuilds (fault
        # recovery, reseed) must never recompile from possibly-edited source.
        if self._library_path is None:
            self._library_path = self._lane.library_path
        self._min_height = MIN_HEIGHT_FRACTION * self._lane.home_root_height

    def _clip_limits(self, targets: np.ndarray) -> np.ndarray:
        lim = self._lane.joint_limits
        return np.clip(targets, lim[:, 0], lim[:, 1])

    # ------------------------------------------------------------------
    def reset(self, mask: np.ndarray | None = None, seed: int | None = None) -> np.ndarray:
        if seed is not None and int(seed) != self._seed:
            # New seed => new deterministic per-env perturbations: rebuild the
            # lane so the initial snapshot embeds them, and reset every env.
            self._lane.close()
            self._seed = int(seed)
            self._build_lane()
            mask = None
        m = np.ones(self.E, bool) if mask is None else np.asarray(mask, bool).reshape(self.E)
        if m.any():
            self._lane.restore(m)
            for e in np.flatnonzero(m):
                rng = _episode_rng(self._seed, int(e), int(self._episode[e]) + 1)
                # oversample the hardest command (0.10: slow walking = longest
                # balance demands; it fails at every clock without extra data)
                draw = rng.random()
                self._command[e] = (COMMANDS_MPS[0] if draw < 0.5
                                    else COMMANDS_MPS[1] if draw < 0.75
                                    else COMMANDS_MPS[2])
                # random gait-phase offset: without it every episode starts in
                # the LEFT clock window, training a left-leading bias
                self._phase0[e] = 2.0 * math.pi * rng.random()
                if self._rz_on:
                    # draws 3-6: scales in fixed order; draw 7: latency;
                    # draw 8 (only if r_gravity > 0): one-sided gravity
                    self._rand_scales[e], self._latency[e] = \
                        _draw_randomization(rng, {
                            **self._rz,
                            "max_latency_steps": self._rz_latency})
                self._episode[e] += 1
            self._t[m] = 0
            self._done[m] = False
            self._prev_action[m] = 0.0
            self._targets[m] = HOME
            self._effective = self._clip_limits(self._targets)
            if self._rz_on:
                self._lane.set_randomization(
                    m, self._rand_scales[:, 0], self._rand_scales[:, 1],
                    self._rand_scales[:, 2], self._rand_scales[:, 3],
                    self._latency, gravity_scale=self._rand_scales[:, 4])
                self._ring[m] = self._effective[m][:, None, :]
            self._tracker.reset(m)
        state = self._lane.read()
        self._prev = self._reward_state(state, self._prev_action,
                                        np.zeros((self.E, ACT)))
        return self._observe(state)

    def set_command(self, commands) -> np.ndarray:
        """Override the per-env commanded forward velocity (m/s) mid-episode.

        Extension beyond the ABC used by walk/eval/capture.py so the strict
        evaluator can pin the +0.10/+0.15/+0.20 m/s episodes. Returns
        refreshed observations."""
        self._command[:] = np.broadcast_to(
            np.asarray(commands, dtype=np.float64), (self.E,))
        return self._observe(self._lane.read())

    @property
    def command(self) -> np.ndarray:
        return self._command.copy()

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray, on_tick=None):
        """Advance one 0.02 s policy step. `on_tick(LaneState)` is an optional
        per-2ms-tick observer used by walk/eval/capture.py (adds reads)."""
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(self.E, ACT), -1.0, 1.0)
        live = ~self._done
        requested = HOME + ACTION_SCALE * a
        new_targets = np.clip(requested, self._targets - MAX_TARGET_INCREMENT,
                              self._targets + MAX_TARGET_INCREMENT)
        # done envs hold their previous targets (no auto-reset; frozen).
        self._targets = np.where(live[:, None], new_targets, self._targets)
        self._effective = self._clip_limits(self._targets)
        if self._rz_on:
            # latency ring: write eff[t], physics consumes eff[t - latency]
            rows = np.arange(self.E)
            t = self._t
            self._ring[rows, t % self._ring_P] = self._effective
            idx = (t + self._ring_P - self._latency) % self._ring_P
            self._applied = self._ring[rows, idx]
        else:
            self._applied = self._effective

        iterations = np.zeros(self.E, np.int32)
        # per-tick foot contact: 2 ms flickers are invisible at the 20 ms
        # policy boundary, but the evaluator's 40 ms continuity windows see
        # them; the reward needs the tick-resolution count to shape them away.
        # Lanes with a native tick_block (CUDA) count contact ticks on device:
        # one launch + one readback per policy step instead of ten.
        if on_tick is None and hasattr(self._lane, "tick_block"):
            rc, diagnostics = self._lane.tick_block(self._applied, TICKS_PER_STEP)
            bad = [d for d in diagnostics if d["native_status"] != 0]
            if rc or bad:
                self._raise_fault(rc, diagnostics, bad, 0, a)
            iterations = np.asarray([d["iterations"] for d in diagnostics], np.int32)
            mid = self._lane.read()
            contact_ticks = np.asarray(mid.contact_ticks, np.int32).copy()
        else:
            contact_ticks = np.zeros((self.E, 2), np.int32)
            for tick in range(TICKS_PER_STEP):
                rc, diagnostics = self._lane.tick(self._applied)
                bad = [d for d in diagnostics if d["native_status"] != 0]
                if rc or bad:
                    self._raise_fault(rc, diagnostics, bad, tick, a)
                iterations = np.maximum(iterations,
                                        [d["iterations"] for d in diagnostics])
                mid = self._lane.read()
                contact_ticks += mid.foot_contact
                if on_tick is not None:
                    on_tick(mid)

        state = mid
        finite = state.finite()
        cur = self._reward_state(state, a, self._torque(state))
        cur["contact_ticks"] = contact_ticks.copy()
        r = reward_mod.reward(self._prev, cur, a, self._command, self._tracker,
                              dt=CONTROL_DT)
        self._prev = cur

        self._t[live] += 1
        tilt = self._tilt(state)
        fell = (state.q[:, 2] < self._min_height) | (tilt > MAX_TILT_RAD) | ~finite
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
    def _torque(self, state: native_lane.LaneState) -> np.ndarray:
        """Boundary PD torque estimate (same law the lane applies per tick)."""
        raw = (self._lane.kp * self._rand_scales[:, 2:3]) \
            * (self._applied - state.q[:, 7:]) \
            - self._lane.kv * state.v[:, 6:]
        return np.clip(raw, -self._lane.effort_cap, self._lane.effort_cap)

    @staticmethod
    def _tilt(state: native_lane.LaneState) -> np.ndarray:
        up = 1.0 - 2.0 * (np.square(state.q[:, 3]) + np.square(state.q[:, 4]))
        return np.arccos(np.clip(up, -1.0, 1.0))

    def _reward_state(self, state, action, torque):
        return {"root_lin_vel": state.v[:, 0:3].copy(),
                "root_ang_vel": state.v[:, 3:6].copy(),
                "foot_contact": state.foot_contact.copy(),
                "sole_height": state.sole_height.copy(),
                "action": np.asarray(action).copy(),
                "torque": np.asarray(torque).copy(),
                "foot_x": state.foot_pos[:, :, 0].copy(),
                "joint_q": state.q[:, 7:].copy(),
                # same clock the policy observes in obs[:, 56:58]
                "phase": self._phase0
                + 2.0 * math.pi
                * (PHASE_HZ_BASE + PHASE_HZ_PER_MPS * self._command)
                * self._t * CONTROL_DT}

    def _observe(self, state: native_lane.LaneState) -> np.ndarray:
        obs = np.zeros((self.E, OBS), np.float32)
        rot = native_lane.quat_to_rot(state.q[:, 3:7])          # body->world
        obs[:, 0:14] = state.q[:, 7:] - HOME
        obs[:, 14:28] = QDOT_OBS_SCALE * state.v[:, 6:]
        obs[:, 28:42] = self._prev_action
        obs[:, 42:45] = -rot[:, 2, :]                            # gravity, body frame
        obs[:, 45:48] = np.einsum("eji,ej->ei", rot, state.v[:, 3:6])
        obs[:, 48:51] = np.einsum("eji,ej->ei", rot, state.v[:, 0:3])
        obs[:, 51] = self._command
        obs[:, 54:56] = state.foot_contact
        phase = self._phase0 + 2.0 * math.pi \
            * (PHASE_HZ_BASE + PHASE_HZ_PER_MPS * self._command) \
            * self._t * CONTROL_DT
        obs[:, 56] = np.sin(phase)
        obs[:, 57] = np.cos(phase)
        return obs

    # ------------------------------------------------------------------
    def _raise_fault(self, rc, diagnostics, bad, tick, action):
        """Persist the failing envs' exact post-rollback state, then raise."""
        FAULT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        failing = sorted({d["environment"] for d in bad}) or list(range(self.E))
        first_path = None
        for e in failing:
            payload = {
                "schema": "duckgridwalk.solver_fault/1",
                "environment": int(e), "status_rc": int(rc),
                "tick_of_policy_step": int(tick),
                "policy_step": int(self._t[e]),
                "command_mps": float(self._command[e]),
                "action": np.asarray(action)[e].tolist(),
                "effective_targets": self._effective[e].tolist(),
                "dt": SIM_DT,
                "max_iterations": native_lane.MAX_SOLVER_ITERATIONS,
                "tolerance": native_lane.IMPULSE_TOLERANCE,
                "diagnostics": [d for d in diagnostics if d["environment"] == e],
                "all_diagnostics": diagnostics,
                "state": self._lane.state_dump(e),
            }
            path = FAULT_DIR / f"{stamp}-env{int(e)}.json"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            first_path = first_path or path
        raise SolverFault(int(failing[0]), str(first_path),
                          f"idv1_step rc={rc} envs={failing}")

    def close(self) -> None:
        self._lane.close()
