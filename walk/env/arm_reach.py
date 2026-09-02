"""ArmReachEnv: batched fixed-base 6-axis arm reach env (both variants).

The arm twin of walk/env/flat.py / humanoid_flat.py on the same env
contract (walk/env/contract.DuckEnvBatch surface: reset(mask, seed) /
step(action) -> (obs, reward, done, info), envs are NOT auto-reset), over a
duck-typed lane (walk/env/arm_native_lane.NativeArmLane = f64 oracle,
walk/env/arm_cuda_lane.CudaArmLane = fp32 serial kernel build). One policy
step = 10 ticks x 0.002 s = 0.02 s (duck-stack cadence).

OBS (27, identical for both variants so one policy can span the family):
    [ 0: 6]  joint q (rad)
    [ 6:12]  QDOT_OBS_SCALE * joint qdot
    [12:15]  active target xyz (m, world; base at the origin)
    [15:18]  tip (flange) xyz (m, world)
    [18:21]  target - tip (m)
    [21:27]  previous action (live envs only)
ACT (6), a in [-1, 1], two header-selected contracts (ACTION_MODES; the
kernel's DW_ENV_ACTION_MODE; the lane reports which it compiled):
  "delta" (default): target_j = clip(target_j + a_j * MAX_TARGET_INCREMENT_j,
    limits) -- a = 0 HOLDS the current target exactly, |a| = 1 commands the
    URDF speed limit (MAX_TARGET_INCREMENT_j = velocity_j * CONTROL_DT).
    Adopted after the absolute contract's v5 CPU runs: with absolute
    targets the exploration noise (std ~0.4 on a full +-3.2 rad range,
    slew-saturated) jittered the tip 1.7 cm/step and a 14-boundary 2 cm
    hold was never sampled; with deltas "hold" is the tanh centre.
  "abs": requested_j = lower_j + (a_j + 1) / 2 * (upper_j - lower_j),
    slew-limited per policy step to +- MAX_TARGET_INCREMENT_j (the v1-v5
    lineage; kept byte-identical so those actors stay evaluable).
Either way the target is held for all 10 ticks and the PD (per-joint
kp/kv/effort tables from arm_lowering.gains) does the rest. `prev action`
in the obs is the previous action in the env's own contract.

TASK: a seeded sequence of targets uniformly sampled in the reachable
workspace (sample_target: uniform joint-space draw over TARGET_JOINT_BOX,
FK, keep if inside the tier ball around the HOME tip and clear of the
judge's proxies -- reachable BY CONSTRUCTION); the active target advances
when the tip has been within the judge's acquisition radius at
ACQ_HOLD_STEPS consecutive policy-step reads (>= the judge's 0.25 s tick
window). Tier (TIER_RADIUS_FRAC, judge-owned) is drawn per episode when
tier=None (training) or pinned (evaluation, pin_tier).

TERMINATION: proxy violation (floor / base-column clause of the frozen
judge, applied to tip / wrist / elbow), non-finite state, or the 400-step
(8 s) horizon. Joint-limit and speed clauses are NOT terminal (regularized
in the reward; the judge fails them).
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import arm_reward as reward_mod
from .contract import DuckEnvBatch, SolverFault

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "arm") not in sys.path:
    sys.path.insert(0, str(ROOT / "arm"))
import arm_lowering as al  # noqa: E402
from walk.eval import arm_reach_judge as judge  # noqa: E402  (frozen clauses)

FAULT_DIR = ROOT / "runs" / "faults"

ACT = al.J
OBS = 27
ACTION_MODES = ("abs", "delta")       # index == DW_ENV_ACTION_MODE
ACTION_MODE = "delta"                 # default contract (header default)
CONTROL_DT = al.CONTROL_DT
SIM_DT = al.SIM_DT
TICKS_PER_STEP = al.TICKS_PER_CONTROL
HORIZON_STEPS = 400
QDOT_OBS_SCALE = 0.25                 # URDF speeds <= 3.6 rad/s -> |obs| <= 0.9
MAX_PERTURBATION_RAD = 0.05           # reset joint perturbation bound
ACQ_HOLD_STEPS = judge.ACQ_HOLD_STEPS       # 14 consecutive boundary reads
# reachable-workspace sampler: uniform joint box (rad), a6 pinned 0 (a pure
# tool roll: the flange lies on the a6 axis, so a6 never moves the tip).
# a3 >= 0.4 keeps the elbow bent >= 23 deg: the straight arm (a3 = 0) is the
# workspace boundary, where a 2 cm target is a singular full-stretch pose
# and the forearm sweeps the floor on the way (measured: proxy crashes).
TARGET_JOINT_BOX = ((-1.2, 1.2), (-2.0, 0.2), (0.4, 2.2),
                    (-0.6, 0.6), (-1.6, 1.6), (0.0, 0.0))
SAMPLE_BATCH = 256                    # rejection-sampling batch (vectorised FK)
SAMPLE_MAX_TRIES = 10000              # batches; tier-0 accepts ~1 % of draws


def _episode_rng(seed: int, env: int, episode: int) -> np.random.Generator:
    """Counter-based RNG (identical scheme to flat.py's)."""
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, int(env), int(episode)])


def episode_draw(spec: al.ArmSpec, seed: int, env: int, episode: int,
                 tier_pin: int | None):
    """The FROZEN per-episode draw order shared by ArmReachEnv.reset() and
    the device-path lane (walk/env/arm_cuda_lane.CudaArmLane.reset_policy):
    rng = _episode_rng(seed, env, episode); tier = rng.integers(len(TIERS))
    (drawn even when pinned, so pinning never shifts the stream); then the
    targets are drawn lazily from `rng` in acquisition order with
    sample_target -- the env draws target k+1 when k is acquired, the lane
    draws one target AHEAD (a queued next target); both consume the same
    rng in the same order, so the presented sequences are bit-identical.
    Returns (rng, tier, first_target)."""
    rng = _episode_rng(seed, env, episode)
    drawn = int(rng.integers(len(judge.TIERS)))
    tier = drawn if tier_pin is None else int(tier_pin)
    return rng, tier, sample_target(spec, rng, tier)


def abs_to_delta_action(spec: al.ArmSpec, a_abs: np.ndarray,
                        targets: np.ndarray) -> np.ndarray:
    """Re-express an ABSOLUTE limit-scaled action as the DELTA action that
    reaches the same slew-limited target from `targets` [E,J] (the env's
    current targets): a_delta = clip((requested - targets) / MAX_INC, +-1).
    Exact up to f64 rounding of the product a_delta * MAX_INC."""
    lim = al.joint_limits(spec)
    inc = al.velocity_limits(spec) * CONTROL_DT
    a = np.clip(np.asarray(a_abs, float), -1.0, 1.0)
    requested = lim[:, 0] + 0.5 * (a + 1.0) * (lim[:, 1] - lim[:, 0])
    return np.clip((requested - np.asarray(targets, float)) / inc, -1.0, 1.0)


class DeltaActionAdapter:
    """Wrap a policy that emits ABSOLUTE actions (e.g. the scripted IK
    baseline) for a DELTA-contract env. Shadows the env's target state
    from the home pose with the env's own arithmetic, so it needs no env
    handle; create one per episode (fresh targets = home)."""

    def __init__(self, abs_policy, spec: al.ArmSpec, environments: int = 1):
        self.policy = abs_policy
        self.spec = spec
        self._lim = al.joint_limits(spec)
        self._inc = al.velocity_limits(spec) * CONTROL_DT
        self._targets = np.tile(np.asarray(spec.home_q, float), (environments, 1))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs)
        if self._targets.shape[0] != obs.shape[0]:
            self._targets = np.tile(np.asarray(self.spec.home_q, float),
                                    (obs.shape[0], 1))
        a = abs_to_delta_action(self.spec, self.policy(obs), self._targets)
        self.register(a)
        return a.astype(np.float32)

    def register(self, a_delta: np.ndarray) -> None:
        """Advance the target shadow by a delta action applied to the env
        from elsewhere (interleaved exploration steps in the parity tests)."""
        a = np.clip(np.asarray(a_delta, float), -1.0, 1.0)
        self._targets = np.clip(self._targets + a * self._inc,
                                self._lim[:, 0], self._lim[:, 1])


def sample_target(spec: al.ArmSpec, rng: np.random.Generator, tier: int
                  ) -> np.ndarray:
    """One reachable target for `tier`: FK of a uniform joint-box draw,
    accepted inside the tier ball around the HOME tip and proxy-clear.
    Rejection sampling in batches of SAMPLE_BATCH; the FIRST accepted
    candidate in draw order is returned, so the distribution (and the
    seeded sequence) is exactly that of one-at-a-time rejection."""
    box = np.asarray(TARGET_JOINT_BOX)
    lim = al.joint_limits(spec)
    lo = np.maximum(box[:, 0], lim[:, 0])
    hi = np.minimum(box[:, 1], lim[:, 1])
    radius = judge.tier_radius(spec.variant, tier)
    center = al.home_tip(spec)
    for _ in range(SAMPLE_MAX_TRIES):
        q = rng.uniform(lo, hi, (SAMPLE_BATCH, al.J))
        tip, origins = al.fk_batch(spec, q)
        ok = np.linalg.norm(tip - center, axis=1) <= radius
        ok &= ~judge.proxy_violation(spec.variant, tip, origins[:, 4],
                                     origins[:, 2])
        hit = np.flatnonzero(ok)
        if hit.size:
            return tip[hit[0]].copy()
    raise RuntimeError("target sampler exhausted (tier %d)" % tier)


def link_origins_from_body_state(spec: al.ArmSpec, body_state: np.ndarray,
                                 link: int) -> np.ndarray:
    """[E,3] world origin of link_k (k = 1..6) = COM - R @ com_offset."""
    b = np.asarray(body_state, float)[:, 1 + link, :]
    rot = np.stack([al.quat_to_rot(qq) for qq in b[:, 3:7]])
    return b[:, :3] - rot @ np.asarray(spec.links[link - 1].com)


class ArmReachEnv(DuckEnvBatch):
    """E parallel fixed-base arms; obs/act layout in the module docstring."""

    OBS = OBS
    ACT = ACT

    def __init__(self, environments: int = 16, seed: int = 0,
                 perturbation_rad: float = 0.0, variant: str = "kr240",
                 tier: int | None = None, library_path=None,
                 lane_factory=None, action_mode: str | None = None):
        if not 0.0 <= float(perturbation_rad) <= MAX_PERTURBATION_RAD:
            raise ValueError(
                f"perturbation_rad must be in [0, {MAX_PERTURBATION_RAD}]")
        self.spec = al.spec(variant)
        self.variant = self.spec.variant
        self.E = int(environments)
        self._perturbation = float(perturbation_rad)
        self._seed = int(seed)
        self._tier_pin = None if tier is None else int(tier)
        self._library_path = library_path
        self._lane_factory = lane_factory or (
            lambda E, offsets: _default_lane(self.variant, E, offsets,
                                             self._library_path))
        if action_mode is not None and action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}")
        self._action_mode_arg = action_mode
        self._build_lane()
        self.reach = al.reach(self.spec)
        self.joint_limits = al.joint_limits(self.spec)
        self.velocity_limits = al.velocity_limits(self.spec)
        self.max_target_increment = self.velocity_limits * CONTROL_DT
        self.acq_radius = judge.ACQ_RADIUS_M[self.variant]
        self.home_q = np.asarray(self.spec.home_q)
        self._episode = np.zeros(self.E, np.int64)
        self._t = np.zeros(self.E, np.int64)
        self._done = np.zeros(self.E, bool)
        self._prev_action = np.zeros((self.E, ACT))
        self._targets = np.tile(self.home_q, (self.E, 1))    # PD slew ref
        self._applied = self._clip_limits(self._targets)
        self._rng = [None] * self.E
        self._tier = np.zeros(self.E, np.int64)
        self._target_index = np.zeros(self.E, np.int64)
        self._target = np.zeros((self.E, 3))
        self._hold = np.zeros(self.E, np.int64)
        self._acquired_total = np.zeros(self.E, np.int64)
        self.reset()

    # ------------------------------------------------------------------
    def _build_lane(self) -> None:
        offsets = np.stack([
            _episode_rng(self._seed, e, 0).uniform(
                -self._perturbation, self._perturbation, ACT)
            for e in range(self.E)]) if self._perturbation else None
        self._lane = self._lane_factory(self.E, offsets)
        if self._library_path is None:
            self._library_path = getattr(self._lane, "library_path", None)
        if getattr(self._lane, "variant", self.variant) != self.variant:
            raise ValueError("lane variant %r != env variant %r"
                             % (self._lane.variant, self.variant))
        # action contract: explicit arg, else the lane's compiled contract
        # (CudaArmLane reads DW_ENV_ACTION_MODE), else the module default;
        # a lane compiled for the other contract is refused.
        lane_mode = getattr(self._lane, "action_mode", None)
        self.action_mode = self._action_mode_arg or lane_mode or ACTION_MODE
        if lane_mode is not None and lane_mode != self.action_mode:
            raise ValueError(f"lane action contract {lane_mode!r} != env "
                             f"action_mode {self.action_mode!r}")

    def _clip_limits(self, targets: np.ndarray) -> np.ndarray:
        lim = al.joint_limits(self.spec)
        return np.clip(targets, lim[:, 0], lim[:, 1])

    def pin_tier(self, tier: int | None) -> None:
        """Evaluation hook: pin the difficulty tier drawn at the NEXT reset
        (None restores per-episode uniform tier draws)."""
        self._tier_pin = None if tier is None else int(tier)

    @property
    def tier(self) -> np.ndarray:
        return self._tier.copy()

    @property
    def target(self) -> np.ndarray:
        return self._target.copy()

    @property
    def target_index(self) -> np.ndarray:
        return self._target_index.copy()

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
                # draw order (frozen, shared with the device-path lane):
                # tier, then targets lazily from rng (episode_draw)
                rng, tier, target = episode_draw(
                    self.spec, self._seed, int(e), int(self._episode[e]) + 1,
                    self._tier_pin)
                self._tier[e] = tier
                self._rng[e] = rng
                self._target[e] = target
                self._episode[e] += 1
            self._t[m] = 0
            self._done[m] = False
            self._prev_action[m] = 0.0
            self._targets[m] = self.home_q
            self._applied = self._clip_limits(self._targets)
            self._target_index[m] = 0
            self._hold[m] = 0
            self._acquired_total[m] = 0
        state = self._lane.read()
        return self._observe(state)

    # ------------------------------------------------------------------
    def _geometry(self, state):
        tip = al.tip_from_body_state(self.spec, state.body_state)
        wrist = link_origins_from_body_state(self.spec, state.body_state, 5)
        elbow = link_origins_from_body_state(self.spec, state.body_state, 3)
        return tip, wrist, elbow

    def step(self, action: np.ndarray, on_tick=None):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(self.E, ACT),
                    -1.0, 1.0)
        live = ~self._done
        lim = self.joint_limits
        if self.action_mode == "delta":
            # target += a * MAX_INC, clamped to the limits (a = 0 holds)
            requested = self._targets + a * self.max_target_increment
            new_targets = np.clip(requested, lim[:, 0], lim[:, 1])
        else:
            requested = lim[:, 0] + 0.5 * (a + 1.0) * (lim[:, 1] - lim[:, 0])
            new_targets = np.clip(requested,
                                  self._targets - self.max_target_increment,
                                  self._targets + self.max_target_increment)
        previous_targets = self._targets
        self._targets = np.where(live[:, None], new_targets, self._targets)
        # commanded speed as a fraction of the URDF limit (reward v3; 0 for
        # done envs whose targets are frozen)
        command_speed = (self._targets - previous_targets) / self.max_target_increment
        self._applied = self._clip_limits(self._targets)

        iterations = np.zeros(self.E, np.int32)
        if on_tick is None and hasattr(self._lane, "tick_block"):
            rc, diagnostics = self._lane.tick_block(self._applied,
                                                    TICKS_PER_STEP)
            bad = [d for d in diagnostics if d["native_status"] != 0]
            if rc or bad:
                self._raise_fault(rc, diagnostics, bad, 0, a)
            iterations = np.asarray([d["iterations"] for d in diagnostics],
                                    np.int32)
        else:
            for tick in range(TICKS_PER_STEP):
                rc, diagnostics = self._lane.tick(self._applied)
                bad = [d for d in diagnostics if d["native_status"] != 0]
                if rc or bad:
                    self._raise_fault(rc, diagnostics, bad, tick, a)
                iterations = np.maximum(
                    iterations, [d["iterations"] for d in diagnostics])
                if on_tick is not None:
                    on_tick(self._lane.read(), self)
        state = self._lane.read()
        finite = state.finite()
        tip, wrist, elbow = self._geometry(state)
        dist = np.linalg.norm(tip - self._target, axis=1)
        inside = dist <= self.acq_radius
        self._hold = np.where(inside, self._hold + 1, 0)
        acquired = live & (self._hold >= ACQ_HOLD_STEPS)
        proxy = judge.proxy_violation(self.variant, tip, wrist, elbow)
        torque = self._torque(state)
        r = reward_mod.reward(
            {"tip_dist": dist, "hold": self._hold, "acquired": acquired,
             "command_speed": command_speed, "torque": torque,
             "qd": state.v[:, 6:], "proxy_violation": proxy},
            a, self._prev_action, self.reach, self._lane.effort_cap,
            self.velocity_limits, self.acq_radius, ACQ_HOLD_STEPS)
        # advance acquired envs to their next target (fresh episode rng draw)
        for e in np.flatnonzero(acquired):
            self._acquired_total[e] += 1
            self._target_index[e] += 1
            self._hold[e] = 0
            self._target[e] = sample_target(self.spec, self._rng[e],
                                            int(self._tier[e]))
        self._t[live] += 1
        crashed = proxy | ~finite
        newly_done = live & (crashed | (self._t >= HORIZON_STEPS))
        r = np.where(live, r, 0.0).astype(np.float32)
        self._done |= newly_done
        self._prev_action = np.where(live[:, None], a, self._prev_action)
        info = {"solver_iterations": iterations,
                "episode_time": (self._t * CONTROL_DT).astype(np.float32),
                "tip_dist": dist.astype(np.float32),
                "acquired": acquired.copy(),
                "acquired_total": self._acquired_total.copy(),
                "target_index": self._target_index.copy(),
                "tier": self._tier.copy(),
                "proxy_violation": proxy.copy()}
        return self._observe(state, tip), r, self._done.copy(), info

    # ------------------------------------------------------------------
    def _torque(self, state) -> np.ndarray:
        raw = self._lane.kp * (self._applied - state.q[:, 7:]) \
            - self._lane.kv * state.v[:, 6:]
        return np.clip(raw, -self._lane.effort_cap, self._lane.effort_cap)

    def _observe(self, state, tip=None) -> np.ndarray:
        if tip is None:
            tip = al.tip_from_body_state(self.spec, state.body_state)
        obs = np.zeros((self.E, OBS), np.float32)
        obs[:, 0:6] = state.q[:, 7:]
        obs[:, 6:12] = QDOT_OBS_SCALE * state.v[:, 6:]
        obs[:, 12:15] = self._target
        obs[:, 15:18] = tip
        obs[:, 18:21] = self._target - tip
        obs[:, 21:27] = self._prev_action
        return obs

    # ------------------------------------------------------------------
    def _raise_fault(self, rc, diagnostics, bad, tick, action):
        FAULT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        failing = sorted({d["environment"] for d in bad}) or list(range(self.E))
        first_path = None
        for e in failing:
            payload = {
                "schema": "duckgridwalk.arm_solver_fault/1",
                "variant": self.variant,
                "environment": int(e), "status_rc": int(rc),
                "tick_of_policy_step": int(tick),
                "policy_step": int(self._t[e]),
                "target_xyz": self._target[e].tolist(),
                "action": np.asarray(action)[e].tolist(),
                "effective_targets": self._applied[e].tolist(),
                "dt": SIM_DT,
                "diagnostics": [d for d in diagnostics
                                if d["environment"] == e],
                "all_diagnostics": diagnostics,
                "state": self._lane.state_dump(e),
            }
            path = FAULT_DIR / f"{stamp}-arm-{self.variant}-env{int(e)}.json"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            first_path = first_path or path
        raise SolverFault(int(failing[0]), str(first_path),
                          f"arm step rc={rc} envs={failing}")

    def close(self) -> None:
        self._lane.close()


def _default_lane(variant, E, offsets, library_path):
    from .arm_native_lane import NativeArmLane  # noqa: PLC0415
    return NativeArmLane(E, variant=variant, joint_offsets=offsets,
                         library_path=library_path)
