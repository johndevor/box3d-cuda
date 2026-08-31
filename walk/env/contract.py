"""Batched duck environment contract shared by env implementations and the trainer.

See PLAN.md. Implementations must be deterministic per seed and must never
return observations produced from a faulted native step.
"""
from __future__ import annotations

import abc
import numpy as np

ACT = 14
# joint q (14) + joint qdot (14) + previous action (14) + gravity in body
# frame (3) + root angular velocity (3) + root linear velocity (3) +
# commanded velocity (3) + foot contact flags (2) + phase clock (2)
OBS = 58


class SolverFault(RuntimeError):
    """Raised after persisting the exact failing native inputs to disk."""

    def __init__(self, env_index: int, saved_problem_path: str, message: str = ""):
        super().__init__(message or f"solver fault in env {env_index}: {saved_problem_path}")
        self.env_index = env_index
        self.saved_problem_path = saved_problem_path


class DuckEnvBatch(abc.ABC):
    """E parallel duck environments stepped in lockstep.

    One policy step = 10 native ticks x 0.002 s = 0.02 s. Actions in [-1, 1];
    targets = HOME + 0.25 * action, joint-limit clipped, slew-limited to
    0.1048 rad per policy step.
    """

    E: int
    OBS: int = OBS
    ACT: int = ACT

    @abc.abstractmethod
    def reset(self, mask: np.ndarray | None = None, seed: int | None = None) -> np.ndarray:
        """Reset masked (or all) envs; return float32 observations [E, OBS]."""

    @abc.abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Advance one policy step.

        action: float32 [E, ACT] in [-1, 1] (implementation clips).
        Returns (obs [E, OBS] f32, reward [E] f32, done [E] bool, info).
        info must include 'solver_iterations' [E] and may include
        'episode_time' [E], 'foot_contact' [E, 2].
        Envs are NOT auto-reset: done[i] stays terminal until reset(mask).
        """

    def close(self) -> None:
        return None


class StubEnv(DuckEnvBatch):
    """Deterministic synthetic env for trainer development and throughput tests.

    Dynamics: a damped double integrator per joint driven by the action, with a
    reward that increases when a fixed linear probe of state matches the
    commanded velocity — enough signal for PPO smoke tests, zero physics.
    """

    def __init__(self, environments: int = 16, seed: int = 0, horizon_steps: int = 400):
        self.E = int(environments)
        self._horizon = int(horizon_steps)
        self._rng = np.random.default_rng(seed)
        self._w = np.asarray(
            np.random.default_rng(1234).standard_normal((OBS,)) / np.sqrt(OBS), np.float32
        )
        self.reset()

    def reset(self, mask=None, seed=None):
        m = np.ones(self.E, bool) if mask is None else np.asarray(mask, bool)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if not hasattr(self, "_q"):
            self._q = np.zeros((self.E, ACT), np.float32)
            self._v = np.zeros((self.E, ACT), np.float32)
            self._prev = np.zeros((self.E, ACT), np.float32)
            self._t = np.zeros(self.E, np.int64)
            self._done = np.zeros(self.E, bool)
        self._q[m] = np.asarray(self._rng.uniform(-0.05, 0.05, (int(m.sum()), ACT)), np.float32)
        self._v[m] = 0
        self._prev[m] = 0
        self._t[m] = 0
        self._done[m] = False
        return self._obs()

    def _obs(self):
        out = np.zeros((self.E, OBS), np.float32)
        out[:, :ACT] = self._q
        out[:, ACT : 2 * ACT] = self._v
        out[:, 2 * ACT : 3 * ACT] = self._prev
        out[:, 44] = -1.0  # gravity z in body frame at rest
        out[:, 51] = 0.1  # commanded vx
        phase = (self._t % 50) / 50.0 * 2 * np.pi
        out[:, 56] = np.sin(phase)
        out[:, 57] = np.cos(phase)
        return out

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        live = ~self._done
        self._v[live] = 0.9 * self._v[live] + 0.1 * a[live]
        self._q[live] += 0.02 * self._v[live]
        self._prev[live] = a[live]
        self._t[live] += 1
        obs = self._obs()
        reward = np.where(
            live, -np.square(obs @ self._w - 0.1).astype(np.float32) - 1e-3 * np.square(a).sum(1), 0.0
        ).astype(np.float32)
        self._done |= self._t >= self._horizon
        return obs, reward, self._done.copy(), {"solver_iterations": np.zeros(self.E, np.int32)}
