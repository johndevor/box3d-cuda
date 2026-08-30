"""Backend-neutral contracts for parallel learners over batched worlds.

The CUDA benchmark uses two explicit batch axes: learner ``L`` and environment
``N``. Physics and camera kernels consume the flattened ``E=L*N`` axis while
policy parameters and rollout statistics retain ``[L,N,...]``. This module is
dependency-free so layout, seeding, and return estimation can be checked
without CUDA or PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


CONTRACT_ID = "box3d.parallel-vision-ppo/v1"


@dataclass(frozen=True)
class ParallelTrainerConfig:
    learners: int = 8
    environments_per_learner: int = 512
    horizon: int = 32
    cameras: int = 2
    camera_width: int = 8
    camera_height: int = 8
    action_dimensions: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2

    def __post_init__(self) -> None:
        integer_bounds = (
            ("learners", self.learners, 1, 256),
            ("environments_per_learner", self.environments_per_learner, 1, 1_048_576),
            ("horizon", self.horizon, 1, 4096),
            ("cameras", self.cameras, 1, 64),
            ("camera_width", self.camera_width, 1, 4096),
            ("camera_height", self.camera_height, 1, 4096),
            ("action_dimensions", self.action_dimensions, 1, 16),
        )
        for name, value, lower, upper in integer_bounds:
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower},{upper}]")
        for name, value, lower, upper in (
            ("gamma", self.gamma, 0.0, 1.0),
            ("gae_lambda", self.gae_lambda, 0.0, 1.0),
            ("clip_ratio", self.clip_ratio, 0.0, 1.0),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if not lower <= float(value) <= upper:
                raise ValueError(f"{name} must be in [{lower},{upper}]")
        if self.total_worlds > 1_048_576:
            raise ValueError("parallel trainer supports at most 1,048,576 worlds")
        if self.rays_per_world > 262_144:
            raise ValueError("camera packet exceeds 262,144 rays per world")
        if self.total_worlds * self.rays_per_world > 1_048_576:
            raise ValueError("camera launch exceeds 1,048,576 total rays")

    @property
    def total_worlds(self) -> int:
        return self.learners * self.environments_per_learner

    @property
    def rays_per_world(self) -> int:
        return self.cameras * self.camera_width * self.camera_height

    @property
    def observation_dimensions(self) -> int:
        # Per-pixel optical depth + instance ID, joint coordinates, goal error.
        return 2 * self.rays_per_world + self.action_dimensions + 1

    def flatten_world(self, learner: int, environment: int) -> int:
        if not 0 <= learner < self.learners:
            raise IndexError("learner index out of range")
        if not 0 <= environment < self.environments_per_learner:
            raise IndexError("environment index out of range")
        return learner * self.environments_per_learner + environment

    def unflatten_world(self, world: int) -> tuple[int, int]:
        if not 0 <= world < self.total_worlds:
            raise IndexError("world index out of range")
        return divmod(world, self.environments_per_learner)


def camera_pixel_packet(config: ParallelTrainerConfig) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Return camera-major, row-major immutable ray topology."""

    cameras: list[int] = []
    pixels: list[tuple[int, int]] = []
    for camera in range(config.cameras):
        for y in range(config.camera_height):
            for x in range(config.camera_width):
                cameras.append(camera)
                pixels.append((x, y))
    return tuple(cameras), tuple(pixels)


def learner_seed(base_seed: int, learner: int) -> int:
    """Stable independent seed with no dependence on Python hash randomization."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    if isinstance(learner, bool) or not isinstance(learner, int) or learner < 0:
        raise ValueError("learner must be a non-negative integer")
    return (base_seed * 0x9E3779B1 + (learner + 1) * 0x85EBCA77) & 0x7FFFFFFF


def generalized_advantage_estimate(
    rewards: Sequence[Sequence[float]],
    values: Sequence[Sequence[float]],
    terminated: Sequence[Sequence[bool]],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[list[float]], list[list[float]]]:
    """Dependency-free GAE for time-major ``[T,N]`` batches.

    ``values`` has ``T+1`` rows. A true termination at row ``t`` prevents both
    bootstrap and advantage propagation across that episode boundary.
    """

    if not rewards or len(values) != len(rewards) + 1 or len(terminated) != len(rewards):
        raise ValueError("GAE requires rewards/terminated [T,N] and values [T+1,N]")
    width = len(rewards[0])
    if width == 0:
        raise ValueError("GAE environment axis must be non-empty")
    rows = [*rewards, *values, *terminated]
    if any(len(row) != width for row in rows):
        raise ValueError("GAE inputs must be rectangular and share N")
    if not all(math.isfinite(float(item)) for row in rewards for item in row):
        raise ValueError("GAE rewards must be finite")
    if not all(math.isfinite(float(item)) for row in values for item in row):
        raise ValueError("GAE values must be finite")
    if not all(0.0 <= float(value) <= 1.0 for value in (gamma, gae_lambda)):
        raise ValueError("GAE gamma and lambda must be in [0,1]")

    advantages = [[0.0] * width for _ in rewards]
    running = [0.0] * width
    for time_index in range(len(rewards) - 1, -1, -1):
        for environment in range(width):
            continuation = 0.0 if terminated[time_index][environment] else 1.0
            delta = (
                float(rewards[time_index][environment])
                + float(gamma) * float(values[time_index + 1][environment]) * continuation
                - float(values[time_index][environment])
            )
            running[environment] = (
                delta
                + float(gamma) * float(gae_lambda) * continuation * running[environment]
            )
            advantages[time_index][environment] = running[environment]
    returns = [
        [advantages[t][n] + float(values[t][n]) for n in range(width)]
        for t in range(len(rewards))
    ]
    return advantages, returns


__all__ = [
    "CONTRACT_ID",
    "ParallelTrainerConfig",
    "camera_pixel_packet",
    "generalized_advantage_estimate",
    "learner_seed",
]
