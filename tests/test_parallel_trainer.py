import math
from pathlib import Path

import pytest

from box3d_cuda.parallel_trainer import (
    ASYNC_CONTRACT_ID,
    CONTRACT_ID,
    ParallelTrainerConfig,
    camera_pixel_packet,
    deterministic_episode_length,
    generalized_advantage_estimate,
    learning_curve_summary,
    learner_seed,
)


ROOT = Path(__file__).resolve().parents[1]


def test_default_layout_is_eight_independent_learners_and_4096_worlds() -> None:
    config = ParallelTrainerConfig()
    assert CONTRACT_ID == "box3d.parallel-vision-ppo/v1"
    assert ASYNC_CONTRACT_ID == "box3d.parallel-vision-ppo-async/v2"
    assert config.total_worlds == 4096
    assert config.rays_per_world == 128
    assert config.observation_dimensions == 259
    assert config.flatten_world(3, 17) == 3 * 512 + 17
    assert config.unflatten_world(config.flatten_world(3, 17)) == (3, 17)


def test_camera_packet_is_camera_major_and_pixel_row_major() -> None:
    config = ParallelTrainerConfig(
        learners=1, environments_per_learner=1, cameras=2,
        camera_width=2, camera_height=2,
    )
    cameras, pixels = camera_pixel_packet(config)
    assert cameras == (0, 0, 0, 0, 1, 1, 1, 1)
    assert pixels == ((0, 0), (1, 0), (0, 1), (1, 1)) * 2


def test_camera_launch_bound_fails_closed() -> None:
    with pytest.raises(ValueError, match="total rays"):
        ParallelTrainerConfig(
            learners=8, environments_per_learner=512,
            cameras=2, camera_width=16, camera_height=16,
        )


def test_gae_stops_at_independent_environment_termination() -> None:
    rewards = [[1.0, 1.0], [1.0, 1.0]]
    values = [[0.5, 0.5], [0.25, 0.25], [4.0, 4.0]]
    terminated = [[False, True], [False, False]]
    advantages, returns = generalized_advantage_estimate(
        rewards, values, terminated, gamma=1.0, gae_lambda=1.0
    )
    assert advantages == [[5.5, 0.5], [4.75, 4.75]]
    assert returns == [[6.0, 1.0], [5.0, 5.0]]


def test_learner_seeds_are_stable_distinct_and_bounded() -> None:
    seeds = [learner_seed(20260829, learner) for learner in range(32)]
    assert seeds == [learner_seed(20260829, learner) for learner in range(32)]
    assert len(set(seeds)) == len(seeds)
    assert all(0 <= seed <= 0x7FFFFFFF for seed in seeds)


def test_episode_lengths_are_deterministic_heterogeneous_and_bounded() -> None:
    lengths = [
        deterministic_episode_length(20260829, learner, environment, episode)
        for learner in range(4)
        for environment in range(8)
        for episode in range(2)
    ]
    assert lengths == [
        deterministic_episode_length(20260829, learner, environment, episode)
        for learner in range(4)
        for environment in range(8)
        for episode in range(2)
    ]
    assert min(lengths) >= 8
    assert max(lengths) <= 24
    assert len(set(lengths)) > 8


def test_learning_curve_requires_multi_seed_consistent_improvement() -> None:
    improving = [[float(update + learner) for learner in range(8)] for update in range(8)]
    summary = learning_curve_summary(improving)
    assert summary["accepted"] is True
    assert summary["improved_learners"] == 8

    mixed = [row[:] for row in improving]
    for update, row in enumerate(mixed):
        for learner in range(3):
            row[learner] = float(-update + learner)
    rejected = learning_curve_summary(mixed)
    assert rejected["accepted"] is False
    assert rejected["improved_learners"] == 5


def test_parallel_trainer_benchmark_contract_is_truthful() -> None:
    source = (ROOT / "benchmark_parallel_trainer.py").read_text()
    for token in (
        "native.coupled_step(",
        "native.camera_rays(",
        "native.ray_cast(",
        "native.camera_depth(",
        "torch.optim.Adam",
        "_gae_cuda",
        "_restore_masked",
        "_masked_restore_exact",
        "_fixed_action_reward_probe",
        "learning_curve_summary",
        '"reported_rgb_or_raster_pixels": False',
        '"asynchronous_partial_episode_reset": args.asynchronous_resets',
    ):
        assert token in source
    assert "torch.cuda.Event" in source
    assert "policy_parameter_delta_per_learner" in source
    assert "depth_pixels_per_second" in source
