"""GPU-resident multi-policy PPO and batched analytic-camera baseline.

This is an execution/throughput gate, not a task-solution claim. It proves that
independent policy parameter sets can collect physics + vision rollouts and
update together without moving rollout tensors through host memory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .benchmark_coupled import _config, make_workload
from .extension import load_extension
from .parallel_trainer import (
    ASYNC_CONTRACT_ID,
    CONTRACT_ID,
    ParallelTrainerConfig,
    camera_pixel_packet,
    learner_seed,
    learning_curve_summary,
)


HIDDEN_DIMENSIONS = 64
PPO_EPOCHS = 2
UPDATES = 2
BASE_SEED = 20260829
GOAL_X_M = 0.85
MAXIMUM_CAMERA_DISTANCE_M = 3.0


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learners", type=int, default=8)
    parser.add_argument("--environments-per-learner", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--ppo-epochs", type=int, default=PPO_EPOCHS)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument(
        "--asynchronous-resets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--minimum-episode-steps", type=int, default=8)
    parser.add_argument("--maximum-episode-steps", type=int, default=24)
    return parser.parse_args()


class BatchedActorCritic:
    """Small independent actor/value networks stored in learner-major tensors."""

    def __init__(self, torch, config: ParallelTrainerConfig, seeds: list[int]) -> None:
        self.torch = torch
        learners = config.learners
        observations = config.observation_dimensions
        actions = config.action_dimensions
        device = torch.device("cuda")
        scale1 = math.sqrt(2.0 / (observations + HIDDEN_DIMENSIONS))
        scale2 = math.sqrt(2.0 / (HIDDEN_DIMENSIONS + actions))
        if len(seeds) != learners:
            raise ValueError("policy requires one explicit seed per learner")

        parameter_block = 0

        def learner_randn(*shape):
            nonlocal parameter_block
            rows = []
            for seed in seeds:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed + 104_729 * parameter_block)
                rows.append(torch.randn(shape, generator=generator, device=device))
            parameter_block += 1
            return torch.stack(rows)

        self.parameters = [
            torch.nn.Parameter(
                learner_randn(observations, HIDDEN_DIMENSIONS) * scale1
            ),
            torch.nn.Parameter(torch.zeros(learners, HIDDEN_DIMENSIONS, device=device)),
            torch.nn.Parameter(
                learner_randn(HIDDEN_DIMENSIONS, actions) * scale2
            ),
            torch.nn.Parameter(torch.zeros(learners, actions, device=device)),
            torch.nn.Parameter(
                learner_randn(HIDDEN_DIMENSIONS, 1)
                * math.sqrt(2.0 / (HIDDEN_DIMENSIONS + 1))
            ),
            torch.nn.Parameter(torch.zeros(learners, 1, device=device)),
            torch.nn.Parameter(torch.full((learners, actions), -1.5, device=device)),
        ]

    def forward(self, observations):
        torch = self.torch
        w1, b1, wa, ba, wv, bv, log_standard_deviation = self.parameters
        hidden = torch.tanh(torch.einsum("lno,loh->lnh", observations, w1) + b1[:, None, :])
        mean = torch.einsum("lnh,lha->lna", hidden, wa) + ba[:, None, :]
        value = (torch.einsum("lnh,lhv->lnv", hidden, wv) + bv[:, None, :]).squeeze(-1)
        return mean, value, log_standard_deviation[:, None, :]


def _normal_log_probability(torch, raw_action, mean, log_standard_deviation):
    inverse_variance_error = (raw_action - mean) * torch.exp(-log_standard_deviation)
    return (
        -0.5 * inverse_variance_error.square()
        - log_standard_deviation
        - 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=-1)


def _squashed_log_probability(torch, raw_action, mean, log_standard_deviation):
    action = torch.tanh(raw_action)
    correction = torch.log(1.0 - action.square() + 1.0e-6).sum(dim=-1)
    return _normal_log_probability(
        torch, raw_action, mean, log_standard_deviation
    ) - correction


def _camera_fixture(torch, config: ParallelTrainerConfig, bundle):
    camera_indices, pixel_xy = camera_pixel_packet(config)
    device = bundle["state"].device
    focal = float(config.camera_width)
    return {
        "body_enabled": torch.ones(
            (config.total_worlds, bundle["state"].shape[1]),
            dtype=torch.uint8,
            device=device,
        ),
        "parent": torch.tensor((-1, 3), dtype=torch.int64, device=device),
        "position": torch.tensor(
            ((0.4, 0.2, -2.0), (0.0, 0.0, -0.4)),
            dtype=torch.float32,
            device=device,
        ),
        "quaternion": torch.tensor(
            ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
            dtype=torch.float32,
            device=device,
        ),
        "intrinsics": torch.tensor(
            (
                (
                    focal,
                    focal,
                    config.camera_width / 2.0,
                    config.camera_height / 2.0,
                    MAXIMUM_CAMERA_DISTANCE_M,
                ),
            )
            * config.cameras,
            dtype=torch.float32,
            device=device,
        ),
        "pixel_camera": torch.tensor(camera_indices, dtype=torch.int64, device=device),
        "pixel_xy": torch.tensor(pixel_xy, dtype=torch.int64, device=device),
    }


def _camera_observation(native, torch, config, bundle, cameras, joint_coordinate):
    origins, directions, maximum, forward = native.camera_rays(
        bundle["state"], cameras["parent"], cameras["position"],
        cameras["quaternion"], cameras["intrinsics"],
        cameras["pixel_camera"], cameras["pixel_xy"],
    )
    distance, body_index, _normal = native.ray_cast(
        bundle["state"], bundle["half"], cameras["body_enabled"], origins,
        directions, maximum,
    )
    depth, _hit_range = native.camera_depth(distance, body_index, forward)
    normalized_depth = depth / MAXIMUM_CAMERA_DISTANCE_M
    normalized_instance = (body_index + 1).to(dtype=torch.float32) / float(
        bundle["state"].shape[1]
    )
    goal_error = (GOAL_X_M - bundle["state"][:, 4, 0:1]).clamp(-1.0, 1.0)
    flat = torch.cat(
        (
            normalized_depth,
            normalized_instance,
            joint_coordinate / math.pi,
            goal_error,
        ),
        dim=1,
    )
    return flat.reshape(
        config.learners, config.environments_per_learner,
        config.observation_dimensions,
    ), depth, body_index


def _native_coupled_step(native, torch, bundle, joint_coordinate, actions, config):
    solver = _config()
    worlds = config.total_worlds
    target = torch.clamp(
        joint_coordinate + 0.35 * actions.reshape(worlds, config.action_dimensions),
        bundle["lower"][None, :], bundle["upper"][None, :],
    )
    effort = torch.tensor(
        (80.0, 60.0), dtype=torch.float32, device=bundle["state"].device
    ).expand(worlds, -1)
    result = native.coupled_step(
        bundle["state"], bundle["inverse_mass"], bundle["half"],
        bundle["inverse_inertia"], bundle["joint_indices"],
        bundle["joint_types"], bundle["parent_anchor"],
        bundle["child_anchor"], bundle["axis"], bundle["reference"],
        bundle["lower"], bundle["upper"], bundle["damping"],
        bundle["motor_enabled"], torch.zeros_like(target), target,
        bundle["stiffness"], effort, bundle["joint_cache"], bundle["pairs"],
        bundle["contact_feature_ids"], bundle["contact_impulse_cache"],
        float(solver.joints.warm_start_factor), float(solver.joints.dt),
        int(solver.joints.substeps), float(solver.joints.gravity_y),
        float(solver.contacts.restitution), float(solver.contacts.friction),
        float(solver.contacts.contact_generation_distance),
        float(solver.contacts.position_slop),
        float(solver.joints.position_correction),
        float(solver.contacts.angular_damping),
        int(solver.joints.solver_iterations), float(solver.contacts.sat_epsilon),
        float(solver.joints.position_slop), float(solver.joints.angular_slop),
        float(solver.joints.maximum_linear_repair_m),
        float(solver.joints.maximum_angular_repair_rad), False, 1.0,
    )
    bundle["state"] = result[0]
    bundle["joint_cache"] = result[7]
    bundle["contact_feature_ids"] = result[10]
    bundle["contact_impulse_cache"] = result[11]
    return result


def _fixed_action_reward_probe(native, torch, config, bundle, reset_snapshot, steps=8):
    """Prove that the reward changes under simple bounded control choices."""

    candidates = (
        (-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0),
        (0.0, -1.0), (0.0, 0.0), (0.0, 1.0),
        (1.0, -1.0), (1.0, 0.0), (1.0, 1.0),
    )
    initial_distance = torch.abs(GOAL_X_M - reset_snapshot["state"][:, 4, 0])
    results = []
    for candidate in candidates:
        _restore(bundle, reset_snapshot)
        joint_coordinate = bundle["initial_q"]
        action = torch.tensor(
            candidate, dtype=torch.float32, device=bundle["state"].device
        ).expand(config.learners, config.environments_per_learner, -1)
        contact_ever = torch.zeros(
            config.total_worlds, dtype=torch.bool, device=bundle["state"].device
        )
        for _ in range(steps):
            step_result = _native_coupled_step(
                native, torch, bundle, joint_coordinate, action, config
            )
            joint_coordinate = step_result[1]
            contact_ever |= step_result[8].to(dtype=torch.bool).any(dim=1)
        final_distance = torch.abs(GOAL_X_M - bundle["state"][:, 4, 0])
        progress = initial_distance - final_distance
        results.append({
            "action": list(candidate),
            "mean_goal_progress_m": float(progress.mean().item()),
            "contact_fraction": float(contact_ever.float().mean().item()),
        })
    _restore(bundle, reset_snapshot)
    progress_values = [result["mean_goal_progress_m"] for result in results]
    return {
        "steps": steps,
        "candidates": results,
        "best_mean_goal_progress_m": max(progress_values),
        "worst_mean_goal_progress_m": min(progress_values),
        "progress_spread_m": max(progress_values) - min(progress_values),
    }


def _gae_cuda(torch, rewards, values, terminated, gamma: float, gae_lambda: float):
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for time_index in range(rewards.shape[0] - 1, -1, -1):
        continuation = (~terminated[time_index]).to(dtype=rewards.dtype)
        delta = (
            rewards[time_index]
            + gamma * values[time_index + 1] * continuation
            - values[time_index]
        )
        running = delta + gamma * gae_lambda * continuation * running
        advantages[time_index] = running
    return advantages, advantages + values[:-1]


def _snapshot(bundle):
    return {
        name: bundle[name].clone()
        for name in (
            "state", "joint_cache", "contact_feature_ids", "contact_impulse_cache"
        )
    }


def _restore(bundle, snapshot) -> None:
    for name, value in snapshot.items():
        bundle[name].copy_(value)


def _restore_masked(torch, bundle, snapshot, environment_mask) -> None:
    for name, reference in snapshot.items():
        value = bundle[name]
        shape = (environment_mask.shape[0],) + (1,) * (value.ndim - 1)
        value.copy_(torch.where(environment_mask.reshape(shape), reference, value))


def _masked_restore_exact(torch, bundle, snapshot, before, environment_mask):
    exact = torch.ones((), dtype=torch.bool, device=environment_mask.device)
    for name, reference in snapshot.items():
        value = bundle[name]
        shape = (environment_mask.shape[0],) + (1,) * (value.ndim - 1)
        expected = torch.where(environment_mask.reshape(shape), reference, before[name])
        exact &= (value == expected).all()
    return exact


def _episode_lengths(
    torch,
    learner_seeds,
    config,
    episode_indices,
    minimum_steps,
    maximum_steps,
    device,
):
    environment = torch.arange(
        config.environments_per_learner, dtype=torch.int64, device=device
    ).repeat(config.learners)
    seeds = torch.tensor(learner_seeds, dtype=torch.int64, device=device).repeat_interleave(
        config.environments_per_learner
    )
    span = maximum_steps - minimum_steps + 1
    mixed = (
        seeds + environment * 1_103_515_245 + episode_indices * 12_345
    ) & 0x7FFFFFFF
    return minimum_steps + mixed.remainder(span)


def _rollout(
    native,
    torch,
    policy,
    config,
    bundle,
    cameras,
    reset_snapshot,
    learner_seeds,
    minimum_episode_steps,
    maximum_episode_steps,
    asynchronous_resets,
):
    observations, raw_actions, log_probabilities = [], [], []
    rewards, values, terminated = [], [], []
    joint_coordinate = bundle["initial_q"]
    previous_distance = torch.abs(GOAL_X_M - bundle["state"][:, 4, 0]).reshape(
        config.learners, config.environments_per_learner
    )
    reset_distance = torch.abs(
        GOAL_X_M - reset_snapshot["state"][:, 4, 0]
    ).reshape_as(previous_distance)
    episode_steps = torch.zeros(
        config.total_worlds, dtype=torch.int64, device=bundle["state"].device
    )
    episode_indices = torch.zeros_like(episode_steps)
    episode_returns = torch.zeros_like(previous_distance)
    completed_return_sum = torch.zeros(
        config.learners, dtype=torch.float32, device=bundle["state"].device
    )
    completed_episode_count = torch.zeros(
        config.learners, dtype=torch.int64, device=bundle["state"].device
    )
    reset_count_by_world = torch.zeros_like(episode_steps)
    masked_restore_exact = torch.ones(
        (), dtype=torch.bool, device=bundle["state"].device
    )
    partial_reset_observed = torch.zeros_like(masked_restore_exact)
    contact_ever = torch.zeros(
        config.total_worlds, dtype=torch.bool, device=bundle["state"].device
    )
    final_depth = final_ids = None
    for time_index in range(config.horizon):
        observation, final_depth, final_ids = _camera_observation(
            native, torch, config, bundle, cameras, joint_coordinate
        )
        mean, value, log_standard_deviation = policy.forward(observation)
        raw_action = mean + torch.exp(log_standard_deviation) * torch.randn_like(mean)
        action = torch.tanh(raw_action)
        log_probability = _squashed_log_probability(
            torch, raw_action, mean, log_standard_deviation
        )
        result = _native_coupled_step(
            native, torch, bundle, joint_coordinate, action, config
        )
        joint_coordinate = result[1]
        distance = torch.abs(GOAL_X_M - bundle["state"][:, 4, 0]).reshape_as(previous_distance)
        contact = result[8].to(dtype=torch.bool).any(dim=1).reshape_as(distance)
        contact_ever |= contact.reshape(-1)
        reward = (
            20.0 * (previous_distance - distance)
            + 0.02 * contact.to(dtype=torch.float32)
            - 0.001 * action.square().sum(dim=-1)
        )
        episode_returns += reward
        episode_steps += 1
        if asynchronous_resets:
            lengths = _episode_lengths(
                torch,
                learner_seeds,
                config,
                episode_indices,
                minimum_episode_steps,
                maximum_episode_steps,
                bundle["state"].device,
            )
            end_flat = episode_steps >= lengths
            end = end_flat.reshape_as(contact)
        else:
            end = torch.full_like(
                contact, time_index + 1 == config.horizon, dtype=torch.bool
            )
            end_flat = end.reshape(-1)
        observations.append(observation)
        raw_actions.append(raw_action)
        log_probabilities.append(log_probability)
        rewards.append(reward)
        values.append(value)
        terminated.append(end)
        if asynchronous_resets:
            completed_return_sum += (episode_returns * end).sum(dim=1)
            completed_episode_count += end.sum(dim=1)
            before = _snapshot(bundle)
            _restore_masked(torch, bundle, reset_snapshot, end_flat)
            partial_reset_observed |= end_flat.any() & (~end_flat).any()
            masked_restore_exact &= _masked_restore_exact(
                torch, bundle, reset_snapshot, before, end_flat
            )
            joint_coordinate = torch.where(
                end_flat[:, None], bundle["initial_q"], joint_coordinate
            )
            previous_distance = torch.where(end, reset_distance, distance)
            episode_returns = torch.where(end, 0.0, episode_returns)
            episode_steps = torch.where(end_flat, 0, episode_steps)
            episode_indices += end_flat.to(dtype=episode_indices.dtype)
            reset_count_by_world += end_flat.to(dtype=reset_count_by_world.dtype)
        else:
            previous_distance = distance
    final_observation, _, _ = _camera_observation(
        native, torch, config, bundle, cameras, joint_coordinate
    )
    _, bootstrap, _ = policy.forward(final_observation)
    return {
        "observations": torch.stack(observations),
        "raw_actions": torch.stack(raw_actions),
        "old_log_probabilities": torch.stack(log_probabilities),
        "rewards": torch.stack(rewards),
        "values": torch.cat((torch.stack(values), bootstrap[None]), dim=0),
        "terminated": torch.stack(terminated),
        "contact_ever": contact_ever,
        "completed_return_sum": completed_return_sum,
        "completed_episode_count": completed_episode_count,
        "reset_count_by_world": reset_count_by_world,
        "partial_reset_observed": partial_reset_observed,
        "masked_restore_exact": masked_restore_exact,
        "final_depth": final_depth,
        "final_ids": final_ids,
    }


def _ppo_update(torch, policy, optimizer, config, rollout, ppo_epochs: int):
    advantages, returns = _gae_cuda(
        torch, rollout["rewards"], rollout["values"], rollout["terminated"],
        config.gamma, config.gae_lambda,
    )
    advantages = (advantages - advantages.mean(dim=(0, 2), keepdim=True)) / (
        advantages.std(dim=(0, 2), keepdim=True, unbiased=False) + 1.0e-6
    )
    observations = rollout["observations"].permute(1, 0, 2, 3).reshape(
        config.learners, -1, config.observation_dimensions
    )
    raw_actions = rollout["raw_actions"].permute(1, 0, 2, 3).reshape(
        config.learners, -1, config.action_dimensions
    )
    old_log_probability = rollout["old_log_probabilities"].permute(1, 0, 2).reshape(
        config.learners, -1
    )
    advantage = advantages.permute(1, 0, 2).reshape(config.learners, -1)
    target_return = returns.permute(1, 0, 2).reshape(config.learners, -1)
    final_loss = None
    for _ in range(ppo_epochs):
        mean, value, log_standard_deviation = policy.forward(observations)
        log_probability = _squashed_log_probability(
            torch, raw_actions, mean, log_standard_deviation
        )
        ratio = torch.exp(log_probability - old_log_probability)
        clipped = torch.clamp(
            ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
        )
        actor_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean(dim=1)
        value_loss = 0.5 * (value - target_return).square().mean(dim=1)
        entropy = (
            log_standard_deviation.squeeze(1)
            + 0.5 * math.log(2.0 * math.pi * math.e)
        ).sum(dim=1)
        learner_loss = actor_loss + 0.5 * value_loss - 0.001 * entropy
        final_loss = learner_loss.mean()
        optimizer.zero_grad(set_to_none=True)
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters, 1.0)
        optimizer.step()
    return advantages, returns, final_loss


def _parameter_delta_per_learner(torch, before, after):
    squared = None
    for initial, current in zip(before, after):
        row = (current.detach() - initial).reshape(initial.shape[0], -1).square().sum(dim=1)
        squared = row if squared is None else squared + row
    return torch.sqrt(squared)


def main() -> int:
    args = arguments()
    if args.updates < 1 or args.ppo_epochs < 1:
        raise ValueError("updates and PPO epochs must be positive")
    if args.base_seed < 0:
        raise ValueError("base seed must be non-negative")
    if (
        args.minimum_episode_steps < 1
        or args.maximum_episode_steps < args.minimum_episode_steps
        or args.maximum_episode_steps > 65_536
    ):
        raise ValueError("episode bounds must satisfy 1 <= minimum <= maximum <= 65536")
    config = ParallelTrainerConfig(
        learners=args.learners,
        environments_per_learner=args.environments_per_learner,
        horizon=args.horizon,
    )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("parallel trainer requires a visible CUDA device")
    torch.manual_seed(args.base_seed)
    torch.cuda.manual_seed_all(args.base_seed)
    learner_seeds = [learner_seed(args.base_seed, learner) for learner in range(config.learners)]
    native = load_extension()
    bundle = make_workload(config.total_worlds, torch.device("cuda"))
    reset_snapshot = _snapshot(bundle)
    cameras = _camera_fixture(torch, config, bundle)
    reward_probe = _fixed_action_reward_probe(
        native, torch, config, bundle, reset_snapshot
    )
    reward_probe_passed = bool(
        reward_probe["best_mean_goal_progress_m"] > 1.0e-4
        and reward_probe["progress_spread_m"] > 1.0e-4
    )
    if not reward_probe_passed:
        raise RuntimeError(
            "bounded controls do not produce a learnable reward difference: "
            + json.dumps(reward_probe, sort_keys=True)
        )
    policy = BatchedActorCritic(torch, config, learner_seeds)
    optimizer = torch.optim.Adam(policy.parameters, lr=3.0e-4)

    initial_observation, initial_depth, initial_ids = _camera_observation(
        native, torch, config, bundle, cameras, bundle["initial_q"]
    )
    replay_observation, _, _ = _camera_observation(
        native, torch, config, bundle, cameras, bundle["initial_q"]
    )
    camera_replay_exact = torch.equal(initial_observation, replay_observation)
    before = [parameter.detach().clone() for parameter in policy.parameters]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    rollout_events = []
    update_events = []
    returns_by_update = []
    completed_episodes_by_update = []
    reset_count_by_update = []
    masked_restore_exact_by_update = []
    partial_reset_observed_by_update = []
    final_rollout = final_loss = final_advantages = None
    for _ in range(args.updates):
        rollout_start = torch.cuda.Event(enable_timing=True)
        rollout_end = torch.cuda.Event(enable_timing=True)
        update_end = torch.cuda.Event(enable_timing=True)
        rollout_start.record()
        _restore(bundle, reset_snapshot)
        with torch.no_grad():
            final_rollout = _rollout(
                native,
                torch,
                policy,
                config,
                bundle,
                cameras,
                reset_snapshot,
                learner_seeds,
                args.minimum_episode_steps,
                args.maximum_episode_steps,
                args.asynchronous_resets,
            )
        rollout_end.record()
        final_advantages, _returns, final_loss = _ppo_update(
            torch, policy, optimizer, config, final_rollout, args.ppo_epochs
        )
        update_end.record()
        rollout_events.append((rollout_start, rollout_end))
        update_events.append((rollout_end, update_end))
        completed = final_rollout["completed_episode_count"]
        completed_return = final_rollout["completed_return_sum"] / completed.clamp_min(1)
        fallback_return = final_rollout["rewards"].sum(dim=0).mean(dim=1)
        returns_by_update.append(torch.where(completed > 0, completed_return, fallback_return))
        completed_episodes_by_update.append(completed)
        reset_count_by_update.append(final_rollout["reset_count_by_world"].sum())
        masked_restore_exact_by_update.append(final_rollout["masked_restore_exact"])
        partial_reset_observed_by_update.append(final_rollout["partial_reset_observed"])
    torch.cuda.synchronize()

    rollout_seconds = sum(
        start.elapsed_time(end) for start, end in rollout_events
    ) / 1000.0
    update_seconds = sum(
        start.elapsed_time(end) for start, end in update_events
    ) / 1000.0
    total_seconds = rollout_seconds + update_seconds
    deltas = _parameter_delta_per_learner(torch, before, policy.parameters)
    learner_returns = torch.stack(returns_by_update)
    completed_episodes = torch.stack(completed_episodes_by_update)
    reset_counts = torch.stack(reset_count_by_update)
    curve_summary = learning_curve_summary(learner_returns.cpu().tolist())
    if final_rollout is None or final_loss is None or final_advantages is None:
        raise RuntimeError("parallel trainer produced no rollout")
    hit = final_rollout["final_ids"] >= 0
    miss = ~hit
    state_changed = not torch.equal(bundle["state"], reset_snapshot["state"])
    gates = {
        "camera_replay_bit_exact": bool(camera_replay_exact),
        "nontrivial_initial_depth": bool((initial_depth > 0.0).any().item()),
        "nontrivial_hit_and_miss_population": bool(hit.any().item() and miss.any().item()),
        "finite_rollout_observations": bool(torch.isfinite(final_rollout["observations"]).all().item()),
        "finite_rewards_and_advantages": bool(
            torch.isfinite(final_rollout["rewards"]).all().item()
            and torch.isfinite(final_advantages).all().item()
        ),
        "finite_policy_loss": bool(torch.isfinite(final_loss).item()),
        "physics_state_changed": bool(state_changed),
        "every_learner_parameter_set_updated": bool(torch.all(deltas > 0.0).item()),
        "learner_returns_are_finite": bool(torch.isfinite(learner_returns).all().item()),
        "instance_ids_are_bounded": bool(
            torch.all((initial_ids >= -1) & (initial_ids < bundle["state"].shape[1])).item()
        ),
        "bounded_control_changes_reward": reward_probe_passed,
        "asynchronous_partial_reset_observed": bool(
            not args.asynchronous_resets or all(partial_reset_observed_by_update)
        ),
        "masked_reset_selected_and_unselected_exact": bool(
            not args.asynchronous_resets or all(masked_restore_exact_by_update)
        ),
        "completed_episodes_are_nonzero": bool(
            not args.asynchronous_resets or torch.all(completed_episodes > 0).item()
        ),
    }
    if not all(gates.values()):
        raise RuntimeError("parallel trainer gate failed: " + json.dumps(gates, sort_keys=True))

    world_steps = args.updates * config.horizon * config.total_worlds
    pixels = world_steps * config.rays_per_world
    payload = {
        "schema_version": ASYNC_CONTRACT_ID if args.asynchronous_resets else CONTRACT_ID,
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "configuration": {
            "learners": config.learners,
            "environments_per_learner": config.environments_per_learner,
            "total_worlds": config.total_worlds,
            "horizon": config.horizon,
            "updates": args.updates,
            "ppo_epochs": args.ppo_epochs,
            "cameras_per_world": config.cameras,
            "camera_width": config.camera_width,
            "camera_height": config.camera_height,
            "rays_per_world": config.rays_per_world,
            "observation_dimensions": config.observation_dimensions,
            "action_dimensions": config.action_dimensions,
            "base_seed": args.base_seed,
            "learner_seeds": learner_seeds,
            "asynchronous_resets": args.asynchronous_resets,
            "minimum_episode_steps": args.minimum_episode_steps,
            "maximum_episode_steps": args.maximum_episode_steps,
        },
        "timing": {
            "rollout_seconds": rollout_seconds,
            "ppo_update_seconds": update_seconds,
            "end_to_end_seconds": total_seconds,
            "world_steps": world_steps,
            "world_steps_per_second": world_steps / rollout_seconds,
            "depth_pixels": pixels,
            "depth_pixels_per_second": pixels / rollout_seconds,
            "policy_actions_per_second": world_steps / rollout_seconds,
            "learner_updates_per_second": (args.updates * config.learners) / total_seconds,
            "peak_torch_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "gates": gates,
        "policy_parameter_delta_per_learner": deltas.cpu().tolist(),
        "mean_episode_return_per_update_and_learner": learner_returns.cpu().tolist(),
        "completed_episodes_per_update_and_learner": completed_episodes.cpu().tolist(),
        "partial_resets_per_update": reset_counts.cpu().tolist(),
        "learning_curve": curve_summary,
        "fixed_action_reward_probe": reward_probe,
        "contacted_worlds_last_rollout": int(final_rollout["contact_ever"].sum().item()),
        "claims": {
            "gpu_resident_rollout_buffers": True,
            "parallel_independent_policy_parameters": True,
            "analytic_depth_and_instance_pixels": True,
            "reported_rgb_or_raster_pixels": False,
            "learning_curve_accepted": curve_summary["accepted"],
            "asynchronous_partial_episode_reset": args.asynchronous_resets,
            "notes": (
                "Rollout timing includes batched actor/value inference, Stage-7 rigid stepping, "
                "two calibrated camera ray packets, linear OBB queries, optical depth, rewards, and buffer writes."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "passed",
        "device": payload["device"],
        "world_steps_per_second": payload["timing"]["world_steps_per_second"],
        "depth_pixels_per_second": payload["timing"]["depth_pixels_per_second"],
        "peak_torch_cuda_memory_bytes": payload["timing"]["peak_torch_cuda_memory_bytes"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
