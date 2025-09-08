"""Custom reward functions for Franka picking environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def trajectory_tracking_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for staying close to reference trajectory."""
    deviation = env.get_trajectory_deviation().squeeze(1)
    # More progressive reward system
    delta_max = 0.08  # Slightly more tolerant maximum deviation

    # Exponential reward that's less punitive for small deviations
    reward = torch.exp(-deviation / 0.04)  # Smoother exponential decay

    # Small penalty for large deviations instead of harsh cutoff
    large_deviation_penalty = torch.where(
        deviation > delta_max,
        -2.0 * (deviation - delta_max),  # Moderate penalty for violations
        torch.zeros_like(deviation),
    )

    return reward + large_deviation_penalty


def target_approach_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for approaching target efficiently with distance-based scaling."""
    ee_pos = env.scene["robot"].data.body_pos_w[:, -1, :3]
    obj_pos = env.scene["object"].data.root_pos_w[:, :3]
    distance = torch.norm(ee_pos - obj_pos, dim=1)

    # Multi-stage reward system
    # Stage 1: Far away (>0.3m) - basic approach reward
    # Stage 2: Medium distance (0.1-0.3m) - increased reward
    # Stage 3: Close (<0.1m) - high precision reward

    reward = torch.zeros_like(distance)

    # Far stage: linear approach reward
    far_mask = distance > 0.3
    reward[far_mask] = (1.0 - torch.clamp(distance[far_mask] / 1.0, 0, 1)) * 0.5

    # Medium stage: exponential approach reward
    medium_mask = (distance <= 0.3) & (distance > 0.1)
    reward[medium_mask] = 0.5 + torch.exp(-(distance[medium_mask] - 0.1) / 0.1) * 2.0

    # Close stage: high precision reward
    close_mask = distance <= 0.1
    reward[close_mask] = 2.5 + torch.exp(-distance[close_mask] / 0.03) * 5.0

    return reward


def success_bonus_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Large bonus for successful task completion with progress consideration."""
    success_mask = env.check_success()
    reward = torch.zeros(env.num_envs, device=env.device)

    # Base success bonus
    base_bonus = 200.0

    # Time efficiency bonus (completed faster = higher bonus)
    progress = getattr(
        env, "task_progress", env.episode_length_buf / env.max_episode_length
    )
    time_bonus = (1.0 - progress) * 100.0  # Up to 100 extra points for fast completion

    total_bonus = base_bonus + time_bonus
    reward[success_mask] = (
        total_bonus[success_mask] if hasattr(time_bonus, "__getitem__") else base_bonus
    )

    return reward


def time_efficiency_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward faster completion while maintaining trajectory with adaptive weight."""
    # Get current progress (0 to 1)
    progress = getattr(
        env, "task_progress", env.episode_length_buf / env.max_episode_length
    )

    # Adaptive time reward based on current trajectory performance
    deviation = env.get_trajectory_deviation().squeeze(1)
    trajectory_quality = torch.exp(-deviation / 0.05)  # 0 to 1 scale

    # Only reward speed if trajectory following is good
    speed_reward = progress * trajectory_quality * 1.0

    # Progressive exploration bonus early in training
    exploration_bonus = torch.where(
        progress < 0.3,  # Early in episode
        trajectory_quality * 0.5,  # Bonus for good trajectory following
        torch.zeros_like(progress),
    )

    return speed_reward + exploration_bonus


def grasp_precision_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for maintaining grasp precision near target."""
    ee_pos = env.scene["robot"].data.body_pos_w[:, -1, :3]
    target_pos = env.scene["object"].data.root_pos_w[:, :3]
    distance = torch.norm(ee_pos - target_pos, dim=1)

    epsilon = 0.02  # Precision tolerance
    reward = torch.where(
        distance <= epsilon,
        10.0,  # High reward for precise positioning
        torch.exp(-distance / 0.05),  # Exponential decay for approach
    )
    return reward


def joint_velocity_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalty for excessive joint velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    vel_penalty = torch.sum(torch.square(joint_vel), dim=1)
    return -vel_penalty * 0.01


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for jerky control actions."""
    # Get current actions - in IsaacLab, actions are stored in the action buffer
    if hasattr(env, "action_buf"):
        current_actions = env.action_buf
    else:
        # Fallback: try to get from action manager's processed actions
        current_actions = getattr(
            env, "_actions", torch.zeros(env.num_envs, 9, device=env.device)
        )

    if hasattr(env, "_last_actions"):
        action_diff = current_actions - env._last_actions
        penalty = torch.sum(torch.square(action_diff), dim=1)
        env._last_actions = current_actions.clone()
        return -penalty * 0.005
    else:
        # Initialize _last_actions with current actions
        env._last_actions = current_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)


def trajectory_violation_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Hard constraint penalty for trajectory violations."""
    deviation = env.get_trajectory_deviation().squeeze(1)
    delta_max = 0.08  # Slightly larger than tracking reward threshold

    violation_penalty = torch.where(
        deviation > delta_max,
        -50.0 * (deviation - delta_max),  # Severe penalty for violations
        torch.zeros_like(deviation),
    )
    return violation_penalty
