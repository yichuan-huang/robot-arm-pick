"""Custom reward functions for Franka picking environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def trajectory_tracking_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Compute trajectory tracking reward."""
    deviation = env.get_trajectory_deviation().squeeze(1)
    # Exponential reward for staying close to trajectory
    reward = torch.exp(-deviation / 0.05)  # Tune decay parameter
    return reward


def target_approach_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Compute target approach reward."""
    ee_pos = env.scene["robot"].data.body_pos_w[:, -1, :3]
    obj_pos = env.scene["object"].data.root_pos_w[:, :3]
    distance = torch.norm(ee_pos - obj_pos, dim=1)
    # Exponential reward for approaching target
    reward = torch.exp(-distance / 0.1)  # Tune decay parameter
    return reward


def success_bonus_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Compute success reward."""
    success_mask = env.check_success()
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[success_mask] = 1.0
    return reward


def time_penalty_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Compute time penalty reward."""
    return -0.01 * torch.ones(env.num_envs, device=env.device)


def joint_velocity_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize large joint velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return -torch.sum(torch.square(joint_vel), dim=1) * 0.001


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large action changes between steps."""
    if hasattr(env, "_last_actions"):
        action_diff = env.actions - env._last_actions
        penalty = -torch.sum(torch.square(action_diff), dim=1) * 0.01
        env._last_actions = env.actions.clone()
        return penalty
    else:
        env._last_actions = env.actions.clone()
        return torch.zeros(env.num_envs, device=env.device)
