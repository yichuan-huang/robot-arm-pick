"""Custom termination functions for Franka target reaching and path tracking environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_reached(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Check if the robot gripper has reached the target using center position."""
    try:
        robot: Articulation = env.scene[robot_cfg.name]
        target: RigidObject = env.scene[target_cfg.name]

        # Get gripper fingertip positions and calculate center
        fingertip1_pos = robot.data.body_pos_w[:, -2, :3]  # First fingertip
        fingertip2_pos = robot.data.body_pos_w[:, -1, :3]  # Second fingertip
        gripper_center_pos = (fingertip1_pos + fingertip2_pos) / 2.0

        target_pos = target.data.root_pos_w[:, :3]

        distance = torch.norm(gripper_center_pos - target_pos, dim=1)
        return distance <= threshold
    except:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, dtype=torch.bool, device=device)


def robot_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Check if the robot gripper is out of workspace bounds."""
    try:
        robot: Articulation = env.scene[asset_cfg.name]

        # Get gripper fingertip positions and calculate center
        fingertip1_pos = robot.data.body_pos_w[:, -2, :3]  # First fingertip
        fingertip2_pos = robot.data.body_pos_w[:, -1, :3]  # Second fingertip
        gripper_center_pos = (fingertip1_pos + fingertip2_pos) / 2.0

        # Define workspace bounds (adjust based on robot's reachable space)
        x_min, x_max = 0.2, 1.0  # Forward/backward limits
        y_min, y_max = -0.6, 0.6  # Left/right limits
        z_min, z_max = 0.1, 1.2  # Up/down limits

        out_of_bounds = (
            (gripper_center_pos[:, 0] < x_min)
            | (gripper_center_pos[:, 0] > x_max)
            | (gripper_center_pos[:, 1] < y_min)
            | (gripper_center_pos[:, 1] > y_max)
            | (gripper_center_pos[:, 2] < z_min)
            | (gripper_center_pos[:, 2] > z_max)
        )

        return out_of_bounds
    except:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, dtype=torch.bool, device=device)


def success_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check if the task is successfully completed."""
    try:
        return env.check_task_completion()
    except:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def task_completion(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check if path tracking task is completed."""
    try:
        return env.check_task_completion()
    except:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def path_violation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check severe path violation."""
    try:
        return env.check_path_violation()
    except:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def constraint_violation_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check if constraint violation threshold is exceeded."""
    try:
        return env.check_constraint_violation()
    except:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def severe_trajectory_violation_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check if severe trajectory violation threshold is exceeded."""
    try:
        return env.check_path_violation()
    except:
        deviation = env.get_trajectory_deviation().squeeze(1)
        severe_threshold = getattr(env.cfg, "severe_violation_threshold", 0.08)
        return deviation > severe_threshold
