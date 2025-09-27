"""Custom observation functions for Franka target reaching and path tracking environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint positions of the robot."""
    try:
        asset: Articulation = env.scene[asset_cfg.name]
        return asset.data.joint_pos[:, asset_cfg.joint_ids]
    except:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        joint_count = (
            7 if hasattr(asset_cfg, "joint_ids") and asset_cfg.joint_ids else 7
        )
        return torch.zeros(num_envs, joint_count, device=device)


def robot_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint velocities of the robot."""
    try:
        asset: Articulation = env.scene[asset_cfg.name]
        return asset.data.joint_vel[:, asset_cfg.joint_ids]
    except:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        joint_count = (
            7 if hasattr(asset_cfg, "joint_ids") and asset_cfg.joint_ids else 7
        )
        return torch.zeros(num_envs, joint_count, device=device)


def gripper_center_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Gripper center position of the robot (midpoint between fingertips)."""
    try:
        asset: Articulation = env.scene[asset_cfg.name]
        # Get gripper fingertip positions and calculate center
        fingertip1_pos = asset.data.body_pos_w[:, -2, :3]  # First fingertip
        fingertip2_pos = asset.data.body_pos_w[:, -1, :3]  # Second fingertip
        gripper_center_pos = (fingertip1_pos + fingertip2_pos) / 2.0
        return gripper_center_pos
    except:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, 3, device=device)


def ee_linear_velocity(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """End-effector linear velocity."""
    try:
        asset: Articulation = env.scene[asset_cfg.name]
        return asset.data.body_lin_vel_w[:, -1, :3]
    except:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, 3, device=device)


def target_sphere_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Position of the target sphere."""
    try:
        asset: RigidObject = env.scene[asset_cfg.name]
        return asset.data.root_pos_w[:, :3]
    except:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, 3, device=device)


def gripper_to_target_distance(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Distance from gripper center to target sphere."""
    try:
        robot: Articulation = env.scene[robot_cfg.name]
        target: RigidObject = env.scene[target_cfg.name]

        # Get gripper fingertip positions and calculate center
        fingertip1_pos = robot.data.body_pos_w[:, -2, :3]  # First fingertip
        fingertip2_pos = robot.data.body_pos_w[:, -1, :3]  # Second fingertip
        gripper_center_pos = (fingertip1_pos + fingertip2_pos) / 2.0

        target_pos = target.data.root_pos_w[:, :3]  # Target position

        distance = torch.norm(gripper_center_pos - target_pos, dim=1, keepdim=True)
        return distance
    except:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.ones(num_envs, 1, device=device)


def task_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Task progress based on distance to target."""
    try:
        robot: Articulation = env.scene["robot"]
        target: RigidObject = env.scene["target_sphere"]

        # Get gripper fingertip positions and calculate center
        fingertip1_pos = robot.data.body_pos_w[:, -2, :3]  # First fingertip
        fingertip2_pos = robot.data.body_pos_w[:, -1, :3]  # Second fingertip
        gripper_center_pos = (fingertip1_pos + fingertip2_pos) / 2.0

        target_pos = target.data.root_pos_w[:, :3]
        distance = torch.norm(gripper_center_pos - target_pos, dim=1)

        # Convert distance to progress (closer = higher progress)
        max_distance = 2.0  # Maximum expected distance in workspace
        progress = torch.clamp(1.0 - distance / max_distance, 0.0, 1.0)
        return progress.unsqueeze(1)
    except:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, 1, device=device)


def reference_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reference trajectory position for path tracking."""
    try:
        return env.get_reference_position()
    except:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, 3, device=device)


def trajectory_deviation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Current trajectory deviation for path tracking."""
    try:
        return env.get_trajectory_deviation()
    except:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, 1, device=device)


def target_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Target position for path tracking."""
    try:
        return env.target_positions
    except:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, 3, device=device)


def success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Success reward for reaching target."""
    try:
        return env.compute_success_reward()
    except:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, device=device)


def task_completion(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check if task is completed."""
    try:
        return env.check_task_completion()
    except:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, dtype=torch.bool, device=device)


def path_violation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check severe path violation."""
    try:
        return env.check_path_violation()
    except:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, dtype=torch.bool, device=device)


def time_remaining(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Time remaining in episode."""
    try:
        return env.get_time_remaining()
    except:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, 1, device=device)
