"""Custom observation functions for Franka picking environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint positions of the robot."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def robot_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint velocities of the robot."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def robot_ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector position of the robot."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_pos_w[:, -1, :3]  # End-effector is the last body


def object_position(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Position of the target object."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, :3]


def reference_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reference trajectory position."""
    return env.get_reference_position()


def trajectory_deviation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Current trajectory deviation."""
    return env.get_trajectory_deviation()


def time_remaining(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Time remaining in episode."""
    return env.get_time_remaining()


def robot_arm_joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Joint positions of the robot arm (first 7 joints)."""
    asset: Articulation = env.scene["robot"]
    return asset.data.joint_pos[:, :7]


def robot_arm_joint_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Joint velocities of the robot arm (first 7 joints)."""
    asset: Articulation = env.scene["robot"]
    return asset.data.joint_vel[:, :7]


def robot_gripper_joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Joint positions of the robot gripper (last 2 joints)."""
    asset: Articulation = env.scene["robot"]
    return asset.data.joint_pos[:, 7:9]


def robot_gripper_joint_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Joint velocities of the robot gripper (last 2 joints)."""
    asset: Articulation = env.scene["robot"]
    return asset.data.joint_vel[:, 7:9]


def task_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Task progress as a fraction of episode completion (0 to 1)."""
    progress = env.episode_length_buf.float() / env.max_episode_length
    return progress.unsqueeze(1)  # Add dimension for consistency


def relative_ee_to_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Relative position from end-effector to target object."""
    ee_pos = env.scene["robot"].data.body_pos_w[:, -1, :3]
    obj_pos = env.scene["object"].data.root_pos_w[:, :3]
    return obj_pos - ee_pos


def ee_to_target_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Distance from end-effector to target object."""
    ee_pos = env.scene["robot"].data.body_pos_w[:, -1, :3]
    obj_pos = env.scene["object"].data.root_pos_w[:, :3]
    distance = torch.norm(ee_pos - obj_pos, dim=1)
    return distance.unsqueeze(1)


def ee_linear_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End-effector linear velocity."""
    asset: Articulation = env.scene["robot"]
    return asset.data.body_lin_vel_w[:, -1, :3]


def relative_ref_to_ee(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Relative position from reference trajectory to end-effector."""
    ref_pos = env.get_reference_position()
    ee_pos = env.scene["robot"].data.body_pos_w[:, -1, :3]
    return ref_pos - ee_pos
