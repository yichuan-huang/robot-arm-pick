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
