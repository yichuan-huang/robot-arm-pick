"""Franka constrained picking environment."""

import torch
import numpy as np
import os
import sys

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv


class TrajectoryGenerator:
    """Simple trajectory generator for Franka picking."""

    def __init__(self, trajectory_type: str = "parabolic"):
        """Initialize trajectory generator."""
        self.trajectory_type = trajectory_type

    def generate_trajectory(
        self, start_pos: torch.Tensor, end_pos: torch.Tensor, duration: float
    ) -> dict:
        """Generate parabolic trajectory."""
        # Create intermediate waypoint above the midpoint
        mid_pos = (start_pos + end_pos) / 2
        mid_pos[:, 2] += 0.12  # Lift 12cm above midpoint (适应新的桌面高度)

        return {
            "start_pos": start_pos,
            "mid_pos": mid_pos,
            "end_pos": end_pos,
            "duration": duration,
        }

    def get_reference_pose(
        self, trajectory: dict, time_step: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get reference position and velocity using quadratic Bezier curve."""
        # Normalize time to [0, 1]
        t_norm = torch.clamp(time_step / trajectory["duration"], 0, 1).unsqueeze(1)

        start_pos = trajectory["start_pos"]
        mid_pos = trajectory["mid_pos"]
        end_pos = trajectory["end_pos"]

        # Quadratic Bezier curve: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
        t_norm_sq = t_norm**2
        one_minus_t = 1 - t_norm
        one_minus_t_sq = one_minus_t**2

        position = (
            one_minus_t_sq * start_pos
            + 2 * one_minus_t * t_norm * mid_pos
            + t_norm_sq * end_pos
        )

        # Derivative for velocity: B'(t) = 2(1-t)(P₁-P₀) + 2t(P₂-P₁)
        velocity = (
            2 * one_minus_t * (mid_pos - start_pos) + 2 * t_norm * (end_pos - mid_pos)
        ) / trajectory["duration"]

        return position, velocity


class FrankaPickingEnv(ManagerBasedRLEnv):
    """Franka constrained picking environment."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment."""
        # Initialize trajectory data before calling super().__init__()
        self.trajectory_generator = TrajectoryGenerator()
        self.start_positions = None
        self.target_positions = None
        self.reference_trajectory = None

        # Initialize constraint parameters
        self.max_deviation = getattr(cfg, "max_trajectory_deviation", 0.08)
        self.target_tolerance = getattr(cfg, "target_tolerance", 0.05)
        self.severe_violation_threshold = getattr(
            cfg, "severe_violation_threshold", 0.15
        )

        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        """Setup the scene."""
        super()._setup_scene()

        # Get robot and object references
        self.robot = self.scene["robot"]
        self.object = self.scene["object"]

    def reset_idx(self, env_ids: torch.Tensor | None = None):
        """Reset environments."""
        super().reset_idx(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Generate new random trajectories
        self._generate_random_trajectories(env_ids)

    def _generate_random_trajectories(self, env_ids: torch.Tensor):
        """Generate random start and target positions with trajectory."""
        num_resets = len(env_ids)

        # Random start positions (机械臂工作空间)
        start_pos = torch.zeros(num_resets, 3, device=self.device)
        start_pos[:, 0] = 0.3 + 0.5 * torch.rand(
            num_resets, device=self.device
        )  # x: 0.3-0.8 (可到达桌面范围)
        start_pos[:, 1] = -0.25 + 0.5 * torch.rand(
            num_resets, device=self.device
        )  # y: -0.25-0.25 (桌面宽度范围内)
        start_pos[:, 2] = 0.7 + 0.3 * torch.rand(
            num_resets, device=self.device
        )  # z: 0.7-1.0 (桌面上方)

        # Random target positions (桌面上的物体位置)
        target_pos = torch.zeros(num_resets, 3, device=self.device)
        # 桌子中心在 x=0.5, 桌子尺寸 0.6x0.5, 所以范围是 x: 0.2-0.8, y: -0.25-0.25
        target_pos[:, 0] = 0.25 + 0.45 * torch.rand(
            num_resets, device=self.device
        )  # x: 0.25-0.70 (在桌面内留边距)
        target_pos[:, 1] = -0.15 + 0.3 * torch.rand(
            num_resets, device=self.device
        )  # y: -0.15-0.15 (在桌面内留边距)
        target_pos[:, 2] = (
            0.575  # 桌面高度 (桌子位置0.5 + 桌子厚度0.025 + 物体半径约0.05)
        )

        # Update full arrays
        if self.start_positions is None:
            self.start_positions = torch.zeros(self.num_envs, 3, device=self.device)
            self.target_positions = torch.zeros(self.num_envs, 3, device=self.device)

        self.start_positions[env_ids] = start_pos
        self.target_positions[env_ids] = target_pos

        # Generate reference trajectory
        self.reference_trajectory = self.trajectory_generator.generate_trajectory(
            self.start_positions, self.target_positions, self.max_episode_length
        )

        # Update object positions
        self.scene["object"].write_root_pose_to_sim(
            torch.cat(
                [
                    target_pos,
                    torch.tensor([[1, 0, 0, 0]], device=self.device).repeat(
                        num_resets, 1
                    ),
                ],
                dim=1,
            ),
            env_ids,
        )

    def get_reference_position(self) -> torch.Tensor:
        """Get current reference position."""
        if self.reference_trajectory is None:
            # Return zeros with appropriate shape and device
            try:
                return torch.zeros(self.num_envs, 3, device=self.device)
            except:
                # Fallback during initialization
                return torch.zeros(2, 3, device="cuda:0")

        current_time = self.episode_length_buf * self.step_dt
        ref_pos, _ = self.trajectory_generator.get_reference_pose(
            self.reference_trajectory, current_time
        )
        return ref_pos.to(self.device)

    def get_trajectory_deviation(self) -> torch.Tensor:
        """Get current trajectory deviation."""
        try:
            ee_pos = self.scene["robot"].data.body_pos_w[:, -1, :3]
            ref_pos = self.get_reference_position()
            deviation = torch.norm(ee_pos - ref_pos, dim=1, keepdim=True)
            return deviation.to(self.device)
        except:
            # Fallback during initialization
            return torch.zeros(2, 1, device="cuda:0")

    def get_time_remaining(self) -> torch.Tensor:
        """Get remaining episode time."""
        try:
            current_time = self.episode_length_buf * self.step_dt
            remaining = (self.max_episode_length - current_time).unsqueeze(1)
            return remaining.to(self.device)
        except:
            # Fallback during initialization
            return torch.zeros(2, 1, device="cuda:0")

    def compute_trajectory_reward(self) -> torch.Tensor:
        """Compute trajectory tracking reward."""
        deviation = self.get_trajectory_deviation().squeeze(1)
        # Exponential reward for staying close to trajectory
        reward = torch.exp(-deviation / 0.05)  # TODO: Tune decay parameter
        return reward

    def compute_approach_reward(self) -> torch.Tensor:
        """Compute target approach reward."""
        ee_pos = self.scene["robot"].data.body_pos_w[:, -1, :3]
        obj_pos = self.scene["object"].data.root_pos_w[:, :3]
        distance = torch.norm(ee_pos - obj_pos, dim=1)
        # Exponential reward for approaching target
        reward = torch.exp(-distance / 0.1)  # TODO: Tune decay parameter
        return reward

    def compute_success_reward(self) -> torch.Tensor:
        """Compute success reward."""
        success_mask = self.check_success()
        reward = torch.zeros(self.num_envs, device=self.device)
        reward[success_mask] = 1.0
        return reward

    def check_success(self) -> torch.Tensor:
        """Check if grasping is successful."""
        ee_pos = self.scene["robot"].data.body_pos_w[:, -1, :3]
        obj_pos = self.scene["object"].data.root_pos_w[:, :3]
        distance = torch.norm(ee_pos - obj_pos, dim=1)
        return distance < self.target_tolerance

    def check_constraint_violation(self) -> torch.Tensor:
        """Check for severe constraint violations."""
        deviation = self.get_trajectory_deviation().squeeze(1)
        return deviation > self.severe_violation_threshold
