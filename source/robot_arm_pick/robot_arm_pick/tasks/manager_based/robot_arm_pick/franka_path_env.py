"""Franka path tracking environment - focused on trajectory following without grasping."""

import torch
import numpy as np
import math

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv


class OptimalTrajectoryGenerator:
    """Optimized trajectory generator focused on shortest time and path constraints."""

    def __init__(self, max_deviation: float = 0.05, smoothness_factor: float = 0.8):
        """
        Initialize trajectory generator.
        Args:
            max_deviation: Maximum allowed path deviation (meters)
            smoothness_factor: Smoothness factor [0,1], higher means smoother
        """
        self.max_deviation = max_deviation
        self.smoothness_factor = smoothness_factor

    def generate_optimal_trajectory(
        self, start_pos: torch.Tensor, end_pos: torch.Tensor, max_time: float
    ) -> dict:
        """
        Generate optimal trajectory: shortest time + path constraints.

        Args:
            start_pos: Starting position [batch_size, 3]
            end_pos: Target position [batch_size, 3]
            max_time: Maximum allowed time

        Returns:
            Trajectory parameter dictionary
        """
        batch_size = start_pos.shape[0]
        device = start_pos.device

        # Calculate direct distance
        direct_distance = torch.norm(end_pos - start_pos, dim=1)

        # Estimate optimal time based on distance and velocity limits
        max_velocity = 0.8  # m/s - Franka arm maximum end-effector velocity
        optimal_time = direct_distance / max_velocity

        # Limit within maximum time range
        trajectory_time = torch.clamp(optimal_time, min=0.5, max=max_time)

        # Generate control points using Bezier curves for smoothness
        control_points = self._generate_control_points(start_pos, end_pos)

        return {
            "start_pos": start_pos,
            "end_pos": end_pos,
            "control_points": control_points,
            "trajectory_time": trajectory_time,
            "direct_distance": direct_distance,
            "max_deviation": self.max_deviation,
        }

    def _generate_control_points(
        self, start_pos: torch.Tensor, end_pos: torch.Tensor
    ) -> torch.Tensor:
        """Generate Bezier curve control points ensuring smooth path with minimal deviation."""
        batch_size = start_pos.shape[0]
        device = start_pos.device

        # Intermediate control point: slightly lift above direct line to avoid collisions
        mid_point = (start_pos + end_pos) / 2

        # Adaptive lift height based on horizontal distance
        horizontal_dist = torch.norm(end_pos[:, :2] - start_pos[:, :2], dim=1)
        lift_height = torch.clamp(horizontal_dist * 0.1, min=0.02, max=0.08)
        mid_point[:, 2] += lift_height  # Vertical lift

        # Four-point Bezier curve control points
        control_points = torch.stack(
            [
                start_pos,  # P0: Start point
                start_pos + (mid_point - start_pos) * 0.33,  # P1: Start control point
                end_pos + (mid_point - end_pos) * 0.33,  # P2: End control point
                end_pos,  # P3: End point
            ],
            dim=1,
        )  # [batch_size, 4, 3]

        return control_points

    def get_reference_state(
        self, trajectory: dict, current_time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get reference state: position, velocity, acceleration.

        Args:
            trajectory: Trajectory parameters
            current_time: Current time [batch_size]

        Returns:
            reference_pos: Reference position [batch_size, 3]
            reference_vel: Reference velocity [batch_size, 3]
            reference_acc: Reference acceleration [batch_size, 3]
        """
        # Normalize time parameter t ∈ [0, 1]
        t = torch.clamp(current_time / trajectory["trajectory_time"], 0, 1)

        # Bezier curve parameters
        control_points = trajectory["control_points"]  # [batch_size, 4, 3]

        # Four-point Bezier curve
        t2 = t * t
        t3 = t2 * t
        one_minus_t = 1 - t
        one_minus_t2 = one_minus_t * one_minus_t
        one_minus_t3 = one_minus_t2 * one_minus_t

        # Position B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
        reference_pos = (
            one_minus_t3.unsqueeze(-1) * control_points[:, 0]
            + 3 * one_minus_t2.unsqueeze(-1) * t.unsqueeze(-1) * control_points[:, 1]
            + 3 * one_minus_t.unsqueeze(-1) * t2.unsqueeze(-1) * control_points[:, 2]
            + t3.unsqueeze(-1) * control_points[:, 3]
        )

        # Velocity B'(t) (first derivative)
        reference_vel = (
            3
            * one_minus_t2.unsqueeze(-1)
            * (control_points[:, 1] - control_points[:, 0])
            + 6
            * one_minus_t.unsqueeze(-1)
            * t.unsqueeze(-1)
            * (control_points[:, 2] - control_points[:, 1])
            + 3 * t2.unsqueeze(-1) * (control_points[:, 3] - control_points[:, 2])
        ) / trajectory["trajectory_time"].unsqueeze(-1)

        # Acceleration B''(t) (second derivative)
        reference_acc = (
            6
            * one_minus_t.unsqueeze(-1)
            * (control_points[:, 2] - 2 * control_points[:, 1] + control_points[:, 0])
            + 6
            * t.unsqueeze(-1)
            * (control_points[:, 3] - 2 * control_points[:, 2] + control_points[:, 1])
        ) / (trajectory["trajectory_time"].unsqueeze(-1) ** 2)

        return reference_pos, reference_vel, reference_acc


class FrankaPathEnv(ManagerBasedRLEnv):
    """Franka path tracking environment - focused on trajectory control without grasping."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        """Initialize environment."""

        # Path control parameters
        self.max_path_deviation = getattr(
            cfg, "max_path_deviation", 0.05
        )  # Max deviation 5cm
        self.target_tolerance = getattr(
            cfg, "target_tolerance", 0.03
        )  # Target tolerance 3cm
        self.max_episode_time = getattr(
            cfg, "max_episode_time", 8.0
        )  # Max task time 8s

        # Trajectory generator
        self.trajectory_generator = OptimalTrajectoryGenerator(
            max_deviation=self.max_path_deviation
        )

        # State variables
        self.current_trajectory = None
        self.start_positions = None
        self.target_positions = None
        self.episode_start_time = None

        # Performance statistics
        self.success_count = 0
        self.episode_count = 0
        self.total_deviation_sum = 0
        self.total_completion_time = 0

        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        """Setup scene."""
        super()._setup_scene()
        self.robot = self.scene["robot"]

    def reset_idx(self, env_ids: torch.Tensor | None = None):
        """Reset environment."""
        super().reset_idx(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Randomize target sphere position
        self._randomize_target_position(env_ids)

        # Reset statistics
        self.episode_count += len(env_ids)

        # Record start time
        if self.episode_start_time is None:
            self.episode_start_time = torch.zeros(self.num_envs, device=self.device)
        self.episode_start_time[env_ids] = 0.0

    def _generate_path_task(self, env_ids: torch.Tensor):
        """Generate path tracking task."""
        num_resets = len(env_ids)

        # Generate start positions (random positions within arm workspace)
        start_pos = torch.zeros(num_resets, 3, device=self.device)
        start_pos[:, 0] = 0.3 + 0.4 * torch.rand(
            num_resets, device=self.device
        )  # x: 0.3-0.7
        start_pos[:, 1] = -0.2 + 0.4 * torch.rand(
            num_resets, device=self.device
        )  # y: -0.2-0.2
        start_pos[:, 2] = 0.6 + 0.3 * torch.rand(
            num_resets, device=self.device
        )  # z: 0.6-0.9

        # Generate target positions on ground (ensure reasonable distance from start)
        target_pos = torch.zeros(num_resets, 3, device=self.device)
        target_pos[:, 0] = 0.4 + 0.3 * torch.rand(
            num_resets, device=self.device
        )  # x: 0.4-0.7
        target_pos[:, 1] = -0.3 + 0.6 * torch.rand(
            num_resets, device=self.device
        )  # y: -0.3-0.3
        target_pos[:, 2] = 0.05  # Fixed on ground (sphere radius)

        # Ensure minimum distance while keeping targets on ground
        min_distance = 0.15  # 15cm minimum distance
        distance = torch.norm(target_pos - start_pos, dim=1)
        too_close = distance < min_distance

        # Regenerate targets that are too close (only in x,y plane to keep on ground)
        if too_close.any():
            # Generate horizontal direction only
            direction_xy = torch.randn(too_close.sum(), 2, device=self.device)
            direction_xy = direction_xy / torch.norm(direction_xy, dim=1, keepdim=True)

            # Update only x,y coordinates, keep z fixed on ground
            target_pos[too_close, :2] = (
                start_pos[too_close, :2] + direction_xy * min_distance
            )
            # Ensure new positions are still within reachable bounds
            target_pos[too_close, 0] = torch.clamp(target_pos[too_close, 0], 0.4, 0.7)
            target_pos[too_close, 1] = torch.clamp(target_pos[too_close, 1], -0.3, 0.3)
            target_pos[too_close, 2] = 0.05  # Keep on ground

        # Update global arrays
        if self.start_positions is None:
            self.start_positions = torch.zeros(self.num_envs, 3, device=self.device)
            self.target_positions = torch.zeros(self.num_envs, 3, device=self.device)

        self.start_positions[env_ids] = start_pos
        self.target_positions[env_ids] = target_pos

        # Generate optimal trajectory
        self.current_trajectory = self.trajectory_generator.generate_optimal_trajectory(
            self.start_positions, self.target_positions, self.max_episode_time
        )

    def _randomize_target_position(self, env_ids: torch.Tensor):
        """Randomize target sphere position within robot's reachable workspace on ground."""
        num_resets = len(env_ids)

        # Define robot's reachable workspace bounds for ground-level objects
        # Franka robot base is at (0,0,0), these are conservative reachable bounds
        x_min, x_max = 0.4, 0.7  # Forward reach (closer range for ground objects)
        y_min, y_max = -0.3, 0.3  # Side reach (reduced for better reachability)
        sphere_radius = 0.05  # Target sphere radius (5cm)

        # Generate random positions within workspace
        target_positions = torch.zeros(num_resets, 3, device=self.device)
        target_positions[:, 0] = x_min + (x_max - x_min) * torch.rand(
            num_resets, device=self.device
        )
        target_positions[:, 1] = y_min + (y_max - y_min) * torch.rand(
            num_resets, device=self.device
        )
        # Fix z-coordinate to ensure sphere rests on ground
        target_positions[:, 2] = sphere_radius  # Sphere center at radius height

        # Set the target sphere positions
        target_sphere = self.scene["target_sphere"]
        target_sphere.write_root_pos_to_sim(target_positions, env_ids)

    def get_current_path_state(self) -> dict:
        """Get current path state."""
        if self.current_trajectory is None:
            return {
                "reference_pos": torch.zeros(self.num_envs, 3, device=self.device),
                "reference_vel": torch.zeros(self.num_envs, 3, device=self.device),
                "current_pos": torch.zeros(self.num_envs, 3, device=self.device),
                "path_deviation": torch.zeros(self.num_envs, device=self.device),
                "progress": torch.zeros(self.num_envs, device=self.device),
            }

        # Current time
        current_time = self.episode_length_buf.float() * self.step_dt

        # Get reference state
        ref_pos, ref_vel, ref_acc = self.trajectory_generator.get_reference_state(
            self.current_trajectory, current_time
        )

        # Current gripper center position
        fingertip1_pos = self.robot.data.body_pos_w[:, -2, :3]  # First fingertip
        fingertip2_pos = self.robot.data.body_pos_w[:, -1, :3]  # Second fingertip
        current_pos = (fingertip1_pos + fingertip2_pos) / 2.0  # Gripper center

        # Path deviation distance
        path_deviation = torch.norm(current_pos - ref_pos, dim=1)

        # Task progress
        progress = torch.clamp(
            current_time / self.current_trajectory["trajectory_time"], 0, 1
        )

        return {
            "reference_pos": ref_pos,
            "reference_vel": ref_vel,
            "current_pos": current_pos,
            "path_deviation": path_deviation,
            "progress": progress,
            "target_distance": torch.norm(current_pos - self.target_positions, dim=1),
        }

    def compute_path_tracking_reward(self) -> torch.Tensor:
        """Compute path tracking reward."""
        path_state = self.get_current_path_state()
        deviation = path_state["path_deviation"]

        # Exponential reward based on deviation distance
        tracking_reward = torch.exp(-deviation / 0.02)  # 2cm standard deviation

        # Violation penalty
        violation_penalty = torch.where(
            deviation > self.max_path_deviation,
            -torch.clamp(
                (deviation - self.max_path_deviation) / 0.02, 0, 2
            ),  # Max -2 penalty
            torch.zeros_like(deviation),
        )

        return tracking_reward + violation_penalty

    def compute_efficiency_reward(self) -> torch.Tensor:
        """Compute efficiency reward - encourage fast target reaching."""
        path_state = self.get_current_path_state()
        current_time = self.episode_length_buf.float() * self.step_dt

        # Time bonus based on remaining time
        time_bonus = torch.clamp(
            (self.current_trajectory["trajectory_time"] - current_time)
            / self.current_trajectory["trajectory_time"],
            0,
            1,
        )

        # Progress-based reward
        progress_reward = path_state["progress"] * 0.5

        return time_bonus * 0.3 + progress_reward

    def compute_success_reward(self) -> torch.Tensor:
        """Compute success reward."""
        path_state = self.get_current_path_state()
        target_distance = path_state["target_distance"]

        # Target reached reward
        success_mask = target_distance < self.target_tolerance
        success_reward = torch.where(
            success_mask,
            torch.ones_like(target_distance),
            torch.zeros_like(target_distance),
        )

        # Update statistics
        self.success_count += success_mask.sum().item()

        return success_reward * 10.0  # Large success reward

    def check_task_completion(self) -> torch.Tensor:
        """Check if task is completed."""
        path_state = self.get_current_path_state()
        return path_state["target_distance"] < self.target_tolerance

    def check_path_violation(self) -> torch.Tensor:
        """Check severe path violation."""
        path_state = self.get_current_path_state()
        return path_state["path_deviation"] > (
            self.max_path_deviation * 2
        )  # Severe violation threshold

    def check_timeout(self) -> torch.Tensor:
        """Check timeout."""
        current_time = self.episode_length_buf.float() * self.step_dt
        return current_time > self.max_episode_time

    def print_statistics(self):
        """Print performance statistics."""
        if self.episode_count > 0:
            success_rate = self.success_count / self.episode_count
            print(
                f"Path tracking statistics: Success rate={success_rate:.3f}, Total tasks={self.episode_count}"
            )

    # Legacy compatibility methods for existing code
    def get_reference_position(self) -> torch.Tensor:
        """Get current reference position (legacy compatibility)."""
        path_state = self.get_current_path_state()
        return path_state["reference_pos"]

    def get_trajectory_deviation(self) -> torch.Tensor:
        """Get current trajectory deviation (legacy compatibility)."""
        path_state = self.get_current_path_state()
        return path_state["path_deviation"].unsqueeze(1)

    def get_time_remaining(self) -> torch.Tensor:
        """Get remaining episode time (legacy compatibility)."""
        current_time = self.episode_length_buf.float() * self.step_dt
        remaining = (self.max_episode_time - current_time).unsqueeze(1)
        return remaining.to(self.device)
