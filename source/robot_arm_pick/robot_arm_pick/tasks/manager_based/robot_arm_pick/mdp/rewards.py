"""Custom reward functions for Franka target reaching and path tracking environments.

Design goals:
- Stable and bounded magnitudes per term (typically in [0, 1] for rewards, [-1, 0] for penalties).
- Avoid very large sparse bonuses vs tiny dense shaping.
- Add optional debug hooks so users can inspect per-term values during training.

Debug hooks:
- If ``env.reward_debug_print_interval`` (int > 0) is set, the mean of each term is printed every N steps.
- Latest per-term tensors are stored in ``env._rew_debug`` dict (non-fatal if unavailable).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# Internal helpers (no-op if env doesn't expose fields)
# -----------------------------------------------------------------------------


def _get_cfg_attr(env: "ManagerBasedRLEnv", name: str, default):
    try:
        return getattr(env.cfg, name, default)
    except Exception:
        return default


def _record_debug(env: "ManagerBasedRLEnv", name: str, values: torch.Tensor):
    """Record per-term tensors and optionally print means at intervals.

    This is intentionally best-effort and won't raise if env lacks fields.
    """
    try:
        # Keep latest values for external inspection
        if not hasattr(env, "_rew_debug"):
            env._rew_debug = {}
        env._rew_debug[name] = values.detach()

        # Optional periodic printing
        interval = int(
            getattr(
                env,
                "reward_debug_print_interval",
                _get_cfg_attr(env, "reward_debug_print_interval", 0),
            )
            or 0
        )
        if interval > 0:
            # Use the maximum step across sub-envs for a single trigger
            step = int(
                getattr(
                    env, "episode_length_buf", torch.tensor([0], device=values.device)
                )
                .max()
                .item()
            )
            if step % interval == 0:
                mean_v = float(values.mean().item())
                print(f"[reward-debug][step={step}] {name}: mean={mean_v:.4f}")
    except Exception:
        # Never fail due to debugging
        pass


# -----------------------------------------------------------------------------
# Reward Functions
# -----------------------------------------------------------------------------


def reach_target_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Reward for reaching close to the target using gripper center position."""
    try:
        robot: Articulation = env.scene[robot_cfg.name]
        target: RigidObject = env.scene[target_cfg.name]

        # Get gripper fingertip positions and calculate center
        fingertip1_pos = robot.data.body_pos_w[:, -2, :3]  # First fingertip
        fingertip2_pos = robot.data.body_pos_w[:, -1, :3]  # Second fingertip
        gripper_center_pos = (fingertip1_pos + fingertip2_pos) / 2.0

        target_pos = target.data.root_pos_w[:, :3]  # Target position

        distance = torch.norm(gripper_center_pos - target_pos, dim=1)

        # Give high reward when within threshold
        reward = torch.where(distance <= threshold, 1.0, 0.0)
        return reward
    except:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, device=device)


def approach_target_reward(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Enhanced reward for approaching the target with stronger gradients using gripper center."""
    try:
        robot: Articulation = env.scene[robot_cfg.name]
        target: RigidObject = env.scene[target_cfg.name]

        # Get gripper fingertip positions and calculate center
        fingertip1_pos = robot.data.body_pos_w[:, -2, :3]  # First fingertip
        fingertip2_pos = robot.data.body_pos_w[:, -1, :3]  # Second fingertip
        gripper_center_pos = (fingertip1_pos + fingertip2_pos) / 2.0

        target_pos = target.data.root_pos_w[:, :3]  # Target position

        distance = torch.norm(gripper_center_pos - target_pos, dim=1)

        # Multi-scale reward with stronger gradients
        max_distance = 1.5  # Reduced for better scaling

        # Close-range exponential reward (strong signal when very close)
        close_reward = torch.exp(-distance / 0.03) * 2.0  # Stronger signal when < 3cm

        # Medium-range linear reward (good gradient for approach)
        medium_reward = torch.clamp(
            2.0 - distance / 0.2, min=0.0, max=2.0
        )  # Linear decay over 20cm

        # Long-range reward for initial approach
        long_reward = torch.clamp(1.0 - distance / max_distance, min=0.0, max=1.0)

        # Combine with distance-based weighting
        if distance.numel() > 0:
            close_weight = torch.exp(-((distance - 0.05) ** 2) / 0.01)  # Peak at 5cm
            medium_weight = torch.exp(-((distance - 0.15) ** 2) / 0.05)  # Peak at 15cm
            long_weight = torch.where(distance > 0.3, 1.0, 0.0)  # For far distances

            reward = (
                close_weight * close_reward
                + medium_weight * medium_reward
                + long_weight * long_reward
            )
        else:
            reward = long_reward

        _record_debug(env, "approach_target_enhanced", reward)
        return reward
    except:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, device=device)


def touch_target_success(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Success bonus for touching the target using gripper center position."""
    try:
        robot: Articulation = env.scene[robot_cfg.name]
        target: RigidObject = env.scene[target_cfg.name]

        # Get gripper fingertip positions and calculate center
        fingertip1_pos = robot.data.body_pos_w[:, -2, :3]  # First fingertip
        fingertip2_pos = robot.data.body_pos_w[:, -1, :3]  # Second fingertip
        gripper_center_pos = (fingertip1_pos + fingertip2_pos) / 2.0

        target_pos = target.data.root_pos_w[:, :3]  # Target position

        distance = torch.norm(gripper_center_pos - target_pos, dim=1)

        # Large bonus for successful touch
        reward = torch.where(distance <= threshold, 1.0, 0.0)
        return reward
    except:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, device=device)


def trajectory_tracking_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Path tracking reward for trajectory following.

    Returns reward for staying close to the reference trajectory path.
    """
    try:
        return env.compute_path_tracking_reward()
    except:
        # Fallback for compatibility
        deviation = env.get_trajectory_deviation().squeeze(1)
        delta_max = float(_get_cfg_attr(env, "max_trajectory_deviation", 0.10))

        # Normalize deviation by tolerance and shape into [0, 1]
        dev_norm = torch.clamp(deviation / max(delta_max, 1e-6), min=0.0, max=5.0)
        reward_pos = torch.exp(-dev_norm)  # 1 at perfect tracking, decays smoothly

        # Gentle penalty when outside tolerance, bounded in [-0.5, 0]
        overflow = torch.relu(dev_norm - 1.0)  # portion exceeding tolerance
        penalty = -torch.clamp(overflow, max=1.0) * 0.5

        out = reward_pos + penalty
        _record_debug(env, "track_trajectory", out)
        return out


def target_approach_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Efficiency reward for path tracking - encourage fast and efficient movement."""
    try:
        return env.compute_efficiency_reward()
    except:
        # Fallback for compatibility - enhanced bounded reward for being close to target
        return torch.zeros(env.num_envs, device=env.device)


def success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Success reward for reaching the target position."""
    try:
        return env.compute_success_reward()
    except:
        return torch.zeros(env.num_envs, device=env.device)


def enhanced_target_approach_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Enhanced reward with stronger gradients and higher magnitude for better learning.

    Returns rewards in range [0, 5] with strong gradients throughout the approach.
    Uses gripper center position for accurate grasp distance calculation.
    """
    try:
        robot: Articulation = env.scene["robot"]
        target: RigidObject = env.scene["target_sphere"]

        # Get gripper fingertip positions and calculate center
        fingertip1_pos = robot.data.body_pos_w[:, -2, :3]  # First fingertip
        fingertip2_pos = robot.data.body_pos_w[:, -1, :3]  # Second fingertip
        gripper_center_pos = (fingertip1_pos + fingertip2_pos) / 2.0

        obj_pos = target.data.root_pos_w[:, :3]  # Target position
        distance = torch.norm(gripper_center_pos - obj_pos, dim=1)

        # Multi-tier reward system with stronger signals

        # Tier 1: Very close (0-5cm) - Exponential reward with high magnitude
        very_close_reward = torch.where(
            distance <= 0.05,
            5.0 * torch.exp(-distance / 0.02),  # Max reward = 5.0
            torch.zeros_like(distance),
        )

        # Tier 2: Close (5-15cm) - Strong linear reward
        close_reward = torch.where(
            (distance > 0.05) & (distance <= 0.15),
            4.0 - 20.0 * (distance - 0.05),  # From 4.0 to 2.0
            torch.zeros_like(distance),
        )

        # Tier 3: Medium (15-30cm) - Moderate linear reward
        medium_reward = torch.where(
            (distance > 0.15) & (distance <= 0.30),
            2.0 - 4.0 * (distance - 0.15),  # From 2.0 to 1.4
            torch.zeros_like(distance),
        )

        # Tier 4: Far (30cm+) - Base approach reward
        far_reward = torch.where(
            distance > 0.30,
            torch.clamp(1.4 - distance / 0.5, min=0.1, max=1.4),
            torch.zeros_like(distance),
        )

        # Combine all tiers
        reward = very_close_reward + close_reward + medium_reward + far_reward

        # Add velocity bonus for active approach using gripper center velocity
        if hasattr(robot.data, "body_lin_vel_w"):
            # Use gripper center velocity (average of fingertip velocities)
            fingertip1_vel = robot.data.body_lin_vel_w[:, -2, :3]
            fingertip2_vel = robot.data.body_lin_vel_w[:, -1, :3]
            gripper_center_vel = (fingertip1_vel + fingertip2_vel) / 2.0

            approach_direction = (obj_pos - gripper_center_pos) / (
                distance.unsqueeze(1) + 1e-6
            )
            velocity_alignment = torch.sum(
                gripper_center_vel * approach_direction, dim=1
            )
            velocity_bonus = torch.clamp(velocity_alignment * 0.5, min=0.0, max=0.5)
            reward = reward + velocity_bonus

        _record_debug(env, "enhanced_approach_target", reward)
        return reward

    except Exception:
        # Fallback in case of any issues
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, device=device)


def success_bonus_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Enhanced success bonus with higher magnitude for stronger learning signal.

    Base bonus ~50 with up to +25 extra for faster completion. Provides much stronger
    positive reinforcement for successful task completion.
    """
    try:
        # Check if target is reached (same logic as termination condition)
        robot: Articulation = env.scene["robot"]
        target: RigidObject = env.scene["target_sphere"]

        # Get gripper fingertip positions and calculate center
        fingertip1_pos = robot.data.body_pos_w[:, -2, :3]  # First fingertip
        fingertip2_pos = robot.data.body_pos_w[:, -1, :3]  # Second fingertip
        gripper_center_pos = (fingertip1_pos + fingertip2_pos) / 2.0

        target_pos = target.data.root_pos_w[:, :3]

        distance = torch.norm(gripper_center_pos - target_pos, dim=1)
        success_mask = distance <= 0.05  # Same threshold as termination

        reward = torch.zeros(env.num_envs, device=env.device)

        base_bonus = float(
            _get_cfg_attr(env, "success_base_bonus", 100.0)
        )  # MASSIVE bonus to overcome all penalties
        extra_bonus = float(
            _get_cfg_attr(env, "success_time_bonus", 50.0)
        )  # HUGE time bonus

        # Time efficiency (0..1); prefer Tensor on device
        progress = getattr(env, "task_progress", None)
        if progress is None:
            progress = env.episode_length_buf.float() / max(
                float(env.max_episode_length), 1.0
            )
        progress = torch.as_tensor(progress, device=env.device, dtype=torch.float32)
        progress = torch.clamp(progress, 0.0, 1.0).view(-1)

        # Enhanced time bonus with exponential scaling for very fast completion
        time_multiplier = torch.exp(-progress * 2.0)  # Exponential bonus for speed
        total_bonus = base_bonus + extra_bonus * time_multiplier

        # Apply only to successful envs
        reward[success_mask] = total_bonus[success_mask]

        # Debug print success count and average bonus
        if success_mask.sum().item() > 0:
            try:
                step = int(getattr(env, "total_steps", 0))
                if step % 500 == 0:  # More frequent reporting
                    avg_bonus = reward[success_mask].mean().item()
                    print(
                        f"[Success Debug] Step {step}: {success_mask.sum().item()}/{env.num_envs} envs succeeded, avg_bonus: {avg_bonus:.2f}"
                    )
            except Exception:
                pass

        _record_debug(env, "success_bonus_enhanced", reward)
        return reward
    except Exception:
        # Fallback in case of any issues
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, device=device)


def time_efficiency_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Enhanced time efficiency reward with stronger magnitude and better shaping.

    Returns values in [0, 2.0]. Provides stronger positive signal throughout the episode
    with emphasis on both speed and progress towards the target.
    """
    try:
        robot: Articulation = env.scene["robot"]
        target: RigidObject = env.scene["target_sphere"]

        progress = getattr(env, "task_progress", None)
        if progress is None:
            progress = env.episode_length_buf.float() / max(
                float(env.max_episode_length), 1.0
            )
        progress = torch.as_tensor(progress, device=env.device, dtype=torch.float32)
        progress = torch.clamp(progress, 0.0, 1.0).view(-1)

        # Distance-based progress for better reward signal using gripper center
        fingertip1_pos = robot.data.body_pos_w[:, -2, :3]  # First fingertip
        fingertip2_pos = robot.data.body_pos_w[:, -1, :3]  # Second fingertip
        gripper_center_pos = (fingertip1_pos + fingertip2_pos) / 2.0

        obj_pos = target.data.root_pos_w[:, :3]
        distance = torch.norm(gripper_center_pos - obj_pos, dim=1)
        distance_progress = torch.clamp(
            1.0 - distance / 0.4, min=0.0, max=1.0
        )  # Reduced max distance

        # Enhanced combined progress signal
        combined_progress = 0.7 * distance_progress + 0.3 * (1.0 - progress)

        # Multi-component reward structure with higher magnitude
        base_reward = combined_progress * 1.0  # Increased from 0.3

        # Speed bonus - reward for making progress quickly
        speed_bonus = torch.where(
            progress < 0.6,
            combined_progress * (1.0 - progress) * 0.8,  # Higher bonus early on
            torch.zeros_like(progress),
        )

        # Distance improvement reward - reward for getting closer
        distance_improvement = torch.where(
            distance < 0.2,  # When within 20cm
            (0.2 - distance) * 3.0,  # Strong reward for being close
            torch.zeros_like(distance),
        )

        out = base_reward + speed_bonus + distance_improvement
        out = torch.clamp(out, min=0.0, max=2.0)  # Bound the reward

        _record_debug(env, "time_efficiency_enhanced", out)
        return out
    except Exception:
        # Fallback in case of any issues
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, device=device)


def grasp_precision_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bounded reward in [0, 1] for precise gripper placement at target."""
    try:
        robot: Articulation = env.scene["robot"]
        target: RigidObject = env.scene["target_sphere"]

        # Get gripper fingertip positions and calculate center for precise grasping
        fingertip1_pos = robot.data.body_pos_w[:, -2, :3]  # First fingertip
        fingertip2_pos = robot.data.body_pos_w[:, -1, :3]  # Second fingertip
        gripper_center_pos = (fingertip1_pos + fingertip2_pos) / 2.0

        target_pos = target.data.root_pos_w[:, :3]
        distance = torch.norm(gripper_center_pos - target_pos, dim=1)

        # Use curriculum-based tolerance if available, otherwise config default
        epsilon = getattr(
            env, "target_tolerance", _get_cfg_attr(env, "target_tolerance", 0.025)
        )
        reward = torch.where(
            distance <= epsilon, torch.ones_like(distance), torch.exp(-distance / 0.05)
        )
        _record_debug(env, "grasp_precision", reward)
        return reward
    except Exception:
        # Fallback in case of any issues
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, device=device)


def joint_velocity_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Bounded penalty in [-1, 0] for excessive joint velocities.

    Uses tanh on the mean absolute velocity to avoid exploding magnitudes.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    # Mean abs velocity -> [0, +inf), then squashed to [0,1)
    mean_abs = torch.mean(torch.abs(joint_vel), dim=1)
    penalty = -torch.tanh(mean_abs * 2.0)
    _record_debug(env, "joint_velocity_penalty", penalty)
    return penalty


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bounded penalty in [-1, 0] for jerky control actions (delta between steps)."""
    # Get current actions - in IsaacLab, actions are stored in the action buffer
    if hasattr(env, "action_buf") and env.action_buf is not None:
        current_actions = env.action_buf
    else:
        # Fallback: try to get from action manager's processed actions
        current_actions = getattr(
            env, "_actions", torch.zeros(env.num_envs, 9, device=env.device)
        )

    if hasattr(env, "_last_actions") and env._last_actions is not None:
        action_diff = current_actions - env._last_actions
        # Mean squared diff per env; squash to [0,1)
        msd = torch.mean(action_diff * action_diff, dim=1)
        penalty = -torch.tanh(msd * 5.0)
    else:
        penalty = torch.zeros(env.num_envs, device=env.device)

    # Update last actions for next step
    try:
        env._last_actions = current_actions.clone()
    except Exception:
        pass

    _record_debug(env, "action_smoothness", penalty)
    return penalty


def trajectory_violation_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bounded penalty in [-1, 0] for exceeding deviation tolerance."""
    deviation = env.get_trajectory_deviation().squeeze(1)
    delta_max = float(_get_cfg_attr(env, "max_trajectory_deviation", 0.10))
    dev_norm = torch.clamp(deviation / max(delta_max, 1e-6), min=0.0, max=5.0)
    overflow = torch.relu(dev_norm - 1.0)
    penalty = -torch.clamp(overflow, max=1.0)
    _record_debug(env, "trajectory_violation", penalty)
    return penalty


def progressive_distance_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Progressive distance reward that prevents reward plateaus and encourages continuous improvement.

    This function addresses the specific issue where training gets stuck around -0.00023 reward
    by providing stronger gradients and avoiding reward saturation.
    Uses gripper center position for accurate grasp distance calculation.
    """
    try:
        robot: Articulation = env.scene["robot"]
        target: RigidObject = env.scene["target_sphere"]

        # Get gripper fingertip positions and calculate center
        fingertip1_pos = robot.data.body_pos_w[:, -2, :3]  # First fingertip
        fingertip2_pos = robot.data.body_pos_w[:, -1, :3]  # Second fingertip
        gripper_center_pos = (fingertip1_pos + fingertip2_pos) / 2.0

        obj_pos = target.data.root_pos_w[:, :3]
        distance = torch.norm(gripper_center_pos - obj_pos, dim=1)

        # BOOSTED progressive scaling to overcome heavy penalties
        # Increased magnitudes to fight against -12.5 baseline penalty

        # Scale 1: Macro approach (1m to 30cm) - BOOSTED Linear scaling
        macro_reward = torch.where(
            distance > 0.30,
            8.0 * (1.0 - torch.clamp(distance / 1.0, min=0.0, max=1.0)),  # 4x increase
            torch.zeros_like(distance),
        )

        # Scale 2: Micro approach (30cm to 10cm) - BOOSTED Quadratic scaling
        micro_reward = torch.where(
            (distance > 0.10) & (distance <= 0.30),
            12.0 * (0.30 - distance) / 0.20,  # 4x increase
            torch.zeros_like(distance),
        )

        # Scale 3: Precision approach (10cm to 3cm) - BOOSTED Exponential scaling
        precision_reward = torch.where(
            (distance > 0.03) & (distance <= 0.10),
            16.0
            * torch.exp(
                -(distance - 0.03) / 0.02
            ),  # 4x increase - strong signal for precision
            torch.zeros_like(distance),
        )

        # Scale 4: Final approach (< 3cm) - MASSIVE reward for success
        final_reward = torch.where(
            distance <= 0.03,
            32.0 * torch.exp(-distance / 0.01),  # 4x increase - huge reward for success
            torch.zeros_like(distance),
        )

        # Combine all scales
        total_reward = macro_reward + micro_reward + precision_reward + final_reward

        # NO baseline addition - let it be zero when far
        # This ensures negative total reward when far from target

        # Allow higher maximum to overcome penalties
        total_reward = torch.clamp(total_reward, min=0.0, max=50.0)

        _record_debug(env, "progressive_distance", total_reward)
        return total_reward

    except Exception as e:
        # Robust fallback
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.full((num_envs,), 0.1, device=device)  # Small baseline reward


def adaptive_learning_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Adaptive reward that scales based on training progress to maintain learning momentum.

    This addresses training plateaus by adapting the reward scale based on recent performance.
    Uses gripper center position for accurate grasp distance calculation.
    """
    try:
        robot: Articulation = env.scene["robot"]
        target: RigidObject = env.scene["target_sphere"]

        # Get gripper fingertip positions and calculate center
        fingertip1_pos = robot.data.body_pos_w[:, -2, :3]  # First fingertip
        fingertip2_pos = robot.data.body_pos_w[:, -1, :3]  # Second fingertip
        gripper_center_pos = (fingertip1_pos + fingertip2_pos) / 2.0

        obj_pos = target.data.root_pos_w[:, :3]
        distance = torch.norm(gripper_center_pos - obj_pos, dim=1)

        # Get current training step for adaptive scaling
        current_step = getattr(env, "common_step_counter", 0)

        # Adaptive scaling factor - increases over time to combat diminishing returns
        if current_step < 50000:
            scale_factor = 1.0
        elif current_step < 200000:
            scale_factor = 2.0  # Double the reward after initial learning
        else:
            scale_factor = 3.0  # Triple for later stage fine-tuning

        # Base reward with strong gradient
        base_reward = torch.where(
            distance < 0.5,
            scale_factor * (0.5 - distance) / 0.5,  # Linear from 0.5m to 0m
            torch.zeros_like(distance),
        )

        # Success probability bonus - reward for getting closer to success
        success_probability = torch.exp(-distance / 0.05)  # High when very close
        prob_bonus = success_probability * scale_factor * 0.5

        total_reward = base_reward + prob_bonus + 0.05  # Small baseline

        _record_debug(env, "adaptive_learning", total_reward)
        return total_reward

    except Exception as e:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.full((num_envs,), 0.05, device=device)


def exploration_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Exploration bonus to break out of local optima and encourage diverse behaviors."""
    try:
        robot: Articulation = env.scene["robot"]

        # Get current joint positions and velocities
        joint_pos = robot.data.joint_pos[:, :7]  # First 7 joints (arm only)
        joint_vel = robot.data.joint_vel[:, :7]

        # Encourage exploration of joint space diversity
        joint_diversity = torch.std(joint_pos, dim=1)  # Higher std = more diverse poses
        velocity_diversity = torch.std(joint_vel, dim=1)  # Movement diversity

        # Combined exploration metric
        exploration_metric = joint_diversity + 0.5 * velocity_diversity

        # Scale to reasonable range [0, 1]
        exploration_reward = torch.tanh(exploration_metric / 2.0)

        # Add step-based decay to reduce exploration over time
        current_step = getattr(env, "common_step_counter", 0)
        decay_factor = max(0.1, 1.0 - current_step / 5000000)  # Decay over 5M steps
        exploration_reward = exploration_reward * decay_factor

        _record_debug(env, "exploration", exploration_reward)
        return exploration_reward

    except Exception:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, device=device)


def curriculum_learning_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Curriculum learning reward that adapts difficulty based on training progress."""
    try:
        robot: Articulation = env.scene["robot"]
        target: RigidObject = env.scene["target_sphere"]

        # Get gripper fingertip positions and calculate center
        fingertip1_pos = robot.data.body_pos_w[:, -2, :3]  # First fingertip
        fingertip2_pos = robot.data.body_pos_w[:, -1, :3]  # Second fingertip
        gripper_center_pos = (fingertip1_pos + fingertip2_pos) / 2.0

        target_pos = target.data.root_pos_w[:, :3]
        distance = torch.norm(gripper_center_pos - target_pos, dim=1)

        # Get current training step for curriculum
        current_step = getattr(env, "common_step_counter", 0)

        # Dynamic tolerance based on training progress
        initial_tol = getattr(env.cfg, "initial_tolerance", 0.02)  # 2cm
        final_tol = getattr(env.cfg, "final_tolerance", 0.005)  # 0.5cm
        curriculum_steps = getattr(env.cfg, "curriculum_steps", 1000000)

        # Linear curriculum progression
        progress = min(current_step / curriculum_steps, 1.0)
        current_tolerance = initial_tol - (initial_tol - final_tol) * progress

        # Reward based on current curriculum level
        success_mask = distance <= current_tolerance
        base_reward = torch.where(
            success_mask, 1.0, torch.exp(-distance / current_tolerance)
        )

        # Bonus for achieving harder tolerances
        difficulty_bonus = (1.0 - progress) * 0.5  # Extra reward for early precision
        total_reward = base_reward + difficulty_bonus

        # Debug curriculum progress occasionally
        if current_step % 100000 == 0:
            try:
                print(
                    f"[Curriculum] Step {current_step}: tolerance={current_tolerance:.4f}, progress={progress:.2f}"
                )
            except Exception:
                pass

        _record_debug(env, "curriculum", total_reward)
        return total_reward

    except Exception:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.ones(num_envs, device=device) * 0.1


def baseline_negative_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant negative baseline reward to ensure initial reward is negative.

    This forces the model to actively learn to overcome the negative baseline,
    providing strong motivation for improvement.
    """
    try:
        # Constant negative reward for all environments
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")

        # Fixed negative baseline
        baseline = torch.full((num_envs,), -1.0, device=device, dtype=torch.float32)

        _record_debug(env, "baseline_negative", baseline)
        return baseline

    except Exception:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.full((num_envs,), -1.0, device=device, dtype=torch.float32)


def distance_penalty_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Heavy penalty for being far from the target using gripper center distance.

    This creates a strong negative signal that the model must overcome
    by getting closer to the target, ensuring negative baseline for random behavior.
    """
    try:
        robot: Articulation = env.scene["robot"]
        target: RigidObject = env.scene["target_sphere"]

        # Get gripper fingertip positions and calculate center
        fingertip1_pos = robot.data.body_pos_w[:, -2, :3]  # First fingertip
        fingertip2_pos = robot.data.body_pos_w[:, -1, :3]  # Second fingertip
        gripper_center_pos = (fingertip1_pos + fingertip2_pos) / 2.0

        target_pos = target.data.root_pos_w[:, :3]
        distance = torch.norm(gripper_center_pos - target_pos, dim=1)

        # Strong negative reward that scales with distance
        # At 1m distance: -1.0, at 0.5m: -0.5, at 0m: 0.0
        max_distance = 1.0  # Maximum expected distance
        normalized_distance = torch.clamp(distance / max_distance, 0.0, 1.0)

        # Quadratic penalty - gets worse quickly with distance
        penalty = -(normalized_distance**2)

        _record_debug(env, "distance_penalty", penalty)
        return penalty

    except Exception:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.full((num_envs,), -1.0, device=device, dtype=torch.float32)


def simple_distance_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Simple, effective distance-based reward to replace complex multi-tier system.

    Provides clear, continuous signal that scales smoothly from 0 to 1 based on distance.
    Designed to overcome 0.2 plateau by providing stronger, clearer gradients.
    Uses gripper fingertip center position for accurate grasp distance calculation.
    """
    try:
        robot: Articulation = env.scene["robot"]
        target: RigidObject = env.scene["target_sphere"]

        # Get gripper fingertip positions (last two bodies are the fingertips)
        # For Franka robot: body_pos_w[:, -2] and body_pos_w[:, -1] are the two fingertips
        fingertip1_pos = robot.data.body_pos_w[:, -2, :3]  # First fingertip
        fingertip2_pos = robot.data.body_pos_w[:, -1, :3]  # Second fingertip

        # Calculate the center point between the two fingertips (grasp center)
        gripper_center_pos = (fingertip1_pos + fingertip2_pos) / 2.0

        target_pos = target.data.root_pos_w[:, :3]
        distance = torch.norm(gripper_center_pos - target_pos, dim=1)

        # Multi-tier reward system emphasizing precision

        # Tier 1: Basic approach (> 20cm) - gentle guidance
        basic_reward = torch.where(
            distance > 0.20,
            0.5 * torch.exp(-distance / 0.3),  # Gentle exponential decay
            torch.zeros_like(distance),
        )

        # Tier 2: Intermediate approach (5cm - 20cm) - stronger signal
        intermediate_reward = torch.where(
            (distance > 0.05) & (distance <= 0.20),
            2.0 * (0.20 - distance) / 0.15,  # Linear increase as we get closer
            torch.zeros_like(distance),
        )

        # Tier 3: Precision zone (1.5cm - 5cm) - very strong signal for final approach
        precision_reward = torch.where(
            (distance > 0.015) & (distance <= 0.05),
            5.0
            * torch.exp(-(distance - 0.015) / 0.01),  # Strong exponential for precision
            torch.zeros_like(distance),
        )

        # Tier 4: Ultra-precision zone (< 1.5cm) - maximum reward for success
        ultra_precision_reward = torch.where(
            distance <= 0.015,
            10.0 * torch.exp(-distance / 0.005),  # Huge reward for achieving precision
            torch.zeros_like(distance),
        )

        # Combine all tiers
        total_reward = (
            basic_reward
            + intermediate_reward
            + precision_reward
            + ultra_precision_reward
        )

        # Clamp to reasonable range
        total_reward = torch.clamp(total_reward, min=0.0, max=15.0)

        _record_debug(env, "simple_distance", total_reward)
        return total_reward

    except Exception:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, device=device)


def reward_debug_monitor(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Monitoring function to track reward progression and detect plateaus.

    This function helps identify when training gets stuck and provides diagnostic information.
    Returns a small reward but mainly serves for debugging.
    Uses gripper center position for accurate grasp distance monitoring.
    """
    try:
        robot: Articulation = env.scene["robot"]
        target: RigidObject = env.scene["target_sphere"]

        # Get gripper fingertip positions and calculate center for monitoring
        fingertip1_pos = robot.data.body_pos_w[:, -2, :3]  # First fingertip
        fingertip2_pos = robot.data.body_pos_w[:, -1, :3]  # Second fingertip
        gripper_center_pos = (fingertip1_pos + fingertip2_pos) / 2.0

        obj_pos = target.data.root_pos_w[:, :3]
        distance = torch.norm(gripper_center_pos - obj_pos, dim=1)

        # Track training progress
        current_step = getattr(env, "common_step_counter", 0)

        # Periodic debugging output
        if current_step % 5000 == 0 and current_step > 0:
            min_dist = distance.min().item()
            max_dist = distance.max().item()
            mean_dist = distance.mean().item()

            # Check for plateau conditions
            if hasattr(env, "_last_mean_distance"):
                improvement = env._last_mean_distance - mean_dist
                if abs(improvement) < 0.001:  # Very small improvement
                    print(
                        f"[PLATEAU WARNING] Step {current_step}: Mean distance improvement: {improvement:.6f}"
                    )
                    print(
                        f"[PLATEAU WARNING] Current mean distance: {mean_dist:.4f}, min: {min_dist:.4f}, max: {max_dist:.4f}"
                    )

            env._last_mean_distance = mean_dist

            print(
                f"[Reward Monitor] Step {current_step}: Distance stats - Mean: {mean_dist:.4f}, Min: {min_dist:.4f}, Max: {max_dist:.4f}"
            )

        # Return a small baseline reward for contribution
        return torch.full_like(distance, 0.001)

    except Exception as e:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, device=device)
