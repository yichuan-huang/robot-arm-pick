"""Custom reward functions for Franka picking environment.

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

from isaaclab.assets import Articulation
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


def trajectory_tracking_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bounded reward for staying close to the reference trajectory.

    Returns values in approximately [-0.5, 1.0]. Positive for good tracking,
    small negative when exceeding the configured deviation limit.
    """
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
    """Bounded reward in [0, 1] for being close to the target object.

    Uses a smooth exponential shaping on the EE-to-target distance.
    """
    ee_pos = env.scene["robot"].data.body_pos_w[:, -1, :3]
    obj_pos = env.scene["object"].data.root_pos_w[:, :3]
    distance = torch.norm(ee_pos - obj_pos, dim=1)

    # Characteristic distance where reward ~ 0.37; tune as needed
    d_char = 0.15
    reward = torch.exp(-distance / d_char)
    _record_debug(env, "approach_target", reward)
    return reward


def success_bonus_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Moderate, consistent success bonus with time-awareness.

    Base bonus ~20 with up to +10 extra for faster completion. Returns zeros for
    non-successful episodes. Output shape: (num_envs,).
    """
    success_mask = env.check_success()
    reward = torch.zeros(env.num_envs, device=env.device)

    base_bonus = float(_get_cfg_attr(env, "success_base_bonus", 20.0))
    extra_bonus = float(_get_cfg_attr(env, "success_time_bonus", 10.0))

    # Time efficiency (0..1); prefer Tensor on device
    progress = getattr(env, "task_progress", None)
    if progress is None:
        progress = env.episode_length_buf.float() / max(
            float(env.max_episode_length), 1.0
        )
    progress = torch.as_tensor(progress, device=env.device, dtype=torch.float32)
    progress = torch.clamp(progress, 0.0, 1.0).view(-1)

    total_bonus = base_bonus + (1.0 - progress) * extra_bonus
    # Apply only to successful envs
    reward[success_mask] = total_bonus[success_mask]

    # Debug print success count periodically
    if success_mask.sum().item() > 0:
        try:
            step = int(getattr(env, "total_steps", 0))
            if step % 1000 == 0:
                print(
                    f"[Reward Debug] Step {step}: {success_mask.sum().item()}/{env.num_envs} envs succeeded this step"
                )
        except Exception:
            pass

    _record_debug(env, "success_bonus", reward)
    return reward


def time_efficiency_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Small positive shaping that decreases over time and prefers good tracking.

    Returns values roughly in [0, 0.2]. Encourages making progress early while
    maintaining trajectory quality.
    """
    progress = getattr(env, "task_progress", None)
    if progress is None:
        progress = env.episode_length_buf.float() / max(
            float(env.max_episode_length), 1.0
        )
    progress = torch.as_tensor(progress, device=env.device, dtype=torch.float32)
    progress = torch.clamp(progress, 0.0, 1.0).view(-1)

    deviation = env.get_trajectory_deviation().squeeze(1)
    trajectory_quality = torch.exp(-deviation / 0.05)  # [0,1]

    # Reward is higher early on and with better quality
    base = (1.0 - progress) * trajectory_quality * 0.15

    # Small exploration bonus early
    exploration = torch.where(
        progress < 0.3, trajectory_quality * 0.05, torch.zeros_like(progress)
    )

    out = base + exploration
    _record_debug(env, "time_efficiency", out)
    return out


def grasp_precision_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bounded reward in [0, 1] for precise end-effector placement at target."""
    ee_pos = env.scene["robot"].data.body_pos_w[:, -1, :3]
    target_pos = env.scene["object"].data.root_pos_w[:, :3]
    distance = torch.norm(ee_pos - target_pos, dim=1)

    # Use curriculum-based tolerance if available, otherwise config default
    epsilon = getattr(
        env, "target_tolerance", _get_cfg_attr(env, "target_tolerance", 0.025)
    )
    reward = torch.where(
        distance <= epsilon, torch.ones_like(distance), torch.exp(-distance / 0.05)
    )
    _record_debug(env, "grasp_precision", reward)
    return reward


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
