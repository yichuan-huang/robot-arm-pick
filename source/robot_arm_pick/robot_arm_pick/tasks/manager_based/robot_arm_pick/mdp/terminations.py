"""Custom termination functions for Franka picking environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def success_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check if the task is successfully completed."""
    return env.check_success()


def constraint_violation_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check if constraint violation threshold is exceeded."""
    return env.check_constraint_violation()
