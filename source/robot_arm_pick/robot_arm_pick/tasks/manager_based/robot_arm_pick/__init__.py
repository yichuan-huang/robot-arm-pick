"""Franka constrained picking environments."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Franka picking environment
gym.register(
    id="Isaac-Franka-Picking-v0",
    entry_point=f"{__name__}.franka_picking_env:FrankaPickingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_picking_env_cfg:FrankaPickingEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
