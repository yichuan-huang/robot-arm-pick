"""Franka path tracking environments."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Franka path tracking environment
gym.register(
    id="Isaac-Franka-Path-v0",
    entry_point=f"{__name__}.franka_path_env:FrankaPathEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_path_env_cfg:FrankaPathEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
