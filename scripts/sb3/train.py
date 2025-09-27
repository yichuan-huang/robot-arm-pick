# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""
Script to train RL agent with Stable Baselines3 - Enhanced for path tracking tasks.
"""

import argparse
import contextlib
import signal
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Train an RL agent with Stable-Baselines3."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="sb3_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--log_interval", type=int, default=50_000, help="Log data every n timesteps."
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Continue the training from checkpoint.",
)
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)
parser.add_argument(
    "--export_io_descriptors",
    action="store_true",
    default=False,
    help="Export IO descriptors.",
)
parser.add_argument(
    "--rew_debug_interval",
    type=int,
    default=0,
    help="If >0, print mean of each reward term every N steps.",
)
parser.add_argument(
    "--success_base_bonus",
    type=float,
    default=None,
    help="Override base success bonus in env cfg.",
)
parser.add_argument(
    "--success_time_bonus",
    type=float,
    default=None,
    help="Override extra time-based success bonus in env cfg.",
)
parser.add_argument(
    "--max_path_deviation",
    type=float,
    default=0.01,
    help="Maximum allowed path deviation in meters (default: 0.01m = 1cm).",
)
parser.add_argument(
    "--target_tolerance",
    type=float,
    default=0.005,
    help="Target tolerance for task completion in meters (default: 0.005m = 5mm).",
)
parser.add_argument(
    "--max_episode_time",
    type=float,
    default=12.0,
    help="Maximum episode time in seconds (default: 12.0s).",
)
parser.add_argument(
    "--curriculum_initial_tolerance",
    type=float,
    default=None,
    help="Override initial target tolerance for curriculum learning.",
)
parser.add_argument(
    "--curriculum_final_tolerance",
    type=float,
    default=None,
    help="Override final target tolerance for curriculum learning.",
)
parser.add_argument(
    "--curriculum_decay_steps",
    type=int,
    default=None,
    help="Override decay steps for curriculum learning.",
)
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def cleanup_pbar(*args):
    """A small helper to stop training and cleanup progress bar properly on ctrl+c"""
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, cleanup_pbar)

import gymnasium as gym
import numpy as np
import os
import random
import torch
from datetime import datetime

import omni
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_arm_pick.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict
):
    """Train with stable-baselines agent."""
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg["seed"] = (
        args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    )
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = (
            args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs
        )

    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    # Apply optional reward debug/bonus overrides
    if hasattr(env_cfg, "reward_debug_print_interval"):
        env_cfg.reward_debug_print_interval = args_cli.rew_debug_interval
    if args_cli.success_base_bonus is not None and hasattr(
        env_cfg, "success_base_bonus"
    ):
        env_cfg.success_base_bonus = args_cli.success_base_bonus
    if args_cli.success_time_bonus is not None and hasattr(
        env_cfg, "success_time_bonus"
    ):
        env_cfg.success_time_bonus = args_cli.success_time_bonus

    # Apply path tracking configuration overrides
    if hasattr(env_cfg, "max_path_deviation"):
        env_cfg.max_path_deviation = args_cli.max_path_deviation
    if hasattr(env_cfg, "target_tolerance"):
        env_cfg.target_tolerance = args_cli.target_tolerance
    if hasattr(env_cfg, "max_episode_time"):
        env_cfg.max_episode_time = args_cli.max_episode_time

    # Apply curriculum learning overrides
    if args_cli.curriculum_initial_tolerance is not None and hasattr(
        env_cfg, "target_tolerance_initial"
    ):
        env_cfg.target_tolerance_initial = args_cli.curriculum_initial_tolerance
    if args_cli.curriculum_final_tolerance is not None and hasattr(
        env_cfg, "target_tolerance_final"
    ):
        env_cfg.target_tolerance_final = args_cli.curriculum_final_tolerance
    if args_cli.curriculum_decay_steps is not None and hasattr(
        env_cfg, "target_tolerance_decay_steps"
    ):
        env_cfg.target_tolerance_decay_steps = args_cli.curriculum_decay_steps

    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
    norm_args = {}
    for key in norm_keys:
        if key in agent_cfg:
            norm_args[key] = agent_cfg.pop(key)

    if norm_args and norm_args.get("normalize_input"):
        print(f"Normalizing input, {norm_args=}")
        env = VecNormalize(
            env,
            training=True,
            norm_obs=norm_args["normalize_input"],
            norm_reward=norm_args.get("normalize_value", False),
            clip_obs=norm_args.get("clip_obs", 100.0),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    if "Franka-Picking" in args_cli.task or "Franka-Path" in args_cli.task:
        print("[INFO] Using breakthrough hyperparameters to overcome plateau at 5e-4.")

        # Detect if we're continuing from a checkpoint (plateau scenario)
        is_continuing = args_cli.checkpoint is not None

        # Simplified training approach - focus on basic learning first
        print("[INFO] Using simplified learning approach for basic reaching task")
        agent_cfg.update(
            {
                "learning_rate": 3e-4,  # Standard learning rate
                "n_steps": 2048,  # Shorter rollout for frequent updates
                "batch_size": 64,  # Smaller batch for more updates
                "n_epochs": 10,  # Standard epochs
                "gamma": 0.99,  # Standard discount
                "gae_lambda": 0.95,  # Standard GAE
                "clip_range": 0.2,  # Conservative clipping
                "ent_coef": 0.01,  # Higher exploration for discovery
                "vf_coef": 0.5,  # Standard value function
                "max_grad_norm": 0.5,  # Conservative gradient clipping
            }
        )

        if policy_arch == "MlpPolicy":
            agent_cfg["policy_kwargs"] = {
                "net_arch": [512, 512, 256, 128],  # Deep network for complex control
                "activation_fn": torch.nn.ReLU,  # ReLU for faster computation
            }

    # Create agent
    agent = PPO(policy_arch, env, verbose=1, tensorboard_log=log_dir, **agent_cfg)
    if args_cli.checkpoint is not None:
        agent = agent.load(args_cli.checkpoint, env, print_system_info=True)
        print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}")
        print("[INFO] Applying plateau-breaking modifications...")

        # Increase learning rate for breakthrough
        original_lr = agent.learning_rate
        if callable(original_lr):
            new_lr = lambda progress: max(original_lr(progress) * 2.0, 1e-5)
        else:
            new_lr = min(original_lr * 2.0, 1e-3)
        agent.learning_rate = new_lr
        print(f"[INFO] Learning rate increased for plateau breakthrough")

    save_freq = (
        32000 if "Franka-Picking" in args_cli.task else 1000
    )  # More frequent saves for precision tasks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq, save_path=log_dir, name_prefix="model", verbose=2
    )
    callbacks = [checkpoint_callback, LogEveryNTimesteps(n_steps=args_cli.log_interval)]

    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(
            total_timesteps=n_timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=None,
        )
    # Save the final model
    final_model_path = os.path.join(log_dir, "model")
    agent.save(final_model_path)
    print("[INFO] Final model saved to:")
    print(f"  {final_model_path}.zip")

    # Also save with timestamp for backup
    timestamp_path = os.path.join(log_dir, f"model_final_{run_info}")
    agent.save(timestamp_path)
    print(f"[INFO] Backup model saved to: {timestamp_path}.zip")

    if isinstance(env, VecNormalize):
        print("Saving normalization")
        env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
