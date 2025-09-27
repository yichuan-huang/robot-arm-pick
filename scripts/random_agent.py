"""Script to run Franka picking environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Random agent for Franka picking environment."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--num_steps", type=int, default=1000, help="Number of steps to run."
)
parser.add_argument(
    "--task", type=str, default="Isaac-Franka-Path-v0", help="Name of the task."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import robot_arm_pick.tasks  # noqa: F401


def main():
    """Random actions agent with Franka picking environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: Running random actions for {args_cli.num_steps} steps...")

    # reset environment
    obs, _ = env.reset()
    total_reward = 0.0
    step_count = 0

    # simulate environment
    while simulation_app.is_running() and step_count < args_cli.num_steps:
        # run everything in inference mode
        with torch.inference_mode():
            # sample random actions using gymnasium's action space bounds
            actions_np = env.action_space.sample()
            # convert to torch tensor and move to correct device
            actions = torch.from_numpy(actions_np).float().to(env.unwrapped.device)
            # apply actions
            obs, rewards, terminations, truncations, infos = env.step(actions)

            # Calculate total reward
            total_reward += torch.sum(rewards).item()
            step_count += 1

            # Print progress
            if step_count % 100 == 0:
                avg_reward = total_reward / step_count
                print(
                    f"Step {step_count}/{args_cli.num_steps}, Avg reward: {avg_reward:.3f}"
                )

    # Final statistics
    if step_count > 0:
        avg_reward = total_reward / step_count
        print(f"\n[INFO] Random agent completed!")
        print(f"[INFO] Average reward per step: {avg_reward:.3f}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
