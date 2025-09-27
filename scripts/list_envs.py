# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the available environments in Isaac Lab.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.

All the environments are registered in the `robot_arm_pick` extension. They start
with `Isaac` in their name.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

import robot_arm_pick.tasks  # noqa: F401


def main():
    """Print all environments registered in `robot_arm_pick` extension."""
    # print all the available environments
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Robot Arm Pick Project"
    # set alignment of table columns
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    # count of environments
    index = 0
    # acquire all robot_arm_pick environments names
    for task_spec in gym.registry.values():
        # Look for environments that are from robot_arm_pick project
        if (
            "robot_arm_pick" in str(task_spec.kwargs.get("env_cfg_entry_point", ""))
            or "Isaac-Franka-Picking" in task_spec.id
            or "Template-Robot-Arm-Pick" in task_spec.id
        ):
            # add details to table
            table.add_row(
                [
                    index + 1,
                    task_spec.id,
                    task_spec.entry_point,
                    task_spec.kwargs["env_cfg_entry_point"],
                ]
            )
            # increment count
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # close the app
        simulation_app.close()
