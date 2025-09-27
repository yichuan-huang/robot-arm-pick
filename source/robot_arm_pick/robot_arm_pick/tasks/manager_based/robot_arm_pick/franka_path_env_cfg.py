"""Franka path tracking environment configuration."""

import math
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp as custom_mdp

##
# Scene definition
##


@configclass
class PathSceneCfg(InteractiveSceneCfg):
    """Simple scene with Franka robot for path tracking."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
    )

    # Target sphere
    target_sphere = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetSphere",
        spawn=sim_utils.SphereCfg(
            radius=0.05,  # 5cm radius
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=False,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # Red color
                metallic=0.0,
                roughness=0.5,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.2, 0.05),  # On ground (z = sphere radius), reachable position
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Franka robot
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # Natural ready pose - arm slightly extended forward
                "panda_joint1": 0.0,  # Base rotation - neutral
                "panda_joint2": -0.785,  # Shoulder - slight forward lean
                "panda_joint3": 0.0,  # Upper arm rotation - neutral
                "panda_joint4": -2.356,  # Elbow - moderate bend (within limits)
                "panda_joint5": 0.0,  # Forearm rotation - neutral
                "panda_joint6": 1.571,  # Wrist pitch - ready position
                "panda_joint7": 0.785,  # Wrist roll - neutral
                "panda_finger_joint1": 0.04,  # Gripper open
                "panda_finger_joint2": 0.04,  # Gripper open
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Empty command config as we use predefined paths."""

    pass


@configclass
class ActionsCfg:
    """Action configuration."""

    # Robot arm joint actions (7 joints + 2 gripper joints)
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], scale=1.0
    )


@configclass
class ObservationsCfg:
    """Observation space configuration - path tracking related."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations for gripper center control."""

        # Robot arm state (7 joints)
        arm_joint_pos = ObsTerm(
            func=custom_mdp.robot_joint_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_ids=[0, 1, 2, 3, 4, 5, 6])
            },
        )
        arm_joint_vel = ObsTerm(
            func=custom_mdp.robot_joint_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_ids=[0, 1, 2, 3, 4, 5, 6])
            },
        )

        # Gripper center position
        gripper_center_pos = ObsTerm(
            func=custom_mdp.gripper_center_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Target sphere position
        target_pos = ObsTerm(
            func=custom_mdp.target_sphere_pos,
            params={"asset_cfg": SceneEntityCfg("target_sphere")},
        )

        # Distance from gripper center to target
        gripper_to_target_distance = ObsTerm(
            func=custom_mdp.gripper_to_target_distance,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "target_cfg": SceneEntityCfg("target_sphere"),
            },
        )

        # Task progress
        task_progress = ObsTerm(func=custom_mdp.task_progress)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Default observation group
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Precision-focused reward configuration for millimeter-level control."""

    # CORE distance reward - simplified and powerful
    distance_reward = RewTerm(
        func=custom_mdp.simple_distance_reward,
        weight=10.0,  # Primary learning signal
    )

    # Success reward - simple and powerful
    success_reward = RewTerm(
        func=custom_mdp.reach_target_reward,
        weight=100.0,  # Huge reward for success
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target_sphere"),
            "threshold": 0.015,  # 1.5cm threshold - requires precision
        },
    )

    # Light penalties - allow learning without overwhelming negative feedback
    action_smoothness = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.1,  # Light penalty
    )

    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.1,  # Light penalty
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Debug monitor (no weight - pure logging)
    debug_monitor = RewTerm(
        func=custom_mdp.reward_debug_monitor,
        weight=0.0,  # No learning impact - pure debugging
    )


@configclass
class TerminationsCfg:
    """Termination conditions configuration."""

    # Task success - target reached (strict precision required)
    success = DoneTerm(
        func=custom_mdp.target_reached,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target_sphere"),
            "threshold": 0.02,  # 2cm precision required for task completion
        },
    )

    # Robot out of workspace bounds
    out_of_bounds = DoneTerm(
        func=custom_mdp.robot_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Robot falling (safety consideration)
    robot_falling = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.5, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class EventsCfg:
    """Events configuration for resetting the environment."""

    reset_scene_to_default = EventTermCfg(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )


@configclass
class CurriculumCfg:
    """Curriculum learning configuration - gradually increase task difficulty."""

    pass


##
# Environment configuration
##


@configclass
class FrankaPathEnvCfg(ManagerBasedRLEnvCfg):
    """Franka path control environment configuration."""

    # Scene configuration
    scene: PathSceneCfg = PathSceneCfg(num_envs=1024, env_spacing=2.5)

    # Basic settings
    episode_length_s = 12.0  # 12 second maximum task time (matches max_episode_time)
    decimation = 2  # Control frequency = sim_freq / decimation

    # MDP components
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    # Environment specific parameters with dynamic curriculum
    max_path_deviation: float = 0.01  # Maximum path deviation 1cm (much stricter)
    target_tolerance: float = 0.005  # Target tolerance 0.5cm (extremely tight)
    max_episode_time: float = 12.0  # Longer time to allow for precision

    # Curriculum learning parameters (NEW)
    curriculum_enabled: bool = True
    initial_tolerance: float = 0.02  # Start easier (2cm)
    final_tolerance: float = 0.005  # End harder (0.5cm)
    curriculum_steps: int = 1000000  # 1M steps to transition
    plateau_detection: bool = True  # Auto-detect plateaus
    exploration_noise: float = 0.1  # Add exploration noise

    def __post_init__(self):
        """Post-processing configuration."""
        # Simulation settings
        self.sim.dt = 1.0 / 120.0  # 120Hz simulation frequency
        self.sim.render_interval = self.decimation

        # Viewer settings
        self.viewer.resolution = (1280, 720)
        self.viewer.eye = (2.5, 2.5, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)


# Quick test configuration - for debugging
@configclass
class FrankaPathEnvCfg_PLAY(FrankaPathEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Play mode settings
        self.scene.num_envs = 50
        self.episode_length_s = 12.0  # Longer test time
        self.sim.render_interval = 1  # Real-time rendering
