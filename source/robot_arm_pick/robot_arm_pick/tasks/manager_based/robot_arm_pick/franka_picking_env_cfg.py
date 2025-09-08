"""Franka constrained picking environment configuration."""

import math
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
class TableSceneCfg(InteractiveSceneCfg):
    """Table scene with Franka robot and object."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # table
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.5], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.5, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.4, 0.3)),
        ),
    )

    # Franka robot
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.05, 0.0, 0.525),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -1.0,
                "panda_joint3": 0.0,
                "panda_joint4": -2.0,
                "panda_joint5": 0.0,
                "panda_joint6": 1.5,
                "panda_joint7": 0.0,
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=150.0,
                velocity_limit=2.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=28.0,
                velocity_limit=2.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=1000.0,
                damping=100.0,
            ),
        },
    )

    # target object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.6, 0.0, 0.575], rot=[1, 0, 0, 0]
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.5, 0.5, 0.5),
        ),
    )


##
# Environment configuration
##


@configclass
class FrankaPickingEnvCfg(ManagerBasedRLEnvCfg):
    """Franka constrained picking environment configuration."""

    # Scene settings
    scene: TableSceneCfg = TableSceneCfg(num_envs=64, env_spacing=3.0)

    # Basic settings - adjusted for better learning
    episode_length_s = 12.0  # Longer episodes for more exploration
    decimation = 2

    # Simulation settings
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=0.01)

    # Observations
    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            # robot state (first 7 joints)
            joint_pos = ObsTerm(func=custom_mdp.robot_arm_joint_pos)
            joint_vel = ObsTerm(func=custom_mdp.robot_arm_joint_vel)

            # gripper state (last 2 joints)
            gripper_pos = ObsTerm(func=custom_mdp.robot_gripper_joint_pos)

            # end-effector position and velocity
            ee_pos = ObsTerm(
                func=custom_mdp.robot_ee_pos,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
            ee_velocity = ObsTerm(func=custom_mdp.ee_linear_velocity)

            # object position
            object_pos = ObsTerm(
                func=custom_mdp.object_position,
                params={"asset_cfg": SceneEntityCfg("object")},
            )

            # relative positions (more informative for learning)
            relative_ee_to_target = ObsTerm(func=custom_mdp.relative_ee_to_target)
            ee_to_target_distance = ObsTerm(func=custom_mdp.ee_to_target_distance)

            # reference trajectory info
            ref_pos = ObsTerm(func=custom_mdp.reference_position)
            relative_ref_to_ee = ObsTerm(func=custom_mdp.relative_ref_to_ee)
            deviation = ObsTerm(func=custom_mdp.trajectory_deviation)

            # task progress and time info
            task_progress = ObsTerm(func=custom_mdp.task_progress)
            time_remaining = ObsTerm(func=custom_mdp.time_remaining)

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        policy: PolicyCfg = PolicyCfg()

    observations: ObservationsCfg = ObservationsCfg()

    # Actions
    @configclass
    class ActionsCfg:
        joint_efforts = mdp.JointEffortActionCfg(
            asset_name="robot", joint_names=["panda_joint.*", "panda_finger_.*"]
        )

    actions: ActionsCfg = ActionsCfg()

    action_space = 9

    # Balanced Rewards - easier to obtain positive rewards but still challenging
    @configclass
    class RewardsCfg:
        # Primary trajectory tracking reward - moderate weight
        track_trajectory = RewTerm(
            func=custom_mdp.trajectory_tracking_reward,
            weight=2.0,  # Moderate weight for guidance without dominating
        )

        # Target approach reward - strong positive signal
        approach_target = RewTerm(
            func=custom_mdp.target_approach_reward, weight=6.0
        )  # Strong positive reward for approaching target

        # Success bonus - highest priority
        success_bonus = RewTerm(
            func=custom_mdp.success_bonus_reward,
            weight=1.0,  # Weight is 1.0, bonus value is large in function
        )

        # Grasp precision reward - good weight
        grasp_precision = RewTerm(
            func=custom_mdp.grasp_precision_reward,
            weight=3.0,  # Reasonable weight for precision
        )

        # Time efficiency reward - small positive incentive
        time_efficiency = RewTerm(
            func=custom_mdp.time_efficiency_reward,
            weight=1.0,  # Small positive incentive
        )

        # Trajectory violation penalty - gentle constraint
        trajectory_violation = RewTerm(
            func=custom_mdp.trajectory_violation_penalty,
            weight=0.3,  # Gentle penalty to maintain some constraint
        )

        # Control penalties - small but present
        joint_velocity_penalty = RewTerm(
            func=custom_mdp.joint_velocity_penalty,
            weight=0.2,  # Small penalty for excessive movement
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_ids=[0, 1, 2, 3, 4, 5, 6])
            },
        )

        action_smoothness = RewTerm(
            func=custom_mdp.action_smoothness_penalty,
            weight=0.2,  # Small smoothness penalty
        )

    rewards: RewardsCfg = RewardsCfg()

    # Terminations
    @configclass
    class TerminationsCfg:
        time_out = DoneTerm(
            func=mdp.time_out,
            time_out=True,
        )

        success = DoneTerm(func=custom_mdp.success_termination)

        # Severe trajectory violation termination
        severe_violation = DoneTerm(
            func=custom_mdp.severe_trajectory_violation_termination
        )

        # Robot safety terminations
        robot_joint_limits = DoneTerm(
            func=mdp.joint_pos_out_of_limit,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_ids=[0, 1, 2, 3, 4, 5, 6])
            },
        )

    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # Constraint parameters - balanced for reasonable challenge
        self.max_trajectory_deviation = (
            0.10  # Balanced tolerance - not too strict, not too lenient
        )
        self.target_tolerance = 0.025  # Reasonable precision requirement
        self.severe_violation_threshold = 0.15  # Reasonable severe violation threshold

        # Trajectory parameters
        self.trajectory_update_rate = 0.1  # How often to update reference trajectory
        self.workspace_bounds = {
            "x_min": 0.2,
            "x_max": 0.8,
            "y_min": -0.3,
            "y_max": 0.3,
            "z_min": 0.55,
            "z_max": 0.8,
        }
