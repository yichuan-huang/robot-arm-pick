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
            size=(0.6, 0.5, 0.05),  # 桌子尺寸调小：60cm x 50cm x 5cm
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
            pos=(0.05, 0.0, 0.525),  # 机械臂位置：桌子前方，不在桌子上
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
            pos=[0.6, 0.0, 0.575], rot=[1, 0, 0, 0]  # 桌面上的位置
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
    scene: TableSceneCfg = TableSceneCfg(
        num_envs=64, env_spacing=3.0
    )  # Small for visualization

    # Basic settings
    episode_length_s = 8.0
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

            # end-effector position
            ee_pos = ObsTerm(
                func=custom_mdp.robot_ee_pos,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )

            # object position
            object_pos = ObsTerm(
                func=custom_mdp.object_position,
                params={"asset_cfg": SceneEntityCfg("object")},
            )

            # reference trajectory info
            ref_pos = ObsTerm(func=custom_mdp.reference_position)
            deviation = ObsTerm(func=custom_mdp.trajectory_deviation)
            time_remaining = ObsTerm(func=custom_mdp.time_remaining)

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        policy: PolicyCfg = PolicyCfg()

    observations: ObservationsCfg = ObservationsCfg()

    # Actions (7 arm joints + 2 gripper joints = 9 DOF)
    @configclass
    class ActionsCfg:
        joint_efforts = mdp.JointEffortActionCfg(
            asset_name="robot", joint_names=["panda_joint.*", "panda_finger_.*"]
        )

    actions: ActionsCfg = ActionsCfg()

    # Actions (9 joint efforts for Franka)
    action_space = 9

    # Rewards
    @configclass
    class RewardsCfg:
        # Time penalty
        time_penalty = RewTerm(
            func=custom_mdp.time_penalty_reward,
            weight=1.0,
        )

        # Trajectory tracking
        track_trajectory = RewTerm(
            func=custom_mdp.trajectory_tracking_reward, weight=5.0
        )

        # Target approach
        approach_target = RewTerm(func=custom_mdp.target_approach_reward, weight=2.0)

        # Success bonus
        success_bonus = RewTerm(func=custom_mdp.success_bonus_reward, weight=100.0)

    rewards: RewardsCfg = RewardsCfg()

    # Terminations
    @configclass
    class TerminationsCfg:
        time_out = DoneTerm(
            func=mdp.time_out,
            time_out=True,
        )
        success = DoneTerm(func=custom_mdp.success_termination)
        constraint_violation = DoneTerm(
            func=custom_mdp.constraint_violation_termination
        )

    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # Constraint parameters
        self.max_trajectory_deviation = (
            0.08  # 8cm max deviation # TODO: Tune this parameter
        )
        self.target_tolerance = 0.05  # 5cm grasp tolerance # TODO: Tune this parameter
        self.severe_violation_threshold = (
            0.15  # 15cm severe violation # TODO: Tune this parameter
        )
