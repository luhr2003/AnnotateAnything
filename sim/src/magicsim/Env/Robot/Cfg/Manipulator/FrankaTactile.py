from typing import Dict

import torch
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from magicsim.Env.Robot.Cfg.Manipulator.Manipulator import (
    ManipulatorCfg,
    ManipulatorActionsCfg,
    ManipulatorObsCfg,
    FrameSensorCfg,
)
import magicsim.Env.Robot.mdp as mdp
from isaaclab.managers import ActionTermCfg as ActionTerm
from dataclasses import MISSING
from magicsim import MAGICSIM_ASSETS
from isaaclab.managers import ObservationTermCfg as ObsTerm

FRANKA_GSMINI_GRIPPER_RIGID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Tactile/Robots/GSminiFranka/GSminiFranka.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


@configclass
class FrankaActionsCfg(ManipulatorActionsCfg):
    available_action: Dict[
        str, Dict[str, ActionTerm]
    ] = {  # Use this paramter to load action manager
        "arm_action": {
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=["panda_joint.*"],
            ),
            "ik_abs": mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="dls"
                ),
                body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
                    pos=[0.0, 0.0, 0.107]
                ),
                action_space=torch.tensor(
                    [
                        [-0.6, -0.6, -1.0, -1.0, -1.0, -1.0, -1.0],
                        [0.6, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0],
                    ]
                ),  # robot workspace limits
            ),
        },
        "eef_action": {
            "binary": mdp.BinaryJointPositionActionCfg(
                joint_names=["panda_finger.*"],
                open_command_expr={"panda_finger_.*": 0.04},
                close_command_expr={"panda_finger_.*": 0.0},
            ),
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=["panda_finger.*"],
            ),
        },
    }


class FrankaFrameCfg(FrameSensorCfg):
    def __post_init__(self):
        self.target_frames = [
            FrameTransformerCfg.FrameCfg(
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1184],
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                name="tool_rightfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                name="tool_leftfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
        ]
        self.visualizer_cfg.prim_path = (
            self.robot_prim_path + "/Visuals/FrameTransformer"
        )
        self.target_frames[0].prim_path = self.robot_prim_path + "/panda_hand"
        self.target_frames[1].prim_path = self.robot_prim_path + "/panda_rightfinger"
        self.target_frames[2].prim_path = self.robot_prim_path + "/panda_leftfinger"
        self.prim_path = self.robot_prim_path + "/panda_link0"


@configclass
class FrankaObsCfg(ManipulatorObsCfg):
    gripper_pos: ObsTerm = MISSING

    def __post_init__(self):
        self.gripper_pos = ObsTerm(
            func=mdp.gripper_pos, params={"robot_cfg": SceneEntityCfg(self.asset_name)}
        )
        super().__post_init__()


@configclass
class FrankaTactileCfg(ManipulatorCfg):
    """Configuration for the Franka robot."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    arm_action_name: str = "joint_pos"
    eef_action_name: str = "binary"
    frame_name: str = "ee_frame"
    robot: ArticulationCfg = MISSING
    action: FrankaActionsCfg = MISSING
    ee_frame: FrankaFrameCfg = MISSING
    obs: FrankaObsCfg = MISSING

    def __post_init__(self):
        self.robot: ArticulationCfg = FRANKA_GSMINI_GRIPPER_RIGID_CFG
        self.robot.prim_path = self.prim_path
        self.action: FrankaActionsCfg = FrankaActionsCfg(
            asset_name=self.asset_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = FrankaFrameCfg(robot_prim_path=self.prim_path)
        self.obs: FrankaObsCfg = FrankaObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        super().__post_init__()
