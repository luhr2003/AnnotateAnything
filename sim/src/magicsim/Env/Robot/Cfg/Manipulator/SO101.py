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
    FrameSensorCfg,
    ManipulatorCfg,
    ManipulatorActionsCfg,
    ManipulatorObsCfg,
    ManipulatorPlannerCfg,
)
import magicsim.Env.Robot.mdp as mdp
from isaaclab.managers import ActionTermCfg as ActionTerm
from dataclasses import MISSING
from isaaclab.managers import ObservationTermCfg as ObsTerm
from magicsim import MAGICSIM_ASSETS
from magicsim.Env.Robot.mdp.curobo_ik_cfg import CuroboIKActionCfg


SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/SO101.usd",
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
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.5,
            "elbow_flex": 0.8,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
    ),
    actuators={
        "so101_arm_base": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex"],
            effort_limit_sim=20.0,
            stiffness=300.0,
            damping=30.0,
        ),
        "so101_arm_wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_flex", "wrist_roll"],
            effort_limit_sim=20.0,
            stiffness=200.0,
            damping=20.0,
        ),
        "so101_gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=20.0,
            stiffness=400.0,
            damping=20.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


@configclass
class SO101ActionsCfg(ManipulatorActionsCfg):
    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "arm_action": {
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=[
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                ],
            ),
            "joint_pos_vel": mdp.JointPositionVelocityToLimitsActionCfg(
                joint_names=[
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                ],
            ),
            "ik_curobo": CuroboIKActionCfg(
                joint_names=[
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                ],
                robot_cfg_file="magicsim_so101.yml",
                # tool_frames is read from the YAML's kinematics.tool_frames.
                action_space=torch.tensor(
                    [
                        [-0.3, -0.3, 0.0, -1.0, -1.0, -1.0, -1.0],
                        [0.3, 0.3, 0.5, 1.0, 1.0, 1.0, 1.0],
                    ]
                ),
                num_seeds=20,
                position_threshold=0.005,
                rotation_threshold=0.05,
                decimation=4,
                diff_ik_method="pinv",
            ),
        },
        "eef_action": {
            "binary": mdp.BinaryJointPositionActionCfg(
                joint_names=["gripper"],
                open_command_expr={"gripper": 1.5},
                close_command_expr={"gripper": 0.0},
            ),
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=["gripper"],
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()


class SO101FrameCfg(FrameSensorCfg):
    def __post_init__(self):
        self.target_frames = [
            FrameTransformerCfg.FrameCfg(
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
            FrameTransformerCfg.FrameCfg(
                name="jaw",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
        ]
        self.visualizer_cfg.prim_path = (
            self.robot_prim_path + "/Visuals/FrameTransformer"
        )
        self.target_frames[0].prim_path = self.robot_prim_path + "/gripper"
        self.target_frames[1].prim_path = self.robot_prim_path + "/jaw"
        self.prim_path = self.robot_prim_path + "/base"


@configclass
class SO101ObsCfg(ManipulatorObsCfg):
    gripper_pos: ObsTerm = MISSING

    def __post_init__(self):
        self.gripper_pos = ObsTerm(
            func=mdp.gripper_pos, params={"robot_cfg": SceneEntityCfg(self.asset_name)}
        )
        super().__post_init__()


@configclass
class SO101PlannerCfg(ManipulatorPlannerCfg):
    max_eef_num: int = 1

    arm_action_dim: Dict[str, int] = {
        "curobo": 7,
        "ik_curobo": 7,
        "default": 5,
    }
    eef_action_dim: Dict[str, int] = {
        "default": 1,
    }
    arm_action_space: Dict[str, torch.Tensor] = {
        "curobo": torch.tensor(
            [
                [-1, -1, -1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ]
        ),
        "default": None,
    }
    eef_action_space: Dict[str, torch.Tensor] = {
        "default": None,
    }


@configclass
class SO101Cfg(ManipulatorCfg):
    """Configuration for the SO-ARM100 (SO101) 5-DoF arm + 1-DoF gripper."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    arm_action_name: str = "joint_pos"
    eef_action_name: str = "binary"
    frame_name: str = "ee_frame"
    robot: ArticulationCfg = MISSING
    action: SO101ActionsCfg = MISSING
    ee_frame: SO101FrameCfg = MISSING
    obs: SO101ObsCfg = MISSING
    planner: SO101PlannerCfg = SO101PlannerCfg()

    def __post_init__(self):
        self.robot: ArticulationCfg = SO101_CFG
        self.robot.prim_path = self.prim_path
        self.action: SO101ActionsCfg = SO101ActionsCfg(
            asset_name=self.asset_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = SO101FrameCfg(robot_prim_path=self.prim_path)
        self.obs: SO101ObsCfg = SO101ObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        super().__post_init__()
