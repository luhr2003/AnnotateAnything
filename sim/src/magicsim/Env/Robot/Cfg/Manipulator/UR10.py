from typing import Dict

import torch
from isaaclab_assets import UR10_CFG, ArticulationCfg

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


UR10_HIGH_PD_CFG = UR10_CFG.copy()
UR10_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
UR10_HIGH_PD_CFG.actuators["arm"].stiffness = 1600.0
UR10_HIGH_PD_CFG.actuators["arm"].damping = 80.0
UR10_HIGH_PD_CFG.actuators["arm"].effort_limit = 176
UR10_HIGH_PD_CFG.actuators["arm"].velocity_limit = 200.0


@configclass
class UR10ActionsCfg(ManipulatorActionsCfg):
    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "arm_action": {
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=[
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ],
            ),
            "ik_abs": mdp.DifferentialInverseKinematicsActionCfg(
                joint_names=[".*"],
                body_name="ee_link",
                controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="pinv"
                ),
                body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
                    pos=[0.0, 0.0, 0.0]
                ),
                action_space=torch.tensor(
                    [
                        [-0.75, -0.75, -0.75, -1.0, -1.0, -1.0, -1.0],
                        [0.75, 0.75, 0.75, 1.0, 1.0, 1.0, 1.0],
                    ]
                ),  # robot workspace limits
            ),
        },
    }


class UR10FrameCfg(FrameSensorCfg):
    def __post_init__(self):
        self.target_frames = [
            FrameTransformerCfg.FrameCfg(
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                ),
            ),
        ]
        self.visualizer_cfg.prim_path = (
            self.robot_prim_path + "/Visuals/FrameTransformer"
        )
        self.target_frames[0].prim_path = self.robot_prim_path + "/ee_link"
        self.prim_path = self.robot_prim_path + "/base_link"


@configclass
class UR10Cfg(ManipulatorCfg):
    """Configuration for the UR10 robot."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    arm_action_name: str = "joint_pos"
    eef_action_name: str = None
    frame_name: str = "ee_frame"
    robot: ArticulationCfg = MISSING
    action: UR10ActionsCfg = MISSING
    ee_frame: UR10FrameCfg = MISSING
    obs: ManipulatorObsCfg = MISSING

    def __post_init__(self):
        if "ik" in self.arm_action_name:
            self.robot: ArticulationCfg = UR10_CFG
        else:
            self.robot: ArticulationCfg = UR10_CFG
        self.robot.prim_path = self.prim_path
        self.action: UR10ActionsCfg = UR10ActionsCfg(
            asset_name=self.asset_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = UR10FrameCfg(robot_prim_path=self.prim_path)
        self.obs: ManipulatorObsCfg = ManipulatorObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        super().__post_init__()
