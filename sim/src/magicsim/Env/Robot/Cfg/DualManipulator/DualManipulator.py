"""
Dual-arm manipulator base configuration.
Inherits from ManipulatorCfg, provides common structure for dual-arm robots
(e.g. Xtrainer: 2 EEFs, 2 grippers, arm_action 14D, eef_action 2D).

Frame order: right first, then left (target_frames[0]=right, target_frames[1]=left).
"""

from typing import Dict

import torch
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.utils import configclass
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from dataclasses import MISSING

import magicsim.Env.Robot.mdp as mdp
from magicsim.Env.Robot.Cfg.Base import ActionsCfg, RobotObsCfg
from magicsim.Env.Robot.Cfg.Manipulator.Manipulator import (
    ManipulatorCfg,
    ManipulatorPlannerCfg,
    FrameSensorCfg,
)


@configclass
class DualManipulatorActionsCfg(ActionsCfg):
    """Action specifications for dual-arm MDP. Same structure as ManipulatorActionsCfg."""

    arm_action: ActionTerm = MISSING
    eef_action: ActionTerm | None = None
    arm_action_name: str = MISSING
    eef_action_name: str = MISSING

    def __post_init__(self):
        self.arm_action = self.available_action["arm_action"][self.arm_action_name]
        self.arm_action.asset_name = self.asset_name
        if self.eef_action_name is not None:
            self.eef_action = self.available_action["eef_action"][self.eef_action_name]
            self.eef_action.asset_name = self.asset_name
        else:
            self.eef_action = None
        super().__post_init__()
        del self.arm_action_name
        del self.eef_action_name


@configclass
class DualManipulatorObsCfg(RobotObsCfg):
    """Observation config for dual-arm. Inherits from RobotObsCfg. Right first, left second."""

    asset_name: str = MISSING
    frame_name: str = MISSING
    right_eef_pos: ObsTerm = MISSING
    right_eef_quat: ObsTerm = MISSING
    right_eef_relative_pos: ObsTerm = MISSING
    right_eef_relative_quat: ObsTerm = MISSING
    left_eef_pos: ObsTerm = MISSING
    left_eef_quat: ObsTerm = MISSING
    left_eef_relative_pos: ObsTerm = MISSING
    left_eef_relative_quat: ObsTerm = MISSING
    gripper_pos: ObsTerm = MISSING
    # For get_robot_state compatibility: eef_pos [N,2,3], eef_quat [N,2,4] (right first, left second)
    eef_pos: ObsTerm = MISSING
    eef_quat: ObsTerm = MISSING

    def __post_init__(self):
        # Right (frame_index=0), left (frame_index=1)
        self.right_eef_pos = ObsTerm(
            func=mdp.ee_frame_pos_at,
            params={"ee_frame_cfg": SceneEntityCfg(self.frame_name), "frame_index": 0},
        )
        self.right_eef_quat = ObsTerm(
            func=mdp.ee_frame_quat_at,
            params={"ee_frame_cfg": SceneEntityCfg(self.frame_name), "frame_index": 0},
        )
        self.right_eef_relative_pos = ObsTerm(
            func=mdp.ee_rel_pos_at,
            params={"ee_frame_cfg": SceneEntityCfg(self.frame_name), "frame_index": 0},
        )
        self.right_eef_relative_quat = ObsTerm(
            func=mdp.ee_rel_quat_at,
            params={"ee_frame_cfg": SceneEntityCfg(self.frame_name), "frame_index": 0},
        )
        self.left_eef_pos = ObsTerm(
            func=mdp.ee_frame_pos_at,
            params={"ee_frame_cfg": SceneEntityCfg(self.frame_name), "frame_index": 1},
        )
        self.left_eef_quat = ObsTerm(
            func=mdp.ee_frame_quat_at,
            params={"ee_frame_cfg": SceneEntityCfg(self.frame_name), "frame_index": 1},
        )
        self.left_eef_relative_pos = ObsTerm(
            func=mdp.ee_rel_pos_at,
            params={"ee_frame_cfg": SceneEntityCfg(self.frame_name), "frame_index": 1},
        )
        self.left_eef_relative_quat = ObsTerm(
            func=mdp.ee_rel_quat_at,
            params={"ee_frame_cfg": SceneEntityCfg(self.frame_name), "frame_index": 1},
        )
        self.gripper_pos = ObsTerm(
            func=mdp.gripper_pos, params={"robot_cfg": SceneEntityCfg(self.asset_name)}
        )
        self.eef_pos = ObsTerm(
            func=mdp.ee_dual_pos,
            params={"ee_frame_cfg": SceneEntityCfg(self.frame_name)},
        )
        self.eef_quat = ObsTerm(
            func=mdp.ee_dual_quat,
            params={"ee_frame_cfg": SceneEntityCfg(self.frame_name)},
        )
        super().__post_init__()
        del self.frame_name


@configclass
class DualManipulatorPlannerCfg(ManipulatorPlannerCfg):
    """Planner config for dual-arm. Default: arm 14D (2×7), eef 2D (2 grippers)."""

    max_eef_num: int = 2

    arm_action_dim: Dict[str, int] = {
        "curobo": 14,
        "default": 12,
        "ik_pink": 14,
    }
    eef_action_dim: Dict[str, int] = {
        "default": 2,
    }
    arm_action_space: Dict[str, torch.Tensor] = {}
    eef_action_space: Dict[str, torch.Tensor] = {}


@configclass
class DualManipulatorFrameCfg(FrameSensorCfg):
    """
    Frame config for dual-arm: 2 EEF frames, order right first then left.
    Subclasses must set left_link_name and right_link_name.
    """

    left_link_name: str = MISSING
    right_link_name: str = MISSING

    def __post_init__(self):
        # Frame order: right first, left second
        self.target_frames = [
            FrameTransformerCfg.FrameCfg(
                name="end_effector_right",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
            FrameTransformerCfg.FrameCfg(
                name="end_effector_left",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
        ]
        self.visualizer_cfg.prim_path = (
            self.robot_prim_path + "/Visuals/FrameTransformer"
        )
        self.target_frames[0].prim_path = (
            self.robot_prim_path + "/" + self.right_link_name
        )
        self.target_frames[1].prim_path = (
            self.robot_prim_path + "/" + self.left_link_name
        )
        self.prim_path = self.robot_prim_path + "/base_link"
        super().__post_init__()


@configclass
class DualManipulatorCfg(ManipulatorCfg):
    """Base configuration for dual-arm manipulators (e.g. Xtrainer)."""

    type: str = "dual_manipulator"
    arm_action_name: str = "joint_pos"
    eef_action_name: str = "binary"
    frame_name: str = "ee_frame"
