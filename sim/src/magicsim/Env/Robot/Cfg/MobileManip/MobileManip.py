from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils import configclass
from dataclasses import MISSING
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
import magicsim.Env.Robot.mdp as mdp
from magicsim.Env.Robot.Cfg.Base import ActionsCfg, RobotCfg, RobotObsCfg
from typing import Dict
import torch


@configclass
class MobileManipActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    base_action: ActionTerm = MISSING
    arm_action: ActionTerm = MISSING
    eef_action: ActionTerm = MISSING
    base_action_name: str = MISSING
    arm_action_name: str = MISSING
    eef_action_name: str = MISSING

    def __post_init__(self):
        self.arm_action = self.available_action["arm_action"][self.arm_action_name]
        self.arm_action.asset_name = self.asset_name
        self.base_action = self.available_action["base_action"][self.base_action_name]
        self.base_action.asset_name = self.asset_name
        if self.eef_action_name is not None:
            self.eef_action = self.available_action["eef_action"][self.eef_action_name]
            self.eef_action.asset_name = self.asset_name
        else:
            self.eef_action = None
        super().__post_init__()
        del self.arm_action_name
        del self.eef_action_name
        del self.base_action_name


@configclass
class MobileManipObsCfg(RobotObsCfg):
    asset_name: str = MISSING
    frame_name: str = MISSING
    eef_pos: ObsTerm = MISSING
    eef_quat: ObsTerm = MISSING
    eef_relative_pos: ObsTerm = MISSING
    eef_relative_quat: ObsTerm = MISSING

    def __post_init__(self):
        """Observations for policy group with state values."""

        self.eef_pos = ObsTerm(
            func=mdp.ee_frame_pos,
            params={"ee_frame_cfg": SceneEntityCfg(self.frame_name)},
        )
        self.eef_quat = ObsTerm(
            func=mdp.ee_frame_quat,
            params={"ee_frame_cfg": SceneEntityCfg(self.frame_name)},
        )
        self.eef_relative_pos = ObsTerm(
            func=mdp.ee_rel_pos,
            params={"ee_frame_cfg": SceneEntityCfg(self.frame_name)},
        )
        self.eef_relative_quat = ObsTerm(
            func=mdp.ee_rel_quat,
            params={"ee_frame_cfg": SceneEntityCfg(self.frame_name)},
        )
        super().__post_init__()
        del self.frame_name


@configclass
class MobileManipCfg(RobotCfg):
    """Configuration for the Base robot."""

    type: str = "mobilemanip"
    base_action_name: str = MISSING
    arm_action_name: str = "joint_pos"
    eef_action_name: str = None
    frame_name: str = "ee_frame"

    robot: ArticulationCfg = MISSING
    action: MobileManipActionsCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    obs: MobileManipObsCfg = MISSING


@configclass
class FrameSensorCfg(FrameTransformerCfg):
    """Configuration for the frame sensor."""

    robot_prim_path: str = MISSING
    visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.copy()
    visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


@configclass
class MobileManipPlannerCfg:
    base_action_dim: Dict[str, int] = MISSING
    base_action_space: Dict[str, torch.Tensor] = MISSING
    max_eef_num: int = 1
