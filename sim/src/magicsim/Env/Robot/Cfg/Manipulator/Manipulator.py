from typing import Dict
import torch
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


@configclass
class ManipulatorActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

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
class ManipulatorObsCfg(RobotObsCfg):
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
class ManipulatorPlannerCfg:
    """Planner layout: ``per_eef_dim = eef_action_dim // max_eef_num`` (from action slice)."""

    arm_action_dim: Dict[str, int] = MISSING
    eef_action_dim: Dict[str, int] = MISSING
    arm_action_space: Dict[str, torch.Tensor] = MISSING
    eef_action_space: Dict[str, torch.Tensor] = MISSING
    # Number of end-effectors in the full robot action (e.g. Franka=1, dual-arm=2).
    max_eef_num: int = 1


@configclass
class ManipulatorCfg(RobotCfg):
    """Configuration for the Base robot."""

    type: str = "manipulator"
    arm_action_name: str = "joint_pos"
    eef_action_name: str = None
    frame_name: str = "ee_frame"

    robot: ArticulationCfg = MISSING
    action: ManipulatorActionsCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    obs: ManipulatorObsCfg = MISSING
    planner: ManipulatorPlannerCfg = MISSING


@configclass
class FrameSensorCfg(FrameTransformerCfg):
    """Configuration for the frame sensor."""

    robot_prim_path: str = MISSING
    visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.copy()
    visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
