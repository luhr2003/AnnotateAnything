from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.managers import ActionTermCfg as ActionTerm
from dataclasses import MISSING
from magicsim.Env.Robot.Cfg.Base import ActionsCfg, RobotCfg, RobotObsCfg
from typing import Callable, Dict, Optional
import torch


@configclass
class HumanoidActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    base_action: ActionTerm = None
    arm_action: ActionTerm = MISSING
    eef_action: ActionTerm = MISSING
    base_action_name: str = MISSING
    arm_action_name: str = MISSING
    eef_action_name: str = MISSING

    def __post_init__(self):
        if self.base_action_name is not None:
            self.base_action = self.available_action["base_action"][
                self.base_action_name
            ]
            self.base_action.asset_name = self.asset_name
        else:
            self.base_action = None
        if self.arm_action_name is not None:
            self.arm_action = self.available_action["arm_action"][self.arm_action_name]
            self.arm_action.asset_name = self.asset_name
        else:
            self.arm_action = None
        if self.eef_action_name is not None:
            self.eef_action = self.available_action["eef_action"][self.eef_action_name]
            self.eef_action.asset_name = self.asset_name
        else:
            self.eef_action = None
        super().__post_init__()
        del self.base_action_name
        del self.arm_action_name
        del self.eef_action_name


@configclass
class HumanoidObsCfg(RobotObsCfg):
    """Observation specifications for the MDP."""

    asset_name: str = MISSING

    def __post_init__(self):
        super().__post_init__()


@configclass
class HumanoidPlannerCfg:
    base_action_dim: Dict[str, int] = MISSING
    base_action_space: Dict[str, torch.Tensor] = MISSING
    arm_action_dim: Dict[str, int] = MISSING
    arm_action_space: Dict[str, torch.Tensor] = MISSING
    eef_action_dim: Dict[str, int] = MISSING
    eef_action_space: Dict[str, torch.Tensor] = MISSING
    # Number of end-effectors in the full action layout; per-EEF dex width = eef_dim // max_eef_num.
    max_eef_num: int = 1
    # Move strategy function for RetractMoveL: converts a straight-line trajectory
    # to a multi-segment trajectory (e.g., stand up -> move -> squat down)
    # Signature: (trajectory: torch.Tensor, robot_state: Dict) -> torch.Tensor
    move_strategy: Optional[Callable] = None
    # Distance threshold for base movement to trigger move strategy
    move_strategy_distance_threshold: float = 0.3


@configclass
class HumanoidCfg(RobotCfg):
    """Configuration for the Base humanoid robot."""

    type: str = "humanoid"
    action_name: str = MISSING

    robot: ArticulationCfg = MISSING
    action: HumanoidActionsCfg = MISSING
    obs: HumanoidObsCfg = MISSING
    planner: HumanoidPlannerCfg = MISSING
