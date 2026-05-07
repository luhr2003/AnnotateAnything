from typing import Dict
import torch
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.managers import ActionTermCfg as ActionTerm
from dataclasses import MISSING
from magicsim.Env.Robot.Cfg.Base import ActionsCfg, RobotCfg, RobotObsCfg


@configclass
class QuadrupedActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    base_action: ActionTerm = MISSING
    base_action_name: str = MISSING

    def __post_init__(self):
        if self.base_action_name is not None:
            self.base_action = self.available_action["base_action"][
                self.base_action_name
            ]
            print("self.base_action: ", self.base_action)
            self.base_action.asset_name = self.asset_name
        else:
            self.base_action = None
        super().__post_init__()
        del self.base_action_name


@configclass
class QuadrupedObsCfg(RobotObsCfg):
    """Observation specifications for the MDP."""

    asset_name: str = MISSING

    def __post_init__(self):
        super().__post_init__()


@configclass
class QuadrupedPlannerCfg:
    """Planner configuration for quadruped robots."""

    quadruped_action_dim: Dict[str, int] = MISSING
    eef_action_dim: Dict[str, int] = MISSING
    quadruped_action_space: Dict[str, torch.Tensor] = MISSING
    eef_action_space: Dict[str, torch.Tensor] = MISSING
    max_eef_num: int = 1


@configclass
class QuadrupedCfg(RobotCfg):
    """Configuration for the Base humanoid robot."""

    type: str = "quadruped"
    base_action_name: str = MISSING

    robot: ArticulationCfg = MISSING
    base_action: QuadrupedActionsCfg = MISSING
    obs: QuadrupedObsCfg = MISSING
