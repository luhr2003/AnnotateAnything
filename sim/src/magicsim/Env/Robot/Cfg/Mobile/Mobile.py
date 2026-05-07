from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
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
class MobileActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    base_action: ActionTerm = MISSING
    base_action_name: str = MISSING

    def __post_init__(self):
        self.base_action = self.available_action["base_action"][self.base_action_name]
        self.base_action.asset_name = self.asset_name
        super().__post_init__()
        del self.base_action_name


@configclass
class MobileObsCfg(RobotObsCfg):
    base_pos: ObsTerm = MISSING
    base_quat: ObsTerm = MISSING
    base_ang_vel: ObsTerm = MISSING
    base_lin_vel: ObsTerm = MISSING

    def __post_init__(self):
        """Observations for policy group with state values."""
        self.base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg(self.asset_name)}
        )
        self.base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg(self.asset_name)}
        )
        super().__post_init__()


@configclass
class novaCarterObsCfg(RobotObsCfg):
    base_pos: ObsTerm = MISSING
    base_quat: ObsTerm = MISSING
    base_ang_vel: ObsTerm = MISSING
    base_lin_vel: ObsTerm = MISSING

    def __post_init__(self):
        """Observations for policy group with state values."""
        self.base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg(self.asset_name)}
        )
        self.base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg(self.asset_name)}
        )
        # self.front_steer = ObsTerm(
        #     func=mdp.front_steer, params={"robot_cfg": SceneEntityCfg(self.asset_name)}
        # )
        super().__post_init__()


@configclass
class MobileCfg(RobotCfg):
    """Configuration for the Base robot."""

    type: str = "mobile"
    base_action_name: str = "differential_drive"

    robot: ArticulationCfg = MISSING
    action: MobileActionsCfg = MISSING
    obs: MobileObsCfg = MISSING


@configclass
class MobilePlannerCfg:
    base_action_dim: Dict[str, int] = MISSING
    base_action_space: Dict[str, torch.Tensor] = MISSING
    max_eef_num: int = 1
