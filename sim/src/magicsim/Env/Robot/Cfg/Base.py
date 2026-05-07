from typing import Dict
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.managers import ActionTermCfg as ActionTerm
from dataclasses import MISSING
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
import magicsim.Env.Robot.mdp as mdp
from isaaclab.utils.noise.noise_cfg import (
    ConstantNoiseCfg,
    UniformNoiseCfg,
    GaussianNoiseCfg,
)
from isaaclab.utils.noise.noise_cfg import NoiseCfg

NOISE_TYPE_DICT: Dict[str, NoiseCfg] = {
    "constant": ConstantNoiseCfg,
    "uniform": UniformNoiseCfg,
    "gaussian": GaussianNoiseCfg,
}


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    asset_name: str = MISSING
    available_action: Dict[str, Dict[str, ActionTerm]] = MISSING

    def __post_init__(self):
        del self.asset_name
        del self.available_action


@configclass
class RobotObsCfg(ObsGroup):
    asset_name: str = MISSING
    joint_pos: ObsTerm = MISSING
    joint_vel: ObsTerm = MISSING
    joint_effort: ObsTerm = MISSING
    base_pos: ObsTerm = MISSING
    base_quat: ObsTerm = MISSING

    def __post_init__(self):
        """Observations for policy group with state values."""
        self.joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg(self.asset_name)},
        )
        self.joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg(self.asset_name)},
        )
        self.joint_effort = ObsTerm(
            func=mdp.joint_effort, params={"asset_cfg": SceneEntityCfg(self.asset_name)}
        )

        self.base_pos = ObsTerm(
            func=mdp.base_pos, params={"robot_cfg": SceneEntityCfg(self.asset_name)}
        )
        self.base_quat = ObsTerm(
            func=mdp.base_quat, params={"robot_cfg": SceneEntityCfg(self.asset_name)}
        )
        self.enable_corruption = False
        self.concatenate_terms = False
        del self.asset_name


@configclass
class RobotCfg:
    """Configuration for the Base robot."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    robot: ArticulationCfg = MISSING
    action: ActionsCfg = MISSING
    obs: RobotObsCfg = MISSING
    type: str = MISSING
