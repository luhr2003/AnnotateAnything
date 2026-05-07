from dataclasses import MISSING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from magicsim.Env.Robot.Cfg.Humanoid.mdp.homie_wbc_action import HomieWBCAction


@configclass
class HomieWBCActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = HomieWBCAction
    """Specifies the action term class type for G1 WBC action."""
    robot_type: str = "g1"
    preserve_order: bool = False
    joint_names: list[str] = MISSING
    wbc_version: str = "homie_v2"
    num_wbc_joints: int = MISSING
    action_space: torch.Tensor = MISSING
    wbc_joint_yaml_path: str = MISSING
    decimation: int = 1
