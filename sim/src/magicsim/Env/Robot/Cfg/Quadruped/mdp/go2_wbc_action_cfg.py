from dataclasses import MISSING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from magicsim.Env.Robot.Cfg.Quadruped.mdp.go2_wbc_action import Go2WBCAction


@configclass
class Go2WBCActionCfg(ActionTermCfg):
    """Configuration for Go2 quadruped WBC action term."""

    class_type: type[ActionTerm] = Go2WBCAction
    """Specifies the action term class type for Go2 WBC action."""
    robot_type: str = "go2"
    preserve_order: bool = False
    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    wbc_version: str = "go2_v1"
    """Version of the WBC controller to use."""
    num_wbc_joints: int = MISSING
    """Number of joints controlled by the WBC."""
    action_space: torch.Tensor = MISSING
    """Action space tensor of shape (2, action_dim)."""
    wbc_joint_yaml_path: str = MISSING
    """Path to the YAML file containing WBC joint configuration."""
