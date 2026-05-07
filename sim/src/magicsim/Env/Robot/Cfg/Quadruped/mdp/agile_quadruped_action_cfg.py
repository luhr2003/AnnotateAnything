# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from gymnasium import spaces

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from magicsim.Env.Robot.Cfg.Quadruped.mdp.agile_quadruped_action import (
    AgileQuadrupedAction,
)


@configclass
class AgileQuadrupedActionCfg(ActionTermCfg):
    """Configuration for the quadruped action term that is based on Agile quadruped RL policy."""

    class_type: type[ActionTerm] = AgileQuadrupedAction
    """The class type for the quadruped action term."""

    joint_names: list[str] = MISSING
    """The names of the joints to control."""

    policy_path: str = MISSING
    """The path to the policy model."""

    policy_output_offset: float = 0.0
    """Offsets the output of the policy."""

    policy_output_scale: float = 1.0
    """Scales the output of the policy."""

    num_wbc_joints: int = MISSING
    """The number of joints to control."""

    action_space: spaces.Box = MISSING
    """The action space for the policy input commands."""
