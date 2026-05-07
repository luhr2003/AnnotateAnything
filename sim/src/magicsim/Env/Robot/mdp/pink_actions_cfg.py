# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import torch

from magicsim.Env.Robot.mdp.pink_ik import PinkIKControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from . import pink_task_space_actions


@configclass
class PinkInverseKinematicsActionCfg(ActionTermCfg):
    """Configuration for Pink inverse kinematics action term.

    This configuration is used to define settings for the Pink inverse kinematics action term,
    which is a inverse kinematics framework.
    """

    class_type: type[ActionTerm] = pink_task_space_actions.PinkInverseKinematicsAction
    """Specifies the action term class type for Pink inverse kinematics action."""

    pink_controlled_joint_names: list[str] = MISSING
    """List of joint names or regular expression patterns that specify the joints controlled by pink IK."""

    hand_joint_names: list[str] = MISSING
    """List of joint names or regular expression patterns that specify the joints controlled by hand retargeting."""

    controller: PinkIKControllerCfg = MISSING
    """Configuration for the Pink IK controller that will be used to solve the inverse kinematics."""

    enable_gravity_compensation: bool = True
    """Whether to compensate for gravity in the Pink IK controller."""

    target_eef_link_names: dict[str, str] = MISSING
    """Dictionary mapping task names to controlled link names for the Pink IK controller.

    This dictionary should map the task names (e.g., 'left_wrist', 'right_wrist') to the
    corresponding link names in the URDF that will be controlled by the IK solver.
    """

    action_space: torch.Tensor = MISSING
    """The action space for the action term. Should be a tuple of (low, high) or a list of such tuples.
    If a list is provided, it should have the same length as the action dimension.
    This defines the valid range for the end-effector poses (and hand joints if present).
    """

    num_joints: int = MISSING
    """The number of joints to control. This is used to set the action space."""

    relative_to_base: bool = False

    fallback_to_current: bool = True
    """When an input action contains NaN and no valid last passed action exists:
    if True, fall back to the current EEF pose / hand joint positions; if False,
    skip that env (drop it from this step's IK solve)."""

    decimation: int = 1
    """IK solve decimation: only recompute IK every N apply_actions calls.
    Cached joint targets are applied every step regardless. Default 1 (no decimation)."""


@configclass
class PinkDualDifferentialInverseKinematicsActionCfg(PinkInverseKinematicsActionCfg):
    """Hybrid Pink IK + per-arm differential IK action.

    Behaves like :class:`PinkInverseKinematicsActionCfg` on decimation fires
    (full Pink IK solve) and like the per-arm Jacobian tracker used by
    :class:`DualDifferentialInverseKinematicsActionCfg` between fires.

    The input layout is identical to the parent class: concatenated pose
    commands for the Pink ``variable_input_tasks`` frames in order, which for
    our dual-arm robots is **right first, left second** (7 + 7 = 14).
    """

    class_type: type[ActionTerm] = (
        pink_task_space_actions.PinkDualDifferentialInverseKinematicsAction
    )

    right_joint_names: list[str] = MISSING
    left_joint_names: list[str] = MISSING
    right_body_name: str = MISSING
    left_body_name: str = MISSING
    right_command_reference_body_name: str | None = None
    left_command_reference_body_name: str | None = None
    right_body_offset: object | None = None
    left_body_offset: object | None = None
    diff_ik_controller: DifferentialIKControllerCfg = MISSING
    """Controller used for the per-arm Jacobian tracker on non-Pink steps."""

    decimation: int = 4
