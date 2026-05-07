# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from magicsim.Env.Robot.mdp.actions import (
    JointPositionToLimitsAction,
    JointPositionVelocityToLimitsAction,
    MultipleJointPositionToLimitsAction,
    BinaryJointPositionAction,
    BinaryJointChoicePositionAction,
    InterpolatedJointPositionAction,
    InterpolatedJointChoicePositionAction,
    MultipleBinaryJointPositionAction,
    MultipleBinaryJointChoicePositionAction,
    MultipleInterpolatedJointPositionAction,
    MultipleInterpolatedJointChoicePositionAction,
    DifferentialInverseKinematicsAction,
    DualDifferentialInverseKinematicsAction,
    HolonomicAction,
    HolonomicVWAction,
    DifferentialAction,
    AckermannSteeringAction,
    HolonomicForQuadrupedAction,
)

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg


@configclass
class JointPositionToLimitsActionCfg(ActionTermCfg):
    """Configuration for the bounded joint position action term.

    See :class:`JointPositionToLimitsAction` for more details.
    """

    class_type: type[ActionTerm] = JointPositionToLimitsAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""

    rescale_to_limits: bool = False
    """Whether to rescale the action to the joint limits. Defaults to True.

    If True, the input actions are rescaled to the joint limits, i.e., the action value in
    the range [-1, 1] corresponds to the joint lower and upper limits respectively.

    Note:
        This operation is performed after applying the scale factor.
    """
    num_joints: int = None

    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False.

    If True, the joint order will match the order specified in joint_names.
    This is useful when you need to directly concatenate joint values in a specific order.
    """


@configclass
class MultipleJointPositionToLimitsActionGroupCfg:
    """Configuration for a single joint group in MultipleJointPositionToLimitsAction."""

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions for this group."""

    scale: float | dict[str, float] = 1.0
    """Scale factor for this group (float or dict of regex expressions)."""

    clip: dict[str, tuple[float, float]] | None = None
    """Optional clip range for this group. dict mapping joint regex to (min, max)."""

    rescale_to_limits: bool = False
    """Whether to rescale actions to joint limits. If True, input [-1,1] maps to limits."""

    num_joints: int | None = None
    """Expected number of joints. If set, validated against resolved joints."""

    preserve_order: bool = False
    """Whether to preserve joint order from joint_names."""


@configclass
class MultipleJointPositionToLimitsActionCfg(ActionTermCfg):
    """Configuration for multiple joint position-to-limits action term.

    Each joint group has its own scale/clip/rescale_to_limits. action_dim = sum of
    num_joints across all groups. NaN handling is per-group: if a group has any NaN,
    it is filled with current joint positions for that group (like pink_task_space_actions).
    """

    class_type: type[ActionTerm] = MultipleJointPositionToLimitsAction

    joint_groups: list[MultipleJointPositionToLimitsActionGroupCfg] = MISSING
    """List of joint groups. Each defines joint_names, scale, clip, rescale_to_limits."""


@configclass
class JointPositionVelocityToLimitsActionCfg(ActionTermCfg):
    """Configuration for the bounded joint position and velocity action term.

    See :class:`JointPositionVelocityToLimitsAction` for more details.
    """

    class_type: type[ActionTerm] = JointPositionVelocityToLimitsAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""

    rescale_to_limits: bool = False
    """Whether to rescale the action to the joint limits. Defaults to False.

    If True, the input actions are rescaled to the joint limits, i.e., the action value in
    the range [-1, 1] corresponds to the joint lower and upper limits respectively.

    Note:
        This operation is performed after applying the scale factor.
    """
    num_joints: int = None

    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False.

    If True, the joint order will match the order specified in joint_names.
    This is useful when you need to directly concatenate joint values in a specific order.
    """


##
# Gripper actions.
##


@configclass
class BinaryJointActionCfg(ActionTermCfg):
    """Configuration for the base binary joint action term.

    See :class:`BinaryJointAction` for more details.
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    open_command_expr: dict[str, float] = MISSING
    """The joint command to move to *open* configuration."""
    close_command_expr: dict[str, float] = MISSING
    """The joint command to move to *close* configuration."""


@configclass
class BinaryJointPositionActionCfg(BinaryJointActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = BinaryJointPositionAction


@configclass
class InterpolatedJointActionCfg(ActionTermCfg):
    """Configuration for the base interpolated joint action term.

    Same fields as BinaryJointActionCfg. Action in [0, 1]: 0=open, 1=close,
    values in between linearly interpolate between open_command and close_command.

    See :class:`InterpolatedJointAction` for more details.
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    open_command_expr: dict[str, float] = MISSING
    """The joint command for *open* configuration."""
    close_command_expr: dict[str, float] = MISSING
    """The joint command for *close* configuration."""


@configclass
class InterpolatedJointPositionActionCfg(InterpolatedJointActionCfg):
    """Configuration for interpolated joint position action term.

    See :class:`InterpolatedJointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = InterpolatedJointPositionAction


@configclass
class MultipleInterpolatedJointActionCfg(ActionTermCfg):
    """Configuration for the base multiple interpolated joint action term.

    Each element in joint_groups is an InterpolatedJointActionCfg.
    action_dim = number of joint groups.

    See :class:`MultipleInterpolatedJointAction` for more details.
    """

    joint_groups: list[InterpolatedJointActionCfg] = MISSING
    """List of joint groups. Each defines joint_names, open_command_expr, close_command_expr."""


@configclass
class MultipleInterpolatedJointPositionActionCfg(MultipleInterpolatedJointActionCfg):
    """Configuration for multiple interpolated joint position action term.

    See :class:`MultipleInterpolatedJointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = MultipleInterpolatedJointPositionAction


@configclass
class BinaryJointChoiceActionCfg(ActionTermCfg):
    """Configuration for binary joint choice action term (no interpolation).

    action_dim = 1. action[0] is integer choice: 0=open, 1..n=close configs.

    See :class:`BinaryJointChoiceAction` for more details.
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions."""
    open_command_expr: dict[str, float] = MISSING
    """The joint command for *open* configuration."""
    close_command_exprs: list[dict[str, float]] = MISSING
    """List of close command configs. Each is a dict like {"joint_.*": 0.0}."""


@configclass
class BinaryJointChoicePositionActionCfg(BinaryJointChoiceActionCfg):
    """Configuration for binary joint choice position action term.

    See :class:`BinaryJointChoicePositionAction` for more details.
    """

    class_type: type[ActionTerm] = BinaryJointChoicePositionAction


@configclass
class MultipleBinaryJointChoiceActionCfg(ActionTermCfg):
    """Configuration for multiple binary joint choice action term.

    action_dim = joint_group_num.
    Action layout: [choice_1, choice_2, ...]

    See :class:`MultipleBinaryJointChoiceAction` for more details.
    """

    joint_groups: list[BinaryJointChoiceActionCfg] = MISSING
    """List of joint groups. Each has joint_names, open_command_expr, close_command_exprs."""


@configclass
class MultipleBinaryJointChoicePositionActionCfg(MultipleBinaryJointChoiceActionCfg):
    """Configuration for multiple binary joint choice position action term.

    See :class:`MultipleBinaryJointChoicePositionAction` for more details.
    """

    class_type: type[ActionTerm] = MultipleBinaryJointChoicePositionAction


@configclass
class InterpolatedJointChoiceActionCfg(ActionTermCfg):
    """Configuration for interpolated joint choice action term.

    Allows multiple close configurations. action_dim = 2:
    - action[0] (joint_group_choice): integer index in [0, num_close_configs-1]
    - action[1] (joint_control): [0,1] -> interpolation 0=open, 1=selected close

    See :class:`InterpolatedJointChoiceAction` for more details.
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions."""
    open_command_expr: dict[str, float] = MISSING
    """The joint command for *open* configuration."""
    close_command_exprs: list[dict[str, float]] = MISSING
    """List of close command configs. Each is a dict like {"joint_.*": 0.0}."""


@configclass
class InterpolatedJointChoicePositionActionCfg(InterpolatedJointChoiceActionCfg):
    """Configuration for interpolated joint choice position action term.

    See :class:`InterpolatedJointChoicePositionAction` for more details.
    """

    class_type: type[ActionTerm] = InterpolatedJointChoicePositionAction


@configclass
class MultipleInterpolatedJointChoiceActionCfg(ActionTermCfg):
    """Configuration for multiple interpolated joint choice action term.

    action_dim = 2 * num_joint_groups.
    Action layout: [choice_1, control_1, choice_2, control_2, ...]

    See :class:`MultipleInterpolatedJointChoiceAction` for more details.
    """

    joint_groups: list[InterpolatedJointChoiceActionCfg] = MISSING
    """List of joint groups. Each has joint_names, open_command_expr, close_command_exprs."""


@configclass
class MultipleInterpolatedJointChoicePositionActionCfg(
    MultipleInterpolatedJointChoiceActionCfg
):
    """Configuration for multiple interpolated joint choice position action term.

    See :class:`MultipleInterpolatedJointChoicePositionAction` for more details.
    """

    class_type: type[ActionTerm] = MultipleInterpolatedJointChoicePositionAction


@configclass
class MultipleBinaryJointActionCfg(ActionTermCfg):
    """Configuration for the base multiple binary joint action term.

    This allows independent control of multiple joint groups, where each group
    can be controlled separately with its own binary action.

    Each element in `joint_groups` is a :class:`BinaryJointActionCfg` that defines
    one joint group with its joint names and open/close commands.

    See :class:`MultipleBinaryJointAction` for more details.
    """

    joint_groups: list[BinaryJointActionCfg] = MISSING
    """List of joint groups. Each element is a :class:`BinaryJointActionCfg` that defines
    one joint group with its joint_names, open_command_expr, and close_command_expr.
    The action dimension will be the number of joint groups.
    """


@configclass
class MultipleBinaryJointPositionActionCfg(MultipleBinaryJointActionCfg):
    """Configuration for multiple binary joint position action term.

    See :class:`MultipleBinaryJointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = MultipleBinaryJointPositionAction


@configclass
class DifferentialInverseKinematicsActionCfg(ActionTermCfg):
    """Configuration for inverse differential kinematics action term.

    See :class:`DifferentialInverseKinematicsAction` for more details.
    """

    @configclass
    class OffsetCfg:
        """The offset pose from parent frame to child frame.

        On many robots, end-effector frames are fictitious frames that do not have a corresponding
        rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
        For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
        "panda_hand" frame.
        """

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    class_type: type[ActionTerm] = DifferentialInverseKinematicsAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    body_name: str = MISSING
    """Name of the body or frame for which IK is performed."""
    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""
    scale: float | tuple[float, ...] = 1.0
    """Scale factor for the action. Defaults to 1.0."""
    controller: DifferentialIKControllerCfg = MISSING
    """The configuration for the differential IK controller."""
    action_space: torch.Tensor = MISSING
    """The action space for the action term. Should be a tuple of (low, high) or a list of such tuples.
    If a list is provided, it should have the same length as the number of joints.
    """
    relative_to_base: bool = False
    """Whether the command is already in the robot base frame. If False, commands in world frame
    are converted to base frame before being sent to the controller.
    """
    command_reference_body_name: str | None = None
    """Body to use as reference when converting command to base frame (e.g. arm base like panda_link0).
    If None, use articulation root (e.g. base_link). Set to arm base link so IK target is relative to arm base."""


@configclass
class DualDifferentialInverseKinematicsActionCfg(ActionTermCfg):
    """Configuration for a dual-arm differential IK action term.

    Treats the two arms as independent kinematic chains: each has its own
    ``joint_names``, ``body_name`` and optional ``command_reference_body_name``.
    The input action concatenates ``[right, left]`` per env — matching the
    ``DualManipulator`` frame convention. Per-arm length is the controller's
    ``action_dim`` (3 for ``command_type='position'``, 7 for ``'pose'``).
    """

    OffsetCfg = DifferentialInverseKinematicsActionCfg.OffsetCfg

    class_type: type[ActionTerm] = DualDifferentialInverseKinematicsAction

    right_joint_names: list[str] = MISSING
    left_joint_names: list[str] = MISSING
    right_body_name: str = MISSING
    left_body_name: str = MISSING
    right_command_reference_body_name: str | None = None
    left_command_reference_body_name: str | None = None
    right_body_offset: "DifferentialInverseKinematicsActionCfg.OffsetCfg | None" = None
    left_body_offset: "DifferentialInverseKinematicsActionCfg.OffsetCfg | None" = None
    controller: DifferentialIKControllerCfg = MISSING
    action_space: torch.Tensor = MISSING
    relative_to_base: bool = False
    """If True the command is already in each arm's reference frame."""
    decimation: int = 1
    """IK recompute cadence: solve every N apply_actions calls, apply cached
    joint targets on intermediate steps."""


@configclass
class HolonomicActionCfg(ActionTermCfg):
    """Configuration for differential drive control action term.

    This config works with the DifferentialDriveAction class,
    which wraps Isaac Sim’s DifferentialController to convert
    (linear_vel, angular_vel) commands into wheel joint velocities.
    """

    # Link to your ActionTerm class
    class_type: type[ActionTerm] = HolonomicAction

    # ---- Robot geometry ----
    joint_names: list[str] = MISSING

    # ---- Optional clipping/scaling ----
    clip: dict[str, tuple[float, float]] | None = None
    """Optional action clipping range for velocity commands."""

    scale: float | tuple[float, ...] = 1.0
    """Scale factor for the action. Defaults to 1.0."""
    rescale_to_limits: bool = False
    action_space: torch.Tensor | None = None
    """The action space for the action term. Should be a tuple of (low, high) or a list of such tuples.
    If a list is provided, it should have the same length as the number of joints.
    """


@configclass
class HolonomicVWActionCfg(ActionTermCfg):
    """Configuration for differential drive control action term.

    This config works with the DifferentialDriveAction class,
    which wraps Isaac Sim’s DifferentialController to convert
    (linear_vel, angular_vel) commands into wheel joint velocities.
    """

    # Link to your ActionTerm class
    class_type: type[ActionTerm] = HolonomicVWAction

    # ---- Optional clipping/scaling ----
    clip: dict[str, tuple[float, float]] | None = None
    """Optional action clipping range for velocity commands."""

    scale: float | tuple[float, ...] = 1.0
    """Scale factor for the action. Defaults to 1.0."""
    rescale_to_limits: bool = False
    action_space: torch.Tensor | None = None
    """The action space for the action term. Should be a tuple of (low, high) or a list of such tuples.
    If a list is provided, it should have the same length as the number of joints.
    """
    joint_names: list[str] = MISSING


@configclass
class DifferentialActionCfg(ActionTermCfg):
    """Configuration for differential drive control action term.

    This config works with the DifferentialDriveAction class,
    which wraps Isaac Sim’s DifferentialController to convert
    (linear_vel, angular_vel) commands into wheel joint velocities.
    """

    # Link to your ActionTerm class
    class_type: type[ActionTerm] = DifferentialAction

    # ---- Optional clipping/scaling ----
    clip: dict[str, tuple[float, float]] | None = None
    """Optional action clipping range for velocity commands."""

    scale: float | tuple[float, ...] = 1.0
    """Scale factor for the action. Defaults to 1.0."""
    rescale_to_limits: bool = False
    action_space: torch.Tensor | None = None
    """The action space for the action term. Should be a tuple of (low, high) or a list of such tuples.
    If a list is provided, it should have the same length as the number of joints.
    """
    wheel_radius = 0.2
    wheel_base = 0.8
    joint_names: list[str] = MISSING


@configclass
class AckermannSteeringActionCfg(ActionTermCfg):
    """Ackermann 转向动作配置"""

    class_type: type[ActionTerm] = (
        AckermannSteeringAction  # 需要设置为 AckermannSteeringAction
    )

    # 关节名称
    wheel_joint_names: list[str] = MISSING
    """轮子关节名称（正则表达式列表）"""

    steering_joint_names: list[str] = MISSING
    """转向关节名称（正则表达式列表）"""

    # 车辆参数
    wheel_radius: float = 0.25
    """轮子半径 (m)"""

    wheel_base: float = 1.65
    """轴距：前后轴之间的距离 (m)"""

    track_width: float = 1.25
    """轮距：左右轮之间的距离 (m)"""

    # 控制限制
    max_speed: float = 10.0
    """最大速度 (m/s)"""

    max_steering_angle: float = 0.5
    """最大转向角 (rad)，约 28.6 度"""

    # 动作空间
    action_space: torch.Tensor = MISSING
    """动作空间 [[throttle_min, steering_min], [throttle_max, steering_max]]"""

    # 可选：是否使用 Ackermann 几何
    use_ackermann_geometry: bool = True
    """是否为左右轮计算不同的转向角（内外轮差）"""

    # 可选：裁剪
    clip: dict[str, tuple[float, float]] | None = None
    """动作裁剪范围，如果为 None 则不裁剪"""


@configclass
class HolonomicForQuadrupedActionCfg(ActionTermCfg):
    """Configuration for differential drive control action term for quadruped robots.

    This config works with the HolonomicForQuadrupedAction class,
    which wraps Isaac Sim’s DifferentialController to convert
    (v, ω) commands into wheel joint velocities.
    """

    class_type: type[ActionTerm] = HolonomicForQuadrupedAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    scale: float | tuple[float, ...] = 1.0
    """Scale factor for the action. Defaults to 1.0."""
    rescale_to_limits: bool = False
    action_space: torch.Tensor | None = None
    """The action space for the action term. Should be a tuple of (low, high) or a list of such tuples.
    If a list is provided, it should have the same length as the number of joints.
    """
