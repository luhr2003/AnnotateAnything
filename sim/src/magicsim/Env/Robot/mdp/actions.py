# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from magicsim.Env.Robot.mdp.action_manager import ActionTerm

if TYPE_CHECKING:
    from magicsim.Env.Environment.Isaac.IsaacRLEnv import IsaacRLEnv

    from magicsim.Env.Robot.mdp import actions_cfg

from .differential_ik import DifferentialIKController

from gymnasium import spaces


class JointPositionToLimitsAction(ActionTerm):
    """Joint position action term that scales the input actions to the joint limits and applies them to the
    articulation's joints.

    This class is similar to the :class:`JointPositionAction` class. However, it performs additional
    re-scaling of input actions to the actuator joint position limits.

    While processing the actions, it performs the following operations:

    1. Apply scaling to the raw actions based on :attr:`actions_cfg.JointPositionToLimitsActionCfg.scale`.
    2. Clip the scaled actions to the range [-1, 1] and re-scale them to the joint limits if
       :attr:`actions_cfg.JointPositionToLimitsActionCfg.rescale_to_limits` is set to True.

    The processed actions are then sent as position commands to the articulation's joints.
    """

    cfg: actions_cfg.JointPositionToLimitsActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    _action_space: spaces.Box
    """The action space of the action term."""

    def __init__(
        self, cfg: actions_cfg.JointPositionToLimitsActionCfg, env: IsaacRLEnv
    ):
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        if self.cfg.num_joints is not None:
            assert self._num_joints == self.cfg.num_joints, (
                f"Expected {self.cfg.num_joints} joints, but got {self._num_joints}, {self._joint_names}"
            )
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                self.cfg.scale, self._joint_names
            )
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(
                f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict."
            )
        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self.action_dim, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict."
                )

        if self.cfg.rescale_to_limits:
            self._action_space_raw = torch.stack(
                [
                    torch.tensor([-1.0] * self.action_dim),
                    torch.tensor([1.0] * self.action_dim),
                ],
                dim=1,
            )  # shape (2, action_dim)
            if self.cfg.clip is not None:
                self._action_space_raw = torch.clamp(
                    self._action_space_raw,
                    min=self.cfg.clip[:, 0],
                    max=self.cfg.clip[:, 1],
                )
            self._action_space = spaces.Box(
                low=self._action_space_raw[0].cpu().numpy(),
                high=self._action_space_raw[1].cpu().numpy(),
                dtype=torch.float32,
            )
        else:
            self._action_space_raw = torch.clone(
                self._asset.data.soft_joint_pos_limits[0, self._joint_ids, :]
            )  # shape (2, action_dim)
            if self.cfg.clip is not None:
                self._action_space_raw = torch.clamp(
                    self._action_space_raw,
                    min=self.cfg.clip[:, 0],
                    max=self.cfg.clip[:, 1],
                )

            self._action_space_raw = self._action_space_raw.T
            self._action_space = spaces.Box(
                low=self._action_space_raw[0].cpu().numpy(),
                high=self._action_space_raw[1].cpu().numpy(),
            )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    """
    Operations.
    """

    def process_actions(
        self, actions: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        if env_ids is None:
            selected_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                selected_env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                selected_env_ids = env_ids
        actions, self._env_ids = self._handle_nan_actions(actions, selected_env_ids)
        if actions is None or len(self._env_ids) == 0:
            return
        assert actions.shape[0] == len(self._env_ids), (
            f"Expected actions shape[0] to be {len(self._env_ids)}, but got {actions.shape[0]}"
        )
        # store the raw actions
        self._raw_actions[self.env_ids] = actions
        # apply affine transformations (in-place to keep _processed_actions at (num_envs, dim))
        self._processed_actions[self.env_ids] = (
            self._raw_actions[self.env_ids] * self._scale
        )
        if self.cfg.clip is not None:
            self._processed_actions[self.env_ids] = torch.clamp(
                self._processed_actions[self.env_ids],
                min=self._clip[self.env_ids, :, 0],
                max=self._clip[self.env_ids, :, 1],
            )
        # rescale the position targets if configured
        # this is useful when the input actions are in the range [-1, 1]
        if self.cfg.rescale_to_limits:
            # clip to [-1, 1]
            actions = self._processed_actions[self.env_ids].clamp(-1.0, 1.0)
            # rescale within the joint limits
            actions = math_utils.unscale_transform(
                actions,
                self._asset.data.soft_joint_pos_limits[
                    self.env_ids.unsqueeze(1), self._joint_ids, 0
                ],
                self._asset.data.soft_joint_pos_limits[
                    self.env_ids.unsqueeze(1), self._joint_ids, 1
                ],
            )
            self._processed_actions[self.env_ids] = actions
        else:
            joint_limits_min = self._asset.data.joint_pos_limits[
                self.env_ids.unsqueeze(1), self._joint_ids, 0
            ]
            joint_limits_max = self._asset.data.joint_pos_limits[
                self.env_ids.unsqueeze(1), self._joint_ids, 1
            ]
            self._processed_actions[self.env_ids] = torch.clamp(
                self._processed_actions[self.env_ids],
                min=joint_limits_min,
                max=joint_limits_max,
            )

    def apply_actions(self):
        # set position targets
        assert self._processed_actions.shape[0] == self.num_envs, (
            f"Expected _processed_actions shape[0] to be {self.num_envs}, but got {self._processed_actions.shape[0]}"
        )
        assert self._processed_actions.shape[1] == self._num_joints, (
            f"Expected processed actions shape[1] to be {self._num_joints}, but got {self.processed_actions.shape[1]}"
        )

        # # DEBUG: 打印发送的位置指令
        # print(f"\n=== JointPositionToLimitsAction.apply_actions DEBUG ===")
        # print(f"  joint_names: {self._joint_names}")
        # print(f"  joint_ids: {self._joint_ids}")
        # print(f"  env_ids: {self.env_ids}")
        # print(f"  processed_actions shape: {self.processed_actions.shape}")
        # print(f"  processed_actions (env 0): {self.processed_actions[0].cpu().numpy()}")

        # # 获取当前关节位置用于对比
        # current_joint_pos = self._asset.data.joint_pos[self.env_ids.unsqueeze(1), self._joint_ids]
        # print(f"  current_joint_pos (env 0): {current_joint_pos[0].cpu().numpy()}")
        # print(f"  position error (env 0): {(self.processed_actions[0] - current_joint_pos[0]).cpu().numpy()}")

        self._asset.set_joint_position_target(
            self._processed_actions[self.env_ids],
            joint_ids=self._joint_ids,
            env_ids=self.env_ids,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass
        self._raw_actions[env_ids] = 0.0


class JointPositionVelocityToLimitsAction(ActionTerm):
    """Joint position and velocity action term that scales the input actions to the joint limits and applies them to the
    articulation's joints.

    This class is similar to the :class:`JointPositionToLimitsAction` class. However, it handles both position and velocity
    actions simultaneously. The input actions are split into two parts: the first half represents position actions and
    the second half represents velocity actions.

    While processing the actions, it performs the following operations:

    1. Split the input actions into position and velocity parts.
    2. Apply scaling to the raw actions based on :attr:`actions_cfg.JointPositionVelocityToLimitsActionCfg.scale`.
    3. Clip the scaled actions and re-scale them to the joint limits if
       :attr:`actions_cfg.JointPositionVelocityToLimitsActionCfg.rescale_to_limits` is set to True.

    The processed actions are then sent as position and velocity commands to the articulation's joints.
    """

    cfg: actions_cfg.JointPositionVelocityToLimitsActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    _action_space: spaces.Box
    """The action space of the action term."""

    def __init__(
        self, cfg: actions_cfg.JointPositionVelocityToLimitsActionCfg, env: IsaacRLEnv
    ):
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        if self.cfg.num_joints is not None:
            assert self._num_joints == self.cfg.num_joints, (
                f"Expected {self.cfg.num_joints} joints, but got {self._num_joints}, {self._joint_names}"
            )
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # create tensors for raw and processed actions
        # action_dim is 2 * num_joints (positions + velocities)
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_pos_actions = torch.zeros(
            self.num_envs, self._num_joints, device=self.device
        )
        self._processed_vel_actions = torch.zeros(
            self.num_envs, self._num_joints, device=self.device
        )

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                self.cfg.scale, self._joint_names
            )
            # apply to both position and velocity parts
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
            self._scale[:, index_list + self._num_joints] = torch.tensor(
                value_list, device=self.device
            )
        else:
            raise ValueError(
                f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict."
            )
        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self.action_dim, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                # apply to both position and velocity parts
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
                self._clip[:, index_list + self._num_joints] = torch.tensor(
                    value_list, device=self.device
                )
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict."
                )

        if self.cfg.rescale_to_limits:
            self._action_space_raw = torch.stack(
                [
                    torch.tensor([-1.0] * self.action_dim),
                    torch.tensor([1.0] * self.action_dim),
                ],
                dim=1,
            )  # shape (2, action_dim)
            if self.cfg.clip is not None:
                self._action_space_raw = torch.clamp(
                    self._action_space_raw,
                    min=self.cfg.clip[:, 0],
                    max=self.cfg.clip[:, 1],
                )
            self._action_space = spaces.Box(
                low=self._action_space_raw[0].cpu().numpy(),
                high=self._action_space_raw[1].cpu().numpy(),
                dtype=torch.float32,
            )
        else:
            # Position limits: shape (2, num_joints)
            pos_limits = torch.clone(
                self._asset.data.soft_joint_pos_limits[0, self._joint_ids, :]
            ).T  # shape (2, num_joints)
            # Velocity limits: shape (2, num_joints) - symmetric around zero
            vel_limits = self._asset.data.soft_joint_vel_limits[
                0, self._joint_ids
            ]  # shape (num_joints,)
            vel_limits = torch.stack(
                [-vel_limits, vel_limits], dim=0
            )  # shape (2, num_joints)
            # Concatenate position and velocity limits
            self._action_space_raw = torch.cat(
                [pos_limits, vel_limits], dim=1
            )  # shape (2, action_dim)
            if self.cfg.clip is not None:
                self._action_space_raw = torch.clamp(
                    self._action_space_raw,
                    min=self.cfg.clip[:, 0],
                    max=self.cfg.clip[:, 1],
                )
            self._action_space = spaces.Box(
                low=self._action_space_raw[0].cpu().numpy(),
                high=self._action_space_raw[1].cpu().numpy(),
            )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2 * self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return torch.cat(
            [self._processed_pos_actions, self._processed_vel_actions], dim=1
        )

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    """
    Operations.
    """

    def process_actions(
        self, actions: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        if env_ids is None:
            selected_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                selected_env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                selected_env_ids = env_ids
        actions, self._env_ids = self._handle_nan_actions(actions, selected_env_ids)
        if actions is None or len(self._env_ids) == 0:
            return
        assert actions.shape[0] == len(self._env_ids), (
            f"Expected actions shape[0] to be {len(self._env_ids)}, but got {actions.shape[0]}"
        )
        assert actions.shape[1] == self.action_dim, (
            f"Expected actions shape[1] to be {self.action_dim}, but got {actions.shape[1]}"
        )
        # store the raw actions
        self._raw_actions[self.env_ids] = actions

        # split into position and velocity parts
        pos_actions = actions[:, : self._num_joints]
        vel_actions = actions[:, self._num_joints :]

        # apply scaling
        if isinstance(self._scale, float):
            pos_actions_scaled = pos_actions * self._scale
            vel_actions_scaled = vel_actions * self._scale
        else:
            pos_scale = self._scale[self.env_ids, : self._num_joints]
            vel_scale = self._scale[self.env_ids, self._num_joints :]
            pos_actions_scaled = pos_actions * pos_scale
            vel_actions_scaled = vel_actions * vel_scale

        # apply clip if configured
        if self.cfg.clip is not None:
            pos_clip = self._clip[self.env_ids, : self._num_joints, :]
            vel_clip = self._clip[self.env_ids, self._num_joints :, :]
            pos_actions_scaled = torch.clamp(
                pos_actions_scaled,
                min=pos_clip[:, :, 0],
                max=pos_clip[:, :, 1],
            )
            vel_actions_scaled = torch.clamp(
                vel_actions_scaled,
                min=vel_clip[:, :, 0],
                max=vel_clip[:, :, 1],
            )

        # process position actions
        if self.cfg.rescale_to_limits:
            # clip to [-1, 1]
            pos_actions_scaled = pos_actions_scaled.clamp(-1.0, 1.0)
            # rescale within the joint limits
            pos_actions = math_utils.unscale_transform(
                pos_actions_scaled,
                self._asset.data.soft_joint_pos_limits[
                    self.env_ids.unsqueeze(1), self._joint_ids, 0
                ],
                self._asset.data.soft_joint_pos_limits[
                    self.env_ids.unsqueeze(1), self._joint_ids, 1
                ],
            )
            self._processed_pos_actions[self.env_ids] = pos_actions
        else:
            joint_limits_min = self._asset.data.joint_pos_limits[
                self.env_ids.unsqueeze(1), self._joint_ids, 0
            ]
            joint_limits_max = self._asset.data.joint_pos_limits[
                self.env_ids.unsqueeze(1), self._joint_ids, 1
            ]
            self._processed_pos_actions[self.env_ids] = torch.clamp(
                pos_actions_scaled,
                min=joint_limits_min,
                max=joint_limits_max,
            )

        # process velocity actions
        if self.cfg.rescale_to_limits:
            # clip to [-1, 1]
            vel_actions_scaled = vel_actions_scaled.clamp(-1.0, 1.0)
            # rescale within the joint velocity limits (symmetric around zero)
            vel_limits = self._asset.data.soft_joint_vel_limits[
                self.env_ids.unsqueeze(1), self._joint_ids
            ]  # shape (num_envs, num_joints)
            self._processed_vel_actions[self.env_ids] = vel_actions_scaled * vel_limits
        else:
            vel_limits = self._asset.data.soft_joint_vel_limits[
                self.env_ids.unsqueeze(1), self._joint_ids
            ]  # shape (num_envs, num_joints)
            self._processed_vel_actions[self.env_ids] = torch.clamp(
                vel_actions_scaled,
                min=-vel_limits,
                max=vel_limits,
            )

    def apply_actions(self):
        # set position and velocity targets
        assert self._processed_pos_actions.shape[0] >= len(self.env_ids), (
            f"Expected processed pos actions shape[0] to be at least {len(self.env_ids)}, "
            f"but got {self._processed_pos_actions.shape[0]}"
        )
        assert self._processed_vel_actions.shape[0] >= len(self.env_ids), (
            f"Expected processed vel actions shape[0] to be at least {len(self.env_ids)}, "
            f"but got {self._processed_vel_actions.shape[0]}"
        )
        assert self._processed_pos_actions.shape[1] == self._num_joints, (
            f"Expected processed pos actions shape[1] to be {self._num_joints}, "
            f"but got {self._processed_pos_actions.shape[1]}"
        )
        assert self._processed_vel_actions.shape[1] == self._num_joints, (
            f"Expected processed vel actions shape[1] to be {self._num_joints}, "
            f"but got {self._processed_vel_actions.shape[1]}"
        )
        self._asset.set_joint_position_target(
            self._processed_pos_actions[self.env_ids],
            joint_ids=self._joint_ids,
            env_ids=self.env_ids,
        )
        self._asset.set_joint_velocity_target(
            self._processed_vel_actions[self.env_ids],
            joint_ids=self._joint_ids,
            env_ids=self.env_ids,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.int32)
        self._raw_actions[env_ids] = 0.0
        self._processed_pos_actions[env_ids] = 0.0
        self._processed_vel_actions[env_ids] = 0.0


class BinaryJointAction(ActionTerm):
    """Base class for binary joint actions.

    This action term maps a binary action to the *open* or *close* joint configurations. These configurations are
    specified through the :class:`BinaryJointActionCfg` object. If the input action is a float vector, the action
    is considered binary based on the sign of the action values.

    Based on above, we follow the following convention for the binary action:

    1. Open action: 1 (bool) or positive values (float).
    2. Close action: 0 (bool) or negative values (float).

    The action term can mostly be used for gripper actions, where the gripper is either open or closed. This
    helps in devising a mimicking mechanism for the gripper, since in simulation it is often not possible to
    add such constraints to the gripper.
    """

    cfg: actions_cfg.BinaryJointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    _action_space: spaces.Box
    """The action space of the action term."""

    def __init__(self, cfg: actions_cfg.BinaryJointActionCfg, env: IsaacRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names
        )
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        # _processed_actions stores joint position targets, shape (num_envs, _num_joints)
        self._processed_actions = torch.zeros(
            self.num_envs, self._num_joints, device=self.device
        )

        # parse open command
        self._open_command = torch.zeros(self._num_joints, device=self.device)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.open_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(
                f"Could not resolve all joints for the action term. Missing: {set(self._joint_names) - set(name_list)}"
            )
        self._open_command[index_list] = torch.tensor(value_list, device=self.device)

        # parse close command
        self._close_command = torch.zeros_like(self._open_command)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.close_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(
                f"Could not resolve all joints for the action term. Missing: {set(self._joint_names) - set(name_list)}"
            )
        self._close_command[index_list] = torch.tensor(value_list, device=self.device)

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self._num_joints, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict."
                )

        self._action_space = spaces.Discrete(2)  # 1 close 0 open

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    """
    Operations.
    """

    def process_actions(
        self, actions: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        if env_ids is None:
            selected_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                selected_env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                selected_env_ids = env_ids
        actions, self._env_ids = self._handle_nan_actions(actions, selected_env_ids)
        if actions is None or len(self._env_ids) == 0:
            return
        assert actions.shape[0] == self._env_ids.shape[0], (
            f"Expected actions shape[0] to be {len(self._env_ids)}, but got {actions.shape[0]}"
        )
        # store the raw actions
        self._raw_actions[self.env_ids] = actions
        # compute the binary mask
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 1
        else:
            # true: close, false: open
            binary_mask = actions >= 1
        # compute the command (in-place to keep _processed_actions at (num_envs, dim))
        self._processed_actions[self.env_ids] = torch.where(
            binary_mask, self._close_command, self._open_command
        )
        if self.cfg.clip is not None:
            self._processed_actions[self.env_ids] = torch.clamp(
                self._processed_actions[self.env_ids],
                min=self._clip[self.env_ids, :, 0],
                max=self._clip[self.env_ids, :, 1],
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class BinaryJointPositionAction(BinaryJointAction):
    """Binary joint action that sets the binary action into joint position targets."""

    cfg: actions_cfg.BinaryJointPositionActionCfg
    """The configuration of the action term."""

    def apply_actions(self):
        if len(self.env_ids) == 0:
            return
        assert self._processed_actions.shape[0] == self.num_envs, (
            f"Expected _processed_actions shape[0] to be {self.num_envs}, but got {self._processed_actions.shape[0]}"
        )

        self._asset.set_joint_position_target(
            self._processed_actions[self.env_ids],
            joint_ids=self._joint_ids,
            env_ids=self.env_ids,
        )


class InterpolatedJointAction(ActionTerm):
    """Base class for interpolated joint actions.

    Linearly interpolates between open and close commands based on action value in [0, 1]:
    - 0 = open (open_command)
    - 1 = close (close_command)
    - Values in between = linear interpolation: open + t * (close - open)

    action_dim = 1, same config as BinaryJointActionCfg.
    """

    cfg: actions_cfg.InterpolatedJointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _clip: torch.Tensor | None
    """The clip applied to the interpolated joint positions (optional)."""

    def __init__(
        self, cfg: actions_cfg.InterpolatedJointActionCfg, env: IsaacRLEnv
    ) -> None:
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names
        )
        self._num_joints = len(self._joint_ids)
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros(
            self.num_envs, self._num_joints, device=self.device
        )

        # parse open command
        self._open_command = torch.zeros(self._num_joints, device=self.device)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.open_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(
                f"Could not resolve all joints for open. Missing: {set(self._joint_names) - set(name_list)}"
            )
        self._open_command[index_list] = torch.tensor(value_list, device=self.device)

        # parse close command
        self._close_command = torch.zeros_like(self._open_command)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.close_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(
                f"Could not resolve all joints for close. Missing: {set(self._joint_names) - set(name_list)}"
            )
        self._close_command[index_list] = torch.tensor(value_list, device=self.device)

        # parse clip
        self._clip = None
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self._num_joints, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported: dict."
                )

        self._action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=torch.float32
        )

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def process_actions(
        self, actions: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        if env_ids is None:
            selected_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                selected_env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                selected_env_ids = env_ids
        actions, self._env_ids = self._handle_nan_actions(actions, selected_env_ids)
        if actions is None or len(self._env_ids) == 0:
            return
        assert actions.shape[0] == self._env_ids.shape[0], (
            f"Expected actions shape[0] to be {len(self._env_ids)}, got {actions.shape[0]}"
        )

        self._raw_actions[self.env_ids] = actions
        # t in [0, 1]: 0=open, 1=close
        t = actions[:, 0:1].clamp(0.0, 1.0)
        # linear interpolation: open + t * (close - open)
        interpolated = self._open_command + t * (
            self._close_command - self._open_command
        )
        self._processed_actions[self.env_ids] = interpolated

        if self._clip is not None:
            self._processed_actions[self.env_ids] = torch.clamp(
                self._processed_actions[self.env_ids],
                min=self._clip[self.env_ids, :, 0],
                max=self._clip[self.env_ids, :, 1],
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class InterpolatedJointPositionAction(InterpolatedJointAction):
    """Interpolated joint action that sets the interpolated action into joint position targets."""

    cfg: actions_cfg.InterpolatedJointPositionActionCfg
    """The configuration of the action term."""

    def apply_actions(self):
        self._asset.set_joint_position_target(
            self._processed_actions[self.env_ids],
            joint_ids=self._joint_ids,
            env_ids=self.env_ids,
        )


class MultipleInterpolatedJointAction(ActionTerm):
    """Base class for multiple interpolated joint actions.

    Each joint group has its own action in [0, 1] that linearly interpolates
    between open and close for that group. action_dim = num_joint_groups.
    """

    cfg: actions_cfg.MultipleInterpolatedJointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _joint_group_configs: list[dict]
    """List of joint group configs: joint_ids, joint_names, open_command, close_command."""

    def __init__(
        self, cfg: actions_cfg.MultipleInterpolatedJointActionCfg, env: IsaacRLEnv
    ) -> None:
        super().__init__(cfg, env)

        self._joint_group_configs = []
        self._num_joint_groups = len(cfg.joint_groups)

        for joint_group_cfg in cfg.joint_groups:
            joint_ids, joint_names = self._asset.find_joints(
                joint_group_cfg.joint_names
            )
            num_joints = len(joint_ids)

            omni.log.info(
                f"Resolved joint names for joint group: {joint_names} [{joint_ids}]"
            )

            open_command = torch.zeros(num_joints, device=self.device)
            index_list, name_list, value_list = (
                string_utils.resolve_matching_names_values(
                    joint_group_cfg.open_command_expr, joint_names
                )
            )
            if len(index_list) != num_joints:
                raise ValueError(
                    f"Could not resolve all joints for joint group. Missing: {set(joint_names) - set(name_list)}"
                )
            open_command[index_list] = torch.tensor(value_list, device=self.device)

            close_command = torch.zeros_like(open_command)
            index_list, name_list, value_list = (
                string_utils.resolve_matching_names_values(
                    joint_group_cfg.close_command_expr, joint_names
                )
            )
            if len(index_list) != num_joints:
                raise ValueError(
                    f"Could not resolve all joints for joint group. Missing: {set(joint_names) - set(name_list)}"
                )
            close_command[index_list] = torch.tensor(value_list, device=self.device)

            self._joint_group_configs.append(
                {
                    "joint_ids": joint_ids,
                    "joint_names": joint_names,
                    "open_command": open_command,
                    "close_command": close_command,
                }
            )

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = [
            torch.zeros(self.num_envs, len(jgc["joint_ids"]), device=self.device)
            for jgc in self._joint_group_configs
        ]

        self._env_ids = torch.arange(self.num_envs, device=self.device)
        self._action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._num_joint_groups,),
            dtype=torch.float32,
        )

    @property
    def action_dim(self) -> int:
        return self._num_joint_groups

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> list[torch.Tensor]:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def process_actions(
        self, actions: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        if env_ids is None:
            selected_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                selected_env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                selected_env_ids = env_ids
        actions, self._env_ids = self._handle_nan_actions(actions, selected_env_ids)
        if actions is None or len(self._env_ids) == 0:
            return

        assert actions.shape[0] == self._env_ids.shape[0], (
            f"Expected actions shape[0] to be {self._env_ids.shape[0]}, got {actions.shape[0]}"
        )
        assert actions.shape[1] == self._num_joint_groups, (
            f"Expected actions shape[1] to be {self._num_joint_groups}, got {actions.shape[1]}"
        )

        self._raw_actions[self.env_ids] = actions

        for joint_group_idx, joint_group_cfg in enumerate(self._joint_group_configs):
            t = actions[:, joint_group_idx : joint_group_idx + 1].clamp(0.0, 1.0)
            interpolated = joint_group_cfg["open_command"] + t * (
                joint_group_cfg["close_command"] - joint_group_cfg["open_command"]
            )
            self._processed_actions[joint_group_idx][self.env_ids] = interpolated

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.int32)

        self._raw_actions[env_ids] = 0.0
        for joint_group_idx, joint_group_cfg in enumerate(self._joint_group_configs):
            self._processed_actions[joint_group_idx][env_ids] = (
                joint_group_cfg["open_command"].unsqueeze(0).expand(len(env_ids), -1)
            )


class MultipleInterpolatedJointPositionAction(MultipleInterpolatedJointAction):
    """Multiple interpolated joint action that sets into joint position targets."""

    cfg: actions_cfg.MultipleInterpolatedJointPositionActionCfg
    """The configuration of the action term."""

    def apply_actions(self):
        for joint_group_idx, joint_group_cfg in enumerate(self._joint_group_configs):
            processed_actions_for_envs = self._processed_actions[joint_group_idx][
                self.env_ids
            ]
            assert processed_actions_for_envs.shape[0] == len(self.env_ids), (
                f"Expected processed actions shape[0] to be {len(self.env_ids)}, "
                f"but got {processed_actions_for_envs.shape[0]}"
            )
            self._asset.set_joint_position_target(
                processed_actions_for_envs,
                joint_ids=joint_group_cfg["joint_ids"],
                env_ids=self.env_ids,
            )


class MultipleJointPositionToLimitsAction(ActionTerm):
    """Multiple joint position action term with per-group limits and group-wise NaN handling.

    Each joint group has its own scale, clip, rescale_to_limits (like JointPositionToLimitsAction).
    action_dim = sum of num_joints across all groups.
    NaN handling is per-group (like pink_task_space_actions): if a group has any NaN,
    fill that group with current joint positions for that group's joints.
    """

    cfg: actions_cfg.MultipleJointPositionToLimitsActionCfg
    _asset: Articulation
    _joint_group_configs: list[dict]

    def __init__(
        self,
        cfg: actions_cfg.MultipleJointPositionToLimitsActionCfg,
        env: IsaacRLEnv,
    ) -> None:
        super().__init__(cfg, env)

        self._joint_group_configs = []
        action_dim_total = 0

        for group_cfg in cfg.joint_groups:
            joint_ids, joint_names = self._asset.find_joints(
                group_cfg.joint_names, preserve_order=group_cfg.preserve_order
            )
            num_joints = len(joint_ids)
            if group_cfg.num_joints is not None:
                assert num_joints == group_cfg.num_joints, (
                    f"Group expected {group_cfg.num_joints} joints, "
                    f"got {num_joints}, {joint_names}"
                )

            omni.log.info(
                f"MultipleJointPositionToLimitsAction group: {joint_names} [{joint_ids}]"
            )

            # Scale
            if isinstance(group_cfg.scale, (float, int)):
                scale = float(group_cfg.scale)
            elif isinstance(group_cfg.scale, dict):
                scale = torch.ones(num_joints, device=self.device)
                idx, _, vals = string_utils.resolve_matching_names_values(
                    group_cfg.scale, joint_names
                )
                scale[idx] = torch.tensor(vals, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported scale type {type(group_cfg.scale)} for group"
                )

            # Clip
            clip = None
            if group_cfg.clip is not None:
                clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, num_joints, 1)
                idx, _, vals = string_utils.resolve_matching_names_values(
                    group_cfg.clip, joint_names
                )
                clip[:, idx] = torch.tensor(vals, device=self.device)

            # Action space for this group
            if group_cfg.rescale_to_limits:
                low = torch.tensor([-1.0] * num_joints, device=self.device)
                high = torch.tensor([1.0] * num_joints, device=self.device)
                if clip is not None:
                    low = torch.clamp(low, min=clip[0, :, 0], max=clip[0, :, 1])
                    high = torch.clamp(high, min=clip[0, :, 0], max=clip[0, :, 1])
            else:
                limits = self._asset.data.soft_joint_pos_limits[0, joint_ids, :].T
                low, high = limits[0], limits[1]
                if clip is not None:
                    low = torch.clamp(low, min=clip[0, :, 0], max=clip[0, :, 1])
                    high = torch.clamp(high, min=clip[0, :, 0], max=clip[0, :, 1])

            self._joint_group_configs.append(
                {
                    "joint_ids": joint_ids,
                    "joint_names": joint_names,
                    "num_joints": num_joints,
                    "scale": scale,
                    "clip": clip,
                    "rescale_to_limits": group_cfg.rescale_to_limits,
                    "action_start": action_dim_total,
                    "action_end": action_dim_total + num_joints,
                }
            )
            action_dim_total += num_joints

        self._action_dim_total = action_dim_total
        self._raw_actions = torch.zeros(
            self.num_envs, action_dim_total, device=self.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._env_ids = torch.arange(self.num_envs, device=self.device)

        lows = []
        highs = []
        for jgc in self._joint_group_configs:
            if jgc["rescale_to_limits"]:
                n = jgc["num_joints"]
                lows.append(torch.tensor([-1.0] * n, device=self.device))
                highs.append(torch.tensor([1.0] * n, device=self.device))
            else:
                lim = self._asset.data.soft_joint_pos_limits[0, jgc["joint_ids"], :].T
                lows.append(lim[0])
                highs.append(lim[1])
        self._action_space = spaces.Box(
            low=torch.cat(lows).cpu().numpy(),
            high=torch.cat(highs).cpu().numpy(),
            dtype=torch.cat(lows).cpu().numpy().dtype,
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim_total

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def _handle_nan_actions(
        self, actions: torch.Tensor, selected_env_ids: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Handle NaN by group: skip all-NaN envs; fill per-group NaN with current joint pos."""
        all_nan_mask = torch.isnan(actions).all(dim=1)
        valid_mask = ~all_nan_mask
        valid_indices = torch.where(valid_mask)[0]
        env_ids = selected_env_ids[valid_indices]

        if len(env_ids) == 0:
            return None, env_ids

        valid_actions = actions[valid_indices].clone()

        for jgc in self._joint_group_configs:
            start, end = jgc["action_start"], jgc["action_end"]
            group_nan_mask = torch.isnan(valid_actions[:, start:end]).any(dim=1)
            if group_nan_mask.any():
                nan_env_ids = env_ids[group_nan_mask]
                current_pos = self._asset.data.joint_pos[nan_env_ids][
                    :, jgc["joint_ids"]
                ]
                valid_actions[group_nan_mask, start:end] = current_pos

        return valid_actions, env_ids

    def process_actions(
        self, actions: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        if env_ids is None:
            selected_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            selected_env_ids = (
                env_ids
                if isinstance(env_ids, torch.Tensor)
                else torch.tensor(env_ids, device=self.device, dtype=torch.int32)
            )

        actions, self._env_ids = self._handle_nan_actions(actions, selected_env_ids)
        if actions is None or len(self._env_ids) == 0:
            return

        assert actions.shape[0] == len(self._env_ids)
        assert actions.shape[1] == self.action_dim

        self._raw_actions[self.env_ids] = actions
        self._processed_actions[self.env_ids] = self._raw_actions[self.env_ids].clone()

        for jgc in self._joint_group_configs:
            start, end = jgc["action_start"], jgc["action_end"]
            scale = jgc["scale"]
            clip = jgc["clip"]
            rescale = jgc["rescale_to_limits"]

            vals = self._processed_actions[self.env_ids, start:end] * scale
            if clip is not None:
                vals = torch.clamp(
                    vals,
                    min=clip[self.env_ids, :, 0],
                    max=clip[self.env_ids, :, 1],
                )

            if rescale:
                vals = vals.clamp(-1.0, 1.0)
                vals = math_utils.unscale_transform(
                    vals,
                    self._asset.data.soft_joint_pos_limits[
                        self.env_ids.unsqueeze(1), jgc["joint_ids"], 0
                    ],
                    self._asset.data.soft_joint_pos_limits[
                        self.env_ids.unsqueeze(1), jgc["joint_ids"], 1
                    ],
                )
            else:
                lim_min = self._asset.data.joint_pos_limits[
                    self.env_ids.unsqueeze(1), jgc["joint_ids"], 0
                ]
                lim_max = self._asset.data.joint_pos_limits[
                    self.env_ids.unsqueeze(1), jgc["joint_ids"], 1
                ]
                vals = torch.clamp(vals, min=lim_min, max=lim_max)

            self._processed_actions[self.env_ids, start:end] = vals

    def apply_actions(self):
        for jgc in self._joint_group_configs:
            start, end = jgc["action_start"], jgc["action_end"]
            self._asset.set_joint_position_target(
                self._processed_actions[self.env_ids, start:end],
                joint_ids=jgc["joint_ids"],
                env_ids=self.env_ids,
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.int32)
        self._raw_actions[env_ids] = 0.0


class BinaryJointChoiceAction(ActionTerm):
    """Base class for binary joint choice actions (no interpolation).

    Selects one among multiple close configurations. No interpolation - directly
    outputs the selected config. Useful when you need discrete grip presets.

    action_dim = 1:
    - action[0]: integer choice index in [0, num_options-1].
      choice 0 = open, choice 1..n = close_configs[0..n-1].
      num_options = 1 + len(close_command_exprs).

    Command: open (if choice=0) else close[choice_idx-1]
    """

    cfg: actions_cfg.BinaryJointChoiceActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _clip: torch.Tensor | None
    """The clip applied to the joint positions (optional)."""

    def __init__(
        self, cfg: actions_cfg.BinaryJointChoiceActionCfg, env: IsaacRLEnv
    ) -> None:
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names
        )
        self._num_joints = len(self._joint_ids)
        self._num_close_configs = len(self.cfg.close_command_exprs)
        if self._num_close_configs < 1:
            raise ValueError(
                "BinaryJointChoiceAction requires at least one close_command_expr"
            )
        self._num_options = 1 + self._num_close_configs  # open + close configs

        omni.log.info(
            f"Resolved joint names for {self.__class__.__name__}: "
            f"{self._joint_names} [{self._joint_ids}], "
            f"num_options={self._num_options} (1 open + {self._num_close_configs} close)"
        )

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros(
            self.num_envs, self._num_joints, device=self.device
        )

        # parse open command
        self._open_command = torch.zeros(self._num_joints, device=self.device)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.open_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(
                f"Could not resolve all joints for open. Missing: {set(self._joint_names) - set(name_list)}"
            )
        self._open_command[index_list] = torch.tensor(value_list, device=self.device)

        # parse close commands
        self._close_commands = []
        for close_expr in self.cfg.close_command_exprs:
            close_cmd = torch.zeros(self._num_joints, device=self.device)
            index_list, name_list, value_list = (
                string_utils.resolve_matching_names_values(
                    close_expr, self._joint_names
                )
            )
            if len(index_list) != self._num_joints:
                raise ValueError(
                    f"Could not resolve all joints for close. Missing: {set(self._joint_names) - set(name_list)}"
                )
            close_cmd[index_list] = torch.tensor(value_list, device=self.device)
            self._close_commands.append(close_cmd)

        # all options: [open, close_0, close_1, ...]
        self._all_commands = [self._open_command] + self._close_commands

        # parse clip
        self._clip = None
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self._num_joints, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported: dict."
                )

        self._action_space = spaces.Box(
            low=0.0, high=float(self._num_options - 1), shape=(1,), dtype=torch.float32
        )

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def process_actions(
        self, actions: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        if env_ids is None:
            selected_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                selected_env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                selected_env_ids = env_ids
        actions, self._env_ids = self._handle_nan_actions(actions, selected_env_ids)
        if actions is None or len(self._env_ids) == 0:
            return
        assert actions.shape[0] == self._env_ids.shape[0], (
            f"Expected actions shape[0] to be {len(self._env_ids)}, got {actions.shape[0]}"
        )
        assert actions.shape[1] >= 1, (
            f"Expected actions shape[1] >= 1, got {actions.shape[1]}"
        )

        self._raw_actions[self.env_ids] = actions[:, :1]

        # choice: integer index in [0, num_options-1], 0=open, 1..n=close configs
        choice_idx = actions[:, 0:1].long().clamp(0, self._num_options - 1)

        # select command for each env
        all_stacked = torch.stack(self._all_commands, dim=0)
        selected = all_stacked[choice_idx.squeeze(1)]
        if selected.dim() == 1:
            selected = selected.unsqueeze(0)
        self._processed_actions[self.env_ids] = selected

        if self._clip is not None:
            self._processed_actions[self.env_ids] = torch.clamp(
                self._processed_actions[self.env_ids],
                min=self._clip[self.env_ids, :, 0],
                max=self._clip[self.env_ids, :, 1],
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.int32)
        self._raw_actions[env_ids, 0] = 0.0


class BinaryJointChoicePositionAction(BinaryJointChoiceAction):
    """Binary joint choice action that sets into joint position targets."""

    cfg: actions_cfg.BinaryJointChoicePositionActionCfg
    """The configuration of the action term."""

    def apply_actions(self):
        self._asset.set_joint_position_target(
            self._processed_actions[self.env_ids],
            joint_ids=self._joint_ids,
            env_ids=self.env_ids,
        )


class MultipleBinaryJointChoiceAction(ActionTerm):
    """Base class for multiple binary joint choice actions (no interpolation).

    Each joint group has 1 action: which config to use.
    action_dim = joint_group_num.

    Action layout: [joint_group_choice_1, joint_group_choice_2, ...]
    - joint_group_choice_i: integer index in [0, num_options_i-1], selects config for group i
      (0=open, 1..n=close configs for that group).
    """

    cfg: actions_cfg.MultipleBinaryJointChoiceActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _joint_group_configs: list[dict]
    """List of configs: joint_ids, joint_names, open_command, close_commands, all_commands."""

    def __init__(
        self, cfg: actions_cfg.MultipleBinaryJointChoiceActionCfg, env: IsaacRLEnv
    ) -> None:
        super().__init__(cfg, env)

        self._joint_group_configs = []
        self._num_joint_groups = len(cfg.joint_groups)

        for joint_group_cfg in cfg.joint_groups:
            joint_ids, joint_names = self._asset.find_joints(
                joint_group_cfg.joint_names
            )
            num_joints = len(joint_ids)
            num_close = len(joint_group_cfg.close_command_exprs)
            if num_close < 1:
                raise ValueError(
                    f"Joint group requires at least one close_command_expr, got {num_close}"
                )
            num_options = 1 + num_close

            omni.log.info(
                f"Resolved joint group: {joint_names} [{joint_ids}], "
                f"num_options={num_options}"
            )

            open_command = torch.zeros(num_joints, device=self.device)
            index_list, name_list, value_list = (
                string_utils.resolve_matching_names_values(
                    joint_group_cfg.open_command_expr, joint_names
                )
            )
            if len(index_list) != num_joints:
                raise ValueError(
                    f"Could not resolve all joints. Missing: {set(joint_names) - set(name_list)}"
                )
            open_command[index_list] = torch.tensor(value_list, device=self.device)

            close_commands = []
            for close_expr in joint_group_cfg.close_command_exprs:
                close_cmd = torch.zeros(num_joints, device=self.device)
                index_list, name_list, value_list = (
                    string_utils.resolve_matching_names_values(close_expr, joint_names)
                )
                if len(index_list) != num_joints:
                    raise ValueError(
                        f"Could not resolve all joints for close. Missing: {set(joint_names) - set(name_list)}"
                    )
                close_cmd[index_list] = torch.tensor(value_list, device=self.device)
                close_commands.append(close_cmd)

            all_commands = [open_command] + close_commands

            self._joint_group_configs.append(
                {
                    "joint_ids": joint_ids,
                    "joint_names": joint_names,
                    "open_command": open_command,
                    "close_commands": close_commands,
                    "all_commands": all_commands,
                    "num_options": num_options,
                }
            )

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = [
            torch.zeros(self.num_envs, len(jgc["joint_ids"]), device=self.device)
            for jgc in self._joint_group_configs
        ]

        self._env_ids = torch.arange(self.num_envs, device=self.device)
        max_options = max(jgc["num_options"] for jgc in self._joint_group_configs)
        self._action_space = spaces.Box(
            low=0.0,
            high=float(max_options - 1),
            shape=(self._num_joint_groups,),
            dtype=torch.float32,
        )

    @property
    def action_dim(self) -> int:
        return self._num_joint_groups

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> list[torch.Tensor]:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def process_actions(
        self, actions: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        if env_ids is None:
            selected_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                selected_env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                selected_env_ids = env_ids
        actions, self._env_ids = self._handle_nan_actions(actions, selected_env_ids)
        if actions is None or len(self._env_ids) == 0:
            return

        assert actions.shape[0] == self._env_ids.shape[0], (
            f"Expected actions shape[0] to be {self._env_ids.shape[0]}, got {actions.shape[0]}"
        )
        assert actions.shape[1] >= self._num_joint_groups, (
            f"Expected actions shape[1] >= {self._num_joint_groups}, got {actions.shape[1]}"
        )

        self._raw_actions[self.env_ids] = actions[:, : self._num_joint_groups]

        for joint_group_idx, joint_group_cfg in enumerate(self._joint_group_configs):
            num_options = joint_group_cfg["num_options"]
            choice_idx = actions[:, joint_group_idx].long().clamp(0, num_options - 1)

            all_stacked = torch.stack(joint_group_cfg["all_commands"], dim=0)
            selected = all_stacked[choice_idx]
            if selected.dim() == 1:
                selected = selected.unsqueeze(0)
            self._processed_actions[joint_group_idx][self.env_ids] = selected

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.int32)

        for i in range(self._num_joint_groups):
            self._raw_actions[env_ids, i] = 0.0
        for joint_group_idx, joint_group_cfg in enumerate(self._joint_group_configs):
            self._processed_actions[joint_group_idx][env_ids] = (
                joint_group_cfg["open_command"].unsqueeze(0).expand(len(env_ids), -1)
            )


class MultipleBinaryJointChoicePositionAction(MultipleBinaryJointChoiceAction):
    """Multiple binary joint choice action that sets into joint position targets."""

    cfg: actions_cfg.MultipleBinaryJointChoicePositionActionCfg
    """The configuration of the action term."""

    def apply_actions(self):
        for joint_group_idx, joint_group_cfg in enumerate(self._joint_group_configs):
            processed_actions_for_envs = self._processed_actions[joint_group_idx][
                self.env_ids
            ]
            assert processed_actions_for_envs.shape[0] == len(self.env_ids), (
                f"Expected processed actions shape[0] to be {len(self.env_ids)}, "
                f"but got {processed_actions_for_envs.shape[0]}"
            )
            self._asset.set_joint_position_target(
                processed_actions_for_envs,
                joint_ids=joint_group_cfg["joint_ids"],
                env_ids=self.env_ids,
            )


class InterpolatedJointChoiceAction(ActionTerm):
    """Base class for interpolated joint actions with multiple close configurations.

    Allows selecting among multiple close joint configurations and interpolating
    between open and the selected close. Useful for grippers with different grip
    strengths (e.g., light grip vs firm grip).

    action_dim = 2:
    - action[0] (joint_group_choice): integer index in [0, num_close_configs-1].
    - action[1] (joint_control): in [0, 1], interpolation factor. 0=open, 1=selected close.

    Command: open + t * (close[choice_idx] - open)
    """

    cfg: actions_cfg.InterpolatedJointChoiceActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _clip: torch.Tensor | None
    """The clip applied to the interpolated joint positions (optional)."""

    def __init__(
        self, cfg: actions_cfg.InterpolatedJointChoiceActionCfg, env: IsaacRLEnv
    ) -> None:
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names
        )
        self._num_joints = len(self._joint_ids)
        self._num_close_configs = len(self.cfg.close_command_exprs)
        if self._num_close_configs < 1:
            raise ValueError(
                "InterpolatedJointChoiceAction requires at least one close_command_expr"
            )

        omni.log.info(
            f"Resolved joint names for {self.__class__.__name__}: "
            f"{self._joint_names} [{self._joint_ids}], "
            f"num_close_configs={self._num_close_configs}"
        )

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros(
            self.num_envs, self._num_joints, device=self.device
        )

        # parse open command
        self._open_command = torch.zeros(self._num_joints, device=self.device)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.open_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(
                f"Could not resolve all joints for open. Missing: {set(self._joint_names) - set(name_list)}"
            )
        self._open_command[index_list] = torch.tensor(value_list, device=self.device)

        # parse close commands (list of configs)
        self._close_commands = []
        for close_expr in self.cfg.close_command_exprs:
            close_cmd = torch.zeros(self._num_joints, device=self.device)
            index_list, name_list, value_list = (
                string_utils.resolve_matching_names_values(
                    close_expr, self._joint_names
                )
            )
            if len(index_list) != self._num_joints:
                raise ValueError(
                    f"Could not resolve all joints for close config. Missing: {set(self._joint_names) - set(name_list)}"
                )
            close_cmd[index_list] = torch.tensor(value_list, device=self.device)
            self._close_commands.append(close_cmd)

        # parse clip
        self._clip = None
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self._num_joints, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported: dict."
                )

        self._action_space = spaces.Box(
            low=[0.0, 0.0],
            high=[float(self._num_close_configs - 1), 1.0],
            shape=(2,),
            dtype=torch.float32,
        )

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def process_actions(
        self, actions: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        if env_ids is None:
            selected_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                selected_env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                selected_env_ids = env_ids
        actions, self._env_ids = self._handle_nan_actions(actions, selected_env_ids)
        if actions is None or len(self._env_ids) == 0:
            return
        assert actions.shape[0] == self._env_ids.shape[0], (
            f"Expected actions shape[0] to be {len(self._env_ids)}, got {actions.shape[0]}"
        )
        assert actions.shape[1] >= 2, (
            f"Expected actions shape[1] >= 2, got {actions.shape[1]}"
        )

        self._raw_actions[self.env_ids] = actions[:, :2]

        # action[0]: integer choice index in [0, num_close_configs-1]
        choice_idx = actions[:, 0:1].long().clamp(0, self._num_close_configs - 1)

        # action[1]: interpolation t in [0, 1]
        t = actions[:, 1:2].clamp(0.0, 1.0)

        # For each env, select close_command by choice_idx and interpolate
        # close_cmd: (num_envs, num_joints) - need to gather from _close_commands
        close_stacked = torch.stack(
            self._close_commands, dim=0
        )  # (num_close, num_joints)
        close_selected = close_stacked[choice_idx.squeeze(1)]  # (num_envs, num_joints)
        if close_selected.dim() == 1:
            close_selected = close_selected.unsqueeze(0)
        interpolated = self._open_command + t * (close_selected - self._open_command)
        self._processed_actions[self.env_ids] = interpolated

        if self._clip is not None:
            self._processed_actions[self.env_ids] = torch.clamp(
                self._processed_actions[self.env_ids],
                min=self._clip[self.env_ids, :, 0],
                max=self._clip[self.env_ids, :, 1],
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.int32)
        self._raw_actions[env_ids, :2] = 0.0


class InterpolatedJointChoicePositionAction(InterpolatedJointChoiceAction):
    """Interpolated joint choice action that sets into joint position targets."""

    cfg: actions_cfg.InterpolatedJointChoicePositionActionCfg
    """The configuration of the action term."""

    def apply_actions(self):
        self._asset.set_joint_position_target(
            self._processed_actions[self.env_ids],
            joint_ids=self._joint_ids,
            env_ids=self.env_ids,
        )


class MultipleInterpolatedJointChoiceAction(ActionTerm):
    """Base class for multiple interpolated joint choice actions.

    Each joint group has 2 action dimensions: (choice, control).
    action_dim = 2 * num_joint_groups.

    Action layout: [joint_group_choice_1, joint_control_1, joint_group_choice_2, joint_control_2, ...]
    - joint_group_choice_i: integer index in [0, num_close_i-1], selects close config for group i
    - joint_control_i: in [0, 1], interpolation between open and selected close for group i
    """

    cfg: actions_cfg.MultipleInterpolatedJointChoiceActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _joint_group_configs: list[dict]
    """List of configs: joint_ids, joint_names, open_command, close_commands (list)."""

    def __init__(
        self, cfg: actions_cfg.MultipleInterpolatedJointChoiceActionCfg, env: IsaacRLEnv
    ) -> None:
        super().__init__(cfg, env)

        self._joint_group_configs = []
        self._num_joint_groups = len(cfg.joint_groups)

        for joint_group_cfg in cfg.joint_groups:
            joint_ids, joint_names = self._asset.find_joints(
                joint_group_cfg.joint_names, preserve_order=True
            )
            num_joints = len(joint_ids)
            num_close = len(joint_group_cfg.close_command_exprs)
            if num_close < 1:
                raise ValueError(
                    f"Joint group requires at least one close_command_expr, got {num_close}"
                )

            omni.log.info(
                f"Resolved joint group: {joint_names} [{joint_ids}], "
                f"num_close_configs={num_close}"
            )

            open_command = torch.zeros(num_joints, device=self.device)
            index_list, name_list, value_list = (
                string_utils.resolve_matching_names_values(
                    joint_group_cfg.open_command_expr, joint_names
                )
            )
            if len(index_list) != num_joints:
                raise ValueError(
                    f"Could not resolve all joints. Missing: {set(joint_names) - set(name_list)}"
                )
            open_command[index_list] = torch.tensor(value_list, device=self.device)

            close_commands = []
            for close_expr in joint_group_cfg.close_command_exprs:
                close_cmd = torch.zeros(num_joints, device=self.device)
                index_list, name_list, value_list = (
                    string_utils.resolve_matching_names_values(close_expr, joint_names)
                )
                if len(index_list) != num_joints:
                    raise ValueError(
                        f"Could not resolve all joints for close. Missing: {set(joint_names) - set(name_list)}"
                    )
                close_cmd[index_list] = torch.tensor(value_list, device=self.device)
                close_commands.append(close_cmd)

            self._joint_group_configs.append(
                {
                    "joint_ids": joint_ids,
                    "joint_names": joint_names,
                    "open_command": open_command,
                    "close_commands": close_commands,
                    "num_close_configs": num_close,
                }
            )

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = [
            torch.zeros(self.num_envs, len(jgc["joint_ids"]), device=self.device)
            for jgc in self._joint_group_configs
        ]

        self._env_ids = torch.arange(self.num_envs, device=self.device)
        max_close = max(jgc["num_close_configs"] for jgc in self._joint_group_configs)
        # choice dims: [0, max_close-1], control dims: [0, 1]; use max for simplicity
        self._action_space = spaces.Box(
            low=0.0,
            high=max(float(max_close - 1), 1.0),
            shape=(2 * self._num_joint_groups,),
            dtype=float,
        )

    @property
    def action_dim(self) -> int:
        return 2 * self._num_joint_groups

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> list[torch.Tensor]:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def process_actions(
        self, actions: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        if env_ids is None:
            selected_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                selected_env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                selected_env_ids = env_ids
        actions, self._env_ids = self._handle_nan_actions(actions, selected_env_ids)
        if actions is None or len(self._env_ids) == 0:
            return

        expected_dim = 2 * self._num_joint_groups
        assert actions.shape[0] == self._env_ids.shape[0], (
            f"Expected actions shape[0] to be {self._env_ids.shape[0]}, got {actions.shape[0]}"
        )
        assert actions.shape[1] >= expected_dim, (
            f"Expected actions shape[1] >= {expected_dim}, got {actions.shape[1]}"
        )

        self._raw_actions[self.env_ids] = actions[:, :expected_dim]

        for joint_group_idx, joint_group_cfg in enumerate(self._joint_group_configs):
            # action[2*i]: integer choice, action[2*i+1]: interpolation
            num_close = joint_group_cfg["num_close_configs"]
            choice_idx = actions[:, 2 * joint_group_idx].long().clamp(0, num_close - 1)
            t = actions[:, 2 * joint_group_idx + 1 : 2 * joint_group_idx + 2].clamp(
                0.0, 1.0
            )

            close_stacked = torch.stack(joint_group_cfg["close_commands"], dim=0)
            close_selected = close_stacked[choice_idx]
            if close_selected.dim() == 1:
                close_selected = close_selected.unsqueeze(0)

            interpolated = joint_group_cfg["open_command"] + t * (
                close_selected - joint_group_cfg["open_command"]
            )
            self._processed_actions[joint_group_idx][self.env_ids] = interpolated

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.int32)

        for i in range(2 * self._num_joint_groups):
            self._raw_actions[env_ids, i] = 0.0
        for joint_group_idx, joint_group_cfg in enumerate(self._joint_group_configs):
            self._processed_actions[joint_group_idx][env_ids] = (
                joint_group_cfg["open_command"].unsqueeze(0).expand(len(env_ids), -1)
            )


class MultipleInterpolatedJointChoicePositionAction(
    MultipleInterpolatedJointChoiceAction
):
    """Multiple interpolated joint choice action that sets into joint position targets."""

    cfg: actions_cfg.MultipleInterpolatedJointChoicePositionActionCfg
    """The configuration of the action term."""

    def apply_actions(self):
        for joint_group_idx, joint_group_cfg in enumerate(self._joint_group_configs):
            processed_actions_for_envs = self._processed_actions[joint_group_idx][
                self.env_ids
            ]
            assert processed_actions_for_envs.shape[0] == len(self.env_ids), (
                f"Expected processed actions shape[0] to be {len(self.env_ids)}, "
                f"but got {processed_actions_for_envs.shape[0]}"
            )
            self._asset.set_joint_position_target(
                processed_actions_for_envs,
                joint_ids=joint_group_cfg["joint_ids"],
                env_ids=self.env_ids,
            )


class MultipleBinaryJointAction(ActionTerm):
    """Base class for multiple binary joint actions.

    This action term maps multiple binary actions to independent joint group configurations.
    Each joint group can be controlled separately with its own binary action (open/close).
    These configurations are specified through the :class:`MultipleBinaryJointActionCfg` object.

    Based on above, we follow the following convention for the binary action:

    1. Open action: 0 (bool) or negative values (float).
    2. Close action: 1 (bool) or positive values (float).
    """

    cfg: actions_cfg.MultipleBinaryJointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _joint_group_configs: list[dict]
    """List of joint group configurations, each containing joint_ids, joint_names, open_command, close_command."""

    def __init__(
        self, cfg: actions_cfg.MultipleBinaryJointActionCfg, env: IsaacRLEnv
    ) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # store joint group configurations
        self._joint_group_configs = []
        self._num_joint_groups = len(cfg.joint_groups)

        # process each joint group
        for joint_group_cfg in cfg.joint_groups:
            # resolve the joints for this joint group
            joint_ids, joint_names = self._asset.find_joints(
                joint_group_cfg.joint_names
            )
            num_joints = len(joint_ids)

            # log the resolved joint names for debugging
            omni.log.info(
                f"Resolved joint names for joint group: {joint_names} [{joint_ids}]"
            )

            # parse open command
            open_command = torch.zeros(num_joints, device=self.device)
            index_list, name_list, value_list = (
                string_utils.resolve_matching_names_values(
                    joint_group_cfg.open_command_expr, joint_names
                )
            )
            if len(index_list) != num_joints:
                raise ValueError(
                    f"Could not resolve all joints for joint group. Missing: {set(joint_names) - set(name_list)}"
                )
            open_command[index_list] = torch.tensor(value_list, device=self.device)

            # parse close command
            close_command = torch.zeros_like(open_command)
            index_list, name_list, value_list = (
                string_utils.resolve_matching_names_values(
                    joint_group_cfg.close_command_expr, joint_names
                )
            )
            if len(index_list) != num_joints:
                raise ValueError(
                    f"Could not resolve all joints for joint group. Missing: {set(joint_names) - set(name_list)}"
                )
            close_command[index_list] = torch.tensor(value_list, device=self.device)

            # store joint group configuration
            self._joint_group_configs.append(
                {
                    "joint_ids": joint_ids,
                    "joint_names": joint_names,
                    "open_command": open_command,
                    "close_command": close_command,
                }
            )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        # processed_actions will be a list of tensors, one per joint group
        self._processed_actions = [
            torch.zeros(
                self.num_envs, len(joint_group_cfg["joint_ids"]), device=self.device
            )
            for joint_group_cfg in self._joint_group_configs
        ]

        # initialize env_ids
        self._env_ids = torch.arange(self.num_envs, device=self.device)

        # action space: MultiDiscrete with 2 options per joint group
        self._action_space = spaces.MultiDiscrete([2] * self._num_joint_groups)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joint_groups

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> list[torch.Tensor]:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        return self._action_space

    """
    Operations.
    """

    def process_actions(
        self, actions: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        if env_ids is None:
            selected_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                selected_env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                selected_env_ids = env_ids
        actions, self._env_ids = self._handle_nan_actions(actions, selected_env_ids)
        if actions is None or len(self._env_ids) == 0:
            return

        assert actions.shape[0] == self._env_ids.shape[0], (
            f"Expected actions shape[0] to be {self._env_ids.shape[0]}, but got {actions.shape[0]}"
        )
        assert actions.shape[1] == self._num_joint_groups, (
            f"Expected actions shape[1] to be {self._num_joint_groups}, but got {actions.shape[1]}"
        )

        # store the raw actions
        self._raw_actions[self.env_ids] = actions

        # process each joint group independently
        for joint_group_idx, joint_group_cfg in enumerate(self._joint_group_configs):
            # get the binary action for this joint group
            joint_group_actions = actions[:, joint_group_idx]

            # compute the binary mask
            if joint_group_actions.dtype == torch.bool:
                # true: close, false: open
                binary_mask = joint_group_actions == 1
            else:
                # true: close, false: open
                binary_mask = joint_group_actions >= 1

            # compute the command for this joint group
            # expand dimensions to match the number of joints in this joint group
            binary_mask_expanded = binary_mask.unsqueeze(1)  # [num_envs, 1]
            open_cmd = joint_group_cfg["open_command"].unsqueeze(0)  # [1, num_joints]
            close_cmd = joint_group_cfg["close_command"].unsqueeze(0)  # [1, num_joints]

            # apply open/close command based on binary mask
            joint_group_command = torch.where(
                binary_mask_expanded,
                close_cmd.expand(len(self.env_ids), -1),
                open_cmd.expand(len(self.env_ids), -1),
            )

            # store processed actions for this joint group
            self._processed_actions[joint_group_idx][self.env_ids] = joint_group_command

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.int32)

        self._raw_actions[env_ids] = 0.0
        for joint_group_idx, joint_group_cfg in enumerate(self._joint_group_configs):
            # reset to open position
            self._processed_actions[joint_group_idx][env_ids] = (
                joint_group_cfg["open_command"].unsqueeze(0).expand(len(env_ids), -1)
            )


class MultipleBinaryJointPositionAction(MultipleBinaryJointAction):
    """Multiple binary joint action that sets the binary actions into joint position targets."""

    cfg: actions_cfg.MultipleBinaryJointPositionActionCfg
    """The configuration of the action term."""

    def apply_actions(self):
        """Apply the processed actions to all joint groups."""
        for joint_group_idx, joint_group_cfg in enumerate(self._joint_group_configs):
            # Get only the processed actions for the current env_ids
            processed_actions_for_envs = self._processed_actions[joint_group_idx][
                self.env_ids
            ]
            assert processed_actions_for_envs.shape[0] == len(self.env_ids), (
                f"Expected processed actions shape[0] to be {len(self.env_ids)}, "
                f"but got {processed_actions_for_envs.shape[0]}"
            )
            self._asset.set_joint_position_target(
                processed_actions_for_envs,
                joint_ids=joint_group_cfg["joint_ids"],
                env_ids=self.env_ids,
            )


class DifferentialInverseKinematicsAction(ActionTerm):
    r"""Inverse Kinematics action term.

    This action term performs pre-processing of the raw actions using scaling transformation.

    .. math::
        \text{action} = \text{scaling} \times \text{input action}
        \text{joint position} = J^{-} \times \text{action}

    where :math:`\text{scaling}` is the scaling applied to the input action, and :math:`\text{input action}`
    is the input action from the user, :math:`J` is the Jacobian over the articulation's actuated joints,
    and \text{joint position} is the desired joint position command for the articulation's joints.
    """

    cfg: actions_cfg.DifferentialInverseKinematicsActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    _action_space: spaces.Box
    """The action space of the action term."""

    def __init__(
        self, cfg: actions_cfg.DifferentialInverseKinematicsActionCfg, env: IsaacRLEnv
    ):
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names
        )
        self._num_joints = len(self._joint_ids)
        # parse the body index
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        # save only the first body index
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        # optional: reference body for command transform (e.g. panda_link0 = arm base)
        self._command_ref_body_idx: int | None = None
        if self.cfg.command_reference_body_name is not None:
            ref_ids, ref_names = self._asset.find_bodies(
                self.cfg.command_reference_body_name
            )
            if len(ref_ids) != 1:
                raise ValueError(
                    f"Expected one match for command_reference_body_name "
                    f"{self.cfg.command_reference_body_name}. Found {len(ref_ids)}: {ref_names}."
                )
            self._command_ref_body_idx = ref_ids[0]
            # omni.log.info(
            #     f"Command-to-base transform uses body {ref_names[0]} [{self._command_ref_body_idx}] (arm base)."
            # )
        # check if articulation is fixed-base
        # if fixed-base then the jacobian for the base is not computed
        # this means that number of bodies is one less than the articulation's number of bodies
        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]

        # log info for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        omni.log.info(
            f"Resolved body name for the action term {self.__class__.__name__}: {self._body_name} [{self._body_idx}]"
        )

        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.controller, num_envs=self.num_envs, device=self.device
        )

        # create tensors for raw and processed actions.
        # ``_raw_actions`` stores the raw EEF pose command (action_dim = 3/6/7).
        # ``_processed_actions`` stores the POST-IK joint position targets
        # (num_joints) so downstream obs terms / consumers can read the joint
        # targets this action term last sent to physics. Shape differs from
        # ``_raw_actions`` — do not use ``zeros_like(raw_actions)``.
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros(
            self.num_envs, self._num_joints, device=self.device
        )

        # save the scale as tensors
        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        # convert the fixed offsets to torch tensors of batched shape
        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(
                self.cfg.body_offset.pos, device=self.device
            ).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(
                self.cfg.body_offset.rot, device=self.device
            ).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self.action_dim, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict."
                )

        assert self.cfg.action_space.shape[0] == 2, (
            "Expected action space to be of shape (2, action_dim)."
        )
        assert self.cfg.action_space.shape[1] == self.action_dim, (
            f"Expected action space to be of shape (2, {self.action_dim})."
        )

        if self.cfg.clip is not None:
            self.cfg.action_space = torch.clamp(
                self.cfg.action_space, min=self.cfg.clip[:, 0], max=self.cfg.clip[:, 1]
            )
        self._action_space = spaces.Box(
            low=self.cfg.action_space[0].cpu().numpy(),
            high=self.cfg.action_space[1].cpu().numpy(),
        )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._ik_controller.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._asset.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._jacobi_joint_ids
        ]

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def jacobian_b(self) -> torch.Tensor:
        """Jacobian in reference frame (arm base when command_reference_body_name set, else root)."""
        jacobian = self.jacobian_w
        if self._command_ref_body_idx is not None:
            ref_quat = self._asset.data.body_quat_w[:, self._command_ref_body_idx]
        else:
            ref_quat = self._asset.data.root_quat_w
        ref_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(ref_quat))
        jacobian[:, :3, :] = torch.bmm(ref_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(ref_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    """
    Operations.
    """

    def _handle_nan_actions(
        self, actions: torch.Tensor, selected_env_ids: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Handle NaN actions by filling with current EEF pose (single EEF, no grouping).

        1. Skip envs where ALL actions are NaN.
        2. For envs with any valid action: if any NaN in the pose, fill with current EEF pose
           so the IK solver keeps the end-effector at its current position.

        Args:
            actions: Input action tensor of shape (num_envs, action_dim).
            selected_env_ids: Selected environment IDs.

        Returns:
            Tuple of (processed_actions, valid_env_ids).
            processed_actions is None if no valid envs exist.
        """
        all_nan_mask = torch.isnan(actions).all(dim=1)
        valid_mask = ~all_nan_mask
        valid_indices = torch.where(valid_mask)[0]
        env_ids = selected_env_ids[valid_indices]

        if len(env_ids) == 0:
            return None, env_ids

        valid_actions = actions[valid_indices].clone()

        # Fill NaN with current EEF pose (single EEF, action_dim is 3, 6, or 7)
        pose_nan_mask = torch.isnan(valid_actions).any(dim=1)
        if pose_nan_mask.any():
            nan_env_ids = env_ids[pose_nan_mask]
            current_pose = self._get_current_eef_pose_for_nan(nan_env_ids)
            valid_actions[pose_nan_mask] = current_pose

        return valid_actions, env_ids

    def _get_current_eef_pose_for_nan(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Get current EEF pose for NaN fill, in same frame/format as command.

        Returns pose in reference frame (base or root). For relative mode (6 dof),
        returns zeros (delta=0 means keep current).
        """
        pose_dim = self.action_dim
        if self.cfg.controller.use_relative_mode:
            # Relative mode: delta=0 means keep current
            return torch.zeros(len(env_ids), pose_dim, device=self.device)
        ee_pos_b, ee_quat_b = self._compute_frame_pose_for_env_ids(env_ids)
        if pose_dim == 3:
            return ee_pos_b
        return torch.cat([ee_pos_b, ee_quat_b], dim=1)

    def _compute_frame_pose_for_env_ids(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute EEF pose in reference frame for given env_ids (for NaN handling)."""
        ee_pos_w = self._asset.data.body_pos_w[env_ids, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[env_ids, self._body_idx]
        if self._command_ref_body_idx is not None:
            ref_pos_w = self._asset.data.body_pos_w[env_ids, self._command_ref_body_idx]
            ref_quat_w = self._asset.data.body_quat_w[
                env_ids, self._command_ref_body_idx
            ]
        else:
            ref_pos_w = self._asset.data.root_pos_w[env_ids]
            ref_quat_w = self._asset.data.root_quat_w[env_ids]
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            ref_pos_w, ref_quat_w, ee_pos_w, ee_quat_w
        )
        if self.cfg.body_offset is not None:
            ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pose_b,
                ee_quat_b,
                self._offset_pos[env_ids],
                self._offset_rot[env_ids],
            )
        return ee_pose_b, ee_quat_b

    def process_actions(
        self, actions: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        if env_ids is None:
            selected_env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            selected_env_ids = torch.tensor(
                env_ids, device=self.device, dtype=torch.int32
            )
        else:
            selected_env_ids = env_ids

        valid_actions, self._env_ids = self._handle_nan_actions(
            actions, selected_env_ids
        )

        if valid_actions is None or len(self._env_ids) == 0:
            return

        # Store the raw EEF pose command in real-env-id space (num_envs, action_dim).
        self._raw_actions[self._env_ids] = valid_actions

        # Build the pose command fed to the IK controller in a K-local scratch
        # tensor — ``_processed_actions`` is reserved for joint targets.
        command = valid_actions.clone()
        if self.cfg.clip is not None:
            command = torch.clamp(
                command,
                min=self._clip[self._env_ids, :, 0],
                max=self._clip[self._env_ids, :, 1],
            )
        if not self.cfg.relative_to_base and not self.cfg.controller.use_relative_mode:
            command = self._transform_command_to_base(command)

        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.set_command(command, ee_pos_curr, ee_quat_curr)

    def apply_actions(self):
        # Index-space convention:
        #   K-local  (size K = len(self.env_ids)): joint_pos, joint_pos_des,
        #     ee_pos_curr, ee_quat_curr, valid_mask, jacobian[self.env_ids]
        #   real env-id (size num_envs): self.env_ids, valid_env_ids,
        #     asset buffers, _raw_actions, _processed_actions
        if len(self.env_ids) == 0:
            return
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()  # (K, 3), (K, 4)
        joint_pos = self._asset.data.joint_pos[
            self.env_ids.unsqueeze(1), self._joint_ids
        ]  # (K, num_joints)

        quat_norm = torch.linalg.norm(ee_quat_curr, dim=1)  # (K,)
        valid_mask = quat_norm > 0  # (K,)
        if not valid_mask.any():
            print("no valid end-effector pose, follow the current joint position")
            return

        jacobian = self._compute_frame_jacobian()[self.env_ids]  # (K, 6, num_joints)
        joint_pos_des = joint_pos.clone()  # (K, num_joints)
        joint_pos_des[valid_mask] = self._ik_controller.compute(
            ee_pos_curr[valid_mask],
            ee_quat_curr[valid_mask],
            jacobian[valid_mask],
            joint_pos[valid_mask],
            ee_pos_des=self._ik_controller.ee_pos_des[valid_mask],
            ee_quat_des=self._ik_controller.ee_quat_des[valid_mask],
        )
        # Clamp to joint limits to prevent IK from requesting out-of-limit
        # positions (which would cause persistent error and divergence).
        joint_limits_low = self._asset.data.joint_pos_limits[0, self._joint_ids, 0]
        joint_limits_high = self._asset.data.joint_pos_limits[0, self._joint_ids, 1]
        joint_pos_des = torch.clamp(
            joint_pos_des, min=joint_limits_low, max=joint_limits_high
        )

        # Lift K-local valid rows into real env-id space for:
        #   1) remembering joint targets in _processed_actions (num_envs, num_joints)
        #   2) issuing set_joint_position_target (takes real env indices)
        # Invalid-mask envs keep whatever _processed_actions and the physics
        # target were from the previous step.
        valid_env_ids = self.env_ids[valid_mask]
        self._processed_actions[valid_env_ids] = joint_pos_des[valid_mask]
        self._asset.set_joint_position_target(
            joint_pos_des[valid_mask], self._joint_ids, valid_env_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    """
    Helper functions.
    """

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the reference frame (arm base or root).

        Uses same reference as _transform_command_to_base (e.g. panda_link0 when set).
        Returns:
            A tuple of the body's position and orientation in the reference frame.
        """
        ee_pos_w = self._asset.data.body_pos_w[self.env_ids, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[self.env_ids, self._body_idx]
        if self._command_ref_body_idx is not None:
            ref_pos_w = self._asset.data.body_pos_w[
                self.env_ids, self._command_ref_body_idx
            ]
            ref_quat_w = self._asset.data.body_quat_w[
                self.env_ids, self._command_ref_body_idx
            ]
        else:
            ref_pos_w = self._asset.data.root_pos_w[self.env_ids]
            ref_quat_w = self._asset.data.root_quat_w[self.env_ids]
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            ref_pos_w, ref_quat_w, ee_pos_w, ee_quat_w
        )
        # account for the offset
        if self.cfg.body_offset is not None:
            ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pose_b,
                ee_quat_b,
                self._offset_pos[self.env_ids],
                self._offset_rot[self.env_ids],
            )

        return ee_pose_b, ee_quat_b

    def _transform_command_to_base(self, command: torch.Tensor) -> torch.Tensor:
        """Transform command from world (or env) frame to reference frame for IK.

        Reference frame: if command_reference_body_name is set (e.g. panda_link0 = arm base),
        use that body pose; otherwise use articulation root (e.g. base_link).
        subtract_frame_transforms(t01, q01, t02, q02) returns frame-2 pose in frame-1.
        If command is in env-local frame, convert to world first: pos_w = env_origin + pos.
        """
        if self._command_ref_body_idx is not None:
            base_pos_w = self._asset.data.body_pos_w[
                self.env_ids, self._command_ref_body_idx
            ]
            base_quat_w = self._asset.data.body_quat_w[
                self.env_ids, self._command_ref_body_idx
            ]
        else:
            base_pos_w = self._asset.data.root_pos_w[self.env_ids]
            base_quat_w = self._asset.data.root_quat_w[self.env_ids]
        env_origins = getattr(self._env.scene, "env_origins", None)

        # Command may be in env-local frame (e.g. cube_pose from get_local_pose).
        # Convert to world so we can express target in base_link consistently.
        if env_origins is not None:
            positions_w = command[:, 0:3] + env_origins[self.env_ids]
        else:
            positions_w = command[:, 0:3]

        if self.cfg.controller.command_type == "position":
            positions_b, _ = math_utils.subtract_frame_transforms(
                base_pos_w, base_quat_w, positions_w, None
            )
            return positions_b

        quats_w = command[:, 3:7]
        positions_b, quats_b = math_utils.subtract_frame_transforms(
            base_pos_w, base_quat_w, positions_w, quats_w
        )
        return torch.cat((positions_b, quats_b), dim=1)

    def _compute_frame_jacobian(self):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # read the parent jacobian
        jacobian = self.jacobian_b
        # account for the offset
        if self.cfg.body_offset is not None:
            # Modify the jacobian to account for the offset
            # -- translational part
            # v_link = v_ee + w_ee x r_link_ee = v_J_ee * q + w_J_ee * q x r_link_ee
            #        = (v_J_ee + w_J_ee x r_link_ee ) * q
            #        = (v_J_ee - r_link_ee_[x] @ w_J_ee) * q
            jacobian[:, 0:3, :] += torch.bmm(
                -math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :]
            )
            # -- rotational part
            # w_link = R_link_ee @ w_ee
            jacobian[:, 3:, :] = torch.bmm(
                math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :]
            )

        return jacobian


class DualDifferentialInverseKinematicsAction(ActionTerm):
    r"""Dual-arm differential IK action term.

    Drives two independent kinematic chains (e.g. a shared torso with a left and
    right arm) with two separate :class:`DifferentialIKController` instances.
    Each arm has its own joint set, end-effector body, and optional arm-base
    reference body; they only share the articulation and PhysX jacobian buffer.

    Input action shape is ``(N, 2 * per_arm_dim)`` with **right first, left
    second** — matching the ``DualManipulator`` frame convention. When
    ``controller.command_type == "pose"`` each arm takes 7 values
    ``(px, py, pz, qw, qx, qy, qz)``; when ``"position"`` each arm takes 3.
    """

    cfg: actions_cfg.DualDifferentialInverseKinematicsActionCfg
    _asset: Articulation

    def __init__(
        self,
        cfg: actions_cfg.DualDifferentialInverseKinematicsActionCfg,
        env: IsaacRLEnv,
    ):
        super().__init__(cfg, env)

        # Resolve joints and bodies for both arms.
        self._right_joint_ids, self._right_joint_names = self._asset.find_joints(
            cfg.right_joint_names
        )
        self._left_joint_ids, self._left_joint_names = self._asset.find_joints(
            cfg.left_joint_names
        )
        self._num_right_joints = len(self._right_joint_ids)
        self._num_left_joints = len(self._left_joint_ids)

        right_body_ids, right_body_names = self._asset.find_bodies(cfg.right_body_name)
        left_body_ids, left_body_names = self._asset.find_bodies(cfg.left_body_name)
        if len(right_body_ids) != 1 or len(left_body_ids) != 1:
            raise ValueError(
                f"Expected one body match each: right={right_body_names}, left={left_body_names}"
            )
        self._right_body_idx = right_body_ids[0]
        self._left_body_idx = left_body_ids[0]

        # Optional per-arm reference body (e.g. L_base_link / R_base_link).
        self._right_ref_idx, self._left_ref_idx = None, None
        if cfg.right_command_reference_body_name is not None:
            ids, _ = self._asset.find_bodies(cfg.right_command_reference_body_name)
            if len(ids) != 1:
                raise ValueError(
                    f"right_command_reference_body_name resolved to {len(ids)} bodies."
                )
            self._right_ref_idx = ids[0]
        if cfg.left_command_reference_body_name is not None:
            ids, _ = self._asset.find_bodies(cfg.left_command_reference_body_name)
            if len(ids) != 1:
                raise ValueError(
                    f"left_command_reference_body_name resolved to {len(ids)} bodies."
                )
            self._left_ref_idx = ids[0]

        # PhysX jacobian index offsets (fixed-base robots drop the root row).
        if self._asset.is_fixed_base:
            self._right_jacobi_body_idx = self._right_body_idx - 1
            self._left_jacobi_body_idx = self._left_body_idx - 1
            self._right_jacobi_joint_ids = self._right_joint_ids
            self._left_jacobi_joint_ids = self._left_joint_ids
        else:
            self._right_jacobi_body_idx = self._right_body_idx
            self._left_jacobi_body_idx = self._left_body_idx
            self._right_jacobi_joint_ids = [i + 6 for i in self._right_joint_ids]
            self._left_jacobi_joint_ids = [i + 6 for i in self._left_joint_ids]

        # One controller per arm — they share cfg.controller since both arms
        # use identical IK settings (pinv/dls/svd, gains, command_type).
        self._right_ik = DifferentialIKController(
            cfg=cfg.controller, num_envs=self.num_envs, device=self.device
        )
        self._left_ik = DifferentialIKController(
            cfg=cfg.controller, num_envs=self.num_envs, device=self.device
        )
        self._per_arm_dim = self._right_ik.action_dim

        # Fixed per-arm offsets.
        self._right_offset_pos, self._right_offset_rot = self._build_offset(
            cfg.right_body_offset
        )
        self._left_offset_pos, self._left_offset_rot = self._build_offset(
            cfg.left_body_offset
        )

        # Buffers.
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)
        # Cached joint targets so we can re-apply between IK recomputes when
        # `decimation > 1`. Indexed in full-articulation joint space.
        self._right_joint_target = torch.zeros(
            self.num_envs, self._num_right_joints, device=self.device
        )
        self._left_joint_target = torch.zeros(
            self.num_envs, self._num_left_joints, device=self.device
        )
        self._has_cached_target = False
        self._step_count = 0

        # Validate action_space.
        assert cfg.action_space.shape == (2, self.action_dim), (
            f"Expected action_space shape (2, {self.action_dim}), got {tuple(cfg.action_space.shape)}."
        )
        self._action_space = spaces.Box(
            low=cfg.action_space[0].cpu().numpy(),
            high=cfg.action_space[1].cpu().numpy(),
        )

        self._env_ids = torch.arange(self.num_envs, device=self.device)

    # ---------- helpers ---------------------------------------------------

    def _build_offset(self, offset_cfg):
        if offset_cfg is None:
            return None, None
        pos = torch.tensor(offset_cfg.pos, device=self.device).repeat(self.num_envs, 1)
        rot = torch.tensor(offset_cfg.rot, device=self.device).repeat(self.num_envs, 1)
        return pos, rot

    @property
    def action_dim(self) -> int:
        return 2 * self._per_arm_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    # ---------- per-arm pose / jacobian ---------------------------------

    def _arm_pose_in_ref(
        self,
        env_ids: torch.Tensor,
        body_idx: int,
        ref_idx: int | None,
        offset_pos: torch.Tensor | None,
        offset_rot: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._asset.data.body_pos_w[env_ids, body_idx]
        ee_quat_w = self._asset.data.body_quat_w[env_ids, body_idx]
        if ref_idx is not None:
            ref_pos_w = self._asset.data.body_pos_w[env_ids, ref_idx]
            ref_quat_w = self._asset.data.body_quat_w[env_ids, ref_idx]
        else:
            ref_pos_w = self._asset.data.root_pos_w[env_ids]
            ref_quat_w = self._asset.data.root_quat_w[env_ids]
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            ref_pos_w, ref_quat_w, ee_pos_w, ee_quat_w
        )
        if offset_pos is not None:
            ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pos_b, ee_quat_b, offset_pos[env_ids], offset_rot[env_ids]
            )
        return ee_pos_b, ee_quat_b

    def _arm_jacobian(
        self,
        env_ids: torch.Tensor,
        jacobi_body_idx: int,
        jacobi_joint_ids: list[int],
        ref_idx: int | None,
        offset_pos: torch.Tensor | None,
        offset_rot: torch.Tensor | None,
    ) -> torch.Tensor:
        jac_w = self._asset.root_physx_view.get_jacobians()[
            :, jacobi_body_idx, :, jacobi_joint_ids
        ][env_ids]
        # Rotate into reference frame.
        if ref_idx is not None:
            ref_quat = self._asset.data.body_quat_w[env_ids, ref_idx]
        else:
            ref_quat = self._asset.data.root_quat_w[env_ids]
        R = math_utils.matrix_from_quat(math_utils.quat_inv(ref_quat))
        jac = jac_w.clone()
        jac[:, 0:3, :] = torch.bmm(R, jac[:, 0:3, :])
        jac[:, 3:, :] = torch.bmm(R, jac[:, 3:, :])
        # Apply body offset to jacobian (same correction as single-arm action).
        if offset_pos is not None:
            skew = math_utils.skew_symmetric_matrix(offset_pos[env_ids])
            jac[:, 0:3, :] += torch.bmm(-skew, jac[:, 3:, :])
            jac[:, 3:, :] = torch.bmm(
                math_utils.matrix_from_quat(offset_rot[env_ids]), jac[:, 3:, :]
            )
        return jac

    def _transform_command_to_base(
        self, command: torch.Tensor, env_ids: torch.Tensor, ref_idx: int | None
    ) -> torch.Tensor:
        """Convert a (N, per_arm_dim) command from env-local world frame into the arm's reference frame."""
        if ref_idx is not None:
            base_pos_w = self._asset.data.body_pos_w[env_ids, ref_idx]
            base_quat_w = self._asset.data.body_quat_w[env_ids, ref_idx]
        else:
            base_pos_w = self._asset.data.root_pos_w[env_ids]
            base_quat_w = self._asset.data.root_quat_w[env_ids]
        env_origins = getattr(self._env.scene, "env_origins", None)
        positions_w = command[:, 0:3]
        if env_origins is not None:
            positions_w = positions_w + env_origins[env_ids]
        if self.cfg.controller.command_type == "position":
            pos_b, _ = math_utils.subtract_frame_transforms(
                base_pos_w, base_quat_w, positions_w, None
            )
            return pos_b
        quats_w = command[:, 3:7]
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            base_pos_w, base_quat_w, positions_w, quats_w
        )
        return torch.cat((pos_b, quat_b), dim=1)

    # ---------- ActionTerm API ------------------------------------------

    def process_actions(
        self, actions: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        if env_ids is None:
            sel = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            sel = torch.tensor(env_ids, device=self.device, dtype=torch.int64)
        else:
            sel = env_ids.to(self.device, dtype=torch.int64)

        # NaN handling: skip envs where ALL values are NaN. For partial NaN, fill
        # per-arm with current EEF pose so IK holds position.
        all_nan = torch.isnan(actions).all(dim=1)
        valid_idx = torch.where(~all_nan)[0]
        self._env_ids = sel[valid_idx]
        if len(self._env_ids) == 0:
            return
        valid = actions[valid_idx].clone()
        d = self._per_arm_dim
        right_cmd, left_cmd = valid[:, :d], valid[:, d : 2 * d]

        # Per-arm NaN fill with current pose (relative mode → zeros).
        right_cmd = self._fill_arm_nan(
            right_cmd,
            self._env_ids,
            self._right_body_idx,
            self._right_ref_idx,
            self._right_offset_pos,
            self._right_offset_rot,
        )
        left_cmd = self._fill_arm_nan(
            left_cmd,
            self._env_ids,
            self._left_body_idx,
            self._left_ref_idx,
            self._left_offset_pos,
            self._left_offset_rot,
        )

        self._raw_actions[self._env_ids] = torch.cat([right_cmd, left_cmd], dim=1)
        self._processed_actions[self._env_ids] = self._raw_actions[self._env_ids]

        # Optionally convert from env-local world frame to each arm's ref frame.
        if not self.cfg.relative_to_base and not self.cfg.controller.use_relative_mode:
            right_cmd = self._transform_command_to_base(
                right_cmd, self._env_ids, self._right_ref_idx
            )
            left_cmd = self._transform_command_to_base(
                left_cmd, self._env_ids, self._left_ref_idx
            )

        # Current EEF pose in ref frame (needed by set_command for rel modes).
        r_pos, r_quat = self._arm_pose_in_ref(
            self._env_ids,
            self._right_body_idx,
            self._right_ref_idx,
            self._right_offset_pos,
            self._right_offset_rot,
        )
        l_pos, l_quat = self._arm_pose_in_ref(
            self._env_ids,
            self._left_body_idx,
            self._left_ref_idx,
            self._left_offset_pos,
            self._left_offset_rot,
        )
        self._right_ik.set_command(right_cmd, r_pos, r_quat)
        self._left_ik.set_command(left_cmd, l_pos, l_quat)

    def _fill_arm_nan(
        self,
        cmd: torch.Tensor,
        env_ids: torch.Tensor,
        body_idx: int,
        ref_idx: int | None,
        offset_pos,
        offset_rot,
    ) -> torch.Tensor:
        """Fill NaN rows with the current EE pose so a single-arm action
        (the other arm slot all NaN) doesn't drive that arm anywhere.

        Frame matters: ``process_actions`` runs ``_transform_command_to_base``
        AFTER this fill when ``relative_to_base=False``, which assumes the
        cmd is in env-local world frame and subtracts the base pose. So the
        fill MUST be in the same frame as the rest of the cmd at this
        pipeline stage:
          - relative_mode  → fill 0 (delta cmd, "no motion")
          - relative_to_base=True → cmd already in base/ref frame; fill in ref
          - relative_to_base=False → cmd in env-local world; fill in env-local world
        """
        nan_rows = torch.isnan(cmd).any(dim=1)
        if not nan_rows.any():
            return cmd
        if self.cfg.controller.use_relative_mode:
            cmd[nan_rows] = 0.0
            return cmd
        rows = torch.where(nan_rows)[0]
        eids = env_ids[rows]
        if self.cfg.relative_to_base:
            # cmd already in ref frame → ref-frame fill matches downstream.
            pos, quat = self._arm_pose_in_ref(
                eids, body_idx, ref_idx, offset_pos, offset_rot
            )
        else:
            # cmd in env-local world → fill with env-local world pose
            # (absolute world from sim minus env_origin). The downstream
            # _transform_command_to_base then correctly converts the whole
            # cmd to base frame in one shot.
            pos = self._asset.data.body_pos_w[eids, body_idx].clone()
            quat = self._asset.data.body_quat_w[eids, body_idx].clone()
            env_origins = getattr(self._env.scene, "env_origins", None)
            if env_origins is not None:
                pos = pos - env_origins[eids]
            # Apply optional body offset in body frame, same as
            # ``_arm_pose_in_ref`` does after subtract — keeps the fill
            # consistent with the action term's "EE + offset" target.
            if offset_pos is not None:
                pos, quat = math_utils.combine_frame_transforms(
                    pos, quat, offset_pos[eids], offset_rot[eids]
                )
        if self._per_arm_dim == 3:
            cmd[rows] = pos
        else:
            cmd[rows] = torch.cat([pos, quat], dim=1)
        return cmd

    def apply_actions(self):
        if len(self._env_ids) == 0:
            return
        self._step_count += 1
        recompute = (
            not self._has_cached_target
            or (self._step_count - 1) % max(1, self.cfg.decimation) == 0
        )

        if recompute:
            # Right arm.
            r_pos, r_quat = self._arm_pose_in_ref(
                self._env_ids,
                self._right_body_idx,
                self._right_ref_idx,
                self._right_offset_pos,
                self._right_offset_rot,
            )
            r_q = self._asset.data.joint_pos[
                self._env_ids.unsqueeze(1), self._right_joint_ids
            ]
            r_jac = self._arm_jacobian(
                self._env_ids,
                self._right_jacobi_body_idx,
                self._right_jacobi_joint_ids,
                self._right_ref_idx,
                self._right_offset_pos,
                self._right_offset_rot,
            )
            r_q_des = self._right_ik.compute(r_pos, r_quat, r_jac, r_q)

            # Left arm.
            l_pos, l_quat = self._arm_pose_in_ref(
                self._env_ids,
                self._left_body_idx,
                self._left_ref_idx,
                self._left_offset_pos,
                self._left_offset_rot,
            )
            l_q = self._asset.data.joint_pos[
                self._env_ids.unsqueeze(1), self._left_joint_ids
            ]
            l_jac = self._arm_jacobian(
                self._env_ids,
                self._left_jacobi_body_idx,
                self._left_jacobi_joint_ids,
                self._left_ref_idx,
                self._left_offset_pos,
                self._left_offset_rot,
            )
            l_q_des = self._left_ik.compute(l_pos, l_quat, l_jac, l_q)

            # Clamp to joint limits.
            r_lo = self._asset.data.joint_pos_limits[0, self._right_joint_ids, 0]
            r_hi = self._asset.data.joint_pos_limits[0, self._right_joint_ids, 1]
            l_lo = self._asset.data.joint_pos_limits[0, self._left_joint_ids, 0]
            l_hi = self._asset.data.joint_pos_limits[0, self._left_joint_ids, 1]
            r_q_des = torch.clamp(r_q_des, min=r_lo, max=r_hi)
            l_q_des = torch.clamp(l_q_des, min=l_lo, max=l_hi)

            self._right_joint_target[self._env_ids] = r_q_des
            self._left_joint_target[self._env_ids] = l_q_des
            self._has_cached_target = True

        self._asset.set_joint_position_target(
            self._right_joint_target[self._env_ids],
            self._right_joint_ids,
            self._env_ids,
        )
        self._asset.set_joint_position_target(
            self._left_joint_target[self._env_ids],
            self._left_joint_ids,
            self._env_ids,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._right_joint_target.zero_()
            self._left_joint_target.zero_()
            self._has_cached_target = False
            self._step_count = 0
        else:
            self._raw_actions[env_ids] = 0.0
            self._right_joint_target[env_ids] = 0.0
            self._left_joint_target[env_ids] = 0.0


class HolonomicAction(ActionTerm):
    """Base-frame holonomic velocity action.

    Input ``actions`` are interpreted as ``[vx_b, vy_b, wz_b]`` in the robot's
    **base frame**: ``vx_b`` is forward (along base_link +x), ``vy_b`` is left
    (along base_link +y), ``wz_b`` is yaw rate around base_link z.

    Before being written to PhysX, the linear components are rotated into the
    world frame using the robot's current yaw, and the three components are
    remapped onto the underlying dummy-base joints (``prismatic_x``,
    ``prismatic_y``, ``revolute_z``) regardless of their order in
    :attr:`cfg.joint_names`. The dummy-base joints themselves live above the
    yaw joint in the URDF and therefore move the robot in world x/y/yaw.
    """

    cfg: actions_cfg.HolonomicActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    _action_space: spaces.Box
    """The action space of the action term."""

    def __init__(self, cfg: actions_cfg.HolonomicActionCfg, env: IsaacRLEnv):
        super().__init__(cfg, env)
        # 先获取 base 关节的 joint_ids，然后只修改这些关节的 stiffness/damping
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names
        )
        # 只修改 base 关节的 stiffness/damping，而不是所有关节
        # 这样不会影响 arm 关节的位置控制能力
        self._asset.write_joint_stiffness_to_sim(
            stiffness=0.0, joint_ids=self._joint_ids
        )
        self._asset.write_joint_damping_to_sim(
            damping=100000.0, joint_ids=self._joint_ids
        )

        # Resolve which column of self._processed_actions (ordered as
        # self._joint_ids / self._joint_names) corresponds to the dummy x / y
        # / yaw joints so we can write a base-frame [vx, vy, wz] input in any
        # joint_names ordering.
        def _find_col(substr: str) -> int:
            for i, n in enumerate(self._joint_names):
                if substr in n:
                    return i
            raise ValueError(
                f"HolonomicAction: no joint matching '{substr}' in {self._joint_names}"
            )

        self._col_x = _find_col("prismatic_x")
        self._col_y = _find_col("prismatic_y")
        self._col_z = _find_col("revolute_z")

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )  # base-frame [vx_b, vy_b, wz_b]
        self._processed_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        # save the scale as tensors
        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self.action_dim, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict."
                )

        assert self.cfg.action_space.shape[0] == 2, (
            "Expected action space to be of shape (2, action_dim)."
        )
        assert self.cfg.action_space.shape[1] == self.action_dim, (
            f"Expected action space to be of shape (2, {self.action_dim})."
        )

        if self.cfg.clip is not None:
            self.cfg.action_space = torch.clamp(
                self.cfg.action_space, min=self.cfg.clip[:, 0], max=self.cfg.clip[:, 1]
            )
        self._action_space = spaces.Box(
            low=self.cfg.action_space[0].cpu().numpy(),
            high=self.cfg.action_space[1].cpu().numpy(),
        )

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def process_actions(self, actions, env_ids=None):
        if env_ids is None:
            selected_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                selected_env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                selected_env_ids = env_ids
        actions, self._env_ids = self._handle_nan_actions(actions, selected_env_ids)
        if actions is None or len(self._env_ids) == 0:
            return
        assert actions.shape[0] == len(self._env_ids), (
            f"Expected actions shape[0] to be {len(self._env_ids)}, but got {actions.shape[0]}"
        )

        # store the raw actions
        self._raw_actions[self.env_ids] = actions
        # apply affine transformations (in-place)
        self._processed_actions[self.env_ids] = self._raw_actions[self.env_ids]

        if self.cfg.clip is not None:
            self._processed_actions[self.env_ids] = torch.clamp(
                self._processed_actions[self.env_ids],
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

    def apply_actions(self):
        """Rotate base-frame ``[vx_b, vy_b, wz_b]`` into world frame and
        write it onto the dummy-base prismatic / revolute joints.

        The raw action is interpreted as ``action[:, 0] = vx_b``,
        ``action[:, 1] = vy_b``, ``action[:, 2] = wz_b`` regardless of the
        order in ``cfg.joint_names``; the output is placed into the correct
        column of the joint-target tensor using the pre-resolved
        ``_col_x / _col_y / _col_z`` indices.
        """
        env_ids = self.env_ids
        action = self._processed_actions[env_ids]  # [M, 3]  base-frame

        vx_b = action[:, 0]
        vy_b = action[:, 1]
        wz_b = action[:, 2]

        # Current yaw from root quaternion (w, x, y, z).
        root_quat = self._asset.data.root_quat_w[env_ids]
        _, _, yaw = math_utils.euler_xyz_from_quat(root_quat)
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)

        vx_w = cos_y * vx_b - sin_y * vy_b
        vy_w = sin_y * vx_b + cos_y * vy_b
        # Dummy revolute_z axis is world z, so wz_b (about base z) == wz_w.
        wz_w = wz_b

        out = torch.empty_like(action)
        out[:, self._col_x] = vx_w
        out[:, self._col_y] = vy_w
        out[:, self._col_z] = wz_w

        self._asset.set_joint_velocity_target(
            target=out,
            joint_ids=self._joint_ids,
            env_ids=env_ids,
        )


class HolonomicForQuadrupedAction(ActionTerm):
    """Applies a differential controller to compute left/right wheel speeds from (v, ω)."""

    cfg: actions_cfg.HolonomicForQuadrupedActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    _action_space: spaces.Box
    """The action space of the action term."""

    def __init__(
        self, cfg: actions_cfg.HolonomicForQuadrupedActionCfg, env: IsaacRLEnv
    ):
        super().__init__(cfg, env)
        # self._asset.write_joint_stiffness_to_sim(stiffness=0.0)
        # self._asset.write_joint_damping_to_sim(damping=100000.0)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names
        )

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )  # [v, ω]
        self._processed_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        # save the scale as tensors
        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self.action_dim, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict."
                )

        assert self.cfg.action_space.shape[0] == 2, (
            "Expected action space to be of shape (2, action_dim)."
        )
        assert self.cfg.action_space.shape[1] == self.action_dim, (
            f"Expected action space to be of shape (2, {self.action_dim})."
        )

        if self.cfg.clip is not None:
            self.cfg.action_space = torch.clamp(
                self.cfg.action_space, min=self.cfg.clip[:, 0], max=self.cfg.clip[:, 1]
            )
        self._action_space = spaces.Box(
            low=self.cfg.action_space[0].cpu().numpy(),
            high=self.cfg.action_space[1].cpu().numpy(),
        )

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def process_actions(self, actions, env_ids=None):
        if env_ids is None:
            selected_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                selected_env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                selected_env_ids = env_ids
        actions, self._env_ids = self._handle_nan_actions(actions, selected_env_ids)
        if actions is None or len(self._env_ids) == 0:
            return
        assert actions.shape[0] == len(self._env_ids), (
            f"Expected actions shape[0] to be {len(self._env_ids)}, but got {actions.shape[0]}"
        )

        # store the raw actions
        self._raw_actions[self.env_ids] = actions
        # apply affine transformations
        self._processed_actions[self.env_ids] = self._raw_actions[self.env_ids]

        if self.cfg.clip is not None:
            self._processed_actions[self.env_ids] = torch.clamp(
                self._processed_actions[self.env_ids],
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

    def apply_actions(self):
        """
        将底盘期望速度 (vx, vy, wz) 转成 Go2 12 个腿关节的关节“位置目标”，并发给 sim。

        思路：
        - 第一次调用时记住当前关节角作为站立姿态 q0
        - 每一帧根据 (vx, vy, wz) 生成一个很小的关节角偏移 delta_q
        - q_des = q0 + delta_q
        - 用 set_joint_position_target 去跟踪 q_des（而不是纯速度）
        """
        device = self.device
        actions = self._processed_actions[self.env_ids]  # [N, 3] = (vx, vy, wz)
        N = actions.shape[0]

        if N == 0 or len(self._joint_ids) == 0:
            return

        # ---------------------------------------------------
        # 1) 底盘期望速度
        # ---------------------------------------------------
        vx = actions[:, 0:1]  # [N, 1] 方便广播
        vy = actions[:, 1:2]  # [N, 1]
        wz = actions[:, 2:3]  # [N, 1]

        num_joints = len(self._joint_ids)

        # ---------------------------------------------------
        # 2) 读取 / 缓存“站立姿态” q0
        # ---------------------------------------------------
        # 第一次调用时，把当前关节角作为 reference 姿态
        if (not hasattr(self, "_q0")) or (self._q0.shape[0] != self.num_envs):
            # IsaacLab 里的关节角一般在 robot.data.joint_pos
            q_all = self._asset.data.joint_pos.to(device)  # [num_envs, total_joints]
            self._q0 = q_all[
                :, self._joint_ids
            ].clone()  # 只取腿的 12 个关节 [num_envs, num_joints]

        # 只取当前 env_ids 对应的 q0（有可能不是所有 env 一起）
        q0 = self._q0[self.env_ids]  # [N, num_joints]

        # ---------------------------------------------------
        # 3) 解析每个关节属于哪条腿、哪个自由度（和你之前一样）
        # ---------------------------------------------------
        if not hasattr(self, "_joint_roles"):
            roles = []  # List[(leg, dof)]，leg ∈ {FL, FR, RL, RR}, dof ∈ {hip, thigh, calf}
            for name in self._joint_names:
                leg = None
                if "FL_" in name:
                    leg = "FL"
                elif "FR_" in name:
                    leg = "FR"
                elif "RL_" in name:
                    leg = "RL"
                elif "RR_" in name:
                    leg = "RR"

                dof = None
                if "hip_joint" in name:
                    dof = "hip"
                elif "thigh_joint" in name:
                    dof = "thigh"
                elif "calf_joint" in name:
                    dof = "calf"

                roles.append((leg, dof))

            self._joint_roles = roles
            # 调试可以看一下顺序：
            # print(list(zip(self._joint_names, self._joint_roles)))

        # 对不同腿设置一个简单的“符号”：
        #   forward_sign: 身体前进时，前腿/后腿反相
        #   lateral_sign: 身体侧移时，左右腿反相
        leg_sign = {
            "FL": (+1.0, +1.0),  # (forward_sign, lateral_sign)
            "FR": (+1.0, -1.0),
            "RL": (-1.0, +1.0),
            "RR": (-1.0, -1.0),
        }

        # ---------------------------------------------------
        # 4) 把 (vx, vy, wz) 映射成一个很小的关节角偏移 delta_q
        #    —— 注意这里是“位置偏移”，不要太大，否则会直接崩
        # ---------------------------------------------------
        delta_q = torch.zeros(N, num_joints, device=device)

        # 相比你之前的速度增益，这里减小很多，避免一下子蹬太狠
        k_hip_f, k_hip_lat, k_hip_yaw = 0.05, 0.03, 0.02
        k_thigh_f, k_thigh_lat, k_thigh_yaw = 0.06, 0.02, 0.02
        k_calf_f, k_calf_lat, k_calf_yaw = -0.04, -0.01, -0.01

        for j, (leg, dof) in enumerate(self._joint_roles):
            if leg is None or dof is None:
                continue

            f_sign, l_sign = leg_sign[leg]

            if dof == "hip":
                dq = (
                    k_hip_f * f_sign * vx
                    + k_hip_lat * l_sign * vy
                    + k_hip_yaw * f_sign * wz
                )
            elif dof == "thigh":
                dq = (
                    k_thigh_f * f_sign * vx
                    + k_thigh_lat * l_sign * vy
                    + k_thigh_yaw * f_sign * wz
                )
            else:  # calf
                dq = (
                    k_calf_f * f_sign * vx
                    + k_calf_lat * l_sign * vy
                    + k_calf_yaw * f_sign * wz
                )

            delta_q[:, j] = dq.squeeze(-1)

        # ---------------------------------------------------
        # 5) 得到目标关节角 q_des = q0 + delta_q，并适当 clamp 一下
        # ---------------------------------------------------
        q_des = q0 + delta_q

        # 简单的角度范围限制，先用一个保守范围，后面可以查真实 joint limit 替换
        q_min = -2.5
        q_max = 2.5
        q_des = torch.clamp(q_des, q_min, q_max)

        print("q_des (first env): ", q_des[0])

        # ---------------------------------------------------
        # 6) 发送关节“位置目标”，而不是纯速度
        # ---------------------------------------------------
        self._asset.set_joint_position_target(
            target=q_des,  # [N, num_joints]
            joint_ids=self._joint_ids,
            env_ids=self.env_ids,
        )

        # # 可选：把这些关节的速度目标拉回 0，防止别的地方还在写 velocity
        # self._asset.set_joint_velocity_target(
        #     target=torch.zeros_like(q_des),
        #     joint_ids=self._joint_ids,
        #     env_ids=self.env_ids,
        # )

    # def reset(self, env_ids: Sequence[int] | None = None) -> None:
    #     pass
    #     self._raw_actions[env_ids] = 0.0


class HolonomicVWAction(ActionTerm):
    cfg: actions_cfg.HolonomicVWActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    _action_space: spaces.Box
    """The action space of the action term."""

    def __init__(self, cfg: actions_cfg.HolonomicVWActionCfg, env: IsaacRLEnv):
        super().__init__(cfg, env)
        # resolve the joints over which the action term is applied

        self._asset.write_joint_stiffness_to_sim(stiffness=0.0)
        self._asset.write_joint_damping_to_sim(damping=100000.0)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names
        )

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )  # [v, ω]
        self._processed_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        # save the scale as tensors
        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self.action_dim, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict."
                )

        assert self.cfg.action_space.shape[0] == 2, (
            "Expected action space to be of shape (2, action_dim)."
        )
        assert self.cfg.action_space.shape[1] == self.action_dim, (
            f"Expected action space to be of shape (2, {self.action_dim})."
        )

        if self.cfg.clip is not None:
            self.cfg.action_space = torch.clamp(
                self.cfg.action_space, min=self.cfg.clip[:, 0], max=self.cfg.clip[:, 1]
            )
        self._action_space = spaces.Box(
            low=self.cfg.action_space[0].cpu().numpy(),
            high=self.cfg.action_space[1].cpu().numpy(),
        )

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def process_actions(self, actions, env_ids=None):
        if env_ids is None:
            selected_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                selected_env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                selected_env_ids = env_ids
        actions, self._env_ids = self._handle_nan_actions(actions, selected_env_ids)
        if actions is None or len(self._env_ids) == 0:
            return
        assert actions.shape[0] == len(self._env_ids), (
            f"Expected actions shape[0] to be {len(self._env_ids)}, but got {actions.shape[0]}"
        )

        # store the raw actions
        self._raw_actions[self.env_ids] = actions
        # apply affine transformations
        self._processed_actions[self.env_ids] = self._raw_actions[self.env_ids]

        if self.cfg.clip is not None:
            self._processed_actions[self.env_ids] = torch.clamp(
                self._processed_actions[self.env_ids],
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

    def apply_actions(self):
        """
        将 [v, ω] 转换为基座 3 个关节的速度：
        - dummy_base_prismatic_x_joint: x_dot (世界坐标系)
        - dummy_base_prismatic_y_joint: y_dot (世界坐标系)
        - dummy_base_revolute_z_joint: yaw_dot

        这里假设:
        - _processed_actions: [B, 2]，每行为 [v, ω]，v 为车体前向速度（body x 轴）
        - _asset.data.root_quat: [num_envs, 4]，(x, y, z, w) 格式
        """
        device = self.device
        # print("joint_stiffness: ", self._asset.data.joint_stiffness)
        # print("joint_damping: ", self._asset.data.joint_damping)
        # 当前这一步控制的 env id 列表，在 process_actions 里已经设置好了
        # print("env_ids: ", self.env_ids)
        # actions: [B, 2] -> v, ω
        actions = self._processed_actions[self.env_ids].to(device)
        v = actions[:, 0]  # [B]
        w = actions[:, 1]  # [B]

        # 取当前 base 的四元数 (假设为 (x,y,z,w) 格式)
        # Articulation 在 Isaac Lab 里通常有 data.root_quat

        # print("actions: ", actions)
        # root_pose = self._asset.data.root_link_pose_w[self._env_ids].clone()  # shape: (num_envs, 6)
        joint_pos = self._asset.data.joint_pos[self.env_ids].to(
            device
        )  # [num_envs, num_dofs]
        yaw_joint_id = self._joint_ids[2]
        yaw = joint_pos[:, yaw_joint_id]  # [num_envs]
        # yaw = yaw_all[env_ids]  # [B]

        # print("yaw: ", yaw)

        # 2) body-frame v 映射到 world-frame x_dot, y_dot
        x_dot = v * torch.cos(yaw)
        y_dot = v * torch.sin(yaw)
        yaw_dot = w

        # 3) 关节顺序是 [prismatic_y, prismatic_x, revolute_z]
        joint_vel_target = torch.stack([x_dot, y_dot, yaw_dot], dim=1)  # [B, 3]

        self._processed_actions[self.env_ids] = joint_vel_target
        self._asset.set_joint_velocity_target(
            target=joint_vel_target,
            joint_ids=self._joint_ids,
            env_ids=self.env_ids,
        )

    # def reset(self, env_ids: Sequence[int] | None = None) -> None:
    #     pass
    #     self._raw_actions[env_ids] = 0.0


class DifferentialAction(ActionTerm):
    """Applies a differential controller to compute left/right wheel speeds from (v, ω)."""

    cfg: actions_cfg.DifferentialActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    _action_space: spaces.Box
    """The action space of the action term."""

    def __init__(self, cfg: actions_cfg.DifferentialActionCfg, env: IsaacRLEnv):
        super().__init__(cfg, env)
        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names
        )

        self._num_joints = len(self._joint_ids)
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )  # [v, ω]
        self._processed_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        # save the scale as tensors
        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self.action_dim, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict."
                )

        assert self.cfg.action_space.shape[0] == 2, (
            "Expected action space to be of shape (2, action_dim)."
        )
        assert self.cfg.action_space.shape[1] == self.action_dim, (
            f"Expected action space to be of shape (2, {self.action_dim})."
        )

        if self.cfg.clip is not None:
            self.cfg.action_space = torch.clamp(
                self.cfg.action_space, min=self.cfg.clip[:, 0], max=self.cfg.clip[:, 1]
            )
        self._action_space = spaces.Box(
            low=self.cfg.action_space[0].cpu().numpy(),
            high=self.cfg.action_space[1].cpu().numpy(),
        )

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def process_actions(self, actions, env_ids=None):
        """Convert [v, ω] commands to left/right wheel speeds."""
        if env_ids is None:
            selected_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                selected_env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                selected_env_ids = env_ids
        actions, self._env_ids = self._handle_nan_actions(actions, selected_env_ids)
        if actions is None or len(self._env_ids) == 0:
            return
        assert actions.shape[0] == len(self._env_ids), (
            f"Expected actions shape[0] to be {len(self._env_ids)}, but got {actions.shape[0]}"
        )

        # store the raw actions
        self._raw_actions[self.env_ids] = actions
        # apply affine transformations
        self._processed_actions[self.env_ids] = self._raw_actions[self.env_ids]

        if self.cfg.clip is not None:
            self._processed_actions[self.env_ids] = torch.clamp(
                self._processed_actions[self.env_ids],
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

    def apply_actions(self):
        """
        将 [v, ω] 转换为左右轮速度并应用
        """
        # 输入：(num_envs, 2) -> [v, ω]
        pa = self._processed_actions[self.env_ids]
        v = pa[:, 0]
        w = pa[:, 1]

        # 差速驱动运动学
        v_left = v - (w * self.cfg.wheel_base / 2.0)
        v_right = v + (w * self.cfg.wheel_base / 2.0)

        # 转换为角速度
        omega_left = v_left / self.cfg.wheel_radius
        omega_right = v_right / self.cfg.wheel_radius

        # ✅ 输出：(num_envs, 2) -> [左轮速度, 右轮速度]
        joint_vel_target = torch.stack([omega_left, omega_right], dim=1)

        self._processed_actions[self.env_ids] = joint_vel_target

        # 应用到关节
        self._asset.set_joint_velocity_target(
            target=joint_vel_target,  # shape: (num_envs, 2)
            joint_ids=self._joint_ids,
            env_ids=self._env_ids,
        )


class AckermannSteeringAction(ActionTerm):
    """
    Ackermann 转向控制器

    输入动作（2维）：
    - actions[:, 0]: 油门命令 [-1, 1]
        - 正值：前进
        - 负值：倒车
        - 0：停止
    - actions[:, 1]: 转向命令 [-1, 1]
        - 正值：左转
        - 负值：右转
        - 0：直行

    输出：
    - 四个轮子的角速度（所有轮子相同）
    - 前轮的转向角度（左右轮可能不同，遵循 Ackermann 几何）
    """

    cfg: actions_cfg.AckermannSteeringActionCfg

    _asset: Articulation

    _clip: torch.Tensor
    """The clip applied to the input action."""

    _action_space: spaces.Box
    """The action space of the action term."""

    def __init__(self, cfg: actions_cfg.AckermannSteeringActionCfg, env: IsaacRLEnv):
        super().__init__(cfg, env)

        # 查找轮子关节
        self._wheel_joint_ids, self._wheel_joint_names = self._asset.find_joints(
            self.cfg.wheel_joint_names
        )

        # 查找转向关节
        self._steering_joint_ids, self._steering_joint_names = self._asset.find_joints(
            self.cfg.steering_joint_names
        )

        # 验证关节数量
        assert len(self._wheel_joint_ids) == 4, (
            f"Expected 4 wheel joints, got {len(self._wheel_joint_ids)}"
        )
        assert len(self._steering_joint_ids) == 2, (
            f"Expected 2 steering joints, got {len(self._steering_joint_ids)}"
        )

        # 动作缓冲区
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self.action_dim, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict."
                )

        assert self.cfg.action_space.shape[0] == 2, (
            "Expected action space to be of shape (2, action_dim)."
        )
        assert self.cfg.action_space.shape[1] == self.action_dim, (
            f"Expected action space to be of shape (2, {self.action_dim})."
        )

        if self.cfg.clip is not None:
            self.cfg.action_space = torch.clamp(
                self.cfg.action_space, min=self.cfg.clip[:, 0], max=self.cfg.clip[:, 1]
            )
        self._action_space = spaces.Box(
            low=self.cfg.action_space[0].cpu().numpy(),
            high=self.cfg.action_space[1].cpu().numpy(),
        )

        # 打印配置信息
        # print(f"\n{'='*60}")
        # print(f"Ackermann Steering Controller 初始化:")
        # print(f"{'='*60}")
        # print(f"轮子关节 ({len(self._wheel_joint_ids)}):")
        # for i, name in enumerate(self._wheel_joint_names):
        #     print(f"  [{i}] {name}")
        # print(f"转向关节 ({len(self._steering_joint_ids)}):")
        # for i, name in enumerate(self._steering_joint_names):
        #     print(f"  [{i}] {name}")
        # print(f"\n车辆参数:")
        # print(f"  轮子半径: {self.cfg.wheel_radius} m")
        # print(f"  轴距: {self.cfg.wheel_base} m")
        # print(f"  轮距: {self.cfg.track_width} m")
        # print(f"  最大速度: {self.cfg.max_speed} m/s")
        # print(f"  最大转向角: {self.cfg.max_steering_angle} rad ({self.cfg.max_steering_angle * 57.3:.1f} deg)")
        # print(f"  使用 Ackermann 几何: {self.cfg.use_ackermann_geometry}")
        # print(f"{'='*60}\n")

    @property
    def action_dim(self) -> int:
        """Ackermann 转向只需要 2 个输入：油门和转向角"""
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def process_actions(self, actions, env_ids=None):
        """
        处理动作输入

        Args:
            actions: shape (num_envs, 2) 的动作张量
                - actions[:, 0]: 油门命令 [-1, 1]
                - actions[:, 1]: 转向命令 [-1, 1]
            env_ids: 环境ID列表，如果为None则使用所有环境
        """
        if env_ids is None:
            selected_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                selected_env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                selected_env_ids = env_ids
        actions, self._env_ids = self._handle_nan_actions(actions, selected_env_ids)
        if actions is None or len(self._env_ids) == 0:
            return

        # 验证动作维度
        assert actions.shape[0] == len(self._env_ids), (
            f"Expected actions shape[0] ({actions.shape[0]}) to match env_ids length ({len(self._env_ids)})"
        )
        assert actions.shape[1] == self.action_dim, (
            f"Expected actions shape[1] to be {self.action_dim} (throttle, steering), but got {actions.shape[1]}"
        )

        # 存储原始动作（只更新选定的环境）
        self._raw_actions[self._env_ids] = actions

        # 应用裁剪（如果有配置）
        if self.cfg.clip is not None:
            self._raw_actions[self._env_ids] = torch.clamp(
                self._raw_actions[self._env_ids],
                min=self._clip[self._env_ids, :, 0],
                max=self._clip[self._env_ids, :, 1],
            )

    def apply_actions(self):
        """
        应用 Ackermann 转向控制

        输入动作：
        - throttle_cmd: 油门命令 [-1, 1]，负值表示倒车
        - steering_cmd: 转向命令 [-1, 1]，正值表示左转
        """
        # 提取动作（只使用前两个维度）
        throttle_cmd = self._raw_actions[
            self._env_ids, 0
        ]  # shape: (num_selected_envs,)
        steering_cmd = self._raw_actions[
            self._env_ids, 1
        ]  # shape: (num_selected_envs,)

        device = self._raw_actions.device
        dtype = self._raw_actions.dtype

        # ========== 1. 计算轮子角速度 ==========
        # 将油门命令缩放到线速度
        linear_speed = throttle_cmd * self.cfg.max_speed

        # 参数检查
        if self.cfg.wheel_radius <= 0.0:
            raise ValueError(
                f"wheel_radius 必须是正数，当前为: {self.cfg.wheel_radius}"
            )

        # 线速度转换为轮子角速度: ω = v / r
        wheel_angular_speed = linear_speed / self.cfg.wheel_radius

        # 所有轮子使用相同的角速度（Ackermann 转向的特性）
        wheel_velocities = torch.stack(
            [
                wheel_angular_speed,  # Front Left
                wheel_angular_speed,  # Front Right
                wheel_angular_speed,  # Rear Left
                wheel_angular_speed,  # Rear Right
            ],
            dim=1,
        )  # shape: (num_selected_envs, 4)

        # ========== 2. 计算转向角度 ==========
        # 将转向命令缩放到目标角度
        target_steering_angle = steering_cmd * self.cfg.max_steering_angle

        if self.cfg.use_ackermann_geometry:
            # Ackermann 几何：左右轮转向角不同
            if self.cfg.wheel_base <= 0.0 or self.cfg.track_width <= 0.0:
                raise ValueError("wheel_base 和 track_width 必须是正数")

            wheel_base = torch.tensor(self.cfg.wheel_base, device=device, dtype=dtype)
            track_width = torch.tensor(self.cfg.track_width, device=device, dtype=dtype)
            half_track = track_width / 2.0
            eps = 1e-6

            # 计算转弯半径: R = L / tan(δ)
            tan_steering = torch.tan(target_steering_angle)
            is_straight = torch.abs(tan_steering) < eps

            turn_radius = torch.where(
                is_straight,
                torch.sign(tan_steering) * 1e9,  # 直行时使用很大的半径
                wheel_base / (tan_steering + eps * torch.sign(tan_steering)),
            )

            # 计算内外轮半径
            inner_radius = torch.clamp(torch.abs(turn_radius) - half_track, min=eps)
            outer_radius = torch.abs(turn_radius) + half_track
            turn_sign = torch.sign(turn_radius)

            # 计算内外轮转向角
            angle_inner = torch.atan2(wheel_base, inner_radius) * turn_sign
            angle_outer = torch.atan2(wheel_base, outer_radius) * turn_sign

            # 根据转向方向分配左右轮
            is_left_turn = turn_radius > 0
            final_left = torch.where(is_left_turn, angle_inner, angle_outer)
            final_right = torch.where(is_left_turn, angle_outer, angle_inner)

            steering_angles = torch.stack([final_left, final_right], dim=1)
        else:
            # 简化模式：左右轮转向角相同
            steering_angles = torch.stack(
                [target_steering_angle, target_steering_angle], dim=1
            )

        # ========== 3. 应用命令到关节 ==========
        self._asset.set_joint_velocity_target(
            target=wheel_velocities,
            joint_ids=self._wheel_joint_ids,
            env_ids=self._env_ids,
        )

        self._asset.set_joint_position_target(
            target=steering_angles,
            joint_ids=self._steering_joint_ids,
            env_ids=self._env_ids,
        )
