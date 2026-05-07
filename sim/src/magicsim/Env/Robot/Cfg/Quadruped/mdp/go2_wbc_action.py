from __future__ import annotations

from gymnasium import spaces

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from magicsim.Env.Robot.mdp.action_manager import ActionTerm
from magicsim.Env.Robot.Cfg.Quadruped.whole_body_controller.wbc_policy_factory import (
    get_wbc_policy,
)
from magicsim.Env.Robot.Cfg.Quadruped.whole_body_controller.Go2WBC.utils import (
    prepare_observations,
)

if TYPE_CHECKING:
    from magicsim.Env.Environment.Isaac.IsaacRLEnv import IsaacRLEnv

    from magicsim.Env.Robot.Cfg.Quadruped.mdp.go2_wbc_action_cfg import (
        Go2WBCActionCfg,
    )


# Constants for Go2 WBC action dimensions
NUM_VELOCITY_CMD = 2  # [vx, vy] - linear velocity commands
NUM_BASE_HEIGHT_CMD = 1  # base height command
NUM_YAW_RATE_CMD = 1  # yaw rate command


class Go2WBCAction(ActionTerm):
    """Action term for the Go2 quadruped WBC policy."""

    cfg: Go2WBCActionCfg
    _asset: Articulation
    _action_space: spaces.Box
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: Go2WBCActionCfg, env: IsaacRLEnv):
        """Initialize the action term.
        Args:
            cfg: The configuration for this action term.
            env: The environment in which the action term will be applied.
        """
        super().__init__(cfg, env)

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        print(
            f"Go2 WBC - joint_ids: {self._joint_ids}, joint_names: {self._joint_names}"
        )
        assert self._num_joints == self.cfg.num_wbc_joints, (
            f"Expected {self.cfg.num_wbc_joints} WBC joints, but got {self._num_joints}"
        )
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        self._processed_actions = torch.zeros(
            [self.num_envs, self._num_joints], device=self.device
        )

        # Initialize WBC policy
        self.wbc_policy = get_wbc_policy(
            self.cfg.robot_type, self.cfg.wbc_version, self.num_envs
        )

        self.wbc_go2_joints_order = self.wbc_policy.wbc_config.wbc_joints_order

        self._wbc_goal = {
            # lin_vel_cmd_x, lin_vel_cmd_y
            "velocity_cmd": np.tile(np.array([[0.0, 0.0]]), (self.num_envs, 1)),
            # base_height_cmd: default height for Go2
            "base_height_command": np.tile(np.array([0.4]), (self.num_envs, 1)),
            # yaw_rate_cmd
            "yaw_rate_cmd": np.tile(np.array([0.0]), (self.num_envs, 1)),
        }

        self._action_space = self.cfg.action_space
        assert self._action_space.shape[0] == 2, (
            f"Expected action space to be of shape (2, {self.action_dim}), but got {self._action_space.shape}"
        )
        assert self._action_space.shape[1] == self.action_dim, (
            f"Expected action space to be of shape (2, {self.action_dim}), but got {self._action_space.shape}"
        )
        self._action_space = spaces.Box(
            low=self._action_space[0].cpu().numpy(),
            high=self._action_space[1].cpu().numpy(),
        )

    # Properties.
    @property
    def num_joints(self) -> int:
        """Get the number of joints."""
        return self._num_joints

    @property
    def velocity_cmd_dim(self) -> int:
        """Dimension of velocity command."""
        return NUM_VELOCITY_CMD

    @property
    def base_height_cmd_dim(self) -> int:
        """Dimension of base height command."""
        return NUM_BASE_HEIGHT_CMD

    @property
    def yaw_rate_cmd_dim(self) -> int:
        """Dimension of yaw rate command."""
        return NUM_YAW_RATE_CMD

    @property
    def action_dim(self) -> int:
        """Dimension of the action space."""
        return NUM_VELOCITY_CMD + NUM_BASE_HEIGHT_CMD + NUM_YAW_RATE_CMD

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def raw_actions(self) -> torch.Tensor:
        """Get the raw actions tensor."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Get the processed actions tensor."""
        return self._processed_actions

    @property
    def get_wbc_version(self):
        return self.cfg.wbc_version

    @property
    def get_wbc_goal(self):
        return self._wbc_goal

    def set_wbc_goal(
        self,
        velocity_cmd: torch.Tensor,
        base_height_cmd: torch.Tensor,
        yaw_rate_cmd: torch.Tensor = None,
    ):
        """Set the WBC goal commands."""
        self._wbc_goal["velocity_cmd"] = velocity_cmd.cpu().numpy()
        self._wbc_goal["base_height_command"] = base_height_cmd.cpu().numpy()
        if yaw_rate_cmd is not None:
            self._wbc_goal["yaw_rate_cmd"] = yaw_rate_cmd.cpu().numpy()
        assert self._wbc_goal["velocity_cmd"].shape == (self.num_envs, NUM_VELOCITY_CMD)
        assert self._wbc_goal["base_height_command"].shape == (
            self.num_envs,
            NUM_BASE_HEIGHT_CMD,
        )
        assert self._wbc_goal["yaw_rate_cmd"].shape == (
            self.num_envs,
            NUM_YAW_RATE_CMD,
        )

    def get_velocity_cmd_from_actions(self, actions: torch.Tensor):
        """Get the velocity command from the actions."""
        return actions[:, :NUM_VELOCITY_CMD]

    def get_base_height_cmd_from_actions(self, actions: torch.Tensor):
        """Get the base height command from the actions."""
        return actions[:, NUM_VELOCITY_CMD : NUM_VELOCITY_CMD + NUM_BASE_HEIGHT_CMD]

    def get_yaw_rate_cmd_from_actions(self, actions: torch.Tensor):
        """Get the yaw rate command from the actions."""
        return actions[:, -NUM_YAW_RATE_CMD:]

    # Operations.
    def process_actions(
        self, actions: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        """Process the input actions and set targets for each task.

        Args:
            actions: The input actions tensor.
            env_ids: The environment indices to process. If None, all environments are processed.
        """

        # Store the raw actions
        self._raw_actions[:] = actions[:, : self.action_dim]

        # Make a copy of actions before modifying so that raw actions are not modified
        actions_clone = actions.clone()

        # Extract commands from actions
        velocity_cmd = self.get_velocity_cmd_from_actions(actions_clone)
        base_height_cmd = self.get_base_height_cmd_from_actions(actions_clone)
        yaw_rate_cmd = self.get_yaw_rate_cmd_from_actions(actions_clone)

        self.set_wbc_goal(velocity_cmd, base_height_cmd, yaw_rate_cmd)
        self.wbc_policy.set_goal(self._wbc_goal)

        """
        **************************************************
        Prepare WBC policy input
        **************************************************
        """
        wbc_obs = prepare_observations(
            self.num_envs, self._asset.data, self.wbc_go2_joints_order
        )
        self.wbc_policy.set_observation(wbc_obs)

        wbc_action = self.wbc_policy.get_action()
        print(f"Go2 WBC - wbc_action: {wbc_action}")
        self._processed_actions = torch.from_numpy(wbc_action)

    def apply_actions(self):
        """Apply the computed joint positions based on the WBC solution."""
        assert self._processed_actions.shape[0] == self.num_envs, (
            f"Expected processed actions shape[0] to be {self.num_envs}, but got {self._processed_actions.shape[0]}"
        )
        assert self._processed_actions.shape[1] == self._num_joints, (
            f"Expected processed actions shape[1] to be {self._num_joints}, but got {self._processed_actions.shape[1]}"
        )
        self._processed_actions = self._processed_actions.to(self.device)
        self._asset.set_joint_position_target(self._processed_actions, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term for specified environments.
        Args:
            env_ids: A list of environment IDs to reset. If None, all environments are reset.
        """
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = torch.zeros(self.action_dim, device=self.device)
        self.wbc_policy.reset()
