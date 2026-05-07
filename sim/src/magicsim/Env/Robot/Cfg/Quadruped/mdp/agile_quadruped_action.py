# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from gymnasium import spaces
import torch
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io.torchscript import load_torchscript_model

if TYPE_CHECKING:
    from magicsim.Env.Environment.Isaac.IsaacRLEnv import IsaacRLEnv

    from magicsim.Env.Robot.Cfg.Quadruped.mdp.agile_quadruped_action_cfg import (
        AgileQuadrupedActionCfg,
    )


class AgileQuadrupedAction(ActionTerm):
    """Action term that is based on Agile quadruped RL policy."""

    cfg: AgileQuadrupedActionCfg
    """The configuration of the action term."""

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: AgileQuadrupedActionCfg, env: IsaacRLEnv):
        super().__init__(cfg, env)

        # Load policy here if needed
        _temp_policy_path = retrieve_file_path(cfg.policy_path)
        self._policy = load_torchscript_model(_temp_policy_path, device=env.device)
        self._env = env

        # Find joint ids for the leg joints
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names
        )
        self._num_joints = len(self._joint_ids)

        assert self._num_joints == cfg.num_wbc_joints, (
            f"Expected {cfg.num_wbc_joints} joints, but got {self._num_joints}"
        )

        # Find joint ids for observation joints (matching typical quadruped observation configs)
        observation_joint_names = [
            ".*_hip_joint",
            ".*_thigh_joint",
            ".*_calf_joint",
        ]
        self._obs_joint_ids, _ = self._asset.find_joints(observation_joint_names)

        # Get the scale and offset from the configuration
        self._policy_output_scale = torch.tensor(
            cfg.policy_output_scale, device=env.device
        )
        self._policy_output_offset = self._asset.data.default_joint_pos[
            :, self._joint_ids
        ].clone()

        # Create tensors to store raw and processed actions
        self._raw_actions = torch.zeros(
            self.num_envs, len(self._joint_ids), device=self.device
        )
        self._processed_actions = torch.zeros(
            self.num_envs, len(self._joint_ids), device=self.device
        )

        # Initialize clip if configured
        self._clip = None
        if cfg.clip is not None:
            # Parse clip configuration (similar to other action terms)
            # This is a placeholder - adjust based on actual clip format needed
            self._clip = torch.zeros(
                (self.num_envs, len(self._joint_ids), 2), device=self.device
            )
            # Set clip values based on cfg.clip
            # TODO: Implement proper clip parsing based on cfg.clip format

        self._action_space = cfg.action_space
        self._action_space = spaces.Box(
            low=self._action_space[0].cpu().numpy(),
            high=self._action_space[1].cpu().numpy(),
        )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Quadruped Action: [vx, vy, wz, base_height]"""
        return 4

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _compose_policy_input(
        self, base_command: torch.Tensor, obs_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Compose the policy input by concatenating repeated commands with observations.

        Args:
            base_command: The base command tensor [vx, vy, wz, base_height].
            obs_tensor: The observation tensor from the environment.

        Returns:
            The composed policy input tensor with repeated commands concatenated to observations.
        """

        history_length = 1

        # Repeat commands based on history length and concatenate with observations
        repeated_commands = (
            base_command.unsqueeze(1)
            .repeat(1, history_length, 1)
            .reshape(base_command.shape[0], -1)
        )
        policy_input = torch.cat([repeated_commands, obs_tensor], dim=-1)

        return policy_input

    def get_observation(self) -> torch.Tensor:
        """Get the observation for the action term by extracting from asset.data."""
        # Extract observations from asset.data
        # Base linear velocity in body frame
        base_lin_vel = self._asset.data.root_lin_vel_b  # [num_envs, 3]

        # Base angular velocity in body frame
        base_ang_vel = self._asset.data.root_ang_vel_b  # [num_envs, 3]

        # Projected gravity in body frame
        projected_gravity = self._asset.data.projected_gravity_b  # [num_envs, 3]

        # Joint positions relative to default (for observation joints)
        joint_pos_rel = (
            self._asset.data.joint_pos[:, self._obs_joint_ids]
            - self._asset.data.default_joint_pos[:, self._obs_joint_ids]
        )  # [num_envs, num_obs_joints]

        # Joint velocities relative to default (for observation joints)
        joint_vel_rel = (
            self._asset.data.joint_vel[:, self._obs_joint_ids]
            - self._asset.data.default_joint_vel[:, self._obs_joint_ids]
        )  # [num_envs, num_obs_joints]
        # Scale joint velocities by 0.1 (as in typical quadruped observation configs)
        joint_vel_rel = joint_vel_rel * 0.1

        # Last action (leg joint positions from previous step)
        # This is already stored in self._raw_actions which gets updated in process_actions
        last_action = self._raw_actions  # [num_envs, num_leg_joints]

        # Concatenate all observations
        obs_tensor = torch.cat(
            [
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                joint_pos_rel,
                joint_vel_rel,
                last_action,
            ],
            dim=-1,
        )

        return obs_tensor

    def process_actions(self, actions: torch.Tensor):
        """Process the input actions using the locomotion policy.

        Args:
            actions: The quadruped commands [vx, vy, wz, base_height].
        """

        # Extract base command from the action tensor
        # Assuming the base command [vx, vy, wz, base_height]
        base_command = actions

        # Get observation tensor from asset.data
        obs_tensor = self.get_observation()

        # Compose policy input using helper function
        policy_input = self._compose_policy_input(base_command, obs_tensor)

        joint_actions = self._policy.forward(policy_input)

        # Store raw actions (used as last_action in next observation)
        self._raw_actions[:] = joint_actions

        # Apply scaling and offset to the raw actions from the policy
        self._processed_actions = (
            joint_actions * self._policy_output_scale + self._policy_output_offset
        )

        # Clip actions if configured
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

    def apply_actions(self):
        """Apply the actions to the environment."""
        # Store the raw actions
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )
