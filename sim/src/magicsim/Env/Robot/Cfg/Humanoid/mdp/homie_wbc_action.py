from __future__ import annotations

from gymnasium import spaces

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from magicsim.Env.Robot.mdp.action_manager import ActionTerm

from magicsim.Env.Robot.Cfg.Humanoid.whole_body_controller.Homie.policy_constants import (
    NUM_BASE_HEIGHT_CMD,
    NUM_NAVIGATE_CMD,
    NUM_TORSO_ORIENTATION_RPY_CMD,
)
from magicsim.Env.Robot.Cfg.Humanoid.whole_body_controller.wbc_policy_factory import (
    get_wbc_policy,
)
from magicsim.Env.Robot.Cfg.Humanoid.whole_body_controller.Homie.utils import (
    prepare_observations,
)

if TYPE_CHECKING:
    from magicsim.Env.Environment.Isaac.IsaacRLEnv import IsaacRLEnv

    from magicsim.Env.Robot.Cfg.Humanoid.mdp.homie_wbc_action_cfg import (
        HomieWBCActionCfg,
    )


class HomieWBCAction(ActionTerm):
    """Action term for the G1 decoupled WBC policy. Upper body direct joint position control, lower body RL-based policy."""

    cfg: HomieWBCActionCfg
    _asset: Articulation
    _action_space: spaces.Box
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: HomieWBCActionCfg, env: IsaacRLEnv):
        """Initialize the action term.
        Args:
            cfg: The configuration for this action term.
            env: The environment in which the action term will be applied.
        """
        self.step_count = 0
        super().__init__(cfg, env)

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._raw_actions[:, 3] = 0.7

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)

        assert self._num_joints == self.cfg.num_wbc_joints, (
            f"Expected {self.cfg.num_wbc_joints} WBC joints, but got {self._num_joints}"
        )
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        self._processed_actions = torch.zeros(
            [self.num_envs, self._num_joints], device=self.device
        )

        self.wbc_policy = get_wbc_policy(
            self.cfg.robot_type, self.cfg.wbc_version, self.num_envs
        )

        self._wbc_goal = {
            # lin_vel_cmd_x, lin_vel_cmd_y, ang_vel_cmd
            "navigate_cmd": np.tile(np.array([[0.0, 0.0, 0.0]]), (self.num_envs, 1)),
            # base_height_cmd: 0.75 as pelvis height
            "base_height_command": np.tile(np.array([0.75]), (self.num_envs, 1)),
            # roll pitch yaw command
            "torso_orientation_rpy_cmd": np.tile(
                np.array([[0.0, 0.0, 0.0]]), (self.num_envs, 1)
            ),
        }

        self.wbc_g1_joints_order = self.wbc_policy.wbc_config.wbc_joints_order

        self._lower_body_joint_ids = self.get_lower_body_joint_ids(
            self.wbc_g1_joints_order
        )

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

    def get_lower_body_joint_ids(
        self, wbc_g1_joints_order: dict[str, int]
    ) -> list[int]:
        """Get the lower body joint ids from the WBC G1 joints order."""
        lower_body_joint_ids = []
        for wbc_joint_name, wbc_joint_index in wbc_g1_joints_order.items():
            if wbc_joint_index > 14:
                break
            if wbc_joint_name not in self._asset.data.joint_names:
                print(f"Joint {wbc_joint_name} not found in asset")
                continue

            sim_joint_index = self._asset.data.joint_names.index(wbc_joint_name)
            lower_body_joint_ids.append(sim_joint_index)
        return lower_body_joint_ids

    # Properties.
    # """
    @property
    def num_joints(self) -> int:
        """Get the number of joints."""
        return self._num_joints

    @property
    def navigate_cmd_dim(self) -> int:
        """Dimension of navigation command."""
        return NUM_NAVIGATE_CMD

    @property
    def base_height_cmd_dim(self) -> int:
        """Dimension of base height command."""
        return NUM_BASE_HEIGHT_CMD

    @property
    def torso_orientation_rpy_cmd_dim(self) -> int:
        """Dimension of torso orientation command."""
        return NUM_TORSO_ORIENTATION_RPY_CMD

    @property
    def action_dim(self) -> int:
        """Dimension of the action space (based on number of tasks and pose dimension)."""
        return NUM_NAVIGATE_CMD + NUM_BASE_HEIGHT_CMD + NUM_TORSO_ORIENTATION_RPY_CMD

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
    def get_wbc_policy(self):
        return self.wbc_policy

    @property
    def get_wbc_goal(self):
        return self._wbc_goal

    def set_wbc_goal(
        self,
        navigate_cmd: torch.Tensor,
        base_height_cmd: torch.Tensor,
        torso_orientation_rpy_cmd: torch.Tensor = None,
    ):
        self._wbc_goal["navigate_cmd"] = navigate_cmd.cpu().numpy()
        self._wbc_goal["base_height_command"] = base_height_cmd.cpu().numpy()
        if self.cfg.wbc_version == "homie_v2" and torso_orientation_rpy_cmd is not None:
            self._wbc_goal["torso_orientation_rpy_cmd"] = (
                torso_orientation_rpy_cmd.cpu().numpy()
            )
        assert self._wbc_goal["navigate_cmd"].shape == (self.num_envs, NUM_NAVIGATE_CMD)
        assert self._wbc_goal["base_height_command"].shape == (
            self.num_envs,
            NUM_BASE_HEIGHT_CMD,
        )
        assert self._wbc_goal["torso_orientation_rpy_cmd"].shape == (
            self.num_envs,
            NUM_TORSO_ORIENTATION_RPY_CMD,
        )

    def get_navigation_cmd_from_actions(self, actions: torch.Tensor):
        """Get the navigation command from the actions."""
        return actions[
            :,
            -NUM_NAVIGATE_CMD
            - NUM_BASE_HEIGHT_CMD
            - NUM_TORSO_ORIENTATION_RPY_CMD : -NUM_BASE_HEIGHT_CMD
            - NUM_TORSO_ORIENTATION_RPY_CMD,
        ]

    def get_base_height_cmd_from_actions(self, actions: torch.Tensor):
        """Get the base height command from the actions."""
        return actions[
            :,
            -NUM_BASE_HEIGHT_CMD
            - NUM_TORSO_ORIENTATION_RPY_CMD : -NUM_TORSO_ORIENTATION_RPY_CMD,
        ]

    def get_torso_orientation_rpy_cmd_from_actions(self, actions: torch.Tensor):
        """Get the torso orientation command from the actions."""
        return actions[:, -NUM_TORSO_ORIENTATION_RPY_CMD:]

    # """
    # Operations.
    # """

    def process_actions(
        self, actions: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        """Process the input actions and set targets for each task.

        Args:
            actions: The input actions tensor.
            env_ids: The environment indices to process. If None, all environments are processed.
        """

        if self.step_count % self.cfg.decimation != 0:
            return

        work = actions[:, : self.action_dim].clone()
        prev = self._raw_actions if env_ids is None else self._raw_actions[env_ids]
        nan_mask = torch.isnan(work)
        if nan_mask.any():
            work[nan_mask] = prev[nan_mask]

        if env_ids is None:
            self._raw_actions[:] = work
        else:
            self._raw_actions[env_ids] = work

        # WBC pipeline operates on all envs
        actions_clone = self._raw_actions.clone()
        """
        **************************************************
        WBC closedloop
        **************************************************
        """
        # extract navigate_cmd  base_height_cmd, and torso_orientation_rpy_cmd from actions
        navigate_cmd = self.get_navigation_cmd_from_actions(actions_clone)
        base_height_cmd = self.get_base_height_cmd_from_actions(actions_clone)
        torso_orientation_rpy_cmd = self.get_torso_orientation_rpy_cmd_from_actions(
            actions_clone
        )

        self.set_wbc_goal(navigate_cmd, base_height_cmd, torso_orientation_rpy_cmd)
        self.wbc_policy.set_goal(self._wbc_goal)

        """
        **************************************************
        Prepare WBC policy input
        **************************************************
        """
        wbc_obs = prepare_observations(
            self.num_envs, self._asset.data, self.wbc_g1_joints_order
        )
        self.wbc_policy.set_observation(wbc_obs)

        wbc_action = self.wbc_policy.get_action()
        self._processed_actions[:] = torch.from_numpy(wbc_action).to(self.device)
        assert self._processed_actions.shape[0] == self.num_envs, (
            f"Expected processed actions shape[0] to be {self.num_envs}, but got {self._processed_actions.shape[0]}"
        )
        assert self._processed_actions.shape[1] == self._num_joints, (
            f"Expected processed actions shape[1] to be {self._num_joints}, but got {self._processed_actions.shape[1]}"
        )

    def apply_actions(self):
        """Apply the computed joint positions based on the WBC solution."""
        self.step_count += 1
        assert self._processed_actions.shape[0] == self.num_envs, (
            f"Expected processed actions shape[0] to be {self.num_envs}, but got {self._processed_actions.shape[0]}"
        )
        assert self._processed_actions.shape[1] == self._num_joints, (
            f"Expected processed actions shape[1] to be {self._num_joints}, but got {self._processed_actions.shape[1]}"
        )
        self._asset.set_joint_position_target(
            self._processed_actions[:, :-3], self._lower_body_joint_ids[:-3]
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term for specified environments.
        Args:
            env_ids: A list of environment IDs to reset. If None, all environments are reset.
        """
        self._raw_actions[env_ids] = torch.zeros(self.action_dim, device=self.device)
        self.wbc_policy.reset(env_ids=env_ids)
