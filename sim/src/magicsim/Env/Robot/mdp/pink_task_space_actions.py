# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from dataclasses import MISSING

from pink.tasks import FrameTask

from gymnasium import spaces

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from magicsim.Env.Robot.mdp.pink_ik import LocalFrameTask, PinkIKController
from magicsim.Env.Robot.mdp.action_manager import ActionTerm
from magicsim.Env.Robot.mdp.differential_ik import DifferentialIKController

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import pink_actions_cfg


class PinkInverseKinematicsAction(ActionTerm):
    r"""Pink Inverse Kinematics action term.

    This action term processes the action tensor and sets these setpoints in the pink IK framework.
    The action tensor is ordered in the order of the tasks defined in PinkIKControllerCfg.
    """

    cfg: pink_actions_cfg.PinkInverseKinematicsActionCfg
    """Configuration for the Pink Inverse Kinematics action term."""

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(
        self, cfg: pink_actions_cfg.PinkInverseKinematicsActionCfg, env: ManagerBasedEnv
    ):
        """Initialize the Pink Inverse Kinematics action term.

        Args:
            cfg: The configuration for this action term.
            env: The environment in which the action term will be applied.
        """
        self._step_count = 0
        super().__init__(cfg, env)

        self._env = env
        self._sim_dt = env.sim.get_physics_dt()

        # Initialize joint information
        self._initialize_joint_info()

        # Initialize IK controllers
        self._initialize_ik_controllers()

        # Initialize action tensors
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        # Last passed (valid) action per env; used when input becomes NaN (first time: use current EEF)
        self._last_passed_action = torch.zeros_like(self._raw_actions)
        # _processed_actions stores joint positions output from IK, shape (num_envs, num_controlled_joints)
        self._processed_actions = torch.zeros(
            self.num_envs, len(self._controlled_joint_ids), device=self.device
        )

        # PhysX Articulation Floating joint indices offset from IsaacLab Articulation joint indices
        self._physx_floating_joint_indices_offset = 6

        # Initialize env_ids (all envs by default)
        self._env_ids = torch.arange(self.num_envs, device=self.device)

        # Pre-allocate tensors for runtime use
        self._initialize_helper_tensors()

        # Initialize action space
        self._action_space = spaces.Box(
            low=self.cfg.action_space[0].cpu().numpy(),
            high=self.cfg.action_space[1].cpu().numpy(),
            dtype=float,
        )

    def _initialize_joint_info(self) -> None:
        """Initialize joint IDs and names based on configuration."""
        # Resolve pink controlled joints
        self._isaaclab_controlled_joint_ids, self._isaaclab_controlled_joint_names = (
            self._asset.find_joints(self.cfg.pink_controlled_joint_names)
        )
        self.cfg.controller.joint_names = self._isaaclab_controlled_joint_names
        self._isaaclab_all_joint_ids = list(range(len(self._asset.data.joint_names)))
        self.cfg.controller.all_joint_names = self._asset.data.joint_names

        # Resolve hand joints (optional)
        if self.cfg.hand_joint_names not in (None, MISSING):
            self._hand_joint_ids, self._hand_joint_names = self._asset.find_joints(
                self.cfg.hand_joint_names
            )
        else:
            self._hand_joint_ids, self._hand_joint_names = [], []

        # Combine all joint information
        self._controlled_joint_ids = (
            self._isaaclab_controlled_joint_ids + self._hand_joint_ids
        )
        self._controlled_joint_names = (
            self._isaaclab_controlled_joint_names + self._hand_joint_names
        )
        assert len(self._controlled_joint_ids) == self.cfg.num_joints, (
            f"Number of controlled joints does not match the number of joints to control: {len(self._controlled_joint_ids)} != {self.cfg.num_joints}"
        )

    def _initialize_ik_controllers(self) -> None:
        """Initialize Pink IK controllers for all environments."""
        assert self._env.num_envs > 0, (
            "Number of environments specified are less than 1."
        )

        self._ik_controllers = []
        robot_cfg = self._asset.cfg
        for _ in range(self._env.num_envs):
            controller_cfg = self.cfg.controller.copy()
            # If urdf_path is not specified in controller config, try to get it from various sources
            if controller_cfg.urdf_path is None:
                # 1. Try to get from asset configuration
                if hasattr(robot_cfg, "urdf_path"):
                    controller_cfg.urdf_path = robot_cfg.urdf_path
                # 2. Try to get from spawn configuration if it's a UsdFileCfg (common in IsaacLab)
                elif hasattr(robot_cfg.spawn, "urdf_path"):
                    controller_cfg.urdf_path = robot_cfg.spawn.urdf_path
                # 3. Try to infer from G1 specific configs if available
                # Note: This is a fallback for G1 robot specific structure
                elif (
                    hasattr(robot_cfg, "spawn")
                    and hasattr(robot_cfg.spawn, "usd_path")
                    and "g1" in str(robot_cfg.spawn.usd_path).lower()
                ):
                    # Attempt to find URDF in expected location relative to asset root or known path
                    pass

            self._ik_controllers.append(
                PinkIKController(
                    cfg=controller_cfg,
                    robot_cfg=robot_cfg,
                    device=self.device,
                    controlled_joint_indices=self._isaaclab_controlled_joint_ids,
                )
            )

    def _initialize_helper_tensors(self) -> None:
        """Pre-allocate tensors and cache values for performance optimization."""
        # Cache frequently used tensor versions of joint IDs to avoid repeated creation
        self._controlled_joint_ids_tensor = torch.tensor(
            self._controlled_joint_ids, device=self.device
        )

        # Cache base link index to avoid string lookup every time
        if self.cfg.controller.base_link_name == "root":
            self._base_link_idx = None
        else:
            articulation_data = self._env.scene[
                self.cfg.controller.articulation_name
            ].data
            self._base_link_idx = articulation_data.body_names.index(
                self.cfg.controller.base_link_name
            )

        # Pre-allocate working tensors
        # Count only FrameTask instances in variable_input_tasks (not all tasks)
        num_frame_tasks = sum(
            1
            for task in self._ik_controllers[0].cfg.variable_input_tasks
            if isinstance(task, FrameTask)
        )
        self._num_frame_tasks = num_frame_tasks
        self._controlled_frame_poses = torch.zeros(
            num_frame_tasks, self.num_envs, 4, 4, device=self.device
        )

        # Pre-allocate tensor for base frame computations
        self._base_link_frame_buffer = torch.zeros(
            self.num_envs, 4, 4, device=self.device
        )

        # Cache EEF body indices for NaN action handling
        # Maps frame task index → body index in articulation data
        self._eef_body_indices: list[int] = []
        if self.cfg.target_eef_link_names not in (None, MISSING):
            for link_name in self.cfg.target_eef_link_names.values():
                body_idx = self._asset.data.body_names.index(link_name)
                self._eef_body_indices.append(body_idx)

    # ==================== Properties ====================

    @property
    def hand_joint_dim(self) -> int:
        """Dimension for hand joint positions."""
        return self.cfg.controller.num_hand_joints

    @property
    def position_dim(self) -> int:
        """Dimension for position (x, y, z)."""
        return 3

    @property
    def orientation_dim(self) -> int:
        """Dimension for orientation (w, x, y, z)."""
        return 4

    @property
    def pose_dim(self) -> int:
        """Total pose dimension (position + orientation)."""
        return self.position_dim + self.orientation_dim

    @property
    def action_dim(self) -> int:
        """Dimension of the action space (based on number of tasks and pose dimension)."""
        # Count only FrameTask instances in variable_input_tasks
        frame_tasks_count = sum(
            1
            for task in self._ik_controllers[0].cfg.variable_input_tasks
            if isinstance(task, FrameTask)
        )
        return frame_tasks_count * self.pose_dim + self.hand_joint_dim

    @property
    def env_ids(self) -> torch.Tensor:
        """Get the current valid environment IDs (excluding NaN action envs)."""
        return self._env_ids

    @property
    def raw_actions(self) -> torch.Tensor:
        """Get the raw actions tensor."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Get the processed actions tensor."""
        return self._processed_actions

    @property
    def action_space(self) -> spaces.Box:
        """Get the action space."""
        return self._action_space

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor of the action term.

        This descriptor is used to describe the action term of the pink inverse kinematics action.
        It adds the following information to the base descriptor:
        - scale: The scale of the action term.
        - offset: The offset of the action term.
        - clip: The clip of the action term.
        - pink_controller_joint_names: The names of the pink controller joints.
        - hand_joint_names: The names of the hand joints.
        - controller_cfg: The configuration of the pink controller.

        Returns:
            The IO descriptor of the action term.
        """
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "PinkInverseKinematicsAction"
        self._IO_descriptor.pink_controller_joint_names = (
            self._isaaclab_controlled_joint_names
        )
        self._IO_descriptor.hand_joint_names = self._hand_joint_names
        self._IO_descriptor.extras["controller_cfg"] = self.cfg.controller.__dict__
        return self._IO_descriptor

    # """
    # Operations.
    # """

    def _handle_nan_actions(
        self, actions: torch.Tensor, selected_env_ids: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Handle NaN actions: use last passed action if available, else current EEF pose.

        Actions are structured as groups of pose_dim (7) per frame task, optionally
        followed by hand joint values. The logic is:

        1. Include all envs by default (do not skip all-NaN envs).
        2. For each group of 7 that has any NaN: use last passed action if valid;
           else (no valid previous action) behavior depends on
           ``cfg.fallback_to_current``:
             - True (default): fall back to current EEF pose.
             - False: skip this env (drop it from the returned batch).
        3. For hand joints: same logic — last passed if valid, else current joint
           positions (or skip when ``fallback_to_current`` is False).

        Args:
            actions: Input action tensor of shape ``(len(selected_env_ids), action_dim)``.
                Rows are **already** the slice for those envs (same order as
                ``selected_env_ids``), as produced by ``ActionManager.process_action``.
                Do not index again with global env ids as row indices.
            selected_env_ids: Selected environment IDs (global indices).

        Returns:
            Tuple of (processed_actions, valid_env_ids).
            processed_actions is None if no valid envs exist.
        """
        env_ids = selected_env_ids
        if len(env_ids) == 0:
            return None, env_ids

        if actions.shape[0] != len(env_ids):
            raise ValueError(
                f"Pink IK actions batch ({actions.shape[0]}) must match len(env_ids) ({len(env_ids)}); "
                "actions should be the env slice in env_ids order, not the full buffer indexed by id."
            )

        valid_actions = actions.clone()
        fallback_to_current = self.cfg.fallback_to_current
        skip_mask = torch.zeros(
            len(env_ids), dtype=torch.bool, device=valid_actions.device
        )

        # For each group of pose_dim (7), fill NaN with last passed action or current EEF pose
        for task_idx in range(self._num_frame_tasks):
            start = task_idx * self.pose_dim
            end = start + self.pose_dim

            group_nan_mask = torch.isnan(valid_actions[:, start:end]).any(dim=1)
            if group_nan_mask.any():
                nan_env_ids = env_ids[group_nan_mask]
                last_poses = self._last_passed_action[nan_env_ids, start:end]
                has_valid_last = (~torch.isnan(last_poses)).all(dim=1) & (
                    last_poses[:, 3:7].abs().sum(dim=1) > 0.01
                )
                if fallback_to_current:
                    current_poses = self._get_current_eef_pose(nan_env_ids, task_idx)
                    fill_values = torch.where(
                        has_valid_last.unsqueeze(1).expand_as(current_poses),
                        last_poses,
                        current_poses,
                    )
                    valid_actions[group_nan_mask, start:end] = fill_values
                else:
                    # Use last_poses only where valid; mark the rest for skipping.
                    valid_actions[group_nan_mask, start:end] = last_poses
                    group_local_idx = group_nan_mask.nonzero(as_tuple=True)[0]
                    skip_mask[group_local_idx[~has_valid_last]] = True

        # Handle NaN in hand joints: last passed if valid, else current joint positions
        if self.hand_joint_dim > 0:
            hand_start = self._num_frame_tasks * self.pose_dim
            hand_nan_mask = torch.isnan(valid_actions[:, hand_start:]).any(dim=1)
            if hand_nan_mask.any():
                nan_env_ids = env_ids[hand_nan_mask]
                last_hand = self._last_passed_action[nan_env_ids, hand_start:]
                has_valid_last_hand = ~torch.isnan(last_hand).any(dim=1)
                if fallback_to_current:
                    current_hand_pos = self._asset.data.joint_pos[nan_env_ids][
                        :, self._hand_joint_ids
                    ]
                    fill_hand = torch.where(
                        has_valid_last_hand.unsqueeze(1).expand_as(current_hand_pos),
                        last_hand,
                        current_hand_pos,
                    )
                    valid_actions[hand_nan_mask, hand_start:] = fill_hand
                else:
                    valid_actions[hand_nan_mask, hand_start:] = last_hand
                    hand_local_idx = hand_nan_mask.nonzero(as_tuple=True)[0]
                    skip_mask[hand_local_idx[~has_valid_last_hand]] = True

        if skip_mask.any():
            keep_mask = ~skip_mask
            valid_actions = valid_actions[keep_mask]
            env_ids = env_ids[keep_mask]
            if len(env_ids) == 0:
                return None, env_ids

        return valid_actions, env_ids

    def _get_current_eef_pose(
        self, env_ids: torch.Tensor, task_idx: int
    ) -> torch.Tensor:
        """Get current EEF pose for specified envs and frame task index.

        Returns the pose as [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
        in the same coordinate frame as the input actions (base-relative when
        ``relative_to_base`` is True, otherwise env-origin-relative world frame).

        Args:
            env_ids: Environment IDs for which to get the current EEF pose.
            task_idx: Frame task index (0-based, counting only FrameTask instances).

        Returns:
            Current EEF pose tensor of shape ``(len(env_ids), 7)``.
        """
        body_idx = self._eef_body_indices[task_idx]

        # Get world-frame pose from articulation data: [pos(3), quat(4)]
        eef_state_w = self._asset.data.body_link_state_w[env_ids, body_idx, :7]
        eef_pos_w = eef_state_w[:, :3]
        eef_quat_w = eef_state_w[:, 3:7]

        # Convert position to env-origin-relative frame
        eef_pos = eef_pos_w - self._env.scene.env_origins[env_ids]

        if self.cfg.relative_to_base:
            # Compute base link pose in env-origin frame
            if self.cfg.controller.base_link_name == "root":
                base_pos = (
                    self._asset.data.root_pos_w[env_ids]
                    - self._env.scene.env_origins[env_ids]
                )
                base_quat = self._asset.data.root_quat_w[env_ids]
            else:
                articulation_data = self._env.scene[
                    self.cfg.controller.articulation_name
                ].data
                base_state = articulation_data.body_link_state_w[
                    env_ids, self._base_link_idx, :7
                ]
                base_pos = base_state[:, :3] - self._env.scene.env_origins[env_ids]
                base_quat = base_state[:, 3:7]

            # Express EEF pose relative to the base link frame
            eef_pos_base, eef_quat_base = math_utils.subtract_frame_transforms(
                base_pos, base_quat, eef_pos, eef_quat_w
            )
            return torch.cat([eef_pos_base, eef_quat_base], dim=1)
        else:
            return torch.cat([eef_pos, eef_quat_w], dim=1)

    def process_actions(self, actions: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Process the input actions and set targets for each task.

        Args:
            actions: The input actions tensor.
            env_ids: The environment IDs to process.
        """
        if self._step_count % self.cfg.decimation != 0:
            return

        # Handle env_ids
        if env_ids is None:
            selected_env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            selected_env_ids = torch.tensor(
                env_ids, device=self.device, dtype=torch.int32
            )
        else:
            selected_env_ids = env_ids

        # Handle NaN actions: skip all-NaN envs, fill partial NaN groups with current EEF pose
        valid_actions, self._env_ids = self._handle_nan_actions(
            actions, selected_env_ids
        )

        # If no valid envs, return early
        if valid_actions is None or len(self._env_ids) == 0:
            return

        # Store raw actions and record last passed action for NaN fallback
        self._raw_actions[self._env_ids] = valid_actions
        self._last_passed_action[self._env_ids] = valid_actions.clone()

        # Extract hand joint positions directly (no cloning needed)
        # Handle case when hand_joint_dim = 0: -0 is same as 0, so slice would be whole tensor
        if self.hand_joint_dim > 0:
            self._target_hand_joint_positions = valid_actions[:, -self.hand_joint_dim :]
        else:
            self._target_hand_joint_positions = torch.zeros(
                (valid_actions.shape[0], 0), device=self.device
            )

        # Get base link frame transformation for valid envs only
        all_base_link_frames = self._get_base_link_frame_transform()
        self.base_link_frame_in_world_rf = all_base_link_frames[self._env_ids]

        # Process controlled frame poses for valid envs
        controlled_frame_poses = self._extract_controlled_frame_poses(valid_actions)

        # DEBUG: Print raw action and controlled frame poses
        # print(f"\n=== Pink IK process_actions DEBUG ===")
        # print(f"Raw action (first env): pos={valid_actions[0, :3].cpu().numpy()}, quat={valid_actions[0, 3:7].cpu().numpy()}")
        # print(f"relative_to_base: {self.cfg.relative_to_base}")
        # print(f"Controlled frame poses (world frame):\n{controlled_frame_poses[0, 0].cpu().numpy()}")

        # NOTE(magics): We assume actions are already in the Base Link Frame (Pelvis) for LocalFrameTask.
        # This bypasses the World-to-Base transformation to allow direct local control.
        # If using standard FrameTask (World Frame), this would imply inputs are also in Local Frame
        # but interpreted as World Frame, which might be incorrect for mixed usage.
        # But for G1 LocalFrameTask usage, this is the correct behavior for Body-relative control.
        if self.cfg.relative_to_base:
            positions, rotation_matrices = math_utils.unmake_pose(
                controlled_frame_poses
            )
            transformed_poses = (positions, rotation_matrices)
        else:
            transformed_poses = self._transform_poses_to_base_link_frame(
                controlled_frame_poses
            )
            # DEBUG: Print after transformation
            # print(f"After transform_to_base_link_frame:")
            # print(f"  positions: {transformed_poses[0].cpu().numpy()}")
            # print(f"  rotation_matrices: {transformed_poses[1][0, 0].cpu().numpy()}")

        # Set targets for valid envs only
        self._set_task_targets(transformed_poses)

    def _get_base_link_frame_transform(self) -> torch.Tensor:
        """Get the base link frame transformation matrix.

        Returns:
            Base link frame transformation matrix.
        """
        if self.cfg.controller.base_link_name == "root":
            base_pos_w = self._asset.data.root_pos_w
            base_quat_w = self._asset.data.root_quat_w
            torch.sub(
                base_pos_w,
                self._env.scene.env_origins,
                out=self._base_link_frame_buffer[:, :3, 3],
            )
            return math_utils.make_pose(
                self._base_link_frame_buffer[:, :3, 3],
                math_utils.matrix_from_quat(base_quat_w),
            )

        # Get base link frame pose in world origin using cached index
        articulation_data = self._env.scene[self.cfg.controller.articulation_name].data
        base_link_frame_in_world_origin = articulation_data.body_link_state_w[
            :, self._base_link_idx, :7
        ]

        # Transform to environment origin frame (reuse buffer to avoid allocation)
        torch.sub(
            base_link_frame_in_world_origin[:, :3],
            self._env.scene.env_origins,
            out=self._base_link_frame_buffer[:, :3, 3],
        )

        # Copy orientation (avoid clone)
        base_link_frame_quat = base_link_frame_in_world_origin[:, 3:7]

        # Create transformation matrix
        return math_utils.make_pose(
            self._base_link_frame_buffer[:, :3, 3],
            math_utils.matrix_from_quat(base_link_frame_quat),
        )

    def _extract_controlled_frame_poses(self, actions: torch.Tensor) -> torch.Tensor:
        """Extract controlled frame poses from action tensor for valid envs.

        Args:
            actions: The action tensor for valid envs, shape (num_valid_envs, action_dim).

        Returns:
            Controlled frame poses tensor, shape (num_frame_tasks, num_valid_envs, 4, 4).
        """
        num_valid_envs = actions.shape[0]
        # Create tensor for valid envs (cannot use pre-allocated when size varies)
        controlled_frame_poses = torch.zeros(
            self._num_frame_tasks, num_valid_envs, 4, 4, device=self.device
        )

        for task_index in range(self._num_frame_tasks):
            # Extract position and orientation for this task
            pos_start = task_index * self.pose_dim
            pos_end = pos_start + self.position_dim
            quat_start = pos_end
            quat_end = (task_index + 1) * self.pose_dim

            position = actions[:, pos_start:pos_end]
            quaternion = actions[:, quat_start:quat_end]

            # Create pose matrix
            controlled_frame_poses[task_index] = math_utils.make_pose(
                position, math_utils.matrix_from_quat(quaternion)
            )

        return controlled_frame_poses

    def _transform_poses_to_base_link_frame(
        self, poses: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform poses from world frame to base link frame.

        Args:
            poses: Poses in world frame.

        Returns:
            Tuple of (positions, rotation_matrices) in base link frame.
        """
        # Transform poses to base link frame
        base_link_inv = math_utils.pose_inv(self.base_link_frame_in_world_rf)
        transformed_poses = math_utils.pose_in_A_to_pose_in_B(poses, base_link_inv)

        # Extract position and rotation
        positions, rotation_matrices = math_utils.unmake_pose(transformed_poses)

        return positions, rotation_matrices

    def _set_task_targets(
        self, transformed_poses: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Set targets for all tasks across valid environments only.

        Args:
            transformed_poses: Tuple of (positions, rotation_matrices) in base link frame,
                               with shape (num_frame_tasks, num_valid_envs, ...).
        """
        positions, rotation_matrices = transformed_poses

        # DEBUG: Print base link frame and transformed target positions
        # print(f"\n=== Pink IK _set_task_targets DEBUG ===")
        # print(f"base_link_frame_in_world_rf (env 0):\n{self.base_link_frame_in_world_rf[0].cpu().numpy()}")
        # print(f"Target positions (in base link frame): {positions.cpu().numpy()}")

        # Iterate only over valid envs
        for valid_idx, env_index in enumerate(self._env_ids.tolist()):
            ik_controller = self._ik_controllers[env_index]
            for frame_task_index, task in enumerate(
                ik_controller.cfg.variable_input_tasks
            ):
                if isinstance(task, LocalFrameTask):
                    target = task.transform_target_to_base
                elif isinstance(task, FrameTask):
                    target = task.transform_target_to_world
                else:
                    continue

                # Use valid_idx to index into transformed_poses
                target.translation = (
                    positions[frame_task_index, valid_idx, :].cpu().numpy()
                )
                target.rotation = (
                    rotation_matrices[frame_task_index, valid_idx, :].cpu().numpy()
                )

                task.set_target(target)

    # ==================== Action Application ====================

    def apply_actions(self) -> None:
        """Apply the computed joint positions based on the inverse kinematics solution.

        IK solve is only performed every ``cfg.decimation`` steps.  Cached
        joint-position targets are applied every step so the robot never
        stalls.
        """
        self._step_count += 1

        if len(self._env_ids) == 0:
            return

        # Recompute IK only on decimation steps (matches process_actions gate)
        if (self._step_count - 1) % self.cfg.decimation == 0:
            ik_joint_positions = self._compute_ik_solutions()

            if self.hand_joint_dim > 0:
                all_joint_positions = torch.cat(
                    (ik_joint_positions, self._target_hand_joint_positions), dim=1
                )
            else:
                all_joint_positions = ik_joint_positions

            self._processed_actions[self._env_ids] = all_joint_positions

        # Always apply cached targets + gravity compensation
        if self.cfg.enable_gravity_compensation:
            self._apply_gravity_compensation()

        self._asset.set_joint_position_target(
            self._processed_actions[self._env_ids],
            self._controlled_joint_ids,
            self._env_ids,
        )

    def _apply_gravity_compensation(self) -> None:
        """Apply gravity compensation to arm joints for valid envs only."""
        if not self._asset.cfg.spawn.rigid_props.disable_gravity:
            # Get gravity compensation forces using cached tensor for valid envs
            all_forces = self._asset.root_physx_view.get_gravity_compensation_forces()
            if self._asset.is_fixed_base:
                gravity = torch.zeros_like(
                    all_forces[self._env_ids][:, self._controlled_joint_ids_tensor]
                )
            else:
                # If floating base, then need to skip the first 6 joints (base)
                gravity = all_forces[self._env_ids][
                    :,
                    self._controlled_joint_ids_tensor
                    + self._physx_floating_joint_indices_offset,
                ]

            # Apply gravity compensation to arm joints for valid envs
            self._asset.set_joint_effort_target(
                gravity, self._controlled_joint_ids, self._env_ids
            )

    def _compute_ik_solutions(self) -> torch.Tensor:
        """Compute IK solutions for valid environments only.

        Returns:
            IK joint positions tensor for valid environments.
        """
        ik_solutions = []

        for env_index in self._env_ids.tolist():
            ik_controller = self._ik_controllers[env_index]
            # Get current joint positions for this environment
            current_joint_pos = self._asset.data.joint_pos.cpu().numpy()[env_index]

            # Compute IK solution
            joint_pos_des = ik_controller.compute(current_joint_pos, self._sim_dt)
            ik_solutions.append(joint_pos_des)

        return torch.stack(ik_solutions)

    # ==================== Reset ====================

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term for specified environments.

        Args:
            env_ids: A list of environment IDs to reset. If None, all environments are reset.
        """
        self._raw_actions[env_ids] = torch.zeros(self.action_dim, device=self.device)
        self._last_passed_action[env_ids] = torch.zeros(
            self.action_dim, device=self.device
        )
        # Also clear ``_processed_actions`` (the cached IK solution applied
        # in ``apply_actions`` every step). Without this, after a reset the
        # next NaN-action tick has ``valid_actions=None`` and
        # ``process_actions`` returns early — but ``apply_actions`` still
        # ships the *previous* episode's IK joint targets to PhysX,
        # snapping the arms back to whatever pose ended the last attempt.
        # Replace with the robot's current joint positions ("hold here")
        # so the freshly-reset default joint pose is what gets commanded.
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.tensor(
                list(env_ids), device=self.device, dtype=torch.long
            )
        if env_ids_tensor.numel() > 0:
            cur_joint_pos = self._asset.data.joint_pos[env_ids_tensor][
                :, self._controlled_joint_ids
            ]
            self._processed_actions[env_ids_tensor] = cur_joint_pos


class PinkDualDifferentialInverseKinematicsAction(PinkInverseKinematicsAction):
    """Hybrid Pink IK + per-arm differential IK action.

    On decimation fires this runs the full Pink IK solve (via the parent
    implementation). Between fires it falls back to per-arm Jacobian tracking
    toward the *same* target, matching the computation in
    :class:`DualDifferentialInverseKinematicsAction`. Hand-joint targets are
    refreshed only on Pink fires; PhysX retains them between steps.

    The action layout is identical to the parent: ``[right_pose(7), left_pose(7)]``
    for dual-arm robots, because the Pink ``variable_input_tasks`` must be
    ordered right-first, left-second.
    """

    cfg: "pink_actions_cfg.PinkDualDifferentialInverseKinematicsActionCfg"

    def __init__(
        self,
        cfg: "pink_actions_cfg.PinkDualDifferentialInverseKinematicsActionCfg",
        env,
    ):
        super().__init__(cfg, env)
        self._setup_dual_diff()

    def _setup_dual_diff(self) -> None:
        cfg = self.cfg

        self._right_joint_ids, _ = self._asset.find_joints(cfg.right_joint_names)
        self._left_joint_ids, _ = self._asset.find_joints(cfg.left_joint_names)

        r_ids, r_names = self._asset.find_bodies(cfg.right_body_name)
        l_ids, l_names = self._asset.find_bodies(cfg.left_body_name)
        if len(r_ids) != 1 or len(l_ids) != 1:
            raise ValueError(
                f"Expected one body match each: right={r_names}, left={l_names}"
            )
        self._right_body_idx = r_ids[0]
        self._left_body_idx = l_ids[0]

        self._right_ref_idx = None
        self._left_ref_idx = None
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

        self._right_offset_pos, self._right_offset_rot = self._build_diff_offset(
            cfg.right_body_offset
        )
        self._left_offset_pos, self._left_offset_rot = self._build_diff_offset(
            cfg.left_body_offset
        )

        self._right_diff_ik = DifferentialIKController(
            cfg=cfg.diff_ik_controller, num_envs=self.num_envs, device=self.device
        )
        self._left_diff_ik = DifferentialIKController(
            cfg=cfg.diff_ik_controller, num_envs=self.num_envs, device=self.device
        )

        if self._num_frame_tasks != 2:
            raise ValueError(
                "PinkDualDifferentialInverseKinematicsAction expects exactly 2 "
                f"FrameTask entries in variable_input_tasks, got {self._num_frame_tasks}."
            )

    def _build_diff_offset(self, offset_cfg):
        if offset_cfg is None:
            return None, None
        pos = torch.tensor(offset_cfg.pos, device=self.device).repeat(self.num_envs, 1)
        rot = torch.tensor(offset_cfg.rot, device=self.device).repeat(self.num_envs, 1)
        return pos, rot

    # --- per-arm geometry helpers (mirror DualDifferentialInverseKinematicsAction)

    def _arm_pose_in_ref(
        self,
        env_ids,
        body_idx,
        ref_idx,
        offset_pos,
        offset_rot,
    ):
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
        env_ids,
        jacobi_body_idx,
        jacobi_joint_ids,
        ref_idx,
        offset_pos,
        offset_rot,
    ):
        jac_w = self._asset.root_physx_view.get_jacobians()[
            :, jacobi_body_idx, :, jacobi_joint_ids
        ][env_ids]
        if ref_idx is not None:
            ref_quat = self._asset.data.body_quat_w[env_ids, ref_idx]
        else:
            ref_quat = self._asset.data.root_quat_w[env_ids]
        R = math_utils.matrix_from_quat(math_utils.quat_inv(ref_quat))
        jac = jac_w.clone()
        jac[:, 0:3, :] = torch.bmm(R, jac[:, 0:3, :])
        jac[:, 3:, :] = torch.bmm(R, jac[:, 3:, :])
        if offset_pos is not None:
            skew = math_utils.skew_symmetric_matrix(offset_pos[env_ids])
            jac[:, 0:3, :] += torch.bmm(-skew, jac[:, 3:, :])
            jac[:, 3:, :] = torch.bmm(
                math_utils.matrix_from_quat(offset_rot[env_ids]), jac[:, 3:, :]
            )
        return jac

    def _transform_command_to_ref(self, command, env_ids, ref_idx):
        """Convert an env-local world-frame pose command into the arm's ref frame."""
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
        if self.cfg.diff_ik_controller.command_type == "position":
            pos_b, _ = math_utils.subtract_frame_transforms(
                base_pos_w, base_quat_w, positions_w, None
            )
            return pos_b
        quats_w = command[:, 3:7]
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            base_pos_w, base_quat_w, positions_w, quats_w
        )
        return torch.cat((pos_b, quat_b), dim=1)

    # --- overridden apply ---------------------------------------------------

    def apply_actions(self) -> None:
        self._step_count += 1

        if len(self._env_ids) == 0:
            return

        is_pink_fire = (self._step_count - 1) % max(1, self.cfg.decimation) == 0

        if is_pink_fire:
            ik_joint_positions = self._compute_ik_solutions()
            if self.hand_joint_dim > 0:
                all_joint_positions = torch.cat(
                    (ik_joint_positions, self._target_hand_joint_positions), dim=1
                )
            else:
                all_joint_positions = ik_joint_positions
            self._processed_actions[self._env_ids] = all_joint_positions

            if self.cfg.enable_gravity_compensation:
                self._apply_gravity_compensation()

            self._asset.set_joint_position_target(
                self._processed_actions[self._env_ids],
                self._controlled_joint_ids,
                self._env_ids,
            )
        else:
            self._apply_dual_diff_ik()

    def _apply_dual_diff_ik(self) -> None:
        """Per-arm Jacobian tracking toward the stored Pink pose target."""
        raw = self._raw_actions[self._env_ids]
        right_cmd = raw[:, : self.pose_dim]
        left_cmd = raw[:, self.pose_dim : 2 * self.pose_dim]

        # Pink interprets actions in env-local world frame when relative_to_base is False.
        if not self.cfg.relative_to_base:
            right_cmd = self._transform_command_to_ref(
                right_cmd, self._env_ids, self._right_ref_idx
            )
            left_cmd = self._transform_command_to_ref(
                left_cmd, self._env_ids, self._left_ref_idx
            )

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

        self._right_diff_ik.set_command(right_cmd, r_pos, r_quat)
        self._left_diff_ik.set_command(left_cmd, l_pos, l_quat)

        r_q = self._asset.data.joint_pos[
            self._env_ids.unsqueeze(1), self._right_joint_ids
        ]
        l_q = self._asset.data.joint_pos[
            self._env_ids.unsqueeze(1), self._left_joint_ids
        ]
        r_jac = self._arm_jacobian(
            self._env_ids,
            self._right_jacobi_body_idx,
            self._right_jacobi_joint_ids,
            self._right_ref_idx,
            self._right_offset_pos,
            self._right_offset_rot,
        )
        l_jac = self._arm_jacobian(
            self._env_ids,
            self._left_jacobi_body_idx,
            self._left_jacobi_joint_ids,
            self._left_ref_idx,
            self._left_offset_pos,
            self._left_offset_rot,
        )

        r_q_des = self._right_diff_ik.compute(r_pos, r_quat, r_jac, r_q)
        l_q_des = self._left_diff_ik.compute(l_pos, l_quat, l_jac, l_q)

        r_lo = self._asset.data.joint_pos_limits[0, self._right_joint_ids, 0]
        r_hi = self._asset.data.joint_pos_limits[0, self._right_joint_ids, 1]
        l_lo = self._asset.data.joint_pos_limits[0, self._left_joint_ids, 0]
        l_hi = self._asset.data.joint_pos_limits[0, self._left_joint_ids, 1]
        r_q_des = torch.clamp(r_q_des, min=r_lo, max=r_hi)
        l_q_des = torch.clamp(l_q_des, min=l_lo, max=l_hi)

        if self.cfg.enable_gravity_compensation:
            self._apply_gravity_compensation()

        self._asset.set_joint_position_target(
            r_q_des, self._right_joint_ids, self._env_ids
        )
        self._asset.set_joint_position_target(
            l_q_des, self._left_joint_ids, self._env_ids
        )
