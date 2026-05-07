# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as PoseUtils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def transform_pose_from_world_to_target_frame(
    env: ManagerBasedEnv,
    target_link_name: str,
    target_frame_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the pose of the target link in the specified target frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    assert target_link_name in asset.data.body_names, (
        f"Target link {target_link_name} not found in asset {asset_cfg.name}"
    )
    assert target_frame_name in asset.data.body_names, (
        f"Target frame {target_frame_name} not found in asset {asset_cfg.name}"
    )

    target_link_pose_w = asset.data.body_link_state_w[
        :, asset.data.body_names.index(target_link_name), :
    ]
    target_frame_pose_w = asset.data.body_link_state_w[
        :, asset.data.body_names.index(target_frame_name), :
    ]

    # Convert to pose matrix
    target_link_position_w = target_link_pose_w[:, :3]
    target_link_rot_mat_w = PoseUtils.matrix_from_quat(target_link_pose_w[:, 3:7])
    target_link_pose_mat_w = PoseUtils.make_pose(
        target_link_position_w, target_link_rot_mat_w
    )

    target_frame_position_w = target_frame_pose_w[:, :3]
    target_frame_rot_mat_w = PoseUtils.matrix_from_quat(target_frame_pose_w[:, 3:7])
    target_frame_pose_mat_w = PoseUtils.make_pose(
        target_frame_position_w, target_frame_rot_mat_w
    )

    # Get target frame inverse transform to convert from world to target frame
    target_frame_pose_inv = PoseUtils.pose_inv(target_frame_pose_mat_w)

    # Transform target link poses from world frame to target frame
    target_link_pose_target_frame = torch.matmul(
        target_frame_pose_inv, target_link_pose_mat_w
    )

    return target_link_pose_target_frame


def get_target_link_position_in_target_frame(
    env: ManagerBasedEnv,
    target_link_name: str = "left_wrist_yaw_link",
    target_frame_name: str = "pelvis",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the position of the target link in the target frame."""
    target_link_pose_target_frame = transform_pose_from_world_to_target_frame(
        env, target_link_name, target_frame_name, asset_cfg
    )
    target_link_position_target_frame, left_target_link_rot_target_frame = (
        PoseUtils.unmake_pose(target_link_pose_target_frame)
    )
    return target_link_position_target_frame


def get_target_link_position_in_world_frame(
    env: ManagerBasedEnv,
    target_link_name: str = "left_wrist_yaw_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the position of the target link in the world frame."""
    return (
        env.scene[asset_cfg.name].data.body_link_state_w[
            :, env.scene[asset_cfg.name].data.body_names.index(target_link_name), :3
        ]
        - env.scene.env_origins
    )


def get_target_link_quaternion_in_world_frame(
    env: ManagerBasedEnv,
    target_link_name: str = "left_wrist_yaw_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the quaternion of the target link in the world frame."""
    return env.scene[asset_cfg.name].data.body_link_state_w[
        :, env.scene[asset_cfg.name].data.body_names.index(target_link_name), 3:7
    ]


def get_target_link_quaternion_in_target_frame(
    env: ManagerBasedEnv,
    target_link_name: str = "left_wrist_yaw_link",
    target_frame_name: str = "pelvis",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the quaternion of the target link in the target frame."""
    target_link_pose_target_frame = transform_pose_from_world_to_target_frame(
        env, target_link_name, target_frame_name, asset_cfg
    )
    target_link_position_target_frame, left_target_link_rot_target_frame = (
        PoseUtils.unmake_pose(target_link_pose_target_frame)
    )
    target_link_quat_target_frame = PoseUtils.quat_from_matrix(
        left_target_link_rot_target_frame
    )
    return target_link_quat_target_frame


def get_dual_link_position_in_world_frame(
    env: ManagerBasedEnv,
    link_name_1: str = "left_hand_palm_link",
    link_name_2: str = "right_hand_palm_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the positions of two links in the world frame and stack them.

    Returns:
        [num_envs, 2, 3] -> [[x1, y1, z1], [x2, y2, z2]]
    """
    data = env.scene[asset_cfg.name].data
    origin = env.scene.env_origins
    pos1 = data.body_link_state_w[:, data.body_names.index(link_name_1), :3] - origin
    pos2 = data.body_link_state_w[:, data.body_names.index(link_name_2), :3] - origin
    return torch.stack([pos1, pos2], dim=1)


def get_dual_link_quaternion_in_world_frame(
    env: ManagerBasedEnv,
    link_name_1: str = "left_hand_palm_link",
    link_name_2: str = "right_hand_palm_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the quaternions of two links in the world frame and stack them.

    Returns:
        [num_envs, 2, 4] -> [[qw1, qx1, qy1, qz1], [qw2, qx2, qy2, qz2]]
    """
    data = env.scene[asset_cfg.name].data
    quat1 = data.body_link_state_w[:, data.body_names.index(link_name_1), 3:7]
    quat2 = data.body_link_state_w[:, data.body_names.index(link_name_2), 3:7]
    return torch.stack([quat1, quat2], dim=1)


def get_target_link_lin_vel_in_world_frame(
    env: ManagerBasedEnv,
    target_link_name: str = "base_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the linear velocity of the target link in the world frame.

    Returns:
        [num_envs, 3] -> (vx, vy, vz)
    """
    return env.scene[asset_cfg.name].data.body_link_state_w[
        :, env.scene[asset_cfg.name].data.body_names.index(target_link_name), 7:10
    ]


def get_target_link_ang_vel_in_world_frame(
    env: ManagerBasedEnv,
    target_link_name: str = "base_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the angular velocity of the target link in the world frame.

    Returns:
        [num_envs, 3] -> (wx, wy, wz)
    """
    return env.scene[asset_cfg.name].data.body_link_state_w[
        :, env.scene[asset_cfg.name].data.body_names.index(target_link_name), 10:13
    ]


def joint_pos_with_root_offset(
    env: ManagerBasedEnv,
    x_joint_name: str = "dummy_base_prismatic_x_joint",
    y_joint_name: str = "dummy_base_prismatic_y_joint",
    yaw_joint_name: str = "dummy_base_revolute_z_joint",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint positions with root-state offset applied to base dummy joints.

    For holonomic mobile manipulators whose base is driven by dummy prismatic/revolute
    joints starting from 0, the actual world-frame base pose lives in
    ``default_root_state``. This function adds the initial root (x, y, yaw) to the
    designated base joint positions so that ``joint_pos`` reflects world-frame values.

    Args:
        x_joint_name: Name of the dummy prismatic x joint.
        y_joint_name: Name of the dummy prismatic y joint.
        yaw_joint_name: Name of the dummy revolute z (yaw) joint.
        asset_cfg: Asset scene entity configuration.

    Returns:
        ``[num_envs, num_joints]`` tensor – same as ``asset.data.joint_pos`` but with
        the base joints offset by the initial root state.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids].clone()

    # Resolve joint indices within the returned joint_ids slice.
    # asset_cfg.joint_ids can be a slice, list, or None.
    num_joints = asset.data.joint_pos.shape[1]
    jids = asset_cfg.joint_ids
    if jids is None:
        joint_ids_list = list(range(num_joints))
    elif isinstance(jids, slice):
        joint_ids_list = list(range(*jids.indices(num_joints)))
    else:
        joint_ids_list = list(jids)
    id_to_col = {jid: col for col, jid in enumerate(joint_ids_list)}

    idx_x, _ = asset.find_joints(x_joint_name)
    idx_y, _ = asset.find_joints(y_joint_name)
    idx_yaw, _ = asset.find_joints(yaw_joint_name)
    col_x = id_to_col[idx_x[0]]
    col_y = id_to_col[idx_y[0]]
    col_yaw = id_to_col[idx_yaw[0]]

    # default_root_state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    root = asset.data.default_root_state  # (N, 13)
    root_x = root[:, 0] - env.scene.env_origins[:, 0]
    root_y = root[:, 1] - env.scene.env_origins[:, 1]
    root_quat = root[:, 3:7]  # (w, x, y, z)
    # Extract yaw from quaternion
    _, _, root_yaw = PoseUtils.euler_xyz_from_quat(root_quat)

    joint_pos[:, col_x] += root_x
    joint_pos[:, col_y] += root_y
    joint_pos[:, col_yaw] += root_yaw

    return joint_pos


def get_navigate_cmd(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Get the navigate command."""
    return env.action_manager.get_term("humanoid_action").navigate_cmd.clone()


def get_asset_position(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the robot position."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w


def get_asset_quaternion(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the robot quaternion."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_quat_w
