# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import yaml
from typing import Any
from isaaclab.assets import ArticulationData


def load_config(config_path: str) -> dict[str, Any]:
    """Load and process the YAML configuration file"""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Convert lists to numpy arrays where needed
    array_keys = ["default_angles", "cmd_scale", "cmd_init"]
    for key in array_keys:
        if key in config:
            config[key] = np.array(config[key], dtype=np.float32)

    return config


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by the inverse of quaternion q

    Args:
        q: Quaternion array of shape (num_envs, 4) or (4,)
        v: Vector array of shape (3,) or (num_envs, 3)

    Returns:
        Rotated vector array of shape (num_envs, 3) or (3,)
    """
    # Handle batch processing
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if v.ndim == 1:
        v = v.reshape(1, -1)

    num_envs = q.shape[0]
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    # Pre-calculate components for batch processing
    q_conj_w = w
    q_conj_x = -x
    q_conj_y = -y
    q_conj_z = -z

    # Ensure v is broadcastable to (num_envs, 3)
    if v.shape[0] == 1 and num_envs > 1:
        v = np.tile(v, (num_envs, 1))
    elif v.shape[0] != num_envs:
        v = v[:num_envs]

    # Compute rotated vector for each environment - result shape (num_envs, 3)
    result = np.zeros((num_envs, 3), dtype=np.float32)

    # Optimized batch calculation
    result[:, 0] = (
        v[:, 0] * (q_conj_w**2 + q_conj_x**2 - q_conj_y**2 - q_conj_z**2)
        + v[:, 1] * 2 * (q_conj_x * q_conj_y - q_conj_w * q_conj_z)
        + v[:, 2] * 2 * (q_conj_x * q_conj_z + q_conj_w * q_conj_y)
    )
    result[:, 1] = (
        v[:, 0] * 2 * (q_conj_x * q_conj_y + q_conj_w * q_conj_z)
        + v[:, 1] * (q_conj_w**2 - q_conj_x**2 + q_conj_y**2 - q_conj_z**2)
        + v[:, 2] * 2 * (q_conj_y * q_conj_z - q_conj_w * q_conj_x)
    )
    result[:, 2] = (
        v[:, 0] * 2 * (q_conj_x * q_conj_z - q_conj_w * q_conj_y)
        + v[:, 1] * 2 * (q_conj_y * q_conj_z + q_conj_w * q_conj_x)
        + v[:, 2] * (q_conj_w**2 - q_conj_x**2 - q_conj_y**2 + q_conj_z**2)
    )

    return result


def get_gravity_orientation(quat: np.ndarray) -> np.ndarray:
    """Get gravity vector in body frame"""
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)


def convert_sim_joint_to_wbc_joint(
    sim_joint_data: np.ndarray,
    sim_joint_names: list[str],
    wbc_joints_order: dict[str, int],
) -> np.ndarray:
    """Convert sim joint observations to WBC joint observations.

    Args:
        sim_joint_data: Sim joint data in Lab's order
        sim_joint_names: Sim joint names in Lab's order
        wbc_joints_order: WBC joint order in policy config yaml

    Returns:
        WBC joint data in WBC joint order
    """
    num_joints = len(wbc_joints_order)
    num_envs = sim_joint_data.shape[0]
    wbc_joint_data = np.zeros((num_envs, num_joints))

    # Check if sim_joint_data is a numpy array, if not, convert from torch tensor to numpy
    if not isinstance(sim_joint_data, np.ndarray):
        sim_joint_data = sim_joint_data.cpu().numpy()

    for sim_joint_name in sim_joint_names:
        sim_joint_index = sim_joint_names.index(sim_joint_name)
        assert sim_joint_name in wbc_joints_order, (
            f"Joint {sim_joint_name} not found in wbc_joints_order"
        )
        wbc_joint_index = wbc_joints_order[sim_joint_name]
        wbc_joint_data[:, wbc_joint_index] = sim_joint_data[:, sim_joint_index]
    return wbc_joint_data


def prepare_observations(
    num_envs: int, robot_data: ArticulationData, wbc_joints_order: dict[str, int]
) -> dict[str, np.ndarray]:
    """Prepare observations for the policy.

    Args:
        num_envs: Number of environments
        robot_data: Robot data
        wbc_joints_order: WBC joint order in policy config yaml

    Returns:
        Observations for the policy
        - q: Joint positions
        - dq: Joint velocities
        - ddq: Joint accelerations
        - floating_base_pose: Floating base pose
        - floating_base_vel: Floating base velocity
        - floating_base_acc: Floating base acceleration
    """
    # Get robot joint observations
    sim_joint_pos = robot_data.joint_pos.cpu().numpy()
    sim_joint_vel = robot_data.joint_vel.cpu().numpy()
    num_joints = len(robot_data.joint_names)

    # Convert joints data from Lab's order to WBC's order saved in config yaml
    wbc_joint_pos = np.zeros((num_envs, num_joints))
    wbc_joint_vel = np.zeros((num_envs, num_joints))
    wbc_joint_acc = np.zeros((num_envs, num_joints))
    wbc_joint_pos = convert_sim_joint_to_wbc_joint(
        sim_joint_pos, robot_data.joint_names, wbc_joints_order
    )
    wbc_joint_vel = convert_sim_joint_to_wbc_joint(
        sim_joint_vel, robot_data.joint_names, wbc_joints_order
    )

    # Prepare obs dict for WBC policy input
    assert (
        wbc_joint_pos.shape
        == wbc_joint_vel.shape
        == wbc_joint_acc.shape
        == (num_envs, num_joints)
    )

    root_link_pos_w = robot_data.root_link_pos_w.cpu().numpy()
    root_link_quat_w = robot_data.root_link_quat_w.cpu().numpy()
    base_pose_w = np.concatenate((root_link_pos_w, root_link_quat_w), axis=1)
    base_lin_vel_b = robot_data.root_link_lin_vel_b.cpu().numpy()
    base_ang_vel_b = robot_data.root_link_ang_vel_b.cpu().numpy()

    base_vel_b = np.concatenate((base_lin_vel_b, base_ang_vel_b), axis=1)

    # Prepare observations
    wbc_obs = {
        "q": wbc_joint_pos,
        "dq": wbc_joint_vel,
        "ddq": np.zeros(
            (num_envs, num_joints)
        ),  # Not used by quadruped locomotion policy
        "tau_est": np.zeros(
            (num_envs, num_joints)
        ),  # Not used by quadruped locomotion policy
        "floating_base_pose": base_pose_w,  # wrt world frame, used to project gravity vector to local frame
        "floating_base_vel": base_vel_b,  # wrt body frame
        "floating_base_acc": np.zeros(
            (num_envs, 6)
        ),  # Not used by quadruped locomotion policy
    }
    return wbc_obs
