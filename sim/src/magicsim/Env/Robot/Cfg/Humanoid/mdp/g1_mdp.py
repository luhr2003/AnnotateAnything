from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_dtype,
    record_joint_names,
    record_shape,
)
from isaaclab.managers import SceneEntityCfg


def _pelvis_pos_yaw(
    asset: Articulation, env: ManagerBasedEnv, target_link_name: str = "pelvis"
) -> torch.Tensor:
    """Return pelvis (or root) position and yaw relative to default root state as [x, y, z, rz]."""
    try:
        link_index = asset.data.body_names.index(target_link_name)
        link_state = asset.data.body_link_state_w[:, link_index]
        pos_w = link_state[:, :3] - env.scene.env_origins
        quat_w = link_state[:, 3:7]
    except ValueError:
        # Fallback to root if pelvis link is not available.
        pos_w = asset.data.root_pos_w - env.scene.env_origins
        quat_w = asset.data.root_quat_w - env.scene.env_origins

    # default_root_state = asset.data.default_root_state
    # default_pos = default_root_state[:, :3]
    # default_quat = default_root_state[:, 3:7]

    # pos_rel = pos_w - default_pos
    # quat_rel = math_utils.quat_mul(quat_w, math_utils.quat_conjugate(default_quat))
    pos_rel = pos_w
    quat_rel = quat_w
    _, _, yaw = math_utils.euler_xyz_from_quat(quat_rel)
    return torch.cat([pos_rel, yaw.unsqueeze(-1)], dim=-1)


@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape],
    units="rad",
)
def joint_pos(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset with pelvis [x, y, z, rz] appended.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    pelvis_pos_yaw = _pelvis_pos_yaw(asset, env)
    return torch.cat([joint_pos, pelvis_pos_yaw], dim=-1)


@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape],
    units="rad/s",
)
def joint_vel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint velocities of the asset with pelvis [x, y, z, rz] appended.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    pelvis_pos_yaw = _pelvis_pos_yaw(asset, env)
    return torch.cat([joint_vel, pelvis_pos_yaw], dim=-1)


@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape],
    units="N.m",
)
def joint_effort(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint applied effort of the robot.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their effort returned.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The joint effort (N or N-m) for joint_names in asset_cfg, shape is [num_env,num_joints].
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]
