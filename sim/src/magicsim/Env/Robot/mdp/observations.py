from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from magicsim.Env.Sensor.frame_transformer import FrameTransformer
from magicsim.Env.Environment.Isaac import IsaacRLEnv


def ee_frame_pos(
    env: IsaacRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(
    env, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def gripper_pos(
    env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)


def ee_rel_pos(
    env: IsaacRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_source[:, 0, :]
    return ee_frame_pos


def ee_rel_pos_arm_base(
    env: IsaacRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # print("ee_frame.data: ",ee_frame.data)
    ee_frame_pos = (
        ee_frame.data.target_pos_source[:, 0, :]
        - ee_frame.data.target_pos_source[:, 1, :]
    )
    # print("ee_frame_pos: ", ee_frame_pos)
    return ee_frame_pos


def ee_rel_quat(
    env: IsaacRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_source[:, 0, :]
    return ee_frame_quat


def ee_frame_pos_at(
    env: IsaacRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    frame_index: int = 0,
) -> torch.Tensor:
    """EE frame pos for dual-arm. frame_index: 0=right, 1=left (right first, left second)."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[:, frame_index, :] - env.scene.env_origins[:, 0:3]


def ee_frame_quat_at(
    env: IsaacRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    frame_index: int = 0,
) -> torch.Tensor:
    """EE frame quat for dual-arm. frame_index: 0=right, 1=left."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[:, frame_index, :]


def ee_rel_pos_at(
    env: IsaacRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    frame_index: int = 0,
) -> torch.Tensor:
    """EE relative pos for dual-arm. frame_index: 0=right, 1=left."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_source[:, frame_index, :]


def ee_rel_quat_at(
    env: IsaacRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    frame_index: int = 0,
) -> torch.Tensor:
    """EE relative quat for dual-arm. frame_index: 0=right, 1=left."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_source[:, frame_index, :]


def ee_dual_pos(
    env: IsaacRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Dual-arm: stacked eef pos [N, 2, 3], right first then left. For get_robot_state compatibility."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    pos = ee_frame.data.target_pos_w[:, :2, :] - env.scene.env_origins[:, None, 0:3]
    return pos


def ee_dual_quat(
    env: IsaacRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Dual-arm: stacked eef quat [N, 2, 4], right first then left. For get_robot_state compatibility."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[:, :2, :]


def base_pos(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    base_pos = robot.data.root_pos_w - env.scene.env_origins[:, 0:3]
    return base_pos


def base_quat(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    base_quat = robot.data.root_quat_w
    return base_quat


def front_steer(
    env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    joint_ids, _ = robot.find_joints(
        ["Knuckle__Upright__Front_Left", "Knuckle__Upright__Front_Right"]
    )
    steers = robot.data.joint_pos[:, joint_ids]
    return steers


def holomonic_base_pos(
    env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    joint_ids, _ = robot.find_joints(
        ["dummy_base_prismatic_y_joint", "dummy_base_prismatic_x_joint"]
    )
    base_pos = robot.data.joint_pos[:, joint_ids]
    return base_pos


def holomonic_base_quat(
    env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    joint_ids, _ = robot.find_joints(["dummy_base_revolute_z_joint"])
    base_quat = robot.data.joint_pos[:, joint_ids]
    return base_quat


def holonomic_base_lin_vel_b(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    从 dummy base 的 prismatic x/y joint 直接读 robot(base) frame 的线速度:
      vx = v(dummy_base_prismatic_x_joint)
      vy = v(dummy_base_prismatic_y_joint)

    返回:
      [num_envs, 2]  -> (vx, vy) in base/body frame
    """
    robot: Articulation = env.scene[robot_cfg.name]

    jx, _ = robot.find_joints(["dummy_base_prismatic_x_joint"])
    jy, _ = robot.find_joints(["dummy_base_prismatic_y_joint"])

    # joint_vel: [num_envs, num_joints]
    vx = robot.data.joint_vel[:, jx].squeeze(-1)  # [N]
    vy = robot.data.joint_vel[:, jy].squeeze(-1)  # [N]

    return torch.stack([vx, vy], dim=-1)  # [N, 2]


def holonomic_base_ang_vel_b(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    返回 base/body frame 的角速度 (rx, ry, rz)

    对于 dummy base 结构：
      rx = 0
      ry = 0
      rz = joint_vel(dummy_base_revolute_z_joint)

    返回:
      [num_envs, 3] -> (rx, ry, rz) in body frame
    """
    robot: Articulation = env.scene[robot_cfg.name]

    # 找 yaw 关节
    jz, _ = robot.find_joints(["dummy_base_revolute_z_joint"])

    rz = robot.data.joint_vel[:, jz].squeeze(-1)  # [N] rad/s

    zeros = torch.zeros_like(rz)

    return torch.stack([zeros, zeros, rz], dim=-1)  # [N, 3]


def get_pos(
    env: IsaacRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_link_name: str = "pelvis",
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    target_link_pos = robot.data.body_link_state_w[
        :, robot.data.body_names.index(target_link_name), :3
    ]
    return target_link_pos


def get_quat(
    env: IsaacRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_link_name: str = "pelvis",
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    target_link_quat = robot.data.body_link_state_w[
        :, robot.data.body_names.index(target_link_name), 3:7
    ]
    return target_link_quat


def get_ang_vel(
    env: IsaacRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_link_name: str = "pelvis",
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    target_link_ang_vel = robot.data.body_link_state_w[
        :, robot.data.body_names.index(target_link_name), 7:10
    ]
    return target_link_ang_vel


def get_lin_vel(
    env: IsaacRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_link_name: str = "pelvis",
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    target_link_lin_vel = robot.data.body_link_state_w[
        :, robot.data.body_names.index(target_link_name), 10:13
    ]
    return target_link_lin_vel
