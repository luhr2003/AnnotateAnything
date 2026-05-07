from typing import List
import torch
from magicsim.Task.TableTop.Env.GraspEnv import GraspEnv
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim.Env.Utils.rotations import quat_to_rot_matrix
import isaacsim.replicator.grasping.ui.grasping_ui_utils as grasping_ui_utils

from pxr import Gf

# Visualization settings
AXIS_LENGTH = 0.05
LINE_THICKNESS = 3
LINE_OPACITY = 0.8


def visualize_grasp_pose(grasp_pose: List[torch.Tensor]):
    grasp_pose = [p.cpu().numpy().tolist() for p in grasp_pose]
    grasp_pose_list = [
        (Gf.Vec3d(p[0], p[1], p[2]), Gf.Quatd(p[3], p[4], p[5], p[6]))
        for p in grasp_pose
    ]
    grasping_ui_utils.draw_grasp_samples_as_axes(
        grasp_poses=grasp_pose_list,
        axis_length=AXIS_LENGTH,
        line_thickness=LINE_THICKNESS,
        line_opacity=LINE_OPACITY,
        clear_existing=True,
    )


@hydra.main(version_base=None, config_path="../../Conf", config_name="grasp_rect")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: GraspEnv = gym.make("GraspEnv-V0", config=cfg, cli_args=None, logger=logger)
    obs, info = env.reset()
    for i in range(40):
        env.sim_step()
    action = env.get_mug_grasp_pose(env_ids=[0])
    print(action)
    action = action[0]["functional_grasp"]["body"]
    visualize_grasp_pose(action)
    action = action[1]
    visualize_grasp_pose([action])
    action = action.cpu().numpy().tolist()
    # action[2] -= 0.1

    for i in range(100):
        pre_action = action.copy()
        # Extract position and quaternion from grasp pose
        grasp_pos = torch.tensor(pre_action[:3], device=env.device)  # [x, y, z]
        grasp_quat = torch.tensor(
            pre_action[3:7], device=env.device
        )  # [qw, qx, qy, qz]

        # Convert quaternion to rotation matrix
        rot_matrix = quat_to_rot_matrix(grasp_quat.unsqueeze(0))  # (1, 3, 3)

        # Extract z-axis direction (grasp direction, typically forward)
        # The z-axis in local frame is [0, 0, 1], after rotation it becomes the third column
        grasp_direction = rot_matrix[0, :, 2]  # (3,) - third column is z-axis

        # Normalize the direction vector to ensure unit length
        grasp_direction_normalized = grasp_direction / torch.norm(grasp_direction)

        # Move 0.15 along the grasp direction (backward, so subtract)
        offset = grasp_direction_normalized * 0.15
        new_pos = grasp_pos - offset
        # Update action with new position
        pre_action[:3] = new_pos.cpu().tolist()
        pre_action = torch.tensor(pre_action, device=env.device).unsqueeze(0)
        env.step(action=pre_action)

    for i in range(60):
        grasp_action = action.copy()
        grasp_action = torch.tensor(grasp_action, device=env.device).unsqueeze(0)
        env.step(action=grasp_action)

    for i in range(20):
        close_action = action.copy()
        close_action.append(1)
        close_action = torch.tensor(close_action, device=env.device).unsqueeze(0)
        env.step(action=close_action)

    for i in range(100):
        up_action = action.copy()
        up_action.append(1)
        # Extract position and quaternion from grasp pose
        grasp_pos = torch.tensor(up_action[:3], device=env.device)  # [x, y, z]
        grasp_quat = torch.tensor(up_action[3:7], device=env.device)  # [qw, qx, qy, qz]

        # Convert quaternion to rotation matrix
        rot_matrix = quat_to_rot_matrix(grasp_quat.unsqueeze(0))  # (1, 3, 3)

        # Extract z-axis direction (grasp direction, typically forward)
        # The z-axis in local frame is [0, 0, 1], after rotation it becomes the third column
        grasp_direction = rot_matrix[0, :, 2]  # (3,) - third column is z-axis

        # Normalize the direction vector to ensure unit length
        grasp_direction_normalized = grasp_direction / torch.norm(grasp_direction)

        # Move 0.5 along the grasp direction (upward, so add)
        offset = grasp_direction_normalized * 0.2
        new_pos = grasp_pos - offset

        # Update action with new position
        up_action[:3] = new_pos.cpu().tolist()
        up_action = torch.tensor(up_action, device=env.device).unsqueeze(0)
        env.step(action=up_action)


if __name__ == "__main__":
    main()
