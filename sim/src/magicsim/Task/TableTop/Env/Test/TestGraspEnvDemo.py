from typing import List
import torch
from magicsim.Task.TableTop.Env.GraspEnv import GraspEnv
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
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


@hydra.main(version_base=None, config_path="../../Conf", config_name="grasp_env")
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

    while 1:
        env.sim_step()


if __name__ == "__main__":
    main()
