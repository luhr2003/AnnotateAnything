"""
Test: xtrainer dual-arm reach with two cubes (one per arm). Uses xtrainer with
ik_pink: target poses are sent as action and Pink IK converts to joint angles
internally. No manual ik_server calls. Scene has cube_left and cube_right.
"""

import gymnasium as gym
import torch
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from loguru import logger as log

from magicsim.Task.TableTop.Env.DualReachEnv import DualReachEnv


@hydra.main(version_base=None, config_path="../../Conf", config_name="dual_reach_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: DualReachEnv = gym.make(
        "DualReachEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    obs, _ = env.reset()

    device = env.device
    num_envs = env.num_envs
    env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
    # ik_pink: arm 14D (left 7D + right 7D), eef 2D -> total 16D
    eef_open = torch.ones(2, device=device, dtype=torch.float32)

    step_count = 0
    while True:
        # One cube per arm (like multi_arm_reacher: two targets)
        # get_object_pose(env_ids, obj_type, obj_name, obj_id); cubes are "geometry", obj_id=0
        right_pose = env.get_object_pose(env_ids, "geometry", "cube_right", 0).to(
            device=device
        )  # [num_envs, 7]
        left_pose = env.get_object_pose(env_ids, "geometry", "cube_left", 0).to(
            device=device
        )  # [num_envs, 7]

        arm_action = torch.cat([right_pose, left_pose], dim=1)  # [num_envs, 14]
        eef = eef_open.unsqueeze(0).expand(num_envs, -1)  # [num_envs, 2]
        action = torch.cat([arm_action, eef], dim=1)  # [num_envs, 16]

        obs, reward, terminated, truncated, info, pending_env_ids = env.step(
            action=action
        )
        step_count += 1
        if step_count % 50 == 0:
            print(
                f"[TestDualReachEnv] step {step_count}, left->cube_left, right->cube_right."
            )


if __name__ == "__main__":
    main()
