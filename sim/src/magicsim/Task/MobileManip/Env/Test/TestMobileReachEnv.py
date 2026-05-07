from magicsim.Task.MobileManip.Env.MobileReachEnv import MobileReachEnv
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
import torch


@hydra.main(version_base=None, config_path="../../Conf", config_name="mobile_reach_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: MobileReachEnv = gym.make(
        "MobileReachEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    obs, info = env.reset()

    step_count = 0
    while True:
        cube_pose = obs["privilege_obs"]["cube_pose"]  # [N, 7]: pos + quat (wxyz)
        num_envs = cube_pose.shape[0]
        device = cube_pose.device

        # Base action: zeros (keep base stationary)
        base_action = torch.zeros(num_envs, 3, device=device)

        # Concatenate all actions
        action = torch.cat([base_action, cube_pose], dim=-1)  # [N, 11]

        # 获取更新后的观测值，这样拖动方块后 cube_pose 会更新
        step_result = env.step(action=action)
        obs = step_result[0]  # 更新 obs 以获取最新的 cube_pose


if __name__ == "__main__":
    main()
