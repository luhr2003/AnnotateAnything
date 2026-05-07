from magicsim.Task.TableTop.Env.ReachEnv import ReachEnv
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from loguru import logger as log


@hydra.main(version_base=None, config_path="../../Conf", config_name="reach_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: ReachEnv = gym.make("ReachEnv-V0", config=cfg, cli_args=None, logger=logger)
    obs, info = env.reset()

    while 1:
        action = obs["privilege_obs"]["cube_pose"]
        obs, reward, terminated, truncated, info, pending_env_ids = env.step(
            action=action
        )
        # we here save the obs from camera 1
        camera_rgb_capture = obs["policy_obs"]["camera_info"][0]["rgb"]


if __name__ == "__main__":
    main()
