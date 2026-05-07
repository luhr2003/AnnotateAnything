from magicsim.Task.MobileManip.Env.MobileGraspEnv import MobileGraspEnv
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from loguru import logger as log


@hydra.main(version_base=None, config_path="../../Conf", config_name="mobile_grasp_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: MobileGraspEnv = gym.make(
        "MobileGraspEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    obs, info = env.reset()

    while True:
        step_result = env.step(action=None)
        obs = step_result[0]


if __name__ == "__main__":
    main()
