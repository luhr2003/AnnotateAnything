from magicsim.Task.Spatial.Env.MotionEnv import MotionEnv
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from loguru import logger as log


@hydra.main(version_base=None, config_path="../../Conf", config_name="motion_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: MotionEnv = gym.make("MotionEnv-V0", config=cfg, cli_args=None, logger=logger)
    env.reset()

    while True:
        env.step()


if __name__ == "__main__":
    main()
