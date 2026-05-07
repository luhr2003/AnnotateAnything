import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.BaseEnv import BaseEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="startup_config")
def main(cfg: DictConfig):
    print(cfg)
    cfg = cfg.sim
    logger = Logger("Env", log)
    env: BaseEnv = gym.make("BaseEnv-V0", config=cfg, cli_args=None, logger=logger)
    env.reset()
    while True:
        env.step()


if __name__ == "__main__":
    main()
