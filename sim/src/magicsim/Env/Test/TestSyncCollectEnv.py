import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.SyncCollectEnv import SyncCollectEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="base_config")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: SyncCollectEnv = gym.make(
        "SyncCollectEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()
    for _ in range(100):
        env.step()

    while True:
        env.step()


if __name__ == "__main__":
    main()
