import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.SyncBaseEnv import SyncBaseEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="fire_config")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: SyncBaseEnv = gym.make(
        "SyncBaseEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    for i in range(150):
        env.step()

    env.reset_idx([0, 2])

    for i in range(50):
        env.sim.sim_step()
    for i in range(150):
        env.step()

    env.reset_idx([0, 1])

    for i in range(50):
        env.step()

    env.reset_idx([2, 3])

    for i in range(50):
        env.step()

    env.reset_idx([0, 1, 3])

    for i in range(50):
        env.step()

    while True:
        env.step()


if __name__ == "__main__":
    main()
