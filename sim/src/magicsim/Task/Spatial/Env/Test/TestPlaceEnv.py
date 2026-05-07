from magicsim.Task.Spatial.Env.SpatialPlaceEnv import SpatialPlaceEnv
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from loguru import logger as log


@hydra.main(version_base=None, config_path="../../Conf", config_name="place_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: SpatialPlaceEnv = gym.make(
        "SpatialPlaceEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    env.step()

    while True:
        env.sim_step()


if __name__ == "__main__":
    main()
