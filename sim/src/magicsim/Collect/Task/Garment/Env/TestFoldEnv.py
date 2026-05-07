"""Auto-collect entrypoint for the bimanual Fold skill on GarmentFoldEnv."""

import hydra
import gymnasium as gym
from loguru import logger as log
from omegaconf import DictConfig

from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.AutoCollectEnv import AutoCollectEnv
from magicsim.Task.Garment.Env.GarmentFoldEnv import GarmentFoldEnv  # noqa: F401 (gym register)


TASK_STRING_DICT = {"Fold": 1.0}


@hydra.main(version_base=None, config_path="../Conf", config_name="fold")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: AutoCollectEnv = gym.make(
        "AutoCollectEnv-V0",
        task_string=TASK_STRING_DICT,
        config=cfg,
        cli_args=None,
        logger=logger,
    )

    env.start_collect()


if __name__ == "__main__":
    main()
