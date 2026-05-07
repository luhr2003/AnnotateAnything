"""Closed-loop bimanual basket grasp via AutoCollect.

Drives :class:`BiGrasp` (Collect/Command/BiGrasp.py) through the
AutoCollectManager — sequentially issues right-arm Grasp then left-arm
Grasp on :class:`BiGraspEnv`. Mirrors :file:`TestGraspEnv.py` in shape;
the only differences are the env id (``BiGraspEnv-V0``), the hydra config
name (``bi_grasp``), and the task name (``BiGrasp``) in TASK_STRING_DICT.
"""

from omegaconf import DictConfig
import hydra
from loguru import logger as log

import gymnasium as gym

from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.AutoCollectEnv import AutoCollectEnv


TASK_STRING_DICT = {"BiGrasp": 1.0}


@hydra.main(version_base=None, config_path="../Conf", config_name="bi_grasp")
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
