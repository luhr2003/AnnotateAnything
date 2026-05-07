"""Closed-loop bimanual handover via AutoCollect.

Drives :class:`Handover` (Collect/Command/Handover.py) through the
AutoCollectManager — issues a single ``Handover`` task on
:class:`HandoverEnv`. Mirrors :file:`TestBiGraspEnv.py` in shape; the
only differences are the env id (``HandoverEnv-V0``), the hydra config
name (``handover``), and the task name (``Handover``) in
``TASK_STRING_DICT``.
"""

from omegaconf import DictConfig
import hydra
from loguru import logger as log

import gymnasium as gym

from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.AutoCollectEnv import AutoCollectEnv


TASK_STRING_DICT = {"Handover": 1.0}


@hydra.main(version_base=None, config_path="../Conf", config_name="handover")
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
