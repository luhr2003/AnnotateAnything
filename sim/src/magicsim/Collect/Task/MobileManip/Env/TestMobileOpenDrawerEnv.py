"""Test MobileOpenDrawer task. Cabinet on the floor + ridgebackFranka.

Reuses the OpenDrawer atomic skill; the atomic_skill config sets
``OpenDrawer.mobile: true`` so MoveL/ServoL are swapped for
MobileMoveL/MobileServoL under the hood.
"""

from omegaconf import DictConfig
import hydra
from loguru import logger as log
from magicsim.Env.Utils.file import Logger
import gymnasium as gym
from magicsim.StardardEnv.Robot.AutoCollectEnv import AutoCollectEnv


TASK_STRING_DICT = {"MobileOpenDrawer": 1.0}


@hydra.main(version_base=None, config_path="../Conf", config_name="mobile_open_drawer")
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
