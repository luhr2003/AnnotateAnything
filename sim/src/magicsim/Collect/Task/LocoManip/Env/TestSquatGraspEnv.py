"""
Test SquatDexGrasp with AutoCollect: bottle on floor, robot squats to grasp.
Uses SquatGraspEnv (bottle z=0.05, termination object_z>0.2).
"""

from omegaconf import DictConfig
import hydra
from loguru import logger as log
from magicsim.Env.Utils.file import Logger
import gymnasium as gym
from magicsim.StardardEnv.Robot.AutoCollectEnv import AutoCollectEnv


TASK_STRING_DICT = {"SquatDexGrasp": 1.0}


@hydra.main(
    version_base=None,
    config_path="../Conf",
    config_name="squat_grasp",
)
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
