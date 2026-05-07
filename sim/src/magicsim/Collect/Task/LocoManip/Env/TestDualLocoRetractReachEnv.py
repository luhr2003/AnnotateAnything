from omegaconf import DictConfig
import hydra
from loguru import logger as log
from magicsim.Env.Utils.file import Logger
import gymnasium as gym
from magicsim.StardardEnv.Robot.AutoCollectEnv import AutoCollectEnv


TASK_STRING_DICT = {"DualLocoRetractReach": 1.0}


@hydra.main(
    version_base=None, config_path="../Conf", config_name="dual_loco_retract_reach"
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
