"""Close-loop dex open-drawer collect driver.

Mirrors :mod:`magicsim.Collect.Task.TableTop.Env.TestOpenDrawerEnv` but uses
the dexterous variant (``DexOpenDrawerEnv-V0`` + ``DexOpenDrawer`` task /
atomic skill) so the Franka + Xhand robot can be auto-collected through the
``xhand_open_by_handle_trajectory`` annotation on Drawer/7120
(source data: ``~/sharpa_bin+xhand_open_by_handle/xhand_open_by_handle``).
"""

from omegaconf import DictConfig
import hydra
from loguru import logger as log
from magicsim.Env.Utils.file import Logger
import gymnasium as gym
from magicsim.StardardEnv.Robot.AutoCollectEnv import AutoCollectEnv

TASK_STRING_DICT = {"DexOpenDrawer": 1.0}


@hydra.main(version_base=None, config_path="../Conf", config_name="dex_open_drawer")
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
