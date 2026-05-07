import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.SyncRobotEnv import SyncRobotEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF
import torch


@hydra.main(
    version_base=None, config_path=MAGICSIM_CONF, config_name="quadruped_config"
)
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: SyncRobotEnv = gym.make(
        "SyncRobotEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    while 1:
        action = torch.tensor(
            [
                0.4,
                0,
                0.3,
                0.2,
            ]
        )
        action = action.unsqueeze(0)
        action = action.repeat(env.num_envs, 1)
        env.step(action=action)

    env.close()


if __name__ == "__main__":
    main()
