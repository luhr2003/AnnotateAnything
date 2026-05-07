from magicsim.Task.LocoManip.Env.LocoReachEnv import LocoReachEnv
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from loguru import logger as log


@hydra.main(
    version_base=None, config_path="../../Conf", config_name="dual_loco_reach_env"
)
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: LocoReachEnv = gym.make(
        "LocoReachEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    obs, info = env.reset()

    while 1:
        env.sim_step()


if __name__ == "__main__":
    main()
