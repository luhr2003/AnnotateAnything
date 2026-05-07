import gymnasium as gym
from omegaconf import DictConfig
from omegaconf import OmegaConf
import hydra
from magicsim.Env.Environment.SyncRobotEnv import SyncRobotEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF
import time


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="base_config")
def main(cfg: DictConfig):
    new_seed = int(time.time())
    print(f"Initializing with seed: {new_seed}")

    OmegaConf.set_struct(cfg, False)
    cfg.sim.seed = new_seed
    OmegaConf.set_struct(cfg, True)

    print(cfg)
    logger = Logger("Env", log)
    env: SyncRobotEnv = gym.make(
        "SyncRobotEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    for i in range(120):
        env.step()
        pose = env.scene_manager.rigid_objects[0]["mug"][0].get_local_pose()
        print(f"Mug pose: {pose}")

    for i in range(5):
        print(f"Reset environments - iteration {i + 1}")
        env.reset_idx([0, 1, 2, 3])

        for step_idx in range(100):
            env.step()

    while True:
        env.step()


if __name__ == "__main__":
    main()
