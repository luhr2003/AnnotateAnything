import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.SyncBaseEnv import SyncBaseEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="rice_config")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: SyncBaseEnv = gym.make(
        "SyncBaseEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    points = None
    for i in range(100):
        env.step()
        if i == 10:
            points, _, _ = env.scene_manager.fluid_objects[2]["fluid_items"][
                0
            ].get_particle_positions(visualize=True)

    env.scene_manager.fluid_objects[2]["fluid_items"][0].set_particle_positions(points)
    for _ in range(100):
        env.step()

    for i in range(5):
        env.reset_idx([0, 1, 2, 3])
        for _ in range(100):
            env.step()

    while True:
        env.step()


if __name__ == "__main__":
    main()
