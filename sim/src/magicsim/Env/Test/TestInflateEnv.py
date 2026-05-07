import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.SyncBaseEnv import SyncBaseEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="inflate_config")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: SyncBaseEnv = gym.make(
        "SyncBaseEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    # Try capture and restore current mesh points of an inflatable object
    inflatable = env.scene_manager.inflatable_objects[0]["inflate_items"][0]

    points = None
    for i in range(200):
        env.step()
        if i == 10:
            points, _, _, _ = inflatable.get_current_mesh_points(
                visualize=True, save=False
            )

    inflatable.set_current_mesh_points(points)

    for i in range(3):
        env.reset_idx([0, 1, 2, 3])
        for _ in range(200):
            env.step()

    env.reset_idx([0, 2], seed=[50, 70])
    for _ in range(200):
        env.step()

    env.reset_idx([0, 2])
    for _ in range(200):
        env.step()

    env.reset_idx([1, 3], seed=[90, 110])
    for _ in range(200):
        env.step()

    env.reset_idx([1, 2], seed=[120, 130])
    for _ in range(200):
        env.step()

    env.reset_idx([0, 3], seed=[140, 150])
    for _ in range(200):
        env.step()

    env.reset_idx([0, 3])

    while True:
        env.step()


if __name__ == "__main__":
    main()
