import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.SyncBaseEnv import SyncBaseEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="femcloth_config")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: SyncBaseEnv = gym.make(
        "SyncBaseEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    cached_points = None
    cached_pos_world = None
    cached_ori_world = None
    fem_cloth_object = None

    for i in range(100):
        env.step()
        if i == 10:
            fem_cloth_object = env.scene_manager.fem_cloth_objects[0]["fem_cloth"][0]
            _, cached_points, cached_pos_world, cached_ori_world = (
                fem_cloth_object.get_current_mesh_points(visualize=True, save=False)
            )

    if fem_cloth_object is not None and cached_points is not None:
        fem_cloth_object.set_current_mesh_points(
            cached_points, cached_pos_world, cached_ori_world
        )

    for _ in range(100):
        env.step()

    for i in range(5):
        env.reset_idx([0, 1, 2, 3])
        for _ in range(100):
            env.step()

    env.reset_idx([0, 2])
    for _ in range(100):
        env.step()

    env.reset_idx([1, 3])
    for _ in range(100):
        env.step()

    env.reset_idx([1, 2])
    for _ in range(100):
        env.step()

    env.reset_idx([0, 3])
    for _ in range(100):
        env.step()

    env.reset_idx([0, 3])

    while True:
        env.step()


if __name__ == "__main__":
    main()
