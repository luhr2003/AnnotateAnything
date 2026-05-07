import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.SyncBaseEnv import SyncBaseEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="deform_config")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: SyncBaseEnv = gym.make(
        "SyncBaseEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset(seed=[10, 20, 30, 40])

    for _ in range(10):
        env.step()

    target_object = env.scene_manager.deformable_objects[0]["deformable_items"][0]
    _, points, pos_world, ori_world = target_object.get_current_mesh_points(
        visualize=True, save=False
    )

    for _ in range(200):
        env.step()

    target_object.set_current_mesh_points(points, pos_world, ori_world)

    for _ in range(100):
        env.step()

    for i in range(5):
        env.reset_idx([0, 1, 2, 3], seed=[20, 30, 40, 50])
        for _ in range(100):
            env.step()

    env.reset_idx([0, 2], seed=[60, 80])
    for _ in range(100):
        env.step()

    env.reset_idx([1, 2])
    for _ in range(100):
        env.step()

    env.reset_idx([1, 3], seed=[90, 110])

    while True:
        env.step()


if __name__ == "__main__":
    main()
