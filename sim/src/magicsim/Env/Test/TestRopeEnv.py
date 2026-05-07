import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.SyncBaseEnv import SyncBaseEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="rope_config")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: SyncBaseEnv = gym.make(
        "SyncBaseEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    rope_object = env.scene_manager.rope_objects[0]["rope_items"][0]
    positions = None
    for i in range(100):
        env.step()
        if i == 10:
            positions, _, _ = rope_object.get_current_mesh_points(
                visualize=True, save=False
            )

    rope_object.set_current_mesh_points(positions, None, None)

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

    env.reset_idx([1, 3])
    for _ in range(100):
        env.step()

    env.reset_idx([1, 3])

    while True:
        env.step()


if __name__ == "__main__":
    main()
