import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.SyncBaseEnv import SyncBaseEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="garment_config")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: SyncBaseEnv = gym.make(
        "SyncBaseEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    scene_mgr = env.scene_manager
    garments = []
    for env_id in range(env.num_envs):
        garment_dict = scene_mgr.garment_objects[env_id]
        for category, garment_list in garment_dict.items():
            garments.extend(garment_list)

    # Settle garments on the table before picking keypoints
    print("Settling garments (50 steps)...")
    for _ in range(50):
        env.step()

    print("Computing and visualizing keypoints...")
    for garment in garments:
        garment.update_keypoint()
        garment.visualize_keypoint()

    print("Running simulation; refreshing keypoint visualization every 500 steps...")
    step = 0
    while True:
        env.step()
        step += 1
        if step % 100 == 0:
            for garment in garments:
                garment.visualize_keypoint()


if __name__ == "__main__":
    main()
