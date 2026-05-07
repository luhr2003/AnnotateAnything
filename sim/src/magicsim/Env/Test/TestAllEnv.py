import gymnasium as gym
from omegaconf import DictConfig
from omegaconf import OmegaConf
import hydra
from magicsim.Env.Environment.SyncBaseEnv import SyncBaseEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF
import time


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="all_config")
def main(cfg: DictConfig):
    new_seed = int(time.time())
    print(f"Initializing with seed: {new_seed}")

    OmegaConf.set_struct(cfg, False)
    cfg.sim.seed = new_seed
    OmegaConf.set_struct(cfg, True)

    print(cfg)
    logger = Logger("AllEnv", log)
    env: SyncBaseEnv = gym.make(
        "SyncBaseEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()
    # Export two JSON files using LayoutManager
    print("\n=== Exporting object layouts and ranges to JSON ===")

    # Export 1: Object layouts with bounding boxes (actual positions)
    layouts_file = "outputs/all_objects_layouts.json"
    env.scene_manager.layout_manager.export_objects_layouts_to_json(
        output_file=layouts_file,
        env_ids=None,  # Export all environments
        timestamp=True,
    )

    # Export 2: Object ranges from configuration (spawn ranges)
    ranges_file = "outputs/all_objects_ranges.json"
    env.scene_manager.layout_manager.export_objects_ranges_to_json(
        output_file=ranges_file, timestamp=True
    )

    print("=== Export completed ===\n")

    for i in range(200):
        env.sim.sim_step()

    for i in range(5):
        print(f"Reset environment [0] - iteration {i + 1}")
        env.reset_idx([0])
        for _ in range(200):
            env.sim.sim_step()

    layout_manager = env.scene_manager.layout_manager
    all_objects = layout_manager.get_objects(0)

    print("\nObjects in environment:")
    for object_key, object_list in all_objects.items():
        if object_list:
            print(f"  {object_key}: {len(object_list)} instances")

    while True:
        rand_action_batched = env.robot_manager.sample_actions(batched=True)
        env.step(action=rand_action_batched)


if __name__ == "__main__":
    main()
