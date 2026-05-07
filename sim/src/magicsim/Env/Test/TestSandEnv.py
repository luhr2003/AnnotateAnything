import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.SyncBaseEnv import SyncBaseEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="sand_config")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: SyncBaseEnv = gym.make(
        "SyncBaseEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset(seed=[10, 20, 30, 40])

    # Run simulation for initial settling
    for _ in range(10):
        env.step()

    # Get sand object and particle positions
    target_object = env.scene_manager.sand_objects[0]["sand_items"][0]
    particle_positions = target_object.get_particle_positions(
        visualize=True, save=False
    )
    print(f"Initial particle count: {len(particle_positions)}")

    # Continue simulation
    for _ in range(200):
        env.step()

    # Get current particle positions
    current_positions = target_object.get_particle_positions(visualize=False)
    print(f"Current particle count: {len(current_positions)}")

    # Reset particles to initial positions
    target_object.set_particle_positions(particle_positions)

    # Continue simulation after reset
    for _ in range(100):
        env.step()

    # Test multiple resets
    for i in range(5):
        env.reset_idx([0, 1], seed=[20 + i * 10, 30 + i * 10])
        for _ in range(100):
            env.step()
        positions = target_object.get_particle_positions(visualize=False)
        print(f"Reset {i + 1}: particle count = {len(positions)}")

    env.reset_idx([0, 1], seed=[60, 70])
    for _ in range(100):
        env.step()

    env.reset_idx([0, 1])
    for _ in range(100):
        env.step()

    env.reset_idx([0, 1], seed=[90, 100])

    # # Continuous simulation
    while True:
        env.step()


if __name__ == "__main__":
    main()
