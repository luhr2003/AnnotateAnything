import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.SyncBaseEnv import SyncBaseEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="flow_config")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: SyncBaseEnv = gym.make(
        "SyncBaseEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    print("\n" + "=" * 80)
    print("FLOW EFFECTS TEST")
    print("=" * 80)
    print("\nRunning simulation...")
    print("You should see:")
    print("  - Dark smoke (left) - turbulent, chaotic rise")
    print("  - White steam (center) - gentle, wispy vapor")
    print("  - Tan dust (right) - particles that settle down")
    print("\nWait 100-150 steps for effects to build up...")
    print("=" * 80 + "\n")

    # Run simulation to let effects build up
    for i in range(150):
        env.step()

    print("\n✓ Effects should be visible now!")

    env.reset_idx([0, 2])

    for i in range(50):
        env.sim.sim_step()
    for i in range(150):
        env.step()

    env.reset_idx([0, 1])

    for i in range(50):
        env.step()

    env.reset_idx([2, 3])

    for i in range(50):
        env.step()

    env.reset_idx([0, 1, 3])

    for i in range(50):
        env.step()

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)
    print("\nFlow Effects Summary:")
    print("  • Smoke: Dark gray, turbulent, hot (layer 4)")
    print("  • Steam: White/wispy, gentle rise, quick fade (layer 2)")
    print("  • Dust: Earth tones, settles down, particulate (layer 1)")
    print("\nCustomization:")
    print("  • Adjust 'intensity' for heavier/lighter effects")
    print("  • Change 'color_brightness' for smoke darkness")
    print("  • Set 'dust_type' to tan/brown/gray/dark")
    print("  • Override physics parameters for custom behavior")
    print("=" * 80 + "\n")

    while True:
        env.step()


if __name__ == "__main__":
    main()
