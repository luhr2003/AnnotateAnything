import gymnasium as gym
from omegaconf import DictConfig
from omegaconf import OmegaConf
import hydra
from magicsim.Env.Environment.SyncCollectEnv import SyncCollectEnv
from magicsim.Env.Utils.file import Logger
from magicsim.Env.Capture.CaptureWriter import write_capture_data
from magicsim.Env.Utils.path import resolve_path
from loguru import logger as log
from magicsim import MAGICSIM_CONF
import time
import os


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="layout_config")
def main(cfg: DictConfig):
    new_seed = int(time.time())
    print(f"Initializing with seed: {new_seed}")

    OmegaConf.set_struct(cfg, False)
    cfg.sim.seed = new_seed
    OmegaConf.set_struct(cfg, True)

    print(cfg)
    logger = Logger("Env", log)
    env: SyncCollectEnv = gym.make(
        "SyncCollectEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    # Stabilize the environment
    print("Stabilizing environment...")
    for i in range(100):
        env.step()

    # Test CaptureWriter
    print("\n" + "=" * 50)
    print("Testing CaptureWriter.write_capture_data()")
    print("=" * 50)

    # Set up output directory
    output_dir = resolve_path("$MAGICSIM_HOME/TestOutput/test_capture_writer")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Capture data for multiple steps
    num_test_steps = 5
    print(f"\nCapturing {num_test_steps} steps...")

    for step_idx in range(num_test_steps):
        # Step the environment
        env.step()

        # Get capture data
        capture_data = env.capture_manager.step()

        if capture_data:
            print(
                f"  Step {step_idx}: Captured data from {len(capture_data)} camera(s)"
            )

            # Write capture data to disk
            write_capture_data(
                capture_data=capture_data,
                step_idx=step_idx,
                path=output_dir,
            )
            print(f"    Saved to: {output_dir}")
        else:
            print(f"  Step {step_idx}: No capture data available")

    print(f"\n✓ Test completed! Data saved to: {output_dir}")
    print("=" * 50 + "\n")

    while True:
        env.step()


if __name__ == "__main__":
    main()
