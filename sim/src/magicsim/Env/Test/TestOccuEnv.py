import gymnasium as gym
from omegaconf import DictConfig
from omegaconf import OmegaConf
import hydra
from magicsim.Env.Environment.SyncBaseEnv import SyncBaseEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF
import time
import numpy as np
import cv2
import os


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="room_config")
def main(cfg: DictConfig):
    new_seed = int(time.time())
    print(f"Initializing with seed: {new_seed}")

    OmegaConf.set_struct(cfg, False)
    cfg.sim.seed = new_seed
    OmegaConf.set_struct(cfg, True)

    print(cfg)
    logger = Logger("Env", log)
    env: SyncBaseEnv = gym.make(
        "SyncBaseEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    print("Waiting for scene to stabilize...")
    for i in range(50):
        env.step()
        if (i + 1) % 10 == 0:
            print(f"  Stabilized {i + 1} steps...")

    occupancy_manager = env.nav_manager.occupancy_manager
    num_envs = occupancy_manager.num_envs

    print(f"\nTesting OccupancyManager with {num_envs} environments")

    room_size = 10.0
    half_size = room_size / 2.0

    boundary = [
        -half_size,
        half_size,
        -half_size,
        half_size,
        -2,
        1,
    ]

    boundaries = [boundary] * num_envs

    # Scan center in LOCAL coordinates (relative to env_origin)
    scan_origin = [0.0, 0.0, 2.2]  # Center of room, height 2.2m
    scan_origins = [scan_origin] * num_envs

    print("Generating occupancy maps...")
    grids = occupancy_manager.generate(
        origin=scan_origins, boundary=boundaries, type="2d", env_ids=None
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    )
    output_dir = os.path.join(project_root, "occupancy_maps")
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(grids, list):
        grid_list = grids
    else:
        grid_list = [grids]

    for env_id, grid in enumerate(grid_list):
        if grid is None:
            print(f"  Env {env_id}: Grid is None")
            continue

        free_cells = np.sum(grid == 0)
        occupied_cells = np.sum(grid == 1)
        total_cells = grid.size
        free_ratio = free_cells / total_cells if total_cells > 0 else 0
        occupied_ratio = occupied_cells / total_cells if total_cells > 0 else 0

        vis_grid = (1 - grid) * 255
        vis_grid = vis_grid.astype(np.uint8)

        png_path = os.path.join(output_dir, f"occupancy_map_env_{env_id}.png")
        cv2.imwrite(png_path, vis_grid)

        npy_path = os.path.join(output_dir, f"occupancy_map_env_{env_id}.npy")
        occupancy_manager.save_grid_npy(npy_path, env_id=env_id)

        print(
            f"  Env {env_id}: shape={grid.shape}, free={free_ratio:.1%}, occupied={occupied_ratio:.1%}"
        )
        print(f"    Saved to: {png_path} and {npy_path}")

    print("\nTest completed successfully!")
    print(f"Occupancy maps saved to: {output_dir}")

    # ========================================
    # Test is_collided functionality
    # ========================================
    print("\n" + "=" * 80)
    print("Testing is_collided() - Path collision detection")
    print("=" * 80)

    test_env_id = 0  # Test with Env 0

    # Test various paths in LOCAL coordinates
    test_cases = [
        ([0.0, 0.0], [1.0, 1.0], "Diagonal path"),
        ([0.0, 0.0], [5.0, 0.0], "Straight path to wall"),
        ([-3.0, 0.0], [-2.5, 0.0], "Path available"),
        ([1.0, 1.0], [1.0, -1.0], "Vertical path"),
    ]

    print(f"\nTesting collision detection for Env {test_env_id}:")
    for start_local, end_local, desc in test_cases:
        try:
            has_collision = occupancy_manager.is_collided(
                src_xy_local=start_local, dst_xy_local=end_local, env_id=test_env_id
            )
            status = "COLLISION" if has_collision else "FREE"
            print(f"  [{status}] {desc}: {start_local} -> {end_local}")
        except Exception as e:
            print(f"  [ERROR] {desc}: {e}")

    # ========================================
    # Test is_escaped functionality
    # ========================================
    print("\n" + "=" * 80)
    print("Testing is_escaped() - Position escape detection")
    print("=" * 80)

    # Test various positions in LOCAL coordinates
    test_positions = [
        ([0.0, 0.0], "Center"),
        ([3.0, 3.0], "Interior point"),
        ([10.0, 10.0], "Far outside boundary"),
        ([-10.0, -10.0], "Far outside boundary"),
    ]

    print(f"\nTesting escape detection for Env {test_env_id}:")
    for pos_local, desc in test_positions:
        try:
            escaped = occupancy_manager.is_escaped(
                xy_local=pos_local, env_id=test_env_id
            )
            status = "ESCAPED" if escaped else "INSIDE"
            print(f"  [{status}] {desc}: {pos_local}")
        except Exception as e:
            print(f"  [ERROR] {desc}: {e}")

    # Test multi-environment collision detection

    print("\n" + "=" * 80)
    print("Testing multi-environment collision detection")
    print("=" * 80)

    # Test same local path in different environments
    start_local = [0.0, 0.0]
    end_local = [2.0, 2.0]

    print(f"\nTesting path {start_local} -> {end_local} in all environments:")
    for env_id in range(num_envs):
        try:
            has_collision = occupancy_manager.is_collided(
                src_xy_local=start_local, dst_xy_local=end_local, env_id=env_id
            )
            status = "COLLISION" if has_collision else "FREE"
            print(
                f"  Env {env_id} [{status}] Local coords: {start_local} -> {end_local}"
            )
        except Exception as e:
            print(f"  Env {env_id} [ERROR]: {e}")

    print("\n" + "=" * 80)
    print("All collision and escape detection tests completed!")
    print("=" * 80)

    print("\nSimulation running... Press Ctrl+C to stop.")

    try:
        while True:
            env.step()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")


if __name__ == "__main__":
    main()
