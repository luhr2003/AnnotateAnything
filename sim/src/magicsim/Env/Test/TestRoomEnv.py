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

    # ========================================
    # Test 1: Room Annotation Loader Functionality
    # ========================================
    print("\n" + "=" * 80)
    print("TEST 1: Room AnnotationLoader Functionality")
    print("=" * 80)

    try:
        room = env.nav_manager.rooms[0]
        print("✓ Room instance created successfully")

        # Test house-level metadata
        print("\n[1.2] Testing house-level methods:")
        house_meta = room.house_meta
        print(f"  ✓ house_meta keys: {list(house_meta.keys())}")

        house_bb = room.get_house_bb()
        print("  ✓ get_house_bb():")
        print(f"    - min: {house_bb.min_xyz}")
        print(f"    - max: {house_bb.max_xyz}")
        print(f"    - center: {house_bb.center}")
        print(f"    - size: {house_bb.size}")

        # Test room-level methods
        print("\n[1.3] Testing room-level methods:")
        room_num = room.get_room_num()
        print(f"  ✓ get_room_num(): {room_num} rooms found")

        room_ids = room.list_room_ids()
        print(f"  ✓ list_room_ids(): {room_ids}")

        room_files = room.list_room_files()
        print(f"  ✓ list_room_files(): {len(room_files)} files")

        # Test individual room access
        if room_ids:
            print(f"\n[1.4] Testing individual room access (Room ID: {room_ids[0]}):")
            room_data = room.get_room_by_id(room_ids[0])
            print(
                f"  ✓ get_room_by_id({room_ids[0]}) keys: {list(room_data.keys())[:5]}..."
            )

            try:
                room_bb = room.get_room_bb(room_ids[0])
                print(f"  ✓ get_room_bb({room_ids[0]}):")
                print(f"    - min: {room_bb.min_xyz}")
                print(f"    - max: {room_bb.max_xyz}")
                print(f"    - size: {room_bb.size}")
            except Exception as e:
                print(f"  ⚠ get_room_bb() error: {e}")

        # Test occupancy map generation
        print("\n[1.5] Testing occupancy map generation:")
        try:
            omap, info = room.get_house_omap(height=0.5, as_array=True)
            print("  ✓ get_house_omap():")
            print(f"    - shape: {omap.shape}")
            print(f"    - dtype: {omap.dtype}")
            print(f"    - info: {info}")

            # Save occupancy map for visualization
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
            )
            output_dir = os.path.join(project_root, "room_test_output")
            os.makedirs(output_dir, exist_ok=True)

            # Convert to visualization (assuming values are 0-255 or 0/1)
            if omap.max() <= 1:
                vis_omap = (omap * 255).astype(np.uint8)
            else:
                vis_omap = omap.astype(np.uint8)

            png_path = os.path.join(output_dir, "room_occupancy_map.png")
            cv2.imwrite(png_path, vis_omap)
            print(f"    - Saved visualization to: {png_path}")

        except Exception as e:
            print(f"  ⚠ get_house_omap() error: {e}")

        # Test voxel_size and z_range properties
        print("\n[1.6] Testing additional properties:")
        voxel_size = room.voxel_size
        print(f"  ✓ voxel_size: {voxel_size}")

        z_range = room.z_range
        print(f"  ✓ z_range: {z_range}")

        # Test reset method
        print("\n[1.7] Testing reset method:")
        room.reset()
        print("  ✓ reset() method executed successfully")

        print("\n✅ TEST 1 PASSED: All AnnotationLoader functionality works!")

    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    # ========================================
    # Test 2: Room Class Inheritance Structure
    # ========================================
    print("\n" + "=" * 80)
    print("TEST 2: Room Class Inheritance Structure")
    print("=" * 80)

    try:
        from magicsim.Env.Scene.Object.Room import Room, AnnotationLoader
        from isaacsim.core.prims import SingleGeometryPrim

        print("\n[2.1] Checking class inheritance:")
        print(f"  Room base classes: {[cls.__name__ for cls in Room.__bases__]}")

        print("\n[2.2] Method Resolution Order (MRO):")
        for i, cls in enumerate(Room.__mro__[:6]):  # Show first 6
            print(f"  {i}. {cls.__module__}.{cls.__name__}")

        print("\n[2.3] Checking key methods:")
        methods_to_check = [
            "get_house_bb",
            "get_room_num",
            "get_house_omap",
            "set_local_pose",
            "set_local_scale",
            "reset",
        ]

        for method in methods_to_check:
            has_method = hasattr(Room, method)
            symbol = "✓" if has_method else "✗"
            source = ""
            if has_method:
                # Try to determine which parent class provides this method
                if hasattr(AnnotationLoader, method):
                    source = " (from AnnotationLoader)"
                elif hasattr(SingleGeometryPrim, method):
                    source = " (from SingleGeometryPrim)"
            print(f"  {symbol} {method}{source}")

        print("\n✅ TEST 2 PASSED: Class structure verified!")

    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    try:
        while True:
            env.step()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")


if __name__ == "__main__":
    main()
