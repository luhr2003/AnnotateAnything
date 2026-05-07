from typing import Dict
import gymnasium as gym
from omegaconf import DictConfig
from omegaconf import OmegaConf
import hydra
from magicsim.Env.Environment.SyncCollectEnv import SyncCollectEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF
import time
from magicsim.Env.Utils.strings import parse_string_to_tuple
import numpy as np


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="spatial_config")
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

    num_envs = env.num_envs
    # ========================================
    # Test 1: Generate paths for ALL environments with DIFFERENT waypoints
    # ========================================
    print(
        f"\n[Test 1] Generating paths for ALL {num_envs} environments with different waypoints"
    )

    start_points_all = []
    coords_all = []

    # Define different paths for each environment
    # Use z=-0.77 (NavMesh surface height in local coordinates)
    navmesh_z = -0.77

    for i in range(num_envs):
        # Different start points for each env
        start = np.array([1.0 + i * 0.5, 1.0 + i * 0.3, navmesh_z])
        start_points_all.append(start)

        # Different waypoints for each env
        if i % 2 == 0:  # Even envs: square path
            waypoints = [
                np.array([2.0, 1.0, navmesh_z]),
                np.array([2.0, 3.0, navmesh_z]),
                np.array([0.5, 3.0, navmesh_z]),
            ]
        else:  # Odd envs: diagonal path
            waypoints = [
                np.array([1.5, 2.0, navmesh_z]),
                np.array([2.5, 2.5, navmesh_z]),
                np.array([3.0, 3.5, navmesh_z]),
            ]
        coords_all.append(waypoints)

    env_ids_all = list(range(num_envs))

    print("Configuration:")
    for i in range(num_envs):
        print(
            f"  Env {i}: start={start_points_all[i]}, waypoints={len(coords_all[i])} points"
        )

    # Initialize variables for Test 4
    paths_all = []
    path_objects_all = []
    paths_all, path_objects_all = env.nav_manager.generate_path(
        start_point=start_points_all,
        coords=coords_all,
        env_ids=env_ids_all,
        visualize=True,
        return_path_objects=True,
    )

    print("\nResults:")
    for i, path in enumerate(paths_all):
        if path and len(path) > 0:
            print(f"  Env {i}: Generated {len(path)} waypoints")
            if path_objects_all and i < len(path_objects_all):
                path_obj_status = (
                    "with path object" if path_objects_all[i] else "no path object"
                )
                print(f"           ({path_obj_status})")
        else:
            print(f"  Env {i}: Failed to generate path")

    for i in range(100):
        env.sim.sim_step()

    # ========================================
    # Test 2: Test nav  manager query_closest_point and path_closest_distance
    # ========================================
    print("\n[Test 2] Testing NavMesh query functions")

    # Test 2.1: query_random_point
    print("\n[2.1] Testing query_random_point:")

    print("  Generating 5 random points on NavMesh:")
    for i in range(5):
        try:
            local_pos, env_idx = env.nav_manager.query_random_point()
            print(f"    Random point {i}: local={local_pos}, env_idx={env_idx}")
        except Exception as e:
            print(f"    Error generating random point {i}: {e}")

    # Test 2.2: query_closest_point
    print("\n[2.2] Testing query_closest_point:")

    # Test with points at different heights to verify projection onto NavMesh
    test_positions = [
        np.array([1.5, 2.3, 0.5]),
        np.array([2.8, 3.1, -0.77]),
        np.array([4.0, 7.0, -0.77]),
    ]

    for pos_idx, test_pos in enumerate(test_positions):
        closest_point = env.nav_manager.query_closest_point(position=test_pos, env_id=0)
        distance_to_navmesh = np.linalg.norm(test_pos - closest_point)
        print(
            f"  Test position {pos_idx}: {test_pos} -> Closest on NavMesh: {closest_point}"
        )
        print(f"    Distance to NavMesh: {distance_to_navmesh:.4f}")

    # ========================================
    # Test 3: Test room manager
    # ========================================
    print("\n[Test 3] Testing room manager")
    room = env.nav_manager.rooms[0]
    print("✓ Room instance created successfully")

    # Test house-level metadata
    print("\n[3.2] Testing house-level methods:")
    house_meta = room.house_meta
    print(f"  ✓ house_meta keys: {list(house_meta.keys())}")

    house_bb = room.get_house_bb()
    print("  ✓ get_house_bb():")
    print(f"    - min: {house_bb.min_xyz}")
    print(f"    - max: {house_bb.max_xyz}")
    print(f"    - center: {house_bb.center}")
    print(f"    - size: {house_bb.size}")

    # Test room-level methods
    print("\n[3.3] Testing room-level methods:")
    room_num = room.get_room_num()
    print(f"  ✓ get_room_num(): {room_num} rooms found")

    room_ids = room.list_room_ids()
    print(f"  ✓ list_room_ids(): {room_ids}")

    room_files = room.list_room_files()
    print(f"  ✓ list_room_files(): {len(room_files)} files")

    # Test individual room access
    if room_ids:
        print(f"\n[3.4] Testing individual room access (Room ID: {room_ids[0]}):")
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
    print("\n[3.5] Testing occupancy map generation:")
    omap = room.get_house_omap(height=0.5)
    print("  ✓ get_house_omap():")
    print(f"    - shape: {omap.shape}")
    print(f"    - dtype: {omap.dtype}")

    print("\n[3.6] Test Room Boundary:")
    room_boundary = room.get_room_boundary(room_ids[0])
    print(f"  ✓ get_room_boundary({room_ids[0]}):")
    print(f"    - boundary: {room_boundary}")

    print("\n[3.7] Test Room BB:")
    room_bb = room.get_room_bb(room_ids[0])
    print(f"  ✓ get_room_bb({room_ids[0]}):")
    print(f"    - min: {room_bb.min_xyz}")
    print(f"    - max: {room_bb.max_xyz}")
    print(f"    - size: {room_bb.size}")
    print(f"    - center: {room_bb.center}")

    print("\n[3.8] Test Room OMap:")
    room_omap = room.get_room_omap(room_ids[0], height=0.5)
    print(f"  ✓ get_room_omap({room_ids[0]}):")
    print(f"    - shape: {room_omap.shape}")
    print(f"    - dtype: {room_omap.dtype}")

    # ========================================
    # Test 4: Test camera manager
    # ========================================
    action_list = ["go ([0, 0, 0], [0, 0, 5])"] * 72
    action_list.insert(0, "move_to ([0, 0, 1], [0, 0, 0])")

    def text2action(text_actions: str) -> Dict:
        # This function is to test all the movings of camera manager. DEBUG ONLY
        line = text_actions.lower().strip()
        action = line.split()[0]
        param = ""
        for i in range(1, len(line.split())):
            param += line.split()[i] + " "

        param = parse_string_to_tuple(param)
        processed_actions = {}
        if param is None:
            param = ([0, 0, 0], [0, 0, 0])
        if action == "go":
            action = {"go": param}
            for env_id in range(num_envs):
                processed_actions[env_id] = {0: action}
            return processed_actions
        elif action == "move":
            action = {"move": param}
            for env_id in range(num_envs):
                processed_actions[env_id] = {0: action}
            return processed_actions
        elif action == "move_to":
            action = {"move_to": param}
            for env_id in range(num_envs):
                processed_actions[env_id] = {0: action}
            return processed_actions
        elif action == "rand":
            action = {"randomize_camera_pose": param}
            for env_id in range(num_envs):
                processed_actions[env_id] = {0: action}
            return processed_actions  # param of randomize_camera_pose is not like others, test later
        elif action == "look_at":
            action = {"look_at": param}
            for env_id in range(num_envs):
                processed_actions[env_id] = {0: action}
            return processed_actions  # look_at haven't done, test later
        else:
            return None

    print("Test 4: Test camera manager")
    print("Test 4.1: Test camera manager step")
    for action in action_list:
        actions = text2action(action)
        env.step(actions)

    print("Test 4.2: Test camera manager reset")
    env.reset_idx()
    action_list[0] = "move_to ([1, 0, 1], [0, 0, 0])"
    for action in action_list:
        actions = text2action(action)
        env.step(actions)

    while True:
        env.sim.sim_step()


if __name__ == "__main__":
    main()
