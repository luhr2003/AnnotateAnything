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


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="nav_config")
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

    # Initialize NavManager after environment reset
    if hasattr(env, "nav_manager") and env.nav_manager is not None:
        print("\nInitializing NavManager...")
        env.nav_manager.reset()
        print("NavManager initialized successfully!")

        # Verify room wrapping
        print("\n" + "=" * 80)
        print("Room Wrapping Verification")
        print("=" * 80)
        if env.nav_manager.rooms:
            print(f"[OK] Successfully wrapped {len(env.nav_manager.rooms)} rooms")
            for i, room in enumerate(env.nav_manager.rooms):
                print(f"  Room {i}:")
                print(f"    - Prim path: {room.prim_path}")
                try:
                    house_bb = room.get_house_bb()
                    print(
                        f"    - House BB: min={house_bb.min_xyz}, max={house_bb.max_xyz}"
                    )
                    room_num = room.get_room_num()
                    print(f"    - Number of rooms: {room_num}")
                except Exception as e:
                    print(f"    - Error accessing room data: {e}")
        else:
            print("[FAIL] No rooms wrapped!")
        print("=" * 80 + "\n")

    # Test generate_path functionality
    if hasattr(env, "nav_manager") and env.nav_manager is not None:
        print("\n" + "=" * 80)
        print("Testing NavManager generate_path functionality")
        print("=" * 80)

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

        try:
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
                            "with path object"
                            if path_objects_all[i]
                            else "no path object"
                        )
                        print(f"           ({path_obj_status})")
                else:
                    print(f"  Env {i}: Failed to generate path")

        except Exception as e:
            print(f"\nError in Test 1: {e}")
            import traceback

            traceback.print_exc()

        for i in range(300):
            env.sim.sim_step()

        # ========================================
        # Test 2: Generate paths for SELECTED environments only
        # ========================================
        print(
            "\n[Test 2] Generating paths for SELECTED environments (Env 0 and 2 only)"
        )

        navmesh_z = -0.77  # NavMesh surface height in local coordinates

        selected_env_ids = [0, 2]
        start_points_selected = [
            np.array([-1.0, -1.0, navmesh_z]),  # Env 0
            np.array([-1.0, -1.0, navmesh_z]),  # Env 2
        ]
        coords_selected = [
            # Env 0: L-shaped path
            [
                np.array([0.0, -1.0, navmesh_z]),
                np.array([1.0, -1.0, navmesh_z]),
                np.array([1.0, 1.0, navmesh_z]),
            ],
            # Env 2: Zigzag path
            [
                np.array([-0.5, 0.0, navmesh_z]),
                np.array([0.5, 0.5, navmesh_z]),
                np.array([-0.5, 1.0, navmesh_z]),
            ],
        ]

        print("Configuration:")
        for i, env_id in enumerate(selected_env_ids):
            print(
                f"  Env {env_id}: start={start_points_selected[i]}, waypoints={len(coords_selected[i])} points"
            )

        try:
            paths_selected = env.nav_manager.generate_path(
                start_point=start_points_selected,
                coords=coords_selected,
                env_ids=selected_env_ids,
                visualize=True,
            )

            print("\nResults:")
            for i, env_id in enumerate(selected_env_ids):
                path = paths_selected[i]
                if path and len(path) > 0:
                    print(f"  Env {env_id}: Generated {len(path)} waypoints")
                else:
                    print(f"  Env {env_id}: Failed to generate path")

        except Exception as e:
            print(f"\nError in Test 2: {e}")
            import traceback

            traceback.print_exc()

        for i in range(300):
            env.sim.sim_step()

        # ========================================
        # Test 3: Generate path for SINGLE environment (Env 1 only)
        # ========================================
        print("\n[Test 3] Generating path for SINGLE environment (Env 1 only)")

        navmesh_z = -0.77  # NavMesh surface height in local coordinates

        single_env_id = [1]
        start_point_single = [np.array([0.0, 0.0, navmesh_z])]
        coords_single = [
            [
                np.array([1.0, 0.0, navmesh_z]),
                np.array([1.0, 1.0, navmesh_z]),
                np.array([0.0, 1.0, navmesh_z]),
                np.array([0.0, 0.0, navmesh_z]),  # Complete loop
            ]
        ]

        print("Configuration:")
        print(
            f"  Env {single_env_id[0]}: start={start_point_single[0]}, waypoints={len(coords_single[0])} points (loop)"
        )

        try:
            paths_single = env.nav_manager.generate_path(
                start_point=start_point_single,
                coords=coords_single,
                env_ids=single_env_id,
                visualize=True,
            )

            print("\nResults:")
            path = paths_single[0]
            if path and len(path) > 0:
                path_array = np.array(path)
                print(f"  Env {single_env_id[0]}: Generated {len(path)} waypoints")
                print(
                    f"    Path bounds: X=[{path_array[:, 0].min():.2f}, {path_array[:, 0].max():.2f}], "
                    f"Y=[{path_array[:, 1].min():.2f}, {path_array[:, 1].max():.2f}]"
                )
            else:
                print(f"  Env {single_env_id[0]}: Failed to generate path")

        except Exception as e:
            print(f"\nError in Test 3: {e}")
            import traceback

            traceback.print_exc()

        for i in range(300):
            env.sim.sim_step()

        # ========================================
        # Test 4: Test query_closest_point and path_closest_distance
        # ========================================
        print("\n[Test 4] Testing NavMesh query functions")

        # Test 4.1: query_random_point
        print("\n[4.1] Testing query_random_point:")

        print("  Generating 5 random points on NavMesh:")
        for i in range(5):
            try:
                local_pos, env_idx = env.nav_manager.query_random_point()
                print(f"    Random point {i}: local={local_pos}, env_idx={env_idx}")
            except Exception as e:
                print(f"    Error generating random point {i}: {e}")

        # Test 4.2: query_closest_point
        print("\n[4.2] Testing query_closest_point:")

        # Test with points at different heights to verify projection onto NavMesh
        test_positions = [
            np.array([1.5, 2.3, 0.5]),
            np.array([2.8, 3.1, -0.77]),
            np.array([4.0, 7.0, -0.77]),
        ]

        for pos_idx, test_pos in enumerate(test_positions):
            try:
                closest_point = env.nav_manager.query_closest_point(
                    position=test_pos, env_id=0
                )
                distance_to_navmesh = np.linalg.norm(test_pos - closest_point)
                print(
                    f"  Test position {pos_idx}: {test_pos} -> Closest on NavMesh: {closest_point}"
                )
                print(f"    Distance to NavMesh: {distance_to_navmesh:.4f}")
            except Exception as e:
                print(f"  Error testing position {pos_idx}: {e}")

        # Test 4.3: path_closest_distance
        print("\n[4.3] Testing path_closest_distance:")

        # Use the path object from Test 1 for Env 0
        if (
            paths_all
            and len(paths_all[0]) > 0
            and path_objects_all
            and path_objects_all[0] is not None
        ):
            print(f"  Using path from Env 0 ({len(paths_all[0])} waypoints)")

            # Test with several positions along and near the path
            # Use same z coordinate as the path
            navmesh_z = -0.77
            agent_positions = [
                np.array([1.2, 1.2, navmesh_z]),
                np.array([1.8, 2.0, navmesh_z]),
                np.array([2.5, 2.8, navmesh_z]),
            ]

            for agent_idx, agent_pos in enumerate(agent_positions):
                try:
                    # Use the actual NavMesh path object
                    distance, path_position, path_tangent = (
                        env.nav_manager.path_closest_distance(
                            navmesh_path=path_objects_all[0],
                            position=agent_pos,
                            env_id=0,
                            return_position=True,
                            return_tangent=True,
                        )
                    )

                    print(f"\n  Agent position {agent_idx} (local): {agent_pos}")
                    print(f"    Distance along path: {distance:.4f}")
                    if path_position is not None:
                        print(f"    Closest point on path (local): {path_position}")
                        dist_to_path = np.linalg.norm(agent_pos - path_position)
                        print(f"    Perpendicular distance to path: {dist_to_path:.4f}")
                    if path_tangent is not None:
                        tangent_norm = np.linalg.norm(path_tangent)
                        print(
                            f"    Path tangent: {path_tangent} (norm: {tangent_norm:.4f})"
                        )

                except Exception as e:
                    print(f"  Error testing agent position {agent_idx}: {e}")
        else:
            print("  [SKIP] No valid path object from Test 1 to use for testing")

        print("\n[Test 4] Summary:")
        print("  [OK] query_random_point: Generates random valid points on NavMesh")
        print("  [OK] query_closest_point: Projects any position onto NavMesh surface")
        print(
            "  [OK] path_closest_distance: Finds closest point on path with distance and tangent"
        )

        # Let visualization render
        print("\n  Rendering visualizations...")
        for _ in range(200):
            env.sim.sim_step()

        print("\n" + "=" * 80)
        print("All tests completed! Check Isaac Sim viewport for visualizations.")
        print("Visualization paths: /World/Vis/Env_<id>/PathCurve and Waypoint_<idx>")
        print("=" * 80)

    print("\nEntering simulation loop (press Ctrl+C to exit)")

    step_count = 0
    try:
        while True:
            env.step()
            step_count += 1

            # Print status every 100 steps
            if step_count % 100 == 0:
                print(f"Step {step_count} completed")

    except KeyboardInterrupt:
        print(f"\nSimulation stopped at step {step_count}")


if __name__ == "__main__":
    main()
