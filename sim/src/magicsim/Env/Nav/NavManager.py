import os
import random
from typing import List, Tuple, Optional, Union
from magicsim.Env.Sensor.OccupancyManager import OccupancyManager
from omegaconf import DictConfig
import carb
from magicsim.Env.Utils.file import Logger
from magicsim.Env.Utils.path import deep_resolve_paths
from isaacsim.core.utils.stage import get_current_stage, add_reference_to_stage
from isaacsim.core.utils.prims import delete_prim
import numpy as np
from pxr import Usd, UsdGeom, Sdf, Gf
import omni
import omni.kit.commands
import omni.anim.navigation.core as nav
from magicsim.Env.Sensor.NavMeshManager import NavMeshManager
from magicsim.Env.Environment.Isaac.IsaacRLEnv import IsaacRLEnv
from magicsim.Env.Scene.Object.Room import Room
import omni.kit.app
from magicsim.Env.Utils.path import _resolve_asset_paths
from isaacsim.core.cloner import Cloner


class NavMeshConfig:
    """Dynamic configuration for NavMesh baking, loaded from yaml config"""

    def __init__(self, nav_config: DictConfig):
        """Initialize NavMeshConfig from yaml config"""

        # USD structure
        self.GEOMETRY_ROOT = nav_config.get("geometry_root", "/Root")
        self.VOLUME_PARENT = nav_config.get("volume_parent", "/Root")
        self.VOLUME_TYPE = nav_config.get("volume_type", 0)

        # Baking parameters
        self.BAKE_TIMEOUT_S = nav_config.get("bake_timeout_s", 600.0)

        # Agent parameters (in centimeters and degrees)
        self.AGENT_HEIGHT_CM = nav_config.get("agent_height_cm", 170.0)
        self.AGENT_MIN_RADIUS_CM = nav_config.get("min_agent_radius_cm", 35.0)
        self.AGENT_MAX_RADIUS_CM = nav_config.get("max_agent_radius_cm", 50.0)
        self.MAX_STEP_HEIGHT_CM = nav_config.get("max_step_height_cm", 25.0)
        self.MAX_FLOOR_SLOPE_DEG = nav_config.get("max_floor_slope_deg", 30.0)
        self.AGENT_ISLAND_RADIUS_CM = nav_config.get("agent_island_radius_cm", 5)

        # NavMesh behavior
        self.AUTO_EXCLUDE_RIGID_BODIES = nav_config.get(
            "auto_exclude_rigid_bodies", False
        )
        # NavMesh transform (in METERS)
        self.NAVMESH_OFFSET_X_M = nav_config.get("navmesh_offset_x_m", 0.0)
        self.NAVMESH_OFFSET_Y_M = nav_config.get("navmesh_offset_y_m", 0.0)
        self.NAVMESH_OFFSET_Z_M = nav_config.get("navmesh_offset_z_m", 0.0)

        # Logging
        self.VERBOSE = nav_config.get("verbose", True)


class NavManager:
    """
    Navigation Manager for handling NavMesh generation and room setup.

    Configuration:
    --------------
    Specify room directory in yaml, system will auto-discover USD and Annotation:

    The system automatically discovers:
    - USD file: Collected_SimpleRoom/*.usd
    - Annotation: Collected_SimpleRoom/Annotation/

    Usage:
    ------
    Access rooms directly via the `rooms` attribute:
    - nav_manager.rooms[env_id] -> Room object for specific environment
    - nav_manager.rooms -> List of all Room objects
    """

    def __init__(
        self, num_envs: int, config: DictConfig, device, logger: Logger = None
    ):
        self.num_envs = num_envs
        self.config = config
        self.device = device
        self.logger = logger
        self.nav_interface = nav.acquire_interface()
        self.stage = get_current_stage()
        self.navmesh = None
        self.volume_path = None

        self.nav_config = config.navmesh
        # Initialize NavMeshConfig from yaml
        self.navmeshconfig = NavMeshConfig(self.nav_config)

        self.room_config = config.room

        self._room_usd_path, self._annotation_dir = self._get_room_usd_path()
        self.rooms: List[Room] = []  # List to store Room objects

        self.occupancy_manager = OccupancyManager(
            self.num_envs,
            self.device,
            cell_size=0.05,
            values=(0.0, 1.0, 2.0),
        )

    def _get_room_usd_path(self) -> Tuple[str, str]:
        """Get room USD path, auto-discover USD and Annotation from room_dir

        Returns:
            Tuple of (Path to USD file, Path to Annotation directory)

        Raises:
            ValueError: If room_dir not configured or files not found
        """
        # Return cached path if already discovered
        ## TODO Adjust Room Structure
        deep_resolve_paths(self.room_config)
        room_categories = [
            k
            for k in self.room_config
            if k not in ["hide_top_walls", "collision", "nav_flag"]
        ]

        if not room_categories:
            return

        cat_name = random.choice(room_categories)
        cat_spec = self.room_config[cat_name]
        asset_list = cat_spec.get("usd", [])
        if asset_list is None:
            return None, None
        if not asset_list:
            raise FileNotFoundError(
                f"The 'usd' list is missing or empty for room category '{cat_name}' in your config."
            )

        # Use centralized asset resolution
        usd_paths = _resolve_asset_paths(asset_list)

        if not usd_paths:
            raise FileNotFoundError(
                f"Failed to find any valid USD files for room category '{cat_name}' from the provided 'usd' list."
            )

        selected_usd_path = random.choice(usd_paths)
        annotation_path = cat_spec.get("annotation", None)
        if annotation_path is None:
            annotation_path = "./Annotation"
        if os.path.isabs(annotation_path):
            annotation_dir = annotation_path
        else:
            annotation_dir = os.path.join(
                os.path.dirname(selected_usd_path), annotation_path
            )
        if not os.path.exists(annotation_dir):
            raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

        return selected_usd_path, annotation_dir

    def _post_setup_scene(self, sim: IsaacRLEnv):
        """Setup navigation rooms for each environment"""
        self.app = omni.kit.app.get_app()
        if self._room_usd_path is None:
            return
        self.nav_cloner = Cloner(stage=get_current_stage())
        self.nav_cloner.define_base_env(base_env_path="/World/NavRoom")
        nav_room_paths = self.nav_cloner.generate_paths(
            root_path="/World/NavRoom/Room", num_paths=self.num_envs
        )

        add_reference_to_stage(
            prim_path="/World/NavRoom/Room_0",
            usd_path=self._room_usd_path,
        )
        self.env_origins = sim.scene.env_origins

        self.nav_cloner.clone(
            source_prim_path=nav_room_paths[0],
            prim_paths=nav_room_paths,
            positions=self.env_origins,
            orientations=None,
            replicate_physics=False,
        )

        self.app.update()

        # Wrap each room with Room class using auto-discovered annotation_dir
        if self._annotation_dir:
            self.rooms = []
            failed_count = 0
            for i, room_path in enumerate(nav_room_paths):
                try:
                    room_obj = Room(
                        annotation_dir=self._annotation_dir,
                        prim_path=room_path,
                        usd_path=None,  # Don't reload USD, prim already exists
                        room_config=self.room_config,
                    )
                    self.rooms.append(room_obj)
                except Exception as e:
                    failed_count += 1
                    print(f"[WARNING] Failed to wrap room {i}: {e}")

            # Only print if there are issues
            if len(self.rooms) < self.num_envs:
                if len(self.rooms) > 0:
                    print(
                        f"[WARNING] Wrapped {len(self.rooms)}/{self.num_envs} rooms ({failed_count} failed)"
                    )
                else:
                    print("[ERROR] Failed to wrap all rooms")
        else:
            print("[WARNING] No annotation directory found, rooms not wrapped")

    def initialize(self, sim: IsaacRLEnv):
        # assert self.navmesh is not None, "NavMesh has not been generated"
        self.sim = sim
        # self.NavMeshManager = NavMeshManager(self.navmesh)
        self.env_origins = sim.scene.env_origins
        self.configure_nav_settings()
        self.volume_path = self.create_navmesh_volume()
        self.configure_volume_bounds(self.volume_path)

    def reset(self, soft: bool = True):
        self.occupancy_manager.initialize(self.sim)
        self.nav_mesh_gen()
        self.navmesh_manager = NavMeshManager(self.navmesh)

    def update_navmesh(self):
        self.nav_mesh_gen()
        self.navmesh_manager = NavMeshManager(self.navmesh)

    def configure_nav_settings(self):
        """Configure navigation agent parameters in Carb settings"""
        from omni.anim.navigation.core import NavMeshSettings

        settings = carb.settings.get_settings()
        settings.set(
            NavMeshSettings.AGENT_MIN_HEIGHT_SETTING_PATH,
            float(self.navmeshconfig.AGENT_HEIGHT_CM),
        )
        settings.set(
            NavMeshSettings.AGENT_MAX_RADIUS_SETTING_PATH,
            float(self.navmeshconfig.AGENT_MAX_RADIUS_CM),
        )
        settings.set(
            NavMeshSettings.AGENT_MIN_RADIUS_SETTING_PATH,
            float(self.navmeshconfig.AGENT_MIN_RADIUS_CM),
        )
        settings.set(
            NavMeshSettings.AGENT_MAX_STEP_HEIGHT_SETTING_PATH,
            float(self.navmeshconfig.MAX_STEP_HEIGHT_CM),
        )
        settings.set(
            NavMeshSettings.AGENT_MAX_FLOOR_SLOPE_SETTING_PATH,
            float(self.navmeshconfig.MAX_FLOOR_SLOPE_DEG),
        )
        settings.set(
            NavMeshSettings.AGENT_MIN_ISLAND_RADIUS_SETTING_PATH,
            float(self.navmeshconfig.AGENT_ISLAND_RADIUS_CM),
        )
        settings.set(NavMeshSettings.AUTO_REBAKE_SETTING_PATH, False)
        if hasattr(NavMeshSettings, "EXCLUDE_RIGID_BODIES_PATH"):
            settings.set(
                NavMeshSettings.EXCLUDE_RIGID_BODIES_PATH,
                self.navmeshconfig.AUTO_EXCLUDE_RIGID_BODIES,
            )

    def create_navmesh_volume(self) -> Sdf.Path:
        """
        Create a NavMeshVolume using the official CreateNavMeshVolumeCommand.

        Args:
            stage: The USD stage

        Returns:
            Path to the created NavMeshVolume prim

        Raises:
            RuntimeError: If the command fails
        """
        layer = self.stage.GetRootLayer()

        success, result = omni.kit.commands.execute(
            "CreateNavMeshVolumeCommand",
            parent_prim_path="/World",
            volume_type=int(self.navmeshconfig.VOLUME_TYPE),
            layer=layer,
        )

        if not success:
            raise RuntimeError("CreateNavMeshVolumeCommand failed")

        # Normalize result to Sdf.Path
        if isinstance(result, (str, Sdf.Path)):
            volume_path = Sdf.Path(str(result))
        else:
            # Fallback to common default name
            volume_path = Sdf.Path("/NavMeshVolume")

        return volume_path

    def compute_all_envs_bounds(self) -> Tuple[Gf.Vec3d, Gf.Vec3d]:
        """
        Compute the bounding box that encompasses all environment rooms.

        Returns:
            Tuple of (min_point, max_point) covering all environments with padding
        """
        if self._room_usd_path is None:
            env_spacing = self.sim.env_spacing
            global_min = self.sim.scene.env_origins.min(dim=0).values - env_spacing
            global_max = self.sim.scene.env_origins.max(dim=0).values + env_spacing
            # padding 5%
            padding = (global_max - global_min) * 0.05
            global_min -= padding
            global_max += padding
            return Gf.Vec3d(global_min[0].item(), global_min[1].item(), -1), Gf.Vec3d(
                global_max[0].item(), global_max[1].item(), 1
            )
        global_min = None
        global_max = None

        # Iterate through all environments
        for env_id in range(self.num_envs):
            room_path = f"/World/NavRoom/Room_{env_id}"

            # Check if the room prim exists
            prim = self.stage.GetPrimAtPath(room_path)
            if not prim or not prim.IsValid():
                continue

            try:
                # Compute bounds for this room
                cache = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(),
                    [UsdGeom.Tokens.render, UsdGeom.Tokens.default_],
                    useExtentsHint=True,
                )
                box = cache.ComputeWorldBound(prim).ComputeAlignedBox()

                if box.IsEmpty():
                    continue

                min_point = box.GetMin()
                max_point = box.GetMax()

                # Update global bounds
                if global_min is None:
                    global_min = min_point
                    global_max = max_point
                else:
                    global_min = Gf.Vec3d(
                        min(global_min[0], min_point[0]),
                        min(global_min[1], min_point[1]),
                        min(global_min[2], min_point[2]),
                    )
                    global_max = Gf.Vec3d(
                        max(global_max[0], max_point[0]),
                        max(global_max[1], max_point[1]),
                        max(global_max[2], max_point[2]),
                    )

            except Exception:
                continue

        # If no valid bounds were found, raise error
        if global_min is None or global_max is None:
            raise RuntimeError(
                "Failed to compute bounds for any environment rooms. "
                "Ensure rooms are properly loaded before computing NavMesh volume."
            )

        # Add 5% padding
        size = global_max - global_min
        padding = size * 0.05
        global_min -= padding
        global_max += padding

        return global_min, global_max

    def configure_volume_bounds(self, volume_path: Sdf.Path) -> None:
        """
        Set the size and position of the NavMeshVolume to match scene bounds.

        Args:
            volume_path: Path to the NavMeshVolume prim
            min_point: Minimum corner of the bounding box
            max_point: Maximum corner of the bounding box

        Raises:
            RuntimeError: If the NavMeshVolume prim is invalid
        """
        # Compute bounds of all environments
        min_point, max_point = self.compute_all_envs_bounds()

        # Calculate center position from bounds
        bounds_center = (min_point + max_point) * 0.5

        # Apply NavMesh offset to the center position (convert meters to stage units)

        center = Gf.Vec3d(
            bounds_center[0] + self.navmeshconfig.NAVMESH_OFFSET_X_M,
            bounds_center[1] + self.navmeshconfig.NAVMESH_OFFSET_Y_M,
            bounds_center[2] + self.navmeshconfig.NAVMESH_OFFSET_Z_M,
        )

        # Calculate scale from bounding box dimensions
        dimensions = max_point - min_point
        scale_values = (
            float(dimensions[0]),
            float(dimensions[1]),
            float(dimensions[2]),
        )

        # SetTranslate requires Vec3d, SetScale requires Vec3f
        scale_f = Gf.Vec3f(scale_values[0], scale_values[1], scale_values[2])

        prim = self.stage.GetPrimAtPath(volume_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"NavMeshVolume prim not found at {volume_path}")

        # Author attributes on root layer to ensure they take precedence
        root_layer = self.stage.GetRootLayer()
        with Usd.EditContext(self.stage, root_layer):
            # Set transform (position and scale)
            try:
                xform = UsdGeom.Xform(prim)
                xform_api = UsdGeom.XformCommonAPI(xform)

                # Set translate (position with offset applied) - requires Vec3d
                xform_api.SetTranslate(center)

                # Set scale to match bounding box dimensions - requires Vec3f
                xform_api.SetScale(scale_f)
            except Exception as e:
                print(f"  WARNING: Failed to set transform: {e}")

            # Set additional helpful attributes
            try:
                prim.CreateAttribute("enabled", Sdf.ValueTypeNames.Bool).Set(True)
                prim.CreateAttribute(
                    "omni:navmesh:volume:type", Sdf.ValueTypeNames.Token
                ).Set("include")
            except Exception as e:
                print(f"  WARNING: Failed to set optional attributes: {e}")

    def nav_mesh_gen(self):
        self.app.update()
        # Bake NavMesh
        bake_success = self.nav_interface.start_navmesh_baking_and_wait()
        if not bake_success:
            print("[ERROR] NavMesh baking failed or timed out")
            return False

        self.app.update()

        # Double-check navmesh exists before saving
        self.navmesh = self.nav_interface.get_navmesh()

        if self.navmesh is None:
            print("[ERROR] NavMesh verification failed")
            return False

    def check_existing_navmesh(self):
        """
        Check if a valid NavMesh already exists in memory.

        Returns:
            True if NavMesh exists and is valid, False otherwise
        """
        try:
            navmesh = self.nav_interface.get_navmesh()
            if navmesh is not None:
                return True
            return False
        except Exception:
            return False

    def find_existing_navmesh_volume(self):
        """
        Find existing NavMeshVolume in the stage.

        Returns:
            Path to NavMeshVolume if found, None otherwise
        """
        from pxr import Sdf

        # Check common locations
        common_paths = [
            f"{self.navmeshconfig.VOLUME_PARENT}/NavMeshVolume",
            "/Root/NavMeshVolume",
            "/NavMeshVolume",
        ]

        for path in common_paths:
            prim = self.stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                if prim.HasAttribute("omni:navmesh:volume:size"):
                    return Sdf.Path(path)

        # Search entire stage as fallback
        for prim in self.stage.Traverse():
            if prim.HasAttribute("omni:navmesh:volume:size"):
                return prim.GetPath()

        return None

    def query_random_point(self) -> Tuple[np.ndarray, int]:
        """Query a random valid point on NavMesh and return local coordinates with env_idx

        Returns:
            Tuple of (local_position, env_idx)
            - local_position: Random point in local coordinates relative to closest env_origin
            - env_idx: ID of the environment whose origin is closest to the random point
        """

        # Query random point from NavMesh
        result = self.navmesh.query_random_point()

        # API returns: (carb.Float3, polygon_id) or just carb.Float3
        if isinstance(result, tuple) and len(result) == 2:
            random_point_global, polygon_id = result
        else:
            random_point_global = result

        # Convert Float3 to numpy array
        if hasattr(random_point_global, "x"):
            random_point_array = np.array(
                [
                    random_point_global.x,
                    random_point_global.y,
                    random_point_global.z,
                ],
                dtype=float,
            )
        else:
            random_point_array = np.array(random_point_global, dtype=float)

        # Find the closest environment origin
        min_distance = float("inf")
        env_idx = 0

        for i in range(self.num_envs):
            env_origin = self.env_origins[i].cpu().numpy()
            env_origin_with_offset = env_origin

            # Calculate distance to this environment's origin
            distance = np.linalg.norm(random_point_array - env_origin_with_offset)

            if distance < min_distance:
                min_distance = distance
                env_idx = i

        # Convert global to local using the closest environment's origin
        env_origin = self.env_origins[env_idx].cpu().numpy()
        env_origin_with_offset = env_origin
        local_position = random_point_array - env_origin_with_offset

        return local_position, env_idx

    def query_closest_point(self, position: np.ndarray, env_id: int) -> np.ndarray:
        """Query closest point on NavMesh from a local position

        Args:
            position: Local position relative to env_origin [x, y, z]
            env_id: Environment ID

        Returns:
            Closest point on NavMesh in local coordinates
        """
        import carb

        # Convert local to global
        env_origin = self.env_origins[env_id].cpu().numpy()
        env_origin_with_offset = env_origin
        position_global = position + env_origin_with_offset

        # Query NavMesh with global coordinates
        pos_carb = carb.Float3(
            float(position_global[0]),
            float(position_global[1]),
            float(position_global[2]),
        )

        result = self.navmesh.query_closest_point(pos_carb)

        # API returns: (carb.Float3, polygon_id)
        if isinstance(result, tuple) and len(result) == 2:
            closest_point_global, polygon_id = result
        else:
            closest_point_global = result

        # Convert Float3 to numpy array
        if hasattr(closest_point_global, "x"):
            closest_point_array = np.array(
                [
                    closest_point_global.x,
                    closest_point_global.y,
                    closest_point_global.z,
                ],
                dtype=float,
            )
        else:
            closest_point_array = np.array(closest_point_global, dtype=float)

        # Convert global back to local
        closest_point_local = closest_point_array - env_origin_with_offset

        return closest_point_local

    def path_closest_distance(
        self,
        navmesh_path,
        position: np.ndarray,
        env_id: int,
        return_position: bool = True,
        return_tangent: bool = False,
    ) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute closest distance on path from a local position

        Args:
            navmesh_path: NavMesh path object returned from generate_path
            position: Local position relative to env_origin [x, y, z]
            env_id: Environment ID
            return_position: Whether to return the closest position on path
            return_tangent: Whether to return the tangent at closest point

        Returns:
            Tuple of (distance, path_position_local, path_tangent_local)
            - distance: Distance along the path to the closest point
            - path_position_local: Closest point on path in local coordinates (if return_position=True)
            - path_tangent_local: Tangent at closest point in local coordinates (if return_tangent=True)
        """
        import carb

        # Convert local to global
        env_origin = self.env_origins[env_id].cpu().numpy()
        env_origin_with_offset = env_origin
        position_global = position + env_origin_with_offset

        # Prepare carb.Float3 for query
        pos_carb = carb.Float3(
            float(position_global[0]),
            float(position_global[1]),
            float(position_global[2]),
        )

        # Prepare output parameters
        out_path_position = carb.Float3() if return_position else None
        out_path_tangent = carb.Float3() if return_tangent else None

        # Query distance with global coordinates
        distance = navmesh_path.closest_distance(
            position=pos_carb,
            out_path_position=out_path_position,
            out_path_tangent=out_path_tangent,
        )

        # Convert global results back to local
        path_position_local = None
        if return_position and out_path_position:
            # Handle both tuple and Float3
            if isinstance(out_path_position, tuple):
                path_position_array = np.array(out_path_position, dtype=float)
            else:
                path_position_array = np.array(
                    [out_path_position.x, out_path_position.y, out_path_position.z],
                    dtype=float,
                )
            path_position_local = path_position_array - env_origin_with_offset

        path_tangent_local = None
        if return_tangent and out_path_tangent:
            # Tangent is a direction vector, not affected by translation
            # Handle both tuple and Float3
            if isinstance(out_path_tangent, tuple):
                path_tangent_local = np.array(out_path_tangent, dtype=float)
            else:
                path_tangent_local = np.array(
                    [out_path_tangent.x, out_path_tangent.y, out_path_tangent.z],
                    dtype=float,
                )

        return distance, path_position_local, path_tangent_local

    def generate_path(
        self,
        start_point: List[np.ndarray],
        coords: List[List[np.ndarray]],
        env_ids: List[int],
        visualize: bool = False,
        return_path_objects: bool = False,
    ) -> Union[List[List[np.ndarray]], Tuple[List[List[np.ndarray]], List]]:
        """Generate navigation paths for multiple environments

        Args:
            start_point: List of start positions in LOCAL coordinates (relative to env_origin) for each env
            coords: List of waypoint lists in LOCAL coordinates (relative to env_origin) for each env
            env_ids: List of environment IDs
            visualize: Whether to visualize the paths
            return_path_objects: Whether to also return NavMesh path objects for advanced queries

        Returns:
            If return_path_objects=False:
                List of path point lists in LOCAL coordinates for each env
            If return_path_objects=True:
                Tuple of (path_points_list, path_objects_list) - path points in LOCAL coordinates
        """
        if not isinstance(env_ids, list):
            env_ids = env_ids.tolist() if hasattr(env_ids, "tolist") else [env_ids]

        paths = []
        path_objects = [] if return_path_objects else None

        for i, env_id in enumerate(env_ids):
            # Convert relative coordinates to world coordinates
            # Apply room position offset to match room's actual position
            env_origin = self.env_origins[env_id].cpu().numpy()
            env_origin_with_offset = env_origin

            start_relative = np.array(start_point[i], dtype=float)
            waypoints_relative = [np.array(wp, dtype=float) for wp in coords[i]]
            # print("start_relative: ", start_relative)
            # print("waypoints_relative: ", waypoints_relative)
            start_world = env_origin_with_offset + start_relative
            waypoints_world = [env_origin_with_offset + wp for wp in waypoints_relative]
            # print("start_world: ", start_world)
            # print("waypoints_world: ", waypoints_world)
            # Generate path using NavMeshManager
            if return_path_objects:
                result = self.navmesh_manager.generate_path(
                    start_point=start_world,
                    coords=waypoints_world,
                    return_path_object=True,
                )
                if result is None or result[0] is None or len(result[0]) == 0:
                    paths.append([])
                    path_objects.append(None)
                    continue
                path_world, path_obj = result
                path_objects.append(path_obj)
            else:
                path_world = self.navmesh_manager.generate_path(
                    start_point=start_world, coords=waypoints_world
                )
                if path_world is None or len(path_world) == 0:
                    paths.append([])
                    continue

            # Convert world coordinates back to local
            path_local = [p - env_origin_with_offset for p in path_world]

            paths.append(path_local)
            if visualize:
                self._visualize_path(path_world, env_id)

        if return_path_objects:
            return paths, path_objects
        return paths

    def clear_all_vis(self):
        """Clear all navigation visualization from the stage"""
        vis_root = "/World/Vis"
        vis_root_prim = self.stage.GetPrimAtPath(vis_root)
        if vis_root_prim.IsValid():
            delete_prim(vis_root)

    def _visualize_path(
        self,
        path: List[np.ndarray],
        env_id: int,
        curve_color: Tuple[float, float, float] = (1.0, 0.2, 0.2),
        waypoint_color: Tuple[float, float, float] = (0.2, 0.7, 1.0),
    ):
        """Visualize a navigation path in the stage

        Args:
            path: List of path points as numpy arrays [x, y, z]
            env_id: Environment ID for naming
            curve_color: RGB color for the path curve
            waypoint_color: RGB color for waypoint spheres
        """
        if not path:
            return

        vis_root = "/World/Vis"
        vis_root_prim = self.stage.GetPrimAtPath(vis_root)
        if not vis_root_prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_path=vis_root,
                prim_type="Xform",
                select_new_prim=False,
            )

        parent_path = f"{vis_root}/Env_{env_id}"
        parent_prim = self.stage.GetPrimAtPath(parent_path)
        if not parent_prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_path=parent_path,
                prim_type="Xform",
                select_new_prim=False,
            )

        # Draw path curve
        curve_path = f"{parent_path}/PathCurve"
        if self.stage.GetPrimAtPath(curve_path).IsValid():
            delete_prim(curve_path)

        curve = UsdGeom.BasisCurves.Define(self.stage, curve_path)
        verts = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in path]
        curve.CreatePointsAttr(verts)
        curve.CreateCurveVertexCountsAttr([len(verts)])
        curve.CreateTypeAttr("linear")
        curve.CreateWrapAttr("nonperiodic")
        curve.CreateWidthsAttr([0.03])
        UsdGeom.Gprim(curve.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*curve_color)])

        # Draw waypoint spheres
        sampled_points = path[:: max(1, len(path) // 10)]
        for i, point in enumerate(sampled_points):
            sphere_path = f"{parent_path}/Waypoint_{i}"
            if self.stage.GetPrimAtPath(sphere_path).IsValid():
                delete_prim(sphere_path)

            pos = Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
            sphere = UsdGeom.Sphere.Define(self.stage, sphere_path)
            sphere.CreateRadiusAttr(0.06)
            sphere_xform = UsdGeom.Xformable(sphere)
            sphere_xform.ClearXformOpOrder()
            translate_op = sphere_xform.AddTranslateOp()
            translate_op.Set(pos)
            sphere.CreateDisplayColorAttr([Gf.Vec3f(*waypoint_color)])
