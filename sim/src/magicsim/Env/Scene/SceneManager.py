from typing import List, Sequence, Dict, Any, Optional
import random
import torch
from omegaconf import DictConfig
from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf

# MagicSim Environment and Utilities
from magicsim.Env.Scene.Object.Light import Light
from magicsim.Env.Scene.Object.Ground import Ground
from magicsim.Env.Scene.Object.Room import Room
from magicsim.Env.Scene.Object.Fire import Fire
from magicsim.Env.Scene.Object.Flow import Flow
from magicsim.Env.Scene.Object.Rigid import RigidObject
from magicsim.Env.Scene.Object.Geometry import GeometryObject
from magicsim.Env.Scene.Object.Deformable import DeformableObject
from magicsim.Env.Scene.Object.Garment import GarmentObject
from magicsim.Env.Scene.Object.Background import Background
from magicsim.Env.Scene.Object.Fluid import FluidObject
from magicsim.Env.Scene.Object.Articulation import ArticulationObject
from magicsim.Env.Scene.Object.Rope import RopeObject
from magicsim.Env.Scene.Object.FEMCloth import FEMClothObject
from magicsim.Env.Scene.Object.Inflatable import InflatableObject
from magicsim.Env.Scene.Object.Sand import SandObject
from magicsim.Env.Environment.Isaac.IsaacRLEnv import IsaacRLEnv
from magicsim.Env.Utils.path import (
    deep_resolve_paths,
    _resolve_asset_paths,
    _select_assets,
)

# Import LayoutManager
from magicsim.Env.Layout.LayoutManager import LayoutManager
from magicsim.Env.Terrain.TerrainManager import TerrainManager

# Import all primitive shape classes
from magicsim.Env.Scene.Object.Primitives import PRIMITIVE_MAP
from magicsim.Env.Environment.Utils.Basic import seed_everywhere


class SceneManager:
    def __init__(
        self,
        num_envs: int,
        config: DictConfig,
        device: torch.device,
        env_spacing: float,
        use_fabric: bool = False,
        seeds_per_env: Sequence[int] | None = None,
        nav_enable: bool = False,
    ):
        """Initialize the SceneManager for managing multiple parallel environments and their objects.

        Args:
            num_envs: Number of parallel environments to manage
            config: Configuration dictionary containing environment and object parameters
            device: PyTorch device (CPU/GPU) for computations
            layout_manager: LayoutManager instance for position management
            use_fabric: Whether to use fabric physics engine
        """
        self.config = config
        self.nav_enable = nav_enable
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.env_spacing = env_spacing
        self.use_fabric = use_fabric
        self.stage = get_current_stage()
        self.sim: Optional[IsaacRLEnv] = None
        if seeds_per_env is not None:
            self.update_env_seeds(seeds_per_env)
        else:
            self._seeds_per_env = None

        # Initialize LayoutManager reference
        self.layout_manager = LayoutManager(self.num_envs, self.config, self.device)

        # Initialize TerrainManager
        self.terrain_manager = TerrainManager(
            self.num_envs,
            self.env_spacing,
            self.config.get("terrain", None),
            self.device,
        )

        # Environment roots and origins
        self.env_roots = [f"/World/envs/env_{env_id}" for env_id in range(num_envs)]

        # Object storage structure: [env_id][cat_name] = [objects]
        self._rigid_objects: List[Dict[str, List[RigidObject]]] = [
            {} for _ in range(num_envs)
        ]
        self._articulation_objects: List[Dict[str, List[ArticulationObject]]] = [
            {} for _ in range(num_envs)
        ]
        self._deformable_objects: List[Dict[str, List[DeformableObject]]] = [
            {} for _ in range(num_envs)
        ]
        self._garment_objects: List[Dict[str, List[GarmentObject]]] = [
            {} for _ in range(num_envs)
        ]
        self._geometry_objects: List[Dict[str, List[GeometryObject]]] = [
            {} for _ in range(num_envs)
        ]
        self._fluid_objects: List[Dict[str, List[FluidObject]]] = [
            {} for _ in range(num_envs)
        ]
        self._rope_objects: List[Dict[str, List[RopeObject]]] = [
            {} for _ in range(num_envs)
        ]
        self._inflatable_objects: List[Dict[str, List[InflatableObject]]] = [
            {} for _ in range(num_envs)
        ]
        self._fem_cloth_objects: List[Dict[str, List[FEMClothObject]]] = [
            {} for _ in range(num_envs)
        ]
        self._sand_objects: List[Dict[str, List[SandObject]]] = [
            {} for _ in range(num_envs)
        ]

        # Rooms and lights remain environment-specific
        self._rooms: List[Optional[Room]] = [None] * num_envs
        self._grounds: List[List[Ground]] = [[] for _ in range(num_envs)]
        self._lights: List[List[Light]] = [[] for _ in range(num_envs)]
        self._fires: List[List[Fire]] = [[] for _ in range(num_envs)]
        self._flows: List[List[Flow]] = [[] for _ in range(num_envs)]

        # Category counters managed by environment and category
        self._category_counters: Dict[int, Dict[str, int]] = {
            env_id: {} for env_id in range(num_envs)
        }
        # Background (shared across all environments)
        self._background: Optional[Background] = None
        self.background_prim_path = "/World/background"
        self.obj_init_buffer = []

    def update_env_seeds(self, seeds: Sequence[int] | None):
        """Update per-environment seed list."""
        if seeds is None:
            self._seeds_per_env = None
            return
        seed_list = [int(s) for s in seeds]
        if len(seed_list) != self.num_envs:
            raise ValueError(
                f"seed list length {len(seed_list)} does not match num_envs {self.num_envs}."
            )
        self._seeds_per_env = seed_list

    def initialize(self, sim: IsaacRLEnv):
        """Initialize the scene manager.
        This function will be called before simulation context creation.
        Put components that cannot be dynamically imported here.
        For CUDA devices, initialize rigid and articulation objects here.
        """
        self.sim = sim
        if self._seeds_per_env:
            seed_everywhere(self._seeds_per_env[0])
        # initialize terrain manager
        if not self.nav_enable:
            self.terrain_manager.initialize(sim)
        self._setup_background()

        if self.device.type == "cuda":
            for env_id in range(self.num_envs):
                self._set_env_seed(env_id)
                # Create dynamic and articulation objects for CUDA
                self._create_objects_by_type_filter(
                    env_id,
                    lambda obj_type: obj_type in ["dynamic", "articulation", "rope"],
                )
        if self.device.type == "cpu":
            for env_id in range(self.num_envs):
                self._set_env_seed(env_id)
                self._create_objects_by_type_filter(
                    env_id,
                    lambda obj_type: obj_type in ["articulation"],
                )

    def post_init(self):
        """Post-initialization logic.
        Initialize objects created in the initialize() method here.
        For CUDA devices, initialize rigid and articulation objects here.
        """
        for env_id in range(self.num_envs):
            # Initialize rigid and articulation objects
            for obj_dict in [
                self._rigid_objects[env_id],
                self._articulation_objects[env_id],
                self._rope_objects[env_id],
            ]:
                for cat_name, obj_list in obj_dict.items():
                    for obj in obj_list:
                        obj.initialize()

    def _set_env_seed(self, env_id: int):
        if self._seeds_per_env is None:
            return
        if env_id >= len(self._seeds_per_env):
            raise IndexError(
                f"Requested env_id {env_id} exceeds configured seeds (len={len(self._seeds_per_env)})."
            )
        seed_everywhere(self._seeds_per_env[env_id])

    def reset(self, soft: bool = False):
        """Reset all environments (batch processing).

        Args:
            soft: If True, perform soft reset; If False, perform hard reset
        """
        self.post_init()
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids, soft=soft)

    def reset_idx(self, env_ids: Sequence[int], soft: bool = True):
        """Reset specified environments (batch processing).

        Args:
            env_ids: Sequence of environment IDs to reset
            soft: If True, perform soft reset; If False, perform hard reset
        """
        if self._background:
            self._background.reset()

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.int32)

        for env_id in env_ids:
            env_id = int(env_id)
            self._set_env_seed(env_id)
            if soft:
                self._soft_reset(env_id)
            else:
                self._hard_reset(env_id)

        if not soft:
            self.sim.sim_step()
            for obj in self.obj_init_buffer:
                obj.initialize()
            self.obj_init_buffer.clear()

    def _soft_reset(self, env_id: int):
        """Perform a soft reset on the specified environment.
        Resets object states without recreating them.

        Args:
            env_id: ID of the environment to reset
        """
        # Reset all object types in this environment
        for obj_dict in [
            self._rigid_objects[env_id],
            self._articulation_objects[env_id],
            self._deformable_objects[env_id],
            self._garment_objects[env_id],
            self._geometry_objects[env_id],
            self._fluid_objects[env_id],
            self._rope_objects[env_id],
            self._inflatable_objects[env_id],
            self._fem_cloth_objects[env_id],
            self._sand_objects[env_id],
        ]:
            for cat_name, obj_list in obj_dict.items():
                for obj in obj_list:
                    obj.reset()

        # Reset room
        if self._rooms[env_id] is not None:
            self._rooms[env_id].reset()
        # Reset lights
        for light in self._lights[env_id]:
            light.reset()

        for fire in self._fires[env_id]:
            fire.reset()

        for flow in self._flows[env_id]:
            flow.reset()
        grounds = self._grounds[env_id]
        if isinstance(grounds, (list, tuple)):
            grounds_list = grounds
        else:
            grounds_list = [grounds]

        for ground in grounds_list:
            ground.reset()

    def _hard_reset(self, env_id: int, object_types: List[str] = None):
        """Perform a hard reset on the specified environment.
        Resets lights, room, and recreates objects.

        Args:
            env_id: ID of the environment to reset
        """
        self.layout_manager.hard_reset(env_id)
        self._reset_lights(env_id)
        self._reset_grounds(env_id)
        if not self.nav_enable:
            self._reset_room(env_id)
        self._reset_fires(env_id)
        self._reset_flows(env_id)
        # self._reset_room(env_id)

        if self.device.type == "cuda":
            # For CUDA: soft reset rigid/articulation, full reset for others
            self._soft_reset_objects_by_type(
                env_id, ["dynamic", "articulation", "rope"]
            )
            self._reset_objects(
                env_id, exclude_types=["dynamic", "articulation", "rope"]
            )
        else:
            self._soft_reset_objects_by_type(env_id, ["articulation"])
            self._reset_objects(env_id, exclude_types=["articulation"])

    def _create_objects_for_category(
        self,
        env_id: int,
        cat_name: str,
        cat_spec: Dict,
        exclude_types: Optional[List[str]] = None,
    ):
        """Create objects for a specific category in an environment.

        Args:
            env_id: Environment ID
            cat_name: Category name
            cat_spec: Category configuration
            exclude_types: Object types to exclude from creation
        """
        exclude_types = exclude_types or []

        deep_resolve_paths(cat_spec)
        asset_list = cat_spec.get("usd", [])
        num_per_env = cat_spec["num_per_env"]
        random_flag = cat_spec["random"]

        if not asset_list:
            selected_assets = [None] * num_per_env
        else:
            # Resolve and select assets
            all_asset_sources = _resolve_asset_paths(asset_list)
            if not all_asset_sources:
                print(
                    f"Warning: No valid assets found for category '{cat_name}' in env {env_id}. Skipping."
                )
                return
            selected_assets = _select_assets(
                all_asset_sources, num_per_env, random_flag, device=self.device
            )

        # Initialize category list for this environment
        self._initialize_category_list(env_id, cat_name)

        # Create object instances
        for i in range(num_per_env):
            asset_to_spawn = selected_assets[i]
            prim_path, inst_name = self._generate_object_paths(
                env_id, cat_name, num_per_env
            )
            inst_cfg = cat_spec.get(inst_name, {})
            obj_type = inst_cfg.get("physics", {}).get("type", "dynamic")

            # Skip excluded types
            if obj_type in exclude_types:
                continue

            # First register object to LayoutManager to get position info
            layout_info = self.layout_manager.register_object_and_get_layout(
                env_id, prim_path, cat_name, inst_cfg, cat_spec, asset_to_spawn
            )

            # Create object with layout info
            obj = self._create_single_object(
                env_id,
                asset_to_spawn,
                prim_path,
                inst_cfg,
                cat_spec,
                cat_name,
                layout_info,
            )
            if obj is not None:
                self.layout_manager._assign_object_to_category(cat_name, env_id, obj)
                self._assign_object_to_category(cat_name, env_id, obj)
                self.obj_init_buffer.append(obj)

    def _create_single_object(
        self,
        env_id: int,
        asset_to_spawn: str,
        prim_path: str,
        inst_cfg: Dict,
        cat_spec: Dict,
        cat_name: str,
        layout_info: Dict,
    ) -> Any:
        """Create a single object instance.

        Args:
            env_id: Environment ID
            asset_to_spawn: Asset to spawn (USD path or primitive name)
            prim_path: Prim path for the object
            inst_cfg: Instance configuration
            cat_spec: Category specification
            cat_name: Category name
            layout_info: Layout information from LayoutManager

        Returns:
            Created object instance or None
        """
        # Get object type first to check if it should be disabled
        obj_type = inst_cfg.get("physics", {}).get("type", "dynamic")

        # Check if garment or inflatable is disabled due to cpu+use_fabric combination
        # Do this BEFORE creating any primitive shapes to avoid creating orphaned primitives
        if self.device.type == "cpu" and self.use_fabric:
            if obj_type in ["garment", "inflatable", "rope", "articulation"]:
                print(
                    f"Warning: {obj_type.capitalize()} objects are disabled when device='cpu' and use_fabric=True. "
                    f"Skipping object creation at '{prim_path}'."
                )
                return None
        if self.device.type.startswith("cuda"):
            if obj_type in ["fluid", "rope", "sand"]:
                print(
                    f"Warning: {obj_type.capitalize()} objects are disabled when using CUDA device. "
                    f"Skipping object creation at '{prim_path}'."
                )
                return None

        usd_path = None

        # Handle primitive shapes
        if asset_to_spawn in PRIMITIVE_MAP:
            self._create_primitive_shape(
                asset_to_spawn, prim_path, inst_cfg, cat_spec, layout_info
            )
        else:
            usd_path = asset_to_spawn

        # Create object wrapper based on physics type
        primitive_type = asset_to_spawn if asset_to_spawn in PRIMITIVE_MAP else None

        return self._create_object_by_type(
            obj_type, prim_path, usd_path, primitive_type, env_id, layout_info
        )

    def _assign_object_to_category(self, cat_name: str, env_id: int, obj: Any):
        """Assign object to appropriate category collection.

        Args:
            cat_name: Category name
            env_id: Environment ID
            obj: Object instance to assign
        """
        # Store in hierarchy: environment -> category -> object list
        if isinstance(obj, RigidObject):
            self._rigid_objects[env_id][cat_name].append(obj)
        elif isinstance(obj, ArticulationObject):
            self._articulation_objects[env_id][cat_name].append(obj)
        elif isinstance(obj, RopeObject):
            self._rope_objects[env_id][cat_name].append(obj)
        elif isinstance(obj, DeformableObject):
            self._deformable_objects[env_id][cat_name].append(obj)
        elif isinstance(obj, GarmentObject):
            self._garment_objects[env_id][cat_name].append(obj)
        elif isinstance(obj, GeometryObject):
            self._geometry_objects[env_id][cat_name].append(obj)
        elif isinstance(obj, FluidObject):
            self._fluid_objects[env_id][cat_name].append(obj)
        elif isinstance(obj, InflatableObject):
            self._inflatable_objects[env_id][cat_name].append(obj)
        elif isinstance(obj, FEMClothObject):
            self._fem_cloth_objects[env_id][cat_name].append(obj)
        elif isinstance(obj, SandObject):
            self._sand_objects[env_id][cat_name].append(obj)

    def _setup_background(self):
        """Set up background for all environments."""
        # Setup shared background
        if self.config.get("background", None) is not None:
            self._background = Background(
                self.background_prim_path, self.config.background
            )

    def _reset_lights(self, env_id: int):
        """Reset lights for the specified environment.

        Args:
            env_id: Environment ID
        """
        light_root = f"{self.env_roots[env_id]}/Light"

        # Delete existing light prims
        if is_prim_path_valid(light_root):
            delete_prim(light_root)

        # Clear light references and recreate
        self._lights[env_id].clear()
        for light_config_key in self.config.light.keys():
            light_cfg = self.config.light[light_config_key]
            light_type = random.choice(light_cfg.types)
            light_prim_path = f"{light_root}/light_{light_config_key}"
            self._lights[env_id].append(
                Light(
                    prim_path=light_prim_path, light_type=light_type, config=light_cfg
                )
            )

    def _reset_grounds(self, env_id: int):
        """Reset ground for the specified environment.

        Args:
            env_id: Environment ID
        """
        # Check if ground configuration exists
        try:
            ground_config_exists = (
                hasattr(self.config, "ground") and self.config.ground is not None
            )
        except (AttributeError, KeyError):
            ground_config_exists = "ground" in self.config

        if not ground_config_exists:
            return

        ground_prim_path = f"{self.env_roots[env_id]}/ground"

        # Delete existing ground if present
        if is_prim_path_valid(ground_prim_path):
            delete_prim(ground_prim_path)

        self._grounds[env_id].clear()
        ground_cfg = self.config.ground
        self._grounds[env_id].append(
            Ground(
                prim_path=ground_prim_path,
                config=ground_cfg,
                env_spacing=self.env_spacing,
            )
        )

    def _reset_fires(self, env_id: int):
        """Reset fires for the specified environment.

        Args:
            env_id: Environment ID
        """
        # Check if fire configuration exists
        try:
            fire_config_exists = (
                hasattr(self.config, "fire") and self.config.fire is not None
            )
        except (AttributeError, KeyError):
            fire_config_exists = "fire" in self.config

        if not fire_config_exists:
            return

        fire_root = f"{self.env_roots[env_id]}/Fire"

        # Delete existing fire prims
        if is_prim_path_valid(fire_root):
            delete_prim(fire_root)

        # Clear fire references and recreate
        self._fires[env_id].clear()

        # Create fires using positions from config
        for fire_config_key in self.config.fire.keys():
            fire_cfg = self.config.fire[fire_config_key]
            fire_prim_path = f"{fire_root}/fire_{fire_config_key}"

            # Get position from config (required field)
            if "pos" not in fire_cfg:
                print(
                    f"Warning: Fire config '{fire_config_key}' missing 'pos' field. Skipping."
                )
                continue

            fire_position = fire_cfg.pos
            print(f"Creating fire at {fire_prim_path} with position {fire_position}")

            self._fires[env_id].append(Fire(prim_path=fire_prim_path, config=fire_cfg))

    def _reset_flows(self, env_id: int):
        """Reset flows (smoke, steam, dust) for the specified environment.

        Args:
            env_id: Environment ID
        """
        # Check if flow configuration exists
        try:
            flow_config_exists = (
                hasattr(self.config, "flow") and self.config.flow is not None
            )
        except (AttributeError, KeyError):
            flow_config_exists = "flow" in self.config

        if not flow_config_exists:
            return

        flow_root = f"{self.env_roots[env_id]}/Flow"

        # Delete existing flow prims
        if is_prim_path_valid(flow_root):
            delete_prim(flow_root)

        # Clear flow references and recreate
        self._flows[env_id].clear()

        # Create flows using positions from config
        for flow_config_key in self.config.flow.keys():
            flow_cfg = self.config.flow[flow_config_key]
            flow_prim_path = f"{flow_root}/flow_{flow_config_key}"

            # Get position from config (required field)
            if "pos" not in flow_cfg:
                print(
                    f"Warning: Flow config '{flow_config_key}' missing 'pos' field. Skipping."
                )
                continue

            flow_position = flow_cfg.pos
            flow_type = flow_cfg.get("type", "smoke")
            print(
                f"Creating {flow_type} at {flow_prim_path} with position {flow_position}"
            )

            self._flows[env_id].append(Flow(prim_path=flow_prim_path, config=flow_cfg))

    def _reset_room(self, env_id: int):
        """Reset room for the specified environment.

        Args:
            env_id: Environment ID
        """
        room_top_cfg = self.config.get("room", None)
        if room_top_cfg is None:
            return
        ## TODO Adjust Room Structure
        env_root = self.env_roots[env_id]
        room_prim_path = f"{env_root}/dynamic/room"

        if is_prim_path_valid(room_prim_path):
            delete_prim(room_prim_path)

        if room_top_cfg is None:
            return
        deep_resolve_paths(room_top_cfg)
        room_categories = [
            k
            for k in room_top_cfg
            if k not in ["hide_top_walls", "collision", "nav_flag"]
        ]

        if not room_categories:
            return

        cat_name = random.choice(room_categories)
        cat_spec = room_top_cfg[cat_name]
        asset_list = cat_spec.get("usd", [])

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
        inst_prim_path = f"{env_root}/dynamic/room/{cat_name}_1"
        self._rooms[env_id] = Room(
            prim_path=inst_prim_path,
            usd_path=selected_usd_path,
            room_config=room_top_cfg,
        )

    def _create_primitive_shape(
        self,
        primitive_name: str,
        prim_path: str,
        inst_cfg: Dict,
        cat_spec: Dict,
        layout_info: Dict,
    ):
        """Create a primitive shape on the stage.

        Args:
            primitive_name: Name of the primitive shape
            prim_path: Prim path for the primitive
            inst_cfg: Instance configuration
            cat_spec: Category specification
            layout_info: Layout information from LayoutManager
        """
        primitive_class = PRIMITIVE_MAP[primitive_name]
        primitive_params_cfg = inst_cfg.get(
            primitive_name, cat_spec.get(primitive_name, {})
        )
        params = {k: v for k, v in primitive_params_cfg.items()}

        # Use position from LayoutManager
        translation = layout_info["pos"]
        params["position"] = Gf.Vec3f(
            float(translation[0]), float(translation[1]), float(translation[2])
        )
        primitive_class(prim_path=prim_path, **params)

    def _create_object_by_type(
        self,
        obj_type: str,
        prim_path: str,
        usd_path: Optional[str],
        primitive_type: Optional[str],
        env_id: int,
        layout_info: Dict,
    ) -> Any:
        """Create object based on its type.

        Args:
            obj_type: Object type (dynamic, geometry, deformable, etc.)
            prim_path: Prim path for the object
            usd_path: USD file path (if applicable)
            primitive_type: Primitive type name (if applicable)
            env_id: Environment ID
            layout_info: Layout information from LayoutManager

        Returns:
            Created object instance
        """
        base_args = {
            "prim_path": prim_path,
            "usd_path": usd_path,
            "config": self.config,
            "primitive_type": primitive_type,
            "layout_manager": self.layout_manager,
            "layout_info": layout_info,
        }

        if obj_type == "dynamic":
            return RigidObject(
                env_origin=self.sim.scene.env_origins[env_id], **base_args
            )
        elif obj_type == "geometry":
            return GeometryObject(
                env_origin=self.sim.scene.env_origins[env_id], **base_args
            )
        elif obj_type == "rope":
            return RopeObject(
                env_origin=self.sim.scene.env_origins[env_id], **base_args
            )
        elif obj_type == "deformable":
            return DeformableObject(
                env_origin=self.sim.scene.env_origins[env_id], **base_args
            )
        elif obj_type == "garment":
            return GarmentObject(
                env_origin=self.sim.scene.env_origins[env_id], **base_args
            )
        elif obj_type == "fluid":
            return FluidObject(
                env_origin=self.sim.scene.env_origins[env_id],
                **{k: v for k, v in base_args.items() if k != "primitive_type"},
            )
        elif obj_type == "articulation":
            return ArticulationObject(
                env_origin=self.sim.scene.env_origins[env_id],
                **{k: v for k, v in base_args.items() if k != "primitive_type"},
            )
        elif obj_type == "inflatable":
            return InflatableObject(
                env_origin=self.sim.scene.env_origins[env_id], **base_args
            )
        elif obj_type == "fem_cloth":
            return FEMClothObject(
                env_origin=self.sim.scene.env_origins[env_id], **base_args
            )
        elif obj_type == "sand":
            return SandObject(
                env_origin=self.sim.scene.env_origins[env_id], **base_args
            )
        else:
            raise ValueError(f"Invalid object type: {obj_type}")

    def _create_objects_by_type_filter(self, env_id: int, type_filter_func):
        """Create objects filtered by a type filter function.

        Args:
            env_id: Environment ID
            type_filter_func: Function to filter object types
        """
        objects_cfg = self.config.objects
        if objects_cfg is None:
            return

        for cat_name in objects_cfg:
            if self._is_global_config_key(cat_name):
                continue

            cat_spec = objects_cfg[cat_name]
            deep_resolve_paths(cat_spec)
            asset_list = cat_spec.get("usd", [])
            num_per_env = cat_spec["num_per_env"]
            random_flag = cat_spec["random"]

            if not asset_list:
                selected_assets = [None] * num_per_env
            else:
                # Resolve and select assets
                all_asset_sources = _resolve_asset_paths(asset_list)
                if not all_asset_sources:
                    print(
                        f"Warning: No valid assets found for category '{cat_name}' in env {env_id}. Skipping."
                    )
                    return
                selected_assets = _select_assets(
                    all_asset_sources, num_per_env, random_flag, device=self.device
                )

            # Initialize category list
            self._initialize_category_list(env_id, cat_name)

            # Create object instances
            for i in range(num_per_env):
                asset_to_spawn = selected_assets[i]
                prim_path, inst_name = self._generate_object_paths(
                    env_id, cat_name, num_per_env
                )
                inst_cfg = cat_spec.get(inst_name, {})
                obj_type = inst_cfg.get("physics", {}).get("type", "dynamic")

                # Apply type filter
                if type_filter_func(obj_type):
                    # First register object to LayoutManager to get position info
                    layout_info = self.layout_manager.register_object_and_get_layout(
                        env_id, prim_path, cat_name, inst_cfg, cat_spec, asset_to_spawn
                    )

                    # Create object with layout info
                    obj = self._create_single_object(
                        env_id,
                        asset_to_spawn,
                        prim_path,
                        inst_cfg,
                        cat_spec,
                        cat_name,
                        layout_info,
                    )
                    if obj is not None:
                        self.layout_manager._assign_object_to_category(
                            cat_name, env_id, obj
                        )
                        self._assign_object_to_category(cat_name, env_id, obj)

    def _reset_objects(self, env_id: int, exclude_types: Optional[List[str]] = None):
        """Reset objects for the specified environment.

        Args:
            env_id: Environment ID
            exclude_types: Object types to exclude from reset
        """
        objects_cfg = self.config.objects
        if objects_cfg is None:
            return
        exclude_types = exclude_types or []

        # Clear existing objects (based on exclude_types)
        self._clear_existing_objects(env_id, exclude_types)

        # Create new objects for this environment
        for cat_name in objects_cfg:
            if self._is_global_config_key(cat_name):
                continue

            self._create_objects_for_category(
                env_id, cat_name, objects_cfg[cat_name], exclude_types
            )

    def _clear_existing_objects(
        self, env_id: int, exclude_types: Optional[List[str]] = None
    ):
        """Clear all existing objects in the specified environment.

        Args:
            env_id: Environment ID
            exclude_types: Object types to exclude from clearing
        """
        exclude_types = exclude_types or []

        # Determine which object collections to clear based on exclude_types
        all_collections = [
            (self._rigid_objects[env_id], "dynamic"),
            (self._articulation_objects[env_id], "articulation"),
            (self._rope_objects[env_id], "rope"),
            (self._deformable_objects[env_id], "deformable"),
            (self._garment_objects[env_id], "garment"),
            (self._geometry_objects[env_id], "geometry"),
            (self._fluid_objects[env_id], "fluid"),
            (self._inflatable_objects[env_id], "inflatable"),
            (self._fem_cloth_objects[env_id], "fem_cloth"),
            (self._sand_objects[env_id], "sand"),
        ]

        collections_to_clear = [
            obj_dict
            for obj_dict, obj_type in all_collections
            if obj_type not in exclude_types
        ]

        for obj_dict in collections_to_clear:
            for cat_name in list(obj_dict.keys()):
                for obj in obj_dict[cat_name]:
                    self._delete_object_by_type(obj)
                obj_dict[cat_name].clear()
                del obj_dict[cat_name]

    def _soft_reset_objects_by_type(self, env_id: int, obj_types: List[str]):
        """Perform soft reset on objects of specified types.

        Args:
            env_id: Environment ID
            obj_types: List of object types to soft reset
        """
        type_mapping = {
            "dynamic": self._rigid_objects,
            "articulation": self._articulation_objects,
            "rope": self._rope_objects,
        }

        for obj_type, obj_collection in type_mapping.items():
            if obj_type in obj_types and env_id < len(obj_collection):
                for cat_name, obj_list in obj_collection[env_id].items():
                    for obj in obj_list:
                        obj.reset_hard()

    def _initialize_category_list(self, env_id: int, cat_name: str):
        """Initialize object list for a specific environment and category.

        Args:
            env_id: Environment ID
            cat_name: Category name
        """
        # Ensure each category in each environment has an empty list to store objects
        if cat_name not in self._rigid_objects[env_id]:
            self._rigid_objects[env_id][cat_name] = []
        if cat_name not in self._articulation_objects[env_id]:
            self._articulation_objects[env_id][cat_name] = []
        if cat_name not in self._rope_objects[env_id]:
            self._rope_objects[env_id][cat_name] = []
        if cat_name not in self._deformable_objects[env_id]:
            self._deformable_objects[env_id][cat_name] = []
        if cat_name not in self._garment_objects[env_id]:
            self._garment_objects[env_id][cat_name] = []
        if cat_name not in self._geometry_objects[env_id]:
            self._geometry_objects[env_id][cat_name] = []
        if cat_name not in self._fluid_objects[env_id]:
            self._fluid_objects[env_id][cat_name] = []
        if cat_name not in self._inflatable_objects[env_id]:
            self._inflatable_objects[env_id][cat_name] = []
        if cat_name not in self._fem_cloth_objects[env_id]:
            self._fem_cloth_objects[env_id][cat_name] = []
        if cat_name not in self._sand_objects[env_id]:
            self._sand_objects[env_id][cat_name] = []

    def _generate_object_paths(
        self, env_id: int, cat_name: str, num_per_env: int
    ) -> tuple[str, str]:
        """Generate sequential prim path and instance name for a category.

        Args:
            env_id: Environment ID
            cat_name: Category name
            num_per_env: Number of objects per environment

        Returns:
            Tuple of (prim_path, inst_name)
        """
        env_root = self.env_roots[env_id]
        obj_id = self._get_next_object_id(env_id, cat_name)
        inst_name = f"{cat_name}_{obj_id}"
        prim_path = f"{env_root}/dynamic/{cat_name}/{inst_name}"
        # Calculate original ID to map to config instance
        original_id = (obj_id - 1) % num_per_env + 1
        inst_name = f"{cat_name}_{original_id}"

        return prim_path, inst_name

    def _get_next_object_id(self, env_id: int, cat_name: str) -> int:
        """Get the next available ID for a category in the specified environment.

        Args:
            env_id: Environment ID
            cat_name: Category name

        Returns:
            Next available integer ID for this category
        """
        if cat_name not in self._category_counters[env_id]:
            self._category_counters[env_id][cat_name] = 1
        else:
            env_root = self.env_roots[env_id]
            counter = self._category_counters[env_id][cat_name]

            # Skip existing IDs to avoid prim path conflicts
            while is_prim_path_valid(
                f"{env_root}/dynamic/{cat_name}/{cat_name}_{counter}"
            ):
                counter += 1

            self._category_counters[env_id][cat_name] = counter

        current_id = self._category_counters[env_id][cat_name]
        self._category_counters[env_id][cat_name] += 1

        return current_id

    def _delete_object_by_type(self, obj: Any):
        """Delete object based on its type using appropriate method.

        Args:
            obj: Object instance to delete
        """
        if self.device.type == "cuda":
            if isinstance(obj, DeformableObject):
                obj.destroy()
            elif isinstance(
                obj, (GarmentObject, InflatableObject, FEMClothObject, SandObject)
            ):
                if hasattr(obj, "usd_prim_path") and is_prim_path_valid(
                    obj.usd_prim_path
                ):
                    delete_prim(obj.usd_prim_path)
            elif isinstance(obj, FluidObject):
                if hasattr(obj, "usd_prim_path") and is_prim_path_valid(
                    obj.usd_prim_path
                ):
                    delete_prim(obj.usd_prim_path)
                # 仅删除由 Fluid 内联创建的 container，不删除 config 里单独导入的 container
                if (
                    getattr(obj, "container_owned", False)
                    and hasattr(obj, "container_prim_path")
                    and obj.container_prim_path
                    and is_prim_path_valid(obj.container_prim_path)
                ):
                    delete_prim(obj.container_prim_path)
            else:
                if hasattr(obj, "prim_path") and is_prim_path_valid(obj.prim_path):
                    delete_prim(obj.prim_path)
        else:
            if isinstance(obj, RigidObject):
                obj.destroy()
            elif isinstance(
                obj,
                (
                    DeformableObject,
                    GarmentObject,
                    InflatableObject,
                    FEMClothObject,
                    SandObject,
                ),
            ):
                if hasattr(obj, "usd_prim_path") and is_prim_path_valid(
                    obj.usd_prim_path
                ):
                    delete_prim(obj.usd_prim_path)
            elif isinstance(obj, FluidObject):
                if hasattr(obj, "usd_prim_path") and is_prim_path_valid(
                    obj.usd_prim_path
                ):
                    delete_prim(obj.usd_prim_path)
                if (
                    getattr(obj, "container_owned", False)
                    and hasattr(obj, "container_prim_path")
                    and obj.container_prim_path
                    and is_prim_path_valid(obj.container_prim_path)
                ):
                    delete_prim(obj.container_prim_path)
            else:
                if hasattr(obj, "prim_path") and is_prim_path_valid(obj.prim_path):
                    delete_prim(obj.prim_path)

    def _is_global_config_key(self, key: str) -> bool:
        """Check if the key is a global configuration key.

        Args:
            key: Configuration key to check

        Returns:
            True if it's a global config key, False otherwise
        """
        global_keys = [
            "common",
            "deformable_material",
            "visual_material",
            "deformable_config",
            "particle_system",
            "particle_material",
            "garment_config",
            "fem_config",
        ]
        return key in global_keys

    def get_objects(
        self,
        env_ids: Optional[List[int]] = None,
        object_name: Optional[str] = None,
        object_type: Optional[str] = None,
    ) -> Dict[str, List[Any]]:
        """Get object instances by environment IDs, object name, and object type.

        Args:
            env_ids: List of environment IDs to query. If None, query all environments
            object_name: Name of the object category to retrieve. If None, return all objects
            object_type: Type of objects to retrieve (rigid, articulation, etc.). If None, return all types

        Returns:
            Dictionary with object collections organized by composite keys (env_{id}_{type}_{category})
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        result = {}
        obj_collections = self._get_object_collections_by_type(object_type)

        for collection_name, env_dict_list in obj_collections.items():
            for env_id in env_ids:
                if env_id >= len(env_dict_list):
                    continue

                env_dict = env_dict_list[env_id]
                for cat_name, obj_list in env_dict.items():
                    if object_name is None or cat_name == object_name:
                        key = f"env_{env_id}_{collection_name}_{cat_name}"
                        result[key] = obj_list

        return result

    def _get_object_collections_by_type(
        self, object_type: Optional[str]
    ) -> Dict[str, List[Dict[str, List[Any]]]]:
        """Get object collections filtered by type.

        Args:
            object_type: Type filter for objects

        Returns:
            Dictionary of object collections
        """
        all_collections = {
            "rigid": self._rigid_objects,
            "articulation": self._articulation_objects,
            "rope": self._rope_objects,
            "deformable": self._deformable_objects,
            "garment": self._garment_objects,
            "geometry": self._geometry_objects,
            "fluid": self._fluid_objects,
            "inflatable": self._inflatable_objects,
            "fem_cloth": self._fem_cloth_objects,
            "sand": self._sand_objects,
        }

        if object_type is None:
            return all_collections
        elif object_type in all_collections:
            return {object_type: all_collections[object_type]}
        else:
            return {}

    def keys(self) -> List[str]:
        """Returns the keys of all scene entities.

        Returns:
            List of all entity keys in the scene
        """
        keys = ["rooms", "lights", "background"]

        # Generate keys for each category in each environment
        for env_id in range(self.num_envs):
            for obj_dict in [
                self._rigid_objects[env_id],
                self._articulation_objects[env_id],
                self._deformable_objects[env_id],
                self._garment_objects[env_id],
                self._geometry_objects[env_id],
                self._fluid_objects[env_id],
                self._inflatable_objects[env_id],
                self._fem_cloth_objects[env_id],
                self._sand_objects[env_id],
            ]:
                for cat_name in obj_dict.keys():
                    keys.append(f"env_{env_id}_{cat_name}")

        return keys

    # -------------------------------------------------------------------------
    # Object Access Properties (Helper Methods)
    # -------------------------------------------------------------------------
    @property
    def rigid_objects(self) -> List[Dict[str, List[RigidObject]]]:
        """List of dictionaries for rigid objects, organized as [env_id][cat_name] = [objects]"""
        return self._rigid_objects

    @property
    def articulation_objects(self) -> List[Dict[str, List[ArticulationObject]]]:
        """List of dictionaries for articulation objects, organized as [env_id][cat_name] = [objects]"""
        return self._articulation_objects

    @property
    def rope_objects(self) -> List[Dict[str, List[RopeObject]]]:
        """List of dictionaries for rope objects, organized as [env_id][cat_name] = [objects]"""
        return self._rope_objects

    @property
    def deformable_objects(self) -> List[Dict[str, List[DeformableObject]]]:
        """List of dictionaries for deformable objects, organized as [env_id][cat_name] = [objects]"""
        return self._deformable_objects

    @property
    def garment_objects(self) -> List[Dict[str, List[GarmentObject]]]:
        """List of dictionaries for garment objects, organized as [env_id][cat_name] = [objects]"""
        return self._garment_objects

    @property
    def geometry_objects(self) -> List[Dict[str, List[GeometryObject]]]:
        """List of dictionaries for geometry objects, organized as [env_id][cat_name] = [objects]"""
        return self._geometry_objects

    @property
    def fluid_objects(self) -> List[Dict[str, List[FluidObject]]]:
        """List of dictionaries for fluid objects, organized as [env_id][cat_name] = [objects]"""
        return self._fluid_objects

    @property
    def inflatable_objects(self) -> List[Dict[str, List[InflatableObject]]]:
        """List of dictionaries for inflatable objects, organized as [env_id][cat_name] = [objects]"""
        return self._inflatable_objects

    @property
    def fem_cloth_objects(self) -> List[Dict[str, List[FEMClothObject]]]:
        """List of dictionaries for fem_cloth objects, organized as [env_id][cat_name] = [objects]"""
        return self._fem_cloth_objects

    @property
    def sand_objects(self) -> List[Dict[str, List[SandObject]]]:
        """List of dictionaries for sand objects, organized as [env_id][cat_name] = [objects]"""
        return self._sand_objects

    @property
    def rooms(self) -> List[Optional[Room]]:
        """List of room objects for each environment."""
        return self._rooms

    @property
    def lights(self) -> List[List[Light]]:
        """List of light collections for each environment."""
        return self._lights

    def get_category(self, str_type: str) -> str:
        if str_type == "rigid":
            return self._rigid_objects
        elif str_type == "articulation":
            return self._articulation_objects
        elif str_type == "rope":
            return self._rope_objects
        elif str_type == "deformable":
            return self._deformable_objects
        elif str_type == "garment":
            return self._garment_objects
        elif str_type == "fluid":
            return self._fluid_objects
        elif str_type == "inflatable":
            return self._inflatable_objects
        elif str_type == "fem_cloth":
            return self._fem_cloth_objects
        elif str_type == "sand":
            return self._sand_objects
        elif str_type == "geometry":
            return self._geometry_objects
        elif str_type == "room":
            return self._rooms
        elif str_type == "light":
            return self._lights
        elif str_type == "fire":
            return self._fires
        elif str_type == "flow":
            return self._flows
        elif str_type == "background":
            return self._background
        else:
            raise ValueError(f"Invalid object type: {str_type}")

    def get_state(
        self, is_relative: bool = False, env_ids: Optional[List[int]] = None
    ) -> dict[
        str, dict[str, dict[str, list[torch.Tensor] | list[dict[str, str | None]]]]
    ]:
        """Get the state of all objects in the scene.

        Reference implementation from interactive_scene.py get_state method.

        Args:
            is_relative: If True, positions are relative to environment origin. Defaults to False.
            env_ids: List of environment IDs to get state for. If None, get state for all environments.

        Returns:
            Dictionary containing all object states with the following format:
            {
                "articulation": {
                    "category_name": {
                        "root_pose": [torch.Tensor, ...],
                        "root_velocity": [torch.Tensor, ...],
                        "joint_position": [torch.Tensor, ...],
                        "joint_velocity": [torch.Tensor, ...],
                        "asset_info": [{"usd_path": str, "primitive_type": str}, ...],
                    },
                },
                "rigid": {
                    "category_name": {
                        "root_pose": [torch.Tensor, ...],
                        "root_velocity": [torch.Tensor, ...],
                        "asset_info": [{"usd_path": str, "primitive_type": str}, ...],
                    },
                },
                "deformable": {
                    "category_name": {
                        "root_pose": [torch.Tensor, ...],
                        "root_velocity": [torch.Tensor, ...],
                        "asset_info": [{"usd_path": str, "primitive_type": str}, ...],
                    },
                },
                ...
            }

            Note:
            - Each state key corresponds to a list, where each element is the state of one object
            - asset_info contains asset information for each object:
              - usd_path: USD file path if object is from USD file, otherwise None
              - primitive_type: Primitive type name if object is a primitive shape, otherwise None
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        state = dict()

        env_origins = None
        if is_relative and self.sim is not None:
            try:
                env_origins = self.sim.scene.env_origins
            except (AttributeError, RuntimeError):
                pass
        state["articulation"] = dict()
        for env_id in env_ids:
            if env_id < len(self._articulation_objects):
                for cat_name, obj_list in self._articulation_objects[env_id].items():
                    if cat_name not in state["articulation"]:
                        state["articulation"][cat_name] = {
                            "root_pose": [],
                            "root_velocity": [],
                            "joint_position": [],
                            "joint_velocity": [],
                            "asset_info": [],
                        }
                    for obj in obj_list:
                        try:
                            obj_state = obj.get_state(is_relative=is_relative)
                            state["articulation"][cat_name]["root_pose"].append(
                                obj_state["root_pose"]
                            )
                            state["articulation"][cat_name]["root_velocity"].append(
                                obj_state["root_velocity"]
                            )
                            state["articulation"][cat_name]["joint_position"].append(
                                obj_state["joint_position"]
                            )
                            state["articulation"][cat_name]["joint_velocity"].append(
                                obj_state["joint_velocity"]
                            )
                            if "asset_info" in obj_state:
                                state["articulation"][cat_name]["asset_info"].append(
                                    obj_state["asset_info"]
                                )
                        except (AttributeError, RuntimeError) as e:
                            print(
                                f"Warning: Failed to get state for articulation {cat_name} in env {env_id}: {e}"
                            )

        state["rigid"] = dict()
        for env_id in env_ids:
            if env_id < len(self._rigid_objects):
                for cat_name, obj_list in self._rigid_objects[env_id].items():
                    if cat_name not in state["rigid"]:
                        state["rigid"][cat_name] = {
                            "root_pose": [],
                            "root_velocity": [],
                            "asset_info": [],
                        }
                    for obj in obj_list:
                        try:
                            obj_state = obj.get_state(is_relative=is_relative)
                            state["rigid"][cat_name]["root_pose"].append(
                                obj_state["root_pose"]
                            )
                            state["rigid"][cat_name]["root_velocity"].append(
                                obj_state["root_velocity"]
                            )
                            if "asset_info" in obj_state:
                                state["rigid"][cat_name]["asset_info"].append(
                                    obj_state["asset_info"]
                                )
                        except (AttributeError, RuntimeError) as e:
                            print(
                                f"Warning: Failed to get state for rigid object {cat_name} in env {env_id}: {e}"
                            )

        state["deformable"] = dict()
        for env_id in env_ids:
            if env_id < len(self._deformable_objects):
                for cat_name, obj_list in self._deformable_objects[env_id].items():
                    if cat_name not in state["deformable"]:
                        state["deformable"][cat_name] = {
                            "root_pose": [],
                            "root_velocity": [],
                            "asset_info": [],
                        }
                    for obj in obj_list:
                        try:
                            obj_state = obj.get_state(is_relative=is_relative)
                            state["deformable"][cat_name]["root_pose"].append(
                                obj_state["root_pose"]
                            )
                            state["deformable"][cat_name]["root_velocity"].append(
                                obj_state["root_velocity"]
                            )
                            if "asset_info" in obj_state:
                                state["deformable"][cat_name]["asset_info"].append(
                                    obj_state["asset_info"]
                                )
                        except (AttributeError, RuntimeError) as e:
                            print(
                                f"Warning: Failed to get state for deformable object {cat_name} in env {env_id}: {e}"
                            )

        state["garment"] = dict()
        for env_id in env_ids:
            if env_id < len(self._garment_objects):
                for cat_name, obj_list in self._garment_objects[env_id].items():
                    if cat_name not in state["garment"]:
                        state["garment"][cat_name] = {
                            "root_pose": [],
                            "root_velocity": [],
                            "asset_info": [],
                        }
                    for obj in obj_list:
                        try:
                            obj_state = obj.get_state(is_relative=is_relative)
                            state["garment"][cat_name]["root_pose"].append(
                                obj_state["root_pose"]
                            )
                            state["garment"][cat_name]["root_velocity"].append(
                                obj_state["root_velocity"]
                            )
                            if "asset_info" in obj_state:
                                state["garment"][cat_name]["asset_info"].append(
                                    obj_state["asset_info"]
                                )
                        except (AttributeError, RuntimeError) as e:
                            print(
                                f"Warning: Failed to get state for garment {cat_name} in env {env_id}: {e}"
                            )

        state["geometry"] = dict()
        for env_id in env_ids:
            if env_id < len(self._geometry_objects):
                for cat_name, obj_list in self._geometry_objects[env_id].items():
                    if cat_name not in state["geometry"]:
                        state["geometry"][cat_name] = {
                            "root_pose": [],
                            "asset_info": [],
                        }
                    for obj in obj_list:
                        try:
                            obj_state = obj.get_state(is_relative=is_relative)
                            state["geometry"][cat_name]["root_pose"].append(
                                obj_state["root_pose"]
                            )
                            if "asset_info" in obj_state:
                                state["geometry"][cat_name]["asset_info"].append(
                                    obj_state["asset_info"]
                                )
                        except (AttributeError, RuntimeError) as e:
                            print(
                                f"Warning: Failed to get state for geometry object {cat_name} in env {env_id}: {e}"
                            )

        state["rope"] = dict()
        for env_id in env_ids:
            if env_id < len(self._rope_objects):
                for cat_name, obj_list in self._rope_objects[env_id].items():
                    if cat_name not in state["rope"]:
                        state["rope"][cat_name] = {
                            "capsule_positions": [],
                            "asset_info": [],
                        }
                    for obj in obj_list:
                        try:
                            obj_state = obj.get_state(is_relative=is_relative)
                            state["rope"][cat_name]["capsule_positions"].append(
                                obj_state["capsule_positions"]
                            )
                            if "asset_info" in obj_state:
                                state["rope"][cat_name]["asset_info"].append(
                                    obj_state["asset_info"]
                                )
                        except (AttributeError, RuntimeError) as e:
                            print(
                                f"Warning: Failed to get state for rope {cat_name} in env {env_id}: {e}"
                            )

        state["fluid"] = dict()
        for env_id in env_ids:
            if env_id < len(self._fluid_objects):
                for cat_name, obj_list in self._fluid_objects[env_id].items():
                    if cat_name not in state["fluid"]:
                        state["fluid"][cat_name] = {
                            "particle_positions": [],
                            "particle_velocities": [],
                            "asset_info": [],
                        }
                    for obj in obj_list:
                        try:
                            obj_state = obj.get_state(is_relative=is_relative)
                            state["fluid"][cat_name]["particle_positions"].append(
                                obj_state["particle_positions"]
                            )
                            state["fluid"][cat_name]["particle_velocities"].append(
                                obj_state["particle_velocities"]
                            )
                            if "asset_info" in obj_state:
                                state["fluid"][cat_name]["asset_info"].append(
                                    obj_state["asset_info"]
                                )
                        except (AttributeError, RuntimeError) as e:
                            print(
                                f"Warning: Failed to get state for fluid {cat_name} in env {env_id}: {e}"
                            )

        state["sand"] = dict()
        for env_id in env_ids:
            if env_id < len(self._sand_objects):
                for cat_name, obj_list in self._sand_objects[env_id].items():
                    if cat_name not in state["sand"]:
                        state["sand"][cat_name] = {
                            "root_pose": [],
                            "root_velocity": [],
                            "particle_positions": [],
                            "asset_info": [],
                        }
                    for obj in obj_list:
                        try:
                            obj_state = obj.get_state(is_relative=is_relative)
                            state["sand"][cat_name]["root_pose"].append(
                                obj_state["root_pose"]
                            )
                            state["sand"][cat_name]["root_velocity"].append(
                                obj_state["root_velocity"]
                            )
                            if "particle_positions" in obj_state:
                                state["sand"][cat_name]["particle_positions"].append(
                                    obj_state["particle_positions"]
                                )
                            if "asset_info" in obj_state:
                                state["sand"][cat_name]["asset_info"].append(
                                    obj_state["asset_info"]
                                )
                        except (AttributeError, RuntimeError) as e:
                            print(
                                f"Warning: Failed to get state for sand {cat_name} in env {env_id}: {e}"
                            )

        state["inflatable"] = dict()
        for env_id in env_ids:
            if env_id < len(self._inflatable_objects):
                for cat_name, obj_list in self._inflatable_objects[env_id].items():
                    if cat_name not in state["inflatable"]:
                        state["inflatable"][cat_name] = {
                            "root_pose": [],
                            "asset_info": [],
                        }
                    for obj in obj_list:
                        try:
                            obj_state = obj.get_state(is_relative=is_relative)
                            state["inflatable"][cat_name]["root_pose"].append(
                                obj_state["root_pose"]
                            )
                            if "asset_info" in obj_state:
                                state["inflatable"][cat_name]["asset_info"].append(
                                    obj_state["asset_info"]
                                )
                        except (AttributeError, RuntimeError) as e:
                            print(
                                f"Warning: Failed to get state for inflatable {cat_name} in env {env_id}: {e}"
                            )

        state["fem_cloth"] = dict()
        for env_id in env_ids:
            if env_id < len(self._fem_cloth_objects):
                for cat_name, obj_list in self._fem_cloth_objects[env_id].items():
                    if cat_name not in state["fem_cloth"]:
                        state["fem_cloth"][cat_name] = {
                            "root_pose": [],
                            "root_velocity": [],
                            "asset_info": [],
                        }
                    for obj in obj_list:
                        try:
                            obj_state = obj.get_state(is_relative=is_relative)
                            state["fem_cloth"][cat_name]["root_pose"].append(
                                obj_state["root_pose"]
                            )
                            state["fem_cloth"][cat_name]["root_velocity"].append(
                                obj_state["root_velocity"]
                            )
                            if "asset_info" in obj_state:
                                state["fem_cloth"][cat_name]["asset_info"].append(
                                    obj_state["asset_info"]
                                )
                        except (AttributeError, RuntimeError) as e:
                            print(
                                f"Warning: Failed to get state for fem_cloth {cat_name} in env {env_id}: {e}"
                            )

        filtered_state = {}
        for obj_type, obj_dict in state.items():
            if obj_dict:
                filtered_categories = {}
                for cat_name, state_data in obj_dict.items():
                    has_data = any(
                        value_list and len(value_list) > 0
                        for value_list in state_data.values()
                    )
                    if has_data:
                        filtered_categories[cat_name] = state_data

                if filtered_categories:
                    filtered_state[obj_type] = filtered_categories

        return filtered_state
