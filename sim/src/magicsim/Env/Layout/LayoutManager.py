from typing import List, Dict, Any, Optional
import random
import torch
import numpy as np
from omegaconf import DictConfig
import json
import os
from datetime import datetime
from magicsim.Env.Environment.Isaac.IsaacRLEnv import IsaacRLEnv
from magicsim.Env.Scene.Object.Rigid import RigidObject
from magicsim.Env.Scene.Object.Geometry import GeometryObject
from magicsim.Env.Scene.Object.Deformable import DeformableObject
from magicsim.Env.Scene.Object.Garment import GarmentObject
from magicsim.Env.Scene.Object.Fluid import FluidObject
from magicsim.Env.Scene.Object.Articulation import ArticulationObject
from magicsim.Env.Scene.Object.Rope import RopeObject
from magicsim.Env.Scene.Object.FEMCloth import FEMClothObject
from magicsim.Env.Scene.Object.Inflatable import InflatableObject
from magicsim.Env.Scene.Object.Room import Room
from magicsim.Env.Scene.Object.Light import Light
from magicsim.Env.Scene.Object.Background import Background
from magicsim.Env.Scene.Object.Primitives import PRIMITIVE_MAP
from magicsim.Env.Animation.Avatar import Avatar
from magicsim.Env.Utils.rotations import euler_angles_to_quat


class LayoutManager:
    """LayoutManager manages position, rotation, and scale information for all objects.

    Responsibilities:
    - Generate and manage pos, scale, ori information
    - Maintain object tables consistent with SceneManager
    - Clear and update object tables during hard reset
    - Register objects before SceneManager creates them
    """

    def __init__(self, num_envs: int, config: DictConfig, device: torch.device):
        """Initialize LayoutManager

        Args:
            num_envs: Number of parallel environments
            config: Configuration dictionary
            device: PyTorch device
        """
        self.config = config
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.common_config = {
            "initial_pos_range": [-0.2, -0.2, 1.0, 0.2, 0.5, 1.2],
            "initial_ori_range": [-30, -30, -30, 30, 30, 30],
            "scale": [1.0, 1.0, 1.0],
        }

        self.env_roots = [f"/World/envs/env_{env_id}" for env_id in range(num_envs)]
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
        self._avatar_objects: List[Dict[str, List[Avatar]]] = [
            {} for _ in range(num_envs)
        ]

        self._rooms: List[Optional[Room]] = [None] * num_envs
        self._lights: List[List[Light]] = [[] for _ in range(num_envs)]
        self._background: Optional[Background] = None

        self._object_layouts: List[Dict[str, Dict[str, Any]]] = [
            {} for _ in range(num_envs)
        ]

        self._category_counters: Dict[int, Dict[str, int]] = {
            env_id: {} for env_id in range(num_envs)
        }

    def initialize(self, sim: IsaacRLEnv):
        """Initialize LayoutManager with simulation reference

        Args:
            sim: Simulation instance
        """
        self.sim = sim

    def _get_timestamped_output_dir(self, output_file: str) -> str:
        """Generate a timestamped output directory path.

        Args:
            output_file: Original output file path (e.g., "outputs/robot_objects_layouts.json")

        Returns:
            Output directory path with timestamp (e.g., "outputs/2025-01-26/14-30-45/")
        """
        now = datetime.now()
        timestamp_dir = now.strftime("%Y-%m-%d/%H-%M-%S")
        base_dir = os.path.dirname(output_file) or "outputs"
        timestamped_dir = os.path.join(base_dir, timestamp_dir)

        return timestamped_dir

    def _get_timestamped_output_path(self, base_name: str) -> str:
        """Generate a full output path with timestamp.

        Args:
            base_name: Base filename (e.g., "robot_objects_layouts.json")

        Returns:
            Full timestamped path (e.g., "outputs/2025-01-26/14-30-45/robot_objects_layouts.json")
        """
        timestamped_dir = self._get_timestamped_output_dir(base_name)
        return os.path.join(timestamped_dir, os.path.basename(base_name))

    def register_object_and_get_layout(
        self,
        env_id: int,
        prim_path: str,
        cat_name: str,
        inst_cfg: Dict,
        cat_spec: Dict,
        asset_to_spawn: str = None,
    ) -> Dict[str, Any]:
        """Register object to LayoutManager and get layout information

        Args:
            env_id: Environment ID
            prim_path: Object prim path
            cat_name: Object category name
            inst_cfg: Instance configuration
            cat_spec: Category configuration
            asset_to_spawn: Asset to spawn (USD path or primitive name)

        Returns:
            Layout information dictionary
        """
        layout_info = self._generate_object_layout_from_config(
            env_id, prim_path, cat_name, inst_cfg, cat_spec, asset_to_spawn
        )

        self._object_layouts[env_id][prim_path] = layout_info
        self._category_counters[env_id].setdefault(cat_name, 0)

        return layout_info

    def _generate_object_layout_from_config(
        self,
        env_id: int,
        prim_path: str,
        cat_name: str,
        inst_cfg: Dict,
        cat_spec: Dict,
        asset_to_spawn: str = None,
    ) -> Dict[str, Any]:
        """Generate position, rotation, and scale information from configuration
        Determines the valid ranges and generates the initial values.

        Args:
            env_id: Environment ID
            prim_path: Object prim path
            cat_name: Object category name
            inst_cfg: Instance configuration
            cat_spec: Category configuration
            asset_to_spawn: Asset to spawn (USD path or primitive name)

        Returns:
            Dictionary containing initial pos, ori, scale, and their ranges
        """
        category_common_config = cat_spec.get("common", {})
        # Normalize empty YAML mapping: `common:` with no children becomes None
        if not isinstance(category_common_config, (dict, DictConfig)):
            category_common_config = {}
        common_config = self.common_config
        initial_pos_range = self._get_pos_range_from_config(
            inst_cfg, category_common_config, common_config, "initial_pos_range"
        )
        initial_ori_range = self._get_ori_range_from_config(
            inst_cfg, category_common_config, common_config, "initial_ori_range"
        )
        initial_scale_range = self._get_scale_range_from_config(
            inst_cfg,
            category_common_config,
            common_config,
            cat_spec,
            asset_to_spawn,
        )
        pos = self._generate_pos_from_range(initial_pos_range)
        ori = self._generate_ori_from_range(initial_ori_range)
        scale = self._generate_scale_from_range(initial_scale_range)

        return {
            "pos": pos,
            "ori": ori,
            "scale": scale,
            "prim_path": prim_path,
            "cat_name": cat_name,
            "env_id": env_id,
            "initial_pos_range": initial_pos_range,
            "initial_ori_range": initial_ori_range,
            "initial_scale_range": initial_scale_range,
        }

    def _get_pos_range_from_config(
        self,
        instance_config: Dict,
        category_config: Dict,
        global_config: Dict,
        range_key: str,
    ) -> List[float]:
        """Get the position range based on config priority."""
        # Normalize possibly None configs
        if not isinstance(category_config, (dict, DictConfig)):
            category_config = {}
        if not isinstance(global_config, (dict, DictConfig)):
            global_config = {}
        if "pos" in instance_config and instance_config["pos"] is not None:
            pos = instance_config["pos"]
            if len(pos) != 3:
                raise ValueError(f"Instance 'pos' must have 3 elements, got {len(pos)}")
            return [pos[0], pos[1], pos[2], pos[0], pos[1], pos[2]]
        pos_range = category_config.get(range_key)
        if pos_range is not None:
            if len(pos_range) != 6:
                raise ValueError(
                    f"Category '{range_key}' must have 6 elements, got {len(pos_range)}"
                )
            return pos_range

        pos_range = global_config.get(range_key)
        if pos_range is not None:
            if len(pos_range) != 6:
                raise ValueError(
                    f"Global '{range_key}' must have 6 elements, got {len(pos_range)}"
                )
            return pos_range

        return [0.0] * 6

    def _get_ori_range_from_config(
        self,
        instance_config: Dict,
        category_config: Dict,
        global_config: Dict,
        range_key: str,
    ) -> List[float]:
        """Get the orientation (euler) range based on config priority."""
        # Normalize possibly None configs
        if not isinstance(category_config, (dict, DictConfig)):
            category_config = {}
        if not isinstance(global_config, (dict, DictConfig)):
            global_config = {}
        if "ori" in instance_config and instance_config["ori"] is not None:
            ori = instance_config["ori"]
            if len(ori) != 3:
                raise ValueError(
                    f"Instance 'ori' must have 3 elements (euler angles), got {len(ori)}"
                )
            return [ori[0], ori[1], ori[2], ori[0], ori[1], ori[2]]

        ori_range = category_config.get(range_key)
        if ori_range is not None:
            if len(ori_range) != 6:
                raise ValueError(
                    f"Category '{range_key}' must have 6 elements, got {len(ori_range)}"
                )
            return ori_range

        ori_range = global_config.get(range_key)
        if ori_range is not None:
            if len(ori_range) != 6:
                raise ValueError(
                    f"Global '{range_key}' must have 6 elements, got {len(ori_range)}"
                )
            return ori_range

        return [0.0] * 6

    def _get_scale_range_from_config(
        self,
        instance_config: Dict,
        category_config: Dict,
        global_config: Dict,
        cat_spec: Dict,
        asset_to_spawn: str = None,
    ) -> List[float]:
        """Get the scale range based on config priority. Scale is always fixed."""
        # Normalize possibly None configs
        if not isinstance(category_config, (dict, DictConfig)):
            category_config = {}
        if not isinstance(global_config, (dict, DictConfig)):
            global_config = {}
        scale_val = None

        is_primitive = asset_to_spawn in PRIMITIVE_MAP
        if is_primitive:
            # Prefer primitive_scale if provided (instance-level first, then category-level)
            primitive_scale_inst = instance_config.get("primitive_scale")
            primitive_scale_cat = category_config.get("primitive_scale")
            if primitive_scale_inst is not None:
                scale_val = primitive_scale_inst
            elif primitive_scale_cat is not None:
                scale_val = primitive_scale_cat

        if scale_val is None:
            # Fallback to instance-level scale first (direct), then visual.scale
            direct_scale = instance_config.get("scale")
            if direct_scale is not None:
                scale_val = direct_scale
            else:
                visual_config = instance_config.get("visual", {})
                if "scale" in visual_config and visual_config["scale"] is not None:
                    scale_val = visual_config["scale"]

        if scale_val is None:
            category_scale = category_config.get("scale")
            if category_scale is not None:
                scale_val = category_scale

        if scale_val is None:
            scale_val = global_config.get("scale", [1.0, 1.0, 1.0])

        if len(scale_val) != 3:
            raise ValueError(f"Scale must have 3 elements, got {len(scale_val)}")
        return [
            scale_val[0],
            scale_val[1],
            scale_val[2],
            scale_val[0],
            scale_val[1],
            scale_val[2],
        ]

    def _generate_value_from_range(self, value_range: List[float]) -> np.ndarray:
        """Generates a 3D value by sampling from a 6-element range."""
        if len(value_range) != 6:
            raise ValueError(
                f"Value range must have 6 elements [min_x, min_y, min_z, max_x, max_y, max_z], got {len(value_range)}"
            )

        val = [
            random.uniform(value_range[0], value_range[3]),
            random.uniform(value_range[1], value_range[4]),
            random.uniform(value_range[2], value_range[5]),
        ]
        return np.array(val, dtype=np.float32)

    def _generate_pos_from_range(self, pos_range: List[float]) -> np.ndarray:
        """Generates a position vector from a 6-element range."""
        return self._generate_value_from_range(pos_range)

    def _generate_ori_from_range(self, ori_range: List[float]) -> np.ndarray:
        """Generates a quaternion orientation from a 6-element euler range."""
        euler_val = self._generate_value_from_range(ori_range)
        return euler_angles_to_quat(euler_val, degrees=True)

    def _generate_scale_from_range(self, scale_range: List[float]) -> np.ndarray:
        """Generates a scale vector from a 6-element range."""
        return self._generate_value_from_range(scale_range)

    # --- End of New Helper Functions ---

    def _assign_object_to_category(self, cat_name: str, env_id: int, obj: Any):
        """Assign object to corresponding category collection

        Args:
            cat_name: Category name
            env_id: Environment ID
            obj: Object instance
        """
        if isinstance(obj, RigidObject):
            self._rigid_objects[env_id].setdefault(cat_name, []).append(obj)
        elif isinstance(obj, ArticulationObject):
            self._articulation_objects[env_id].setdefault(cat_name, []).append(obj)
        elif isinstance(obj, RopeObject):
            self._rope_objects[env_id].setdefault(cat_name, []).append(obj)
        elif isinstance(obj, DeformableObject):
            self._deformable_objects[env_id].setdefault(cat_name, []).append(obj)
        elif isinstance(obj, GarmentObject):
            self._garment_objects[env_id].setdefault(cat_name, []).append(obj)
        elif isinstance(obj, GeometryObject):
            self._geometry_objects[env_id].setdefault(cat_name, []).append(obj)
        elif isinstance(obj, FluidObject):
            self._fluid_objects[env_id].setdefault(cat_name, []).append(obj)
        elif isinstance(obj, InflatableObject):
            self._inflatable_objects[env_id].setdefault(cat_name, []).append(obj)
        elif isinstance(obj, FEMClothObject):
            self._fem_cloth_objects[env_id].setdefault(cat_name, []).append(obj)
        elif isinstance(obj, Avatar):
            self._avatar_objects[env_id].setdefault(cat_name, []).append(obj)

    def get_object_layout(
        self, env_id: int, prim_path: str
    ) -> Optional[Dict[str, Any]]:
        """Get layout information for specified object

        Args:
            env_id: Environment ID
            prim_path: Object prim path

        Returns:
            Layout information dictionary, None if not found
        """
        return self._object_layouts[env_id].get(prim_path)

    def update_object_layout(
        self, env_id: int, prim_path: str, layout_info: Dict[str, Any]
    ):
        """Update layout information for specified object

        Args:
            env_id: Environment ID
            prim_path: Object prim path
            layout_info: New layout information
        """
        self._object_layouts[env_id][prim_path] = layout_info

    def generate_new_layout(
        self, env_id: int, prim_path: str, reset_type: str = "soft"
    ) -> Optional[Dict[str, Any]]:
        """Generate new layout information for specified object

        Args:
            env_id: Environment ID
            prim_path: Object prim path
            reset_type: Reset type (not used anymore, kept for compatibility)

        Returns:
            New layout information dictionary
        """
        if prim_path not in self._object_layouts[env_id]:
            return None

        layout_info = self._object_layouts[env_id][prim_path]
        cat_name = layout_info["cat_name"]

        # Always use initial ranges
        pos_range = layout_info.get("initial_pos_range", [0.0] * 6)
        ori_range = layout_info.get("initial_ori_range", [0.0] * 6)
        scale_range = layout_info.get(
            "initial_scale_range", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )

        # Generate new values from the selected ranges
        new_pos = self._generate_pos_from_range(pos_range)
        new_ori = self._generate_ori_from_range(ori_range)
        new_scale = self._generate_scale_from_range(scale_range)
        new_layout = layout_info.copy()
        new_layout.update(
            {
                "pos": new_pos,
                "ori": new_ori,
                "scale": new_scale,
                "prim_path": prim_path,
                "cat_name": cat_name,
                "env_id": env_id,
            }
        )

        self._object_layouts[env_id][prim_path] = new_layout

        return new_layout

    def _find_object_by_prim_path(self, env_id: int, prim_path: str) -> Optional[Any]:
        """Find object instance by prim_path

        Args:
            env_id: Environment ID
            prim_path: Object prim path

        Returns:
            Object instance, None if not found
        """
        all_collections = [
            self._rigid_objects[env_id],
            self._articulation_objects[env_id],
            self._rope_objects[env_id],
            self._deformable_objects[env_id],
            self._garment_objects[env_id],
            self._geometry_objects[env_id],
            self._fluid_objects[env_id],
            self._inflatable_objects[env_id],
            self._fem_cloth_objects[env_id],
            self._avatar_objects[env_id],
        ]

        for obj_dict in all_collections:
            for cat_name, obj_list in obj_dict.items():
                for obj in obj_list:
                    if hasattr(obj, "prim_path") and obj.prim_path == prim_path:
                        return obj
                    elif (
                        hasattr(obj, "usd_prim_path") and obj.usd_prim_path == prim_path
                    ):
                        return obj

        return None

    def hard_reset(self, env_id: int):
        """Hard reset: clear and update object tables

        Args:
            env_id: Environment ID
        """
        is_cuda = self.device.type == "cuda"

        if is_cuda:
            prim_paths_to_keep = set()
            cat_names_to_keep = set()

            collections_to_keep = [
                self._rigid_objects[env_id],
                self._articulation_objects[env_id],
                self._rope_objects[env_id],
            ]

            for obj_dict in collections_to_keep:
                cat_names_to_keep.update(obj_dict.keys())

                for cat_name, obj_list in obj_dict.items():
                    for obj in obj_list:
                        if hasattr(obj, "prim_path") and obj.prim_path:
                            prim_paths_to_keep.add(obj.prim_path)
                        elif hasattr(obj, "usd_prim_path") and obj.usd_prim_path:
                            prim_paths_to_keep.add(obj.usd_prim_path)

            all_layout_paths = list(self._object_layouts[env_id].keys())
            for prim_path in all_layout_paths:
                if prim_path not in prim_paths_to_keep:
                    del self._object_layouts[env_id][prim_path]

            all_cat_names = list(self._category_counters[env_id].keys())
            for cat_name in all_cat_names:
                if cat_name not in cat_names_to_keep:
                    del self._category_counters[env_id][cat_name]

            self._clear_objects(env_id)

        else:
            prim_paths_to_keep = set()
            cat_names_to_keep = set()
            obj_dict = self._articulation_objects[env_id]
            cat_names_to_keep.update(obj_dict.keys())
            for cat_name, obj_list in obj_dict.items():
                for obj in obj_list:
                    if hasattr(obj, "prim_path") and obj.prim_path:
                        prim_paths_to_keep.add(obj.prim_path)
                    elif hasattr(obj, "usd_prim_path") and obj.usd_prim_path:
                        prim_paths_to_keep.add(obj.usd_prim_path)
            all_layout_paths = list(self._object_layouts[env_id].keys())
            for prim_path in all_layout_paths:
                if prim_path not in prim_paths_to_keep:
                    del self._object_layouts[env_id][prim_path]
            all_cat_names = list(self._category_counters[env_id].keys())
            for cat_name in all_cat_names:
                if cat_name not in cat_names_to_keep:
                    del self._category_counters[env_id][cat_name]
            self._clear_objects(env_id)

    def _clear_objects(self, env_id: int):
        """Clear all objects in specified environment

        Args:
            env_id: Environment ID
        """

        is_cuda = self.device.type == "cuda"

        if not is_cuda:
            self._rigid_objects[env_id].clear()
            self._rope_objects[env_id].clear()
        self._deformable_objects[env_id].clear()
        self._garment_objects[env_id].clear()
        self._geometry_objects[env_id].clear()
        self._fluid_objects[env_id].clear()
        self._inflatable_objects[env_id].clear()
        self._fem_cloth_objects[env_id].clear()
        self._avatar_objects[env_id].clear()

        self._rooms[env_id] = None
        self._lights[env_id].clear()

    def get_objects(
        self,
        env_id: int,
        object_name: Optional[str] = None,
        object_type: Optional[str] = None,
    ) -> Dict[str, List[Any]]:
        """Get object instances for specified environment

        Args:
            env_id: Environment ID
            object_name: Object category name, None for all objects
            object_type: Object type, None for all types

        Returns:
            Object dictionary with category names as keys and object lists as values
        """
        result = {}
        obj_collections = self._get_object_collections_by_type(object_type)

        for collection_name, env_dict_list in obj_collections.items():
            if env_id >= len(env_dict_list):
                continue

            env_dict = env_dict_list[env_id]
            for cat_name, obj_list in env_dict.items():
                if object_name is None or cat_name == object_name:
                    key = f"{collection_name}_{cat_name}"
                    result[key] = obj_list

        return result

    def _get_object_collections_by_type(
        self, object_type: Optional[str]
    ) -> Dict[str, List[Dict[str, List[Any]]]]:
        """Get object collections by type

        Args:
            object_type: Object type filter

        Returns:
            Object collections dictionary
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
        }

        if object_type is None:
            return all_collections
        elif object_type in all_collections:
            return {object_type: all_collections[object_type]}
        else:
            return {}

    @property
    def rigid_objects(self) -> List[Dict[str, List[RigidObject]]]:
        return self._rigid_objects

    @property
    def articulation_objects(self) -> List[Dict[str, List[ArticulationObject]]]:
        return self._articulation_objects

    @property
    def rope_objects(self) -> List[Dict[str, List[RopeObject]]]:
        return self._rope_objects

    @property
    def deformable_objects(self) -> List[Dict[str, List[DeformableObject]]]:
        return self._deformable_objects

    @property
    def garment_objects(self) -> List[Dict[str, List[GarmentObject]]]:
        return self._garment_objects

    @property
    def geometry_objects(self) -> List[Dict[str, List[GeometryObject]]]:
        return self._geometry_objects

    @property
    def fluid_objects(self) -> List[Dict[str, List[FluidObject]]]:
        return self._fluid_objects

    @property
    def inflatable_objects(self) -> List[Dict[str, List[InflatableObject]]]:
        return self._inflatable_objects

    @property
    def fem_cloth_objects(self) -> List[Dict[str, List[FEMClothObject]]]:
        return self._fem_cloth_objects

    @property
    def rooms(self) -> List[Optional[Room]]:
        return self._rooms

    @property
    def lights(self) -> List[List[Light]]:
        return self._lights

    @property
    def background(self) -> Optional[Background]:
        return self._background

    def export_objects_layouts_to_json(
        self,
        output_file: str = "objects_layouts.json",
        env_ids: Optional[List[int]] = None,
        timestamp: bool = True,
    ) -> Dict[str, Any]:
        """Export all objects' layouts to JSON file and convert to bbox format.

        Args:
            output_file: Path to output JSON file
            env_ids: List of environment IDs to export. If None, export all environments.
            timestamp: If True, add timestamp to output directory (default: True)

        Returns:
            Dictionary containing all objects layouts in both original and bbox format
        """
        # Apply timestamp if enabled
        if timestamp:
            output_file = self._get_timestamped_output_path(output_file)
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        export_data = {
            "metadata": {
                "num_envs": len(env_ids),
                "env_ids": env_ids,
                "export_format": "layouts_and_bbox",
            },
            "environments": {},
        }

        for env_id in env_ids:
            env_data = {"objects": [], "bboxes": []}

            # Collect all objects in this environment
            all_objects = {}
            for obj_type, obj_collection in [
                ("rigid", self._rigid_objects),
                ("articulation", self._articulation_objects),
                ("rope", self._rope_objects),
                ("deformable", self._deformable_objects),
                ("garment", self._garment_objects),
                ("geometry", self._geometry_objects),
                ("fluid", self._fluid_objects),
                ("inflatable", self._inflatable_objects),
                ("fem_cloth", self._fem_cloth_objects),
                ("avatar", self._avatar_objects),
            ]:
                if env_id < len(obj_collection):
                    for cat_name, obj_list in obj_collection[env_id].items():
                        for obj in obj_list:
                            if obj not in all_objects.values():
                                all_objects[f"{obj_type}_{cat_name}"] = obj

            def convert_value(val):
                import torch

                if isinstance(val, torch.Tensor):
                    if val.numel() == 1:
                        return float(val.item())
                    else:
                        return val.tolist()
                elif isinstance(val, np.ndarray):
                    return val.tolist()
                elif hasattr(val, "item") and hasattr(val, "numel"):
                    if val.numel() == 1:
                        return float(val.item())
                    else:
                        return val.tolist()
                elif isinstance(val, (list, tuple)):
                    result = []
                    for item in val:
                        if isinstance(item, (torch.Tensor, np.ndarray)) or (
                            hasattr(item, "item") and hasattr(item, "numel")
                        ):
                            result.append(convert_value(item))
                        else:
                            result.append(item)
                    return result
                else:
                    return val

            for prim_path, layout_info in self._object_layouts[env_id].items():
                pos = layout_info.get("pos", None)
                ori = layout_info.get("ori", None)
                scale = layout_info.get("scale", None)
                cat_name = layout_info.get("cat_name", "unknown")

                pos_list = convert_value(pos) if pos is not None else None
                ori_list = convert_value(ori) if ori is not None else None
                scale_list = convert_value(scale) if scale is not None else None

                obj_layout = {
                    "prim_path": prim_path,
                    "category": cat_name,
                    "position": pos_list,
                    "orientation": ori_list,
                    "scale": scale_list,
                }
                env_data["objects"].append(obj_layout)

                if pos is not None and ori is not None and scale is not None:
                    bbox = self._layout_to_bbox(pos_list, ori_list, scale_list)
                    bbox_entry = {
                        "prim_path": prim_path,
                        "category": cat_name,
                        "bbox": bbox,
                    }
                    env_data["bboxes"].append(bbox_entry)

            export_data["environments"][f"env_{env_id}"] = env_data

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {
                    key: convert_to_serializable(value) for key, value in obj.items()
                }
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)

        export_data_serializable = convert_to_serializable(export_data)

        with open(output_file, "w") as f:
            json.dump(export_data_serializable, f, indent=2)

        print(f"Exported layouts to {output_file}")
        return export_data

    def _layout_to_bbox(self, pos, ori, scale) -> Dict[str, Any]:
        """Convert layout information (position, orientation, scale) to bounding box format.

        Bbox format: [center, extent, orientation]
        - center: 3D position (x, y, z)
        - extent: 3D size (width, height, depth)
        - orientation: quaternion (w, x, y, z)

        Args:
            pos: Position array [x, y, z] (already converted to list)
            ori: Orientation quaternion [w, x, y, z] (already converted to list)
            scale: Scale array [x, y, z] (already converted to list)

        Returns:
            Dictionary with bbox information
        """
        center = (
            pos if isinstance(pos, list) else list(pos) if pos is not None else None
        )
        extent = (
            scale
            if isinstance(scale, list)
            else list(scale)
            if scale is not None
            else None
        )
        orientation = (
            ori if isinstance(ori, list) else list(ori) if ori is not None else None
        )

        return {
            "center": center,
            "extent": extent,
            "orientation": orientation,
            "type": "axis_aligned",
        }

    def export_objects_ranges_to_json(
        self, output_file: str = "objects_ranges.json", timestamp: bool = True
    ) -> Dict[str, Any]:
        """Export object position/orientation ranges (from config) to JSON.

        This exports, for each category's each item (instance), the configured
        ranges that define where objects can be spawned, not the actual positions.

        Args:
            output_file: Path to output JSON file
            timestamp: If True, add timestamp to output directory (default: True)

        Returns:
            Dictionary containing configuration ranges
        """
        # Apply timestamp if enabled
        if timestamp:
            output_file = self._get_timestamped_output_path(output_file)

        # Resolve global/common config fallbacks
        global_common_config = self.common_config
        objects_cfg = None
        try:
            # Support both dict and DictConfig
            if isinstance(self.config, (dict, DictConfig)) and "objects" in self.config:
                objects_cfg = self.config["objects"]
        except Exception:
            objects_cfg = None

        if isinstance(objects_cfg, (dict, DictConfig)) and ("common" in objects_cfg):
            try:
                common_cfg = objects_cfg["common"]
                global_common_config = (
                    dict(common_cfg)
                    if isinstance(common_cfg, DictConfig)
                    else common_cfg
                )
            except Exception:
                global_common_config = self.common_config

        ranges_data = {
            "metadata": {
                "num_envs": self.num_envs,
                "description": "Per-item spawn ranges from configuration",
            },
            "categories": {},
        }

        # Extract ranges per item using helper functions so priority is correct
        if isinstance(objects_cfg, (dict, DictConfig)):
            for cat_name, cat_spec in objects_cfg.items():
                if self._is_global_config_key(cat_name):
                    continue

                # Allow both dict and OmegaConf DictConfig
                is_mapping = isinstance(cat_spec, (dict, DictConfig))
                category_common_config = (
                    cat_spec.get("common", {}) if is_mapping else {}
                )
                if not isinstance(category_common_config, (dict, DictConfig)):
                    category_common_config = {}

                # collect instance entries (item_1, item_2, ...)
                instances: Dict[str, Dict[str, Any]] = {}
                if is_mapping:
                    for key, value in cat_spec.items():
                        if key in (
                            "common",
                            "num_per_env",
                            "random",
                            "usd",
                            "semantic_label",
                        ) or key.startswith("_"):
                            continue
                        if isinstance(value, (dict, DictConfig)):
                            instances[key] = value

                items_out: Dict[str, Dict[str, Any]] = {}
                for inst_key, inst_cfg in instances.items():
                    # Use helper functions to compute ranges consistently with runtime
                    initial_pos_range = self._get_pos_range_from_config(
                        inst_cfg,
                        category_common_config,
                        global_common_config,
                        "initial_pos_range",
                    )
                    initial_ori_range = self._get_ori_range_from_config(
                        inst_cfg,
                        category_common_config,
                        global_common_config,
                        "initial_ori_range",
                    )
                    initial_scale_range = self._get_scale_range_from_config(
                        inst_cfg,
                        category_common_config,
                        global_common_config,
                        cat_spec,
                        None,
                    )

                    items_out[inst_key] = {
                        "initial_pos_range": initial_pos_range,
                        "initial_ori_range": initial_ori_range,
                        "initial_scale_range": initial_scale_range,
                    }

                ranges_data["categories"][cat_name] = {"items": items_out}

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(ranges_data, f, indent=2)

        print(f"Exported ranges to {output_file}")
        return ranges_data

    def _is_global_config_key(self, key: str) -> bool:
        """Check if key is a global configuration key."""
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
