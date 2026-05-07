import json
import os
from pathlib import Path
import random
import re

import isaacsim.core.utils.prims as prims_utils
import numpy as np
import omni.kit.commands
import omni.usd
import torch
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.prims import (
    add_reference_to_stage,
    get_prim_at_path,
    is_prim_path_valid,
)
from isaacsim.core.utils.semantics import add_labels, remove_labels
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.string import find_unique_string_name
from magicsim.Env.Planner.Utils import quat_mul
from magicsim.Env.Utils.path import get_usd_paths_from_folder
from magicsim.Env.Utils.rotations import quat_to_rot_matrix
from omegaconf import DictConfig
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
import omni.kit.app


class ArticulationObject(SingleArticulation):
    """
    ArticulationObject class that wraps the Isaac Sim SingleArticulation functionality.
    This class inherits from the Isaac Sim SingleArticulation class and can be extended.
    Supports semantic labeling for object identification and scene understanding.
    """

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        config: DictConfig,
        env_origin: torch.Tensor,
        layout_manager=None,
        layout_info=None,
    ):
        """
        Initialize the ArticulationObject with position, orientation, and configuration.
        """
        self._device = SimulationManager.get_physics_sim_device()
        if usd_path:
            prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        else:
            prim = get_prim_at_path(prim_path)

        if not prim or not prim.IsValid():
            error_message = (
                f"Failed to load USD from {usd_path} to {prim_path}"
                if usd_path
                else f"Failed to find an existing prim at path {prim_path}"
            )
            raise RuntimeError(error_message)

        self.stage = get_current_stage()
        self._current_color = None
        self._prim_path = prim_path
        prim_path_parts = prim_path.split("/")
        self.category_name = prim_path_parts[-2]
        self.instance_name = prim_path_parts[-1]
        self.num_per_env = config.objects[self.category_name].get("num_per_env")
        self.instance_name = self._re_instance_name(self.instance_name)

        self.global_config = config
        # Use layout_manager.common_config if available, otherwise fall back to config.objects.common
        if layout_manager and hasattr(layout_manager, "common_config"):
            self.global_common_config = layout_manager.common_config
        else:
            self.global_common_config = (
                self.global_config.objects.common
                if hasattr(self.global_config.objects, "common")
                else {}
            )

        self.category_config = config.objects[self.category_name]
        category_common_config_val = self.category_config.get("common")
        self.category_common_config = (
            category_common_config_val if category_common_config_val is not None else {}
        )

        self.instance_config = self.category_config.get(self.instance_name, {})
        self.visual_cfg = self.instance_config.get("visual", {})
        self.physics_cfg = self.instance_config.get("physics", {})

        # Apply ratio-based randomization to physics parameters
        # self.physics_cfg = self._apply_physics_ratio_randomization(self.physics_cfg)

        inst_visual_material_cfg = self.visual_cfg.get("visual_material", {})

        self.usd_prim_path = prim_path
        self.usd_path = usd_path
        self.prim_name = prim_path_parts[-1]
        self.layout_manager = layout_manager
        self.layout_info = layout_info

        if self.layout_info:
            # Use provided layout info
            pos_from_layout = self.layout_info["pos"]
            # Convert torch tensor to numpy if needed
            if torch.is_tensor(pos_from_layout):
                pos_from_layout = pos_from_layout.cpu().numpy()
            self.init_pos = pos_from_layout
            ori_from_layout = self.layout_info["ori"]
            if torch.is_tensor(ori_from_layout):
                ori_from_layout = ori_from_layout.cpu().numpy()
            self.init_ori = ori_from_layout
            scale_from_layout = self.layout_info["scale"]
            if torch.is_tensor(scale_from_layout):
                scale_from_layout = scale_from_layout.cpu().numpy()
            self.init_scale = scale_from_layout
        else:
            # Must have layout_manager
            if not self.layout_manager:
                raise RuntimeError(
                    f"LayoutManager is required for {self._prim_path}. All position information must come from LayoutManager."
                )

            env_id = self._extract_env_id_from_prim_path()
            if env_id is None:
                raise ValueError(
                    f"Could not extract env_id from prim path: {self._prim_path}"
                )

            layout_info = self.layout_manager.get_object_layout(
                env_id=env_id, prim_path=self._prim_path
            )
            if layout_info is None:
                # If not registered yet, register now using config and keep consistent prim_path
                layout_info = self.layout_manager.register_object_and_get_layout(
                    env_id=env_id,
                    prim_path=self._prim_path,
                    cat_name=self.category_name,
                    inst_cfg=self.instance_config,
                    cat_spec=self.category_config,
                    asset_to_spawn=None,
                )

            if layout_info is None:
                raise RuntimeError(
                    f"LayoutManager failed to generate/retrieve layout for {self._prim_path}"
                )

            pos_from_layout = layout_info["pos"]
            # Convert torch tensor to numpy if needed
            if torch.is_tensor(pos_from_layout):
                pos_from_layout = pos_from_layout.cpu().numpy()
            self.init_pos = pos_from_layout
            ori_from_layout = layout_info["ori"]
            if torch.is_tensor(ori_from_layout):
                ori_from_layout = ori_from_layout.cpu().numpy()
            self.init_ori = ori_from_layout
            scale_from_layout = layout_info["scale"]
            if torch.is_tensor(scale_from_layout):
                scale_from_layout = scale_from_layout.cpu().numpy()
            self.init_scale = scale_from_layout

        super().__init__(
            prim_path=self._prim_path,
            name=self.prim_name,
            translation=self.init_pos,
            orientation=self.init_ori,
            scale=self.init_scale,
        )
        print(
            f"ArticulationObject initialized with prim_path: {self._prim_path}, usd_path: {self.usd_path}"
        )
        print(
            f"Initial position: {self.init_pos}, initial orientation: {self.init_ori}, initial scale: {self.init_scale}"
        )

        self.color_list = self.visual_cfg.get("color")
        if self.color_list is not None and isinstance(self.color_list[0], (int, float)):
            self.color_list = [self.color_list]

        self.visual_material_usd_folder = None
        self.color_material_path = None

        if self.color_list:
            self._current_color = random.choice(self.color_list)
            self._apply_color_material(self._current_color)
        else:
            # Fallback to single color or material folder
            color = self.visual_cfg.get("color")
            if color:  # If a single color is provided
                material_path = find_unique_string_name(
                    initial_name=f"{self.usd_prim_path}/Looks/color_material",
                    is_unique_fn=lambda x: not is_prim_path_valid(x),
                )
                self.color_material_path = material_path
                material = PreviewSurface(
                    prim_path=material_path, color=torch.tensor(color)
                )
                # Apply to all visualizable geometry children
                for child_prim in Usd.PrimRange(self.prim):
                    # Check for common geometry types
                    if (
                        child_prim.IsA(UsdGeom.Mesh)
                        or child_prim.IsA(UsdGeom.Capsule)
                        or child_prim.IsA(UsdGeom.Sphere)
                        or child_prim.IsA(UsdGeom.Cube)
                        or child_prim.IsA(UsdGeom.Cylinder)
                        or child_prim.IsA(UsdGeom.Cone)
                    ):
                        omni.kit.commands.execute(
                            "BindMaterialCommand",
                            prim_path=child_prim.GetPath(),
                            material_path=material_path,
                            strength=UsdShade.Tokens.strongerThanDescendants,
                        )
            else:  # No color list, no single color -> check material folder
                self.visual_material_usd_folder = inst_visual_material_cfg.get(
                    "material_usd_folder"
                )
                if self.visual_material_usd_folder is not None:
                    self.visual_usd_paths = get_usd_paths_from_folder(
                        folder_path=self.visual_material_usd_folder,
                        skip_keywords=[".thumbs"],
                    )
                    if self.visual_usd_paths:
                        selected_path = random.choice(self.visual_usd_paths)
                        # Use the specific material application logic for articulations
                        self._apply_visual_material_from_file_original_logic(
                            selected_path
                        )
                    else:
                        print(
                            f"⚠️ Warning: No visual materials found in '{self.visual_material_usd_folder}'. Skipping material application for {self.usd_prim_path}."
                        )
        visible = self.visual_cfg.get("visible", True)
        if not visible:
            imageable = UsdGeom.Imageable(self.prim)
            if imageable:
                imageable.MakeInvisible()

        # Handle semantic labels
        self._handle_semantic_labels()

        # Load annotations from Annotation folder
        self._load_annotations()

        # Ensure this articulation is tracked in LayoutManager collections so CUDA hard resets preserve its layout
        try:
            if self.layout_manager:
                env_id_for_assign = self._extract_env_id_from_prim_path()
                if env_id_for_assign is not None:
                    self.layout_manager._initialize_category_list(
                        env_id_for_assign, self.category_name
                    )
                    self.layout_manager._assign_object_to_category(
                        self.category_name, env_id_for_assign, self
                    )
        except Exception:
            pass

    # ── Annotation loading & access ──────────────────────────────────────

    def _load_annotations(self):
        """
        Load annotations from Annotation folder if it exists at the same level as USD file.
        Annotations are loaded from JSON files in the Annotation folder, sorted alphabetically.
        Each JSON file should contain a dictionary structure.

        Stores annotations in self.annotations as a dictionary where keys are JSON filenames
        (without extension) and values are the loaded JSON dictionaries.
        """
        self.annotations = {}

        if not self.usd_path:
            return

        try:
            usd_file_path = Path(self.usd_path)

            if not usd_file_path.is_absolute():
                if os.path.exists(self.usd_path):
                    usd_file_path = Path(os.path.abspath(self.usd_path))
                else:
                    usd_file_path = Path(self.usd_path)

            usd_dir = usd_file_path.parent
            annotation_dir = usd_dir / "Annotation"

            if not annotation_dir.exists() or not annotation_dir.is_dir():
                return

            json_files = sorted(annotation_dir.glob("*.json"))
            if not json_files:
                return

            for json_file in json_files:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        annotation_data = json.load(f)
                        annotation_key = json_file.stem
                        self.annotations[annotation_key.lower()] = annotation_data
                except json.JSONDecodeError as e:
                    print(
                        f"⚠️ Warning: Failed to parse JSON file '{json_file}': {e}. Skipping."
                    )
                except Exception as e:
                    print(
                        f"⚠️ Warning: Failed to load annotation file '{json_file}': {e}. Skipping."
                    )
        except Exception as e:
            print(f"⚠️ Warning: Failed to load annotations for {self.usd_path}: {e}")

    def get_annotation(self, annotation_name: str, default=None):
        """
        Get annotation data by name (JSON filename without extension).

        Args:
            annotation_name: Name of the annotation (JSON filename without .json extension)
            default: Default value to return if annotation not found

        Returns:
            Annotation dictionary or default value if not found
        """
        return self.annotations.get(annotation_name, default)

    def has_annotation(self, annotation_name: str) -> bool:
        """
        Check if a specific annotation exists.

        Args:
            annotation_name: Name of the annotation (JSON filename without .json extension)

        Returns:
            True if annotation exists, False otherwise
        """
        return annotation_name in self.annotations

    def list_annotations(self) -> list:
        """
        Get list of all available annotation names.

        Returns:
            List of annotation names (sorted alphabetically)
        """
        return sorted(self.annotations.keys())

    # ── Pose transformation helpers ───────────────────────────────────────

    def _convert_pose_list_to_tensor(self, pose_list, device):
        """Convert list of poses to tensor: [N, 7] with [x, y, z, qw, qx, qy, qz]."""
        if not isinstance(pose_list, list) or len(pose_list) == 0:
            return None
        return torch.tensor(pose_list, dtype=torch.float32, device=device)  # [N, 7]

    def _transform_poses_batch(self, poses_tensor, obj_pos, obj_quat, obj_rot_matrix):
        """Transform poses tensor from local to world coordinate using matrix operations.

        Args:
            poses_tensor: [N, 7] with [x, y, z, qw, qx, qy, qz]
            obj_pos: [3] object world position
            obj_quat: [4] object world quaternion [qw, qx, qy, qz]
            obj_rot_matrix: [3, 3] object rotation matrix

        Returns:
            [N, 7] transformed poses in world coordinate
        """
        local_positions = poses_tensor[:, :3]  # [N, 3]
        local_quats = poses_tensor[:, 3:]  # [N, 4] - [qw, qx, qy, qz]

        # Transform positions: world_pos = obj_rot @ local_pos + obj_pos
        world_positions = (obj_rot_matrix @ local_positions.T).T + obj_pos.unsqueeze(0)

        # Transform quaternions: world_quat = obj_quat * local_quat
        obj_quat_expanded = obj_quat.unsqueeze(0).expand(local_quats.shape[0], -1)
        world_quats = quat_mul(obj_quat_expanded, local_quats)

        return torch.cat([world_positions, world_quats], dim=1)  # [N, 7]

    def _get_object_transform(self, device=None):
        """Get the object's current pose as tensors for transformation.

        Args:
            device: Target device. If None, inferred from object pose.

        Returns:
            (obj_pos, obj_quat, obj_rot_matrix, target_device)
        """
        obj_translation, obj_orientation = self.get_local_pose()

        if device is not None:
            target_device = device
        elif isinstance(obj_translation, torch.Tensor):
            target_device = obj_translation.device
        elif isinstance(obj_orientation, torch.Tensor):
            target_device = obj_orientation.device
        else:
            target_device = "cpu"

        if isinstance(obj_translation, torch.Tensor):
            obj_pos = obj_translation.to(
                dtype=torch.float32, device=target_device
            ).squeeze()[:3]
        else:
            obj_pos = torch.tensor(
                obj_translation, dtype=torch.float32, device=target_device
            )[:3]

        if isinstance(obj_orientation, torch.Tensor):
            obj_quat = obj_orientation.to(
                dtype=torch.float32, device=target_device
            ).squeeze()[:4]
        else:
            obj_quat = torch.tensor(
                obj_orientation, dtype=torch.float32, device=target_device
            )[:4]

        obj_rot_matrix = quat_to_rot_matrix(obj_quat)  # [3, 3]
        return obj_pos, obj_quat, obj_rot_matrix, target_device

    # ── Trajectory pose extraction ────────────────────────────────────────

    def get_trajectory_poses(
        self,
        annotation_name: str = "open_by_handle_trajectory",
        joint_name: str = None,
        transform_to_world: bool = True,
        device: str = None,
    ):
        """
        Get trajectory poses from an annotation file, optionally transformed
        from the object's local coordinate frame to world coordinates.

        The returned dictionary preserves the original annotation structure.
        Inside ``trajectories``, every leaf list of 7-element waypoints
        is converted to a ``torch.Tensor`` of shape ``(N, 7)`` with
        ``[x, y, z, qw, qx, qy, qz]``.

        Args:
            annotation_name: Name of the annotation (JSON filename without
                ``.json`` extension, lowercased).  Default is
                ``"open_by_handle_trajectory"``.
            joint_name: If given, only return trajectories for this joint
                (e.g. ``"joint_1"``).  If ``None`` (default), return all joints.
            transform_to_world: If ``True`` (default), transform every waypoint
                from the object's local frame to the world frame using the
                object's current ``get_local_pose()``.
            device: Torch device for the output tensors.  If ``None``, the
                device is inferred from the object's pose tensors or defaults
                to ``"cpu"``.

        Returns:
            A dictionary that mirrors the annotation structure, e.g.::

                {
                    "type": "Oven",
                    "bottom_center": {"x": ..., "y": ..., "z": ...},
                    "trajectories": {
                        "joint_1": {
                            "1": Tensor (N, 7),
                            "2": Tensor (N, 7),
                            ...
                        },
                        ...
                    },
                    "target_angle_deg": 72.0,
                }

            Returns ``None`` if the annotation does not exist.
        """
        annotation_data = self.get_annotation(annotation_name)
        if annotation_data is None:
            return None

        # Get object transform for coordinate conversion
        obj_pos, obj_quat, obj_rot_matrix, target_device = self._get_object_transform(
            device
        )
        # Deep-copy the top-level structure so we don't mutate cached data
        result = {}
        for key, value in annotation_data.items():
            if self._is_trajectory_dict(value):
                result[key] = self._process_trajectory_dict(
                    value,
                    obj_pos,
                    obj_quat,
                    obj_rot_matrix,
                    transform_to_world,
                    target_device,
                    joint_name,
                )
            else:
                # Keep non-trajectory fields as-is (type, bottom_center, target_angle_deg …)
                result[key] = value

        return result

    @staticmethod
    def _is_trajectory_dict(value):
        """Check whether *value* looks like a trajectory dict.

        A trajectory dict has the structure::

            { joint_name: { traj_id: [[7-element list], …], … }, … }

        i.e. a dict whose values are themselves dicts containing lists of
        7-element pose lists.  This distinguishes it from simple metadata
        dicts like ``bottom_center`` (``{"x": …, "y": …, "z": …}``).
        """
        if not isinstance(value, dict):
            return False
        for v in value.values():
            if isinstance(v, dict):
                for inner in v.values():
                    if (
                        isinstance(inner, list)
                        and len(inner) > 0
                        and isinstance(inner[0], list)
                        and len(inner[0]) == 7
                    ):
                        return True
        return False

    def _process_trajectory_dict(
        self,
        trajectories_dict,
        obj_pos,
        obj_quat,
        obj_rot_matrix,
        transform_to_world,
        device,
        joint_name=None,
    ):
        """Process the ``trajectories`` dict and convert waypoints to tensors.

        Args:
            trajectories_dict: ``{ joint_name: { traj_id: [[7], …], … }, … }``
            obj_pos, obj_quat, obj_rot_matrix: Object pose (world).
            transform_to_world: Whether to apply the local→world transform.
            device: Target torch device.
            joint_name: Optional filter – return only this joint.

        Returns:
            Dictionary with the same nesting but leaf lists replaced by
            ``torch.Tensor (N, 7)``.
        """
        if not isinstance(trajectories_dict, dict):
            return None

        result = {}
        for jname, traj_group in trajectories_dict.items():
            if joint_name is not None and jname != joint_name:
                continue

            if not isinstance(traj_group, dict):
                result[jname] = traj_group
                continue

            processed_group = {}
            for traj_id, waypoints in traj_group.items():
                # Each waypoints should be a list of [7] poses
                if (
                    isinstance(waypoints, list)
                    and len(waypoints) > 0
                    and isinstance(waypoints[0], list)
                    and len(waypoints[0]) == 7
                ):
                    poses_tensor = self._convert_pose_list_to_tensor(waypoints, device)
                    if poses_tensor is not None and transform_to_world:
                        poses_tensor = self._transform_poses_batch(
                            poses_tensor, obj_pos, obj_quat, obj_rot_matrix
                        )
                    processed_group[traj_id] = poses_tensor
                else:
                    processed_group[traj_id] = waypoints

            result[jname] = processed_group

        return result

    def _apply_color_material(self, color):
        """Creates or updates the PreviewSurface material with the specified color."""
        if self.color_material_path is None:
            self.color_material_path = find_unique_string_name(
                initial_name=f"{self.usd_prim_path}/Looks/color_material",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )

        material_prim = get_prim_at_path(self.color_material_path)
        if not material_prim:
            material = PreviewSurface(
                prim_path=self.color_material_path, color=torch.tensor(color)
            )
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.prim.GetPath(),
                material_path=self.color_material_path,
                strength=UsdShade.Tokens.strongerThanDescendants,
            )
        else:
            material = PreviewSurface(prim_path=self.color_material_path)
            try:
                material.set_color(np.array(color))
            except Exception as e:
                print(
                    f"Error setting color for material {self.color_material_path}: {e}"
                )

    def _apply_visual_material_from_file_original_logic(self, material_path: str):
        """Applies a visual material from a USD file using the original logic."""
        visual_material_path = find_unique_string_name(
            self.usd_prim_path + "/visual_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        add_reference_to_stage(usd_path=material_path, prim_path=visual_material_path)
        visual_material_ref_prim = prims_utils.get_prim_at_path(visual_material_path)
        if not visual_material_ref_prim or not visual_material_ref_prim.IsValid():
            return
        material_children = prims_utils.get_prim_children(visual_material_ref_prim)
        if not material_children:
            return

        material_prim = material_children[0]
        material_prim_path_str = material_prim.GetPath().pathString

        for child_prim in Usd.PrimRange(self.prim):
            if (
                child_prim.IsA(UsdGeom.Mesh)
                or child_prim.IsA(UsdGeom.Capsule)
                or child_prim.IsA(UsdGeom.Sphere)
                or child_prim.IsA(UsdGeom.Cube)
                or child_prim.IsA(UsdGeom.Cylinder)
                or child_prim.IsA(UsdGeom.Cone)
            ):
                omni.kit.commands.execute(
                    "BindMaterialCommand",
                    prim_path=child_prim.GetPath(),
                    material_path=material_prim_path_str,
                    strength=UsdShade.Tokens.strongerThanDescendants,
                )

    def _re_instance_name(self, inst_name):
        parts = inst_name.split("_")
        cat_name_extracted = "_".join(parts[:-1])
        obj_id_str = parts[-1]
        obj_id = int(obj_id_str)
        original_id = (obj_id - 1) % self.num_per_env + 1
        inst_name = f"{cat_name_extracted}_{original_id}"
        return inst_name

    def hide_prim(self, prim_path: str):
        try:
            path = Sdf.Path(prim_path)
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                return
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()
            else:
                visibility_attribute = prim.GetAttribute("visibility")
                if visibility_attribute:
                    visibility_attribute.Set("invisible")
        except Exception as e:
            print(f"Warning: Failed to hide prim {prim_path}: {e}")

    def _find_first_mesh_in_hierarchy(self, prim_path: str) -> str:
        start_prim = get_prim_at_path(prim_path)
        if not start_prim:
            return None
        for prim in Usd.PrimRange(start_prim):
            if prim.IsA(UsdGeom.Mesh):
                return prim.GetPath().pathString
        return None

    def get_current_mesh_points(
        self,
        visualize: bool = False,
        save: bool = False,
        save_path: str = "./pointcloud.ply",
    ):
        """
        Get current mesh vertex positions for the first mesh under this articulation.

        Returns (points_world, points_local, pos_world, ori_world).
        """
        mesh_path = self._find_first_mesh_in_hierarchy(self._prim_path)
        if mesh_path is None:
            return np.array([]), np.array([]), None, None

        mesh_prim = UsdGeom.Mesh.Get(self.stage, mesh_path)
        points_local = np.array(mesh_prim.GetPointsAttr().Get(), dtype=np.float32)
        world_tf = omni.usd.get_world_transform_matrix(mesh_prim.GetPrim())

        points_world = np.array(
            [
                list(
                    world_tf.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
                )
                for p in points_local
            ],
            dtype=np.float32,
        )

        rot_quat = world_tf.ExtractRotationQuat()
        pos_world = np.array(world_tf.ExtractTranslation(), dtype=np.float32)
        ori_world = np.array(
            [rot_quat.GetReal(), *rot_quat.GetImaginary()], dtype=np.float32
        )

        if visualize or save:
            try:
                import open3d as o3d

                if points_world.size > 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points_world)
                    if visualize:
                        o3d.visualization.draw_geometries([pcd])
                    if save:
                        o3d.io.write_point_cloud(save_path, pcd)
            except Exception as e:
                print(
                    f"Error during visualization/saving articulation point cloud: {e}"
                )

        return points_world, points_local, pos_world, ori_world

    def set_current_mesh_points(
        self, mesh_points: np.ndarray, pos_world=None, ori_world=None
    ):
        """
        Set current mesh vertex positions (local space) back to the first mesh under this articulation.
        """
        mesh_path = self._find_first_mesh_in_hierarchy(self._prim_path)
        if mesh_path is None:
            return
        mesh_prim = UsdGeom.Mesh.Get(self.stage, mesh_path)
        try:
            mesh_prim.GetPointsAttr().Set(
                Vt.Vec3fArray.FromNumpy(np.asarray(mesh_points, dtype=np.float32))
            )
        except Exception as e:
            print(f"Error setting articulation mesh points: {e}")

    def initialize(self):
        print("ArticulationObject initialize")
        self.physics_sim_view = SimulationManager.get_physics_sim_view()
        super().initialize(physics_sim_view=self.physics_sim_view)
        self.upper_joint_positions = self.dof_properties["upper"].copy()
        self.lower_joint_positions = self.dof_properties["lower"].copy()
        self.initial_joint_positions = self.get_current_joint_positions()
        self.app = omni.kit.app.get_app()
        self.app.update()

    @property
    def num_joints(self) -> int:
        """Number of joints (DOFs) in this articulation."""
        if hasattr(self, "upper_joint_positions"):
            return len(self.upper_joint_positions)
        if hasattr(self, "dof_properties") and self.dof_properties is not None:
            return len(self.dof_properties["upper"])
        return 0

    def get_current_joint_positions(self):
        return self.get_joint_positions()

    def set_current_joint_positions(self, positions):
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float32)
        self.set_joint_positions(positions)

    def _extract_env_id_from_prim_path(self):
        """ "get env_id from prim_path"""
        try:
            parts = self._prim_path.split("/")
            for part in parts:
                if part.startswith("env_"):
                    return int(part.split("_")[1])
        except (ValueError, IndexError):
            pass
        return None

    def reset(self, soft=False):
        """Reset articulation pose using LayoutManager."""
        self.set_current_joint_positions(self.initial_joint_positions)

        if not self.layout_manager:
            raise RuntimeError(
                f"LayoutManager is required for {self._prim_path}. All position information must come from LayoutManager."
            )

        env_id = self._extract_env_id_from_prim_path()
        if env_id is None:
            print(
                f"Warning: Could not extract env_id for {self._prim_path}. Cannot perform reset."
            )
            return

        reset_type = "soft" if soft else "hard"
        new_layout = self.layout_manager.generate_new_layout(
            env_id=env_id, prim_path=self._prim_path, reset_type=reset_type
        )

        if not new_layout:
            print(
                f"Warning: LayoutManager did not provide new layout for {self._prim_path}. Cannot perform reset."
            )
            return

        pos = new_layout["pos"]
        # Convert torch tensor to numpy if needed
        if torch.is_tensor(pos):
            pos = pos.cpu().numpy()
        ori = new_layout["ori"]
        if torch.is_tensor(ori):
            ori = ori.cpu().numpy()
        scale = new_layout["scale"]
        if torch.is_tensor(scale):
            scale = scale.cpu().numpy()

        self.set_local_pose(pos, ori)
        self.set_local_scale(np.array(scale))
        self.app.update()

    def _handle_semantic_labels(self):
        """Manage semantic labeling: clear existing labels and apply new ones."""

        remove_labels(self.prim, include_descendants=True)
        semantic_label = self._get_semantic_label()
        if semantic_label:
            add_labels(self.prim, [semantic_label])
            self.semantic_label = semantic_label

    def _get_semantic_label(self) -> str:
        """Generate semantic label from configuration or USD filename."""

        if (
            hasattr(self.category_config, "semantic_label")
            and self.category_config.semantic_label
        ):
            return self.category_config.semantic_label

        if not self.usd_path:
            return ""

        regex_pattern = self.category_config.get("semantic_regex_pattern", r".*")
        regex_replacement = self.category_config.get("semantic_regex_repl", r"\g<0>")
        filename = os.path.basename(self.usd_path)
        filename_without_ext = os.path.splitext(filename)[0]
        return re.sub(regex_pattern, regex_replacement, filename_without_ext)

    def reset_hard(self, soft: bool = False):
        """Reset articulation pose and optionally randomize appearance.

        Mirrors Rigid.reset_hard: fetch new layout from LayoutManager, apply
        translation/orientation/scale, then randomize color or visual material.
        """
        self.set_current_joint_positions(self.initial_joint_positions)

        if not self.layout_manager:
            raise RuntimeError(
                f"LayoutManager is required for {self._prim_path}. All position information must come from LayoutManager."
            )

        env_id = self._extract_env_id_from_prim_path()
        if env_id is None:
            print(
                f"Warning: Could not extract env_id for {self._prim_path}. Cannot perform reset."
            )
            return

        reset_type = "soft" if soft else "hard"
        new_layout = self.layout_manager.generate_new_layout(
            env_id=env_id, prim_path=self._prim_path, reset_type=reset_type
        )

        if not new_layout:
            print(
                f"Warning: LayoutManager did not provide new layout for {self._prim_path}. Cannot perform reset."
            )
            return

        pos = torch.tensor(new_layout["pos"], dtype=torch.float32)
        ori = torch.tensor(new_layout["ori"], dtype=torch.float32)
        scale = torch.tensor(new_layout["scale"], dtype=torch.float32)

        self.set_local_pose(pos, ori)
        self.set_local_scale(np.array(scale))

        visible = self.visual_cfg.get("visible", True)
        imageable = UsdGeom.Imageable(self.prim)
        if imageable:
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()

        if self.color_list:
            random_color = random.choice(self.color_list)
            self._apply_color_material(random_color)
        elif (
            self.color_list is None
            and hasattr(self, "visual_material_usd_folder")
            and self.visual_material_usd_folder
            and hasattr(self, "visual_usd_paths")
            and self.visual_usd_paths
        ):
            selected_path = random.choice(self.visual_usd_paths)
            self._apply_visual_material_from_file_original_logic(selected_path)
        self.app.update()

    def get_state(self, is_relative: bool = False) -> dict[str, torch.Tensor]:
        """Get the state of the articulation object.

        Args:
            is_relative: If True, positions are relative to environment origin. Defaults to False.

        Returns:
            Dictionary containing:
                - root_pose: torch.Tensor, shape (7,), position (3) and quaternion (4)
                - root_velocity: torch.Tensor, shape (6,), linear velocity (3) and angular velocity (3)
                - joint_position: torch.Tensor, shape (num_joints,), joint positions
                - joint_velocity: torch.Tensor, shape (num_joints,), joint velocities
                - asset_info: dict with usd_path and primitive_type
        """
        try:
            world_tf = omni.usd.get_world_transform_matrix(self.prim)
            translation = np.array(world_tf.ExtractTranslation(), dtype=np.float32)
            rot_quat = world_tf.ExtractRotationQuat()
            orientation = np.array(
                [rot_quat.GetReal(), *rot_quat.GetImaginary()], dtype=np.float32
            )
        except (AttributeError, RuntimeError) as e:
            try:
                translation, orientation = self.get_current_pose()
                translation = np.array(translation, dtype=np.float32)
                orientation = np.array(orientation, dtype=np.float32)
            except (AttributeError, RuntimeError):
                print(f"Warning: Failed to get pose for {self._prim_path}: {e}")
                translation = np.zeros(3, dtype=np.float32)
                orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        if not isinstance(translation, torch.Tensor):
            translation = torch.tensor(
                translation,
                dtype=torch.float32,
                device=self.device if hasattr(self, "device") else "cpu",
            )
        if not isinstance(orientation, torch.Tensor):
            orientation = torch.tensor(
                orientation, dtype=torch.float32, device=translation.device
            )

        if translation.dim() > 1:
            translation = translation.squeeze()
        if orientation.dim() > 1:
            orientation = orientation.squeeze()

        root_pose = torch.cat([translation, orientation])

        if is_relative and hasattr(self, "env_origin"):
            env_origin_tensor = (
                torch.tensor(
                    self.env_origin, dtype=torch.float32, device=root_pose.device
                )
                if isinstance(self.env_origin, np.ndarray)
                else self.env_origin
            )
            if env_origin_tensor.dim() == 0:
                env_origin_tensor = env_origin_tensor.unsqueeze(0)
            if env_origin_tensor.shape[0] < 3:
                env_origin_tensor = torch.cat(
                    [
                        env_origin_tensor,
                        torch.zeros(
                            3 - env_origin_tensor.shape[0],
                            device=env_origin_tensor.device,
                        ),
                    ]
                )
            root_pose[:3] -= env_origin_tensor[:3]

        try:
            if (
                hasattr(self, "_articulation_prim_view")
                and self._articulation_prim_view is not None
            ):
                linear_vel = self._articulation_prim_view.get_linear_velocities()
                angular_vel = self._articulation_prim_view.get_angular_velocities()
                if linear_vel is not None and angular_vel is not None:
                    if not isinstance(linear_vel, torch.Tensor):
                        linear_vel = torch.tensor(
                            linear_vel, dtype=torch.float32, device=root_pose.device
                        )
                    if not isinstance(angular_vel, torch.Tensor):
                        angular_vel = torch.tensor(
                            angular_vel, dtype=torch.float32, device=root_pose.device
                        )
                    if linear_vel.dim() > 1:
                        linear_vel = linear_vel.squeeze()
                    if angular_vel.dim() > 1:
                        angular_vel = angular_vel.squeeze()
                    root_velocity = torch.cat([linear_vel, angular_vel])
                else:
                    root_velocity = torch.zeros(
                        6, dtype=torch.float32, device=root_pose.device
                    )
            else:
                linear_vel = self.get_linear_velocity()
                angular_vel = self.get_angular_velocity()
                if not isinstance(linear_vel, torch.Tensor):
                    linear_vel = torch.tensor(
                        linear_vel, dtype=torch.float32, device=root_pose.device
                    )
                if not isinstance(angular_vel, torch.Tensor):
                    angular_vel = torch.tensor(
                        angular_vel, dtype=torch.float32, device=root_pose.device
                    )
                if linear_vel.dim() > 1:
                    linear_vel = linear_vel.squeeze()
                if angular_vel.dim() > 1:
                    angular_vel = angular_vel.squeeze()
                root_velocity = torch.cat([linear_vel, angular_vel])
        except (AttributeError, RuntimeError):
            root_velocity = torch.zeros(6, dtype=torch.float32, device=root_pose.device)

        joint_position = self.get_current_joint_positions()
        if not isinstance(joint_position, torch.Tensor):
            joint_position = torch.tensor(
                joint_position, dtype=torch.float32, device=root_pose.device
            )
        if joint_position.dim() > 1:
            joint_position = joint_position.squeeze()

        try:
            joint_velocity = self.get_joint_velocities()
            if not isinstance(joint_velocity, torch.Tensor):
                joint_velocity = torch.tensor(
                    joint_velocity, dtype=torch.float32, device=root_pose.device
                )
            if joint_velocity.dim() > 1:
                joint_velocity = joint_velocity.squeeze()
        except (AttributeError, RuntimeError):
            joint_velocity = torch.zeros_like(joint_position)

        asset_info = {
            "usd_path": self.usd_path if hasattr(self, "usd_path") else None,
            "primitive_type": None,
        }

        return {
            "root_pose": root_pose,
            "root_velocity": root_velocity,
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "asset_info": asset_info,
        }

    def set_root_transform(self, pos: torch.Tensor, ori: torch.Tensor):
        self._articulation_view.set_world_poses(
            positions=pos, orientations=ori, indices=[0]
        )

    def destroy(self):
        self.set_world_pose(position=torch.randint(1000, 1099511627776, size=(3,)))
        self._articulation_view.destroy()
