import random
import re
import os
import json
from pathlib import Path
from typing import Optional
import numpy as np
from omegaconf import DictConfig
import torch
from pxr import PhysxSchema, UsdGeom, Sdf, UsdPhysics, UsdShade, Vt, Gf, Usd
import omni.kit.commands
from magicsim.Env.Utils.path import (
    get_usd_paths_from_folder,
    resolve_mdl_paths,
    resolve_path,
)
from magicsim.Env.Planner.Utils import quat_mul
from magicsim.Env.Utils.rotations import quat_to_rot_matrix
import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.prims import SingleRigidPrim, SingleGeometryPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.prims import get_prim_at_path
from magicsim.Env.Scene.Object.physics_material import PhysicsMaterial
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.semantics import add_labels, remove_labels
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.replicator.behavior.utils.scene_utils import create_mdl_material
import omni.usd


class RigidObject(SingleRigidPrim, SingleGeometryPrim):
    """Rigid body object with physical properties, collision detection, and semantic labeling.
    Combines geometry (visual/collision) and rigid body dynamics, with support for semantic tagging.
    """

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        config: DictConfig,
        env_origin: torch.Tensor,
        layout_manager=None,
        layout_info=None,
        primitive_type: str = None,
    ):
        print(usd_path)
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

        """Initialize a rigid body object with geometry and physics properties."""
        self.stage = prim.GetStage()
        self._prim_path = prim_path
        prim_path_parts = prim_path.split("/")
        self.category_name = prim_path_parts[-2]
        self.instance_name = prim_path_parts[-1]
        self.num_per_env = config.objects[self.category_name].get("num_per_env")
        self.instance_name = self._re_instance_name(self.instance_name)

        self.primitive_type = primitive_type
        self.global_config = config
        self.category_config = config.objects[self.category_name]
        self.instance_config = self.category_config.get(self.instance_name, {})
        category_common_config_val = self.category_config.get("common")
        self.category_common_config = (
            category_common_config_val if category_common_config_val is not None else {}
        )
        # Use layout_manager.common_config if available, otherwise fall back to config.objects.common
        if layout_manager and hasattr(layout_manager, "common_config"):
            self.global_common_config = layout_manager.common_config
        else:
            self.global_common_config = (
                self.global_config.objects.common
                if hasattr(self.global_config.objects, "common")
                else {}
            )
        self.visual_config = self.instance_config.get("visual", {})
        self.physics_config = self.instance_config.get("physics", {})

        # Apply ratio-based randomization to physics parameters
        self.physics_config = self._apply_physics_ratio_randomization(
            self.physics_config
        )

        inst_physics_material_cfg = self.physics_config.get("physics_material", {})
        inst_visual_material_cfg = self.visual_config.get("visual_material", {})
        self.layout_manager = layout_manager
        self.layout_info = layout_info

        self.usd_path = usd_path
        self.usd_prim_path = prim_path
        self._current_color = None

        if self.layout_info:
            # Use provided layout info
            translation = self.layout_info["pos"]
            orientation = self.layout_info["ori"]
            scale = self.layout_info["scale"]
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
                raise RuntimeError(
                    f"LayoutManager failed to generate/retrieve layout for {self._prim_path}"
                )

            translation = layout_info["pos"]
            orientation = layout_info["ori"]
            scale = layout_info["scale"]

        self.physics_material_path = find_unique_string_name(
            prim_path + "/physcis_material",
            lambda x: not is_prim_path_valid(x),
        )
        self.physics_material = PhysicsMaterial(
            prim_path=self.physics_material_path,
            static_friction=inst_physics_material_cfg.get("static_friction", 1.0),
            dynamic_friction=inst_physics_material_cfg.get("dynamic_friction", 1.0),
            restitution=inst_physics_material_cfg.get("restitution", 0.5),
            friction_combine_mode=inst_physics_material_cfg.get(
                "friction_combine_mode", "multiply"
            ),
            restitution_combine_mode=inst_physics_material_cfg.get(
                "restitution_combine_mode", "multiply"
            ),
        )

        SingleGeometryPrim.__init__(
            self,
            prim_path=prim_path,
            name=self.instance_name,
            scale=scale,
            visible=self.visual_config.get("visible", True),
            collision=self.physics_config.get("collision", True),
            track_contact_forces=False,
        )

        SingleRigidPrim.__init__(
            self,
            prim_path=prim_path,
            name=self.instance_name,
            translation=translation,
            orientation=orientation,
            scale=scale,
        )

        self._default_linear_velocity = None
        self._default_angular_velocity = None

        self.color_list = self.visual_config.get("color")
        if self.color_list is not None and isinstance(self.color_list[0], (int, float)):
            self.color_list = [self.color_list]

        self.color_material_path = None
        self.visual_material_usd_folder = None
        self.visual_material_mdl_path = None
        self.visual_material_mdl_folder = None

        default_material = self.category_config.get("default_material")
        if self.color_list:
            self._current_color = random.choice(self.color_list)
            self._apply_color_material(self._current_color)
        else:
            # Fallback to single color or material folder/file
            color = self.visual_config.get("color")
            if color is not None:
                # Note: _apply_color_as_material might create a new material each time.
                # Consider modifying it or using _apply_color_material if consistent material path is needed.
                self._apply_color_as_material(color)
            else:  # color is None
                # Check for MDL support
                self.visual_material_mdl_path = inst_visual_material_cfg.get("mdl_path")
                self.visual_material_mdl_folder = inst_visual_material_cfg.get(
                    "mdl_folder"
                )

                if self.visual_material_mdl_path:
                    self._apply_mdl_material(
                        mdl_path=self.visual_material_mdl_path,
                        mdl_name=inst_visual_material_cfg.get("mdl_name"),
                    )
                elif self.visual_material_mdl_folder:
                    resolved_mdl_paths = resolve_mdl_paths(
                        self.visual_material_mdl_folder
                    )
                    if resolved_mdl_paths:
                        selected_path = random.choice(resolved_mdl_paths)
                        self._apply_mdl_material(mdl_path=selected_path)
                    else:
                        print(
                            f"⚠️ Warning: No MDL files found in folder: {self.visual_material_mdl_folder}"
                        )
                else:
                    # Fallback to USD material folder
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
                            self._apply_visual_material_from_file(selected_path)
                        else:
                            print(
                                f"⚠️ Warning: No visual materials found in '{self.visual_material_usd_folder}'. Skipping material application for {self.usd_prim_path}."
                            )
                    else:
                        # No color, MDL or USD visual_material folder
                        # Check default_material setting
                        if default_material is True:
                            # Apply default material when default_material is True
                            self._apply_default_material(
                                material_name="default_material"
                            )
                        elif default_material is False:
                            # Do nothing - use USD's own material when default_material is False
                            pass
                        # If default_material is None/not set, maintain original behavior (no material binding)

        self._setup_physics()
        self._handle_semantic_labels()
        self._load_annotations()

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
            # Get the directory containing the USD file
            usd_file_path = Path(self.usd_path)

            # Handle both absolute and relative paths
            if not usd_file_path.is_absolute():
                # Try to resolve relative path
                if os.path.exists(self.usd_path):
                    usd_file_path = Path(os.path.abspath(self.usd_path))
                else:
                    # If path doesn't exist, try to construct from workspace
                    # This handles cases where usd_path might be relative to project root
                    usd_file_path = Path(self.usd_path)

            usd_dir = usd_file_path.parent
            annotation_dir = usd_dir / "Annotation"

            # Check if Annotation folder exists
            if not annotation_dir.exists() or not annotation_dir.is_dir():
                return

            # Find all JSON files in Annotation folder, sorted alphabetically
            json_files = sorted(annotation_dir.glob("*.json"))

            if not json_files:
                return

            # Load each JSON file
            for json_file in json_files:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        annotation_data = json.load(f)
                        # Use filename without extension as key
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

    def _convert_pose_list_to_tensor(self, pose_list, device):
        """Convert list of poses to tensor: [N, 7] with [x, y, z, qw, qx, qy, qz]."""
        if not isinstance(pose_list, list) or len(pose_list) == 0:
            return None
        return torch.tensor(pose_list, dtype=torch.float32, device=device)  # [N, 7]

    def _transform_poses_batch(self, poses_tensor, obj_pos, obj_quat, obj_rot_matrix):
        """Transform poses tensor from local to world coordinate using matrix operations."""
        # poses_tensor: [N, 7] with [x, y, z, qw, qx, qy, qz]
        local_positions = poses_tensor[:, :3]  # [N, 3] - [x, y, z]
        local_quats = poses_tensor[
            :, 3:
        ]  # [N, 4] - [qw, qx, qy, qz] where qw=w, so [w, x, y, z]

        # Transform positions: world_pos = obj_rot @ local_pos + obj_pos
        world_positions = (obj_rot_matrix @ local_positions.T).T + obj_pos.unsqueeze(
            0
        )  # [N, 3]

        # Transform quaternions: world_quat = obj_quat * local_quat
        obj_quat_expanded = obj_quat.unsqueeze(0).expand(
            local_quats.shape[0], -1
        )  # [N, 4]
        world_quats = quat_mul(obj_quat_expanded, local_quats)  # [N, 4] - [w, x, y, z]

        # Concatenate back to [N, 7] format: [x, y, z, w, x, y, z]
        return torch.cat([world_positions, world_quats], dim=1)  # [N, 7]

    @staticmethod
    def transform_pose_to_world(
        local_pose: torch.Tensor,
        obj_pos: torch.Tensor,
        obj_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Transform 7D pose [x,y,z,qw,qx,qy,qz] from object frame to world frame."""
        obj_rot = quat_to_rot_matrix(obj_quat.unsqueeze(0))[0]
        local_pos = local_pose[:3]
        local_quat = local_pose[3:7]
        world_pos = (obj_rot @ local_pos) + obj_pos
        world_quat = quat_mul(obj_quat.unsqueeze(0), local_quat.unsqueeze(0)).squeeze(0)
        return torch.cat([world_pos, world_quat], dim=0)

    def _process_grasp_dict(
        self, grasp_dict, obj_pos, obj_quat, obj_rot_matrix, transform_to_world, device
    ):
        """Process grasp dictionary. Handles both formats:
        - grasp_pose: part -> list of [7D poses]
        - xhand_grasp_pose: part -> list of {coarse_grasp/fine_grasp/final_grasp: {position, orientation, joints}}
        Uses batched _transform_poses_batch for both.
        """
        if not isinstance(grasp_dict, dict):
            return None

        result_dict = {}
        for key, value in grasp_dict.items():
            if not isinstance(value, list) or len(value) == 0:
                result_dict[key] = value
                continue

            first = value[0]
            # xhand format: list of {coarse_grasp, fine_grasp, final_grasp}
            if isinstance(first, dict) and "coarse_grasp" in first:
                pose_list = []
                for raw in value:
                    for phase in ("coarse_grasp", "fine_grasp", "final_grasp"):
                        pd = raw.get(phase)
                        if pd is not None:
                            pose_list.append(
                                list(pd["position"]) + list(pd["orientation"])
                            )
                if not pose_list:
                    result_dict[key] = []
                    continue
                poses_tensor = torch.tensor(
                    pose_list, dtype=torch.float32, device=device
                )
                if transform_to_world and obj_rot_matrix is not None:
                    poses_tensor = self._transform_poses_batch(
                        poses_tensor, obj_pos, obj_quat, obj_rot_matrix
                    )
                # Reassemble
                idx = 0
                candidates = []
                for raw in value:
                    candidate = {}
                    for phase in ("coarse_grasp", "fine_grasp", "final_grasp"):
                        if raw.get(phase) is None:
                            continue
                        candidate[phase] = {
                            "position": poses_tensor[idx, :3],
                            "orientation": poses_tensor[idx, 3:7],
                            "joints": torch.tensor(
                                raw[phase]["joints"],
                                dtype=torch.float32,
                                device=device,
                            ),
                        }
                        idx += 1
                    if candidate:
                        candidates.append(candidate)
                result_dict[key] = candidates
            # grasp_pose format: list of [7D poses]
            elif isinstance(first, list) and len(first) == 7:
                poses_tensor = self._convert_pose_list_to_tensor(value, device)
                if (
                    transform_to_world
                    and obj_rot_matrix is not None
                    and poses_tensor is not None
                ):
                    poses_tensor = self._transform_poses_batch(
                        poses_tensor, obj_pos, obj_quat, obj_rot_matrix
                    )
                result_dict[key] = poses_tensor
            else:
                result_dict[key] = value
        return result_dict

    def get_grasp_poses(
        self,
        grasp_type: str = None,
        transform_to_world: bool = True,
        device: str = None,
        hand_type: str = None,
    ):
        """
        Get grasp poses from annotation.

        Args:
            grasp_type: Type of grasp to retrieve.
                hand_type=None: "functional_grasp", "grasp", or None (all).
                hand_type set: same keys, or part name (e.g. "body") to get that part's list.
            transform_to_world: If True, transform grasp poses from local coordinate
                                to world coordinate using object's current pose.
            device: Device for tensor operations (e.g., "cpu", "cuda:0").
            hand_type: None = use "grasp_pose" annotation;
                      "xhand" = use "xhand_grasp_pose" annotation (i.e. {hand_type}_grasp_pose).

        Returns:
            hand_type=None: dict with "functional_grasp"/"grasp" (or grasp_type subset)
            hand_type set: same structure {"functional_grasp": {part: [candidates]}, "grasp": {...}}
                           each candidate has coarse_grasp/fine_grasp/final_grasp
            None if annotation doesn't exist
        """
        annotation_name = (
            "grasp_pose" if hand_type is None else f"{hand_type}_grasp_pose"
        )
        grasp_pose_data = self.get_annotation(annotation_name)
        if grasp_pose_data is None:
            return None

        # Get object's current pose once
        obj_translation, obj_orientation = self.get_local_pose()

        # Determine target device
        if device is not None:
            target_device = device
        elif isinstance(obj_translation, torch.Tensor):
            target_device = obj_translation.device
        elif isinstance(obj_orientation, torch.Tensor):
            target_device = obj_orientation.device
        else:
            target_device = "cpu"

        # Convert object pose to torch tensors on target device
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

        obj_rot_matrix = quat_to_rot_matrix(obj_quat) if transform_to_world else None

        # hand_type set: for loop + _process_grasp_dict (batched per part)
        if hand_type is not None:
            result = {"functional_grasp": {}, "grasp": {}}
            for top_key in ("functional_grasp", "grasp"):
                top_data = grasp_pose_data.get(top_key, {})
                if isinstance(top_data, dict):
                    processed = self._process_grasp_dict(
                        top_data,
                        obj_pos,
                        obj_quat,
                        obj_rot_matrix,
                        transform_to_world,
                        target_device,
                    )
                    if processed:
                        result[top_key] = processed
            if grasp_type is not None:
                if grasp_type in ("functional_grasp", "grasp"):
                    return result[grasp_type] if result[grasp_type] else None
                for top_key in ("functional_grasp", "grasp"):
                    if grasp_type in result.get(top_key, {}):
                        return result[top_key][grasp_type]
                return None
            return result if (result["functional_grasp"] or result["grasp"]) else None

        # hand_type=None: process grasp_pose format (functional_grasp, grasp)
        if grasp_type is None:
            # Return all grasp data, always include both keys (empty dict if missing)
            result = {"functional_grasp": {}, "grasp": {}}
            if "functional_grasp" in grasp_pose_data:
                functional_grasp_processed = self._process_grasp_dict(
                    grasp_pose_data["functional_grasp"],
                    obj_pos,
                    obj_quat,
                    obj_rot_matrix,
                    transform_to_world,
                    target_device,
                )
                if functional_grasp_processed:
                    result["functional_grasp"] = functional_grasp_processed
            if "grasp" in grasp_pose_data:
                grasp_processed = self._process_grasp_dict(
                    grasp_pose_data["grasp"],
                    obj_pos,
                    obj_quat,
                    obj_rot_matrix,
                    transform_to_world,
                    target_device,
                )
                if grasp_processed:
                    result["grasp"] = grasp_processed
            return result
        else:
            # Return specific grasp type
            if grasp_type not in grasp_pose_data:
                return None
            grasp_data = grasp_pose_data.get(grasp_type)
            if grasp_data is None:
                return None
            return self._process_grasp_dict(
                grasp_data,
                obj_pos,
                obj_quat,
                obj_rot_matrix,
                transform_to_world,
                target_device,
            )

    def _find_first_mesh_in_hierarchy(self, prim_path: str) -> str:
        start_prim = get_prim_at_path(prim_path)
        if not start_prim:
            return None
        for prim in Usd.PrimRange(start_prim):
            if prim.IsA(UsdGeom.Mesh):
                return prim.GetPath().pathString
        return None

    def _get_mesh_volume_local(self) -> Optional[float]:
        """Compute mesh volume in local space (no scale). Returns None if mesh not found or invalid."""
        mesh_path = self._find_first_mesh_in_hierarchy(self._prim_path)
        if mesh_path is None:
            return None
        mesh_prim = UsdGeom.Mesh.Get(self.stage, mesh_path)
        points_attr = mesh_prim.GetPointsAttr()
        face_counts_attr = mesh_prim.GetFaceVertexCountsAttr()
        face_indices_attr = mesh_prim.GetFaceVertexIndicesAttr()
        if not points_attr or not face_counts_attr or not face_indices_attr:
            return None
        points = np.array(points_attr.Get(), dtype=np.float64)
        face_counts = face_counts_attr.Get()
        face_indices = face_indices_attr.Get()
        if len(points) == 0 or len(face_counts) == 0:
            return None
        volume = 0.0
        idx = 0
        for n in face_counts:
            if n == 3:
                i0, i1, i2 = (
                    face_indices[idx],
                    face_indices[idx + 1],
                    face_indices[idx + 2],
                )
                p0, p1, p2 = points[i0], points[i1], points[i2]
                volume += np.dot(np.cross(p1 - p0, p2 - p0), p0) / 6.0
            elif n == 4:
                i0, i1, i2, i3 = (
                    face_indices[idx],
                    face_indices[idx + 1],
                    face_indices[idx + 2],
                    face_indices[idx + 3],
                )
                p0, p1, p2, p3 = points[i0], points[i1], points[i2], points[i3]
                volume += np.dot(np.cross(p1 - p0, p2 - p0), p0) / 6.0
                volume += np.dot(np.cross(p2 - p0, p3 - p0), p0) / 6.0
            else:
                for k in range(1, n - 1):
                    i0, i1, i2 = (
                        face_indices[idx],
                        face_indices[idx + k],
                        face_indices[idx + k + 1],
                    )
                    p0, p1, p2 = points[i0], points[i1], points[i2]
                    volume += np.dot(np.cross(p1 - p0, p2 - p0), p0) / 6.0
            idx += n
        return float(abs(volume)) if volume != 0.0 else None

    def get_current_mesh_points(
        self,
        visualize: bool = False,
        save: bool = False,
        save_path: str = "./pointcloud.ply",
    ):
        """
        Get current mesh vertex positions for this rigid object.

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
                print(f"Error during visualization/saving rigid point cloud: {e}")

        return points_world, points_local, pos_world, ori_world

    def set_current_mesh_points(
        self, mesh_points: np.ndarray, pos_world=None, ori_world=None
    ):
        """
        Set current mesh vertex positions (local space) back to the mesh. Pose update is optional.
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
            print(f"Error setting rigid mesh points: {e}")

    def hide_prim(self, prim_path: str):
        """
        Hide a prim by setting its visibility to invisible.
        This will make the prim and all its children invisible and ignored by physics.

        Args:
            prim_path: The prim path to hide
        """
        try:
            path = Sdf.Path(prim_path)
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                print(f"Warning: Invalid prim path {prim_path}")
                return

            # Set visibility to invisible
            visibility_attribute = prim.GetAttribute("visibility")
            if visibility_attribute is None:
                # Create the visibility attribute if it doesn't exist
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()
            else:
                visibility_attribute.Set("invisible")

        except Exception as e:
            print(f"Warning: Failed to hide prim {prim_path}: {e}")

    def _apply_color_material(self, color_rgb):
        """Creates or updates a simple PreviewSurface material with the given color and binds it."""
        if self.color_material_path is None:
            self.color_material_path = find_unique_string_name(
                initial_name=self.usd_prim_path + "/Looks/color_material",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )
        material_prim = get_prim_at_path(self.color_material_path)
        if not material_prim:
            material = PreviewSurface(
                prim_path=self.color_material_path, color=torch.tensor(color_rgb)
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
                material.set_color(np.array(color_rgb))
            except Exception as e:
                print(
                    f"Error setting color for material {self.color_material_path}: {e}"
                )

    def _apply_color_as_material(self, color_rgb):
        """Creates a simple PreviewSurface material with the given color and binds it."""
        material_path = find_unique_string_name(
            initial_name=self.usd_prim_path + "/color_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        PreviewSurface(prim_path=material_path, color=np.array(color_rgb))
        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=self.prim.GetPath(),
            material_path=material_path,
            strength=UsdShade.Tokens.strongerThanDescendants,
        )
        # Also bind to any mesh children to ensure override
        children_prims = prims_utils.get_prim_children(self.prim)
        for prim in children_prims:
            if prim.IsA(UsdGeom.Gprim):
                omni.kit.commands.execute(
                    "BindMaterialCommand",
                    prim_path=prim.GetPath(),
                    material_path=material_path,
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

    def _apply_visual_material_from_file(self, material_path: str):
        self.visual_material_path = find_unique_string_name(
            self.usd_prim_path + "/visual_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        add_reference_to_stage(
            usd_path=material_path, prim_path=self.visual_material_path
        )
        visual_material_ref_prim = prims_utils.get_prim_at_path(
            self.visual_material_path
        )
        material_children = prims_utils.get_prim_children(visual_material_ref_prim)
        if not material_children:
            print(
                f"Warning: Material USD at {material_path} has no child prims to bind."
            )
            return

        self.material_prim = material_children[0]
        self.material_prim_path = self.material_prim.GetPath()
        self.visual_material = PreviewSurface(self.material_prim_path)
        object_prim = self.prim
        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=object_prim.GetPath(),
            material_path=self.material_prim_path,
            strength=UsdShade.Tokens.strongerThanDescendants,
        )
        children_prims = prims_utils.get_prim_children(object_prim)
        for prim in children_prims:
            if prim.GetTypeName() in ["Mesh", "GeomSubset"]:
                omni.kit.commands.execute(
                    "BindMaterialCommand",
                    prim_path=prim.GetPath(),
                    material_path=self.material_prim_path,
                    strength=UsdShade.Tokens.strongerThanDescendants,
                )

    def _apply_mdl_material(self, mdl_path: str, mdl_name: str = None):
        """Apply an MDL material to the rigid object.

        Args:
            mdl_path: Path to the MDL file
            mdl_name: Name of the material in the MDL file (optional)
        """
        resolved_mdl_path = resolve_path(mdl_path)
        if not resolved_mdl_path:
            print(f"Warning: MDL material path not found: {mdl_path}")
            return

        if not mdl_name:
            mdl_name = os.path.splitext(os.path.basename(resolved_mdl_path))[0]

        # Create unique material path under object's Looks
        material_path = find_unique_string_name(
            initial_name=f"{self.usd_prim_path}/Looks/mdl_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )

        try:
            # Create the MDL material
            create_mdl_material(resolved_mdl_path, mdl_name, material_path)

            # Bind material to the prim and its children
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.prim.GetPath(),
                material_path=material_path,
                strength=UsdShade.Tokens.strongerThanDescendants,
            )

            children_prims = prims_utils.get_prim_children(self.prim)
            for prim in children_prims:
                if prim.GetTypeName() in ["Mesh", "GeomSubset"]:
                    omni.kit.commands.execute(
                        "BindMaterialCommand",
                        prim_path=prim.GetPath(),
                        material_path=material_path,
                        strength=UsdShade.Tokens.strongerThanDescendants,
                    )

            self.visual_material_path = material_path
            self.visual_material = PreviewSurface(material_path)

        except Exception as e:
            print(f"Warning: Failed to apply MDL material {mdl_path}: {e}")

    def _apply_default_material(
        self, material_name: str = "default_material", prim_path: str = None
    ) -> str:
        """Creates a default PreviewSurface material (no color) and binds it to the prim.

        This ensures the object always has a material bound, even if no color or material
        is specified in the configuration.

        Args:
            material_name: Name for the material (default: "default_material")
            prim_path: Prim path to bind the material to. If None, uses self.prim.GetPath()

        Returns:
            str: The path of the created material
        """
        if prim_path is None:
            prim_path = self.prim.GetPath()

        opaque_mtl_path = f"{self.usd_prim_path}/Looks/{material_name}"
        PreviewSurface(prim_path=opaque_mtl_path)
        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=prim_path,
            material_path=opaque_mtl_path,
            strength=UsdShade.Tokens.strongerThanDescendants,
        )
        return opaque_mtl_path

    def _extract_env_id_from_prim_path(self):
        """get env_id from prim_path"""
        try:
            parts = self._prim_path.split("/")
            for part in parts:
                if part.startswith("env_"):
                    return int(part.split("_")[1])
        except (ValueError, IndexError):
            pass
        return None

    def _setup_physics(self):
        """Configure physics properties (rigid type, mass) from instance config.
        Uses mass if provided and > 0; otherwise uses density * volume (with scale) to compute mass.
        Uses self.physics_config (already randomized by ratio) so ratio-based randomization applies.
        """
        self.rigid_schema_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(self.prim)
        self.rigid_api = UsdPhysics.RigidBodyAPI(self.prim)
        physics_config = self.physics_config
        self.rigid_type = physics_config.get("type", "dynamic")
        assert self.rigid_type == "dynamic", (
            f"{self.prim_path} from {self.usd_path} is not a dynamic rigid body"
        )
        mass_cfg = physics_config.get("mass")
        if mass_cfg is not None and mass_cfg > 0:
            self.mass = float(mass_cfg)
        else:
            density = physics_config.get("density")
            if density is not None and density > 0:
                scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
                if self.layout_info and "scale" in self.layout_info:
                    scale = np.array(self.layout_info["scale"], dtype=np.float64)
                vol_local = self._get_mesh_volume_local()
                if vol_local is not None and vol_local > 0:
                    scale_product = scale[0] * scale[1] * scale[2]
                    self.mass = float(density) * vol_local * scale_product
                else:
                    self.mass = 0.01
            else:
                self.mass = 0.01
        if self.rigid_type == "dynamic" and self.mass <= 0:
            self.mass = 0.01
        self.set_mass(self.mass)

        lin_vel = physics_config.get("linear_velocity")
        if lin_vel is not None:
            self._default_linear_velocity = np.array(lin_vel, dtype=np.float32)
            self.set_linear_velocity(torch.tensor(self._default_linear_velocity))

        ang_vel = physics_config.get("angular_velocity")
        if ang_vel is not None:
            self._default_angular_velocity = np.array(ang_vel, dtype=np.float32)
            self.set_angular_velocity(torch.tensor(self._default_angular_velocity))

        if (
            self._default_linear_velocity is not None
            or self._default_angular_velocity is not None
        ):
            self.set_default_state(
                linear_velocity=self._default_linear_velocity,
                angular_velocity=self._default_angular_velocity,
            )
        if self.physics_material is not None:
            self.apply_physics_material(physics_material=self.physics_material)

        self.rigid_schema_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(self.prim)
        self.rigid_schema_body_api.CreateContactSlopCoefficientAttr().Set(2.0)
        if self.prim.HasAPI(UsdPhysics.RigidBodyAPI):
            self.rigid_api = UsdPhysics.RigidBodyAPI(self.prim)
        else:
            self.rigid_api = UsdPhysics.RigidBodyAPI.Apply(self.prim)

    def _set_initial_pose(self):
        """Set initial local pose (translation/orientation) from LayoutManager."""

        if self.layout_info:
            # Use provided layout info
            translation = self.layout_info["pos"]
            orientation = self.layout_info["ori"]
            scale = self.layout_info["scale"]
            self.set_local_scale(np.array(scale))
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
                raise RuntimeError(
                    f"LayoutManager failed to generate/retrieve layout for {self._prim_path}"
                )

            translation = layout_info["pos"]
            orientation = layout_info["ori"]
            scale = layout_info["scale"]

            self.set_local_scale(np.array(scale))

        self.set_local_pose(translation=translation, orientation=orientation)

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

        if self.primitive_type:
            return self.primitive_type

        if not self.usd_path:
            return ""

        regex_pattern = self.category_config.get("semantic_regex_pattern", r".*")
        regex_replacement = self.category_config.get("semantic_regex_repl", r"\g<0>")
        filename = os.path.basename(self.usd_path)
        filename_without_ext = os.path.splitext(filename)[0]
        return re.sub(regex_pattern, regex_replacement, filename_without_ext)

    def reset(self, soft: bool = False):
        """Reset object pose and velocity."""

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

        if new_layout:
            translation = new_layout["pos"]
            orientation = new_layout["ori"]
            scale = new_layout["scale"]
            self.set_local_scale(np.array(scale))
        else:
            print(
                f"Warning: LayoutManager did not provide new layout for {self._prim_path}. Cannot perform reset."
            )
            return

        self.set_local_pose(translation=translation, orientation=orientation)
        self._apply_default_velocities()

    def reset_hard(self, soft: bool = False):
        """Reset object pose and velocity."""

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

        if new_layout:
            translation = new_layout["pos"]
            orientation = new_layout["ori"]
            scale = new_layout["scale"]
            self.set_local_scale(np.array(scale))
        else:
            print(
                f"Warning: LayoutManager did not provide new layout for {self._prim_path}. Cannot perform reset."
            )
            return

        self.set_local_pose(translation=translation, orientation=orientation)
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
            self._apply_visual_material_from_file(selected_path)
        self._apply_default_velocities()

    def initialize(self):
        self.physics_sim_view = SimulationManager.get_physics_sim_view()
        super().initialize(physics_sim_view=self.physics_sim_view)

    def _apply_default_velocities(self):
        """Reset linear/angular velocity on every reset.

        Default to ZERO so a soft reset wipes any residual motion from
        the previous episode (e.g. spin imparted by a failed grasp,
        drop velocity after a successful lift). If the scene config set
        explicit ``linear_velocity`` / ``angular_velocity`` defaults at
        spawn time, those override the zero baseline here.
        """
        lin = (
            self._default_linear_velocity
            if self._default_linear_velocity is not None
            else np.zeros(3, dtype=np.float32)
        )
        ang = (
            self._default_angular_velocity
            if self._default_angular_velocity is not None
            else np.zeros(3, dtype=np.float32)
        )
        self.set_linear_velocity(torch.tensor(lin))
        self.set_angular_velocity(torch.tensor(ang))

    def _apply_physics_ratio_randomization(self, physics_config):
        """Apply ratio-based randomization to physics parameters.

        Args:
            physics_config: Original physics configuration dictionary

        Returns:
            Modified physics configuration with randomized values
        """
        # Create a copy to avoid modifying the original config
        modified_config = physics_config.copy()

        # Get ratio from physics config, default to 1.0 if not specified
        ratio = modified_config.get("ratio", 1.0)

        # If ratio is 1.0, no randomization needed
        if ratio == 1.0:
            return modified_config

        # List of physics parameters to randomize
        physics_params_to_randomize = [
            "mass",
            "density",
            "linear_velocity",
            "angular_velocity",
        ]

        # Randomize physics parameters
        for param in physics_params_to_randomize:
            if param in modified_config and modified_config[param] is not None:
                original_value = modified_config[param]
                if isinstance(original_value, (int, float)):
                    # Calculate random range: original_value ± (original_value * (ratio - 1))
                    variation = original_value * (ratio - 1)
                    min_val = original_value - variation
                    max_val = original_value + variation
                    modified_config[param] = random.uniform(min_val, max_val)
                elif isinstance(original_value, list) and len(original_value) > 0:
                    # Handle list values (e.g., velocity vectors)
                    randomized_list = []
                    for val in original_value:
                        if isinstance(val, (int, float)):
                            variation = val * (ratio - 1)
                            min_val = val - variation
                            max_val = val + variation
                            randomized_list.append(random.uniform(min_val, max_val))
                        else:
                            randomized_list.append(val)
                    modified_config[param] = randomized_list

        # Randomize physics material parameters
        if "physics_material" in modified_config:
            physics_material = modified_config["physics_material"].copy()
            material_params_to_randomize = [
                "static_friction",
                "dynamic_friction",
                "restitution",
            ]

            for param in material_params_to_randomize:
                if param in physics_material and physics_material[param] is not None:
                    original_value = physics_material[param]
                    if isinstance(original_value, (int, float)):
                        variation = original_value * (ratio - 1)
                        min_val = original_value - variation
                        max_val = original_value + variation
                        physics_material[param] = random.uniform(min_val, max_val)

            modified_config["physics_material"] = physics_material

        return modified_config

    def destroy(self):
        self._rigid_prim_view.disable_gravities()
        self._geometry_prim_view.disable_collision()
        self._rigid_prim_view.set_visibilities([False])
        self._geometry_prim_view.set_visibilities([False])
        self.set_world_pose(position=np.array([100.0, 100.0, 100.0]))
        self.hide_prim(self._prim_path)
