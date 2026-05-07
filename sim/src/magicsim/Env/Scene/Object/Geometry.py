# Object/Geometry.py

import random
import re
import os
from omegaconf import DictConfig
import omni.kit.commands
from magicsim.Env.Utils.path import (
    get_usd_paths_from_folder,
    resolve_mdl_paths,
    resolve_path,
)
import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.api.materials.preview_surface import PreviewSurface
import torch
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Vt, Gf
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.replicator.behavior.utils.scene_utils import create_mdl_material
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.semantics import add_labels, remove_labels
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.simulation_manager import SimulationManager
import omni.usd
import open3d as o3d


class GeometryObject(SingleGeometryPrim):
    """Static geometry object with collision support and semantic labeling.

    Represents static objects (e.g., rooms, walls, fixed structures) with collision detection
    and semantic tagging, configured entirely through the provided configuration.
    """

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        config: DictConfig,
        env_origin: torch.Tensor,
        layout_manager=None,
        layout_info=None,
        config_root: str = "objects",
        primitive_type: str = None,
    ):
        """Initialize a static geometry object.

        Args:
            prim_path: USD prim path for the object (format: .../{config_root}/{category}/{instance})
            usd_path: Path to the USD asset file. Can be None for procedurally generated objects.
            config: Global configuration containing object properties
            config_root: Root node name for object settings in config, default is "objects"
            primitive_type: The name of the primitive shape (e.g., "Cube"), if applicable.
        """
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

        prim_path_parts = prim_path.split("/")
        self._prim_path = prim_path
        self.category_name = prim_path_parts[-2]
        self.instance_name = prim_path_parts[-1]

        self.primitive_type = primitive_type
        self.config_root = config_root
        self.global_config = config
        self.usd_path = usd_path
        self.usd_prim_path = prim_path
        self._current_color = None
        self.layout_manager = layout_manager
        self.layout_info = layout_info
        root_section = config[self.config_root]

        self.category_config = root_section[self.category_name]
        self.num_per_env = self.category_config.get("num_per_env")
        category_common_config_val = self.category_config.get("common")
        self.category_common_config = (
            category_common_config_val if category_common_config_val is not None else {}
        )

        if self.num_per_env is None:
            num_per_env_val = self.category_common_config.get("num_per_env")
            if num_per_env_val is not None:
                self.num_per_env = num_per_env_val
        # Use layout_manager.common_config if available, otherwise fall back to config.objects.common
        if layout_manager and hasattr(layout_manager, "common_config"):
            default_common_config = layout_manager.common_config
        else:
            default_common_config = (
                self.global_config.objects.common
                if hasattr(self.global_config.objects, "common")
                else {}
            )
        self.global_common_config = root_section.get("common", default_common_config)

        if self.num_per_env is None:
            num_per_env_global = self.global_common_config.get("num_per_env")
            if num_per_env_global is not None:
                self.num_per_env = num_per_env_global

        if self.num_per_env is None:
            print(
                f"Warning: Could not determine 'num_per_env' for {self._prim_path}. Defaulting to 1."
            )
            self.num_per_env = 1
        instance_name_for_config = self._re_instance_name(self.instance_name)

        self.instance_config = self.category_config.get(instance_name_for_config, {})

        self.visual_config = self.instance_config.get("visual", {})
        self.physics_config = self.instance_config.get("physics", {})

        # Apply ratio-based randomization to physics parameters
        self.physics_config = self._apply_physics_ratio_randomization(
            self.physics_config
        )

        inst_physics_material_cfg = self.physics_config.get("physics_material", {})
        inst_visual_material_cfg = self.visual_config.get("visual_material", {})

        collision_val = self.physics_config.get("collision")
        collision = collision_val if collision_val is not None else False

        track_forces_val = self.physics_config.get("track_contact_forces")
        track_contact_forces = (
            track_forces_val if track_forces_val is not None else False
        )

        prepare_sensor_val = self.physics_config.get("prepare_contact_sensor")
        prepare_contact_sensor = (
            prepare_sensor_val if prepare_sensor_val is not None else False
        )

        disable_stab_val = self.physics_config.get("disable_stablization")
        disable_stablization = (
            disable_stab_val if disable_stab_val is not None else True
        )

        contact_filter_val = self.physics_config.get("contact_filter_prim_paths_expr")
        contact_filter_prim_paths_expr = (
            contact_filter_val if contact_filter_val is not None else []
        )
        if self.layout_info:
            # Use provided layout info
            self.init_pos = self.layout_info["pos"]
            self.init_ori = self.layout_info["ori"]
            self.init_scale = self.layout_info["scale"]
            final_scale = self.init_scale
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

            self.init_pos = layout_info["pos"]
            self.init_ori = layout_info["ori"]
            self.init_scale = layout_info["scale"]
            final_scale = self.init_scale

        self.physics_material = PhysicsMaterial(
            prim_path=self.usd_prim_path + "/physcis_material",
            static_friction=inst_physics_material_cfg.get("static_friction", 0.0),
            dynamic_friction=inst_physics_material_cfg.get("dynamic_friction", 0.0),
            restitution=inst_physics_material_cfg.get("restitution", 0.5),
        )

        super().__init__(
            prim_path=prim_path,
            name=self.instance_name,
            translation=self.init_pos,
            orientation=self.init_ori,
            scale=final_scale,
            visible=self.visual_config.get("visible", True),
            collision=collision,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensor=prepare_contact_sensor,
            disable_stablization=disable_stablization,
            contact_filter_prim_paths_expr=contact_filter_prim_paths_expr,
        )

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
        self._handle_semantic_labels()

    def _find_first_mesh_in_hierarchy(self, prim_path: str) -> str:
        """Recursively searches for the first prim of type UsdGeom.Mesh under the given path."""
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
        Get current mesh vertex positions for this geometry object.

        Returns (points_world, points_local, pos_world, ori_world).
        """
        mesh_path = self._find_first_mesh_in_hierarchy(self.usd_prim_path)
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
                if points_world.size > 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points_world)
                    if visualize:
                        o3d.visualization.draw_geometries([pcd])
                    if save:
                        o3d.io.write_point_cloud(save_path, pcd)
            except Exception as e:
                print(f"Error during visualization/saving geometry point cloud: {e}")

        return points_world, points_local, pos_world, ori_world

    def set_current_mesh_points(
        self, mesh_points: np.ndarray, pos_world=None, ori_world=None
    ):
        """
        Set current mesh vertex positions (local space) back to the mesh. Pose update is optional.
        """
        mesh_path = self._find_first_mesh_in_hierarchy(self.usd_prim_path)
        if mesh_path is None:
            return
        mesh_prim = UsdGeom.Mesh.Get(self.stage, mesh_path)
        try:
            mesh_prim.GetPointsAttr().Set(
                Vt.Vec3fArray.FromNumpy(np.asarray(mesh_points, dtype=np.float32))
            )
        except Exception as e:
            print(f"Error setting geometry mesh points: {e}")

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

        children_prims = prims_utils.get_prim_children(self.prim)
        for prim in children_prims:
            if prim.IsA(UsdGeom.Gprim):
                omni.kit.commands.execute(
                    "BindMaterialCommand",
                    prim_path=prim.GetPath(),
                    material_path=material_path,
                    strength=UsdShade.Tokens.strongerThanDescendants,
                )

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
        """Apply an MDL material to the geometry object.

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

        # Create unique material path under geometry's Looks
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

    def reset(self, soft: bool = False):
        """Reset object pose to initial or random values."""

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

    def _re_instance_name(self, inst_name):
        parts = inst_name.split("_")
        cat_name_extracted = "_".join(parts[:-1])
        obj_id_str = parts[-1]
        obj_id = int(obj_id_str)
        original_id = (obj_id - 1) % self.num_per_env + 1
        inst_name = f"{cat_name_extracted}_{original_id}"
        return inst_name

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

    def get_state(self, is_relative: bool = False) -> dict[str, torch.Tensor]:
        """Get the state of the geometry object.

        Args:
            is_relative: If True, positions are relative to environment origin. Defaults to False.

        Returns:
            Dictionary containing:
                - root_pose: torch.Tensor, shape (7,), position (3) and quaternion (4)
                - asset_info: dict with usd_path and primitive_type
        """
        try:
            import omni.usd

            world_tf = omni.usd.get_world_transform_matrix(self.prim)
            translation = np.array(world_tf.ExtractTranslation(), dtype=np.float32)
            rot_quat = world_tf.ExtractRotationQuat()
            orientation = np.array(
                [rot_quat.GetReal(), *rot_quat.GetImaginary()], dtype=np.float32
            )
        except (AttributeError, RuntimeError) as e:
            try:
                translation = self.get_translation()
                orientation = self.get_orientation()
                translation = np.array(translation, dtype=np.float32)
                orientation = np.array(orientation, dtype=np.float32)
            except (AttributeError, RuntimeError):
                print(f"Warning: Failed to get pose for {self._prim_path}: {e}")
                translation = np.zeros(3, dtype=np.float32)
                orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        if not isinstance(translation, torch.Tensor):
            translation = torch.tensor(translation, dtype=torch.float32)
        if not isinstance(orientation, torch.Tensor):
            orientation = torch.tensor(orientation, dtype=torch.float32)

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

        asset_info = {
            "usd_path": self.usd_path if hasattr(self, "usd_path") else None,
            "primitive_type": self.primitive_type
            if hasattr(self, "primitive_type")
            else None,
        }

        return {
            "root_pose": root_pose,
            "asset_info": asset_info,
        }

    def initialize(self):
        self.physics_sim_view = SimulationManager.get_physics_sim_view()
        super().initialize(physics_sim_view=self.physics_sim_view)
