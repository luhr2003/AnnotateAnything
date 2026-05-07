import torch
import numpy as np
import random
import re
import os
import omni.kit.commands
import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.prims import SingleDeformablePrim
from isaacsim.core.api.materials.deformable_material import DeformableMaterial
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.semantics import add_labels, remove_labels
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Vt, UsdGeom, Usd, PhysxSchema, UsdShade, Sdf
from magicsim.Env.Utils.path import (
    get_usd_paths_from_folder,
    resolve_mdl_paths,
    resolve_path,
)
from omegaconf import DictConfig
from isaacsim.replicator.behavior.utils.scene_utils import create_mdl_material


class DeformableObject(SingleDeformablePrim):
    """
    DeformableObject class that wraps the Isaac Sim deformable prim functionality.
    Inherits from Isaac Sim's SingleDeformablePrim class and can be extended with custom functionality.
    Supports semantic labeling for object identification and scene understanding.
    """

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        config: DictConfig,
        env_origin: torch.Tensor,
        layout_manager=None,
        primitive_type: str = None,
        layout_info=None,
    ):
        """
        Initialize the DeformableObject with USD reference and physics setup.

        Args:
            prim_path: Path to the prim in the stage
            usd_path: Path to the USD asset file for this object
            config: Configuration dictionary containing object properties
            env_origin: Origin position of the environment
            primitive_type: Type of primitive if using a primitive shape
        """
        if primitive_type == "Plane":
            raise ValueError(
                f"DeformableObject '{prim_path}' does not support the 'Plane' primitive type. "
                "Planes are 2D and cannot be used for volumetric deformation."
            )

        # Parameters Configuration
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

        visual_cfg_val = self.instance_config.get("visual")
        self.visual_cfg = visual_cfg_val if visual_cfg_val is not None else {}
        physics_cfg_val = self.instance_config.get("physics")
        self.physics_cfg = physics_cfg_val if physics_cfg_val is not None else {}

        # Apply ratio-based randomization to physics parameters
        self.physics_cfg = self._apply_physics_ratio_randomization(self.physics_cfg)

        inst_deformable_cfg_val = self.physics_cfg.get("deformable_config")
        self.inst_deformable_cfg = (
            inst_deformable_cfg_val if inst_deformable_cfg_val is not None else {}
        )

        inst_visual_material_cfg_val = self.visual_cfg.get("visual_material")
        self.inst_visual_material_cfg = (
            inst_visual_material_cfg_val
            if inst_visual_material_cfg_val is not None
            else {}
        )

        # USD path configurations
        self.usd_prim_path = prim_path
        self.usd_path = usd_path
        self.prim_name = prim_path.split("/")[-1]
        self.config = config
        self.objects_config = config.get("objects")
        self._current_color = None
        self.is_primitive_asset = False

        if self.primitive_type is not None:
            self.is_primaitive_asset = True
        elif usd_path and "/Object/Primitive/" in usd_path:
            self.is_primitive_asset = True

        self.env_origin = env_origin.detach().cpu().numpy()
        self.layout_manager = layout_manager
        self.layout_info = layout_info

        if self.layout_info:
            # Use provided layout info
            pos_from_layout = self.layout_info["pos"]
            self.init_pos = (
                np.array(pos_from_layout, dtype=np.float32) + self.env_origin
            )
            self.init_ori = self.layout_info["ori"]
            self.init_scale = self.layout_info["scale"]
        else:
            # Must have layout_manager
            if not self.layout_manager:
                raise RuntimeError(
                    f"LayoutManager is required for {self.usd_prim_path}. All position information must come from LayoutManager."
                )

            env_id = self._extract_env_id_from_prim_path()
            if env_id is None:
                raise ValueError(
                    f"Could not extract env_id from prim path: {self.usd_prim_path}"
                )

            layout_info = self.layout_manager.get_object_layout(
                env_id=env_id, prim_path=self.usd_prim_path
            )
            if layout_info is None:
                raise RuntimeError(
                    f"LayoutManager failed to generate/retrieve layout for {self.usd_prim_path}"
                )

            pos_from_layout = layout_info["pos"]
            self.init_pos = (
                np.array(pos_from_layout, dtype=np.float32) + self.env_origin
            )
            self.init_ori = layout_info["ori"]
            self.init_scale = layout_info["scale"]

        # Load USD asset as reference
        if usd_path:
            add_reference_to_stage(usd_path=usd_path, prim_path=self.usd_prim_path)
            self.mesh_prim_path = self._find_first_mesh_in_hierarchy(self.usd_prim_path)
            if self.mesh_prim_path is None:
                raise RuntimeError(
                    f"Could not find a UsdGeom.Mesh prim under the referenced asset at {self.usd_prim_path}"
                )
        else:
            self.mesh_prim_path = self.usd_prim_path

        # Configure simulation resolution
        simulation_hex_res = self.inst_deformable_cfg.get(
            "simulation_hexahedral_resolution", 24
        )

        if self.primitive_type is not None:
            primitive_sim_hex_res = self.category_common_config.get(
                "primitive_simulation_hexahedral_resolution"
            )
            if primitive_sim_hex_res is not None:
                simulation_hex_res = primitive_sim_hex_res
        if self.is_primitive_asset:
            primitive_sim_hex_res = self.category_common_config.get(
                "primitive_simulation_hexahedral_resolution"
            )
            if primitive_sim_hex_res is not None:
                simulation_hex_res = primitive_sim_hex_res

        # Setup materials and initialize parent class
        self._setup_deformable_material()

        super().__init__(
            prim_path=self.mesh_prim_path,
            deformable_material=self.deformable_material,
            scale=self.init_scale,
            name=self.prim_name,
            vertex_velocity_damping=self.inst_deformable_cfg.get(
                "vertex_velocity_damping", 0.0
            ),
            sleep_damping=self.inst_deformable_cfg.get("sleep_damping", 0.10),
            sleep_threshold=self.inst_deformable_cfg.get("sleep_threshold", 0.15),
            settling_threshold=self.inst_deformable_cfg.get("settling_threshold", 0.15),
            self_collision=self.inst_deformable_cfg.get("self_collision", True),
            solver_position_iteration_count=self.inst_deformable_cfg.get(
                "solver_position_iteration_count", 16
            ),
            simulation_hexahedral_resolution=simulation_hex_res,
            kinematic_enabled=self.inst_deformable_cfg.get("kinematic_enabled", False),
            collision_simplification=self.inst_deformable_cfg.get(
                "collision_simplification", True
            ),
            collision_simplification_remeshing=self.inst_deformable_cfg.get(
                "collision_simplification_remeshing", True
            ),
            collision_simplification_remeshing_resolution=self.inst_deformable_cfg.get(
                "collision_simplification_remeshing_resolution", 16
            ),
        )
        self.stage = self.prim.GetStage()
        visible = self.visual_cfg.get("visible", True)
        if not visible:
            imageable = UsdGeom.Imageable(self.prim)
            imageable.MakeInvisible()
        self.color_list = self.visual_cfg.get("color")
        if self.color_list is not None and isinstance(self.color_list[0], (int, float)):
            self.color_list = [self.color_list]

        self.visual_material_usd_folder = None
        self.visual_material_mdl_path = None
        self.visual_material_mdl_folder = None

        if self.color_list:
            self._current_color = random.choice(self.color_list)
            self._apply_color_material(self._current_color)
        else:  # color_list is None
            # Check for MDL support
            self.visual_material_mdl_path = self.inst_visual_material_cfg.get(
                "mdl_path"
            )
            self.visual_material_mdl_folder = self.inst_visual_material_cfg.get(
                "mdl_folder"
            )

            if self.visual_material_mdl_path:
                self._apply_mdl_material(
                    mdl_path=self.visual_material_mdl_path,
                    mdl_name=self.inst_visual_material_cfg.get("mdl_name"),
                )
            elif self.visual_material_mdl_folder:
                resolved_mdl_paths = resolve_mdl_paths(self.visual_material_mdl_folder)
                if resolved_mdl_paths:
                    selected_path = random.choice(resolved_mdl_paths)
                    self._apply_mdl_material(mdl_path=selected_path)
                else:
                    print(
                        f"⚠️ Warning: No MDL files found in folder: {self.visual_material_mdl_folder}"
                    )
            else:
                # Fallback to USD material folder
                self.visual_material_usd_folder = self.inst_visual_material_cfg.get(
                    "material_usd_folder",
                    "$MAGICSIM_ASSETS/Material/Garment",  # Default folder
                )
                if self.visual_material_usd_folder is not None:
                    self.visual_usd_paths = get_usd_paths_from_folder(
                        folder_path=self.visual_material_usd_folder,
                        skip_keywords=[".thumbs"],
                    )
                    if self.visual_usd_paths:
                        # Select and apply random material from folder
                        selected_indices = torch.randint(
                            low=0,
                            high=len(self.visual_usd_paths),
                            size=(1,),
                        ).tolist()
                        self.visual_usd_path = self.visual_usd_paths[
                            selected_indices[0]
                        ]
                        self._apply_visual_material(self.visual_usd_path)
                    else:
                        print(
                            f"Warning: No material USDs found in {self.visual_material_usd_folder}"
                        )

        self._set_contact_offset(self.inst_deformable_cfg.get("contact_offset", 0.01))
        self._set_rest_offset(self.inst_deformable_cfg.get("rest_offset", 0.008))

        # Set initial pose
        self.set_world_pose(position=self.init_pos, orientation=self.init_ori)

        # Handle semantic labels
        self._handle_semantic_labels()

    def _apply_color_material(self, color):
        """Creates or updates the PreviewSurface material with the specified color."""
        material_path = find_unique_string_name(
            initial_name=f"{self.usd_prim_path}/Looks/color_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        material_prim = get_prim_at_path(material_path)
        if not material_prim:
            material = PreviewSurface(
                prim_path=material_path, color=torch.tensor(color)
            )
            # Bind the new material to the geometry prim
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.mesh_prim_path,
                material_path=material_path,
                strength=UsdShade.Tokens.strongerThanDescendants,
            )
        else:
            material = PreviewSurface(prim_path=material_path)
            material.set_color(np.array(color))

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

    def _find_first_mesh_in_hierarchy(self, prim_path: str) -> str:
        """Recursively searches for the first prim of type UsdGeom.Mesh under the given path."""
        start_prim = get_prim_at_path(prim_path)
        if not start_prim:
            return None

        for prim in Usd.PrimRange(start_prim):
            if prim.IsA(UsdGeom.Mesh):
                return prim.GetPath().pathString

        return None

    def _re_instance_name(self, inst_name):
        """Reformats the instance name to ensure consistent numbering."""
        parts = inst_name.split("_")
        cat_name_extracted = "_".join(parts[:-1])
        obj_id_str = parts[-1]
        obj_id = int(obj_id_str)
        original_id = (obj_id - 1) % self.num_per_env + 1
        return f"{cat_name_extracted}_{original_id}"

    def initialize(self):
        """
        Initialize the deformable object by capturing initial particle information.
        """
        self._get_initial_info()

    def reset(self, soft=False):
        """
        Perform reset by restoring initial particle positions and setting new pose using LayoutManager.

        Args:
            soft: If True, use soft reset ranges; otherwise use initial ranges
        """
        if self._device == "cpu":
            if isinstance(self.initial_points_positions, torch.Tensor):
                initial_pos_np = self.initial_points_positions.cpu().numpy()
            else:
                initial_pos_np = self.initial_points_positions
            self._prim.GetAttribute("points").Set(
                Vt.Vec3fArray.FromNumpy(initial_pos_np)
            )
        else:
            if hasattr(self, "_deformable_prim_view") and self._deformable_prim_view:
                if isinstance(self.initial_points_positions, np.ndarray):
                    initial_pos_tensor = torch.from_numpy(
                        self.initial_points_positions
                    ).to(self._device)
                else:
                    initial_pos_tensor = self.initial_points_positions.to(self._device)
                expected_shape = self._deformable_prim_view.get_simulation_mesh_nodal_positions().shape
                if initial_pos_tensor.shape != expected_shape:
                    if len(expected_shape) == 3 and initial_pos_tensor.ndim == 2:
                        initial_pos_tensor = initial_pos_tensor.unsqueeze(0)
                    if initial_pos_tensor.shape != expected_shape:
                        print(
                            f"Warning/Error: Shape mismatch for {self.name} reset. Expected {expected_shape}, got {initial_pos_tensor.shape}. Skipping particle reset."
                        )
                        initial_pos_tensor = None

                if initial_pos_tensor is not None:
                    try:
                        self._deformable_prim_view.set_simulation_mesh_nodal_positions(
                            initial_pos_tensor
                        )
                    except Exception as e:
                        print(
                            f"Error setting simulation mesh nodal positions in reset: {e}"
                        )
                        print(f"  Expected shape: {expected_shape}")
                        print(f"  Provided tensor shape: {initial_pos_tensor.shape}")
            else:
                print(
                    f"Warning: _deformable_prim_view not initialized for {self.name} on device {self._device}. Skipping particle position reset."
                )

        if not self.layout_manager:
            raise RuntimeError(
                f"LayoutManager is required for {self.usd_prim_path}. All position information must come from LayoutManager."
            )

        env_id = self._extract_env_id_from_prim_path()
        if env_id is None:
            print(
                f"Warning: Could not extract env_id for {self.usd_prim_path}. Cannot perform reset."
            )
            return

        reset_type = "soft" if soft else "hard"
        new_layout = self.layout_manager.generate_new_layout(
            env_id=env_id, prim_path=self.usd_prim_path, reset_type=reset_type
        )

        if new_layout:
            position = new_layout["pos"]
            orientation = new_layout["ori"]
            scale = new_layout["scale"]

            position = np.array(position, dtype=np.float32) + self.env_origin
            position[2] += random.uniform(-0.0001, 0.0001)
            self.set_world_pose(position, orientation)
            if hasattr(self, "set_local_scale"):
                self.set_local_scale(scale)
            return
        else:
            print(
                f"Warning: LayoutManager did not provide new layout for {self.usd_prim_path}. Cannot perform reset."
            )
            return

    def get_current_mesh_points(
        self, visualize=False, save=False, save_path="./pointcloud.ply"
    ):
        """
        Get the current mesh points of the deformable object.

        Args:
            visualize: Whether to visualize the mesh points using Open3D
            save: Whether to save the mesh points to a file
            save_path: Path to save the point cloud if save=True

        Returns:
            transformed_points: Mesh points in world space
            mesh_points: Original mesh points in local space
            pos_world: World position (CPU only)
            ori_world: World orientation (CPU only)
        """
        if self._device == "cpu":
            pos_world, ori_world = self.get_world_pose()
            scale_world = self.get_world_scale()
            mesh_points = self._get_points_pose().detach().cpu().numpy()
            transformed_mesh_points = self.transform_points(
                mesh_points,
                pos_world.detach().cpu().numpy(),
                ori_world.detach().cpu().numpy(),
                scale_world.detach().cpu().numpy(),
            )
        else:
            mesh_points = (
                self._deformable_prim_view.get_simulation_mesh_nodal_positions()
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            transformed_mesh_points = mesh_points
            pos_world = None
            ori_world = None

        # Visualization and saving
        if visualize or save:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(transformed_mesh_points)

            if visualize:
                o3d.visualization.draw_geometries([pcd])
            if save:
                o3d.io.write_point_cloud(save_path, pcd)

        return transformed_mesh_points, mesh_points, pos_world, ori_world

    def set_current_mesh_points(self, mesh_points, pos_world=None, ori_world=None):
        """
        Set the current mesh points of the deformable object.

        Args:
            mesh_points: Original mesh points in local space
            pos_world: World position (required for CPU device)
            ori_world: World orientation (required for CPU device)
        """
        if self._device == "cpu":
            if pos_world is None or ori_world is None:
                raise ValueError(
                    "pos_world and ori_world must be provided for CPU device"
                )
            self._prim.GetAttribute("points").Set(Vt.Vec3fArray.FromNumpy(mesh_points))
            self.set_world_pose(pos_world, ori_world)
        else:
            mesh_points_tensor = (
                torch.from_numpy(mesh_points).to(self._device).unsqueeze(0)
            )
            if hasattr(self, "_deformable_prim_view") and self._deformable_prim_view:
                expected_shape = self._deformable_prim_view.get_simulation_mesh_nodal_positions().shape
                if mesh_points_tensor.shape != expected_shape:
                    if len(expected_shape) == 3 and mesh_points_tensor.ndim == 2:
                        mesh_points_tensor = mesh_points_tensor.unsqueeze(0)
                    if mesh_points_tensor.shape != expected_shape:
                        print(
                            f"Warning/Error: Shape mismatch for {self.name} set_current_mesh_points. Expected {expected_shape}, got {mesh_points_tensor.shape}. Skipping."
                        )
                        mesh_points_tensor = None

                if mesh_points_tensor is not None:
                    try:
                        self._deformable_prim_view.set_simulation_mesh_nodal_positions(
                            mesh_points_tensor
                        )
                    except Exception as e:
                        print(
                            f"Error setting simulation mesh nodal positions in set_current_mesh_points: {e}"
                        )
            else:
                print(
                    f"Warning: _deformable_prim_view not initialized for {self.name} on device {self._device}. Skipping set_current_mesh_points."
                )

    def _setup_deformable_material(self):
        """Configure the deformable material properties from the configuration."""
        inst_deformable_material_cfg = self.physics_cfg.get("deformable_material", {})

        self.deformable_material_prim_path = find_unique_string_name(
            self.usd_prim_path + "/deformable_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )

        self.deformable_material = DeformableMaterial(
            prim_path=self.deformable_material_prim_path,
            damping_scale=inst_deformable_material_cfg.get("damping_scale", 0.15),
            dynamic_friction=inst_deformable_material_cfg.get("dynamic_friction", 1.0),
            elasticity_damping=inst_deformable_material_cfg.get(
                "elasticity_damping", 0.0
            ),
            poissons_ratio=inst_deformable_material_cfg.get("poissons_ratio", 0.0),
            youngs_modulus=inst_deformable_material_cfg.get("youngs_modulus", 1e8),
        )

    def _set_contact_offset(self, contact_offset: float = 0.01):
        """Set the collision contact offset."""
        self.collsionapi = PhysxSchema.PhysxCollisionAPI.Apply(self.prim)
        self.collsionapi.GetContactOffsetAttr().Set(contact_offset)

    def _set_rest_offset(self, rest_offset: float = 0.008):
        """Set the collision rest offset."""
        self.collsionapi = PhysxSchema.PhysxCollisionAPI.Apply(self.prim)
        self.collsionapi.GetRestOffsetAttr().Set(rest_offset)

    def _apply_visual_material(self, material_path: str):
        """Apply a visual material to the deformable mesh."""

        self.visual_material_prim_path = find_unique_string_name(
            self.usd_prim_path + "/Looks/visual_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )

        add_reference_to_stage(
            usd_path=material_path, prim_path=self.visual_material_prim_path
        )

        self.visual_material_prim = prims_utils.get_prim_at_path(
            self.visual_material_prim_path
        )
        if not self.visual_material_prim or not self.visual_material_prim.IsValid():
            print(
                f"Warning: Could not get valid prim at {self.visual_material_prim_path}"
            )
            return
        children = prims_utils.get_prim_children(self.visual_material_prim)
        if not children:
            print(
                f"Warning: Material prim at {self.visual_material_prim_path} has no children."
            )
            return

        self.material_prim = children[0]
        self.material_prim_path = self.material_prim.GetPath()
        self.visual_material = PreviewSurface(self.material_prim_path)

        self.deformable_mesh_prim = prims_utils.get_prim_at_path(self.mesh_prim_path)
        if not self.deformable_mesh_prim or not self.deformable_mesh_prim.IsValid():
            print(
                f"Warning: Could not find mesh prim at {self.mesh_prim_path} to bind material."
            )
            return
        self.deformable_submesh = prims_utils.get_prim_children(
            self.deformable_mesh_prim
        )

        # Apply material to main mesh and submeshes with strongerThanDescendants
        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=self.mesh_prim_path,
            material_path=self.material_prim_path,
            strength=UsdShade.Tokens.strongerThanDescendants,
        )

        for prim in self.deformable_submesh:
            if prim.IsA(UsdGeom.Gprim):
                omni.kit.commands.execute(
                    "BindMaterialCommand",
                    prim_path=prim.GetPath(),
                    material_path=self.material_prim_path,
                    strength=UsdShade.Tokens.strongerThanDescendants,
                )

    def _apply_mdl_material(self, mdl_path: str, mdl_name: str = None):
        """Apply an MDL material to the deformable object.

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

            # Bind material to the deformable mesh
            self.deformable_mesh_prim = prims_utils.get_prim_at_path(
                self.mesh_prim_path
            )
            if not self.deformable_mesh_prim or not self.deformable_mesh_prim.IsValid():
                print(
                    f"Warning: Could not find mesh prim at {self.mesh_prim_path} to bind material."
                )
                return

            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.mesh_prim_path,
                material_path=material_path,
                strength=UsdShade.Tokens.strongerThanDescendants,
            )

            # Bind material to submeshes if any
            self.deformable_submesh = prims_utils.get_prim_children(
                self.deformable_mesh_prim
            )
            for prim in self.deformable_submesh:
                if prim.IsA(UsdGeom.Gprim):
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

    def _extract_env_id_from_prim_path(self):
        """get env_id from prim_path"""
        try:
            parts = self.usd_prim_path.split("/")
            for part in parts:
                if part.startswith("env_"):
                    return int(part.split("_")[1])
        except (ValueError, IndexError):
            pass
        return None

    def destroy(self):
        """
        Deactivates the deformable object by moving it to a far-away location
        and disabling its visibility.
        """
        try:
            far_away_pos = np.array([100.0, 100.0, 100.0])
            self.set_world_pose(position=far_away_pos)
            self.hide_prim(self.usd_prim_path)

        except Exception as e:
            print(
                f"Warning: Failed to destroy/move deformable prim {self.usd_prim_path}: {e}"
            )

    def _get_initial_info(self):
        """Capture initial particle positions for reset functionality."""
        if self._device == "cpu":
            self.initial_points_positions = (
                self._get_points_pose().detach().cpu().numpy()
            )
        else:
            self.physics_sim_view = SimulationManager.get_physics_sim_view()
            self._deformable_prim_view.initialize(self.physics_sim_view)
            self.initial_points_positions = (
                self._deformable_prim_view.get_simulation_mesh_nodal_positions()
            )

    def transform_points(self, points, pos, ori, scale):
        """
        Transform local points to world space using position, orientation, and scale.

        Args:
            points: (N, 3) array of local points
            pos: (3,) position vector
            ori: (4,) quaternion orientation
            scale: Scale factor

        Returns:
            (N, 3) array of transformed points in world space
        """
        ori_matrix = quat_to_rot_matrix(ori)
        scaled_points = points * scale
        transformed_points = scaled_points @ ori_matrix.T + pos
        return transformed_points

    def inverse_transform_points(self, transformed_points, pos, ori, scale):
        """
        Transform world space points back to local space.

        Args:
            transformed_points: (N, 3) array of world space points
            pos: (3,) position vector
            ori: (4,) quaternion orientation
            scale: Scale factor

        Returns:
            (N, 3) array of points in local space
        """
        ori_matrix = quat_to_rot_matrix(ori)
        shifted_points = transformed_points - pos
        rotated_points = shifted_points @ ori_matrix
        original_points = rotated_points / scale
        return original_points

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

        # Randomize deformable_config parameters
        if "deformable_config" in modified_config:
            deformable_config = modified_config["deformable_config"].copy()
            deformable_params_to_randomize = [
                "vertex_velocity_damping",
                "sleep_damping",
                "sleep_threshold",
                "settling_threshold",
                "solver_position_iteration_count",
                "contact_offset",
                "rest_offset",
            ]

            for param in deformable_params_to_randomize:
                if param in deformable_config and deformable_config[param] is not None:
                    original_value = deformable_config[param]
                    if isinstance(original_value, (int, float)):
                        variation = original_value * (ratio - 1)
                        min_val = original_value - variation
                        max_val = original_value + variation
                        deformable_config[param] = random.uniform(min_val, max_val)

            modified_config["deformable_config"] = deformable_config

        # Randomize deformable_material parameters
        if "deformable_material" in modified_config:
            deformable_material = modified_config["deformable_material"].copy()
            material_params_to_randomize = [
                "damping_scale",
                "dynamic_friction",
                "elasticity_damping",
                "poissons_ratio",
                "youngs_modulus",
            ]

            for param in material_params_to_randomize:
                if (
                    param in deformable_material
                    and deformable_material[param] is not None
                ):
                    original_value = deformable_material[param]
                    if isinstance(original_value, (int, float)):
                        variation = original_value * (ratio - 1)
                        min_val = original_value - variation
                        max_val = original_value + variation
                        deformable_material[param] = random.uniform(min_val, max_val)

            modified_config["deformable_material"] = deformable_material

        return modified_config

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

    def get_state(self, is_relative: bool = False) -> dict[str, torch.Tensor]:
        """Get the state of the deformable object.

        Args:
            is_relative: If True, positions are relative to environment origin. Defaults to False.

        Returns:
            Dictionary containing:
                - root_pose: torch.Tensor, shape (7,), root pose [pos(3), quat(4)]
                - root_velocity: torch.Tensor, shape (6,), root velocity [lin_vel(3), ang_vel(3)]
                - asset_info: dict with usd_path and primitive_type
        """
        # Get root pose using get_world_pose() (center of mass position)
        try:
            pos_world, ori_world = self.get_world_pose()

            # Convert to tensors
            if isinstance(pos_world, np.ndarray):
                pos_tensor = torch.tensor(pos_world, dtype=torch.float32)
            elif isinstance(pos_world, torch.Tensor):
                pos_tensor = pos_world.clone().detach().to(dtype=torch.float32)
            else:
                pos_tensor = torch.tensor(pos_world, dtype=torch.float32)

            if isinstance(ori_world, np.ndarray):
                ori_tensor = torch.tensor(ori_world, dtype=torch.float32)
            elif isinstance(ori_world, torch.Tensor):
                ori_tensor = ori_world.clone().detach().to(dtype=torch.float32)
            else:
                ori_tensor = torch.tensor(ori_world, dtype=torch.float32)

            # Ensure correct shapes
            if pos_tensor.ndim == 0:
                pos_tensor = pos_tensor.unsqueeze(0)
            if pos_tensor.shape[0] < 3:
                pos_tensor = torch.cat(
                    [
                        pos_tensor,
                        torch.zeros(3 - pos_tensor.shape[0], dtype=torch.float32),
                    ]
                )
            pos_tensor = pos_tensor[:3]  # Take only first 3 elements

            if ori_tensor.ndim == 0:
                ori_tensor = ori_tensor.unsqueeze(0)
            if ori_tensor.shape[0] < 4:
                ori_tensor = torch.cat(
                    [
                        ori_tensor,
                        torch.zeros(4 - ori_tensor.shape[0], dtype=torch.float32),
                    ]
                )
            ori_tensor = ori_tensor[:4]  # Take only first 4 elements

            # Combine position and orientation into root_pose [pos(3), quat(4)]
            root_pose = torch.cat([pos_tensor, ori_tensor])

            # Apply relative transformation if needed
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
                                dtype=torch.float32,
                            ),
                        ]
                    )
                root_pose[:3] -= env_origin_tensor[:3]
        except (AttributeError, RuntimeError, Exception):
            # If get_world_pose fails, return zeros
            root_pose = torch.zeros(7, dtype=torch.float32)

        # Get root velocity (mean of nodal velocities if available, otherwise zeros)
        try:
            if (
                hasattr(self, "_deformable_prim_view")
                and self._deformable_prim_view is not None
            ):
                # Try to get nodal velocities and compute mean
                nodal_velocity = (
                    self._deformable_prim_view.get_simulation_mesh_nodal_velocities()
                )
                if not isinstance(nodal_velocity, torch.Tensor):
                    nodal_velocity = torch.tensor(nodal_velocity, dtype=torch.float32)

                # Handle batched tensor
                if nodal_velocity.ndim == 3 and nodal_velocity.shape[0] == 1:
                    nodal_velocity = nodal_velocity.squeeze(0)
                elif nodal_velocity.ndim == 3 and nodal_velocity.shape[0] > 1:
                    nodal_velocity = nodal_velocity[0]

                # Compute mean velocity (root velocity)
                if nodal_velocity.numel() > 0 and nodal_velocity.shape[0] > 0:
                    root_lin_vel = torch.mean(
                        nodal_velocity, dim=0
                    )  # Mean of all nodes
                    if root_lin_vel.shape[0] < 3:
                        root_lin_vel = torch.cat(
                            [
                                root_lin_vel,
                                torch.zeros(
                                    3 - root_lin_vel.shape[0], dtype=torch.float32
                                ),
                            ]
                        )
                    root_lin_vel = root_lin_vel[:3]
                else:
                    root_lin_vel = torch.zeros(3, dtype=torch.float32)
            else:
                root_lin_vel = torch.zeros(3, dtype=torch.float32)

            # Angular velocity is typically zero for deformable objects (no rigid body rotation)
            root_ang_vel = torch.zeros(3, dtype=torch.float32)

            # Combine linear and angular velocity [lin_vel(3), ang_vel(3)]
            root_velocity = torch.cat([root_lin_vel, root_ang_vel])
        except (AttributeError, RuntimeError, Exception):
            # If velocity calculation fails, return zeros
            root_velocity = torch.zeros(6, dtype=torch.float32)

        asset_info = {
            "usd_path": self.usd_path if hasattr(self, "usd_path") else None,
            "primitive_type": self.primitive_type
            if hasattr(self, "primitive_type")
            else None,
        }

        return {
            "root_pose": root_pose,
            "root_velocity": root_velocity,
            "asset_info": asset_info,
        }
