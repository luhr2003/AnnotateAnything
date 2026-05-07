import torch
import numpy as np
import random
import re
import os
import omni.kit.commands
import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.prims import SingleClothPrim, SingleParticleSystem
from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
from isaacsim.core.api.materials.particle_material import ParticleMaterial
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.semantics import add_labels, remove_labels
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Vt, Usd, UsdGeom, UsdShade
from magicsim.Env.Utils.path import (
    get_usd_paths_from_folder,
    resolve_mdl_paths,
    resolve_path,
)
from omegaconf import DictConfig
from isaacsim.replicator.behavior.utils.scene_utils import create_mdl_material


class GarmentObject(SingleClothPrim):
    """
    GarmentObject class that wraps the Isaac Sim SingleCloth prim functionality.
    This class inherits from the Isaac Sim SingleClothPrim class and can be extended
    to add custom garment-specific behaviors.
    Supports semantic labeling for object identification and scene understanding.
    """

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        config: DictConfig,
        env_origin: torch.Tensor,
        primitive_type: str = None,
        layout_manager=None,
        layout_info=None,
    ):
        """
        Initialize the GarmentObject with position, orientation, and configuration.

        Args:
            prim_path: Path to the prim in the stage
            usd_path: Path to the USD asset file for this object
            config: Configuration dictionary containing object properties
            env_origin: Origin position of the environment
            primitive_type: Type of primitive (only 'Plane' is supported)
        """
        if primitive_type is not None and primitive_type != "Plane":
            raise ValueError(
                f"GarmentObject '{prim_path}' only supports the 'Plane' primitive type, "
                f"but received '{primitive_type}'"
            )

        # Parse prim path components
        prim_path_parts = prim_path.split("/")
        self.category_name = prim_path_parts[-2]
        self.instance_name = prim_path_parts[-1]
        self.env_name = prim_path_parts[-4]
        self.num_per_env = config.objects[self.category_name].get("num_per_env")
        self.instance_name = self._re_instance_name(self.instance_name)

        # Configuration setup
        self.primitive_type = primitive_type
        self.global_config = config
        self.category_config = config.objects[self.category_name]
        self.instance_config = self.category_config.get(self.instance_name, {})
        self.stage = get_current_stage()
        self._current_color = None

        # Common configurations
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

        # Visual and physics configurations
        self.visual_cfg = self.instance_config.get("visual", {})
        self.physics_cfg = self.instance_config.get("physics", {})

        # Apply ratio-based randomization to physics parameters
        self.physics_cfg = self._apply_physics_ratio_randomization(self.physics_cfg)

        # Component-specific configurations
        self.inst_garment_cfg = self.physics_cfg.get("garment_config", {})
        self.inst_particle_material_cfg = self.physics_cfg.get("particle_material", {})
        self.inst_particle_system_cfg = self.physics_cfg.get("particle_system", {})
        self.inst_visual_material_cfg = self.visual_cfg.get("visual_material", {})

        # USD path configurations
        self.usd_prim_path = prim_path
        self.usd_path = usd_path
        self.prim_name = prim_path.split("/")[-1]
        self.config = config
        self.objects_config = config.get("objects")
        self.layout_manager = layout_manager

        # Initial pose calculation
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

        if usd_path:
            add_reference_to_stage(usd_path=usd_path, prim_path=self.usd_prim_path)
            self.mesh_prim_path = self._find_first_mesh_in_hierarchy(self.usd_prim_path)
            if self.mesh_prim_path is None:
                raise RuntimeError(
                    f"Could not find a UsdGeom.Mesh prim under the referenced asset at {self.usd_prim_path}"
                )
        else:
            self.mesh_prim_path = self.usd_prim_path

        # Setup particle system
        interaction_flag = self.category_config.get("interaction_with_fluid", False)
        if interaction_flag:
            self.particle_system_path = (
                f"/Particle_Attribute/{self.env_name}/particle_system"
            )
        else:
            self.particle_system_path = (
                f"/Particle_Attribute/{self.env_name}/garment_particle_system"
            )

        # Initialize or reuse particle system
        if is_prim_path_valid(self.particle_system_path):
            self.particle_system = SingleParticleSystem(
                prim_path=self.particle_system_path
            )
        else:
            self.particle_system = SingleParticleSystem(
                prim_path=self.particle_system_path,
                particle_system_enabled=self.inst_particle_system_cfg.get(
                    "particle_system_enabled", True
                ),
                enable_ccd=self.inst_particle_system_cfg.get("enable_ccd", True),
                solver_position_iteration_count=self.inst_particle_system_cfg.get(
                    "solver_position_iteration_count", 16
                ),
                max_depenetration_velocity=self.inst_particle_system_cfg.get(
                    "max_depenetration_velocity", None
                ),
                global_self_collision_enabled=self.inst_particle_system_cfg.get(
                    "global_self_collision_enabled", True
                ),
                non_particle_collision_enabled=self.inst_particle_system_cfg.get(
                    "non_particle_collision_enabled", True
                ),
                contact_offset=self.inst_particle_system_cfg.get(
                    "contact_offset", 0.01
                ),
                rest_offset=self.inst_particle_system_cfg.get("rest_offset", 0.0075),
                particle_contact_offset=self.inst_particle_system_cfg.get(
                    "particle_contact_offset", 0.01
                ),
                fluid_rest_offset=self.inst_particle_system_cfg.get(
                    "fluid_rest_offset", 0.0075
                ),
                solid_rest_offset=self.inst_particle_system_cfg.get(
                    "solid_rest_offset", 0.0075
                ),
                wind=self.inst_particle_system_cfg.get("wind", None),
                max_neighborhood=self.inst_particle_system_cfg.get(
                    "max_neighborhood", None
                ),
                max_velocity=self.inst_particle_system_cfg.get("max_velocity", None),
            )

        # Setup particle material
        self.particle_material_path = find_unique_string_name(
            self.usd_prim_path + "/particle_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        self.particle_material = ParticleMaterial(
            prim_path=self.particle_material_path,
            adhesion=self.inst_particle_material_cfg.get("adhesion", 0.1),
            adhesion_offset_scale=self.inst_particle_material_cfg.get(
                "adhesion_offset_scale", 0.0
            ),
            cohesion=self.inst_particle_material_cfg.get("cohesion", 0.0),
            particle_adhesion_scale=self.inst_particle_material_cfg.get(
                "particle_adhesion_scale", 0.5
            ),
            particle_friction_scale=self.inst_particle_material_cfg.get(
                "particle_friction_scale", 0.5
            ),
            drag=self.inst_particle_material_cfg.get("drag", 0.0),
            lift=self.inst_particle_material_cfg.get("lift", 0.0),
            friction=self.inst_particle_material_cfg.get("friction", 10.0),
            damping=self.inst_particle_material_cfg.get("damping", 0.0),
            gravity_scale=self.inst_particle_material_cfg.get("gravity_scale", 1.0),
            viscosity=self.inst_particle_material_cfg.get("viscosity", None),
            vorticity_confinement=self.inst_particle_material_cfg.get(
                "vorticity_confinement", None
            ),
            surface_tension=self.inst_particle_material_cfg.get(
                "surface_tension", None
            ),
        )

        # Initialize parent class
        super().__init__(
            name=self.usd_prim_path,
            scale=self.init_scale,
            prim_path=self.mesh_prim_path,
            particle_system=self.particle_system,
            particle_material=self.particle_material,
            particle_mass=self.inst_garment_cfg.get("particle_mass", 1e-2),
            self_collision=self.inst_garment_cfg.get("self_collision", True),
            self_collision_filter=self.inst_garment_cfg.get(
                "self_collision_filter", True
            ),
            stretch_stiffness=self.inst_garment_cfg.get("stretch_stiffness", 1e8),
            bend_stiffness=self.inst_garment_cfg.get("bend_stiffness", 1000.0),
            shear_stiffness=self.inst_garment_cfg.get("shear_stiffness", 1000.0),
            spring_damping=self.inst_garment_cfg.get("spring_damping", 10.0),
        )

        # --- Visual Material and Visibility Setup ---
        # Set visibility based on config. self.prim is available from SingleClothPrim.
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

        default_material = self.category_config.get("default_material")
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
                    "material_usd_folder", "$MAGICSIM_ASSETS/Material/Garment"
                )
                if self.visual_material_usd_folder is not None:
                    self.visual_usd_paths = get_usd_paths_from_folder(
                        folder_path=self.visual_material_usd_folder,
                        skip_keywords=[".thumbs"],
                    )
                    if self.visual_usd_paths:
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
                            f"⚠️ Warning: No visual materials found in '{self.visual_material_usd_folder}'. Skipping material application for {self.usd_prim_path}."
                        )
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
                else:
                    # No color, MDL or USD visual_material folder
                    # Check default_material setting
                    if default_material is True:
                        # Apply default material when default_material is True
                        self._apply_default_material(material_name="default_material")
                    elif default_material is False:
                        # Do nothing - use USD's own material when default_material is False
                        pass
                    # If default_material is None/not set, maintain original behavior (no material binding)

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
            # Bind the new material to the geometry prim and its submeshes
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.mesh_prim_path,  # Bind to the mesh prim
                material_path=material_path,
                strength=UsdShade.Tokens.strongerThanDescendants,
            )
            # Also bind to submeshes if they exist
            mesh_prim_to_bind = prims_utils.get_prim_at_path(self.mesh_prim_path)
            if mesh_prim_to_bind:
                garment_submesh = prims_utils.get_prim_children(mesh_prim_to_bind)
                if len(garment_submesh) > 0:
                    for sub_prim in garment_submesh:
                        if sub_prim.IsA(UsdGeom.Gprim):
                            omni.kit.commands.execute(
                                "BindMaterialCommand",
                                prim_path=sub_prim.GetPath(),
                                material_path=material_path,
                                strength=UsdShade.Tokens.strongerThanDescendants,
                            )
        else:
            material = PreviewSurface(prim_path=material_path)
            material.set_color(np.array(color))

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
        Initialize the object by capturing initial particle information
        and setting up initial state.
        """
        self._get_initial_info()

    def _detect_garment_type(self) -> str:
        """Detect garment type from USD path or primitive type.

        Returns one of: 'tops', 'dress', 'pants', 'plane', 'unknown'.
        """
        if self.primitive_type == "Plane":
            return "plane"
        if self.usd_path:
            path_lower = self.usd_path.lower()
            if "/tops/" in path_lower:
                return "tops"
            elif "/dress/" in path_lower:
                return "dress"
            elif "/pants/" in path_lower:
                return "pants"
        return "unknown"

    def _to_canonical_frame(self, mesh_points: np.ndarray) -> np.ndarray:
        """Transform world-space mesh points to canonical frame.

        Centers the points and undoes the init_ori rotation so that the
        garment is in its original USD-authored orientation.
        """
        centroid = mesh_points.mean(axis=0)
        centered = mesh_points - centroid

        ori = np.array(self.init_ori, dtype=np.float32)
        inv_ori = np.array([ori[0], -ori[1], -ori[2], -ori[3]], dtype=np.float32)
        rot_mat = quat_to_rot_matrix(inv_ori)
        rot_mat = np.asarray(rot_mat)
        if rot_mat.shape == (4, 4):
            rot_mat = rot_mat[:3, :3]

        canonical = (rot_mat @ centered.T).T
        # Swap X and Y so that X=sleeve direction, Y=body direction
        # Negate new Y so neckline at +Y, bottom at -Y (visually correct: head up)
        canonical = canonical[:, [1, 0, 2]]
        canonical[:, 1] = -canonical[:, 1]
        return canonical

    def _compute_keypoint_indices(self, mesh_points: np.ndarray) -> dict:
        """Run the type-specific algorithm and return {name: vertex_index}."""
        garment_type = self._detect_garment_type()
        if garment_type == "plane":
            return self._keypoint_plane(mesh_points)
        canonical = self._to_canonical_frame(mesh_points)
        if garment_type in ("tops", "dress"):
            return self._keypoint_upper_body(mesh_points, canonical)
        if garment_type == "pants":
            return self._keypoint_pants(mesh_points, canonical)
        return {}

    def get_keypoint(self) -> dict:
        """Compute keypoint positions from current mesh vertices.

        Positions are expressed in the local env frame (world position minus
        ``self.env_origin``) so downstream task code can stay env-agnostic.
        Reuses cached vertex indices from ``update_keypoint`` when available
        to avoid re-running the per-frame keypoint detection algorithm.

        Returns:
            Dict mapping keypoint name -> (3,) numpy array of env-local position.
        """
        mesh_points, _, _, _ = self.get_current_mesh_points()
        if mesh_points is None or len(mesh_points) == 0:
            return {}
        indices = getattr(self, "_keypoint_indices", None)
        if not indices:
            indices = self._compute_keypoint_indices(mesh_points)
            self._keypoint_indices = indices
        env_origin = np.asarray(self.env_origin, dtype=np.float32)
        return {name: (mesh_points[idx] - env_origin) for name, idx in indices.items()}

    def update_keypoint(self) -> dict:
        """Recompute and cache keypoint vertex indices from the current mesh.

        The cached indices are reused by ``visualize_keypoint`` so that the
        spheres track the same vertices across simulation steps.

        Returns:
            The cached {name: vertex_index} dict.
        """
        mesh_points, _, _, _ = self.get_current_mesh_points()
        if mesh_points is None or len(mesh_points) == 0:
            self._keypoint_indices = {}
        else:
            self._keypoint_indices = self._compute_keypoint_indices(mesh_points)
        return self._keypoint_indices

    def visualize_keypoint(
        self,
        radius: float = 0.02,
        color: tuple = (1.0, 0.0, 0.0),
    ) -> None:
        """Create or refresh VisualSphere markers at cached keypoint vertices.

        Uses the indices cached by ``update_keypoint`` and the latest vertex
        positions. Spheres live under an independent ``/debug_keypoints`` root
        (outside the garment prim) so they do not interfere with cloth physics
        or get driven by the garment's transform.
        """
        from isaacsim.core.api.objects import VisualSphere

        indices = getattr(self, "_keypoint_indices", None)
        if not indices:
            print(
                f"[{self.usd_prim_path}] visualize_keypoint: no cached indices, "
                "call update_keypoint() first"
            )
            return

        mesh_points, _, _, _ = self.get_current_mesh_points()
        if mesh_points is None or len(mesh_points) == 0:
            return

        if not hasattr(self, "_keypoint_spheres"):
            self._keypoint_spheres = {}

        color_arr = np.array(color, dtype=np.float32)
        kp_root = f"/debug_keypoints/{self.env_name}/{self.category_name}/{self.instance_name}"

        for name, idx in indices.items():
            pos = mesh_points[int(idx)]
            world_pos = np.array(
                [float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32
            )
            sphere_path = f"{kp_root}/kp_{name}"
            sphere = self._keypoint_spheres.get(name)
            if sphere is None:
                sphere = VisualSphere(
                    prim_path=sphere_path,
                    name=f"{self.env_name}_{self.category_name}_{self.instance_name}_kp_{name}",
                    radius=radius,
                    color=color_arr,
                )
                self._keypoint_spheres[name] = sphere
            sphere.set_world_pose(position=world_pos)

    def visualize_canonical_frame(self, save_path="./canonical_frame.ply"):
        """Visualize the canonical frame points with Open3D for debugging."""
        import open3d as o3d

        mesh_points, _, _, _ = self.get_current_mesh_points()
        if mesh_points is None or len(mesh_points) == 0:
            print("No mesh points available")
            return

        canonical = self._to_canonical_frame(mesh_points)

        print("Canonical frame stats:")
        print(f"  X range: [{canonical[:, 0].min():.4f}, {canonical[:, 0].max():.4f}]")
        print(f"  Y range: [{canonical[:, 1].min():.4f}, {canonical[:, 1].max():.4f}]")
        print(f"  Z range: [{canonical[:, 2].min():.4f}, {canonical[:, 2].max():.4f}]")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(canonical)
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"Saved canonical frame to {save_path}")
        o3d.visualization.draw_geometries([pcd])

    def _keypoint_upper_body(
        self, mesh_points: np.ndarray, canonical: np.ndarray
    ) -> dict:
        """Keypoints for tops and dress using canonical frame.

        Produces 11 keypoints using the reference get_keypoint_groups algorithm.
        Canonical frame has neckline at +Y, bottom at -Y (visually correct).
        Internally negates Y so bottom=max y, matching the reference convention.
        """
        # Negate Y to match reference convention: bottom at max y, neckline at min y
        pts = canonical.copy()
        pts[:, 1] = -pts[:, 1]

        x = pts[:, 0]
        y = pts[:, 1]

        cloth_height = float(np.max(y) - np.min(y))
        cloth_width = float(np.max(x) - np.min(x))

        # Shoulders via rate of change of garment height
        max_ys, min_ys = [], []
        num_bins = 40
        x_min, x_max = np.min(x), np.max(x)
        mid = (x_min + x_max) / 2
        lin = np.linspace(mid, x_max, num=num_bins)
        for xleft, xright in zip(lin[:-1], lin[1:]):
            mask = np.where((xleft < x) & (x < xright))
            if len(mask[0]) > 0:
                max_ys.append(-1 * y[mask].min())
                min_ys.append(-1 * y[mask].max())
            else:
                max_ys.append(max_ys[-1] if max_ys else 0.0)
                min_ys.append(min_ys[-1] if min_ys else 0.0)

        diff = np.array(max_ys) - np.array(min_ys)
        roc = diff[1:] - diff[:-1]

        begin_offset = num_bins // 5
        end_offset = num_bins // 10
        roc[:begin_offset] = np.max(roc[:begin_offset])
        roc[-end_offset:] = np.max(roc[-end_offset:])

        right_x = (x_max - mid) * (np.argmin(roc) / num_bins) + mid

        # Right shoulder
        pts_copy = pts.copy()
        pts_copy[np.where(np.abs(pts[:, 0] - right_x) > 0.01), 1] = 10
        right_shoulder_idx = int(np.argmin(pts_copy[:, 1]))
        right_shoulder_pos = pts[right_shoulder_idx, :]

        # Left shoulder via mirror
        left_shoulder_query = np.array(
            [-right_shoulder_pos[0], right_shoulder_pos[1], right_shoulder_pos[2]]
        )
        left_shoulder_idx = int(
            np.linalg.norm(pts - left_shoulder_query, axis=1).argmin()
        )

        # Sleeve tips: extreme X in the upper half of the garment
        y_median = np.median(y)
        upper_mask = y < y_median
        upper_indices = np.where(upper_mask)[0]
        top_right_idx = int(upper_indices[np.argmax(x[upper_mask])])
        top_left_idx = int(upper_indices[np.argmin(x[upper_mask])])

        # Bottom corners
        pickpoint_bottom = int(np.argmax(y))
        diff_bottom = pts[pickpoint_bottom, 1] - pts[:, 1]
        idx = diff_bottom < 0.1
        locations = np.where(diff_bottom < 0.1)
        points_near_bottom = pts[idx, :]
        x_bot = points_near_bottom[:, 0]
        y_bot = points_near_bottom[:, 1]
        bottom_right_idx = int(locations[0][np.argmax(x_bot + y_bot)])
        bottom_left_idx = int(locations[0][np.argmax(-x_bot + y_bot)])

        # Middle point: nearest to centroid (0,0) in XY plane
        xy_dist = np.sqrt(x**2 + y**2)
        middle_point_idx = int(np.argmin(xy_dist))

        # Top and bottom points: extreme Y in a band around X=0 (body center)
        x_mid = (np.min(x) + np.max(x)) / 2.0
        x_middle_band_mask = np.abs(x - x_mid) < 0.1 * cloth_width
        x_middle_band_indices = np.where(x_middle_band_mask)[0]
        top_point_idx = int(x_middle_band_indices[np.argmin(y[x_middle_band_mask])])
        bottom_point_idx = int(x_middle_band_indices[np.argmax(y[x_middle_band_mask])])

        return {
            "bottom_left": bottom_left_idx,
            "bottom_right": bottom_right_idx,
            "top_left": top_left_idx,
            "top_right": top_right_idx,
            "left_shoulder": left_shoulder_idx,
            "right_shoulder": right_shoulder_idx,
            "middle_point": middle_point_idx,
            "top_point": top_point_idx,
            "bottom_point": bottom_point_idx,
        }

    def _keypoint_pants(self, mesh_points: np.ndarray, canonical: np.ndarray) -> dict:
        """Keypoints for pants using canonical frame.

        Canonical orientation: X = left-right, Y = waist-to-legs.
        Uses the same corner-scoring algorithm as tops/dress:
        pickpoint + 0.1 band + (±x±y) scoring, plus top_point / bottom_point
        from the x-middle band y extremes.
        """
        pts = canonical
        x = pts[:, 0]
        y = pts[:, 1]

        cloth_width = float(np.max(x) - np.min(x))

        # Bottom corners: pickpoint at max y, band within 0.1, corner scoring
        pickpoint_bottom = int(np.argmax(y))
        diff_bottom = pts[pickpoint_bottom, 1] - pts[:, 1]
        idx_b = diff_bottom < 0.1
        locations_b = np.where(idx_b)
        near_b = pts[idx_b, :]
        xb, yb = near_b[:, 0], near_b[:, 1]
        bottom_right_idx = int(locations_b[0][np.argmax(xb + yb)])
        bottom_left_idx = int(locations_b[0][np.argmax(-xb + yb)])

        # Top corners: pickpoint at min y, band within 0.1, mirrored scoring
        pickpoint_top = int(np.argmin(y))
        diff_top = pts[:, 1] - pts[pickpoint_top, 1]
        idx_t = diff_top < 0.1
        locations_t = np.where(idx_t)
        near_t = pts[idx_t, :]
        xt, yt = near_t[:, 0], near_t[:, 1]
        top_right_idx = int(locations_t[0][np.argmax(xt - yt)])
        top_left_idx = int(locations_t[0][np.argmax(-xt - yt)])

        # Top / bottom points: extreme Y in a band around X=0 (body center)
        x_mid = (np.min(x) + np.max(x)) / 2.0
        x_middle_band_mask = np.abs(x - x_mid) < 0.1 * cloth_width
        x_middle_band_indices = np.where(x_middle_band_mask)[0]
        top_point_idx = int(x_middle_band_indices[np.argmin(y[x_middle_band_mask])])
        bottom_point_idx = int(x_middle_band_indices[np.argmax(y[x_middle_band_mask])])

        return {
            "bottom_left": bottom_left_idx,
            "bottom_right": bottom_right_idx,
            "top_left": top_left_idx,
            "top_right": top_right_idx,
            "top_point": top_point_idx,
            "bottom_point": bottom_point_idx,
        }

    def _keypoint_plane(self, mesh_points: np.ndarray) -> dict:
        """Compute 4 corner keypoints for Plane primitives.

        Uses PCA to find the two principal axes of the plane,
        then corner scoring to find the 4 corners. Rotation-invariant.
        """
        xy = mesh_points[:, :2]
        centroid = xy.mean(axis=0)
        centered = xy - centroid
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        p0 = centered @ vt[0]
        p1 = centered @ vt[1]

        r0 = p0.max() - p0.min()
        r1 = p1.max() - p1.min()
        n0 = (p0 - p0.min()) / max(r0, 1e-8)
        n1 = (p1 - p1.min()) / max(r1, 1e-8)

        return {
            "corner_0": int(np.argmax(n0 + n1)),
            "corner_1": int(np.argmax(n0 + (1.0 - n1))),
            "corner_2": int(np.argmax((1.0 - n0) + n1)),
            "corner_3": int(np.argmax((1.0 - n0) + (1.0 - n1))),
        }

    def reset(self, soft=False):
        """
        Perform reset by restoring initial particle positions and setting new pose using LayoutManager.
        Clears cached keypoint indices so they are recomputed after settling.

        Args:
            soft: If True, use soft reset ranges; otherwise use initial ranges
        """
        # Drop cached keypoint indices so they are recomputed after settling
        self._keypoint_indices = {}

        # Reset particle positions first
        if self._device == "cpu":
            self._prim.GetAttribute("points").Set(
                Vt.Vec3fArray.FromNumpy(self.initial_points_positions)
            )
        else:
            if hasattr(self, "_cloth_prim_view") and self._cloth_prim_view:
                if isinstance(self.initial_points_positions, np.ndarray):
                    initial_pos_tensor = torch.from_numpy(
                        self.initial_points_positions
                    ).to(self._device)
                    if initial_pos_tensor.ndim == 2:
                        initial_pos_tensor = initial_pos_tensor.unsqueeze(0)
                else:
                    initial_pos_tensor = self.initial_points_positions.to(self._device)

                expected_shape_prefix = (
                    self._cloth_prim_view.get_world_positions().shape[:-1]
                )
                if initial_pos_tensor.shape[:-1] != expected_shape_prefix:
                    if len(expected_shape_prefix) == 2 and initial_pos_tensor.ndim == 2:
                        initial_pos_tensor = initial_pos_tensor.unsqueeze(0)
                try:
                    self._cloth_prim_view.set_world_positions(initial_pos_tensor)
                except Exception as e:
                    print(f"Error setting world positions in reset: {e}")
                    print(f"  Expected shape prefix: {expected_shape_prefix}")
                    print(f"  Provided tensor shape: {initial_pos_tensor.shape}")
            else:
                print(
                    f"Warning: _cloth_prim_view not initialized for {self.name} on device {self._device}. Skipping particle position reset."
                )
        if self.layout_manager is not None:
            env_id = self._extract_env_id_from_prim_path()
            if env_id is not None:
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

    def get_current_mesh_points(
        self, visualize=False, save=False, save_path="./pointcloud.ply"
    ):
        """
        Get the current mesh points of the garment.

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
                self._cloth_prim_view.get_world_positions()
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

    def set_current_mesh_points(self, mesh_points, pos_world, ori_world):
        """
        Set the current mesh points of the garment object.

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
            current_mesh_points = (
                torch.from_numpy(mesh_points).to(self._device).unsqueeze(0)
            )
            self._cloth_prim_view.set_world_positions(current_mesh_points)

    def _apply_visual_material(self, material_path: str):
        """Apply a visual material to the garment mesh."""
        self.visual_material_path = find_unique_string_name(
            self.usd_prim_path + "/Looks/visual_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )

        add_reference_to_stage(
            usd_path=material_path, prim_path=self.visual_material_path
        )

        self.visual_material_prim = prims_utils.get_prim_at_path(
            self.visual_material_path
        )
        # Check if visual_material_prim is valid and has children
        if not self.visual_material_prim or not self.visual_material_prim.IsValid():
            print(f"Warning: Could not get valid prim at {self.visual_material_path}")
            return
        children = prims_utils.get_prim_children(self.visual_material_prim)
        if not children:
            print(
                f"Warning: Material prim at {self.visual_material_path} has no children."
            )
            return

        self.material_prim = children[0]
        self.material_prim_path = self.material_prim.GetPath()
        self.visual_material = PreviewSurface(self.material_prim_path)

        mesh_prim_to_bind = prims_utils.get_prim_at_path(self.mesh_prim_path)
        if not mesh_prim_to_bind:
            print(
                f"Warning: Could not find mesh prim at {self.mesh_prim_path} to bind material."
            )
            return

        # Apply material to main mesh with strongerThanDescendants
        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=self.mesh_prim_path,
            material_path=self.material_prim_path,
            strength=UsdShade.Tokens.strongerThanDescendants,
        )

        # Apply material to submeshes if any, also with strongerThanDescendants
        garment_submesh = prims_utils.get_prim_children(mesh_prim_to_bind)
        if len(garment_submesh) > 0:
            for prim in garment_submesh:
                if prim.IsA(UsdGeom.Gprim):
                    omni.kit.commands.execute(
                        "BindMaterialCommand",
                        prim_path=prim.GetPath(),
                        material_path=self.material_prim_path,
                        strength=UsdShade.Tokens.strongerThanDescendants,
                    )

    def _apply_mdl_material(self, mdl_path: str, mdl_name: str = None):
        """Apply an MDL material to the garment object.

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

            # Bind material to the garment mesh
            mesh_prim_to_bind = prims_utils.get_prim_at_path(self.mesh_prim_path)
            if not mesh_prim_to_bind:
                print(
                    f"Warning: Could not find mesh prim at {self.mesh_prim_path} to bind material."
                )
                return

            # Apply material to main mesh with strongerThanDescendants
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.mesh_prim_path,
                material_path=material_path,
                strength=UsdShade.Tokens.strongerThanDescendants,
            )

            # Apply material to submeshes if any, also with strongerThanDescendants
            garment_submesh = prims_utils.get_prim_children(mesh_prim_to_bind)
            if len(garment_submesh) > 0:
                for sub_prim in garment_submesh:
                    if sub_prim.IsA(UsdGeom.Gprim):
                        omni.kit.commands.execute(
                            "BindMaterialCommand",
                            prim_path=sub_prim.GetPath(),
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
        """Creates a default PreviewSurface material (no color) and binds it to the garment mesh.

        This ensures the garment always has a material bound, even if no color or material
        is specified in the configuration.

        Args:
            material_name: Name for the material (default: "default_material")
            prim_path: Prim path to bind the material to. If None, uses self.mesh_prim_path

        Returns:
            str: The path of the created material
        """
        if prim_path is None:
            prim_path = self.mesh_prim_path

        opaque_mtl_path = f"{self.usd_prim_path}/Looks/{material_name}"
        PreviewSurface(prim_path=opaque_mtl_path)

        # Bind material to the garment mesh
        mesh_prim_to_bind = prims_utils.get_prim_at_path(prim_path)
        if not mesh_prim_to_bind:
            print(
                f"Warning: Could not find mesh prim at {prim_path} to bind default material."
            )
            return opaque_mtl_path

        # Apply material to main mesh with strongerThanDescendants
        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=prim_path,
            material_path=opaque_mtl_path,
            strength=UsdShade.Tokens.strongerThanDescendants,
        )

        # Apply material to submeshes if any, also with strongerThanDescendants
        garment_submesh = prims_utils.get_prim_children(mesh_prim_to_bind)
        if len(garment_submesh) > 0:
            for sub_prim in garment_submesh:
                if sub_prim.IsA(UsdGeom.Gprim):
                    omni.kit.commands.execute(
                        "BindMaterialCommand",
                        prim_path=sub_prim.GetPath(),
                        material_path=opaque_mtl_path,
                        strength=UsdShade.Tokens.strongerThanDescendants,
                    )

        return opaque_mtl_path

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

    def _get_initial_info(self):
        """Capture initial particle positions for reset functionality."""
        if self._device == "cpu":
            self.initial_points_positions = (
                self._get_points_pose().detach().cpu().numpy()
            )
        else:
            self.physics_sim_view = SimulationManager.get_physics_sim_view()
            self._cloth_prim_view.initialize(self.physics_sim_view)
            self.initial_points_positions = self._cloth_prim_view.get_world_positions()

    def transform_points(self, points, pos, ori, scale):
        """
        Transform local points to world space using position, orientation, and scale.

        Args:
            points: (N, 3) array of local points
            pos: (3,) position vector
            ori: (4,) quaternion orientation
            scale: Scale factor (numpy array)

        Returns:
            (N, 3) array of transformed points in world space
        """
        ori_matrix = quat_to_rot_matrix(ori)  # Expects numpy array, returns numpy array
        scaled_points = (
            points * scale
        )  # element-wise multiplication if scale is numpy array
        transformed_points = scaled_points @ ori_matrix.T + pos
        return transformed_points

    def inverse_transform_points(self, transformed_points, pos, ori, scale):
        """
        Transform world space points back to local space.

        Args:
            transformed_points: (N, 3) array of world space points
            pos: (3,) position vector
            ori: (4,) quaternion orientation
            scale: Scale factor (numpy array)

        Returns:
            (N, 3) array of points in local space
        """
        ori_matrix = quat_to_rot_matrix(ori)  # Expects numpy array, returns numpy array
        shifted_points = transformed_points - pos
        rotated_points = shifted_points @ ori_matrix
        original_points = (
            rotated_points / scale
        )  # element-wise division if scale is numpy array
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

        # Randomize garment_config parameters
        if "garment_config" in modified_config:
            garment_config = modified_config["garment_config"].copy()
            garment_params_to_randomize = [
                "particle_mass",
                "stretch_stiffness",
                "bend_stiffness",
                "shear_stiffness",
                "spring_damping",
            ]

            for param in garment_params_to_randomize:
                if param in garment_config and garment_config[param] is not None:
                    original_value = garment_config[param]
                    if isinstance(original_value, (int, float)):
                        variation = original_value * (ratio - 1)
                        min_val = original_value - variation
                        max_val = original_value + variation
                        garment_config[param] = random.uniform(min_val, max_val)

            modified_config["garment_config"] = garment_config

        # Randomize particle_system parameters
        if "particle_system" in modified_config:
            particle_system = modified_config["particle_system"].copy()
            particle_system_params_to_randomize = [
                "solver_position_iteration_count",
                "max_depenetration_velocity",
                "contact_offset",
                "rest_offset",
                "particle_contact_offset",
                "fluid_rest_offset",
                "solid_rest_offset",
                "max_neighborhood",
                "max_velocity",
            ]

            for param in particle_system_params_to_randomize:
                if param in particle_system and particle_system[param] is not None:
                    original_value = particle_system[param]
                    if isinstance(original_value, (int, float)):
                        variation = original_value * (ratio - 1)
                        min_val = original_value - variation
                        max_val = original_value + variation
                        particle_system[param] = random.uniform(min_val, max_val)
                    elif isinstance(original_value, list) and len(original_value) > 0:
                        # Handle list values (e.g., wind vector)
                        randomized_list = []
                        for val in original_value:
                            if isinstance(val, (int, float)):
                                variation = val * (ratio - 1)
                                min_val = val - variation
                                max_val = val + variation
                                randomized_list.append(random.uniform(min_val, max_val))
                            else:
                                randomized_list.append(val)
                        particle_system[param] = randomized_list

            modified_config["particle_system"] = particle_system

        # Randomize particle_material parameters
        if "particle_material" in modified_config:
            particle_material = modified_config["particle_material"].copy()
            material_params_to_randomize = [
                "adhesion",
                "adhesion_offset_scale",
                "cohesion",
                "particle_adhesion_scale",
                "particle_friction_scale",
                "drag",
                "lift",
                "friction",
                "damping",
                "gravity_scale",
                "viscosity",
                "vorticity_confinement",
                "surface_tension",
            ]

            for param in material_params_to_randomize:
                if param in particle_material and particle_material[param] is not None:
                    original_value = particle_material[param]
                    if isinstance(original_value, (int, float)):
                        variation = original_value * (ratio - 1)
                        min_val = original_value - variation
                        max_val = original_value + variation
                        particle_material[param] = random.uniform(min_val, max_val)

            modified_config["particle_material"] = particle_material

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
        """Get the state of the garment object.

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
            if hasattr(self, "_cloth_prim_view") and self._cloth_prim_view is not None:
                # Try to get nodal velocities and compute mean
                nodal_velocity = self._cloth_prim_view.get_velocities()
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

            # Angular velocity is typically zero for garment objects (no rigid body rotation)
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
