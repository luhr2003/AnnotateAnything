"""Ground object with material domain randomization support."""

import random
import numpy as np
import os
import omni.kit.commands
import isaacsim.core.utils.prims as prims_utils
from omegaconf import DictConfig
from typing import Optional, List, Dict
from pxr import Gf, UsdGeom, UsdShade, Sdf

from isaacsim.core.utils.prims import (
    delete_prim,
    is_prim_path_valid,
    get_prim_at_path,
)
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.api.materials import PreviewSurface
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.replicator.behavior.utils.scene_utils import create_mdl_material
from isaacsim.core.prims import SingleGeometryPrim

from magicsim.Env.Utils.path import resolve_mdl_paths, resolve_path
from magicsim.Env.Scene.Object.physics_material import PhysicsMaterial


class Ground:
    """Ground plane with material domain randomization.

    This class creates a ground plane for each parallel environment and supports
    material randomization through MDL files or color changes.

    Args:
        prim_path: USD prim path for the ground
        config: Configuration containing ground parameters including:
            - size: Ground plane size (should match env_spacing)
            - materials: List of material configurations for randomization
            - axis: Ground plane axis (default "Z")
            - position: Ground plane position (default [0, 0, 0])
            - thickness: Ground plane thickness (default 0.1)
            - geometry: Geometry type for ground (default "cube", options: "cube", "plane")
            - scale: Multiplier for in-plane extent vs ``env_spacing`` (default 1.0). Applied to
              the two horizontal axes (XY for Z-up ground); thickness is unchanged for cubes.
    """

    def __init__(self, prim_path: str, config: DictConfig, env_spacing: float):
        """Initialize the Ground object.

        Args:
            prim_path: USD prim path for the ground
            config: Ground configuration
            env_spacing: Environment spacing (used as ground size)
        """
        self.prim_path = prim_path
        self.config = config
        self.env_spacing = env_spacing
        self.stage = get_current_stage()

        # Extract configuration parameters
        self.axis = config.get("axis", "Z")
        self.position = config.get("position", [0.0, 0.0, 0.0])
        self.base_color = config.get("base_color", [0.8, 0.8, 0.8])
        self.thickness = config.get("thickness", 0.1)  # Ground thickness
        self.geometry_type = config.get("geometry", "cube")  # cube or plane
        self.scale_xy = float(config.get("scale", 1.0))

        # Material randomization configuration
        self.materials = config.get("materials", None)

        # Process materials to expand folder paths
        if self.materials:
            self.materials = self._process_material_configs(self.materials)

        self.use_material_randomization = (
            self.materials is not None and len(self.materials) > 0
        )

        # Current material state
        self.current_material_path = None
        self.current_visual_material = None

        # Physics material configuration
        self.physics_material_config = config.get("physics_material", {})

        # Geometry wrapper (will be set in create)
        self.geometry_prim = None

        self.create()
        self.initialize()

    def _process_material_configs(self, material_configs: List[Dict]) -> List[Dict]:
        """Process material configurations to expand folder paths into individual files.

        Args:
            material_configs: List of material configurations

        Returns:
            Expanded list of material configurations with individual MDL files
        """
        expanded_configs = []

        for config in material_configs:
            material_type = config.get("type", "color")

            if material_type == "color":
                # Color materials don't need expansion
                expanded_configs.append(config)

            elif material_type == "mdl":
                # MDL materials may need folder expansion
                mdl_path = resolve_path(config.get("mdl_path"))

                if not mdl_path:
                    print("Warning: MDL material config missing 'mdl_path'. Skipping.")
                    continue

                # Check if the path is a single file or a folder
                resolved_paths = []

                # Check if it's a single MDL file
                if os.path.isfile(mdl_path) and mdl_path.lower().endswith(".mdl"):
                    # Single MDL file - use it directly
                    resolved_paths = [mdl_path]
                else:
                    # Folder path - resolve all MDL files in the folder
                    resolved_paths = resolve_mdl_paths(mdl_path)

                if not resolved_paths:
                    print(f"Warning: No MDL files found for path: {mdl_path}")
                    continue

                # Create a config for each resolved MDL file
                for file_path in resolved_paths:
                    expanded_config = {
                        "type": "mdl",
                        "mdl_path": file_path,
                        # Preserve mdl_name if explicitly provided, otherwise it will be auto-generated
                        "mdl_name": config.get("mdl_name", None),
                    }
                    expanded_configs.append(expanded_config)

            else:
                # Unknown type, keep as is
                expanded_configs.append(config)

        return expanded_configs

    def create(self):
        """Create the ground plane in the USD stage."""
        # Delete existing ground if present
        if is_prim_path_valid(self.prim_path):
            delete_prim(self.prim_path)

        # Prepare physics material
        physics_material_path = find_unique_string_name(
            initial_name=f"{self.prim_path}/physics_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        physics_material = PhysicsMaterial(
            prim_path=physics_material_path,
            config=self.physics_material_config,
        )

        # Create ground using cube or plane geometry
        if self.geometry_type == "cube":
            self._create_cube_ground(physics_material)
        elif self.geometry_type == "plane":
            self._create_plane_ground(physics_material)
        else:
            print(f"Warning: Unknown geometry type '{self.geometry_type}', using cube.")
            self._create_cube_ground(physics_material)

    def _create_cube_ground(self, physics_material):
        """Create ground using a Mesh primitive (Cube shape) with UVs."""
        # Create mesh instead of Cube to support UVs
        mesh_prim = UsdGeom.Mesh.Define(self.stage, self.prim_path)

        # Calculate size based on axis and env_spacing
        if self.axis.upper() == "Z":
            size_x = self.env_spacing
            size_y = self.env_spacing
            size_z = self.thickness
            position = Gf.Vec3d(
                float(self.position[0]),
                float(self.position[1]),
                float(self.position[2]) - self.thickness / 2,
            )
        elif self.axis.upper() == "Y":
            size_x = self.env_spacing
            size_y = self.thickness
            size_z = self.env_spacing
            position = Gf.Vec3d(
                float(self.position[0]),
                float(self.position[1]) - self.thickness / 2,
                float(self.position[2]),
            )
        elif self.axis.upper() == "X":
            size_x = self.thickness
            size_y = self.env_spacing
            size_z = self.env_spacing
            position = Gf.Vec3d(
                float(self.position[0]) - self.thickness / 2,
                float(self.position[1]),
                float(self.position[2]),
            )
        else:
            print(f"Warning: Unknown axis '{self.axis}', using Z.")
            size_x = self.env_spacing
            size_y = self.env_spacing
            size_z = self.thickness
            position = Gf.Vec3d(
                float(self.position[0]),
                float(self.position[1]),
                float(self.position[2]) - self.thickness / 2,
            )

        # Scale horizontal footprint (two in-plane axes); keep thickness axis as-is for Z-up cube.
        if self.axis.upper() == "Z":
            size_x *= self.scale_xy
            size_y *= self.scale_xy
        elif self.axis.upper() == "Y":
            size_x *= self.scale_xy
            size_z *= self.scale_xy
        elif self.axis.upper() == "X":
            size_y *= self.scale_xy
            size_z *= self.scale_xy
        else:
            size_x *= self.scale_xy
            size_y *= self.scale_xy

        # Define a cube of size 2.0 (extents -1 to 1) to match original UsdGeom.Cube(2.0) logic
        points = [
            Gf.Vec3f(-1, -1, -1),  # 0
            Gf.Vec3f(1, -1, -1),  # 1
            Gf.Vec3f(-1, 1, -1),  # 2
            Gf.Vec3f(1, 1, -1),  # 3
            Gf.Vec3f(-1, -1, 1),  # 4
            Gf.Vec3f(1, -1, 1),  # 5
            Gf.Vec3f(-1, 1, 1),  # 6
            Gf.Vec3f(1, 1, 1),  # 7
        ]

        face_vertex_counts = [4] * 6
        face_vertex_indices = [
            4,
            5,
            7,
            6,  # +Z
            1,
            0,
            2,
            3,  # -Z
            2,
            6,
            7,
            3,  # +Y
            0,
            1,
            5,
            4,  # -Y
            5,
            1,
            3,
            7,  # +X
            0,
            4,
            6,
            2,  # -X
        ]

        normals = [
            Gf.Vec3f(0, 0, 1),  # +Z
            Gf.Vec3f(0, 0, -1),  # -Z
            Gf.Vec3f(0, 1, 0),  # +Y
            Gf.Vec3f(0, -1, 0),  # -Y
            Gf.Vec3f(1, 0, 0),  # +X
            Gf.Vec3f(-1, 0, 0),  # -X
        ]

        # UVs (FaceVarying - 4 per face)
        uvs = [
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),
        ] * 6

        # Set attributes
        mesh_prim.GetPointsAttr().Set(points)
        mesh_prim.GetFaceVertexCountsAttr().Set(face_vertex_counts)
        mesh_prim.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
        mesh_prim.GetNormalsAttr().Set(normals)
        mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.uniform)

        # Set UVs
        primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
        uv_primvar = primvars_api.CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        uv_primvar.Set(uvs)

        # Apply scale to match desired size (base size is 2.0)
        scale = Gf.Vec3f(size_x / 2.0, size_y / 2.0, size_z / 2.0)
        mesh_prim.AddScaleOp().Set(scale)

        # Set position
        mesh_prim.AddTranslateOp().Set(position)

        # Wrap with SingleGeometryPrim
        self.geometry_prim = SingleGeometryPrim(
            prim_path=self.prim_path,
            name="ground_cube",
            collision=True,
        )
        # Apply physics material
        if physics_material is not None:
            self.geometry_prim.apply_physics_material(physics_material)

        # Set collision approximation to none (use mesh as-is)
        self.geometry_prim.set_collision_approximation("none")

    def _create_plane_ground(self, physics_material: PhysicsMaterial):
        """Create ground using a Plane (mesh) primitive wrapped with SingleGeometryPrim.

        Args:
            physics_material: Physics material to apply
        """
        # Create mesh for plane
        mesh_prim = UsdGeom.Mesh.Define(self.stage, self.prim_path)

        # Define plane vertices based on env_spacing (horizontal extent × scale_xy)
        half_size = self.env_spacing * self.scale_xy / 2

        if self.axis.upper() == "Z":
            # Ground is XY plane
            vertices = [
                Gf.Vec3f(-half_size, -half_size, 0.0),  # Bottom-left
                Gf.Vec3f(half_size, -half_size, 0.0),  # Bottom-right
                Gf.Vec3f(half_size, half_size, 0.0),  # Top-right
                Gf.Vec3f(-half_size, half_size, 0.0),  # Top-left
            ]
            normals = [Gf.Vec3f(0, 0, 1)] * 4
        elif self.axis.upper() == "Y":
            # Ground is XZ plane
            vertices = [
                Gf.Vec3f(-half_size, 0.0, -half_size),
                Gf.Vec3f(half_size, 0.0, -half_size),
                Gf.Vec3f(half_size, 0.0, half_size),
                Gf.Vec3f(-half_size, 0.0, half_size),
            ]
            normals = [Gf.Vec3f(0, 1, 0)] * 4
        else:  # X
            # Ground is YZ plane
            vertices = [
                Gf.Vec3f(0.0, -half_size, -half_size),
                Gf.Vec3f(0.0, half_size, -half_size),
                Gf.Vec3f(0.0, half_size, half_size),
                Gf.Vec3f(0.0, -half_size, half_size),
            ]
            normals = [Gf.Vec3f(1, 0, 0)] * 4

        # Set mesh attributes
        mesh_prim.GetPointsAttr().Set(vertices)
        mesh_prim.GetFaceVertexCountsAttr().Set([4])  # One quad face
        mesh_prim.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        mesh_prim.GetNormalsAttr().Set(normals)

        # Set UVs
        uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
        primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
        uv_primvar = primvars_api.CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        uv_primvar.Set(uvs)

        # Set position
        position = np.array(
            [
                float(self.position[0]),
                float(self.position[1]),
                float(self.position[2]),
            ]
        )
        mesh_prim.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], position[2]))

        # Set display color
        color_attr = mesh_prim.GetDisplayColorAttr()
        if not color_attr:
            color_attr = mesh_prim.CreateDisplayColorAttr()
        color_attr.Set(
            [
                Gf.Vec3f(
                    float(self.base_color[0]),
                    float(self.base_color[1]),
                    float(self.base_color[2]),
                )
            ]
        )

        # Wrap with SingleGeometryPrim
        self.geometry_prim = SingleGeometryPrim(
            prim_path=self.prim_path,
            name="ground_plane",
            collision=True,
        )

        # Apply physics material
        if physics_material is not None:
            self.geometry_prim.apply_physics_material(physics_material)

        # Set collision approximation to none (use mesh as-is)
        self.geometry_prim.set_collision_approximation("none")

    def initialize(self):
        """Initialize the ground with the configuration after the first hard reset."""
        self.reset()

    def reset(self):
        """Soft Reset Ground and Perform Material Randomization Here."""
        if self.use_material_randomization:
            self._randomize_material()

    def _randomize_material(self):
        """Randomize ground material.

        This method supports two types of material randomization:
        1. MDL materials: Apply MDL material files
        2. Color materials: Apply random colors using PreviewSurface
        """
        if not self.materials:
            return

        # Select a random material configuration
        material_config = random.choice(self.materials)
        material_type = material_config.get("type", "color")

        if material_type == "mdl":
            self._apply_mdl_material(material_config)
        elif material_type == "color":
            self._apply_color_material(material_config)
        else:
            print(
                f"Warning: Unknown material type '{material_type}' for ground. Skipping."
            )

    def _apply_mdl_material(self, material_config: dict):
        """Apply an MDL material to the ground using create_mdl_material.

        Args:
            material_config: Material configuration containing:
                - mdl_path: Path to the MDL file (e.g., "./Assets/Material/Base/Wood.mdl")
                - mdl_name: Name of the material in the MDL file (optional, defaults to filename)
        """
        mdl_path = resolve_path(material_config.get("mdl_path"))

        if not mdl_path:
            print("Warning: MDL material requires 'mdl_path'. Skipping.")
            return

        # Extract material name from mdl_path if not provided
        mdl_name = material_config.get("mdl_name")
        if not mdl_name:
            mdl_name = os.path.splitext(os.path.basename(mdl_path))[0]

        # Create unique material path under ground's Looks
        looks_path = f"{self.prim_path}/Looks"
        material_path = find_unique_string_name(
            initial_name=f"{looks_path}/mdl_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )

        # Use create_mdl_material to create the material (MDL material is already complete)
        try:
            # Create the MDL material
            create_mdl_material(mdl_path, mdl_name, material_path)
            self.current_material_path = material_path
            self.current_visual_material = PreviewSurface(material_path)

            # Bind material to the ground prim
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.prim_path,
                material_path=material_path,
                strength=UsdShade.Tokens.strongerThanDescendants,
            )

            children_prims = prims_utils.get_prim_children(self.geometry_prim.prim)
            for prim in children_prims:
                if prim.GetTypeName() in ["Mesh", "GeomSubset"]:
                    omni.kit.commands.execute(
                        "BindMaterialCommand",
                        prim_path=prim.GetPath(),
                        material_path=material_path,
                        strength=UsdShade.Tokens.strongerThanDescendants,
                    )

        except Exception as e:
            print(f"Warning: Failed to apply MDL material: {e}")
            # Fallback to color material
            self._apply_color_material({"color": self.base_color})

    def _apply_color_material(self, material_config: dict):
        """Apply a color material to the ground using PreviewSurface.

        Args:
            material_config: Material configuration containing:
                - color: RGB color as list [r, g, b] (values 0-1)
        """
        if self.geometry_prim is None:
            print("Warning: Geometry prim not initialized. Cannot apply material.")
            return

        color = material_config.get("color", self.base_color)

        # Create unique material path
        material_path = find_unique_string_name(
            initial_name=f"{self.prim_path}/Looks/color_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )

        # Check if material already exists
        material_prim = get_prim_at_path(material_path)

        if not material_prim:
            # Create new PreviewSurface material
            material = PreviewSurface(
                prim_path=material_path, color=np.array(color, dtype=np.float32)
            )
        else:
            # Update existing material color
            material = PreviewSurface(prim_path=material_path)
            material.set_color(np.array(color, dtype=np.float32))

        # Bind material using geometry_prim's apply_visual_material (like cuboid.py)
        try:
            self.geometry_prim.apply_visual_material(material)
        except Exception as e:
            print(f"Warning: Failed to apply visual material: {e}")

        self.current_material_path = material_path
        self.current_visual_material = material

    def get_prim_path(self) -> str:
        """Get the USD prim path of the ground.

        Returns:
            Ground prim path
        """
        return self.prim_path

    def get_current_material(self) -> Optional[str]:
        """Get the current material path applied to the ground.

        Returns:
            Current material path or None
        """
        return self.current_material_path

    def get_material_count(self) -> int:
        """Get the total number of materials available for randomization.

        Returns:
            Number of materials (after folder expansion)
        """
        return len(self.materials) if self.materials else 0
