import os
import random
import re
import numpy as np
import torch
import omni.kit.commands
import carb
from magicsim.Env.Scene.Object.Fluid import generate_particles_in_convex_mesh
from omni.physx.scripts import particleUtils, physicsUtils
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path
from isaacsim.core.utils.semantics import add_labels, remove_labels
from pxr import Usd, UsdGeom, Sdf, Gf, Vt, PhysxSchema, UsdShade, UsdPhysics
from omegaconf import DictConfig
from magicsim.Env.Utils.path import (
    get_usd_paths_from_folder,
    resolve_mdl_paths,
    resolve_path,
)
from isaacsim.replicator.behavior.utils.scene_utils import create_mdl_material
from isaacsim.core.api.materials.preview_surface import PreviewSurface
import isaacsim.core.utils.prims as prims_utils


class SandObject:
    """
    SandObject class for simulating sand particles in Isaac Sim.
    Uses PhysX particle system to simulate granular materials like sand.
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
        Initialize the SandObject with configuration, USD assets, and physics setup.

        Args:
            prim_path: Path to the prim in the stage
            usd_path: Path to the USD asset file for emitter mesh (or None for primitive)
            config: Configuration dictionary containing sand properties
            env_origin: Origin position of the environment
            layout_manager: LayoutManager instance for position management
            primitive_type: Type of primitive shape if using a primitive emitter
            layout_info: Pre-computed layout information (optional)
        """
        # Enable CPU particle updates for visualization
        carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
        carb.settings.get_settings().set_bool("/physics/updateParticlesToUsd", True)

        # --- 1. Configuration Parsing ---
        self.prim_path = prim_path
        prim_path_parts = prim_path.split("/")
        self.category_name = prim_path_parts[-2]
        self.instance_name = prim_path_parts[-1]
        self.num_per_env = config.objects[self.category_name].get("num_per_env")
        self.instance_name = self._re_instance_name(self.instance_name)
        self.env_name = prim_path_parts[-4] if len(prim_path_parts) >= 4 else "World"

        self.primitive_type = primitive_type
        self.global_config = config
        self.category_config = config.objects[self.category_name]
        self.instance_config = self.category_config.get(self.instance_name, {})

        category_common_config_val = self.category_config.get("common")
        self.category_common_config = (
            category_common_config_val if category_common_config_val is not None else {}
        )

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

        inst_particle_system_cfg_val = self.physics_cfg.get("particle_system", {})
        self.inst_particle_system_cfg = (
            inst_particle_system_cfg_val
            if inst_particle_system_cfg_val is not None
            else {}
        )

        inst_particle_material_cfg_val = self.physics_cfg.get("particle_material", {})
        self.inst_particle_material_cfg = (
            inst_particle_material_cfg_val
            if inst_particle_material_cfg_val is not None
            else {}
        )

        inst_emitter_cfg_val = self.physics_cfg.get("emitter_config", {})
        self.inst_emitter_cfg = (
            inst_emitter_cfg_val if inst_emitter_cfg_val is not None else {}
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
        self.prim_name = self.instance_name
        self.stage = get_current_stage()

        self.layout_manager = layout_manager
        self.layout_info = layout_info
        self.env_origin = env_origin.detach().cpu().numpy()

        # --- 2. Initial Pose ---
        if self.layout_info:
            pos_from_layout = self.layout_info["pos"]
            self.init_pos = (
                np.array(pos_from_layout, dtype=np.float32) + self.env_origin
            )
            self.init_ori = self.layout_info["ori"]
            self.init_scale = self.layout_info["scale"]
        else:
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

        # --- 3. Create Points Prim for Particles ---
        self.points_prim_path = self.usd_prim_path + "/Points"
        if not is_prim_path_valid(self.points_prim_path):
            self.points_prim = UsdGeom.Points.Define(self.stage, self.points_prim_path)
        else:
            self.points_prim = UsdGeom.Points.Get(self.stage, self.points_prim_path)

        # Set initial empty positions (will be filled by emitter)
        self.points_prim.GetPointsAttr().Set(Vt.Vec3fArray([]))

        # --- 4. Create Emitter Mesh (if USD path provided) ---
        self.emitter_prim_path = None
        if usd_path:
            self.emitter_prim_path = self.usd_prim_path + "/Emitter"
            add_reference_to_stage(usd_path=usd_path, prim_path=self.emitter_prim_path)
            self.emitter_prim_path = self._find_first_mesh_in_hierarchy(
                self.emitter_prim_path
            )
        elif primitive_type:
            # Create primitive emitter
            self.emitter_prim_path = self.usd_prim_path + "/Emitter"
            self._create_primitive_emitter(primitive_type)

        # Note: Emitter transform is not set here because:
        # 1. The emitter is only used to generate initial particle positions
        # 2. The actual particle positions will be transformed by the PointInstancer's transform
        # 3. Trying to modify existing USD file transform operations can cause conflicts
        # If you need to transform the emitter mesh, do it before loading the USD file,
        # or use the PointInstancer transform which is set later in the code

        # --- 5. Create Collider (optional) ---
        collider_cfg = self.inst_emitter_cfg.get("collider", {})
        self.collider_enabled = collider_cfg.get("enabled", False)
        self.collider_prim_path = None
        if self.collider_enabled:
            collider_usd = collider_cfg.get("usd_path", None)
            if collider_usd:
                self.collider_prim_path = self.usd_prim_path + "/Collider"
                add_reference_to_stage(
                    usd_path=collider_usd, prim_path=self.collider_prim_path
                )
                # Make collider physics-enabled
                collider_prim = get_prim_at_path(self.collider_prim_path)
                if collider_prim:
                    UsdPhysics.CollisionAPI.Apply(collider_prim)
                    UsdPhysics.RigidBodyAPI.Apply(collider_prim)

        # --- 6. Particle System Setup ---
        interaction_flag = self.category_config.get("interaction_with_object", False)
        if interaction_flag:
            self.particle_system_path = (
                f"/Particle_Attribute/{self.env_name}/particle_system"
            )
        else:
            self.particle_system_path = (
                f"/Particle_Attribute/{self.env_name}/sand_particle_system"
            )

        if not is_prim_path_valid(self.particle_system_path):
            self.particle_system = PhysxSchema.PhysxParticleSystem.Define(
                self.stage, self.particle_system_path
            )
        else:
            prim = self.stage.GetPrimAtPath(self.particle_system_path)
            self.particle_system = PhysxSchema.PhysxParticleSystem(prim)

        # Configure particle system properties (optimized for granular materials)
        self.particle_system.CreateParticleContactOffsetAttr().Set(
            self.inst_particle_system_cfg.get("particle_contact_offset", 0.01)
        )
        self.particle_system.CreateContactOffsetAttr().Set(
            self.inst_particle_system_cfg.get("contact_offset", 0.015)
        )
        self.particle_system.CreateRestOffsetAttr().Set(
            self.inst_particle_system_cfg.get("rest_offset", 0.008)
        )
        self.particle_system.CreateFluidRestOffsetAttr().Set(
            self.inst_particle_system_cfg.get("fluid_rest_offset", 0.008)
        )
        self.particle_system.CreateSolidRestOffsetAttr().Set(
            self.inst_particle_system_cfg.get("solid_rest_offset", 0.008)
        )
        self.particle_system.CreateMaxVelocityAttr().Set(
            self.inst_particle_system_cfg.get("max_velocity", 10.0)
        )

        # Apply optional particle system APIs
        if self.inst_particle_system_cfg.get("smoothing", False):
            PhysxSchema.PhysxParticleSmoothingAPI.Apply(self.particle_system.GetPrim())
        if self.inst_particle_system_cfg.get("anisotropy", False):
            PhysxSchema.PhysxParticleAnisotropyAPI.Apply(self.particle_system.GetPrim())
        if self.inst_particle_system_cfg.get("isosurface", False):
            PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(self.particle_system.GetPrim())

        # --- 7. Particle Material Setup ---
        self.particle_material_path = find_unique_string_name(
            self.usd_prim_path + "/particle_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )

        particleUtils.add_pbd_particle_material(
            stage=self.stage,
            path=self.particle_material_path,
            cohesion=self.inst_particle_material_cfg.get("cohesion", 0.2),
            friction=self.inst_particle_material_cfg.get("friction", 0.75),
            damping=self.inst_particle_material_cfg.get("damping", 0.1),
            adhesion=self.inst_particle_material_cfg.get("adhesion", 0.0),
            particle_adhesion_scale=self.inst_particle_material_cfg.get(
                "particle_adhesion_scale", 0.5
            ),
            particle_friction_scale=self.inst_particle_material_cfg.get(
                "particle_friction_scale", 0.75
            ),
        )

        # --- 8. Particle Instancing ---
        particle_radius = self.inst_emitter_cfg.get("particle_radius", 0.5)
        particle_spacing = self.inst_emitter_cfg.get("particle_spacing", 1.0)
        max_particles = self.inst_emitter_cfg.get("max_particles", 100000)

        # Generate initial particle positions from emitter mesh
        if self.emitter_prim_path:
            emitter_mesh = UsdGeom.Mesh.Get(
                self.stage, Sdf.Path(self.emitter_prim_path)
            )
            if emitter_mesh:
                mesh_points = np.array(emitter_mesh.GetPointsAttr().Get())
                if len(mesh_points) >= 4:
                    # Use the same particle generation as Fluid
                    # sphere_diameter should be the spacing between particles
                    sphere_diameter = particle_spacing
                    try:
                        particle_positions, particle_velocities = (
                            generate_particles_in_convex_mesh(
                                vertices=mesh_points,
                                sphere_diameter=sphere_diameter,
                                visualize=False,
                            )
                        )
                        if len(particle_positions) > 0:
                            # Convert from Gf.Vec3f list to numpy array
                            if isinstance(particle_positions[0], Gf.Vec3f):
                                positions_np = np.array(
                                    [[p[0], p[1], p[2]] for p in particle_positions],
                                    dtype=np.float32,
                                )
                            else:
                                positions_np = np.array(
                                    particle_positions, dtype=np.float32
                                )
                            self.init_particle_positions = positions_np
                            # Limit to max_particles
                            if len(self.init_particle_positions) > max_particles:
                                self.init_particle_positions = (
                                    self.init_particle_positions[:max_particles]
                                )
                            print(
                                f"Generated {len(self.init_particle_positions)} particles using convex mesh method"
                            )
                        else:
                            # Fallback to simple generation
                            print(
                                "Warning: Convex mesh method generated 0 particles, falling back to simple generation"
                            )
                            self.init_particle_positions = (
                                self._generate_particles_from_emitter(
                                    particle_spacing, max_particles
                                )
                            )
                    except Exception as e:
                        print(
                            f"Warning: Failed to generate particles using convex mesh method: {e}"
                        )
                        # Fallback to simple generation
                        self.init_particle_positions = (
                            self._generate_particles_from_emitter(
                                particle_spacing, max_particles
                            )
                        )
                else:
                    # Fallback to simple generation
                    print(
                        f"Warning: Emitter mesh has only {len(mesh_points)} points, using simple generation"
                    )
                    self.init_particle_positions = (
                        self._generate_particles_from_emitter(
                            particle_spacing, max_particles
                        )
                    )
            else:
                # Fallback to simple generation
                print("Warning: Could not get emitter mesh, using simple generation")
                self.init_particle_positions = self._generate_particles_from_emitter(
                    particle_spacing, max_particles
                )
        else:
            # Fallback: create a small grid of particles at origin
            # Create a 3x3x3 grid as default
            default_grid_size = 3
            default_spacing = 0.1
            particles = []
            for i in range(default_grid_size):
                for j in range(default_grid_size):
                    for k in range(default_grid_size):
                        pos = np.array(
                            [
                                (i - 1) * default_spacing,
                                (j - 1) * default_spacing,
                                (k - 1) * default_spacing,
                            ],
                            dtype=np.float32,
                        )
                        particles.append(pos)
            self.init_particle_positions = np.array(particles, dtype=np.float32)
            print(
                f"Generated {len(self.init_particle_positions)} particles in default grid (no emitter)"
            )

        print(f"Initial particle count: {len(self.init_particle_positions)}")

        self.particle_point_instancer_path = Sdf.Path(self.usd_prim_path).AppendChild(
            "particles"
        )

        # Convert numpy arrays to Vt.Vec3fArray format
        # Ensure positions are in float32 format and convert to list of tuples or use FromNumpy
        positions_array = self.init_particle_positions.astype(np.float32)
        velocities_array = np.zeros_like(self.init_particle_positions, dtype=np.float32)

        # Use FromNumpy if available, otherwise convert to list of tuples
        try:
            positions_vt = Vt.Vec3fArray.FromNumpy(positions_array)
            velocities_vt = Vt.Vec3fArray.FromNumpy(velocities_array)
        except (AttributeError, TypeError):
            # Fallback: convert to list of tuples
            positions_list = [tuple(pos) for pos in positions_array]
            velocities_list = [tuple(vel) for vel in velocities_array]
            positions_vt = Vt.Vec3fArray(positions_list)
            velocities_vt = Vt.Vec3fArray(velocities_list)

        particleUtils.add_physx_particleset_pointinstancer(
            stage=self.stage,
            path=self.particle_point_instancer_path,
            positions=positions_vt,
            velocities=velocities_vt,
            particle_system_path=self.particle_system_path,
            self_collision=True,
            fluid=False,  # Sand is granular, not fluid
            particle_group=0,
            particle_mass=self.inst_particle_material_cfg.get("particle_mass", 0.001),
            density=0.0,
        )

        # Bind particle material to the particle system
        system_prim = self.stage.GetPrimAtPath(self.particle_system_path)
        if system_prim and system_prim.IsValid():
            physicsUtils.add_physics_material_to_prim(
                self.stage, system_prim, self.particle_material_path
            )

        # Get point instancer for setting up prototype and transform
        self.point_instancer = UsdGeom.PointInstancer.Get(
            self.stage, self.particle_point_instancer_path
        )

        # Set transform for point instancer
        init_scale_array = np.array(self.init_scale, dtype=np.float32)
        physicsUtils.set_or_add_scale_orient_translate(
            self.point_instancer,
            translate=Gf.Vec3f([float(v) for v in self.init_pos]),
            orient=Gf.Quatf(1.0, Gf.Vec3f(0, 0, 0)),  # Identity quaternion
            scale=Gf.Vec3f([float(v) for v in init_scale_array]),
        )

        # Create particle prototype geometry (sphere) for visualization
        particle_radius = self.inst_emitter_cfg.get("particle_radius", 0.5)
        rest_offset = self.inst_particle_system_cfg.get("rest_offset", 0.008)

        proto_path = self.particle_point_instancer_path.AppendChild(
            "particlePrototype0"
        )
        self.particle_prototype_path = proto_path

        # Create sphere prototype for particles
        if not is_prim_path_valid(proto_path):
            particle_prototype_sphere = UsdGeom.Sphere.Define(self.stage, proto_path)
        else:
            particle_prototype_sphere = UsdGeom.Sphere.Get(self.stage, proto_path)

        # Set radius (use rest_offset * 2 for diameter, or particle_radius)
        rest_offset = self.inst_particle_system_cfg.get("rest_offset", 0.008)
        # Use rest_offset * 2 for sphere diameter, or use particle_radius if specified and reasonable
        if particle_radius > 0.001 and particle_radius < 0.1:
            sphere_radius = float(particle_radius)
        else:
            sphere_radius = float(rest_offset * 2)
        particle_prototype_sphere.CreateRadiusAttr().Set(sphere_radius)

        # Ensure prototype is bound to point instancer (add_physx_particleset_pointinstancer may have created it already)
        # Check if prototypes are already set, if not, set them manually
        try:
            existing_prototypes = self.point_instancer.GetPrototypesRel().GetTargets()
            if not existing_prototypes or len(existing_prototypes) == 0:
                # Manually set prototype relationship
                self.point_instancer.GetPrototypesRel().SetTargets([proto_path])
                # Set proto indices for all particles (all use prototype 0)
                num_particles = len(self.init_particle_positions)
                self.point_instancer.GetProtoIndicesAttr().Set([0] * num_particles)
        except Exception as e:
            print(f"Warning: Could not set particle prototypes: {e}")

        # Hide emitter mesh if it exists (we only want to see particles)
        if self.emitter_prim_path:
            emitter_prim = get_prim_at_path(self.emitter_prim_path)
            if emitter_prim:
                imageable = UsdGeom.Imageable(emitter_prim)
                imageable.MakeInvisible()

        # --- 9. Visual Setup ---
        visible = self.visual_cfg.get("visible", True)
        if not visible and self.points_prim:
            imageable = UsdGeom.Imageable(self.points_prim)
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
                            f"Warning: No USD paths found in material folder: {self.visual_material_usd_folder}"
                        )

        # --- 10. Semantic Labels ---
        self._handle_semantic_labels()

        # Store initial state for reset
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _create_primitive_emitter(self, primitive_type: str):
        """Create a primitive mesh as emitter."""
        if primitive_type == "Box":
            self._create_box_emitter()
        elif primitive_type == "Sphere":
            self._create_sphere_emitter()
        elif primitive_type == "Cylinder":
            self._create_cylinder_emitter()
        else:
            raise ValueError(
                f"Unsupported primitive type for emitter: {primitive_type}"
            )

    def _create_box_emitter(self):
        """Create a box mesh emitter."""
        size = self.inst_emitter_cfg.get("size", [1.0, 1.0, 1.0])
        mesh = UsdGeom.Mesh.Define(self.stage, self.emitter_prim_path)
        # Simple box vertices (8 vertices for a box)
        half_size = [s / 2.0 for s in size]
        points = [
            [-half_size[0], -half_size[1], -half_size[2]],
            [half_size[0], -half_size[1], -half_size[2]],
            [half_size[0], half_size[1], -half_size[2]],
            [-half_size[0], half_size[1], -half_size[2]],
            [-half_size[0], -half_size[1], half_size[2]],
            [half_size[0], -half_size[1], half_size[2]],
            [half_size[0], half_size[1], half_size[2]],
            [-half_size[0], half_size[1], half_size[2]],
        ]
        face_vertex_counts = [4, 4, 4, 4, 4, 4]
        face_vertex_indices = [
            0,
            1,
            2,
            3,  # bottom
            4,
            7,
            6,
            5,  # top
            0,
            4,
            5,
            1,  # front
            2,
            6,
            7,
            3,  # back
            0,
            3,
            7,
            4,  # left
            1,
            5,
            6,
            2,  # right
        ]
        mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_vertex_counts))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_vertex_indices))

    def _create_sphere_emitter(self):
        """Create a sphere mesh emitter."""
        radius = self.inst_emitter_cfg.get("radius", 0.5)
        resolution = self.inst_emitter_cfg.get("resolution", 16)
        # Create a simple sphere approximation
        mesh = UsdGeom.Mesh.Define(self.stage, self.emitter_prim_path)
        # For simplicity, use a basic sphere generation
        # In practice, you might want to use a more sophisticated method
        points = []
        indices = []
        # Simplified sphere generation
        for i in range(resolution):
            theta = 2.0 * np.pi * i / resolution
            for j in range(resolution // 2):
                phi = np.pi * j / (resolution // 2)
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.cos(phi)
                z = radius * np.sin(phi) * np.sin(theta)
                points.append([x, y, z])
        # This is a simplified version - full sphere generation would be more complex
        mesh.GetPointsAttr().Set(Vt.Vec3fArray(points[:8]))  # Minimal set

    def _create_cylinder_emitter(self):
        """Create a cylinder mesh emitter."""
        radius = self.inst_emitter_cfg.get("radius", 0.5)
        height = self.inst_emitter_cfg.get("height", 1.0)
        resolution = self.inst_emitter_cfg.get("resolution", 16)
        mesh = UsdGeom.Mesh.Define(self.stage, self.emitter_prim_path)
        # Simplified cylinder
        points = []
        for i in range(resolution):
            angle = 2.0 * np.pi * i / resolution
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            points.append([x, -height / 2, z])
            points.append([x, height / 2, z])
        mesh.GetPointsAttr().Set(Vt.Vec3fArray(points[:16]))  # Minimal set

    def _find_first_mesh_in_hierarchy(self, prim_path: str) -> str:
        """Recursively searches for the first prim of type UsdGeom.Mesh under the given path."""
        start_prim = get_prim_at_path(prim_path)
        if not start_prim:
            return None

        for prim in Usd.PrimRange(start_prim):
            if prim.IsA(UsdGeom.Mesh):
                return prim.GetPath().pathString

        return None

    def _generate_particles_from_emitter(
        self, spacing: float, max_particles: int
    ) -> np.ndarray:
        """Generate particle positions within the emitter mesh."""
        emitter_mesh = UsdGeom.Mesh.Get(self.stage, Sdf.Path(self.emitter_prim_path))
        if not emitter_mesh:
            return np.array([[0, 0, 0]], dtype=np.float32)

        mesh_points = np.array(emitter_mesh.GetPointsAttr().Get())
        if len(mesh_points) == 0:
            return np.array([[0, 0, 0]], dtype=np.float32)

        # Simple particle generation: sample points within bounding box
        min_bounds = mesh_points.min(axis=0)
        max_bounds = mesh_points.max(axis=0)
        bbox_size = max_bounds - min_bounds

        print(
            f"Emitter bounding box: min={min_bounds}, max={max_bounds}, size={bbox_size}"
        )
        print(
            f"Particle spacing: {spacing}, rest_offset: {self.inst_particle_system_cfg.get('rest_offset', 0.008)}"
        )

        # If spacing is too large compared to bbox, use rest_offset-based spacing
        rest_offset = self.inst_particle_system_cfg.get("rest_offset", 0.008)
        if spacing > np.max(bbox_size) * 0.5:
            # Spacing is too large, use rest_offset * 2.5 as spacing
            effective_spacing = rest_offset * 2.5
            print(
                f"Warning: particle_spacing ({spacing}) is too large for emitter size ({np.max(bbox_size)}). Using rest_offset-based spacing: {effective_spacing}"
            )
            spacing = effective_spacing

        # Generate grid of particles
        num_x = max(1, int((max_bounds[0] - min_bounds[0]) / spacing) + 1)
        num_y = max(1, int((max_bounds[1] - min_bounds[1]) / spacing) + 1)
        num_z = max(1, int((max_bounds[2] - min_bounds[2]) / spacing) + 1)

        print(
            f"Generating particle grid: {num_x} x {num_y} x {num_z} = {num_x * num_y * num_z} particles"
        )

        particles = []
        for i in range(num_x):
            for j in range(num_y):
                for k in range(num_z):
                    pos = np.array(
                        [
                            min_bounds[0] + i * spacing,
                            min_bounds[1] + j * spacing,
                            min_bounds[2] + k * spacing,
                        ]
                    )
                    particles.append(pos)
                    if len(particles) >= max_particles:
                        break
                if len(particles) >= max_particles:
                    break
            if len(particles) >= max_particles:
                break

        if len(particles) == 0:
            print("Warning: No particles generated, using center point")
            return np.array([[(min_bounds + max_bounds) / 2]], dtype=np.float32)

        print(f"Generated {len(particles)} particles using simple method")
        return np.array(particles, dtype=np.float32)

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
        else:
            material = PreviewSurface(prim_path=material_path)
            material.set_color(np.array(color))

        # Bind material to particle prototype (this is what's actually visible)
        if hasattr(self, "particle_prototype_path") and self.particle_prototype_path:
            if is_prim_path_valid(self.particle_prototype_path):
                omni.kit.commands.execute(
                    "BindMaterialCommand",
                    prim_path=self.particle_prototype_path,
                    material_path=material_path,
                    strength=UsdShade.Tokens.strongerThanDescendants,
                )

    def _apply_mdl_material(self, mdl_path: str, mdl_name: str = None):
        """Apply an MDL material to the sand particle prototype.

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

            # Bind material to particle prototype (this is what's actually visible)
            if (
                hasattr(self, "particle_prototype_path")
                and self.particle_prototype_path
            ):
                if is_prim_path_valid(self.particle_prototype_path):
                    omni.kit.commands.execute(
                        "BindMaterialCommand",
                        prim_path=self.particle_prototype_path,
                        material_path=material_path,
                        strength=UsdShade.Tokens.strongerThanDescendants,
                    )

            self.visual_material_path = material_path
            self.visual_material = PreviewSurface(material_path)

        except Exception as e:
            print(f"Warning: Failed to apply MDL material {mdl_path}: {e}")

    def _apply_visual_material(self, material_path: str):
        """Apply a visual USD material to the sand particle prototype.

        Args:
            material_path: Path to the USD material file
        """
        visual_material_prim_path = find_unique_string_name(
            initial_name=f"{self.usd_prim_path}/Looks/visual_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )

        add_reference_to_stage(
            usd_path=material_path, prim_path=visual_material_prim_path
        )

        visual_material_prim = prims_utils.get_prim_at_path(visual_material_prim_path)
        if not visual_material_prim or not visual_material_prim.IsValid():
            print(f"Warning: Could not get valid prim at {visual_material_prim_path}")
            return

        children = prims_utils.get_prim_children(visual_material_prim)
        if not children:
            print(
                f"Warning: Material prim at {visual_material_prim_path} has no children."
            )
            return

        self.material_prim = children[0]
        self.material_prim_path = self.material_prim.GetPath()
        self.visual_material = PreviewSurface(self.material_prim_path)

        # Bind material to particle prototype (this is what's actually visible)
        if hasattr(self, "particle_prototype_path") and self.particle_prototype_path:
            if is_prim_path_valid(self.particle_prototype_path):
                omni.kit.commands.execute(
                    "BindMaterialCommand",
                    prim_path=self.particle_prototype_path,
                    material_path=self.material_prim_path,
                    strength=UsdShade.Tokens.strongerThanDescendants,
                )

    def _re_instance_name(self, inst_name):
        """Reformats the instance name to ensure consistent numbering."""
        parts = inst_name.split("_")
        cat_name_extracted = "_".join(parts[:-1])
        obj_id_str = parts[-1]
        obj_id = int(obj_id_str)
        original_id = (obj_id - 1) % self.num_per_env + 1
        return f"{cat_name_extracted}_{original_id}"

    def _extract_env_id_from_prim_path(self):
        """Extract env_id from prim_path."""
        try:
            parts = self.usd_prim_path.split("/")
            for part in parts:
                if part.startswith("env_"):
                    return int(part.split("_")[1])
        except (ValueError, IndexError):
            pass
        return None

    def _apply_physics_ratio_randomization(self, physics_config):
        """Apply ratio-based randomization to physics parameters."""
        modified_config = physics_config.copy()
        ratio = modified_config.get("ratio", 1.0)

        if ratio == 1.0:
            return modified_config

        # Randomize particle_system parameters
        if "particle_system" in modified_config:
            particle_system = modified_config["particle_system"].copy()
            params_to_randomize = [
                "particle_contact_offset",
                "contact_offset",
                "rest_offset",
                "max_velocity",
            ]

            for param in params_to_randomize:
                if param in particle_system and particle_system[param] is not None:
                    original_value = particle_system[param]
                    if isinstance(original_value, (int, float)):
                        variation = original_value * (ratio - 1)
                        min_val = original_value - variation
                        max_val = original_value + variation
                        particle_system[param] = random.uniform(min_val, max_val)

            modified_config["particle_system"] = particle_system

        # Randomize particle_material parameters
        if "particle_material" in modified_config:
            particle_material = modified_config["particle_material"].copy()
            params_to_randomize = [
                "cohesion",
                "friction",
                "damping",
                "adhesion",
                "particle_mass",
            ]

            for param in params_to_randomize:
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
        if self.points_prim:
            remove_labels(self.points_prim.GetPrim(), include_descendants=True)
            semantic_label = self._get_semantic_label()
            if semantic_label:
                add_labels(self.points_prim.GetPrim(), [semantic_label])
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
            return "sand"

        regex_pattern = self.category_config.get("semantic_regex_pattern", r".*")
        regex_replacement = self.category_config.get("semantic_regex_repl", r"\g<0>")
        filename = os.path.basename(self.usd_path)
        filename_without_ext = os.path.splitext(filename)[0]
        return re.sub(regex_pattern, regex_replacement, filename_without_ext)

    def initialize(self):
        """Initialize the sand object by capturing initial particle information."""
        pass  # Particles are already set up in __init__

    def reset(self, soft=False):
        """Reset sand particles by restoring initial positions."""
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

            # Reset particle positions
            self._reset_particle_positions(self.init_particle_positions.copy())

    def _reset_particle_positions(self, positions: np.ndarray):
        """Reset particle positions in the point instancer."""
        point_instancer_prim = self.stage.GetPrimAtPath(
            self.particle_point_instancer_path
        )
        if point_instancer_prim:
            point_instancer = UsdGeom.PointInstancer(point_instancer_prim)
            # Convert numpy array to Vt.Vec3fArray format
            positions_array = np.array(positions).astype(np.float32)
            try:
                positions_vt = Vt.Vec3fArray.FromNumpy(positions_array)
            except (AttributeError, TypeError):
                # Fallback: convert to list of tuples
                positions_list = [tuple(pos) for pos in positions_array]
                positions_vt = Vt.Vec3fArray(positions_list)
            point_instancer.GetPositionsAttr().Set(positions_vt)

    def get_particle_positions(
        self, visualize=False, save=False, save_path="./sand_particles.ply"
    ):
        """
        Get the current particle positions.

        Args:
            visualize: Whether to visualize the particles using Open3D
            save: Whether to save the particles to a file
            save_path: Path to save the point cloud if save=True

        Returns:
            particle_positions: Current particle positions
        """
        point_instancer_prim = self.stage.GetPrimAtPath(
            self.particle_point_instancer_path
        )
        if point_instancer_prim:
            point_instancer = UsdGeom.PointInstancer(point_instancer_prim)
            positions = point_instancer.GetPositionsAttr().Get()
            if positions:
                particle_positions = np.array(positions)
            else:
                particle_positions = self.init_particle_positions.copy()
        else:
            particle_positions = self.init_particle_positions.copy()

        # Visualization and saving
        if visualize or save:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(particle_positions)

            if visualize:
                o3d.visualization.draw_geometries([pcd])
            if save:
                o3d.io.write_point_cloud(save_path, pcd)

        return particle_positions

    def set_particle_positions(self, positions: np.ndarray):
        """Set particle positions."""
        self._reset_particle_positions(positions)

    def get_state(self, is_relative: bool = False) -> dict:
        """Get the state of the sand object."""
        particle_positions = self.get_particle_positions()

        # Compute center of mass
        if len(particle_positions) > 0:
            com = np.mean(particle_positions, axis=0)
        else:
            com = self.init_pos

        # Convert to tensors
        com_tensor = torch.tensor(com, dtype=torch.float32)
        if is_relative and hasattr(self, "env_origin"):
            env_origin_tensor = torch.tensor(
                self.env_origin, dtype=torch.float32, device=com_tensor.device
            )
            com_tensor -= env_origin_tensor

        # Default orientation (no rotation for particles)
        ori_tensor = torch.tensor([0, 0, 0, 1], dtype=torch.float32)

        # Combine into root_pose [pos(3), quat(4)]
        root_pose = torch.cat([com_tensor, ori_tensor])

        # Root velocity (zeros for now)
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
            "particle_positions": torch.tensor(particle_positions, dtype=torch.float32),
        }
