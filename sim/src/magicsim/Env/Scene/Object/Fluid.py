import os
import random
import re
import numpy as np
import omni.kit.commands
import carb
from scipy.spatial import Delaunay
import open3d as o3d
import torch
from omni.physx.scripts import particleUtils, physicsUtils
from isaacsim.replicator.behavior.utils.scene_utils import create_mdl_material
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.prims import is_prim_path_valid, delete_prim
from isaacsim.core.utils.semantics import add_labels, remove_labels
from isaacsim.core.prims import SingleGeometryPrim

from pxr import UsdGeom, Sdf, Gf, Vt, PhysxSchema
from omegaconf import DictConfig


class FluidObject:
    """
    FluidObject class for simulating fluid particles in Isaac Sim.
    Manages fluid particle systems, containers, materials, and physics properties.
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
        Initialize the FluidObject with configuration, USD assets, and physics setup.

        Args:
            prim_path: Path to the prim in the stage
            usd_path: Path to the USD asset file for this fluid
            config: Configuration dictionary containing fluid properties
        """
        # Enable CPU fluid updates
        carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
        carb.settings.get_settings().set_bool("/physics/updateParticlesToUsd", True)

        # --- 1. Configuration Parsing ---
        self.prim_path = prim_path
        prim_path_parts = prim_path.split("/")
        self.category_name = prim_path_parts[-2]
        self.instance_name = prim_path_parts[-1]
        self.num_per_env = config.objects[self.category_name].get("num_per_env")
        self.instance_name = self._re_instance_name(self.instance_name)

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
        self.physics_cfg = self.instance_config.get("physics", {})

        # Apply ratio-based randomization to physics parameters
        self.physics_cfg = self._apply_physics_ratio_randomization(self.physics_cfg)

        self.layout_manager = layout_manager
        self.layout_info = layout_info
        self.env_origin = env_origin.detach().cpu().numpy()

        # --- 2. Prim and Path Initialization ---
        self.usd_prim_path = prim_path
        self.usd_path = usd_path
        self.prim_name = self.instance_name
        self.env_name = prim_path_parts[-4]
        self.mesh_prim_path = self.usd_prim_path + "/mesh"
        self.stage = get_current_stage()

        # --- 3. Initial Pose and Asset Loading ---
        if self.layout_info:
            # Use provided layout info
            self.init_pos = self.layout_info["pos"]
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

            self.init_pos = layout_info["pos"]
            self.init_ori = layout_info["ori"]
            self.init_scale = layout_info["scale"]

        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        # --- 4. Container Setup ---
        # Two modes: (1) inline container under fluid (enabled + usd);
        # (2) reference container imported separately in config (ref + category), do not create here.
        container_cfg = self.instance_config.get("container", {})
        self.container_offset = container_cfg.get("offset", [0.0, 0.0, 0.0])
        self.container_owned = False  # Whether this Fluid created the container and is responsible for deleting it

        self.container = None
        self.container_prim_path = None
        self.container_position = None

        # Mode 1: Reference container imported separately in config (created by another object in the scene)
        container_ref = container_cfg.get("ref")
        container_category = container_cfg.get("category")
        if container_ref and container_category:
            env_id = self._extract_env_id_from_prim_path()
            if env_id is not None:
                env_root = self._get_env_root_from_prim_path()
                if env_root:
                    container_path = (
                        f"{env_root}/dynamic/{container_category}/{container_ref}"
                    )
                    if is_prim_path_valid(container_path):
                        self.container_prim_path = container_path
                        self.container_owned = False
                        self.container = SingleGeometryPrim(
                            prim_path=container_path,
                            name=f"fluid_container_{self.prim_name}",
                            collision=True,
                        )
                        # Read initial position from existing prim; offset can still be used for later sync
                        prim = self.stage.GetPrimAtPath(container_path)
                        if prim and prim.IsValid():
                            xform = UsdGeom.Xformable(prim)
                            world_xform = xform.ComputeLocalToWorldTransform(0)
                            t = world_xform.ExtractTranslation()
                            self.container_position = Gf.Vec3d(
                                float(t[0]), float(t[1]), float(t[2])
                            )
                    else:
                        carb.log_warn(
                            f"Fluid {self.usd_prim_path}: container ref "
                            f"'{container_category}/{container_ref}' not found at {container_path}"
                        )
            else:
                carb.log_warn(
                    f"Fluid {self.usd_prim_path}: cannot resolve container ref (env_id unknown)."
                )
        else:
            # Mode 2: Inline container creation (original logic)
            container_enabled = container_cfg.get("enabled", False)
            container_usd_path = container_cfg.get(
                "usd", "./Assets/Object/mugs/8567/Object.usd"
            )
            container_scale = container_cfg.get(
                "scale",
                self.instance_config.get("visual", {}).get(
                    "container_scale", [1.0, 1.0, 1.0]
                ),
            )
            if container_enabled and container_usd_path:
                container_path = find_unique_string_name(
                    initial_name=os.path.dirname(prim_path) + "/container",
                    is_unique_fn=lambda x: not is_prim_path_valid(x),
                )
                self.container_prim_path = container_path
                self.container_owned = True
                add_reference_to_stage(
                    usd_path=container_usd_path, prim_path=container_path
                )
                self.container_position = Gf.Vec3d(
                    float(self.init_pos[0] + self.container_offset[0]),
                    float(self.init_pos[1] + self.container_offset[1]),
                    float(self.container_offset[2]),
                )
                self.container = SingleGeometryPrim(
                    prim_path=container_path,
                    name=f"fluid_container_{self.prim_name}",
                    collision=True,
                    scale=container_scale,
                )

        # --- 5. Particle System Setup ---
        inst_particle_system_cfg = self.physics_cfg.get("particle_system", {})
        interaction_flag = self.category_config.get("interaction_with_object", False)

        if interaction_flag:
            self.particle_system_path = (
                f"/Particle_Attribute/{self.env_name}/particle_system"
            )
        else:
            self.particle_system_path = (
                f"/Particle_Attribute/{self.env_name}/fluid_particle_system"
            )

        if not is_prim_path_valid(self.particle_system_path):
            self.particle_system = PhysxSchema.PhysxParticleSystem.Define(
                self.stage, self.particle_system_path
            )
        else:
            prim = self.stage.GetPrimAtPath(self.particle_system_path)
            self.particle_system = PhysxSchema.PhysxParticleSystem(prim)

        # Configure particle system properties
        self.particle_system.CreateParticleContactOffsetAttr().Set(
            inst_particle_system_cfg.get("particle_contact_offset", 0.025)
        )
        self.particle_system.CreateContactOffsetAttr().Set(
            inst_particle_system_cfg.get("contact_offset", 0.025)
        )
        self.particle_system.CreateRestOffsetAttr().Set(
            inst_particle_system_cfg.get("rest_offset", 0.0225)
        )
        self.particle_system.CreateFluidRestOffsetAttr().Set(
            inst_particle_system_cfg.get("fluid_rest_offset", 0.0135)
        )
        self.particle_system.CreateSolidRestOffsetAttr().Set(
            inst_particle_system_cfg.get("solid_rest_offset", 0.0225)
        )
        self.particle_system.CreateMaxVelocityAttr().Set(
            inst_particle_system_cfg.get("max_velocity", 2.5)
        )

        # Apply optional particle system APIs
        if inst_particle_system_cfg.get("smoothing", False):
            PhysxSchema.PhysxParticleSmoothingAPI.Apply(self.particle_system.GetPrim())
        if inst_particle_system_cfg.get("anisotropy", False):
            PhysxSchema.PhysxParticleAnisotropyAPI.Apply(self.particle_system.GetPrim())
        if inst_particle_system_cfg.get("isosurface", True):
            PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(self.particle_system.GetPrim())

        # --- 6. Particle Generation and Instancing ---
        # Check whether to generate particles from container mesh
        use_container_mesh = container_cfg.get("use_container_mesh", False)
        cloud_points = None  # Initialize so variable always exists

        if (
            use_container_mesh
            and self.container_prim_path
            and is_prim_path_valid(self.container_prim_path)
        ):
            # Generate particles from container mesh
            container_mesh_path = self._find_first_mesh_in_hierarchy(
                self.container_prim_path
            )
            if container_mesh_path:
                container_mesh = UsdGeom.Mesh.Get(
                    self.stage, Sdf.Path(container_mesh_path)
                )
                if container_mesh:
                    try:
                        cloud_points_base = np.array(
                            container_mesh.GetPointsAttr().Get()
                        )
                        # Container mesh vertices are in container local frame; transform to fluid local frame
                        container_prim = self.stage.GetPrimAtPath(
                            self.container_prim_path
                        )
                        fluid_prim = self.stage.GetPrimAtPath(self.usd_prim_path)

                        if (
                            container_prim
                            and container_prim.IsValid()
                            and fluid_prim
                            and fluid_prim.IsValid()
                        ):
                            # Get transform matrices
                            container_xform = UsdGeom.Xformable(
                                container_prim
                            ).ComputeLocalToWorldTransform(0)
                            fluid_xform = UsdGeom.Xformable(
                                fluid_prim
                            ).ComputeLocalToWorldTransform(0)
                            fluid_xform_inv = fluid_xform.GetInverse()

                            # Transform container vertices into fluid local frame
                            cloud_points_world = []
                            for p in cloud_points_base:
                                world_p = container_xform.Transform(
                                    Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))
                                )
                                local_p = fluid_xform_inv.Transform(world_p)
                                cloud_points_world.append(
                                    [local_p[0], local_p[1], local_p[2]]
                                )
                            cloud_points_base = np.array(
                                cloud_points_world, dtype=np.float32
                            )

                        # Apply scale and volume multiplier
                        visual_scale = (
                            np.array(
                                self.instance_config.get("visual", {}).get(
                                    "scale", [1.0, 1.0, 1.0]
                                ),
                                dtype=np.float32,
                            )
                            if isinstance(
                                self.instance_config.get("visual", {}).get(
                                    "scale", [1.0, 1.0, 1.0]
                                ),
                                (list, tuple, np.ndarray),
                            )
                            else np.array([1.0, 1.0, 1.0], dtype=np.float32)
                        )
                        self.visual_scale = visual_scale
                        fluid_volume_multiplier = self.physics_cfg.get(
                            "fluid_volume", self.physics_cfg.get("fluid_volumn", 1.0)
                        )
                        # For container mesh, scale down slightly to fit interior (avoid particles on walls)
                        container_scale_factor = container_cfg.get(
                            "mesh_scale_factor", 0.95
                        )
                        cloud_points = (
                            cloud_points_base
                            * fluid_volume_multiplier
                            * self.visual_scale
                            * container_scale_factor
                        )
                        carb.log_info(
                            f"Using container mesh from {self.container_prim_path} for particle generation"
                        )
                    except Exception as e:
                        carb.log_warn(
                            f"Error processing container mesh: {e}. Falling back to fluid mesh."
                        )
                        use_container_mesh = False
                else:
                    carb.log_warn(
                        f"Container mesh at {container_mesh_path} is not valid. Falling back to fluid mesh."
                    )
                    use_container_mesh = False
            else:
                carb.log_warn(
                    f"Could not find mesh in container {self.container_prim_path}. Falling back to fluid mesh."
                )
                use_container_mesh = False

        # If use_container_mesh failed or is disabled, use default fluid mesh
        if not use_container_mesh or cloud_points is None:
            # Default: generate particles from fluid USD mesh (original logic)
            fluid_mesh = UsdGeom.Mesh.Get(self.stage, Sdf.Path(self.mesh_prim_path))
            fluid_volume_multiplier = self.physics_cfg.get(
                "fluid_volume", self.physics_cfg.get("fluid_volumn", 1.0)
            )
            cloud_points_base = np.array(fluid_mesh.GetPointsAttr().Get())
            visual_scale = (
                np.array(
                    self.instance_config.get("visual", {}).get(
                        "scale", [1.0, 1.0, 1.0]
                    ),
                    dtype=np.float32,
                )
                if isinstance(
                    self.instance_config.get("visual", {}).get(
                        "scale", [1.0, 1.0, 1.0]
                    ),
                    (list, tuple, np.ndarray),
                )
                else np.array([1.0, 1.0, 1.0], dtype=np.float32)
            )
            self.visual_scale = visual_scale
            cloud_points = (
                cloud_points_base * fluid_volume_multiplier * self.visual_scale
            )

        fluid_rest_offset = inst_particle_system_cfg.get("fluid_rest_offset", 0.0135)
        particleSpacing = 2.0 * fluid_rest_offset

        self.init_particle_positions, self.init_particle_velocities = (
            generate_particles_in_convex_mesh(
                vertices=cloud_points, sphere_diameter=particleSpacing, visualize=False
            )
        )
        self.stage.GetPrimAtPath(self.mesh_prim_path).SetActive(False)

        self.particle_point_instancer_path = Sdf.Path(self.usd_prim_path).AppendChild(
            "particles"
        )

        particleUtils.add_physx_particleset_pointinstancer(
            stage=self.stage,
            path=self.particle_point_instancer_path,
            positions=Vt.Vec3fArray(self.init_particle_positions),
            velocities=Vt.Vec3fArray(self.init_particle_velocities),
            particle_system_path=self.particle_system_path,
            self_collision=True,
            fluid=True,
            particle_group=0,
            particle_mass=0.001,
            density=0.0,
        )

        self.point_instancer = UsdGeom.PointInstancer.Get(
            self.stage, self.particle_point_instancer_path
        )

        init_scale_array = np.array(self.init_scale, dtype=np.float32)
        combined_scale = init_scale_array * self.visual_scale

        physicsUtils.set_or_add_scale_orient_translate(
            self.point_instancer,
            translate=Gf.Vec3f([float(v) for v in self.init_pos]),
            orient=Gf.Quatf(
                float(self.init_ori[0]),
                Gf.Vec3f(
                    float(self.init_ori[1]),
                    float(self.init_ori[2]),
                    float(self.init_ori[3]),
                ),
            ),
            scale=Gf.Vec3f([float(v) for v in combined_scale]),
        )

        proto_path = self.particle_point_instancer_path.AppendChild(
            "particlePrototype0"
        )
        self.particle_prototype_path = proto_path

        particle_prototype_sphere = UsdGeom.Sphere.Get(self.stage, proto_path)
        particle_prototype_sphere.CreateRadiusAttr().Set(fluid_rest_offset)
        if inst_particle_system_cfg.get("isosurface", True):
            UsdGeom.Imageable(particle_prototype_sphere).MakeInvisible()

        # --- 7. Initial Material and Physics Properties Setup ---
        self._apply_random_material()

        # Handle semantic labels
        self._handle_semantic_labels()

    def _apply_random_material(self):
        """
        Selects a random material from configuration, creates it, and binds it
        to the particle system and particle prototypes. Also sets physics properties.
        """
        visual_cfg = self.instance_config.get("visual", {})
        material_cfg = visual_cfg.get("visual_material", {})
        material_list = material_cfg.get("material_usd_folder", [])

        if material_list:
            material_url = random.choice(material_list)
        else:
            material_url = "./Assets/Material/Base/Textiles/Linen_Blue.mdl"

        material_name = os.path.splitext(os.path.basename(material_url))[0]
        looks_path = f"{os.path.dirname(self.prim_path)}/Looks"  # Consistent Looks path
        if is_prim_path_valid(f"{looks_path}/material"):  # Check specific material path
            delete_prim(f"{looks_path}/material")

        unique_material_name = find_unique_string_name(
            initial_name=f"{looks_path}/material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        color_material_path = unique_material_name
        create_mdl_material(material_url, material_name, color_material_path)

        inst_particle_material_cfg = self.physics_cfg.get("particle_material", {})
        particleUtils.add_pbd_particle_material(
            stage=self.stage,
            path=color_material_path,
            adhesion=inst_particle_material_cfg.get("adhesion"),
            adhesion_offset_scale=inst_particle_material_cfg.get(
                "adhesion_offset_scale"
            ),
            cohesion=inst_particle_material_cfg.get("cohesion"),
            particle_adhesion_scale=inst_particle_material_cfg.get(
                "particle_adhesion_scale"
            ),
            particle_friction_scale=inst_particle_material_cfg.get(
                "particle_friction_scale"
            ),
            drag=inst_particle_material_cfg.get("drag"),
            lift=inst_particle_material_cfg.get("lift"),
            friction=inst_particle_material_cfg.get("friction"),
            damping=inst_particle_material_cfg.get("damping"),
            gravity_scale=inst_particle_material_cfg.get("gravity_scale", 1.0),
            viscosity=inst_particle_material_cfg.get("viscosity"),
            vorticity_confinement=inst_particle_material_cfg.get(
                "vorticity_confinement"
            ),
            surface_tension=inst_particle_material_cfg.get("surface_tension"),
            density=inst_particle_material_cfg.get("density"),
            cfl_coefficient=inst_particle_material_cfg.get("cfl_coefficient"),
        )

        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=self.particle_system_path,
            material_path=color_material_path,
        )

        if hasattr(self, "particle_prototype_path") and is_prim_path_valid(
            self.particle_prototype_path
        ):
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.particle_prototype_path,
                material_path=color_material_path,
            )

    def _re_instance_name(self, inst_name):
        """Reformats the instance name to ensure consistent numbering."""
        parts = inst_name.split("_")
        cat_name_extracted = "_".join(parts[:-1])
        obj_id_str = parts[-1]
        obj_id = int(obj_id_str)
        original_id = (obj_id - 1) % self.num_per_env + 1
        return f"{cat_name_extracted}_{original_id}"

    def initialize(self):
        """Initialize the fluid container position."""
        # Ensure container exists before setting pose
        if (
            hasattr(self, "container")
            and self.container
            and self.container_position is not None
        ):
            self.container.set_local_pose(translation=self.container_position)
        else:
            print(
                f"Warning: Container not initialized for {self.prim_path}, cannot set initial pose."
            )

    def reset(self, soft=False):
        """
        Reset the fluid system to initial state with new position and orientation.

        Args:
            soft: If True, use soft reset ranges; otherwise use initial ranges
        """
        self._apply_random_material()

        if not self.layout_manager:
            raise RuntimeError(
                f"LayoutManager is required for {self.prim_path}. All position information must come from LayoutManager."
            )

        env_id = self._extract_env_id_from_prim_path()
        if env_id is None:
            print(
                f"Warning: Could not extract env_id for {self.prim_path}. Cannot perform reset."
            )
            return

        reset_type = "soft" if soft else "hard"
        new_layout = self.layout_manager.generate_new_layout(
            env_id=env_id, prim_path=self.usd_prim_path, reset_type=reset_type
        )

        if not new_layout:
            print(
                f"Warning: LayoutManager did not provide new layout for {self.prim_path}. Cannot perform reset."
            )
            return

        pos = new_layout["pos"]
        ori_quat = new_layout["ori"]

        # Update container pose - ensure type conversion for Gf.Vec3d
        if self.container is not None:
            self.container_position = Gf.Vec3d(
                float(pos[0] + self.container_offset[0]),
                float(pos[1] + self.container_offset[1]),
                float(self.container_offset[2]),
            )
        # Ensure container exists before setting pose
        if (
            hasattr(self, "container")
            and self.container
            and self.container_position is not None
        ):
            # Assuming container uses local pose relative to env root
            self.container.set_local_pose(
                translation=self.container_position
            )  # Orientation might not be needed for container usually
        else:
            print(
                f"Warning: Container not initialized for {self.prim_path}, cannot reset pose."
            )

        # Reset particle positions and instancer transform
        self.set_particle_positions(self.init_particle_positions)

        # Ensure type conversion for Gf.Vec3f and Gf.Quatf
        physicsUtils.set_or_add_translate_op(
            self.point_instancer, translate=Gf.Vec3f([float(v) for v in pos])
        )

        physicsUtils.set_or_add_orient_op(
            self.point_instancer,
            orient=Gf.Quatf(
                float(ori_quat[0]),
                Gf.Vec3f(float(ori_quat[1]), float(ori_quat[2]), float(ori_quat[3])),
            ),
        )

    def get_particle_positions(self, visualize: bool = True):
        """
        Get current positions of all fluid particles.

        Args:
            visualize: Whether to visualize particles using Open3D

        Returns:
            positions: Array of particle positions
        """
        # Ensure point_instancer is valid before accessing attributes
        if not hasattr(self, "point_instancer") or not self.point_instancer:
            print(
                f"Warning: point_instancer not valid for {self.prim_path}. Cannot get positions."
            )
            return np.array([]), None, None  # Return empty array

        positions_attr = self.point_instancer.GetPositionsAttr()
        if not positions_attr:
            print(
                f"Warning: Could not get PositionsAttr for {self.particle_point_instancer_path}."
            )
            return np.array([]), None, None

        positions = np.array(positions_attr.Get(), dtype=np.float32)

        if visualize:
            if positions.size == 0:
                print("Warning: No particle positions to visualize.")
                return positions, None, None
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(positions)
                o3d.visualization.draw_geometries([pcd])
            except Exception as e:
                print(f"Error during visualization: {e}")

        # Fluid doesn't have a single world pose like rigid objects
        return positions, None, None

    def set_particle_positions(self, positions: np.ndarray):
        """
        Set positions of all fluid particles.

        Args:
            positions: Array of new particle positions
        """
        # Ensure point_instancer is valid before setting attributes
        if not hasattr(self, "point_instancer") or not self.point_instancer:
            print(
                f"Warning: point_instancer not valid for {self.prim_path}. Cannot set positions."
            )
            return

        # Ensure positions is a numpy array
        if not isinstance(positions, np.ndarray):
            try:
                # Attempt conversion if it's list-like (e.g., list of Gf.Vec3f from init)
                if positions and isinstance(positions[0], Gf.Vec3f):
                    positions = np.array(
                        [[p[0], p[1], p[2]] for p in positions], dtype=np.float32
                    )
                else:
                    positions = np.array(positions, dtype=np.float32)
            except Exception as e:
                print(
                    f"Error converting positions to numpy array: {e}. Positions type: {type(positions)}"
                )
                return

        if positions.ndim != 2 or positions.shape[1] != 3:
            print(
                f"Error: Invalid shape for positions array: {positions.shape}. Expected (N, 3)."
            )
            return

        # Convert numpy array to Vt.Vec3fArray, ensuring float type
        try:
            positions_vt = Vt.Vec3fArray.FromNumpy(positions.astype(np.float32))
        except Exception as e:
            print(f"Error converting numpy array to Vt.Vec3fArray: {e}")
            return

        positions_attr = self.point_instancer.GetPositionsAttr()
        if not positions_attr:
            print(
                f"Warning: Could not get PositionsAttr for {self.particle_point_instancer_path} to set positions."
            )
            return

        positions_attr.Set(positions_vt)

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
            "fluid_volumn",
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

        # Randomize particle_system parameters
        if "particle_system" in modified_config:
            particle_system = modified_config["particle_system"].copy()
            particle_system_params_to_randomize = [
                "particle_contact_offset",
                "contact_offset",
                "rest_offset",
                "fluid_rest_offset",
                "solid_rest_offset",
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
                "density",
                "cfl_coefficient",
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

    def _get_env_root_from_prim_path(self):
        """Get current env root path from prim_path, e.g. /World/envs/env_0"""
        parts = self.usd_prim_path.split("/")
        for i, part in enumerate(parts):
            if part.startswith("env_"):
                return "/".join(parts[: i + 1])
        return None

    def _find_first_mesh_in_hierarchy(self, prim_path: str) -> str:
        """Recursively find the first UsdGeom.Mesh prim under the given path."""
        from isaacsim.core.utils.prims import get_prim_at_path
        from pxr import Usd

        start_prim = get_prim_at_path(prim_path)
        if not start_prim:
            return None

        for prim in Usd.PrimRange(start_prim):
            if prim.IsA(UsdGeom.Mesh):
                return prim.GetPath().pathString

        return None

    def _handle_semantic_labels(self):
        """Manage semantic labeling: clear existing labels and apply new ones."""

        # Apply semantic labels to the particle point instancer (where the actual geometry is)
        prim = self.stage.GetPrimAtPath(self.particle_point_instancer_path)
        if prim and prim.IsValid():
            remove_labels(prim, include_descendants=True)
            semantic_label = self._get_semantic_label()
            if semantic_label:
                add_labels(prim, [semantic_label])
                self.semantic_label = semantic_label

    def _get_semantic_label(self) -> str:
        """Generate semantic label from configuration or USD filename.

        Priority: (1) category semantic_label if non-empty, (2) regex on USD filename.
        semantic_regex_pattern / semantic_regex_repl must be set at category level (e.g. fluid_items).
        """
        semantic_label = self.category_config.get("semantic_label")
        if (
            semantic_label
            and isinstance(semantic_label, str)
            and semantic_label.strip()
        ):
            return semantic_label.strip()

        if not self.usd_path:
            return ""

        regex_pattern = self.category_config.get("semantic_regex_pattern", r".*")
        regex_replacement = self.category_config.get("semantic_regex_repl", r"\g<0>")
        filename = os.path.basename(self.usd_path)
        filename_without_ext = os.path.splitext(filename)[0]
        return re.sub(regex_pattern, regex_replacement, filename_without_ext)

    def get_state(self, is_relative: bool = False) -> dict[str, torch.Tensor]:
        """Get the state of the fluid object.

        Args:
            is_relative: If True, positions are relative to environment origin. Defaults to False.

        Returns:
            Dictionary containing:
                - particle_positions: torch.Tensor, shape (num_particles, 3), particle positions
                - particle_velocities: torch.Tensor, shape (num_particles, 3), particle velocities
                - asset_info: dict with usd_path and primitive_type
        """
        try:
            positions, _, _ = self.get_particle_positions(visualize=False)
            if positions is None or (
                isinstance(positions, np.ndarray) and positions.size == 0
            ):
                positions = torch.zeros(0, 3, dtype=torch.float32)
            elif not isinstance(positions, torch.Tensor):
                positions = torch.tensor(positions, dtype=torch.float32)
            if positions.dim() == 1 and positions.shape[0] == 3:
                positions = positions.unsqueeze(0)
            elif positions.dim() == 0 or positions.numel() == 0:
                positions = torch.zeros(0, 3, dtype=torch.float32)
        except (AttributeError, RuntimeError):
            positions = torch.zeros(0, 3, dtype=torch.float32)

        if is_relative and hasattr(self, "env_origin") and positions.numel() > 0:
            env_origin_tensor = (
                torch.tensor(
                    self.env_origin, dtype=torch.float32, device=positions.device
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
            positions[:, :3] -= env_origin_tensor[:3]

        try:
            velocities = self.get_particle_velocities()
            if not isinstance(velocities, torch.Tensor):
                velocities = torch.tensor(
                    velocities, dtype=torch.float32, device=positions.device
                )
            if velocities.dim() == 1 and velocities.shape[0] == 3:
                velocities = velocities.unsqueeze(0)
            elif velocities.dim() == 0 or velocities.numel() == 0:
                velocities = torch.zeros_like(positions)
        except (AttributeError, RuntimeError):
            velocities = torch.zeros_like(positions)

        asset_info = {
            "usd_path": self.usd_path if hasattr(self, "usd_path") else None,
            "primitive_type": None,
        }

        return {
            "particle_positions": positions,
            "particle_velocities": velocities,
            "asset_info": asset_info,
        }


def generate_particles_in_convex_mesh(
    vertices: np.ndarray, sphere_diameter: float, visualize: bool = False
):
    """
    Generate particles within a convex mesh using Delaunay triangulation.

    Args:
        vertices: Vertices of the convex mesh
        sphere_diameter: Diameter of particles to generate
        visualize: Whether to visualize the particles and mesh vertices

    Returns:
        List of particle positions and velocities (zero-initialized)
    """
    # Ensure vertices is a numpy array
    if not isinstance(vertices, np.ndarray):
        vertices = np.array(vertices)

    # Check for sufficient vertices
    if vertices.shape[0] < 4:
        print(
            "Warning: Need at least 4 vertices for Delaunay triangulation. Returning empty."
        )
        return [], []

    try:
        min_bound = np.min(vertices, axis=0)
        max_bound = np.max(vertices, axis=0)

        # Add small jitter if points are coplanar or degenerate
        if np.linalg.matrix_rank(vertices) < 3:
            vertices += np.random.rand(*vertices.shape) * 1e-6

        hull = Delaunay(vertices)
    except Exception as e:
        print(
            f"Error during Delaunay triangulation: {e}. Vertices shape: {vertices.shape}. Returning empty."
        )
        return [], []

    # Create grid of sample points
    # Add a small epsilon to max_bound to include boundary points if desired
    epsilon = sphere_diameter * 0.01
    x_vals = np.arange(min_bound[0], max_bound[0] + epsilon, sphere_diameter)
    y_vals = np.arange(min_bound[1], max_bound[1] + epsilon, sphere_diameter)
    z_vals = np.arange(min_bound[2], max_bound[2] + epsilon, sphere_diameter)

    # Handle cases where ranges might be empty
    if x_vals.size == 0 or y_vals.size == 0 or z_vals.size == 0:
        print("Warning: Empty dimension range for particle grid. Returning empty.")
        return [], []

    samples = np.stack(
        np.meshgrid(x_vals, y_vals, z_vals, indexing="ij"), axis=-1
    ).reshape(-1, 3)

    # Find points inside the convex hull
    # Use tolerance to handle points near the boundary
    inside_mask = hull.find_simplex(samples, tol=1e-6) >= 0
    inside_points = samples[inside_mask]

    # Initialize velocities to zero
    velocity = np.zeros_like(inside_points)

    # Visualization
    if visualize:
        if inside_points.size == 0:
            print("Warning: No inside points found to visualize.")
        else:
            try:
                particle_pcd = o3d.geometry.PointCloud()
                particle_pcd.points = o3d.utility.Vector3dVector(inside_points)
                particle_pcd.paint_uniform_color([0.2, 0.4, 1.0])

                vertex_pcd = o3d.geometry.PointCloud()
                vertex_pcd.points = o3d.utility.Vector3dVector(vertices)
                vertex_pcd.paint_uniform_color([1.0, 0.1, 0.1])

                o3d.visualization.draw_geometries(
                    [particle_pcd, vertex_pcd],
                    window_name="Convex Mesh Particle Filling",
                )
            except Exception as e:
                print(f"Error during visualization: {e}")

    # Convert numpy points/velocities to list of Gf.Vec3f
    positions_gf = [
        Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in inside_points
    ]
    velocities_gf = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in velocity]

    return positions_gf, velocities_gf
