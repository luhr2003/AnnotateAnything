import random
import re
import os
import omni.kit.commands
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.prims import (
    is_prim_path_valid,
    delete_prim,
    get_prim_at_path,
)
from isaacsim.core.utils.semantics import add_labels, remove_labels
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Vt, Sdf
from omni.physx.scripts import physicsUtils, particleUtils
from omegaconf import DictConfig
import omni.usd
import numpy as np
import torch
from isaacsim.core.api.materials import PreviewSurface
from isaacsim.core.utils.string import find_unique_string_name
from magicsim.Env.Utils.path import (
    get_usd_paths_from_folder,
    resolve_mdl_paths,
    resolve_path,
)
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.replicator.behavior.utils.scene_utils import create_mdl_material


class RopeObject:
    """
    RopeObject class. This version uses a robust reset mechanism that deletes and
    recreates the internal physics components to ensure a clean state.
    It correctly handles multi-environment offsets by relying on the scene graph transform hierarchy.
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
        self.stage = get_current_stage()
        self.prim_path = prim_path
        self.point_instancer_path = f"{self.prim_path}/rigidBodyInstancer"
        self.joint_instancer_path = f"{self.prim_path}/jointInstancer"

        prim_path_parts = prim_path.split("/")
        self.category_name = prim_path_parts[-2]
        instance_name_from_path = prim_path_parts[-1]
        self.num_per_env = config.objects[self.category_name].get("num_per_env")

        self.instance_name = self._re_instance_name(instance_name_from_path)

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
        self.visual_cfg = self.instance_config.get("visual", {})
        self.physics_cfg = self.instance_config.get("physics", {})

        # Apply ratio-based randomization to physics parameters
        self.physics_cfg = self._apply_physics_ratio_randomization(self.physics_cfg)

        self.layout_manager = layout_manager
        self.layout_info = layout_info
        self.env_origin = env_origin.detach().cpu().numpy()
        self.usd_path = usd_path

        # Read the 'visible' property from the config, default to True if not specified
        self.visible = self.visual_cfg.get("visible", True)
        self.color_list = self.visual_cfg.get("color")
        if self.color_list is not None and isinstance(self.color_list[0], (int, float)):
            self.color_list = [self.color_list]
        self._current_color = None
        self._current_material_path = None
        self.visual_material_mdl_path = None
        self.visual_material_mdl_folder = None
        self._current_mdl_path = None

        inst_visual_material_cfg_val = self.visual_cfg.get("visual_material")
        self.inst_visual_material_cfg = (
            inst_visual_material_cfg_val
            if inst_visual_material_cfg_val is not None
            else {}
        )

        if self.color_list:
            # (Priority 1: Color List)
            self._current_color = random.choice(self.color_list)
        else:
            # (Priority 2: Single Color)
            color = self.visual_cfg.get("color")
            if color is not None:
                self._current_color = color
            else:  # color is None
                # (Priority 3: MDL Material)
                self.visual_material_mdl_path = self.inst_visual_material_cfg.get(
                    "mdl_path"
                )
                self.visual_material_mdl_folder = self.inst_visual_material_cfg.get(
                    "mdl_folder"
                )

                if self.visual_material_mdl_path:
                    self._current_mdl_path = self.visual_material_mdl_path
                elif self.visual_material_mdl_folder:
                    resolved_mdl_paths = resolve_mdl_paths(
                        self.visual_material_mdl_folder
                    )
                    if resolved_mdl_paths:
                        self._current_mdl_path = random.choice(resolved_mdl_paths)
                    else:
                        print(
                            f"⚠️ Warning: No MDL files found in folder: {self.visual_material_mdl_folder}"
                        )
                else:
                    # (Priority 4: USD Material Folder)
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
                            # Select and store random material path
                            selected_indices = torch.randint(
                                low=0,
                                high=len(self.visual_usd_paths),
                                size=(1,),
                            ).tolist()
                            self._current_material_path = self.visual_usd_paths[
                                selected_indices[0]
                            ]
                        else:
                            print(
                                f"Warning: No material USDs found in {self.visual_material_usd_folder} for Rope {self.prim_path}"
                            )

    def initialize(self):
        """
        Called by SceneManager after the object is first created.
        Gets initial pose and creates the rope instance.
        """
        if self.layout_info:
            # Use provided layout info
            init_pos = self.layout_info["pos"]
            init_ori = self.layout_info["ori"]
            init_scale = self.layout_info["scale"]
        else:
            # Must have layout_manager
            if not self.layout_manager:
                raise RuntimeError(
                    f"LayoutManager is required for {self.prim_path}. All position information must come from LayoutManager."
                )

            env_id = self._extract_env_id_from_prim_path()
            if env_id is None:
                raise ValueError(
                    f"Could not extract env_id from prim path: {self.prim_path}"
                )

            layout_info = self.layout_manager.get_object_layout(
                env_id=env_id, prim_path=self.prim_path
            )
            if layout_info is None:
                raise RuntimeError(
                    f"LayoutManager failed to generate/retrieve layout for {self.prim_path}"
                )

            init_pos = layout_info["pos"]
            init_ori = layout_info["ori"]
            init_scale = layout_info["scale"]

        initial_color = self._current_color
        self._recreate_rope_instance(init_pos, init_ori, init_scale, initial_color)

        # Handle semantic labels
        self._handle_semantic_labels()

    def hide_prim(self, prim_path: str):
        try:
            path = Sdf.Path(prim_path)
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                return
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()
        except Exception as e:
            print(f"Warning: Failed to hide prim {prim_path}: {e}")

    def _re_instance_name(self, inst_name):
        parts = inst_name.split("_")
        cat_name_extracted = "_".join(parts[:-1])
        obj_id_str = parts[-1]
        obj_id = int(obj_id_str)
        original_id = (obj_id - 1) % self.num_per_env + 1
        return f"{cat_name_extracted}_{original_id}"

    def reset(self, soft=False):
        """
        Performs a thorough reset by recreating the rope at a new LOCAL pose generated by LayoutManager.
        """
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
            env_id=env_id, prim_path=self.prim_path, reset_type=reset_type
        )

        if new_layout:
            pos = new_layout["pos"]
            ori = new_layout["ori"]
            scale = new_layout["scale"]

            reset_color = self._current_color
            self._recreate_rope_instance(pos, ori, np.array(scale), reset_color)
        else:
            print(
                f"Warning: LayoutManager did not provide new layout for {self.prim_path}. Cannot perform reset."
            )
            return

    def reset_hard(self, soft=False):
        """
        Performs a thorough reset by recreating the rope at a new LOCAL pose generated by LayoutManager.
        """
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
            env_id=env_id, prim_path=self.prim_path, reset_type=reset_type
        )

        if new_layout:
            pos = new_layout["pos"]
            ori = new_layout["ori"]
            scale = new_layout["scale"]

            reset_color = random.choice(self.color_list)
            self._recreate_rope_instance(pos, ori, np.array(scale), reset_color)
        else:
            print(
                f"Warning: LayoutManager did not provide new layout for {self.prim_path}. Cannot perform reset."
            )
            return

    def _extract_env_id_from_prim_path(self):
        """get env_id from prim_path"""
        try:
            parts = self.prim_path.split("/")
            for part in parts:
                if part.startswith("env_"):
                    return int(part.split("_")[1])
        except (ValueError, IndexError):
            pass
        return None

    def _calculate_local_space_poses(self):
        """
        Calculates the local positions and orientations of the capsule segments relative to the rope's origin (0,0,0).
        """
        ropeLength = self.physics_cfg.get("rope_length")
        numLinks = self.physics_cfg.get("num_links")
        single_link_length = ropeLength / numLinks
        linkHalfLength = single_link_length / 2.0
        linkRadius = self.physics_cfg.get("link_radius", 0.02)
        max_radius = linkHalfLength * 0.9
        if linkRadius > max_radius:
            linkRadius = max_radius
        linkSeparation = 2.0 * linkHalfLength - linkRadius
        xStart = -numLinks * linkSeparation * 0.5

        positions = []
        for linkInd in range(numLinks):
            local_pos = Gf.Vec3f(xStart + linkInd * linkSeparation, 0, 0)
            positions.append(local_pos)

        orientations = [Gf.Quath(1.0, 0.0, 0.0, 0.0)] * numLinks

        return positions, orientations

    def _recreate_rope_instance(self, pos, ori, scale, color=None):
        """
        Helper method to delete old rope components and create new ones based on a given LOCAL pose and color.
        """
        if is_prim_path_valid(self.point_instancer_path):
            delete_prim(self.point_instancer_path)
        if is_prim_path_valid(self.joint_instancer_path):
            delete_prim(self.joint_instancer_path)

        local_positions, local_orientations = self._calculate_local_space_poses()

        rboInstancer = UsdGeom.PointInstancer.Define(
            self.stage, self.point_instancer_path
        )

        instancer_prim = self.stage.GetPrimAtPath(self.point_instancer_path)

        if not self.visible:
            imageable = UsdGeom.Imageable(instancer_prim)
            imageable.MakeInvisible()

        xformable = UsdGeom.Xformable(instancer_prim)
        physicsUtils.set_or_add_scale_orient_translate(
            xformable,
            scale=Gf.Vec3f([float(v) for v in scale]),
            orient=Gf.Quatf(
                float(ori[0]), Gf.Vec3f(float(ori[1]), float(ori[2]), float(ori[3]))
            ),
            translate=Gf.Vec3f([float(v) for v in pos]),
        )

        ropeLength = self.physics_cfg.get("rope_length")
        numLinks = self.physics_cfg.get("num_links")
        single_link_length = ropeLength / numLinks
        linkHalfLength = single_link_length / 2.0
        linkRadius = self.physics_cfg.get("link_radius", 0.02)
        max_radius = linkHalfLength * 0.9
        if linkRadius > max_radius:
            linkRadius = max_radius

        capsulePath = f"{self.point_instancer_path}/capsulePrototype"
        self._create_capsule_prototype(capsulePath, linkHalfLength, linkRadius, color)

        rboInstancer.GetPrototypesRel().AddTarget(capsulePath)
        rboInstancer.GetProtoIndicesAttr().Set([0] * numLinks)
        rboInstancer.GetPositionsAttr().Set(local_positions)
        rboInstancer.GetOrientationsAttr().Set(local_orientations)

        jointInstancer = PhysxSchema.PhysxPhysicsJointInstancer.Define(
            self.stage, self.joint_instancer_path
        )
        jointPath = f"{self.joint_instancer_path}/jointPrototype"
        self._create_joint_prototype(jointPath)

        body0indices = list(range(numLinks - 1))
        body1indices = list(range(1, numLinks))

        jointX = linkHalfLength - 0.5 * linkRadius
        localPos0 = [Gf.Vec3f(jointX, 0, 0)] * (numLinks - 1)
        localPos1 = [Gf.Vec3f(-jointX, 0, 0)] * (numLinks - 1)
        localRot = [Gf.Quath(1.0)] * (numLinks - 1)

        jointInstancer.GetPhysicsPrototypesRel().AddTarget(jointPath)
        jointInstancer.GetPhysicsProtoIndicesAttr().Set([0] * (numLinks - 1))
        jointInstancer.GetPhysicsBody0sRel().SetTargets([self.point_instancer_path])
        jointInstancer.GetPhysicsBody0IndicesAttr().Set(body0indices)
        jointInstancer.GetPhysicsLocalPos0sAttr().Set(localPos0)
        jointInstancer.GetPhysicsLocalRot0sAttr().Set(localRot)
        jointInstancer.GetPhysicsBody1sRel().SetTargets([self.point_instancer_path])
        jointInstancer.GetPhysicsBody1IndicesAttr().Set(body1indices)
        jointInstancer.GetPhysicsLocalPos1sAttr().Set(localPos1)
        jointInstancer.GetPhysicsLocalRot1sAttr().Set(localRot)

    def _create_joint_prototype(self, jointPath):
        joint = UsdPhysics.Joint.Define(self.stage, jointPath)
        d6Prim = joint.GetPrim()
        coneAngleLimit = self.physics_cfg.get("cone_angle_limit", 110.0)
        rope_damping = self.physics_cfg.get("damping", 10.0)
        rope_stiffness = self.physics_cfg.get("stiffness", 1.0)
        for dof in ["transX", "transY", "transZ", "rotX"]:
            limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, dof)
            limitAPI.CreateLowAttr(1.0)
            limitAPI.CreateHighAttr(-1.0)
        for dof in ["rotY", "rotZ"]:
            limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, dof)
            limitAPI.CreateLowAttr(-coneAngleLimit)
            limitAPI.CreateHighAttr(coneAngleLimit)
            driveAPI = UsdPhysics.DriveAPI.Apply(d6Prim, dof)
            driveAPI.CreateTypeAttr("force")
            driveAPI.CreateDampingAttr(rope_damping)
            driveAPI.CreateStiffnessAttr(rope_stiffness)

    def _create_capsule_prototype(self, path, linkHalfLength, linkRadius, color=None):
        capsuleGeom = UsdGeom.Capsule.Define(self.stage, path)
        capsuleGeom.CreateHeightAttr(linkHalfLength)
        capsuleGeom.CreateRadiusAttr(linkRadius)
        capsuleGeom.CreateAxisAttr("X")

        if color is not None:
            # --- (Priority 1: Color) ---
            material_path = find_unique_string_name(
                initial_name=f"{path}/Looks/color_material",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )
            material_prim = get_prim_at_path(material_path)
            if not material_prim:
                material = PreviewSurface(
                    prim_path=material_path, color=torch.tensor(color)
                )
                omni.kit.commands.execute(
                    "BindMaterialCommand",
                    prim_path=path,
                    material_path=material_path,
                    strength=UsdShade.Tokens.strongerThanDescendants,
                )
            else:
                material = PreviewSurface(prim_path=material_path)
                material.set_color(np.array(color))

        # --- (Priority 3: MDL Material) ---
        elif hasattr(self, "_current_mdl_path") and self._current_mdl_path is not None:
            self._apply_mdl_material_to_capsule(path, self._current_mdl_path)

        # --- (Priority 4: USD Material) ---
        elif (
            hasattr(self, "_current_material_path")
            and self._current_material_path is not None
        ):
            visual_material_prim_path = find_unique_string_name(
                initial_name=f"{path}/Looks/visual_material",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )
            add_reference_to_stage(
                usd_path=self._current_material_path,
                prim_path=visual_material_prim_path,
            )
            visual_material_prim = get_prim_at_path(visual_material_prim_path)

            if visual_material_prim and visual_material_prim.IsValid():
                children = visual_material_prim.GetChildren()
                if children:
                    material_prim_path = children[0].GetPath()
                    omni.kit.commands.execute(
                        "BindMaterialCommand",
                        prim_path=path,
                        material_path=material_prim_path,
                        strength=UsdShade.Tokens.strongerThanDescendants,
                    )
                else:
                    print(
                        f"Warning: Material prim at {visual_material_prim_path} has no children."
                    )
            else:
                print(
                    f"Warning: Could not get valid prim at {visual_material_prim_path}"
                )

        UsdPhysics.CollisionAPI.Apply(capsuleGeom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(capsuleGeom.GetPrim())
        massAPI = UsdPhysics.MassAPI.Apply(capsuleGeom.GetPrim())
        massAPI.CreateDensityAttr().Set(self.physics_cfg.get("density", 0.00005))

        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(capsuleGeom.GetPrim())
        physxCollisionAPI.CreateRestOffsetAttr().Set(0.0)
        contactOffset = self.physics_cfg.get("contact_offset", 0.005)
        physxCollisionAPI.CreateContactOffsetAttr().Set(contactOffset)
        physicsMaterialPath = f"{self.prim_path}/PhysicsMaterial"
        if not is_prim_path_valid(physicsMaterialPath):
            UsdShade.Material.Define(self.stage, physicsMaterialPath)
            material = UsdPhysics.MaterialAPI.Apply(
                self.stage.GetPrimAtPath(physicsMaterialPath)
            )
            material.CreateStaticFrictionAttr().Set(
                self.physics_cfg.get("static_friction", 0.5)
            )
            material.CreateDynamicFrictionAttr().Set(
                self.physics_cfg.get("dynamic_friction", 0.5)
            )
            material.CreateRestitutionAttr().Set(self.physics_cfg.get("restitution", 0))
        physicsUtils.add_physics_material_to_prim(
            self.stage, capsuleGeom.GetPrim(), physicsMaterialPath
        )

    def _apply_mdl_material_to_capsule(
        self, capsule_path: str, mdl_path: str, mdl_name: str = None
    ):
        """Apply an MDL material to a capsule prim.

        Args:
            capsule_path: Path to the capsule prim
            mdl_path: Path to the MDL file
            mdl_name: Name of the material in the MDL file (optional)
        """
        resolved_mdl_path = resolve_path(mdl_path)
        if not resolved_mdl_path:
            print(f"Warning: MDL material path not found: {mdl_path}")
            return

        if not mdl_name:
            mdl_name = os.path.splitext(os.path.basename(resolved_mdl_path))[0]

        # Create unique material path under capsule's Looks
        material_path = find_unique_string_name(
            initial_name=f"{capsule_path}/Looks/mdl_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )

        try:
            # Create the MDL material
            create_mdl_material(resolved_mdl_path, mdl_name, material_path)

            # Bind material to the capsule prim
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=capsule_path,
                material_path=material_path,
                strength=UsdShade.Tokens.strongerThanDescendants,
            )

        except Exception as e:
            print(
                f"Warning: Failed to apply MDL material {mdl_path} to capsule {capsule_path}: {e}"
            )

    def get_capsule_positions(self):
        """
        Correctly retrieve the world positions of the capsules from the PointInstancer.
        """
        if not is_prim_path_valid(self.point_instancer_path):
            return None

        try:
            computed_positions = particleUtils.get_particle_instancer_positions(
                self.stage, self.point_instancer_path
            )
            if computed_positions is not None and len(computed_positions) > 0:
                return np.array(computed_positions, dtype=np.float32)
        except (ImportError, AttributeError):
            print(
                "Warning: particleUtils.get_particle_instancer_positions not available or failed. Falling back to USD attributes."
            )
            pass  # Fallback to reading USD attributes

        instancer_prim = self.stage.GetPrimAtPath(self.point_instancer_path)
        point_instancer = UsdGeom.PointInstancer(instancer_prim)
        local_positions = point_instancer.GetPositionsAttr().Get()
        if not local_positions:
            return np.array([])
        world_transform = omni.usd.get_world_transform_matrix(instancer_prim)
        world_positions = [
            list(world_transform.Transform(Gf.Vec3d(pos))) for pos in local_positions
        ]
        return np.array(world_positions, dtype=np.float32)

    def set_capsule_positions(self, world_positions: np.ndarray):
        """
        Sets the world positions of each instance in the PointInstancer.
        Note: This overrides physics simulation and is mainly for state resetting.
        """
        if not is_prim_path_valid(self.point_instancer_path):
            return

        instancer_prim = self.stage.GetPrimAtPath(self.point_instancer_path)
        point_instancer = UsdGeom.PointInstancer(instancer_prim)

        world_transform = omni.usd.get_world_transform_matrix(instancer_prim)
        inverse_transform = world_transform.GetInverse()

        local_positions = [
            Gf.Vec3f(
                inverse_transform.Transform(
                    Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))
                )
            )
            for p in world_positions
        ]

        point_instancer.GetPositionsAttr().Set(Vt.Vec3fArray(local_positions))

    def get_current_mesh_points(
        self,
        visualize: bool = True,
        save: bool = False,
        save_path: str = "./pointcloud.ply",
    ):
        """
        Get the world-space point cloud of rope capsules generated by PointInstancer.

        Aligned with Fluid semantics: there is no single local mesh frame.
        Returns (positions, None, None), where positions is an Nx3 array in world coordinates.
        """
        positions = self.get_capsule_positions()
        if positions is None:
            positions = np.array([])

        if visualize or save:
            try:
                import open3d as o3d

                if positions.size > 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(positions)
                    if visualize:
                        o3d.visualization.draw_geometries([pcd])
                    if save:
                        o3d.io.write_point_cloud(save_path, pcd)
            except Exception as e:
                print(f"Error during visualization/saving rope point cloud: {e}")

        # Align with Fluid.get_particle_positions: no single object pose to return
        return positions, None, None

    def set_current_mesh_points(self, mesh_points, pos_world=None, ori_world=None):
        """
        Set the world-space positions of rope capsules (overrides current instance positions).

        Semantics match Fluid: pass the world-space Nx3 array directly.
        """
        if mesh_points is None:
            return
        if isinstance(mesh_points, np.ndarray):
            positions = mesh_points
        else:
            try:
                positions = np.array(mesh_points, dtype=np.float32)
            except Exception:
                print("Warning: invalid mesh_points for rope set_current_mesh_points")
                return
        self.set_capsule_positions(positions)

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
            "rope_length",
            "link_radius",
            "cone_angle_limit",
            "damping",
            "stiffness",
            "density",
            "contact_offset",
            "static_friction",
            "dynamic_friction",
            "restitution",
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
                    # Handle list values
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

        return modified_config

    def destroy(self):
        """Hides the prims associated with the rope to effectively remove it from the scene."""
        self.hide_prim(self.point_instancer_path)
        self.hide_prim(self.joint_instancer_path)

    def _handle_semantic_labels(self):
        """Manage semantic labeling: clear existing labels and apply new ones."""

        # Apply semantic labels to the point instancer (where the actual geometry is)
        prim = get_prim_at_path(self.point_instancer_path)
        if prim and prim.IsValid():
            remove_labels(prim, include_descendants=True)
            semantic_label = self._get_semantic_label()
            if semantic_label:
                add_labels(prim, [semantic_label])
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

    def get_state(self, is_relative: bool = False) -> dict[str, torch.Tensor]:
        """Get the state of the rope object.

        Args:
            is_relative: If True, positions are relative to environment origin. Defaults to False.

        Returns:
            Dictionary containing:
                - capsule_positions: torch.Tensor, shape (num_capsules, 3), capsule positions
                - asset_info: dict with usd_path and primitive_type
        """
        try:
            positions = self.get_capsule_positions()
            if not isinstance(positions, torch.Tensor):
                positions = torch.tensor(positions, dtype=torch.float32)
            if positions.dim() == 1 and positions.shape[0] == 3:
                positions = positions.unsqueeze(0)
            elif positions.dim() == 0:
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

        asset_info = {
            "usd_path": self.usd_path if hasattr(self, "usd_path") else None,
            "primitive_type": None,
        }

        return {
            "capsule_positions": positions,
            "asset_info": asset_info,
        }
