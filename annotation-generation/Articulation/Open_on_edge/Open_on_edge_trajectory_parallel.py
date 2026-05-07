import asyncio
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
from scipy.spatial.transform import Rotation as R
from omni.isaac.core.utils import stage as stage_utils
from omni.isaac.core.utils import transformations as transform_utils
from pxr import Gf, UsdGeom, UsdShade, Sdf, UsdPhysics, Usd, PhysxSchema, UsdLux, PhysicsSchemaTools
import omni.usd
import omni.kit.app
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.cloner import GridCloner 
from omni.timeline import get_timeline_interface
ext_manager = omni.kit.app.get_app().get_extension_manager()
if not ext_manager.is_extension_enabled("isaacsim.replicator.grasping"):
    ext_manager.set_extension_enabled_immediate("isaacsim.replicator.grasping", True)
timeline = get_timeline_interface()
import isaacsim.replicator.grasping.transform_utils as transform_utils
from omni.physx import get_physx_scene_query_interface, get_physx_interface

# Constants
APPROACH_DISTANCE = 0.7  # Distance to step back before approaching (m)
MOVE_STEPS = 200          # Steps for linear interpolation
CLOSE_STEPS = 64         # Steps for closing gripper
HOLD_STEPS = 20          # Extra physics steps after closing to let contacts settle
CONTACTSLOPCOEFF = 0.1    # PhysX contact slop coefficient
TRAJECTORY_SIM_STEPS_PER_WAYPOINT = 3  # Physics steps per trajectory waypoint
NUM_COPIES = 10         # Number of copies to create in cloner
CLONE_SPACING = 5.0    # Spacing between clones in cloner grid (m)
TRAJ_OVERSHOOT_CHECKS   = 5 
GRIPPER_FINGERTIP_OFFSET = 0.18
INTERIOR_REJECTION_RATIO = 0.7 # If the majority of grasp poses is inside the object, we ignore the case
                                # as it may require some pre-grasping manipulation to pull the handle out first

# Filtering / geometry normal settings
USE_GEOM_DOOR_NORMAL = True
DOOR_NORMAL_MAX_POINTS = 8000
# Gripper approach axis used for filtering (local axis in gripper pose frame)
GRIPPER_APPROACH_LOCAL = (0, 0, 1)
GRIPPER_FINGER_LINE_LOCAL = (0, 1, 0) 

APPROACH_POSITION_THRESHOLD = 0.005
JOINT_SUCCESS_THRESHOLD = 0.95
OVERSHOOT_TOLERANCE_DEG = 10.0   
OVERSHOOT_TOLERANCE_M   = 0.01  

INITIAL_JOINT_ANGLE = 15.0
INITIAL_JOINT_POSITION_M = 0.1
OPENING_FRACTION = 0.7
SURFACE_SAMPLES_PER_MESH = 300
EDGE_SAMPLE_POINTS = 500 


# =======================
# Processing Mode Configuration
# =======================
_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _THIS_DIR.parents[1]

def _path_from_env(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default))).expanduser()

PROCESSING_MODE = "dataset"  # Options: "single" or "dataset"

# For single object mode
SINGLE_OBJECT_USD = _path_from_env("OPEN_EDGE_OBJECT_USD", _THIS_DIR / "45213" / "Object.usd")

INPUT_DATASET_PATH = _path_from_env("OPEN_EDGE_DATASET_PATH", _THIS_DIR)
GRIPPER_USD = _path_from_env("OPEN_EDGE_GRIPPER_USD", _WORKSPACE_ROOT / "Flying_hand_probe_pro.usd")
LOG_FILE = INPUT_DATASET_PATH / "open_on_edge_completed_objects.txt"

#Isaac Sim Stage Paths
ENV_BASE_PATH = "/World/Envs"
ENV_ROOT_PREFIX = f"{ENV_BASE_PATH}/env"
PHYSICS_SCENE_PATH = "/World/physicsScene"

def env_path(i: int) -> str:
    # env paths are /World/Envs/env_0, /World/Envs/env_1, ...
    return f"{ENV_ROOT_PREFIX}_{i}"

def obj_wrap(i: int) -> str:
    return f"{env_path(i)}/Object"

def obj_ref(i: int) -> str:
    return f"{env_path(i)}/Object/ref"

def grip_wrap(i: int) -> str:
    return f"{env_path(i)}/Flying_hand_probe_pro"

def grip_ref(i: int) -> str:
    return f"{env_path(i)}/Flying_hand_probe_pro/ref"

def grip_base(i: int) -> str:
    return f"{grip_ref(i)}/panda_hand"

OBJECT_WRAPPER_PATH = obj_wrap(0)
OBJECT_REF_PATH = obj_ref(0)
GRIPPER_WRAPPER_PATH = grip_wrap(0)
GRIPPER_REF_PATH = grip_ref(0)

# =======================
# Helper Functions
# =======================

def get_completed_objects(log_file: Path) -> set:
    """Read list of already processed objects from log file."""
    if not log_file.exists():
        return set()
    
    with open(log_file, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def mark_object_completed(log_file: Path, obj_id: str):
    """Append object ID to completion log."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(f"{obj_id}\n")


def find_all_objects(dataset_path: Path) -> List[Tuple[Path, str]]:
    """
    Find all Object.usd files in the dataset.
    
    Returns:
        List of (obj_usd_path, obj_id) tuples
    """
    objects = []
    
    # Iterate through all subdirectories
    for obj_dir in sorted(dataset_path.iterdir()):
        if not obj_dir.is_dir():
            continue
        
        obj_usd = obj_dir / "Object.usd"
        
        if obj_usd.exists():
            obj_id = obj_dir.name
            objects.append((obj_usd, obj_id))
    
    return objects

async def step_simulation(steps: int):
    for _ in range(steps):
        await omni.kit.app.get_app().next_update_async()

async def ensure_timeline_playing():
    """Ensure timeline is playing - force restart if stopped"""
    if not timeline.is_playing():
        print(f"[DEBUG] Timeline stopped, restarting...")
        timeline.play()
        await omni.kit.app.get_app().next_update_async()

def _get_or_create_attr_local(prim, name: str, sdf_type):
    attr = prim.GetAttribute(name)
    if not attr or not attr.IsValid():
        attr = prim.CreateAttribute(name, sdf_type)
    return attr

# Apply PhysX overrides to existing rigid bodies under the referenced object (links/base).
# IMPORTANT: Do NOT apply RigidBodyAPI to the wrapper prim, otherwise you create a rigid-body hierarchy
# (wrapper rigid body + child link rigid bodies), which PhysX warns is invalid unless you add xformStack resets.
#
# This function walks the referenced object subtree and applies PhysxRigidBodyAPI overrides ONLY
# to prims that already have UsdPhysics.RigidBodyAPI (i.e., your Object/base and Object/link_*).

def apply_object_physx_overrides(stage, object_ref_path: str, disable_gravity: bool = True):
    """
    Apply PhysX overrides to the existing rigid bodies under the referenced object (links/base).

    IMPORTANT: We only touch prims that already have UsdPhysics.RigidBodyAPI.
    We do NOT add a RigidBodyAPI to the wrapper prim to avoid rigid-body hierarchies.
    """
    root = stage.GetPrimAtPath(object_ref_path)
    if not root.IsValid():
        raise RuntimeError(f"[apply_object_physx_overrides] Invalid prim: {object_ref_path}")

    changed = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsValid():
            continue

        # Only touch prims that are already rigid bodies in the asset.
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue

        # Ensure the PhysX rigid body API exists so we can set PhysX attributes.
        if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

        # Keep rigid body enabled (asset should already do this, but be explicit)
        try:
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr().Set(True)
        except Exception:
            pass

        physx_rb = PhysxSchema.PhysxRigidBodyAPI(prim)

        # Toggle gravity for the articulated object.
        physx_rb.CreateDisableGravityAttr().Set(bool(disable_gravity))

        # Contact slop coefficient
        try:
            physx_rb.CreateContactSlopCoefficientAttr().Set(float(CONTACTSLOPCOEFF))
        except Exception:
            _get_or_create_attr_local(
                prim,
                "physxRigidBody:contactSlopCoefficient",
                Sdf.ValueTypeNames.Float,
            ).Set(float(CONTACTSLOPCOEFF))

        changed += 1

    print(
        f"[INFO] Applied runtime PhysX overrides under {object_ref_path}: "
        f"rigid_bodies_touched={changed}, "
        f"disableGravity={bool(disable_gravity)}, "
        f"contactSlopCoefficient={CONTACTSLOPCOEFF}"
    )

def setup_physics_scene(stage):
    prim = stage.GetPrimAtPath(PHYSICS_SCENE_PATH)
    if not prim.IsValid():
        prim = stage.DefinePrim(PHYSICS_SCENE_PATH, "PhysicsScene")

    if not prim.HasAPI(UsdPhysics.Scene):
        scene = UsdPhysics.Scene.Define(stage, PHYSICS_SCENE_PATH)
    else:
        scene = UsdPhysics.Scene(prim)
    
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    if not prim.HasAPI(PhysxSchema.PhysxSceneAPI):
        PhysxSchema.PhysxSceneAPI.Apply(prim)
        
    physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(prim)
    physx_scene_api.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(32768)
    physx_scene_api.CreateGpuTotalAggregatePairsCapacityAttr().Set(32768)
    return PHYSICS_SCENE_PATH

def get_bbox_bottom_center(stage, prim_path: str):
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    
    bbox = bbox_cache.ComputeWorldBound(prim)
    bbox_range = bbox.GetRange()
    
    min_point = bbox_range.GetMin()
    max_point = bbox_range.GetMax()
    
    bottom_center = Gf.Vec3d(
        (min_point[0] + max_point[0]) / 2.0,
        (min_point[1] + max_point[1]) / 2.0,
        min_point[2]
    )
    
    return bottom_center


def disable_instanceable_for_grasp_generation(stage, object_ref_path: str) -> List[str]:
    """Disable instanceable on ALL prims under object_ref_path.

    Many PartNet assets mark nested prims (links/visuals/World/mesh) as instanceable,
    not necessarily the reference prim itself. Disabling only the reference prim is often a no-op.

    Returns:
        changed_paths: list of prim path strings that were changed from instanceable=True to False.
        Use this list to restore instanceable later.
    """
    root = stage.GetPrimAtPath(object_ref_path)

    if not root.IsValid():
        print(f"[ERROR] Invalid object reference path: {object_ref_path}")
        return []

    changed_paths: List[str] = []

    # Disable instanceable across the entire subtree.
    for prim in Usd.PrimRange(root):
        if not prim.IsValid():
            continue
        if prim.IsInstanceable():
            try:
                prim.SetInstanceable(False)
                changed_paths.append(prim.GetPath().pathString)
            except Exception as e:
                print(f"[WARN] Failed to SetInstanceable(False) on {prim.GetPath()}: {e}")

    print(f"[INFO] Disabled instanceable on {len(changed_paths)} prim(s) under {object_ref_path}")
    return changed_paths


def restore_instanceable(stage, changed_paths: List[str]):
    """Restore original instanceable state on prims we temporarily made editable."""
    if not changed_paths:
        print("[INFO] No instanceable prims to restore")
        return

    # Restore deepest prims first so parent instanceability does not interfere
    # with restoring authored state on descendants.
    unique_paths = sorted(set(changed_paths), key=lambda p: (p.count("/"), p), reverse=True)

    cleared = 0
    restored = 0
    fallback_restored = 0
    unresolved: List[str] = []

    for p in unique_paths:
        prim = stage.GetPrimAtPath(p)
        if not prim.IsValid():
            continue
        try:
            if prim.HasAuthoredMetadata("instanceable"):
                prim.ClearInstanceable()
                cleared += 1
        except Exception as e:
            print(f"[WARN] Failed to clear instanceable override on {p}: {e}")
            unresolved.append(p)

    for p in unique_paths:
        if p in unresolved:
            continue
        prim = stage.GetPrimAtPath(p)
        if not prim.IsValid():
            continue
        if prim.IsInstanceable():
            restored += 1
            continue
        try:
            prim.SetInstanceable(True)
            if prim.IsInstanceable():
                fallback_restored += 1
            else:
                unresolved.append(p)
        except Exception as e:
            print(f"[WARN] Failed to restore instanceable on {p}: {e}")
            unresolved.append(p)

    total_restored = restored + fallback_restored
    print(
        f"[INFO] Restored instanceable on {total_restored}/{len(unique_paths)} prim(s) "
        f"(cleared {cleared} local override(s), fallback-authored {fallback_restored})"
    )
    if unresolved:
        sample = ", ".join(unresolved[:5])
        if len(unresolved) > 5:
            sample += ", ..."
        print(f"[WARN] Instanceable still not restored on {len(unresolved)} prim(s): {sample}")

def set_object_gravity_enabled(stage, object_ref_path: str, enabled: bool = True):
    """Enable/disable gravity for all rigid bodies inside the referenced articulated object.

    PhysX schema uses `disableGravity=True` to mean gravity OFF.
    """
    root = stage.GetPrimAtPath(object_ref_path)
    if not root.IsValid():
        raise RuntimeError(f"[set_object_gravity_enabled] Invalid prim: {object_ref_path}")

    changed = 0
    disable_gravity = not bool(enabled)
    for prim in Usd.PrimRange(root):
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        PhysxSchema.PhysxRigidBodyAPI(prim).CreateDisableGravityAttr().Set(disable_gravity)
        changed += 1

    print(
        f"[INFO] Updated gravity under {object_ref_path}: "
        f"rigid_bodies_touched={changed}, enabled={bool(enabled)}"
    )

def save_trajectories_to_json(
    trajectories: List[Dict],
    obj_usd_path: Path,
    bottom_center: List[float]
) -> Path:
    """
    Save validated trajectories to JSON format grouped by joint.
    
    Args:
        trajectories: List of validated trajectory dicts from physics_validation_loop
        obj_usd_path: Path to Object.usd file
        bottom_center: [x, y, z] bottom center coordinates
    
    Returns:
        Path to saved JSON file
    """
    # Extract obj_id from path
    obj_id = obj_usd_path.parent.name
    
    # For dataset mode with category structure
    if PROCESSING_MODE == "dataset":
        try:
            obj_cat = obj_usd_path.parent.parent.name
            type_str = f"{obj_cat}/{obj_id}/Object.usd"
        except:
            type_str = f"{obj_id}/Object.usd"
    else:
        # Single object mode
        obj_cat = obj_usd_path.parent.parent.name
        type_str = f"{obj_id}/Object.usd"
    
    # Create Annotation directory at same level as Object.usd
    annotation_dir = obj_usd_path.parent / "Annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect per-joint initial angles and target displacements (in degrees or meters)
    initial_joint_angles: Dict[str, float] = {}
    target_joint_displacements: Dict[str, float] = {}

    for traj in trajectories:
        jname = traj.get("joint_name", "unknown")
        jtype = traj.get("joint_type", "revolute")
        if jname not in initial_joint_angles:
            if jtype == "revolute":
                initial_joint_angles[jname] = float(INITIAL_JOINT_ANGLE)  # already in degrees
                target_joint_displacements[jname] = float(np.degrees(traj.get("target_displacement", 0.0)))
            else:
                initial_joint_angles[jname] = float(INITIAL_JOINT_POSITION_M)  # meters
                target_joint_displacements[jname] = float(traj.get("target_displacement", 0.0))

    # Build JSON structure
    data = {
        "type": obj_cat,
        "bottom_center": {
            "x": float(bottom_center[0]),
            "y": float(bottom_center[1]),
            "z": float(bottom_center[2])
        },
        "initial_joint_angles": initial_joint_angles,
        "target_joint_displacements": target_joint_displacements,
        "trajectories": {}
    }
    
    # Group trajectories by joint
    trajectories_by_joint = {}
    
    # Process each trajectory
    for idx, traj in enumerate(trajectories):
        trajectory_positions = np.asarray(traj["trajectory_positions"], dtype=np.float64)
        trajectory_orientations = np.asarray(traj["trajectory_orientations"], dtype=np.float64)
        
        # Get joint information
        joint_name = traj.get("joint_name", "unknown")
        
        # Check if trajectory terminated early
        termination_step = traj.get("termination_step", None)
        if termination_step is not None:
            trajectory_positions = trajectory_positions[:termination_step + 1]
            trajectory_orientations = trajectory_orientations[:termination_step + 1]
            print(f"[INFO] Trajectory {idx} ({joint_name}): "
                  f"Truncated to {termination_step + 1} waypoints (early termination at step {termination_step})")
        
        # Build waypoint list: each waypoint is [x, y, z, qw, qx, qy, qz]
        waypoints = []
        for pos, quat in zip(trajectory_positions, trajectory_orientations):
            waypoint = [
                float(pos[0]),  # x
                float(pos[1]),  # y
                float(pos[2]),  # z
                float(quat[0]), # w
                float(quat[1]), # x
                float(quat[2]), # y
                float(quat[3])  # z
            ]
            waypoints.append(waypoint)
        
        # Group by joint_name
        if joint_name not in trajectories_by_joint:
            trajectories_by_joint[joint_name] = []
        trajectories_by_joint[joint_name].append(waypoints)
    
    # Build the nested structure: joint_{num} -> { "1": [...], "2": [...], ... }
    for joint_name, joint_trajectories in trajectories_by_joint.items():
        data["trajectories"][joint_name] = {}
        for traj_num, waypoints in enumerate(joint_trajectories, start=1):
            data["trajectories"][joint_name][str(traj_num)] = waypoints
    
    # Save to JSON
    json_path = annotation_dir / "open_on_edge_trajectory.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"[INFO] Saved {len(trajectories)} trajectories to {json_path}")
    print(f"[INFO] Trajectories organized by joint:")
    for joint_name, joint_trajectories in trajectories_by_joint.items():
        print(f"  - {joint_name}: {len(joint_trajectories)} trajectories")
    
    return json_path

# =======================
# Fix Object Base to World (FixedJoint)
# =======================
def fix_object_base_to_world(stage, object_ref_path: str, base_link_name: str = "base", joint_path: str = "/World/ObjectFixedToWorld"):
    """Fix the articulated object's base rigid body to the world using a FixedJoint.
    
    The joint is positioned at the base body's current world location.
    """
    base_path = f"{object_ref_path}/{base_link_name}"
    base_prim = stage.GetPrimAtPath(base_path)
    if not base_prim.IsValid():
        # Fallback: try to find a single rigid body directly under the ref root.
        root = stage.GetPrimAtPath(object_ref_path)
        if not root.IsValid():
            print(f"[WARN] fix_object_base_to_world: invalid object ref path {object_ref_path}")
            return
        candidate = None
        for prim in root.GetChildren():
            if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                candidate = prim
                break
        if candidate is None:
            print(f"[WARN] fix_object_base_to_world: could not find base rigid body under {object_ref_path}")
            return
        base_path = candidate.GetPath().pathString
        base_prim = candidate

    # Ensure the base is a rigid body (required for joints)
    if not base_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print(f"[WARN] fix_object_base_to_world: base prim is not a rigid body: {base_path}")
        return

    # Get the current world transform of the base body
    xformable = UsdGeom.Xformable(base_prim)
    world_xf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    
    # Extract position and rotation from world transform
    translation = world_xf.ExtractTranslation()
    rotation = world_xf.ExtractRotation()
    
    # Convert rotation to quaternion
    rotation_quat = rotation.GetQuat()
    
    print(f"[DEBUG] Base body world position: ({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f})")

    # Create/overwrite a fixed joint prim
    joint_prim = stage.GetPrimAtPath(joint_path)
    if joint_prim.IsValid():
        try:
            stage.RemovePrim(Sdf.Path(joint_path))
        except Exception:
            pass

    fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)

    # World anchor: set Body0 = base rigid body, leave Body1 empty (world)
    fixed_joint.CreateBody0Rel().SetTargets([Sdf.Path(base_path)])

    # CRITICAL: Set the joint frame to the body's CURRENT world position
    # This keeps the object where it is instead of moving it to origin
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))  # Local to body0
    fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))  # Local to body0
    
    # Set Body1 frame (world) to the body's current world position
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(
        float(translation[0]), 
        float(translation[1]), 
        float(translation[2])
    ))
    fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(
        float(rotation_quat.GetReal()),
        Gf.Vec3f(
            float(rotation_quat.GetImaginary()[0]),
            float(rotation_quat.GetImaginary()[1]),
            float(rotation_quat.GetImaginary()[2])
        )
    ))

    print(f"[INFO] Fixed object base to world at current position: base={base_path} joint={joint_path}")

# =======================
# Grasp Generation
# =======================

def find_revolute_prismatic_joints_in_urdf(urdf_path: str) -> List[Tuple[str, str, str, float, float, np.ndarray]]:
    """Parse URDF to find movable joints and their child links.
    
    Returns:
        List of (joint_name, joint_type, child_link_name, lower_limit, upper_limit, joint_axis)
        
    Note: We no longer track handle_visual_indices since we sample all surfaces of the child link.
    """
    joints_info = []
    
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        for joint in root.findall('joint'):
            joint_type = joint.get('type')
            if joint_type not in ['revolute', 'prismatic']:
                continue
            
            joint_name = joint.get('name')
            child_link = joint.find('child')
            child_link_name = child_link.get('link') if child_link is not None else None
            
            if not child_link_name:
                continue
            
            # Get joint axis
            axis_elem = joint.find('axis')
            if axis_elem is not None:
                xyz_str = axis_elem.get('xyz', '0 0 1')
                joint_axis = np.array([float(x) for x in xyz_str.split()])
            else:
                joint_axis = np.array([0, 0, 1])
            
            # Get joint limits
            limit_elem = joint.find('limit')
            if limit_elem is not None:
                lower_limit = float(limit_elem.get('lower', '0.0'))
                upper_limit = float(limit_elem.get('upper', '0.0'))
            else:
                lower_limit = 0.0
                upper_limit = np.pi / 2 if joint_type == 'revolute' else 0.5
            
            # No longer tracking handle_visual_indices - we'll sample entire child link
            joints_info.append((joint_name, joint_type, child_link_name, lower_limit, upper_limit, joint_axis))
            
            print(f"[DEBUG] Found joint: {joint_name} ({joint_type}) -> {child_link_name}")
    
    except Exception as e:
        print(f"[ERROR] Failed to parse URDF: {e}")
    
    return joints_info

def collect_all_child_link_surface_samples(stage, object_ref_path: str, child_link_name: str, n_points_per_mesh: int = 500) -> np.ndarray:
    """Sample surface points from ALL visual geometries under a child link using area-weighted sampling.
    
    Args:
        stage: USD stage
        object_ref_path: Path to object reference (e.g., /World/Envs/env_0/Object/ref)
        child_link_name: Name of the child link (e.g., 'link_0')
        n_points_per_mesh: Number of points to sample per mesh
    
    Returns:
        [N, 3] array of world-space surface sample points from all meshes under the child link
    """
    asset_root = resolve_asset_root_under_ref(stage, object_ref_path)
    
    # Path to the child link's visual geometry
    # Typical structure: {asset_root}/{child_link_name}/visuals
    visuals_root_path = f"{asset_root}/{child_link_name}/visuals"
    
    visuals_prim = stage.GetPrimAtPath(visuals_root_path)
    if not visuals_prim.IsValid():
        print(f"[WARNING] No visuals found at {visuals_root_path}")
        return np.zeros((0, 3), dtype=np.float64)
    
    # Ensure payloads are loaded
    try:
        visuals_prim.Load()
    except Exception:
        pass
    
    all_sampled_points = []
    mesh_count = 0
    
    # Collect and sample points from all mesh prims under visuals
    for prim in Usd.PrimRange(visuals_prim, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        
        mesh_count += 1
        mesh_path = prim.GetPath().pathString
        
        # Use area-weighted surface sampling
        sampled_pts = sample_mesh_surface_points(stage, mesh_path, n_points=n_points_per_mesh)
        
        if sampled_pts.shape[0] > 0:
            all_sampled_points.append(sampled_pts)
            print(f"[INFO]   Sampled {sampled_pts.shape[0]} points from mesh: {prim.GetName()}")
    
    if not all_sampled_points:
        print(f"[WARNING] No mesh points sampled from {visuals_root_path} ({mesh_count} meshes found)")
        return np.zeros((0, 3), dtype=np.float64)
    
    all_sampled_points = np.concatenate(all_sampled_points, axis=0)
    print(f"[INFO] Total sampled points from {child_link_name}: {all_sampled_points.shape[0]} (from {mesh_count} meshes)")
    
    return all_sampled_points

def compute_object_center(stage, object_wrapper_path: str) -> np.ndarray:
    """Compute center of entire object."""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    prim = stage.GetPrimAtPath(object_wrapper_path)
    
    if not prim.IsValid():
        return np.array([0, 0, 0])
    
    bbox = bbox_cache.ComputeWorldBound(prim)
    bbox_range = bbox.GetRange()
    center = bbox_range.GetMidpoint()
    
    return np.array([float(center[0]), float(center[1]), float(center[2])])


def get_joint_hinge_position(stage, object_ref_path: str, child_link_name: str) -> Optional[np.ndarray]:
    """Get hinge position (origin) of the joint in world coordinates."""
    link_path = f"{object_ref_path}/{child_link_name}"
    link_prim = stage.GetPrimAtPath(link_path)
    
    if not link_prim.IsValid():
        return None
    
    xformable = UsdGeom.Xformable(link_prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    
    hinge_local = Gf.Vec3d(0, 0, 0)
    hinge_world = world_transform.Transform(hinge_local)
    
    return np.array([float(hinge_world[0]), float(hinge_world[1]), float(hinge_world[2])])


def get_link_local_bbox_center(stage, link_path: str) -> Optional[np.ndarray]:
    """Get center of link's local bounding box in world coordinates."""
    link_prim = stage.GetPrimAtPath(link_path)
    
    if not link_prim.IsValid():
        return None
    
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    local_bbox = bbox_cache.ComputeLocalBound(link_prim)
    local_range = local_bbox.GetRange()
    local_center = local_range.GetMidpoint()
    
    xformable = UsdGeom.Xformable(link_prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_center = world_transform.Transform(local_center)
    
    return np.array([float(world_center[0]), float(world_center[1]), float(world_center[2])])


# =======================
# Door surface normal from geometry (door mesh plane) + handle sign
# =======================

def _gvec_to_np(v) -> np.ndarray:
    return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)



def _world_xf_of_prim(prim) -> Gf.Matrix4d:
    xf = UsdGeom.Xformable(prim)
    return xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())


# Helper: resolve internal asset root under a reference prim (e.g., to find /partnet_<hash>)
def resolve_asset_root_under_ref(stage, object_ref_path: str) -> str:
    ref_prim = stage.GetPrimAtPath(object_ref_path)
    if not ref_prim.IsValid():
        return object_ref_path

    # Go one level deeper unconditionally: pick the first valid child that
    # itself has children (the scene root), regardless of its name.
    # This handles partnet_* and any other naming convention.
    children = [c for c in ref_prim.GetChildren() if c.IsValid()]
    if len(children) == 1:
        return children[0].GetPath().pathString
    for child in children:
        if list(child.GetChildren()):
            return child.GetPath().pathString
    # Fallback: ref IS the root (no wrapper layer present)
    return object_ref_path


def collect_mesh_points_world(stage, root_path: str, max_points: int = 6000) -> np.ndarray:
    """Collect (subsampled) mesh vertices under root_path, transformed to world.

    Works for paths like:
      .../visuals/visual_mesh_0/World/mesh
    and higher-level roots like:
      .../link_0/visuals

    Also attempts to load payloads so PrimRange can see Mesh prims.
    """
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return np.zeros((0, 3), dtype=np.float64)

    # Ensure payloads are loaded (safe no-op if none)
    try:
        root_prim.Load()
    except Exception:
        pass

    pts_all = []

    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if not pts:
            continue

        xf = _world_xf_of_prim(prim)
        pts_np = np.asarray([_gvec_to_np(xf.Transform(p)) for p in pts], dtype=np.float64)
        pts_all.append(pts_np)

    if not pts_all:
        return np.zeros((0, 3), dtype=np.float64)

    pts_all = np.concatenate(pts_all, axis=0)

    if pts_all.shape[0] > max_points:
        idx = np.random.choice(pts_all.shape[0], size=max_points, replace=False)
        pts_all = pts_all[idx]

    return pts_all


def plane_normal_pca(points_world: np.ndarray) -> Optional[np.ndarray]:
    """Fit a plane by PCA; return unit normal (smallest eigenvector)."""
    if points_world is None or points_world.shape[0] < 200:
        return None

    centroid = points_world.mean(axis=0)
    X = points_world - centroid
    cov = (X.T @ X) / max(X.shape[0] - 1, 1)

    # eigenvalues ascending; smallest eigenvector is plane normal
    w, v = np.linalg.eigh(cov)
    n = v[:, 0]
    n = n / (np.linalg.norm(n) + 1e-12)
    return n

def sample_mesh_surface_points(stage, mesh_path, n_points=500):
    """Sample points uniformly on mesh surface using area-weighted triangle sampling.
    
    Args:
        stage: USD stage
        mesh_path: Path to mesh prim
        n_points: Number of points to sample
    
    Returns:
        [n_points, 3] array of world-space surface points
    """
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        print(f"[WARNING] Invalid mesh prim at {mesh_path}")
        return np.zeros((0, 3), dtype=np.float64)
    
    mesh = UsdGeom.Mesh(prim)

    verts = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    
    if not verts or not counts or not indices:
        print(f"[WARNING] Incomplete mesh data at {mesh_path}")
        return np.zeros((0, 3), dtype=np.float64)
    
    verts = np.asarray(verts, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int32)
    indices = np.asarray(indices, dtype=np.int32)

    # Check if mesh is triangulated (all faces are triangles)
    if not np.all(counts == 3):
        print(f"[WARNING] Mesh at {mesh_path} is not triangulated (has non-triangle faces)")
        # Filter to only use triangular faces
        tri_mask = counts == 3
        if not np.any(tri_mask):
            print(f"[ERROR] No triangular faces found in mesh")
            return np.zeros((0, 3), dtype=np.float64)
        
        # Rebuild indices for triangles only
        new_indices = []
        idx_ptr = 0
        for i, count in enumerate(counts):
            if count == 3:
                new_indices.extend(indices[idx_ptr:idx_ptr+3])
            idx_ptr += count
        indices = np.array(new_indices, dtype=np.int32)
    
    # Reshape to triangles
    try:
        tris = verts[indices].reshape(-1, 3, 3)
    except Exception as e:
        print(f"[ERROR] Failed to reshape mesh vertices: {e}")
        return np.zeros((0, 3), dtype=np.float64)

    if len(tris) == 0:
        print(f"[WARNING] No triangles in mesh at {mesh_path}")
        return np.zeros((0, 3), dtype=np.float64)

    # Area-weighted sampling
    edge1 = tris[:,1] - tris[:,0]
    edge2 = tris[:,2] - tris[:,0]
    areas = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)
    
    total_area = areas.sum()
    if total_area < 1e-12:
        print(f"[WARNING] Mesh at {mesh_path} has near-zero total area")
        return np.zeros((0, 3), dtype=np.float64)
    
    probs = areas / total_area

    # Choose triangles
    tri_idx = np.random.choice(len(tris), min(n_points, len(tris)), p=probs)
    t = tris[tri_idx]

    # Barycentric random points
    n_actual = len(tri_idx)
    u = np.random.rand(n_actual, 1)
    v = np.random.rand(n_actual, 1)
    mask = u + v > 1
    u[mask], v[mask] = 1-u[mask], 1-v[mask]
    w = 1 - u - v

    pts = t[:,0] * w + t[:,1] * u + t[:,2] * v

    # Transform to world
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pts_world = np.asarray([_gvec_to_np(xf.Transform(Gf.Vec3d(p[0], p[1], p[2]))) for p in pts], dtype=np.float64)

    return pts_world

# =======================
# Quaternion / approach helpers for filtering
# =======================

def quat_to_xyzw(quat) -> np.ndarray:
    """Convert Gf.Quat* or sequence to numpy [x,y,z,w]."""
    # pxr Gf.Quatf / Gf.Quatd
    if hasattr(quat, "GetReal") and hasattr(quat, "GetImaginary"):
        w = float(quat.GetReal())
        im = quat.GetImaginary()
        x, y, z = float(im[0]), float(im[1]), float(im[2])
        return np.array([x, y, z, w], dtype=np.float64)

    arr = np.array(quat, dtype=np.float64).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"Unexpected quat size {arr.size}: {quat}")

    # Assumption: sequences are already xyzw.
    # If your sequence is wxyz in practice, swap here:
    # w, x, y, z = arr
    # return np.array([x, y, z, w], dtype=np.float64)
    return arr

def grasp_approach_dir_world(quat, approach_local=(0, 0, 1)) -> np.ndarray:
    """Compute world approach direction given grasp quaternion and local approach axis."""
    q_xyzw = quat_to_xyzw(quat)
    rot = R.from_quat(q_xyzw)  # scipy expects [x,y,z,w]
    v = rot.apply(np.array(approach_local, dtype=np.float64))
    v = v / (np.linalg.norm(v) + 1e-12)
    return v

def compute_door_outward_normal_revolute(stage, 
                                object_wrapper_path: str,
                                object_ref_path: str, 
                                child_link_name: str,
                                joint_axis: np.ndarray) -> Optional[np.ndarray]:
    """Compute door's outward normal using hinge-to-door-center vector."""
    link_path = f"{object_ref_path}/{child_link_name}"
    
    hinge_pos = get_joint_hinge_position(stage, object_ref_path, child_link_name)
    if hinge_pos is None:
        print(f"[WARNING] Could not get hinge position")
        return None
    
    door_center = get_link_local_bbox_center(stage, link_path)
    if door_center is None:
        print(f"[WARNING] Could not get door center")
        return None
    
    print(f"[DEBUG] Hinge position: {hinge_pos}")
    print(f"[DEBUG] Door center: {door_center}")
    
    hinge_to_door = door_center - hinge_pos
    joint_axis_norm = joint_axis / (np.linalg.norm(joint_axis) + 1e-8)
    
    # Project perpendicular to hinge axis
    parallel_component = np.dot(hinge_to_door, joint_axis_norm) * joint_axis_norm
    door_normal = hinge_to_door - parallel_component
    
    door_normal_norm = np.linalg.norm(door_normal)
    if door_normal_norm < 1e-6:
        print(f"[WARNING] Door normal has zero length")
        return None
    
    door_normal = door_normal / door_normal_norm
    
    # Choose correct sign using object center
    object_center = compute_object_center(stage, object_wrapper_path)
    to_door = door_center - object_center
    
    if np.dot(door_normal, to_door) < 0:
        door_normal = -door_normal
    
    print(f"[DEBUG] Door outward normal: {door_normal}")
    
    return door_normal

def compute_prismatic_door_normal(stage, object_wrapper_path, object_ref_path, child_link_name, sliding_axis):
    """For prismatic joints, door normal is parallel to the hinge axis, oriented outward."""
    hinge_pos = get_joint_hinge_position(stage, object_ref_path, child_link_name)
    if hinge_pos is None:
        print(f"[WARNING] Could not get hinge position for prismatic normal")
        return sliding_axis / (np.linalg.norm(sliding_axis) + 1e-12)
    
    link_path = f"{object_ref_path}/{child_link_name}"
    door_center = get_link_local_bbox_center(stage, link_path)
    object_center = compute_object_center(stage, object_wrapper_path)
    
    # Normal is parallel to the hinge (sliding) axis
    axis_norm = sliding_axis / (np.linalg.norm(sliding_axis) + 1e-12)
    
    # Orient outward from object center
    if door_center is not None:
        to_door = door_center - object_center
        if np.dot(axis_norm, to_door) < 0:
            axis_norm = -axis_norm
    
    print(f"[DEBUG] Prismatic door normal (parallel to hinge): {axis_norm}")
    return axis_norm

#filter surface points based on distance to joint axis
def point_to_line_distance(p: np.ndarray, pivot: np.ndarray, axis: np.ndarray) -> float:
    
    v = p - pivot
    
    # Component of v parallel to axis
    parallel = np.dot(v, axis) * axis
    
    # Component of v perpendicular to axis
    perpendicular = v - parallel
    
    # Distance is magnitude of perpendicular component
    distance = np.linalg.norm(perpendicular)
    
    return distance

def debug_draw_vector(start, direction, length=0.3):
    """
    Draw a single debug line in the stage with no dependencies.
    """
    stage = omni.usd.get_context().get_stage()

    # Normalize
    d = direction / (np.linalg.norm(direction) + 1e-12)
    end = start + d * length

    # Create a unique path for each line
    path = f"/World/_debug_line_{np.random.randint(1e9)}"
    curve = UsdGeom.BasisCurves.Define(stage, path)

    curve.CreatePointsAttr([
        Gf.Vec3f(*start),
        Gf.Vec3f(*end)
    ])

    curve.CreateCurveVertexCountsAttr([2])
    curve.CreateWidthsAttr([0.005])  # thin line
    curve.CreateTypeAttr("linear")

def _dbg_draw_three_vectors_simple(
    hinge_origin: np.ndarray,
    hinge_axis: np.ndarray,
    radial: np.ndarray,
    door_normal: np.ndarray,
    length: float = 0.35,
    width: float = 3.0,
    clear_first: bool = True,
):
    """
    Isaac Sim 5.1.0 debug draw:
      - hinge axis (red), radial (green), door normal (blue)
    """
    from isaacsim.util.debug_draw import _debug_draw

    dd = _debug_draw.acquire_debug_draw_interface()

    if clear_first:
        dd.clear_lines()

    def nhat(v):
        v = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(v)
        if n < 1e-12:
            return None
        return v / n

    o = np.asarray(hinge_origin, dtype=np.float64)
    a = nhat(hinge_axis)
    r = nhat(radial)
    n = nhat(door_normal)
    if a is None or r is None or n is None:
        return

    p0_axis = o - a * length
    p1_axis = o + a * length
    p0_rad  = o
    p1_rad  = o + r * length
    p0_n    = o
    p1_n    = o + n * length

    starts = [
        (float(p0_axis[0]), float(p0_axis[1]), float(p0_axis[2])),
        (float(p0_rad[0]),  float(p0_rad[1]),  float(p0_rad[2])),
        (float(p0_n[0]),    float(p0_n[1]),    float(p0_n[2])),
    ]
    ends = [
        (float(p1_axis[0]), float(p1_axis[1]), float(p1_axis[2])),
        (float(p1_rad[0]),  float(p1_rad[1]),  float(p1_rad[2])),
        (float(p1_n[0]),    float(p1_n[1]),    float(p1_n[2])),
    ]
    colors = [
        (1.0, 0.2, 0.2, 1.0),  # red   — hinge axis
        (0.2, 1.0, 0.2, 1.0),  # green — radial
        (0.2, 0.6, 1.0, 1.0),  # blue  — door normal
    ]
    sizes = [float(width), float(width), float(width)]
    dd.draw_lines(starts, ends, colors, sizes)


def compute_door_normal_from_joint(
    stage,
    joint_prim: Usd.Prim,
    joint_params: Dict,
) -> Optional[np.ndarray]:
    """
    Compute revolute door outward normal from joint geometry.

    Returns the cross product (hinge_axis × radial), where radial is the
    projection of (door_center - hinge_origin) onto the plane perpendicular
    to the hinge axis.  Also draws debug vectors.
    """
    if joint_params.get("joint_type") != "revolute":
        return None

    j = UsdPhysics.Joint(joint_prim)

    # --- body1 (child / door link) ---
    body1_targets = []
    try:
        rel1 = j.GetBody1Rel()
        if rel1 and rel1.IsValid():
            body1_targets = rel1.GetTargets()
    except Exception:
        pass

    if not body1_targets:
        print(f"[JointNormal] No body1 on {joint_prim.GetPath()}")
        return None

    body1_path = body1_targets[0].pathString
    body1_prim = stage.GetPrimAtPath(body1_path)
    if not body1_prim.IsValid():
        print(f"[JointNormal] body1 invalid: {body1_path}")
        return None

    # --- Recompute hinge axis from body1 frame ---
    body1_world_xf = UsdGeom.Xformable(body1_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    lr1 = (
        j.GetLocalRot1Attr().Get()
        if (j.GetLocalRot1Attr() and j.GetLocalRot1Attr().IsValid())
        else Gf.Quatf(1, 0, 0, 0)
    )
    lp1 = (
        j.GetLocalPos1Attr().Get()
        if (j.GetLocalPos1Attr() and j.GetLocalPos1Attr().IsValid())
        else Gf.Vec3f(0, 0, 0)
    )
    joint_frame_body1 = body1_world_xf * _make_transform_gf(
        np.array([float(lp1[0]), float(lp1[1]), float(lp1[2])], dtype=np.float64),
        lr1,
    )
    axis_map = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}
    axis_local = axis_map.get(joint_params.get("axis_token", "Z"), Gf.Vec3d(0, 0, 1))
    axis_world_gf = joint_frame_body1.TransformDir(axis_local)
    hinge_axis = np.array(
        [float(axis_world_gf[0]), float(axis_world_gf[1]), float(axis_world_gf[2])],
        dtype=np.float64,
    )
    hinge_axis /= np.linalg.norm(hinge_axis) + 1e-12

    # --- Hinge origin from joint_params ---
    hinge_origin = np.asarray(joint_params["pivot"], dtype=np.float64)

    # --- Door center: body1 bbox midpoint ---
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_],
    )
    bbox = bbox_cache.ComputeWorldBound(body1_prim)
    mid = bbox.GetRange().GetMidpoint()
    door_center = np.array([float(mid[0]), float(mid[1]), float(mid[2])], dtype=np.float64)

    # --- Radial: project (door_center - hinge_origin) onto plane ⊥ hinge_axis ---
    v = door_center - hinge_origin
    v_radial = v - np.dot(v, hinge_axis) * hinge_axis
    rmag = np.linalg.norm(v_radial)
    if rmag < 1e-4:
        print(f"[JointNormal] door_center lies on hinge axis — cannot determine radial")
        return None
    r_hat = v_radial / rmag

    # --- Door normal: hinge_axis × radial ---
    door_normal = np.cross(hinge_axis, r_hat)
    nmag = np.linalg.norm(door_normal)
    if nmag < 1e-4:
        print(f"[JointNormal] door_normal degenerate")
        return None
    door_normal = door_normal / nmag

    # --- Debug draw ---
    _dbg_draw_three_vectors_simple(
        hinge_origin=hinge_origin,
        hinge_axis=hinge_axis,
        radial=v_radial,
        door_normal=door_normal,
        length=0.35,
        width=3.0,
        clear_first=True,
    )

    print(
        f"[JointNormal] axis_token={joint_params.get('axis_token')}, "
        f"hinge_axis={hinge_axis}, hinge_origin={hinge_origin}, "
        f"door_center={door_center}, radial={r_hat}, door_normal={door_normal}"
    )
    return door_normal

def filter_revolute_edge_points(
    stage,
    joint_prim: Usd.Prim,
    joint_params: Dict,
    surface_points: np.ndarray,
    pivot: np.ndarray,
    axis: np.ndarray,
    epsilon: float = 0.005,
    slice_thickness: float = 0.02
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # Added third return: tangents
    """
    Filter revolute joint surface points to get outer rim band with radial normals.
    
    Returns:
        filtered_points: [M, 3] points on outer rim
        normals: [M, 3] radial normals (from axis to point)
        tangents: [M, 3] edge tangent directions (perpendicular to both axis and normal)
    """
    if surface_points.shape[0] == 0:
        return (np.zeros((0, 3), dtype=np.float64), 
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64))
    
    # Ensure axis is unit vector
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    
    # -------------------------------------------
    # First pass: axis positions & radial stuff
    # -------------------------------------------
    axis_positions = []
    radial_distances = []
    perpendiculars = []
    
    for point in surface_points:
        v = point - pivot
        
        # Component parallel to axis (scalar projection)
        axis_pos = np.dot(v, axis)
        axis_positions.append(axis_pos)
        
        # Component perpendicular to axis (radial)
        parallel_vec = axis_pos * axis
        perpendicular = v - parallel_vec
        
        radial_dist = np.linalg.norm(perpendicular)
        radial_distances.append(radial_dist)
        perpendiculars.append(perpendicular)
    
    axis_positions   = np.array(axis_positions)
    radial_distances = np.array(radial_distances)
    perpendiculars   = np.array(perpendiculars)
    
    # -------------------------------------------
    # Compute door normal from joint geometry
    # -------------------------------------------
    door_normal = compute_door_normal_from_joint(stage, joint_prim, joint_params)

    if door_normal is None:
        # fallback: axis × world-up (or axis × world-X if nearly vertical)
        helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(helper, axis)) > 0.9:
            helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        door_normal = np.cross(axis, helper)
        dn_norm = np.linalg.norm(door_normal)
        door_normal = door_normal / (dn_norm + 1e-12)

    print(f"[DEBUG] Revolute door normal (world): {door_normal}")
    
    # -------------------------------------------
    # Angle filter vs door normal (handle rejection)
    # We want v_dir ⟂ door_normal → angle ~ 90°
    # Keep if |angle - 90°| <= angle_tolerance_deg
    # <=> |dot| <= sin(angle_tolerance_deg)
    # -------------------------------------------
    angle_tolerance_deg = 3.0
    angle_tol_rad = np.deg2rad(angle_tolerance_deg)
    max_abs_dot = np.sin(angle_tol_rad)
    
    angle_ok_mask = []
    for point in surface_points:
        v = point - pivot
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-8:
            angle_ok_mask.append(False)
        else:
            v_dir = v / v_norm
            dot = abs(np.dot(v_dir, door_normal))
            angle_ok_mask.append(dot <= max_abs_dot)
    angle_ok_mask = np.array(angle_ok_mask, dtype=bool)
    
    # -------------------------------------------
    # Divide into slices along the axis
    # -------------------------------------------
    axis_min = axis_positions.min()
    axis_max = axis_positions.max()
    
    num_slices = max(int(np.ceil((axis_max - axis_min) / slice_thickness)), 1)
    slice_edges = np.linspace(axis_min, axis_max, num_slices + 1)
    
    print(f"[DEBUG] Revolute: Axis range [{axis_min:.4f}, {axis_max:.4f}]m, {num_slices} slices")
    
    # For each slice, find local max radius and filter points
    keep_mask = np.zeros(len(surface_points), dtype=bool)
    
    for i in range(num_slices):
        slice_start = slice_edges[i]
        slice_end = slice_edges[i + 1]
        
        in_slice = (axis_positions >= slice_start) & (axis_positions < slice_end)
        if i == num_slices - 1:
            in_slice = (axis_positions >= slice_start) & (axis_positions <= slice_end)
        
        if not np.any(in_slice):
            continue
        
        # Also require angle_ok in this slice (reject handle points)
        valid_in_slice = in_slice & angle_ok_mask
        if not np.any(valid_in_slice):
            print(f"[DEBUG]   Slice {i}: all points rejected by angle filter")
            continue
        
        slice_radii = radial_distances[valid_in_slice]
        local_max_radius = slice_radii.max()
        local_threshold = local_max_radius - epsilon
        
        slice_indices = np.where(valid_in_slice)[0]
        for idx in slice_indices:
            if radial_distances[idx] >= local_threshold:
                keep_mask[idx] = True
        
        points_in_slice = np.sum(in_slice)
        points_valid    = np.sum(valid_in_slice)
        points_kept     = np.sum(keep_mask[valid_in_slice])
        print(
            f"[DEBUG]   Slice {i}: axis=[{slice_start:.3f}, {slice_end:.3f}], "
            f"max_r={local_max_radius:.4f}m, "
            f"valid_by_angle={points_valid}/{points_in_slice}, "
            f"kept {points_kept}/{points_valid}"
        )
    
    filtered_points = surface_points[keep_mask]
    filtered_perpendiculars = perpendiculars[keep_mask]
    
    print(f"[INFO] Revolute edge filter: {filtered_points.shape[0]}/{surface_points.shape[0]} points kept")
    
    # Compute normals and tangents
    normals = []
    tangents = []
    
    for perp in filtered_perpendiculars:
        norm = np.linalg.norm(perp)
        if norm < 1e-6:
            # Point is on axis, use arbitrary perpendicular
            normal = np.array([1.0, 0.0, 0.0])
            tangent = np.array([0.0, 1.0, 0.0])
        else:
            # Normal: radial direction (outward from axis)
            normal = perp / norm
            
            # Tangent: perpendicular to both axis and normal
            # tangent = axis × normal (right-hand rule)
            tangent = np.cross(axis, normal)
            tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
        
        normals.append(normal)
        tangents.append(tangent)
    
    normals = np.array(normals, dtype=np.float64)
    tangents = np.array(tangents, dtype=np.float64)
    
    return filtered_points, normals, tangents

def filter_prismatic_edge_points(
    surface_points: np.ndarray,
    pivot: np.ndarray,
    axis: np.ndarray,
    stage,
    object_wrapper_path: str,
    object_ref_path: str,
    child_link_name: str,
    epsilon: float = 0.005
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # Added third return: edge_directions
    """
    Filter prismatic joint surface points to get edge line with upward normals.
    
    Returns:
        filtered_points: [M, 3] points on edge line
        normals: [M, 3] upward normals
        edge_directions: [M, 3] edge line directions (sliding axis)
    """
    if surface_points.shape[0] == 0:
        return (np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64))
    
    # Ensure axis is unit vector
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    
    # Step 1: Compute door outward normal (inline)
    door_normal = compute_prismatic_door_normal(
        stage,
        object_wrapper_path,
        object_ref_path,
        child_link_name,
        axis
    )
    
    if door_normal is None:
        print(f"[WARNING] Could not compute door normal, using axis as fallback")
        door_normal = axis.copy()
    
    door_normal = door_normal / (np.linalg.norm(door_normal) + 1e-12)
    
    print(f"[DEBUG] Prismatic door normal (world): {door_normal}")
    debug_draw_vector(pivot, door_normal, length=0.3)
    
    # Step 2: Verify axis and door_normal are parallel
    dot_product = abs(np.dot(axis, door_normal))
    print(f"[DEBUG] Prismatic axis·door_normal = {dot_product:.3f} (1=parallel)")
    
    if dot_product < 0.9:
        print(f"[WARNING] Axis and door normal are not parallel (dot={dot_product:.3f})")
    
    # Step 3: Rotate axis 90° "upward"
    v = axis
    g = np.array([0.0, 0.0, 1.0])  # World up
    
    u = g - np.dot(g, v) * v
    
    if np.linalg.norm(u) < 1e-6:
        print(f"[DEBUG] Axis is vertical, using world X")
        g = np.array([1.0, 0.0, 0.0])
        u = g - np.dot(g, v) * v
    
    upward_normal = u / (np.linalg.norm(u) + 1e-12)
    
    print(f"[DEBUG] Sliding axis: {axis}")
    print(f"[DEBUG] Door normal: {door_normal}")
    print(f"[DEBUG] Upward normal: {upward_normal}")
    
    # Angle filter vs door normal (handle rejection)
    angle_tolerance_deg = 3.0
    angle_tol_rad = np.deg2rad(angle_tolerance_deg)
    max_abs_dot = np.sin(angle_tol_rad)
    
    # Step 4: Filter points along upward direction + angle constraint
    upward_projections = []
    angle_ok_mask = []
    
    for point in surface_points:
        projection = np.dot(point, upward_normal)
        upward_projections.append(projection)
        
        v = point - pivot
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-8:
            angle_ok_mask.append(False)
        else:
            v_dir = v / v_norm
            dot = abs(np.dot(v_dir, door_normal))
            angle_ok_mask.append(dot <= max_abs_dot)
    
    upward_projections = np.array(upward_projections)
    angle_ok_mask = np.array(angle_ok_mask, dtype=bool)
    
    if np.any(angle_ok_mask):
        max_projection = upward_projections[angle_ok_mask].max()
    else:
        # If angle filter kills everything, fall back to all points for height
        print("[WARNING] All points rejected by angle filter; ignoring angle for height threshold")
        max_projection = upward_projections.max()
    
    threshold = max_projection - epsilon
    keep_mask = (upward_projections >= threshold) & angle_ok_mask
    
    filtered_points = surface_points[keep_mask]
    
    print(
        f"[INFO] Prismatic edge filter: {filtered_points.shape[0]}/"
        f"{surface_points.shape[0]} points kept "
        f"(angle_ok={angle_ok_mask.sum()}/{surface_points.shape[0]})"
    )
    
    # Step 5: Return upward normal and edge direction (sliding axis)
    normals = np.tile(upward_normal, (filtered_points.shape[0], 1))
    edge_directions = np.tile(axis, (filtered_points.shape[0], 1))  # Edge follows sliding axis
    
    return filtered_points, normals, edge_directions

def normal_and_tangent_to_gripper_quaternion(
    normal: np.ndarray,
    tangent: np.ndarray
) -> np.ndarray:
    """
    Convert surface normal and edge tangent to gripper orientation quaternion.
    
    The gripper is oriented such that:
    - GRIPPER_APPROACH_LOCAL (0,0,1) aligns with normal
    - GRIPPER_FINGER_LINE_LOCAL (0,1,0) is perpendicular to tangent
    
    Args:
        normal: [3] unit normal vector (gripper approach direction)
        tangent: [3] unit tangent vector (edge line direction)
    
    Returns:
        [4] quaternion [w, x, y, z] for gripper orientation
    
    Raises:
        ValueError: If tangent is parallel to normal (degenerate configuration)
    """
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
    
    # Target gripper frame axes in world coordinates:
    # - Gripper local Z (0,0,1) should align with normal
    # - Gripper local Y (0,1,0) should be aligned to tangent
    
    z_world = -normal  # Gripper's local Z in world frame
    
    y_world = tangent
    y_norm = np.linalg.norm(y_world)
    
    if y_norm < 1e-6:
        # Tangent is parallel to normal - degenerate case
        # This should never happen if edge filtering is correct
        raise ValueError(
            f"[ERROR] Tangent is parallel to normal - cannot constrain gripper orientation!\n"
            f"  normal: {normal}\n"
            f"  tangent: {tangent}\n"
            f"  dot product: {np.dot(normal, tangent):.6f}\n"
            f"This indicates a bug in edge point filtering."
        )
    
    y_world = y_world / y_norm
    
    # Gripper's local X completes the right-handed frame
    # x_world = y_world × z_world
    x_world = np.cross(y_world, z_world)
    x_world = x_world / (np.linalg.norm(x_world) + 1e-12)
    
    # Build rotation matrix: columns are the world-frame representations
    # of the gripper's local X, Y, Z axes
    rotation_matrix = np.column_stack([x_world, y_world, z_world])
    
    # Convert rotation matrix to quaternion
    rot = R.from_matrix(rotation_matrix)
    quat_xyzw = rot.as_quat()  # [x, y, z, w]
    
    # Convert to [w, x, y, z] format
    quat_wxyz = np.array([
        quat_xyzw[3],  # w
        quat_xyzw[0],  # x
        quat_xyzw[1],  # y
        quat_xyzw[2]   # z
    ])
    
    return quat_wxyz

def generate_and_filter_grasps_for_joint(stage,
                                         joint_name: str,
                                         child_link_name: str,
                                         joint_axis: np.ndarray,
                                         object_wrapper_path: str,
                                         object_ref_path: str,
                                         gripper_wrapper_path: str,
                                         num_samples: int = 500) -> list:
    """Generate grasp poses by sampling surfaces and filtering edge points."""
    
    print(f"\n[INFO] Generating grasps for {joint_name} -> {child_link_name}")
    
    # Step 1: Collect surface samples
    child_link_points = collect_all_child_link_surface_samples(
        stage, 
        object_ref_path, 
        child_link_name,
        n_points_per_mesh=SURFACE_SAMPLES_PER_MESH
    )
    
    if child_link_points.shape[0] == 0:
        print(f"[ERROR] No surface points sampled")
        return []
    
    print(f"[INFO] Sampled {child_link_points.shape[0]} surface points")
    
    # Step 2: Get joint parameters
    joint_prim = find_usd_joint_prim(stage, object_ref_path, joint_name)
    if joint_prim is None:
        print(f"[ERROR] Could not find USD joint")
        return []
    
    joint_params = get_joint_world_parameters(stage, joint_prim)
    if joint_params is None:
        print(f"[ERROR] Could not extract joint parameters")
        return []
    
    pivot = joint_params["pivot"]
    axis = joint_params["axis"]
    joint_type = joint_params["joint_type"]
    
    # Step 3: Filter edge points and get normals + tangents
    root_path = resolve_asset_root_under_ref(stage, object_ref_path)
    if joint_type == "revolute":
        print(f"[INFO] Filtering revolute edge points...")
        filtered_points, normals, tangents = filter_revolute_edge_points(
            stage,
            joint_prim,
            joint_params,
            child_link_points,
            pivot,
            axis,
            epsilon=0.01
        )
    elif joint_type == "prismatic":  # prismatic
        print(f"[INFO] Filtering prismatic edge points...")
        filtered_points, normals, edge_directions = filter_prismatic_edge_points(
            child_link_points,
            pivot,
            axis,
            stage,
            object_wrapper_path,
            root_path,
            child_link_name,
            epsilon=0.01
        )
        tangents = edge_directions  # For prismatic, tangent = sliding axis
    
    if filtered_points.shape[0] == 0:
        print(f"[ERROR] No edge points after filtering")
        return []
    
    print(f"[INFO] Filtered to {filtered_points.shape[0]} edge points")
    
    # Step 4: Convert (point, normal, tangent) → (position, quaternion)
    grasp_poses = []
    skipped_count = 0
    for point, normal, tangent in zip(filtered_points, normals, tangents):
        
        try:
            # Use both normal and tangent to fully constrain orientation
            quaternion = normal_and_tangent_to_gripper_quaternion(normal, tangent)
            position, _ = offset_pose_along_local_z(point, quaternion, -GRIPPER_FINGERTIP_OFFSET)
            grasp_poses.append((position, quaternion))
        except ValueError as e:
            print(f"[ERROR] Skipping grasp: {e}")
            skipped_count += 1
            continue
    
    if skipped_count > 0:
        print(f"[WARNING] Skipped {skipped_count}/{filtered_points.shape[0]} grasps due to degenerate orientations")
    print(f"[INFO] Generated {len(grasp_poses)} fully-constrained grasp poses")
    
    #step 5: Filter grasps that are inside the object that may need pre-procedures like opening the door first
    print(f"[INFO] Checking if grasp fingertips are inside the object...")
    grasp_poses = filter_interior_grasps(
        stage,
        grasp_poses,
        object_ref_path,
        object_wrapper_path,
        joint_name,
    )
    
    return grasp_poses

def filter_interior_grasps(
    stage,
    grasp_poses: list,
    object_ref_path: str,
    object_wrapper_path: str,
    joint_name: str,
    rejection_ratio: float = INTERIOR_REJECTION_RATIO,
    surface_sample_count: int = 5000,
    cone_half_angle_deg: float = 30.0,
) -> list:
    """
    Check if grasp fingertips are inside the object mesh.
    If more than rejection_ratio of them are interior, return empty list (reject entire joint).
    Otherwise return only the exterior grasps.
    
    Then we compare: is the fingertip closer to the object center than the
    surrounding surface shell? If yes → interior.
    
    Args:
        stage: USD stage
        grasp_poses: List of (position, quaternion) tuples
        object_ref_path: Path to object reference
        object_wrapper_path: Path to object wrapper
        joint_name: Name of the joint (for logging)
        rejection_ratio: If this fraction are interior, reject ALL grasps for the joint
        surface_sample_count: Number of surface points to sample
        cone_half_angle_deg: Half-angle of directional cone for surface lookup
    
    Returns:
        Filtered list of (position, quaternion) — or empty list if joint is rejected
    """
    # Get object center
    obj_center = compute_object_center(stage, object_wrapper_path)
    
    # Collect surface points from entire object
    surface_points = collect_mesh_points_world(stage, object_ref_path, max_points=surface_sample_count)
    
    if surface_points.shape[0] < 100:
        print(f"[INTERIOR-CHECK] Too few surface points ({surface_points.shape[0]}), skipping check")
        return grasp_poses
    
    # Precompute normalized directions from center to each surface point
    surface_vecs = surface_points - obj_center
    surface_dists = np.linalg.norm(surface_vecs, axis=1)
    valid_mask = surface_dists > 1e-6
    surface_points = surface_points[valid_mask]
    surface_vecs = surface_vecs[valid_mask]
    surface_dists = surface_dists[valid_mask]
    surface_dirs = surface_vecs / surface_dists[:, None]
    
    cos_threshold = np.cos(np.radians(cone_half_angle_deg))
    
    exterior_grasps = []
    inside_count = 0
    
    for pos, quat in grasp_poses:
        pos_np = np.asarray(pos, dtype=np.float64).reshape(3)
        test_point = pos_np

        # Vector from object center to test point
        to_point = test_point - obj_center
        dist_to_point = np.linalg.norm(to_point)
        
        if dist_to_point < 1e-6:
            # Exactly at center — definitely inside
            inside_count += 1
            continue
        
        test_dir = to_point / dist_to_point
        
        # Find surface points in similar direction (within cone)
        cos_angles = surface_dirs @ test_dir
        cone_mask = cos_angles > cos_threshold
        
        if not np.any(cone_mask):
            # No nearby surface points — can't determine, assume outside
            exterior_grasps.append((pos, quat))
            continue
        
        # Shell distance = 90th percentile of surface distances in this direction
        nearby_surface_dists = surface_dists[cone_mask]
        shell_dist = np.percentile(nearby_surface_dists, 90)
        
        # If fingertip is closer to center than the shell, it's inside
        if dist_to_point < shell_dist * 0.85:
            inside_count += 1
        else:
            exterior_grasps.append((pos, quat))
    
    num_grasps = len(grasp_poses)
    interior_ratio = inside_count / num_grasps if num_grasps > 0 else 0.0
    
    print(f"[INTERIOR-CHECK] Joint '{joint_name}': "
          f"{inside_count}/{num_grasps} interior ({interior_ratio:.1%}), "
          f"{len(exterior_grasps)} exterior")
    
    if interior_ratio >= rejection_ratio:
        print(f"[INTERIOR-REJECT] Rejecting ALL grasps for joint '{joint_name}' "
              f"({interior_ratio:.1%} >= {rejection_ratio:.1%} threshold)")
        return []
    
    print(f"[INTERIOR-KEEP] Keeping {len(exterior_grasps)}/{num_grasps} exterior grasps "
          f"for joint '{joint_name}'")
    return exterior_grasps

def filter_joints(
    stage,
    object_ref_path: str,
    joints_info: List[Tuple[str, str, str, float, float, np.ndarray]],
) -> List[Tuple[str, str, str, float, float, np.ndarray]]:
    """
    Filter out "tray-like" joints whose CHILD link geometry is completely inside the
    object's overall bounding box (i.e., not touching the outside shell).

    joints_info tuples:
      (joint_name, joint_type, child_link_name, lower_limit, upper_limit, joint_axis)

    Rule:
      - If only one joint, keep it (no filtering).
      - Otherwise, discard a joint if child's world AABB is fully contained inside
        the object's world AABB after shrinking the object AABB by a safety margin.

    Notes:
      - This does NOT require the parent link. It relies on "child fully inside the object"
        which works well for internal sliders/trays.
    """
    if len(joints_info) <= 1:
        return joints_info

    # ---- helpers ----
    def _compute_world_aabb(prim_path: str):
        from pxr import UsdGeom, Usd

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None

        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        wb = cache.ComputeWorldBound(prim)
        box = wb.GetBox()
        mn = box.GetMin()
        mx = box.GetMax()
        return (
            np.array([mn[0], mn[1], mn[2]], dtype=np.float64),
            np.array([mx[0], mx[1], mx[2]], dtype=np.float64),
        )

    def _get_link_bbox(child_link_name: str):
        # Your USD layout often has .../{link}/visuals
        asset_root = resolve_asset_root_under_ref(stage, object_ref_path)
        candidates = [
            f"{asset_root}/{child_link_name}/visuals",
            f"{asset_root}/{child_link_name}",
        ]
        for p in candidates:
            bb = _compute_world_aabb(p)
            if bb is not None:
                return bb
        return None

    # Prefer "visuals" bbox if present; else whole object.
    asset_root = resolve_asset_root_under_ref(stage, object_ref_path)
    obj_bb = _compute_world_aabb(f"{asset_root}/visuals")
    if obj_bb is None:
        obj_bb = _compute_world_aabb(asset_root)

    if obj_bb is None:
        print("[WARN] filter_joints: cannot compute object bbox. No filtering.")
        return joints_info

    obj_min, obj_max = obj_bb
    obj_diag = float(np.linalg.norm(obj_max - obj_min))

    # Margin: shrink object bbox inward before containment test.
    # - abs: ~1cm
    # - ratio: ~2% of object diagonal
    margin_abs = 0.01
    margin_ratio = 0.01
    margin = max(margin_abs, margin_ratio * obj_diag)

    # If shrink would invert the box, skip filtering.
    obj_min_s = obj_min + margin
    obj_max_s = obj_max - margin
    if np.any(obj_min_s >= obj_max_s):
        print("[WARN] filter_joints: object bbox too small after margin shrink. No filtering.")
        return joints_info

    # Optional extra guard: avoid discarding large "near-shell" parts accidentally
    # by requiring the child to be "meaningfully smaller" than the object.
    # (Good for keeping big doors/panels that might barely be inside numerically.)
    vol_obj = float(np.prod(np.maximum(obj_max - obj_min, 1e-12)))
    keep: List[Tuple[str, str, str, float, float, np.ndarray]] = []

    for (joint_name, joint_type, child_link_name, lower, upper, axis) in joints_info:
        child_bb = _get_link_bbox(child_link_name)
        if child_bb is None:
            # Can't evaluate -> keep
            keep.append((joint_name, joint_type, child_link_name, lower, upper, axis))
            continue

        cmin, cmax = child_bb
        inside = (np.all(cmin >= obj_min_s) and np.all(cmax <= obj_max_s))
        
        print("\n====================================================")
        print(f"[DEBUG] Joint: {joint_name}")
        print(f"Child link: {child_link_name}")

        print("\nObject bbox (shrunk):")
        print(f"  min: {obj_min_s}")
        print(f"  max: {obj_max_s}")

        print("\nChild bbox:")
        print(f"  min: {cmin}")
        print(f"  max: {cmax}")

        print(f"\nFinal inside decision: {inside}")
        print("====================================================")

        # Size guard: child must be relatively small to be considered a tray-like internal part
        vol_child = float(np.prod(np.maximum(cmax - cmin, 1e-12)))
        rel = vol_child / max(vol_obj, 1e-12)
        small_enough = (rel < 0.35)  # tune: 0.2~0.4 typically

        if inside:
            print(f"[SKIP] {joint_name}: child '{child_link_name}' fully inside object bbox (rel_vol={rel:.3f})")
            continue

        keep.append((joint_name, joint_type, child_link_name, lower, upper, axis))

    # If filtering removed everything (rare), fall back to original.
    return keep if keep else joints_info
      

def generate_grasps_for_all_joints(stage,
                                   joints_info,
                                   object_wrapper_path: str,
                                   object_ref_path: str,
                                   gripper_wrapper_path: str,
                                   num_samples_per_joint: int = 500) -> Dict[str, Dict]:
    """Generate filtered grasp poses for all articulated joints."""
    
    if not joints_info:
        print(f"[ERROR] No movable joints found")
        return {}
    
    print(f"\n[INFO] Found {len(joints_info)} movable joint(s)")
    
    all_grasps = {}
    
    # Updated unpacking - no handle_visual_indices
    for joint_name, joint_type, child_link_name, lower_limit, upper_limit, joint_axis in joints_info:
        print(f"\n{'='*80}")
        print(f"Processing: {joint_name} ({joint_type}) -> {child_link_name}")
        print(f"  Joint axis: {joint_axis}")
        print(f"  Will sample entire child link surface")
        print(f"{'='*80}")
        
        grasp_poses = generate_and_filter_grasps_for_joint(
            stage,
            joint_name,
            child_link_name,
            joint_axis,
            object_wrapper_path,
            object_ref_path,
            gripper_wrapper_path,
            num_samples_per_joint
        )
        
        if grasp_poses:
            all_grasps[joint_name] = {
                'grasp_poses': grasp_poses,
                'joint_type': joint_type,
                'child_link_name': child_link_name,
                'lower_limit': lower_limit,
                'upper_limit': upper_limit,
                'joint_axis': joint_axis
            }
            print(f"[SUCCESS] Generated {len(grasp_poses)} filtered grasp(es)")
        else:
            print(f"[WARNING] No valid grasps after filtering")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: Generated grasps for {len(all_grasps)}/{len(joints_info)} joint(s)")
    print(f"{'='*80}\n")
    
    return all_grasps

def overshoot_reject(
    stage,
    joint_prim,
    initial_joint_pos: float,
    should_displacement: float,
    target_displacement: float,
    joint_type: str,
) -> bool:
    cur = get_joint_current_position(stage, joint_prim)
    if cur is None:
        return True  # conservative reject

    open_dir = 1.0 if target_displacement >= 0.0 else -1.0
    desired = initial_joint_pos + should_displacement
    overshoot = open_dir * (float(cur) - desired)

    tol = (
        np.deg2rad(OVERSHOOT_TOLERANCE_DEG)
        if joint_type == "revolute"
        else OVERSHOOT_TOLERANCE_M
    )
    
    if overshoot > tol:
        print(
            f"[OVERSHOOT-REJECT] joint={joint_prim.GetPath()} "
            f"cur={cur:.6f} desired={desired:.6f} should_disp={should_displacement:.6f} "
            f"target_disp={target_displacement:.6f} overshoot={overshoot:.6f} tol={tol:.6f}"
        )
        return True

    return False

# =======================
# Trajectory Planning
# =======================

def get_body_world_position(stage, body_path: str) -> Optional[np.ndarray]:
    """Get world-space position (translation) of a body/link prim, matching the GUI transform."""
    prim = stage.GetPrimAtPath(body_path)
    if not prim.IsValid():
        return None

    xformable = UsdGeom.Xformable(prim)
    world_xf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = world_xf.ExtractTranslation()

    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)

def get_body_env_local_position(stage, body_path: str, env_root_path: str) -> Optional[np.ndarray]:
    """
    Returns the position of `body_path` in the *environment's local frame*.
    Uses USD xform math directly, avoiding world space altogether.
    """
    body_prim = stage.GetPrimAtPath(body_path)
    env_prim  = stage.GetPrimAtPath(env_root_path)

    if not body_prim.IsValid() or not env_prim.IsValid():
        return None

    body_xf = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    env_xf  = UsdGeom.Xformable(env_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # Convert body → env-local by removing env's world transform
    env_local_xf = body_xf * env_xf.GetInverse()

    # Extract translation
    t = env_local_xf.ExtractTranslation()
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)

def find_usd_joint_prim(stage, object_ref_path: str, joint_name_from_urdf: str) -> Optional[Usd.Prim]:
    """
    Find USD joint prim matching URDF joint name.

    Joints are located at: {object_ref_path}/partnet_.../joints/joint_{num}

    Args:
        stage: USD stage
        object_ref_path: Path to object reference (e.g., /World/Envs/env_0/Object/ref)
        joint_name_from_urdf: Joint name from URDF (e.g., 'joint_0', 'joint_1')

    Returns:
        Joint prim or None
    """
    # Extract joint number from URDF name
    joint_name_clean = joint_name_from_urdf.lower().replace("joint_", "").replace("joint", "")

    asset_root = resolve_asset_root_under_ref(stage, object_ref_path)
    joints_root = f"{asset_root}/joints"
    direct_joint_path = f"{joints_root}/joint_{joint_name_clean}"

    direct_prim = stage.GetPrimAtPath(direct_joint_path)
    if direct_prim.IsValid() and (
        direct_prim.IsA(UsdPhysics.RevoluteJoint) or direct_prim.IsA(UsdPhysics.PrismaticJoint)
    ):
        print(f"[INFO] Found USD joint via direct path: {direct_joint_path}")
        return direct_prim

    # Fallback: search under /joints
    joints_root_prim = stage.GetPrimAtPath(joints_root)
    if not joints_root_prim.IsValid():
        print(f"[ERROR] No /joints folder found at {joints_root}")
        return None

    for prim in joints_root_prim.GetChildren():
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
            continue

        prim_name = prim.GetName().lower()

        if (
            joint_name_from_urdf.lower() in prim_name
            or joint_name_clean in prim_name
            or f"joint_{joint_name_clean}" == prim_name
        ):
            print(f"[INFO] Found USD joint via search: {prim.GetPath()}")
            return prim

    print(f"[WARNING] No USD joint found for URDF joint: {joint_name_from_urdf}")
    print(f"[DEBUG] Searched in: {joints_root}")
    print(f"[DEBUG] Available joints under {joints_root}:")
    for prim in joints_root_prim.GetChildren():
        print(f"  - {prim.GetName()} (type: {prim.GetTypeName()})")

    return None


def _gfquat_to_mat3d(q) -> Gf.Matrix3d:
    """Convert pxr Gf.Quat* or 4-seq to a 3x3 rotation matrix."""
    if hasattr(q, "GetReal") and hasattr(q, "GetImaginary"):
        w = float(q.GetReal())
        im = q.GetImaginary()
        x, y, z = float(im[0]), float(im[1]), float(im[2])
    else:
        arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if arr.size != 4:
            raise ValueError(f"Unexpected quat size {arr.size}: {q}")
        # Interpret as [w, x, y, z]
        w, x, y, z = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])

    rot = R.from_quat([x, y, z, w])  # scipy expects [x, y, z, w]
    m = rot.as_matrix()
    return Gf.Matrix3d(
        m[0, 0],
        m[0, 1],
        m[0, 2],
        m[1, 0],
        m[1, 1],
        m[1, 2],
        m[2, 0],
        m[2, 1],
        m[2, 2],
    )


def _make_transform_gf(t: np.ndarray, q) -> Gf.Matrix4d:
    """Make a Gf.Matrix4d from translation (3,) and quaternion q."""
    t = np.asarray(t, dtype=np.float64).reshape(3)
    r3 = _gfquat_to_mat3d(q)
    m = Gf.Matrix4d(1.0)
    m.SetTranslateOnly(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))
    # Set rotation block
    m.SetRow3(0, Gf.Vec3d(r3[0][0], r3[0][1], r3[0][2]))
    m.SetRow3(1, Gf.Vec3d(r3[1][0], r3[1][1], r3[1][2]))
    m.SetRow3(2, Gf.Vec3d(r3[2][0], r3[2][1], r3[2][2]))
    return m


def _get_body_world_xf(stage, body_path: str) -> Optional[Gf.Matrix4d]:
    if not body_path:
        return None
    prim = stage.GetPrimAtPath(body_path)
    if not prim.IsValid():
        return None
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return xf


def _compute_joint_frame_world(stage, joint_prim: Usd.Prim) -> Optional[Tuple[Gf.Matrix4d, str]]:
    """Compute joint frame in world using (body0, localPos0/localRot0) if available, else body1."""
    if not joint_prim.IsValid():
        return None

    j = UsdPhysics.Joint(joint_prim)

    # Resolve body0/body1 targets
    body0_targets = []
    body1_targets = []
    try:
        rel0 = j.GetBody0Rel()
        if rel0 and rel0.IsValid():
            body0_targets = rel0.GetTargets()
    except Exception:
        pass
    try:
        rel1 = j.GetBody1Rel()
        if rel1 and rel1.IsValid():
            body1_targets = rel1.GetTargets()
    except Exception:
        pass

    body0_path = body0_targets[0].pathString if body0_targets else ""
    body1_path = body1_targets[0].pathString if body1_targets else ""

    # localPos/localRot for each body
    lp0 = (
        j.GetLocalPos0Attr().Get()
        if (j.GetLocalPos0Attr() and j.GetLocalPos0Attr().IsValid())
        else Gf.Vec3f(0, 0, 0)
    )
    lr0 = (
        j.GetLocalRot0Attr().Get()
        if (j.GetLocalRot0Attr() and j.GetLocalRot0Attr().IsValid())
        else Gf.Quatf(1, 0, 0, 0)
    )

    lp1 = (
        j.GetLocalPos1Attr().Get()
        if (j.GetLocalPos1Attr() and j.GetLocalPos1Attr().IsValid())
        else Gf.Vec3f(0, 0, 0)
    )
    lr1 = (
        j.GetLocalRot1Attr().Get()
        if (j.GetLocalRot1Attr() and j.GetLocalRot1Attr().IsValid())
        else Gf.Quatf(1, 0, 0, 0)
    )

    # Prefer body0 frame; fall back to body1
    if body0_path:
        body_xf = _get_body_world_xf(stage, body0_path)
        if body_xf is not None:
            local_xf = _make_transform_gf(np.array([lp0[0], lp0[1], lp0[2]], dtype=np.float64), lr0)
            return body_xf * local_xf, body0_path

    if body1_path:
        body_xf = _get_body_world_xf(stage, body1_path)
        if body_xf is not None:
            local_xf = _make_transform_gf(np.array([lp1[0], lp1[1], lp1[2]], dtype=np.float64), lr1)
            return body_xf * local_xf, body1_path

    return None


def get_joint_world_parameters(stage, joint_prim: Usd.Prim) -> Optional[Dict]:
    """Extract joint parameters in world coordinates.

    IMPORTANT:
      USD Physics joints define their constraint frames via body0/body1 and localPos*/localRot*.
      The joint prim's own Xform is often NOT the physical joint frame.

    Returns:
        Dict with:
        - joint_type: 'revolute' or 'prismatic'
        - axis: [3] world-frame axis
        - pivot: [3] world-frame pivot/reference point (child body center)
        - lower_limit: float (radians for revolute, meters for prismatic)
        - upper_limit: float (radians for revolute, meters for prismatic)
        - limit_units: 'rad', 'deg', or 'm'
        - joint_frame_world: Gf.Matrix4d
        - axis_token: 'X', 'Y', or 'Z'
        - used_body_path: which body was used for pivot
    """
    if not joint_prim.IsValid():
        return None

    # Determine joint type
    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        joint_usd = UsdPhysics.RevoluteJoint(joint_prim)
        joint_type = "revolute"
    elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
        joint_usd = UsdPhysics.PrismaticJoint(joint_prim)
        joint_type = "prismatic"
    else:
        print(f"[ERROR] Unknown joint type: {joint_prim.GetPath()}")
        return None

    # Compute physical joint frame in world using body0/body1 + localPose
    jf = _compute_joint_frame_world(stage, joint_prim)
    if jf is None:
        print(f"[ERROR] Could not compute joint frame from body0/body1 for {joint_prim.GetPath()}")
        return None
    joint_frame_world, frame_body_path = jf

    # Axis token (X/Y/Z) is defined in the JOINT FRAME
    axis_attr = joint_usd.GetAxisAttr()
    if axis_attr and axis_attr.IsValid():
        axis_token = str(axis_attr.Get())
    else:
        axis_token = "Z"

    axis_map = {
        "X": np.array([1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "Z": np.array([0.0, 0.0, 1.0]),
    }
    axis_local = axis_map.get(axis_token, axis_map["Z"]).astype(np.float64)

    # Transform axis from joint frame -> world
    axis_world_gf = joint_frame_world.TransformDir(
        Gf.Vec3d(float(axis_local[0]), float(axis_local[1]), float(axis_local[2]))
    )
    axis_world = np.array(
        [float(axis_world_gf[0]), float(axis_world_gf[1]), float(axis_world_gf[2])],
        dtype=np.float64,
    )
    axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-12)

    # Prefer child body (body1) world position as pivot for motion planning.
    body1_targets = []
    try:
        rel1 = joint_usd.GetBody1Rel()
        if rel1 and rel1.IsValid():
            body1_targets = rel1.GetTargets()
    except Exception:
        pass

    body1_path = body1_targets[0].pathString if body1_targets else ""

    pivot_world = None
    used_body_path = ""

    if body1_path:
        pivot_center = get_body_world_position(stage, body1_path)
        if pivot_center is not None:
            pivot_world = pivot_center
            used_body_path = body1_path

    # Fallback: use the origin of the joint frame if body1 position is unavailable
    if pivot_world is None:
        pivot_world_gf = joint_frame_world.Transform(Gf.Vec3d(0, 0, 0))
        pivot_world = np.array(
            [float(pivot_world_gf[0]), float(pivot_world_gf[1]), float(pivot_world_gf[2])],
            dtype=np.float64,
        )
        used_body_path = frame_body_path

    # Limits
    lower_attr = joint_usd.GetLowerLimitAttr()
    upper_attr = joint_usd.GetUpperLimitAttr()
    lower_raw = lower_attr.Get() if (lower_attr and lower_attr.IsValid()) else None
    upper_raw = upper_attr.Get() if (upper_attr and upper_attr.IsValid()) else None

    if joint_type == "revolute":
        if lower_raw is None or upper_raw is None:
            lower = -np.pi
            upper = np.pi
            limit_units = "rad"
        else:
            lower_f = float(lower_raw)
            upper_f = float(upper_raw)
            # degree-like if clearly > 2*pi
            if max(abs(lower_f), abs(upper_f)) > 6.5:
                lower = np.radians(lower_f)
                upper = np.radians(upper_f)
                limit_units = "deg"
            else:
                lower = lower_f
                upper = upper_f
                limit_units = "rad"
    else:  # prismatic
        lower = float(lower_raw) if lower_raw is not None else -0.5
        upper = float(upper_raw) if upper_raw is not None else 0.5
        limit_units = "m"

    print(f"[DEBUG] Joint frame from {used_body_path}: {joint_prim.GetPath()}")
    print(f"[DEBUG]   axis_token={axis_token}, axis_world={axis_world}, pivot_world={pivot_world}")
    print(f"[DEBUG]   limits=({lower_raw},{upper_raw}) -> ({lower},{upper}) units={limit_units}")

    return {
        "joint_prim": joint_prim,
        "joint_type": joint_type,
        "axis": axis_world,
        "pivot": pivot_world,
        "lower_limit": lower,
        "upper_limit": upper,
        "limit_units": limit_units,
        "joint_frame_world": joint_frame_world,
        "axis_token": axis_token,
        "used_body_path": used_body_path,
    }


def compute_target_joint_displacement(joint_params: Dict, opening_fraction: float = 0.7) -> float:
    """Compute signed delta motion (angle or distance) from joint limits."""
    lower = float(joint_params["lower_limit"])
    upper = float(joint_params["upper_limit"])
    joint_range = upper - lower

    # Plan from the "closed" end (lower) toward opening.
    OPEN_SIGN = 1.0
    target_delta = OPEN_SIGN * float(opening_fraction) * float(joint_range)

    if joint_params["joint_type"] == "revolute":
        # Cap to 90 degrees for safety
        max_opening = np.radians(90) * float(opening_fraction)
        target_delta = np.clip(target_delta, -max_opening, max_opening)
        print(f"[DEBUG] Target revolute delta: {np.degrees(target_delta):.1f}°")
    else:
        print(f"[DEBUG] Target prismatic delta: {target_delta:.4f}m")

    return float(target_delta)

def compute_gripper_orientation_for_trajectory(
    trajectory: np.ndarray,
    initial_grasp_quat: np.ndarray,
    joint_params: Optional[Dict] = None,
    method: str = "fixed",
) -> np.ndarray:
    """
    Compute gripper orientation at each trajectory waypoint.

    Args:
        trajectory: [num_steps, 3] position waypoints
        initial_grasp_quat: [4] initial quaternion [w, x, y, z]
        joint_params: Dict with joint_type, axis, pivot (needed for revolute)
        method: 'fixed' or 'revolute_follow'

    Returns:
        orientations: [num_steps, 4] quaternions [w, x, y, z]
    """
    num_steps = trajectory.shape[0]

    if method == "fixed":
        return np.tile(initial_grasp_quat, (num_steps, 1))

    elif method == "revolute_follow":
        if joint_params is None or joint_params["joint_type"] != "revolute":
            return np.tile(initial_grasp_quat, (num_steps, 1))
        
        pivot = np.asarray(joint_params["pivot"], dtype=np.float64)
        axis = np.asarray(joint_params["axis"], dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        
        # Vector from pivot to initial grasp position
        r0 = trajectory[0] - pivot
        # Project perpendicular to axis (component in rotation plane)
        r0_perp = r0 - np.dot(r0, axis) * axis
        
        orientations = []
        
        for i in range(num_steps):
            # Current position vector from pivot
            r_current = trajectory[i] - pivot
            r_current_perp = r_current - np.dot(r_current, axis) * axis
            
            # Calculate rotation angle from initial to current position
            norm_r0 = np.linalg.norm(r0_perp)
            norm_rc = np.linalg.norm(r_current_perp)
            
            if norm_r0 < 1e-6 or norm_rc < 1e-6:
                # On axis, no rotation
                orientations.append(initial_grasp_quat)
                continue
            
            # Angle between r0_perp and r_current_perp
            cos_theta = np.dot(r0_perp, r_current_perp) / (norm_r0 * norm_rc)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            
            # Use cross product to determine sign
            cross = np.cross(r0_perp, r_current_perp)
            sin_theta = np.dot(cross, axis) / (norm_r0 * norm_rc)
            
            theta = np.arctan2(sin_theta, cos_theta)
            
            # Create incremental rotation quaternion around axis
            # Rotation axis-angle to quaternion: q = [cos(θ/2), sin(θ/2) * axis]
            half_theta = theta / 2.0
            quat_increment_xyzw = np.array([
                np.sin(half_theta) * axis[0],
                np.sin(half_theta) * axis[1],
                np.sin(half_theta) * axis[2],
                np.cos(half_theta)
            ])
            
            # Convert initial grasp quat from [w,x,y,z] to [x,y,z,w] for scipy
            initial_quat_xyzw = np.array([
                initial_grasp_quat[1],
                initial_grasp_quat[2],
                initial_grasp_quat[3],
                initial_grasp_quat[0]
            ])
            
            # Compose rotations: new_rot = increment_rot * initial_rot
            rot_increment = R.from_quat(quat_increment_xyzw)
            rot_initial = R.from_quat(initial_quat_xyzw)
            rot_new = rot_increment * rot_initial
            
            # Convert back to [w,x,y,z]
            new_quat_xyzw = rot_new.as_quat()
            new_quat_wxyz = np.array([
                new_quat_xyzw[3],  # w
                new_quat_xyzw[0],  # x
                new_quat_xyzw[1],  # y
                new_quat_xyzw[2]   # z
            ])
            
            orientations.append(new_quat_wxyz)
        
        return np.array(orientations)
    
    else:
        raise ValueError(f"Unknown orientation method: {method}")

def plan_revolute_joint_trajectory(
    grasp_position: np.ndarray,
    joint_params: Dict,
    target_angle: float,
    num_steps: int = 200,
) -> np.ndarray:
    """
    Plan circular arc trajectory for revolute joint (door opening).
    Gripper base follows circular path around hinge.

    Args:
        grasp_position: [3] gripper base position in world frame
        joint_params: Dict from get_joint_world_parameters()
        target_angle: Desired opening angle in radians
        num_steps: Number of trajectory waypoints

    Returns:
        trajectory: [num_steps, 3] position waypoints
    """
    pivot = joint_params["pivot"]
    axis = joint_params["axis"]

    print(f"[DEBUG] Revolute trajectory:")
    print(f"  Pivot: {pivot}")
    print(f"  Axis: {axis}")
    print(f"  Angle: {np.degrees(target_angle):.1f}°")

    # Vector from pivot to grasp
    r_vec = grasp_position - pivot

    # Decompose: r_vec = r_parallel + r_perp
    r_parallel = np.dot(r_vec, axis) * axis
    r_perp = r_vec - r_parallel

    radius = np.linalg.norm(r_perp)
    print(f"  Radius: {radius:.4f}m")

    if radius < 1e-6:
        print(f"[WARNING] Grasp on rotation axis - no circular motion")
        return np.tile(grasp_position, (num_steps, 1))

    # Basis in rotation plane
    e1 = r_perp / radius  # Radial at t=0
    e2 = np.cross(axis, e1)  # Tangential at t=0
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)

    trajectory = np.zeros((num_steps, 3), dtype=np.float64)

    for i in range(num_steps):
        theta = (i / max(num_steps - 1, 1)) * target_angle
        r_rotated = radius * (np.cos(theta) * e1 + np.sin(theta) * e2)
        trajectory[i] = pivot + r_rotated + r_parallel

    return trajectory


def plan_prismatic_joint_trajectory(
    grasp_position: np.ndarray,
    joint_params: Dict,
    target_distance: float,
    num_steps: int = 200,
) -> np.ndarray:
    """
    Plan linear trajectory for prismatic joint (drawer opening).
    Gripper base moves in straight line along the joint axis.

    Args:
        grasp_position: [3] gripper base position in world frame
        joint_params: Dict from get_joint_world_parameters()
        target_distance: Desired displacement in meters
        num_steps: Number of trajectory waypoints

    Returns:
        trajectory: [num_steps, 3] position waypoints
    """
    axis = joint_params["axis"]

    print(f"[DEBUG] Prismatic trajectory:")
    print(f"  Axis: {axis}")
    print(f"  Distance: {target_distance:.4f}m")

    trajectory = np.zeros((num_steps, 3), dtype=np.float64)

    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        displacement = alpha * target_distance
        trajectory[i] = grasp_position + displacement * axis

    return trajectory


def offset_pose_along_local_z(
    position: np.ndarray,
    quaternion: np.ndarray,
    offset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Offset position along gripper's local Z-axis (approach direction).

    Args:
        position: [3] position
        quaternion: [4] quaternion [w,x,y,z]
        offset: Distance (positive = forward, negative = backward)

    Returns:
        (new_position, quaternion)
    """
    # Convert to scipy format [x,y,z,w]
    quat_xyzw = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rot = R.from_quat(quat_xyzw)

    local_z = np.array([0, 0, 1], dtype=np.float64)
    world_z = rot.apply(local_z)

    new_position = position + world_z * offset
    return new_position, quaternion


def generate_trajectories_for_all_grasps(
    stage,
    all_grasps: Dict[str, Dict],
    object_ref_path: str,
    num_trajectory_steps: int = 200,
    opening_fraction: float = 0.7,
) -> List[Dict]:
    """
    Generate trajectories for all grasp poses.

    Args:
        stage: USD stage
        all_grasps: Output from generate_grasps_for_all_joints()
        object_ref_path: Path to object reference
        num_trajectory_steps: Waypoints per trajectory
        opening_fraction: Fraction of joint range to open
        orientation_method: 'fixed' or 'tangent'

    Returns:
        List of trajectory dicts with:
        - joint_name, joint_type, grasp_index
        - grasp_position, grasp_quaternion
        - trajectory_positions: [num_steps, 3]
        - trajectory_orientations: [num_steps, 4]
        - target_displacement, joint_motion, joint_pivot_world
    """
    all_trajectories = []

    for joint_name, joint_data in all_grasps.items():
        print(f"\n[INFO] Planning trajectories for: {joint_name}")

        # Find joint in USD
        joint_prim = find_usd_joint_prim(stage, object_ref_path, joint_name)
        if joint_prim is None:
            print(f"[ERROR] Could not find USD joint for {joint_name}")
            continue

        # Get joint parameters from USD
        joint_params = get_joint_world_parameters(stage, joint_prim)
        if joint_params is None:
            print(f"[ERROR] Could not extract parameters for {joint_name}")
            continue

        # Compute target displacement (angle or distance)
        target_displacement = compute_target_joint_displacement(joint_params, opening_fraction)

        # Plan trajectory for each grasp
        grasp_poses = joint_data["grasp_poses"]

        for grasp_idx, (grasp_pos, grasp_quat) in enumerate(grasp_poses):
            try:
                # Convert grasp position
                if isinstance(grasp_pos, (Gf.Vec3d, Gf.Vec3f)):
                    grasp_pos_np = np.array(
                        [grasp_pos[0], grasp_pos[1], grasp_pos[2]], dtype=np.float64
                    )
                else:
                    grasp_pos_np = np.asarray(grasp_pos, dtype=np.float64)

                # Convert quaternion to [w,x,y,z]
                if isinstance(grasp_quat, (Gf.Quatd, Gf.Quatf)):
                    w = float(grasp_quat.GetReal())
                    im = grasp_quat.GetImaginary()
                    grasp_quat_np = np.array(
                        [w, im[0], im[1], im[2]], dtype=np.float64
                    )
                else:
                    grasp_quat_np = np.asarray(grasp_quat, dtype=np.float64)

                # (Optional) step-back logic could go here if you want
                start_pos = grasp_pos_np

                # Position trajectory
                if joint_params["joint_type"] == "revolute":
                    traj_positions = plan_revolute_joint_trajectory(
                        start_pos,
                        joint_params,
                        target_displacement,
                        num_trajectory_steps,
                    )
                    orientation_method_to_use = "revolute_follow"
                else:
                    traj_positions = plan_prismatic_joint_trajectory(
                        start_pos,
                        joint_params,
                        target_displacement,
                        num_trajectory_steps,
                    )
                    orientation_method_to_use = "fixed"

                # Orientation trajectory
                orientation_method_to_use = "revolute_follow" if joint_params["joint_type"] == "revolute" else "fixed"
                traj_orientations = compute_gripper_orientation_for_trajectory(
                    traj_positions,
                    grasp_quat_np,
                    joint_params=joint_params,
                    method=orientation_method_to_use,
                )

                # Compact motion info for downstream planners
                joint_motion = {
                    "joint_type": joint_params["joint_type"],
                    "axis": joint_params["axis"],
                    "lower_limit": joint_params["lower_limit"],
                    "upper_limit": joint_params["upper_limit"],
                    "limit_units": joint_params["limit_units"],
                    "axis_token": joint_params["axis_token"],
                    "used_body_path": joint_params["used_body_path"],
                }

                print(
                    f"[TRAJ] joint={joint_name}, "
                    f"type={joint_params['joint_type']}, "
                    f"grasp_idx={grasp_idx}, "
                    f"used_body={joint_params['used_body_path']}, "
                    f"axis={joint_params['axis']}"
                )

                all_trajectories.append(
                    {
                        "joint_name": joint_name,
                        "joint_type": joint_params["joint_type"],
                        "grasp_index": grasp_idx,
                        "grasp_position": grasp_pos_np,
                        "grasp_quaternion": grasp_quat_np,
                        "trajectory_positions": traj_positions,
                        "trajectory_orientations": traj_orientations,
                        "target_displacement": target_displacement,
                        # For motion planning:
                        "joint_motion": joint_motion,
                        # For visualization:
                        "joint_pivot_world": joint_params["pivot"],
                    }
                )

            except Exception as e:
                print(f"[ERROR] Failed to plan trajectory for grasp {grasp_idx}: {e}")
                import traceback

                traceback.print_exc()
                continue

        print(
            f"[INFO] Generated {len([t for t in all_trajectories if t['joint_name'] == joint_name])} "
            f"trajectories for joint {joint_name}"
        )

    print(f"\n[INFO] Total trajectories: {len(all_trajectories)}")
    return all_trajectories

def create_trajectory_batches(trajectories: List[Dict], num_envs: int) -> List[List[Dict]]:
    """
    Distribute trajectories across environments for batched validation.
    
    Args:
        trajectories: List of all trajectories to validate
        num_envs: Number of cloned environments (NUM_COPIES)
    
    Returns:
        List of batches, where each batch contains trajectories for one round of parallel execution
    """
    batches = []
    
    # Process in chunks of num_envs
    for i in range(0, len(trajectories), num_envs):
        batch = trajectories[i:i + num_envs]
        batches.append(batch)
    
    print(f"[INFO] Split {len(trajectories)} trajectories into {len(batches)} batches")
    print(f"[INFO] Batch sizes: {[len(b) for b in batches]}")
    
    return batches

def create_and_bind_high_friction_material(
    stage, 
    root_prim_path: str,
    static_friction: float = 2.0,
    dynamic_friction: float = 2.0,
    restitution: float = 0.0,
    pre_disabled_instanceable_paths: Optional[List[str]] = None,
):
    """
    Create a high-friction physics material and bind it above instanceable subtrees
    so the binding survives when instanceable is restored later.
    
    Args:
        stage: USD stage
        root_prim_path: Root prim path to apply material to (e.g., OBJECT_REF_PATH)
        static_friction: Static friction coefficient (default: 2.0)
        dynamic_friction: Dynamic friction coefficient (default: 2.0)
        restitution: Bounciness (default: 0.0 for no bounce)
    
    Returns:
        Tuple[int, List[str]]: Number of prims the material was applied to and
        prim paths whose instanceable flag was disabled for authoring.
    """
    # Create material if it doesn't exist
    material_path = "/World/Physics_Materials/HighFrictionMaterial"
    
    if not stage.GetPrimAtPath(material_path).IsValid():
        UsdShade.Material.Define(stage, material_path)
    
    mat_prim = stage.GetPrimAtPath(material_path)
    
    # Apply UsdPhysics.MaterialAPI
    if not mat_prim.HasAPI(UsdPhysics.MaterialAPI):
        UsdPhysics.MaterialAPI.Apply(mat_prim)
    
    p_mat = UsdPhysics.MaterialAPI(mat_prim)
    p_mat.CreateStaticFrictionAttr().Set(float(static_friction))
    p_mat.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    p_mat.CreateRestitutionAttr().Set(float(restitution))

    # Apply PhysxSchema.PhysxMaterialAPI for combine mode
    if not mat_prim.HasAPI(PhysxSchema.PhysxMaterialAPI):
        PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
    
    physx_mat = PhysxSchema.PhysxMaterialAPI(mat_prim)
    physx_mat.CreateFrictionCombineModeAttr().Set("multiply")
    
    # Get root prim
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim.IsValid():
        print(f"[ERROR] Invalid root prim path: {root_prim_path}")
        return 0, []
    
    instanceable_disabled_paths: List[str] = []
    candidate_prim_paths: List[str] = []
    bound_target_paths: List[str] = []
    
    # First pass: make the hierarchy editable and collect candidate collision prims.
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsValid():
            continue
        
        # Disable instanceable if set
        if prim.IsInstanceable():
            try:
                prim.SetInstanceable(False)
                instanceable_disabled_paths.append(prim.GetPath().pathString)
            except Exception as e:
                print(f"[WARN] Failed to disable instanceable on {prim.GetPath()}: {e}")
        
        if prim.IsA(UsdGeom.Mesh):
            candidate_prim_paths.append(prim.GetPath().pathString)
        elif prim.HasAPI(UsdPhysics.CollisionAPI):
            candidate_prim_paths.append(prim.GetPath().pathString)

    disabled_set = set(instanceable_disabled_paths)
    if pre_disabled_instanceable_paths:
        disabled_set.update(pre_disabled_instanceable_paths)
    bound_target_set = set()

    # Second pass: bind on a stable parent prim instead of deep mesh prims.
    for prim_path in candidate_prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue

        bind_target = prim
        nearest_disabled_ancestor = None
        cursor = prim
        while cursor and cursor.IsValid():
            cursor_path = cursor.GetPath().pathString
            if cursor_path in disabled_set:
                nearest_disabled_ancestor = cursor
                break
            if cursor.GetPath() == root_prim.GetPath():
                break
            cursor = cursor.GetParent()

        if nearest_disabled_ancestor is not None:
            parent = nearest_disabled_ancestor.GetParent()
            if parent and parent.IsValid():
                bind_target = parent
            else:
                bind_target = nearest_disabled_ancestor
        elif prim.IsA(UsdGeom.Mesh):
            parent = prim.GetParent()
            if parent and parent.IsValid():
                bind_target = parent

        bind_target_path = bind_target.GetPath().pathString
        if bind_target_path in bound_target_set:
            continue

        try:
            api = UsdShade.MaterialBindingAPI(bind_target)
            if not api:
                api = UsdShade.MaterialBindingAPI.Apply(bind_target)

            api.Bind(
                UsdShade.Material(mat_prim),
                materialPurpose="physics"
            )
            bound_target_set.add(bind_target_path)
            bound_target_paths.append(bind_target_path)
        except Exception as e:
            print(f"[WARN] Failed to bind material to {bind_target.GetPath()} (from {prim_path}): {e}")
    
    print(f"[INFO] Physics material '{material_path}' applied:")
    print(f"  - Candidate mesh/collision prims: {len(candidate_prim_paths)}")
    print(f"  - Bound to {len(bound_target_paths)} prim(s)")
    print(f"  - Disabled instanceable on {len(instanceable_disabled_paths)} prim(s)")
    print(f"  - Static friction: {static_friction}")
    print(f"  - Dynamic friction: {dynamic_friction}")
    print(f"  - Restitution: {restitution}")
    
    return len(bound_target_paths), instanceable_disabled_paths

def add_lighting(stage):
    light = UsdLux.DomeLight.Define(stage, "/World/defaultDomeLight")
    light.CreateIntensityAttr().Set(1000.0)
    d_light = UsdLux.DistantLight.Define(stage, "/World/distantLight")
    d_light.CreateIntensityAttr().Set(2000.0)
    UsdGeom.Xformable(d_light).AddRotateXYZOp().Set(Gf.Vec3f(45, 45, 0))
# =======================
# Processing One Object
# =======================

async def process_one_object(obj_usd: Path, obj_id: str, output_dir: Path, object_wrapper_path: str):
    
    ctx = omni.usd.get_context()
    if ctx.get_stage():
        print(f"[INFO] Closing previous stage...")
        await ctx.close_stage_async()
        await omni.kit.app.get_app().next_update_async()
        
        import gc
        gc.collect()
        
        await step_simulation(10)
        
    print(f"[INFO] Creating new stage")
    await ctx.new_stage_async()
    stage = ctx.get_stage()
    
    world = stage.GetPrimAtPath("/World")
    if not world.IsValid():
        world = UsdGeom.Xform.Define(stage, Sdf.Path("/World")).GetPrim()
        stage.SetDefaultPrim(world)

    if not stage.GetPrimAtPath(ENV_BASE_PATH).IsValid():
        UsdGeom.Scope.Define(stage, ENV_BASE_PATH)
    if not stage.GetPrimAtPath(env_path(0)).IsValid():
        UsdGeom.Xform.Define(stage, env_path(0))
    
    add_lighting(stage)
    await omni.kit.app.get_app().next_update_async()
    # Add object to stage
    object_wrapper_xform = UsdGeom.Xform.Define(stage, object_wrapper_path)
    print(f"[INFO] Created wrapper at {object_wrapper_path}")
    
    object_ref_prim = add_reference_to_stage(str(obj_usd), OBJECT_REF_PATH)
    
    if not object_ref_prim:
        print(f"[ERROR] Failed to add object reference")
        return
    print(f"[INFO] Added reference at {OBJECT_REF_PATH}")
    
    gripper_wrapper_xform = UsdGeom.Xform.Define(stage, GRIPPER_WRAPPER_PATH)
    print(f"[INFO]   Created wrapper at {GRIPPER_WRAPPER_PATH}")
    
    gripper_ref_prim = add_reference_to_stage(str(GRIPPER_USD), GRIPPER_REF_PATH)
    if not gripper_ref_prim:
        print(f"[ERROR] Failed to add gripper reference")
        return
    print(f"[INFO]   Added reference at {GRIPPER_REF_PATH}")
    
    await omni.kit.app.get_app().next_update_async()
    
    #add d6 joint
    setup_gripper_d6_control(stage, GRIPPER_WRAPPER_PATH, env_path(0))
    
    await omni.kit.app.get_app().next_update_async()
    set_gripper_world_pose(stage, GRIPPER_WRAPPER_PATH, position = [0, 0, 3], quaternion = [1, 0, 0, 0])
    set_kinematic_target_pose(stage, f"{env_path(0)}/GripperTarget", position = [0, 0, 3], quaternion = [1, 0, 0, 0])
    
    await omni.kit.app.get_app().next_update_async()
    
    # =====================
    # DISABLE INSTANCEABLE FOR GRASP GENERATION
    # =====================
    print(f"\n[INFO] Disabling instanceable for grasp generation...")
    changed_instanceable_paths = disable_instanceable_for_grasp_generation(stage, OBJECT_REF_PATH)
    await omni.kit.app.get_app().next_update_async()
    
    
    print(f"\n[INFO] Applying high-friction physics material to object...")
    num_prims_with_material, object_material_instanceable_paths = create_and_bind_high_friction_material(
        stage,
        OBJECT_REF_PATH,
        static_friction=2.0,
        dynamic_friction=2.0,
        restitution=0.0,
        pre_disabled_instanceable_paths=changed_instanceable_paths,
    )
    
    if num_prims_with_material == 0:
        print(f"[WARN] No prims found to apply physics material")
    
    await omni.kit.app.get_app().next_update_async()
    
    # Apply PhysX overrides to the existing rigid bodies inside the referenced asset.
    # Do NOT put a rigid body on the wrapper, otherwise PhysX reports invalid rigid-body hierarchy.
    print(f"[INFO] Applying PhysX overrides to object link rigid bodies...")
    apply_object_physx_overrides(stage, OBJECT_REF_PATH, disable_gravity=True)
    await omni.kit.app.get_app().next_update_async()
    
    # Setup physics scene
    print(f"[INFO] Setting up physics scene...")
    physics_scene_path = setup_physics_scene(stage)
    await omni.kit.app.get_app().next_update_async()
    ps_prim = stage.GetPrimAtPath(physics_scene_path)
    if not ps_prim.IsValid():
        raise RuntimeError(f"Failed to create valid physics scene at {physics_scene_path}")
    print(f"[INFO] Physics scene validated at {physics_scene_path}")
    
    # Get bottom center and add ground plane
    bottom_center = get_bbox_bottom_center(stage, object_wrapper_path)
    if bottom_center:
        bottom_center_list = [float(bottom_center[0]), float(bottom_center[1]), float(bottom_center[2])]
        print(f"[INFO] Object bottom center: {bottom_center_list}")
    else:
        bottom_center_list = None
        print(f"[WARN] Could not compute bottom center")
        
    GroundPlane(prim_path="/World/GroundPlane", z_position=bottom_center[2] - 0.001)
    await omni.kit.app.get_app().next_update_async()
    
    # =====================
    # SET INITIAL OBJECT POSE
    # =====================
    
    dataset_dir = obj_usd.parent
    print(f"[INFO] Setting initial joint positions before grasp generation...")
    timeline.play()
    await step_simulation(5)

    urdf_path = os.path.join(str(dataset_dir), 'mobility.urdf')
    joints_info = find_revolute_prismatic_joints_in_urdf(urdf_path)
    joints_info = filter_joints(stage, f"{object_wrapper_path}/ref", joints_info)

    for joint_name, joint_type, child_link_name, lower, upper, joint_axis in joints_info:
        joint_prim = find_usd_joint_prim(stage, OBJECT_REF_PATH, joint_name)
        if joint_prim is None:
            continue
        
        if joint_type == "revolute":
            initial_pos = INITIAL_JOINT_ANGLE
        else:
            initial_pos = INITIAL_JOINT_POSITION_M
        
        drive_kind = "angular" if joint_type == "revolute" else "linear"
        set_usd_joint_drive_target(stage, joint_prim.GetPath().pathString, initial_pos, drive_kind)
        set_joint_position_direct(stage, joint_prim, initial_pos)

    await step_simulation(60)  # Let joints settle

    print(f"[INFO] Joints set to initial positions, ready for grasp generation")
    
    # =====================
    # GENERATE GRASPS
    # =====================
    print(f"\n[INFO] Generating grasps...")
    
    all_grasps = generate_grasps_for_all_joints(
        stage,
        joints_info,
        OBJECT_WRAPPER_PATH,
        OBJECT_REF_PATH,
        GRIPPER_WRAPPER_PATH,
        num_samples_per_joint=100
    )
    
    if not all_grasps:
        print(f"[ERROR] No grasps generated")
        return

    # =====================
    # Enable gravity for physics validation
    # =====================
    print("[INFO] Enabling gravity on articulated object for physics validation...")
    set_object_gravity_enabled(stage, OBJECT_REF_PATH, enabled=True)
    await omni.kit.app.get_app().next_update_async()
    
    # =====================
    # TRAJECTORY GENERATION 
    # =====================
    print("[INFO] Generating trajectories for all grasps...")
    all_trajectories = generate_trajectories_for_all_grasps(
        stage,
        all_grasps,
        OBJECT_REF_PATH,
        num_trajectory_steps=200,
        opening_fraction=OPENING_FRACTION
    )
    
    # =====================
    # RESTORE INSTANCEABLE
    # =====================
    print(f"\n[INFO] Restoring instanceable state...")
    restore_instanceable(stage, object_material_instanceable_paths + changed_instanceable_paths)
    await omni.kit.app.get_app().next_update_async()
    
    #Restore initial joint positions before trajectory validation
    for joint_name, joint_type, child_link_name, lower, upper, joint_axis in joints_info:
        joint_prim = find_usd_joint_prim(stage, OBJECT_REF_PATH, joint_name)
        if joint_prim is None:
            continue
        
        initial_pos = 0.0
        drive_kind = "angular" if joint_type == "revolute" else "linear"
        set_usd_joint_drive_target(stage, joint_prim.GetPath().pathString, initial_pos, drive_kind)
        set_joint_position_direct(stage, joint_prim, initial_pos)
    
    await step_simulation(60) 
    timeline.stop()
    
    # =====================
    # VISUALIZE GRASP POINTS + DIRECTION LINES
    # =====================
    # print(f"\n[INFO] Visualizing grasp poses with direction lines...")
    # debug_root = "/World/Debug/GraspPoints"
    # if not stage.GetPrimAtPath("/World/Debug").IsValid():
    #     UsdGeom.Scope.Define(stage, "/World/Debug")
    # UsdGeom.Scope.Define(stage, debug_root)

    # colors = {
    #     "revolute": Gf.Vec3f(0.0, 1.0, 0.0),
    #     "prismatic": Gf.Vec3f(0.0, 0.5, 1.0),
    # }
    # line_length = 0.08

    # grasp_count = 0
    # for joint_name, joint_data in all_grasps.items():
    #     jtype = joint_data["joint_type"]
    #     color = colors.get(jtype, Gf.Vec3f(1.0, 1.0, 0.0))

    #     for gi, (pos, quat) in enumerate(joint_data["grasp_poses"]):
    #         pos_np = np.asarray(pos, dtype=np.float64).reshape(3)
    #         quat_np = np.asarray(quat, dtype=np.float64).reshape(4)

    #         # Compute approach direction (gripper local Z in world)
    #         q_xyzw = [quat_np[1], quat_np[2], quat_np[3], quat_np[0]]
    #         rot = R.from_quat(q_xyzw)
    #         approach_dir = rot.apply([0, 0, 1])
    #         end_pt = pos_np + approach_dir * line_length

    #         # Grasp point sphere
    #         sphere_path = f"{debug_root}/{joint_name}_g{gi}_pt"
    #         sphere = UsdGeom.Sphere.Define(stage, sphere_path)
    #         sphere.CreateRadiusAttr(0.004)
    #         sphere.CreateDisplayColorAttr([color])
    #         xf = UsdGeom.Xformable(sphere)
    #         xf.AddTranslateOp().Set(Gf.Vec3d(float(pos_np[0]), float(pos_np[1]), float(pos_np[2])))

    #         # Direction line (BasisCurves)
    #         line_path = f"{debug_root}/{joint_name}_g{gi}_dir"
    #         curve = UsdGeom.BasisCurves.Define(stage, line_path)
    #         curve.CreateTypeAttr().Set("linear")
    #         curve.CreatePointsAttr().Set([
    #             Gf.Vec3f(float(pos_np[0]), float(pos_np[1]), float(pos_np[2])),
    #             Gf.Vec3f(float(end_pt[0]), float(end_pt[1]), float(end_pt[2])),
    #         ])
    #         curve.CreateCurveVertexCountsAttr().Set([2])
    #         curve.CreateWidthsAttr().Set([0.002, 0.002])
    #         curve.CreateDisplayColorAttr([color])

    #         grasp_count += 1

    # print(f"[INFO] Visualized {grasp_count} grasp poses (sphere + direction line)")
    # print(f"[INFO]   Green = revolute, Blue = prismatic")
    # print(f"[INFO]   Line = gripper approach direction (local Z)")
    # await omni.kit.app.get_app().next_update_async()
    
    # =====================
    # GRID CLONER
    # =====================
    print(f"[INFO] Cloning template env_0 into grid (NUM_COPIES={NUM_COPIES}, spacing={CLONE_SPACING})...")
    cloner = GridCloner(spacing=CLONE_SPACING)
    cloner.define_base_env(ENV_BASE_PATH)
    env_paths = cloner.generate_paths(ENV_ROOT_PREFIX, NUM_COPIES)
    print(f"[DEBUG] Cloner env_paths[0..min]: {env_paths[:min(5, len(env_paths))]}")
    cloner.clone(source_prim_path=env_path(0), prim_paths=env_paths)
    await omni.kit.app.get_app().next_update_async()
    
    # Debug: confirm env transforms differ (so clones are spatially separated)
    try:
        p0, _ = transform_utils.get_prim_world_pose(stage.GetPrimAtPath(env_path(0)))
        p1, _ = transform_utils.get_prim_world_pose(stage.GetPrimAtPath(env_path(1))) if NUM_COPIES > 1 else (None, None)
        print(f"[DEBUG] env_0 world pos: {p0}")
        if p1 is not None:
            print(f"[DEBUG] env_1 world pos: {p1} (delta ~ {p1 - p0})")
    except Exception as e:
        print(f"[WARN] Could not print env transforms: {e}")
    
    print(f"[INFO] Cloning complete.")
    
    #Fix object base to world in all envs
    print(f"\n[INFO] Adding fixed joints to all {NUM_COPIES} environments...")
    for i in range(NUM_COPIES):
        env_obj_ref = obj_ref(i)
        fixed_joint_path = f"{env_path(i)}/ObjectFixedToWorld"
        env_obj_root = resolve_asset_root_under_ref(stage, env_obj_ref)
        
        print(f"[INFO] Fixing object in env_{i}...")
        fix_object_base_to_world(
            stage, 
            env_obj_root, 
            base_link_name="base", 
            joint_path=fixed_joint_path
        )
    
    await omni.kit.app.get_app().next_update_async()
    print(f"[INFO] All environments fixed to world")

    # =====================
    # PHYSICS VALIDATION
    # =====================
    valid_trajectories = []
    if all_trajectories:
        print("[INFO] Running physics validation on trajectories (step-back + approach)...")
        valid_trajectories = await physics_validation_loop(
            stage,
            all_trajectories,
        )
        print(
            f"[INFO] Physics validation kept {len(valid_trajectories)}/"
            f"{len(all_trajectories)} trajectories"
        )
        
        # Save validated trajectories to JSON
        json_path = None
        if valid_trajectories and bottom_center_list:
            try:
                json_path = save_trajectories_to_json(
                    valid_trajectories,
                    obj_usd,
                    bottom_center_list
                )
                print(f"[SUCCESS] Trajectory data saved to {json_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save trajectory JSON: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("[WARN] No trajectories generated; skipping physics validation")
    
    # =====================
    # SUMMARY TABLE
    # =====================
    n_joints_found   = len(joints_info)
    n_joints_grasped = len(all_grasps)
    n_trajectories   = len(all_trajectories)
    n_valid          = len(valid_trajectories)
    pct_valid        = (n_valid / n_trajectories * 100) if n_trajectories > 0 else 0.0

    print(f"\n{'='*80}")
    print(f"  SUMMARY: {obj_id}")
    print(f"{'='*80}")
    print(f"  {'Joints found:':<30} {n_joints_found}")
    print(f"  {'Joints with grasps:':<30} {n_joints_grasped}")
    print(f"  {'Total trajectories:':<30} {n_trajectories}")
    print(f"  {'Valid trajectories:':<30} {n_valid}  ({pct_valid:.1f}%)")
    print(f"  {'JSON saved:':<30} {json_path if json_path else '(none)'}")
    print(f"{'='*80}\n")

    if PROCESSING_MODE == "dataset":
        if valid_trajectories:
            mark_object_completed(LOG_FILE, obj_id)
        else:
            print(f"[WARN] No valid trajectories for {obj_id}, not marking as completed")

# =======================
# Main Pipeline Functions
# =======================

async def run_pipeline():
    """Main pipeline - processes either single object or entire dataset"""
    
    print(f"\n{'#'*80}")
    print(f"# Grasp Generation and Validation Pipeline")
    print(f"# Mode: {PROCESSING_MODE.upper()}")
    print(f"{'#'*80}\n")
    
    if PROCESSING_MODE == "single":
        # =====================
        # SINGLE OBJECT MODE
        # =====================
        print(f"[INFO] Processing single object: {SINGLE_OBJECT_USD}")
        
        if not SINGLE_OBJECT_USD.exists():
            print(f"[ERROR] Object file not found: {SINGLE_OBJECT_USD}")
            return
        
        obj_id = SINGLE_OBJECT_USD.parent.name
        output_dir = SINGLE_OBJECT_USD.parent
        
        try:
            await process_one_object(SINGLE_OBJECT_USD, obj_id, output_dir, OBJECT_WRAPPER_PATH)
            print(f"\n[SUCCESS] Single object processing complete!")
        except Exception as e:
            print(f"\n[ERROR] Failed to process object: {e}")
            import traceback
            traceback.print_exc()
    
    elif PROCESSING_MODE == "dataset":
        # =====================
        # DATASET MODE
        # =====================
        print(f"[INFO] Dataset path: {INPUT_DATASET_PATH}")
        print(f"[INFO] Log file: {LOG_FILE}")
        
        if not INPUT_DATASET_PATH.exists():
            print(f"[ERROR] Dataset path not found: {INPUT_DATASET_PATH}")
            return
        
        # Find all objects
        all_objects = find_all_objects(INPUT_DATASET_PATH)
        print(f"[INFO] Found {len(all_objects)} objects in dataset")
        
        if not all_objects:
            print(f"[ERROR] No objects found in {INPUT_DATASET_PATH}")
            return
        
        # Get completed objects
        completed = get_completed_objects(LOG_FILE)
        print(f"[INFO] Already completed: {len(completed)} objects")
        
        # Filter out completed objects
        remaining = [(usd, oid) for usd, oid in all_objects if oid not in completed]
        print(f"[INFO] Remaining to process: {len(remaining)} objects")
        
        if not remaining:
            print(f"[INFO] All objects already processed!")
            return
        
        # Process each object
        success_count = 0
        fail_count = 0
        
        for idx, (obj_usd, obj_id) in enumerate(remaining, start=1):
            print(f"\n{'='*80}")
            print(f"Processing {idx}/{len(remaining)}: {obj_id}")
            print(f"{'='*80}\n")
            
            try:
                output_dir = obj_usd.parent
                await process_one_object(obj_usd, obj_id, output_dir, OBJECT_WRAPPER_PATH)
                success_count += 1
                
            except Exception as e:
                print(f"\n[ERROR] Failed to process {obj_id}: {e}")
                import traceback
                traceback.print_exc()
                fail_count += 1
                print(f"[INFO] Continuing to next object...")
                continue
        
        print(f"\n{'#'*80}")
        print(f"# Dataset Processing Complete")
        print(f"# Successfully processed: {success_count}/{len(remaining)}")
        print(f"# Failed: {fail_count}/{len(remaining)}")
        print(f"{'#'*80}\n")
    
    else:
        print(f"[ERROR] Invalid PROCESSING_MODE: '{PROCESSING_MODE}'")
        print(f"[ERROR] Must be either 'single' or 'dataset'")
        return
    
    print(f"\n{'#'*80}")
    print(f"# Pipeline Complete")
    print(f"{'#'*80}\n")

def main():
    """Entry point"""
    print("[INFO] Starting pipeline...")
    
    try:
        loop = asyncio.get_event_loop()
        task = asyncio.ensure_future(run_pipeline())
        
        while not task.done():
            simulation_app.update()
        
        if task.exception():
            raise task.exception()

        app = omni.kit.app.get_app()

        WAIT_SECONDS = 60
        start = time.time()

        while time.time() - start < WAIT_SECONDS:
            app.update()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] Closing simulation app...")
        simulation_app.close()
        print("[INFO] Done")


# =======================
# Physics validation loop
# =======================
def setup_gripper_d6_control(stage, gripper_wrapper_path: str, env_path: str):
    """
    Create kinematic target + D6 joint for physics-based gripper control.
    Uses the working pattern from the test file.
    
    Args:
        stage: USD stage
        gripper_wrapper_path: Path to gripper wrapper
        env_path: Path to environment root
    
    Returns:
        (target_path, d6_joint_path): Paths to created prims
    """
    gripper_ref_path = f"{gripper_wrapper_path}/ref"
    gripper_base_link = f"{gripper_ref_path}/panda_hand"
    
    # 1. Ensure gripper base has RigidBodyAPI
    gripper_base_prim = stage.GetPrimAtPath(gripper_base_link)
    if not gripper_base_prim.IsValid():
        raise RuntimeError(f"Gripper base link not found: {gripper_base_link}")
    
    if not gripper_base_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(gripper_base_prim)
    
    # 2. Create kinematic target
    target_path = f"{env_path}/GripperTarget"
    target_xform = UsdGeom.Xform.Define(stage, target_path)
    target_prim = stage.GetPrimAtPath(target_path)
    
    UsdPhysics.RigidBodyAPI.Apply(target_prim)
    rb = UsdPhysics.RigidBodyAPI(target_prim)
    rb.CreateKinematicEnabledAttr().Set(True)
    
    # Get gripper's current pose
    gripper_prim = stage.GetPrimAtPath(gripper_wrapper_path)
    if gripper_prim.IsValid():
        gripper_xf = UsdGeom.Xformable(gripper_prim)
        world_xf = gripper_xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pos = world_xf.ExtractTranslation()
        rot = world_xf.ExtractRotation().GetQuat()
        
        target_xformable = UsdGeom.Xformable(target_prim)
        target_xformable.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
        target_xformable.AddOrientOp().Set(Gf.Quatf(
            rot.GetReal(),
            Gf.Vec3f(rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2])
        ))
    else:
        # Default position
        target_xformable = UsdGeom.Xformable(target_prim)
        target_xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))
        target_xformable.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    
    print("✓ Created kinematic target")
    
    # 3. Create D6-style joint with drives (soft arm)
    joint_path = f"{env_path}/D6Joint"
    joint = UsdPhysics.Joint.Define(stage, joint_path)
    joint_prim = joint.GetPrim()
    
    joint.CreateBody0Rel().SetTargets([Sdf.Path(target_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(gripper_base_link)])
    
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    
    joint.CreateExcludeFromArticulationAttr().Set(True)
    
    # 4. Configure all 6 DOF drives
    axes = [
        UsdPhysics.Tokens.transX,
        UsdPhysics.Tokens.transY,
        UsdPhysics.Tokens.transZ,
        UsdPhysics.Tokens.rotX,
        UsdPhysics.Tokens.rotY,
        UsdPhysics.Tokens.rotZ,
    ]
    
    # Use parameters from working code
    lin_stiffness = 5e1
    lin_damping = 5
    ang_stiffness = 2e1
    ang_damping = 2e1
    max_force = 1e2
    
    for axis in axes:
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, axis)
        drive.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
        
        if axis in (UsdPhysics.Tokens.transX, UsdPhysics.Tokens.transY, UsdPhysics.Tokens.transZ):
            drive.CreateStiffnessAttr().Set(lin_stiffness)
            drive.CreateDampingAttr().Set(lin_damping)
        else:
            drive.CreateStiffnessAttr().Set(ang_stiffness)
            drive.CreateDampingAttr().Set(ang_damping)
        
        drive.CreateMaxForceAttr().Set(max_force)
    
    print("✓ Created D6 soft-arm joint")
    
    #setup another fixedjoint for later swicthes
    fixed_joint_path = f"{env_path}/GripperFixedJoint"
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, fixed_joint_path)
    fixed_joint_prim = fixed_joint.GetPrim()

    fixed_joint.CreateBody0Rel().SetTargets([])
    fixed_joint.CreateBody1Rel().SetTargets([])
    fixed_joint.CreateBreakForceAttr().Set(1000.0)     # example: 1000 N
    fixed_joint.CreateBreakTorqueAttr().Set(1000.0)  
    
    return target_path, joint_path, fixed_joint_path

def switch_to_fixed_joint(stage, env_idx: int):
    """Disconnect D6 from panda_hand, connect FixedJoint to panda_hand."""
    d6_path = f"{env_path(env_idx)}/D6Joint"
    fixed_path = f"{env_path(env_idx)}/GripperFixedJoint"
    hand_path = grip_base(env_idx)

    # Disconnect D6: clear body1
    d6_prim = stage.GetPrimAtPath(d6_path)
    if d6_prim.IsValid():
        UsdPhysics.Joint(d6_prim).GetBody1Rel().SetTargets([])

    # Connect FixedJoint: set body1 to panda_hand
    fixed_prim = stage.GetPrimAtPath(fixed_path)
    if fixed_prim.IsValid():
        UsdPhysics.Joint(fixed_prim).GetBody1Rel().SetTargets([Sdf.Path(hand_path)])

    print(f"[ENV {env_idx}] Switched to FixedJoint (rigid mode)")


def switch_to_d6_joint(stage, env_idx: int):
    """Disconnect FixedJoint from panda_hand, reconnect D6 to panda_hand."""
    d6_path = f"{env_path(env_idx)}/D6Joint"
    fixed_path = f"{env_path(env_idx)}/GripperFixedJoint"
    hand_path = grip_base(env_idx)

    # Disconnect FixedJoint: clear body1
    fixed_prim = stage.GetPrimAtPath(fixed_path)
    if fixed_prim.IsValid():
        UsdPhysics.Joint(fixed_prim).GetBody1Rel().SetTargets([])

    # Reconnect D6: set body1 to panda_hand
    d6_prim = stage.GetPrimAtPath(d6_path)
    if d6_prim.IsValid():
        UsdPhysics.Joint(d6_prim).GetBody1Rel().SetTargets([Sdf.Path(hand_path)])

    print(f"[ENV {env_idx}] Switched to D6Joint (soft mode)")

def set_kinematic_target_pose(stage, target_path: str, position: np.ndarray, quaternion: np.ndarray):
    """Move the kinematic target prim to a new world pose."""
    prim = stage.GetPrimAtPath(target_path)
    if not prim.IsValid():
        return
    xformable = UsdGeom.Xformable(prim)
    ops = list(xformable.GetOrderedXformOps())
    translate_op = orient_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient_op = op
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    if orient_op is None:
        orient_op = xformable.AddOrientOp()
    
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    w, x, y, z = float(quaternion[0]), float(quaternion[1]), float(quaternion[2]), float(quaternion[3])
    translate_op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
    orient_op.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))

def get_gripper_current_pose(stage, gripper_wrapper_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Get current gripper pose from USD.
    
    Returns:
        (position, quaternion) where quaternion is [w, x, y, z]
        Returns (None, None) if gripper prim is invalid
    """
    prim = stage.GetPrimAtPath(gripper_wrapper_path)
    if not prim.IsValid():
        return None, None
    
    xformable = UsdGeom.Xformable(prim)
    world_xf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    
    # Extract position
    translation = world_xf.ExtractTranslation()
    position = np.array([translation[0], translation[1], translation[2]], dtype=np.float64)
    
    # Extract rotation as quaternion
    rotation = world_xf.ExtractRotation()
    quat_gf = rotation.GetQuat()
    quaternion = np.array([
        quat_gf.GetReal(),           # w
        quat_gf.GetImaginary()[0],   # x
        quat_gf.GetImaginary()[1],   # y
        quat_gf.GetImaginary()[2]    # z
    ], dtype=np.float64)
    
    return position, quaternion

# Helper: Set USD joint drive target (for gripper closing etc.)
def set_usd_joint_drive_target(stage, joint_prim_path: str, target_value: float, drive_kind: str = "linear"):
    """Set a USD Physics drive target on a joint prim.

    Used for closing the gripper during physics validation.

    Args:
        stage: USD stage
        joint_prim_path: Path to a USD joint prim (e.g., .../panda_finger_joint1)
        target_value: target position (meters for prismatic, radians for revolute)
        drive_kind: "linear" for prismatic joints, "angular" for revolute joints
    """
    prim = stage.GetPrimAtPath(joint_prim_path)
    if not prim.IsValid():
        print(f"[WARN] set_usd_joint_drive_target: invalid joint prim {joint_prim_path}")
        return

    try:
        drive = UsdPhysics.DriveAPI.Apply(prim, drive_kind)

        tp = drive.GetTargetPositionAttr()
        if not tp or not tp.IsValid():
            tp = drive.CreateTargetPositionAttr()
        tp.Set(float(target_value))

        # If the asset already has gains, these may already be set.
        # We set defaults only when they are missing.
        st = drive.GetStiffnessAttr()
        if not st or not st.IsValid():
            st = drive.CreateStiffnessAttr()
        if st.Get() is None:
            st.Set(400.0)

        dm = drive.GetDampingAttr()
        if not dm or not dm.IsValid():
            dm = drive.CreateDampingAttr()
        if dm.Get() is None:
            dm.Set(80.0)

        mf = drive.GetMaxForceAttr()
        if not mf or not mf.IsValid():
            mf = drive.CreateMaxForceAttr()
        if mf.Get() is None:
            mf.Set(20.0)

    except Exception as e:
        print(f"[WARN] Failed to set drive target on {joint_prim_path} ({drive_kind}): {e}")

def set_joint_position_direct(stage, joint_prim: Usd.Prim, position: float):
    """
    Directly set joint position using the Joint State API.
    This corresponds to GUI: Joint State -> Angular/Linear -> Position
    
    Args:
        stage: USD stage
        joint_prim: Joint prim (RevoluteJoint or PrismaticJoint)
        position: Target position (RADIANS for revolute, meters for prismatic)
        also_set_drive: If True, also set drive target to match (keeps drive consistent)
    """
    if not joint_prim.IsValid():
        print(f"[WARN] set_joint_position_direct: invalid joint prim")
        return
    
    joint_type = None
    drive_kind = None
    state_api_kind = None
    
    # Determine joint type, drive kind, and state API kind
    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        joint_type = "revolute"
        drive_kind = "angular"
        state_api_kind = "angular"
    elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
        joint_type = "prismatic"
        drive_kind = "linear"
        state_api_kind = "linear"
    else:
        print(f"[WARN] set_joint_position_direct: unknown joint type {joint_prim.GetTypeName()}")
        return
    
    # Set the joint state position using PhysxSchema.JointStateAPI
    try:
        # Apply JointStateAPI if not already present
        if not joint_prim.HasAPI(PhysxSchema.JointStateAPI):
            PhysxSchema.JointStateAPI.Apply(joint_prim, state_api_kind)
        
        joint_state = PhysxSchema.JointStateAPI(joint_prim, state_api_kind)
        
        # Set the position attribute
        pos_attr = joint_state.GetPositionAttr()
        if not pos_attr or not pos_attr.IsValid():
            pos_attr = joint_state.CreatePositionAttr()
        pos_attr.Set(float(position))
        
        print(f"[DEBUG] Set joint state position: {joint_prim.GetPath()} = {position:.4f} "
              f"({'deg' if joint_type == 'revolute' else 'm' if state_api_kind == 'linear' else 'rad'})")
    
    except Exception as e:
        print(f"[ERROR] Failed to set joint state position on {joint_prim.GetPath()}: {e}")
        import traceback
        traceback.print_exc()

def set_gripper_world_pose(stage, gripper_wrapper_path: str, position: np.ndarray, quaternion: np.ndarray):
    """
    Set the gripper wrapper prim's world pose using a translate + orient op.

    Args:
        stage: USD stage
        gripper_wrapper_path: Path to the gripper wrapper Xform (e.g., GRIPPER_WRAPPER_PATH)
        position: [3] world position in meters
        quaternion: [4] quaternion [w, x, y, z]
    """
    prim = stage.GetPrimAtPath(gripper_wrapper_path)
    if not prim.IsValid():
        print(f"[WARN] set_gripper_world_pose: invalid prim {gripper_wrapper_path}")
        return

    xformable = UsdGeom.Xformable(prim)
    ops = list(xformable.GetOrderedXformOps())

    translate_op = None
    orient_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient_op = op

    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    if orient_op is None:
        orient_op = xformable.AddOrientOp()

    pos = np.asarray(position, dtype=np.float64).reshape(3)
    wxyz = np.asarray(quaternion, dtype=np.float64).reshape(4)
    w, x, y, z = float(wxyz[0]), float(wxyz[1]), float(wxyz[2]), float(wxyz[3])

    translate_op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
    orient_op.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))

async def validate_batch_parallel(
    stage,
    batch: List[Dict],
    batch_index: int,
) -> List[Tuple[int, bool]]:
    """Validate a batch of trajectories in parallel across environments."""
    print(f"\n[BATCH {batch_index}] Validating {len(batch)} trajectories in parallel...")
    
    if timeline.is_playing() or timeline.is_stopped() == False:
        timeline.stop()
        await step_simulation(10)
        
    # Step 0: Reset gripper's positions to avoid physics instability
    for env_idx in range(len(batch)):
        gripper_wrapper_path = grip_wrap(env_idx)
        # gripper_ref_path = grip_ref(env_idx)
        
        # hand_path = f"{gripper_ref_path}/panda_hand"
        # hand_prim = stage.GetPrimAtPath(hand_path)
        # if hand_prim:
        #     xformable = UsdGeom.Xformable(hand_prim)
        #     xformable.ClearXformOpOrder() 
                # Teleport BOTH to same pose while paused
        d6_path = f"{env_path(env_idx)}/D6Joint"
        
        # disconnect D6: set body1 to panda_hand
        d6_prim = stage.GetPrimAtPath(d6_path)
        if d6_prim.IsValid():
            UsdPhysics.Joint(d6_prim).GetBody1Rel().SetTargets([])
            
        hand_prim = stage.GetPrimAtPath(grip_base(env_idx))
        # Set linear and angular velocities to zero
        hand_prim.GetAttribute("physics:velocity").Set((0, 0, 0))
        hand_prim.GetAttribute("physics:angularVelocity").Set((0, 0, 0))
        # hand_prim.GetAttribute("xformOp:translate").Set((0.0, 0.0, 0.0))
        # hand_prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        set_gripper_world_pose(stage, gripper_wrapper_path, position=[0, 0, 3], quaternion=[1, 0, 0, 0])
        set_kinematic_target_pose(stage, f"{env_path(env_idx)}/GripperTarget", position=[0, 0, 3], quaternion=[1, 0, 0, 0])
    
    await step_simulation(2)

    # Step 1: Reset all environments to initial state using constants
    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        object_ref_path = obj_ref(env_idx)
        joint_name = traj.get('joint_name', '')
        
        # Reset joint to initial position (not lower limit)
        joint_prim = find_usd_joint_prim(stage, object_ref_path, joint_name)
        if joint_prim:
            joint_params = get_joint_world_parameters(stage, joint_prim)
            if joint_params:
                # **USE NEW CONSTANTS FOR INITIAL POSITION**
                if joint_params["joint_type"] == "revolute":
                    initial_pos = INITIAL_JOINT_ANGLE
                else:  # prismatic
                    initial_pos = INITIAL_JOINT_POSITION_M
                
                drive_kind = "angular" if joint_params["joint_type"] == "revolute" else "linear"
                set_usd_joint_drive_target(stage, joint_prim.GetPath().pathString, initial_pos, drive_kind)
                set_joint_position_direct(stage, joint_prim, initial_pos)
    
    await step_simulation(2) 
    
    await ensure_timeline_playing()
    await step_simulation(60)  # Let resets settle
    timeline.pause()
    await step_simulation(2)
    
    # Step 2: Record initial states for all envs
    initial_states = []
    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        object_wrapper_path = obj_wrap(env_idx)
        object_ref_path = obj_ref(env_idx)
        joint_name = traj.get('joint_name', '')
        
        # Record initial joint position
        joint_prim = find_usd_joint_prim(stage, object_ref_path, joint_name)
        initial_joint_pos = get_joint_current_position(stage, joint_prim) if joint_prim else 0.0
        
        initial_states.append({
            'initial_joint_pos': initial_joint_pos,
            'joint_prim': joint_prim,
        })
    
    # Step 3: compute approach trajectories for all envs
    all_approach_data = []
    target_paths = []
    
    for env_idx in range(len(batch)):
        target_path = f"{env_path(env_idx)}/GripperTarget"
        target_paths.append(target_path)

    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        grasp_pos = np.asarray(traj["grasp_position"], dtype=np.float64)
        grasp_quat = np.asarray(traj["grasp_quaternion"], dtype=np.float64)
        
        # Compute step-back position (along gripper's local -Z)
        grasp_pos_np = np.asarray(grasp_pos, dtype=np.float64).reshape(3)
        grasp_quat_np = np.asarray(grasp_quat, dtype=np.float64).reshape(4)
        
        stepback_pos, _ = offset_pose_along_local_z(grasp_pos_np, grasp_quat_np, -APPROACH_DISTANCE)
        
        # Linear interpolation from step-back to grasp position
        positions = []
        for t in range(MOVE_STEPS + 1):
            alpha = t / float(MOVE_STEPS) if MOVE_STEPS > 0 else 1.0
            cur = stepback_pos * (1.0 - alpha) + grasp_pos_np * alpha
            positions.append(cur)
        
        approach_positions = np.stack(positions, axis=0)
        approach_quats = np.tile(grasp_quat_np, (approach_positions.shape[0], 1))
        
        all_approach_data.append({
            'stepback_position': stepback_pos,
            'stepback_quaternion': grasp_quat_np,
            'approach_positions': approach_positions,
            'approach_quats': approach_quats,
        })

    env_failed = [False] * len(batch)

    # Step 4: Execute approach phase using D6 joints
    open_target = 0.04

    # Pause physics for safe teleport
    if timeline.is_playing():
        timeline.stop()
        await step_simulation(2)

    for env_idx in range(len(batch)):
        if env_failed[env_idx]:
            continue
        gripper_wrapper_path = grip_wrap(env_idx)
        stepback_pos = all_approach_data[env_idx]['stepback_position']
        stepback_quat = all_approach_data[env_idx]['stepback_quaternion']
        gripper_ref_path = grip_ref(env_idx)
        
        # Set fingers open
        finger_joint1 = f"{gripper_ref_path}/panda_hand/panda_finger_joint1"
        finger_joint2 = f"{gripper_ref_path}/panda_hand/panda_finger_joint2"
        set_usd_joint_drive_target(stage, finger_joint1, open_target, "linear")
        set_usd_joint_drive_target(stage, finger_joint2, open_target, "linear")
        
        set_kinematic_target_pose(stage, target_paths[env_idx], stepback_pos, stepback_quat)
        set_gripper_world_pose(stage, gripper_wrapper_path, stepback_pos, stepback_quat)

    await step_simulation(5)  # Let USD process Xform changes
    
    # NOW reconnect D6 — offset ≈ 0
    for env_idx in range(len(batch)):
        if env_failed[env_idx]:
            continue
        switch_to_d6_joint(stage, env_idx)
    await step_simulation(2)
    
    # Start physics — panda_hand initializes at wrapper Xform
    timeline.play()
    await step_simulation(10)  # Settle with D6 spring at near-zero offset

    # Now step through approach waypoints in lockstep
    max_approach_len = max(d['approach_positions'].shape[0] for d in all_approach_data)

    for wp_idx in range(max_approach_len):
        for env_idx in range(len(batch)):
            if env_failed[env_idx]:
                continue
            approach_positions = all_approach_data[env_idx]['approach_positions']
            approach_quats = all_approach_data[env_idx]['approach_quats']
            if wp_idx >= approach_positions.shape[0]:
                continue
            set_kinematic_target_pose(stage, target_paths[env_idx], 
                                      approach_positions[wp_idx], approach_quats[wp_idx])
        await step_simulation(TRAJECTORY_SIM_STEPS_PER_WAYPOINT)
        
        # Check position error periodically + at final step
        if wp_idx % 30 == 20 or wp_idx == max_approach_len - 1:
            for env_idx in range(len(batch)):
                if env_failed[env_idx]:
                    continue
                approach_positions = all_approach_data[env_idx]['approach_positions']
                if wp_idx >= approach_positions.shape[0]:
                    continue
                
                hand_pos = get_body_env_local_position(stage, grip_base(env_idx), env_path(env_idx))
                if hand_pos is None:
                    env_failed[env_idx] = True
                    print(f"[REJECT] Env {env_idx}: Could not read gripper at wp {wp_idx}")
                    continue
                
                target_pos = approach_positions[wp_idx]
                pos_error = np.linalg.norm(hand_pos - target_pos)
                print(f"[BATCH {batch_index}][Env {env_idx}] Approach wp {wp_idx}: pos error = {pos_error:.4f}m")
                
                if pos_error > APPROACH_POSITION_THRESHOLD:
                    env_failed[env_idx] = True
                    # Park target at hand position to kill D6 spring force
                    hand_quat = all_approach_data[env_idx]['approach_quats'][wp_idx]
                    set_kinematic_target_pose(stage, target_paths[env_idx], hand_pos, hand_quat)
                    print(f"[REJECT] Env {env_idx}: Approach error {pos_error:.4f}m at wp {wp_idx}, frozen")
    
    # >>> EARLY EXIT AFTER APPROACH IF ALL FAILED
    if all(env_failed):
        print(f"[BATCH {batch_index}] All environments failed during approach. Skipping remaining steps.")
        results = []
        for env_idx in range(len(batch)):
            traj = batch[env_idx]
            original_idx = traj.get('original_index', -1)
            results.append((original_idx, False))
        return results

    # Step 5: Close gripper phase (parallel)
    
    #Swicth to fixed joint control after approach phase
    timeline.stop()
    close_target = 0.0
    for env_idx in range(len(batch)):
        if env_failed[env_idx]:
            continue
        gripper_wrapper_path = grip_wrap(env_idx)
        gripper_ref_path = grip_ref(env_idx)
        
        # Get the final approach position = grasp position
        grasp_pos = np.asarray(batch[env_idx]["grasp_position"], dtype=np.float64)
        grasp_quat = np.asarray(batch[env_idx]["grasp_quaternion"], dtype=np.float64)
        
        switch_to_fixed_joint(stage, env_idx)
        set_gripper_world_pose(stage, gripper_wrapper_path, grasp_pos, grasp_quat)
        set_kinematic_target_pose(stage, f"{env_path(env_idx)}/GripperTarget", grasp_pos, grasp_quat)
        
        # Set fingers to close target
        finger_joint1 = f"{gripper_ref_path}/panda_hand/panda_finger_joint1"
        finger_joint2 = f"{gripper_ref_path}/panda_hand/panda_finger_joint2"
        set_usd_joint_drive_target(stage, finger_joint1, close_target, "linear")
        set_usd_joint_drive_target(stage, finger_joint2, close_target, "linear")
        
    await step_simulation(5)  # Let PhysX pick up the joint change
    await ensure_timeline_playing()
    
    for _ in range(CLOSE_STEPS):
        for env_idx in range(len(batch)):
            # >>> SKIP FAILED ENVS
            if env_failed[env_idx]:
                continue
            
            gripper_ref_path = grip_ref(env_idx)
            finger_joint1 = f"{gripper_ref_path}/panda_hand/panda_finger_joint1"
            finger_joint2 = f"{gripper_ref_path}/panda_hand/panda_finger_joint2"
            
            set_usd_joint_drive_target(stage, finger_joint1, close_target, drive_kind="linear")
            set_usd_joint_drive_target(stage, finger_joint2, close_target, drive_kind="linear")
        
        await step_simulation(1)
    
    for env_idx in range(len(batch)):
        if env_failed[env_idx]:
            continue

        traj = batch[env_idx]
        if overshoot_reject(
            stage,
            initial_states[env_idx]['joint_prim'],
            initial_states[env_idx]['initial_joint_pos'],
            should_displacement=0.0,
            target_displacement=traj.get("target_displacement", 0.0),
            joint_type=traj.get("joint_type", ""),
        ):
            env_failed[env_idx] = True

    # >>> OPTIONAL: another early exit here if you want
    if all(env_failed):
        print(f"[BATCH {batch_index}] All environments failed after close phase. Skipping remaining steps.")
        results = []
        for env_idx in range(len(batch)):
            traj = batch[env_idx]
            original_idx = traj.get('original_index', -1)
            results.append((original_idx, False))
        return results
    
    # Step 6: Hold phase (parallel)
    for _ in range(HOLD_STEPS):
        for env_idx in range(len(batch)):
            # >>> SKIP FAILED ENVS
            if env_failed[env_idx]:
                continue
            gripper_ref_path = grip_ref(env_idx)
            finger_joint1 = f"{gripper_ref_path}/panda_hand/panda_finger_joint1"
            finger_joint2 = f"{gripper_ref_path}/panda_hand/panda_finger_joint2"
            
            set_usd_joint_drive_target(stage, finger_joint1, close_target, drive_kind="linear")
            set_usd_joint_drive_target(stage, finger_joint2, close_target, drive_kind="linear")
        
        await step_simulation(1)
    
    # Step 7: Execute trajectory phase (parallel)
    # Pre-cache trajectory arrays to avoid repeated np.asarray per step
    cached_traj_pos = []
    cached_traj_ori = []
    cached_traj_len = []
    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        tp = np.asarray(traj.get("trajectory_positions", []), dtype=np.float64)
        to = np.asarray(traj.get("trajectory_orientations", []), dtype=np.float64)
        cached_traj_pos.append(tp)
        cached_traj_ori.append(to)
        cached_traj_len.append(tp.shape[0])

    max_traj_length = max(cached_traj_len) if cached_traj_len else 0

    if max_traj_length > 0:
        traj_checkpoints = set(np.linspace(0, max_traj_length - 1, TRAJ_OVERSHOOT_CHECKS, dtype=int).tolist())
    else:
        traj_checkpoints = set()

    for traj_step in range(max_traj_length):
        for env_idx in range(len(batch)):
            if env_failed[env_idx]:
                continue

            if traj_step >= cached_traj_len[env_idx]:
                continue

            pos = cached_traj_pos[env_idx][traj_step]
            quat = cached_traj_ori[env_idx][traj_step]

            # Move kinematic target (NOT gripper wrapper teleport)
            gripper_wrapper_path = grip_wrap(env_idx)
            set_gripper_world_pose(stage, gripper_wrapper_path, pos, quat)
            set_kinematic_target_pose(stage, target_paths[env_idx], pos, quat)

            gripper_ref_path = grip_ref(env_idx)
            finger_joint1 = f"{gripper_ref_path}/panda_hand/panda_finger_joint1"
            finger_joint2 = f"{gripper_ref_path}/panda_hand/panda_finger_joint2"
            set_usd_joint_drive_target(stage, finger_joint1, close_target, drive_kind="linear")
            set_usd_joint_drive_target(stage, finger_joint2, close_target, drive_kind="linear")

        await step_simulation(TRAJECTORY_SIM_STEPS_PER_WAYPOINT)

        # Only check overshoot at evenly-spaced checkpoints
        if traj_step in traj_checkpoints:
            for env_idx in range(len(batch)):
                if env_failed[env_idx]:
                    continue

                traj = batch[env_idx]
                target_disp = traj.get("target_displacement", 0.0)
                local_len = cached_traj_len[env_idx]

                if traj_step >= local_len:
                    continue

                if local_len <= 1:
                    alpha_env = 1.0
                else:
                    alpha_env = float(traj_step) / float(local_len - 1)

                should_disp = alpha_env * target_disp

                if overshoot_reject(
                    stage,
                    initial_states[env_idx]['joint_prim'],
                    initial_states[env_idx]['initial_joint_pos'],
                    should_displacement=should_disp,
                    target_displacement=target_disp,
                    joint_type=traj.get("joint_type", ""),
                ):
                    env_failed[env_idx] = True
    
    # Step 8: Measure final joint positions and determine success
    results = []        
    
    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        original_idx = traj.get('original_index', -1)  # We'll add this in the batching step
        if env_failed[env_idx]:
            results.append((original_idx, False))
            continue
        
        joint_prim = initial_states[env_idx]['joint_prim']
        initial_joint_pos = initial_states[env_idx]['initial_joint_pos']
        target_displacement = traj.get("target_displacement", 0.0)
        joint_type = traj.get("joint_type", "")
        
        final_joint_pos = get_joint_current_position(stage, joint_prim)
        if final_joint_pos is None:
            results.append((original_idx, False))
            continue
        
        actual_displacement = abs(final_joint_pos - initial_joint_pos)
        required_displacement = JOINT_SUCCESS_THRESHOLD * abs(target_displacement)
        
        is_valid = actual_displacement >= required_displacement
        
        if joint_type == "revolute":
            print(f"[BATCH {batch_index}] Env {env_idx}: "
                  f"Initial={np.degrees(initial_joint_pos):.1f}°, "
                  f"Final={np.degrees(final_joint_pos):.1f}°, "
                  f"Valid={is_valid}")
        else:
            print(f"[BATCH {batch_index}] Env {env_idx}: "
                  f"Initial={initial_joint_pos:.4f}m, "
                  f"Final={final_joint_pos:.4f}m, "
                  f"Valid={is_valid}")
        
        results.append((original_idx, is_valid))
    
    return results

def get_joint_current_position(stage, joint_prim: Usd.Prim) -> Optional[float]:
    """Read current joint position by checking the computed joint value attribute.
    
    Returns:
        Current position (radians for revolute, meters for prismatic)
    """
    if not joint_prim.IsValid():
        return None
    
    try:
        
        # Check if joint has PhysX joint state API
        if joint_prim.HasAPI(PhysxSchema.PhysxJointAPI):
            physx_joint = PhysxSchema.PhysxJointAPI(joint_prim)
            
            # Try to get the joint position from PhysX schema
            # The actual position might be stored differently
            pass
        
        # Alternative: Read from articulation using dc (dynamic control) interface
        import omni.isaac.core.utils.numpy.rotations as rot_utils
        from omni.isaac.core.utils.stage import get_current_stage
        
        # Get DC interface
        from omni.isaac.dynamic_control import _dynamic_control
        dc = _dynamic_control.acquire_dynamic_control_interface()
        
        # Get articulation
        articulation = dc.get_articulation(joint_prim.GetPath().pathString)
        if articulation == _dynamic_control.INVALID_HANDLE:
            # Try parent
            parent = joint_prim.GetParent()
            while parent and parent.IsValid():
                articulation = dc.get_articulation(parent.GetPath().pathString)
                if articulation != _dynamic_control.INVALID_HANDLE:
                    break
                parent = parent.GetParent()
        
        if articulation == _dynamic_control.INVALID_HANDLE:
            print(f"[WARN] Could not get articulation handle")
            return None
        
        # Get joint handle
        dof_ptr = dc.find_articulation_dof(articulation, joint_prim.GetName())
        
        if dof_ptr != _dynamic_control.INVALID_HANDLE:
            # Read DOF position
            dof_state = dc.get_dof_state(dof_ptr, _dynamic_control.STATE_POS)
            position = dof_state.pos
            return float(position)
        
        print(f"[WARN] Could not get DOF handle for joint {joint_prim.GetName()}")
        return None
        
    except Exception as e:
        print(f"[ERROR] Failed to read joint position: {e}")
        import traceback
        traceback.print_exc()
        return None

async def physics_validation_loop(
    stage,
    trajectories: List[Dict],
) -> List[Dict]:
    """
    Run batched parallel physics validation across all cloned environments.
    """
    if not trajectories:
        return []
    
    # Add original index to each trajectory for tracking
    for idx, traj in enumerate(trajectories):
        traj['original_index'] = idx
    
    # Create batches
    batches = create_trajectory_batches(trajectories, NUM_COPIES)
    
    # Track results
    all_results = {}  # {original_index: success_bool}
    
    # Process each batch
    for batch_idx, batch in enumerate(batches):
            
        results = await validate_batch_parallel(stage, batch, batch_idx)
        
        for orig_idx, success in results:
            all_results[orig_idx] = success
    
    # Filter to valid trajectories
    valid = [traj for traj in trajectories if all_results.get(traj['original_index'], False)]
    
    print(f"\n[INFO] Batched validation: {len(valid)}/{len(trajectories)} trajectories passed")
    return valid

if __name__ == "__main__":
    main()
