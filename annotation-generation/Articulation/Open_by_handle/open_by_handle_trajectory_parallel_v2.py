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
from omni.isaac.core import World
from omni.isaac.core.utils import stage as stage_utils
from omni.isaac.core.utils import transformations as transform_utils
from omni.isaac.core.prims import XFormPrim
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
from isaacsim.replicator.grasping.grasping_manager import GraspingManager
import isaacsim.replicator.grasping.transform_utils as transform_utils
from omni.physx import get_physx_scene_query_interface, get_physx_interface

# Constants
APPROACH_DISTANCE = 0.15  # Distance to step back before approaching (m)
MOVE_STEPS = 100          # Steps for linear interpolation
CLOSE_STEPS = 64         # Steps for closing gripper
HOLD_STEPS = 60          # Extra physics steps after closing to let contacts settle
CONTACTSLOPCOEFF = 0.1    # PhysX contact slop coefficient
TRAJECTORY_SIM_STEPS_PER_WAYPOINT = 3  # Physics steps per trajectory waypoint
NUM_COPIES = 100         # Number of copies to create in cloner
CLONE_SPACING = 5.0    # Spacing between clones in cloner grid (m)
TRAJ_OVERSHOOT_CHECKS   = 5 
NUM_SAMPLES_PER_JOINT = 3000


# Filtering / geometry normal settings
USE_GEOM_DOOR_NORMAL = True
DOOR_NORMAL_MAX_POINTS = 8000
# Gripper approach axis used for filtering (local axis in gripper pose frame)
GRIPPER_APPROACH_LOCAL = (0, 0, 1)
# Keep grasps where approach_dir · door_normal < DOT_THRESHOLD (door_normal is outward)
DOT_THRESHOLD = 0.1
TARGET_OPEN_RATIO = 0.7
JOINT_SUCCESS_THRESHOLD = 0.95
OVERSHOOT_TOLERANCE_DEG = 2.0   
OVERSHOOT_TOLERANCE_M   = 0.01  


# =======================
# Processing Mode Configuration
# =======================
_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _THIS_DIR.parents[1]

def _path_from_env(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default))).expanduser()

PROCESSING_MODE = "single"  # Options: "single" or "dataset"

# For single object mode
SINGLE_OBJECT_USD = _path_from_env("OPEN_HANDLE_OBJECT_USD", _THIS_DIR / "7120" / "Object.usd")

INPUT_DATASET_PATH = _path_from_env("OPEN_HANDLE_DATASET_PATH", _THIS_DIR)
GRIPPER_USD = _path_from_env("OPEN_HANDLE_GRIPPER_USD", _WORKSPACE_ROOT / "Flying_hand_probe_pro.usd")
LOG_FILE = INPUT_DATASET_PATH / "open_by_handle_completed_objects.txt"

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
    Save validated trajectories to JSON format.
    
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
        # Try to get category (parent of obj_id folder)
        try:
            obj_cat = obj_usd_path.parent.parent.name
            type_str = f"{obj_cat}/{obj_id}/Object.usd"
        except:
            type_str = f"{obj_id}/Object.usd"
    else:
        # Single object mode
        try:
            obj_cat = obj_usd_path.parent.parent.name
        except:
            obj_cat = f"{obj_id}/Object.usd"
    
    # Create Annotation directory at same level as Object.usd
    annotation_dir = obj_usd_path.parent / "Annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    
    # Build JSON structure
    data = {
        "type": obj_cat,
        "bottom_center": {
            "x": float(bottom_center[0]),
            "y": float(bottom_center[1]),
            "z": float(bottom_center[2])
        },
        "trajectories": {}
    }
    if trajectories:
        first_traj = trajectories[0]
        joint_type = first_traj.get("joint_type", "")
        target_disp = float(first_traj.get("target_displacement", 0.0))
        
        if joint_type == "revolute":
            data["target_angle_deg"] = float(np.degrees(target_disp))
        else:
            data["target_distance_m"] = target_disp
            
    # Process each trajectory and Group trajectories by joint name
    
    from collections import defaultdict
    joint_groups = defaultdict(list)
    for traj in trajectories:
        joint_name = traj.get("joint_name", "unknown")
        joint_groups[joint_name].append(traj)
    
    # Process each joint group
    for joint_name, joint_trajs in joint_groups.items():
        joint_dict = {}
        for idx, traj in enumerate(joint_trajs, start=1):
            trajectory_positions = np.asarray(traj["trajectory_positions"], dtype=np.float64)
            trajectory_orientations = np.asarray(traj["trajectory_orientations"], dtype=np.float64)
            
            waypoints = []
            for pos, quat in zip(trajectory_positions, trajectory_orientations):
                waypoint = [
                    float(pos[0]), float(pos[1]), float(pos[2]),
                    float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
                ]
                waypoints.append(waypoint)
            
            joint_dict[str(idx)] = waypoints
        
        data["trajectories"][joint_name] = joint_dict
    
    # Save to JSON
    json_path = annotation_dir / "open_by_handle_trajectory.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"[INFO] Saved {len(trajectories)} trajectories to {json_path}")
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

def find_revolute_prismatic_joints_in_urdf(urdf_path: str) -> List[Tuple[str, str, str, float, float, np.ndarray, List[int]]]:
    """Parse URDF to find movable joints with handles."""
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
            
            # Find link definition
            link_elem = None
            for link in root.findall('link'):
                if link.get('name') == child_link_name:
                    link_elem = link
                    break
            
            if link_elem is None:
                continue
            
            # Find handle visual indices (position under link)
            handle_visual_indices = []
            all_visuals = link_elem.findall('visual')
            
            for idx, visual in enumerate(all_visuals):
                visual_name = visual.get('name', '').lower()
                if 'handle' in visual_name:
                    handle_visual_indices.append(idx)
            
            if not handle_visual_indices:
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
            
            joints_info.append((joint_name, joint_type, child_link_name, lower_limit, upper_limit, joint_axis, handle_visual_indices))
    
    except Exception as e:
        print(f"[ERROR] Failed to parse URDF: {e}")
    
    return joints_info

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
    obj_root_path= resolve_asset_root_under_ref(stage, object_ref_path)
    link_path = f"{obj_root_path}/{child_link_name}"
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


def compute_handle_center_world(stage, handle_paths: List[str]) -> np.ndarray:
    """Compute handle center proxy using world BBox midpoints. Assumes handle exists."""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    centers = []
    for p in handle_paths:
        prim = stage.GetPrimAtPath(p)
        if not prim.IsValid():
            continue
        try:
            prim.Load()
        except Exception:
            pass
        bbox = bbox_cache.ComputeWorldBound(prim)
        c = bbox.GetRange().GetMidpoint()
        centers.append(np.array([float(c[0]), float(c[1]), float(c[2])], dtype=np.float64))

    if not centers:
        # User asked to assume handle exists; still guard to avoid crash.
        return np.zeros(3, dtype=np.float64)

    return np.mean(np.stack(centers, axis=0), axis=0)


def compute_door_surface_normal_from_geometry(
    stage,
    object_ref_path: str,
    child_link_name: str,
    handle_paths: List[str],
    max_points: int = 8000,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute door surface normal by plane-fitting door link mesh points.
    Choose outward sign by pointing toward handle center (handle typically protrudes outward).
    Returns (door_normal, door_center).
    """
    asset_root = resolve_asset_root_under_ref(stage, object_ref_path)
    door_visuals_path = f"{asset_root}/{child_link_name}/visuals"

    door_pts = collect_mesh_points_world(stage, door_visuals_path, max_points=max_points)
    if door_pts.shape[0] < 200:
        print(f"[WARN] Not enough door mesh points for plane fit: {door_pts.shape[0]}")
        return None

    n = plane_normal_pca(door_pts)
    if n is None:
        print("[WARN] PCA plane fit failed")
        return None

    door_center = door_pts.mean(axis=0)
    handle_center = compute_handle_center_world(stage, handle_paths)

    # Choose sign: outward should point AWAY from the handle (your convention)
    if np.dot(n, (handle_center - door_center)) > 0:
        n = -n

    n = n / (np.linalg.norm(n) + 1e-12)
    print(f"[DEBUG] Door surface normal (geom+handle): {n}")
    return n, door_center


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

def compute_door_outward_normal(stage, 
                                object_wrapper_path: str,
                                object_ref_path: str, 
                                child_link_name: str,
                                joint_axis: np.ndarray) -> Optional[np.ndarray]:
    """Compute door's outward normal using hinge-to-door-center vector."""
    obj_root_path = resolve_asset_root_under_ref(stage, object_ref_path)
    link_path = f"{obj_root_path}/{child_link_name}"
    
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



def filter_grasps_by_door_orientation(
    grasp_poses: list,
    door_normal: np.ndarray,
    dot_threshold: float = DOT_THRESHOLD,
    gripper_approach_local=GRIPPER_APPROACH_LOCAL,
) -> list:
    """Filter grasps to keep only those where gripper approaches *into* the door surface.

    door_normal is assumed to point OUTWARD from the door.
    We keep grasps where approach_dir · door_normal < dot_threshold (typically negative).
    """
    if door_normal is None:
        print(f"[WARNING] No door normal, keeping all grasps")
        return grasp_poses

    door_normal = door_normal / (np.linalg.norm(door_normal) + 1e-12)

    filtered = []
    for pos, quat in grasp_poses:
        approach_dir = grasp_approach_dir_world(quat, gripper_approach_local)
        dot = float(np.dot(approach_dir, door_normal))
        if dot < dot_threshold:
            filtered.append((pos, quat))

    print(f"[INFO] Filtered by orientation: {len(grasp_poses)} → {len(filtered)}")
    return filtered


def get_handle_mesh_paths(stage, object_ref_path: str, child_link_name: str, handle_visual_indices: List[int]) -> List[str]:
    """Get USD paths to handle meshes based on visual indices."""
    handle_paths = []
    
    for visual_idx in handle_visual_indices:
        # Expected mesh path pattern:
        #   {asset_root}/{child_link_name}/visuals/visual_mesh_Y/World/mesh
        visual_mesh_name = f"visual_mesh_{visual_idx}"
        asset_root = resolve_asset_root_under_ref(stage, object_ref_path)
        visual_mesh_path = f"{asset_root}/{child_link_name}/visuals/{visual_mesh_name}/World/mesh"
        
        # Check if this path exists
        visual_prim = stage.GetPrimAtPath(visual_mesh_path)
        if visual_prim.IsValid():
            handle_paths.append(visual_mesh_path)
            print(f"[DEBUG] Found handle visual: {visual_mesh_path}")
        else:
            print(f"[WARNING] Handle visual not found: {visual_mesh_path}")
    
    return handle_paths


def create_yaml_config(yaml_path: Path, handle_paths: List[str], gripper_wrapper_path: str, num_candidates: int = 500):
    """Create YAML config for GraspingManager targeting ONLY handle meshes."""
    gripper_ref_path = f"{gripper_wrapper_path}/ref"
    
    # Use first handle path as object_path (GraspingManager will target this)
    # If multiple handles, we'll generate for each separately
    object_path = handle_paths[0] if handle_paths else ""
    if not object_path:
        raise RuntimeError("[create_yaml_config] Empty object_path for GraspingManager (no handle paths)")
    
    config = {
        "object_path": object_path,
        "gripper_path": gripper_wrapper_path,
        "num_orientations": 5,
        "joint_pregrasp_states": {
            f"{gripper_ref_path}/panda_hand/panda_finger_joint1": 0.039737965911626816,
            f"{gripper_ref_path}/panda_hand/panda_finger_joint2": 0.03973797708749771
        },
        "sampler_config": {
            "num_candidates": num_candidates,
            "num_orientations": 5,
            "grasp_align_axis": [0, 1, 0],
            "orientation_sample_axis": [0, 1, 0],
            "gripper_approach_direction": [0, 0, 1],
            "gripper_maximum_aperture": 0.08,
            "gripper_standoff_fingertips": 0.19,
            "lateral_sigma": 0.05,
            "random_seed": 42,
            "sampler_type": "antipodal",
            "verbose": False
        },
        "grasp_phases": [
            {
                "name": "Open",
                "simulation_steps": 32,
                "simulation_step_dt": 0.016666666666666666,
                "joint_drive_targets": {
                    f"{gripper_ref_path}/panda_hand/panda_finger_joint1": 0.04
                }
            },
            {
                "name": "Close",
                "simulation_steps": 32,
                "simulation_step_dt": 0.016666666666666666,
                "joint_drive_targets": {
                    f"{gripper_ref_path}/panda_hand/panda_finger_joint1": 0.0
                }
            }
        ]
    }
    
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    
    return yaml_path


def generate_and_filter_grasps_for_joint(stage,
                                         grasping_manager,
                                         yaml_path: Path,
                                         joint_name: str,
                                         child_link_name: str,
                                         joint_axis: np.ndarray,
                                         handle_visual_indices: List[int],
                                         object_wrapper_path: str,
                                         object_ref_path: str,
                                         gripper_wrapper_path: str,
                                         num_samples: int = 500) -> list:
    """Generate grasp poses ONLY on handle meshes, then filter by door orientation."""
    print(f"\n[INFO] Generating grasps for {joint_name} -> {child_link_name}")
    print(f"[INFO] Handle visual indices: {handle_visual_indices}")
    
    # Get handle mesh paths
    handle_paths = get_handle_mesh_paths(stage, object_ref_path, child_link_name, handle_visual_indices)
    asset_root_dbg = resolve_asset_root_under_ref(stage, object_ref_path)
    print(f"[DEBUG] Resolved asset root under ref: {asset_root_dbg}")
    
    if not handle_paths:
        print(f"[ERROR] No handle meshes found for {child_link_name}")
        return []
    
    all_grasp_poses = []
    
    # Generate grasps for each handle mesh
    for handle_path in handle_paths:
        print(f"[INFO] Generating grasps on: {handle_path}")
        
        # Update YAML to target this handle mesh
        create_yaml_config(yaml_path, [handle_path], gripper_wrapper_path, num_samples)
        
        if not grasping_manager.load_config(str(yaml_path)):
            print(f"[ERROR] Failed to load config for {handle_path}")
            continue
        
        if not grasping_manager.generate_grasp_poses():
            print(f"[WARNING] No poses generated for {handle_path}")
            continue
        
        poses = grasping_manager.get_grasp_poses(in_world_frame=True)
        all_grasp_poses.extend(poses)
        print(f"[INFO] Generated {len(poses)} poses on this handle mesh")
    
    print(f"[INFO] Total raw poses from all handles: {len(all_grasp_poses)}")

    door_center = None

    if USE_GEOM_DOOR_NORMAL:
        result = compute_door_surface_normal_from_geometry(
            stage,
            object_ref_path,
            child_link_name,
            handle_paths,
            max_points=DOOR_NORMAL_MAX_POINTS,
        )
        if result is None:
            door_normal = None
        else:
            door_normal, door_center = result
    else:
        # Legacy method (kept as fallback)
        door_normal = compute_door_outward_normal(
            stage, object_wrapper_path, object_ref_path, child_link_name, joint_axis
        )

    # Draw the computed door normal as a bright yellow arrow in the stage (if available)
    if door_normal is not None:
        if door_center is None:
            link_path = f"{object_ref_path}/{child_link_name}"
            door_center = get_link_local_bbox_center(stage, link_path)

    # Filter by door orientation
    filtered_poses = filter_grasps_by_door_orientation(
        all_grasp_poses,
        door_normal,
        dot_threshold=DOT_THRESHOLD,
        gripper_approach_local=GRIPPER_APPROACH_LOCAL,
    )

    return filtered_poses


def generate_grasps_for_all_joints(stage,
                                   grasping_manager,
                                   yaml_path: Path,
                                   dataset_dir: str,
                                   object_wrapper_path: str,
                                   object_ref_path: str,
                                   gripper_wrapper_path: str,
                                   num_samples_per_joint: int = 500) -> Dict[str, Dict]:
    """Generate filtered grasp poses for all articulated joints."""
    urdf_path = os.path.join(dataset_dir, 'mobility.urdf')
    
    if not os.path.exists(urdf_path):
        print(f"[ERROR] URDF not found at {urdf_path}")
        return {}
    
    joints_info = find_revolute_prismatic_joints_in_urdf(urdf_path)
    
    if not joints_info:
        print(f"[ERROR] No movable joints with handles found")
        return {}
    
    print(f"\n[INFO] Found {len(joints_info)} joint(s) with handles")
    
    all_grasps = {}
    
    for joint_name, joint_type, child_link_name, lower_limit, upper_limit, joint_axis, handle_visual_indices in joints_info:
        print(f"\n{'='*80}")
        print(f"Processing: {joint_name} ({joint_type}) -> {child_link_name}")
        print(f"  Joint axis: {joint_axis}")
        print(f"  Handle visual indices: {handle_visual_indices}")
        print(f"{'='*80}")
        
        grasp_poses = generate_and_filter_grasps_for_joint(
            stage,
            grasping_manager,
            yaml_path,
            joint_name,
            child_link_name,
            joint_axis,
            handle_visual_indices,
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


def find_usd_joint_prim(stage, object_ref_path: str, joint_name_from_urdf: str) -> Optional[Usd.Prim]:
    """
    Find USD joint prim matching URDF joint name.

    Joints are located at: {object_ref_path}/joints/joint_{num}

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


def compute_target_joint_displacement(joint_params: Dict, opening_fraction: float = 0.8) -> float:
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
    opening_fraction: float = 0.8,
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
    
    # First pass: make the hierarchy editable.
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

    disabled_set = set(instanceable_disabled_paths)
    if pre_disabled_instanceable_paths:
        disabled_set.update(pre_disabled_instanceable_paths)

    # Second pass: now that instanceable subtrees are editable, collect candidates.
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsValid():
            continue
        if prim.IsA(UsdGeom.Mesh):
            candidate_prim_paths.append(prim.GetPath().pathString)
        elif prim.HasAPI(UsdPhysics.CollisionAPI):
            candidate_prim_paths.append(prim.GetPath().pathString)

    bound_target_set = set()

    # Third pass: bind on a stable parent prim instead of deep mesh prims.
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
    
    print(f"\n[INFO] Applying high-friction physics material to object...")
    num_prims_with_material, object_material_instanceable_paths = create_and_bind_high_friction_material(
        stage,
        OBJECT_REF_PATH,
        static_friction=2.0,
        dynamic_friction=2.0,
        restitution=0.0
    )
    
    if num_prims_with_material == 0:
        print(f"[WARN] No prims found to apply physics material")
    
    await omni.kit.app.get_app().next_update_async()
    
    # =====================
    # DISABLE INSTANCEABLE FOR GRASP GENERATION
    # =====================
    print(f"\n[INFO] Disabling instanceable for grasp generation...")
    changed_instanceable_paths = disable_instanceable_for_grasp_generation(stage, OBJECT_REF_PATH)
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
    # =====================
    # FIX OBJECT BASE TO WORLD (prevent whole-asset drift/oscillation)
    # =====================
    
    # Get bottom center and add ground plane
    bottom_center = get_bbox_bottom_center(stage, object_wrapper_path)
    if bottom_center:
        bottom_center_list = [float(bottom_center[0]), float(bottom_center[1]), float(bottom_center[2])]
        print(f"[INFO] Object bottom center: {bottom_center_list}")
    else:
        bottom_center_list = None
        print(f"[WARN] Could not compute bottom center")
        
    GroundPlane(prim_path="/World/GroundPlane", z_position=bottom_center[2] - 0.0001)
    await omni.kit.app.get_app().next_update_async()
    
    # =====================
    # GENERATE GRASPS
    # =====================
    dataset_dir = obj_usd.parent
    yaml_path = output_dir / "grasp_config.yaml"
    
    print(f"\n[INFO] Generating grasps...")
    grasping_manager = GraspingManager()
    
    all_grasps = generate_grasps_for_all_joints(
        stage,
        grasping_manager,
        yaml_path,
        str(dataset_dir),
        OBJECT_WRAPPER_PATH,
        OBJECT_REF_PATH,
        GRIPPER_WRAPPER_PATH,
        num_samples_per_joint = NUM_SAMPLES_PER_JOINT
    )
    
    if not all_grasps:
        print(f"[ERROR] No grasps generated")
        return
    
    # =====================
    # Filtering grasps to avoid overlaps
    # =====================
    print(f"\n[INFO] Filtering grasps via overlap detection...")

    # 1. Disable instanceable on gripper so we can manipulate prims
    gripper_instanceable_paths = disable_instanceable_for_grasp_generation(stage, GRIPPER_REF_PATH)
    await omni.kit.app.get_app().next_update_async()

    # 2. Collect all gripper mesh prims, record which already had CollisionAPI.
    #    We then DISABLE the collider during the query so the gripper's own
    #    collision geometry doesn't self-report — only overlaps with the object count.
    gripper_root_prim = stage.GetPrimAtPath(GRIPPER_REF_PATH)
    gripper_mesh_paths: List[str] = []          # meshes we touched
    gripper_had_collision: List[bool] = []       # whether they already had CollisionAPI

    for prim in Usd.PrimRange(gripper_root_prim):
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            continue
        had = prim.HasAPI(UsdPhysics.CollisionAPI)
        gripper_had_collision.append(had)
        gripper_mesh_paths.append(prim.GetPath().pathString)

        # Add CollisionAPI + convex hull if not already present
        if not had:
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr().Set("convexHull")
        
        # Always DISABLE the collider — we re-enable per-mesh only during its query
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)

    print(f"[INFO] Prepared {len(gripper_mesh_paths)} gripper mesh collider(s) (disabled)")
    await omni.kit.app.get_app().next_update_async()
    
    physx_iface = get_physx_interface()
    sq_iface = get_physx_scene_query_interface()

    def _overlap_at_pose(pos_np: np.ndarray, quat_np: np.ndarray) -> bool:
        """
        Place gripper at pose, then for each mesh, query
        overlap_shape_any. Return True if ANY mesh overlaps.
        
        overlap_shape_any returns True if the shape itself has a collider (self-hit)
        OR overlaps another collider. By querying each mesh that has collider disabled,
        the only possible hits are against other object colliders.
        """
        set_gripper_world_pose(stage, GRIPPER_WRAPPER_PATH, pos_np, quat_np)
        physx_iface.force_load_physics_from_usd()

        for mesh_path in gripper_mesh_paths:
            mesh_prim = stage.GetPrimAtPath(mesh_path)
            if not mesh_prim.IsValid():
                continue

            enc0, enc1 = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(mesh_path))
            hit = sq_iface.overlap_shape_any(enc0, enc1)

            if hit:
                return True

        return False

    # 3. Filter each joint's grasps
    filtered_all_grasps: Dict[str, Dict] = {}

    for joint_name, joint_data in all_grasps.items():
        grasp_poses = joint_data["grasp_poses"]
        kept = []
        rejected = 0

        for pos, quat in grasp_poses:
            # Normalise types
            if isinstance(pos, (Gf.Vec3d, Gf.Vec3f)):
                pos_np = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float64)
            else:
                pos_np = np.asarray(pos, dtype=np.float64)

            if isinstance(quat, (Gf.Quatd, Gf.Quatf)):
                w = float(quat.GetReal()); im = quat.GetImaginary()
                quat_np = np.array([w, float(im[0]), float(im[1]), float(im[2])], dtype=np.float64)
            else:
                quat_np = np.asarray(quat, dtype=np.float64)

            if _overlap_at_pose(pos_np, quat_np):
                rejected += 1
            else:
                kept.append((pos, quat))

        print(f"[INFO] Joint '{joint_name}': {len(grasp_poses)} → {len(kept)} "
              f"(rejected {rejected} overlapping grasps)")

        if kept:
            filtered_all_grasps[joint_name] = {**joint_data, "grasp_poses": kept}

    all_grasps = filtered_all_grasps

    # 4. Restore gripper colliders to their original state
    for mesh_path, had_collision in zip(gripper_mesh_paths, gripper_had_collision):
        prim = stage.GetPrimAtPath(mesh_path)
        if not prim.IsValid():
            continue
        if had_collision:
            # It had a collider before — re-enable it
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(True)
        else:
            # We added the collider — remove it entirely
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            prim.RemoveAPI(UsdPhysics.CollisionAPI)

    print(f"[INFO] Restored gripper collider state on {len(gripper_mesh_paths)} mesh(es)")

    # 5. Restore gripper instanceable state
    restore_instanceable(stage, gripper_instanceable_paths)
    await omni.kit.app.get_app().next_update_async()
    
    # 6. Final PhysX reload to evict any temporary colliders from the simulation
    physx_iface.force_load_physics_from_usd()
    await omni.kit.app.get_app().next_update_async()

    print(f"[INFO] Overlap filtering complete. "
          f"Joints with valid grasps: {len(all_grasps)}")
        
    # =====================
    # Enable gravity for physics validation
    # =====================
    print("[INFO] Enabling gravity on articulated object for physics validation...")
    set_object_gravity_enabled(stage, OBJECT_REF_PATH, enabled=True)
    await omni.kit.app.get_app().next_update_async()

    
    # =====================
    # TRAJECTORY GENERATION + PHYSICS VALIDATION
    # =====================
    print("[INFO] Generating trajectories for all grasps...")
    all_trajectories = generate_trajectories_for_all_grasps(
        stage,
        all_grasps,
        OBJECT_REF_PATH,
        num_trajectory_steps=200,
        opening_fraction=TARGET_OPEN_RATIO
    )

    # =====================
    # RESTORE INSTANCEABLE
    # =====================
    print(f"\n[INFO] Restoring instanceable state...")
    restore_instanceable(stage, object_material_instanceable_paths + changed_instanceable_paths)
    await omni.kit.app.get_app().next_update_async()
    
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
        env_root_path = resolve_asset_root_under_ref(stage, env_obj_ref)
        fixed_joint_path = f"{env_path(i)}/ObjectFixedToWorld"
        
        print(f"[INFO] Fixing object in env_{i}...")
        fix_object_base_to_world(
            stage, 
            env_root_path, 
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
            grasping_manager
        )
        print(
            f"[INFO] Physics validation kept {len(valid_trajectories)}/"
            f"{len(all_trajectories)} trajectories"
        )
        
        # Save validated trajectories to JSON
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
    
    # Cleanup
    grasping_manager.clear()
    if yaml_path.exists():
        yaml_path.unlink()
        
    if PROCESSING_MODE == "dataset":
        if valid_trajectories:
            mark_object_completed(LOG_FILE, obj_id)
        else:
            print(f"[WARN] No valid trajectories for {obj_id}, not marking as completed")
    
    print(f"\n{'='*80}")
    print(f"Completed: {obj_id}")
    print(f"{'='*80}\n")


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
        task = asyncio.ensure_future(run_pipeline())
        
        while not task.done():
            simulation_app.update()
        
        if task.exception():
            raise task.exception()

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


def compute_stepback_and_approach_poses(
    grasp_position: np.ndarray,
    grasp_quaternion: np.ndarray,
    approach_distance: float,
    move_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a simple step-back + linear-approach path in world coordinates.

    Args:
        grasp_position: [3] grasp position (world)
        grasp_quaternion: [4] grasp quaternion [w, x, y, z]
        approach_distance: distance to step back along local Z (meters)
        move_steps: number of interpolation steps

    Returns:
        positions: [N, 3] positions from step-back to grasp
        quaternions: [N, 4] same orientation at all waypoints
    """
    grasp_position = np.asarray(grasp_position, dtype=np.float64).reshape(3)
    grasp_quaternion = np.asarray(grasp_quaternion, dtype=np.float64).reshape(4)

    # Use existing helper: offset along local Z by -approach_distance (step back)
    back_pos, _ = offset_pose_along_local_z(grasp_position, grasp_quaternion, -approach_distance)

    positions = []
    for t in range(move_steps + 1):
        alpha = t / float(move_steps) if move_steps > 0 else 1.0
        cur = back_pos * (1.0 - alpha) + grasp_position * alpha
        positions.append(cur)

    positions = np.stack(positions, axis=0)
    quats = np.tile(grasp_quaternion, (positions.shape[0], 1))
    return positions, quats

async def validate_batch_parallel(
    stage,
    batch: List[Dict],  # trajectories for this batch
    batch_index: int,
) -> List[Tuple[int, bool]]:
    """
    Validate a batch of trajectories in parallel across environments.
    
    Args:
        stage: USD stage
        batch: List of trajectory dicts (length <= NUM_COPIES)
        batch_index: Which batch number this is (for logging)
    
    Returns:
        List of (original_trajectory_index, success) tuples
    """
    print(f"\n[BATCH {batch_index}] Validating {len(batch)} trajectories in parallel...")
    
    if timeline.is_playing():
        timeline.stop()
        await step_simulation(5)
    # Step 1: Reset all environments to initial state
    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        object_ref_path = obj_ref(env_idx)
        joint_name = traj.get('joint_name', '')
        
        # Reset joint to closed position
        joint_prim = find_usd_joint_prim(stage, object_ref_path, joint_name)
        if joint_prim:
            joint_params = get_joint_world_parameters(stage, joint_prim)
            if joint_params:
                lower_limit = joint_params["lower_limit"]
                drive_kind = "angular" if joint_params["joint_type"] == "revolute" else "linear"
                set_usd_joint_drive_target(stage, joint_prim.GetPath().pathString, lower_limit, drive_kind)
                set_joint_position_direct(stage, joint_prim, lower_limit)
    
    await ensure_timeline_playing()
    await step_simulation(60)  # Let resets settle

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
    
    # Step 3: Compute approach trajectories for all envs
    all_approach_data = []
    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        grasp_pos = np.asarray(traj["grasp_position"], dtype=np.float64)
        grasp_quat = np.asarray(traj["grasp_quaternion"], dtype=np.float64)
        
        approach_positions, approach_quats = compute_stepback_and_approach_poses(
            grasp_pos, grasp_quat, approach_distance=APPROACH_DISTANCE, move_steps=MOVE_STEPS
        )
        
        all_approach_data.append({
            'positions': approach_positions,
            'quats': approach_quats,
        })
    env_failed = [False] * len(batch)
    
    # Step 4: Execute approach phase in parallel (with displacement checks)
    open_target = 0.04
    
    for step_idx in range(MOVE_STEPS + 1):
        # Set poses for ALL environments simultaneously
        for env_idx in range(len(batch)):
            
            gripper_wrapper_path = grip_wrap(env_idx)
            gripper_ref_path = grip_ref(env_idx)
            
            pos = all_approach_data[env_idx]['positions'][step_idx]
            quat = all_approach_data[env_idx]['quats'][step_idx]
            
            set_gripper_world_pose(stage, gripper_wrapper_path, pos, quat)
            
            finger_joint1 = f"{gripper_ref_path}/panda_hand/panda_finger_joint1"
            finger_joint2 = f"{gripper_ref_path}/panda_hand/panda_finger_joint2"
            set_usd_joint_drive_target(stage, finger_joint1, open_target, drive_kind="linear")
            set_usd_joint_drive_target(stage, finger_joint2, open_target, drive_kind="linear")
        
        # Step simulation ONCE for all environments
        await step_simulation(1)
    
    #Check if any env overshot during approach
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
            
    # Step 5: Close gripper phase (parallel)
    close_target = 0.0
    
    for _ in range(CLOSE_STEPS):
        for env_idx in range(len(batch)):
            
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
    
    # Step 6: Hold phase (parallel)
    for _ in range(HOLD_STEPS):
        for env_idx in range(len(batch)):
            
            gripper_ref_path = grip_ref(env_idx)
            finger_joint1 = f"{gripper_ref_path}/panda_hand/panda_finger_joint1"
            finger_joint2 = f"{gripper_ref_path}/panda_hand/panda_finger_joint2"
            
            set_usd_joint_drive_target(stage, finger_joint1, close_target, drive_kind="linear")
            set_usd_joint_drive_target(stage, finger_joint2, close_target, drive_kind="linear")
        
        await step_simulation(1)
    
    # Step 7: Execute trajectory phase (parallel)
    max_traj_length = 0
    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        traj_positions = np.asarray(traj.get("trajectory_positions", []), dtype=np.float64)
        max_traj_length = max(max_traj_length, traj_positions.shape[0])
    
    if max_traj_length > 0:
        traj_checkpoints = set(np.linspace(0, max_traj_length - 1, TRAJ_OVERSHOOT_CHECKS, dtype=int).tolist())
    else:
        traj_checkpoints = set()
    
    for traj_step in range(max_traj_length):
        for env_idx in range(len(batch)):
            
            traj = batch[env_idx]
            traj_positions = np.asarray(traj.get("trajectory_positions", []), dtype=np.float64)
            traj_orientations = np.asarray(traj.get("trajectory_orientations", []), dtype=np.float64)
            
            if traj_step >= traj_positions.shape[0]:
                continue  # This env's trajectory is shorter
            
            gripper_wrapper_path = grip_wrap(env_idx)
            gripper_ref_path = grip_ref(env_idx)
            
            pos = traj_positions[traj_step]
            quat = traj_orientations[traj_step]
            
            set_gripper_world_pose(stage, gripper_wrapper_path, pos, quat)
            
            finger_joint1 = f"{gripper_ref_path}/panda_hand/panda_finger_joint1"
            finger_joint2 = f"{gripper_ref_path}/panda_hand/panda_finger_joint2"
            set_usd_joint_drive_target(stage, finger_joint1, close_target, drive_kind="linear")
            set_usd_joint_drive_target(stage, finger_joint2, close_target, drive_kind="linear")
        
        await step_simulation(TRAJECTORY_SIM_STEPS_PER_WAYPOINT)
        
        for env_idx in range(len(batch)):
            if env_failed[env_idx]:
                continue

            traj = batch[env_idx]
            target_disp = traj.get("target_displacement", 0.0)

            # Per-trajectory waypoints
            traj_positions = np.asarray(traj.get("trajectory_positions", []), dtype=np.float64)
            local_len = traj_positions.shape[0]

            # If this traj is shorter, skip when we're past its end
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
    grasping_manager = None,
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
