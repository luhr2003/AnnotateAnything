import os
os.environ["OMNI_LOG_LEVEL_DEFAULT"] = "error"

import asyncio
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import carb
carb.settings.get_settings().set("/log/level", 3)
carb.settings.get_settings().set("/log/fileLogLevel", 3)
carb.settings.get_settings().set("/rtx/instanceLogging", False)

import warnings
warnings.filterwarnings("ignore")

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
import isaacsim.replicator.grasping.transform_utils as transform_utils
from omni.physx import get_physx_scene_query_interface, get_physx_interface

# Constants
APPROACH_DISTANCE = 0.10  # Distance to step back before approaching (m)
MOVE_STEPS = 200          # Steps for linear interpolation
CLOSE_STEPS = 64         # Steps for closing gripper
HOLD_STEPS = 60          # Extra physics steps after closing to let contacts settle
CONTACTSLOPCOEFF = 0.1    # PhysX contact slop coefficient
TRAJECTORY_SIM_STEPS_PER_WAYPOINT = 3  # Physics steps per trajectory waypoint
NUM_COPIES = 100         # Number of copies to create in cloner
CLONE_SPACING = 5.0    # Spacing between clones in cloner grid (m)
TRAJ_OVERSHOOT_CHECKS   = 5 
NUM_SAMPLES_PER_JOINT = 1000


# Filtering / geometry normal settings
USE_GEOM_DOOR_NORMAL = True
DOOR_NORMAL_MAX_POINTS = 8000
# Gripper approach axis used for filtering (local axis in gripper pose frame)
HAND_PALM_DIRECTION = (0, 1, 0)
HAND_FINGER_DIRECTION = (1, 0, 0)
HAND_FORWARD_LEAN_DEG = 15.0
# Keep grasps where approach_dir · door_normal < DOT_THRESHOLD (door_normal is outward)
DOT_THRESHOLD = 0.1
TARGET_OPEN_RATIO = 0.7
JOINT_SUCCESS_THRESHOLD = 0.95
OVERSHOOT_TOLERANCE_DEG = 2.0   
OVERSHOOT_TOLERANCE_M   = 0.01  

#HANDLE OVERIDE FOR DOORS
MANUAL_HANDLE_OVERRIDES = {
    # obj_id: [(door_joint_name, handle_link_name)]
    # door_joint_name  = the revolute joint whose motion to drive (the hinge)
    # handle_link_name = the child link containing the graspable surface
    "8867": [("joint_1", "link_2", 0)],  # joint_0 is the door hinge, link_2 has the handle
    "8893": [("joint_2", "link_1", None)],
    "8897": [("joint_1", "link_2", 1)],
    "8903": [("joint_2", "link_1", 0)],
    "8930": [
        ("joint_3", "link_1", None),  
        ("joint_4", "link_2", None),
    ],
    "8961": [
        ("joint_2", "link_2", 1),
        ("joint_1", "link_1", 0)
    ],
    "8966":[("joint_1", "link_2", 2)],
    "8994": [("joint_2", "link_1", 3)],
    "9003": [("joint_1", "link_2", 2)],
    "9127":[("joint_1", "link_2", 1)],
    "9164":[("joint_1", "link_2", 0)],
    "9168":[
        ("joint_1", "link_1", 0),
        ("joint_0", "link_0", 0)
    ],
    "9263": [("joint_1", "link_2", 1)],
    "9277": [("joint_2", "link_1", 0)],
    "9280": [("joint_2", "link_2", 0)],
    "9281": [("joint_1", "link_2", 0)],
    "9288": [("joint_0", "link_2", 0)],
    "9388": [("joint_0", "link_0", 11)],
    "9393": [("joint_1", "link_2", 0)],
    "9410": [("joint_1", "link_2", 0)]   
}

# =======================
# Processing Mode Configuration
# =======================
_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _THIS_DIR.parents[1]

def _path_from_env(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default))).expanduser()

PROCESSING_MODE = "single"  # Options: "single" or "dataset"

# For single object mode
SINGLE_OBJECT_USD = _path_from_env("DEX3_OPEN_HANDLE_OBJECT_USD", _THIS_DIR / "7120" / "Object.usd")

INPUT_DATASET_PATH = _path_from_env("DEX3_OPEN_HANDLE_DATASET_PATH", _THIS_DIR)
HAND_USD = _path_from_env("DEX3_OPEN_HANDLE_HAND_USD", _THIS_DIR / "dex3_1_right.usd")
LOG_FILE = INPUT_DATASET_PATH / "dex3_1_open_by_handle_completed_objects.txt"

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
    return f"{env_path(i)}/Hand"

def grip_ref(i: int) -> str:
    return f"{env_path(i)}/Hand/ref"

OBJECT_WRAPPER_PATH = obj_wrap(0)
OBJECT_REF_PATH = obj_ref(0)
GRIPPER_WRAPPER_PATH = grip_wrap(0)
GRIPPER_REF_PATH = grip_ref(0)

# =======================
# Hand Closing HardCode
# =======================
RIGHT_HAND_OPEN = {
    "right_hand_index_0_joint":  0.0,
    "right_hand_index_1_joint":  0.0,
    "right_hand_middle_0_joint": 0.0,
    "right_hand_middle_1_joint": 0.0,
    "right_hand_thumb_0_joint":  0.0,
    "right_hand_thumb_1_joint":  0.0,
    "right_hand_thumb_2_joint":  0.0,
}

RIGHT_HAND_CLOSE_WBC = {
    "right_hand_index_0_joint":  0.6,
    "right_hand_index_1_joint":  1.2,
    "right_hand_middle_0_joint": 0.6,
    "right_hand_middle_1_joint": 1.2,
    "right_hand_thumb_0_joint":  0.0,
    "right_hand_thumb_1_joint": -0.7,
    "right_hand_thumb_2_joint": -0.7,
}

RIGHT_HAND_CLOSE_INDEX = {
    "right_hand_index_0_joint":  1.5,
    "right_hand_index_1_joint":  1.5,
    "right_hand_middle_0_joint": 0.6,
    "right_hand_middle_1_joint": 1.5,
    "right_hand_thumb_0_joint": -0.5,
    "right_hand_thumb_1_joint": -0.7,
    "right_hand_thumb_2_joint": -0.7,
}

RIGHT_HAND_CLOSE_MIDDLE = {
    "right_hand_index_0_joint":  1.0,
    "right_hand_index_1_joint":  1.5,
    "right_hand_middle_0_joint": 1.0,
    "right_hand_middle_1_joint": 1.5,
    "right_hand_thumb_0_joint":  0.0,
    "right_hand_thumb_1_joint": -0.7,
    "right_hand_thumb_2_joint": -0.7,
}

RIGHT_HAND_CLOSE_RING = {
    "right_hand_index_0_joint":  0.6,
    "right_hand_index_1_joint":  1.5,
    "right_hand_middle_0_joint": 1.5,
    "right_hand_middle_1_joint": 1.5,
    "right_hand_thumb_0_joint":  0.5,
    "right_hand_thumb_1_joint": -0.7,
    "right_hand_thumb_2_joint": -0.7,
}

# Ordered list matching the joint_names order in the config
HAND_CLOSE_CONFIG = "wbc"
RIGHT_HAND_JOINT_NAMES = [
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]

def hand_pose_to_tensor(pose: dict, device="cpu") -> torch.Tensor:
    """Convert a hand pose dict to an ordered [7] tensor."""
    return torch.tensor(
        [pose[j] for j in RIGHT_HAND_JOINT_NAMES],
        dtype=torch.float32,
        device=device,
    )

# Pre-built tensors
RIGHT_HAND_OPEN_T         = hand_pose_to_tensor(RIGHT_HAND_OPEN)
RIGHT_HAND_CLOSE_WBC_T    = hand_pose_to_tensor(RIGHT_HAND_CLOSE_WBC)
RIGHT_HAND_CLOSE_INDEX_T  = hand_pose_to_tensor(RIGHT_HAND_CLOSE_INDEX)
RIGHT_HAND_CLOSE_MIDDLE_T = hand_pose_to_tensor(RIGHT_HAND_CLOSE_MIDDLE)
RIGHT_HAND_CLOSE_RING_T   = hand_pose_to_tensor(RIGHT_HAND_CLOSE_RING)

# Lookup by name
RIGHT_HAND_POSES = {
    "open":   RIGHT_HAND_OPEN,
    "wbc":    RIGHT_HAND_CLOSE_WBC,
    "index":  RIGHT_HAND_CLOSE_INDEX,
    "middle": RIGHT_HAND_CLOSE_MIDDLE,
    "ring":   RIGHT_HAND_CLOSE_RING,
}

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
    
def ensure_object_colliders(stage, object_ref_path: str) -> List[str]:
    """
    Ensure all meshes under object_ref_path have CollisionAPI with convex decomposition.
    Disables instanceable on any prim that needs to be modified.
    
    Returns:
        List of prim paths that had instanceable disabled (to exclude from restore).
    """
    root = stage.GetPrimAtPath(object_ref_path)
    if not root.IsValid():
        print(f"[ERROR] ensure_object_colliders: invalid path {object_ref_path}")
        return []

    instanceable_disabled: List[str] = []
    meshes_touched = 0
    meshes_already_ok = 0

    for prim in Usd.PrimRange(root):
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            continue

        # Disable instanceable BEFORE modifying — otherwise edits are ignored
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
            instanceable_disabled.append(prim.GetPath().pathString)

        # Also disable on all ancestors up to root (instanceable is inherited)
        ancestor = prim.GetParent()
        while ancestor and ancestor.IsValid() and ancestor.GetPath() != root.GetPath():
            if ancestor.IsInstanceable():
                ancestor.SetInstanceable(False)
                instanceable_disabled.append(ancestor.GetPath().pathString)
            ancestor = ancestor.GetParent()

        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr().Set("convexHull")
            meshes_touched += 1
        else:
            # Already has collision — ensure approximation is set correctly
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr().Set("convexHull")
            meshes_already_ok += 1

    print(f"[INFO] ensure_object_colliders under {object_ref_path}:")
    print(f"  - Colliders added/updated: {meshes_touched}")
    print(f"  - Already had collision:   {meshes_already_ok}")
    print(f"  - Instanceable disabled:   {len(instanceable_disabled)}")
    return instanceable_disabled

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
    bottom_center: List[float],
    approach_data: Dict[int, Dict],  # {original_index: {positions, quats, joint_angles}}
) -> Path:
    """
    Save validated trajectories to JSON format.
    Each waypoint: [x, y, z, w, qx, qy, qz, j0..j6 (radians)]
    Full sequence = approach waypoints + trajectory waypoints (continuous).
    """
    obj_id = obj_usd_path.parent.name
    try:
        obj_cat = obj_usd_path.parent.parent.name
    except:
        obj_cat = obj_id

    annotation_dir = obj_usd_path.parent / "Annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "type": obj_cat,
        "bottom_center": {
            "x": float(bottom_center[0]),
            "y": float(bottom_center[1]),
            "z": float(bottom_center[2])
        },
        "joint_names": RIGHT_HAND_JOINT_NAMES,
        "waypoint_format": "x, y, z, w, qx, qy, qz, j0..j6 (radians)",
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

    from collections import defaultdict
    joint_groups = defaultdict(list)
    for traj in trajectories:
        joint_name = traj.get("joint_name", "unknown")
        joint_groups[joint_name].append(traj)

    for joint_name, joint_trajs in joint_groups.items():
        joint_dict = {}
        for idx, traj in enumerate(joint_trajs, start=1):
            orig_idx = traj.get("original_index", -1)

            # --- Approach phase ---
            approach = approach_data.get(orig_idx)
            approach_waypoints = []
            if approach is not None:
                app_positions    = np.asarray(approach["positions"],    dtype=np.float64)
                app_orientations = np.asarray(approach["quats"],        dtype=np.float64)
                app_joints_rad   = np.asarray(approach["joint_angles"], dtype=np.float64)

                for pos, quat, joints in zip(app_positions, app_orientations, app_joints_rad):
                    waypoint = [
                        float(pos[0]),  float(pos[1]),  float(pos[2]),
                        float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]),
                        *[float(j) for j in joints]
                    ]
                    approach_waypoints.append(waypoint)

            # --- Trajectory phase (hand fully closed) ---
            traj_positions    = np.asarray(traj["trajectory_positions"],    dtype=np.float64)
            traj_orientations = np.asarray(traj["trajectory_orientations"], dtype=np.float64)
            close_joints_rad  = hand_pose_to_tensor(RIGHT_HAND_POSES[HAND_CLOSE_CONFIG]).numpy()

            traj_waypoints = []
            for pos, quat in zip(traj_positions, traj_orientations):
                waypoint = [
                    float(pos[0]),  float(pos[1]),  float(pos[2]),
                    float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]),
                    *[float(j) for j in close_joints_rad]
                ]
                traj_waypoints.append(waypoint)

            joint_dict[str(idx)] = {
                "approach": approach_waypoints,
                "trajectory": traj_waypoints,
            }

        data["trajectories"][joint_name] = joint_dict

    json_path = annotation_dir / "dex3_1_open_by_handle_trajectory.json"
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

def get_hand_eef_local_offset(
    stage,
    hand_ref_path: str,
    eef_link_name: str = "eef",
) -> Optional[np.ndarray]:
    """
    Compute EEF offset as a 3D vector in the hand root's LOCAL frame.
    Returns None if EEF link not found.
    """
    eef_path = f"{hand_ref_path}/{eef_link_name}"
    eef_prim = stage.GetPrimAtPath(eef_path)

    if not eef_prim.IsValid():
        print(f"[WARN] EEF link not found at {eef_path}")
        return None

    root_prim = stage.GetPrimAtPath(hand_ref_path)
    if not root_prim.IsValid():
        return None

    root_xf = UsdGeom.Xformable(root_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    eef_xf  = UsdGeom.Xformable(eef_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    root_pos = np.array([float(root_xf[3][0]), float(root_xf[3][1]), float(root_xf[3][2])])
    eef_pos  = np.array([float(eef_xf[3][0]),  float(eef_xf[3][1]),  float(eef_xf[3][2])])

    world_offset = eef_pos - root_pos

    # Rotation matrix of root frame: rows are world-space basis vectors
    root_rot_mat = np.array([
        [float(root_xf[0][0]), float(root_xf[0][1]), float(root_xf[0][2])],
        [float(root_xf[1][0]), float(root_xf[1][1]), float(root_xf[1][2])],
        [float(root_xf[2][0]), float(root_xf[2][1]), float(root_xf[2][2])],
    ])
    local_offset = root_rot_mat @ world_offset

    print(f"[INFO] EEF local offset (hand root -> '{eef_link_name}'): {local_offset}")
    return local_offset

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
    Isaac Sim 5.1.0 debug draw (simple + reliable):
      - uses draw_lines(starts, ends, colors, sizes)
      - uses clear_lines() with NO args
    """
    from isaacsim.util.debug_draw import _debug_draw

    dd = _debug_draw.acquire_debug_draw_interface()

    if clear_first:
        dd.clear_lines()  # 5.1.0: no arguments  [oai_citation:1‡Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/utilities/debugging/ext_isaacsim_util_debug_draw.html)

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

    # 3 segments:
    # 1) hinge axis (red): centered at hinge origin
    p0_axis = o - a * length
    p1_axis = o + a * length

    # 2) radial (green): from hinge origin
    p0_rad = o
    p1_rad = o + r * length

    # 3) door normal (blue): from hinge origin (keep same origin to compare)
    p0_n = o
    p1_n = o + n * length

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
        (1.0, 0.2, 0.2, 1.0),  # red
        (0.2, 1.0, 0.2, 1.0),  # green
        (0.2, 0.6, 1.0, 1.0),  # blue
    ]
    sizes = [float(width), float(width), float(width)]

    dd.draw_lines(starts, ends, colors, sizes)

def compute_door_normal_from_joint(
    stage,
    joint_prim: Usd.Prim,
    joint_params: Dict,
) -> Optional[np.ndarray]:
    if joint_params.get("joint_type") != "revolute":
        return None

    j = UsdPhysics.Joint(joint_prim)

    # --- Get body1 ---
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

    # --- Recompute hinge axis from body1 side ---
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

    # --- Hinge origin (pivot already in world frame from joint_params) ---
    hinge_origin = np.asarray(joint_params["pivot"], dtype=np.float64)

    # --- Door center: body1 bbox center ---
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_],
    )
    bbox = bbox_cache.ComputeWorldBound(body1_prim)
    mid = bbox.GetRange().GetMidpoint()
    door_center = np.array([float(mid[0]), float(mid[1]), float(mid[2])], dtype=np.float64)

    # --- Radial: project (door_center - hinge_origin) onto plane ⟂ hinge_axis ---
    v = door_center - hinge_origin
    v_radial = v - np.dot(v, hinge_axis) * hinge_axis
    rmag = np.linalg.norm(v_radial)
    if rmag < 1e-4:
        print(f"[JointNormal] door_center lies on hinge axis — cannot determine radial")
        return None
    r_hat = v_radial / rmag

    # --- Door normal (YOUR CURRENT COMPUTATION) ---
    # This is the "direction on directed circle" tangent if you wanted that.
    # If you want a different formula, only change this block.
    door_normal = np.cross(hinge_axis, r_hat)  # (left-hand tangent); swap order for opposite
    nmag = np.linalg.norm(door_normal)
    if nmag < 1e-4:
        print(f"[JointNormal] door_normal degenerate")
        return None
    door_normal = door_normal / nmag
    
    L = 0.35

    _dbg_draw_three_vectors_simple(
        hinge_origin=hinge_origin,
        hinge_axis=hinge_axis,
        radial=v_radial,          # IMPORTANT: pass the raw radial (projected) vector
        door_normal=door_normal,  # whatever you computed
        length=L,
        width=3.0,
        clear_first=True,         # if you call this many times, it will overwrite each time
    )

    print(
        f"[JointNormal] axis_token={joint_params.get('axis_token')}, "
        f"hinge_axis(world)={hinge_axis}, hinge_origin={hinge_origin}, "
        f"door_center={door_center}, radial={r_hat}, door_normal={door_normal}"
    )

    return door_normal

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

def visualize_door_normal(stage, door_center: np.ndarray, door_normal: np.ndarray, length: float = 0.3, prim_path: str = "/World/Debug/DoorNormal"):
    """Draw a line (thin cylinder) from door_center along door_normal to visualize surface normal."""

    tip = door_center + door_normal * length
    mid = (door_center + tip) / 2.0
    direction = tip - door_center
    
    # Build rotation from +Y (cylinder default axis) to door_normal
    up = np.array([0.0, 1.0, 0.0])
    axis = np.cross(up, direction / (np.linalg.norm(direction) + 1e-12))
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        # Already aligned or anti-aligned
        if np.dot(up, door_normal) < 0:
            quat_wxyz = np.array([0.0, 1.0, 0.0, 0.0])  # 180° around X
        else:
            quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(up, door_normal), -1, 1))
        half = angle / 2.0
        quat_wxyz = np.array([np.cos(half), *(np.sin(half) * axis)])

    # Define cylinder prim
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        stage.RemovePrim(prim_path)
    
    cylinder = UsdGeom.Cylinder.Define(stage, prim_path)
    cylinder.CreateRadiusAttr(0.01)
    cylinder.CreateHeightAttr(length)

    xf = UsdGeom.Xformable(cylinder)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2])))
    xf.AddOrientOp().Set(Gf.Quatf(float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])))

    # Color it red
    gprim = UsdGeom.Gprim(cylinder)
    gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])

    print(f"[DEBUG] Door normal visualized: center={door_center}, normal={door_normal}, path={prim_path}")

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
    gripper_approach_local=HAND_PALM_DIRECTION,
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

def rotate_quat_around_local_axis(quat_wxyz: np.ndarray, local_axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate a quaternion by angle_deg around a local axis (expressed in the hand's local frame).
    
    Args:
        quat_wxyz: [4] quaternion [w,x,y,z]
        local_axis: [3] axis in hand local frame to rotate around
        angle_deg: rotation angle in degrees
    
    Returns:
        [4] rotated quaternion [w,x,y,z]
    """
    # Convert to scipy [x,y,z,w]
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rot = R.from_quat(quat_xyzw)
    
    # Transform local axis to world frame
    world_axis = rot.apply(local_axis / (np.linalg.norm(local_axis) + 1e-12))
    
    # Build incremental rotation around that world axis
    delta_rot = R.from_rotvec(np.radians(angle_deg) * world_axis)
    new_rot = delta_rot * rot
    
    new_xyzw = new_rot.as_quat()
    return np.array([new_xyzw[3], new_xyzw[0], new_xyzw[1], new_xyzw[2]])

def compute_handle_axis_pca(
    stage,
    handle_paths: List[str],
    max_points: int = 4000,
) -> Optional[np.ndarray]:
    """
    Compute the principal elongation axis of the handle geometry using PCA.
    Returns the largest eigenvector (world frame, unit length), or None if insufficient points.
    """
    all_pts = []
    for path in handle_paths:
        pts = collect_mesh_points_world(stage, path, max_points=max_points)
        if pts.shape[0] > 0:
            all_pts.append(pts)

    if not all_pts:
        print("[WARN] compute_handle_axis_pca: no mesh points found")
        return None

    pts = np.concatenate(all_pts, axis=0)
    if pts.shape[0] < 10:
        print(f"[WARN] compute_handle_axis_pca: too few points ({pts.shape[0]})")
        return None

    centroid = pts.mean(axis=0)
    X = pts - centroid
    cov = (X.T @ X) / max(X.shape[0] - 1, 1)

    # eigenvalues ascending; LARGEST eigenvector = elongation axis
    w, v = np.linalg.eigh(cov)
    axis = v[:, 2]  # largest eigenvalue
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    print(f"[INFO] Handle PCA axis: {axis}  (eigenvalues: {w})")
    return axis

def sample_eef_poses_on_handle(
    handle_paths: List[str],
    door_normal: np.ndarray,
    handle_axis: np.ndarray,
    stage,
    eef_local_offset: np.ndarray,
    num_samples: int = 500,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Sample surface points on handle meshes and construct hand root EEF poses.

    Orientation is fully determined by two constraints:
      1. HAND_PALM_DIRECTION (local) aligns with -door_normal
         (palm faces INTO the door surface)
      2. HAND_FINGER_DIRECTION (local) aligns with handle_axis projected
         perpendicular to door_normal (fingers run along the handle)

    The hand root is offset so the EEF (not the root) lands at each surface point.

    Args:
        handle_paths:     USD paths to handle mesh prims
        door_normal:      [3] outward door surface normal (world frame)
        handle_axis:      [3] joint axis in world frame
        stage:            USD stage
        eef_local_offset: [3] offset from hand root to EEF in hand local frame
        num_samples:      number of surface points to sample

    Returns:
        List of (position [3], quaternion [w,x,y,z]) tuples — position is hand ROOT pose.
        For each sampled target point, poses are appended in order:
        1. existing "finger up" pose
        2. additional "finger down" pose with the same palm direction
    """
    # ------------------------------------------------------------------
    # 1. Collect and subsample surface points
    # ------------------------------------------------------------------
    all_pts = []
    for handle_path in handle_paths:
        pts = collect_mesh_points_world(stage, handle_path, max_points=num_samples)
        if pts.shape[0] > 0:
            all_pts.append(pts)

    if not all_pts:
        print(f"[WARN] sample_eef_poses_on_handle: no mesh points found")
        return []

    all_pts = np.concatenate(all_pts, axis=0)
    if all_pts.shape[0] > num_samples:
        idx = np.random.choice(all_pts.shape[0], size=num_samples, replace=False)
        all_pts = all_pts[idx]

    print(f"[INFO] Sampled {all_pts.shape[0]} surface points for EEF pose generation")

    # ------------------------------------------------------------------
    # 2. Build world-frame target axes
    # ------------------------------------------------------------------
    door_normal = door_normal / (np.linalg.norm(door_normal) + 1e-12)

    # HAND_PALM_DIRECTION maps to -door_normal (palm INTO surface)
    world_palm_dir = -door_normal

    # Project handle_axis perpendicular to door_normal → finger direction
    pca_axis = compute_handle_axis_pca(stage, handle_paths)
    if pca_axis is not None:
        print(f"[INFO] Using PCA handle axis: {pca_axis}")
        handle_axis = pca_axis
    else:
        print(f"[WARN] PCA failed, falling back to joint axis: {handle_axis}")
        handle_axis = handle_axis / (np.linalg.norm(handle_axis) + 1e-12)
    handle_axis_perp = handle_axis - np.dot(handle_axis, door_normal) * door_normal
    handle_axis_perp_norm = np.linalg.norm(handle_axis_perp)

    if handle_axis_perp_norm < 1e-6:
        print(f"[WARN] Handle axis parallel to door normal, using fallback finger direction")
        fallback = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(fallback, door_normal)) > 0.9:
            fallback = np.array([1.0, 0.0, 0.0])
        handle_axis_perp = fallback - np.dot(fallback, door_normal) * door_normal
        handle_axis_perp = handle_axis_perp / (np.linalg.norm(handle_axis_perp) + 1e-12)
    else:
        handle_axis_perp = handle_axis_perp / handle_axis_perp_norm

    world_finger_dir = np.cross(world_palm_dir, handle_axis_perp)
    world_finger_dir_norm = np.linalg.norm(world_finger_dir)
    if world_finger_dir_norm < 1e-6:
        # Degenerate: palm direction and handle axis are parallel
        # Fall back to handle_axis_perp itself
        print(f"[WARN] Palm dir parallel to handle axis perp, using fallback finger direction")
        world_finger_dir = handle_axis_perp
    else:
        world_finger_dir = world_finger_dir / world_finger_dir_norm

    # ------------------------------------------------------------------
    # 3. Build rotation matrix R: local frame -> world frame
    #    R @ HAND_PALM_DIRECTION   = world_palm_dir
    #    R @ HAND_FINGER_DIRECTION = world_finger_dir
    #    R = W @ L^T
    # ------------------------------------------------------------------
    palm_local   = np.array(HAND_PALM_DIRECTION,   dtype=np.float64)
    finger_local = np.array(HAND_FINGER_DIRECTION, dtype=np.float64)
    palm_local   = palm_local   / (np.linalg.norm(palm_local)   + 1e-12)
    finger_local = finger_local / (np.linalg.norm(finger_local) + 1e-12)
    third_local  = np.cross(palm_local, finger_local)
    third_local  = third_local  / (np.linalg.norm(third_local)  + 1e-12)

    third_world = np.cross(world_palm_dir, world_finger_dir)
    third_world = third_world / (np.linalg.norm(third_world) + 1e-12)

    L = np.stack([palm_local,     finger_local,    third_local],  axis=1)  # [3,3]
    W = np.stack([world_palm_dir, world_finger_dir, third_world], axis=1)  # [3,3]
    rot_mat = W @ L.T  # [3,3]

    tilt_axis = np.cross(
        np.array(HAND_FINGER_DIRECTION, dtype=np.float64),
        np.array(HAND_PALM_DIRECTION,   dtype=np.float64),
    )

    rot = R.from_matrix(rot_mat)
    quat_xyzw = rot.as_quat()  # scipy: [x,y,z,w]
    quat_base_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    quat_up_wxyz = rotate_quat_around_local_axis(
        quat_base_wxyz,
        tilt_axis,
        angle_deg=HAND_FORWARD_LEAN_DEG,
    )

    quat_down_wxyz = rotate_quat_around_local_axis(quat_base_wxyz, palm_local, angle_deg=180.0)
    quat_down_wxyz = rotate_quat_around_local_axis(
        quat_down_wxyz,
        tilt_axis,
        angle_deg=HAND_FORWARD_LEAN_DEG,
    )

    orientation_variants = []
    for quat_wxyz in (quat_up_wxyz, quat_down_wxyz):
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
        world_eef_offset = R.from_quat(quat_xyzw).as_matrix() @ eef_local_offset
        orientation_variants.append((quat_wxyz.copy(), world_eef_offset))

    poses = []
    for pt in all_pts:
        for quat_wxyz, world_eef_offset in orientation_variants:
            root_pos = pt - world_eef_offset
            poses.append((root_pos.copy(), quat_wxyz.copy()))

    print(f"[INFO] Generated {len(poses)} EEF poses ({len(orientation_variants)} orientations per point)")
    return poses

def get_handle_mesh_paths(stage, object_ref_path: str, child_link_name: str, handle_visual_indices: List[int]) -> List[str]:
    """Get USD paths to handle meshes based on visual indices."""

    asset_root = resolve_asset_root_under_ref(stage, object_ref_path)
    visuals_root = f"{asset_root}/{child_link_name}/visuals"
    if not handle_visual_indices:
        # Full link visuals root as grasp surface
        if stage.GetPrimAtPath(visuals_root).IsValid():
            print(f"[INFO] Using full visuals root as grasp surface: {visuals_root}")
            return [visuals_root]
        link_root = f"{asset_root}/{child_link_name}"
        if stage.GetPrimAtPath(link_root).IsValid():
            print(f"[INFO] Falling back to link root: {link_root}")
            return [link_root]
        print(f"[ERROR] Neither visuals root nor link root found for '{child_link_name}'")
        return []
    
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

def _build_joints_info_from_overrides(
    urdf_path: str,
    overrides: List[Tuple],
) -> List[Tuple]:
    result = []
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        joint_map = {j.get('name'): j for j in root.findall('joint')}

        for joint_name, child_link_name, visual_idx in overrides:
            joint_elem = joint_map.get(joint_name)
            if joint_elem is None:
                print(f"[WARN] Override joint '{joint_name}' not found in URDF")
                continue

            joint_type = joint_elem.get('type', 'revolute')
            axis_elem = joint_elem.find('axis')
            joint_axis = np.array([float(x) for x in axis_elem.get('xyz', '0 0 1').split()]) \
                if axis_elem is not None else np.array([0.0, 0.0, 1.0])

            limit_elem = joint_elem.find('limit')
            if limit_elem is not None:
                lower = float(limit_elem.get('lower', '0.0'))
                upper = float(limit_elem.get('upper', '0.0'))
            else:
                lower = 0.0
                upper = np.pi / 2 if joint_type == 'revolute' else 0.5

            handle_visual_indices = [visual_idx] if visual_idx is not None else []

            print(f"[INFO] Override: joint='{joint_name}', link='{child_link_name}', "
                  f"visual_idx={visual_idx}, type={joint_type}, limits=({lower:.3f}, {upper:.3f})")
            result.append((joint_name, joint_type, child_link_name, lower, upper, joint_axis, handle_visual_indices))

    except Exception as e:
        print(f"[ERROR] _build_joints_info_from_overrides failed: {e}")

    return result


def generate_and_filter_grasps_for_joint(
    stage,
    joint_name: str,
    child_link_name: str,
    joint_axis: np.ndarray,
    handle_visual_indices: List[int],
    object_wrapper_path: str,
    object_ref_path: str,
    num_samples: int = 500,
) -> list:
    """Generate grasp poses on handle meshes using hand EEF offset, then filter by door orientation."""
    print(f"\n[INFO] Generating grasps for {joint_name} -> {child_link_name}")
    print(f"[INFO] Handle visual indices: {handle_visual_indices}")

    handle_paths = get_handle_mesh_paths(stage, object_ref_path, child_link_name, handle_visual_indices)
    asset_root_dbg = resolve_asset_root_under_ref(stage, object_ref_path)
    print(f"[DEBUG] Resolved asset root under ref: {asset_root_dbg}")

    if not handle_paths:
        print(f"[ERROR] No handle meshes found for {child_link_name}")
        return []

    # ------------------------------------------------------------------
    # Compute door normal FIRST (needed for both pose generation and filtering)
    # ------------------------------------------------------------------
    door_center = None
    
    joint_prim = find_usd_joint_prim(stage, object_ref_path, joint_name)
    if joint_prim is None:
        print(f"[ERROR] Could not find USD joint")
        return []
    
    joint_params = get_joint_world_parameters(stage, joint_prim)
    if joint_params is None:
        print(f"[ERROR] Could not extract joint parameters")
        return []

    if not USE_GEOM_DOOR_NORMAL:
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
        door_normal = compute_door_normal_from_joint(stage, joint_prim, joint_params)

    if door_normal is None:
        print(f"[ERROR] Could not compute door normal for {joint_name}, skipping")
        return []

    if door_center is None:
        link_path = f"{object_ref_path}/{child_link_name}"
        door_center = get_link_local_bbox_center(stage, link_path)

    # ------------------------------------------------------------------
    # Get EEF local offset from hand USD
    # ------------------------------------------------------------------
    eef_local_offset = get_hand_eef_local_offset(stage, GRIPPER_REF_PATH)
    if eef_local_offset is None:
        print(f"[ERROR] Could not compute EEF local offset for {joint_name}, skipping")
        return []

    # ------------------------------------------------------------------
    # Sample EEF poses — orientation fixed by door_normal + joint_axis
    # ------------------------------------------------------------------
    all_grasp_poses = sample_eef_poses_on_handle(
        handle_paths,
        door_normal,
        handle_axis=joint_axis,
        stage=stage,
        eef_local_offset=eef_local_offset,
        num_samples=num_samples,
    )
    print(f"[INFO] Total raw poses from all handles: {len(all_grasp_poses)}")

    # ------------------------------------------------------------------
    # Filter by door orientation
    # ------------------------------------------------------------------
    # filtered_poses = filter_grasps_by_door_orientation(
    #     all_grasp_poses,
    #     door_normal,
    #     dot_threshold=DOT_THRESHOLD,
    #     gripper_approach_local=HAND_PALM_DIRECTION,
    # )

    return all_grasp_poses

def generate_grasps_for_all_joints(stage,
                                   dataset_dir: str,
                                   object_wrapper_path: str,
                                   object_ref_path: str,
                                   gripper_ref_path: str,
                                   num_samples_per_joint: int = 500) -> Dict[str, Dict]:
    """Generate filtered grasp poses for all articulated joints."""
    urdf_path = os.path.join(dataset_dir, 'mobility.urdf')
    
    if not os.path.exists(urdf_path):
        print(f"[ERROR] URDF not found at {urdf_path}")
        return {}
    
    joints_info = find_revolute_prismatic_joints_in_urdf(urdf_path)
    obj_id = Path(dataset_dir).name
    if not joints_info:
        print(f"[WARN] No joints with handles found via URDF for '{obj_id}'.")
        overrides = MANUAL_HANDLE_OVERRIDES.get(obj_id, [])
        if not overrides:
            print(f"[ERROR] No manual override for '{obj_id}' either. Skipping.")
            return {}
        print(f"[INFO] Using manual handle override for '{obj_id}': {overrides}")
        joints_info = _build_joints_info_from_overrides(urdf_path, overrides)
        if not joints_info:
            print(f"[ERROR] Could not build joint info from overrides for '{obj_id}'. Skipping.")
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
            joint_name,
            child_link_name,
            joint_axis,
            handle_visual_indices,
            object_wrapper_path,
            object_ref_path,
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

def slerp(q1, q2, t):
    q = (1.0 - t) * q1 + t * q2
    return q / np.linalg.norm(q)

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

    world_z = rot.apply(np.array(HAND_PALM_DIRECTION, dtype=np.float64))

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
    restitution: float = 0.0
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
    
    gripper_ref_prim = add_reference_to_stage(str(HAND_USD), GRIPPER_REF_PATH)
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
    num_prims_with_material_hand, hand_material_instanceable_paths = create_and_bind_high_friction_material(
        stage,
        GRIPPER_REF_PATH,
        static_friction=4.0,
        dynamic_friction=4.0,
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
    
    print(f"\n[INFO] Generating grasps...")
    
    all_grasps = generate_grasps_for_all_joints(
        stage,
        str(dataset_dir),
        OBJECT_WRAPPER_PATH,
        OBJECT_REF_PATH,
        GRIPPER_REF_PATH,
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
            UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr().Set("convexDecomposition")
        
        # Always DISABLE the collider — we re-enable per-mesh only during its query
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)

    print(f"[INFO] Prepared {len(gripper_mesh_paths)} gripper mesh collider(s) (disabled)")
    await omni.kit.app.get_app().next_update_async()
    
    
    #Ensure colliders + convex decomp for the object
    print(f"[INFO] Ensuring object colliders with convex decomposition...")
    collider_instanceable_paths = ensure_object_colliders(stage, OBJECT_REF_PATH)
    await omni.kit.app.get_app().next_update_async()
    # physx_iface = get_physx_interface()
    # sq_iface = get_physx_scene_query_interface()

    # def _overlap_at_pose(pos_np: np.ndarray, quat_np: np.ndarray) -> bool:
    #     """
    #     Place gripper at pose, then for each mesh, query
    #     overlap_shape_any. Return True if ANY mesh overlaps.
        
    #     overlap_shape_any returns True if the shape itself has a collider (self-hit)
    #     OR overlaps another collider. By querying each mesh that has collider disabled,
    #     the only possible hits are against other object colliders.
    #     """
    #     set_gripper_world_pose(stage, GRIPPER_WRAPPER_PATH, pos_np, quat_np)
    #     physx_iface.force_load_physics_from_usd()

    #     for mesh_path in gripper_mesh_paths:
    #         mesh_prim = stage.GetPrimAtPath(mesh_path)
    #         if not mesh_prim.IsValid():
    #             continue

    #         enc0, enc1 = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(mesh_path))
    #         hit = sq_iface.overlap_shape_any(enc0, enc1)

    #         if hit:
    #             return True

    #     return False

    # # 3. Filter each joint's grasps
    # filtered_all_grasps: Dict[str, Dict] = {}

    # for joint_name, joint_data in all_grasps.items():
    #     grasp_poses = joint_data["grasp_poses"]
    #     kept = []
    #     rejected = 0

    #     for pos, quat in grasp_poses:
    #         # Normalise types
    #         if isinstance(pos, (Gf.Vec3d, Gf.Vec3f)):
    #             pos_np = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float64)
    #         else:
    #             pos_np = np.asarray(pos, dtype=np.float64)

    #         if isinstance(quat, (Gf.Quatd, Gf.Quatf)):
    #             w = float(quat.GetReal()); im = quat.GetImaginary()
    #             quat_np = np.array([w, float(im[0]), float(im[1]), float(im[2])], dtype=np.float64)
    #         else:
    #             quat_np = np.asarray(quat, dtype=np.float64)

    #         if _overlap_at_pose(pos_np, quat_np):
    #             rejected += 1
    #         else:
    #             kept.append((pos, quat))

    #     print(f"[INFO] Joint '{joint_name}': {len(grasp_poses)} → {len(kept)} "
    #           f"(rejected {rejected} overlapping grasps)")

    #     if kept:
    #         filtered_all_grasps[joint_name] = {**joint_data, "grasp_poses": kept}

    # all_grasps = filtered_all_grasps

    # # 4. Restore gripper colliders to their original state
    # for mesh_path, had_collision in zip(gripper_mesh_paths, gripper_had_collision):
    #     prim = stage.GetPrimAtPath(mesh_path)
    #     if not prim.IsValid():
    #         continue
    #     if had_collision:
    #         # It had a collider before — re-enable it
    #         UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(True)
    #     else:
    #         # We added the collider — remove it entirely
    #         prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
    #         prim.RemoveAPI(UsdPhysics.CollisionAPI)

    # print(f"[INFO] Restored gripper collider state on {len(gripper_mesh_paths)} mesh(es)")

    # # 5. Restore gripper instanceable state
    restore_instanceable(stage, hand_material_instanceable_paths + gripper_instanceable_paths)
    await omni.kit.app.get_app().next_update_async()
    
    # # 6. Final PhysX reload to evict any temporary colliders from the simulation
    # physx_iface.force_load_physics_from_usd()
    # await omni.kit.app.get_app().next_update_async()

    # print(f"[INFO] Overlap filtering complete. "
    #       f"Joints with valid grasps: {len(all_grasps)}")
        
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
    restore_instanceable(
        stage,
        object_material_instanceable_paths + changed_instanceable_paths + collider_instanceable_paths,
    )
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
        valid_trajectories, apporach_data_map = await physics_validation_loop(
            stage,
            all_trajectories,
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
                    bottom_center_list,
                    apporach_data_map,
                )
                print(f"[SUCCESS] Trajectory data saved to {json_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save trajectory JSON: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("[WARN] No trajectories generated; skipping physics validation")
    
    # Cleanup
        
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
    hand_open_pose: np.ndarray = None,   # [7] joint angles at step-back
    hand_close_pose: np.ndarray = None,  # [7] joint angles at grasp
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute step-back position along palm direction, then linearly interpolate
    positions and slerp quaternions from step-back pose to grasp pose.

    Returns:
        positions:   [N, 3]  world positions
        quaternions: [N, 4]  slerped quaternions [w,x,y,z]
        joint_angles:[N, 7]  linearly interpolated right hand joint angles
    """
    grasp_position  = np.asarray(grasp_position,  dtype=np.float64).reshape(3)
    grasp_quaternion = np.asarray(grasp_quaternion, dtype=np.float64).reshape(4)

    # --- Step back along palm direction (local Y = HAND_PALM_DIRECTION) ---
    back_pos, _ = offset_pose_along_local_z(
        grasp_position, grasp_quaternion, -approach_distance
    )

    # At the step-back pose we want the hand OPEN,
    # at the grasp pose we want it in pre-grasp (still open here,
    # closing happens in the separate close phase).
    # So both endpoints are open — but the signature lets you pass
    # any two joint configs if you want a different interpolation.
    if hand_open_pose is None:
        hand_open_pose  = RIGHT_HAND_OPEN_T.numpy()
    if hand_close_pose is None:
        hand_close_pose = RIGHT_HAND_OPEN_T.numpy()  # still open at grasp point

    N = move_steps + 1
    positions    = np.zeros((N, 3),  dtype=np.float64)
    quaternions  = np.zeros((N, 4),  dtype=np.float64)
    joint_angles = np.zeros((N, 7),  dtype=np.float64)

    # Normalise quaternions for slerp
    q_start = grasp_quaternion / (np.linalg.norm(grasp_quaternion) + 1e-12)  # same orientation at back
    q_end   = grasp_quaternion / (np.linalg.norm(grasp_quaternion) + 1e-12)  # same orientation at grasp
    # NOTE: orientation is constant during approach (palm always faces door).
    # If you ever want to slerp from a different start orientation, change q_start.

    for i in range(N):
        alpha = i / float(move_steps) if move_steps > 0 else 1.0

        # --- Position: linear interpolation ---
        positions[i] = back_pos * (1.0 - alpha) + grasp_position * alpha

        # --- Orientation: slerp ---
        quaternions[i] = slerp(q_start, q_end, alpha)

        # --- Joint angles: piecewise-linear close schedule during approach ---
        # Reach 70% of the close pose over the first 90% of the approach,
        # then finish the remaining 30% in the last 10% of the motion.
        if alpha <= 0.9:
            joint_alpha = alpha * (0.7 / 0.9)
        else:
            joint_alpha = 0.7 + (alpha - 0.9) * (0.3 / 0.1)
        joint_alpha = min(max(joint_alpha, 0.0), 1.0)
        joint_angles[i] = hand_open_pose * (1.0 - joint_alpha) + hand_close_pose * joint_alpha

    return positions, quaternions, joint_angles

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
    env_failed = [False] * len(batch)
    # Step 3: Compute approach trajectories for all envs
    all_approach_data = []
    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        grasp_pos  = np.asarray(traj["grasp_position"],  dtype=np.float64)
        grasp_quat = np.asarray(traj["grasp_quaternion"], dtype=np.float64)

        approach_positions, approach_quats, approach_joints = compute_stepback_and_approach_poses(
            grasp_pos,
            grasp_quat,
            approach_distance=APPROACH_DISTANCE,
            move_steps=MOVE_STEPS,
            hand_open_pose=RIGHT_HAND_OPEN_T.numpy(),
            hand_close_pose=hand_pose_to_tensor(RIGHT_HAND_POSES[HAND_CLOSE_CONFIG]).numpy(),  # still open when arriving at grasp
        )

        all_approach_data.append({
            'positions':    approach_positions,   # [N, 3]
            'quats':        approach_quats,       # [N, 4]
            'joint_angles': approach_joints,      # [N, 7]
        })
    
    approach_data_by_orig_idx = {}
    for env_idx, traj in enumerate(batch):
        orig_idx = traj.get("original_index", -1)
        approach_data_by_orig_idx[orig_idx] = all_approach_data[env_idx]
    # Step 4: Execute approach phase in parallel (with displacement checks)
    for step_idx in range(MOVE_STEPS + 1):
        for env_idx in range(len(batch)):
            gripper_wrapper_path = grip_wrap(env_idx)
            gripper_ref_path     = grip_ref(env_idx)

            pos           = all_approach_data[env_idx]['positions'][step_idx]
            quat          = all_approach_data[env_idx]['quats'][step_idx]
            joint_targets = all_approach_data[env_idx]['joint_angles'][step_idx]  # [7]

            # --- Wrist pose ---
            set_gripper_world_pose(stage, gripper_wrapper_path, pos, quat)

            # --- Drive all 7 right hand joints to interpolated open angles ---
            for joint_name, target_angle in zip(RIGHT_HAND_JOINT_NAMES, joint_targets):
                joint_path = f"{gripper_ref_path}/joints/{joint_name}"
                set_usd_joint_drive_target(
                    stage, joint_path, float(np.degrees(target_angle)), drive_kind="angular"
                )

        await step_simulation(1)
            
    # Step 5: Close gripper phase (parallel) — interpolate open → chosen close config
    close_joints = hand_pose_to_tensor(RIGHT_HAND_POSES[HAND_CLOSE_CONFIG]).numpy()  # [7]

    # for close_step in range(CLOSE_STEPS):
    #     alpha = close_step / float(CLOSE_STEPS - 1) if CLOSE_STEPS > 1 else 1.0
    #     current_joints = RIGHT_HAND_OPEN_T.numpy() * (1.0 - alpha) + close_joints * alpha

    #     for env_idx in range(len(batch)):
    #         gripper_ref_path = grip_ref(env_idx)
    #         for joint_name, target_angle in zip(RIGHT_HAND_JOINT_NAMES, current_joints):
    #             joint_path = f"{gripper_ref_path}/joints/{joint_name}"
    #             set_usd_joint_drive_target(
    #                 stage, joint_path, float(target_angle), drive_kind="angular"
    #             )

    #     await step_simulation(1)

    # Step 6: Hold phase — keep hand fully closed at chosen config
    for _ in range(HOLD_STEPS):
        for env_idx in range(len(batch)):
            gripper_ref_path = grip_ref(env_idx)
            for joint_name, target_angle in zip(RIGHT_HAND_JOINT_NAMES, close_joints):
                joint_path = f"{gripper_ref_path}/joints/{joint_name}"
                set_usd_joint_drive_target(
                    stage, joint_path, float(np.degrees(target_angle)), drive_kind="angular"
                )

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
                continue

            gripper_wrapper_path = grip_wrap(env_idx)
            gripper_ref_path = grip_ref(env_idx)

            pos  = traj_positions[traj_step]
            quat = traj_orientations[traj_step]
            set_gripper_world_pose(stage, gripper_wrapper_path, pos, quat)

            # Keep hand fully closed at chosen config throughout trajectory
            for joint_name, target_angle in zip(RIGHT_HAND_JOINT_NAMES, close_joints):
                joint_path = f"{gripper_ref_path}/joints/{joint_name}"
                set_usd_joint_drive_target(
                    stage, joint_path, float(np.degrees(target_angle)), drive_kind="angular"
                )

        await step_simulation(TRAJECTORY_SIM_STEPS_PER_WAYPOINT)
        
        for env_idx in range(len(batch)):

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
    
    # Step 8: Measure final joint positions and determine success
    results = []        
    
    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        original_idx = traj.get('original_index', -1)  # We'll add this in the batching step
        
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

        joint_motion = traj.get("joint_motion", {})
        upper_limit = joint_motion.get("upper_limit", None)
        lower_limit = joint_motion.get("lower_limit", None)
        exceeds_upper = (upper_limit is not None) and (final_joint_pos > upper_limit)
        exceeds_lower = (lower_limit is not None) and (final_joint_pos < lower_limit)
        out_of_bounds = exceeds_upper or exceeds_lower

        is_valid = (actual_displacement >= required_displacement) and (not out_of_bounds)

        if joint_type == "revolute":
            if exceeds_upper:
                extra = f" [exceeds upper {np.degrees(upper_limit):.1f}°]"
            elif exceeds_lower:
                extra = f" [below lower {np.degrees(lower_limit):.1f}°, physics explosion?]"
            else:
                extra = ""
            print(f"[BATCH {batch_index}] Env {env_idx}: "
                  f"Initial={np.degrees(initial_joint_pos):.1f}°, "
                  f"Final={np.degrees(final_joint_pos):.1f}°, "
                  f"Valid={is_valid}{extra}")
        else:
            if exceeds_upper:
                extra = f" [exceeds upper {upper_limit:.4f}m]"
            elif exceeds_lower:
                extra = f" [below lower {lower_limit:.4f}m, physics explosion?]"
            else:
                extra = ""
            print(f"[BATCH {batch_index}] Env {env_idx}: "
                  f"Initial={initial_joint_pos:.4f}m, "
                  f"Final={final_joint_pos:.4f}m, "
                  f"Valid={is_valid}{extra}")
        
        results.append((original_idx, is_valid))
    
    return results, approach_data_by_orig_idx

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
) -> List[Dict]:
    """
    Run batched parallel physics validation across all cloned environments.
    """
    if not trajectories:
        return [], {}
    
    # Add original index to each trajectory for tracking
    for idx, traj in enumerate(trajectories):
        traj['original_index'] = idx
    
    # Create batches
    batches = create_trajectory_batches(trajectories, NUM_COPIES)
    
    # Track results
    all_results = {}  # {original_index: success_bool}
    all_approach_data_map: Dict[int, Dict] = {}
    
    # Process each batch
    num_batches = len(batches)
    for batch_idx, batch in enumerate(batches):
        print(f"\n[INFO] Processing batch {batch_idx + 1}/{num_batches} ({len(batch)} trajectories)...")
        results, approach_data = await validate_batch_parallel(stage, batch, batch_idx)
        
        for orig_idx, success in results:
            all_results[orig_idx] = success
        all_approach_data_map.update(approach_data)
        
    # Filter to valid trajectories
    valid = [traj for traj in trajectories if all_results.get(traj['original_index'], False)]
    
    print(f"\n[INFO] Batched validation: {len(valid)}/{len(trajectories)} trajectories passed")
    return valid, all_approach_data_map

if __name__ == "__main__":
    main()
