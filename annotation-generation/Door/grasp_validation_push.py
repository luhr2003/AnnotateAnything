import asyncio
import json
import os
from pathlib import Path
import time
from typing import List, Tuple, Optional, Dict

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

# Start Kit before importing NumPy-heavy third-party packages so Isaac Sim can
# establish its own Python paths first.
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from omni.isaac.core import World
from omni.isaac.core.utils import stage as stage_utils
from omni.isaac.core.prims import XFormPrim
from pxr import Gf, UsdGeom, UsdShade, Sdf, UsdPhysics, Usd, PhysxSchema, UsdLux, PhysicsSchemaTools
import omni.usd
import omni.kit.app
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.cloner import GridCloner 
from omni.timeline import get_timeline_interface
timeline = get_timeline_interface()
from omni.physx import get_physx_scene_query_interface, get_physx_interface

# Constants
APPROACH_DISTANCE = 0.20  # Distance to step back before approaching (m)
MOVE_STEPS = 200          # Steps for linear interpolation
CLOSE_STEPS = 64         # Steps for closing gripper
HOLD_STEPS = 60          # Extra physics steps after closing to let contacts settle
CONTACTSLOPCOEFF = 0.1    # PhysX contact slop coefficient
TRAJECTORY_SIM_STEPS_PER_WAYPOINT = 3  # Physics steps per trajectory waypoint
NUM_COPIES = 20        # Number of copies to create in cloner
CLONE_SPACING = 25.0    # Spacing between clones in cloner grid (m)
TRAJ_OVERSHOOT_CHECKS   = 5
NUM_SAMPLES_PER_JOINT = 200
GRIPPER_PARK_POSITION = np.array([0.0, 0.0, 3.0])  # Safe out-of-scene position for reset


# Gripper local axes
HAND_PALM_DIRECTION = (0, -1, 0)
HAND_FINGER_DIRECTION = (1, 0, 0)
HAND_FORWARD_LEAN_DEG = 15.0

# Hand joint drive parameters
JOINT_STIFFNESS = 20.0
JOINT_DAMPING = 2.0
JOINT_MAX_FORCE = 300.0
JOINT_ARMATURE = 0.01
JOINT_VELOCITY_LIMIT = 100.0
TARGET_OPEN_RATIO = 0.7
HANDLE_ROTATE_RATIO = 1.0
PRIMARY_ROTATE_ACCEPT_RATIO = 1.0
SECONDARY_ROTATE_ACCEPT_RATIO = 0.6
JOINT_SUCCESS_THRESHOLD = 0.90
OVERSHOOT_TOLERANCE_DEG = 2.0   
OVERSHOOT_TOLERANCE_M   = 0.01  
DOOR_OPEN_OVERSHOOT_REJECT_DEG = 10.0
HAND_RUNTIME_COLLISION_ALLOW_TOKENS = ("thumb", "index_1", "middle_1")
HAND_RUNTIME_COLLISION_BLOCK_TOKENS = ("palm", "eef")

# Generated-door naming convention from Doorman/scripts/generate_door_standalone.py
GENERATED_DOOR_ROOT_LINK_NAME = "root"
GENERATED_DOOR_PANEL_LINK_NAME = "door_panel"
GENERATED_DOOR_HANDLE_LINK_NAME = "door_handle"
GENERATED_DOOR_GRASP_TARGET_NAME = "grasp_target"
GENERATED_DOOR_PUSH_JOINT_NAME = "hinge_joint"
GENERATED_DOOR_ROTATE_JOINT_NAME = "handle_joint"
HAND_EEF_LINK_NAME = "eef"
HAND_EEF2_LINK_NAME = "eef2"

GENERATED_DOOR_PRIMARY_GRASP_MODE = "palm_in_finger_up"
GENERATED_DOOR_SECONDARY_GRASP_MODE = "palm_out_finger_down"


def get_generated_door_grasp_mode_specs() -> Tuple[Dict[str, object], ...]:
    return (
        {
            "name": GENERATED_DOOR_PRIMARY_GRASP_MODE,
            "eef_link_name": HAND_EEF_LINK_NAME,
            "palm_sign": 1.0,
            "vertical_sign": 1.0,
            "lean_sign": 1.0,
            "approach_local_direction": np.array(HAND_PALM_DIRECTION, dtype=np.float64),
            "close_during_approach": True,
            "rotate_accept_ratio": PRIMARY_ROTATE_ACCEPT_RATIO,
        },
        {
            "name": GENERATED_DOOR_SECONDARY_GRASP_MODE,
            "eef_link_name": HAND_EEF2_LINK_NAME,
            "palm_sign": -1.0,
            "vertical_sign": -1.0,
            "lean_sign": -1.0,
            "approach_local_direction": np.array(HAND_FINGER_DIRECTION, dtype=np.float64),
            "close_during_approach": False,
            "rotate_accept_ratio": SECONDARY_ROTATE_ACCEPT_RATIO,
        },
    )


def _get_prim_world_pose(prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray] | None:
    if prim is None or not prim.IsValid():
        return None
    world_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = world_transform.ExtractTranslation()
    rotation = world_transform.ExtractRotationQuat()
    position = np.array([translation[0], translation[1], translation[2]], dtype=float)
    quat_wxyz = np.array([rotation.GetReal(), *rotation.GetImaginary()], dtype=float)
    return position, quat_wxyz


def get_hand_eef_local_offset(
    stage,
    hand_ref_path: str,
    eef_link_name: str = HAND_EEF_LINK_NAME,
) -> Optional[np.ndarray]:
    """Return the EEF position expressed in the hand root's local frame."""
    hand_root_prim = stage.GetPrimAtPath(hand_ref_path)
    if not hand_root_prim.IsValid():
        print(f"[WARN] Hand root not found at {hand_ref_path}")
        return None

    eef_prim = stage.GetPrimAtPath(f"{hand_ref_path}/Xform/{eef_link_name}")
    if not eef_prim.IsValid():
        print(f"[WARN] Hand EEF link not found at {hand_ref_path}/Xform/{eef_link_name}")
        return None

    hand_root_xf = UsdGeom.Xformable(hand_root_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    eef_xf = UsdGeom.Xformable(eef_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    eef_world = eef_xf.ExtractTranslation()
    eef_local = hand_root_xf.GetInverse().Transform(eef_world)
    local_offset = np.array([float(eef_local[0]), float(eef_local[1]), float(eef_local[2])], dtype=np.float64)
    print(f"[INFO] Hand EEF local offset ({eef_link_name} in {hand_ref_path}): {np.round(local_offset, 5)}")
    return local_offset


def compute_hand_base_position_from_eef_target(
    target_eef_position: np.ndarray,
    hand_quaternion_wxyz: np.ndarray,
    eef_local_offset: np.ndarray,
) -> np.ndarray:
    """Convert a desired EEF world position into the hand wrapper/root world position."""
    target_eef_position = np.asarray(target_eef_position, dtype=np.float64).reshape(3)
    hand_quaternion_wxyz = np.asarray(hand_quaternion_wxyz, dtype=np.float64).reshape(4)
    eef_local_offset = np.asarray(eef_local_offset, dtype=np.float64).reshape(3)
    quat_xyzw = np.array(
        [
            hand_quaternion_wxyz[1],
            hand_quaternion_wxyz[2],
            hand_quaternion_wxyz[3],
            hand_quaternion_wxyz[0],
        ],
        dtype=np.float64,
    )
    rot = R.from_quat(quat_xyzw)
    eef_world_offset = rot.apply(eef_local_offset)
    return target_eef_position - eef_world_offset


def convert_eef_positions_to_hand_base_positions(
    eef_positions: np.ndarray,
    hand_quaternions_wxyz: np.ndarray,
    eef_local_offset: np.ndarray,
) -> np.ndarray:
    eef_positions = np.asarray(eef_positions, dtype=np.float64)
    hand_quaternions_wxyz = np.asarray(hand_quaternions_wxyz, dtype=np.float64)
    if eef_positions.ndim == 1:
        return compute_hand_base_position_from_eef_target(
            eef_positions, hand_quaternions_wxyz, eef_local_offset
        )

    base_positions = np.zeros_like(eef_positions, dtype=np.float64)
    for i in range(eef_positions.shape[0]):
        base_positions[i] = compute_hand_base_position_from_eef_target(
            eef_positions[i], hand_quaternions_wxyz[i], eef_local_offset
        )
    return base_positions


def get_cached_hand_eef_local_offset(
    stage,
    hand_ref_path: str,
    eef_link_name: str,
    cache: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    if eef_link_name in cache:
        return cache[eef_link_name]

    offset = get_hand_eef_local_offset(stage, hand_ref_path, eef_link_name=eef_link_name)
    if offset is None:
        return None
    cache[eef_link_name] = offset
    return offset


def rotate_quat_around_local_axis(
    quat_wxyz: np.ndarray,
    local_axis: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    local_axis = np.asarray(local_axis, dtype=np.float64).reshape(3)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    rot = R.from_quat(quat_xyzw)

    world_axis = rot.apply(local_axis / (np.linalg.norm(local_axis) + 1e-12))
    delta_rot = R.from_rotvec(np.radians(angle_deg) * world_axis)
    new_rot = delta_rot * rot

    new_xyzw = new_rot.as_quat()
    return np.array([new_xyzw[3], new_xyzw[0], new_xyzw[1], new_xyzw[2]], dtype=np.float64)

# =======================
# Processing Mode Configuration
# =======================
PROCESSING_MODE = "single"  # Options: "single" or "dataset"

# For single object mode
_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _THIS_DIR.parent

def _path_from_env(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default))).expanduser()

SINGLE_OBJECT_USD = _path_from_env("DOOR_PUSH_OBJECT_USD", _THIS_DIR / "data" / "door_push" / "Object.usd")

INPUT_DATASET_PATH = _path_from_env("DOOR_PUSH_DATASET_PATH", _THIS_DIR / "data")
HAND_USD = _path_from_env(
    "DOOR_PUSH_HAND_USD",
    _WORKSPACE_ROOT / "Articulation" / "Dex_hand" / "dex3_1_right.usd",
)
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
    return f"{env_path(i)}/Dex3"

def grip_ref(i: int) -> str:
    return f"{env_path(i)}/Dex3/ref"

OBJECT_WRAPPER_PATH = obj_wrap(0)
OBJECT_REF_PATH = obj_ref(0)
GRIPPER_WRAPPER_PATH = grip_wrap(0)
GRIPPER_REF_PATH = grip_ref(0)

# =======================
# Hand Closing HardCode
# =======================
OPPOSITE_POSE = -1 #-1 for opposite and 1 for normal
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

_HAND_JOINT_PATH_CACHE = {}  # {(stage_id, container_path): {joint_name: joint_path}}

def resolve_hand_joint_paths(stage, container_path: str) -> Dict[str, str]:
    """Walk the prim tree to find actual joint prim paths for RIGHT_HAND_JOINT_NAMES.

    Joints in the new hand are distributed under link prims rather than being
    in a flat /joints/ directory, so this function discovers them dynamically
    and caches the result.
    """
    cache_key = (stage.GetRootLayer().identifier, container_path)
    cached = _HAND_JOINT_PATH_CACHE.get(cache_key)
    if cached is not None:
        if all(stage.GetPrimAtPath(p).IsValid() for p in cached.values()):
            return cached

    container_prim = stage.GetPrimAtPath(container_path)
    if not container_prim.IsValid():
        print(f"[WARN] resolve_hand_joint_paths: invalid container {container_path}")
        return {}

    joint_paths: Dict[str, str] = {}
    for prim in Usd.PrimRange(container_prim):
        if not prim.IsValid():
            continue
        name = prim.GetName()
        if prim.IsA(UsdPhysics.Joint) and name in RIGHT_HAND_JOINT_NAMES and name not in joint_paths:
            joint_paths[name] = prim.GetPath().pathString

    missing = [n for n in RIGHT_HAND_JOINT_NAMES if n not in joint_paths]
    if missing:
        print(f"[WARN] resolve_hand_joint_paths: could not find joints under {container_path}: {missing}")

    _HAND_JOINT_PATH_CACHE[cache_key] = joint_paths
    return joint_paths


def configure_hand_joint_drives(stage, container_path: str) -> int:
    """Apply PD-gain and limit parameters to all joints under container_path."""
    container_prim = stage.GetPrimAtPath(container_path)
    if not container_prim.IsValid():
        print(f"[WARN] configure_hand_joint_drives: invalid container {container_path}")
        return 0

    joint_count = 0
    for prim in Usd.PrimRange(container_prim):
        if not prim.IsValid():
            continue

        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive_kind = "angular"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            drive_kind = "linear"
        else:
            continue

        try:
            drive = UsdPhysics.DriveAPI.Apply(prim, drive_kind)

            st = drive.GetStiffnessAttr() or drive.CreateStiffnessAttr()
            st.Set(float(JOINT_STIFFNESS))

            dm = drive.GetDampingAttr() or drive.CreateDampingAttr()
            dm.Set(float(JOINT_DAMPING))

            mf = drive.GetMaxForceAttr() or drive.CreateMaxForceAttr()
            mf.Set(float(JOINT_MAX_FORCE))

            tp = drive.GetTargetPositionAttr() or drive.CreateTargetPositionAttr()
            tp.Set(0.0)

            tv = drive.GetTargetVelocityAttr() or drive.CreateTargetVelocityAttr()
            tv.Set(0.0)

            if not prim.HasAPI(PhysxSchema.PhysxJointAPI):
                PhysxSchema.PhysxJointAPI.Apply(prim)
            physx_joint = PhysxSchema.PhysxJointAPI(prim)
            mv = physx_joint.GetMaxJointVelocityAttr() or physx_joint.CreateMaxJointVelocityAttr()
            mv.Set(float(JOINT_VELOCITY_LIMIT))
            arm = physx_joint.GetArmatureAttr() or physx_joint.CreateArmatureAttr()
            arm.Set(float(JOINT_ARMATURE))

            joint_count += 1
        except Exception as e:
            print(f"[WARN] configure_hand_joint_drives: {prim.GetPath()}: {e}")

    print(f"[INFO] Configured drive params for {joint_count} hand joints under {container_path}")
    return joint_count


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
    For all meshes under object_ref_path that already have CollisionAPI,
    update the approximation type to convexDecomposition.
    Does NOT add CollisionAPI to meshes that don't already have it.
    Disables instanceable on any prim that needs to be modified.

    Returns:
        List of prim paths that had instanceable disabled (to exclude from restore).
    """
    root = stage.GetPrimAtPath(object_ref_path)
    if not root.IsValid():
        print(f"[ERROR] ensure_object_colliders: invalid path {object_ref_path}")
        return []

    instanceable_disabled: List[str] = []
    meshes_updated = 0

    for prim in Usd.PrimRange(root):
        if not prim.IsValid():
            continue

        # Only act on prims that already have CollisionAPI (regardless of prim type)
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
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

        # Update approximation type to convexDecomposition
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr().Set("convexDecomposition")
        meshes_updated += 1

    print(f"[INFO] ensure_object_colliders under {object_ref_path}:")
    print(f"  - Existing colliders updated to convexDecomposition: {meshes_updated}")
    print(f"  - Instanceable disabled:                             {len(instanceable_disabled)}")
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
    """Restore instanceable=True on prims changed by disable_instanceable_for_grasp_generation."""
    if not changed_paths:
        print("[INFO] No instanceable prims to restore")
        return

    restored = 0
    for p in changed_paths:
        prim = stage.GetPrimAtPath(p)
        if not prim.IsValid():
            continue
        try:
            prim.SetInstanceable(True)
            restored += 1
        except Exception as e:
            print(f"[WARN] Failed to restore instanceable on {p}: {e}")

    print(f"[INFO] Restored instanceable on {restored}/{len(changed_paths)} prim(s)")

def set_object_gravity_enabled(stage, object_ref_path: str, enabled: bool = True):
    """Enable/disable gravity for all rigid bodies inside the referenced articulated object.

    PhysX schema uses `disableGravity=True` to mean gravity OFF.
    """
    apply_object_physx_overrides(stage, object_ref_path, disable_gravity=(not bool(enabled)))

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
        "phases": ["approach", "rotate", "push"],
        "trajectories": {}
    }

    if trajectories:
        first_traj = trajectories[0]
        joint_type  = first_traj.get("joint_type", "")
        push_target = float(first_traj.get("target_displacement", 0.0))
        rot_target  = float(first_traj.get("rotate_target_displacement", 0.0))
        if joint_type == "revolute":
            data["push_target_angle_deg"]   = float(np.degrees(push_target))
            data["rotate_target_angle_deg"] = float(np.degrees(rot_target))
        else:
            data["push_target_distance_m"]   = push_target
            data["rotate_target_distance_m"] = rot_target

    from collections import defaultdict
    joint_groups = defaultdict(list)
    for traj in trajectories:
        joint_name = traj.get("joint_name", "unknown")
        joint_groups[joint_name].append(traj)

    close_joints_rad = np.radians(
        hand_pose_to_tensor(RIGHT_HAND_POSES[HAND_CLOSE_CONFIG]).numpy()
    )

    def _make_waypoints(positions, orientations, joint_angles_rad=None):
        """Build list of [x,y,z,w,qx,qy,qz,j0..j6] waypoints."""
        waypoints = []
        positions    = np.asarray(positions,    dtype=np.float64)
        orientations = np.asarray(orientations, dtype=np.float64)
        for i, (pos, quat) in enumerate(zip(positions, orientations)):
            if joint_angles_rad is not None:
                joints = joint_angles_rad[i] if joint_angles_rad.ndim == 2 else joint_angles_rad
            else:
                joints = close_joints_rad
            waypoints.append([
                float(pos[0]),  float(pos[1]),  float(pos[2]),
                float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]),
                *[float(j) for j in joints],
            ])
        return waypoints

    for joint_name, joint_trajs in joint_groups.items():
        joint_dict = {}
        for idx, traj in enumerate(joint_trajs, start=1):
            orig_idx = traj.get("original_index", -1)

            # ── Approach phase (open → closed hand) ──────────────────────────
            approach = approach_data.get(orig_idx)
            approach_waypoints = []
            if approach is not None:
                app_joints_rad = np.radians(np.asarray(approach["joint_angles"], dtype=np.float64))
                approach_waypoints = _make_waypoints(
                    approach["positions"], approach["quats"], app_joints_rad
                )

            # ── Rotate phase (hand fully closed) ─────────────────────────────
            rotate_waypoints = _make_waypoints(
                traj["rotate_positions"], traj["rotate_orientations"]
            )

            # ── Push phase (hand fully closed) ────────────────────────────────
            push_waypoints = _make_waypoints(
                traj["trajectory_positions"], traj["trajectory_orientations"]
            )

            joint_dict[str(idx)] = {
                "rotate_joint": traj.get("rotate_joint_name"),
                "approach":     approach_waypoints,
                "rotate":       rotate_waypoints,
                "push":         push_waypoints,
            }

        data["trajectories"][joint_name] = joint_dict

    json_path = annotation_dir / "dex3_1_rotate_and_push_trajectory.json"
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
        generated_root_path = f"{object_ref_path}/{GENERATED_DOOR_ROOT_LINK_NAME}"
        generated_root_prim = stage.GetPrimAtPath(generated_root_path)
        if generated_root_prim.IsValid() and generated_root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            base_path = generated_root_path
            base_prim = generated_root_prim

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

# Helper: resolve internal asset root under a reference prim (e.g., to find /partnet_<hash>)
def resolve_asset_root_under_ref(stage, object_ref_path: str) -> str:
    ref_prim = stage.GetPrimAtPath(object_ref_path)
    if not ref_prim.IsValid():
        return object_ref_path

    generated_asset_root = _find_generated_doorman_asset_root(stage, object_ref_path)
    if generated_asset_root is not None:
        return generated_asset_root

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


def get_prim_custom_data(stage, prim_path: str) -> Dict:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return {}
    if prim.HasMetadata("customData"):
        data = prim.GetMetadata("customData")
        if isinstance(data, dict):
            return data
    return {}


def _is_generated_doorman_asset_root(stage, prim_path: str) -> bool:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False

    custom_data = get_prim_custom_data(stage, prim_path)
    if "doorHandleType" not in custom_data:
        return False

    required_child_names = (
        GENERATED_DOOR_ROOT_LINK_NAME,
        GENERATED_DOOR_PANEL_LINK_NAME,
        GENERATED_DOOR_HANDLE_LINK_NAME,
        GENERATED_DOOR_GRASP_TARGET_NAME,
    )
    return all(stage.GetPrimAtPath(f"{prim_path}/{name}").IsValid() for name in required_child_names)


def _find_generated_doorman_asset_root(stage, object_ref_path: str) -> Optional[str]:
    ref_prim = stage.GetPrimAtPath(object_ref_path)
    if not ref_prim.IsValid():
        return None

    if _is_generated_doorman_asset_root(stage, object_ref_path):
        return object_ref_path

    for prim in Usd.PrimRange(ref_prim):
        if not prim.IsValid():
            continue
        prim_path = prim.GetPath().pathString
        if _is_generated_doorman_asset_root(stage, prim_path):
            return prim_path

    return None


def get_generated_doorman_door_info(stage, object_ref_path: str) -> Optional[Dict]:
    """Detect a standalone generated door and return its canonical prim/joint paths."""
    asset_root = _find_generated_doorman_asset_root(stage, object_ref_path)
    if asset_root is None:
        return None

    custom_data = get_prim_custom_data(stage, asset_root)

    info = {
        "asset_root": asset_root,
        "custom_data": custom_data,
        "root_path": f"{asset_root}/{GENERATED_DOOR_ROOT_LINK_NAME}",
        "panel_path": f"{asset_root}/{GENERATED_DOOR_PANEL_LINK_NAME}",
        "handle_path": f"{asset_root}/{GENERATED_DOOR_HANDLE_LINK_NAME}",
        "grasp_target_path": f"{asset_root}/{GENERATED_DOOR_GRASP_TARGET_NAME}",
        "push_joint_path": f"{asset_root}/{GENERATED_DOOR_ROOT_LINK_NAME}/{GENERATED_DOOR_PUSH_JOINT_NAME}",
        "rotate_joint_path": f"{asset_root}/{GENERATED_DOOR_PANEL_LINK_NAME}/{GENERATED_DOOR_ROTATE_JOINT_NAME}",
        "push_joint_name": GENERATED_DOOR_PUSH_JOINT_NAME,
        "handle_link_name": GENERATED_DOOR_HANDLE_LINK_NAME,
        "handle_type": str(custom_data.get("doorHandleType", "")),
    }

    required_paths = [
        info["root_path"],
        info["panel_path"],
        info["handle_path"],
        info["grasp_target_path"],
        info["push_joint_path"],
    ]
    if not all(stage.GetPrimAtPath(path).IsValid() for path in required_paths):
        return None

    info["rotate_joint_name"] = (
        GENERATED_DOOR_ROTATE_JOINT_NAME
        if info["handle_type"] == "lever" and stage.GetPrimAtPath(info["rotate_joint_path"]).IsValid()
        else None
    )
    return info


def get_prim_world_pose_wxyz(stage, prim_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None

    xformable = UsdGeom.Xformable(prim)
    world_xf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = world_xf.ExtractTranslation()
    quat = world_xf.ExtractRotationQuat()

    pos = np.array([float(translation[0]), float(translation[1]), float(translation[2])], dtype=np.float64)
    quat_wxyz = np.array(
        [
            float(quat.GetReal()),
            float(quat.GetImaginary()[0]),
            float(quat.GetImaginary()[1]),
            float(quat.GetImaginary()[2]),
        ],
        dtype=np.float64,
    )
    return pos, quat_wxyz


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


def get_handle_rotate_axis_from_body1(
    stage,
    joint_prim: Usd.Prim,
    draw_debug: bool = True,
    line_length: float = 0.3,
    line_width: float = 4.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute the rotation axis and pivot for a handle revolute joint using
    body1's world transform (child-link side of the joint).

    The physical rotation happens:
      pivot = body1 world position (centre of the child link)
      axis  = joint-frame axis token (X/Y/Z) rotated into world space via
              body1_world_xf * localRot1

    Args:
        stage:       USD stage
        joint_prim:  the revolute joint prim for the handle rotation
        draw_debug:  if True, draw the axis as a magenta line in the viewport
        line_length: half-length of the debug line (metres)
        line_width:  pixel width of the debug line

    Returns:
        (pivot [3], axis [3]) both in world frame, or None on failure.
    """
    if not joint_prim.IsValid():
        return None

    j = UsdPhysics.Joint(joint_prim)

    # ── Resolve body1 ──────────────────────────────────────────────────────
    body1_targets = []
    try:
        rel1 = j.GetBody1Rel()
        if rel1 and rel1.IsValid():
            body1_targets = rel1.GetTargets()
    except Exception:
        pass

    if not body1_targets:
        print(f"[RotateAxis] No body1 on {joint_prim.GetPath()}")
        return None

    body1_path = body1_targets[0].pathString
    body1_prim = stage.GetPrimAtPath(body1_path)
    if not body1_prim.IsValid():
        print(f"[RotateAxis] body1 invalid: {body1_path}")
        return None

    body1_world_xf = UsdGeom.Xformable(body1_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )

    # ── Joint local frame on body1 side ────────────────────────────────────
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

    joint_frame_body1 = body1_world_xf * _make_transform_gf(
        np.array([float(lp1[0]), float(lp1[1]), float(lp1[2])], dtype=np.float64),
        lr1,
    )

    # ── Axis token → world ──────────────────────────────────────────────────
    joint_usd = UsdPhysics.RevoluteJoint(joint_prim)
    axis_attr = joint_usd.GetAxisAttr()
    axis_token = str(axis_attr.Get()) if (axis_attr and axis_attr.IsValid()) else "Z"

    axis_map = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}
    axis_local_gf = axis_map.get(axis_token, Gf.Vec3d(0, 0, 1))

    axis_world_gf = joint_frame_body1.TransformDir(axis_local_gf)
    axis_world = np.array(
        [float(axis_world_gf[0]), float(axis_world_gf[1]), float(axis_world_gf[2])],
        dtype=np.float64,
    )
    axis_world /= np.linalg.norm(axis_world) + 1e-12

    # ── Pivot = body1 world position ────────────────────────────────────────
    t = body1_world_xf.ExtractTranslation()
    pivot = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)

    print(
        f"[RotateAxis] joint={joint_prim.GetPath().name}, "
        f"axis_token={axis_token}, "
        f"axis_world={np.round(axis_world, 4)}, "
        f"pivot={np.round(pivot, 4)}"
    )

    # ── Debug draw (magenta = body1 joint axis; actual rotate axis = door normal) ─
    if draw_debug:
        try:
            from isaacsim.util.debug_draw import _debug_draw
            dd = _debug_draw.acquire_debug_draw_interface()
            p0 = pivot - axis_world * line_length
            p1 = pivot + axis_world * line_length
            dd.draw_lines(
                [(float(p0[0]), float(p0[1]), float(p0[2]))],
                [(float(p1[0]), float(p1[1]), float(p1[2]))],
                [(1.0, 0.0, 1.0, 1.0)],   # magenta
                [float(line_width)],
            )
        except Exception as e:
            print(f"[RotateAxis] debug draw failed: {e}")

    return pivot, axis_world


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

def _build_joints_info_from_generated_door(stage, object_ref_path: str) -> List[Tuple]:
    """Build joint info directly from the generated door's canonical prim layout."""
    generated_info = get_generated_doorman_door_info(stage, object_ref_path)
    if generated_info is None:
        return []

    push_joint_prim = find_usd_joint_prim(stage, object_ref_path, generated_info["push_joint_name"])
    if push_joint_prim is None:
        print(f"[ERROR] Generated door push joint not found: {generated_info['push_joint_name']}")
        return []

    push_joint_params = get_joint_world_parameters(stage, push_joint_prim)
    if push_joint_params is None:
        print(f"[ERROR] Failed to read generated door push joint parameters")
        return []

    rotate_joint_name = generated_info["rotate_joint_name"]
    if rotate_joint_name is not None:
        rotate_joint_prim = find_usd_joint_prim(stage, object_ref_path, rotate_joint_name)
        rotate_joint_params = get_joint_world_parameters(stage, rotate_joint_prim) if rotate_joint_prim else None
        if rotate_joint_params is None:
            rotate_joint_name = None
        else:
            joint_span = abs(float(rotate_joint_params["upper_limit"]) - float(rotate_joint_params["lower_limit"]))
            if joint_span < 1e-6:
                rotate_joint_name = None

    print(
        "[INFO] Detected generated door asset: "
        f"handle_type={generated_info['handle_type']}, "
        f"push_joint={generated_info['push_joint_name']}, "
        f"rotate_joint={rotate_joint_name}"
    )

    return [
        (
            rotate_joint_name,
            generated_info["push_joint_name"],
            push_joint_params["joint_type"],
            generated_info["handle_link_name"],
            float(push_joint_params["lower_limit"]),
            float(push_joint_params["upper_limit"]),
            np.asarray(push_joint_params["axis"], dtype=np.float64),
        )
    ]


def _get_generated_grasp_patch_extents(custom_data: Dict) -> Tuple[float, float]:
    handle_type = str(custom_data.get("doorHandleType", "lever"))
    handle_length = float(custom_data.get("handleLength", 0.12))
    handle_radius = float(custom_data.get("handleRadius", 0.015))

    if handle_type == "pushbar":
        long_extent = min(0.45 * handle_length, 0.10)
        short_extent = max(3.0 * handle_radius, 0.02)
    elif handle_type == "handle":
        long_extent = min(0.35 * handle_length, 0.06)
        short_extent = max(2.5 * handle_radius, 0.012)
    else:
        long_extent = min(0.35 * handle_length, 0.05)
        short_extent = max(2.0 * handle_radius, 0.01)

    return float(long_extent), float(short_extent)


def _build_gripper_quat_from_generated_target_frame(
    grasp_target_quat_wxyz: np.ndarray,
    palm_sign: float = 1.0,
    vertical_sign: float = 1.0,
    lean_sign: float = 1.0,
) -> np.ndarray:
    """Convert the authored grasp_target frame into a hand pose.

    Generated Doorman doors use grasp_target as a contact frame:
      - local +X points toward the door surface
      - local +Y follows the main handle span

    Our hand convention uses:
      - local +Y as the palm-facing direction
      - local +X as the finger span direction

    Desired grasp convention for generated doors:
      hand +Y <- +/- target +X
      hand +X <- projected +/- world Z
    """
    grasp_target_quat_wxyz = np.asarray(grasp_target_quat_wxyz, dtype=np.float64).reshape(4)
    target_quat_xyzw = np.array(
        [
            grasp_target_quat_wxyz[1],
            grasp_target_quat_wxyz[2],
            grasp_target_quat_wxyz[3],
            grasp_target_quat_wxyz[0],
        ],
        dtype=np.float64,
    )
    target_rot = R.from_quat(target_quat_xyzw)

    palm_world = float(palm_sign) * target_rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float64))
    palm_world = palm_world / (np.linalg.norm(palm_world) + 1e-12)

    world_vertical = np.array([0.0, 0.0, float(vertical_sign)], dtype=np.float64)
    finger_world = world_vertical - np.dot(world_vertical, palm_world) * palm_world
    if np.linalg.norm(finger_world) < 1e-6:
        # Fallback near singular cases: use the target's handle-span axis.
        finger_world = target_rot.apply(np.array([0.0, 1.0, 0.0], dtype=np.float64))
        if np.dot(finger_world, world_vertical) < 0.0:
            finger_world *= -1.0
        finger_world = finger_world - np.dot(finger_world, palm_world) * palm_world
    finger_world = finger_world / (np.linalg.norm(finger_world) + 1e-12)

    hand_z_world = np.cross(finger_world, palm_world)
    hand_z_world = hand_z_world / (np.linalg.norm(hand_z_world) + 1e-12)
    palm_world = np.cross(hand_z_world, finger_world)
    palm_world = palm_world / (np.linalg.norm(palm_world) + 1e-12)

    hand_rot_world = np.column_stack([finger_world, palm_world, hand_z_world])
    hand_quat_xyzw = R.from_matrix(hand_rot_world).as_quat()
    hand_quat_wxyz = np.array(
        [
            hand_quat_xyzw[3],
            hand_quat_xyzw[0],
            hand_quat_xyzw[1],
            hand_quat_xyzw[2],
        ],
        dtype=np.float64,
    )
    tilt_axis = np.cross(
        np.array(HAND_FINGER_DIRECTION, dtype=np.float64),
        np.array(HAND_PALM_DIRECTION, dtype=np.float64),
    )
    return rotate_quat_around_local_axis(
        hand_quat_wxyz,
        tilt_axis,
        angle_deg=float(lean_sign) * HAND_FORWARD_LEAN_DEG,
    )


def _sample_grasp_poses_around_generated_target(
    stage,
    generated_info: Dict,
    num_samples: int,
    mode_specs: Optional[Tuple[Dict[str, object], ...]] = None,
) -> List[Dict[str, object]]:
    pose = get_prim_world_pose_wxyz(stage, generated_info["grasp_target_path"])
    if pose is None:
        return []

    grasp_pos, contact_quat = pose
    num_samples = max(1, int(num_samples))
    if mode_specs is None:
        mode_specs = get_generated_door_grasp_mode_specs()

    quat_xyzw = np.array([contact_quat[1], contact_quat[2], contact_quat[3], contact_quat[0]], dtype=np.float64)
    rot = R.from_quat(quat_xyzw)

    # Treat grasp_target local X as the approach/normal axis and sample in the local YZ tangent plane.
    tangent_u = rot.apply(np.array([0.0, 1.0, 0.0], dtype=np.float64))
    tangent_v = rot.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))

    long_extent, short_extent = _get_generated_grasp_patch_extents(generated_info["custom_data"])

    remaining = num_samples - 1
    cols = int(np.ceil(np.sqrt(remaining)))
    rows = int(np.ceil(remaining / cols))
    u_vals = np.linspace(-long_extent, long_extent, cols)
    v_vals = np.linspace(-short_extent, short_extent, rows)

    offsets = [(float(u), float(v)) for v in v_vals for u in u_vals]
    offsets.sort(key=lambda uv: (uv[0] * uv[0] + uv[1] * uv[1], abs(uv[1]), abs(uv[0])))
    offsets = [uv for uv in offsets if not (abs(uv[0]) < 1e-12 and abs(uv[1]) < 1e-12)]

    sampled_positions = [grasp_pos.astype(np.float64, copy=True)]
    for u_off, v_off in offsets[:remaining]:
        pos = grasp_pos + u_off * tangent_u + v_off * tangent_v
        sampled_positions.append(pos.astype(np.float64, copy=True))

    poses: List[Dict[str, object]] = []
    for mode_spec in mode_specs:
        grasp_quat = _build_gripper_quat_from_generated_target_frame(
            contact_quat,
            palm_sign=float(mode_spec.get("palm_sign", 1.0)),
            vertical_sign=float(mode_spec.get("vertical_sign", 1.0)),
            lean_sign=float(mode_spec.get("lean_sign", 1.0)),
        )
        for pos in sampled_positions:
            poses.append(
                {
                    "position": pos.copy(),
                    "quaternion": grasp_quat.copy(),
                    "mode_name": str(mode_spec.get("name", GENERATED_DOOR_PRIMARY_GRASP_MODE)),
                    "eef_link_name": str(mode_spec.get("eef_link_name", HAND_EEF_LINK_NAME)),
                    "approach_local_direction": np.asarray(
                        mode_spec.get("approach_local_direction", HAND_PALM_DIRECTION),
                        dtype=np.float64,
                    ).copy(),
                    "close_during_approach": bool(mode_spec.get("close_during_approach", True)),
                    "rotate_accept_ratio": float(mode_spec.get("rotate_accept_ratio", 1.0)),
                }
            )

    return poses


def generate_and_filter_grasps_for_joint(
    stage,
    joint_name: str,
    child_link_name: str,
    object_ref_path: str,
    num_samples: int = 1,
) -> list:
    """Return grasp_target-centered grasp poses from a generated Doorman door."""
    print(f"\n[INFO] Generating grasps for {joint_name} -> {child_link_name}")

    generated_info = get_generated_doorman_door_info(stage, object_ref_path)
    if generated_info is None:
        print(
            "[ERROR] This validation script now expects a generated Doorman door "
            "with canonical naming and a grasp_target prim."
        )
        return []

    grasp_poses = _sample_grasp_poses_around_generated_target(stage, generated_info, num_samples=num_samples)
    if not grasp_poses:
        print(f"[ERROR] Generated door grasp target not found: {generated_info['grasp_target_path']}")
        return []
    print(
        f"[INFO] Sampled {len(grasp_poses)} grasp pose(s) around generated door grasp target: "
        f"{generated_info['grasp_target_path']} (handle_type={generated_info['handle_type']})"
    )
    return grasp_poses

def generate_grasps_for_all_joints(stage,
                                   object_ref_path: str,
                                   num_samples_per_joint: int = 1) -> Dict[str, Dict]:
    """Generate grasp poses for the generated Doorman door joints."""
    joints_info = _build_joints_info_from_generated_door(stage, object_ref_path)
    if not joints_info:
        print(
            "[ERROR] No generated Doorman door layout detected. "
            "Expected canonical joints and grasp_target from generate_door_standalone.py."
        )
        return {}

    print(f"\n[INFO] Found {len(joints_info)} joint(s) with handles")

    all_grasps = {}

    for rotate_joint_name, joint_name, joint_type, child_link_name, lower_limit, upper_limit, joint_axis in joints_info:
        print(f"\n{'='*80}")
        print(f"Processing: rotate={rotate_joint_name}, push={joint_name} ({joint_type}) -> {child_link_name}")
        print(f"  Joint axis: {joint_axis}")
        print(f"{'='*80}")

        grasp_poses = generate_and_filter_grasps_for_joint(
            stage,
            joint_name,
            child_link_name,
            object_ref_path,
            num_samples=num_samples_per_joint,
        )

        if grasp_poses:
            all_grasps[joint_name] = {
                'grasp_poses': grasp_poses,
                'joint_type': joint_type,
                'child_link_name': child_link_name,
                'lower_limit': lower_limit,
                'upper_limit': upper_limit,
                'joint_axis': joint_axis,
                'rotate_joint_name': rotate_joint_name,   # may be None
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
    tolerance_override: Optional[float] = None,
) -> bool:
    cur = get_joint_current_position(stage, joint_prim)
    if cur is None:
        return True  # conservative reject

    open_dir = 1.0 if target_displacement >= 0.0 else -1.0
    desired = initial_joint_pos + should_displacement
    overshoot = open_dir * (float(cur) - desired)

    tol = (
        float(tolerance_override)
        if tolerance_override is not None
        else (
            np.deg2rad(OVERSHOOT_TOLERANCE_DEG)
            if joint_type == "revolute"
            else OVERSHOOT_TOLERANCE_M
        )
    )
    
    if overshoot > tol:
        print(
            f"[OVERSHOOT-REJECT] joint={joint_prim.GetPath()} "
            f"cur={cur:.6f} desired={desired:.6f} should_disp={should_displacement:.6f} "
            f"target_disp={target_displacement:.6f} overshoot={overshoot:.6f} tol={tol:.6f}"
        )
        return True

    return False


def build_trajectory_checkpoint_steps(num_steps: int, num_checks: int) -> List[int]:
    """Return evenly spaced trajectory checkpoints, excluding step 0."""
    num_steps = int(num_steps)
    num_checks = max(0, int(num_checks))
    if num_steps <= 1 or num_checks <= 0:
        return []
    checkpoints = np.linspace(1, num_steps - 1, num=min(num_checks, num_steps - 1))
    return sorted({int(round(v)) for v in checkpoints})

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
    if not joint_name_from_urdf:
        return None

    def _is_supported_joint(prim: Usd.Prim) -> bool:
        return prim.IsValid() and (
            prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
        )

    # Extract joint number from URDF name
    joint_name_clean = joint_name_from_urdf.lower().replace("joint_", "").replace("joint", "")

    asset_root = resolve_asset_root_under_ref(stage, object_ref_path)
    direct_candidates = [
        f"{asset_root}/{joint_name_from_urdf}",
        f"{asset_root}/{GENERATED_DOOR_ROOT_LINK_NAME}/{joint_name_from_urdf}",
        f"{asset_root}/{GENERATED_DOOR_PANEL_LINK_NAME}/{joint_name_from_urdf}",
        f"{asset_root}/{GENERATED_DOOR_HANDLE_LINK_NAME}/{joint_name_from_urdf}",
        f"{asset_root}/fake_door/{joint_name_from_urdf}",
    ]
    for candidate_path in direct_candidates:
        candidate_prim = stage.GetPrimAtPath(candidate_path)
        if _is_supported_joint(candidate_prim):
            print(f"[INFO] Found USD joint via canonical path: {candidate_path}")
            return candidate_prim

    asset_root_prim = stage.GetPrimAtPath(asset_root)
    if asset_root_prim.IsValid():
        for prim in Usd.PrimRange(asset_root_prim):
            if not _is_supported_joint(prim):
                continue
            if prim.GetName().lower() == joint_name_from_urdf.lower():
                print(f"[INFO] Found USD joint via recursive exact-name search: {prim.GetPath()}")
                return prim

    joints_root = f"{asset_root}/joints"
    direct_joint_path = f"{joints_root}/joint_{joint_name_clean}"

    direct_prim = stage.GetPrimAtPath(direct_joint_path)
    if _is_supported_joint(direct_prim):
        print(f"[INFO] Found USD joint via direct path: {direct_joint_path}")
        return direct_prim

    # Fallback: search under /joints
    joints_root_prim = stage.GetPrimAtPath(joints_root)
    if not joints_root_prim.IsValid():
        print(f"[ERROR] No /joints folder found at {joints_root}")
        return None

    for prim in joints_root_prim.GetChildren():
        if not _is_supported_joint(prim):
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
        - pivot: [3] world-frame joint-frame origin (true hinge/spindle reference point)
        - lower_limit: float (radians for revolute, meters for prismatic)
        - upper_limit: float (radians for revolute, meters for prismatic)
        - limit_units: 'rad', 'deg', or 'm'
        - joint_frame_world: Gf.Matrix4d
        - axis_token: 'X', 'Y', or 'Z'
        - used_body_path: which body path provided the joint frame
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

    # Use the physical joint-frame origin as the pivot for motion planning.
    # For generated doors this is the true hinge/spindle location, which is
    # what revolute trajectory planning should follow.
    body1_targets = []
    try:
        rel1 = joint_usd.GetBody1Rel()
        if rel1 and rel1.IsValid():
            body1_targets = rel1.GetTargets()
    except Exception:
        pass

    body1_path = body1_targets[0].pathString if body1_targets else ""

    pivot_world_gf = joint_frame_world.Transform(Gf.Vec3d(0, 0, 0))
    pivot_world = np.array(
        [float(pivot_world_gf[0]), float(pivot_world_gf[1]), float(pivot_world_gf[2])],
        dtype=np.float64,
    )
    used_body_path = frame_body_path

    body1_center = None
    if body1_path:
        body1_center = get_body_world_position(stage, body1_path)

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
    print(
        f"[DEBUG]   axis_token={axis_token}, axis_world={axis_world}, "
        f"pivot_world={pivot_world}, body1_center={body1_center}"
    )
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
        "body1_center": body1_center,
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


def offset_pose_along_local_axis(
    position: np.ndarray,
    quaternion: np.ndarray,
    offset: float,
    local_direction: np.ndarray = np.array(HAND_PALM_DIRECTION, dtype=np.float64),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Offset position along a local hand-frame direction.

    Args:
        position: [3] position
        quaternion: [4] quaternion [w,x,y,z]
        offset: Distance (positive = forward, negative = backward)
        local_direction: Local hand-frame direction to offset along

    Returns:
        (new_position, quaternion)
    """
    # Convert to scipy format [x,y,z,w]
    quat_xyzw = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rot = R.from_quat(quat_xyzw)

    world_axis = rot.apply(np.asarray(local_direction, dtype=np.float64))

    new_position = position + world_axis * offset
    return new_position, quaternion


def generate_trajectories_for_all_grasps(
    stage,
    all_grasps: Dict[str, Dict],
    object_ref_path: str,
    num_trajectory_steps: int = 200,
    opening_fraction: float = 0.8,
) -> List[Dict]:
    """
    Generate three-phase trajectories for every grasp pose:
      1. Approach  — computed separately in physics_validation_loop
      2. Rotate    — arc around the handle's own joint axis (body1-derived pivot+axis)
      3. Push      — arc/slide along the door-hinge (push) joint

    Each returned dict contains:
      joint_name / joint_type              — push joint identity
      grasp_position / grasp_quaternion    — end of approach, start of rotate
      rotate_joint_name                    — may be None
      rotate_positions  [N,3]              — rotate-phase waypoints
      rotate_orientations [N,4]
      rotate_target_displacement           — signed target for rotate joint
      trajectory_positions  [N,3]          — push-phase waypoints
      trajectory_orientations [N,4]
      target_displacement                  — signed target for push joint
      joint_motion / joint_pivot_world     — metadata
    """
    all_trajectories = []
    eef_local_offset_cache: Dict[str, np.ndarray] = {}

    for joint_name, joint_data in all_grasps.items():
        print(f"\n[INFO] Planning trajectories for push joint: {joint_name}")

        # ── Push joint ────────────────────────────────────────────────────────
        push_joint_prim = find_usd_joint_prim(stage, object_ref_path, joint_name)
        if push_joint_prim is None:
            print(f"[ERROR] Could not find USD push joint for {joint_name}")
            continue

        push_joint_params = get_joint_world_parameters(stage, push_joint_prim)
        if push_joint_params is None:
            print(f"[ERROR] Could not extract parameters for push joint {joint_name}")
            continue

        push_target = compute_target_joint_displacement(push_joint_params, opening_fraction)

        # ── Rotate joint (optional) ───────────────────────────────────────────
        rotate_joint_name = joint_data.get("rotate_joint_name")
        rotate_joint_params = None   # full params for evaluation
        rotate_pivot = None          # body1 world position
        rotate_axis  = None          # handle-joint axis in world frame

        if rotate_joint_name:
            rotate_joint_prim = find_usd_joint_prim(stage, object_ref_path, rotate_joint_name)
            if rotate_joint_prim is not None:
                # Get full params for limit / type info
                rotate_joint_params = get_joint_world_parameters(stage, rotate_joint_prim)

                # Pivot = body1 world position (from joint frame on body1 side)
                body1_result = get_handle_rotate_axis_from_body1(
                    stage, rotate_joint_prim, draw_debug=False
                )
                if body1_result is not None:
                    rotate_pivot, body1_axis = body1_result
                    # Use the authored handle_joint axis directly for the rotate phase.
                    rotate_axis = body1_axis

                    # Draw the actual rotate axis (magenta) at the body1 pivot
                    try:
                        from isaacsim.util.debug_draw import _debug_draw
                        _dd = _debug_draw.acquire_debug_draw_interface()
                        _ll = 0.3
                        _p0 = rotate_pivot - rotate_axis * _ll
                        _p1 = rotate_pivot + rotate_axis * _ll
                        _dd.draw_lines(
                            [(float(_p0[0]), float(_p0[1]), float(_p0[2]))],
                            [(float(_p1[0]), float(_p1[1]), float(_p1[2]))],
                            [(1.0, 0.0, 1.0, 1.0)],  # magenta
                            [4.0],
                        )
                    except Exception as _e:
                        print(f"[RotateAxis] debug draw failed: {_e}")

                    # Patch params so plan_revolute_joint_trajectory uses the correct values
                    if rotate_joint_params is not None:
                        rotate_joint_params = dict(rotate_joint_params)
                        rotate_joint_params["pivot"] = rotate_pivot
                        rotate_joint_params["axis"]  = rotate_axis

                if rotate_joint_params is not None:
                    print(
                        f"[INFO] Rotate joint '{rotate_joint_name}': "
                        f"pivot={np.round(rotate_pivot, 4)}, "
                        f"axis (handle joint)={np.round(rotate_axis, 4)}, "
                        f"range={np.degrees(float(rotate_joint_params['upper_limit']) - float(rotate_joint_params['lower_limit'])):.1f}° "
                        f"(per-mode rotate acceptance applied per grasp)"
                    )
            else:
                print(f"[WARN] Rotate joint '{rotate_joint_name}' not found in USD; skipping rotate phase")

        # ── Per-grasp trajectory planning ─────────────────────────────────────
        grasp_poses = joint_data["grasp_poses"]

        for grasp_idx, grasp_entry in enumerate(grasp_poses):
            try:
                if isinstance(grasp_entry, dict):
                    grasp_pos = grasp_entry["position"]
                    grasp_quat = grasp_entry["quaternion"]
                    grasp_mode = str(grasp_entry.get("mode_name", GENERATED_DOOR_PRIMARY_GRASP_MODE))
                    eef_link_name = str(grasp_entry.get("eef_link_name", HAND_EEF_LINK_NAME))
                    approach_local_direction = np.asarray(
                        grasp_entry.get("approach_local_direction", HAND_PALM_DIRECTION),
                        dtype=np.float64,
                    )
                    close_during_approach = bool(grasp_entry.get("close_during_approach", True))
                    rotate_accept_ratio = float(grasp_entry.get("rotate_accept_ratio", 1.0))
                else:
                    grasp_pos, grasp_quat = grasp_entry
                    grasp_mode = GENERATED_DOOR_PRIMARY_GRASP_MODE
                    eef_link_name = HAND_EEF_LINK_NAME
                    approach_local_direction = np.array(HAND_PALM_DIRECTION, dtype=np.float64)
                    close_during_approach = True
                    rotate_accept_ratio = 1.0

                # Normalise types
                if isinstance(grasp_pos, (Gf.Vec3d, Gf.Vec3f)):
                    grasp_pos_np = np.array([float(grasp_pos[0]), float(grasp_pos[1]), float(grasp_pos[2])], dtype=np.float64)
                else:
                    grasp_pos_np = np.asarray(grasp_pos, dtype=np.float64)

                if isinstance(grasp_quat, (Gf.Quatd, Gf.Quatf)):
                    w = float(grasp_quat.GetReal())
                    im = grasp_quat.GetImaginary()
                    grasp_quat_np = np.array([w, float(im[0]), float(im[1]), float(im[2])], dtype=np.float64)
                else:
                    grasp_quat_np = np.asarray(grasp_quat, dtype=np.float64)

                eef_local_offset = get_cached_hand_eef_local_offset(
                    stage, GRIPPER_REF_PATH, eef_link_name, eef_local_offset_cache
                )
                if eef_local_offset is None:
                    print(
                        f"[WARN] Skipping grasp {grasp_idx} for joint {joint_name}: "
                        f"missing hand frame '{eef_link_name}'"
                    )
                    continue

                # Generated door grasp poses are authored for the actual hand EEF.
                grasp_eef_pos_np = grasp_pos_np.copy()

                # ── Phase 2: Rotate trajectory ────────────────────────────────
                # Uses body1-derived pivot + axis from get_handle_rotate_axis_from_body1.
                if rotate_joint_params is not None:
                    rotate_target = compute_target_joint_displacement(
                        rotate_joint_params, HANDLE_ROTATE_RATIO
                    )
                    if rotate_joint_params["joint_type"] == "revolute":
                        rotate_positions_eef = plan_revolute_joint_trajectory(
                            grasp_eef_pos_np, rotate_joint_params, rotate_target, num_trajectory_steps
                        )
                        rotate_orient_method = "revolute_follow"
                    else:
                        rotate_positions_eef = plan_prismatic_joint_trajectory(
                            grasp_eef_pos_np, rotate_joint_params, rotate_target, num_trajectory_steps
                        )
                        rotate_orient_method = "fixed"
                    rotate_orientations = compute_gripper_orientation_for_trajectory(
                        rotate_positions_eef, grasp_quat_np,
                        joint_params=rotate_joint_params, method=rotate_orient_method,
                    )
                else:
                    # No rotate joint — stay at grasp position for 1 waypoint
                    rotate_positions_eef = grasp_eef_pos_np.reshape(1, 3)
                    rotate_orientations = grasp_quat_np.reshape(1, 4)

                # ── Phase 3: Push trajectory (starts from end of rotate) ───────
                push_start_pos  = rotate_positions_eef[-1]
                push_start_quat = rotate_orientations[-1]

                if push_joint_params["joint_type"] == "revolute":
                    push_positions_eef = plan_revolute_joint_trajectory(
                        push_start_pos, push_joint_params, push_target, num_trajectory_steps
                    )
                    push_orient_method = "revolute_follow"
                else:
                    push_positions_eef = plan_prismatic_joint_trajectory(
                        push_start_pos, push_joint_params, push_target, num_trajectory_steps
                    )
                    push_orient_method = "fixed"

                push_orientations = compute_gripper_orientation_for_trajectory(
                    push_positions_eef, push_start_quat,
                    joint_params=push_joint_params, method=push_orient_method,
                )

                grasp_hand_pos_np = compute_hand_base_position_from_eef_target(
                    grasp_eef_pos_np, grasp_quat_np, eef_local_offset
                )
                rotate_positions = convert_eef_positions_to_hand_base_positions(
                    rotate_positions_eef, rotate_orientations, eef_local_offset
                )
                push_positions = convert_eef_positions_to_hand_base_positions(
                    push_positions_eef, push_orientations, eef_local_offset
                )

                joint_motion = {
                    "joint_type":     push_joint_params["joint_type"],
                    "axis":           push_joint_params["axis"],
                    "lower_limit":    push_joint_params["lower_limit"],
                    "upper_limit":    push_joint_params["upper_limit"],
                    "limit_units":    push_joint_params["limit_units"],
                    "axis_token":     push_joint_params["axis_token"],
                    "used_body_path": push_joint_params["used_body_path"],
                }

                print(
                    f"[TRAJ] push={joint_name}, rotate={rotate_joint_name}, "
                    f"grasp_idx={grasp_idx}, mode={grasp_mode}, eef={eef_link_name}, "
                    f"push_type={push_joint_params['joint_type']}, "
                    f"rotate_accept_ratio={rotate_accept_ratio:.0%}"
                )

                all_trajectories.append({
                    # Identity
                    "joint_name":                joint_name,
                    "joint_type":                push_joint_params["joint_type"],
                    "grasp_index":               grasp_idx,
                    "grasp_mode":                grasp_mode,
                    "eef_link_name":             eef_link_name,
                    "approach_local_direction":  approach_local_direction.copy(),
                    "close_during_approach":     close_during_approach,
                    # Grasp pose (start of rotate phase), wrapper/base pose used for execution
                    "grasp_position":            grasp_hand_pos_np,
                    "grasp_eef_position":        grasp_eef_pos_np,
                    "grasp_quaternion":          grasp_quat_np,
                    # Rotate phase
                    "rotate_joint_name":         rotate_joint_name,
                    "rotate_positions":          rotate_positions,
                    "rotate_eef_positions":      rotate_positions_eef,
                    "rotate_orientations":       rotate_orientations,
                    "rotate_target_displacement": rotate_target,
                    "rotate_accept_ratio":       rotate_accept_ratio,
                    # Push phase  ('trajectory_*' keys kept for downstream compat)
                    "trajectory_positions":      push_positions,
                    "trajectory_eef_positions":  push_positions_eef,
                    "trajectory_orientations":   push_orientations,
                    "target_displacement":        push_target,
                    # Metadata
                    "joint_motion":              joint_motion,
                    "joint_pivot_world":         push_joint_params["pivot"],
                })

            except Exception as e:
                print(f"[ERROR] Failed to plan trajectory for grasp {grasp_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        count = len([t for t in all_trajectories if t["joint_name"] == joint_name])
        print(f"[INFO] Generated {count} trajectories for push joint {joint_name}")

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
    Create a high-friction physics material and bind it to all collision meshes 
    under root_prim_path. Also disables instanceable on all prims in the hierarchy.
    
    Args:
        stage: USD stage
        root_prim_path: Root prim path to apply material to (e.g., OBJECT_REF_PATH)
        static_friction: Static friction coefficient (default: 2.0)
        dynamic_friction: Dynamic friction coefficient (default: 2.0)
        restitution: Bounciness (default: 0.0 for no bounce)
    
    Returns:
        int: Number of prims the material was applied to
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
        return 0
    
    count = 0
    instanceable_disabled = 0
    
    # Iterate through all prims in the hierarchy
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsValid():
            continue
        
        # Disable instanceable if set
        if prim.IsInstanceable():
            try:
                prim.SetInstanceable(False)
                instanceable_disabled += 1
            except Exception as e:
                print(f"[WARN] Failed to disable instanceable on {prim.GetPath()}: {e}")
        
        # Bind material to meshes and collision prims
        should_bind = False
        
        if prim.IsA(UsdGeom.Mesh):
            should_bind = True
        elif prim.HasAPI(UsdPhysics.CollisionAPI):
            should_bind = True
        
        if should_bind:
            try:
                # Use MaterialBindingAPI to bind the physics material
                api = UsdShade.MaterialBindingAPI(prim)
                if not api:
                    api = UsdShade.MaterialBindingAPI.Apply(prim)
                
                api.Bind(
                    UsdShade.Material(mat_prim), 
                    materialPurpose="physics"
                )
                count += 1
            except Exception as e:
                print(f"[WARN] Failed to bind material to {prim.GetPath()}: {e}")
    
    print(f"[INFO] Physics material '{material_path}' applied:")
    print(f"  - Bound to {count} prim(s)")
    print(f"  - Disabled instanceable on {instanceable_disabled} prim(s)")
    print(f"  - Static friction: {static_friction}")
    print(f"  - Dynamic friction: {dynamic_friction}")
    print(f"  - Restitution: {restitution}")
    
    return count


def _should_enable_runtime_hand_collision(prim_path: str) -> bool:
    low = prim_path.lower()
    if any(tok in low for tok in HAND_RUNTIME_COLLISION_BLOCK_TOKENS):
        return False
    return any(tok in low for tok in HAND_RUNTIME_COLLISION_ALLOW_TOKENS)


def configure_runtime_gripper_collisions(stage, gripper_root_path: str) -> int:
    """Keep collisions only on the intended grasping links to avoid palm-gap snags."""
    root_prim = stage.GetPrimAtPath(gripper_root_path)
    if not root_prim.IsValid():
        print(f"[WARN] Invalid gripper root path for collision filtering: {gripper_root_path}")
        return 0

    enabled_count = 0
    disabled_count = 0

    for prim in Usd.PrimRange(root_prim):
        if not prim.IsValid():
            continue

        prim_path = prim.GetPath().pathString
        is_mesh = prim.IsA(UsdGeom.Mesh)
        had_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        should_enable = _should_enable_runtime_hand_collision(prim_path)

        if not is_mesh and not had_collision:
            continue

        if should_enable and is_mesh and not had_collision:
            UsdPhysics.CollisionAPI.Apply(prim)
            had_collision = True

        if should_enable and is_mesh:
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(prim)
            # Convex hull is less snag-prone than convex decomposition for the hand links.
            UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr().Set("convexHull")

        if had_collision:
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(bool(should_enable))
            if should_enable:
                enabled_count += 1
            else:
                disabled_count += 1

    print(f"[INFO] Runtime gripper collision filtering under {gripper_root_path}:")
    print(f"  - Enabled collision on {enabled_count} prim(s)")
    print(f"  - Disabled collision on {disabled_count} prim(s)")
    print(f"  - Allowed tokens: {HAND_RUNTIME_COLLISION_ALLOW_TOKENS}")
    print(f"  - Blocked tokens: {HAND_RUNTIME_COLLISION_BLOCK_TOKENS}")
    return enabled_count

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

    configure_hand_joint_drives(stage, GRIPPER_REF_PATH)

    await omni.kit.app.get_app().next_update_async()

    print(f"\n[INFO] Applying high-friction physics material to object...")
    num_prims_with_material = create_and_bind_high_friction_material(
        stage,
        OBJECT_REF_PATH,
        static_friction=2.0,
        dynamic_friction=2.0,
        restitution=0.0
    )
    num_prims_with_material_hand = create_and_bind_high_friction_material(
        stage,
        GRIPPER_REF_PATH,
        static_friction=2.0,
        dynamic_friction=2.0,
        restitution=0.0
    )
    # Keep the authored hand colliders as-is.
    # We do not rewrite runtime hand collision approximations or masks here.
    
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
    print(f"\n[INFO] Generating grasps...")
    
    all_grasps = generate_grasps_for_all_joints(
        stage,
        OBJECT_REF_PATH,
        num_samples_per_joint=NUM_SAMPLES_PER_JOINT,
    )
    if not all_grasps:
        print(f"[ERROR] No grasps generated")
        return
    
    # =====================
    # Preserve authored colliders and instanceability
    # =====================
    print(f"\n[INFO] Skipping runtime overlap-filter collider edits and instanceable changes.")
        
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

    # # =====================
    # # DEBUG PAUSE — inspect the rotate axis (magenta line) in the viewport.
    # # The axis drawn by get_handle_rotate_axis_from_body1 is visible now.
    # # Remove or reduce this step count once the axis looks correct.
    # # =====================
    # print("[DEBUG] Pausing for axis inspection — check the magenta line in the viewport.")
    # print("[DEBUG]   rotate_joint axes are drawn; verify pivot + direction before proceeding.")
    # await step_simulation(10000)
    # print("[DEBUG] Pause complete, continuing...")

    # =====================
    # RESTORE INSTANCEABLE
    # =====================
    # print(f"\n[INFO] Restoring instanceable state...")
    # restore_instanceable(stage, changed_instanceable_paths)
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
        pose0 = _get_prim_world_pose(stage.GetPrimAtPath(env_path(0)))
        pose1 = _get_prim_world_pose(stage.GetPrimAtPath(env_path(1))) if NUM_COPIES > 1 else None
        if pose0 is not None:
            p0, _ = pose0
            print(f"[DEBUG] env_0 world pos: {p0}")
            if pose1 is not None:
                p1, _ = pose1
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


def _thumb_closure_progress(alpha: float) -> float:
    """Delayed thumb closure schedule for the approach phase.

    - 0-30% of the approach: thumb stays open
    - 30-90%: thumb moves from 0% to 50% closure
    - 90-100%: thumb moves from 50% to 100% closure
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.30:
        return 0.0
    if alpha <= 0.90:
        return 0.5 * ((alpha - 0.30) / 0.60)
    return 0.5 + 0.5 * ((alpha - 0.90) / 0.10)


def compute_stepback_and_approach_poses(
    grasp_position: np.ndarray,
    grasp_quaternion: np.ndarray,
    approach_distance: float,
    move_steps: int,
    hand_open_pose: np.ndarray = None,   # [7] joint angles at step-back
    hand_close_pose: np.ndarray = None,  # [7] joint angles at grasp
    approach_local_direction: np.ndarray = np.array(HAND_PALM_DIRECTION, dtype=np.float64),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute step-back position along the requested approach direction, then linearly interpolate
    positions and slerp quaternions from step-back pose to grasp pose.

    Returns:
        positions:   [N, 3]  world positions
        quaternions: [N, 4]  slerped quaternions [w,x,y,z]
        joint_angles:[N, 7]  right hand joint angles with delayed thumb closure
    """
    grasp_position  = np.asarray(grasp_position,  dtype=np.float64).reshape(3)
    grasp_quaternion = np.asarray(grasp_quaternion, dtype=np.float64).reshape(4)

    # --- Step back along the requested local approach direction ---
    back_pos, _ = offset_pose_along_local_axis(
        grasp_position, grasp_quaternion, -approach_distance, local_direction=approach_local_direction
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

        # --- Joint angles: linear interpolation, except delayed thumb closure ---
        joint_angles[i] = hand_open_pose * (1.0 - alpha) + hand_close_pose * alpha
        thumb_alpha = _thumb_closure_progress(alpha)
        for joint_idx, joint_name in enumerate(RIGHT_HAND_JOINT_NAMES):
            if "thumb" in joint_name:
                joint_angles[i, joint_idx] = (
                    hand_open_pose[joint_idx] * (1.0 - thumb_alpha)
                    + hand_close_pose[joint_idx] * thumb_alpha
                )

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

    eef_local_offset_cache: Dict[str, np.ndarray] = {}
    for traj in batch:
        eef_link_name = str(traj.get("eef_link_name", HAND_EEF_LINK_NAME))
        offset = get_cached_hand_eef_local_offset(stage, GRIPPER_REF_PATH, eef_link_name, eef_local_offset_cache)
        if offset is None:
            print(f"[WARN] Falling back to zero hand->{eef_link_name} offset during validation.")
            eef_local_offset_cache[eef_link_name] = np.zeros(3, dtype=np.float64)

    # ── Step 1: Park grippers away from the object, then reset joints ─────────
    # Move all grippers to a safe out-of-scene position first so they don't
    # collide with the door while joints are snapped back to their lower limits.
    park_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity orientation
    for env_idx in range(len(batch)):
        park_pos = GRIPPER_PARK_POSITION.copy()
        park_pos[0] += env_idx * CLONE_SPACING   # offset per env to match clone grid
        set_gripper_world_pose(stage, grip_wrap(env_idx), park_pos, park_quat)
    await step_simulation(5)  # let gripper reach park position before joint reset

    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        object_ref_path = obj_ref(env_idx)

        # Push joint
        push_joint_prim = find_usd_joint_prim(stage, object_ref_path, traj.get('joint_name', ''))
        if push_joint_prim:
            push_params = get_joint_world_parameters(stage, push_joint_prim)
            if push_params:
                lower = push_params["lower_limit"]
                dk = "angular" if push_params["joint_type"] == "revolute" else "linear"
                set_usd_joint_drive_target(stage, push_joint_prim.GetPath().pathString, lower, dk)
                set_joint_position_direct(stage, push_joint_prim, lower)

        # Rotate joint (if present)
        rot_joint_name = traj.get('rotate_joint_name')
        if rot_joint_name:
            rot_joint_prim = find_usd_joint_prim(stage, object_ref_path, rot_joint_name)
            if rot_joint_prim:
                rot_params = get_joint_world_parameters(stage, rot_joint_prim)
                if rot_params:
                    lower = rot_params["lower_limit"]
                    dk = "angular" if rot_params["joint_type"] == "revolute" else "linear"
                    set_usd_joint_drive_target(stage, rot_joint_prim.GetPath().pathString, lower, dk)
                    set_joint_position_direct(stage, rot_joint_prim, lower)

    await ensure_timeline_playing()
    await step_simulation(60)  # let resets settle

    # ── Step 2: Record initial joint positions for BOTH joints ────────────────
    initial_states = []
    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        object_ref_path = obj_ref(env_idx)

        push_joint_prim = find_usd_joint_prim(stage, object_ref_path, traj.get('joint_name', ''))
        initial_push_pos = get_joint_current_position(stage, push_joint_prim) if push_joint_prim else 0.0

        rot_joint_name = traj.get('rotate_joint_name')
        rot_joint_prim = None
        initial_rot_pos = 0.0
        rot_joint_range = 0.0
        if rot_joint_name:
            rot_joint_prim = find_usd_joint_prim(stage, object_ref_path, rot_joint_name)
            if rot_joint_prim:
                initial_rot_pos = get_joint_current_position(stage, rot_joint_prim) or 0.0
                rot_params = get_joint_world_parameters(stage, rot_joint_prim)
                if rot_params:
                    rot_joint_range = float(rot_params["upper_limit"]) - float(rot_params["lower_limit"])

        initial_states.append({
            'push_joint_prim':  push_joint_prim,
            'initial_push_pos': initial_push_pos,
            'rot_joint_prim':   rot_joint_prim,
            'initial_rot_pos':  initial_rot_pos,
            'rot_joint_range':  rot_joint_range,
        })

    # ── Step 3: Compute approach trajectories ────────────────────────────────
    all_approach_data = []
    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        grasp_eef_pos = np.asarray(traj.get("grasp_eef_position", traj["grasp_position"]), dtype=np.float64)
        grasp_quat = np.asarray(traj["grasp_quaternion"], dtype=np.float64)
        eef_link_name = str(traj.get("eef_link_name", HAND_EEF_LINK_NAME))
        eef_local_offset = eef_local_offset_cache[eef_link_name]
        approach_local_direction = np.asarray(
            traj.get("approach_local_direction", HAND_PALM_DIRECTION),
            dtype=np.float64,
        )
        close_during_approach = bool(traj.get("close_during_approach", True))
        hand_open_pose = RIGHT_HAND_OPEN_T.numpy()
        hand_close_pose = (
            hand_pose_to_tensor(RIGHT_HAND_POSES[HAND_CLOSE_CONFIG]).numpy()
            if close_during_approach
            else hand_open_pose.copy()
        )

        approach_eef_positions, approach_quats, approach_joints = compute_stepback_and_approach_poses(
            grasp_eef_pos, grasp_quat,
            approach_distance=APPROACH_DISTANCE,
            move_steps=MOVE_STEPS,
            hand_open_pose=hand_open_pose,
            hand_close_pose=hand_close_pose,
            approach_local_direction=approach_local_direction,
        )
        approach_positions = convert_eef_positions_to_hand_base_positions(
            approach_eef_positions, approach_quats, eef_local_offset
        )
        all_approach_data.append({
            'positions':    approach_positions,
            'eef_positions': approach_eef_positions,
            'quats':        approach_quats,
            'joint_angles': approach_joints,
            'eef_link_name': eef_link_name,
            'approach_local_direction': approach_local_direction.copy(),
        })
        print(f"[DEBUG] Env {env_idx} approach joints:")
        print(f"  Step 0 (open):   {dict(zip(RIGHT_HAND_JOINT_NAMES, approach_joints[0]))}")
        print(f"  Step -1 (close): {dict(zip(RIGHT_HAND_JOINT_NAMES, approach_joints[-1]))}")

    approach_data_by_orig_idx = {
        traj.get("original_index", -1): all_approach_data[env_idx]
        for env_idx, traj in enumerate(batch)
    }

    # ── Step 4: Approach phase ────────────────────────────────────────────────
    for step_idx in range(MOVE_STEPS + 1):
        for env_idx in range(len(batch)):
            pos           = all_approach_data[env_idx]['positions'][step_idx]
            quat          = all_approach_data[env_idx]['quats'][step_idx]
            joint_targets = all_approach_data[env_idx]['joint_angles'][step_idx]
            set_gripper_world_pose(stage, grip_wrap(env_idx), pos, quat)
            _joint_paths = resolve_hand_joint_paths(stage, grip_ref(env_idx))
            for jn, ta in zip(RIGHT_HAND_JOINT_NAMES, joint_targets):
                if jn in _joint_paths:
                    set_usd_joint_drive_target(
                        stage, _joint_paths[jn],
                        float(np.degrees(ta)), drive_kind="angular"
                    )
        await step_simulation(1)

    # ── Step 5: Hold phase — contacts settle, hand fully closed ──────────────
    close_joints = hand_pose_to_tensor(RIGHT_HAND_POSES[HAND_CLOSE_CONFIG]).numpy()
    for _ in range(HOLD_STEPS):
        for env_idx in range(len(batch)):
            _joint_paths = resolve_hand_joint_paths(stage, grip_ref(env_idx))
            for jn, ta in zip(RIGHT_HAND_JOINT_NAMES, close_joints):
                if jn in _joint_paths:
                    set_usd_joint_drive_target(
                        stage, _joint_paths[jn],
                        float(np.degrees(ta)), drive_kind="angular"
                    )
        await step_simulation(1)

    # ── Step 6: Rotate phase ──────────────────────────────────────────────────
    # Per-env done flag: True from the start for envs with no rotate joint.
    # Grid-cloner pattern: every env receives an explicit pose command every step.
    # Done envs hold (freeze) at their last commanded pose rather than being skipped.
    rotate_done        = []
    rotate_cutoff_step = []
    # Last commanded gripper pose per env — initialised at grasp pose (end of hold phase)
    last_gripper_pos  = [np.asarray(batch[i]["grasp_position"],  dtype=np.float64) for i in range(len(batch))]
    last_gripper_quat = [np.asarray(batch[i]["grasp_quaternion"], dtype=np.float64) for i in range(len(batch))]

    for env_idx in range(len(batch)):
        rot_prim        = initial_states[env_idx]['rot_joint_prim']
        rot_joint_range = initial_states[env_idx]['rot_joint_range']
        no_rotate = rot_prim is None or abs(rot_joint_range) < 1e-6
        rotate_done.append(no_rotate)
        rotate_cutoff_step.append(None)

    max_rot_length = max(
        np.asarray(traj.get("rotate_positions", []), dtype=np.float64).shape[0]
        for traj in batch
    )
    for traj_step in range(max_rot_length):
        for env_idx in range(len(batch)):
            traj     = batch[env_idx]
            rot_pos  = np.asarray(traj.get("rotate_positions",    []), dtype=np.float64)
            rot_quat = np.asarray(traj.get("rotate_orientations", []), dtype=np.float64)

            if rotate_done[env_idx] or traj_step >= rot_pos.shape[0]:
                # Freeze: hold at last commanded pose
                set_gripper_world_pose(stage, grip_wrap(env_idx),
                                       last_gripper_pos[env_idx], last_gripper_quat[env_idx])
            else:
                # Advance to next waypoint and update last pose
                set_gripper_world_pose(stage, grip_wrap(env_idx), rot_pos[traj_step], rot_quat[traj_step])
                last_gripper_pos[env_idx]  = rot_pos[traj_step]
                last_gripper_quat[env_idx] = rot_quat[traj_step]

            _joint_paths = resolve_hand_joint_paths(stage, grip_ref(env_idx))
            for jn, ta in zip(RIGHT_HAND_JOINT_NAMES, close_joints):
                if jn in _joint_paths:
                    set_usd_joint_drive_target(
                        stage, _joint_paths[jn],
                        float(np.degrees(ta)), drive_kind="angular"
                    )
        await step_simulation(TRAJECTORY_SIM_STEPS_PER_WAYPOINT)

        # Check per-env whether the rotate joint has reached the target
        for env_idx in range(len(batch)):
            if rotate_done[env_idx]:
                continue
            rot_prim        = initial_states[env_idx]['rot_joint_prim']
            init_rot        = initial_states[env_idx]['initial_rot_pos']
            rotate_target   = abs(float(batch[env_idx].get("rotate_target_displacement", 0.0)))
            rotate_accept_ratio = float(batch[env_idx].get("rotate_accept_ratio", 1.0))
            required_rot    = rotate_accept_ratio * rotate_target
            cur = get_joint_current_position(stage, rot_prim)
            if cur is not None and abs(cur - init_rot) >= required_rot:
                rotate_done[env_idx]        = True
                rotate_cutoff_step[env_idx] = traj_step
                print(f"[BATCH {batch_index}] Env {env_idx}: rotate target reached at step {traj_step} "
                      f"(joint={np.degrees(cur):.1f}°, moved={np.degrees(abs(cur - init_rot)):.1f}°, "
                      f"required={np.degrees(required_rot):.1f}° "
                      f"({rotate_accept_ratio:.0%} of planned {np.degrees(rotate_target):.1f}°)")

        # All envs done — proceed directly to push phase without waiting
        if all(rotate_done):
            print(f"[BATCH {batch_index}] All envs reached rotate target at step {traj_step}, "
                  f"skipping remaining rotate steps")
            break

    # Trim rotate trajectories to exclude steps at and after the cutoff
    # (waypoints sent after target was reached are not saved)
    for env_idx in range(len(batch)):
        cutoff = rotate_cutoff_step[env_idx]
        if cutoff is not None:
            traj     = batch[env_idx]
            rot_pos  = np.asarray(traj["rotate_positions"],    dtype=np.float64)
            rot_eef_pos = np.asarray(traj.get("rotate_eef_positions", rot_pos), dtype=np.float64)
            rot_quat = np.asarray(traj["rotate_orientations"], dtype=np.float64)
            orig_len = rot_pos.shape[0]
            traj["rotate_positions"]    = rot_pos[:cutoff]
            traj["rotate_eef_positions"] = rot_eef_pos[:cutoff]
            traj["rotate_orientations"] = rot_quat[:cutoff]
            print(f"[BATCH {batch_index}] Env {env_idx}: trimmed rotate traj "
                  f"{orig_len} -> {cutoff} steps")

    # Re-plan push trajectory from the actual end-of-rotate pose.
    # The pre-planned push was derived from rotate_positions[-1] at generation
    # time, which is now stale if the rotate was cut short. We replan here so
    # the push always starts consistently from where the hand actually ended up.
    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        eef_link_name = str(traj.get("eef_link_name", HAND_EEF_LINK_NAME))
        eef_local_offset = eef_local_offset_cache[eef_link_name]

        rot_pos  = np.asarray(traj["rotate_positions"],    dtype=np.float64)
        rot_eef_pos = np.asarray(traj.get("rotate_eef_positions", rot_pos), dtype=np.float64)
        rot_quat = np.asarray(traj["rotate_orientations"], dtype=np.float64)
        if rot_eef_pos.shape[0] > 0:
            push_start_eef_pos = rot_eef_pos[-1]
            push_start_quat = rot_quat[-1]
        else:
            push_start_eef_pos = np.asarray(traj.get("grasp_eef_position", traj["grasp_position"]), dtype=np.float64)
            push_start_quat = np.asarray(traj["grasp_quaternion"], dtype=np.float64)

        joint_type   = traj["joint_motion"]["joint_type"]
        push_target  = float(traj["target_displacement"])
        push_params  = {
            "joint_type": joint_type,
            "axis":       np.asarray(traj["joint_motion"]["axis"],    dtype=np.float64),
            "pivot":      np.asarray(traj["joint_pivot_world"],       dtype=np.float64),
        }
        num_push_steps = np.asarray(traj["trajectory_positions"], dtype=np.float64).shape[0]

        if joint_type == "revolute":
            new_push_eef_pos = plan_revolute_joint_trajectory(push_start_eef_pos, push_params, push_target, num_push_steps)
            orient_method  = "revolute_follow"
        else:
            new_push_eef_pos = plan_prismatic_joint_trajectory(push_start_eef_pos, push_params, push_target, num_push_steps)
            orient_method  = "fixed"

        new_push_quat = compute_gripper_orientation_for_trajectory(
            new_push_eef_pos, push_start_quat, joint_params=push_params, method=orient_method
        )
        new_push_pos = convert_eef_positions_to_hand_base_positions(
            new_push_eef_pos, new_push_quat, eef_local_offset
        )

        traj["trajectory_positions"]    = new_push_pos
        traj["trajectory_eef_positions"] = new_push_eef_pos
        traj["trajectory_orientations"] = new_push_quat
        print(f"[BATCH {batch_index}] Env {env_idx}: re-planned push traj from "
              f"rotate end EEF pos {np.round(push_start_eef_pos, 3)}")

    # ── Check rotate success before proceeding to push ────────────────────────
    rotate_ok_per_env = []
    for env_idx in range(len(batch)):
        rot_prim        = initial_states[env_idx]['rot_joint_prim']
        init_rot        = initial_states[env_idx]['initial_rot_pos']
        rot_joint_range = initial_states[env_idx]['rot_joint_range']

        if rot_prim is None or abs(rot_joint_range) < 1e-6:
            rotate_ok_per_env.append(True)
            print(f"[BATCH {batch_index}] Env {env_idx}: rotate=N/A (no rotate joint)")
        else:
            cur_rot      = get_joint_current_position(stage, rot_prim)
            actual_rot   = abs((cur_rot or 0.0) - init_rot)
            rotate_target = abs(float(batch[env_idx].get("rotate_target_displacement", 0.0)))
            rotate_accept_ratio = float(batch[env_idx].get("rotate_accept_ratio", 1.0))
            required_rot = rotate_accept_ratio * rotate_target
            rot_ok       = actual_rot >= required_rot
            rotate_ok_per_env.append(rot_ok)
            print(f"[BATCH {batch_index}] Env {env_idx}: "
                  f"rotate init={np.degrees(init_rot):.1f}°, "
                  f"cur={np.degrees(cur_rot or 0.0):.1f}°, "
                  f"required={np.degrees(required_rot):.1f}° "
                  f"({rotate_accept_ratio:.0%} of planned {np.degrees(rotate_target):.1f}°, "
                  f"joint range {np.degrees(rot_joint_range):.1f}°), "
                  f"rotate_ok={rot_ok}")
            if not rot_ok:
                print(f"[BATCH {batch_index}] Env {env_idx}: rotate FAILED — skipping push phase")

    # ── Step 7: Push phase ────────────────────────────────────────────────────
    # Rotate-failed envs freeze at their last rotate-end pose (grid-cloner pattern).
    push_reject_per_env = [False for _ in range(len(batch))]
    push_checkpoint_steps = []
    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        push_len = np.asarray(traj.get("trajectory_positions", []), dtype=np.float64).shape[0]
        push_checkpoint_steps.append(set(build_trajectory_checkpoint_steps(push_len, TRAJ_OVERSHOOT_CHECKS)))

    max_push_length = max(
        np.asarray(traj.get("trajectory_positions", []), dtype=np.float64).shape[0]
        for traj in batch
    )
    for traj_step in range(max_push_length):
        for env_idx in range(len(batch)):
            traj      = batch[env_idx]
            push_pos  = np.asarray(traj.get("trajectory_positions",    []), dtype=np.float64)
            push_quat = np.asarray(traj.get("trajectory_orientations", []), dtype=np.float64)

            if (not rotate_ok_per_env[env_idx]) or push_reject_per_env[env_idx] or traj_step >= push_pos.shape[0]:
                # Freeze: hold at last commanded pose (rotate-end for failed, last push for exhausted)
                set_gripper_world_pose(stage, grip_wrap(env_idx),
                                       last_gripper_pos[env_idx], last_gripper_quat[env_idx])
            else:
                set_gripper_world_pose(stage, grip_wrap(env_idx), push_pos[traj_step], push_quat[traj_step])
                last_gripper_pos[env_idx]  = push_pos[traj_step]
                last_gripper_quat[env_idx] = push_quat[traj_step]

            _joint_paths = resolve_hand_joint_paths(stage, grip_ref(env_idx))
            for jn, ta in zip(RIGHT_HAND_JOINT_NAMES, close_joints):
                if jn in _joint_paths:
                    set_usd_joint_drive_target(
                        stage, _joint_paths[jn],
                        float(np.degrees(ta)), drive_kind="angular"
                    )
        await step_simulation(TRAJECTORY_SIM_STEPS_PER_WAYPOINT)

        for env_idx in range(len(batch)):
            if (not rotate_ok_per_env[env_idx]) or push_reject_per_env[env_idx]:
                continue

            traj = batch[env_idx]
            push_prim = initial_states[env_idx]['push_joint_prim']
            init_push = initial_states[env_idx]['initial_push_pos']
            push_target = float(traj.get("target_displacement", 0.0))
            joint_type = str(traj.get("joint_type", ""))
            push_len = np.asarray(traj.get("trajectory_positions", []), dtype=np.float64).shape[0]

            if push_prim is None or joint_type != "revolute" or push_len <= 1:
                continue
            if traj_step not in push_checkpoint_steps[env_idx]:
                continue

            alpha = traj_step / float(max(push_len - 1, 1))
            should_displacement = push_target * alpha
            if overshoot_reject(
                stage,
                push_prim,
                initial_joint_pos=float(init_push),
                should_displacement=float(should_displacement),
                target_displacement=float(push_target),
                joint_type=joint_type,
                tolerance_override=float(np.deg2rad(DOOR_OPEN_OVERSHOOT_REJECT_DEG)),
            ):
                push_reject_per_env[env_idx] = True
                print(
                    f"[BATCH {batch_index}] Env {env_idx}: push overshoot reject at step {traj_step} "
                    f"(threshold={DOOR_OPEN_OVERSHOOT_REJECT_DEG:.1f} deg)"
                )

    # ── Step 8: Evaluate push success and combine ─────────────────────────────
    results = []
    for env_idx in range(len(batch)):
        traj         = batch[env_idx]
        original_idx = traj.get('original_index', -1)
        joint_type   = traj.get("joint_type", "")

        # Rotate already checked — fail fast without running push eval
        if not rotate_ok_per_env[env_idx]:
            results.append((original_idx, False))
            continue
        if push_reject_per_env[env_idx]:
            print(f"[BATCH {batch_index}] Env {env_idx}: push FAILED — overshoot rejection triggered")
            results.append((original_idx, False))
            continue

        push_prim    = initial_states[env_idx]['push_joint_prim']
        init_push    = initial_states[env_idx]['initial_push_pos']
        push_target  = traj.get("target_displacement", 0.0)

        final_push = get_joint_current_position(stage, push_prim)
        if final_push is None:
            results.append((original_idx, False))
            continue

        actual_push   = abs(final_push - init_push)
        required_push = JOINT_SUCCESS_THRESHOLD * abs(push_target)
        push_ok = actual_push >= required_push

        is_valid = push_ok

        if joint_type == "revolute":
            print(f"[BATCH {batch_index}] Env {env_idx}: "
                  f"push init={np.degrees(init_push):.1f}°, "
                  f"final={np.degrees(final_push):.1f}°, "
                  f"required={np.degrees(required_push):.1f}°, "
                  f"push_ok={push_ok}, valid={is_valid}")
        else:
            print(f"[BATCH {batch_index}] Env {env_idx}: "
                  f"push init={init_push:.4f}m, final={final_push:.4f}m, "
                  f"required={required_push:.4f}m, "
                  f"push_ok={push_ok}, valid={is_valid}")

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
    for batch_idx, batch in enumerate(batches):
            
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
