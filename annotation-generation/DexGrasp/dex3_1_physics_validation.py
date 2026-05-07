import os
os.environ["OMNI_LOG_LEVEL_DEFAULT"] = "error"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import carb
carb.settings.get_settings().set("/log/level", 3)           # 3 = error only
carb.settings.get_settings().set("/log/fileLogLevel", 3)
carb.settings.get_settings().set("/rtx/instanceLogging", False)

import warnings
warnings.filterwarnings("ignore")

import asyncio
import json
import numpy as np
from pathlib import Path
import sys
import shutil
from collections import defaultdict

import omni.kit.app
import omni.usd
import omni.physics.tensors
from pxr import Usd, UsdGeom, Sdf, Gf, UsdPhysics, UsdShade, PhysxSchema, UsdLux
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage_id
from omni.timeline import get_timeline_interface
from isaacsim.core.api.robots import Robot
from isaacsim.core.cloner import GridCloner
from isaacsim.core.simulation_manager import SimulationManager


# ================= OBJECT TRANSFORM HELPERS =================
def log_info(msg):
    print(f"[INFO] {msg}")

def log_state(state_name, elapsed, duration):
    if int(elapsed * 10) % 5 == 0:
        print(f"  -> State: {state_name} | Progress: {elapsed:.2f}/{duration:.1f}s")

def quat_mul(q1, q2):
    # q = [w, x, y, z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quat_rotate_vector(q, v):
    # v: [x, y, z], q: [w, x, y, z]
    s = q[0]
    r = q[1:]
    return v + 2 * np.cross(r, s * v + np.cross(r, v))

def transform_pose_to_world(rel_pos, rel_rot, obj_pos, obj_rot):
    # World Position = Obj_Pos + Obj_Rot * Rel_Pos
    world_pos = obj_pos + quat_rotate_vector(obj_rot, rel_pos)
    # World Rotation = Obj_Rot * Rel_Rot
    world_rot = quat_mul(obj_rot, rel_rot)
    return world_pos, world_rot

def validate_pose(pos, rot):
    if np.any(np.isnan(pos)) or np.any(np.isnan(rot)):
        return False
    if np.linalg.norm(rot) < 1e-6: # Quaternion should not be zero
        return False
    return True

def normalize_quaternion(quat):
    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape != (4,) or np.any(np.isnan(quat)):
        return None

    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return None

    return quat / norm

def quat_world_z_axis_z_component(quat):
    quat = normalize_quaternion(quat)
    if quat is None:
        return None

    world_z_axis = quat_rotate_vector(quat, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    return float(world_z_axis[2])

def filter_grasps_by_stage_z_down(grasp_list, stage_key="final_grasp", threshold=-0.2):
    filtered_grasps = []
    invalid_quat_count = 0

    for grasp_idx, grasp_data in enumerate(grasp_list):
        if not isinstance(grasp_data, dict):
            log_info(f"Skipping grasp {grasp_idx}: expected dict, got {type(grasp_data).__name__}")
            continue

        stage_data = grasp_data.get(stage_key)
        if not isinstance(stage_data, dict):
            log_info(f"Skipping grasp {grasp_idx}: missing '{stage_key}' stage")
            continue

        z_component = quat_world_z_axis_z_component(stage_data.get("orientation"))
        if z_component is None:
            invalid_quat_count += 1
            continue

        if z_component > threshold:
            filtered_grasps.append(grasp_data)

    log_info(
        f"Z-down filter on '{stage_key}' kept {len(filtered_grasps)}/{len(grasp_list)} grasps "
        f"(world z-axis component < {threshold:.3f})"
    )
    if invalid_quat_count:
        log_info(f"Skipped {invalid_quat_count} grasps with invalid quaternions during Z-down filtering")

    return filtered_grasps

def run_transform_unit_tests():
    print("[Unit Test] Running Pose Transform Tests...")
    # Case 1: Identity
    obj_p = np.array([0., 0., 0.])
    obj_r = np.array([1., 0., 0., 0.])
    rel_p = np.array([1., 2., 3.])
    rel_r = np.array([1., 0., 0., 0.])
    res_p, res_r = transform_pose_to_world(rel_p, rel_r, obj_p, obj_r)
    assert np.allclose(res_p, rel_p), "Identity pos failed"
    assert np.allclose(res_r, rel_r), "Identity rot failed"

    # Case 2: Translation
    obj_p = np.array([10., 0., 0.])
    res_p, res_r = transform_pose_to_world(rel_p, rel_r, obj_p, obj_r)
    assert np.allclose(res_p, np.array([11., 2., 3.])), "Translation pos failed"

    # Case 3: Rotation (90 deg around Z)
    val = np.sqrt(2)/2
    obj_r = np.array([val, 0., 0., val])
    obj_p = np.array([0., 0., 0.])
    rel_p = np.array([1., 0., 0.]) # Should become [0, 1, 0]
    res_p, res_r = transform_pose_to_world(rel_p, rel_r, obj_p, obj_r)
    assert np.allclose(res_p, np.array([0., 1., 0.]), atol=1e-6), "Rotation pos failed"
    assert np.allclose(res_r, obj_r, atol=1e-6), "Rotation rot failed"
    
    print("[Unit Test] All Tests Passed.")

def interp_pos(p1, p2, t): return p1 * (1.0 - t) + p2 * t
def interp_joints(j1, j2, t): return j1 * (1.0 - t) + j2 * t

def load_grasp_data_json(json_path):
    log_info(f"Loading grasp data from JSON: {json_path}")
    if not os.path.exists(json_path):
        log_info(f"File not found: {json_path}")
        return None
        
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if "grasps" not in data:
        log_info("Invalid JSON format: 'grasps' key missing")
        return None
        
    grasps = data["grasps"]
    log_info(f"Loaded {len(grasps)} grasps.")
    return grasps

def load_grasp_data_json_second_pass(json_path):
    log_info(f"Loading grasp data from JSON: {json_path}")
    if not os.path.exists(json_path):
        log_info(f"File not found: {json_path}")
        return None
        
    with open(json_path, 'r') as f:
        data = json.load(f)
    grasps = data["functional_grasp"]["body"]
    log_info(f"Loaded {len(grasps)} previously successful grasps.")
    return grasps

# ================= CONFIGURATIONS =================
_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _THIS_DIR.parent

def _path_from_env(env_name: str, default: Path) -> str:
    return str(Path(os.environ.get(env_name, str(default))).expanduser())

ROBOT_USD_PATH = _path_from_env(
    "DEX3_ROBOT_USD",
    _WORKSPACE_ROOT / "Bimanual_dexhand" / "assets" / "hands" / "dex3_1" / "asset" / "dex3_1_r.usd",
)
CONTAINER_PATH_BASE = "/World/Envs/env"
PALM_LINK_NAME = "right_hand_palm_link"

GRASP_JSON_PATH = _path_from_env(
    "DEX3_GRASP_JSON",
    _THIS_DIR / "bottle_2" / "Annotation" / "dex3_1_grasp_pose.json",
)
OBJECT_USD_PATH = _path_from_env("DEX3_OBJECT_USD", _THIS_DIR / "bottle_2" / "Object.usd")
PHYSICS_SCENE_PATH = "/World/physicsScene"

# CLONER PARAMETERS
NUM_ENVS = 20  # Number of parallel environments
CLONE_SPACING = 3.0
ENV_BASE_PATH = "/World/Envs"
ENV_ROOT_PREFIX = f"{ENV_BASE_PATH}/env"

# MOTION/FILTER PARAMTERS
LIFT_HEIGHT = 0.6
MAX_DISPLACEMENT_THRESHOLD = 0.05    
DROP_HEIGHT_THRESHOLD = 0.02
CONTACT_THRESHOLD = 0.08 
RETRIEVAL_DIST = LIFT_HEIGHT

RETRIEVAL_FORCE_TEST_MODE = "all"
# options: "none", "all", "pos_x", "neg_x", "pos_y", "neg_y", "neg_z"
RETRIEVAL_FORCE_UNITS = "newton"
# "body_weight": force magnitude = RETRIEVAL_FORCE_MAG * m * g
# "newton": force magnitude = RETRIEVAL_FORCE_MAG in Newtons
RETRIEVAL_FORCE_MAG = 0.0
GRAVITY_ACCEL = 9.81
RETRIEVAL_FORCE_ALL_MODES = ("neg_z", "pos_x", "neg_x", "pos_y", "neg_y")
FILTER_GRASPS_Z_DOWN_ONLY = True
GRASP_Z_DOWN_STAGE = "final_grasp"
GRASP_Z_DOWN_THRESHOLD = -0.15

# SIMULATION STEPS
COARSE_STEPS = 60       
FINE_STEPS = 60         
FINAL_STEPS = 60        
CHECK_STEPS = 60        
HOLD_STEPS = 50         
LIFT_STEPS = 200  

# JOINT PARAMETERS
JOINT_STIFFNESS = 20.0
JOINT_DAMPING = 2.0
JOINT_MAX_FORCE = 300.0
JOINT_ARMATURE = 0.01
JOINT_VELOCITY_LIMIT = 100.0
CONVEX_DECOMP_MIN_THICKNESS = 0.002

#Part detection parameters
MAX_PARTS = 10

RIGHT_HAND_JOINT_NAMES = [
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]

_HAND_STRUCTURE_CACHE = {}

INVERSE_GRAVITY = False #Set to True for an additional validation round with inverse gravity

# ================= PATH HELPERS =================
def env_path(i: int) -> str:
    # env paths are /World/Envs/env_0, /World/Envs/env_1, ...
    return f"{ENV_ROOT_PREFIX}_{i}"

def obj_wrap(i: int) -> str:
    return f"{env_path(i)}/TargetObject"

def obj_ref(i: int) -> str:
    return f"{env_path(i)}/TargetObject/ref"

def hand_wrap(i: int) -> str:
    return f"{env_path(i)}/Dex3"

def hand_ref(i: int) -> str:
    return f"{env_path(i)}/Dex3/ref"


# PROBES = [
#     f"{hand_ref(0)}/right_hand_index_0_link/Probe1",
#     f"{hand_ref(0)}/right_hand_index_0_link/Probe2",
#     f"{hand_ref(0)}/right_hand_index_1_link/Probe3",
#     f"{hand_ref(0)}/right_hand_middle_0_link/Probe4",
#     f"{hand_ref(0)}/right_hand_middle_0_link/Probe5",
#     f"{hand_ref(0)}/right_hand_middle_1_link/Probe6",
#     f"{hand_ref(0)}/right_hand_thumb_1_link/Probe7",
#     f"{hand_ref(0)}/right_hand_thumb_1_link/Probe8",
#     f"{hand_ref(0)}/right_hand_thumb_2_link/Probe9",  
# ]
PROBES = []

# ================= SETUP STAGE =================
def compute_bottom_center(stage, prim_path):
    """Compute bottom center of object bounding box in world space."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return [0.0, 0.0, 0.0]
    
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    bbox = bbox_cache.ComputeWorldBound(prim)
    box_range = bbox.GetRange()
    
    min_pt = box_range.GetMin()
    max_pt = box_range.GetMax()
    
    center_x = (min_pt[0] + max_pt[0]) / 2.0
    center_y = (min_pt[1] + max_pt[1]) / 2.0
    bottom_z  = min_pt[2]
    
    return [float(center_x), float(center_y), float(bottom_z)]

def add_lighting(stage):
    light = UsdLux.DomeLight.Define(stage, "/World/defaultDomeLight")
    light.CreateIntensityAttr().Set(1000.0)
    d_light = UsdLux.DistantLight.Define(stage, "/World/distantLight")
    d_light.CreateIntensityAttr().Set(2000.0)
    UsdGeom.Xformable(d_light).AddRotateXYZOp().Set(Gf.Vec3f(45, 45, 0))


def resolve_hand_structure(stage, container_path):
    """Resolve palm and named joints under a referenced hand asset."""
    cache_key = (stage.GetRootLayer().identifier, container_path)
    cached = _HAND_STRUCTURE_CACHE.get(cache_key)
    if cached is not None:
        palm_path = cached.get("palm_path")
        ordered_joint_paths = cached.get("ordered_joint_paths", [])
        if palm_path and stage.GetPrimAtPath(palm_path).IsValid():
            if all(stage.GetPrimAtPath(p).IsValid() for p in ordered_joint_paths):
                return cached

    container_prim = stage.GetPrimAtPath(container_path)
    if not container_prim.IsValid():
        print(f"[WARN] Invalid hand container path: {container_path}")
        return None

    palm_path = None
    joint_paths = {}

    for prim in Usd.PrimRange(container_prim):
        if not prim.IsValid():
            continue
        name = prim.GetName()
        if palm_path is None and name == PALM_LINK_NAME:
            palm_path = prim.GetPath().pathString
        if prim.IsA(UsdPhysics.Joint) and name in RIGHT_HAND_JOINT_NAMES and name not in joint_paths:
            joint_paths[name] = prim.GetPath().pathString

    missing = [name for name in RIGHT_HAND_JOINT_NAMES if name not in joint_paths]
    if palm_path is None:
        print(f"[WARN] Could not find palm link '{PALM_LINK_NAME}' under {container_path}")
        return None
    if missing:
        print(f"[WARN] Missing hand joints under {container_path}: {missing}")
        return None

    structure = {
        "palm_path": palm_path,
        "joint_paths_by_name": joint_paths,
        "ordered_joint_paths": [joint_paths[name] for name in RIGHT_HAND_JOINT_NAMES],
    }
    _HAND_STRUCTURE_CACHE[cache_key] = structure
    print(f"[INFO] Resolved hand structure under {container_path}: palm={palm_path}")
    for joint_name in RIGHT_HAND_JOINT_NAMES:
        print(f"  [joint] {joint_name} -> {joint_paths[joint_name]}")
    return structure


def set_joint_drive_target(joint_prim: Usd.Prim, target_position: float) -> bool:
    """Set joint position target using proven DriveAPI pattern."""
    if not joint_prim.IsValid():
        return False

    # Determine drive type
    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        drive_kind = "angular"
    elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
        drive_kind = "linear"
    else:
        return False

    try:
        # Use Apply() - creates if missing, gets if exists
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_kind)
        
        # Set target position
        tp = drive.GetTargetPositionAttr()
        if not tp or not tp.IsValid():
            tp = drive.CreateTargetPositionAttr()
        tp.Set(float(target_position))
        
        return True
        
    except Exception as e:
        print(f"[WARN] Failed to set drive target on {joint_prim.GetPath()}: {e}")
        return False


def set_hand_joint_targets(stage, container_path, joint_values_radians):
    """Set joint position targets using the authored hand joint names."""
    structure = resolve_hand_structure(stage, container_path)
    if structure is None:
        print(f"[WARN] Could not resolve hand joints under {container_path}")
        return

    # Print joint order on first call
    if not hasattr(set_hand_joint_targets, '_printed_order'):
        set_hand_joint_targets._printed_order = True
        print("[JOINT ORDER]")
        for i, joint_path in enumerate(structure["ordered_joint_paths"]):
            print(f"  [{i}] {joint_path}")

    for joint_idx, joint_path in enumerate(structure["ordered_joint_paths"]):
        if joint_idx >= len(joint_values_radians):
            break
        prim = stage.GetPrimAtPath(joint_path)
        target_rad = float(joint_values_radians[joint_idx])

        if prim.IsA(UsdPhysics.RevoluteJoint):
            target_deg = target_rad * 57.29577951308232
            success = set_joint_drive_target(prim, target_deg)
        else:
            success = set_joint_drive_target(prim, target_rad)

        if not success:
            print(f"[WARN] Failed to set target for joint {joint_idx}: {joint_path}")


def set_joint_position_direct_zero(joint_prim: Usd.Prim) -> bool:
    """Directly reset a joint's authored state position to zero."""
    if not joint_prim.IsValid():
        return False

    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        state_api_kind = "angular"
    elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
        state_api_kind = "linear"
    else:
        return False

    try:
        if not joint_prim.HasAPI(PhysxSchema.JointStateAPI):
            PhysxSchema.JointStateAPI.Apply(joint_prim, state_api_kind)

        joint_state = PhysxSchema.JointStateAPI(joint_prim, state_api_kind)
        pos_attr = joint_state.GetPositionAttr()
        if not pos_attr or not pos_attr.IsValid():
            pos_attr = joint_state.CreatePositionAttr()
        pos_attr.Set(0.0)
        return True
    except Exception as e:
        print(f"[WARN] Failed to directly reset joint state on {joint_prim.GetPath()}: {e}")
        return False


def set_hand_fully_open(stage, container_path):
    """Reset the hand to the fully-open pose and keep drive targets aligned."""
    structure = resolve_hand_structure(stage, container_path)
    if structure is None:
        print(f"[WARN] Could not resolve hand structure for open reset under {container_path}")
        return

    for joint_path in structure["ordered_joint_paths"]:
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsValid():
            continue
        set_joint_position_direct_zero(joint_prim)
        set_joint_drive_target(joint_prim, 0.0)


def configure_hand_joint_drives(stage, container_path):
    """Apply the configured drive parameters to the authored hand joints only."""
    structure = resolve_hand_structure(stage, container_path)
    if structure is None:
        print(f"[WARN] Could not resolve hand structure for drive setup under {container_path}")
        return 0

    joint_count = 0
    for joint_path in structure["ordered_joint_paths"]:
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsValid():
            continue

        if joint_prim.IsA(UsdPhysics.RevoluteJoint):
            drive_kind = "angular"
            stiffness = float(JOINT_STIFFNESS) * np.pi / 180.0
            damping = float(JOINT_DAMPING) * np.pi / 180.0
            max_velocity = float(JOINT_VELOCITY_LIMIT) * 180.0 / np.pi
        elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
            drive_kind = "linear"
            stiffness = float(JOINT_STIFFNESS)
            damping = float(JOINT_DAMPING)
            max_velocity = float(JOINT_VELOCITY_LIMIT)
        else:
            continue

        try:
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_kind)

            st = drive.GetStiffnessAttr() or drive.CreateStiffnessAttr()
            st.Set(stiffness)

            dm = drive.GetDampingAttr() or drive.CreateDampingAttr()
            dm.Set(damping)

            mf = drive.GetMaxForceAttr() or drive.CreateMaxForceAttr()
            mf.Set(float(JOINT_MAX_FORCE))

            tp = drive.GetTargetPositionAttr() or drive.CreateTargetPositionAttr()
            tp.Set(0.0)

            tv = drive.GetTargetVelocityAttr() or drive.CreateTargetVelocityAttr()
            tv.Set(0.0)

            if not joint_prim.HasAPI(PhysxSchema.PhysxJointAPI):
                PhysxSchema.PhysxJointAPI.Apply(joint_prim)
            physx_joint = PhysxSchema.PhysxJointAPI(joint_prim)
            mv = physx_joint.GetMaxJointVelocityAttr() or physx_joint.CreateMaxJointVelocityAttr()
            mv.Set(max_velocity)
            arm = physx_joint.GetArmatureAttr() or physx_joint.CreateArmatureAttr()
            arm.Set(float(JOINT_ARMATURE))

            joint_count += 1
        except Exception as e:
            print(f"[WARN] Failed to configure drive for {joint_path}: {e}")

    log_info(f"Configured drive params for {joint_count} hand joints under {container_path}")
    return joint_count


def setup_physics_scene(stage, inverse_gravity=False):
    sign = -1.0 if inverse_gravity else 1.0
    
    prim = stage.GetPrimAtPath(PHYSICS_SCENE_PATH)
    if not prim.IsValid():
        prim = stage.DefinePrim(PHYSICS_SCENE_PATH, "PhysicsScene")

    if not prim.HasAPI(UsdPhysics.Scene):
        scene = UsdPhysics.Scene.Define(stage, PHYSICS_SCENE_PATH)
    else:
        scene = UsdPhysics.Scene(prim)
    
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0 * sign))
    scene.CreateGravityMagnitudeAttr().Set(20)

    if not prim.HasAPI(PhysxSchema.PhysxSceneAPI):
        PhysxSchema.PhysxSceneAPI.Apply(prim)
        
    physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(prim)
    physx_scene_api.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(32768)
    physx_scene_api.CreateGpuTotalAggregatePairsCapacityAttr().Set(32768)
    return PHYSICS_SCENE_PATH

def make_prims_editable(stage, root_prim_path):
    """
    Remove instanceable flag from all prims in hierarchy to allow editing,
    and return the list of prim paths that were instanceable so we can
    restore them later.
    """
    root_prim = stage.GetPrimAtPath(root_prim_path)
    changed_paths = []

    if not root_prim.IsValid():
        print(f"make_prims_editable: invalid root prim {root_prim_path}")
        return changed_paths

    for prim in Usd.PrimRange(root_prim):
        if prim.IsInstanceable():
            changed_paths.append(prim.GetPath().pathString)
            prim.SetInstanceable(False)

    log_info(f"Made {len(changed_paths)} instanceable prims editable under {root_prim_path}")
    return changed_paths

def restore_prims_instanceable(stage, prim_paths):
    """
    Restore instanceable=True on the given list of prim paths.
    This is meant to be used with the output of make_prims_editable.
    """
    restored = 0
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        prim.SetInstanceable(True)
        restored += 1

    log_info(f"Restored {restored} prims to instanceable")
    return restored

def set_prim_instanceable(stage, prim_path, instanceable=True):
    """Set instanceable flag on a specific prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        prim.SetInstanceable(instanceable)
        return True
    return False

def create_and_bind_high_friction_material(stage, root_prim_path):
    """Create physics material and bind to collision prims."""
    # Create material
    material_path = "/World/Physics_Materials/SuperGripMat"
    if not stage.GetPrimAtPath(material_path).IsValid():
        UsdShade.Material.Define(stage, material_path)
    mat_prim = stage.GetPrimAtPath(material_path)
    
    # Set physics properties
    if not mat_prim.HasAPI(UsdPhysics.MaterialAPI):
        UsdPhysics.MaterialAPI.Apply(mat_prim)
    
    p_mat = UsdPhysics.MaterialAPI(mat_prim)
    p_mat.CreateStaticFrictionAttr().Set(4)
    p_mat.CreateDynamicFrictionAttr().Set(4)
    p_mat.CreateRestitutionAttr().Set(0.0)

    if not mat_prim.HasAPI(PhysxSchema.PhysxMaterialAPI):
        PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
    
    physx_mat = PhysxSchema.PhysxMaterialAPI(mat_prim)
    physx_mat.CreateFrictionCombineModeAttr().Set("multiply") 

    # Bind to all mesh/collision prims
    root_prim = stage.GetPrimAtPath(root_prim_path)
    count = 0
    
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI):
            api = UsdShade.MaterialBindingAPI(prim)
            api.Bind(UsdShade.Material(mat_prim), materialPurpose="physics")
            count += 1
    
    log_info(f"Physics Material Applied to {count} Prims under {root_prim_path}")
    return count
    
def capture_object_state(stage, root_prim_path):
    snapshot = {}
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim.IsValid(): return snapshot
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Xformable):
            xform = UsdGeom.Xformable(prim)
            ops_data = {}
            for op in xform.GetOrderedXformOps():
                op_val = op.Get()
                if op_val is not None:
                    ops_data[op.GetName()] = op_val
            snapshot[str(prim.GetPath())] = ops_data
    return snapshot

def restore_object_state(stage, snapshot):
    for path, ops_data in snapshot.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid(): continue
        xform = UsdGeom.Xformable(prim)
        for op in xform.GetOrderedXformOps():
            if op.GetName() in ops_data:
                op.Set(ops_data[op.GetName()])

def setup_object_physics(stage, root_prim_path):
    """Setup physics on object - assuming it's already loaded."""
    root_prim = stage.GetPrimAtPath(root_prim_path)
    
    if root_prim.IsA(UsdGeom.Xformable):
        xform = UsdGeom.Xformable(root_prim)
        xform.ClearXformOpOrder() 

    if not root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
    
    rb_api = UsdPhysics.RigidBodyAPI(root_prim)
    rb_api.CreateKinematicEnabledAttr().Set(False)
    
    if not root_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
    
    physx_rb = PhysxSchema.PhysxRigidBodyAPI(root_prim)
    physx_rb.CreateDisableGravityAttr().Set(True)
    physx_rb.CreateContactSlopCoefficientAttr().Set(2)

    if not root_prim.HasAPI(UsdPhysics.MassAPI):
        UsdPhysics.MassAPI.Apply(root_prim)
    UsdPhysics.MassAPI(root_prim).CreateMassAttr().Set(1)

    count = 0
    # Remove physics from children, setup collision
    for prim in Usd.PrimRange(root_prim):
        path_str = str(prim.GetPath())
        prim_name = prim.GetName().lower()
        
        if path_str == root_prim_path:
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
        if prim.HasAPI(UsdPhysics.MassAPI):
            prim.RemoveAPI(UsdPhysics.MassAPI)

        # Keep purely visual meshes out of the collision authoring pass.
        if prim.IsA(UsdGeom.Mesh) and "visuals" in prim_name:
            continue

        if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdGeom.Mesh):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
             
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_col_api = UsdPhysics.MeshCollisionAPI(prim)
            mesh_col_api.CreateApproximationAttr().Set("convexDecomposition")

            if not prim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI):
                PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            decomp_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI(prim)
            decomp_api.CreateMinThicknessAttr().Set(CONVEX_DECOMP_MIN_THICKNESS)

            if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                PhysxSchema.PhysxCollisionAPI.Apply(prim)
            phys_col = PhysxSchema.PhysxCollisionAPI(prim)
            phys_col.CreateContactOffsetAttr().Set(0.002) 
            phys_col.CreateRestOffsetAttr().Set(0)
            count += 1

    # Set initial pose
    xformable = UsdGeom.Xformable(root_prim)
    xformable.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0)) 
    xformable.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
    
    log_info(f"Applied collision to {count} object meshes")
    return

def load_target_usd_env0(stage, usd_path):
    """Load object into env_0 with wrapper/ref structure."""
    log_info(f"Loading Object USD into env_0: {usd_path}")
    if not os.path.exists(usd_path): 
        return None, None

    # Create wrapper (non-instanceable)
    wrapper_path = obj_wrap(0)
    ref_path = obj_ref(0)
    
    UsdGeom.Xform.Define(stage, wrapper_path)
    # Add reference under ref prim
    add_reference_to_stage(usd_path=usd_path, prim_path=ref_path)
    
    # Make ref editable temporarily to setup physics
    make_prims_editable(stage, ref_path)
    
    # Setup physics on the ref
    setup_object_physics(stage, wrapper_path)
    
    # Apply material
    create_and_bind_high_friction_material(stage, wrapper_path)
    
    # Capture initial state
    initial_snapshot = capture_object_state(stage, wrapper_path)
    
    # NOW make ref instanceable
    #set_prim_instanceable(stage, ref_path, instanceable=True)
    
    return ref_path, initial_snapshot

def set_all_gravity(stage, paths, enable_gravity):
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
            
        if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            
        physx_rb = PhysxSchema.PhysxRigidBodyAPI(prim)
        physx_rb.CreateDisableGravityAttr().Set(not enable_gravity)
        
def clear_all_velocities(stage, paths):
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
            
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(prim)
            
        rb = UsdPhysics.RigidBodyAPI(prim)
        rb.CreateVelocityAttr().Set(Gf.Vec3f(0,0,0))
        rb.CreateAngularVelocityAttr().Set(Gf.Vec3f(0,0,0))

def set_local_pose(stage, prim_path, pos, quat):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    xform = UsdGeom.Xformable(prim)

    # Get or add local translate/orient ops
    ops = xform.GetOrderedXformOps()
    translate_op = None
    orient_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient_op = op

    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    if orient_op is None:
        orient_op = xform.AddOrientOp()

    translate_op.Set(Gf.Vec3f(*pos))
    orient_op.Set(Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))

def get_prim_pose_robust(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid(): return np.zeros(3), np.array([1,0,0,0])
    xformable = UsdGeom.Xformable(prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = world_transform.ExtractTranslation()
    r = world_transform.ExtractRotationQuat()
    return np.array([t[0], t[1], t[2]]), np.array([r.GetReal(), r.GetImaginary()[0], r.GetImaginary()[1], r.GetImaginary()[2]])

def load_hand_env0(stage, usd_path):
    """Load hand into env_0 with wrapper/ref structure."""
    log_info(f"Loading Hand USD into env_0: {usd_path}")
    
    wrapper_path = hand_wrap(0)
    ref_path = hand_ref(0)
    UsdGeom.Xform.Define(stage, wrapper_path)
    
    # Add reference
    add_reference_to_stage(usd_path=usd_path, prim_path=ref_path)
    
    # Allow authored descendants to receive runtime material-binding overrides.
    make_prims_editable(stage, ref_path)
    
    # Apply material
    create_and_bind_high_friction_material(stage, ref_path)

    # Keep the authored hand geometry/collision, but ensure the runtime joint
    # drives use the configured PD/limit values expected by this script.
    configure_hand_joint_drives(stage, ref_path)

    hand_structure = resolve_hand_structure(stage, ref_path)
    if hand_structure is None:
        raise RuntimeError(f"Failed to resolve hand structure in {ref_path}")

    num_dofs = len(hand_structure["ordered_joint_paths"])
    return ref_path, num_dofs

def slerp(q1, q2, t):
    q = (1.0 - t) * q1 + t * q2
    return q / np.linalg.norm(q)

#Part detection and annotation helpers
def precompute_mesh_samples(stage, object_ref_path: str, max_points_per_mesh: int = 8000) -> dict:
    """Sample mesh vertices in LOCAL (template) coordinates before instancing"""
    mesh_samples = {}
    
    ref_prim = stage.GetPrimAtPath(object_ref_path)
    if not ref_prim.IsValid():
        return mesh_samples
    
    for prim in Usd.PrimRange(ref_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        if len(points) == 0:
            continue
        
        # Downsample if too many points
        if len(points) > max_points_per_mesh:
            step = max(1, len(points) // max_points_per_mesh)
            points = points[::step]
        
        # Store in LOCAL coordinates
        local_points = np.array([[float(p[0]), float(p[1]), float(p[2])] 
                                  for p in points], dtype=np.float32)
        
        # Relative path for matching across clones
        rel_path = str(prim.GetPath()).replace(object_ref_path, "")
        mesh_samples[rel_path] = local_points
    
    return mesh_samples

def compute_probe_offsets_in_gripper_frame(stage, gripper_ref_path: str) -> np.ndarray:
    """Compute probe positions relative to gripper base in LOCAL gripper coordinates"""
    hand_structure = resolve_hand_structure(stage, gripper_ref_path)
    if hand_structure is None:
        return np.empty((0, 3), dtype=np.float32)

    gripper_base_path = hand_structure["palm_path"]
    gripper_base = stage.GetPrimAtPath(gripper_base_path)
    
    if not gripper_base.IsValid():
        return np.empty((0, 3), dtype=np.float32)
    
    probe_offsets = []
    for probe_path in PROBES:
        probe_prim = stage.GetPrimAtPath(probe_path)
        if not probe_prim.IsValid():
            continue
        
        # Get transforms
        base_xform = UsdGeom.Xformable(gripper_base).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        probe_xform = UsdGeom.Xformable(probe_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
        # Probe position in gripper-local frame
        probe_world = probe_xform.Transform(Gf.Vec3d(0, 0, 0))
        base_inv = base_xform.GetInverse()
        probe_local = base_inv.Transform(probe_world)
        
        probe_offsets.append([float(probe_local[0]), float(probe_local[1]), float(probe_local[2])])
    
    return np.array(probe_offsets, dtype=np.float32)

def transform_probes_by_grasp_pose(probe_offsets: np.ndarray, 
                                    grasp_pos: Gf.Vec3d, 
                                    grasp_quat: Gf.Quatd) -> np.ndarray:
    """Transform probe offsets from gripper frame to object-local frame"""
    rot = Gf.Rotation(grasp_quat)
    probe_positions = np.empty_like(probe_offsets)
    
    for i, offset in enumerate(probe_offsets):
        offset_vec = Gf.Vec3d(float(offset[0]), float(offset[1]), float(offset[2]))
        rotated = rot.TransformDir(offset_vec)
        world_pos = grasp_pos + rotated
        probe_positions[i] = [float(world_pos[0]), float(world_pos[1]), float(world_pos[2])]
    
    return probe_positions

def batch_match_probes_to_meshes_local(probe_positions: np.ndarray,
                                       mesh_samples: dict) -> list:
    """Match probes to meshes - all in same local coordinate frame"""
    results = []
    
    for probe_pos in probe_positions:
        best_mesh = None
        best_dist = np.inf
        
        for mesh_rel_path, points in mesh_samples.items():
            if points.shape[0] == 0:
                continue
            
            diffs = points - probe_pos[None, :]
            distances = np.sqrt(np.sum(diffs**2, axis=1))
            min_dist = float(np.min(distances))
            
            if min_dist < best_dist:
                best_dist = min_dist
                best_mesh = mesh_rel_path
        
        results.append((best_mesh, best_dist))
    
    return results

def extract_part_from_mesh_path(mesh_rel_path: str, parts_list: list) -> str:
    """
    Extract part name from relative mesh path.
    Match against known parts to find which part this mesh belongs to.
    """
    if not mesh_rel_path:
        return "body"
    
    # Try to match mesh path against known parts
    for part_path, part_name in parts_list:
        # Check if mesh path contains the part name
        if part_name in mesh_rel_path:
            return part_name
    
    # Fallback: try to extract from path structure
    parts = mesh_rel_path.strip("/").split("/")
    # Could be /World/Scan/handle/mesh or /Scan/handle/mesh or /category/handle/mesh
    # Look for the deepest non-mesh name
    for i in range(len(parts) - 1, -1, -1):
        part_candidate = parts[i]
        if not part_candidate.startswith("mesh") and not part_candidate.startswith("Mesh"):
            # Check if this matches any known part
            for _, part_name in parts_list:
                if part_candidate == part_name or part_candidate.lower() == part_name.lower():
                    return part_name
    
    return "body"

def get_parts_list(stage, object_ref_path: str) -> list:
    """
    Get list of (part_path, part_name) tuples.
    Structure: object_ref_path -> child (e.g., /World or /Scan) -> parts
    """
    parts = []
    ref_prim = stage.GetPrimAtPath(object_ref_path)
    
    if not ref_prim.IsValid():
        return parts
    
    # Get children of object_ref
    children = list(ref_prim.GetChildren())
    
    if len(children) == 0:
        return parts

    parts_container = None
    for child in children:
        # Look for a prim that has mesh-containing children
        grandchildren = list(child.GetChildren())
        if len(grandchildren) > 1:  # Multiple parts indicate this is the container
            parts_container = child
            break
    
    if not parts_container:
        # Fallback: if only one child, check if it has parts
        if len(children) == 1:
            parts_container = children[0]
    
    if not parts_container:
        return parts
    
    # Now get parts from the container
    for part_prim in parts_container.GetChildren():
        if not (part_prim.IsA(UsdGeom.Xform) or part_prim.IsA(UsdGeom.Boundable)):
            continue
        
        has_mesh = any(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(part_prim))
        if not has_mesh:
            continue
        
        part_name = part_prim.GetName()
        
        part_path = part_prim.GetPath().pathString
        parts.append((part_path, part_name))
    
    if len(parts) > MAX_PARTS:
        print(f"[INFO] Limiting parts from {len(parts)} to {MAX_PARTS}")
        parts = parts[:MAX_PARTS]
    
    return parts


# ================= BATCHED EVALUATION =================

class EnvState:
    """State for a single environment."""
    def __init__(self):
        self.state = -1  # Current state
        self.step_counter = 0
        self.is_failed = False
        self.fail_reason = ""
        self.contact_checked = False
        
        self.initial_z_height = 0.0
        
        # Current commands
        self.cmd_pos = np.zeros(3)
        self.cmd_rot = np.array([1,0,0,0])
        self.cmd_joints = None  # Will be initialized
        
        # Start poses for interpolation
        self.start_pos = np.zeros(3)
        self.start_rot = np.array([1,0,0,0])
        self.start_joints = None
        
        # Frozen joints for holding
        self.frozen_joints = None
        
        # Grasp stages for current grasp
        self.g_coarse_pos = None
        self.g_coarse_rot = None
        self.g_coarse_joints = None
        
        self.g_fine_pos = None
        self.g_fine_rot = None
        self.g_fine_joints = None
        
        self.g_final_pos = None
        self.g_final_rot = None
        self.g_final_joints = None
        
        # Success tracking
        self.success = False
        self.contacted_part = "body"


def _to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()
    if hasattr(data, "numpy"):
        return data.numpy()
    return np.asarray(data)


def _get_retrieval_force_direction(mode):
    if mode == "pos_x":
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if mode == "neg_x":
        return np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    if mode == "pos_y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if mode == "neg_y":
        return np.array([0.0, -1.0, 0.0], dtype=np.float32)
    if mode == "neg_z":
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)


def _resolve_retrieval_force_mode(mode, frame_idx=None, total_frames=None):
    if mode != "all":
        return mode
    if total_frames is None or total_frames <= 0 or frame_idx is None:
        raise ValueError("'all' retrieval force mode requires frame_idx and total_frames")
    segment_idx = min(
        int(frame_idx * len(RETRIEVAL_FORCE_ALL_MODES) / float(total_frames)),
        len(RETRIEVAL_FORCE_ALL_MODES) - 1,
    )
    return RETRIEVAL_FORCE_ALL_MODES[segment_idx]


def build_simple_retrieval_force_array(
    active,
    num_copies,
    mode,
    mag,
    units="newton",
    masses=None,
    frame_idx=None,
    total_frames=None,
):
    forces = np.zeros((num_copies, 3), dtype=np.float32)
    resolved_mode = _resolve_retrieval_force_mode(mode, frame_idx=frame_idx, total_frames=total_frames)
    direction = _get_retrieval_force_direction(resolved_mode)
    if not np.any(direction):
        return forces

    if units == "body_weight":
        if masses is None:
            raise ValueError("masses are required when RETRIEVAL_FORCE_UNITS == 'body_weight'")
        magnitudes = _to_numpy(masses).reshape(num_copies) * float(mag) * GRAVITY_ACCEL
    else:
        magnitudes = np.full((num_copies,), float(mag), dtype=np.float32)

    for k in range(min(len(active), num_copies)):
        if active[k]:
            forces[k] = direction * magnitudes[k]

    return forces


def create_object_force_tensor_view(num_copies: int):
    backend = SimulationManager.get_backend()
    sim_view = omni.physics.tensors.create_simulation_view(
        backend,
        stage_id=get_current_stage_id(),
    )
    sim_view.set_subspace_roots("/")

    rigid_body_view = sim_view.create_rigid_body_view([obj_wrap(i) for i in range(num_copies)])
    if rigid_body_view.count != num_copies:
        raise RuntimeError(
            f"object_force_view count mismatch: expected {num_copies}, got {rigid_body_view.count}"
        )

    return {
        "sim_view": sim_view,
        "rigid_body_view": rigid_body_view,
        "indices": np.arange(num_copies, dtype=np.uint32),
        "masses": _to_numpy(rigid_body_view.get_masses()).reshape(num_copies),
        "num_copies": num_copies,
    }


def apply_retrieval_forces(force_view_state, forces):
    try:
        force_view_state["rigid_body_view"].apply_forces(
            forces,
            force_view_state["indices"],
            True,
        )
        return force_view_state
    except Exception as exc:
        msg = str(exc)
        if "Failed to apply forces" not in msg and "invalidated" not in msg:
            raise

        rebound_state = create_object_force_tensor_view(force_view_state["num_copies"])
        rebound_state["rigid_body_view"].apply_forces(
            forces,
            rebound_state["indices"],
            True,
        )
        return rebound_state

async def evaluate_grasps_batched(stage, timeline, grasp_list, num_dofs, initial_snapshot, inverse_gravity=False):
    """Evaluate grasps in batches using parallel environments."""
    
    total_grasps = len(grasp_list)
    batch_size = NUM_ENVS
    num_batches = (total_grasps + batch_size - 1) // batch_size
    
    log_info(f"Starting batched evaluation: {total_grasps} grasps, {num_batches} batches")
    
    # # --- Part detection pre-computation (done once, before batch loop) ---
    # mesh_samples = precompute_mesh_samples(stage, obj_ref(0))
    # parts_list = get_parts_list(stage, obj_ref(0))
    # probe_offsets = compute_probe_offsets_in_gripper_frame(stage, hand_ref(0))
    # log_info(f"Part detection ready: {len(mesh_samples)} meshes, {len(parts_list)} parts, {len(probe_offsets)} probes")
        # Stubs so the rest of the code doesn't break
    mesh_samples = {}
    parts_list = []
    probe_offsets = np.empty((0, 3), dtype=np.float32)
    results = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_grasps)
        batch_grasps = grasp_list[start_idx:end_idx]
        batch_size_actual = len(batch_grasps)
        
        print(f"\n{'='*60}")
        print(f"BATCH {batch_idx + 1}/{num_batches}: Grasps {start_idx} to {end_idx-1}")
        print(f"{'='*60}")
        
        # Initialize environment states
        env_states = [EnvState() for _ in range(batch_size_actual)]
        
        for i in range(batch_size_actual):
            env_states[i].cmd_joints = np.zeros(num_dofs)
            env_states[i].start_joints = np.zeros(num_dofs)
            env_states[i].frozen_joints = np.zeros(num_dofs)
        
        # Parse grasp stages for all environments in batch
        nominal_obj_pos = np.array([0.0, 0.0, 0.0])
        nominal_obj_rot = np.array([1.0, 0.0, 0.0, 0.0])
        
        for i, grasp_data in enumerate(batch_grasps):
            env_i = env_states[i]
            
            # Parse all three stages
            coarse_data = grasp_data["coarse_grasp"]
            fine_data = grasp_data["fine_grasp"]
            final_data = grasp_data["final_grasp"]
            
            # Transform to world space using nominal pose
            def transform_stage(stage_data):
                rel_pos = np.array(stage_data["position"])
                rel_rot = np.array(stage_data["orientation"])
                joints = np.array(stage_data["joints"])
                
                world_pos, world_rot = transform_pose_to_world(
                    rel_pos, rel_rot, nominal_obj_pos, nominal_obj_rot
                )
                
                return world_pos, world_rot, joints
            
            env_i.g_coarse_pos, env_i.g_coarse_rot, env_i.g_coarse_joints = transform_stage(coarse_data)
            env_i.g_fine_pos, env_i.g_fine_rot, env_i.g_fine_joints = transform_stage(fine_data)
            env_i.g_final_pos, env_i.g_final_rot, env_i.g_final_joints = transform_stage(final_data)
        
        # Reset hand pose and joints before playback so the first live frame starts open.
        for env_idx in range(batch_size_actual):
            env = env_states[env_idx]
            set_local_pose(stage, hand_wrap(env_idx), env.g_coarse_pos, env.g_coarse_rot)
            set_hand_fully_open(stage, hand_ref(env_idx))

        # Run batch evaluation
        all_done = False
        
        timeline.play()
        object_force_state = None
        
        while not all_done:
            # Update all environments
            for env_idx in range(batch_size_actual):
                env = env_states[env_idx]
                
                # Skip if already done
                if env.state == 999:
                    continue
                
                # Get paths for this environment
                obj_path = obj_wrap(env_idx)
                hand_path = hand_ref(env_idx)
                
                # Handle failure - move to next grasp
                if env.is_failed and env.state != -1:
                    print(f"  [Env {env_idx}] Failed: {env.fail_reason}")
                    env.state = 999  # Mark as done
                    env.success = False
                    continue
                
                # State -1: RESET
                if env.state == -1:
                    set_all_gravity(stage, [obj_path], enable_gravity=False)
                    
                    if env.step_counter == 0:
                        print(f"  [Env {env_idx}] RESET")
                        env.contact_checked = False
                        set_hand_fully_open(stage, hand_path)
                    
                    # Restore object state
                    # Note: For cloned instanceable refs, we reset via parent wrapper
                    obj_wrapper = obj_wrap(env_idx)
                    obj_wrapper_prim = stage.GetPrimAtPath(obj_wrapper)
                    if obj_wrapper_prim.IsValid() and obj_wrapper_prim.IsA(UsdGeom.Xformable):
                        xformable = UsdGeom.Xformable(obj_wrapper_prim)
                        # Clear any existing transforms
                        xformable.ClearXformOpOrder()
                        # Set to identity
                        xformable.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0))
                        xformable.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
                    
                    clear_all_velocities(stage, [obj_path])
                    
                    env.cmd_pos = env.g_coarse_pos
                    env.cmd_rot = env.g_coarse_rot
                    env.cmd_joints = np.zeros(num_dofs)
                    
                    if env.step_counter > 60:
                        p_reset, _ = get_prim_pose_robust(stage, obj_path)
                        env.initial_z_height = p_reset[2]
                        
                        env.state = 1
                        env.step_counter = 0
                        env.start_pos = env.cmd_pos
                        env.start_rot = env.cmd_rot
                        env.start_joints = env.cmd_joints
                
                # State 1: MOVE TO COARSE
                elif env.state == 1:
                    alpha = min(env.step_counter / float(COARSE_STEPS), 1.0)
                    
                    env.cmd_pos = interp_pos(env.start_pos, env.g_coarse_pos, alpha)
                    env.cmd_rot = slerp(env.start_rot, env.g_coarse_rot, alpha)
                    env.cmd_joints = interp_joints(env.start_joints, env.g_coarse_joints, alpha)
                    
                    if env.step_counter >= COARSE_STEPS:
                        env.state = 2
                        env.step_counter = 0
                        env.start_pos = env.g_coarse_pos
                        env.start_rot = env.g_coarse_rot
                        env.start_joints = env.g_coarse_joints
                
                # State 2: MOVE TO FINE
                elif env.state == 2:
                    alpha = min(env.step_counter / float(FINE_STEPS), 1.0)
                    
                    env.cmd_pos = interp_pos(env.start_pos, env.g_fine_pos, alpha)
                    env.cmd_rot = slerp(env.start_rot, env.g_fine_rot, alpha)
                    env.cmd_joints = interp_joints(env.start_joints, env.g_fine_joints, alpha)
                    
                    if env.step_counter >= FINE_STEPS:
                        env.state = 3
                        env.step_counter = 0
                        env.start_pos = env.g_fine_pos
                        env.start_rot = env.g_fine_rot
                        env.start_joints = env.g_fine_joints
                
                # State 3: MOVE TO FINAL
                elif env.state == 3:
                    alpha = min(env.step_counter / float(FINAL_STEPS), 1.0)
                    
                    env.cmd_pos = interp_pos(env.start_pos, env.g_final_pos, alpha)
                    env.cmd_rot = slerp(env.start_rot, env.g_final_rot, alpha)
                    env.cmd_joints = interp_joints(env.start_joints, env.g_final_joints, alpha)
                    
                    if env.step_counter >= FINAL_STEPS:
                        env.frozen_joints = env.g_final_joints
                        env.step_counter = 0
                        set_all_gravity(stage, [obj_path], enable_gravity=True)
                        # # --- Part detection ---
                        # if probe_offsets.shape[0] > 0 and mesh_samples:
                        #     r = env.g_final_rot  # [w, x, y, z]
                        #     gf_pos = Gf.Vec3d(float(env.g_final_pos[0]),
                        #                       float(env.g_final_pos[1]),
                        #                       float(env.g_final_pos[2]))
                        #     gf_quat = Gf.Quatd(float(r[0]),
                        #                        Gf.Vec3d(float(r[1]), float(r[2]), float(r[3])))
                        #     probe_world = transform_probes_by_grasp_pose(probe_offsets, gf_pos, gf_quat)
                        #     matched = batch_match_probes_to_meshes_local(probe_world, mesh_samples)
                        #     if matched:
                        #         best_mesh_path, _ = min(matched, key=lambda x: x[1])
                        #         env.contacted_part = extract_part_from_mesh_path(best_mesh_path, parts_list)
                        #     print(f"  [Env {env_idx}] Contacted part: {env.contacted_part}")

                        env.state = 3.5
                
                # State 3.5: SETTLE WITH GRAVITY
                elif env.state == 3.5:
                    env.cmd_pos = env.g_final_pos
                    env.cmd_rot = env.g_final_rot
                    env.cmd_joints = env.frozen_joints
                    
                    if env.step_counter >= CHECK_STEPS:
                        p_now, _ = get_prim_pose_robust(stage, obj_path)
                        if inverse_gravity:
                            if p_now[2] > (env.initial_z_height + DROP_HEIGHT_THRESHOLD):
                                env.is_failed = True
                                env.fail_reason = "Dropped during settling"
                            else:
                                env.state = 4
                                env.step_counter = 0
                        else:
                            if p_now[2] < (env.initial_z_height - DROP_HEIGHT_THRESHOLD):
                                env.is_failed = True
                                env.fail_reason = "Dropped during settling"
                            else:
                                env.state = 4
                                env.step_counter = 0
                
                # State 4: FREEZE & CHECK
                elif env.state == 4:
                    env.cmd_pos = env.g_final_pos
                    env.cmd_rot = env.g_final_rot
                    env.cmd_joints = env.frozen_joints
                    
                    if env.step_counter >= HOLD_STEPS:
                        env.state = 6
                        env.step_counter = 0
                        env.start_pos = env.g_final_pos
                        env.start_rot = env.g_final_rot
                
                # State 6: LIFT TEST
                elif env.state == 6:
                    alpha = min(env.step_counter / float(LIFT_STEPS), 1.0)
                    if inverse_gravity:
                        target_lift_pos = env.g_final_pos + np.array([0, 0, -LIFT_HEIGHT])
                    else:
                        target_lift_pos = env.g_final_pos + np.array([0, 0, LIFT_HEIGHT])
                    env.cmd_pos = interp_pos(env.g_final_pos, target_lift_pos, alpha)
                    env.cmd_rot = env.g_final_rot
                    env.cmd_joints = env.frozen_joints
                    
                    if env.step_counter >= LIFT_STEPS:
                        p_now, _ = get_prim_pose_robust(stage, obj_path)
                        if inverse_gravity:
                            expected_z = env.initial_z_height - LIFT_HEIGHT
                            if p_now[2] > (expected_z + DROP_HEIGHT_THRESHOLD):
                                env.is_failed = True
                                env.fail_reason = "Dropped during lift"
                            else:
                                env.success = True
                                print(f"  [Env {env_idx}] SUCCESS!")
                            
                            env.state = 999  # Mark as done
                            env.step_counter = 0
                        else:                           
                            expected_z = env.initial_z_height + LIFT_HEIGHT
                            if p_now[2] < (expected_z - DROP_HEIGHT_THRESHOLD):
                                env.is_failed = True
                                env.fail_reason = "Dropped during lift"
                            else:
                                env.success = True
                                print(f"  [Env {env_idx}] SUCCESS!")
                            
                            env.state = 999  # Mark as done
                            env.step_counter = 0
                
                # Apply commands for active states
                if env.state in [-1, 1, 2, 3, 6]:
                    hand_wrapper_path = hand_wrap(env_idx)
                    set_local_pose(stage, hand_wrapper_path, env.cmd_pos, env.cmd_rot)
                
                set_hand_joint_targets(stage, hand_path, env.cmd_joints)
                
                env.step_counter += 1
            
            # Single physics step for ALL environments
            active_retrieval = [False] * NUM_ENVS
            retrieval_frame_idx = None
            for env_idx in range(batch_size_actual):
                env = env_states[env_idx]
                if env.state == 6 and not env.is_failed:
                    active_retrieval[env_idx] = True
                    if retrieval_frame_idx is None:
                        retrieval_frame_idx = env.step_counter

            if retrieval_frame_idx is not None:
                if object_force_state is None:
                    # Tensor views require an active PhysX scene, so defer creation
                    # until retrieval starts after the simulation is already running.
                    object_force_state = create_object_force_tensor_view(NUM_ENVS)
                forces = build_simple_retrieval_force_array(
                    active=active_retrieval,
                    num_copies=NUM_ENVS,
                    mode=RETRIEVAL_FORCE_TEST_MODE,
                    mag=RETRIEVAL_FORCE_MAG,
                    units=RETRIEVAL_FORCE_UNITS,
                    masses=object_force_state["masses"],
                    frame_idx=retrieval_frame_idx,
                    total_frames=LIFT_STEPS + 1,
                )
                object_force_state = apply_retrieval_forces(object_force_state, forces)

            await omni.kit.app.get_app().next_update_async()
            
            # Check if all done
            all_done = all(env.state == 999 for env in env_states)
        
        timeline.stop()
        
        # Collect results
        for i, env in enumerate(env_states):
            grasp_idx = start_idx + i
            results.append({
                "grasp_index": grasp_idx,
                "success": env.success,
                "fail_reason": env.fail_reason if not env.success else "",
                "contacted_part": env.contacted_part
            })
        
        print(f"\nBatch {batch_idx + 1} complete: {sum(1 for e in env_states if e.success)}/{batch_size_actual} succeeded")
    
    return results

async def run_first_pass():
    import gc
    ctx = omni.usd.get_context()
    timeline = get_timeline_interface()
    if ctx.get_stage():
        await ctx.close_stage_async()
        await omni.kit.app.get_app().next_update_async()
        gc.collect()
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
    await ctx.new_stage_async()
    stage = ctx.get_stage()

    setup_physics_scene(stage)
    # Add ground and lighting
    GroundPlane("/World/GroundPlane", z_position=-10.0)
    add_lighting(stage)
    
    # Create base environment container and env_0 root (match working cloner example)
    if not stage.GetPrimAtPath(ENV_BASE_PATH).IsValid():
        UsdGeom.Scope.Define(stage, ENV_BASE_PATH)
    if not stage.GetPrimAtPath(env_path(0)).IsValid():
        UsdGeom.Xform.Define(stage, env_path(0))
    
    # ============ LOAD ENV_0 ============
    log_info("Setting up Environment 0...")
    
    # Load object into env_0
    obj_ref_0, initial_snapshot = load_target_usd_env0(stage, OBJECT_USD_PATH)
    if obj_ref_0 is None:
        log_info("Failed to load object")
        return
    
    bottom_center = compute_bottom_center(stage, obj_wrap(0))
    log_info(f"Computed bottom_center: {bottom_center}")
    
    # Load hand into env_0
    hand_ref_0, num_dofs = load_hand_env0(stage, ROBOT_USD_PATH)
    log_info(f"Hand has {num_dofs} DOFs")
    
    # ============ CLONE ENVIRONMENTS ============
    log_info(f"Cloning template env_0 into grid (NUM_ENVS={NUM_ENVS}, spacing={CLONE_SPACING})...")
    
    cloner = GridCloner(spacing=CLONE_SPACING)
    cloner.define_base_env(ENV_BASE_PATH)
    
    # /World/Envs/env_0 ... /World/Envs/env_{NUM_ENVS-1}
    env_paths = cloner.generate_paths(ENV_ROOT_PREFIX, NUM_ENVS)
    log_info(f"[DEBUG] Cloner env_paths: {env_paths}")
    
    cloner.clone(
        source_prim_path=env_path(0),
        prim_paths=env_paths,
    )
    
    # Give the stage a couple of updates so transforms propagate
    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()
    
    # Debug: confirm env transforms / spacing
    p0, _ = get_prim_pose_robust(stage, env_path(0))
    if NUM_ENVS > 1:
        p1, _ = get_prim_pose_robust(stage, env_path(1))
        log_info(f"[DEBUG] env_1 world pos: {p1} (delta ~ {p1 - p0})")
    
    log_info(f"Created {NUM_ENVS} parallel environments")
    
    # ============ LOAD GRASPS ============
    grasp_list = load_grasp_data_json(GRASP_JSON_PATH)
    if grasp_list is None or len(grasp_list) == 0:
        log_info("No grasps to evaluate")
        return

    if FILTER_GRASPS_Z_DOWN_ONLY:
        grasp_list = filter_grasps_by_stage_z_down(
            grasp_list,
            stage_key=GRASP_Z_DOWN_STAGE,
            threshold=GRASP_Z_DOWN_THRESHOLD,
        )
        if len(grasp_list) == 0:
            log_info("No grasps to evaluate after Z-down filtering")
            return
    
    # ============ RUN BATCHED EVALUATION ============
    results = await evaluate_grasps_batched(
        stage, 
        timeline, 
        grasp_list, 
        num_dofs, 
        initial_snapshot
    )
    
    successful_grasps = [
        grasp_list[r["grasp_index"]]
        for r in results if r["success"]
    ]
    log_info(f"First pass survivors: {len(successful_grasps)}/{len(grasp_list)}")
    
    successes = len(successful_grasps)
    total_input = len(grasp_list)
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total Input Grasps: {total_input}")
    print(f"Final Successes: {successes}")
    print(f"Success Rate: {100.0 * successes / total_input:.1f}%")
    print(f"{'='*60}")
    
    # Save results
    output_file = Path(GRASP_JSON_PATH).parent / "dex3_1_grasp_pose.json"

    output_data = {
        "type": Path(OBJECT_USD_PATH).parent.name,
        "bottom_center": bottom_center,
        "functional_grasp": {
            "body": successful_grasps
        },
        "grasp": {}
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    log_info(f"Results saved to: {output_file}")

async def run_second_pass():
    
    import gc
    ctx = omni.usd.get_context()
    if ctx.get_stage():
        await ctx.close_stage_async()
        await omni.kit.app.get_app().next_update_async()
        gc.collect()
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
    await ctx.new_stage_async()
    stage = ctx.get_stage()
    timeline = get_timeline_interface()
    setup_physics_scene(stage, INVERSE_GRAVITY)
    
    # Add ground and lighting
    # GroundPlane("/World/GroundPlane", z_position=10.0)
    add_lighting(stage)
    
    # Create base environment container and env_0 root (match working cloner example)
    if not stage.GetPrimAtPath(ENV_BASE_PATH).IsValid():
        UsdGeom.Scope.Define(stage, ENV_BASE_PATH)
    if not stage.GetPrimAtPath(env_path(0)).IsValid():
        UsdGeom.Xform.Define(stage, env_path(0))
    
    # ============ LOAD ENV_0 ============
    log_info("Setting up Environment 0...")
    
    # Load object into env_0
    obj_ref_0, initial_snapshot = load_target_usd_env0(stage, OBJECT_USD_PATH)
    if obj_ref_0 is None:
        log_info("Failed to load object")
        return
    
    bottom_center = compute_bottom_center(stage, obj_wrap(0))
    log_info(f"Computed bottom_center: {bottom_center}")
    
    # Load hand into env_0
    hand_ref_0, num_dofs = load_hand_env0(stage, ROBOT_USD_PATH)
    log_info(f"Hand has {num_dofs} DOFs")
    
    # ============ CLONE ENVIRONMENTS ============
    log_info(f"Cloning template env_0 into grid (NUM_ENVS={NUM_ENVS}, spacing={CLONE_SPACING})...")
    
    cloner = GridCloner(spacing=CLONE_SPACING)
    cloner.define_base_env(ENV_BASE_PATH)
    
    # /World/Envs/env_0 ... /World/Envs/env_{NUM_ENVS-1}
    env_paths = cloner.generate_paths(ENV_ROOT_PREFIX, NUM_ENVS)
    log_info(f"[DEBUG] Cloner env_paths: {env_paths}")
    
    cloner.clone(
        source_prim_path=env_path(0),
        prim_paths=env_paths,
    )
    
    # Give the stage a couple of updates so transforms propagate
    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()
    
    # Debug: confirm env transforms / spacing
    p0, _ = get_prim_pose_robust(stage, env_path(0))
    if NUM_ENVS > 1:
        p1, _ = get_prim_pose_robust(stage, env_path(1))
        log_info(f"[DEBUG] env_1 world pos: {p1} (delta ~ {p1 - p0})")
    
    log_info(f"Created {NUM_ENVS} parallel environments")
    
    # ============ LOAD GRASPS ============
    output_file = Path(GRASP_JSON_PATH).parent / "dex3_1_grasp_pose.json"
    grasp_list = load_grasp_data_json_second_pass(output_file)
    if grasp_list is None or len(grasp_list) == 0:
        log_info("No grasps to evaluate")
        return

    if FILTER_GRASPS_Z_DOWN_ONLY:
        grasp_list = filter_grasps_by_stage_z_down(
            grasp_list,
            stage_key=GRASP_Z_DOWN_STAGE,
            threshold=GRASP_Z_DOWN_THRESHOLD,
        )
        if len(grasp_list) == 0:
            log_info("No grasps to evaluate after Z-down filtering")
            return
    
    # ============ RUN BATCHED EVALUATION ============
    results = await evaluate_grasps_batched(
        stage, 
        timeline, 
        grasp_list, 
        num_dofs, 
        initial_snapshot,
        INVERSE_GRAVITY
    )
    
    successful_grasps = [
        grasp_list[r["grasp_index"]]
        for r in results if r["success"]
    ]
    log_info(f"Second pass survivors: {len(successful_grasps)}/{len(grasp_list)}")
    
    successes = len(successful_grasps)
    total_input = len(grasp_list)
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total Input Grasps: {total_input}")
    print(f"Final Successes: {successes}")
    print(f"Success Rate: {100.0 * successes / total_input:.1f}%")
    print(f"{'='*60}")
    
    # Save results
    output_file = Path(GRASP_JSON_PATH).parent / "dex3_1_grasp_pose.json"

    output_data = {
        "type": Path(OBJECT_USD_PATH).parent.name,
        "bottom_center": bottom_center,
        "functional_grasp": {
            "body": successful_grasps
        },
        "grasp": {}
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    log_info(f"Results saved to: {output_file}")

async def run_pipeline():
    run_transform_unit_tests()
    await run_first_pass()
    if INVERSE_GRAVITY:
        await run_second_pass()

# ============ MAIN FUNCTION ============
def main():
    # Schedule the async pipeline
    task = asyncio.ensure_future(run_pipeline())
        
    while not task.done():
        simulation_app.update()
    
    if task.exception():
        raise task.exception()
    
    simulation_app.close()

# ============ ENTRY POINT ============
if __name__ == "__main__":
    main()
