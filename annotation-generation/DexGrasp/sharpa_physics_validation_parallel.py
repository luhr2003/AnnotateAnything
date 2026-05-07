from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

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
import os
import shutil
from collections import defaultdict

import omni.kit.app
import omni.usd
from pxr import Usd, UsdGeom, Sdf, Gf, UsdPhysics, UsdShade, PhysxSchema, UsdLux
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
from omni.timeline import get_timeline_interface
from isaacsim.core.api.robots import Robot
from isaacsim.core.cloner import GridCloner


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
    for g in grasps:
        for stage_key in ("coarse_grasp", "fine_grasp", "final_grasp"):
            joints = g[stage_key]["joints"]
            arr = joints_to_array(joints)  # handles both dict {joint_name: val} and list
            g[stage_key]["joints"] = [arr[i] for i in INVERSE_JOINT_REORDER_INDICES]
    log_info(f"Loaded {len(grasps)} previously successful grasps.")
    return grasps

# ================= CONFIGURATIONS =================
_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _THIS_DIR.parent

def _path_from_env(env_name: str, default: Path) -> str:
    return str(Path(os.environ.get(env_name, str(default))).expanduser())

ROBOT_USD_PATH = _path_from_env(
    "SHARPA_ROBOT_USD",
    _WORKSPACE_ROOT / "Bimanual_dexhand" / "assets" / "hands" / "sharpa" / "asset" / "sharpa_right.usd",
)
CONTAINER_PATH_BASE = "/World/Envs/env"
PALM_LINK_NAME = "right_hand_C_MC"

GRASP_JSON_PATH = _path_from_env(
    "SHARPA_GRASP_JSON",
    _THIS_DIR / "bottle_2" / "Annotation" / "Sharpa_grasp_pose.json",
)
OBJECT_USD_PATH = _path_from_env("SHARPA_OBJECT_USD", _THIS_DIR / "bottle_2" / "Object.usd")
PHYSICS_SCENE_PATH = "/World/physicsScene"

# CLONER PARAMETERS
NUM_ENVS = 1  # Number of parallel environments
CLONE_SPACING = 3.0
ENV_BASE_PATH = "/World/Envs"
ENV_ROOT_PREFIX = f"{ENV_BASE_PATH}/env"

# MOTION/FILTER PARAMTERS
LIFT_HEIGHT = 0.5
MAX_DISPLACEMENT_THRESHOLD = 0.05    
DROP_HEIGHT_THRESHOLD = 0.02
CONTACT_THRESHOLD = 0.08 

# SIMULATION STEPS
COARSE_STEPS = 60
FINE_STEPS = 60
FINAL_STEPS = 60
SETTLE_STEPS = 20  # extra steps after approach to let fingers reach desired pos
CHECK_STEPS = 60
HOLD_STEPS = 50
LIFT_STEPS = 150

# PREAPPROACH PARAMETERS
PALM_DIRECTION_LOCAL = np.array([1.0, 0.0, 0.0])  # Palm forward direction in hand local frame
PREAPPROACH_STEP_BACK_DIST = 0.10                  # Metres to step back along negative palm direction
PREAPPROACH_STEPS = 60                              # Steps to move from preapproach to coarse pose

# JOINT PARAMETERS
JOINT_STIFFNESS = 80.0      
JOINT_DAMPING = 20.0         
JOINT_MAX_FORCE = 300.0 
JOINT_ARMATURE = 0.01     
JOINT_VELOCITY_LIMIT = 100.0 
CONVEX_DECOMP_MIN_THICKNESS = 0.002

#Part detection parameters
MAX_PARTS = 10

#Inverse gravity flag
INVERSE_GRAVITY = True

#Joints Ordering
SHARPA_JOINT_NAMES = [
    'right_index_MCP_FE',
    'right_index_MCP_AA',
    'right_index_PIP',
    'right_index_DIP',
    'right_thumb_CMC_FE',
    'right_thumb_CMC_AA',
    'right_thumb_MCP_FE',
    'right_thumb_MCP_AA',
    'right_thumb_IP',
    'right_middle_MCP_FE',
    'right_middle_MCP_AA',
    'right_middle_PIP',
    'right_middle_DIP',
    'right_ring_MCP_FE',
    'right_ring_MCP_AA',
    'right_ring_PIP',
    'right_ring_DIP',
    'right_pinky_CMC',
    'right_pinky_MCP_FE',
    'right_pinky_MCP_AA',
    'right_pinky_PIP',
    'right_pinky_DIP',
]

OUTPUT_JOINT_ORDER = [
    'right_index_MCP_FE',
    'right_index_MCP_AA',
    'right_index_PIP',
    'right_index_DIP',
    'right_thumb_CMC_FE',
    'right_thumb_CMC_AA',
    'right_thumb_MCP_FE',
    'right_thumb_MCP_AA',
    'right_thumb_IP',
    'right_middle_MCP_FE',
    'right_middle_MCP_AA',
    'right_middle_PIP',
    'right_middle_DIP',
    'right_ring_MCP_FE',
    'right_ring_MCP_AA',
    'right_ring_PIP',
    'right_ring_DIP',
    'right_pinky_CMC',
    'right_pinky_MCP_FE',
    'right_pinky_MCP_AA',
    'right_pinky_PIP',
    'right_pinky_DIP',
]

JOINT_REORDER_INDICES = [SHARPA_JOINT_NAMES.index(name) for name in OUTPUT_JOINT_ORDER]
INVERSE_JOINT_REORDER_INDICES = [OUTPUT_JOINT_ORDER.index(name) for name in SHARPA_JOINT_NAMES]

def joints_to_array(joints):
    """Convert joints to numpy array ordered by SHARPA_JOINT_NAMES.
    Accepts either a list (legacy) or a dict {joint_name: value}."""
    if isinstance(joints, dict):
        return np.array([joints[name] for name in SHARPA_JOINT_NAMES])
    return np.array(joints)

# ================= PATH HELPERS =================
def env_path(i: int) -> str:
    # env paths are /World/Envs/env_0, /World/Envs/env_1, ...
    return f"{ENV_ROOT_PREFIX}_{i}"

def obj_wrap(i: int) -> str:
    return f"{env_path(i)}/TargetObject"

def obj_ref(i: int) -> str:
    return f"{env_path(i)}/TargetObject/ref"

def hand_wrap(i: int) -> str:
    return f"{env_path(i)}/Xhand"

def hand_ref(i: int) -> str:
    return f"{env_path(i)}/Xhand/ref"


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

def reorder_grasp_joints(grasp_data):
    """Return a deep copy of grasp_data with joints reordered to OUTPUT_JOINT_ORDER.
    Handles both list (legacy) and dict {joint_name: value} formats."""
    import copy
    g = copy.deepcopy(grasp_data)
    for stage_key in ("coarse_grasp", "fine_grasp", "final_grasp"):
        joints = g[stage_key]["joints"]
        if isinstance(joints, dict):
            ordered = joints_to_array(joints).tolist()
        else:
            ordered = [joints[i] for i in JOINT_REORDER_INDICES]
        g[stage_key]["joints"] = ordered
    return g

def add_lighting(stage):
    light = UsdLux.DomeLight.Define(stage, "/World/defaultDomeLight")
    light.CreateIntensityAttr().Set(1000.0)
    d_light = UsdLux.DistantLight.Define(stage, "/World/distantLight")
    d_light.CreateIntensityAttr().Set(2000.0)
    UsdGeom.Xformable(d_light).AddRotateXYZOp().Set(Gf.Vec3f(45, 45, 0))


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


def setup_joint_drives(stage, container_path, 
                      stiffness=20.0,
                      damping=2.0,
                      max_force=300.0, 
                      armature=0.001,
                      velocity_limit=100.0):

    container_prim = stage.GetPrimAtPath(container_path)
    joint_count = 0
    
    for prim in Usd.PrimRange(container_prim):
        if not prim.IsA(UsdPhysics.Joint):
            continue
            
        # Determine drive type
        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive_kind = "angular"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            drive_kind = "linear"
        else:
            continue
        
        try:
            # 1. Apply Drive API
            drive = UsdPhysics.DriveAPI.Apply(prim, drive_kind)
            
            # 2. Set PD gains
            st = drive.GetStiffnessAttr() or drive.CreateStiffnessAttr()
            st.Set(float(stiffness))
            
            dm = drive.GetDampingAttr() or drive.CreateDampingAttr()
            dm.Set(float(damping))
            
            # 3. Set effort limit
            mf = drive.GetMaxForceAttr() or drive.CreateMaxForceAttr()
            mf.Set(float(max_force))
            
            # 4. Set initial target to 0
            tp = drive.GetTargetPositionAttr() or drive.CreateTargetPositionAttr()
            tp.Set(0.0)
            
            # 5. Set target velocity to 0 
            tv = drive.GetTargetVelocityAttr() or drive.CreateTargetVelocityAttr()
            tv.Set(0.0)
            
            # 6. Set PhysX joint friction
            if not prim.HasAPI(PhysxSchema.PhysxJointAPI):
                PhysxSchema.PhysxJointAPI.Apply(prim)
            
            physx_joint = PhysxSchema.PhysxJointAPI(prim)
            physx_joint.CreateJointFrictionAttr().Set(0.05)
            
            if not prim.HasAPI(PhysxSchema.PhysxJointAPI):
                PhysxSchema.PhysxJointAPI.Apply(prim)
            physx_joint = PhysxSchema.PhysxJointAPI(prim)
            physx_joint.CreateArmatureAttr().Set(float(armature))
            
            joint_count += 1
            
        except Exception as e:
            print(f"[WARN] Failed to setup drive for {prim.GetPath()}: {e}")
    
    log_info(f"Configured drives for {joint_count} joints")
    return joint_count

def set_hand_joint_targets(stage, container_path, joint_values_radians):
    """Set joint position targets by matching joint name to SHARPA_JOINT_NAMES list."""
    container_prim = stage.GetPrimAtPath(container_path)
    
    for prim in Usd.PrimRange(container_prim):
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        
        joint_name = prim.GetName()
        if joint_name not in SHARPA_JOINT_NAMES:
            continue

        idx = SHARPA_JOINT_NAMES.index(joint_name)
        if idx >= len(joint_values_radians):
            continue
        
        target_deg = float(joint_values_radians[idx]) * 57.29577951308232
        success = set_joint_drive_target(prim, target_deg)
        if not success:
            print(f"[WARN] Failed to set target for joint '{joint_name}' (idx {idx})")

def setup_direct_control(stage, container_path, palm_name):
    container_prim = stage.GetPrimAtPath(container_path)
    
    for prim in Usd.PrimRange(container_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            
            physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            physx_api.CreateLinearDampingAttr().Set(0.5)
            physx_api.CreateAngularDampingAttr().Set(0.5)
            physx_api.CreateMaxLinearVelocityAttr().Set(2.0)
            physx_api.CreateMaxAngularVelocityAttr().Set(2.0)
            physx_api.CreateContactSlopCoefficientAttr().Set(0.2)
            physx_api.CreateMaxDepenetrationVelocityAttr().Set(2.0)
            physx_api.CreateSolverPositionIterationCountAttr().Set(16)
            physx_api.CreateSolverVelocityIterationCountAttr().Set(2)
    
    setup_joint_drives(
        stage, 
        container_path, 
        stiffness=JOINT_STIFFNESS,
        damping=JOINT_DAMPING,
        max_force=JOINT_MAX_FORCE,
        armature=JOINT_ARMATURE,
        velocity_limit=JOINT_VELOCITY_LIMIT
    )
    
    return container_path

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
    scene.CreateGravityMagnitudeAttr().Set(30)

    if not prim.HasAPI(PhysxSchema.PhysxSceneAPI):
        PhysxSchema.PhysxSceneAPI.Apply(prim)
        
    physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(prim)
    physx_scene_api.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(32768)
    physx_scene_api.CreateGpuTotalAggregatePairsCapacityAttr().Set(32768)
    physx_scene_api.CreateSolverTypeAttr().Set("TGS")
    physx_scene_api.CreateBounceThresholdAttr().Set(0.2)
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
    p_mat.CreateStaticFrictionAttr().Set(3)
    p_mat.CreateDynamicFrictionAttr().Set(3)
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
    physx_rb.CreateContactSlopCoefficientAttr().Set(0.2)
    physx_rb.CreateMaxDepenetrationVelocityAttr().Set(2.0)
    physx_rb.CreateSolverPositionIterationCountAttr().Set(16)
    physx_rb.CreateSolverVelocityIterationCountAttr().Set(2)

    if not root_prim.HasAPI(UsdPhysics.MassAPI):
        UsdPhysics.MassAPI.Apply(root_prim)
    UsdPhysics.MassAPI(root_prim).CreateMassAttr().Set(0.5)

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
            phys_col.CreateContactOffsetAttr().Set(0.004)
            phys_col.CreateRestOffsetAttr().Set(0.001)
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

def setup_hand_collision(stage, container_path):
    """Set contact/rest offsets on existing PhysxCollisionAPI instances."""
    container_prim = stage.GetPrimAtPath(container_path)
    if not container_prim.IsValid():
        print(f"[ERROR] Invalid container path: {container_path}")
        return 0

    collision_count = 0

    for prim in Usd.PrimRange(container_prim):
        if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            continue

        physx_col_api = PhysxSchema.PhysxCollisionAPI(prim)
        physx_col_api.GetContactOffsetAttr().Set(0.004)
        physx_col_api.GetRestOffsetAttr().Set(0.001)
        collision_count += 1

    log_info(f"Set collision offsets on {collision_count} hand prims under {container_path}")
    return collision_count

def load_hand_env0(stage, usd_path):
    """Load hand into env_0 with wrapper/ref structure."""
    log_info(f"Loading Hand USD into env_0: {usd_path}")
    
    wrapper_path = hand_wrap(0)
    ref_path = hand_ref(0)
    UsdGeom.Xform.Define(stage, wrapper_path)
    
    # Add reference
    add_reference_to_stage(usd_path=usd_path, prim_path=ref_path)
    
    # Make editable temporarily
    changed_instanceables = make_prims_editable(stage, ref_path)
    
    # Setup physics control
    setup_direct_control(stage, ref_path, PALM_LINK_NAME)
    
    # Apply material
    create_and_bind_high_friction_material(stage, ref_path)
    
    # Setup collision
    setup_hand_collision(stage, ref_path)
    
    # Count DOFs
    container_prim = stage.GetPrimAtPath(ref_path)
    num_dofs = sum(1 for p in Usd.PrimRange(container_prim) 
                   if p.IsA(UsdPhysics.Joint) and not p.IsA(UsdPhysics.FixedJoint))
    
    # NOW make instanceable
    #restore_prims_instanceable(stage, changed_instanceables)
    
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
    gripper_base_path = f"{gripper_ref_path}/{PALM_LINK_NAME}"
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

async def evaluate_grasps_batched(stage, timeline, grasp_list, num_dofs, inverse_gravity=False):
    """Evaluate grasps in batches using parallel environments.

    All envs proceed through each phase together (barrier synchronization):
      0. Preapproach - hand steps back PREAPPROACH_STEP_BACK_DIST along negative palm dir, moves to coarse
      1. Reset    - gravity off, hand at preapproach pose, settle
      2. Approach - coarse -> fine -> final (all envs simultaneous)
      3. Hold     - gravity on, settle then hold at final pose
      4. Lift     - all active envs lift together
      5. Check success
    """
    total_grasps = len(grasp_list)
    batch_size = NUM_ENVS
    num_batches = (total_grasps + batch_size - 1) // batch_size

    log_info(f"Starting batched evaluation: {total_grasps} grasps, {num_batches} batches")

    results = []
    nominal_obj_pos = np.array([0.0, 0.0, 0.0])
    nominal_obj_rot = np.array([1.0, 0.0, 0.0, 0.0])
    zero_joints = np.zeros(num_dofs)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_grasps)
        batch_grasps = grasp_list[start_idx:end_idx]
        K = len(batch_grasps)

        print(f"\n{'='*60}")
        print(f"BATCH {batch_idx + 1}/{num_batches}: Grasps {start_idx} to {end_idx-1}")
        print(f"{'='*60}")

        # --- Parse grasp stages for this batch ---
        coarse_pos, coarse_rot, coarse_joints = [], [], []
        fine_pos,   fine_rot,   fine_joints   = [], [], []
        final_pos,  final_rot,  final_joints  = [], [], []

        for grasp_data in batch_grasps:
            def parse_stage(sd):
                wp, wr = transform_pose_to_world(
                    np.array(sd["position"]), np.array(sd["orientation"]),
                    nominal_obj_pos, nominal_obj_rot
                )
                return wp, wr, joints_to_array(sd["joints"])

            cp, cr, cj = parse_stage(grasp_data["coarse_grasp"])
            fp, fr, fj = parse_stage(grasp_data["fine_grasp"])
            np_, nr, nj = parse_stage(grasp_data["final_grasp"])

            coarse_pos.append(cp);  coarse_rot.append(cr);  coarse_joints.append(cj)
            fine_pos.append(fp);    fine_rot.append(fr);    fine_joints.append(fj)
            final_pos.append(np_);  final_rot.append(nr);   final_joints.append(nj)

        # Preapproach: step back from coarse position along negative palm direction
        preapproach_pos = [
            coarse_pos[k] - PREAPPROACH_STEP_BACK_DIST * quat_rotate_vector(coarse_rot[k], PALM_DIRECTION_LOCAL)
            for k in range(K)
        ]

        # --- RESET: stop timeline, restore objects, gravity off, hand at preapproach ---
        if timeline.is_playing():
            timeline.stop()
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

        for k in range(K):
            obj_wrapper_prim = stage.GetPrimAtPath(obj_wrap(k))
            if obj_wrapper_prim.IsValid() and obj_wrapper_prim.IsA(UsdGeom.Xformable):
                xf = UsdGeom.Xformable(obj_wrapper_prim)
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0))
                xf.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
            set_all_gravity(stage, [obj_wrap(k)], enable_gravity=False)
            clear_all_velocities(stage, [obj_wrap(k)])
            set_local_pose(stage, hand_wrap(k), preapproach_pos[k], coarse_rot[k])
            set_hand_joint_targets(stage, hand_ref(k), zero_joints)

        if not timeline.is_playing():
            timeline.play()
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()

        # Settle at reset pose
        for _ in range(60):
            await omni.kit.app.get_app().next_update_async()

        # Record initial z heights after settling
        initial_z = []
        for k in range(K):
            p, _ = get_prim_pose_robust(stage, obj_wrap(k))
            initial_z.append(p[2])

        active = [True] * K

        # --- PHASE 0: PREAPPROACH step_back -> coarse ---
        print(f"  [Batch {batch_idx+1}] Preapproach: step_back -> coarse ({PREAPPROACH_STEPS} steps)")
        for t in range(PREAPPROACH_STEPS + 1):
            alpha = min(t / float(PREAPPROACH_STEPS), 1.0)
            for k in range(K):
                if not active[k]:
                    continue
                set_local_pose(stage, hand_wrap(k),
                               interp_pos(preapproach_pos[k], coarse_pos[k], alpha),
                               coarse_rot[k])
                set_hand_joint_targets(stage, hand_ref(k), coarse_joints[k])
            await omni.kit.app.get_app().next_update_async()

        # --- PHASE 1: APPROACH coarse -> fine ---
        print(f"  [Batch {batch_idx+1}] Approach: coarse -> fine ({FINE_STEPS} steps)")
        for t in range(FINE_STEPS + 1):
            alpha = min(t / float(FINE_STEPS), 1.0)
            for k in range(K):
                if not active[k]:
                    continue
                set_local_pose(stage, hand_wrap(k),
                               interp_pos(coarse_pos[k], fine_pos[k], alpha),
                               slerp(coarse_rot[k], fine_rot[k], alpha))
                set_hand_joint_targets(stage, hand_ref(k),
                                       interp_joints(coarse_joints[k], fine_joints[k], alpha))
            await omni.kit.app.get_app().next_update_async()

        # --- PHASE 2: APPROACH fine -> final (grasp close) ---
        print(f"  [Batch {batch_idx+1}] Approach: fine -> final ({FINAL_STEPS} steps)")
        for t in range(FINAL_STEPS + 1):
            alpha = min(t / float(FINAL_STEPS), 1.0)
            for k in range(K):
                if not active[k]:
                    continue
                set_local_pose(stage, hand_wrap(k),
                               interp_pos(fine_pos[k], final_pos[k], alpha),
                               slerp(fine_rot[k], final_rot[k], alpha))
                set_hand_joint_targets(stage, hand_ref(k),
                                       interp_joints(fine_joints[k], final_joints[k], alpha))
            await omni.kit.app.get_app().next_update_async()

        # Settle: hold final pose for a few steps so fingers reach desired pos
        print(f"  [Batch {batch_idx+1}] Settle after approach ({SETTLE_STEPS} steps)")
        for _ in range(SETTLE_STEPS):
            for k in range(K):
                if not active[k]:
                    continue
                set_hand_joint_targets(stage, hand_ref(k), final_joints[k])
            await omni.kit.app.get_app().next_update_async()

        # --- PHASE 3: HOLD with gravity on (settle) ---
        print(f"  [Batch {batch_idx+1}] Gravity on, settling ({CHECK_STEPS} steps)")
        for k in range(K):
            if active[k]:
                set_all_gravity(stage, [obj_wrap(k)], enable_gravity=True)

        for _ in range(CHECK_STEPS):
            for k in range(K):
                if not active[k]:
                    continue
                set_hand_joint_targets(stage, hand_ref(k), final_joints[k])
            await omni.kit.app.get_app().next_update_async()

        # Gate: reject envs where object dropped during settling
        for k in range(K):
            if not active[k]:
                continue
            p_now, _ = get_prim_pose_robust(stage, obj_wrap(k))
            dropped = (p_now[2] > initial_z[k] + DROP_HEIGHT_THRESHOLD) if inverse_gravity \
                 else (p_now[2] < initial_z[k] - DROP_HEIGHT_THRESHOLD)
            if dropped:
                active[k] = False
                print(f"  [Env {k}] REJECTED: dropped during settling")

        # --- PHASE 4: HOLD at final pose ---
        print(f"  [Batch {batch_idx+1}] Hold ({HOLD_STEPS} steps)")
        for _ in range(HOLD_STEPS):
            for k in range(K):
                if not active[k]:
                    continue
                set_hand_joint_targets(stage, hand_ref(k), final_joints[k])
            await omni.kit.app.get_app().next_update_async()

        # --- PHASE 5: LIFT ---
        lift_dir = -1.0 if inverse_gravity else 1.0
        target_lift = [final_pos[k] + np.array([0, 0, lift_dir * LIFT_HEIGHT]) for k in range(K)]

        print(f"  [Batch {batch_idx+1}] Lift ({LIFT_STEPS} steps)")
        for t in range(LIFT_STEPS + 1):
            alpha = min(t / float(LIFT_STEPS), 1.0)
            for k in range(K):
                if not active[k]:
                    continue
                set_local_pose(stage, hand_wrap(k),
                               interp_pos(final_pos[k], target_lift[k], alpha),
                               final_rot[k])
                set_hand_joint_targets(stage, hand_ref(k), final_joints[k])
            await omni.kit.app.get_app().next_update_async()

        # Gate: check lift success
        success = [False] * K
        fail_reasons = [""] * K
        for k in range(K):
            if not active[k]:
                fail_reasons[k] = "Failed earlier"
                continue
            p_now, _ = get_prim_pose_robust(stage, obj_wrap(k))
            expected_z = initial_z[k] + lift_dir * LIFT_HEIGHT
            dropped = (p_now[2] > expected_z + DROP_HEIGHT_THRESHOLD) if inverse_gravity \
                 else (p_now[2] < expected_z - DROP_HEIGHT_THRESHOLD)
            if dropped:
                fail_reasons[k] = "Dropped during lift"
                print(f"  [Env {k}] FAILED: dropped during lift")
            else:
                success[k] = True
                print(f"  [Env {k}] SUCCESS!")

        # Cleanup: gravity off
        for k in range(K):
            set_all_gravity(stage, [obj_wrap(k)], enable_gravity=False)

        timeline.stop()

        for i in range(K):
            results.append({
                "grasp_index": start_idx + i,
                "success": success[i],
                "fail_reason": fail_reasons[i],
                "contacted_part": "body"
            })

        print(f"\nBatch {batch_idx+1} complete: {sum(success)}/{K} succeeded")

    return results

async def run_first_pass():
    # ============ STAGE SETUP ============
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
    
    # Physics scene setup
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
    
    # ============ RUN BATCHED EVALUATION ============
    results = await evaluate_grasps_batched(
        stage, 
        timeline, 
        grasp_list, 
        num_dofs
    )
    
    successful_grasps = [
        grasp_list[r["grasp_index"]]
        for r in results if r["success"]
    ]
    log_info(f"First pass survivors: {len(successful_grasps)}/{len(grasp_list)}")
    
    # ============ SUMMARY ============
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
    output_file = Path(GRASP_JSON_PATH).parent / "evaluation_results.json"
    output_successful_grasps = [reorder_grasp_joints(g) for g in successful_grasps]
    output_data = {
        "type": Path(OBJECT_USD_PATH).parent.name,
        "bottom_center": bottom_center,
        "functional_grasp": {
            "body": output_successful_grasps
        },
        "grasp": {}
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    log_info(f"Results saved to: {output_file}")
    
async def run_second_pass():
    # ============ STAGE SETUP ============
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
    
    # Physics scene setup
    setup_physics_scene(stage, INVERSE_GRAVITY)
    
    # Add ground and lighting
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
    output_file = Path(GRASP_JSON_PATH).parent / "evaluation_results.json"
    grasp_list = load_grasp_data_json_second_pass(output_file)
    if grasp_list is None or len(grasp_list) == 0:
        log_info("No grasps to evaluate")
        return
    
    # ============ RUN BATCHED EVALUATION ============
    results = await evaluate_grasps_batched(
        stage, 
        timeline, 
        grasp_list, 
        num_dofs,
        INVERSE_GRAVITY
    )
    
    successful_grasps = [
        grasp_list[r["grasp_index"]]
        for r in results if r["success"]
    ]
    log_info(f"Second pass survivors: {len(successful_grasps)}/{len(grasp_list)}")
    
    # ============ SUMMARY ============
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
    output_file = Path(GRASP_JSON_PATH).parent / "evaluation_results.json"
    output_successful_grasps = [reorder_grasp_joints(g) for g in successful_grasps]
    output_data = {
        "type": Path(OBJECT_USD_PATH).parent.name,
        "bottom_center": bottom_center,
        "functional_grasp": {
            "body": output_successful_grasps
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
