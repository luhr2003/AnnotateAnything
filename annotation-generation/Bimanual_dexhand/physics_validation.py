import os
os.environ["OMNI_LOG_LEVEL_DEFAULT"] = "error"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
carb.settings.get_settings().set("/log/level", 3)
carb.settings.get_settings().set("/log/fileLogLevel", 3)
carb.settings.get_settings().set("/rtx/instanceLogging", False)

import warnings
warnings.filterwarnings("ignore")

import asyncio
import json
import yaml
import numpy as np
from pathlib import Path

import omni.kit.app
import omni.usd
from pxr import Usd, UsdGeom, Gf, UsdPhysics, UsdShade, PhysxSchema, UsdLux
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
from omni.timeline import get_timeline_interface
from isaacsim.core.cloner import GridCloner

from validation_hand_runtime import (
    configure_authored_hand_joint_drives,
    set_authored_hand_joint_targets,
)


# ================= LOGGING =================
def log_info(msg):
    print(f"[INFO] {msg}")


# ================= MATH HELPERS =================
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
    world_pos = obj_pos + quat_rotate_vector(obj_rot, rel_pos)
    world_rot = quat_mul(obj_rot, rel_rot)
    return world_pos, world_rot

def slerp(q1, q2, t):
    q = (1.0 - t) * q1 + t * q2
    n = np.linalg.norm(q)
    return q / n if n > 1e-8 else q

def interp_pos(p1, p2, t):
    return p1 * (1.0 - t) + p2 * t

def interp_joints(j1, j2, t):
    """Interpolate two joint dicts {name: radians}."""
    return {name: float(j1[name]) * (1.0 - t) + float(j2[name]) * t for name in j1}

def xyzw_to_wxyz(q):
    """Convert quaternion from [x, y, z, w] to [w, x, y, z]."""
    return np.array([q[3], q[0], q[1], q[2]])


# ================= HAND CONFIG LOADING =================
_THIS_DIR = Path(__file__).parent
HANDS_BASE_DIR = _THIS_DIR / "assets" / "hands"

def _path_from_env(env_name: str, default: Path) -> str:
    return str(Path(os.environ.get(env_name, str(default))).expanduser())


def load_hand_yaml_config(hand_type: str, side: str) -> dict:
    """
    Load hand configuration from YAML for the given hand type and side.

    Returns a dict with:
      usd_path              - absolute path to the hand USD file
      palm_link             - name of the palm/root link
      approach_direction_local - local palm-facing direction used for pre-approach step-back
      controllable_joints   - ordered list of controllable joint names
    """
    config_path = HANDS_BASE_DIR / hand_type / "config" / f"{side}_hand.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Hand config not found: {config_path}")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    # Resolve USD path relative to the config directory
    config_dir = config_path.parent
    usd_path = (config_dir / raw["asset"]["usd_path"]).resolve()
    return {
        "side": raw["side"],
        "usd_path": str(usd_path),
        "palm_link": raw["root"]["palm_link"],
        # Use the palm normal for retreat/approach so validation approaches from
        # the hand's palm-facing direction rather than sliding along finger-forward.
        "approach_direction_local": np.array(raw["frame_convention"]["palm_normal_local"], dtype=np.float64),
        "controllable_joints": list(raw["joints"]["controllable"]),
    }


# ================= CONFIGURATIONS =================

# ---- Hand selection ----
# Set HAND_TYPE to the name of a subdirectory under assets/hands/
# (e.g. "dex3_1", "sharpa", "xhand").  Everything else is derived from its YAML.
HAND_TYPE = "dex3_1"
HAND_RUNTIME_MODE = "authored_usd"  # "legacy" or "authored_usd"
# For newer dex3_1 validation hands with authored collision models, keep
# HAND_TYPE="dex3_1" and point these overrides at the new left/right USDs.
HAND_USD_OVERRIDES = {
    "left": os.environ.get("BIMANUAL_LEFT_HAND_USD"),
    "right": os.environ.get("BIMANUAL_RIGHT_HAND_USD"),
}

# HAND_USD_OVERRIDES = {
#     "left": str(_THIS_DIR / "assets" / "hands" / "sharpa" / "asset" / "sharpa_left.usd"),
#     "right": str(_THIS_DIR / "assets" / "hands" / "sharpa" / "asset" / "sharpa_right.usd"),
# }

HAND_CFG = {
    "left":  load_hand_yaml_config(HAND_TYPE, "left"),
    "right": load_hand_yaml_config(HAND_TYPE, "right"),
}
HAND_STRUCTURE_CACHE = {}

# ---- Object / grasp paths (update per object) ----
OBJECT_USD_PATH = _path_from_env("BIMANUAL_OBJECT_USD", _THIS_DIR / "assets" / "objects" / "Object.usd")
GRASP_JSON_PATH = _path_from_env(
    "BIMANUAL_GRASP_JSON",
    _THIS_DIR / "outputs" / "physics_validation_inputs" / "cat1_bimanual_1000.json",
)

PHYSICS_SCENE_PATH = "/World/physicsScene"

# ---- Cloner parameters ----
NUM_ENVS      = 20        # parallel environments (two hands each → keep lower than single-hand)
CLONE_SPACING = 1.0       # metres between env origins
ENV_BASE_PATH   = "/World/Envs"
ENV_ROOT_PREFIX = f"{ENV_BASE_PATH}/env"

# ---- Motion / filter parameters ----
LIFT_HEIGHT           = 0.3
DROP_HEIGHT_THRESHOLD = 0.02

# ---- Simulation steps ----
PREAPPROACH_STEPS = 60   # step-back  →  pregrasp pose
APPROACH_STEPS    = 60   # pregrasp   →  grasp pose
SQUEEZE_STEPS     = 60   # grasp      →  squeeze pose
SETTLE_STEPS      = 100  # hold final joints before gravity (long enough for fingers to fully close)
CHECK_STEPS       = 60   # gravity on, settle
HOLD_STEPS        = 50   # hold at final pose
LIFT_STEPS        = 150  # lift to target height

# ---- Pre-approach parameters ----
PREAPPROACH_STEP_BACK_DIST = 0.1  # metres to step back along negative palm-normal direction

# Temporary validation experiment: keep the final/squeeze wrist pose, but do
# not apply the final_grasp joint targets. This lets us test whether the final
# joint close is causing penetration or instability in physics validation.
IGNORE_FINAL_GRASP_JOINTS = False

# ---- Joint drive parameters ----
JOINT_STIFFNESS    = 10.0
JOINT_DAMPING      = 2.0
JOINT_MAX_FORCE    = 300.0
JOINT_ARMATURE     = 0.01
JOINT_VELOCITY_LIMIT = 100.0

# ---- Second-pass flag ----
INVERSE_GRAVITY = False


def _uses_authored_hand_runtime(side: str) -> bool:
    return HAND_RUNTIME_MODE == "authored_usd" and bool(HAND_USD_OVERRIDES.get(side))


def _runtime_hand_usd_path(side: str) -> str:
    return str(HAND_USD_OVERRIDES.get(side) or HAND_CFG[side]["usd_path"])


def _hand_output_label() -> str:
    return f"{HAND_TYPE}_authored" if any(_uses_authored_hand_runtime(side) for side in ("left", "right")) else HAND_TYPE


# ================= PATH HELPERS =================
def env_path(i: int) -> str:
    return f"{ENV_ROOT_PREFIX}_{i}"

def obj_wrap(i: int) -> str:
    return f"{env_path(i)}/TargetObject"

def obj_ref(i: int) -> str:
    return f"{env_path(i)}/TargetObject/ref"

def hand_wrap(i: int, side: str) -> str:
    label = "LeftHand" if side == "left" else "RightHand"
    return f"{env_path(i)}/{label}"

def hand_ref(i: int, side: str) -> str:
    label = "LeftHand" if side == "left" else "RightHand"
    return f"{env_path(i)}/{label}/ref"


# ================= USD / PHYSICS HELPERS =================
def add_lighting(stage):
    light = UsdLux.DomeLight.Define(stage, "/World/defaultDomeLight")
    light.CreateIntensityAttr().Set(1000.0)
    d_light = UsdLux.DistantLight.Define(stage, "/World/distantLight")
    d_light.CreateIntensityAttr().Set(2000.0)
    UsdGeom.Xformable(d_light).AddRotateXYZOp().Set(Gf.Vec3f(45, 45, 0))


def compute_bottom_center(stage, prim_path):
    """Return [cx, cy, z_min] of the world bounding box of a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return [0.0, 0.0, 0.0]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    bbox = bbox_cache.ComputeWorldBound(prim)
    box_range = bbox.GetRange()
    min_pt = box_range.GetMin()
    max_pt = box_range.GetMax()
    return [
        float((min_pt[0] + max_pt[0]) / 2.0),
        float((min_pt[1] + max_pt[1]) / 2.0),
        float(min_pt[2]),
    ]


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
    root_prim = stage.GetPrimAtPath(root_prim_path)
    changed_paths = []
    if not root_prim.IsValid():
        return changed_paths
    for prim in Usd.PrimRange(root_prim):
        if prim.IsInstanceable():
            changed_paths.append(prim.GetPath().pathString)
            prim.SetInstanceable(False)
    log_info(f"Made {len(changed_paths)} instanceable prims editable under {root_prim_path}")
    return changed_paths


def restore_prims_instanceable(stage, prim_paths):
    restored = 0
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        prim.SetInstanceable(True)
        restored += 1
    log_info(f"Restored {restored} prims to instanceable")
    return restored


def create_and_bind_high_friction_material(stage, root_prim_path):
    material_path = "/World/Physics_Materials/SuperGripMat"
    if not stage.GetPrimAtPath(material_path).IsValid():
        UsdShade.Material.Define(stage, material_path)
    mat_prim = stage.GetPrimAtPath(material_path)
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
    root_prim = stage.GetPrimAtPath(root_prim_path)
    count = 0
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI):
            api = UsdShade.MaterialBindingAPI(prim)
            api.Bind(UsdShade.Material(mat_prim), materialPurpose="physics")
            count += 1
    log_info(f"Physics Material Applied to {count} prims under {root_prim_path}")
    return count


def setup_object_physics(stage, root_prim_path, gravity_on=False):
    """
    Preserve the authored object collider setup.

    We only:
    - ensure the wrapper/root behaves as a rigid body for validation
    - remove stray child rigid-body state if present
    - tune runtime parameters on prims that already have authored collision

    We do not author new colliders or replace the authored collision
    approximation here.
    """
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if root_prim.IsA(UsdGeom.Xformable):
        UsdGeom.Xformable(root_prim).ClearXformOpOrder()
    if not root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rb_api = UsdPhysics.RigidBodyAPI(root_prim)
    rb_api.CreateKinematicEnabledAttr().Set(False)
    if not root_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
    physx_rb = PhysxSchema.PhysxRigidBodyAPI(root_prim)
    physx_rb.CreateDisableGravityAttr().Set(not gravity_on)
    physx_rb.CreateContactSlopCoefficientAttr().Set(0.0)
    physx_rb.CreateMaxDepenetrationVelocityAttr().Set(0.5)  # gentle depenetration to avoid explosive correction
    physx_rb.CreateSolverPositionIterationCountAttr().Set(32)
    physx_rb.CreateSolverVelocityIterationCountAttr().Set(4)
    if not root_prim.HasAPI(UsdPhysics.MassAPI):
        UsdPhysics.MassAPI.Apply(root_prim)
    UsdPhysics.MassAPI(root_prim).CreateMassAttr().Set(0.5)
    tuned_collider_count = 0
    for prim in Usd.PrimRange(root_prim):
        path_str = str(prim.GetPath())
        if path_str == root_prim_path:
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
        if prim.HasAPI(UsdPhysics.MassAPI):
            prim.RemoveAPI(UsdPhysics.MassAPI)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            # Parameter API only; leave the authored collider geometry intact.
            if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                PhysxSchema.PhysxCollisionAPI.Apply(prim)
            phys_col = PhysxSchema.PhysxCollisionAPI(prim)
            phys_col.CreateContactOffsetAttr().Set(0.008)  # wider contact zone prevents finger tunneling
            phys_col.CreateRestOffsetAttr().Set(0.002)
            tuned_collider_count += 1
    xformable = UsdGeom.Xformable(root_prim)
    xformable.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0))
    xformable.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
    log_info(f"Tuned runtime params on {tuned_collider_count} existing object collider prims")


def set_joint_drive_target(joint_prim: Usd.Prim, target_position: float) -> bool:
    if not joint_prim.IsValid():
        return False
    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        drive_kind = "angular"
    elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
        drive_kind = "linear"
    else:
        return False
    try:
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_kind)
        tp = drive.GetTargetPositionAttr()
        if not tp or not tp.IsValid():
            tp = drive.CreateTargetPositionAttr()
        tp.Set(float(target_position))
        return True
    except Exception as e:
        print(f"[WARN] Failed to set drive target on {joint_prim.GetPath()}: {e}")
        return False


def setup_joint_drives(stage, container_path,
                       stiffness=20.0, damping=2.0,
                       max_force=300.0, armature=0.001, velocity_limit=100.0):
    container_prim = stage.GetPrimAtPath(container_path)
    joint_count = 0
    for prim in Usd.PrimRange(container_prim):
        if not prim.IsA(UsdPhysics.Joint):
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
            st.Set(float(stiffness))
            dm = drive.GetDampingAttr() or drive.CreateDampingAttr()
            dm.Set(float(damping))
            mf = drive.GetMaxForceAttr() or drive.CreateMaxForceAttr()
            mf.Set(float(max_force))
            tp = drive.GetTargetPositionAttr() or drive.CreateTargetPositionAttr()
            tp.Set(0.0)
            tv = drive.GetTargetVelocityAttr() or drive.CreateTargetVelocityAttr()
            tv.Set(0.0)
            if not prim.HasAPI(PhysxSchema.PhysxJointAPI):
                PhysxSchema.PhysxJointAPI.Apply(prim)
            physx_joint = PhysxSchema.PhysxJointAPI(prim)
            physx_joint.CreateJointFrictionAttr().Set(0.05)
            physx_joint.CreateArmatureAttr().Set(float(armature))
            joint_count += 1
        except Exception as e:
            print(f"[WARN] Failed to setup drive for {prim.GetPath()}: {e}")
    log_info(f"Configured drives for {joint_count} joints")
    return joint_count


def setup_hand_collision(stage, container_path):
    container_prim = stage.GetPrimAtPath(container_path)
    if not container_prim.IsValid():
        return 0
    count = 0
    for prim in Usd.PrimRange(container_prim):
        if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            continue
        physx_col_api = PhysxSchema.PhysxCollisionAPI(prim)
        physx_col_api.GetContactOffsetAttr().Set(0.008)
        physx_col_api.GetRestOffsetAttr().Set(0.002)
        count += 1
    log_info(f"Set collision offsets on {count} hand prims under {container_path}")
    return count


def setup_direct_control(stage, container_path, palm_link_name):
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
            physx_api.CreateContactSlopCoefficientAttr().Set(0.0)
            physx_api.CreateMaxDepenetrationVelocityAttr().Set(0.5)  # consistent with object
            physx_api.CreateSolverPositionIterationCountAttr().Set(32)
            physx_api.CreateSolverVelocityIterationCountAttr().Set(4)
    setup_joint_drives(
        stage, container_path,
        stiffness=JOINT_STIFFNESS,
        damping=JOINT_DAMPING,
        max_force=JOINT_MAX_FORCE,
        armature=JOINT_ARMATURE,
        velocity_limit=JOINT_VELOCITY_LIMIT,
    )
    return container_path


def set_local_pose(stage, prim_path, pos, quat):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    translate_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    orient_op    = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeOrient),    None)
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    if orient_op is None:
        orient_op = xform.AddOrientOp()
    translate_op.Set(Gf.Vec3f(*pos))
    orient_op.Set(Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))


def get_prim_pose_robust(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return np.zeros(3), np.array([1, 0, 0, 0])
    xformable = UsdGeom.Xformable(prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = world_transform.ExtractTranslation()
    r = world_transform.ExtractRotationQuat()
    return (
        np.array([t[0], t[1], t[2]]),
        np.array([r.GetReal(), r.GetImaginary()[0], r.GetImaginary()[1], r.GetImaginary()[2]]),
    )


def set_all_gravity(stage, paths, enable_gravity):
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        PhysxSchema.PhysxRigidBodyAPI(prim).CreateDisableGravityAttr().Set(not enable_gravity)


def clear_all_velocities(stage, paths):
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(prim)
        rb = UsdPhysics.RigidBodyAPI(prim)
        rb.CreateVelocityAttr().Set(Gf.Vec3f(0, 0, 0))
        rb.CreateAngularVelocityAttr().Set(Gf.Vec3f(0, 0, 0))


# ================= JOINT HELPERS =================
def zero_joints(side: str) -> dict:
    """Return a zero-position dict for all controllable joints of the given side."""
    return {name: 0.0 for name in HAND_CFG[side]["controllable_joints"]}


def set_hand_joint_targets_dict(stage, container_path, joint_positions_rad: dict, side: str):
    """Set joint position targets (radians) from a {joint_name: value} dict."""
    if _uses_authored_hand_runtime(side):
        set_authored_hand_joint_targets(
            stage,
            container_path,
            joint_positions_rad,
            palm_link_name=HAND_CFG[side]["palm_link"],
            controllable_joint_names=HAND_CFG[side]["controllable_joints"],
            set_joint_drive_target_fn=set_joint_drive_target,
            cache=HAND_STRUCTURE_CACHE,
            logger=log_info,
        )
        return

    container_prim = stage.GetPrimAtPath(container_path)
    for prim in Usd.PrimRange(container_prim):
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint_name = prim.GetName()
        if joint_name not in joint_positions_rad:
            continue
        target_deg = float(joint_positions_rad[joint_name]) * 57.29577951308232
        set_joint_drive_target(prim, target_deg)


# ================= GRASP JSON I/O =================
def load_grasp_data_first_pass(json_path: str):
    """
    Load bimanual grasps for the first validation pass.

    Expected input formats:
    {
      "grasps": [
        {
          "left":  { "pregrasp": {...}, "grasp": {...}, "squeeze": {...} },
          "right": { "pregrasp": {...}, "grasp": {...}, "squeeze": {...} }
        },
        ...
      ]
    }

    or the legacy / bimanual-export style:
    {
      "functional_grasp": {
        "body": [
          {
            "left_hand":  { "coarse_grasp": {...}, "fine_grasp": {...}, "final_grasp": {...} },
            "right_hand": { "coarse_grasp": {...}, "fine_grasp": {...}, "final_grasp": {...} }
          }
        ]
      }
    }

    Each stage dict has:
      either:
      "wrist_position":        [x, y, z]
      "wrist_quaternion_wxyz": [w, x, y, z]
      "joint_positions":       { joint_name: radians, ... }

      or:
      "position":    [x, y, z]
      "orientation": [w, x, y, z]
      "joints":      [radians in controllable-joint order]
    """
    if not os.path.exists(json_path):
        log_info(f"File not found: {json_path}")
        return None
    with open(json_path) as f:
        data = json.load(f)
    if "grasps" in data:
        grasps = data["grasps"]
    elif "functional_grasp" in data and isinstance(data["functional_grasp"], dict) and "body" in data["functional_grasp"]:
        grasps = data["functional_grasp"]["body"]
    else:
        log_info("Invalid JSON: expected top-level 'grasps' or 'functional_grasp.body'")
        return None
    log_info(f"Loaded {len(grasps)} grasps.")
    return grasps


def load_grasp_data_second_pass(json_path: str):
    """
    Load grasps from the first-pass output file (functional_grasp.body format).
    Same per-grasp schema as load_grasp_data_first_pass.
    """
    if not os.path.exists(json_path):
        log_info(f"File not found: {json_path}")
        return None
    with open(json_path) as f:
        data = json.load(f)
    grasps = data["functional_grasp"]["body"]
    log_info(f"Loaded {len(grasps)} previously successful grasps.")
    return grasps


def _joint_values_to_dict(side: str, joint_values):
    if isinstance(joint_values, dict):
        return {str(k): float(v) for k, v in joint_values.items()}
    if isinstance(joint_values, (list, tuple)):
        ordered_names = HAND_CFG[side]["controllable_joints"]
        if len(joint_values) != len(ordered_names):
            raise ValueError(
                f"Joint list length mismatch for side '{side}': "
                f"expected {len(ordered_names)}, got {len(joint_values)}"
            )
        return {
            joint_name: float(joint_values[idx])
            for idx, joint_name in enumerate(ordered_names)
        }
    raise TypeError(f"Unsupported joint payload for side '{side}': {type(joint_values)!r}")


def _parse_stage(sd: dict, side: str):
    """
    Parse one stage dict into (pos_wxyz, quat_wxyz, joints_dict).
    Preferred serialized format is wrist_quaternion_wxyz.
    Legacy wrist_quaternion_xyzw is also accepted.
    """
    if "wrist_position" in sd:
        pos = np.array(sd["wrist_position"], dtype=float)
    else:
        pos = np.array(sd["position"], dtype=float)

    if "wrist_quaternion_wxyz" in sd:
        quat = np.array(sd["wrist_quaternion_wxyz"], dtype=float)
    else:
        if "wrist_quaternion_xyzw" in sd:
            quat = xyzw_to_wxyz(np.array(sd["wrist_quaternion_xyzw"], dtype=float))
        else:
            quat = np.array(sd["orientation"], dtype=float)

    if "joint_positions" in sd:
        joints = _joint_values_to_dict(side, sd["joint_positions"])
    else:
        joints = _joint_values_to_dict(side, sd["joints"])
    return pos, quat, joints


def _get_stage_entry(side_data: dict, primary_name: str, *fallback_names: str):
    for stage_name in (primary_name, *fallback_names):
        if stage_name in side_data:
            return side_data[stage_name]
    available = ", ".join(sorted(side_data.keys()))
    raise KeyError(f"Missing stage '{primary_name}' (fallbacks: {fallback_names}); available keys: {available}")


def _get_side_entry(grasp_entry: dict, side: str):
    if side in grasp_entry:
        return grasp_entry[side]
    legacy_key = f"{side}_hand"
    if legacy_key in grasp_entry:
        return grasp_entry[legacy_key]
    available = ", ".join(sorted(grasp_entry.keys()))
    raise KeyError(f"Missing side '{side}' (or '{legacy_key}'); available keys: {available}")


def _stage_to_dict(pos, quat_wxyz, joints):
    """Serialise one stage using wxyz, while keeping xyzw for compatibility."""
    return {
        "wrist_position": pos.tolist(),
        "wrist_quaternion_wxyz": [float(quat_wxyz[0]), float(quat_wxyz[1]),
                                   float(quat_wxyz[2]), float(quat_wxyz[3])],
        "wrist_quaternion_xyzw": [float(quat_wxyz[1]), float(quat_wxyz[2]),
                                   float(quat_wxyz[3]), float(quat_wxyz[0])],
        "joint_positions": {k: float(v) for k, v in joints.items()},
    }


# ================= ENV-0 LOADERS =================
def load_target_usd_env0(stage, usd_path, gravity_on=False):
    log_info(f"Loading Object USD into env_0: {usd_path}")
    if not os.path.exists(usd_path):
        return None

    wrapper_path = obj_wrap(0)
    ref_path     = obj_ref(0)

    UsdGeom.Xform.Define(stage, wrapper_path)
    add_reference_to_stage(usd_path=usd_path, prim_path=ref_path)

    changed = make_prims_editable(stage, ref_path)
    setup_object_physics(stage, wrapper_path, gravity_on=gravity_on)
    create_and_bind_high_friction_material(stage, wrapper_path)
    restore_prims_instanceable(stage, changed)

    return ref_path


def load_hand_env0(stage, side: str):
    """Load one hand (left or right) into env_0, returns (ref_path, num_dofs)."""
    cfg = HAND_CFG[side]
    usd_path = _runtime_hand_usd_path(side)
    log_info(f"Loading {side} hand USD into env_0: {usd_path}")

    wrap_path = hand_wrap(0, side)
    ref_path  = hand_ref(0, side)

    UsdGeom.Xform.Define(stage, wrap_path)
    add_reference_to_stage(usd_path=usd_path, prim_path=ref_path)

    changed = make_prims_editable(stage, ref_path)
    create_and_bind_high_friction_material(stage, ref_path)

    if _uses_authored_hand_runtime(side):
        log_info(f"Using authored collision-model runtime for {side} hand")
        hand_structure = configure_authored_hand_joint_drives(
            stage,
            ref_path,
            palm_link_name=cfg["palm_link"],
            controllable_joint_names=cfg["controllable_joints"],
            joint_stiffness=JOINT_STIFFNESS,
            joint_damping=JOINT_DAMPING,
            joint_max_force=JOINT_MAX_FORCE,
            joint_armature=JOINT_ARMATURE,
            joint_velocity_limit=JOINT_VELOCITY_LIMIT,
            cache=HAND_STRUCTURE_CACHE,
            logger=log_info,
        )
        if hand_structure is None:
            raise RuntimeError(f"Failed to resolve authored hand structure for side '{side}' under {ref_path}")
        num_dofs = len(hand_structure["ordered_joint_paths"])
    else:
        setup_direct_control(stage, ref_path, cfg["palm_link"])
        setup_hand_collision(stage, ref_path)
        container_prim = stage.GetPrimAtPath(ref_path)
        num_dofs = sum(
            1 for p in Usd.PrimRange(container_prim)
            if p.IsA(UsdPhysics.Joint) and not p.IsA(UsdPhysics.FixedJoint)
        )

    restore_prims_instanceable(stage, changed)
    return ref_path, num_dofs


# ================= BATCHED EVALUATION =================
async def evaluate_grasps_batched(stage, timeline, grasp_list, inverse_gravity=False, gravity_always_on=False):
    """
    Evaluate bimanual grasps in batches of NUM_ENVS parallel environments.

    Each bimanual grasp has "left" and "right" sides, each with three stages:
      pregrasp  (approach / hand open)
      grasp     (start closing)
      squeeze   (tight close / hold)

    Simulation sequence per batch:
      Phase 0: step-back  → pregrasp   (PREAPPROACH_STEPS)
      Phase 1: pregrasp   → grasp      (APPROACH_STEPS)
      Phase 2: grasp      → squeeze    (SQUEEZE_STEPS)
      Settle at squeeze                (SETTLE_STEPS)
      Gravity on, settle               (CHECK_STEPS)   → gate: drop check
      Hold at squeeze                  (HOLD_STEPS)
      Lift                             (LIFT_STEPS)    → success check
    """
    total   = len(grasp_list)
    n_batch = (total + NUM_ENVS - 1) // NUM_ENVS
    log_info(f"Batched evaluation: {total} grasps in {n_batch} batch(es) of up to {NUM_ENVS}")

    results       = []
    nominal_obj_pos = np.array([0.0, 0.0, 0.0])
    nominal_obj_rot = np.array([1.0, 0.0, 0.0, 0.0])
    SIDES = ("left", "right")

    for batch_idx in range(n_batch):
        s_idx = batch_idx * NUM_ENVS
        e_idx = min(s_idx + NUM_ENVS, total)
        batch = grasp_list[s_idx:e_idx]
        K = len(batch)

        print(f"\n{'='*60}")
        print(f"BATCH {batch_idx+1}/{n_batch}: grasps {s_idx}–{e_idx-1}")
        print(f"{'='*60}")

        # ---- Parse stages for both sides ----
        if IGNORE_FINAL_GRASP_JOINTS:
            log_info("Temporary mode: final_grasp joints are ignored; using grasp/fine_grasp joints at final pose.")
        pregrasp_pos  = {s: [] for s in SIDES}
        pregrasp_rot  = {s: [] for s in SIDES}
        pregrasp_jts  = {s: [] for s in SIDES}
        grasp_pos     = {s: [] for s in SIDES}
        grasp_rot     = {s: [] for s in SIDES}
        grasp_jts     = {s: [] for s in SIDES}
        squeeze_pos   = {s: [] for s in SIDES}
        squeeze_rot   = {s: [] for s in SIDES}
        squeeze_jts   = {s: [] for s in SIDES}

        for g in batch:
            for side in SIDES:
                side_data = _get_side_entry(g, side)
                pp, pr, pj = _parse_stage(_get_stage_entry(side_data, "pregrasp", "coarse_grasp"), side)
                gp, gr, gj = _parse_stage(_get_stage_entry(side_data, "grasp", "fine_grasp"), side)
                sp, sr, sj = _parse_stage(_get_stage_entry(side_data, "squeeze", "final_grasp"), side)
                if IGNORE_FINAL_GRASP_JOINTS:
                    sj = dict(gj)

                wp, wr = transform_pose_to_world(pp, pr, nominal_obj_pos, nominal_obj_rot)
                gp, gr = transform_pose_to_world(gp, gr, nominal_obj_pos, nominal_obj_rot)
                sp, sr = transform_pose_to_world(sp, sr, nominal_obj_pos, nominal_obj_rot)

                pregrasp_pos[side].append(wp);  pregrasp_rot[side].append(wr);  pregrasp_jts[side].append(pj)
                grasp_pos[side].append(gp);     grasp_rot[side].append(gr);     grasp_jts[side].append(gj)
                squeeze_pos[side].append(sp);   squeeze_rot[side].append(sr);   squeeze_jts[side].append(sj)

        # Pre-approach: step back along negative approach direction in hand local frame
        preapproach_pos = {}
        for side in SIDES:
            approach_dir = HAND_CFG[side]["approach_direction_local"]
            preapproach_pos[side] = [
                pregrasp_pos[side][k] - PREAPPROACH_STEP_BACK_DIST
                    * quat_rotate_vector(pregrasp_rot[side][k], approach_dir)
                for k in range(K)
            ]

        # ---- RESET ----
        if timeline.is_playing():
            timeline.stop()
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

        for k in range(K):
            # Reset object pose
            obj_prim = stage.GetPrimAtPath(obj_wrap(k))
            if obj_prim.IsValid() and obj_prim.IsA(UsdGeom.Xformable):
                xf = UsdGeom.Xformable(obj_prim)
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0))
                xf.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
            if not gravity_always_on:
                set_all_gravity(stage, [obj_wrap(k)], enable_gravity=False)
            clear_all_velocities(stage, [obj_wrap(k)])

            # Place both hands at preapproach
            for side in SIDES:
                set_local_pose(stage, hand_wrap(k, side),
                               preapproach_pos[side][k], pregrasp_rot[side][k])
                set_hand_joint_targets_dict(stage, hand_ref(k, side), zero_joints(side), side)

        if not timeline.is_playing():
            timeline.play()
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()

        # Initial settle at reset pose
        for _ in range(60):
            await omni.kit.app.get_app().next_update_async()

        # Record initial object z heights
        initial_z = []
        for k in range(K):
            p, _ = get_prim_pose_robust(stage, obj_wrap(k))
            initial_z.append(p[2])

        active = [True] * K

        # ---- PHASE 0: step-back → pregrasp ----
        print(f"  [Batch {batch_idx+1}] Phase 0: preapproach → pregrasp ({PREAPPROACH_STEPS} steps)")
        for t in range(PREAPPROACH_STEPS + 1):
            alpha = min(t / float(PREAPPROACH_STEPS), 1.0)
            for k in range(K):
                if not active[k]:
                    continue
                for side in SIDES:
                    set_local_pose(stage, hand_wrap(k, side),
                                   interp_pos(preapproach_pos[side][k], pregrasp_pos[side][k], alpha),
                                   pregrasp_rot[side][k])
                    set_hand_joint_targets_dict(stage, hand_ref(k, side), pregrasp_jts[side][k], side)
            await omni.kit.app.get_app().next_update_async()

        # ---- PHASE 1: pregrasp → grasp ----
        print(f"  [Batch {batch_idx+1}] Phase 1: pregrasp → grasp ({APPROACH_STEPS} steps)")
        for t in range(APPROACH_STEPS + 1):
            alpha = min(t / float(APPROACH_STEPS), 1.0)
            for k in range(K):
                if not active[k]:
                    continue
                for side in SIDES:
                    set_local_pose(stage, hand_wrap(k, side),
                                   interp_pos(pregrasp_pos[side][k], grasp_pos[side][k], alpha),
                                   slerp(pregrasp_rot[side][k], grasp_rot[side][k], alpha))
                    set_hand_joint_targets_dict(stage, hand_ref(k, side),
                                                interp_joints(pregrasp_jts[side][k], grasp_jts[side][k], alpha), side)
            await omni.kit.app.get_app().next_update_async()

        # ---- PHASE 2: grasp → squeeze ----
        print(f"  [Batch {batch_idx+1}] Phase 2: grasp → squeeze ({SQUEEZE_STEPS} steps)")
        for t in range(SQUEEZE_STEPS + 1):
            alpha = min(t / float(SQUEEZE_STEPS), 1.0)
            for k in range(K):
                if not active[k]:
                    continue
                for side in SIDES:
                    set_local_pose(stage, hand_wrap(k, side),
                                   interp_pos(grasp_pos[side][k], squeeze_pos[side][k], alpha),
                                   slerp(grasp_rot[side][k], squeeze_rot[side][k], alpha))
                    set_hand_joint_targets_dict(stage, hand_ref(k, side),
                                                interp_joints(grasp_jts[side][k], squeeze_jts[side][k], alpha), side)
            await omni.kit.app.get_app().next_update_async()

        # ---- SETTLE at squeeze ----
        print(f"  [Batch {batch_idx+1}] Settle ({SETTLE_STEPS} steps)")
        for _ in range(SETTLE_STEPS):
            for k in range(K):
                if not active[k]:
                    continue
                for side in SIDES:
                    set_hand_joint_targets_dict(stage, hand_ref(k, side), squeeze_jts[side][k], side)
            await omni.kit.app.get_app().next_update_async()

        # ---- GRAVITY ON + CHECK ----
        print(f"  [Batch {batch_idx+1}] Gravity on, settling ({CHECK_STEPS} steps)")
        if not gravity_always_on:
            for k in range(K):
                if active[k]:
                    set_all_gravity(stage, [obj_wrap(k)], enable_gravity=True)

        for _ in range(CHECK_STEPS):
            for k in range(K):
                if not active[k]:
                    continue
                for side in SIDES:
                    set_hand_joint_targets_dict(stage, hand_ref(k, side), squeeze_jts[side][k], side)
            await omni.kit.app.get_app().next_update_async()

        # Gate: reject envs where object already dropped during settling
        for k in range(K):
            if not active[k]:
                continue
            p_now, _ = get_prim_pose_robust(stage, obj_wrap(k))
            dropped = (p_now[2] > initial_z[k] + DROP_HEIGHT_THRESHOLD) if inverse_gravity \
                 else (p_now[2] < initial_z[k] - DROP_HEIGHT_THRESHOLD)
            if dropped:
                active[k] = False
                print(f"  [Env {k}] REJECTED: dropped during settling")

        # ---- HOLD ----
        print(f"  [Batch {batch_idx+1}] Hold ({HOLD_STEPS} steps)")
        for _ in range(HOLD_STEPS):
            for k in range(K):
                if not active[k]:
                    continue
                for side in SIDES:
                    set_hand_joint_targets_dict(stage, hand_ref(k, side), squeeze_jts[side][k], side)
            await omni.kit.app.get_app().next_update_async()

        # ---- LIFT ----
        lift_dir   = -1.0 if inverse_gravity else 1.0
        lift_delta = np.array([0, 0, lift_dir * LIFT_HEIGHT])
        target_lift = {
            side: [squeeze_pos[side][k] + lift_delta for k in range(K)]
            for side in SIDES
        }

        print(f"  [Batch {batch_idx+1}] Lift ({LIFT_STEPS} steps)")
        for t in range(LIFT_STEPS + 1):
            alpha = min(t / float(LIFT_STEPS), 1.0)
            for k in range(K):
                if not active[k]:
                    continue
                for side in SIDES:
                    set_local_pose(stage, hand_wrap(k, side),
                                   interp_pos(squeeze_pos[side][k], target_lift[side][k], alpha),
                                   squeeze_rot[side][k])
                    set_hand_joint_targets_dict(stage, hand_ref(k, side), squeeze_jts[side][k], side)
            await omni.kit.app.get_app().next_update_async()

        # ---- SUCCESS CHECK ----
        success      = [False] * K
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

        # Disable gravity for clean reset next batch (skip if gravity is always on)
        if not gravity_always_on:
            for k in range(K):
                set_all_gravity(stage, [obj_wrap(k)], enable_gravity=False)

        timeline.stop()

        for i in range(K):
            results.append({
                "grasp_index": s_idx + i,
                "success":     success[i],
                "fail_reason": fail_reasons[i],
            })

        print(f"\nBatch {batch_idx+1} done: {sum(success)}/{K} succeeded")

    return results


# ================= STAGE SETUP HELPERS =================
async def _setup_stage(stage, inverse_gravity=False):
    """Common stage setup: physics scene, lighting, env containers."""
    setup_physics_scene(stage, inverse_gravity)
    add_lighting(stage)
    if not stage.GetPrimAtPath(ENV_BASE_PATH).IsValid():
        UsdGeom.Scope.Define(stage, ENV_BASE_PATH)
    if not stage.GetPrimAtPath(env_path(0)).IsValid():
        UsdGeom.Xform.Define(stage, env_path(0))


async def _build_cloned_envs(stage, gravity_on=False):
    """Load env_0 assets and clone to NUM_ENVS environments. Returns bottom_center."""
    log_info("Setting up Environment 0 …")

    obj_ref_0 = load_target_usd_env0(stage, OBJECT_USD_PATH, gravity_on=gravity_on)
    if obj_ref_0 is None:
        raise RuntimeError(f"Failed to load object USD: {OBJECT_USD_PATH}")

    bottom_center = compute_bottom_center(stage, obj_wrap(0))
    log_info(f"Object bottom_center: {bottom_center}")

    if gravity_on:
        # Place ground plane flush with the object bottom so it rests naturally
        GroundPlane("/World/GroundPlane", z_position=float(bottom_center[2]))

    for side in ("left", "right"):
        ref_path, num_dofs = load_hand_env0(stage, side)
        log_info(f"{side} hand: {num_dofs} DOFs  ({ref_path})")

    log_info(f"Cloning {NUM_ENVS} environments (spacing={CLONE_SPACING} m) …")
    cloner = GridCloner(spacing=CLONE_SPACING)
    cloner.define_base_env(ENV_BASE_PATH)
    env_paths = cloner.generate_paths(ENV_ROOT_PREFIX, NUM_ENVS)
    cloner.clone(source_prim_path=env_path(0), prim_paths=env_paths)

    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()

    p0, _ = get_prim_pose_robust(stage, env_path(0))
    if NUM_ENVS > 1:
        p1, _ = get_prim_pose_robust(stage, env_path(1))
        log_info(f"env_1 world pos: {p1}  (delta ≈ {p1 - p0})")

    log_info(f"Created {NUM_ENVS} parallel environments")
    return bottom_center


def _output_path(suffix: str) -> Path:
    """
    Build output path next to GRASP_JSON_PATH, tagged with HAND_TYPE and suffix.
    e.g. /path/to/grasps.json  →  /path/to/grasps_validated_dex3_1.json
    """
    src = Path(GRASP_JSON_PATH)
    return src.parent / f"{src.stem}_validated_{_hand_output_label()}{suffix}.json"


# ================= PIPELINE PASSES =================
async def run_first_pass():
    import gc
    ctx = omni.usd.get_context()
    if ctx.get_stage():
        await ctx.close_stage_async()
        await omni.kit.app.get_app().next_update_async()
        gc.collect()
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
    await ctx.new_stage_async()
    stage    = ctx.get_stage()
    timeline = get_timeline_interface()

    await _setup_stage(stage, inverse_gravity=False)
    bottom_center = await _build_cloned_envs(stage, gravity_on=True)

    grasp_list = load_grasp_data_first_pass(GRASP_JSON_PATH)
    if not grasp_list:
        log_info("No grasps to evaluate – aborting")
        return

    results = await evaluate_grasps_batched(stage, timeline, grasp_list, inverse_gravity=False, gravity_always_on=True)

    successful = [grasp_list[r["grasp_index"]] for r in results if r["success"]]
    log_info(f"First pass survivors: {len(successful)}/{len(grasp_list)}")

    total  = len(grasp_list)
    passes = len(successful)
    print(f"\n{'='*60}")
    print(f"FIRST PASS COMPLETE")
    print(f"Total Input:   {total}")
    print(f"Survived:      {passes}")
    print(f"Success Rate:  {100.0 * passes / max(total, 1):.1f}%")
    print(f"{'='*60}")

    out_file = _output_path("")
    output = {
        "type":           Path(OBJECT_USD_PATH).parent.name,
        "hand_type":      _hand_output_label(),
        "bottom_center":  bottom_center,
        "functional_grasp": {"body": successful},
        "grasp":          {},
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    log_info(f"First-pass results saved to: {out_file}")


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
    stage    = ctx.get_stage()
    timeline = get_timeline_interface()

    await _setup_stage(stage, inverse_gravity=True)
    bottom_center = await _build_cloned_envs(stage)

    first_pass_file = _output_path("")
    grasp_list = load_grasp_data_second_pass(str(first_pass_file))
    if not grasp_list:
        log_info("No first-pass survivors to re-evaluate – aborting")
        return

    results = await evaluate_grasps_batched(stage, timeline, grasp_list, inverse_gravity=True)

    successful = [grasp_list[r["grasp_index"]] for r in results if r["success"]]
    log_info(f"Second pass survivors: {len(successful)}/{len(grasp_list)}")

    total  = len(grasp_list)
    passes = len(successful)
    print(f"\n{'='*60}")
    print(f"SECOND PASS COMPLETE  (inverse gravity)")
    print(f"Total Input:   {total}")
    print(f"Survived:      {passes}")
    print(f"Success Rate:  {100.0 * passes / max(total, 1):.1f}%")
    print(f"{'='*60}")

    out_file = _output_path("_final")
    output = {
        "type":           Path(OBJECT_USD_PATH).parent.name,
        "hand_type":      _hand_output_label(),
        "bottom_center":  bottom_center,
        "functional_grasp": {"body": successful},
        "grasp":          {},
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    log_info(f"Second-pass results saved to: {out_file}")


async def run_pipeline():
    await run_first_pass()
    if INVERSE_GRAVITY:
        await run_second_pass()


# ================= ENTRY POINT =================
def main():
    try:
        task = asyncio.ensure_future(run_pipeline())
        while not task.done():
            simulation_app.update()
        if task.exception():
            raise task.exception()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
