from simulation_app_runtime import create_simulation_app, cleanup_simulation_runtime_cache
simulation_app = create_simulation_app(headless=False, script_name="single_hand_physics_validation")

import carb
carb.settings.get_settings().set("/log/level", 3)
carb.settings.get_settings().set("/log/fileLogLevel", 3)
carb.settings.get_settings().set("/rtx/instanceLogging", False)

import warnings
warnings.filterwarnings("ignore")

import asyncio
import gc
import json
import os
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.usd
import yaml
from isaacsim.core.cloner import GridCloner
from isaacsim.core.utils.stage import add_reference_to_stage
from omni.timeline import get_timeline_interface
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from validation_hand_runtime import (
    configure_authored_hand_joint_drives,
    set_authored_hand_joint_targets,
)


def log_info(msg):
    print(f"[INFO] {msg}")


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_rotate_vector(q, v):
    s = q[0]
    r = q[1:]
    return v + 2 * np.cross(r, s * v + np.cross(r, v))


def transform_pose_to_world(rel_pos, rel_rot, obj_pos, obj_rot):
    world_pos = obj_pos + quat_rotate_vector(obj_rot, rel_pos)
    world_rot = quat_mul(obj_rot, rel_rot)
    return world_pos, world_rot


def slerp(q1, q2, t):
    q = (1.0 - t) * np.asarray(q1, dtype=np.float64) + t * np.asarray(q2, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    return q / norm if norm > 1e-8 else q


def interp_pos(p1, p2, t):
    return np.asarray(p1, dtype=np.float64) * (1.0 - t) + np.asarray(p2, dtype=np.float64) * t


def interp_joints(j1, j2, t):
    keys = list(j1.keys())
    return {
        name: float(j1.get(name, 0.0)) * (1.0 - t) + float(j2.get(name, 0.0)) * t
        for name in keys
    }


def xyzw_to_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


_THIS_DIR = Path(__file__).parent
HANDS_BASE_DIR = _THIS_DIR / "assets" / "hands"

def _path_from_env(env_name: str, default: Path) -> str:
    return str(Path(os.environ.get(env_name, str(default))).expanduser())


def load_hand_yaml_config(hand_type: str, side: str) -> dict:
    config_path = HANDS_BASE_DIR / hand_type / "config" / f"{side}_hand.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Hand config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    config_dir = config_path.parent
    usd_path = (config_dir / raw["asset"]["usd_path"]).resolve()
    return {
        "side": raw["side"],
        "usd_path": str(usd_path),
        "palm_link": raw["root"]["palm_link"],
        "approach_direction_local": np.array(raw["frame_convention"]["palm_normal_local"], dtype=np.float64),
        "controllable_joints": list(raw["joints"]["controllable"]),
    }


def _infer_side_from_grasp_json_path(json_path: str) -> str | None:
    stem = Path(json_path).stem.lower()
    if "_left_" in f"_{stem}_":
        return "left"
    if "_right_" in f"_{stem}_":
        return "right"
    return None


def _effective_hand_side(configured_side: str, json_path: str) -> str:
    resolved = str(configured_side).lower()
    if resolved in {"left", "right"}:
        return resolved
    inferred = _infer_side_from_grasp_json_path(json_path)
    return inferred or "right"


# ---- Hand / object selection ----
HAND_TYPE = "dex3_1"
HAND_RUNTIME_MODE = "legacy"  # "legacy" or "authored_usd"
# For newer dex3_1 validation hands with authored collision models, keep
# HAND_TYPE="dex3_1" and point the active side at the new authored USD.
HAND_USD_OVERRIDES = {
    "left": os.environ.get("SINGLE_HAND_LEFT_USD"),
    "right": os.environ.get("SINGLE_HAND_RIGHT_USD"),
}
HAND_SIDE = "auto"  # "left", "right", or "auto" to infer from GRASP_JSON_PATH

OBJECT_USD_PATH = _path_from_env("SINGLE_HAND_OBJECT_USD", _THIS_DIR / "assets" / "objects" / "Object.usd")
GRASP_JSON_PATH = _path_from_env(
    "SINGLE_HAND_GRASP_JSON",
    _THIS_DIR / "outputs" / "physics_validation_inputs" / "cat4" / "cat4_right_apple_100_grasps.json",
)

ACTIVE_SIDE = _effective_hand_side(HAND_SIDE, GRASP_JSON_PATH)
HAND_CFG = load_hand_yaml_config(HAND_TYPE, ACTIVE_SIDE)
HAND_STRUCTURE_CACHE = {}

PHYSICS_SCENE_PATH = "/World/physicsScene"

# ---- Parallel envs ----
NUM_ENVS = 2
CLONE_SPACING = 3.0
ENV_BASE_PATH = "/World/Envs"
ENV_ROOT_PREFIX = f"{ENV_BASE_PATH}/env"

# ---- Motion / filter parameters ----
LIFT_HEIGHT = 0.3
DROP_HEIGHT_THRESHOLD = 0.02

# ---- Simulation steps ----
PREAPPROACH_STEPS = 60
APPROACH_STEPS = 60
SQUEEZE_STEPS = 60
SETTLE_STEPS = 40
CHECK_STEPS = 60
HOLD_STEPS = 50
LIFT_STEPS = 150

# ---- Pre-approach parameters ----
PREAPPROACH_STEP_BACK_DIST = 0.015

# ---- Joint drive parameters ----
JOINT_STIFFNESS = 80.0
JOINT_DAMPING = 20.0
JOINT_MAX_FORCE = 300.0
JOINT_ARMATURE = 0.01
JOINT_VELOCITY_LIMIT = 100.0

# ---- Optional second pass ----
INVERSE_GRAVITY = False


def env_path(i: int) -> str:
    return f"{ENV_ROOT_PREFIX}_{i}"


def obj_wrap(i: int) -> str:
    return f"{env_path(i)}/TargetObject"


def obj_ref(i: int) -> str:
    return f"{env_path(i)}/TargetObject/ref"


def hand_wrap(i: int) -> str:
    return f"{env_path(i)}/Hand"


def hand_ref(i: int) -> str:
    return f"{env_path(i)}/Hand/ref"


def _uses_authored_hand_runtime() -> bool:
    return HAND_RUNTIME_MODE == "authored_usd" and bool(HAND_USD_OVERRIDES.get(ACTIVE_SIDE))


def _runtime_hand_usd_path() -> str:
    return str(HAND_USD_OVERRIDES.get(ACTIVE_SIDE) or HAND_CFG["usd_path"])


def _hand_output_label() -> str:
    return f"{HAND_TYPE}_authored" if _uses_authored_hand_runtime() else HAND_TYPE


def add_lighting(stage):
    light = UsdLux.DomeLight.Define(stage, "/World/defaultDomeLight")
    light.CreateIntensityAttr().Set(1000.0)
    d_light = UsdLux.DistantLight.Define(stage, "/World/distantLight")
    d_light.CreateIntensityAttr().Set(2000.0)
    UsdGeom.Xformable(d_light).AddRotateXYZOp().Set(Gf.Vec3f(45, 45, 0))


def compute_bottom_center(stage, prim_path):
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


def make_prims_editable(stage, root_prim_path):
    root_prim = stage.GetPrimAtPath(root_prim_path)
    changed_paths = []
    if not root_prim.IsValid():
        return changed_paths
    for prim in Usd.PrimRange(root_prim):
        if prim.IsInstanceable():
            changed_paths.append(prim.GetPath().pathString)
            prim.SetInstanceable(False)
    return changed_paths


def restore_prims_instanceable(stage, prim_paths):
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.SetInstanceable(True)


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
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI(prim).Bind(UsdShade.Material(mat_prim), materialPurpose="physics")


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
    physx_rb.CreateMaxDepenetrationVelocityAttr().Set(0.5)
    physx_rb.CreateSolverPositionIterationCountAttr().Set(32)
    physx_rb.CreateSolverVelocityIterationCountAttr().Set(4)
    if not root_prim.HasAPI(UsdPhysics.MassAPI):
        UsdPhysics.MassAPI.Apply(root_prim)
    UsdPhysics.MassAPI(root_prim).CreateMassAttr().Set(0.5)
    for prim in Usd.PrimRange(root_prim):
        if str(prim.GetPath()) == root_prim_path:
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
            phys_col.CreateContactOffsetAttr().Set(0.008)
            phys_col.CreateRestOffsetAttr().Set(0.002)
    xformable = UsdGeom.Xformable(root_prim)
    xformable.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0))
    xformable.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))


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
    except Exception as exc:
        print(f"[WARN] Failed to set drive target on {joint_prim.GetPath()}: {exc}")
        return False


def setup_joint_drives(stage, container_path):
    container_prim = stage.GetPrimAtPath(container_path)
    for prim in Usd.PrimRange(container_prim):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive_kind = "angular"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            drive_kind = "linear"
        else:
            continue
        drive = UsdPhysics.DriveAPI.Apply(prim, drive_kind)
        (drive.GetStiffnessAttr() or drive.CreateStiffnessAttr()).Set(float(JOINT_STIFFNESS))
        (drive.GetDampingAttr() or drive.CreateDampingAttr()).Set(float(JOINT_DAMPING))
        (drive.GetMaxForceAttr() or drive.CreateMaxForceAttr()).Set(float(JOINT_MAX_FORCE))
        (drive.GetTargetPositionAttr() or drive.CreateTargetPositionAttr()).Set(0.0)
        (drive.GetTargetVelocityAttr() or drive.CreateTargetVelocityAttr()).Set(0.0)
        if not prim.HasAPI(PhysxSchema.PhysxJointAPI):
            PhysxSchema.PhysxJointAPI.Apply(prim)
        physx_joint = PhysxSchema.PhysxJointAPI(prim)
        physx_joint.CreateJointFrictionAttr().Set(0.05)
        physx_joint.CreateArmatureAttr().Set(float(JOINT_ARMATURE))


def setup_hand_collision(stage, container_path):
    container_prim = stage.GetPrimAtPath(container_path)
    if not container_prim.IsValid():
        return
    for prim in Usd.PrimRange(container_prim):
        if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            continue
        physx_col_api = PhysxSchema.PhysxCollisionAPI(prim)
        physx_col_api.GetContactOffsetAttr().Set(0.008)
        physx_col_api.GetRestOffsetAttr().Set(0.002)


def setup_direct_control(stage, container_path):
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
            physx_api.CreateMaxDepenetrationVelocityAttr().Set(0.5)
            physx_api.CreateSolverPositionIterationCountAttr().Set(32)
            physx_api.CreateSolverVelocityIterationCountAttr().Set(4)
    setup_joint_drives(stage, container_path)


def set_local_pose(stage, prim_path, pos, quat):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    translate_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    orient_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    if orient_op is None:
        orient_op = xform.AddOrientOp()
    translate_op.Set(Gf.Vec3f(*[float(x) for x in pos]))
    orient_op.Set(Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))


def get_prim_pose_robust(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    xformable = UsdGeom.Xformable(prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = world_transform.ExtractTranslation()
    r = world_transform.ExtractRotationQuat()
    return (
        np.array([t[0], t[1], t[2]], dtype=np.float64),
        np.array([r.GetReal(), r.GetImaginary()[0], r.GetImaginary()[1], r.GetImaginary()[2]], dtype=np.float64),
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


def zero_joints():
    return {name: 0.0 for name in HAND_CFG["controllable_joints"]}


def set_hand_joint_targets_dict(stage, container_path, joint_positions_rad: dict):
    if _uses_authored_hand_runtime():
        set_authored_hand_joint_targets(
            stage,
            container_path,
            joint_positions_rad,
            palm_link_name=HAND_CFG["palm_link"],
            controllable_joint_names=HAND_CFG["controllable_joints"],
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


def _joint_values_to_dict(joint_values):
    if isinstance(joint_values, dict):
        return {str(k): float(v) for k, v in joint_values.items()}
    if isinstance(joint_values, (list, tuple)):
        ordered_names = HAND_CFG["controllable_joints"]
        if len(joint_values) != len(ordered_names):
            raise ValueError(
                f"Joint list length mismatch: expected {len(ordered_names)}, got {len(joint_values)}"
            )
        return {
            joint_name: float(joint_values[idx])
            for idx, joint_name in enumerate(ordered_names)
        }
    raise TypeError(f"Unsupported joint payload: {type(joint_values)!r}")


def _parse_stage(stage_dict: dict):
    if "wrist_position" in stage_dict:
        pos = np.array(stage_dict["wrist_position"], dtype=np.float64)
    else:
        pos = np.array(stage_dict["position"], dtype=np.float64)

    if "wrist_quaternion_wxyz" in stage_dict:
        quat = np.array(stage_dict["wrist_quaternion_wxyz"], dtype=np.float64)
    elif "wrist_quaternion_xyzw" in stage_dict:
        quat = xyzw_to_wxyz(np.array(stage_dict["wrist_quaternion_xyzw"], dtype=np.float64))
    else:
        quat = np.array(stage_dict["orientation"], dtype=np.float64)

    if "joint_positions" in stage_dict:
        joints = _joint_values_to_dict(stage_dict["joint_positions"])
    else:
        joints = _joint_values_to_dict(stage_dict["joints"])
    return pos, quat, joints


def load_single_hand_grasp_payload(json_path: str):
    if not os.path.exists(json_path):
        log_info(f"File not found: {json_path}")
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "functional_grasp" not in data or "body" not in data["functional_grasp"]:
        log_info("Invalid JSON: expected top-level 'functional_grasp.body'")
        return None
    grasps = data["functional_grasp"]["body"]
    log_info(f"Loaded {len(grasps)} single-hand grasps for side '{ACTIVE_SIDE}'.")
    return data


def _output_payload(input_payload: dict, successful_grasps: list[dict], computed_bottom_center: list[float]) -> dict:
    object_type = input_payload.get("type", Path(OBJECT_USD_PATH).parent.name)
    bottom_center = input_payload.get("bottom_center", computed_bottom_center)
    return {
        "type": object_type,
        "bottom_center": [float(x) for x in bottom_center],
        "functional_grasp": {"body": successful_grasps},
        "grasp": input_payload.get("grasp", {}),
    }


def _output_path(suffix: str) -> Path:
    src = Path(GRASP_JSON_PATH)
    return src.parent / f"{src.stem}_validated_{_hand_output_label()}_{ACTIVE_SIDE}{suffix}.json"


def load_target_usd_env0(stage, usd_path):
    if not os.path.exists(usd_path):
        return None
    wrapper_path = obj_wrap(0)
    ref_path = obj_ref(0)
    UsdGeom.Xform.Define(stage, wrapper_path)
    add_reference_to_stage(usd_path=usd_path, prim_path=ref_path)
    changed = make_prims_editable(stage, ref_path)
    setup_object_physics(stage, wrapper_path, gravity_on=False)
    create_and_bind_high_friction_material(stage, wrapper_path)
    restore_prims_instanceable(stage, changed)
    return ref_path


def load_hand_env0(stage):
    wrap_path = hand_wrap(0)
    ref_path = hand_ref(0)
    UsdGeom.Xform.Define(stage, wrap_path)
    add_reference_to_stage(usd_path=_runtime_hand_usd_path(), prim_path=ref_path)
    changed = make_prims_editable(stage, ref_path)
    create_and_bind_high_friction_material(stage, ref_path)
    if _uses_authored_hand_runtime():
        log_info(f"Using authored collision-model runtime for {ACTIVE_SIDE} hand")
        hand_structure = configure_authored_hand_joint_drives(
            stage,
            ref_path,
            palm_link_name=HAND_CFG["palm_link"],
            controllable_joint_names=HAND_CFG["controllable_joints"],
            joint_stiffness=JOINT_STIFFNESS,
            joint_damping=JOINT_DAMPING,
            joint_max_force=JOINT_MAX_FORCE,
            joint_armature=JOINT_ARMATURE,
            joint_velocity_limit=JOINT_VELOCITY_LIMIT,
            cache=HAND_STRUCTURE_CACHE,
            logger=log_info,
        )
        if hand_structure is None:
            raise RuntimeError(f"Failed to resolve authored hand structure under {ref_path}")
    else:
        setup_direct_control(stage, ref_path)
        setup_hand_collision(stage, ref_path)
    restore_prims_instanceable(stage, changed)
    return ref_path


async def evaluate_grasps_batched(stage, timeline, grasp_list, *, inverse_gravity=False):
    total = len(grasp_list)
    n_batch = (total + NUM_ENVS - 1) // NUM_ENVS
    log_info(f"Batched single-hand evaluation: {total} grasps in {n_batch} batch(es) of up to {NUM_ENVS}")

    results = []
    nominal_obj_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    nominal_obj_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    approach_dir = HAND_CFG["approach_direction_local"]

    for batch_idx in range(n_batch):
        s_idx = batch_idx * NUM_ENVS
        e_idx = min(s_idx + NUM_ENVS, total)
        batch = grasp_list[s_idx:e_idx]
        K = len(batch)

        print(f"\n{'='*60}")
        print(f"SINGLE-HAND BATCH {batch_idx+1}/{n_batch}: grasps {s_idx}–{e_idx-1}")
        print(f"{'='*60}")

        coarse_pos, coarse_rot, coarse_jts = [], [], []
        fine_pos, fine_rot, fine_jts = [], [], []
        final_pos, final_rot, final_jts = [], [], []

        for grasp in batch:
            cp, cr, cj = _parse_stage(grasp["coarse_grasp"])
            fp, fr, fj = _parse_stage(grasp["fine_grasp"])
            sp, sr, sj = _parse_stage(grasp["final_grasp"])

            cwp, cwr = transform_pose_to_world(cp, cr, nominal_obj_pos, nominal_obj_rot)
            fwp, fwr = transform_pose_to_world(fp, fr, nominal_obj_pos, nominal_obj_rot)
            swp, swr = transform_pose_to_world(sp, sr, nominal_obj_pos, nominal_obj_rot)

            coarse_pos.append(cwp)
            coarse_rot.append(cwr)
            coarse_jts.append(cj)
            fine_pos.append(fwp)
            fine_rot.append(fwr)
            fine_jts.append(fj)
            final_pos.append(swp)
            final_rot.append(swr)
            final_jts.append(sj)

        preapproach_pos = [
            coarse_pos[k] - PREAPPROACH_STEP_BACK_DIST * quat_rotate_vector(coarse_rot[k], approach_dir)
            for k in range(K)
        ]

        if timeline.is_playing():
            timeline.stop()
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

        for k in range(K):
            obj_prim = stage.GetPrimAtPath(obj_wrap(k))
            if obj_prim.IsValid() and obj_prim.IsA(UsdGeom.Xformable):
                xf = UsdGeom.Xformable(obj_prim)
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0))
                xf.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
            set_all_gravity(stage, [obj_wrap(k)], enable_gravity=False)
            clear_all_velocities(stage, [obj_wrap(k)])
            set_local_pose(stage, hand_wrap(k), preapproach_pos[k], coarse_rot[k])
            set_hand_joint_targets_dict(stage, hand_ref(k), zero_joints())

        if not timeline.is_playing():
            timeline.play()
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()
        for _ in range(60):
            await omni.kit.app.get_app().next_update_async()

        initial_z = []
        for k in range(K):
            p, _ = get_prim_pose_robust(stage, obj_wrap(k))
            initial_z.append(float(p[2]))

        active = [True] * K

        print(f"  [Batch {batch_idx+1}] Phase 0: preapproach → coarse ({PREAPPROACH_STEPS} steps)")
        for t in range(PREAPPROACH_STEPS + 1):
            alpha = min(t / float(PREAPPROACH_STEPS), 1.0)
            for k in range(K):
                if not active[k]:
                    continue
                set_local_pose(stage, hand_wrap(k), interp_pos(preapproach_pos[k], coarse_pos[k], alpha), coarse_rot[k])
                set_hand_joint_targets_dict(stage, hand_ref(k), coarse_jts[k])
            await omni.kit.app.get_app().next_update_async()

        print(f"  [Batch {batch_idx+1}] Phase 1: coarse → fine ({APPROACH_STEPS} steps)")
        for t in range(APPROACH_STEPS + 1):
            alpha = min(t / float(APPROACH_STEPS), 1.0)
            for k in range(K):
                if not active[k]:
                    continue
                set_local_pose(
                    stage,
                    hand_wrap(k),
                    interp_pos(coarse_pos[k], fine_pos[k], alpha),
                    slerp(coarse_rot[k], fine_rot[k], alpha),
                )
                set_hand_joint_targets_dict(stage, hand_ref(k), interp_joints(coarse_jts[k], fine_jts[k], alpha))
            await omni.kit.app.get_app().next_update_async()

        print(f"  [Batch {batch_idx+1}] Phase 2: fine → final ({SQUEEZE_STEPS} steps)")
        for t in range(SQUEEZE_STEPS + 1):
            alpha = min(t / float(SQUEEZE_STEPS), 1.0)
            for k in range(K):
                if not active[k]:
                    continue
                set_local_pose(
                    stage,
                    hand_wrap(k),
                    interp_pos(fine_pos[k], final_pos[k], alpha),
                    slerp(fine_rot[k], final_rot[k], alpha),
                )
                set_hand_joint_targets_dict(stage, hand_ref(k), interp_joints(fine_jts[k], final_jts[k], alpha))
            await omni.kit.app.get_app().next_update_async()

        print(f"  [Batch {batch_idx+1}] Settle at final ({SETTLE_STEPS} steps)")
        for _ in range(SETTLE_STEPS):
            for k in range(K):
                if active[k]:
                    set_hand_joint_targets_dict(stage, hand_ref(k), final_jts[k])
            await omni.kit.app.get_app().next_update_async()

        print(f"  [Batch {batch_idx+1}] Gravity on, settling ({CHECK_STEPS} steps)")
        for k in range(K):
            if active[k]:
                set_all_gravity(stage, [obj_wrap(k)], enable_gravity=True)
        for _ in range(CHECK_STEPS):
            for k in range(K):
                if active[k]:
                    set_hand_joint_targets_dict(stage, hand_ref(k), final_jts[k])
            await omni.kit.app.get_app().next_update_async()

        for k in range(K):
            if not active[k]:
                continue
            p_now, _ = get_prim_pose_robust(stage, obj_wrap(k))
            dropped = (p_now[2] > initial_z[k] + DROP_HEIGHT_THRESHOLD) if inverse_gravity else (
                p_now[2] < initial_z[k] - DROP_HEIGHT_THRESHOLD
            )
            if dropped:
                active[k] = False
                print(f"  [Env {k}] REJECTED: dropped during settling")

        print(f"  [Batch {batch_idx+1}] Hold ({HOLD_STEPS} steps)")
        for _ in range(HOLD_STEPS):
            for k in range(K):
                if active[k]:
                    set_hand_joint_targets_dict(stage, hand_ref(k), final_jts[k])
            await omni.kit.app.get_app().next_update_async()

        lift_dir = -1.0 if inverse_gravity else 1.0
        target_lift = [final_pos[k] + np.array([0.0, 0.0, lift_dir * LIFT_HEIGHT], dtype=np.float64) for k in range(K)]

        print(f"  [Batch {batch_idx+1}] Lift ({LIFT_STEPS} steps)")
        for t in range(LIFT_STEPS + 1):
            alpha = min(t / float(LIFT_STEPS), 1.0)
            for k in range(K):
                if not active[k]:
                    continue
                set_local_pose(stage, hand_wrap(k), interp_pos(final_pos[k], target_lift[k], alpha), final_rot[k])
                set_hand_joint_targets_dict(stage, hand_ref(k), final_jts[k])
            await omni.kit.app.get_app().next_update_async()

        success = [False] * K
        fail_reasons = [""] * K
        for k in range(K):
            if not active[k]:
                fail_reasons[k] = "Failed earlier"
                continue
            p_now, _ = get_prim_pose_robust(stage, obj_wrap(k))
            expected_z = initial_z[k] + lift_dir * LIFT_HEIGHT
            dropped = (p_now[2] > expected_z + DROP_HEIGHT_THRESHOLD) if inverse_gravity else (
                p_now[2] < expected_z - DROP_HEIGHT_THRESHOLD
            )
            if dropped:
                fail_reasons[k] = "Dropped during lift"
                print(f"  [Env {k}] FAILED: dropped during lift")
            else:
                success[k] = True
                print(f"  [Env {k}] SUCCESS!")

        for k in range(K):
            set_all_gravity(stage, [obj_wrap(k)], enable_gravity=False)

        timeline.stop()

        for i in range(K):
            results.append(
                {
                    "grasp_index": s_idx + i,
                    "success": success[i],
                    "fail_reason": fail_reasons[i],
                }
            )

        print(f"\nBatch {batch_idx+1} done: {sum(success)}/{K} succeeded")

    return results


async def _setup_stage(stage, inverse_gravity=False):
    setup_physics_scene(stage, inverse_gravity=inverse_gravity)
    add_lighting(stage)
    if not stage.GetPrimAtPath(ENV_BASE_PATH).IsValid():
        UsdGeom.Scope.Define(stage, ENV_BASE_PATH)
    if not stage.GetPrimAtPath(env_path(0)).IsValid():
        UsdGeom.Xform.Define(stage, env_path(0))


async def _build_cloned_envs(stage):
    log_info("Setting up single-hand Environment 0 …")
    obj_ref_0 = load_target_usd_env0(stage, OBJECT_USD_PATH)
    if obj_ref_0 is None:
        raise RuntimeError(f"Failed to load object USD: {OBJECT_USD_PATH}")
    bottom_center = compute_bottom_center(stage, obj_wrap(0))
    log_info(f"Object bottom_center: {bottom_center}")
    hand_ref_0 = load_hand_env0(stage)
    log_info(f"{ACTIVE_SIDE} hand loaded at {hand_ref_0}")
    cloner = GridCloner(spacing=CLONE_SPACING)
    cloner.define_base_env(ENV_BASE_PATH)
    env_paths = cloner.generate_paths(ENV_ROOT_PREFIX, NUM_ENVS)
    cloner.clone(source_prim_path=env_path(0), prim_paths=env_paths)
    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()
    log_info(f"Created {NUM_ENVS} parallel single-hand environments")
    return bottom_center


async def _new_clean_stage():
    ctx = omni.usd.get_context()
    if ctx.get_stage():
        await ctx.close_stage_async()
        await omni.kit.app.get_app().next_update_async()
        gc.collect()
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
    await ctx.new_stage_async()
    return ctx.get_stage(), get_timeline_interface()


async def run_first_pass():
    stage, timeline = await _new_clean_stage()
    await _setup_stage(stage, inverse_gravity=False)
    computed_bottom_center = await _build_cloned_envs(stage)

    input_payload = load_single_hand_grasp_payload(GRASP_JSON_PATH)
    if not input_payload:
        log_info("No grasps to evaluate – aborting")
        return

    grasp_list = input_payload["functional_grasp"]["body"]
    results = await evaluate_grasps_batched(stage, timeline, grasp_list, inverse_gravity=False)
    successful = [grasp_list[r["grasp_index"]] for r in results if r["success"]]
    log_info(f"First pass survivors: {len(successful)}/{len(grasp_list)}")

    print(f"\n{'='*60}")
    print("SINGLE-HAND VALIDATION COMPLETE")
    print(f"Total Input:   {len(grasp_list)}")
    print(f"Survived:      {len(successful)}")
    print(f"Success Rate:  {100.0 * len(successful) / max(len(grasp_list), 1):.1f}%")
    print(f"{'='*60}")

    out_file = _output_path("")
    output = _output_payload(input_payload, successful, computed_bottom_center)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    log_info(f"Single-hand validation results saved to: {out_file}")


async def run_second_pass():
    stage, timeline = await _new_clean_stage()
    await _setup_stage(stage, inverse_gravity=True)
    computed_bottom_center = await _build_cloned_envs(stage)

    first_pass_file = _output_path("")
    input_payload = load_single_hand_grasp_payload(str(first_pass_file))
    if not input_payload:
        log_info("No first-pass survivors to re-evaluate – aborting")
        return

    grasp_list = input_payload["functional_grasp"]["body"]
    results = await evaluate_grasps_batched(stage, timeline, grasp_list, inverse_gravity=True)
    successful = [grasp_list[r["grasp_index"]] for r in results if r["success"]]
    log_info(f"Second pass survivors: {len(successful)}/{len(grasp_list)}")

    print(f"\n{'='*60}")
    print("SINGLE-HAND SECOND PASS COMPLETE  (inverse gravity)")
    print(f"Total Input:   {len(grasp_list)}")
    print(f"Survived:      {len(successful)}")
    print(f"Success Rate:  {100.0 * len(successful) / max(len(grasp_list), 1):.1f}%")
    print(f"{'='*60}")

    out_file = _output_path("_final")
    output = _output_payload(input_payload, successful, computed_bottom_center)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    log_info(f"Second-pass results saved to: {out_file}")


async def run_pipeline():
    await run_first_pass()
    if INVERSE_GRAVITY:
        await run_second_pass()


def main():
    try:
        task = asyncio.ensure_future(run_pipeline())
        while not task.done():
            simulation_app.update()
        if task.exception():
            raise task.exception()
    finally:
        simulation_app.close()
        cleanup_simulation_runtime_cache()


if __name__ == "__main__":
    main()
