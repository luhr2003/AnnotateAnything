"""
bi_gripper_reeval.py
====================
Re-run physics validation on poses stored in bi_gripper_grasp_pose.json.

Pipeline
--------
1. Read JSON → extract stored (left, right) target poses
2. Set up Isaac Sim stage: Object + GripperL + GripperR
3. GridCloner → NUM_COPIES parallel envs
4. Per batch: step-back → approach (both) → close (both) → lift (both) → Z-rise check
5. Write a new JSON that contains only the poses that pass validation

Approach direction recovery
---------------------------
The generator applied a 180-deg rotation around local X to the anchor frame:
    gripper_rot = anchor_rot * R(pi, X)
The anchor EEF z-axis equals the surface normal, so after the flip:
    gripper_rot * [0,0,1] = -surface_normal
We therefore recover the step-back direction as:
    surface_normal = -R(target_quat).apply([0,0,1])

Usage
-----
    Edit INPUT_JSON and OBJECT_USD below, then:
    python bi_gripper_reeval.py
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import asyncio
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciRot
import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
from isaacsim.core.cloner import GridCloner
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
from omni.timeline import get_timeline_interface

timeline = get_timeline_interface()

# ============================================================
# INPUT — edit these two paths before running
# ============================================================
_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _THIS_DIR.parent

def _path_from_env(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default))).expanduser()

INPUT_JSON = _path_from_env(
    "BIGRIPPER_INPUT_JSON",
    _THIS_DIR / "Annotation" / "bi_gripper_grasp_pose.json",
)
OBJECT_USD = _path_from_env("BIGRIPPER_OBJECT_USD", _THIS_DIR / "Object.usd")

# ============================================================
# CONFIG — keep in sync with bi_gripper_grasp_gen.py
# ============================================================
GRIPPER_USD = _path_from_env("BIGRIPPER_GRIPPER_USD", _WORKSPACE_ROOT / "Flying_hand_probe_pro.usd")

GRIPPER_FINGERTIP_OFFSET = 0.14

NUM_COPIES      = 1
CLONE_SPACING   = 3.0
GROUND_Z        = -10.0
ENV_BASE_PATH   = "/World/Envs"
ENV_ROOT_PREFIX = f"{ENV_BASE_PATH}/env"
PHYSICS_SCENE_PATH = "/World/physicsScene"

APPROACH_DISTANCE = 0.20
APPROACH_STEPS    = 80

CLOSE_STEPS       = 32
HOLD_STEPS        = 30
GRIPPER_OPEN_POS  = 0.04
GRIPPER_CLOSE_POS = 0.0

LIFT_DISTANCE   = 0.3
LIFT_STEPS      = 80
LIFT_SUCCESS_Z  = 0.28

HIGH_STATIC_FRICTION  = 2.0
HIGH_DYNAMIC_FRICTION = 2.0


# ============================================================
# USD path helpers
# ============================================================
def env_path(i):   return f"{ENV_ROOT_PREFIX}_{i}"
def obj_wrap(i):   return f"{env_path(i)}/Object"
def obj_ref(i):    return f"{env_path(i)}/Object/ref"
def gripl_wrap(i): return f"{env_path(i)}/GripperL"
def gripl_ref(i):  return f"{env_path(i)}/GripperL/ref"
def gripr_wrap(i): return f"{env_path(i)}/GripperR"
def gripr_ref(i):  return f"{env_path(i)}/GripperR/ref"


# ============================================================
# Physics helpers (identical to bi_gripper_grasp_gen.py)
# ============================================================
async def step_sim(n: int = 1):
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


def _get_world_pos(prim) -> np.ndarray:
    xfc = UsdGeom.XformCache(Usd.TimeCode.Default())
    mat = xfc.GetLocalToWorldTransform(prim)
    return np.array([mat[3][0], mat[3][1], mat[3][2]])


def _setup_physics_scene(stage):
    prim = stage.GetPrimAtPath(PHYSICS_SCENE_PATH)
    if not prim.IsValid():
        prim = stage.DefinePrim(PHYSICS_SCENE_PATH, "PhysicsScene")
    if not prim.HasAPI(UsdPhysics.Scene):
        scene = UsdPhysics.Scene.Define(stage, PHYSICS_SCENE_PATH)
    else:
        scene = UsdPhysics.Scene(prim)
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    if not prim.HasAPI(PhysxSchema.PhysxSceneAPI):
        PhysxSchema.PhysxSceneAPI.Apply(prim)
    physx = PhysxSchema.PhysxSceneAPI.Apply(prim)
    physx.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(65536)
    physx.CreateGpuTotalAggregatePairsCapacityAttr().Set(65536)


def _apply_convex_decomp(stage, root_path: str):
    root = stage.GetPrimAtPath(root_path)
    count = 0
    for p in Usd.PrimRange(root):
        if not p.IsA(UsdGeom.Mesh):
            continue
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(p)
        mapi = UsdPhysics.MeshCollisionAPI.Apply(p)
        mapi.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)
        count += 1
    print(f"[INFO]   Applied convex decomp to {count} meshes under {root_path}")


def _get_or_create_attr(prim, name: str, sdf_type):
    attr = prim.GetAttribute(name)
    if not attr or not attr.IsValid():
        attr = prim.CreateAttribute(name, sdf_type)
    return attr


def _create_physics_material(stage, mat_path: str) -> UsdShade.Material:
    parent = "/".join(mat_path.split("/")[:-1])
    if parent and not stage.GetPrimAtPath(parent).IsValid():
        UsdGeom.Xform.Define(stage, parent)
    mat = UsdShade.Material.Define(stage, mat_path)
    mat_prim = mat.GetPrim()
    pm = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)

    try:
        pm.CreateStaticFrictionAttr().Set(float(HIGH_STATIC_FRICTION))
    except Exception:
        _get_or_create_attr(mat_prim, "physxMaterial:staticFriction",
                            Sdf.ValueTypeNames.Float).Set(float(HIGH_STATIC_FRICTION))
    try:
        pm.CreateDynamicFrictionAttr().Set(float(HIGH_DYNAMIC_FRICTION))
    except Exception:
        _get_or_create_attr(mat_prim, "physxMaterial:dynamicFriction",
                            Sdf.ValueTypeNames.Float).Set(float(HIGH_DYNAMIC_FRICTION))
    try:
        pm.CreateRestitutionAttr().Set(0.0)
    except Exception:
        _get_or_create_attr(mat_prim, "physxMaterial:restitution",
                            Sdf.ValueTypeNames.Float).Set(0.0)

    for attr_name, token_val in (
        ("physxMaterial:frictionCombineMode",    "multiply"),
        ("physxMaterial:restitutionCombineMode", "multiply"),
    ):
        a = mat_prim.GetAttribute(attr_name)
        if not a or not a.IsValid():
            a = mat_prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Token)
        a.Set(token_val)
    return mat


def _bind_physics_material(stage, root_path: str, mat: UsdShade.Material):
    root = stage.GetPrimAtPath(root_path)
    for p in Usd.PrimRange(root):
        if not p.IsA(UsdGeom.Mesh):
            continue
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(p)
        api = UsdShade.MaterialBindingAPI(p)
        try:
            api.Bind(mat, materialPurpose="physics")
        except Exception:
            api.Bind(mat)


def _make_object_rigid(stage, wrapper_path: str):
    prim = stage.GetPrimAtPath(wrapper_path)
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr().Set(True)
    if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb = PhysxSchema.PhysxRigidBodyAPI(prim)
    rb.CreateDisableGravityAttr().Set(True)
    rb.CreateContactSlopCoefficientAttr().Set(2.0)


def _set_gripper_pose(stage, wrapper_path: str,
                      local_pos: np.ndarray, quat_wxyz: np.ndarray):
    prim = stage.GetPrimAtPath(wrapper_path)
    if not prim.IsValid():
        return
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    t_op = next((o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    r_op = next((o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
    if t_op is None:
        t_op = xf.AddTranslateOp()
    if r_op is None:
        r_op = xf.AddOrientOp()
    t_op.Set(Gf.Vec3d(float(local_pos[0]), float(local_pos[1]), float(local_pos[2])))
    w, x, y, z = (float(v) for v in quat_wxyz)
    quat = (Gf.Quatf(w, x, y, z)
            if r_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat
            else Gf.Quatd(w, x, y, z))
    r_op.Set(quat)


def _set_gripper_fingers(stage, gripper_ref_path: str, target: float):
    for joint_path in (
        f"{gripper_ref_path}/panda_hand/panda_finger_joint1",
        f"{gripper_ref_path}/panda_hand/panda_finger_joint2",
    ):
        prim = stage.GetPrimAtPath(joint_path)
        if not prim.IsValid():
            continue
        for prop in prim.GetProperties():
            if "drive" in prop.GetName() and "targetPosition" in prop.GetName():
                prop.Set(float(target))


def _set_obj_gravity(stage, wrapper_path: str, disable: bool):
    prim = stage.GetPrimAtPath(wrapper_path)
    if not prim.IsValid():
        return
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(disable)


def _reset_obj_pose(stage, wrapper_path: str):
    prim = stage.GetPrimAtPath(wrapper_path)
    if not prim.IsValid():
        return
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    t_op = next((o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    if t_op is None:
        t_op = xf.AddTranslateOp()
    t_op.Set(Gf.Vec3d(0, 0, 0))
    r_op = next((o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
    if r_op is not None:
        identity = (Gf.Quatf(1, 0, 0, 0)
                    if r_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat
                    else Gf.Quatd(1, 0, 0, 0))
        r_op.Set(identity)
    for attr_name, sdf_type in [
        ("physics:velocity",        Sdf.ValueTypeNames.Vector3f),
        ("physics:angularVelocity", Sdf.ValueTypeNames.Vector3f),
    ]:
        a = prim.GetAttribute(attr_name)
        if not (a and a.IsValid()):
            a = prim.CreateAttribute(attr_name, sdf_type)
        a.Set(Gf.Vec3f(0, 0, 0))


# ============================================================
# Approach direction recovery
# ============================================================
def _approach_normal_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Recover the step-back direction (surface normal) from the stored quaternion.

    During generation:
        gripper_rot = anchor_rot * R(pi around local X)
        anchor EEF z = surface_normal
        → gripper_rot * [0,0,1] = -surface_normal
    So: surface_normal = -R(target_quat).apply([0,0,1])
    """
    w, x, y, z = quat_wxyz
    rot = SciRot.from_quat([x, y, z, w])   # scipy convention: [x,y,z,w]
    normal = -rot.apply(np.array([0.0, 0.0, 1.0]))
    return normal / (np.linalg.norm(normal) + 1e-12)


# ============================================================
# Main async pipeline
# ============================================================
async def run():
    ctx = omni.usd.get_context()

    # ------------------------------------------------------------------
    # 1. Read JSON
    # ------------------------------------------------------------------
    with INPUT_JSON.open() as f:
        data = json.load(f)

    object_type   = data.get("type", "Unknown")
    bottom_center = data.get("bottom_center", [0.0, 0.0, 0.0])
    grasp_body    = data.get("functional_grasp", {}).get("body", [])
    print(f"[INFO] Loaded {len(grasp_body)} poses from {INPUT_JSON}")

    poses_left  = [np.array(e["left"],  dtype=np.float64) for e in grasp_body]
    poses_right = [np.array(e["right"], dtype=np.float64) for e in grasp_body]

    # ------------------------------------------------------------------
    # 2. Isaac Sim stage
    # ------------------------------------------------------------------
    print("[INFO] Creating Isaac Sim stage...")
    if ctx.get_stage():
        await ctx.close_stage_async()
        await step_sim(5)
    await ctx.new_stage_async()
    stage = ctx.get_stage()

    world = stage.GetPrimAtPath("/World")
    if not world.IsValid():
        world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        stage.SetDefaultPrim(world)

    UsdGeom.Scope.Define(stage, ENV_BASE_PATH)
    UsdGeom.Xform.Define(stage, env_path(0))

    # ------------------------------------------------------------------
    # 3. Object in env_0
    # ------------------------------------------------------------------
    UsdGeom.Xform.Define(stage, obj_wrap(0))
    add_reference_to_stage(str(OBJECT_USD), obj_ref(0))
    await step_sim(2)

    ref_prim = stage.GetPrimAtPath(obj_ref(0))
    if ref_prim.IsValid():
        for p in Usd.PrimRange(ref_prim):
            for api_cls in [UsdPhysics.RigidBodyAPI, PhysxSchema.PhysxRigidBodyAPI]:
                if p.HasAPI(api_cls):
                    p.RemoveAPI(api_cls)

    _apply_convex_decomp(stage, obj_ref(0))
    _make_object_rigid(stage, obj_wrap(0))

    # ------------------------------------------------------------------
    # 4. Two grippers in env_0
    # ------------------------------------------------------------------
    for wrap_fn, ref_fn in [(gripl_wrap, gripl_ref), (gripr_wrap, gripr_ref)]:
        UsdGeom.Xform.Define(stage, wrap_fn(0))
        add_reference_to_stage(str(GRIPPER_USD), ref_fn(0))
    await step_sim(2)

    # ------------------------------------------------------------------
    # 5. Physics scene + ground + high-friction material
    # ------------------------------------------------------------------
    _setup_physics_scene(stage)
    GroundPlane(prim_path="/World/GroundPlane", z_position=GROUND_Z)

    mat = _create_physics_material(stage, "/World/PhysicsMat/HF")
    _bind_physics_material(stage, obj_ref(0), mat)
    await step_sim(2)

    # ------------------------------------------------------------------
    # 6. Make refs instanceable, then clone
    # ------------------------------------------------------------------
    stage.GetPrimAtPath(obj_ref(0)).SetInstanceable(True)
    await step_sim(2)

    print(f"[INFO] Cloning {NUM_COPIES} envs (spacing={CLONE_SPACING} m)...")
    cloner = GridCloner(spacing=CLONE_SPACING)
    cloner.define_base_env(ENV_BASE_PATH)
    env_paths_list = cloner.generate_paths(ENV_ROOT_PREFIX, NUM_COPIES)
    cloner.clone(source_prim_path=env_path(0), prim_paths=env_paths_list)
    await step_sim(4)
    print("[INFO] Cloning done.")
    obj_prims = [stage.GetPrimAtPath(obj_wrap(k)) for k in range(NUM_COPIES)]

    # ------------------------------------------------------------------
    # 7. Start simulation
    # ------------------------------------------------------------------
    timeline.play()
    await step_sim(5)
    print("[INFO] Simulation running.")

    # ------------------------------------------------------------------
    # 8. Batch validation loop
    # ------------------------------------------------------------------
    valid_poses = []
    n_total     = len(poses_left)

    for base in range(0, n_total, NUM_COPIES):
        batch_idx = list(range(base, min(base + NUM_COPIES, n_total)))
        K = len(batch_idx)
        print(f"[INFO] Batch {base // NUM_COPIES + 1}: "
              f"poses {base + 1}..{base + K} / {n_total}")

        target_l = [poses_left[i][:3]  for i in batch_idx]
        target_r = [poses_right[i][:3] for i in batch_idx]
        quat_l   = [poses_left[i][3:]  for i in batch_idx]
        quat_r   = [poses_right[i][3:] for i in batch_idx]

        nrm_l  = [_approach_normal_from_quat(quat_l[k]) for k in range(K)]
        nrm_r  = [_approach_normal_from_quat(quat_r[k]) for k in range(K)]
        start_l = [target_l[k] + APPROACH_DISTANCE * nrm_l[k] for k in range(K)]
        start_r = [target_r[k] + APPROACH_DISTANCE * nrm_r[k] for k in range(K)]

        # ---- Reset envs ----
        for k in range(K):
            _reset_obj_pose(stage, obj_wrap(k))
            _set_obj_gravity(stage, obj_wrap(k), disable=True)
            _set_gripper_fingers(stage, gripl_ref(k), GRIPPER_OPEN_POS)
            _set_gripper_fingers(stage, gripr_ref(k), GRIPPER_OPEN_POS)
            _set_gripper_pose(stage, gripl_wrap(k), start_l[k], quat_l[k])
            _set_gripper_pose(stage, gripr_wrap(k), start_r[k], quat_r[k])
        await step_sim(5)

        z_pre = [float(_get_world_pos(obj_prims[k])[2]) for k in range(K)]

        # ---- Approach ----
        for t in range(APPROACH_STEPS + 1):
            alpha = t / float(APPROACH_STEPS)
            for k in range(K):
                cl = start_l[k] * (1 - alpha) + target_l[k] * alpha
                cr = start_r[k] * (1 - alpha) + target_r[k] * alpha
                _set_gripper_pose(stage, gripl_wrap(k), cl, quat_l[k])
                _set_gripper_pose(stage, gripr_wrap(k), cr, quat_r[k])
            await omni.kit.app.get_app().next_update_async()

        await step_sim(5)

        # Gate: reject if approach disturbed object
        active = [abs(float(_get_world_pos(obj_prims[k])[2]) - z_pre[k]) <= 0.03
                  for k in range(K)]

        # ---- Close grippers ----
        for _ in range(CLOSE_STEPS):
            for k in range(K):
                if not active[k]:
                    continue
                _set_gripper_fingers(stage, gripl_ref(k), GRIPPER_CLOSE_POS)
                _set_gripper_fingers(stage, gripr_ref(k), GRIPPER_CLOSE_POS)
            await omni.kit.app.get_app().next_update_async()

        await step_sim(HOLD_STEPS)

        z_contact = [
            float(_get_world_pos(obj_prims[k])[2]) if active[k] else None
            for k in range(K)
        ]

        for k in range(K):
            if active[k]:
                _set_obj_gravity(stage, obj_wrap(k), disable=False)
        await step_sim(3)

        # ---- Lift ----
        lift_l = [target_l[k] + np.array([0.0, 0.0, LIFT_DISTANCE]) for k in range(K)]
        lift_r = [target_r[k] + np.array([0.0, 0.0, LIFT_DISTANCE]) for k in range(K)]

        for t in range(LIFT_STEPS + 1):
            alpha = t / float(LIFT_STEPS)
            for k in range(K):
                if not active[k]:
                    continue
                cl = target_l[k] * (1 - alpha) + lift_l[k] * alpha
                cr = target_r[k] * (1 - alpha) + lift_r[k] * alpha
                _set_gripper_pose(stage, gripl_wrap(k), cl, quat_l[k])
                _set_gripper_pose(stage, gripr_wrap(k), cr, quat_r[k])
            await omni.kit.app.get_app().next_update_async()

        await step_sim(5)

        # ---- Check success ----
        for k in range(K):
            if not active[k] or z_contact[k] is None:
                continue
            z_after = float(_get_world_pos(obj_prims[k])[2])
            if (z_after - z_contact[k]) >= LIFT_SUCCESS_Z:
                valid_poses.append(grasp_body[batch_idx[k]])

        for k in range(K):
            _set_obj_gravity(stage, obj_wrap(k), disable=True)

    print(f"[INFO] {len(valid_poses)} / {n_total} poses passed re-validation.")

    # ------------------------------------------------------------------
    # 9. Write output JSON (overwrites input)
    # ------------------------------------------------------------------
    result = {
        "type":             object_type,
        "bottom_center":    bottom_center,
        "functional_grasp": {"body": valid_poses},
    }
    INPUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"[INFO] Saved {len(valid_poses)} re-validated poses → {INPUT_JSON}")

    timeline.stop()


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        task = asyncio.ensure_future(run())
        while not task.done():
            simulation_app.update()
        if task.exception():
            raise task.exception()
    except KeyboardInterrupt:
        print("[INFO] Interrupted.")
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()
