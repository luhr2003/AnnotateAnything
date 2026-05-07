"""
bi_gripper_grasp_gen.py
=======================
Physics validation for bi-gripper EEF pairs using Isaac Sim grid cloner.

Pipeline
--------
1. Load mesh (pure pxr) → compute edge-opposite EEF pairs in mapped XY
   (no Isaac Sim stage needed)
2. Set up Isaac Sim stage: env_0 = Object + GripperL + GripperR
3. GridCloner → NUM_COPIES parallel envs
4. Per batch: step-back → approach (both) → close (both) → lift (both) → Z-rise check
5. Save valid pairs to <object_dir>/Annotation/bi_gripper_grasp_pose.json

Output format
-------------
{
  "type": "<object_type>",
  "bottom_center": [x, y, z],
  "functional_grasp": {
    "body": [
      {"left": [x, y, z, qw, qx, qy, qz], "right": [x, y, z, qw, qx, qy, qz]},
      ...
    ]
  }
}

Usage
-----
    python bi_gripper_grasp_gen.py --object_usd /path/to/Object.usd --object_type Bin
    python bi_gripper_grasp_gen.py --dataset_dir /path/to/INPUT_DIR
    python bi_gripper_grasp_gen.py --dataset_dir /path/to/INPUT_DIR --obj_cat Bin
    python bi_gripper_grasp_gen.py --dataset_dir /path/to/INPUT_DIR --obj_cat Bin --obj_id 0001
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--object_usd", type=str, default=None)
parser.add_argument("--dataset_dir", type=str, default=None)
parser.add_argument("--obj_cat", type=str, default=None)
parser.add_argument("--obj_id", type=str, default=None)
parser.add_argument("--object_type", type=str, default=None)
parser.add_argument("--root_prim_path", type=str, default=None)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciRot
import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Vt
from isaacsim.core.cloner import GridCloner
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
from omni.timeline import get_timeline_interface

sys.path.insert(0, str(Path(__file__).parent))
from find_rim_anchors import get_eef_pairs, load_mesh_from_usd, _sample_surface

timeline = get_timeline_interface()

# ============================================================
# CONFIG — tune here
# ============================================================
_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _THIS_DIR.parent

def _path_from_env(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default))).expanduser()

GRIPPER_USD = _path_from_env("BIGRIPPER_GRIPPER_USD", _WORKSPACE_ROOT / "Flying_hand_probe_pro.usd")

# Gripper kinematics
GRIPPER_LOCAL_Z         = np.array([0.0, 0.0, 1.0])  # local approach axis
GRIPPER_FINGERTIP_OFFSET = 0.14   # metres: gripper base → fingertip distance
GRIPPER_AABB_PADDING    = 0.005  # metres — conservative inflation for coarse pair filtering

# Grid cloner
NUM_COPIES      = 1
CLONE_SPACING   = 3.0
GROUND_Z        = -10.0
ENV_BASE_PATH   = "/World/Envs"
ENV_ROOT_PREFIX = f"{ENV_BASE_PATH}/env"
PHYSICS_SCENE_PATH = "/World/physicsScene"

# Approach: step back from target pose, then interpolate to contact
APPROACH_DISTANCE = 0.20   # metres — extra step-back from target pose
APPROACH_STEPS    = 80
APPROACH_DISTURB  = 0.01   # metres — max allowed object Z shift during approach

# Finger closing at contact before lift
CLOSE_STEPS       = 32
HOLD_STEPS        = 50
GRIPPER_OPEN_POS  = 0.04
GRIPPER_CLOSE_POS = 0.0

# Lift: both grippers move straight up simultaneously
LIFT_DISTANCE     = 1   # metres
LIFT_STEPS        = 200
LIFT_SUCCESS_Z    = 0.95   # object must rise at least this much to count as success

# Physics material
HIGH_STATIC_FRICTION  = 2.0
HIGH_DYNAMIC_FRICTION = 2.0
OUTPUT_JSON_NAME      = "bi_gripper_grasp_pose.json"

_GRIPPER_LOCAL_AABB_CACHE = None


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
# Utilities
# ============================================================
def _resolve_single_object_input(parsed_args):
    object_usd = Path(parsed_args.object_usd).resolve()
    object_type = parsed_args.object_type or "Bin"
    return object_usd, object_type


def _quat_wxyz_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    return SciRot.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()


def _local_aabb_corners(aabb_min: np.ndarray, aabb_max: np.ndarray) -> np.ndarray:
    aabb_min = np.asarray(aabb_min, dtype=np.float64)
    aabb_max = np.asarray(aabb_max, dtype=np.float64)
    return np.asarray(
        [
            [aabb_min[0], aabb_min[1], aabb_min[2]],
            [aabb_min[0], aabb_min[1], aabb_max[2]],
            [aabb_min[0], aabb_max[1], aabb_min[2]],
            [aabb_min[0], aabb_max[1], aabb_max[2]],
            [aabb_max[0], aabb_min[1], aabb_min[2]],
            [aabb_max[0], aabb_min[1], aabb_max[2]],
            [aabb_max[0], aabb_max[1], aabb_min[2]],
            [aabb_max[0], aabb_max[1], aabb_max[2]],
        ],
        dtype=np.float64,
    )


def _load_gripper_local_aabb() -> tuple[np.ndarray, np.ndarray]:
    global _GRIPPER_LOCAL_AABB_CACHE
    if _GRIPPER_LOCAL_AABB_CACHE is not None:
        return _GRIPPER_LOCAL_AABB_CACHE
    vertices, _ = load_mesh_from_usd(GRIPPER_USD)
    aabb_min = np.min(vertices, axis=0)
    aabb_max = np.max(vertices, axis=0)
    _GRIPPER_LOCAL_AABB_CACHE = (aabb_min, aabb_max)
    return _GRIPPER_LOCAL_AABB_CACHE


def _world_aabb_from_pose(
    local_aabb_min: np.ndarray,
    local_aabb_max: np.ndarray,
    position: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    padding: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    corners_local = _local_aabb_corners(local_aabb_min, local_aabb_max)
    R = _quat_wxyz_to_rotation_matrix(quat_wxyz)
    pos = np.asarray(position, dtype=np.float64)
    corners_world = (R @ corners_local.T).T + pos[None, :]
    padding = float(max(0.0, padding))
    return np.min(corners_world, axis=0) - padding, np.max(corners_world, axis=0) + padding


def _aabb_intersects(
    min_a: np.ndarray,
    max_a: np.ndarray,
    min_b: np.ndarray,
    max_b: np.ndarray,
) -> bool:
    overlap = np.minimum(np.asarray(max_a, dtype=np.float64), np.asarray(max_b, dtype=np.float64)) - np.maximum(
        np.asarray(min_a, dtype=np.float64), np.asarray(min_b, dtype=np.float64)
    )
    return bool(np.all(overlap > 0.0))


def _filter_pairs_by_gripper_overlap(
    pairs,
    *,
    padding: float = GRIPPER_AABB_PADDING,
):
    local_aabb_min, local_aabb_max = _load_gripper_local_aabb()
    filtered_pairs = []
    rejected = 0
    for anchor_a, anchor_b in pairs:
        target_a, quat_a, _ = _anchor_to_target_pose(anchor_a)
        target_b, quat_b, _ = _anchor_to_target_pose(anchor_b)
        aabb_a = _world_aabb_from_pose(local_aabb_min, local_aabb_max, target_a, quat_a, padding=padding)
        aabb_b = _world_aabb_from_pose(local_aabb_min, local_aabb_max, target_b, quat_b, padding=padding)
        if _aabb_intersects(aabb_a[0], aabb_a[1], aabb_b[0], aabb_b[1]):
            rejected += 1
            continue
        filtered_pairs.append((anchor_a, anchor_b))
    return filtered_pairs, {
        "input_pairs": int(len(pairs)),
        "kept_pairs": int(len(filtered_pairs)),
        "rejected_pairs": int(rejected),
        "gripper_aabb_padding": float(padding),
    }


def _compute_edge_opposite_pairs(vertices: np.ndarray, faces: np.ndarray):
    """
    Build bi-gripper anchor pairs using the mapped-XY edge rule.

    The rule is:
    - determine whether the primary anchor lies on an x-parallel or y-parallel edge
    - preserve the along-edge coordinate
    - choose an anchor on the opposite side with the largest across-edge separation

    `get_eef_pairs()` performs the actual selection; this wrapper makes the
    geometry assumption explicit in this file too.
    """
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center_xy = np.array(
        [
            float((bbox_min[0] + bbox_max[0]) * 0.5),
            float((bbox_min[1] + bbox_max[1]) * 0.5),
        ],
        dtype=np.float64,
    )
    pairs = get_eef_pairs(vertices, faces, center_xy=center_xy)
    filtered_pairs, overlap_info = _filter_pairs_by_gripper_overlap(pairs, padding=GRIPPER_AABB_PADDING)
    print(
        "[INFO] Pair rule: preserve along-edge XY coordinate and maximize across-edge separation "
        f"about object center {center_xy.tolist()}."
    )
    print(
        "[INFO] Coarse gripper-overlap filter: "
        f"kept {overlap_info['kept_pairs']} / {overlap_info['input_pairs']} "
        f"pairs (padding={overlap_info['gripper_aabb_padding']:.3f} m)."
    )
    return filtered_pairs, bbox_min, bbox_max, center_xy


def _collect_dataset_targets(parsed_args):
    if parsed_args.obj_id and not parsed_args.obj_cat:
        raise ValueError("Dataset mode requires --obj_cat when --obj_id is provided.")

    dataset_dir = Path(parsed_args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_dir}")

    if parsed_args.obj_cat:
        cat_dirs = [dataset_dir / parsed_args.obj_cat]
        if not cat_dirs[0].is_dir():
            raise FileNotFoundError(f"Object category directory does not exist: {cat_dirs[0]}")
    else:
        cat_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())

    targets = []
    for cat_dir in cat_dirs:
        if parsed_args.obj_id:
            obj_dirs = [cat_dir / parsed_args.obj_id]
            if not obj_dirs[0].is_dir():
                raise FileNotFoundError(f"Object id directory does not exist: {obj_dirs[0]}")
        else:
            obj_dirs = sorted(path for path in cat_dir.iterdir() if path.is_dir())

        for obj_dir in obj_dirs:
            object_usd = obj_dir / "Object.usd"
            object_type = parsed_args.object_type or cat_dir.name
            targets.append((cat_dir.name, obj_dir.name, object_usd, object_type))

    if not targets:
        raise FileNotFoundError(f"No object directories found under dataset directory: {dataset_dir}")

    return targets


async def _close_stage_if_needed():
    ctx = omni.usd.get_context()
    if ctx.get_stage() is not None:
        await ctx.close_stage_async()
        await step_sim(2)


def _get_world_pos(prim) -> np.ndarray:
    """Return world-space XYZ translation of a prim via XformCache."""
    xfc = UsdGeom.XformCache(Usd.TimeCode.Default())
    mat = xfc.GetLocalToWorldTransform(prim)
    return np.array([mat[3][0], mat[3][1], mat[3][2]])


async def step_sim(n: int = 1):
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


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
        _get_or_create_attr(mat_prim, "physxMaterial:staticFriction", Sdf.ValueTypeNames.Float).Set(float(HIGH_STATIC_FRICTION))

    try:
        pm.CreateDynamicFrictionAttr().Set(float(HIGH_DYNAMIC_FRICTION))
    except Exception:
        _get_or_create_attr(mat_prim, "physxMaterial:dynamicFriction", Sdf.ValueTypeNames.Float).Set(float(HIGH_DYNAMIC_FRICTION))

    try:
        pm.CreateRestitutionAttr().Set(0.0)
    except Exception:
        _get_or_create_attr(mat_prim, "physxMaterial:restitution", Sdf.ValueTypeNames.Float).Set(0.0)

    for attr_name, token_val in (
        ("physxMaterial:frictionCombineMode", "multiply"),
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
    rb.CreateDisableGravityAttr().Set(True)   # gravity off until lift phase
    rb.CreateContactSlopCoefficientAttr().Set(2.0)



def _set_gripper_pose(stage, wrapper_path: str,
                      local_pos: np.ndarray, quat_wxyz: np.ndarray):
    """Set gripper pose in env-local coordinates."""
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
    quat = Gf.Quatf(w, x, y, z) if r_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat else Gf.Quatd(w, x, y, z)
    r_op.Set(quat)


def _apply_joint_targets(stage, joint_targets: dict):
    for joint_path, target_val in joint_targets.items():
        prim = stage.GetPrimAtPath(joint_path)
        if not prim.IsValid():
            continue
        for prop in prim.GetProperties():
            name = prop.GetName()
            if "drive" in name and "targetPosition" in name:
                prop.Set(float(target_val))


def _set_gripper_fingers(stage, gripper_ref_path: str, target: float):
    _apply_joint_targets(stage, {
        f"{gripper_ref_path}/panda_hand/panda_finger_joint1": float(target),
        f"{gripper_ref_path}/panda_hand/panda_finger_joint2": float(target),
    })


def _set_obj_gravity(stage, wrapper_path: str, disable: bool):
    prim = stage.GetPrimAtPath(wrapper_path)
    if not prim.IsValid():
        return
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(disable)


def _reset_obj_pose(stage, wrapper_path: str):
    """Teleport object wrapper back to env-local origin for next batch."""
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
        identity = Gf.Quatf(1, 0, 0, 0) if r_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat else Gf.Quatd(1, 0, 0, 0)
        r_op.Set(identity)
    # Zero velocities
    for attr_name, sdf_type in [
        ("physics:velocity",        Sdf.ValueTypeNames.Vector3f),
        ("physics:angularVelocity", Sdf.ValueTypeNames.Vector3f),
    ]:
        a = prim.GetAttribute(attr_name)
        if not (a and a.IsValid()):
            a = prim.CreateAttribute(attr_name, sdf_type)
        a.Set(Gf.Vec3f(0, 0, 0))




# ============================================================
# Gripper pose utilities
# ============================================================

def _anchor_to_target_pose(anchor):
    """Convert an anchor contact frame into the gripper target pose.

    Pair selection may use projected XY positions, but the target pose itself is
    built from the original 3D anchor point and surface normal. The gripper
    keeps the anchor x-axis, flips its local z-axis to oppose the surface
    normal, then steps its base back by the fingertip offset.
    """
    contact_pos = np.array(anchor["position"], dtype=np.float64)
    surface_normal = np.array(anchor["normal"], dtype=np.float64)
    surface_normal /= np.linalg.norm(surface_normal) + 1e-12

    anchor_quat = np.array(anchor["eef_quaternion"], dtype=np.float64)
    anchor_rot = SciRot.from_quat([anchor_quat[1], anchor_quat[2], anchor_quat[3], anchor_quat[0]])

    # Rotate 180 deg about local X so the finger axis stays fixed while Y/Z flip.
    gripper_rot = anchor_rot * SciRot.from_rotvec([np.pi, 0.0, 0.0])
    q = gripper_rot.as_quat()  # [x, y, z, w]
    target_quat = np.array([q[3], q[0], q[1], q[2]])

    target_pos = contact_pos + GRIPPER_FINGERTIP_OFFSET * surface_normal
    return target_pos, target_quat, surface_normal


# ============================================================
# Visualization
# ============================================================
def _visualize_eef_pairs(stage, vertices: np.ndarray, faces: np.ndarray,
                          pairs) -> None:
    """Add a debug point cloud and pair spheres to the current stage.
    anchor_a → red spheres, anchor_b → green spheres."""
    VIZ_ROOT = "/World/DebugViz"
    for path in [VIZ_ROOT, f"{VIZ_ROOT}/Surface", f"{VIZ_ROOT}/Pairs"]:
        p = stage.GetPrimAtPath(path)
        if p and p.IsValid():
            stage.RemovePrim(path)

    UsdGeom.Xform.Define(stage, VIZ_ROOT)

    # -- Surface point cloud (area-weighted sampling for uniform density) --
    pts, _ = _sample_surface(vertices, faces, num_points=3000, seed=0)
    pts = pts.astype(np.float32)
    cloud = UsdGeom.Points.Define(stage, f"{VIZ_ROOT}/Surface")
    cloud.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p.tolist()) for p in pts]))
    cloud.GetWidthsAttr().Set(Vt.FloatArray([0.003] * len(pts)))
    cloud.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.4, 0.8, 1.0)]))

    # -- EEF pair spheres --
    UsdGeom.Xform.Define(stage, f"{VIZ_ROOT}/Pairs")
    for i, (anch_a, anch_b) in enumerate(pairs):
        for label, anchor, colour in [
            ("a", anch_a, (1.0, 0.15, 0.15)),
            ("b", anch_b, (0.15, 1.0, 0.15)),
        ]:
            sp = UsdGeom.Sphere.Define(stage, f"{VIZ_ROOT}/Pairs/pair_{i:04d}_{label}")
            sp.CreateRadiusAttr(0.006)
            UsdGeom.Xformable(sp.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3d(*anchor["position"])
            )
            sp.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*colour)]))

    print(f"[VIZ] {len(pts)} surface pts, {len(pairs)} EEF pairs drawn at /World/DebugViz")


# ============================================================
# Main async pipeline
# ============================================================
async def run(object_usd_path: Path, object_type: str, root_prim_path=None):
    ctx = omni.usd.get_context()
    output_dir = object_usd_path.parent

    # ------------------------------------------------------------------
    # 1. Geometry: load mesh + compute edge-opposite EEF pairs (pure pxr, no stage)
    # ------------------------------------------------------------------
    print("[INFO] Loading mesh for anchor computation...")
    vertices, faces = load_mesh_from_usd(object_usd_path, root_prim_path=root_prim_path)
    print(f"[INFO] Mesh: {len(vertices)} verts, {len(faces)} faces")

    pairs, bbox_min, bbox_max, center_xy = _compute_edge_opposite_pairs(vertices, faces)
    print(f"[INFO] {len(pairs)} EEF pairs found")
    if not pairs:
        print("[WARN] No EEF pairs found — nothing to validate.")
        return

    bottom_center = [
        float((bbox_min[0] + bbox_max[0]) / 2),
        float((bbox_min[1] + bbox_max[1]) / 2),
        float(bbox_min[2]),
    ]

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
    add_reference_to_stage(str(object_usd_path), obj_ref(0))
    await step_sim(2)

    # Strip any baked rigid bodies from reference
    ref_prim = stage.GetPrimAtPath(obj_ref(0))
    if ref_prim.IsValid():
        for p in Usd.PrimRange(ref_prim):
            for api_cls in [UsdPhysics.RigidBodyAPI, PhysxSchema.PhysxRigidBodyAPI]:
                if p.HasAPI(api_cls):
                    p.RemoveAPI(api_cls)

    _apply_convex_decomp(stage, obj_ref(0))
    _make_object_rigid(stage, obj_wrap(0))

    # # ------------------------------------------------------------------
    # # 4. Visualize EEF pairs (headless=False only — press Enter to proceed)
    # # ------------------------------------------------------------------
    # _visualize_eef_pairs(stage, vertices, faces, pairs)
    # await step_sim(10000)

    # ------------------------------------------------------------------
    # 5. Two grippers in env_0
    # ------------------------------------------------------------------
    for wrap_fn, ref_fn in [(gripl_wrap, gripl_ref), (gripr_wrap, gripr_ref)]:
        UsdGeom.Xform.Define(stage, wrap_fn(0))
        add_reference_to_stage(str(GRIPPER_USD), ref_fn(0))
    await step_sim(2)

    # ------------------------------------------------------------------
    # 6. Physics scene + ground + high-friction material
    # ------------------------------------------------------------------
    _setup_physics_scene(stage)
    GroundPlane(prim_path="/World/GroundPlane", z_position=GROUND_Z)

    mat = _create_physics_material(stage, "/World/PhysicsMat/HF")
    _bind_physics_material(stage, obj_ref(0), mat)
    await step_sim(2)

    # ------------------------------------------------------------------
    # 7. Make refs instanceable, then clone
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
    # 8. Batch loop
    # ------------------------------------------------------------------
    valid_pairs = []
    batch_size  = NUM_COPIES

    for base in range(0, len(pairs), batch_size):
        batch = pairs[base: base + batch_size]
        K = len(batch)
        print(f"[INFO] Batch {base // batch_size + 1}: "
              f"pairs {base + 1}..{base + K} / {len(pairs)}")

        # Unpack anchors
        anch_a = [batch[k][0] for k in range(K)]
        anch_b = [batch[k][1] for k in range(K)]

        pose_a = [_anchor_to_target_pose(a) for a in anch_a]
        pose_b = [_anchor_to_target_pose(a) for a in anch_b]

        target_a = [pose[0] for pose in pose_a]
        target_b = [pose[0] for pose in pose_b]
        quat_a = [pose[1] for pose in pose_a]
        quat_b = [pose[1] for pose in pose_b]
        nrm_a = [pose[2] for pose in pose_a]
        nrm_b = [pose[2] for pose in pose_b]

        # Step back further along the surface normal for the approach phase.
        start_a = [target_a[k] + APPROACH_DISTANCE * nrm_a[k] for k in range(K)]
        start_b = [target_b[k] + APPROACH_DISTANCE * nrm_b[k] for k in range(K)]

        # ---- Reset envs ----
        for k in range(K):
            _reset_obj_pose(stage, obj_wrap(k))
            _set_obj_gravity(stage, obj_wrap(k), disable=True)
            _set_gripper_fingers(stage, gripl_ref(k), GRIPPER_OPEN_POS)
            _set_gripper_fingers(stage, gripr_ref(k), GRIPPER_OPEN_POS)
            _set_gripper_pose(stage, gripl_wrap(k), start_a[k], quat_a[k])
            _set_gripper_pose(stage, gripr_wrap(k), start_b[k], quat_b[k])
        await step_sim(5)

        # Record object Z before approach
        z_pre = []
        for k in range(K):
            p = _get_world_pos(obj_prims[k])
            z_pre.append(float(p[2]))

        # ---- Approach: interpolate both grippers to the target pose ----
        for t in range(APPROACH_STEPS + 1):
            alpha = t / float(APPROACH_STEPS)
            for k in range(K):
                ca = start_a[k] * (1 - alpha) + target_a[k] * alpha
                cb = start_b[k] * (1 - alpha) + target_b[k] * alpha
                _set_gripper_pose(stage, gripl_wrap(k), ca, quat_a[k])
                _set_gripper_pose(stage, gripr_wrap(k), cb, quat_b[k])
            await omni.kit.app.get_app().next_update_async()

        await step_sim(5)

        # Gate: reject if approach disturbed object
        active = []
        for k in range(K):
            p = _get_world_pos(obj_prims[k])
            disturbed = abs(float(p[2]) - z_pre[k]) > 0.03
            active.append(not disturbed)

        # Close grippers at contact while gravity is still off.
        for _ in range(CLOSE_STEPS):
            for k in range(K):
                if not active[k]:
                    continue
                _set_gripper_fingers(stage, gripl_ref(k), GRIPPER_CLOSE_POS)
                _set_gripper_fingers(stage, gripr_ref(k), GRIPPER_CLOSE_POS)
            await omni.kit.app.get_app().next_update_async()

        await step_sim(HOLD_STEPS)

        # Record object Z at post-close contact (gravity still off)
        z_contact = []
        for k in range(K):
            p = _get_world_pos(obj_prims[k])
            z_contact.append(float(p[2]) if active[k] else None)

        # Enable gravity for lift test
        for k in range(K):
            if active[k]:
                _set_obj_gravity(stage, obj_wrap(k), disable=False)
        await step_sim(3)

        # ---- Lift: both grippers move straight up simultaneously ----
        lift_a = [target_a[k] + np.array([0.0, 0.0, LIFT_DISTANCE]) for k in range(K)]
        lift_b = [target_b[k] + np.array([0.0, 0.0, LIFT_DISTANCE]) for k in range(K)]

        for t in range(LIFT_STEPS + 1):
            alpha = t / float(LIFT_STEPS)
            for k in range(K):
                if not active[k]:
                    continue
                ca = target_a[k] * (1 - alpha) + lift_a[k] * alpha
                cb = target_b[k] * (1 - alpha) + lift_b[k] * alpha
                _set_gripper_pose(stage, gripl_wrap(k), ca, quat_a[k])
                _set_gripper_pose(stage, gripr_wrap(k), cb, quat_b[k])
            await omni.kit.app.get_app().next_update_async()

        await step_sim(5)

        # ---- Check success: object Z rise ----
        for k in range(K):
            if not active[k] or z_contact[k] is None:
                continue
            p = _get_world_pos(obj_prims[k])
            z_after = float(p[2])
            if (z_after - z_contact[k]) >= LIFT_SUCCESS_Z:
                valid_pairs.append({
                    "left":  list(target_a[k]) + list(quat_a[k]),
                    "right": list(target_b[k]) + list(quat_b[k]),
                })

        # Disable gravity for next batch (objects stay put)
        for k in range(K):
            _set_obj_gravity(stage, obj_wrap(k), disable=True)

    print(f"[INFO] {len(valid_pairs)} valid bi-gripper grasps "
          f"out of {len(pairs)} candidate pairs.")

    # ------------------------------------------------------------------
    # 9. Save JSON
    # ------------------------------------------------------------------
    annotation_dir = output_dir / "Annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    body_list = list(valid_pairs)

    result = {
        "type": object_type,
        "bottom_center": bottom_center,
        "functional_grasp": {"body": body_list},
    }

    out_path = annotation_dir / OUTPUT_JSON_NAME
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[INFO] Saved {len(body_list)} grasp poses → {out_path}")

    timeline.stop()


# ============================================================
# Entry point
# ============================================================
def main():
    async def _run():
        if args.object_usd:
            object_usd, object_type = _resolve_single_object_input(args)
            if not object_usd.exists():
                raise FileNotFoundError(f"Object USD does not exist: {object_usd}")
            await run(object_usd, object_type, root_prim_path=args.root_prim_path)
            await _close_stage_if_needed()
            return

        if args.dataset_dir:
            targets = _collect_dataset_targets(args)
            completed = 0
            skipped = 0
            failed = 0

            for idx, (obj_cat, obj_id, object_usd, object_type) in enumerate(targets, start=1):
                print(f"[INFO] Dataset object {idx}/{len(targets)}: {obj_cat}/{obj_id}")
                if not object_usd.exists():
                    print(f"[WARN] Missing Object.usd, skipping: {object_usd}")
                    skipped += 1
                    continue

                try:
                    await run(object_usd, object_type, root_prim_path=args.root_prim_path)
                    completed += 1
                except Exception as exc:
                    import traceback

                    failed += 1
                    print(f"[ERROR] Failed on {obj_cat}/{obj_id}: {exc}")
                    traceback.print_exc()
                finally:
                    timeline.stop()
                    await _close_stage_if_needed()

            print("\n" + "=" * 72)
            print("Bi-gripper grasp generation complete")
            print(f"Completed: {completed}")
            print(f"Skipped:   {skipped}")
            print(f"Failed:    {failed}")
            print("=" * 72)
            return

        raise ValueError("Provide either --object_usd or --dataset_dir.")

    try:
        loop = asyncio.get_event_loop()
        task = asyncio.ensure_future(_run())
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


if __name__ == "__main__":
    main()
