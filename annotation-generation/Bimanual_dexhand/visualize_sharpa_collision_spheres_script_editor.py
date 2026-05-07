"""
Paste this whole file into Isaac Sim's Script Editor, or run it from there with
exec(open("Bimanual_dexhand/visualize_sharpa_collision_spheres_script_editor.py").read()).

It loads the local Sharpa USD assets, applies a cat3 fingers-down opposing
pose, and draws the configured collision spheres for both hands.
"""

import sys
import os
from pathlib import Path

import omni.usd
from pxr import Gf, Sdf, UsdGeom, Vt

_DEFAULT_REPO = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd() / "Bimanual_dexhand"
REPO = Path(os.environ.get("BIMANUAL_REPO", str(_DEFAULT_REPO))).expanduser()
HAND_DIR = REPO / "assets/hands/sharpa"
CONFIG_DIR = REPO / "configs"
OBJECT_USD = REPO / "assets/objects/Object.usd"
ROOT = "/World/SharpaCat3CollisionCheck"
POSTURE_NAME = "cat3_fingers_down"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import run_single_hand_staged_pose_preview as preview

preview._import_runtime_math()
np = preview.np

from src.config_loader import load_all_for_hand
from src.hand_kinematics import load_hand_kinematics_model, make_hand_pose
from src.seed_generation import _build_hand_semantic_basis


def _clear(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(Sdf.Path(prim_path))


def _quat_xyzw(rotation_matrix):
    return preview._quat_xyzw_from_rotation_matrix(rotation_matrix)


def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else np.zeros_like(v)


def _rotation_from_semantics(cfg, *, finger_world, palm_world):
    x_axis = _normalize(finger_world)
    z_axis = _normalize(palm_world)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    z_axis = _normalize(np.cross(x_axis, y_axis))
    world_basis = np.stack([x_axis, y_axis, z_axis], axis=1)
    return world_basis @ _build_hand_semantic_basis(cfg).T


def _draw_line(stage, path, p0, p1, color, width=0.006):
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr(Vt.IntArray([2]))
    curve.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p0)), Gf.Vec3f(*map(float, p1))]))
    curve.CreateWidthsAttr(Vt.FloatArray([float(width), float(width)]))
    curve.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, color))]))


def _draw_hand(stage, side, wrist_position, finger_world, palm_world, color):
    cfg = load_all_for_hand(HAND_DIR, CONFIG_DIR, side, "cat3")
    model = load_hand_kinematics_model(cfg)
    rotation = _rotation_from_semantics(cfg, finger_world=finger_world, palm_world=palm_world)
    joint_positions = dict(cfg.hand.default_postures[POSTURE_NAME])
    hand_pose = make_hand_pose(
        wrist_position=wrist_position,
        wrist_quaternion_xyzw=_quat_xyzw(rotation),
        wrist_rotation=rotation,
        joint_positions=joint_positions,
    )

    hand_prim_path = f"{ROOT}/{side.capitalize()}Hand"
    preview._add_reference(stage, cfg.hand.asset.usd_path, hand_prim_path)
    preview._set_hand_container_from_wrist_pose(
        stage,
        hand_prim_path=hand_prim_path,
        wrist_link_name=cfg.hand.root.wrist_link,
        wrist_position=np.asarray(wrist_position, dtype=np.float64),
        wrist_quaternion_xyzw=_quat_xyzw(rotation),
    )
    preview._set_hand_joint_targets(stage, hand_prim_path, joint_positions)

    spheres_root = f"{ROOT}/{side.capitalize()}CollisionSpheres"
    for sphere in model.collision_spheres_world(cfg, hand_pose):
        safe_link = sphere.link_name.replace("/", "_")
        sphere_path = f"{spheres_root}/{safe_link}_{sphere.sphere_index:02d}"
        prim = UsdGeom.Sphere.Define(stage, sphere_path)
        prim.CreateRadiusAttr(float(sphere.radius))
        prim.CreateDisplayColorAttr().Set([Gf.Vec3f(*map(float, color))])
        prim.CreateDisplayOpacityAttr().Set([0.38])
        preview._set_translate_orient_ops(
            UsdGeom.Xformable(prim.GetPrim()),
            np.asarray(sphere.center_world, dtype=np.float64),
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )

    local_finger = np.asarray(cfg.hand.frame_convention.finger_forward_local, dtype=np.float64)
    local_palm = np.asarray(cfg.hand.frame_convention.palm_normal_local, dtype=np.float64)
    actual_finger = _normalize(rotation @ local_finger)
    actual_palm = _normalize(rotation @ local_palm)
    _draw_line(stage, f"{ROOT}/{side.capitalize()}FingerForward", wrist_position, wrist_position + 0.13 * actual_finger, (0.1, 0.9, 0.2))
    _draw_line(stage, f"{ROOT}/{side.capitalize()}PalmNormal", wrist_position, wrist_position + 0.11 * actual_palm, (1.0, 0.2, 0.1))
    print(f"{side} config:")
    print(f"  local finger_forward = {cfg.hand.frame_convention.finger_forward_local}")
    print(f"  local palm_normal    = {cfg.hand.frame_convention.palm_normal_local}")
    print(f"  world finger_forward = {actual_finger.round(4).tolist()}")
    print(f"  world palm_normal    = {actual_palm.round(4).tolist()}")
    return actual_finger, actual_palm


stage = omni.usd.get_context().get_stage()
preview._ensure_world(stage)
_clear(stage, ROOT)
UsdGeom.Xform.Define(stage, ROOT)

preview._add_reference(stage, OBJECT_USD, f"{ROOT}/Object")

right_finger, right_palm = _draw_hand(
    stage,
    "right",
    np.asarray([0.28, 0.0, 0.62], dtype=np.float64),
    finger_world=np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
    palm_world=np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
    color=(0.1, 0.45, 1.0),
)
left_finger, left_palm = _draw_hand(
    stage,
    "left",
    np.asarray([-0.28, 0.0, 0.62], dtype=np.float64),
    finger_world=np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
    palm_world=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    color=(1.0, 0.78, 0.1),
)

print("cat3 pair checks:")
print(f"  finger_forward dot = {float(np.dot(right_finger, left_finger)):.4f}")
print(f"  palm_normal dot    = {float(np.dot(right_palm, left_palm)):.4f}")
