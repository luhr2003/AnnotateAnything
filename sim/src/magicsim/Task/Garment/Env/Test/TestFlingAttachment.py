"""Fling test that physically attaches the cloth to the closing gripper.

Same trajectory as :mod:`TestFlingEnv`, but at the END of close_gripper
(fingers fully clamped) we author one ``PhysxPhysicsAttachment`` per
finger — pinning every cloth particle within ``ATTACH_OVERLAP_OFFSET``
of the finger collision shape to that finger rigid body. That way the
cloth follows the fingers via a real PhysX constraint, no
``set_world_positions`` needed (works on both CPU and GPU pipelines).

Gravity stays at the scene default for the whole run — no per-phase
``set_gravity_scale`` flips, so this test is purely about attachment
behavior.

Phase plan::

    reach → close_gripper → [author attachments at last step]
        → lift → fling → drop → [disable attachments] → open_gripper
"""

from magicsim.Task.Garment.Env.FlingEnv import FlingEnv
import math
from typing import List

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from loguru import logger as log

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, Vt
import omni.kit.commands
import omni.physx
import omni.usd

from magicsim.Env.Utils.file import Logger
from magicsim.Env.Utils.viz import draw_grasp_samples_as_axes, draw_waypoints  # noqa: F401


# ----- knobs (mirror TestFlingEnv) ---------------------------------------

REACH_Z_OFFSET = -0.022
LIFT_HEIGHT = 0.28
FLING_DISTANCE = -0.10
FLING_APEX = 0.10
DROP_HEIGHT = 0.05
DROP_DISTANCE = -0.20
GRIPPER_LENGTH = 0.2
INWARD_SHIFT = 0.05
GRASP_KP_X_SHIFT = 0.02

GRASP_QUAT_STD = [0.0, 1.0, 0.0, 0.0]
LEFT_ARM_YAW_DEG = -90.0
RIGHT_ARM_YAW_DEG = 90.0

# Robot prim path pattern. dual_franka was registered with
# articulation_name="Robot_0" → ``/World/envs/env_<id>/Robot_0``.
# Attach to the FINGER bodies (where the cloth actually gets pinched),
# not panda_hand — panda_hand origin sits ~``GRIPPER_LENGTH`` (0.2m)
# back from the fingertip, so attaching there hangs the cloth off the
# wrist instead of clamping it between the fingers.
ROBOT_NAME = "Robot_0"
# One finger per hand is enough — a single explicit-point attachment
# per side pins one cloth vertex to one rigid finger body. Using both
# fingers on the same kp would over-constrain (two anchors fighting as
# fingers slip on close).
LEFT_FINGER_LINK = "L_panda_leftfinger"
RIGHT_FINGER_LINK = "R_panda_leftfinger"

# Auto-attach radius: any cloth vertex whose world position sits within
# this many meters of the finger collision shape will be attached when
# the attachment prim is authored. Bumped to 5cm so a slight kp /
# fingertip misalignment still lets PhysX catch the kp particle —
# otherwise auto-attach silently produces zero attached vertices.
ATTACH_OVERLAP_OFFSET = 0.05

# Filter cloth ↔ finger collisions inside this radius so the attached
# vertices don't fight the finger-collision shape they're pinned to.
ATTACH_COLLISION_FILTER_OFFSET = 0.02


# ----- quaternion helpers -------------------------------------------------


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def _rotate_world_quat_by_yaw(q_std: List[float], yaw_deg: float) -> List[float]:
    half = math.radians(yaw_deg) * 0.5
    q_yaw = [math.cos(half), 0.0, 0.0, math.sin(half)]
    return _quat_mul(q_yaw, list(q_std))


# ----- garment helpers ----------------------------------------------------


def _collect_garments(env):
    scene_mgr = env.scene.scene_manager
    garments = []
    for env_id in range(env.num_envs):
        for _cat, glist in scene_mgr.garment_objects[env_id].items():
            garments.extend(glist)
    return garments


def _pick_sleeve_keypoints(garment):
    """Returns (left_kp_world, right_kp_world, left_kp_idx, right_kp_idx)
    with L/R assigned by world Y (larger Y → left arm)."""
    garment.update_keypoint()
    garment.visualize_keypoint()
    kp = garment.get_keypoint()
    indices = getattr(garment, "_keypoint_indices", None)
    if not indices or "top_left" not in kp or "top_right" not in kp:
        raise RuntimeError(
            f"need top_left/top_right kps + indices; got {list(kp.keys())}"
        )
    a_pos = np.asarray(kp["top_left"], dtype=np.float32)
    b_pos = np.asarray(kp["top_right"], dtype=np.float32)
    a_idx = int(indices["top_left"])
    b_idx = int(indices["top_right"])
    if a_pos[1] >= b_pos[1]:
        return a_pos, b_pos, a_idx, b_idx
    return b_pos, a_pos, b_idx, a_idx


# ----- attachment authoring ----------------------------------------------


def _world_pos_of_prim(stage: Usd.Stage, prim_path: str):
    """USD-only path; only reflects authoring-time transform.
    Articulation runtime poses won't be visible here unless the sim is
    configured with physics:updateToUsd=True (CPU pipeline default is
    OFF). Use ``_articulation_body_pose`` instead for live finger pose.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    xf = UsdGeom.Xformable(prim)
    if not xf:
        return None
    m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=np.float32)


def _articulation_body_pose(asset, env_id: int, body_name: str):
    """Returns (pos_w[3], quat_wxyz[4]) for an articulation link via
    the live PhysX state. ``asset`` is an isaaclab Articulation
    (e.g. ``env.scene.articulations["Robot_0"]``)."""
    ids, _ = asset.find_bodies(body_name)
    if not ids:
        return None, None
    body_idx = int(ids[0])
    pos = asset.data.body_pos_w[env_id, body_idx].detach().cpu().numpy()
    quat = asset.data.body_quat_w[env_id, body_idx].detach().cpu().numpy()
    return pos.astype(np.float32), quat.astype(np.float32)


def _world_to_local_with_pose(
    pos_world: np.ndarray, ori_wxyz: np.ndarray, pt_world: np.ndarray
) -> np.ndarray:
    rel = np.asarray(pt_world, dtype=np.float32) - pos_world.astype(np.float32)
    return _quat_rotate_inv_np(ori_wxyz.astype(np.float32), rel)


def _quat_rotate_inv_np(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    qx, qy, qz = -x, -y, -z
    qw = w
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    rx = vx + qw * tx + (qy * tz - qz * ty)
    ry = vy + qw * ty + (qz * tx - qx * tz)
    rz = vz + qw * tz + (qx * ty - qy * tx)
    return np.array([rx, ry, rz], dtype=np.float32)


def _world_to_local(
    stage: Usd.Stage, prim_path: str, world_pt: np.ndarray
) -> np.ndarray:
    """Map a world-frame point into the prim's local frame using its
    current LocalToWorld transform (rotation + translation, no scale)."""
    prim = stage.GetPrimAtPath(prim_path)
    xf = UsdGeom.Xformable(prim)
    m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    rot = Gf.Quatf(m.ExtractRotationQuat())
    re, im = rot.GetReal(), rot.GetImaginary()
    q_wxyz = np.array([re, im[0], im[1], im[2]], dtype=np.float32)
    rel = np.asarray(world_pt, dtype=np.float32) - np.array(
        [t[0], t[1], t[2]], dtype=np.float32
    )
    return _quat_rotate_inv_np(q_wxyz, rel)


def _author_attachment_explicit(
    stage: Usd.Stage,
    attachment_path: str,
    cloth_mesh_path: str,
    rigid_body_path: str,
    cloth_local_point: np.ndarray,
    rigid_local_point: np.ndarray,
) -> "PhysxSchema.PhysxPhysicsAttachment":
    """Explicit single-vertex attachment (no auto cooking).

    Sets ``points0`` / ``points1`` directly so PhysX doesn't have to run
    the auto-attach overlap query (which silently produces 0 attached
    vertices when its mid-sim listener doesn't fire). With explicit
    points, the constraint geometry is fully specified at authoring
    time — PhysX just has to build the joint, no overlap search.
    """
    if stage.GetPrimAtPath(attachment_path):
        stage.RemovePrim(attachment_path)
    attachment = PhysxSchema.PhysxPhysicsAttachment.Define(
        stage, Sdf.Path(attachment_path)
    )
    attachment.GetActor0Rel().SetTargets([Sdf.Path(cloth_mesh_path)])
    attachment.GetActor1Rel().SetTargets([Sdf.Path(rigid_body_path)])
    attachment.CreateAttachmentEnabledAttr(True)
    p0 = Vt.Vec3fArray.FromNumpy(cloth_local_point.reshape(1, 3).astype(np.float32))
    p1 = Vt.Vec3fArray.FromNumpy(rigid_local_point.reshape(1, 3).astype(np.float32))
    attachment.CreatePoints0Attr(p0)
    attachment.CreatePoints1Attr(p1)
    print(
        f"[attach] explicit-pts attachment\n"
        f"           {attachment_path}\n"
        f"           cloth ← {cloth_mesh_path} pt0_local={cloth_local_point.tolist()}\n"
        f"           rigid ← {rigid_body_path} pt1_local={rigid_local_point.tolist()}"
    )
    # Hint PhysX cooking to refresh on this prim. Tricky bit: the pybind
    # binding for ``add_prim_to_cooking_refresh_set`` was built against
    # one USD build (``pxrInternal_v0_24__pxrReserved__``) and our
    # ``pxr.Sdf.Path`` import comes from another build, so passing
    # ``Sdf.Path(...)`` or a raw string both fail the type check. The
    # workaround: read the path back off the live stage prim — that
    # SdfPath was minted by the same USD instance the binding uses.
    try:
        cooking_priv = omni.physx.get_physx_cooking_private_interface()
        att_prim = stage.GetPrimAtPath(attachment_path)
        cooking_priv.add_prim_to_cooking_refresh_set(att_prim.GetPath())
        print(f"[attach]   cooking refresh requested for {attachment_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[attach]   WARN: cooking refresh skipped: {e}")
    return attachment


# Auto-attach variant kept here for fallback / comparison; main path is
# the explicit-points version above.
def _author_attachment(
    stage: Usd.Stage,
    attachment_path: str,
    cloth_mesh_path: str,
    rigid_body_path: str,
    overlap_offset: float,
    collision_filter_offset: float,
) -> "PhysxSchema.PhysxPhysicsAttachment":
    if stage.GetPrimAtPath(attachment_path):
        stage.RemovePrim(attachment_path)
    omni.kit.commands.execute(
        "CreatePhysicsAttachment",
        target_attachment_path=Sdf.Path(attachment_path),
        actor0_path=Sdf.Path(cloth_mesh_path),
        actor1_path=Sdf.Path(rigid_body_path),
    )
    attachment = PhysxSchema.PhysxPhysicsAttachment.Get(
        stage, Sdf.Path(attachment_path)
    )
    if not attachment:
        return None
    attachment.CreateAttachmentEnabledAttr(True)
    auto = PhysxSchema.PhysxAutoAttachmentAPI.Get(stage, Sdf.Path(attachment_path))
    if not auto:
        auto = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
    auto.CreateEnableDeformableVertexAttachmentsAttr(True)
    auto.CreateDeformableVertexOverlapOffsetAttr(float(overlap_offset))
    auto.CreateEnableRigidSurfaceAttachmentsAttr(False)
    auto.CreateEnableCollisionFilteringAttr(True)
    auto.CreateCollisionFilteringOffsetAttr(float(collision_filter_offset))
    return attachment


def _set_attachments_enabled(attachments, enabled: bool):
    for att in attachments:
        att.GetAttachmentEnabledAttr().Set(bool(enabled))
    print(f"[attach] enabled -> {enabled} (×{len(attachments)})")


# ----- action builder -----------------------------------------------------


def _build_16d_action(
    right_xyz, left_xyz, right_quat, left_quat, gripper_close, num_envs, device
) -> torch.Tensor:
    arm = (
        list(right_xyz.tolist())
        + list(right_quat)
        + list(left_xyz.tolist())
        + list(left_quat)
        + [float(gripper_close), float(gripper_close)]
    )
    assert len(arm) == 16
    row = torch.tensor(arm, dtype=torch.float32, device=device)
    return row.unsqueeze(0).repeat(num_envs, 1)


def _draw_phase_viz(left_kp, right_kp, right_xyz, left_xyz, q_right, q_left, color):
    draw_waypoints(
        [left_kp.tolist(), right_kp.tolist()],
        point_size=14.0,
        color=(1.0, 0.0, 0.0, 1.0),
        clear_existing=True,
    )
    draw_waypoints(
        [right_xyz.tolist(), left_xyz.tolist()],
        point_size=10.0,
        color=color,
        clear_existing=False,
    )
    pose = torch.tensor(
        [
            list(right_xyz.tolist()) + list(q_right),
            list(left_xyz.tolist()) + list(q_left),
        ],
        dtype=torch.float32,
    )
    draw_grasp_samples_as_axes(
        pose,
        axis_length=0.06,
        line_thickness=2,
        line_opacity=0.9,
        clear_existing=True,
    )


# ----- main ---------------------------------------------------------------


@hydra.main(version_base=None, config_path="../../Conf", config_name="fling_env")
def main(cfg: DictConfig):
    print(cfg)
    # Force PhysX to mirror runtime articulation transforms back to USD
    # so attachment-time pose lookups (and PhysX's own actor-transform
    # caching for attachment cooking) see the real finger pose, not the
    # authoring-time home pose. CPU pipeline defaults this OFF.
    import carb

    carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
    logger = Logger("Env", log)
    env: FlingEnv = gym.make("FlingEnv-V0", config=cfg, cli_args=None, logger=logger)
    env.reset()

    print("[attach] settling cloth (50 steps)...")
    for _ in range(50):
        env.step(action=None)

    garments = _collect_garments(env)
    if not garments:
        raise RuntimeError("no garments")
    garment = garments[0]

    left_kp_raw, right_kp_raw, left_kp_idx, right_kp_idx = _pick_sleeve_keypoints(
        garment
    )
    print(f"[attach] kp indices L={left_kp_idx} R={right_kp_idx}")
    midpoint = (left_kp_raw + right_kp_raw) * 0.5
    lr_vec = right_kp_raw - left_kp_raw
    lr_dist = float(np.linalg.norm(lr_vec))
    if lr_dist > 1e-4 and INWARD_SHIFT > 0:
        step = INWARD_SHIFT / lr_dist
        left_kp = left_kp_raw + lr_vec * step
        right_kp = right_kp_raw - lr_vec * step
    else:
        left_kp, right_kp = left_kp_raw.copy(), right_kp_raw.copy()
    if GRASP_KP_X_SHIFT != 0.0:
        left_kp[0] += GRASP_KP_X_SHIFT
        right_kp[0] += GRASP_KP_X_SHIFT
    print(f"[attach] grasp kp L={left_kp.tolist()} R={right_kp.tolist()}")

    q_left = _rotate_world_quat_by_yaw(GRASP_QUAT_STD, LEFT_ARM_YAW_DEG)
    q_right = _rotate_world_quat_by_yaw(GRASP_QUAT_STD, RIGHT_ARM_YAW_DEG)

    gl = GRIPPER_LENGTH
    reach_off = np.array([0.0, 0.0, REACH_Z_OFFSET + gl], dtype=np.float32)
    lift_off = np.array([0.0, 0.0, LIFT_HEIGHT + gl], dtype=np.float32)
    fling_off = np.array(
        [FLING_DISTANCE, 0.0, LIFT_HEIGHT + FLING_APEX + gl], dtype=np.float32
    )
    drop_off = np.array([DROP_DISTANCE, 0.0, DROP_HEIGHT + gl], dtype=np.float32)

    LIFT_FROM = (left_kp + reach_off, right_kp + reach_off)
    FLING_FROM = (left_kp + lift_off, right_kp + lift_off)
    DROP_FROM = (left_kp + fling_off, right_kp + fling_off)
    phases = [
        (
            "reach",
            left_kp + reach_off,
            right_kp + reach_off,
            0.0,
            60,
            (1.0, 0.8, 0.1, 0.9),
            None,
        ),
        (
            "close_gripper",
            left_kp + reach_off,
            right_kp + reach_off,
            1.0,
            80,
            (1.0, 0.8, 0.1, 0.9),
            None,
        ),
        (
            "lift_up",
            left_kp + lift_off,
            right_kp + lift_off,
            1.0,
            240,
            (0.2, 0.9, 0.2, 0.9),
            LIFT_FROM,
        ),
        (
            "fling_forward",
            left_kp + fling_off,
            right_kp + fling_off,
            1.0,
            120,
            (0.2, 0.6, 1.0, 0.9),
            FLING_FROM,
        ),
        (
            "drop",
            left_kp + drop_off,
            right_kp + drop_off,
            1.0,
            120,
            (0.9, 0.4, 0.9, 0.9),
            DROP_FROM,
        ),
        (
            "open_gripper",
            left_kp + drop_off,
            right_kp + drop_off,
            0.0,
            20,
            (0.9, 0.4, 0.9, 0.9),
            None,
        ),
    ]

    # NOTE: This test intentionally leaves gravity at the scene default
    # for every phase — we want to observe attachment behavior alone,
    # without the cloth artificially floating during lift/fling/drop.

    # Resolve prim paths for attachment authoring.
    stage = omni.usd.get_context().get_stage()
    cloth_mesh_path = garment.mesh_prim_path
    print(f"[attach] cloth mesh prim = {cloth_mesh_path}")
    # One attachment per (env, hand). Test setups typically run num_envs=1
    # but scale to whatever's there.
    # Grab the articulation handle so we can read live finger pose from
    # PhysX rather than from USD (which only reflects authoring-time
    # transforms unless ``/physics/updateToUsd`` is on).
    # ``env.scene`` is the inner SyncRobotEnv; the Isaac sim handle is
    # at ``env.scene.sim`` and articulations are registered under
    # ``env.scene.sim.scene.articulations[asset_name]`` where asset_name
    # is "Robot_0", "Robot_1", ... (per RobotManager). The top-level
    # ``env.scene.robot_manager.robots`` is keyed by the user's config
    # name (e.g. "DualFranka_0"), not asset name.
    robot_asset = None
    sim_arts = env.scene.sim.scene.articulations
    if ROBOT_NAME in sim_arts:
        robot_asset = sim_arts[ROBOT_NAME]
    else:
        # Fall back: any articulation whose body list contains a
        # gripper finger we know about.
        for key, art in sim_arts.items():
            if any(LEFT_FINGER_LINK in b for b in art.data.body_names):
                robot_asset = art
                print(f"[attach] using articulation key='{key}' (matched by body name)")
                break
    if robot_asset is None:
        rm_keys = (
            list(env.scene.robot_manager.robots.keys())
            if hasattr(env.scene, "robot_manager")
            else []
        )
        raise RuntimeError(
            f"could not locate articulation '{ROBOT_NAME}'. "
            f"sim.scene.articulations keys = {list(sim_arts.keys())}, "
            f"robot_manager.robots keys = {rm_keys}"
        )
    print(f"[attach] robot_asset bodies = {list(robot_asset.data.body_names)}")

    # One attachment per (env, hand). Each spec carries the cloth-side kp
    # info so we can author with explicit ``Points0`` (cloth-local) and
    # ``Points1`` (rigid-local) — no auto-attach overlap query needed.
    attach_specs = []  # (attachment_path, rigid_body_path, body_name, env_id, kp_idx, kp_world)
    for env_id in range(env.num_envs):
        env_root = f"/World/envs/env_{env_id}"
        attach_specs.append(
            (
                f"{cloth_mesh_path}/attach_L_env{env_id}",
                f"{env_root}/{ROBOT_NAME}/{LEFT_FINGER_LINK}",
                LEFT_FINGER_LINK,
                env_id,
                left_kp_idx,
                left_kp,
            )
        )
        attach_specs.append(
            (
                f"{cloth_mesh_path}/attach_R_env{env_id}",
                f"{env_root}/{ROBOT_NAME}/{RIGHT_FINGER_LINK}",
                RIGHT_FINGER_LINK,
                env_id,
                right_kp_idx,
                right_kp,
            )
        )

    attachments = []  # filled at close_gripper

    for name, left_xyz, right_xyz, grip, n_steps, color, interp_from in phases:
        print(f"[attach] phase={name} steps={n_steps} grip={grip}")
        _draw_phase_viz(left_kp, right_kp, right_xyz, left_xyz, q_right, q_left, color)

        if interp_from is None:
            action = _build_16d_action(
                right_xyz,
                left_xyz,
                q_right,
                q_left,
                grip,
                env.num_envs,
                env.device,
            )
            for k in range(n_steps):
                env.step(action=action)
                # Author attachments at the END of close_gripper —
                # fingers are fully clamped on the kp vertices by then,
                # so the auto-attach overlap query catches the cloth
                # right between the fingertips (where it's actually
                # pinched), not 0.2m up at the wrist.
                if name == "close_gripper" and not attachments and k == n_steps - 1:
                    print("[attach] >>> authoring explicit-point attachments")
                    # Read cloth points (mesh_local is in cloth local frame).
                    # Read finger pose from PhysX runtime state (NOT
                    # USD — USD doesn't reflect articulation runtime
                    # transforms in CPU pipeline).
                    _, mesh_local, _, _ = garment.get_current_mesh_points()
                    mesh_local = np.asarray(mesh_local, dtype=np.float32)
                    # Also use the LIVE cloth kp world position (the
                    # cloth has settled / been gripped since we cached
                    # left_kp/right_kp; using stale kp_world would also
                    # introduce error).
                    transformed_world, _, _, _ = garment.get_current_mesh_points()
                    transformed_world = np.asarray(transformed_world, dtype=np.float32)
                    for (
                        att_path,
                        rb_path,
                        body_name,
                        eid,
                        kp_idx,
                        _kp_cached,
                    ) in attach_specs:
                        if not stage.GetPrimAtPath(rb_path):
                            print(f"[attach] WARN: no rigid body at {rb_path}")
                            continue
                        finger_pos, finger_quat = _articulation_body_pose(
                            robot_asset,
                            eid,
                            body_name,
                        )
                        if finger_pos is None:
                            print(
                                f"[attach] WARN: cannot find body {body_name} on robot"
                            )
                            continue
                        kp_world_live = transformed_world[kp_idx]
                        d = float(np.linalg.norm(finger_pos - kp_world_live))
                        print(
                            f"[attach]   finger {body_name} runtime_world={finger_pos.tolist()} "
                            f"kp_live={kp_world_live.tolist()} d={d:.4f}"
                        )
                        cloth_local_pt = mesh_local[kp_idx]
                        rigid_local_pt = _world_to_local_with_pose(
                            finger_pos,
                            finger_quat,
                            kp_world_live,
                        )
                        att = _author_attachment_explicit(
                            stage,
                            att_path,
                            cloth_mesh_path,
                            rb_path,
                            cloth_local_pt,
                            rigid_local_pt,
                        )
                        if att is not None:
                            attachments.append(att)
                    print(f"[attach] total attachments authored: {len(attachments)}")
                    # Give cooking a couple of frames to absorb the new
                    # prims before the trajectory continues.
                    for _ in range(3):
                        env.step(action=action)
                # Disable attachments right before opening the gripper,
                # so the cloth releases when fingers start to part.
                if name == "open_gripper" and k == 0 and attachments:
                    print("[attach] <<< disabling particle attachments (release)")
                    _set_attachments_enabled(attachments, False)
        else:
            left_start, right_start = interp_from
            for k in range(n_steps):
                alpha = (k + 1) / max(1, n_steps)
                cur_l = left_start + (left_xyz - left_start) * alpha
                cur_r = right_start + (right_xyz - right_start) * alpha
                action = _build_16d_action(
                    cur_r,
                    cur_l,
                    q_right,
                    q_left,
                    grip,
                    env.num_envs,
                    env.device,
                )
                env.step(action=action)

    print("[attach] all phases done; idling...")
    while True:
        env.step(action=None)


if __name__ == "__main__":
    main()
