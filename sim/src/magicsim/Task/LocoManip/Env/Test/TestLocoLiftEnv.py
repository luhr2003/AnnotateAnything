"""
Open-loop bimanual box squeeze on :class:`LocoLiftEnv`.

Instead of reading paired grasp annotations, this test derives per-arm IK
targets directly from the bin's local AABB (via :meth:`LocoLiftEnv.
get_target_bbox_half_extents`, which wraps
:func:`mesh_utils.get_local_bbox_half_extents`). Targets are transformed
to world frame with the bin's current pose, so they stay correct after
the bin settles on the table.

Per-phase IK targets are visualized as axes in Isaac (two poses per step,
one per arm) using :func:`draw_grasp_samples_as_axes`.

Action layout (43D):

    [p_ctrl(15), right_arm(7), left_arm(7), right_hand(7), left_hand(7)]

``p_ctrl`` is all-NaN (no mobile base). Hand slot order matches
``G1.eef_action['joint_pos']`` joint groups — right first, left second.
"""

from typing import List, Tuple
from magicsim.Task.LocoManip.Env.LocoLiftEnv import LocoLiftEnv
import hydra
import torch
from loguru import logger as log
from omegaconf import DictConfig
import gymnasium as gym
from pxr import Gf

from magicsim.Env.Scene.Object.Rigid import RigidObject
from magicsim.Env.Utils.file import Logger
from magicsim.Env.Utils.viz import draw_grasp_samples_as_axes


AXIS_LENGTH = 0.05
LINE_THICKNESS = 3
LINE_OPACITY = 0.8


def visualize_grasp_poses(
    poses: List[torch.Tensor], env_origin: torch.Tensor | None = None
):
    """Draw axis gizmos for env-local 7D poses (pos + wxyz quat).

    Isaac debug draw consumes true-world coordinates, but our IK targets
    live in env-local frame (``RigidObject.transform_pose_to_world`` returns
    object→env-local, despite the name). Pass ``env_origin`` so the viz
    lands on the same spot the arms are actually commanded to.
    """
    shifted = []
    for p in poses:
        if env_origin is not None:
            q = p.clone()
            q[:3] = p[:3] + env_origin[:3].to(p.device)
            shifted.append(q)
        else:
            shifted.append(p)
    poses_cpu = [p.cpu().numpy().tolist() for p in shifted]
    samples = [
        (Gf.Vec3d(p[0], p[1], p[2]), Gf.Quatd(p[3], p[4], p[5], p[6]))
        for p in poses_cpu
    ]
    draw_grasp_samples_as_axes(
        grasp_poses=samples,
        axis_length=AXIS_LENGTH,
        line_thickness=LINE_THICKNESS,
        line_opacity=LINE_OPACITY,
        clear_existing=True,
    )


# Wrist orientation. With identity (1,0,0,0) the closed fist's *thumb*
# sticks out toward the bin. Roll each wrist 90° around its forearm axis
# (world x → robot's forward direction) so the *knuckles* face the bin
# instead. Right and left mirror — same axis, opposite sense.
# 90° about +x:  (cos45, sin45, 0, 0)
# 90° about -x:  (cos45, -sin45, 0, 0)
_R = 0.7071067811865476
RIGHT_QUAT_WXYZ: Tuple[float, float, float, float] = (_R, +_R, 0.0, 0.0)
LEFT_QUAT_WXYZ: Tuple[float, float, float, float] = (_R, -_R, 0.0, 0.0)

# Fully-closed hand joints for the Dex3. Joint order matches
# ``G1.eef_action['joint_pos']`` group joint_names (G1.py:851 for right,
# G1.py:864 for left): index_0, index_1, middle_0, middle_1, thumb_0,
# thumb_1, thumb_2. Right hand flexes with positive values, left hand is
# the mirror (negative index/middle, positive thumb_1/2). Values large
# enough to hit soft joint limits — :class:`MultipleJointPositionToLimitsAction`
# clamps for us.
RIGHT_CLOSE_HAND: Tuple[float, ...] = (1.5, 1.7, 1.5, 1.7, 0.0, -0.7, -0.7)
LEFT_CLOSE_HAND: Tuple[float, ...] = (-1.5, -1.7, -1.5, -1.7, 0.0, 0.7, 0.7)


def _build_action(
    device: torch.device,
    right_arm: torch.Tensor,
    left_arm: torch.Tensor,
    right_hand: torch.Tensor,
    left_hand: torch.Tensor,
) -> torch.Tensor:
    """43D action, hand order right-first-left-second (G1.py:847)."""
    p_ctrl = torch.full((15,), torch.nan, device=device, dtype=torch.float32)
    return torch.cat([p_ctrl, right_arm, left_arm, right_hand, left_hand], dim=0)


def _step_n(env: LocoLiftEnv, action_1d: torch.Tensor, steps: int):
    batched = action_1d.unsqueeze(0).repeat(env.num_envs, 1)
    for _ in range(steps):
        env.step(action=batched)


def _local_to_world(
    local_pose7: torch.Tensor, obj_pos: torch.Tensor, obj_quat: torch.Tensor
) -> torch.Tensor:
    """Transform a 7D ``[pos, quat_wxyz]`` pose from object-local to world."""
    return RigidObject.transform_pose_to_world(local_pose7, obj_pos, obj_quat)


def _compute_squeeze_targets(
    half: Tuple[float, float, float],
    bin_pos_w: torch.Tensor,
    bin_quat_w: torch.Tensor,
    device: torch.device,
    gap: float = 0.02,
    pre_gap: float = 0.15,
    forward_ratio: float = 0.3,
    down_ratio: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive world-frame 7D targets for pre-grasp and squeeze from bbox.

    All offsets are bbox-relative so the targets auto-scale with the bin's
    actual geometry (half-extents ``hx``/``hy``/``hz``):

    * ``x_local = forward_ratio * hx`` — wrist is at the bin's x center
      pushed forward by ``forward_ratio`` of the x half-extent (i.e. the
      wrist sits on the forward half of the bin so the forearms wrap
      around the forward side).
    * ``z_local = -down_ratio * hz`` — same but dropped below the bin
      center by ``down_ratio`` of the z half-extent.
    * y-component:
        - pre-grasp: ``±(hy + pre_gap)`` — ``pre_gap`` metres outside the
          ``±y`` faces, ready to close straight in along y.
        - squeeze:   ``±(hy - gap)``   — wrists ``gap`` metres inside the
          ``±y`` faces so the forearms press the bin walls.

    Wrist orientation is the rest-pose identity quat.

    Returns ``(r_pre, l_pre, r_grasp, l_grasp)``.
    """
    hx, hy, hz = float(half[0]), float(half[1]), float(half[2])
    x_local = forward_ratio * hx
    z_local = -down_ratio * hz

    def _pack(y_local: float) -> Tuple[torch.Tensor, torch.Tensor]:
        # Robot faces +x → its right hand sits on -y, left hand on +y.
        # Variable r_* must land on -y so it matches the right_hand task
        # slot (Pink IK: right is task[0], left is task[1] in G1.py:135).
        # Per-hand wrist quat (knuckle face inward, mirrored).
        r = torch.tensor(
            [x_local, -y_local, z_local, *RIGHT_QUAT_WXYZ],
            device=device,
            dtype=torch.float32,
        )
        left_pose = torch.tensor(
            [x_local, +y_local, z_local, *LEFT_QUAT_WXYZ],
            device=device,
            dtype=torch.float32,
        )
        return _local_to_world(r, bin_pos_w, bin_quat_w), _local_to_world(
            left_pose, bin_pos_w, bin_quat_w
        )

    r_pre, l_pre = _pack(hy + pre_gap)
    r_grasp, l_grasp = _pack(hy - gap)
    return r_pre, l_pre, r_grasp, l_grasp


def _pose_upward(pose7: torch.Tensor, dz: float) -> torch.Tensor:
    out = pose7.clone()
    out[2] += dz
    return out


@hydra.main(version_base=None, config_path="../../Conf", config_name="loco_lift_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: LocoLiftEnv = gym.make(
        "LocoLiftEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()
    device = env.device

    # ---- 1. Settle to neutral pose so the bin drops onto the table ----
    neutral_right = torch.tensor(
        [-0.5, -0.20, 1.05, 1, 0, 0, 0], device=device, dtype=torch.float32
    )
    neutral_left = torch.tensor(
        [-0.5, 0.20, 1.05, 1, 0, 0, 0], device=device, dtype=torch.float32
    )
    open_hand = torch.zeros(7, device=device, dtype=torch.float32)
    neutral_action = _build_action(
        device, neutral_right, neutral_left, open_hand, open_hand
    )
    _step_n(env, neutral_action, 200)

    env.scene.planner_manager.update_obstacles(
        obstacle_avoidance_path_list=["dynamic"],
        obstacle_ignore_path_list=["bin"],
        env_ids=[0],
    )

    # ---- 2. Read bin bbox + current world pose ----
    half = env.get_target_bbox_half_extents(obj_name="bin")
    if half is None:
        log.error("Could not read bin bbox — is the bin prim loaded?")
        return
    log.info("Bin local AABB half-extents: {}", half)

    bin_pose_w = env.get_target_world_pose(obj_name="bin")
    if bin_pose_w is None:
        log.error("Could not read bin world pose.")
        return
    bin_pos_w = bin_pose_w[:3]
    bin_quat_w = bin_pose_w[3:7]
    log.info("Bin world pose: pos={} quat={}", bin_pos_w.tolist(), bin_quat_w.tolist())

    # ---- 3. Build per-phase IK targets from bbox ----
    # pre-grasp: arms parked outside ±y faces (ready to close inward along y).
    # squeeze:   wrists at ±y faces, forearms press into bin walls.
    r_pre, l_pre, r_grasp, l_grasp = _compute_squeeze_targets(
        half,
        bin_pos_w,
        bin_quat_w,
        device,
        gap=0.12,
        pre_gap=0.15,
        forward_ratio=-1.0,
        down_ratio=-0.5,
    )
    # Retrieval: lift 20cm above grasp after closing.
    r_ret = _pose_upward(r_grasp, 0.2)
    l_ret = _pose_upward(l_grasp, 0.2)

    right_close = torch.tensor(RIGHT_CLOSE_HAND, device=device, dtype=torch.float32)
    left_close = torch.tensor(LEFT_CLOSE_HAND, device=device, dtype=torch.float32)

    # ---- 4. Run phases (with axes viz for each phase's right/left targets) ----
    # Hands stay fully closed throughout — fingers curl in to keep the bin
    # from squirting upward when the forearms compress its sides.
    # (name, right_arm, left_arm, right_hand, left_hand, steps)
    phases = [
        ("pre_grasp", r_pre, l_pre, right_close, left_close, 250),
        ("squeeze", r_grasp, l_grasp, right_close, left_close, 400),
        ("lift", r_ret, l_ret, right_close, left_close, 300),
    ]
    env_origin_0 = env.scene.env_origins[0]
    if env_origin_0.ndim > 1:
        env_origin_0 = env_origin_0[0]
    env_origin_0 = env_origin_0.to(device).flatten()[:3]
    log.info("env_origins[0] used for viz offset: {}", env_origin_0.tolist())
    for name, rp, lp, rh, lh, n_steps in phases:
        print(f"current phase: {name}")
        log.info(
            "[viz] {} right_target(env-local)={} left_target(env-local)={} "
            "-> world={} / {}",
            name,
            rp[:3].tolist(),
            lp[:3].tolist(),
            (rp[:3] + env_origin_0).tolist(),
            (lp[:3] + env_origin_0).tolist(),
        )
        visualize_grasp_poses([rp, lp], env_origin=env_origin_0)
        action = _build_action(device, rp, lp, rh, lh)
        _step_n(env, action, n_steps)

    # ---- 5. Hold ----
    hold = _build_action(device, r_ret, l_ret, right_close, left_close)
    batched_hold = hold.unsqueeze(0).repeat(env.num_envs, 1)
    while True:
        env.step(action=batched_hold)


if __name__ == "__main__":
    main()
