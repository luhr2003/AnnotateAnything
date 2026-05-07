from typing import Dict, Tuple
import torch
from dataclasses import MISSING

from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply
from magicsim.Env.Planner.Utils import quat_mul
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.managers import ActionTermCfg as ActionTerm

from magicsim.Env.Robot.Cfg.Manipulator.Manipulator import FrameSensorCfg
from magicsim.Env.Robot.Cfg.MobileManip.MobileManip import MobileManipActionsCfg
import magicsim.Env.Robot.mdp as mdp
from magicsim.Env.Robot.terms import transforms as transforms_terms
from magicsim.Env.Robot.Cfg.Base import (
    RobotCfg,
    RobotObsCfg,
)
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils

from magicsim import MAGICSIM_ASSETS
from magicsim.Env.Robot.Cfg.MobileManip.MobileManip import MobileManipPlannerCfg
from magicsim.Env.Robot.mdp.pink_ik import (
    LocalFrameTask,
)
from magicsim.Env.Robot.mdp.pink_actions_cfg import (
    PinkIKControllerCfg,
    PinkInverseKinematicsActionCfg,
)
from magicsim.Env.Robot.mdp.curobo_ik_cfg import DualCuroboIKActionCfg


# ================================
#  Joint name patterns
# ================================

_L_ARM_JOINTS = [f"L_arm_j{i}" for i in range(1, 8)]
_R_ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]
_ARM_JOINTS = _L_ARM_JOINTS + _R_ARM_JOINTS  # 14 arm joints (7 L + 7 R)
_TORSO_JOINTS = ["torso_j1", "torso_j2", "torso_j3"]
_HEAD_JOINTS = ["head_j1", "head_j2", "head_j3"]

# Pink IK is expected to control the torso too (the torso DOF lives between
# the mobile base and ``arm_center`` so both arms share it), so torso joints
# are bundled into the pink-controlled set alongside the 14 arm joints.
_PINK_CONTROLLED_JOINTS = _TORSO_JOINTS + _ARM_JOINTS  # 3 + 14 = 17

# Sharpa hand finger joint names — 22 right + 22 left = 44 revolute DOFs.
# Right-hand order matches the upstream sharpa annotation's
# ``OUTPUT_JOINT_ORDER`` exactly so a 22-vec ``coarse_grasp.joints`` /
# ``final_grasp.joints`` from ``sharpa_grasp_pose.json`` lands in
# ``eef_action[0:22]`` without any permutation. Left-hand mirrors the
# right list with ``right_`` → ``left_`` so ``eef_action[22:44]`` follows
# the same per-finger structure.
#
# IMPORTANT: paired with ``preserve_order=True`` on the
# JointPositionToLimitsActionCfg below — otherwise IsaacLab falls back
# to articulation index order (interleaved L/R) and the layout below is
# meaningless.
_R_HAND_JOINT_NAMES = [
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
]
_L_HAND_JOINT_NAMES = [n.replace("right_", "left_") for n in _R_HAND_JOINT_NAMES]
_SHARPA_HAND_JOINT_NAMES = _R_HAND_JOINT_NAMES + _L_HAND_JOINT_NAMES
_NUM_SHARPA_HAND_JOINTS = len(_SHARPA_HAND_JOINT_NAMES)  # 44
_NUM_ARM_JOINTS = 14  # 7 L + 7 R
# Pink IK solves torso + arms (17 joints). The Sharpa fingers are driven
# separately through ``eef_action`` (joint-position control), matching the
# G1 pattern where the palm IK targets are independent of the hand joints.
_NUM_PINK_IK_JOINTS = len(_PINK_CONTROLLED_JOINTS)


# ================================
#  USD-aligned drive parameters
# ================================
#
# All values below are read directly from
# ``Assets/Robots/vega_1p_sharpa.usd`` (PhysicsRevoluteJoint /
# PhysicsPrismaticJoint drive props). Keep these in sync with the USD —
# any tuning done in Isaac Sim's authoring UI lives there, not here, so
# diverging the cfg silently re-tunes the physics.
#
# Re-extract with::
#     python -c "from pxr import Usd; s=Usd.Stage.Open('Assets/Robots/vega_1p_sharpa.usd'); ..."
# (see ``drive:angular:physics:{stiffness,damping,maxForce,targetPosition}``).
# Note: USD revolute targets are in DEGREES; converted to radians here.

# Torso: USD target_deg = (60, 90, -25). Same kp/kd, per-joint effort ladder.
_TORSO_DEFAULT_POS = {  # rad
    "torso_j1": 1.0472,  # 60°
    "torso_j2": 1.5708,  # 90°
    "torso_j3": -0.4363,  # -25°
}
_TORSO_KP = 800000.0
_TORSO_KD = 36000.0
_TORSO_EFFORT = {
    "torso_j1": 700000.0,
    "torso_j2": 380000.0,
    "torso_j3": 380000.0,
}

# Head: per-joint kp/kd/effort. USD has no targetPosition — defaults to 0.
_HEAD_KP = {"head_j1": 45.22, "head_j2": 53.39, "head_j3": 30.43}
_HEAD_KD = {"head_j1": 0.01809, "head_j2": 0.02136, "head_j3": 0.01217}
_HEAD_EFFORT = {"head_j1": 6.0, "head_j2": 2.5, "head_j3": 6.0}

# Arms: per-joint kp ladder (USD's tuned values, ~2500 → ~113 from base
# to wrist). USD authored damping = 150 (L) / 100 (R), but we override
# both sides to **kd=150** so a single actuator group covers all 14
# joints; the difference was small enough not to matter for control.
# R_arm_j7 carries a 250 N effort cap (lighter right wrist); the other
# 13 joints are 1500.
_ARM_KP = {
    "L_arm_j1": 2503.5105,
    "L_arm_j2": 2174.9553,
    "L_arm_j3": 1889.6868,
    "L_arm_j4": 1059.5746,
    "L_arm_j5": 659.2767,
    "L_arm_j6": 210.2241,
    "L_arm_j7": 113.6074,
    "R_arm_j1": 2448.9192,
    "R_arm_j2": 2174.8486,
    "R_arm_j3": 1885.5143,
    "R_arm_j4": 1056.1354,
    "R_arm_j5": 661.4667,
    "R_arm_j6": 210.5952,
    "R_arm_j7": 113.3994,
}
_ARM_KD = 150.0  # unified for both sides
_ARM_EFFORT = {
    "L_arm_j[1-7]": 1500.0,
    "R_arm_j[1-6]": 1500.0,
    "R_arm_j7": 250.0,
}
_ARM_DEFAULT_POS = {
    "L_arm_j1": -0.7854,  # -45°
    "R_arm_j1": 0.7854,  # +45°
}

# Hands: 44 finger DOFs. USD-authored per-joint kp/kd are tiny
# (~0.008–2.86 / 4e-4 × kp) and were causing visible oscillation under
# the 100 Hz sim — the under-damped low-stiffness combination lets the
# fingers wobble around the joint-pos target instead of settling.
# Override with a stable uniform tuning:
#   stiffness = 20.0, damping = 2.0, maxForce = 300.0
# (kept alongside the USD values for reference; if a future tuning
# pass lands closer to USD, swap the actuator block back to the
# per-joint dicts).
_HAND_KP = 20.0
_HAND_KD = 2.0
_HAND_EFFORT = 300.0


# ================================
#  P-controller helper (holonomic base + torso via arm_center)
# ================================
#
# The Vega 1P curobo yaml uses ``info_links = [vega_1p_base, arm_center, R_ee,
# L_ee]`` → MotionGen emits D=28 trajectories with block layout:
#     [ vega_1p_base(7) | arm_center(7) | R_ee(7) | L_ee(7) ]
# This matches G1's ``[pelvis | torso | right_palm | left_palm]`` exactly, with
# the link mapping:
#     vega_1p_base ↔ pelvis      (holonomic mobile base)
#     arm_center   ↔ torso       (IMU in torso)
#     R_ee         ↔ right_palm
#     L_ee         ↔ left_palm
#
# So the preprocess / pcontroller logic is **identical** to G1's 15-dim action
# [vega_1p_base_pose(7) | arm_center_pose(7) | lock_flag(1)]. We reuse
# :class:`G1PControllerHelper` verbatim and only override ``move_strategy`` to
# drop G1's stand-up / squat segments — Vega's base Z is constant.


Vega1pSharpaRestPose7 = Tuple[float, float, float, float, float, float, float]


class Vega1pSharpaPControllerHelper:
    """Vega 1P + SharpaWave P-controller helper.

    Accepts the **same 15-dim G1-style input** that curobo emits for the
    vega trajectory (``info_links=[vega_1p_base, arm_center, R_ee, L_ee]``):

        ``[ vega_1p_base(7) | arm_center(7) | lock_flag(1) ]``

    but returns a **4-dim output** ``[x, y, heading, mode_flag]`` because the
    wheeled base only exposes 3 drive joints (``dummy_base_prismatic_x/y`` +
    ``dummy_base_revolute_z``) — there is no WBC / torso-velocity channel to
    pass through, so ``p_controller_n_extra_dims = 0``. The arm_center block
    (columns 7–13) is parsed only for NaN-fallback bookkeeping; it does not
    drive base motion.

    Stateful NaN fallbacks mirror G1's helper so ``MobileMoveL`` IK-wait rows
    (all-NaN) resolve to ``lock_skip`` with the last valid base pose.
    """

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        self._last_target_xy = torch.full(
            (num_envs, 2), float("nan"), device=device, dtype=torch.float32
        )
        self._last_target_yaw = torch.full(
            (num_envs,), float("nan"), device=device, dtype=torch.float32
        )

    def preprocess(
        self,
        action: torch.Tensor,
        robot_state: Dict,
        env_ids: torch.Tensor,
        device: torch.device = torch.device("cuda:0"),
    ) -> torch.Tensor:
        """15-dim G1-style input → 4-dim P-controller format.

        Input ``action[:, :7]`` = vega_1p_base pose (xyz + wxyz).
        Input ``action[:, 7:14]`` = arm_center pose (ignored for base motion).
        Input ``action[:, 14]`` = lock_flag, ``{-2 skip, -1 lock_skip, 0 nav, 1 turning}``.

        mode_flag rules:

        * ``-2 / -1`` — pass through (skip / lock_skip).
        * ``1`` — pass through (upstream ``move_strategy`` explicitly requested
          turning, e.g. rotation-padding segment).
        * ``0`` — nav; **upgrade to ``1`` (turning) when the base is already
          within ``position_threshold`` of the target** (mirrors
          ``RidgebackFrankaPControllerHelper``). This lets the PController
          rotate in place instead of chasing residual position error once the
          XY has converged.
        * All-NaN row → forced ``-1`` (lock_skip) with last-target / current
          pose fallback.

        Output ``[target_x, target_y, target_heading, mode_flag]``. Because
        the base is holonomic, ``target_heading`` always equals the base yaw
        extracted from ``action[:, 3:7]``; the ``turning`` vs ``nav``
        distinction does not change heading (no torso-decoupling needed).
        """
        action = action.to(device)
        N = action.shape[0]

        current_pos = robot_state["base_pos"][env_ids]
        current_quat = robot_state["base_quat"][env_ids]
        current_xy = current_pos[:, :2]
        _, _, current_yaw = euler_xyz_from_quat(current_quat)

        nan_mask = torch.isnan(action).all(dim=1)

        lock_flag = action[:, 14]
        # All-NaN row: force lock_skip (not skip) so held pose comes from
        # _last_* fallback rather than a zero last_command.
        lock_flag = torch.where(nan_mask, torch.tensor(-1.0, device=device), lock_flag)

        target_x = action[:, 0]
        target_y = action[:, 1]

        quat = action[:, 3:7].clone()
        nan_quat_mask = torch.isnan(quat).any(dim=1)
        if torch.any(nan_quat_mask):
            quat[nan_quat_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        _, _, target_yaw = euler_xyz_from_quat(quat)

        if torch.any(nan_mask):
            last_xy = self._last_target_xy[env_ids].to(device=device)
            last_xy_nan = torch.isnan(last_xy).any(dim=1, keepdim=True).expand(-1, 2)
            xy_fallback = torch.where(last_xy_nan, current_xy, last_xy)
            last_yaw = self._last_target_yaw[env_ids].to(device=device)
            yaw_fallback = torch.where(torch.isnan(last_yaw), current_yaw, last_yaw)
            target_x = torch.where(nan_mask, xy_fallback[:, 0], target_x)
            target_y = torch.where(nan_mask, xy_fallback[:, 1], target_y)
            target_yaw = torch.where(nan_mask, yaw_fallback, target_yaw)

        # mode_flag: upstream lock_flag is passed through for the explicit
        # states (-2 skip, -1 lock_skip, 1 turning). When upstream is 0 (nav)
        # we additionally apply the RidgebackFranka distance-threshold rule:
        # if the base is already within ``position_threshold`` of the target,
        # upgrade to ``1`` (turning) so the PController rotates in place
        # instead of chasing residual position error.
        skip_mask = torch.abs(lock_flag + 2.0) < 0.5
        lock_skip_mask = torch.abs(lock_flag + 1.0) < 0.5
        turning_input_mask = torch.abs(lock_flag - 1.0) < 0.5
        nav_input_mask = torch.abs(lock_flag) < 0.5

        position_threshold = 0.1
        dxy = torch.stack(
            [target_x - current_xy[:, 0], target_y - current_xy[:, 1]], dim=1
        )
        dist_to_target = torch.norm(dxy, dim=1)
        nav_upgrade_mask = nav_input_mask & (dist_to_target < position_threshold)

        mode_flag = torch.zeros(N, device=device)
        mode_flag[skip_mask] = -2.0
        mode_flag[lock_skip_mask] = -1.0
        mode_flag[turning_input_mask] = 1.0
        mode_flag[nav_upgrade_mask] = 1.0

        result = torch.stack([target_x, target_y, target_yaw, mode_flag], dim=1)

        self._last_target_xy[env_ids] = torch.stack([target_x, target_y], dim=1).to(
            device=self._last_target_xy.device, dtype=self._last_target_xy.dtype
        )
        self._last_target_yaw[env_ids] = target_yaw.to(
            device=self._last_target_yaw.device, dtype=self._last_target_yaw.dtype
        )
        return result

    def reset_idx(self, env_ids) -> None:
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        self._last_target_xy[env_ids] = float("nan")
        self._last_target_yaw[env_ids] = float("nan")

    @staticmethod
    def move_strategy(
        trajectory: torch.Tensor,
        robot_state: Dict[str, torch.Tensor],
        hand_id: int = -1,
        lock_xy_steps: int = 10,
        num_rotation_steps: int = 50,
        lock_fwd_offset: float = 0.12,
        lock_perp_offset: float = 0.3,
        yaw_axis_correction: float = 0.8,
        left_rest_pose: Vega1pSharpaRestPose7 = (
            0.25,
            0.25,
            0.60,
            0.0,
            1.0,
            0.0,
            0.0,
        ),
        right_rest_pose: Vega1pSharpaRestPose7 = (
            0.25,
            -0.25,
            0.60,
            0.0,
            1.0,
            0.0,
            0.0,
        ),
    ) -> torch.Tensor:
        return vega1p_sharpa_move_strategy(
            trajectory,
            robot_state,
            hand_id=hand_id,
            lock_xy_steps=lock_xy_steps,
            num_rotation_steps=num_rotation_steps,
            lock_fwd_offset=lock_fwd_offset,
            lock_perp_offset=lock_perp_offset,
            yaw_axis_correction=yaw_axis_correction,
            left_rest_pose=left_rest_pose,
            right_rest_pose=right_rest_pose,
        )


def vega1p_sharpa_move_strategy(
    trajectory: torch.Tensor,
    robot_state: Dict[str, torch.Tensor],
    hand_id: int = -1,
    lock_xy_steps: int = 10,
    num_rotation_steps: int = 50,
    lock_fwd_offset: float = 0.12,
    lock_perp_offset: float = 0.3,
    yaw_axis_correction: float = 0.8,
    left_rest_pose: Vega1pSharpaRestPose7 = (
        0.25,
        0.25,
        0.60,
        0.0,
        1.0,
        0.0,
        0.0,
    ),
    right_rest_pose: Vega1pSharpaRestPose7 = (
        0.25,
        -0.25,
        0.60,
        0.0,
        1.0,
        0.0,
        0.0,
    ),
) -> torch.Tensor:
    """
    Vega 1P + Sharpa move strategy — wheeled adaptation of :func:`g1_move_strategy`.

    Trajectory layout (D=28):
        ``[ vega_1p_base(7) | arm_center(7) | R_ee(7) | L_ee(7) ]``

    Segments:
        1. Horizontal move — ``lock_flag = 0`` (nav). Base XY interpolates from
           the planned start to ``locked_xy`` (a perpendicular/forward-offset
           hold pose near the final grasp); last ``lock_xy_steps`` frames pin
           XY to ``locked_xy``.
        2. Rotation padding (optional) — ``lock_flag = 1`` (turning). Holds
           ``locked_xy`` while the base rotates to the final target yaw.

    No stand-up / squat segments: Vega's base Z is fixed and torso
    roll/pitch/yaw is plumbed through the ``arm_center`` block of every
    waypoint (same as G1's torso block), so posture is already tracked by
    curobo's plan.

    Rest poses for the inactive arm are expressed in the ``vega_1p_base``
    frame (analogue of G1's pelvis frame).
    """
    device = trajectory.device
    dtype = trajectory.dtype
    D = trajectory.shape[1]
    current_base_pos = robot_state["base_pos"]
    if current_base_pos.ndim > 1:
        current_base_pos = current_base_pos[0]

    if trajectory.shape[0] == 0:
        return torch.zeros(0, D + 1, device=device, dtype=dtype)

    start_base_pose = trajectory[0, :7].clone()
    target_base_pose = trajectory[-1, :7].clone()

    if yaw_axis_correction > 0:
        _qw, _qx = target_base_pose[3], target_base_pose[4]
        _qy, _qz = target_base_pose[5], target_base_pose[6]
        _yaw = torch.atan2(
            2.0 * (_qw * _qz + _qx * _qy),
            1.0 - 2.0 * (_qy**2 + _qz**2),
        )
        _nearest = torch.round(_yaw / torch.pi) * torch.pi
        _corrected_yaw = _yaw + yaw_axis_correction * (_nearest - _yaw)
        _half = _corrected_yaw * 0.5
        target_base_pose[3] = torch.cos(_half)
        target_base_pose[4] = 0.0
        target_base_pose[5] = 0.0
        target_base_pose[6] = torch.sin(_half)

    # Vega 1P base Z is held by the holonomic mobile base; there is no
    # stand-up / squat, so ``base_z_hold`` is just the start Z.
    base_z_hold = float(current_base_pos[2].item())

    arm_dim = 14
    arm_start = D - arm_dim  # last 14 = right(7) + left(7)
    base_block_dim = arm_start  # vega_1p_base(7) + arm_center(7) = 14
    left_rest_t = torch.tensor(left_rest_pose, device=device, dtype=dtype)
    right_rest_t = torch.tensor(right_rest_pose, device=device, dtype=dtype)

    def _base_frame_rest_to_world(
        base_pose: torch.Tensor, rest_pose_7: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p_l, q_l = rest_pose_7[0:3], rest_pose_7[3:7]
        p_w = base_pose[0:3] + quat_apply(base_pose[3:7], p_l)
        q_w = quat_mul(base_pose[3:7], q_l)
        return p_w, q_w

    def _make_waypoint(lock_flag: float) -> torch.Tensor:
        wp = torch.full((D + 1,), float("nan"), device=device, dtype=dtype)
        wp[-1] = lock_flag
        return wp

    def _fill_eef_height_relative_to_base(
        wp: torch.Tensor, traj: torch.Tensor, traj_idx: int, base_pose: torch.Tensor
    ) -> None:
        if arm_start < 0 or arm_start + arm_dim > D:
            return
        base_orig = traj[traj_idx, 0:3]

        def _fill_arm(k: int, rest_pose_7: torch.Tensor | None) -> None:
            eef_o = traj[traj_idx, arm_start + k : arm_start + k + 7]
            if rest_pose_7 is not None:
                p_w, q_w = _base_frame_rest_to_world(base_pose, rest_pose_7)
                wp[arm_start + k : arm_start + k + 3] = p_w
                wp[arm_start + k + 3 : arm_start + k + 7] = q_w
            else:
                rel_z = eef_o[2] - base_orig[2]
                wp[arm_start + k + 0] = eef_o[0]
                wp[arm_start + k + 1] = eef_o[1]
                wp[arm_start + k + 2] = base_pose[2] + rel_z
                wp[arm_start + k + 3 : arm_start + k + 7] = eef_o[3:7]

        if hand_id == 0:
            _fill_arm(0, None)
            _fill_arm(7, left_rest_t)
        elif hand_id == 1:
            _fill_arm(0, right_rest_t)
            _fill_arm(7, None)
        else:
            for k in range(0, arm_dim, 7):
                _fill_arm(k, None)

    segments = []

    # Segment 1: Horizontal move (lock_flag = 0, nav)
    num_move_points = trajectory.shape[0]
    tgt_qw, tgt_qx = target_base_pose[3], target_base_pose[4]
    tgt_qy, tgt_qz = target_base_pose[5], target_base_pose[6]
    target_yaw = torch.atan2(
        2.0 * (tgt_qw * tgt_qz + tgt_qx * tgt_qy),
        1.0 - 2.0 * (tgt_qy**2 + tgt_qz**2),
    )
    cos_yaw = torch.cos(target_yaw)
    sin_yaw = torch.sin(target_yaw)
    fwd = torch.stack([cos_yaw, sin_yaw])
    right = torch.stack([sin_yaw, -cos_yaw])

    if hand_id == 0:
        grasp_xy = trajectory[-1, arm_start : arm_start + 2].clone()
        locked_xy = grasp_xy - lock_perp_offset * right - lock_fwd_offset * fwd
    elif hand_id == 1:
        grasp_xy = trajectory[-1, arm_start + 7 : arm_start + 9].clone()
        locked_xy = grasp_xy + lock_perp_offset * right - lock_fwd_offset * fwd
    else:
        grasp_xy_r = trajectory[-1, arm_start : arm_start + 2]
        grasp_xy_l = trajectory[-1, arm_start + 7 : arm_start + 9]
        grasp_xy = (grasp_xy_r + grasp_xy_l) / 2.0
        locked_xy = grasp_xy - lock_fwd_offset * fwd

    lock_start_idx = (
        max(0, num_move_points - lock_xy_steps)
        if lock_xy_steps > 0
        else num_move_points
    )
    move_start_xy = start_base_pose[:2].clone()

    for i in range(num_move_points):
        wp = _make_waypoint(0.0)
        if base_block_dim > 0:
            wp[:base_block_dim] = trajectory[i, :base_block_dim].clone()

        if i >= lock_start_idx:
            wp[:2] = locked_xy
        else:
            t_i = i / max(lock_start_idx - 1, 1)
            wp[:2] = move_start_xy + t_i * (locked_xy - move_start_xy)

        wp[2] = base_z_hold  # Vega base Z is constant
        wp[3:7] = trajectory[i, 3:7]  # keep planned base orientation
        _fill_eef_height_relative_to_base(wp, trajectory, i, wp[0:7])
        segments.append(wp)

    # Segment 2: Rotation padding (lock_flag = 1, turning)
    if num_rotation_steps > 0:
        target_orientation = target_base_pose[3:7].clone()
        for i in range(num_rotation_steps):
            wp = _make_waypoint(1.0)
            if base_block_dim > 0:
                wp[:base_block_dim] = trajectory[-1, :base_block_dim].clone()
            wp[:2] = locked_xy
            wp[2] = base_z_hold
            wp[3:7] = target_orientation
            _fill_eef_height_relative_to_base(wp, trajectory, -1, wp[0:7])
            segments.append(wp)

    if len(segments) == 0:
        wp = _make_waypoint(-1.0)
        wp[:D] = trajectory[-1, :D].clone()
        wp[2] = base_z_hold
        wp[-1] = -1.0
        return wp.unsqueeze(0)

    result = torch.stack(segments, dim=0)
    assert result.shape[1] == D + 1, f"result.shape: {result.shape}"

    # Terminal EEF: active hand = last plan frame; inactive = base-frame rest → world.
    if result.shape[0] > 0 and arm_start >= 0 and arm_start + arm_dim <= D:
        tr_last = trajectory[-1, arm_start : arm_start + arm_dim].clone()
        last_base_pose = result[-1, 0:7]
        if hand_id == -1:
            result[-1, arm_start : arm_start + arm_dim] = tr_last
        elif hand_id == 0:
            result[-1, arm_start : arm_start + 7] = tr_last[:7]
            p_w, q_w = _base_frame_rest_to_world(last_base_pose, left_rest_t)
            result[-1, arm_start + 7 : arm_start + 10] = p_w
            result[-1, arm_start + 10 : arm_start + 14] = q_w
        else:
            p_w, q_w = _base_frame_rest_to_world(last_base_pose, right_rest_t)
            result[-1, arm_start : arm_start + 3] = p_w
            result[-1, arm_start + 3 : arm_start + 7] = q_w
            result[-1, arm_start + 7 : arm_start + 14] = tr_last[7:14]

    return result


# ================================
#  Articulation configuration
# ================================


VEGA_1P_SHARPA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/vega_1p_sharpa.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Spawn pose mirrors the USD's ``drive:angular:physics:targetPosition``
        # (vega_1p_sharpa.usd) so high-stiffness joints (torso kp=8e5)
        # don't yank from spawn=0 to target=60°/90°/-25° on the first
        # sim step. Five joints carry non-zero USD targets:
        #   torso_j1=60°, torso_j2=90°, torso_j3=-25°,
        #   L_arm_j1=-45°, R_arm_j1=45°
        # Note: this DIFFERS from cuRobo's retract in
        # ``magicsim_vega1p_sharpa{,_mobile}.yml``
        # (torso_j1=0.3, torso_j2=0.5, rest=0). If MotionGen cold-starts
        # diverge after this change, sync the curobo yml retract to
        # match these defaults.
        # Pattern order matters but overlap is rejected by IsaacLab's
        # resolve_matching_names_values, so the catch-all regex excludes
        # the explicitly-listed joints.
        joint_pos={
            **_TORSO_DEFAULT_POS,
            **_ARM_DEFAULT_POS,
            "^(?!torso_j[123]$|[LR]_arm_j1$).*$": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # All kp / kd / effort values mirror the USD's authored drive
        # props (see _TORSO_*, _HEAD_*, _ARM_*, _HAND_* constants above).
        # Per-joint dicts are used where the USD tunes joints
        # non-uniformly (torso effort ladder, head, arms, hands).
        "base": ImplicitActuatorCfg(
            joint_names_expr=["dummy_base_.*"],
            effort_limit_sim=4800.0,
            stiffness=0.0,
            damping=1e5,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_j.*"],
            effort_limit_sim=_TORSO_EFFORT,
            stiffness=_TORSO_KP,
            damping=_TORSO_KD,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_j.*"],
            effort_limit_sim=_HEAD_EFFORT,
            stiffness=_HEAD_KP,
            damping=_HEAD_KD,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["L_arm_j.*", "R_arm_j.*"],
            effort_limit_sim=_ARM_EFFORT,
            stiffness=_ARM_KP,
            damping=_ARM_KD,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_(thumb|index|middle|ring|pinky)_.*",
                "right_(thumb|index|middle|ring|pinky)_.*",
            ],
            effort_limit_sim=_HAND_EFFORT,
            stiffness=_HAND_KP,
            damping=_HAND_KD,
        ),
    },
    # The vega_1p_sharpa.usd already carries ArticulationRootAPI on the
    # ``/root_joint`` PhysicsFixedJoint that anchors ``vega_1p_mobile`` to the
    # world. Leave the override unset so IsaacLab keeps that fixed-base setup;
    # the holonomic motion is delivered through the dummy base prismatic /
    # revolute joints below the fixed root.
    articulation_root_prim_path=None,
)


# ================================
#  Pink IK controller configuration (dual arm)
# ================================
#
# Structure mirrors ``G1_PINK_IK_CONTROLLER_CFG``: two ``LocalFrameTask`` targets
# (left/right end-effectors) expressed in the mobile base frame, plus a shared
# ``NullSpacePostureTask`` biasing the 14 arm joints toward their rest.
#
# The simplified URDF is expected at ``Assets/Robots/URDF/vega_1p_sharpa.urdf``
# (mirrors ``franka_panda.urdf`` pattern used by RidgebackFranka). If only the
# full curobo URDF is available it can be symlinked or copied here.

VEGA_1P_SHARPA_PINK_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="Robot_0",
    base_link_name="vega_1p_base",
    num_hand_joints=0,  # fingers handled by eef_action, not pink IK
    show_ik_warnings=True,
    # Simplified URDF: arms + torso + head only. Sharpa fingers and wheels
    # are stripped so pinocchio's ``model.nq`` matches Isaac Lab's joint-count
    # (no continuous joints → no nq/nv mismatch).
    urdf_path=f"{MAGICSIM_ASSETS}/Robots/URDF/vega_1p.urdf",
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        LocalFrameTask(
            "R_ee",
            base_link_frame_name="vega_1p_base",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        LocalFrameTask(
            "L_ee",
            base_link_frame_name="vega_1p_base",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
    ],
    fixed_input_tasks=[],
    amplify_factor=1.0,
)


# ================================
#  Action configuration
# ================================


@configclass
class Vega1pSharpaActionsCfg(MobileManipActionsCfg):
    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "base_action": {
            "holonomic_action": mdp.HolonomicActionCfg(
                joint_names=[
                    "dummy_base_prismatic_y_joint",
                    "dummy_base_prismatic_x_joint",
                    "dummy_base_revolute_z_joint",
                ],
                action_space=torch.tensor(
                    [
                        [-2, -2, -2],
                        [2, 2, 2],
                    ]
                ),
            ),
            "holonomic_vw_action": mdp.HolonomicVWActionCfg(
                joint_names=[
                    "dummy_base_prismatic_y_joint",
                    "dummy_base_prismatic_x_joint",
                    "dummy_base_revolute_z_joint",
                ],
                action_space=torch.tensor(
                    [
                        [-5, -5],
                        [5, 5],
                    ]
                ),
            ),
        },
        "arm_action": {
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=_ARM_JOINTS,
            ),
            # Dual-arm Pink IK over the 14 arm joints only. The per-env action
            # layout is ``[ right_wrist pose (7) | left_wrist pose (7) ]`` —
            # Sharpa finger joints are driven by ``eef_action`` (mirrors the
            # G1 pattern, where palm IK and hand control are decoupled).
            "ik_pink": PinkInverseKinematicsActionCfg(
                pink_controlled_joint_names=_PINK_CONTROLLED_JOINTS,
                num_joints=_NUM_PINK_IK_JOINTS,
                hand_joint_names=None,
                target_eef_link_names={
                    "right_wrist": "R_ee",
                    "left_wrist": "L_ee",
                },
                action_space=torch.tensor(
                    [
                        [
                            0.1,
                            -0.7,
                            0.3,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            0.1,
                            -0.7,
                            0.3,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                        ],
                        [
                            0.9,
                            0.7,
                            1.4,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.9,
                            0.7,
                            1.4,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                    ]
                ),
                controller=VEGA_1P_SHARPA_PINK_IK_CONTROLLER_CFG,
                relative_to_base=False,
            ),
            # Dual-arm cuRobo IK action — alternative to ``ik_pink`` when
            # the pinocchio-based pink solver is suspect. Loads its own
            # ``InverseKinematics`` from ``magicsim_vega1p_sharpa.yml``
            # (locked-base — no virtual ``dummy_base_*`` joints in cspace,
            # so the action assumes the IsaacLab-side ``vega_1p_base`` is
            # static). 14-dim action layout matches pink IK:
            # ``[right_pose(7) | left_pose(7)]`` in env-origin world frame.
            # Per-slice NaN fallback: missing arm holds last-valid / current
            # FK (mirrors pink IK behaviour).  ``diff_ik_method=None``
            # disables the inter-decimation Jacobian refinement so each
            # tick is a pure cuRobo solve.
            "ik_dual_curobo": DualCuroboIKActionCfg(
                # cuRobo's locked-base yml has cspace = torso(3) + head(3)
                # + L_arm(7) + R_arm(7) = 20 active DOFs. The action term
                # asserts every cuRobo controlled joint appears in
                # ``joint_names`` (which __post_init__ flattens as
                # right + left), so we bundle torso + head into
                # ``right_joint_names`` alongside the right arm. cuRobo
                # solves all 20 jointly; the per-arm diff-IK split is moot
                # since ``diff_ik_method=None`` below.
                right_joint_names=_TORSO_JOINTS + _HEAD_JOINTS + _R_ARM_JOINTS,
                left_joint_names=_L_ARM_JOINTS,
                right_eef_link_name="R_ee",
                left_eef_link_name="L_ee",
                robot_cfg_file="magicsim_vega1p_sharpa.yml",
                action_space=torch.tensor(
                    [
                        [
                            0.1,
                            -0.7,
                            0.3,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            0.1,
                            -0.7,
                            0.3,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                        ],
                        [
                            0.9,
                            0.7,
                            1.4,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.9,
                            0.7,
                            1.4,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                    ]
                ),
                num_seeds=8,
                position_threshold=0.005,
                rotation_threshold=0.05,
                self_collision_check=True,
                fallback_to_current_on_fail=True,
                decimation=1,
                diff_ik_method=None,
                # Vega is parked at world (-1, 0, 0.1), NOT at env_origin,
                # AND the cuRobo solver's base link (``vega_1p_base``) is
                # below the articulation root (``vega_1p_mobile``) +
                # dummy_base virtual joints. Reading ``root_pos_w`` would
                # give us ``vega_1p_mobile`` — the wrong frame for cuRobo.
                # Enable the live world→base transform with the explicit
                # base-link body name so the action reads
                # ``body_link_state_w[vega_1p_base_idx]`` each tick
                # (matches pink IK's ``_get_base_link_frame_transform``).
                world_to_base_frame=True,
                base_link_name="vega_1p_base",
            ),
        },
        "eef_action": {
            # Direct joint-position control over the 44 Sharpa finger
            # joints. Layout = ``[right_22 (OUTPUT_JOINT_ORDER) |
            # left_22 (mirror)]`` — locked in by passing the explicit
            # name list + ``preserve_order=True`` so IsaacLab ships the
            # action in the order this list is written, NOT in
            # articulation order (which would be L/R interleaved).
            # Right-22 matches the annotation's ``OUTPUT_JOINT_ORDER``
            # so ``eef_action[0:22]`` is a drop-in for
            # ``coarse_grasp/final_grasp.joints`` from
            # ``sharpa_grasp_pose.json``.
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=_SHARPA_HAND_JOINT_NAMES,
                preserve_order=True,
            ),
        },
    }

    def __post_init__(self):
        if self.arm_action_name is not None:
            self.arm_action = self.available_action["arm_action"][self.arm_action_name]
            self.arm_action.asset_name = self.asset_name
        else:
            self.arm_action = None

        if self.eef_action_name is not None:
            self.eef_action = self.available_action["eef_action"][self.eef_action_name]
            self.eef_action.asset_name = self.asset_name
        else:
            self.eef_action = None

        if self.base_action_name is not None:
            self.base_action = self.available_action["base_action"][
                self.base_action_name
            ]
            self.base_action.asset_name = self.asset_name
        else:
            self.base_action = None

        del self.available_action
        del self.asset_name
        del self.arm_action_name
        del self.eef_action_name
        del self.base_action_name


# ================================
#  Frame configuration (dual EEF)
# ================================


class Vega1pSharpaFrameCfg(FrameSensorCfg):
    def __post_init__(self):
        self.target_frames = [
            FrameTransformerCfg.FrameCfg(
                name="left_end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
            FrameTransformerCfg.FrameCfg(
                name="right_end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
            FrameTransformerCfg.FrameCfg(name="arm_base", offset=OffsetCfg()),
        ]
        self.visualizer_cfg.prim_path = (
            self.robot_prim_path + "/Visuals/FrameTransformer"
        )
        self.target_frames[0].prim_path = self.robot_prim_path + "/L_ee"
        self.target_frames[1].prim_path = self.robot_prim_path + "/R_ee"
        self.target_frames[2].prim_path = self.robot_prim_path + "/vega_1p_base"
        self.prim_path = self.robot_prim_path + "/vega_1p_base"


# ================================
#  Planner configuration
# ================================


@configclass
class Vega1pSharpaPlannerCfg(MobileManipPlannerCfg):
    max_eef_num: int = 2

    base_action_dim: Dict[str, int] = {
        "dwb_differential": 8,
        "dwb_holonomic": 8,
        "default": 8,
        # 15-dim G1-style input: [vega_1p_base_pose(7), arm_center_pose(7), lock_flag(1)]
        "p_controller": 15,
    }
    base_action_space: Dict[str, torch.Tensor] = {
        "dwb_differential": torch.tensor(
            [
                [-5, -5],
                [5, 5],
            ],
        ),
        "dwb_holonomic": torch.tensor(
            [
                [-2, -2, -2],
                [2, 2, 2],
            ],
        ),
        "default": torch.tensor(
            [
                [-100, -100, 0, -1, -1, -1, -1, -1],
                [100, 100, 0, 1, 1, 1, 1, 0],
            ],
        ),
        # 15-dim G1-style: [base(7), arm_center(7), lock_flag]. lock_flag in
        # {-2 skip, -1 lock_skip, 0 nav, 1 turning} — see G1PControllerHelper.
        "p_controller": torch.tensor(
            [
                [
                    -100,
                    -100,
                    0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -100,
                    -100,
                    0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -2.0,
                ],
                [
                    100,
                    100,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    100,
                    100,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
            ],
        ),
    }
    # Planner-side arm action shape. The MotionGen / curobo layer still plans
    # in dual-EEF world-pose space (14 dims = right(7) + left(7)); the 44
    # Sharpa finger targets are attached by the high-level skill and only
    # materialize at the Isaac Lab action-term boundary (see
    # ``Vega1pSharpaActionsCfg.ik_pink``).
    arm_action_dim: Dict[str, int] = {
        "default": 14,
        "curobo": 14,
    }
    arm_action_space: Dict[str, torch.Tensor] = {
        "default": torch.tensor(
            [
                [-1] * 14,
                [1] * 14,
            ],
        ),
        "curobo": torch.tensor(
            [
                [-1] * 14,
                [1] * 14,
            ],
        ),
    }
    # No separate eef planner channel — Sharpa hand joints are folded into the
    # pink IK action term.
    eef_action_dim: Dict[str, int] = {
        "joint_pos": _NUM_SHARPA_HAND_JOINTS,
    }
    eef_action_space: Dict[str, torch.Tensor] = {
        "joint_pos": torch.tensor(
            [
                [-1.0] * _NUM_SHARPA_HAND_JOINTS,
                [1.0] * _NUM_SHARPA_HAND_JOINTS,
            ],
        ),
    }
    p_controller_helper = Vega1pSharpaPControllerHelper
    # Vega's wheeled base only drives 3 joints (x, y, yaw) — no WBC torso
    # channel, so n_extra_dims=0 and preprocess outputs [x, y, heading, mode_flag].
    # The torso block of the 15-dim input (arm_center pose) is parsed for NaN
    # fallback only; it does not propagate through PController.
    p_controller_n_extra_dims: int = 0
    move_strategy = Vega1pSharpaPControllerHelper.move_strategy
    move_strategy_distance_threshold: float = 0.1


# ================================
#  Observation configuration (dual EEF)
# ================================


@configclass
class Vega1pSharpaObsCfg(ObsGroup):
    asset_name: str = MISSING
    frame_name: str = MISSING
    joint_pos: ObsTerm = MISSING
    joint_vel: ObsTerm = MISSING
    joint_effort: ObsTerm = MISSING
    left_eef_pos: ObsTerm = MISSING
    left_eef_quat: ObsTerm = MISSING
    right_eef_pos: ObsTerm = MISSING
    right_eef_quat: ObsTerm = MISSING
    eef_pos: ObsTerm = MISSING
    eef_quat: ObsTerm = MISSING
    left_eef_relative_pos: ObsTerm = MISSING
    left_eef_relative_quat: ObsTerm = MISSING
    right_eef_relative_pos: ObsTerm = MISSING
    right_eef_relative_quat: ObsTerm = MISSING
    base_pos: ObsTerm = MISSING
    base_quat: ObsTerm = MISSING
    base_ang_vel: ObsTerm = MISSING
    base_lin_vel: ObsTerm = MISSING

    def __post_init__(self):
        asset_name = self.asset_name
        self.joint_pos = ObsTerm(
            func=transforms_terms.joint_pos_with_root_offset,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "x_joint_name": "dummy_base_prismatic_x_joint",
                "y_joint_name": "dummy_base_prismatic_y_joint",
                "yaw_joint_name": "dummy_base_revolute_z_joint",
            },
        )
        self.joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg(asset_name)},
        )
        self.joint_effort = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg(asset_name)},
        )
        # Dual-arm EEF world poses.
        self.left_eef_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "L_ee",
            },
        )
        self.left_eef_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "L_ee",
            },
        )
        self.right_eef_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "R_ee",
            },
        )
        self.right_eef_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "R_ee",
            },
        )
        # Dual-link EEF in world frame (right first, then left — matches G1).
        self.eef_pos = ObsTerm(
            func=transforms_terms.get_dual_link_position_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "link_name_1": "R_ee",
                "link_name_2": "L_ee",
            },
        )
        self.eef_quat = ObsTerm(
            func=transforms_terms.get_dual_link_quaternion_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "link_name_1": "R_ee",
                "link_name_2": "L_ee",
            },
        )
        # EEF relative to base (for local-frame policies). The default
        # ``target_frame_name`` in transforms_terms is ``"pelvis"`` (G1-specific),
        # so we override it to Vega's mobile-base link ``vega_1p_base``.
        self.left_eef_relative_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "L_ee",
                "target_frame_name": "vega_1p_base",
            },
        )
        self.left_eef_relative_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "L_ee",
                "target_frame_name": "vega_1p_base",
            },
        )
        self.right_eef_relative_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "R_ee",
                "target_frame_name": "vega_1p_base",
            },
        )
        self.right_eef_relative_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "R_ee",
                "target_frame_name": "vega_1p_base",
            },
        )
        self.base_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "vega_1p_base",
            },
        )
        self.base_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "vega_1p_base",
            },
        )
        self.base_ang_vel = ObsTerm(
            func=transforms_terms.get_target_link_ang_vel_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "vega_1p_base",
            },
        )
        self.base_lin_vel = ObsTerm(
            func=transforms_terms.get_target_link_lin_vel_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "vega_1p_base",
            },
        )
        self.enable_corruption = False
        self.concatenate_terms = False
        del self.asset_name
        del self.frame_name


# ================================
#  Composite robot configuration
# ================================


@configclass
class Vega1pSharpaCfg(RobotCfg):
    """Configuration for the mobile dual-arm Vega 1P with SharpaWave hands."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    base_action_name: str = "holonomic_action"
    arm_action_name: str = "ik_pink"
    # Sharpa hand joints are now decoupled from pink IK and driven via a
    # dedicated joint-position action term (mirrors G1 hand control).
    eef_action_name: str | None = "joint_pos"
    frame_name: str = "ee_frame"

    action: Vega1pSharpaActionsCfg = MISSING
    ee_frame: Vega1pSharpaFrameCfg = MISSING
    obs: RobotObsCfg = MISSING
    planner: Vega1pSharpaPlannerCfg = Vega1pSharpaPlannerCfg()

    def __post_init__(self):
        self.robot: ArticulationCfg = VEGA_1P_SHARPA_CFG

        self.robot.prim_path = self.prim_path
        self.action = Vega1pSharpaActionsCfg(
            asset_name=self.asset_name,
            base_action_name=self.base_action_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = Vega1pSharpaFrameCfg(robot_prim_path=self.robot.prim_path)
        self.obs = Vega1pSharpaObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        self.type = "mobilemanip"
