"""Genie1 (formerly G1_120s) — mobile dual-arm robot configuration.

Structure mirrors :mod:`magicsim.Env.Robot.Cfg.MobileManip.vega1psharpa`:

* 3-DOF holonomic dummy base (``dummy_base_prismatic_x_joint``,
  ``dummy_base_prismatic_y_joint``, ``dummy_base_revolute_z_joint``) rooted at
  ``genie1_mobile`` and anchoring ``base_link``.
* 2-DOF torso (``idx01_body_joint1`` prismatic riser + ``idx02_body_joint2``
  yaw) and 2-DOF head (``idx11_head_joint1``, ``idx12_head_joint2``) — the
  Genie1 torso/head has one fewer joint per chain than Vega's, but the
  kinematic role is identical: everything between the chassis and the two
  arm roots.
* 7+7 DOF dual arms branching from the shared ``arm_base_link`` pivot
  (analogous to Vega's ``arm_center``) to ``arm_l_end_link`` /
  ``arm_r_end_link``.
* Parallel-jaw grippers (16 revolute joints total, 8 per side) controlled
  through ``eef_action`` with joint-position targets; URDF mimic constraints
  keep the finger chain consistent. The grippers are **not** part of the
  pink-IK cspace (following G1's hand-IK decoupling).

The curobo trajectory layout matches Vega exactly so the P-controller and
``move_strategy`` logic can be reused verbatim:

    ``info_links = [base_link, arm_base_link, arm_r_end_link, arm_l_end_link]``
    → D=28: ``[ base_link(7) | arm_base_link(7) | R_ee(7) | L_ee(7) ]``
"""

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
    DampingTask,
    LocalFrameTask,
    NullSpacePostureTask,
)
from magicsim.Env.Robot.mdp.pink_actions_cfg import (
    PinkIKControllerCfg,
    PinkInverseKinematicsActionCfg,
)


# ================================
#  Joint name patterns
# ================================

_L_ARM_JOINTS = [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
_R_ARM_JOINTS = [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
_ARM_JOINTS = _L_ARM_JOINTS + _R_ARM_JOINTS  # 14 arm joints (7 L + 7 R)
_TORSO_JOINTS = ["idx01_body_joint1", "idx02_body_joint2"]
_HEAD_JOINTS = ["idx11_head_joint1", "idx12_head_joint2"]

# Pink IK controls torso + dual arms (17 joints in Vega; 16 here because the
# Genie1 torso has one fewer revolute). Head joints are left unmanaged —
# they're operator-driven, not part of manipulation IK.
_PINK_CONTROLLED_JOINTS = _TORSO_JOINTS + _ARM_JOINTS  # 2 + 14 = 16

# Parallel-jaw gripper joints (120s gripper; same closed-loop topology as the
# Isaac Lab ``Agibot A2D`` parallel-jaw hand). Each side has 8 revolute joints
# grouped by drive role:
#
#     primary driver  — ``idx41`` (L) / ``idx81`` (R), ``outer_joint1``
#     support drivers — ``idx42`` / ``idx32`` (L) and ``idx82`` / ``idx72`` (R):
#                       ``*_outer_joint3`` / ``*_inner_joint3``. Pre-loaded at
#                       low PD gain to keep the parallelogram taut.
#     passive mimics  — all remaining finger joints (URDF ``<mimic>``-linked).
#                       Driven at stiffness=damping=0 so PhysX's joint equality
#                       keeps them in sync with the primary without a redundant
#                       actuator pulling against it.
_L_GRIPPER_PRIMARY_REGEX = r"idx41_gripper_l_outer_joint1"
_L_GRIPPER_SUPPORT_REGEX = r"idx(32_gripper_l_inner_joint3|42_gripper_l_outer_joint3)"
_L_GRIPPER_PASSIVE_REGEX = r"idx(31|33|39|43|49)_gripper_l_.*"
_R_GRIPPER_PRIMARY_REGEX = r"idx81_gripper_r_outer_joint1"
_R_GRIPPER_SUPPORT_REGEX = r"idx(72_gripper_r_inner_joint3|82_gripper_r_outer_joint3)"
_R_GRIPPER_PASSIVE_REGEX = r"idx(71|73|79|83|89)_gripper_r_.*"
# Command regex covers every gripper revolute (primary + support + passive) so
# the ``eef_action`` joint-pos term scales 16-dim [-1, 1] commands against every
# driven joint — PhysX keeps the passive mimics consistent via joint equality.
_L_GRIPPER_JOINT_REGEX = r"idx(31|32|33|39|41|42|43|49)_gripper_l_.*"
_R_GRIPPER_JOINT_REGEX = r"idx(71|72|73|79|81|82|83|89)_gripper_r_.*"
_NUM_GRIPPER_JOINTS = 16  # 8 left + 8 right
_NUM_ARM_JOINTS = 14  # 7 L + 7 R
_NUM_PINK_IK_JOINTS = len(_PINK_CONTROLLED_JOINTS)


# ================================
#  P-controller helper (holonomic base + torso via arm_base_link)
# ================================
#
# The curobo yaml uses ``info_links = [base_link, arm_base_link, arm_r_end_link,
# arm_l_end_link]`` → MotionGen emits D=28 trajectories:
#     [ base_link(7) | arm_base_link(7) | R_ee(7) | L_ee(7) ]
# This matches G1/Vega's layout exactly, with the link mapping:
#     base_link       ↔ pelvis / vega_1p_base   (holonomic chassis)
#     arm_base_link   ↔ torso / arm_center      (shared pivot for both arms)
#     arm_r_end_link  ↔ right_palm / R_ee
#     arm_l_end_link  ↔ left_palm / L_ee
# So the preprocess / pcontroller logic is identical to Vega's 15-dim action.


Genie1RestPose7 = Tuple[float, float, float, float, float, float, float]


class Genie1PControllerHelper:
    """Genie1 P-controller helper (mirrors ``Vega1pSharpaPControllerHelper``).

    Accepts the same 15-dim G1-style input curobo emits with
    ``info_links=[base_link, arm_base_link, arm_r_end_link, arm_l_end_link]``:

        ``[ base_link(7) | arm_base_link(7) | lock_flag(1) ]``

    Returns a 4-dim output ``[x, y, heading, mode_flag]`` — the wheeled base
    only exposes 3 drive joints (``dummy_base_prismatic_x/y`` +
    ``dummy_base_revolute_z``) so ``p_controller_n_extra_dims = 0``. The
    ``arm_base_link`` block (columns 7–13) is parsed only for NaN-fallback
    bookkeeping; it does not drive base motion.

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

        Layout and lock_flag semantics identical to Vega's helper (see
        :class:`Vega1pSharpaPControllerHelper.preprocess`). ``mode_flag`` in
        ``{-2 skip, -1 lock_skip, 0 nav, 1 turning}`` is passed through, with
        ``0 → 1`` auto-upgrade when the base is within ``position_threshold``
        of its XY target (rotate in place instead of chasing residual error).
        """
        action = action.to(device)
        N = action.shape[0]

        current_pos = robot_state["base_pos"][env_ids]
        current_quat = robot_state["base_quat"][env_ids]
        current_xy = current_pos[:, :2]
        _, _, current_yaw = euler_xyz_from_quat(current_quat)

        nan_mask = torch.isnan(action).all(dim=1)

        lock_flag = action[:, 14]
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
        left_rest_pose: Genie1RestPose7 = (
            0.25,
            0.25,
            0.60,
            0.0,
            1.0,
            0.0,
            0.0,
        ),
        right_rest_pose: Genie1RestPose7 = (
            0.25,
            -0.25,
            0.60,
            0.0,
            1.0,
            0.0,
            0.0,
        ),
    ) -> torch.Tensor:
        return genie1_move_strategy(
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


def genie1_move_strategy(
    trajectory: torch.Tensor,
    robot_state: Dict[str, torch.Tensor],
    hand_id: int = -1,
    lock_xy_steps: int = 10,
    num_rotation_steps: int = 50,
    lock_fwd_offset: float = 0.12,
    lock_perp_offset: float = 0.3,
    yaw_axis_correction: float = 0.8,
    left_rest_pose: Genie1RestPose7 = (
        0.25,
        0.25,
        0.60,
        0.0,
        1.0,
        0.0,
        0.0,
    ),
    right_rest_pose: Genie1RestPose7 = (
        0.25,
        -0.25,
        0.60,
        0.0,
        1.0,
        0.0,
        0.0,
    ),
) -> torch.Tensor:
    """Genie1 wheeled-base move strategy — ported from ``vega1p_sharpa_move_strategy``.

    Trajectory layout (D=28):
        ``[ base_link(7) | arm_base_link(7) | R_ee(7) | L_ee(7) ]``

    Segments:
        1. Horizontal move — ``lock_flag = 0`` (nav). Base XY interpolates
           from the planned start to ``locked_xy`` (a perpendicular/forward
           hold pose near the final grasp); last ``lock_xy_steps`` frames
           pin XY to ``locked_xy``.
        2. Rotation padding — ``lock_flag = 1`` (turning). Holds
           ``locked_xy`` while the base rotates to the final target yaw.

    Genie1's base Z is constant (wheeled chassis, no squat), and torso
    yaw/pitch is tracked through the ``arm_base_link`` block of every
    waypoint — so there are no stand-up / squat segments.

    Rest poses for the inactive arm are expressed in the ``base_link``
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

    base_z_hold = float(current_base_pos[2].item())

    arm_dim = 14
    arm_start = D - arm_dim  # last 14 = right(7) + left(7)
    base_block_dim = arm_start  # base_link(7) + arm_base_link(7) = 14
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

        wp[2] = base_z_hold  # Genie1 base Z is constant
        wp[3:7] = trajectory[i, 3:7]
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


GENIE1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/genie1.usd",
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
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["dummy_base_.*"],
            effort_limit_sim=1000.0,
            stiffness=0.0,
            damping=1e5,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["idx0[12]_body_joint.*"],
            effort_limit_sim={
                "idx01_body_joint1": 100.0,
                "idx02_body_joint2": 100.0,
            },
            stiffness={
                "idx01_body_joint1": 5000.0,
                "idx02_body_joint2": 1000.0,
            },
            damping={
                "idx01_body_joint1": 300.0,
                "idx02_body_joint2": 100.0,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["idx1[12]_head_joint.*"],
            effort_limit_sim=50.0,
            stiffness=100.0,
            damping=10.0,
        ),
        # Arms 1–4 carry load; joints 5–7 are wrist DOFs with lower effort.
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["idx2._arm_l_joint.*", "idx6._arm_r_joint.*"],
            effort_limit_sim={
                "idx2[1-4]_arm_l_joint.*": 60.0,
                "idx2[5-7]_arm_l_joint.*": 30.0,
                "idx6[1-4]_arm_r_joint.*": 60.0,
                "idx6[5-7]_arm_r_joint.*": 30.0,
            },
            stiffness={
                "idx2[1-4]_arm_l_joint.*": 1000.0,
                "idx2[5-7]_arm_l_joint.*": 500.0,
                "idx6[1-4]_arm_r_joint.*": 1000.0,
                "idx6[5-7]_arm_r_joint.*": 500.0,
            },
            damping={
                "idx2[1-4]_arm_l_joint.*": 50.0,
                "idx2[5-7]_arm_l_joint.*": 25.0,
                "idx6[1-4]_arm_r_joint.*": 50.0,
                "idx6[5-7]_arm_r_joint.*": 25.0,
            },
        ),
        # 120s parallel-jaw gripper — same closed-loop topology as the Isaac
        # Lab ``Agibot A2D`` gripper. Primary driver (``outer_joint1``) closes
        # the jaw; two support joints (``outer_joint3``, ``inner_joint3``) are
        # pre-loaded at a low PD gain to keep the four-bar linkage taut; all
        # other finger joints are passive (mimic via URDF + PhysX joint
        # equality, so PD=0 avoids actuators fighting the closure).
        "gripper_primary": ImplicitActuatorCfg(
            joint_names_expr=[_L_GRIPPER_PRIMARY_REGEX, _R_GRIPPER_PRIMARY_REGEX],
            effort_limit_sim=10.0,
            stiffness=20.0,
            damping=0.5,
        ),
        "gripper_support": ImplicitActuatorCfg(
            joint_names_expr=[_L_GRIPPER_SUPPORT_REGEX, _R_GRIPPER_SUPPORT_REGEX],
            effort_limit_sim=2.0,
            stiffness=2.0,
            damping=0.05,
        ),
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=[_L_GRIPPER_PASSIVE_REGEX, _R_GRIPPER_PASSIVE_REGEX],
            effort_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
    articulation_root_prim_path=None,
)


# ================================
#  Pink IK controller configuration (dual arm)
# ================================
#
# Two ``LocalFrameTask`` targets (left/right end effectors in ``base_link``
# frame) plus a shared ``NullSpacePostureTask`` biasing the 16 pink-controlled
# joints (2 torso + 14 arms) toward their rest pose. The grippers are not
# part of this cspace — they're handled through ``eef_action`` below.
#
# The pink-IK URDF is the ``genie1_simple.urdf`` variant with grippers and
# loop joints stripped; ``pinocchio.model.nq`` matches IsaacLab's joint
# count because every remaining joint is prismatic or revolute with finite
# limits.

GENIE1_PINK_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="Robot_0",
    base_link_name="base_link",
    num_hand_joints=0,  # grippers handled by eef_action, not pink IK
    show_ik_warnings=True,
    urdf_path=f"{MAGICSIM_ASSETS}/Robots/URDF/genie1_simple.urdf",
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        LocalFrameTask(
            "arm_r_end_link",
            base_link_frame_name="base_link",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        LocalFrameTask(
            "arm_l_end_link",
            base_link_frame_name="base_link",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        NullSpacePostureTask(
            cost=0.5,
            lm_damping=1,
            controlled_frames=["arm_r_end_link", "arm_l_end_link"],
            controlled_joints=_PINK_CONTROLLED_JOINTS,
            gain=0.3,
        ),
        DampingTask(
            cost=0.8,
        ),
    ],
    fixed_input_tasks=[],
    amplify_factor=1.0,
)


# ================================
#  Action configuration
# ================================


@configclass
class Genie1ActionsCfg(MobileManipActionsCfg):
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
            # Dual-arm Pink IK. Per-env action layout:
            # ``[ right_ee pose(7) | left_ee pose(7) ]`` — gripper joints are
            # driven by ``eef_action`` (mirrors the G1/Vega pattern).
            "ik_pink": PinkInverseKinematicsActionCfg(
                pink_controlled_joint_names=_PINK_CONTROLLED_JOINTS,
                num_joints=_NUM_PINK_IK_JOINTS,
                hand_joint_names=None,
                target_eef_link_names={
                    "right_wrist": "arm_r_end_link",
                    "left_wrist": "arm_l_end_link",
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
                controller=GENIE1_PINK_IK_CONTROLLER_CFG,
                relative_to_base=False,
            ),
        },
        "eef_action": {
            # Direct joint-position control over the 16 parallel-jaw gripper
            # joints (8 left + 8 right). URDF mimic constraints keep the
            # inner/outer finger chains consistent when IsaacLab converts
            # them through PhysX.
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=[_L_GRIPPER_JOINT_REGEX, _R_GRIPPER_JOINT_REGEX],
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


class Genie1FrameCfg(FrameSensorCfg):
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
        self.target_frames[0].prim_path = self.robot_prim_path + "/arm_l_end_link"
        self.target_frames[1].prim_path = self.robot_prim_path + "/arm_r_end_link"
        self.target_frames[2].prim_path = self.robot_prim_path + "/base_link"
        self.prim_path = self.robot_prim_path + "/base_link"


# ================================
#  Planner configuration
# ================================


@configclass
class Genie1PlannerCfg(MobileManipPlannerCfg):
    max_eef_num: int = 2

    base_action_dim: Dict[str, int] = {
        "dwb_differential": 8,
        "dwb_holonomic": 8,
        "default": 8,
        # 15-dim G1-style input: [base_link(7), arm_base_link(7), lock_flag(1)]
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
        # 15-dim G1-style: [base(7), arm_base_link(7), lock_flag]. lock_flag
        # in {-2 skip, -1 lock_skip, 0 nav, 1 turning}.
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
    # Planner-side arm action shape: dual-EEF world-pose (14 dims = R(7) + L(7)).
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
    # Gripper action channel — 16 parallel-jaw joints as direct joint-pos.
    eef_action_dim: Dict[str, int] = {
        "joint_pos": _NUM_GRIPPER_JOINTS,
    }
    eef_action_space: Dict[str, torch.Tensor] = {
        "joint_pos": torch.tensor(
            [
                [-1.0] * _NUM_GRIPPER_JOINTS,
                [1.0] * _NUM_GRIPPER_JOINTS,
            ],
        ),
    }
    p_controller_helper = Genie1PControllerHelper
    # Wheeled base: 3 drive joints, no torso-velocity channel → n_extra_dims=0.
    p_controller_n_extra_dims: int = 0
    move_strategy = Genie1PControllerHelper.move_strategy
    move_strategy_distance_threshold: float = 0.1


# ================================
#  Observation configuration (dual EEF)
# ================================


@configclass
class Genie1ObsCfg(ObsGroup):
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
                "target_link_name": "arm_l_end_link",
            },
        )
        self.left_eef_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "arm_l_end_link",
            },
        )
        self.right_eef_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "arm_r_end_link",
            },
        )
        self.right_eef_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "arm_r_end_link",
            },
        )
        # Dual-link EEF in world frame (right first, then left — matches G1/Vega).
        self.eef_pos = ObsTerm(
            func=transforms_terms.get_dual_link_position_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "link_name_1": "arm_r_end_link",
                "link_name_2": "arm_l_end_link",
            },
        )
        self.eef_quat = ObsTerm(
            func=transforms_terms.get_dual_link_quaternion_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "link_name_1": "arm_r_end_link",
                "link_name_2": "arm_l_end_link",
            },
        )
        # EEF relative to base (for local-frame policies).
        self.left_eef_relative_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "arm_l_end_link",
                "target_frame_name": "base_link",
            },
        )
        self.left_eef_relative_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "arm_l_end_link",
                "target_frame_name": "base_link",
            },
        )
        self.right_eef_relative_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "arm_r_end_link",
                "target_frame_name": "base_link",
            },
        )
        self.right_eef_relative_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "arm_r_end_link",
                "target_frame_name": "base_link",
            },
        )
        self.base_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "base_link",
            },
        )
        self.base_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "base_link",
            },
        )
        self.base_ang_vel = ObsTerm(
            func=transforms_terms.get_target_link_ang_vel_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "base_link",
            },
        )
        self.base_lin_vel = ObsTerm(
            func=transforms_terms.get_target_link_lin_vel_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "base_link",
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
class Genie1Cfg(RobotCfg):
    """Configuration for the mobile dual-arm Genie1 (G1_120s) with parallel
    jaw grippers."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    base_action_name: str = "holonomic_action"
    arm_action_name: str = "ik_pink"
    eef_action_name: str | None = "joint_pos"
    frame_name: str = "ee_frame"

    action: Genie1ActionsCfg = MISSING
    ee_frame: Genie1FrameCfg = MISSING
    obs: RobotObsCfg = MISSING
    planner: Genie1PlannerCfg = Genie1PlannerCfg()

    def __post_init__(self):
        self.robot: ArticulationCfg = GENIE1_CFG

        self.robot.prim_path = self.prim_path
        self.action = Genie1ActionsCfg(
            asset_name=self.asset_name,
            base_action_name=self.base_action_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = Genie1FrameCfg(robot_prim_path=self.robot.prim_path)
        self.obs = Genie1ObsCfg(asset_name=self.asset_name, frame_name=self.frame_name)
        self.type = "mobilemanip"
