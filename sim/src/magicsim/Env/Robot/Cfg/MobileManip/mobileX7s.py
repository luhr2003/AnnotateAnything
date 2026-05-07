"""Mobile X7S: 3-DOF holonomic floating base + dual 7-DOF arms + dual 2-DOF grippers.

Joint layout (see Assets/Robots/URDF/x7s_mobile/x7s_mobile.urdf):

* ``dummy_base_prismatic_x_joint`` / ``dummy_base_prismatic_y_joint`` /
  ``dummy_base_revolute_z_joint`` — dummy holonomic base (ridgeback-style).
* ``joint1`` — torso prismatic lift.
* ``joint2`` — torso/shoulder rotate.
* ``joint3`` / ``joint4`` — head/gimbal sub-chain (not part of either arm).
* ``joint5..joint11`` — left arm (7 DOF), EEF tracked at wrist flange ``link11``.
  (joint5 attaches to ``link2`` at ``y=+0.1424`` ⇒ +Y / left side under ROS REP-103.)
* ``joint12`` / ``joint13`` — left gripper fingers (prismatic).
* ``joint14..joint20`` — right arm (7 DOF), EEF tracked at wrist flange ``link20``.
  (joint14 attaches to ``link2`` at ``y=-0.0336`` ⇒ -Y / right side.)
* ``joint21`` / ``joint22`` — right gripper fingers (prismatic).
"""

from typing import Dict, Tuple
import torch
from dataclasses import MISSING

from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply
from magicsim.Env.Planner.Utils import quat_mul
import isaaclab.sim as sim_utils

from magicsim import MAGICSIM_ASSETS
from magicsim.Env.Robot.Cfg.Manipulator.Manipulator import FrameSensorCfg
from magicsim.Env.Robot.Cfg.MobileManip.MobileManip import (
    MobileManipActionsCfg,
    MobileManipPlannerCfg,
)
from magicsim.Env.Robot.Cfg.Base import RobotCfg, RobotObsCfg
from magicsim.Env.Robot.mdp.pink_ik import (
    DampingTask,
    LocalFrameTask,
    NullSpacePostureTask,
)
from magicsim.Env.Robot.mdp.pink_actions_cfg import (
    PinkIKControllerCfg,
    PinkInverseKinematicsActionCfg,
)
from magicsim.Env.Robot.terms import transforms as transforms_terms
import magicsim.Env.Robot.mdp as mdp


# ================================
#  Joint name constants
# ================================

_BASE_JOINTS = [
    "dummy_base_prismatic_x_joint",
    "dummy_base_prismatic_y_joint",
    "dummy_base_revolute_z_joint",
]
_TORSO_JOINTS = ["joint1", "joint2"]
_HEAD_JOINTS = ["joint3", "joint4"]
# Physical right arm = -Y branch of link2 (joint14..joint20, link14..link20).
# Physical left  arm = +Y branch of link2 (joint5..joint11,  link5..link11).
_R_ARM_JOINTS = [f"joint{i}" for i in range(14, 21)]  # joint14..joint20
_R_GRIPPER_JOINTS = ["joint21", "joint22"]
_L_ARM_JOINTS = [f"joint{i}" for i in range(5, 12)]  # joint5..joint11
_L_GRIPPER_JOINTS = ["joint12", "joint13"]

_ARM_JOINTS = _R_ARM_JOINTS + _L_ARM_JOINTS  # 14
_GRIPPER_JOINTS = _R_GRIPPER_JOINTS + _L_GRIPPER_JOINTS  # 4
_ALL_NON_BASE_JOINTS = (
    _TORSO_JOINTS + _HEAD_JOINTS + _ARM_JOINTS + _GRIPPER_JOINTS
)  # 2 + 2 + 14 + 4 = 22
_NUM_ALL_NON_BASE_JOINTS = len(_ALL_NON_BASE_JOINTS)

# Pink IK plans over torso (joint1..2) + both arms (joint5..11, joint14..20).
# Grippers and head are excluded: fingers are driven via eef_action and the
# head sub-chain is not part of either manipulator.
_PINK_CONTROLLED_JOINTS = _TORSO_JOINTS + _ARM_JOINTS  # 2 + 14 = 16
_NUM_PINK_IK_JOINTS = len(_PINK_CONTROLLED_JOINTS)

R_EE_LINK = "link20_tip"
L_EE_LINK = "link11_tip"
BASE_LINK = "base_link"
# ``link2`` is the shared torso→arms junction (both arms branch from it via
# joint5 / joint14). It plays the role of ``arm_center`` in vega and
# ``torso`` in G1 — the second block of the D=28 curobo trajectory.
ARM_CENTER_LINK = "link2"


# ================================
#  P-controller helper (holonomic base + torso via arm_center)
# ================================
#
# Mirrors :class:`Vega1pSharpaPControllerHelper` 1:1. The curobo yamls list
# ``info_links = [base_link, link2, link20_tip, link11_tip]`` so MotionGen
# emits D=28 trajectories with block layout:
#     [ base_link(7) | link2(7) | link20_tip(7) | link11_tip(7) ]
# Link mapping to vega / G1:
#     base_link    ↔ vega_1p_base ↔ pelvis       (holonomic mobile base)
#     link2        ↔ arm_center   ↔ torso        (shared torso→arms junction)
#     link20_tip   ↔ R_ee         ↔ right_palm   (physical right, joint14..20)
#     link11_tip   ↔ L_ee         ↔ left_palm    (physical left,  joint5..11)
#
# The pcontroller input is therefore the **same 15-dim G1-style layout**:
#     [ base_link_pose(7) | link2_pose(7) | lock_flag(1) ]
# and the 4-dim output ``[x, y, heading, mode_flag]`` is identical (the
# wheeled base has no WBC / torso-velocity channel, so
# ``p_controller_n_extra_dims = 0``).


Vega1pSharpaRestPose7 = Tuple[float, float, float, float, float, float, float]


class MobileX7sPControllerHelper:
    """Mobile X7S P-controller helper — 15-dim vega-aligned preprocess.

    Accepts the same 15-dim G1-style input curobo emits for the x7s
    trajectory (``info_links = [base_link, link2, link20_tip, link11_tip]``):

        ``[ base_link(7) | link2(7) | lock_flag(1) ]``

    but returns a **4-dim output** ``[x, y, heading, mode_flag]`` because the
    wheeled base only drives 3 joints (``dummy_base_prismatic_x/y`` +
    ``dummy_base_revolute_z``) — there is no WBC / torso-velocity channel to
    pass through. The arm_center block (columns 7–13) is parsed only for
    NaN-fallback bookkeeping; it does not drive base motion.

    Stateful NaN fallbacks mirror vega's helper so ``MobileMoveL`` IK-wait
    rows (all-NaN) resolve to ``lock_skip`` with the last valid base pose.
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

        Input ``action[:, :7]`` = base_link pose (xyz + wxyz).
        Input ``action[:, 7:14]`` = link2 (arm_center) pose (ignored for
        base motion; retained only so curobo's trajectory layout matches).
        Input ``action[:, 14]`` = lock_flag,
        ``{-2 skip, -1 lock_skip, 0 nav, 1 turning}``.

        mode_flag rules:

        * ``-2 / -1`` — pass through (skip / lock_skip).
        * ``1`` — pass through (upstream ``move_strategy`` requested turning).
        * ``0`` — nav; **upgrade to ``1`` (turning) when the base is already
          within ``position_threshold`` of the target** (mirrors
          ``RidgebackFrankaPControllerHelper``).
        * All-NaN row → forced ``-1`` (lock_skip) with last-target / current
          pose fallback.
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
        return mobile_x7s_move_strategy(
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


def mobile_x7s_move_strategy(
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
    """Mobile X7S move strategy — 1:1 port of ``vega1p_sharpa_move_strategy``.

    Trajectory layout (D=28):
        ``[ base_link(7) | link2(7) | link20_tip(7) | link11_tip(7) ]``
        i.e. slot 0 = physical right (link20_tip), slot 1 = physical left
        (link11_tip). This matches the move_strategy convention where
        ``hand_id == 0`` shifts the base to the robot's left so the right
        arm can naturally extend to the grasp point.

    Segments:
        1. Horizontal move — ``lock_flag = 0`` (nav). Base XY interpolates from
           the planned start to ``locked_xy`` (a perpendicular/forward-offset
           hold pose near the final grasp); last ``lock_xy_steps`` frames pin
           XY to ``locked_xy``.
        2. Rotation padding (optional) — ``lock_flag = 1`` (turning). Holds
           ``locked_xy`` while the base rotates to the final target yaw.

    No stand-up / squat segments: x7s base Z is fixed (holonomic mobile
    base) and the torso lift/rotate (``joint1`` / ``joint2``) is plumbed
    through the ``link2`` block of every waypoint — same role as vega's
    ``arm_center``.

    Rest poses for the inactive arm are expressed in the ``base_link``
    frame (analogue of vega's ``vega_1p_base`` frame).
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
    base_block_dim = arm_start  # base_link(7) + link2(7) = 14
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

        wp[2] = base_z_hold  # x7s base Z is constant
        wp[3:7] = trajectory[i, 3:7]  # keep planned base orientation
        _fill_eef_height_relative_to_base(wp, trajectory, i, wp[0:7])
        segments.append(wp)

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
#  Articulation configuration (URDF spawn)
# ================================

MOBILE_X7S_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/x7s_mobile.usd",
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
            joint_names_expr=["joint1", "joint2"],
            effort_limit_sim=700.0,
            stiffness=5000.0,
            damping=250.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["joint3", "joint4"],
            effort_limit_sim=100.0,
            stiffness=2000.0,
            damping=100.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                r"joint([5-9]|10|11|1[4-9]|20)",
            ],
            effort_limit_sim=300.0,
            stiffness=5000.0,
            damping=250.0,
        ),
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=[r"joint(12|13|21|22)"],
            effort_limit_sim=100.0,
            stiffness=2000.0,
            damping=100.0,
        ),
    },
    articulation_root_prim_path=None,
)


# ================================
#  Pink IK controller configuration (dual arm)
# ================================
#
# Mirrors ``VEGA_1P_SHARPA_PINK_IK_CONTROLLER_CFG``: two ``LocalFrameTask``
# targets for the left/right wrist tips in the mobile base frame, plus a shared
# ``NullSpacePostureTask`` biasing the torso + 14 arm joints. The simplified
# URDF at ``Assets/Robots/URDF/x7s_mobile_fix.urdf`` strips visuals/collisions
# and freezes the three ``dummy_base_*`` joints as fixed so pinocchio sees a
# locked base.

X7S_PINK_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="Robot_0",
    base_link_name=BASE_LINK,
    num_hand_joints=0,
    show_ik_warnings=True,
    urdf_path=f"{MAGICSIM_ASSETS}/Robots/URDF/x7s_mobile_fix.urdf",
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        LocalFrameTask(
            R_EE_LINK,
            base_link_frame_name=BASE_LINK,
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        LocalFrameTask(
            L_EE_LINK,
            base_link_frame_name=BASE_LINK,
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        NullSpacePostureTask(
            cost=0.5,
            lm_damping=1,
            controlled_frames=[R_EE_LINK, L_EE_LINK],
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
class MobileX7sActionsCfg(MobileManipActionsCfg):
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
            # Joint-position control over torso + head + both arms + both
            # grippers (22 joints). The reacher auto test drives this directly.
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=_ALL_NON_BASE_JOINTS,
            ),
            # Dual-arm Pink IK over torso (joint1..2) + 14 arm joints. Per-env
            # layout: ``[ right_wrist pose (7) | left_wrist pose (7) ]``.
            # Gripper fingers are driven through ``eef_action``.
            "ik_pink": PinkInverseKinematicsActionCfg(
                pink_controlled_joint_names=_PINK_CONTROLLED_JOINTS,
                num_joints=_NUM_PINK_IK_JOINTS,
                hand_joint_names=None,
                target_eef_link_names={
                    "right_wrist": R_EE_LINK,
                    "left_wrist": L_EE_LINK,
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
                controller=X7S_PINK_IK_CONTROLLER_CFG,
                relative_to_base=False,
            ),
        },
        "eef_action": {
            # Dedicated gripper channel so high-level code can open/close the
            # hands without touching the arm joints.
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=_GRIPPER_JOINTS,
            ),
            # Dual-hand binary open/close: [right, left] in {0, 1}. Mirrors
            # DualPiper's pattern — one BinaryJointActionCfg per hand,
            # grouped under MultipleBinaryJointPositionActionCfg. Prismatic
            # finger joints travel in [0, 0.044]; 0.044 = open, 0.0 = close.
            "binary": mdp.MultipleBinaryJointPositionActionCfg(
                joint_groups=[
                    mdp.BinaryJointActionCfg(
                        joint_names=_R_GRIPPER_JOINTS,
                        open_command_expr={"joint21": 0.044, "joint22": 0.044},
                        close_command_expr={"joint21": 0.0, "joint22": 0.0},
                    ),
                    mdp.BinaryJointActionCfg(
                        joint_names=_L_GRIPPER_JOINTS,
                        open_command_expr={"joint12": 0.044, "joint13": 0.044},
                        close_command_expr={"joint12": 0.0, "joint13": 0.0},
                    ),
                ],
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


class MobileX7sFrameCfg(FrameSensorCfg):
    def __post_init__(self):
        self.target_frames = [
            FrameTransformerCfg.FrameCfg(
                name="left_end_effector", offset=OffsetCfg(pos=[0.0, 0.0, 0.0])
            ),
            FrameTransformerCfg.FrameCfg(
                name="right_end_effector", offset=OffsetCfg(pos=[0.0, 0.0, 0.0])
            ),
            FrameTransformerCfg.FrameCfg(name="arm_base", offset=OffsetCfg()),
        ]
        self.visualizer_cfg.prim_path = (
            self.robot_prim_path + "/Visuals/FrameTransformer"
        )
        self.target_frames[0].prim_path = self.robot_prim_path + f"/{L_EE_LINK}"
        self.target_frames[1].prim_path = self.robot_prim_path + f"/{R_EE_LINK}"
        self.target_frames[2].prim_path = self.robot_prim_path + f"/{BASE_LINK}"
        self.prim_path = self.robot_prim_path + f"/{BASE_LINK}"


# ================================
#  Planner configuration
# ================================


@configclass
class MobileX7sPlannerCfg(MobileManipPlannerCfg):
    max_eef_num: int = 2

    base_action_dim: Dict[str, int] = {
        "dwb_differential": 8,
        "dwb_holonomic": 8,
        "default": 8,
        # 15-dim vega-style input: [base_link(7), link2(7), lock_flag(1)].
        # The curobo yaml lists info_links = [base_link, link2, link20_tip,
        # link11_tip] so MotionGen emits D=28 trajectories; the first 14 dims
        # feed the p-controller (base + arm_center) and the lock_flag comes
        # from the move_strategy segment metadata.
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
        # 15-dim: [base(7), link2(7), lock_flag]. lock_flag ∈
        # {-2 skip, -1 lock_skip, 0 nav, 1 turning} — mirrors vega / G1.
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
    # Planner-side arm action shape. For the curobo/IK planning path both
    # EEFs are tracked as world poses (14 = right(7) + left(7)); the 22 Isaac
    # Lab joint-position targets only show up at the action-term boundary
    # (see ``MobileX7sActionsCfg.available_action['arm_action']['joint_pos']``).
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
    # Dual gripper: 4 finger joints total (right joint12/13 + left joint21/22).
    eef_action_dim: Dict[str, int] = {
        "joint_pos": len(_GRIPPER_JOINTS),
        "binary": 2,
    }
    eef_action_space: Dict[str, torch.Tensor] = {
        "joint_pos": torch.tensor(
            [
                [-1.0] * len(_GRIPPER_JOINTS),
                [1.0] * len(_GRIPPER_JOINTS),
            ],
        ),
        # Dual-hand binary open/close: [right_close, left_close] in {0, 1}.
        "binary": torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
            ],
        ),
    }
    p_controller_helper = MobileX7sPControllerHelper
    # Holonomic base only drives 3 joints (x/y/yaw); no torso/WBC channel, so
    # preprocess output is plain [x, y, heading, mode_flag].
    p_controller_n_extra_dims: int = 0
    move_strategy = MobileX7sPControllerHelper.move_strategy
    move_strategy_distance_threshold: float = 0.1


# ================================
#  Observation configuration (dual EEF)
# ================================


@configclass
class MobileX7sObsCfg(ObsGroup):
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
        self.left_eef_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": L_EE_LINK,
            },
        )
        self.left_eef_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": L_EE_LINK,
            },
        )
        self.right_eef_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": R_EE_LINK,
            },
        )
        self.right_eef_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": R_EE_LINK,
            },
        )
        self.eef_pos = ObsTerm(
            func=transforms_terms.get_dual_link_position_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "link_name_1": R_EE_LINK,
                "link_name_2": L_EE_LINK,
            },
        )
        self.eef_quat = ObsTerm(
            func=transforms_terms.get_dual_link_quaternion_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "link_name_1": R_EE_LINK,
                "link_name_2": L_EE_LINK,
            },
        )
        self.left_eef_relative_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": L_EE_LINK,
                "target_frame_name": BASE_LINK,
            },
        )
        self.left_eef_relative_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": L_EE_LINK,
                "target_frame_name": BASE_LINK,
            },
        )
        self.right_eef_relative_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": R_EE_LINK,
                "target_frame_name": BASE_LINK,
            },
        )
        self.right_eef_relative_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": R_EE_LINK,
                "target_frame_name": BASE_LINK,
            },
        )
        self.base_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": BASE_LINK,
            },
        )
        self.base_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": BASE_LINK,
            },
        )
        self.base_ang_vel = ObsTerm(
            func=transforms_terms.get_target_link_ang_vel_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": BASE_LINK,
            },
        )
        self.base_lin_vel = ObsTerm(
            func=transforms_terms.get_target_link_lin_vel_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": BASE_LINK,
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
class MobileX7sCfg(RobotCfg):
    """Configuration for the mobile dual-arm X7S."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    base_action_name: str = "holonomic_action"
    arm_action_name: str = "joint_pos"
    # Default eef channel is dual-hand binary open/close (one bit per hand)
    # — matches DualPiper / DualSO101 / DualArxX5 convention. Override to
    # ``"joint_pos"`` to drive the 4 finger joints directly.
    eef_action_name: str | None = "binary"
    frame_name: str = "ee_frame"

    action: MobileX7sActionsCfg = MISSING
    ee_frame: MobileX7sFrameCfg = MISSING
    obs: RobotObsCfg = MISSING
    planner: MobileX7sPlannerCfg = MobileX7sPlannerCfg()

    def __post_init__(self):
        self.robot: ArticulationCfg = MOBILE_X7S_CFG
        self.robot.prim_path = self.prim_path
        self.action = MobileX7sActionsCfg(
            asset_name=self.asset_name,
            base_action_name=self.base_action_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = MobileX7sFrameCfg(robot_prim_path=self.robot.prim_path)
        self.obs = MobileX7sObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        self.type = "mobilemanip"
