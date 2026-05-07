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
#
# ARX Lift2 URDF layout (see Assets/Robots/URDF/lift2.urdf):
#   joint1/2/3  continuous wheels (ignored at planner level)
#   joint4      prismatic lift (0..0.46m, body4 riser — analogue of vega torso)
#   joint5/6    head yaw/pitch
#   joint11..16 left arm revolute (6-DoF, R5a kinematics)
#   joint17/18  left gripper prismatic fingers (2-finger, binary)
#   joint21..26 right arm revolute (6-DoF, R5a kinematics)
#   joint27/28  right gripper prismatic fingers

_L_ARM_JOINTS = [f"joint{i}" for i in range(11, 17)]  # joint11..joint16
_R_ARM_JOINTS = [f"joint{i}" for i in range(21, 27)]  # joint21..joint26
_ARM_JOINTS = _L_ARM_JOINTS + _R_ARM_JOINTS  # 12 arm joints (6 L + 6 R)

_LIFT_JOINTS = ["joint4"]  # prismatic body4 lift
_HEAD_JOINTS = ["joint5", "joint6"]  # head yaw/pitch
_WHEEL_JOINTS_REGEX = "joint[1-3]"  # continuous wheels

_L_FINGER_JOINTS = ["joint17", "joint18"]
_R_FINGER_JOINTS = ["joint27", "joint28"]
_FINGER_JOINTS = _L_FINGER_JOINTS + _R_FINGER_JOINTS  # 4 gripper joints

# Pink IK controls lift + head + arms (1 + 2 + 12 = 15). The trajectory layout
# from curobo (info_links = [base_link, arm_center, R_ee, L_ee]) is D=28 with
# block layout [base(7) | arm_center(7) | R_ee(7) | L_ee(7)] — identical to
# vega_1p_sharpa, so the P-controller helper below is a near-verbatim port.
_PINK_CONTROLLED_JOINTS = _LIFT_JOINTS + _HEAD_JOINTS + _ARM_JOINTS  # 15
_NUM_ARM_JOINTS = len(_ARM_JOINTS)  # 12
_NUM_PINK_IK_JOINTS = len(_PINK_CONTROLLED_JOINTS)  # 15


Lift2RestPose7 = Tuple[float, float, float, float, float, float, float]


# ================================
#  P-controller helper (ported from Vega1pSharpaPControllerHelper)
# ================================


class Lift2PControllerHelper:
    """ARX Lift2 P-controller helper.

    Identical contract to :class:`Vega1pSharpaPControllerHelper`: consumes a
    15-dim G1-style input ``[base_link(7) | arm_center(7) | lock_flag(1)]``
    and returns ``[target_x, target_y, target_heading, mode_flag]``.

    The trajectory layout from curobo is
    ``[ base_link(7) | arm_center(7) | R_ee(7) | L_ee(7) ]`` (D=28) because
    ``info_links`` in ``magicsim_lift2_mobile.yml`` mirrors vega_1p_sharpa's
    ordering. Only ``base_link`` drives the wheels; ``arm_center`` is parsed
    for NaN-fallback bookkeeping.
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
        left_rest_pose: Lift2RestPose7 = (
            0.25,
            0.25,
            0.60,
            0.0,
            1.0,
            0.0,
            0.0,
        ),
        right_rest_pose: Lift2RestPose7 = (
            0.25,
            -0.25,
            0.60,
            0.0,
            1.0,
            0.0,
            0.0,
        ),
    ) -> torch.Tensor:
        return lift2_move_strategy(
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


def lift2_move_strategy(
    trajectory: torch.Tensor,
    robot_state: Dict[str, torch.Tensor],
    hand_id: int = -1,
    lock_xy_steps: int = 10,
    num_rotation_steps: int = 50,
    lock_fwd_offset: float = 0.12,
    lock_perp_offset: float = 0.3,
    yaw_axis_correction: float = 0.8,
    left_rest_pose: Lift2RestPose7 = (
        0.25,
        0.25,
        0.60,
        0.0,
        1.0,
        0.0,
        0.0,
    ),
    right_rest_pose: Lift2RestPose7 = (
        0.25,
        -0.25,
        0.60,
        0.0,
        1.0,
        0.0,
        0.0,
    ),
) -> torch.Tensor:
    """Lift2 move strategy (ported verbatim from ``vega1p_sharpa_move_strategy``).

    Trajectory layout (D=28): ``[base_link(7) | arm_center(7) | R_ee(7) | L_ee(7)]``.

    No stand-up / squat — Lift2's base Z is held by the holonomic mobile base.
    Segment structure and rest-pose semantics match vega's (see
    :func:`vega1p_sharpa_move_strategy`).
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
    arm_start = D - arm_dim
    base_block_dim = arm_start  # base_link(7) + arm_center(7) = 14
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

    segments = []
    num_move_points = trajectory.shape[0]

    for i in range(num_move_points):
        flag = 0.0  # nav
        wp = _make_waypoint(flag)
        wp[:7] = trajectory[i, :7]
        wp[2] = base_z_hold
        # Carry arm_center block through (rows 7..14) so posture is tracked.
        if base_block_dim >= 14:
            wp[7:14] = trajectory[i, 7:14]
        segments.append(wp)

    # Terminal rotation-padding (lock_flag = 1) to align yaw once XY converges.
    if num_rotation_steps > 0 and trajectory.shape[0] > 0:
        for _ in range(num_rotation_steps):
            wp = _make_waypoint(1.0)
            wp[:7] = target_base_pose
            wp[2] = base_z_hold
            segments.append(wp)

    if len(segments) == 0:
        wp = _make_waypoint(-1.0)
        wp[:7] = target_base_pose
        wp[2] = base_z_hold
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
            # Left active — keep right (first 7) from plan, compute left rest.
            result[-1, arm_start : arm_start + 7] = tr_last[:7]
            p_w, q_w = _base_frame_rest_to_world(last_base_pose, left_rest_t)
            result[-1, arm_start + 7 : arm_start + 10] = p_w
            result[-1, arm_start + 10 : arm_start + 14] = q_w
        else:
            # Right active — compute right rest from base frame, keep left from plan.
            p_w, q_w = _base_frame_rest_to_world(last_base_pose, right_rest_t)
            result[-1, arm_start : arm_start + 3] = p_w
            result[-1, arm_start + 3 : arm_start + 7] = q_w
            result[-1, arm_start + 7 : arm_start + 14] = tr_last[7:14]

    return result


# ================================
#  Articulation configuration
# ================================


LIFT2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/lift2.usd",
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
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[_WHEEL_JOINTS_REGEX],
            effort_limit_sim=17.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "lift": ImplicitActuatorCfg(
            joint_names_expr=_LIFT_JOINTS,
            effort_limit_sim=400.0,
            stiffness=5000.0,
            damping=300.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=_HEAD_JOINTS,
            effort_limit_sim=7.0,
            stiffness=200.0,
            damping=10.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=_ARM_JOINTS,
            effort_limit_sim=27.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=_FINGER_JOINTS,
            effort_limit_sim=300.0,
            stiffness=400.0,
            damping=80.0,
        ),
    },
    articulation_root_prim_path=None,
)


# ================================
#  Pink IK controller configuration (dual arm)
# ================================

LIFT2_PINK_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="Robot_0",
    base_link_name="base_link",
    num_hand_joints=0,
    show_ik_warnings=True,
    urdf_path=f"{MAGICSIM_ASSETS}/Robots/URDF/lift2.urdf",
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        LocalFrameTask(
            "R_ee",
            base_link_frame_name="base_link",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        LocalFrameTask(
            "L_ee",
            base_link_frame_name="base_link",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        NullSpacePostureTask(
            cost=0.5,
            lm_damping=1,
            controlled_frames=["R_ee", "L_ee"],
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
class Lift2ActionsCfg(MobileManipActionsCfg):
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
                controller=LIFT2_PINK_IK_CONTROLLER_CFG,
                relative_to_base=False,
            ),
        },
        "eef_action": {
            # Binary dual-gripper: one action dim per gripper, mirrored to both
            # fingers. Finger closed at 0.0m, open at 0.044m (URDF upper limit).
            "binary": mdp.BinaryJointPositionActionCfg(
                joint_names=_FINGER_JOINTS,
                open_command_expr={j: 0.044 for j in _FINGER_JOINTS},
                close_command_expr={j: 0.0 for j in _FINGER_JOINTS},
            ),
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=_FINGER_JOINTS,
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


class Lift2FrameCfg(FrameSensorCfg):
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
        self.target_frames[2].prim_path = self.robot_prim_path + "/base_link"
        self.prim_path = self.robot_prim_path + "/base_link"


# ================================
#  Planner configuration
# ================================


@configclass
class Lift2PlannerCfg(MobileManipPlannerCfg):
    max_eef_num: int = 2

    base_action_dim: Dict[str, int] = {
        "dwb_differential": 8,
        "dwb_holonomic": 8,
        "default": 8,
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
    # Binary eef: one bit per gripper (2 dims).
    eef_action_dim: Dict[str, int] = {
        "binary": 2,
        "joint_pos": 4,
    }
    eef_action_space: Dict[str, torch.Tensor] = {
        "binary": torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
            ],
        ),
        "joint_pos": torch.tensor(
            [
                [-1.0] * 4,
                [1.0] * 4,
            ],
        ),
    }
    p_controller_helper = Lift2PControllerHelper
    p_controller_n_extra_dims: int = 0
    move_strategy = Lift2PControllerHelper.move_strategy
    move_strategy_distance_threshold: float = 0.1


# ================================
#  Observation configuration
# ================================


@configclass
class Lift2ObsCfg(ObsGroup):
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
        self.left_eef_relative_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "L_ee",
                "target_frame_name": "base_link",
            },
        )
        self.left_eef_relative_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "L_ee",
                "target_frame_name": "base_link",
            },
        )
        self.right_eef_relative_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "R_ee",
                "target_frame_name": "base_link",
            },
        )
        self.right_eef_relative_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "R_ee",
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
class Lift2Cfg(RobotCfg):
    """Configuration for the ARX Lift2 mobile dual-arm robot."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    base_action_name: str = "holonomic_action"
    arm_action_name: str = "ik_pink"
    eef_action_name: str | None = "binary"
    frame_name: str = "ee_frame"

    action: Lift2ActionsCfg = MISSING
    ee_frame: Lift2FrameCfg = MISSING
    obs: RobotObsCfg = MISSING
    planner: Lift2PlannerCfg = Lift2PlannerCfg()

    def __post_init__(self):
        self.robot: ArticulationCfg = LIFT2_CFG

        self.robot.prim_path = self.prim_path
        self.action = Lift2ActionsCfg(
            asset_name=self.asset_name,
            base_action_name=self.base_action_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = Lift2FrameCfg(robot_prim_path=self.robot.prim_path)
        self.obs = Lift2ObsCfg(asset_name=self.asset_name, frame_name=self.frame_name)
        self.type = "mobilemanip"
