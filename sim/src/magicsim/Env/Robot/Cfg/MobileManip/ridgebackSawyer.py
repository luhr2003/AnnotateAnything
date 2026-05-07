from typing import Dict
import torch
from dataclasses import MISSING

from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat
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
from pink.tasks import FrameTask
from magicsim.Env.Robot.mdp.pink_ik import NullSpacePostureTask
from magicsim.Env.Robot.mdp.pink_actions_cfg import (
    PinkIKControllerCfg,
    PinkInverseKinematicsActionCfg,
)
from magicsim.Env.Utils.transforms import quat_inverse, quat_mul


class RidgebackSawyerPControllerHelper:
    """
    RidgebackSawyer P-controller preprocess helper (stateful).

    Mirrors :class:`RidgebackFrankaPControllerHelper` exactly: 8-dim base action
    ``[x, y, z, qw, qx, qy, qz, lock_flag]`` is mapped to the 4-dim P-controller
    command ``[target_x, target_y, target_heading, mode_flag]`` as a pure 1:1
    pass-through of ``lock_flag`` ∈ ``{-1, 0}`` from ``MobileMoveL``. All-NaN
    rows (IK wait) force ``lock_flag = -1`` and the target pose falls back
    through the cached ``_last_target_*`` buffers to the current base pose.
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
        """Preprocess 8-dim base action to 4-dim P-controller format (stateful)."""
        return ridgebacksawyer_preprocess_p_controller_action(
            action,
            robot_state,
            env_ids,
            device=device,
            last_target_xy=self._last_target_xy,
            last_target_yaw=self._last_target_yaw,
        )

    @staticmethod
    def move_strategy(
        trajectory: torch.Tensor,
        robot_state: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Stateless wrapper around :func:`ridgebacksawyer_move_strategy`."""
        return ridgebacksawyer_move_strategy(trajectory, robot_state)

    def reset_idx(self, env_ids) -> None:
        """Clear cached last target for reset environments."""
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        self._last_target_xy[env_ids] = float("nan")
        self._last_target_yaw[env_ids] = float("nan")


def ridgebacksawyer_preprocess_p_controller_action(
    action: torch.Tensor,
    robot_state: Dict,
    env_ids: torch.Tensor,
    device: torch.device = torch.device("cuda:0"),
    last_target_xy: "torch.Tensor | None" = None,
    last_target_yaw: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Preprocess 8-dim RidgebackSawyer base action to 4-dim P-controller format.

    1:1 pass-through of ``lock_flag`` produced by ``MobileMoveL``
    (``_LOCK_FLAG_LOCK_SKIP=-1``, ``_LOCK_FLAG_NAV=0``). Stateful NaN fallbacks
    fall back first to the cached last target and only then to the current
    base pose. Buffers are updated in place at the end of the call.

    Input:
        action: [N, 8] with ``[x, y, z, qw, qx, qy, qz, lock_flag]``.
            * ``lock_flag = -1``: lock_skip (hold, base is locked).
            * ``lock_flag = 0`` : nav (free base).
            * All-NaN row: treated as ``lock_flag = -1`` with cached/current
              pose fallback.

    Returns:
        [N, 4] tensor ``[target_x, target_y, target_heading, mode_flag]`` where
        ``mode_flag`` is either ``-1`` (lock_skip) or ``0`` (nav), with an
        upgrade to ``1`` (turning-in-place) once close enough in xy.
    """
    action = action.to(device)
    N = action.shape[0]

    current_pos = robot_state["base_pos"][env_ids]  # [N, 3]
    current_quat = robot_state["base_quat"][env_ids]  # [N, 4] (w, x, y, z)
    current_xy = current_pos[:, :2]  # [N, 2]
    _, _, current_yaw = euler_xyz_from_quat(current_quat)

    nan_mask = torch.isnan(action).all(dim=1)  # [N]

    target_x = action[:, 0]
    target_y = action[:, 1]

    # Extract yaw from quaternion (qw, qx, qy, qz).
    # Remove Ry(-90°) offset between arm base and base_link in the URDF.
    quat = action[:, 3:7].clone()  # [N, 4] (w, x, y, z)
    nan_quat_mask = torch.isnan(quat).any(dim=1)
    if torch.any(nan_quat_mask):
        quat[nan_quat_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    q_offset = torch.tensor([0.70711, 0.0, -0.70711, 0.0], device=device)
    q_offset_inv = quat_inverse(q_offset).unsqueeze(0).expand_as(quat)
    quat = torch.as_tensor(
        quat_mul(q_offset_inv, quat), dtype=torch.float32, device=device
    )
    _, _, target_yaw = euler_xyz_from_quat(quat)

    if torch.any(nan_mask):
        if last_target_xy is not None:
            last_xy = last_target_xy[env_ids].to(device=device)
            last_xy_nan = torch.isnan(last_xy).any(dim=1, keepdim=True).expand(-1, 2)
            xy_fallback = torch.where(last_xy_nan, current_xy, last_xy)
        else:
            xy_fallback = current_xy
        if last_target_yaw is not None:
            last_yaw = last_target_yaw[env_ids].to(device=device)
            yaw_fallback = torch.where(torch.isnan(last_yaw), current_yaw, last_yaw)
        else:
            yaw_fallback = current_yaw

        target_x = torch.where(nan_mask, xy_fallback[:, 0], target_x)
        target_y = torch.where(nan_mask, xy_fallback[:, 1], target_y)
        target_yaw = torch.where(nan_mask, yaw_fallback, target_yaw)

    lock_flag = action[:, 7]
    lock_flag = torch.where(nan_mask, torch.tensor(-1.0, device=device), lock_flag)

    position_threshold = 0.1
    lock_skip_mask = lock_flag < -0.5
    mode_flag = torch.zeros(N, device=device)
    mode_flag[lock_skip_mask] = -1.0

    dxy = torch.stack([target_x - current_xy[:, 0], target_y - current_xy[:, 1]], dim=1)
    dist_to_target = torch.norm(dxy, dim=1)
    close_mask = (~lock_skip_mask) & (dist_to_target < position_threshold)
    mode_flag[close_mask] = 1.0

    target_heading = target_yaw

    result = torch.stack([target_x, target_y, target_heading, mode_flag], dim=1)

    if last_target_xy is not None:
        last_target_xy[env_ids] = torch.stack([target_x, target_y], dim=1).to(
            device=last_target_xy.device, dtype=last_target_xy.dtype
        )
    if last_target_yaw is not None:
        last_target_yaw[env_ids] = target_yaw.to(
            device=last_target_yaw.device, dtype=last_target_yaw.dtype
        )
    return result


def ridgebacksawyer_move_strategy(
    trajectory: torch.Tensor,
    robot_state: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Convert a MotionGen trajectory to a movement trajectory for RidgebackSawyer.

    Mirrors :func:`ridgebackfranka_move_strategy`: keep the EEF in a retracted
    pending pose in front of the base while following base waypoints.

    Args:
        trajectory: [N, D] tensor from MotionGen where first 7 dims are base pose
                   (x, y, z, qw, qx, qy, qz) and remaining dims are EEF poses.
        robot_state: Dict containing 'base_pos', 'base_quat', etc.

    Returns:
        [M, D+1] tensor with trajectory. Last dim is lock_flag (0 or -1).
    """
    device = trajectory.device
    dtype = trajectory.dtype
    D = trajectory.shape[1]

    if trajectory.shape[0] == 0:
        return torch.zeros(0, D + 1, device=device, dtype=dtype)

    eef_offset = torch.tensor(
        [0.4, 0.0, 0.8, 0.0, 1.0, 0.0, 0.0], device=device, dtype=dtype
    )

    def _make_waypoint(lock_flag: float) -> torch.Tensor:
        wp = torch.full((D + 1,), float("nan"), device=device, dtype=dtype)
        wp[-1] = lock_flag
        return wp

    def _fill_eef_pose(wp: torch.Tensor) -> None:
        if D >= 14:
            wp[7:10] = wp[0:3] + eef_offset[:3]
            wp[10:14] = eef_offset[3:]

    segments = []
    num_move_points = trajectory.shape[0]

    for i in range(num_move_points):
        flag = -1.0 if i == num_move_points - 1 else 0.0
        wp = _make_waypoint(flag)
        wp[:7] = trajectory[i, :7]
        _fill_eef_pose(wp)
        segments.append(wp)

    if len(segments) == 0:
        wp = _make_waypoint(-1.0)
        wp[:7] = trajectory[-1, :7]
        _fill_eef_pose(wp)
        return wp.unsqueeze(0)

    result = torch.stack(segments, dim=0)
    assert result.shape[1] == D + 1, f"result.shape: {result.shape}"

    return result


RIDGEBACK_SAWYER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/ridgeback_sawyer.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "dummy_base_prismatic_y_joint": 0.0,
            "dummy_base_prismatic_x_joint": 0.0,
            "dummy_base_revolute_z_joint": 0.0,
            "head_pan": 0.0,
            "right_j0": 0.0,
            "right_j1": -0.785,
            "right_j2": 0.0,
            "right_j3": 1.04,
            "right_j4": 0.0,
            "right_j5": 1.309,
            "right_j6": 0.0,
            "right_gripper_l_finger_joint": 0.0,
            "right_gripper_r_finger_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["dummy_base_.*"],
            effort_limit_sim=1000.0,
            stiffness=0.0,
            damping=1e5,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_pan"],
            effort_limit_sim=8.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "sawyer_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_j[0-6]"],
            effort_limit_sim=80.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "sawyer_hand": ImplicitActuatorCfg(
            joint_names_expr=["right_gripper_.*_finger_joint"],
            effort_limit_sim=20.0,
            stiffness=400.0,
            damping=80.0,
        ),
    },
    articulation_root_prim_path="/base_link",
)


RIDGEBACK_SAWYER_PINK_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="Robot_0",
    base_link_name="right_arm_base_link",
    num_hand_joints=0,
    show_ik_warnings=True,
    urdf_path=f"{MAGICSIM_ASSETS}/Robots/URDF/sawyer.urdf",
    fail_on_joint_limit_violation=False,
    all_joint_names=["right_j.*"],
    variable_input_tasks=[
        FrameTask(
            "right_l6",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        NullSpacePostureTask(
            cost=0.5,
            lm_damping=1,
            controlled_frames=["right_l6"],
            controlled_joints=[
                "right_j0",
                "right_j1",
                "right_j2",
                "right_j3",
                "right_j4",
                "right_j5",
                "right_j6",
            ],
            gain=0.3,
        ),
    ],
    fixed_input_tasks=[],
    amplify_factor=1.0,
)


# ================================
#  Action Configuration
# ================================


@configclass
class RidgebackSawyerActionsCfg(MobileManipActionsCfg):
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
                joint_names=["right_j.*"],
            ),
            "ik_abs": mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["right_j.*"],
                body_name="right_l6",
                command_reference_body_name="right_arm_base_link",
                controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=False,
                    ik_method="dls",
                    ik_params={"lambda_val": 0.05},
                ),
                body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
                    pos=[0.0, 0.0, 0],
                ),
                action_space=torch.tensor(
                    [
                        [-0.6, -0.6, -1.0, -1.0, -1.0, -1.0, -1.0],
                        [0.6, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0],
                    ]
                ),
            ),
            "ik_pink": PinkInverseKinematicsActionCfg(
                pink_controlled_joint_names=["right_j.*"],
                num_joints=7,
                hand_joint_names=None,
                target_eef_link_names={"eef": "right_l6"},
                action_space=torch.tensor(
                    [
                        [-0.6, -0.6, 0.0, -1.0, -1.0, -1.0, -1.0],
                        [0.6, 0.6, 1.2, 1.0, 1.0, 1.0, 1.0],
                    ]
                ),
                controller=RIDGEBACK_SAWYER_PINK_IK_CONTROLLER_CFG,
                relative_to_base=False,
            ),
        },
        "eef_action": {
            "binary": mdp.BinaryJointPositionActionCfg(
                joint_names=["right_gripper_.*_finger_joint"],
                open_command_expr={"right_gripper_.*_finger_joint": 0.02},
                close_command_expr={"right_gripper_.*_finger_joint": 0.0},
            ),
        },
    }

    def __post_init__(self):
        if self.arm_action_name is not None:
            self.arm_action = self.available_action["arm_action"][self.arm_action_name]
            self.arm_action.asset_name = self.asset_name
        else:
            self.arm_action = None

        if (
            self.eef_action_name is not None
            and self.eef_action_name in self.available_action["eef_action"]
        ):
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
#  Frame Configuration
# ================================


class RidgebackSawyerFrameCfg(FrameSensorCfg):
    def __post_init__(self):
        self.target_frames = [
            FrameTransformerCfg.FrameCfg(
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
            FrameTransformerCfg.FrameCfg(name="arm_base", offset=OffsetCfg()),
        ]
        self.visualizer_cfg.prim_path = (
            self.robot_prim_path + "/Visuals/FrameTransformer"
        )
        self.target_frames[0].prim_path = self.robot_prim_path + "/right_l6"
        self.target_frames[1].prim_path = self.robot_prim_path + "/right_arm_base_link"
        self.prim_path = self.robot_prim_path + "/base_link"


# ================================
#  Planner Configuration
# ================================


@configclass
class RidgebackSawyerPlannerCfg(MobileManipPlannerCfg):
    base_action_dim: Dict[str, int] = {
        "dwb_differential": 8,
        "dwb_holonomic": 8,
        "default": 8,
        "p_controller": 8,
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
        # "default": torch.tensor(
        #     [
        #         [-100, -100, 0, -1, -1, -1, -1, -1],
        #         [100, 100, 0, 1, 1, 1, 1, 0],
        #     ],
        # ),
        "p_controller": torch.tensor(
            [
                [-100, -100, 0, -1, -1, -1, -1, -1],
                [100, 100, 0, 1, 1, 1, 1, 0],
            ],
        ),
    }
    arm_action_dim: Dict[str, int] = {
        "default": 7,
    }
    arm_action_space: Dict[str, torch.Tensor] = {
        "default": torch.tensor(
            [
                [-1] * 7,
                [1] * 7,
            ],
        ),
    }
    eef_action_dim: Dict[str, int] = {
        "default": 1,
    }
    eef_action_space: Dict[str, torch.Tensor] = {
        "default": torch.tensor([0, 1]),
    }
    # P-controller preprocessor: class with preprocess, reset_idx
    p_controller_helper = RidgebackSawyerPControllerHelper
    p_controller_n_extra_dims: int = 0
    move_strategy = RidgebackSawyerPControllerHelper.move_strategy
    # Distance threshold for base movement to trigger move strategy
    move_strategy_distance_threshold: float = 0.1


# ================================
#  Observation Configuration
# ================================


@configclass
class RidgebackSawyerObsCfg(ObsGroup):
    asset_name: str = MISSING
    frame_name: str = MISSING
    joint_pos: ObsTerm = MISSING
    joint_vel: ObsTerm = MISSING
    joint_effort: ObsTerm = MISSING
    eef_pos: ObsTerm = MISSING
    eef_quat: ObsTerm = MISSING
    gripper_pos: ObsTerm = MISSING
    eef_relative_pos: ObsTerm = MISSING
    eef_relative_quat: ObsTerm = MISSING
    base_pos: ObsTerm = MISSING
    base_quat: ObsTerm = MISSING
    eef_relatvie_pos_arm_base: ObsTerm = MISSING
    base_ang_vel: ObsTerm = MISSING
    base_lin_vel: ObsTerm = MISSING

    def __post_init__(self):
        self.joint_pos = ObsTerm(
            func=transforms_terms.joint_pos_with_root_offset,
            params={
                "asset_cfg": SceneEntityCfg(self.asset_name),
                "x_joint_name": "dummy_base_prismatic_x_joint",
                "y_joint_name": "dummy_base_prismatic_y_joint",
                "yaw_joint_name": "dummy_base_revolute_z_joint",
            },
        )

        self.joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg(self.asset_name)},
        )
        self.joint_effort = ObsTerm(
            func=mdp.joint_effort, params={"asset_cfg": SceneEntityCfg(self.asset_name)}
        )

        self.eef_pos = ObsTerm(
            func=mdp.ee_frame_pos,
            params={"ee_frame_cfg": SceneEntityCfg(self.frame_name)},
        )
        self.eef_quat = ObsTerm(
            func=mdp.ee_frame_quat,
            params={"ee_frame_cfg": SceneEntityCfg(self.frame_name)},
        )
        self.gripper_pos = ObsTerm(
            func=mdp.gripper_pos, params={"robot_cfg": SceneEntityCfg(self.asset_name)}
        )
        self.eef_relative_pos = ObsTerm(
            func=mdp.ee_rel_pos,
            params={"ee_frame_cfg": SceneEntityCfg(self.frame_name)},
        )
        self.eef_relative_quat = ObsTerm(
            func=mdp.ee_rel_quat,
            params={"ee_frame_cfg": SceneEntityCfg(self.frame_name)},
        )
        self.base_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(self.asset_name),
                "target_link_name": "base_link",
            },
        )
        self.base_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(self.asset_name),
                "target_link_name": "base_link",
            },
        )
        self.eef_relatvie_pos_arm_base = ObsTerm(
            func=mdp.ee_rel_pos_arm_base,
            params={"ee_frame_cfg": SceneEntityCfg(self.frame_name)},
        )
        self.base_ang_vel = ObsTerm(
            func=transforms_terms.get_target_link_ang_vel_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(self.asset_name),
                "target_link_name": "base_link",
            },
        )
        self.base_lin_vel = ObsTerm(
            func=transforms_terms.get_target_link_lin_vel_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(self.asset_name),
                "target_link_name": "base_link",
            },
        )
        self.enable_corruption = False
        self.concatenate_terms = False
        del self.asset_name
        del self.frame_name


# ================================
#  Composite Robot Configuration
# ================================


@configclass
class RidgebackSawyerCfg(RobotCfg):
    """Configuration for a mobile manipulator: Ridgeback + Sawyer"""

    prim_path: str = MISSING
    asset_name: str = "robot"
    base_action_name: str = "differential_drive"
    arm_action_name: str = "joint_pos"
    eef_action_name: str = "binary"
    frame_name: str = "ee_frame"

    # Subconfigs
    action: RidgebackSawyerActionsCfg = MISSING
    ee_frame: RidgebackSawyerFrameCfg = MISSING
    obs: RobotObsCfg = MISSING
    planner: RidgebackSawyerPlannerCfg = RidgebackSawyerPlannerCfg()

    def __post_init__(self):
        self.robot: ArticulationCfg = RIDGEBACK_SAWYER_CFG

        self.robot.prim_path = self.prim_path
        self.action = RidgebackSawyerActionsCfg(
            asset_name=self.asset_name,
            base_action_name=self.base_action_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )

        self.ee_frame = RidgebackSawyerFrameCfg(robot_prim_path=self.robot.prim_path)

        self.obs = RidgebackSawyerObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        self.type = "mobilemanip"
