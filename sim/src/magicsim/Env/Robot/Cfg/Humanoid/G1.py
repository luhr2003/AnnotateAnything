from typing import Dict, Tuple

from isaaclab.utils.math import quat_apply
from magicsim.Env.Planner.Utils import quat_mul

# Inactive-hand rest in **pelvis frame**: ``(x,y,z, qw,qx,qy,qz)``; mapped to world with
# ``p_w = p + quat_apply(q, p_rest)``, ``q_w = quat_mul(q, q_rest)``.
G1EefRestPose7 = Tuple[float, float, float, float, float, float, float]

from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from magicsim.Env.Robot import mdp
from magicsim.Env.Robot.Cfg.Base import RobotObsCfg
import torch
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from dataclasses import MISSING
from magicsim.Env.Robot.terms import transforms as transforms_terms
from magicsim.Env.Robot.Cfg.Humanoid.Humanoid import (
    HumanoidActionsCfg,
    HumanoidCfg,
    HumanoidPlannerCfg,
)
from magicsim.Env.Robot.Cfg.Humanoid.mdp.homie_wbc_action_cfg import (
    HomieWBCActionCfg,
)
from magicsim.Env.Robot.mdp.actions_cfg import (
    JointPositionToLimitsActionCfg,
    JointPositionVelocityToLimitsActionCfg,
    MultipleJointPositionToLimitsActionCfg,
    MultipleJointPositionToLimitsActionGroupCfg,
)
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.camera.camera_cfg import CameraCfg
from magicsim.Env.Robot.mdp.pink_ik import (
    DampingTask,
    LocalFrameTask,
    NullSpacePostureTask,
)
from magicsim import MAGICSIM_ASSETS
from magicsim.Env.Robot.mdp.pink_actions_cfg import (
    PinkIKControllerCfg,
    PinkInverseKinematicsActionCfg,
)
from magicsim.Env.Robot.Cfg.Humanoid.mdp import g1_mdp
from magicsim.Env.Utils.rotations import (
    quat_to_rot_matrix,
    quat_to_euler_angles,
    matrix_to_euler_angles,
    euler_to_rot_matrix,
)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def g1_postprocess_p_controller_action(
    action: torch.Tensor,
    mode_flag: torch.Tensor,
    robot_state: Dict,
    env_ids: torch.Tensor,
    default_height: float = 0.7,
    min_height: float = 0.3,
    max_scale: float = 0.5,
) -> torch.Tensor:
    """Scale velocity components in action based on current height using exponential interpolation.

    When height >= default_height: no scaling (scale = 1.0).
    When height = min_height: velocities amplified by max_scale.
    Between min_height and default_height: exponential interpolation.

    Args:
        action: [N, 7] tensor [lin_vel_x, lin_vel_y, ang_vel, height, torso_roll, torso_pitch, torso_yaw]
        mode_flag: [N] **Output** mode from preprocess / PController (last column of 8-dim).
            Same meaning as ``PController`` (see ``PController.step``). Used here only to
            decide whether height-based scaling applies:

            * ``-2`` (skip) — **No** velocity scaling (scale = 1); output velocities are
              typically replaced upstream by ``last_command``.
            * ``-1`` (lock_skip) — **No** velocity scaling; base velocities are zero from PD.
            * ``0`` (nav) — **Yes**: scale ``lin_vel_*`` and ``ang_vel`` when squatting.
            * ``1`` (turning) — **Yes**: scale yaw rate when squatting.

            "Active" for scaling: ``mode_flag >= -0.5`` (i.e. nav and turning only).
        robot_state: Dict with robot state, must contain "base_pos" [num_envs, 3]
        env_ids: [N] environment indices to index into robot_state
        default_height: Default standing height (no scaling at this height)
        min_height: Minimum height (maximum scaling at this height)
        max_scale: Maximum velocity amplification factor at min_height

    Returns:
        [N, 7] action tensor with scaled velocity components (same layout as input).
    """
    import math

    active_mask = mode_flag >= -0.5  # nav (0) or turning (1)

    # Get current height from robot_state base_pos z-component
    current_height = robot_state["base_pos"][env_ids, 2]

    # Normalized ratio: 0 at default_height, 1 at min_height
    t = (default_height - current_height) / (default_height - min_height)
    t = torch.clamp(t, 0.0, 1.0)

    # Exponential interpolation: scale = exp(ln(max_scale) * t)
    # At t=0 (default height): scale = 1.0
    # At t=1 (min height): scale = max_scale
    scale = torch.exp(math.log(max_scale) * t)

    # Only apply to active environments, and ensure scale >= 1.0
    scale = torch.where(active_mask, scale, torch.ones_like(scale))
    scale = torch.maximum(scale, torch.ones_like(scale))

    # Scale velocity dims: lin_vel_x(0), lin_vel_y(1), ang_vel(2)
    result = action.clone()
    result[:, 0] = action[:, 0] * scale
    result[:, 1] = action[:, 1] * scale
    result[:, 2] = action[:, 2] * scale
    return result


# Pink IK Controller Configuration for G1
G1_PINK_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="Robot_0",
    base_link_name="pelvis",
    num_hand_joints=0,  # currently no hand joints (if needed, should change this value)
    show_ik_warnings=True,
    urdf_path=f"{MAGICSIM_ASSETS}/Robots/URDF/g1.urdf",
    fail_on_joint_limit_violation=False,  # if fails, go to find suboptimal/closest solution
    variable_input_tasks=[
        LocalFrameTask(
            "right_hand_palm_link",
            base_link_frame_name="pelvis_contour_link",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        LocalFrameTask(
            "left_hand_palm_link",
            base_link_frame_name="pelvis_contour_link",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        NullSpacePostureTask(
            cost=0.5,
            lm_damping=1,
            controlled_frames=[
                "left_hand_palm_link",
                "right_hand_palm_link",
            ],
            controlled_joints=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
            ],
            gain=0.3,
        ),
        DampingTask(
            cost=0.8,  # [cost] * [s] / [rad], 增加这个值会让速度更小
        ),
    ],
    fixed_input_tasks=[],
)


@configclass
class G1HeadCameraCfg(TiledCameraCfg):
    robot_prim_path: str = MISSING
    offset = CameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0), rot=(1, 0.0, 0.0, 0.0), convention="world"
    )
    # Explicit list[str]; nested lists break TiledCamera._check_supported_data_types (set(cfg.data_types)).
    data_types: list[str] = ["rgb", "depth"]
    spawn = sim_utils.PinholeCameraCfg(
        focal_length=1.88,
        focus_distance=0.5,
        horizontal_aperture=2.6035,
        vertical_aperture=1.4621,
        clipping_range=(0.1, 20.0),
    )
    width = 1920
    height = 1080

    def __post_init__(self):
        self.prim_path = f"{self.robot_prim_path}/d435_link/ego_camera"
        super().__post_init__()


# Joint order and ImplicitActuatorCfg fields (stiffness, damping, armature, friction, effort/velocity
# sim limits) match isaac_playground ``g1_scene_cfg`` (Doorman ``tasks/locomanip/g1_scene_cfg.py``).

G1_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hand_index_0_joint",
    "left_hand_middle_0_joint",
    "left_hand_thumb_0_joint",
    "right_hand_index_0_joint",
    "right_hand_middle_0_joint",
    "right_hand_thumb_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_1_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_thumb_2_joint",
]

_MAGIC_G1_ACTUATOR_STIFFNESS = {
    "left_hip_pitch_joint": 150,
    "left_hip_roll_joint": 150,
    "left_hip_yaw_joint": 150,
    "left_knee_joint": 200,
    "left_ankle_pitch_joint": 40,
    "left_ankle_roll_joint": 40,
    "right_hip_pitch_joint": 150,
    "right_hip_roll_joint": 150,
    "right_hip_yaw_joint": 150,
    "right_knee_joint": 200,
    "right_ankle_pitch_joint": 40,
    "right_ankle_roll_joint": 40,
    "waist_yaw_joint": 1000,
    "waist_roll_joint": 1000,
    "waist_pitch_joint": 1000,
    "left_shoulder_pitch_joint": 400,
    "left_shoulder_roll_joint": 400,
    "left_shoulder_yaw_joint": 160,
    "left_elbow_joint": 160,
    "left_wrist_roll_joint": 80,
    "left_wrist_pitch_joint": 80,
    "left_wrist_yaw_joint": 80,
    "right_shoulder_pitch_joint": 400,
    "right_shoulder_roll_joint": 400,
    "right_shoulder_yaw_joint": 160,
    "right_elbow_joint": 160,
    "right_wrist_roll_joint": 80,
    "right_wrist_pitch_joint": 80,
    "right_wrist_yaw_joint": 80,
    "left_hand_index_0_joint": 20.0,
    "left_hand_middle_0_joint": 20.0,
    "left_hand_thumb_0_joint": 20.0,
    "right_hand_index_0_joint": 20.0,
    "right_hand_middle_0_joint": 20.0,
    "right_hand_thumb_0_joint": 20.0,
    "left_hand_index_1_joint": 20.0,
    "left_hand_middle_1_joint": 20.0,
    "left_hand_thumb_1_joint": 20.0,
    "right_hand_index_1_joint": 20.0,
    "right_hand_middle_1_joint": 20.0,
    "right_hand_thumb_1_joint": 20.0,
    "left_hand_thumb_2_joint": 20.0,
    "right_hand_thumb_2_joint": 20.0,
}

_MAGIC_G1_ACTUATOR_DAMPING = {
    "left_hip_pitch_joint": 2,
    "left_hip_roll_joint": 2,
    "left_hip_yaw_joint": 2,
    "left_knee_joint": 4,
    "left_ankle_pitch_joint": 2,
    "left_ankle_roll_joint": 2,
    "right_hip_pitch_joint": 2,
    "right_hip_roll_joint": 2,
    "right_hip_yaw_joint": 2,
    "right_knee_joint": 4,
    "right_ankle_pitch_joint": 2,
    "right_ankle_roll_joint": 2,
    "waist_yaw_joint": 10,
    "waist_roll_joint": 10,
    "waist_pitch_joint": 10,
    "left_shoulder_pitch_joint": 10,
    "left_shoulder_roll_joint": 10,
    "left_shoulder_yaw_joint": 4,
    "left_elbow_joint": 4,
    "left_wrist_roll_joint": 4,
    "left_wrist_pitch_joint": 4,
    "left_wrist_yaw_joint": 4,
    "right_shoulder_pitch_joint": 10,
    "right_shoulder_roll_joint": 10,
    "right_shoulder_yaw_joint": 4,
    "right_elbow_joint": 4,
    "right_wrist_roll_joint": 4,
    "right_wrist_pitch_joint": 4,
    "right_wrist_yaw_joint": 4,
    "left_hand_index_0_joint": 2.0,
    "left_hand_middle_0_joint": 2.0,
    "left_hand_thumb_0_joint": 2.0,
    "right_hand_index_0_joint": 2.0,
    "right_hand_middle_0_joint": 2.0,
    "right_hand_thumb_0_joint": 2.0,
    "left_hand_index_1_joint": 2.0,
    "left_hand_middle_1_joint": 2.0,
    "left_hand_thumb_1_joint": 2.0,
    "right_hand_index_1_joint": 2.0,
    "right_hand_middle_1_joint": 2.0,
    "right_hand_thumb_1_joint": 2.0,
    "left_hand_thumb_2_joint": 2.0,
    "right_hand_thumb_2_joint": 2.0,
}

_MAGIC_G1_ACTUATOR_ARMATURE = {
    "left_hip_pitch_joint": 0.01017752004,
    "left_hip_roll_joint": 0.025101925,
    "left_hip_yaw_joint": 0.01017752004,
    "left_knee_joint": 0.025101925,
    "left_ankle_pitch_joint": 0.00721945,
    "left_ankle_roll_joint": 0.00721945,
    "right_hip_pitch_joint": 0.01017752004,
    "right_hip_roll_joint": 0.025101925,
    "right_hip_yaw_joint": 0.01017752004,
    "right_knee_joint": 0.025101925,
    "right_ankle_pitch_joint": 0.00721945,
    "right_ankle_roll_joint": 0.00721945,
    "waist_yaw_joint": 0.01017752004,
    "waist_roll_joint": 0.00721945,
    "waist_pitch_joint": 0.00721945,
    "left_shoulder_pitch_joint": 0.003609725,
    "left_shoulder_roll_joint": 0.003609725,
    "left_shoulder_yaw_joint": 0.003609725,
    "left_elbow_joint": 0.003609725,
    "left_wrist_roll_joint": 0.003609725,
    "left_wrist_pitch_joint": 0.00425,
    "left_wrist_yaw_joint": 0.00425,
    "right_shoulder_pitch_joint": 0.003609725,
    "right_shoulder_roll_joint": 0.003609725,
    "right_shoulder_yaw_joint": 0.003609725,
    "right_elbow_joint": 0.003609725,
    "right_wrist_roll_joint": 0.003609725,
    "right_wrist_pitch_joint": 0.00425,
    "right_wrist_yaw_joint": 0.00425,
    "left_hand_index_0_joint": 0.01,
    "left_hand_middle_0_joint": 0.01,
    "left_hand_thumb_0_joint": 0.01,
    "right_hand_index_0_joint": 0.01,
    "right_hand_middle_0_joint": 0.01,
    "right_hand_thumb_0_joint": 0.01,
    "left_hand_index_1_joint": 0.01,
    "left_hand_middle_1_joint": 0.01,
    "left_hand_thumb_1_joint": 0.01,
    "right_hand_index_1_joint": 0.01,
    "right_hand_middle_1_joint": 0.01,
    "right_hand_thumb_1_joint": 0.01,
    "left_hand_thumb_2_joint": 0.01,
    "right_hand_thumb_2_joint": 0.01,
}

_MAGIC_G1_ACTUATOR_FRICTION = {
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0,
    "left_ankle_pitch_joint": 0.0,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.0,
    "right_ankle_pitch_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "left_hand_index_0_joint": 0.0,
    "left_hand_middle_0_joint": 0.0,
    "left_hand_thumb_0_joint": 0.0,
    "right_hand_index_0_joint": 0.0,
    "right_hand_middle_0_joint": 0.0,
    "right_hand_thumb_0_joint": 0.0,
    "left_hand_index_1_joint": 0.0,
    "left_hand_middle_1_joint": 0.0,
    "left_hand_thumb_1_joint": 0.0,
    "right_hand_index_1_joint": 0.0,
    "right_hand_middle_1_joint": 0.0,
    "right_hand_thumb_1_joint": 0.0,
    "left_hand_thumb_2_joint": 0.0,
    "right_hand_thumb_2_joint": 0.0,
}

_MAGIC_G1_ACTUATOR_EFFORT = {
    "left_hip_pitch_joint": 88.0,
    "left_hip_roll_joint": 139.0,
    "left_hip_yaw_joint": 88.0,
    "left_knee_joint": 139.0,
    "left_ankle_pitch_joint": 35.0,
    "left_ankle_roll_joint": 35.0,
    "right_hip_pitch_joint": 88.0,
    "right_hip_roll_joint": 139.0,
    "right_hip_yaw_joint": 88.0,
    "right_knee_joint": 139.0,
    "right_ankle_pitch_joint": 35.0,
    "right_ankle_roll_joint": 35.0,
    "waist_yaw_joint": 352.0,
    "waist_roll_joint": 140.0,
    "waist_pitch_joint": 140.0,
    "left_shoulder_pitch_joint": 100.0,
    "left_shoulder_roll_joint": 100.0,
    "left_shoulder_yaw_joint": 100.0,
    "left_elbow_joint": 100.0,
    "left_wrist_roll_joint": 100.0,
    "left_wrist_pitch_joint": 100.0,
    "left_wrist_yaw_joint": 100.0,
    "right_shoulder_pitch_joint": 100.0,
    "right_shoulder_roll_joint": 100.0,
    "right_shoulder_yaw_joint": 100.0,
    "right_elbow_joint": 100.0,
    "right_wrist_roll_joint": 100.0,
    "right_wrist_pitch_joint": 100.0,
    "right_wrist_yaw_joint": 100.0,
    "left_hand_index_0_joint": 300.0,
    "left_hand_middle_0_joint": 300.0,
    "left_hand_thumb_0_joint": 300.0,
    "right_hand_index_0_joint": 300.0,
    "right_hand_middle_0_joint": 300.0,
    "right_hand_thumb_0_joint": 300.0,
    "left_hand_index_1_joint": 300.0,
    "left_hand_middle_1_joint": 300.0,
    "left_hand_thumb_1_joint": 300.0,
    "right_hand_index_1_joint": 300.0,
    "right_hand_middle_1_joint": 300.0,
    "right_hand_thumb_1_joint": 300.0,
    "left_hand_thumb_2_joint": 300.0,
    "right_hand_thumb_2_joint": 300.0,
}

_MAGIC_G1_ACTUATOR_VEL = {
    "left_hip_pitch_joint": 32.0,
    "left_hip_roll_joint": 20.0,
    "left_hip_yaw_joint": 32.0,
    "left_knee_joint": 20.0,
    "left_ankle_pitch_joint": 30.0,
    "left_ankle_roll_joint": 30.0,
    "right_hip_pitch_joint": 32.0,
    "right_hip_roll_joint": 20.0,
    "right_hip_yaw_joint": 32.0,
    "right_knee_joint": 20.0,
    "right_ankle_pitch_joint": 30.0,
    "right_ankle_roll_joint": 30.0,
    "waist_yaw_joint": 32.0,
    "waist_roll_joint": 30.0,
    "waist_pitch_joint": 30.0,
    "left_shoulder_pitch_joint": 37.0,
    "left_shoulder_roll_joint": 37.0,
    "left_shoulder_yaw_joint": 37.0,
    "left_elbow_joint": 37.0,
    "left_wrist_roll_joint": 37.0,
    "left_wrist_pitch_joint": 22.0,
    "left_wrist_yaw_joint": 22.0,
    "right_shoulder_pitch_joint": 37.0,
    "right_shoulder_roll_joint": 37.0,
    "right_shoulder_yaw_joint": 37.0,
    "right_elbow_joint": 37.0,
    "right_wrist_roll_joint": 37.0,
    "right_wrist_pitch_joint": 22.0,
    "right_wrist_yaw_joint": 22.0,
    "left_hand_index_0_joint": 100.0,
    "left_hand_middle_0_joint": 100.0,
    "left_hand_thumb_0_joint": 100.0,
    "right_hand_index_0_joint": 100.0,
    "right_hand_middle_0_joint": 100.0,
    "right_hand_thumb_0_joint": 100.0,
    "left_hand_index_1_joint": 100.0,
    "left_hand_middle_1_joint": 100.0,
    "left_hand_thumb_1_joint": 100.0,
    "right_hand_index_1_joint": 100.0,
    "right_hand_middle_1_joint": 100.0,
    "right_hand_thumb_1_joint": 100.0,
    "left_hand_thumb_2_joint": 100.0,
    "right_hand_thumb_2_joint": 100.0,
}

MAGIC_G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/g1_new.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    prim_path="/World/envs/env_.*/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.8, -1.38, 0.78),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            # "left_shoulder_roll_joint": 0.0,
            "left_shoulder_roll_joint": 0.785,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            # "right_shoulder_roll_joint": 0.0,
            "right_shoulder_roll_joint": -0.785,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=G1_DOF_NAMES,
            stiffness=_MAGIC_G1_ACTUATOR_STIFFNESS,
            damping=_MAGIC_G1_ACTUATOR_DAMPING,
            armature=_MAGIC_G1_ACTUATOR_ARMATURE,
            friction=_MAGIC_G1_ACTUATOR_FRICTION,
            effort_limit_sim=_MAGIC_G1_ACTUATOR_EFFORT,
            velocity_limit_sim=_MAGIC_G1_ACTUATOR_VEL,
        ),
    },
)


@configclass
class G1ActionsCfg(HumanoidActionsCfg):
    """Action specifications for the MDP."""

    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "base_action": {
            "wbc": HomieWBCActionCfg(
                joint_names=[
                    ".*_hip_.*_joint",
                    ".*_knee_joint",
                    ".*_ankle_.*_joint",
                    "waist_.*_joint",
                ],
                num_wbc_joints=15,
                action_space=torch.tensor(
                    [
                        [
                            -0.8,
                            -0.8,
                            -0.4,
                            0.2,
                            -torch.pi / 2,
                            -torch.pi / 2,
                            -torch.pi / 2,
                        ],
                        [
                            0.8,
                            0.8,
                            0.4,
                            1.0,
                            torch.pi / 2,
                            torch.pi / 2,
                            torch.pi / 2,
                        ],
                    ]
                ),
                wbc_joint_yaml_path="/home/magics/magicsim/MagicSim/src/magicsim/Env/Robot/Cfg/Humanoid/mdp/G1/homie_wbc.yaml",
                decimation=4,
            ),
            "joint_pos": JointPositionToLimitsActionCfg(
                joint_names=[
                    ".*_hip_.*_joint",
                    ".*_knee_joint",
                    ".*_ankle_.*_joint",
                    "waist_.*_joint",
                ],
                num_joints=15,
            ),
        },
        "arm_action": {
            "joint_pos": JointPositionToLimitsActionCfg(
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                    "waist_.*_joint",
                ],
                num_joints=17,
            ),
            "joint_pos_vel": JointPositionVelocityToLimitsActionCfg(
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    "waist_.*_joint",
                ],
                num_joints=17,
            ),
            "ik_abs": PinkInverseKinematicsActionCfg(
                pink_controlled_joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_yaw_joint",
                    "waist_.*_joint",
                ],
                num_joints=17,
                hand_joint_names=None,
                target_eef_link_names={
                    "right_wrist": "right_hand_palm_link",
                    "left_wrist": "left_hand_palm_link",
                },
                action_space=torch.tensor(
                    [
                        # Lower limits
                        [
                            # Right wrist pose (xyz, xyzw) - broad workspace limits
                            0.2,
                            -0.6,
                            0.4,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            # Left wrist pose (xyz, xyzw)
                            0.2,
                            -0.6,
                            0.4,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                        ],
                        # Upper limits
                        [
                            # Right wrist pose
                            0.8,
                            0.6,
                            1.2,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            # Left wrist pose
                            0.8,
                            0.6,
                            1.2,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                    ]
                ),
                controller=G1_PINK_IK_CONTROLLER_CFG,
                relative_to_base=False,
                fallback_to_current=False,
                decimation=4,
            ),
        },
        "eef_action": {
            "interpolated": mdp.MultipleInterpolatedJointChoicePositionActionCfg(
                joint_groups=[
                    mdp.InterpolatedJointChoiceActionCfg(
                        joint_names=[
                            "left_hand_index_0_joint",
                            "left_hand_index_1_joint",
                            "left_hand_middle_0_joint",
                            "left_hand_middle_1_joint",
                            "left_hand_thumb_0_joint",
                            "left_hand_thumb_1_joint",
                            "left_hand_thumb_2_joint",
                        ],
                        open_command_expr={
                            "left_hand_index_0_joint": 0.0,
                            "left_hand_index_1_joint": 0.0,
                            "left_hand_middle_0_joint": 0.0,
                            "left_hand_middle_1_joint": 0.0,
                            "left_hand_thumb_0_joint": 0.0,
                            "left_hand_thumb_1_joint": 0.0,
                            "left_hand_thumb_2_joint": 0.0,
                        },
                        close_command_exprs=[
                            {  # wbc close
                                "left_hand_index_0_joint": -0.6,
                                "left_hand_index_1_joint": -1.2,
                                "left_hand_middle_0_joint": -0.6,
                                "left_hand_middle_1_joint": -1.2,
                                "left_hand_thumb_0_joint": 0.0,
                                "left_hand_thumb_1_joint": 0.7,
                                "left_hand_thumb_2_joint": 0.7,
                            },
                            {  # index close
                                "left_hand_index_0_joint": -1.5,
                                "left_hand_index_1_joint": -1.5,
                                "left_hand_middle_0_joint": -0.6,
                                "left_hand_middle_1_joint": -1.5,
                                "left_hand_thumb_0_joint": -0.5,
                                "left_hand_thumb_1_joint": 0.7,
                                "left_hand_thumb_2_joint": 0.7,
                            },
                            {  # middle close
                                "left_hand_index_0_joint": -1.0,
                                "left_hand_index_1_joint": -1.5,
                                "left_hand_middle_0_joint": -1.0,
                                "left_hand_middle_1_joint": -1.5,
                                "left_hand_thumb_0_joint": 0.0,
                                "left_hand_thumb_1_joint": 0.7,
                                "left_hand_thumb_2_joint": 0.7,
                            },
                            {  # ring close
                                "left_hand_index_0_joint": -0.6,
                                "left_hand_index_1_joint": -1.5,
                                "left_hand_middle_0_joint": -1.5,
                                "left_hand_middle_1_joint": -1.5,
                                "left_hand_thumb_0_joint": 0.5,
                                "left_hand_thumb_1_joint": 0.7,
                                "left_hand_thumb_2_joint": 0.7,
                            },
                        ],
                    ),
                    mdp.InterpolatedJointChoiceActionCfg(
                        joint_names=[
                            "right_hand_index_0_joint",
                            "right_hand_index_1_joint",
                            "right_hand_middle_0_joint",
                            "right_hand_middle_1_joint",
                            "right_hand_thumb_0_joint",
                            "right_hand_thumb_1_joint",
                            "right_hand_thumb_2_joint",
                        ],
                        open_command_expr={
                            "right_hand_index_0_joint": 0.0,
                            "right_hand_index_1_joint": 0.0,
                            "right_hand_middle_0_joint": 0.0,
                            "right_hand_middle_1_joint": 0.0,
                            "right_hand_thumb_0_joint": 0.0,
                            "right_hand_thumb_1_joint": 0.0,
                            "right_hand_thumb_2_joint": 0.0,
                        },
                        close_command_exprs=[
                            {  # wbc close (negated)
                                "right_hand_index_0_joint": 0.6,
                                "right_hand_index_1_joint": 1.2,
                                "right_hand_middle_0_joint": 0.6,
                                "right_hand_middle_1_joint": 1.2,
                                "right_hand_thumb_0_joint": 0.0,
                                "right_hand_thumb_1_joint": -0.7,
                                "right_hand_thumb_2_joint": -0.7,
                            },
                            {  # index close (negated)
                                "right_hand_index_0_joint": 1.5,
                                "right_hand_index_1_joint": 1.5,
                                "right_hand_middle_0_joint": 0.6,
                                "right_hand_middle_1_joint": 1.5,
                                "right_hand_thumb_0_joint": -0.5,
                                "right_hand_thumb_1_joint": -0.7,
                                "right_hand_thumb_2_joint": -0.7,
                            },
                            {  # middle close (negated)
                                "right_hand_index_0_joint": 1.0,
                                "right_hand_index_1_joint": 1.5,
                                "right_hand_middle_0_joint": 1.0,
                                "right_hand_middle_1_joint": 1.5,
                                "right_hand_thumb_0_joint": 0.0,
                                "right_hand_thumb_1_joint": -0.7,
                                "right_hand_thumb_2_joint": -0.7,
                            },
                            {  # ring close (negated)
                                "right_hand_index_0_joint": 0.6,
                                "right_hand_index_1_joint": 1.5,
                                "right_hand_middle_0_joint": 1.5,
                                "right_hand_middle_1_joint": 1.5,
                                "right_hand_thumb_0_joint": 0.5,
                                "right_hand_thumb_1_joint": -0.7,
                                "right_hand_thumb_2_joint": -0.7,
                            },
                        ],
                    ),
                ],
            ),
            "joint_pos": MultipleJointPositionToLimitsActionCfg(
                joint_groups=[
                    MultipleJointPositionToLimitsActionGroupCfg(
                        joint_names=[
                            "right_hand_index_0_joint",
                            "right_hand_index_1_joint",
                            "right_hand_middle_0_joint",
                            "right_hand_middle_1_joint",
                            "right_hand_thumb_0_joint",
                            "right_hand_thumb_1_joint",
                            "right_hand_thumb_2_joint",
                        ],
                        num_joints=7,
                        preserve_order=True,
                    ),
                    MultipleJointPositionToLimitsActionGroupCfg(
                        joint_names=[
                            "left_hand_index_0_joint",
                            "left_hand_index_1_joint",
                            "left_hand_middle_0_joint",
                            "left_hand_middle_1_joint",
                            "left_hand_thumb_0_joint",
                            "left_hand_thumb_1_joint",
                            "left_hand_thumb_2_joint",
                        ],
                        num_joints=7,
                        preserve_order=True,
                    ),
                ],
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()


@configclass
class G1ObsCfg(RobotObsCfg):
    sensor_name: str = MISSING
    left_eef_pos: ObsTerm = MISSING
    left_eef_quat: ObsTerm = MISSING
    right_eef_pos: ObsTerm = MISSING
    right_eef_quat: ObsTerm = MISSING
    pelvis_pos: ObsTerm = MISSING
    pelvis_quat: ObsTerm = MISSING
    torso_pos: ObsTerm = MISSING
    torso_quat: ObsTerm = MISSING
    base_ang_vel: ObsTerm = MISSING
    base_lin_vel: ObsTerm = MISSING
    pelvis_ang_vel: ObsTerm = MISSING
    pelvis_lin_vel: ObsTerm = MISSING
    torso_ang_vel: ObsTerm = MISSING
    torso_lin_vel: ObsTerm = MISSING
    # head_cam_rgb: ObsTerm = MISSING
    # head_cam_depth: ObsTerm = MISSING

    def __post_init__(self):
        asset_name = self.asset_name
        sensor_name = self.sensor_name
        super().__post_init__()
        # Mimic required observations (from G1WBCPinkObservationsCfg)
        self.base_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "pelvis",
            },
        )
        self.base_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "pelvis",
            },
        )
        self.left_eef_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "left_hand_palm_link",
            },
        )
        self.left_eef_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "left_hand_palm_link",
            },
        )
        self.left_eef_world_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "left_hand_palm_link",
            },
        )
        self.left_eef_world_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "left_hand_palm_link",
            },
        )
        self.right_eef_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "right_hand_palm_link",
            },
        )
        self.right_eef_world_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "right_hand_palm_link",
            },
        )
        self.eef_pos = ObsTerm(
            func=transforms_terms.get_dual_link_position_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "link_name_1": "right_hand_palm_link",
                "link_name_2": "left_hand_palm_link",
            },
        )
        self.eef_quat = ObsTerm(
            func=transforms_terms.get_dual_link_quaternion_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "link_name_1": "right_hand_palm_link",
                "link_name_2": "left_hand_palm_link",
            },
        )
        self.right_eef_world_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_world_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "right_hand_palm_link",
            },
        )
        self.right_eef_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_target_frame,
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "right_hand_palm_link",
            },
        )
        self.pelvis_pos = ObsTerm(
            func=mdp.get_pos,
            params={
                "robot_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "pelvis",
            },
        )
        self.pelvis_quat = ObsTerm(
            func=mdp.get_quat,
            params={
                "robot_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "pelvis",
            },
        )

        self.torso_pos = ObsTerm(
            func=mdp.get_pos,
            params={
                "robot_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "torso_link",
            },
        )
        self.torso_quat = ObsTerm(
            func=mdp.get_quat,
            params={
                "robot_cfg": SceneEntityCfg(asset_name),
                "target_link_name": "torso_link",
            },
        )
        self.base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg(asset_name)},
        )
        self.base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg(asset_name)},
        )
        self.pelvis_ang_vel = ObsTerm(
            func=mdp.get_ang_vel,
            params={"robot_cfg": SceneEntityCfg(asset_name)},
        )
        self.pelvis_lin_vel = ObsTerm(
            func=mdp.get_lin_vel,
            params={"robot_cfg": SceneEntityCfg(asset_name)},
        )
        self.torso_ang_vel = ObsTerm(
            func=mdp.get_ang_vel,
            params={"robot_cfg": SceneEntityCfg(asset_name)},
        )
        self.torso_lin_vel = ObsTerm(
            func=mdp.get_lin_vel,
            params={"robot_cfg": SceneEntityCfg(asset_name)},
        )
        self.joint_pos = ObsTerm(
            func=g1_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg(asset_name)},
        )
        self.joint_vel = ObsTerm(
            func=g1_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg(asset_name)},
        )
        self.joint_effort = ObsTerm(
            func=g1_mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg(asset_name)},
        )
        # self.head_cam_rgb = ObsTerm(
        #     func=mdp.image,
        #     params={"sensor_cfg": SceneEntityCfg(sensor_name), "data_type": "rgb", "normalize": False},
        # )
        # self.head_cam_depth = ObsTerm(
        #     func=mdp.image,
        #     params={"sensor_cfg": SceneEntityCfg(sensor_name), "data_type": "depth", "normalize": False},
        # )
        del self.sensor_name


def g1_move_strategy(
    trajectory: torch.Tensor,
    robot_state: Dict[str, torch.Tensor],
    hand_id: int = -1,
    lock_xy_steps: int = 20,
    num_rotation_steps: int = 50,
    lock_fwd_offset: float = 0.3,
    lock_perp_offset: float = 0.35,
    yaw_axis_correction: float = 0.8,
    clip_height: float | None = None,
    # Defaults: pelvis frame; Translate + Orien (° XYZ) → wxyz via ``quat_from_euler_xyz`` convention.
    left_rest_pose: G1EefRestPose7 = (
        0.2413,
        0.28537,
        0.15985,
        0.923866170836651,
        0.3827156762719941,
        -7.370950992571046e-05,
        -6.392341093299826e-05,
    ),
    right_rest_pose: G1EefRestPose7 = (
        0.2413,
        -0.28536,
        0.15985,
        0.9238695106257977,
        -0.38270761400415704,
        -7.370895208590888e-05,
        6.392405416738334e-05,
    ),
) -> torch.Tensor:
    """
    Convert a straight-line MotionGen trajectory to a multi-segment G1 trajectory.

    **Output ``lock_flag``** (last column of each waypoint) is the **15-dim action** field
    consumed by ``G1PControllerHelper.preprocess``; values align with **PController**
    ``mode_flag`` semantics. Typical timeline:

    * **-1** — Stand-up (if needed) and squat / hold-at-height: lock_skip (no nav/turn PD).
    * **0** — Horizontal motion along the MotionGen path (every waypoint, including last).
    * **1** — Rotation padding: fixed XY + standing height, orientation target from plan.

    Segments in order:

    1. Stand up (only if ``start_height < squat_threshold``): each waypoint uses ``lock_flag=-1``.
    2. Move: synthetic pelvis XY/Z with Z = standing height; ``lock_flag=0`` always.
       Last ``lock_xy_steps`` indices pin XY to ``locked_xy`` (still ``lock_flag=0``).
    2.5. If ``num_rotation_steps > 0``: hold ``locked_xy`` and standing height; ``lock_flag=1``.
    3. Squat (if standing vs target height differ): interpolate Z; ``lock_flag=-1``, then padding.

    Each segment copies the motiongen **base block** (pelvis + torso) from the appropriate
    ``trajectory`` row, then overwrites the synthesized pelvis fields; EEF targets are filled
    via height-relative-to-pelvis rule (see module docstring above).

    End-effector poses (last ``arm_dim`` values) are **not** copied verbatim: for each
    waypoint we keep the **world XY** and **orientation** from the corresponding
    ``trajectory`` frame, and set **Z** so the vertical offset to the **pelvis** matches
    the original plan: ``eef_z = pelvis_z_new + (eef_z_orig - pelvis_z_orig)``, where
    ``pelvis_*`` are the first 3 dims of the motiongen row and ``*_orig`` is from the
    trajectory index used for that segment (0, ``i``, or ``-1``). This accounts for
    synthetic pelvis height / horizontal changes without shifting EEF XY when the pelvis
    path is modified.

    If ``hand_id`` is ``0`` or ``1``, the **inactive** arm uses ``left_rest_pose`` /
    ``right_rest_pose`` in the **pelvis frame** (7-dim like base pose: xyz + wxyz). World
    targets: ``p_w = p_pelvis + quat_apply(q_pelvis, p_rest)``,
    ``q_w = quat_mul(q_pelvis, q_rest)`` at each waypoint's pelvis pose.

    Args:
        trajectory: [N, D] MotionGen plan; first 7 = base pose, rest includes arm.
        robot_state: ``base_pos`` / ``base_quat`` for current height and segment 1.
        hand_id: Active hand for grasp-point-based locked_xy computation.
            0 = right hand (EEF at ``arm_start:arm_start+7``),
            1 = left hand (EEF at ``arm_start+7:arm_start+14``),
            -1 = both / legacy (approach-direction-based offset from pelvis target).
        left_rest_pose: Inactive **left** arm when ``hand_id==0`` (pelvis-frame 7-dim).
        right_rest_pose: Inactive **right** arm when ``hand_id==1`` (pelvis-frame 7-dim).
        lock_xy_steps: Final segment-2 steps with XY held to ``locked_xy`` (0 disables).
        num_rotation_steps: Count of rotation-padding waypoints (0 disables segment 2.5).
        num_squat_steps_padding: Hold at target height after squat interpolation.
        yaw_axis_correction: Blend factor for snapping the target yaw toward the
            nearest world x-axis direction (0° or ±180°). 0 = no correction,
            1 = fully axis-aligned. Default 0.5.

    Returns:
        [M, D+1] Waypoints; **last dim = ``lock_flag``** in ``{-1, 0, 1}`` (this function never
        emits ``-2``; that is only for explicit skip in ``lock_flag`` elsewhere).

    The **last row**'s active-hand EEF matches ``trajectory[-1]``; if ``hand_id`` is ``0`` or
    ``1``, the inactive arm uses the same pelvis-frame rest pose as intermediate waypoints.

    Pelvis height (waypoint index 2) is never commanded below ``min_height`` (0.2 m): the plan's
    target height is clamped before computing ``standing_height``, and stand-up interpolation
    uses the same floor.
    """
    # G1-specific parameters
    default_height = 0.7
    min_height = 0.1
    squat_threshold = 0.5
    num_standup_steps = 10
    num_squat_steps = 20

    device = trajectory.device
    dtype = trajectory.dtype
    D = trajectory.shape[1]
    current_base_pos = robot_state["base_pos"]
    if current_base_pos.ndim > 1:
        current_base_pos = current_base_pos[0]

    # Get start and end base poses from trajectory
    if trajectory.shape[0] == 0:
        return torch.zeros(0, D + 1, device=device, dtype=dtype)

    # First waypoint is current state, last is target
    start_base_pose = trajectory[0, :7].clone()  # [7] (x, y, z, qw, qx, qy, qz)
    target_base_pose = trajectory[-1, :7].clone()  # [7]

    # Correct target yaw toward nearest x-axis direction (0 or ±π)
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
        target_base_pose[3] = torch.cos(_half)  # qw
        target_base_pose[4] = 0.0  # qx (no roll)
        target_base_pose[5] = 0.0  # qy (no pitch)
        target_base_pose[6] = torch.sin(_half)  # qz

    start_height = current_base_pos[2].item()
    target_height = target_base_pose[2].item()
    target_xy = target_base_pose[:2].clone()

    # Determine the intermediate standing height
    standing_height = max(default_height, target_height)

    # Arm is 14 dims; trajectory layout: base block + arm (14) = D
    # For G1: base block = pelvis(7) + torso(7) = 14, arm = dual EEF at [:, 14:28]
    arm_dim = 14
    arm_start = D - arm_dim  # arm is last 14 dims of trajectory
    base_block_dim = arm_start  # motiongen base (pelvis + torso) before EEF block
    left_rest_t = torch.tensor(left_rest_pose, device=device, dtype=dtype)
    right_rest_t = torch.tensor(right_rest_pose, device=device, dtype=dtype)

    def _pelvis_frame_rest_to_world(
        pelvis_pose: torch.Tensor, rest_pose_7: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``rest_pose_7``: pelvis-frame position + quaternion (wxyz). ``pelvis_pose``: world [7]."""
        p_l, q_l = rest_pose_7[0:3], rest_pose_7[3:7]
        p_w = pelvis_pose[0:3] + quat_apply(pelvis_pose[3:7], p_l)
        q_w = quat_mul(pelvis_pose[3:7], q_l)
        return p_w, q_w

    def _make_waypoint(lock_flag: float) -> torch.Tensor:
        """Build one waypoint; last element is ``lock_flag`` (-1 / 0 / 1). See ``g1_move_strategy`` doc."""
        wp = torch.full((D + 1,), float("nan"), device=device, dtype=dtype)
        wp[-1] = lock_flag
        return wp

    def _fill_eef_height_relative_to_pelvis(
        wp: torch.Tensor, traj: torch.Tensor, traj_idx: int, pelvis_pose: torch.Tensor
    ) -> None:
        """Set EEF from ``traj[traj_idx]``; inactive arm uses pelvis-frame rest pose (7-dim)."""
        if arm_start < 0 or arm_start + arm_dim > D:
            return
        pelvis_orig = traj[traj_idx, 0:3]

        def _fill_arm(k: int, rest_pose_7: torch.Tensor | None) -> None:
            eef_o = traj[traj_idx, arm_start + k : arm_start + k + 7]
            if rest_pose_7 is not None:
                p_w, q_w = _pelvis_frame_rest_to_world(pelvis_pose, rest_pose_7)
                wp[arm_start + k : arm_start + k + 3] = p_w
                wp[arm_start + k + 3 : arm_start + k + 7] = q_w
            else:
                rel_z = eef_o[2] - pelvis_orig[2]
                wp[arm_start + k + 0] = eef_o[0]
                wp[arm_start + k + 1] = eef_o[1]
                wp[arm_start + k + 2] = pelvis_pose[2] + rel_z
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

    # Build segments
    segments = []

    # Segment 1: Stand up (if needed) — lock_flag = -1 (lock_skip)
    if start_height < squat_threshold:
        for i in range(num_standup_steps):
            t = (i + 1) / num_standup_steps
            wp = _make_waypoint(-1.0)
            if base_block_dim > 0:
                wp[:base_block_dim] = trajectory[0, :base_block_dim].clone()
            wp[0] = current_base_pos[0]  # x stays
            wp[1] = current_base_pos[1]  # y stays
            wp[2] = max(
                start_height + t * (standing_height - start_height),
                min_height,
            )
            wp[3:7] = start_base_pose[3:7]
            _fill_eef_height_relative_to_pelvis(wp, trajectory, 0, wp[0:7])
            segments.append(wp)

    # Segment 2: Move horizontally — lock_flag = 0 (nav) for all points
    # Compute locked_xy in the robot's local frame defined by its final orientation:
    #   local y = forward (robot facing), local x = perpendicular (right).
    # "y aligned": locked position at same forward distance as target (no fwd offset).
    # "x offset": locked position offset by d perpendicular to forward,
    #   so the target ends up to the robot's side.
    num_move_points = trajectory.shape[0]

    # Extract yaw from target orientation (qw, qx, qy, qz)
    tgt_qw, tgt_qx = target_base_pose[3], target_base_pose[4]
    tgt_qy, tgt_qz = target_base_pose[5], target_base_pose[6]
    target_yaw = torch.atan2(
        2.0 * (tgt_qw * tgt_qz + tgt_qx * tgt_qy),
        1.0 - 2.0 * (tgt_qy**2 + tgt_qz**2),
    )

    cos_yaw = torch.cos(target_yaw)
    sin_yaw = torch.sin(target_yaw)
    fwd = torch.stack([cos_yaw, sin_yaw])  # robot forward in world
    right = torch.stack([sin_yaw, -cos_yaw])  # robot right in world

    if hand_id == 0:
        # Right hand → anchor on right EEF, pelvis goes LEFT (−right) to make room
        grasp_xy = trajectory[-1, arm_start : arm_start + 2].clone()
        locked_xy = grasp_xy - lock_perp_offset * right - lock_fwd_offset * fwd
    elif hand_id == 1:
        # Left hand → anchor on left EEF, pelvis goes RIGHT (+right) to make room
        grasp_xy = trajectory[-1, arm_start + 7 : arm_start + 9].clone()
        locked_xy = grasp_xy + lock_perp_offset * right - lock_fwd_offset * fwd
    else:
        # Both hands (hand_id=-1) → anchor on midpoint of both EEFs, only fwd offset
        grasp_xy_r = trajectory[-1, arm_start : arm_start + 2]
        grasp_xy_l = trajectory[-1, arm_start + 7 : arm_start + 9]
        grasp_xy = (grasp_xy_r + grasp_xy_l) / 2.0
        locked_xy = grasp_xy - lock_fwd_offset * fwd

    lock_start_idx = (
        max(0, num_move_points - lock_xy_steps)
        if lock_xy_steps > 0
        else num_move_points
    )

    # Linearly interpolate XY from trajectory start to locked_xy,
    # then hold at locked_xy for the remaining lock_xy_steps.
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

        wp[2] = standing_height  # override Z to standing height
        wp[3:7] = trajectory[i, 3:7]  # keep original pelvis orientation
        _fill_eef_height_relative_to_pelvis(wp, trajectory, i, wp[0:7])
        segments.append(wp)

    # Segment 2.5: Rotation padding — lock_flag = 1 (turning)
    # Stay at the last locked XY + standing height, rotate to target orientation
    if num_rotation_steps > 0:
        target_orientation = target_base_pose[3:7].clone()
        for i in range(num_rotation_steps):
            wp = _make_waypoint(1.0)
            if base_block_dim > 0:
                wp[:base_block_dim] = trajectory[-1, :base_block_dim].clone()
            wp[:2] = locked_xy
            wp[2] = standing_height
            wp[3:7] = target_orientation
            _fill_eef_height_relative_to_pelvis(wp, trajectory, -1, wp[0:7])
            segments.append(wp)

    # Segment 3: Squat down to target height (if needed) — lock_flag = -1 (lock_skip)
    if abs(standing_height - target_height) > 0.01:
        for i in range(num_squat_steps):
            t = (i + 1) / num_squat_steps
            wp = _make_waypoint(-1.0)
            if base_block_dim > 0:
                wp[:base_block_dim] = trajectory[-1, :base_block_dim].clone()
            wp[0] = locked_xy[0]
            wp[1] = locked_xy[1]
            wp[2] = standing_height + t * (
                max(target_height, min_height) - standing_height
            )
            wp[3:7] = target_base_pose[3:7]
            _fill_eef_height_relative_to_pelvis(wp, trajectory, -1, wp[0:7])
            segments.append(wp)

    if len(segments) == 0:
        wp = _make_waypoint(-1.0)
        wp[:D] = trajectory[-1, :D].clone()
        wp[2] = max(float(wp[2]), min_height)
        wp[-1] = -1.0
        return wp.unsqueeze(0)

    result = torch.stack(segments, dim=0)
    assert result.shape[1] == D + 1, f"result.shape: {result.shape}"

    result[:, 2] = torch.maximum(result[:, 2], torch.tensor(min_height, device=device))
    # Optional **lower bound** on commanded pelvis Z. The robot can stand
    # back up freely and translate at any height above ``clip_height``; only
    # commanded squats deeper than this get raised back to ``clip_height``.
    # Used when the env spawns the target on the ground and the upstream
    # planner would otherwise push the pelvis below the squat the robot can
    # physically reach. ``clip_height = None`` -> no extra clamp (default).
    if clip_height is not None:
        result[:, 2] = torch.maximum(
            result[:, 2], torch.tensor(float(clip_height), device=device)
        )

    # Terminal EEF: active hand = last plan frame; inactive = pelvis-frame rest → world.
    if result.shape[0] > 0 and arm_start >= 0 and arm_start + arm_dim <= D:
        tr_last = trajectory[-1, arm_start : arm_start + arm_dim].clone()
        last_pelvis_pose = result[-1, 0:7]
        if hand_id == -1:
            result[-1, arm_start : arm_start + arm_dim] = tr_last
        elif hand_id == 0:
            result[-1, arm_start : arm_start + 7] = tr_last[:7]
            p_w, q_w = _pelvis_frame_rest_to_world(last_pelvis_pose, left_rest_t)
            result[-1, arm_start + 7 : arm_start + 10] = p_w
            result[-1, arm_start + 10 : arm_start + 14] = q_w
        else:
            p_w, q_w = _pelvis_frame_rest_to_world(last_pelvis_pose, right_rest_t)
            result[-1, arm_start : arm_start + 3] = p_w
            result[-1, arm_start + 3 : arm_start + 7] = q_w
            result[-1, arm_start + 7 : arm_start + 14] = tr_last[7:14]

    return result


def g1_dehatch_strategy(
    phase: str,
    robot_state: Dict[str, "torch.Tensor"],
    retract_distance: float = 0.5,
    num_standup_steps: int = 100,
    num_retract_steps: int = 120,
    default_height: float = 0.8,
    left_rest_pose: G1EefRestPose7 = (0.24127, 0.15165, 0.14523, 1, 0, 0, 0),
    right_rest_pose: G1EefRestPose7 = (0.24127, -0.15164, 0.14523, 1, 0, 0, 0),
    env_id: int = 0,
    **kwargs,
) -> tuple:
    """
    G1 dehatch strategy — phase-based dispatcher.

    Called once per phase by the ``Dehatch`` AtomicSkill. Returns a tuple
    ``(global_planner_key, data)`` that the skill forwards to the GlobalPlannerManager.

    Phases
    ------
    ``"standup"``
        Pelvis Z rises from current height to ``default_height``. EEF positions keep a
        fixed offset relative to the pelvis (only Z shifts).
        Returns ``("MobileServoL", waypoints_tensor)``.

    ``"retract_hands"``
        Computes rest-pose targets in world frame at the current pelvis pose, then
        returns ``("MobileMoveL", target_14d)`` with ``hand_id=-1`` and
        ``planner_mode=1`` (force fixed-base MotionGen, no IK).

    ``"retract_base"``
        Pelvis walks backward by ``retract_distance``; EEF follows at rest poses.
        Returns ``("MobileServoL", waypoints_tensor)``.
    """
    base_pos = robot_state["base_pos"]
    base_quat = robot_state["base_quat"]
    if base_pos.ndim > 1:
        base_pos = base_pos[env_id]
        base_quat = base_quat[env_id]

    device = base_pos.device
    dtype = base_pos.dtype

    left_rest_t = torch.tensor(left_rest_pose, device=device, dtype=dtype)
    right_rest_t = torch.tensor(right_rest_pose, device=device, dtype=dtype)
    nan7 = torch.full((7,), float("nan"), device=device, dtype=dtype)

    def _rest_to_world(
        pelvis_pose7: "torch.Tensor", rest7: "torch.Tensor"
    ) -> "torch.Tensor":
        p_w = pelvis_pose7[:3] + quat_apply(pelvis_pose7[3:7], rest7[:3])
        q_w = quat_mul(pelvis_pose7[3:7], rest7[3:7])
        return torch.cat([p_w, q_w], dim=0)

    def _make_row_with_eef(
        pelvis7: "torch.Tensor",
        right_eef7: "torch.Tensor",
        left_eef7: "torch.Tensor",
        lock_flag: float,
    ) -> "torch.Tensor":
        return torch.cat(
            [
                pelvis7,
                nan7,
                right_eef7,
                left_eef7,
                torch.tensor([lock_flag], device=device, dtype=dtype),
            ],
            dim=0,
        )

    def _make_row(pelvis7: "torch.Tensor", lock_flag: float) -> "torch.Tensor":
        right_world = _rest_to_world(pelvis7, right_rest_t)
        left_world = _rest_to_world(pelvis7, left_rest_t)
        return _make_row_with_eef(pelvis7, right_world, left_world, lock_flag)

    # ---- Shared geometry ----
    current_z = base_pos[2].item()
    current_xy = base_pos[:2].clone()
    standing_z = float(default_height)

    qw, qx, qy, qz = base_quat[0], base_quat[1], base_quat[2], base_quat[3]
    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy**2 + qz**2),
    )

    # ------------------------------------------------------------------ #
    if phase == "standup":
        right_eef_world_pos = robot_state["right_eef_world_pos"]
        right_eef_world_quat = robot_state["right_eef_world_quat"]
        left_eef_world_pos = robot_state["left_eef_world_pos"]
        left_eef_world_quat = robot_state["left_eef_world_quat"]
        if right_eef_world_pos.ndim > 1:
            right_eef_world_pos = right_eef_world_pos[env_id]
            right_eef_world_quat = right_eef_world_quat[env_id]
            left_eef_world_pos = left_eef_world_pos[env_id]
            left_eef_world_quat = left_eef_world_quat[env_id]

        right_eef_z_offset = right_eef_world_pos[2].item() - current_z
        left_eef_z_offset = left_eef_world_pos[2].item() - current_z
        current_right_eef7 = torch.cat(
            [right_eef_world_pos, right_eef_world_quat], dim=0
        )
        current_left_eef7 = torch.cat([left_eef_world_pos, left_eef_world_quat], dim=0)

        segments = []
        for i in range(num_standup_steps):
            t = (i + 1) / num_standup_steps
            z = (
                current_z + t * (standing_z - current_z)
                if current_z < standing_z
                else current_z
            )
            pelvis7 = torch.cat(
                [current_xy, torch.tensor([z], device=device, dtype=dtype), base_quat],
                dim=0,
            )
            r_eef7 = current_right_eef7.clone()
            r_eef7[2] = z + right_eef_z_offset
            l_eef7 = current_left_eef7.clone()
            l_eef7[2] = z + left_eef_z_offset
            segments.append(_make_row_with_eef(pelvis7, r_eef7, l_eef7, -1.0))

        if not segments:
            pelvis7 = torch.cat([base_pos, base_quat], dim=0)
            segments.append(_make_row(pelvis7, -1.0))

        return ("MobileServoL", torch.stack(segments, dim=0))

    # ------------------------------------------------------------------ #
    elif phase == "retract_hands":
        # Compute rest poses in world frame at current pelvis
        pelvis7 = torch.cat([base_pos, base_quat], dim=0)
        rest_right_world7 = _rest_to_world(pelvis7, right_rest_t)
        rest_left_world7 = _rest_to_world(pelvis7, left_rest_t)
        # 14D target: right(7) + left(7) — world frame
        target_14d = torch.cat([rest_right_world7, rest_left_world7], dim=0)
        return ("MobileMoveL", target_14d)

    # ------------------------------------------------------------------ #
    elif phase == "retract_base":
        after_standup_z = standing_z if current_z < standing_z else current_z
        backward = torch.stack([-torch.cos(yaw), -torch.sin(yaw)])
        start_xy = current_xy.clone()
        end_xy = start_xy + retract_distance * backward

        segments = []
        for i in range(num_retract_steps):
            t = (i + 1) / num_retract_steps
            xy = start_xy + t * (end_xy - start_xy)
            pelvis7 = torch.cat(
                [
                    xy,
                    torch.tensor([after_standup_z], device=device, dtype=dtype),
                    base_quat,
                ],
                dim=0,
            )
            segments.append(_make_row(pelvis7, 0.0))

        if not segments:
            pelvis7 = torch.cat([base_pos, base_quat], dim=0)
            segments.append(_make_row(pelvis7, 0.0))

        return ("MobileServoL", torch.stack(segments, dim=0))

    else:
        raise ValueError(f"Unknown dehatch phase: {phase!r}")


class G1PControllerHelper:
    """
    G1 P-controller preprocess/postprocess.

    ``preprocess`` maps 15-dim ``lock_flag`` (index 14) to 8-dim ``mode_flag`` (last column):
    ``-2`` skip, ``-1`` lock_skip, ``0`` nav, ``1`` turning — same as ``PController``.
    All-NaN rows (IK wait) force ``lock_flag=-1`` (lock_skip), not ``-2``, so height is not
    replayed from zero ``last_command``. Stateful NaN fallbacks use ``_last_*`` when set.
    """

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        self._last_target_height = torch.full(
            (num_envs,), float("nan"), device=device, dtype=torch.float32
        )
        self._last_target_pelvis_pos = torch.full(
            (num_envs, 3), float("nan"), device=device, dtype=torch.float32
        )
        self._last_target_pelvis_quat = torch.full(
            (num_envs, 4), float("nan"), device=device, dtype=torch.float32
        )
        self._last_torso_roll = torch.full(
            (num_envs,), float("nan"), device=device, dtype=torch.float32
        )
        self._last_torso_pitch = torch.full(
            (num_envs,), float("nan"), device=device, dtype=torch.float32
        )
        self._last_torso_yaw_rel = torch.full(
            (num_envs,), float("nan"), device=device, dtype=torch.float32
        )

    def preprocess(
        self,
        action: torch.Tensor,
        robot_state: Dict,
        env_ids: torch.Tensor,
        device: torch.device = torch.device("cpu"),
        heading_threshold: float = 0.1,
        position_threshold: float = 0.1,
    ) -> torch.Tensor:
        """
        Preprocess 15-dim G1 action to 8-dim P-controller format (stateful).

        **Input ``lock_flag``** — ``action[:, 14]``, same semantics as ``PController`` command modes:

        * ``-2`` — **skip**: do not advance / ignore step semantics used elsewhere.
        * ``-1`` — **lock_skip**: hold pose (e.g. stand-up, squat, IK-wait); pelvis targets fall back to
          last valid or current state when the row is all-NaN.
        * ``0`` — **nav**: normal locomotion; ``target_heading`` tracks pelvis yaw.
        * ``1`` — **turning**: ``target_heading`` follows torso yaw (or pelvis yaw if torso is invalid).

        **All-NaN row (IK wait)** — if every element of ``action[i]`` is NaN, the value in
        ``action[i, 14]`` is **ignored**; the effective ``lock_flag`` is forced to ``-1`` (not ``-2``)
        so height is not replayed from a zero ``last_command``. In that case ``mode_flag`` is ``-1``
        even if column 14 would have been ``-2`` or any other number when read literally.

        **Output ``mode_flag``** — last column of the returned tensor. Normally matches the
        **effective** ``lock_flag`` (after the all-NaN override), with one additional
        auto-upgrade rule copied from ``RidgebackFrankaPControllerHelper``:

        * ``-2`` skip, ``-1`` lock_skip, ``1`` turning — passed through.
        * ``0`` nav — **upgraded to ``1`` (turning) when the pelvis XY is already within
          ``position_threshold`` of the target** so the base rotates in place once XY has
          converged instead of chasing residual position error. The heading source for
          those upgraded rows switches from pelvis yaw to ``torso_yaw_for_turning``.
        * All-NaN rows → forced ``-1`` (lock_skip).

        Returns
            ``[target_x, target_y, target_heading, target_height, torso_roll, torso_pitch, torso_yaw_rel, mode_flag]``.
        """
        action = action.to(device)
        current_pelvis_pos = robot_state["base_pos"][env_ids]
        current_pelvis_quat = robot_state["base_quat"][env_ids]
        N = action.shape[0]

        nan_mask = torch.isnan(action).all(dim=1)
        lock_flag = action[:, 14]
        # All-NaN row: ignore action[:,14]; force -1 so mode_flag matches lock_skip (not -2/skip).
        # Avoids zero last_command height; see preprocess docstring "All-NaN row (IK wait)".
        lock_flag = torch.where(nan_mask, torch.tensor(-1.0, device=device), lock_flag)
        skip_mask = torch.abs(lock_flag + 2.0) < 0.5
        lock_skip_mask = torch.abs(lock_flag + 1.0) < 0.5
        turning_input_mask = torch.abs(lock_flag - 1.0) < 0.5

        target_pelvis_pos = action[:, :3].clone()
        target_pelvis_quat = action[:, 3:7].clone()
        if torch.any(nan_mask):
            last_pos = self._last_target_pelvis_pos[env_ids]
            last_quat = self._last_target_pelvis_quat[env_ids]
            pos_fallback = torch.where(
                torch.isnan(last_pos).any(dim=1, keepdim=True).expand(-1, 3),
                current_pelvis_pos,
                last_pos,
            )
            quat_fallback = torch.where(
                torch.isnan(last_quat).any(dim=1, keepdim=True).expand(-1, 4),
                current_pelvis_quat,
                last_quat,
            )
            target_pelvis_pos = torch.where(
                nan_mask.unsqueeze(1), pos_fallback, target_pelvis_pos
            )
            target_pelvis_quat = torch.where(
                nan_mask.unsqueeze(1), quat_fallback, target_pelvis_quat
            )

        target_torso_pos = action[:, 7:10]
        target_torso_quat = action[:, 10:14]
        torso_pos_is_nan = torch.isnan(target_torso_pos).any(dim=1)

        # Yaw = third component of extrinsic XYZ Euler angles (same as atan2(2(wz+xy), 1-2(y²+z²)) for wxyz).
        action_pelvis_yaw = quat_to_euler_angles(
            target_pelvis_quat, degrees=False, extrinsic=True
        )[:, 2]

        target_x = target_pelvis_pos[:, 0]
        target_y = target_pelvis_pos[:, 1]
        target_height = target_pelvis_pos[:, 2]
        height_nan_mask = torch.isnan(target_height)
        if torch.any(height_nan_mask):
            current_height = current_pelvis_pos[:, 2]
            last_h = self._last_target_height[env_ids]
            fallback = torch.where(torch.isnan(last_h), current_height, last_h)
            target_height = torch.where(height_nan_mask, fallback, target_height)
        target_pelvis_yaw = action_pelvis_yaw.clone()

        zeros = torch.zeros(N, device=device)
        torso_roll = zeros.clone()
        torso_pitch = zeros.clone()
        torso_yaw_rel = zeros.clone()
        target_torso_yaw_world = target_pelvis_yaw.clone()

        torso_valid_mask = ~torso_pos_is_nan
        if torch.any(torso_valid_mask):
            valid_torso_quat = target_torso_quat[torso_valid_mask]
            valid_pelvis_yaw = action_pelvis_yaw[torso_valid_mask]
            valid_N = valid_torso_quat.shape[0]
            valid_torso_yaw_world = quat_to_euler_angles(
                valid_torso_quat, degrees=False, extrinsic=True
            )[:, 2]
            target_torso_yaw_world[torso_valid_mask] = valid_torso_yaw_world
            valid_zeros = torch.zeros(valid_N, device=device)
            pelvis_yaw_angles = torch.stack(
                [valid_pelvis_yaw, valid_zeros, valid_zeros], dim=1
            )
            pelvis_yaw_only_rot = euler_to_rot_matrix(
                pelvis_yaw_angles, degrees=False, extrinsic=True
            )
            torso_rot = quat_to_rot_matrix(valid_torso_quat)
            pelvis_yaw_only_rot_T = pelvis_yaw_only_rot.transpose(-2, -1)
            relative_rot = torch.bmm(pelvis_yaw_only_rot_T, torso_rot)
            relative_rpy = matrix_to_euler_angles(
                relative_rot, degrees=False, extrinsic=True
            )
            torso_roll[torso_valid_mask] = relative_rpy[:, 0]
            torso_pitch[torso_valid_mask] = relative_rpy[:, 1]
            torso_yaw_rel[torso_valid_mask] = relative_rpy[:, 2]

        if torch.any(torso_pos_is_nan):
            last_r = self._last_torso_roll[env_ids]
            last_p = self._last_torso_pitch[env_ids]
            last_y = self._last_torso_yaw_rel[env_ids]
            r_fallback = torch.where(torch.isnan(last_r), zeros, last_r)
            p_fallback = torch.where(torch.isnan(last_p), zeros, last_p)
            y_fallback = torch.where(torch.isnan(last_y), zeros, last_y)
            torso_roll = torch.where(torso_pos_is_nan, r_fallback, torso_roll)
            torso_pitch = torch.where(torso_pos_is_nan, p_fallback, torso_pitch)
            torso_yaw_rel = torch.where(torso_pos_is_nan, y_fallback, torso_yaw_rel)

        torso_yaw_for_turning = torch.where(
            torch.isnan(target_torso_yaw_world),
            target_pelvis_yaw,
            target_torso_yaw_world,
        )

        mode_flag = torch.zeros(N, device=device)
        mode_flag[skip_mask] = -2.0
        mode_flag[lock_skip_mask] = -1.0
        mode_flag[turning_input_mask] = 1.0

        # Nav → turning auto-upgrade (Ridgeback-style): once the pelvis XY
        # has converged to within ``position_threshold`` of the target, stop
        # walking and switch to in-place rotation. Only applies to rows where
        # upstream ``lock_flag`` was nav (0).
        nav_input_mask = ~(skip_mask | lock_skip_mask | turning_input_mask)
        current_pelvis_xy = current_pelvis_pos[:, :2]
        dxy = torch.stack(
            [target_x - current_pelvis_xy[:, 0], target_y - current_pelvis_xy[:, 1]],
            dim=1,
        )
        dist_to_target = torch.norm(dxy, dim=1)
        nav_upgrade_mask = nav_input_mask & (dist_to_target < position_threshold)
        mode_flag[nav_upgrade_mask] = 1.0

        target_heading = torch.where(
            mode_flag > 0.5,
            torso_yaw_for_turning,
            target_pelvis_yaw,
        )

        result = torch.stack(
            [
                target_x,
                target_y,
                target_heading,
                target_height,
                torso_roll,
                torso_pitch,
                torso_yaw_rel,
                mode_flag,
            ],
            dim=1,
        )
        self._last_target_height[env_ids] = result[:, 3]
        self._last_target_pelvis_pos[env_ids] = target_pelvis_pos
        self._last_target_pelvis_quat[env_ids] = target_pelvis_quat
        self._last_torso_roll[env_ids] = torso_roll
        self._last_torso_pitch[env_ids] = torso_pitch
        self._last_torso_yaw_rel[env_ids] = torso_yaw_rel
        return result

    def postprocess(
        self,
        action: torch.Tensor,
        mode_flag: torch.Tensor,
        robot_state: Dict,
        env_ids: torch.Tensor,
        default_height: float = 0.7,
        min_height: float = 0.3,
        max_scale: float = 1,
    ) -> torch.Tensor:
        """
        Height-based velocity scaling after ``PController``.

        Args:
            action: [N, 7] velocities + extras from P-controller output.
            mode_flag: [N] same as preprocess output last column; only **nav (0)** and
                **turning (1)** get squat-related scaling — see ``g1_postprocess_p_controller_action``.
            robot_state, env_ids: For current height lookup.
            default_height, min_height, max_scale: Passed through to ``g1_postprocess_p_controller_action``.

        Returns:
            [N, 7] scaled action.
        """
        return g1_postprocess_p_controller_action(
            action,
            mode_flag,
            robot_state,
            env_ids,
            default_height=default_height,
            min_height=min_height,
            max_scale=max_scale,
        )

    @staticmethod
    def move_strategy(
        trajectory: torch.Tensor,
        robot_state: Dict[str, torch.Tensor],
        hand_id: int = -1,
        lock_xy_steps: int = 10,
        num_rotation_steps: int = 50,
        clip_height: float | None = None,
        left_rest_pose: G1EefRestPose7 = (
            0.2413,
            0.28537,
            0.15985,
            0.923866170836651,
            0.3827156762719941,
            -7.370950992571046e-05,
            -6.392341093299826e-05,
        ),
        right_rest_pose: G1EefRestPose7 = (
            0.2413,
            -0.28536,
            0.15985,
            0.9238695106257977,
            -0.38270761400415704,
            -7.370895208590888e-05,
            6.392405416738334e-05,
        ),
    ) -> torch.Tensor:
        """Stateless wrapper around ``g1_move_strategy``.

        ``clip_height``: optional upper clamp on commanded pelvis Z across the
        whole strategy output. ``None`` (default) = unchanged behaviour. Set
        to e.g. 0.45 to keep the robot crouched while moving toward a target
        on the ground.

        Output last column is ``lock_flag`` for the 15-dim p-controller
        command; semantics are documented on ``g1_move_strategy``.
        """
        return g1_move_strategy(
            trajectory,
            robot_state,
            hand_id=hand_id,
            lock_xy_steps=lock_xy_steps,
            num_rotation_steps=num_rotation_steps,
            clip_height=clip_height,
            left_rest_pose=left_rest_pose,
            right_rest_pose=right_rest_pose,
        )

    def reset_idx(self, env_ids) -> None:
        """Clear last target state for reset environments."""
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        self._last_target_height[env_ids] = float("nan")
        self._last_target_pelvis_pos[env_ids] = float("nan")
        self._last_target_pelvis_quat[env_ids] = float("nan")
        self._last_torso_roll[env_ids] = float("nan")
        self._last_torso_pitch[env_ids] = float("nan")
        self._last_torso_yaw_rel[env_ids] = float("nan")


@configclass
class G1PlannerCfg(HumanoidPlannerCfg):
    max_eef_num: int = 2

    base_action_dim: Dict[str, int] = {
        "dwb_holonomic": 4,
        "p_controller": 15,  # Input: [pelvis_pose(7), torso_pose(7), lock_flag(1)]
        # Nav shares PController's 15-dim channel; last col is mode_flag ∈ {-2,-1,0,1,2}
        # ({-2,-1,0,1} -> PController, 2 -> Dwb, first 4 cols -> dwb_holonomic path).
        "nav": 15,
        "default": 7,
    }
    base_action_space: Dict[str, torch.Tensor] = {
        "dwb_holonomic": torch.tensor(
            [
                [-0.8, -0.8, -0.4, 0.3],
                [0.8, 0.8, 0.4, 0.8],
            ],
        ),
        # Input action space for 15-dim: [pelvis_pose(7), torso_pose(7), lock_flag(1)]
        # Each pose: [x, y, z, qw, qx, qy, qz]
        # lock_flag: -2..1 (skip / lock_skip / nav / turning); see G1PControllerHelper.preprocess
        # This is preprocessed internally to 8-dim:
        # [target_x, target_y, target_heading, target_height, torso_roll, torso_pitch, torso_yaw, mode_flag]
        "p_controller": torch.tensor(
            [
                # Lower bounds: [pelvis(7), torso(7), lock_flag]
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
                # Upper bounds
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
            ]
        ),
        # Nav = PController input + Dwb route flag (last col upper bound 2.0).
        # First 4 cols double as the dwb_holonomic path slice when mode_flag == 2.
        "nav": torch.tensor(
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
                    2.0,
                ],
            ]
        ),
        "default": torch.tensor(
            [
                [-0.4, -0.8, -torch.pi, 0.3, -1, -1, -1],
                [0.8, 0.8, torch.pi, 0.7, 1, 1, 1],
            ]
        ),
    }
    # P-controller helper: class with preprocess, postprocess, reset_idx
    p_controller_helper = G1PControllerHelper
    arm_action_dim: Dict[str, int] = {
        "default": 14,
        "curobo": 14,
    }
    arm_action_space: Dict[str, torch.Tensor] = {
        "default": None,
        "curobo": torch.tensor(
            [
                [
                    -1,
                    -1,
                    -1,
                    1,
                    1,
                    1,
                    1,
                    -1,
                    -1,
                    -1,
                    1,
                    1,
                    1,
                    1,
                ],  # left EE (7) + right EE (7)
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
            ]
        ),
    }
    eef_action_dim: Dict[str, int] = {
        "default": 14,
    }
    eef_action_space: Dict[str, torch.Tensor] = {
        "default": None,
    }
    # Move strategy for RetractMoveL: stand up -> move horizontally -> squat down
    move_strategy = G1PControllerHelper.move_strategy
    # Distance threshold for base movement to trigger move strategy
    move_strategy_distance_threshold: float = 0.3
    # Dehatch strategy: stand up -> retract hands -> step backward
    dehatch_strategy = staticmethod(g1_dehatch_strategy)


@configclass
class G1Cfg(HumanoidCfg):
    """Configuration for the G1 robot."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    base_action_name: str = MISSING
    arm_action_name: str = MISSING
    eef_action_name: str = MISSING
    robot: ArticulationCfg = MISSING
    action: G1ActionsCfg = MISSING
    sensor: G1HeadCameraCfg = G1HeadCameraCfg()
    obs: G1ObsCfg = MISSING
    planner: G1PlannerCfg = G1PlannerCfg()

    def __post_init__(self):
        self.robot: ArticulationCfg = MAGIC_G1_CFG
        # self.robot.spawn.articulation_props.fix_root_link = True
        self.robot.prim_path = self.prim_path
        self.action: G1ActionsCfg = G1ActionsCfg(
            asset_name=self.asset_name,
            base_action_name=self.base_action_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.sensor: G1HeadCameraCfg = G1HeadCameraCfg(
            robot_prim_path=self.prim_path,
        )
        self.obs: G1ObsCfg = G1ObsCfg(
            asset_name=self.asset_name, sensor_name=f"{self.asset_name}_head_camera"
        )
        super().__post_init__()
