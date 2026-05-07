"""Dual-arm Franka UMI robot. L_panda_* = left arm (y=+0.5, yaw=-90°),
R_panda_* = right arm (y=-0.5, yaw=+90°).

Spacing matches the MagicSim fling / fold scene (two Frankas facing each
other across the table). Single-server-per-robot planner layout (see
Env/Planner/Services/README.md §1–§5 / §9): curobo YAML
``magicsim_dual_franka.yml`` declares both arms' tool_frames in one
articulation and 14 arm joints in cspace.

Frame order: right first, left second (DualManipulator convention).
"""

import os
from typing import Dict

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab.managers import ActionTermCfg as ActionTerm
from dataclasses import MISSING

from magicsim import MAGICSIM_ASSETS
from magicsim.Env.Robot.Cfg.DualManipulator.DualManipulator import (
    DualManipulatorCfg,
    DualManipulatorActionsCfg,
    DualManipulatorObsCfg,
    DualManipulatorPlannerCfg,
    DualManipulatorFrameCfg,
)
import magicsim.Env.Robot.mdp as mdp
from magicsim.Env.Robot.mdp.pink_ik import (
    DampingTask,
    LocalFrameTask,
    NullSpacePostureTask,
)
from magicsim.Env.Robot.mdp.pink_actions_cfg import (
    PinkIKControllerCfg,
    PinkInverseKinematicsActionCfg,
    PinkDualDifferentialInverseKinematicsActionCfg,
)
from magicsim.Env.Robot.mdp.curobo_ik_cfg import DualCuroboIKActionCfg


DUAL_FRANKA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/dual_franka.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Home pose mirrors FrankaUMICfg (Env/Robot/Cfg/Manipulator/FrankaUMI.py).
        joint_pos={
            "L_panda_joint1": 0.0,
            "L_panda_joint2": -1.3,
            "L_panda_joint3": 0.0,
            "L_panda_joint4": -2.5,
            "L_panda_joint5": 0.0,
            "L_panda_joint6": 1.0,
            "L_panda_joint7": 0.0,
            "L_panda_finger_joint1": 0.04,
            "L_panda_finger_joint2": 0.04,
            "R_panda_joint1": 0.0,
            "R_panda_joint2": -1.3,
            "R_panda_joint3": 0.0,
            "R_panda_joint4": -2.5,
            "R_panda_joint5": 0.0,
            "R_panda_joint6": 1.0,
            "R_panda_joint7": 0.0,
            "R_panda_finger_joint1": 0.04,
            "R_panda_finger_joint2": 0.04,
        },
    ),
    actuators={
        "dual_franka_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "dual_franka_forearm": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "dual_franka_gripper": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_panda_finger_joint.*"],
            effort_limit_sim=20.0,
            stiffness=400.0,
            damping=80.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


_L_ARM_JOINTS = [f"L_panda_joint{i}" for i in range(1, 8)]
_R_ARM_JOINTS = [f"R_panda_joint{i}" for i in range(1, 8)]
_ALL_ARM_JOINTS = _L_ARM_JOINTS + _R_ARM_JOINTS

_L_FINGER_JOINTS = ["L_panda_finger_joint1", "L_panda_finger_joint2"]
_R_FINGER_JOINTS = ["R_panda_finger_joint1", "R_panda_finger_joint2"]


# Kinematics-only URDF (no visual / collision) for Pink IK. Lives alongside
# the other Assets/Robots/URDF sources so conversions stay reproducible.
DUAL_FRANKA_PINK_URDF_PATH = os.path.join(
    MAGICSIM_ASSETS, "Robots", "URDF", "dual_franka_pink.urdf"
)


_ARM_LOW = [-1.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0]
_ARM_HIGH = [1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0]
_ARM_ACTION_SPACE_14 = torch.tensor([_ARM_LOW + _ARM_LOW, _ARM_HIGH + _ARM_HIGH])


DUAL_FRANKA_PINK_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="Robot_0",
    base_link_name="base_link",
    num_hand_joints=0,
    show_ik_warnings=True,
    urdf_path=DUAL_FRANKA_PINK_URDF_PATH,
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        LocalFrameTask(
            "R_panda_hand",
            base_link_frame_name="base_link",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        LocalFrameTask(
            "L_panda_hand",
            base_link_frame_name="base_link",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        NullSpacePostureTask(
            cost=0.5,
            lm_damping=1,
            controlled_frames=["R_panda_hand", "L_panda_hand"],
            controlled_joints=_ALL_ARM_JOINTS,
            gain=0.3,
        ),
        DampingTask(cost=0.8),
    ],
    fixed_input_tasks=[],
    amplify_factor=2,
)


@configclass
class DualFrankaActionsCfg(DualManipulatorActionsCfg):
    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "arm_action": {
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=_ALL_ARM_JOINTS,
            ),
            "ik_pink": PinkInverseKinematicsActionCfg(
                pink_controlled_joint_names=_ALL_ARM_JOINTS,
                num_joints=14,
                hand_joint_names=[],
                target_eef_link_names={
                    "right_wrist": "R_panda_hand",
                    "left_wrist": "L_panda_hand",
                },
                action_space=_ARM_ACTION_SPACE_14,
                controller=DUAL_FRANKA_PINK_IK_CONTROLLER_CFG,
                relative_to_base=False,
                decimation=4,
            ),
            "ik_dual_diff": mdp.DualDifferentialInverseKinematicsActionCfg(
                right_joint_names=_R_ARM_JOINTS,
                left_joint_names=_L_ARM_JOINTS,
                right_body_name="R_panda_hand",
                left_body_name="L_panda_hand",
                # The URDF base_to_[LR]_panda_link0 fixed joints collapse
                # panda_link0 into base_link, so panda_link0 is not a
                # rigid body in the articulation. Use base_link (identity-
                # aligned with world) as the per-arm ref body for both
                # arms; the Fling skill bakes the per-arm yaw into the
                # target quaternion (Rz(±90°)·[0,1,0,0]).
                right_command_reference_body_name="base_link",
                left_command_reference_body_name="base_link",
                controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="dls"
                ),
                action_space=_ARM_ACTION_SPACE_14,
                relative_to_base=False,
                # decimation=1: solve DLS IK every physics tick (120 Hz).
                # Trades a bit of compute for finer joint setpoint
                # tracking — useful for the Handover skill where rapid
                # left-arm swings need tight wrist trajectory following
                # to keep the closed-finger contact on the mug.
                decimation=1,
            ),
            "ik_dual_curobo": DualCuroboIKActionCfg(
                right_joint_names=_R_ARM_JOINTS,
                left_joint_names=_L_ARM_JOINTS,
                robot_cfg_file="magicsim_dual_franka.yml",
                right_eef_link_name="R_panda_hand",
                left_eef_link_name="L_panda_hand",
                action_space=_ARM_ACTION_SPACE_14,
                num_seeds=20,
                position_threshold=0.005,
                rotation_threshold=0.05,
                decimation=4,
                diff_ik_method="dls",
            ),
            "ik_pink_diff": PinkDualDifferentialInverseKinematicsActionCfg(
                pink_controlled_joint_names=_ALL_ARM_JOINTS,
                num_joints=14,
                hand_joint_names=[],
                target_eef_link_names={
                    "right_wrist": "R_panda_hand",
                    "left_wrist": "L_panda_hand",
                },
                action_space=_ARM_ACTION_SPACE_14,
                controller=DUAL_FRANKA_PINK_IK_CONTROLLER_CFG,
                relative_to_base=False,
                decimation=4,
                right_joint_names=_R_ARM_JOINTS,
                left_joint_names=_L_ARM_JOINTS,
                right_body_name="R_panda_hand",
                left_body_name="L_panda_hand",
                # panda_link0 gets collapsed into base_link (fixed joint);
                # use base_link as the ref body for both arms.
                right_command_reference_body_name="base_link",
                left_command_reference_body_name="base_link",
                diff_ik_controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="dls"
                ),
            ),
        },
        "eef_action": {
            "binary": mdp.MultipleBinaryJointPositionActionCfg(
                joint_groups=[
                    mdp.BinaryJointActionCfg(
                        joint_names=_R_FINGER_JOINTS,
                        open_command_expr={
                            "R_panda_finger_joint1": 0.04,
                            "R_panda_finger_joint2": 0.04,
                        },
                        close_command_expr={
                            "R_panda_finger_joint1": 0.0,
                            "R_panda_finger_joint2": 0.0,
                        },
                    ),
                    mdp.BinaryJointActionCfg(
                        joint_names=_L_FINGER_JOINTS,
                        open_command_expr={
                            "L_panda_finger_joint1": 0.04,
                            "L_panda_finger_joint2": 0.04,
                        },
                        close_command_expr={
                            "L_panda_finger_joint1": 0.0,
                            "L_panda_finger_joint2": 0.0,
                        },
                    ),
                ],
            ),
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=_L_FINGER_JOINTS + _R_FINGER_JOINTS,
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()


class DualFrankaFrameCfg(DualManipulatorFrameCfg):
    """Right arm (R_panda_hand) frame_index=0, left arm (L_panda_hand) frame_index=1."""

    left_link_name: str = "L_panda_hand"
    right_link_name: str = "R_panda_hand"

    def __post_init__(self):
        super().__post_init__()


@configclass
class DualFrankaObsCfg(DualManipulatorObsCfg):
    pass


@configclass
class DualFrankaPlannerCfg(DualManipulatorPlannerCfg):
    arm_action_dim: Dict[str, int] = {
        "curobo": 14,
        "default": 14,
        "ik_pink": 14,
        "ik_dual_diff": 14,
        "ik_dual_curobo": 14,
        "ik_pink_diff": 14,
    }
    eef_action_dim: Dict[str, int] = {
        "default": 2,
    }
    arm_action_space: Dict[str, torch.Tensor] = {
        "curobo": torch.tensor(
            [
                [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ),
        "default": None,
        "ik_pink": _ARM_ACTION_SPACE_14,
        "ik_dual_diff": _ARM_ACTION_SPACE_14,
        "ik_pink_diff": _ARM_ACTION_SPACE_14,
        "ik_dual_curobo": _ARM_ACTION_SPACE_14,
    }
    eef_action_space: Dict[str, torch.Tensor] = {
        "default": None,
    }


@configclass
class DualFrankaCfg(DualManipulatorCfg):
    """Configuration for Dual Franka UMI (left y=+0.5, right y=-0.5, arms facing each other)."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    arm_action_name: str = "joint_pos"
    eef_action_name: str = "binary"
    frame_name: str = "ee_frame"
    robot: ArticulationCfg = MISSING
    action: DualFrankaActionsCfg = MISSING
    ee_frame: DualFrankaFrameCfg = MISSING
    obs: DualFrankaObsCfg = MISSING
    planner: DualFrankaPlannerCfg = MISSING

    def __post_init__(self):
        self.robot: ArticulationCfg = DUAL_FRANKA_CFG
        self.robot.prim_path = self.prim_path
        self.action: DualFrankaActionsCfg = DualFrankaActionsCfg(
            asset_name=self.asset_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = DualFrankaFrameCfg(
            robot_prim_path=self.prim_path,
            left_link_name="L_panda_hand",
            right_link_name="R_panda_hand",
        )
        self.obs: DualFrankaObsCfg = DualFrankaObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        self.planner: DualFrankaPlannerCfg = DualFrankaPlannerCfg()
        super().__post_init__()
