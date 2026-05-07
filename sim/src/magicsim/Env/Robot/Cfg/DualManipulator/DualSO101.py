"""Dual-arm SO-ARM100 (SO101). L_* = left arm (y=+0.23), R_* = right arm (y=-0.23).

Two SO101 5-DoF arms + 1-DoF jaw gripper each, mounted on a shared ``base_link``.
Frame order follows the DualManipulator convention: right first (frame_index=0),
left second (frame_index=1).
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

from magicsim import MAGICSIM_ASSETS, MAGICSIM_HOME
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


DUAL_SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/dual_so101.usd",
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
        joint_pos={
            "L_shoulder_pan": 0.0,
            "L_shoulder_lift": -0.5,
            "L_elbow_flex": 0.8,
            "L_wrist_flex": 0.0,
            "L_wrist_roll": 0.0,
            "L_joints_gripper": 0.0,
            "R_shoulder_pan": 0.0,
            "R_shoulder_lift": -0.5,
            "R_elbow_flex": 0.8,
            "R_wrist_flex": 0.0,
            "R_wrist_roll": 0.0,
            "R_joints_gripper": 0.0,
        },
    ),
    actuators={
        "dual_so101_arm_base": ImplicitActuatorCfg(
            joint_names_expr=[
                "[LR]_shoulder_pan",
                "[LR]_shoulder_lift",
                "[LR]_elbow_flex",
            ],
            effort_limit_sim=20.0,
            stiffness=300.0,
            damping=30.0,
        ),
        "dual_so101_arm_wrist": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_wrist_flex", "[LR]_wrist_roll"],
            effort_limit_sim=20.0,
            stiffness=200.0,
            damping=20.0,
        ),
        "dual_so101_gripper": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_joints_gripper"],
            effort_limit_sim=20.0,
            stiffness=400.0,
            damping=20.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


_L_ARM_JOINTS = [
    "L_shoulder_pan",
    "L_shoulder_lift",
    "L_elbow_flex",
    "L_wrist_flex",
    "L_wrist_roll",
]
_R_ARM_JOINTS = [
    "R_shoulder_pan",
    "R_shoulder_lift",
    "R_elbow_flex",
    "R_wrist_flex",
    "R_wrist_roll",
]
_ALL_ARM_JOINTS = _L_ARM_JOINTS + _R_ARM_JOINTS
_ALL_GRIPPER_JOINTS = ["L_joints_gripper", "R_joints_gripper"]


DUAL_SO101_URDF_DIR = os.path.join(
    MAGICSIM_HOME,
    "Third_Party",
    "curobo",
    "src",
    "curobo",
    "content",
    "assets",
    "robot",
    "dual_so101",
)
DUAL_SO101_URDF_PATH = os.path.join(DUAL_SO101_URDF_DIR, "dual_so101.urdf")
DUAL_SO101_MESH_PATH = DUAL_SO101_URDF_DIR


DUAL_SO101_PINK_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="Robot_0",
    base_link_name="base_link",
    num_hand_joints=0,
    show_ik_warnings=True,
    urdf_path=DUAL_SO101_URDF_PATH,
    mesh_path=DUAL_SO101_MESH_PATH,
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        LocalFrameTask(
            "R_SO101_gripper",
            base_link_frame_name="base_link",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        LocalFrameTask(
            "L_SO101_gripper",
            base_link_frame_name="base_link",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        NullSpacePostureTask(
            cost=0.5,
            lm_damping=1,
            controlled_frames=["R_SO101_gripper", "L_SO101_gripper"],
            controlled_joints=_ALL_ARM_JOINTS,
            gain=0.3,
        ),
        DampingTask(cost=0.8),
    ],
    fixed_input_tasks=[],
    amplify_factor=2,
)


@configclass
class DualSO101ActionsCfg(DualManipulatorActionsCfg):
    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "arm_action": {
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=_ALL_ARM_JOINTS,
            ),
            "ik_pink": PinkInverseKinematicsActionCfg(
                pink_controlled_joint_names=_ALL_ARM_JOINTS,
                num_joints=10,
                hand_joint_names=[],
                target_eef_link_names={
                    "right_wrist": "R_SO101_gripper",
                    "left_wrist": "L_SO101_gripper",
                },
                action_space=torch.tensor(
                    [
                        [
                            -0.3,
                            -0.3,
                            0.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -0.3,
                            -0.3,
                            0.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                        ],
                        [
                            0.3,
                            0.3,
                            0.5,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.3,
                            0.3,
                            0.5,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                    ]
                ),
                controller=DUAL_SO101_PINK_IK_CONTROLLER_CFG,
                relative_to_base=False,
                decimation=4,
            ),
            "ik_dual_diff": mdp.DualDifferentialInverseKinematicsActionCfg(
                right_joint_names=_R_ARM_JOINTS,
                left_joint_names=_L_ARM_JOINTS,
                right_body_name="R_SO101_gripper",
                left_body_name="L_SO101_gripper",
                right_command_reference_body_name="R_base",
                left_command_reference_body_name="L_base",
                controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="dls"
                ),
                action_space=torch.tensor(
                    [
                        [
                            -0.3,
                            -0.3,
                            0.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -0.3,
                            -0.3,
                            0.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                        ],
                        [
                            0.3,
                            0.3,
                            0.5,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.3,
                            0.3,
                            0.5,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                    ]
                ),
                relative_to_base=False,
                decimation=4,
            ),
            # Two kinematically-independent arms: cuRobo batches both EEFs in
            # one solve (multi-tool-frame), while inter-decimation diff-IK
            # runs per-arm on each arm's own Jacobian slice. Requires a
            # dual-arm YAML declaring both EEFs from a shared root.
            # TODO: author ``magicsim_dual_so101.yml``.
            "ik_dual_curobo": DualCuroboIKActionCfg(
                right_joint_names=_R_ARM_JOINTS,
                left_joint_names=_L_ARM_JOINTS,
                right_eef_link_name="R_SO101_gripper",
                left_eef_link_name="L_SO101_gripper",
                robot_cfg_file="magicsim_dual_so101.yml",
                action_space=torch.tensor(
                    [
                        [
                            -0.3,
                            -0.3,
                            0.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -0.3,
                            -0.3,
                            0.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                        ],
                        [
                            0.3,
                            0.3,
                            0.5,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.3,
                            0.3,
                            0.5,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                    ]
                ),
                num_seeds=20,
                position_threshold=0.005,
                rotation_threshold=0.05,
                decimation=4,
                diff_ik_method="dls",
            ),
            "ik_pink_diff": PinkDualDifferentialInverseKinematicsActionCfg(
                pink_controlled_joint_names=_ALL_ARM_JOINTS,
                num_joints=10,
                hand_joint_names=[],
                target_eef_link_names={
                    "right_wrist": "R_SO101_gripper",
                    "left_wrist": "L_SO101_gripper",
                },
                action_space=torch.tensor(
                    [
                        [
                            -0.3,
                            -0.3,
                            0.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -0.3,
                            -0.3,
                            0.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                        ],
                        [
                            0.3,
                            0.3,
                            0.5,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.3,
                            0.3,
                            0.5,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                    ]
                ),
                controller=DUAL_SO101_PINK_IK_CONTROLLER_CFG,
                relative_to_base=False,
                decimation=4,
                right_joint_names=_R_ARM_JOINTS,
                left_joint_names=_L_ARM_JOINTS,
                right_body_name="R_SO101_gripper",
                left_body_name="L_SO101_gripper",
                right_command_reference_body_name="R_base",
                left_command_reference_body_name="L_base",
                diff_ik_controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="dls"
                ),
            ),
        },
        "eef_action": {
            "binary": mdp.MultipleBinaryJointPositionActionCfg(
                joint_groups=[
                    mdp.BinaryJointActionCfg(
                        joint_names=["R_joints_gripper"],
                        open_command_expr={"R_joints_gripper": 1.5},
                        close_command_expr={"R_joints_gripper": 0.0},
                    ),
                    mdp.BinaryJointActionCfg(
                        joint_names=["L_joints_gripper"],
                        open_command_expr={"L_joints_gripper": 1.5},
                        close_command_expr={"L_joints_gripper": 0.0},
                    ),
                ],
            ),
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=_ALL_GRIPPER_JOINTS,
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()


class DualSO101FrameCfg(DualManipulatorFrameCfg):
    """Right arm (R_SO101_gripper) frame_index=0, left arm (L_SO101_gripper) frame_index=1."""

    left_link_name: str = "L_SO101_gripper"
    right_link_name: str = "R_SO101_gripper"

    def __post_init__(self):
        super().__post_init__()


@configclass
class DualSO101ObsCfg(DualManipulatorObsCfg):
    pass


@configclass
class DualSO101PlannerCfg(DualManipulatorPlannerCfg):
    arm_action_dim: Dict[str, int] = {
        "curobo": 14,
        "default": 10,
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
        "ik_pink": torch.tensor(
            [
                [
                    -0.3,
                    -0.3,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -0.3,
                    -0.3,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ],
                [0.3, 0.3, 0.5, 1.0, 1.0, 1.0, 1.0, 0.3, 0.3, 0.5, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
        "ik_dual_diff": torch.tensor(
            [
                [
                    -0.3,
                    -0.3,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -0.3,
                    -0.3,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ],
                [0.3, 0.3, 0.5, 1.0, 1.0, 1.0, 1.0, 0.3, 0.3, 0.5, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
        "ik_pink_diff": torch.tensor(
            [
                [
                    -0.3,
                    -0.3,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -0.3,
                    -0.3,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ],
                [0.3, 0.3, 0.5, 1.0, 1.0, 1.0, 1.0, 0.3, 0.3, 0.5, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
        "ik_dual_curobo": torch.tensor(
            [
                [
                    -0.3,
                    -0.3,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -0.3,
                    -0.3,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ],
                [0.3, 0.3, 0.5, 1.0, 1.0, 1.0, 1.0, 0.3, 0.3, 0.5, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
    }
    eef_action_space: Dict[str, torch.Tensor] = {
        "default": None,
    }


@configclass
class DualSO101Cfg(DualManipulatorCfg):
    """Configuration for Dual SO101 (left y=+0.23, right y=-0.23)."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    arm_action_name: str = "joint_pos"
    eef_action_name: str = "binary"
    frame_name: str = "ee_frame"
    robot: ArticulationCfg = MISSING
    action: DualSO101ActionsCfg = MISSING
    ee_frame: DualSO101FrameCfg = MISSING
    obs: DualSO101ObsCfg = MISSING
    planner: DualSO101PlannerCfg = MISSING

    def __post_init__(self):
        self.robot: ArticulationCfg = DUAL_SO101_CFG
        self.robot.prim_path = self.prim_path
        self.action: DualSO101ActionsCfg = DualSO101ActionsCfg(
            asset_name=self.asset_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = DualSO101FrameCfg(
            robot_prim_path=self.prim_path,
            left_link_name="L_SO101_gripper",
            right_link_name="R_SO101_gripper",
        )
        self.obs: DualSO101ObsCfg = DualSO101ObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        self.planner: DualSO101PlannerCfg = DualSO101PlannerCfg()
        super().__post_init__()
