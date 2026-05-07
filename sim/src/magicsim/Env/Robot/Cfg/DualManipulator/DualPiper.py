"""Dual-arm Piper X robot. L_* = left arm (y=+0.35), R_* = right arm (y=-0.35).

Frame order: right first, left second (matches DualManipulator convention).
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


DUAL_PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/dual_piper.usd",
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
            "L_joint1": 0.0,
            "L_joint2": 1.2,
            "L_joint3": -1.2,
            "L_joint4": 0.0,
            "L_joint5": 0.7,
            "L_joint6": 0.0,
            "L_joint7": 0.025,
            "L_joint8": -0.025,
            "R_joint1": 0.0,
            "R_joint2": 1.2,
            "R_joint3": -1.2,
            "R_joint4": 0.0,
            "R_joint5": 0.7,
            "R_joint6": 0.0,
            "R_joint7": 0.025,
            "R_joint8": -0.025,
        },
    ),
    actuators={
        "dual_piper_arm_base": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_joint[1-4]"],
            effort_limit_sim=200.0,
            stiffness=1500.0,
            damping=120.0,
        ),
        "dual_piper_arm_wrist": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_joint[56]"],
            effort_limit_sim=200.0,
            stiffness=600.0,
            damping=120.0,
        ),
        "dual_piper_gripper": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_joint[78]"],
            effort_limit_sim=200.0,
            stiffness=5000.0,
            damping=200.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


_L_ARM_JOINTS = [f"L_joint{i}" for i in range(1, 7)]
_R_ARM_JOINTS = [f"R_joint{i}" for i in range(1, 7)]
_ALL_ARM_JOINTS = _L_ARM_JOINTS + _R_ARM_JOINTS


DUAL_PIPER_URDF_DIR = os.path.join(
    MAGICSIM_HOME,
    "Third_Party",
    "curobo",
    "src",
    "curobo",
    "content",
    "assets",
    "robot",
    "dual_piper_description",
)
DUAL_PIPER_URDF_PATH = os.path.join(DUAL_PIPER_URDF_DIR, "urdf", "dual_piper.urdf")
DUAL_PIPER_MESH_PATH = os.path.join(DUAL_PIPER_URDF_DIR, "urdf")


DUAL_PIPER_PINK_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="Robot_0",
    base_link_name="base_link",
    num_hand_joints=0,
    show_ik_warnings=True,
    urdf_path=DUAL_PIPER_URDF_PATH,
    mesh_path=DUAL_PIPER_MESH_PATH,
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        LocalFrameTask(
            "R_link6",  # right
            base_link_frame_name="base_link",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        LocalFrameTask(
            "L_link6",  # left
            base_link_frame_name="base_link",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        NullSpacePostureTask(
            cost=0.5,
            lm_damping=1,
            controlled_frames=["R_link6", "L_link6"],
            controlled_joints=_ALL_ARM_JOINTS,
            gain=0.3,
        ),
        DampingTask(cost=0.8),
    ],
    fixed_input_tasks=[],
    amplify_factor=2,
)


@configclass
class DualPiperActionsCfg(DualManipulatorActionsCfg):
    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "arm_action": {
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=_ALL_ARM_JOINTS,
            ),
            "ik_pink": PinkInverseKinematicsActionCfg(
                pink_controlled_joint_names=_ALL_ARM_JOINTS,
                num_joints=12,
                hand_joint_names=[],
                target_eef_link_names={
                    "right_wrist": "R_link6",
                    "left_wrist": "L_link6",
                },
                action_space=torch.tensor(
                    [
                        [
                            -0.6,
                            -0.6,
                            0.2,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -0.6,
                            -0.6,
                            0.2,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                        ],
                        [
                            0.6,
                            0.6,
                            1.2,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.6,
                            0.6,
                            1.2,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                    ]
                ),
                controller=DUAL_PIPER_PINK_IK_CONTROLLER_CFG,
                relative_to_base=False,
                decimation=4,
            ),
            "ik_dual_diff": mdp.DualDifferentialInverseKinematicsActionCfg(
                right_joint_names=_R_ARM_JOINTS,
                left_joint_names=_L_ARM_JOINTS,
                right_body_name="R_link6",
                left_body_name="L_link6",
                right_command_reference_body_name="R_base_link",
                left_command_reference_body_name="L_base_link",
                controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="dls"
                ),
                action_space=torch.tensor(
                    [
                        [
                            -0.6,
                            -0.6,
                            0.2,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -0.6,
                            -0.6,
                            0.2,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                        ],
                        [
                            0.6,
                            0.6,
                            1.2,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.6,
                            0.6,
                            1.2,
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
            # dual-arm YAML that declares tool_frames=["R_link6", "L_link6"]
            # from a shared articulation root.
            # TODO: author ``magicsim_dual_piper_x.yml``.
            "ik_dual_curobo": DualCuroboIKActionCfg(
                right_joint_names=_R_ARM_JOINTS,
                left_joint_names=_L_ARM_JOINTS,
                right_eef_link_name="R_link6",
                left_eef_link_name="L_link6",
                robot_cfg_file="magicsim_dual_piper_x.yml",
                action_space=torch.tensor(
                    [
                        [
                            -0.6,
                            -0.6,
                            0.2,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -0.6,
                            -0.6,
                            0.2,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                        ],
                        [
                            0.6,
                            0.6,
                            1.2,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.6,
                            0.6,
                            1.2,
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
                num_joints=12,
                hand_joint_names=[],
                target_eef_link_names={
                    "right_wrist": "R_link6",
                    "left_wrist": "L_link6",
                },
                action_space=torch.tensor(
                    [
                        [
                            -0.6,
                            -0.6,
                            0.2,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -0.6,
                            -0.6,
                            0.2,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                        ],
                        [
                            0.6,
                            0.6,
                            1.2,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.6,
                            0.6,
                            1.2,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                    ]
                ),
                controller=DUAL_PIPER_PINK_IK_CONTROLLER_CFG,
                relative_to_base=False,
                decimation=4,
                right_joint_names=_R_ARM_JOINTS,
                left_joint_names=_L_ARM_JOINTS,
                right_body_name="R_link6",
                left_body_name="L_link6",
                right_command_reference_body_name="R_base_link",
                left_command_reference_body_name="L_base_link",
                diff_ik_controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="dls"
                ),
            ),
        },
        "eef_action": {
            "binary": mdp.MultipleBinaryJointPositionActionCfg(
                joint_groups=[
                    mdp.BinaryJointActionCfg(
                        joint_names=["R_joint7", "R_joint8"],
                        open_command_expr={"R_joint7": 0.05, "R_joint8": -0.05},
                        close_command_expr={"R_joint7": 0.0, "R_joint8": 0.0},
                    ),
                    mdp.BinaryJointActionCfg(
                        joint_names=["L_joint7", "L_joint8"],
                        open_command_expr={"L_joint7": 0.05, "L_joint8": -0.05},
                        close_command_expr={"L_joint7": 0.0, "L_joint8": 0.0},
                    ),
                ],
            ),
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=["L_joint7", "L_joint8", "R_joint7", "R_joint8"],
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()


class DualPiperFrameCfg(DualManipulatorFrameCfg):
    """Right arm (R_link6) frame_index=0, left arm (L_link6) frame_index=1."""

    left_link_name: str = "L_link6"
    right_link_name: str = "R_link6"

    def __post_init__(self):
        super().__post_init__()


@configclass
class DualPiperObsCfg(DualManipulatorObsCfg):
    pass


@configclass
class DualPiperPlannerCfg(DualManipulatorPlannerCfg):
    arm_action_dim: Dict[str, int] = {
        "curobo": 14,
        "default": 12,
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
                    -0.6,
                    -0.6,
                    0.2,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -0.6,
                    -0.6,
                    0.2,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ],
                [0.6, 0.6, 1.2, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 1.2, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
        "ik_dual_diff": torch.tensor(
            [
                [
                    -0.6,
                    -0.6,
                    0.2,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -0.6,
                    -0.6,
                    0.2,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ],
                [0.6, 0.6, 1.2, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 1.2, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
        "ik_pink_diff": torch.tensor(
            [
                [
                    -0.6,
                    -0.6,
                    0.2,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -0.6,
                    -0.6,
                    0.2,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ],
                [0.6, 0.6, 1.2, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 1.2, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
        "ik_dual_curobo": torch.tensor(
            [
                [
                    -0.6,
                    -0.6,
                    0.2,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -0.6,
                    -0.6,
                    0.2,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ],
                [0.6, 0.6, 1.2, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 1.2, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
    }
    eef_action_space: Dict[str, torch.Tensor] = {
        "default": None,
    }


@configclass
class DualPiperCfg(DualManipulatorCfg):
    """Configuration for Dual Piper X (left y=+0.35, right y=-0.35)."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    arm_action_name: str = "joint_pos"
    eef_action_name: str = "binary"
    frame_name: str = "ee_frame"
    robot: ArticulationCfg = MISSING
    action: DualPiperActionsCfg = MISSING
    ee_frame: DualPiperFrameCfg = MISSING
    obs: DualPiperObsCfg = MISSING
    planner: DualPiperPlannerCfg = MISSING

    def __post_init__(self):
        self.robot: ArticulationCfg = DUAL_PIPER_CFG
        self.robot.prim_path = self.prim_path
        self.action: DualPiperActionsCfg = DualPiperActionsCfg(
            asset_name=self.asset_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = DualPiperFrameCfg(
            robot_prim_path=self.prim_path,
            left_link_name="L_link6",
            right_link_name="R_link6",
        )
        self.obs: DualPiperObsCfg = DualPiperObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        self.planner: DualPiperPlannerCfg = DualPiperPlannerCfg()
        super().__post_init__()
