"""
Xtrainer dual-arm robot. J1 = right arm, J2 = left arm.
Frame order: right (J1_6) first, left (J2_6) second.
"""

import os
from typing import Dict

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.utils import configclass
from magicsim.Env.Robot.Cfg.DualManipulator.DualManipulator import (
    DualManipulatorCfg,
    DualManipulatorActionsCfg,
    DualManipulatorObsCfg,
    DualManipulatorPlannerCfg,
    DualManipulatorFrameCfg,
)
import magicsim.Env.Robot.mdp as mdp
from isaaclab.managers import ActionTermCfg as ActionTerm
from dataclasses import MISSING
from magicsim import MAGICSIM_ASSETS, MAGICSIM_HOME
from magicsim.Env.Robot.mdp.pink_ik import (
    DampingTask,
    LocalFrameTask,
    NullSpacePostureTask,
)
from magicsim.Env.Robot.mdp.pink_actions_cfg import (
    PinkIKControllerCfg,
    PinkInverseKinematicsActionCfg,
)

XTRAINER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/xtrainer.usd",
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
            # Right arm (J1) initial positions
            "J1_1_joint": 0.0,
            "J1_2_joint": 0.0,
            "J1_3_joint": 0.0,
            "J1_4_joint": 0.0,
            "J1_5_joint": 0.0,
            "J1_6_joint": 0.0,
            "J1_7_joint": 0.0,  # Gripper finger 1
            "J1_8_joint": 0.0,  # Gripper finger 2
            # Left arm (J2) initial positions
            "J2_1_joint": 0.0,
            "J2_2_joint": 0.0,
            "J2_3_joint": 0.0,
            "J2_4_joint": 0.0,
            "J2_5_joint": 0.0,
            "J2_6_joint": 0.0,
            "J2_7_joint": 0.0,  # Gripper finger 1
            "J2_8_joint": 0.0,  # Gripper finger 2
        },
    ),
    actuators={
        # Right arm (J1) actuators
        "xtrainer_j1_1": ImplicitActuatorCfg(
            joint_names_expr=["J1_1_joint"],
            effort_limit_sim=67.0,
            stiffness=600.0,
            damping=60.0,
        ),
        "xtrainer_j1_2": ImplicitActuatorCfg(
            joint_names_expr=["J1_2_joint"],
            effort_limit_sim=67.0,
            stiffness=600.0,
            damping=20.0,
        ),
        "xtrainer_j1_3": ImplicitActuatorCfg(
            joint_names_expr=["J1_3_joint"],
            effort_limit_sim=34.2,
            stiffness=300.0,
            damping=12.0,
        ),
        "xtrainer_j1_4": ImplicitActuatorCfg(
            joint_names_expr=["J1_4_joint"],
            effort_limit_sim=18.6,
            stiffness=180.0,
            damping=8.0,
        ),
        "xtrainer_j1_5": ImplicitActuatorCfg(
            joint_names_expr=["J1_5_joint"],
            effort_limit_sim=18.6,
            stiffness=180.0,
            damping=8.0,
        ),
        "xtrainer_j1_6": ImplicitActuatorCfg(
            joint_names_expr=["J1_6_joint"],
            effort_limit_sim=18.6,
            stiffness=180.0,
            damping=8.0,
        ),
        "xtrainer_j1_gripper": ImplicitActuatorCfg(
            joint_names_expr=["J1_7_joint", "J1_8_joint"],
            effort_limit_sim=20.0,
            stiffness=200.0,
            damping=40.0,
        ),
        # Left arm (J2) actuators
        "xtrainer_j2_1": ImplicitActuatorCfg(
            joint_names_expr=["J2_1_joint"],
            effort_limit_sim=67.0,
            stiffness=600.0,
            damping=60.0,
        ),
        "xtrainer_j2_2": ImplicitActuatorCfg(
            joint_names_expr=["J2_2_joint"],
            effort_limit_sim=67.0,
            stiffness=600.0,
            damping=20.0,
        ),
        "xtrainer_j2_3": ImplicitActuatorCfg(
            joint_names_expr=["J2_3_joint"],
            effort_limit_sim=34.2,
            stiffness=300.0,
            damping=12.0,
        ),
        "xtrainer_j2_4": ImplicitActuatorCfg(
            joint_names_expr=["J2_4_joint"],
            effort_limit_sim=18.6,
            stiffness=180.0,
            damping=8.0,
        ),
        "xtrainer_j2_5": ImplicitActuatorCfg(
            joint_names_expr=["J2_5_joint"],
            effort_limit_sim=18.6,
            stiffness=180.0,
            damping=8.0,
        ),
        "xtrainer_j2_6": ImplicitActuatorCfg(
            joint_names_expr=["J2_6_joint"],
            effort_limit_sim=18.6,
            stiffness=180.0,
            damping=8.0,
        ),
        "xtrainer_j2_gripper": ImplicitActuatorCfg(
            joint_names_expr=["J2_7_joint", "J2_8_joint"],
            effort_limit_sim=20.0,
            stiffness=200.0,
            damping=40.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

XTRAINER_URDF_DIR = os.path.join(
    MAGICSIM_HOME,
    "Third_Party",
    "curobo",
    "src",
    "curobo",
    "content",
    "assets",
    "robot",
    "xtrainer",
)
XTRAINER_URDF_PATH = os.path.join(XTRAINER_URDF_DIR, "urdf", "xtrainer.urdf")
XTRAINER_MESH_PATH = os.path.join(XTRAINER_URDF_DIR, "urdf")

XTRAINER_PINK_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="Robot_0",
    base_link_name="base_link",
    num_hand_joints=0,
    show_ik_warnings=True,
    urdf_path=XTRAINER_URDF_PATH,
    mesh_path=XTRAINER_MESH_PATH,
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        LocalFrameTask(
            "J1_6",  # right
            base_link_frame_name="base_link",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        LocalFrameTask(
            "J2_6",  # left
            base_link_frame_name="base_link",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        NullSpacePostureTask(
            cost=0.5,
            lm_damping=1,
            controlled_frames=["J1_6", "J2_6"],
            controlled_joints=[
                "J1_1_joint",
                "J1_2_joint",
                "J1_3_joint",
                "J1_4_joint",
                "J1_5_joint",
                "J1_6_joint",
                "J2_1_joint",
                "J2_2_joint",
                "J2_3_joint",
                "J2_4_joint",
                "J2_5_joint",
                "J2_6_joint",
            ],
            gain=0.3,
        ),
        DampingTask(cost=0.8),
    ],
    fixed_input_tasks=[],
    amplify_factor=2,
)


@configclass
class XtrainerActionsCfg(DualManipulatorActionsCfg):
    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "arm_action": {
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=[
                    "J1_1_joint",
                    "J1_2_joint",
                    "J1_3_joint",
                    "J1_4_joint",
                    "J1_5_joint",
                    "J1_6_joint",
                    "J2_1_joint",
                    "J2_2_joint",
                    "J2_3_joint",
                    "J2_4_joint",
                    "J2_5_joint",
                    "J2_6_joint",
                ],
            ),
            "ik_pink": PinkInverseKinematicsActionCfg(
                pink_controlled_joint_names=[
                    "J1_1_joint",
                    "J1_2_joint",
                    "J1_3_joint",
                    "J1_4_joint",
                    "J1_5_joint",
                    "J1_6_joint",
                    "J2_1_joint",
                    "J2_2_joint",
                    "J2_3_joint",
                    "J2_4_joint",
                    "J2_5_joint",
                    "J2_6_joint",
                ],
                num_joints=12,
                hand_joint_names=[],
                target_eef_link_names={
                    "right_wrist": "J1_6",
                    "left_wrist": "J2_6",
                },
                action_space=torch.tensor(
                    [
                        # Lower: left EE (7) + right EE (7)
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
                        # Upper
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
                controller=XTRAINER_PINK_IK_CONTROLLER_CFG,
                relative_to_base=False,
            ),
        },
        "eef_action": {
            "binary": mdp.MultipleBinaryJointPositionActionCfg(
                joint_groups=[
                    # right gripper (J1), order: right first, left second
                    mdp.BinaryJointActionCfg(
                        joint_names=["J1_7_joint", "J1_8_joint"],
                        open_command_expr={
                            "J1_7_joint": 0.01,
                            "J1_8_joint": -0.01,
                        },
                        close_command_expr={
                            "J1_7_joint": -0.037,
                            "J1_8_joint": 0.038,
                        },
                    ),
                    # left gripper (J2)
                    mdp.BinaryJointActionCfg(
                        joint_names=["J2_7_joint", "J2_8_joint"],
                        open_command_expr={
                            "J2_7_joint": 0.01,
                            "J2_8_joint": -0.01,
                        },
                        close_command_expr={
                            "J2_7_joint": -0.037,
                            "J2_8_joint": 0.038,
                        },
                    ),
                ],
            ),
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=["J1_7_joint", "J1_8_joint", "J2_7_joint", "J2_8_joint"],
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()


class XtrainerFrameCfg(DualManipulatorFrameCfg):
    """J1=right, J2=left. Frame order: right first, left second."""

    left_link_name: str = "J2_6"
    right_link_name: str = "J1_6"

    def __post_init__(self):
        super().__post_init__()


@configclass
class XtrainerObsCfg(DualManipulatorObsCfg):
    pass


@configclass
class XtrainerPlannerCfg(DualManipulatorPlannerCfg):
    arm_action_dim: Dict[str, int] = {
        "curobo": 14,
        "default": 12,
        "ik_pink": 14,
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
    }
    eef_action_space: Dict[str, torch.Tensor] = {
        "default": None,
    }


@configclass
class XtrainerCfg(DualManipulatorCfg):
    """Configuration for the Xtrainer dual-arm robot. J1=right, J2=left."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    arm_action_name: str = "joint_pos"
    eef_action_name: str = "binary"
    frame_name: str = "ee_frame"
    robot: ArticulationCfg = MISSING
    action: XtrainerActionsCfg = MISSING
    ee_frame: XtrainerFrameCfg = MISSING
    obs: XtrainerObsCfg = MISSING
    planner: XtrainerPlannerCfg = MISSING

    def __post_init__(self):
        self.robot: ArticulationCfg = XTRAINER_CFG
        self.robot.prim_path = self.prim_path
        self.action: XtrainerActionsCfg = XtrainerActionsCfg(
            asset_name=self.asset_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = XtrainerFrameCfg(
            robot_prim_path=self.prim_path,
            left_link_name="J2_6",
            right_link_name="J1_6",
        )
        self.obs: XtrainerObsCfg = XtrainerObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        self.planner: XtrainerPlannerCfg = XtrainerPlannerCfg()
        super().__post_init__()
