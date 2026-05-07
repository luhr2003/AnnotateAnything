"""Bimanual OpenArm (two 7-DoF arms mounted on a shared ``openarm_body_link``).

Mirrors :mod:`isaaclab_assets.robots.openarm.OPENARM_BI_CFG` but points at the
local ``Assets/Robots/openarm_bimanual.usd`` so the asset path is portable.

USD layout (default prim ``/openarm``):

    * base link         : ``openarm_body_link``
    * left  arm joints  : ``openarm_left_joint[1-7]``
    * right arm joints  : ``openarm_right_joint[1-7]``
    * left  hand / tool : ``openarm_left_hand``
    * right hand / tool : ``openarm_right_hand``
    * left  fingers     : ``openarm_left_finger_joint[1-2]`` (→ ``openarm_left_{left,right}_finger``)
    * right fingers     : ``openarm_right_finger_joint[1-2]`` (→ ``openarm_right_{left,right}_finger``)

Frame order: right first, left second (DualManipulator convention).
"""

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
from magicsim.Env.Robot.mdp.curobo_ik_cfg import DualCuroboIKActionCfg


DUAL_OPENARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/openarm_bimanual.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "openarm_left_joint.*": 0.0,
            "openarm_right_joint.*": 0.0,
            "openarm_left_finger_joint.*": 0.044,
            "openarm_right_finger_joint.*": 0.044,
        },
    ),
    actuators={
        "openarm_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "openarm_left_joint[1-7]",
                "openarm_right_joint[1-7]",
            ],
            velocity_limit_sim={
                "openarm_left_joint[1-2]": 2.175,
                "openarm_right_joint[1-2]": 2.175,
                "openarm_left_joint[3-4]": 2.175,
                "openarm_right_joint[3-4]": 2.175,
                "openarm_left_joint[5-7]": 2.61,
                "openarm_right_joint[5-7]": 2.61,
            },
            effort_limit_sim={
                "openarm_left_joint[1-2]": 40.0,
                "openarm_right_joint[1-2]": 40.0,
                "openarm_left_joint[3-4]": 27.0,
                "openarm_right_joint[3-4]": 27.0,
                "openarm_left_joint[5-7]": 7.0,
                "openarm_right_joint[5-7]": 7.0,
            },
            stiffness=400.0,
            damping=80.0,
        ),
        "openarm_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "openarm_left_finger_joint.*",
                "openarm_right_finger_joint.*",
            ],
            velocity_limit_sim=0.2,
            effort_limit_sim=333.33,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


_L_ARM_JOINTS = [f"openarm_left_joint{i}" for i in range(1, 8)]
_R_ARM_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]
_ALL_ARM_JOINTS = _L_ARM_JOINTS + _R_ARM_JOINTS

_L_FINGER_JOINTS = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
_R_FINGER_JOINTS = ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]


_ARM_LOW = [-1.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0]
_ARM_HIGH = [1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0]
_ARM_ACTION_SPACE_14 = torch.tensor([_ARM_LOW + _ARM_LOW, _ARM_HIGH + _ARM_HIGH])


@configclass
class DualOpenarmActionsCfg(DualManipulatorActionsCfg):
    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "arm_action": {
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=_ALL_ARM_JOINTS,
            ),
            "ik_dual_diff": mdp.DualDifferentialInverseKinematicsActionCfg(
                right_joint_names=_R_ARM_JOINTS,
                left_joint_names=_L_ARM_JOINTS,
                right_body_name="openarm_right_hand",
                left_body_name="openarm_left_hand",
                right_command_reference_body_name="openarm_body_link",
                left_command_reference_body_name="openarm_body_link",
                controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="dls"
                ),
                action_space=_ARM_ACTION_SPACE_14,
                relative_to_base=False,
                decimation=4,
            ),
            "ik_dual_curobo": DualCuroboIKActionCfg(
                right_joint_names=_R_ARM_JOINTS,
                left_joint_names=_L_ARM_JOINTS,
                robot_cfg_file="magicsim_dual_openarm.yml",
                right_eef_link_name="openarm_right_hand",
                left_eef_link_name="openarm_left_hand",
                action_space=_ARM_ACTION_SPACE_14,
                num_seeds=20,
                position_threshold=0.005,
                rotation_threshold=0.05,
                decimation=4,
                diff_ik_method="dls",
            ),
        },
        "eef_action": {
            "binary": mdp.MultipleBinaryJointPositionActionCfg(
                joint_groups=[
                    mdp.BinaryJointActionCfg(
                        joint_names=_R_FINGER_JOINTS,
                        open_command_expr={
                            "openarm_right_finger_joint1": 0.044,
                            "openarm_right_finger_joint2": 0.044,
                        },
                        close_command_expr={
                            "openarm_right_finger_joint1": 0.0,
                            "openarm_right_finger_joint2": 0.0,
                        },
                    ),
                    mdp.BinaryJointActionCfg(
                        joint_names=_L_FINGER_JOINTS,
                        open_command_expr={
                            "openarm_left_finger_joint1": 0.044,
                            "openarm_left_finger_joint2": 0.044,
                        },
                        close_command_expr={
                            "openarm_left_finger_joint1": 0.0,
                            "openarm_left_finger_joint2": 0.0,
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


class DualOpenarmFrameCfg(DualManipulatorFrameCfg):
    """Right arm (openarm_right_hand) frame_index=0, left arm frame_index=1."""

    left_link_name: str = "openarm_left_hand"
    right_link_name: str = "openarm_right_hand"

    def __post_init__(self):
        super().__post_init__()
        # Bimanual base link is ``openarm_body_link`` (not ``base_link``); override.
        self.prim_path = self.robot_prim_path + "/openarm_body_link"


@configclass
class DualOpenarmObsCfg(DualManipulatorObsCfg):
    pass


@configclass
class DualOpenarmPlannerCfg(DualManipulatorPlannerCfg):
    arm_action_dim: Dict[str, int] = {
        "curobo": 14,
        "default": 14,
        "ik_dual_diff": 14,
        "ik_dual_curobo": 14,
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
        "ik_dual_diff": _ARM_ACTION_SPACE_14,
        "ik_dual_curobo": _ARM_ACTION_SPACE_14,
    }
    eef_action_space: Dict[str, torch.Tensor] = {
        "default": None,
    }


@configclass
class DualOpenarmCfg(DualManipulatorCfg):
    """Configuration for the Enactic Bimanual OpenArm (2×7-DoF + 2×gripper)."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    arm_action_name: str = "joint_pos"
    eef_action_name: str = "binary"
    frame_name: str = "ee_frame"
    robot: ArticulationCfg = MISSING
    action: DualOpenarmActionsCfg = MISSING
    ee_frame: DualOpenarmFrameCfg = MISSING
    obs: DualOpenarmObsCfg = MISSING
    planner: DualOpenarmPlannerCfg = MISSING

    def __post_init__(self):
        self.robot: ArticulationCfg = DUAL_OPENARM_CFG
        self.robot.prim_path = self.prim_path
        self.action: DualOpenarmActionsCfg = DualOpenarmActionsCfg(
            asset_name=self.asset_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = DualOpenarmFrameCfg(
            robot_prim_path=self.prim_path,
            left_link_name="openarm_left_hand",
            right_link_name="openarm_right_hand",
        )
        self.obs: DualOpenarmObsCfg = DualOpenarmObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        self.planner: DualOpenarmPlannerCfg = DualOpenarmPlannerCfg()
        super().__post_init__()
