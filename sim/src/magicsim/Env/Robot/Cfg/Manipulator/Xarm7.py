"""UFACTORY xArm 7 (7-DoF arm + xArm parallel gripper).

Pulled from ``Assets/Robots/xarm7.usd`` (flattened from
``cross_embodiement/Collected_xarm7``). The cuRobo YAML
``magicsim_xarm7.yml`` and the URDF under
``curobo/content/assets/robot/xarm7_description/`` are produced by
``Script/Robot/import_new_robot.py`` and verified by its built-in
motion_gen smoke test.

Kinematic chain (after USD->URDF):

    * base       : ``world`` (USD root prim of the arm chain)
    * arm joints : ``joint1..7``
    * tool frame : ``xarm_gripper_base_link`` (rigid offset off ``link7``)
    * gripper    : 6 revolute joints; ``drive_joint`` is the master, the
      other 5 mirror it. We lock them at 0 in the cuRobo YAML and drive
      them as a single binary group in the MagicSim action term.
"""

from typing import Dict

import torch
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from magicsim.Env.Robot.Cfg.Manipulator.Manipulator import (
    FrameSensorCfg,
    ManipulatorCfg,
    ManipulatorActionsCfg,
    ManipulatorObsCfg,
    ManipulatorPlannerCfg,
)
import magicsim.Env.Robot.mdp as mdp
from isaaclab.managers import ActionTermCfg as ActionTerm
from dataclasses import MISSING
from isaaclab.managers import ObservationTermCfg as ObsTerm
from magicsim import MAGICSIM_ASSETS
from magicsim.Env.Robot.mdp.curobo_ik_cfg import CuroboIKActionCfg


_ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
_GRIPPER_JOINTS = [
    "drive_joint",
    "left_inner_knuckle_joint",
    "right_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "left_finger_joint",
    "right_finger_joint",
]


XARM7_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/xarm7.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": -0.3,
            "joint3": 0.0,
            "joint4": 0.3,
            "joint5": 0.0,
            "joint6": 0.6,
            "joint7": 0.0,
            "drive_joint": 0.0,
            "left_inner_knuckle_joint": 0.0,
            "right_inner_knuckle_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            "left_finger_joint": 0.0,
            "right_finger_joint": 0.0,
        },
    ),
    actuators={
        "xarm7_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-7]"],
            effort_limit_sim=87.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "xarm7_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "drive_joint",
                ".*inner_knuckle_joint",
                "right_outer_knuckle_joint",
                ".*finger_joint",
            ],
            effort_limit_sim=20.0,
            stiffness=400.0,
            damping=20.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


@configclass
class Xarm7ActionsCfg(ManipulatorActionsCfg):
    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "arm_action": {
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=_ARM_JOINTS,
            ),
            "joint_pos_vel": mdp.JointPositionVelocityToLimitsActionCfg(
                joint_names=_ARM_JOINTS,
            ),
            "ik_curobo": CuroboIKActionCfg(
                joint_names=_ARM_JOINTS,
                robot_cfg_file="magicsim_xarm7.yml",
                action_space=torch.tensor(
                    [
                        [-0.6, -0.6, 0.0, -1.0, -1.0, -1.0, -1.0],
                        [0.6, 0.6, 1.2, 1.0, 1.0, 1.0, 1.0],
                    ]
                ),
                num_seeds=20,
                position_threshold=0.005,
                rotation_threshold=0.05,
                decimation=4,
                diff_ik_method="dls",
            ),
        },
        "eef_action": {
            "binary": mdp.BinaryJointPositionActionCfg(
                joint_names=_GRIPPER_JOINTS,
                # xArm gripper opens at 0, closes around 0.85 rad.
                open_command_expr={j: 0.0 for j in _GRIPPER_JOINTS},
                close_command_expr={j: 0.85 for j in _GRIPPER_JOINTS},
            ),
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=_GRIPPER_JOINTS,
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()


class Xarm7FrameCfg(FrameSensorCfg):
    def __post_init__(self):
        self.target_frames = [
            FrameTransformerCfg.FrameCfg(
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
        ]
        self.visualizer_cfg.prim_path = (
            self.robot_prim_path + "/Visuals/FrameTransformer"
        )
        self.target_frames[0].prim_path = (
            self.robot_prim_path + "/xarm_gripper_base_link"
        )
        self.prim_path = self.robot_prim_path + "/world"


@configclass
class Xarm7ObsCfg(ManipulatorObsCfg):
    gripper_pos: ObsTerm = MISSING

    def __post_init__(self):
        self.gripper_pos = ObsTerm(
            func=mdp.gripper_pos, params={"robot_cfg": SceneEntityCfg(self.asset_name)}
        )
        super().__post_init__()


@configclass
class Xarm7PlannerCfg(ManipulatorPlannerCfg):
    max_eef_num: int = 1

    arm_action_dim: Dict[str, int] = {
        "curobo": 7,
        "ik_curobo": 7,
        "default": 7,
    }
    eef_action_dim: Dict[str, int] = {
        "default": 1,
    }
    arm_action_space: Dict[str, torch.Tensor] = {
        "curobo": torch.tensor(
            [
                [-1, -1, -1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ]
        ),
        "default": None,
    }
    eef_action_space: Dict[str, torch.Tensor] = {
        "default": None,
    }


@configclass
class Xarm7Cfg(ManipulatorCfg):
    """Configuration for the UFACTORY xArm 7 (7-DoF arm + parallel gripper)."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    arm_action_name: str = "joint_pos"
    eef_action_name: str = "binary"
    frame_name: str = "ee_frame"
    robot: ArticulationCfg = MISSING
    action: Xarm7ActionsCfg = MISSING
    ee_frame: Xarm7FrameCfg = MISSING
    obs: Xarm7ObsCfg = MISSING
    planner: Xarm7PlannerCfg = Xarm7PlannerCfg()

    def __post_init__(self):
        self.robot: ArticulationCfg = XARM7_CFG
        self.robot.prim_path = self.prim_path
        self.action: Xarm7ActionsCfg = Xarm7ActionsCfg(
            asset_name=self.asset_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = Xarm7FrameCfg(robot_prim_path=self.prim_path)
        self.obs: Xarm7ObsCfg = Xarm7ObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        super().__post_init__()
