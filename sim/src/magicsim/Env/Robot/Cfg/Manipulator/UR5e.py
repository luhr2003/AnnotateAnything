"""Universal Robots UR5e + Robotiq 2F-85 parallel gripper.

Pulled from ``Assets/Robots/ur5e.usd`` (flattened from
``cross_embodiement/Collected_ur5e``). The cuRobo YAML
``magicsim_ur5e.yml`` and the URDF under
``curobo/content/assets/robot/ur5e_description/`` are produced by
``Script/Robot/import_new_robot.py`` and verified by its built-in
motion_gen smoke test.

Kinematic chain (after USD->URDF):

    * base       : ``base_link``
    * arm joints : ``shoulder_pan_joint``, ``shoulder_lift_joint``,
      ``elbow_joint``, ``wrist_1_joint``, ``wrist_2_joint``,
      ``wrist_3_joint`` (6-DoF)
    * tool frame : ``wrist_3_link`` (the Robotiq base link is rigidly
      mounted to it via ``robot_gripper_joint``)
    * gripper    : Robotiq 2F-85 — ``finger_joint`` is the master plus
      5 mirrored revolutes. We lock them all at 0 in the cuRobo YAML and
      drive them as a single binary action group in MagicSim.
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


_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
_GRIPPER_JOINTS = [
    "finger_joint",
    "right_outer_knuckle_joint",
    "right_inner_finger_joint",
    "right_inner_finger_knuckle_joint",
    "left_inner_finger_knuckle_joint",
    "left_inner_finger_joint",
]


UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/ur5e.usd",
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
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.57,
            "elbow_joint": 1.57,
            "wrist_1_joint": -1.57,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": 0.0,
            "finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            "right_inner_finger_joint": 0.0,
            "right_inner_finger_knuckle_joint": 0.0,
            "left_inner_finger_knuckle_joint": 0.0,
            "left_inner_finger_joint": 0.0,
        },
    ),
    actuators={
        "ur5e_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_[1-3]_joint",
            ],
            effort_limit_sim=150.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "ur5e_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "finger_joint",
                "right_outer_knuckle_joint",
                ".*inner_finger_joint",
                ".*inner_finger_knuckle_joint",
            ],
            effort_limit_sim=20.0,
            stiffness=400.0,
            damping=20.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


@configclass
class UR5eActionsCfg(ManipulatorActionsCfg):
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
                robot_cfg_file="magicsim_ur5e.yml",
                action_space=torch.tensor(
                    [
                        [-0.85, -0.85, 0.0, -1.0, -1.0, -1.0, -1.0],
                        [0.85, 0.85, 1.2, 1.0, 1.0, 1.0, 1.0],
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
                # 2F-85 closes around 0.7 rad on finger_joint.
                open_command_expr={j: 0.0 for j in _GRIPPER_JOINTS},
                close_command_expr={j: 0.7 for j in _GRIPPER_JOINTS},
            ),
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=_GRIPPER_JOINTS,
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()


class UR5eFrameCfg(FrameSensorCfg):
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
        self.target_frames[0].prim_path = self.robot_prim_path + "/wrist_3_link"
        self.prim_path = self.robot_prim_path + "/base_link"


@configclass
class UR5eObsCfg(ManipulatorObsCfg):
    gripper_pos: ObsTerm = MISSING

    def __post_init__(self):
        self.gripper_pos = ObsTerm(
            func=mdp.gripper_pos, params={"robot_cfg": SceneEntityCfg(self.asset_name)}
        )
        super().__post_init__()


@configclass
class UR5ePlannerCfg(ManipulatorPlannerCfg):
    max_eef_num: int = 1

    arm_action_dim: Dict[str, int] = {
        "curobo": 7,
        "ik_curobo": 7,
        "default": 6,
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
class UR5eCfg(ManipulatorCfg):
    """Configuration for the Universal Robots UR5e (6-DoF + Robotiq 2F-85)."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    arm_action_name: str = "joint_pos"
    eef_action_name: str = "binary"
    frame_name: str = "ee_frame"
    robot: ArticulationCfg = MISSING
    action: UR5eActionsCfg = MISSING
    ee_frame: UR5eFrameCfg = MISSING
    obs: UR5eObsCfg = MISSING
    planner: UR5ePlannerCfg = UR5ePlannerCfg()

    def __post_init__(self):
        self.robot: ArticulationCfg = UR5E_CFG
        self.robot.prim_path = self.prim_path
        self.action: UR5eActionsCfg = UR5eActionsCfg(
            asset_name=self.asset_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = UR5eFrameCfg(robot_prim_path=self.prim_path)
        self.obs: UR5eObsCfg = UR5eObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        super().__post_init__()
