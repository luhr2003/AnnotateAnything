"""Unimanual OpenArm (7-DoF + parallel gripper).

Mirrors :mod:`isaaclab_assets.robots.openarm.OPENARM_UNI_CFG` but points at the
local ``Assets/Robots/openarm_unimanual.usd`` so the asset path is portable.

Links / joints (from the USD default prim ``/openarm``):

    * base link        : ``openarm_link0``
    * arm joints       : ``openarm_joint[1-7]``
    * hand link        : ``openarm_hand`` (tool frame used by Pink / cuRobo IK)
    * finger joints    : ``openarm_finger_joint[1-2]`` (prismatic, ≤ 0.044 m)
    * fixed frames     : ``openarm_hand_joint``, ``openarm_ee_tcp_joint`` are
      revolute in the USD but have identical parent/child orientation — they
      are locked at 0 in the cuRobo YAML.

See ``Script/Robot/README.md`` for the full import-a-new-robot workflow that
regenerates the URDF + cuRobo YAML + collision spheres that back this cfg.
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


# 7-DoF Enactic OpenArm. Velocity / effort limits taken from the DM motor
# spec sheets (see isaaclab_assets/robots/openarm.py header for links).
OPENARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/openarm_unimanual.usd",
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
            "openarm_joint1": 1.57,
            "openarm_joint2": 0.0,
            "openarm_joint3": -1.57,
            "openarm_joint4": 1.57,
            "openarm_joint5": 0.0,
            "openarm_joint6": 0.0,
            "openarm_joint7": 0.0,
            "openarm_finger_joint.*": 0.044,
        },
    ),
    actuators={
        "openarm_arm": ImplicitActuatorCfg(
            joint_names_expr=["openarm_joint[1-7]"],
            velocity_limit_sim={
                "openarm_joint[1-2]": 2.175,
                "openarm_joint[3-4]": 2.175,
                "openarm_joint[5-7]": 2.61,
            },
            effort_limit_sim={
                "openarm_joint[1-2]": 40.0,
                "openarm_joint[3-4]": 27.0,
                "openarm_joint[5-7]": 7.0,
            },
            stiffness=400.0,
            damping=80.0,
        ),
        "openarm_gripper": ImplicitActuatorCfg(
            joint_names_expr=["openarm_finger_joint.*"],
            velocity_limit_sim=0.2,
            effort_limit_sim=333.33,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


_ARM_JOINTS = [f"openarm_joint{i}" for i in range(1, 8)]
_FINGER_JOINTS = ["openarm_finger_joint1", "openarm_finger_joint2"]


@configclass
class OpenarmActionsCfg(ManipulatorActionsCfg):
    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "arm_action": {
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=_ARM_JOINTS,
            ),
            "joint_pos_vel": mdp.JointPositionVelocityToLimitsActionCfg(
                joint_names=_ARM_JOINTS,
            ),
            "ik_abs": mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=_ARM_JOINTS,
                body_name="openarm_hand",
                controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="dls"
                ),
                body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
                    pos=[0.0, 0.0, 0.0]
                ),
                action_space=torch.tensor(
                    [
                        [-0.6, -0.6, 0.0, -1.0, -1.0, -1.0, -1.0],
                        [0.6, 0.6, 1.2, 1.0, 1.0, 1.0, 1.0],
                    ]
                ),
            ),
            "ik_curobo": CuroboIKActionCfg(
                joint_names=_ARM_JOINTS,
                robot_cfg_file="magicsim_openarm.yml",
                # tool_frames is read from the YAML's kinematics.tool_frames.
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
                joint_names=_FINGER_JOINTS,
                open_command_expr={
                    "openarm_finger_joint1": 0.044,
                    "openarm_finger_joint2": 0.044,
                },
                close_command_expr={
                    "openarm_finger_joint1": 0.0,
                    "openarm_finger_joint2": 0.0,
                },
            ),
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=_FINGER_JOINTS,
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()


class OpenarmFrameCfg(FrameSensorCfg):
    def __post_init__(self):
        self.target_frames = [
            FrameTransformerCfg.FrameCfg(
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
            FrameTransformerCfg.FrameCfg(
                name="tool_rightfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
            FrameTransformerCfg.FrameCfg(
                name="tool_leftfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ]
        self.visualizer_cfg.prim_path = (
            self.robot_prim_path + "/Visuals/FrameTransformer"
        )
        self.target_frames[0].prim_path = self.robot_prim_path + "/openarm_hand"
        self.target_frames[1].prim_path = self.robot_prim_path + "/openarm_right_finger"
        self.target_frames[2].prim_path = self.robot_prim_path + "/openarm_left_finger"
        self.prim_path = self.robot_prim_path + "/openarm_link0"


@configclass
class OpenarmObsCfg(ManipulatorObsCfg):
    gripper_pos: ObsTerm = MISSING

    def __post_init__(self):
        self.gripper_pos = ObsTerm(
            func=mdp.gripper_pos, params={"robot_cfg": SceneEntityCfg(self.asset_name)}
        )
        super().__post_init__()


@configclass
class OpenarmPlannerCfg(ManipulatorPlannerCfg):
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
class OpenarmCfg(ManipulatorCfg):
    """Configuration for the Enactic OpenArm (unimanual, 7-DoF + gripper)."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    arm_action_name: str = "joint_pos"
    eef_action_name: str = "binary"
    frame_name: str = "ee_frame"
    robot: ArticulationCfg = MISSING
    action: OpenarmActionsCfg = MISSING
    ee_frame: OpenarmFrameCfg = MISSING
    obs: OpenarmObsCfg = MISSING
    planner: OpenarmPlannerCfg = OpenarmPlannerCfg()

    def __post_init__(self):
        self.robot: ArticulationCfg = OPENARM_CFG
        self.robot.prim_path = self.prim_path
        self.action: OpenarmActionsCfg = OpenarmActionsCfg(
            asset_name=self.asset_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = OpenarmFrameCfg(robot_prim_path=self.prim_path)
        self.obs: OpenarmObsCfg = OpenarmObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        super().__post_init__()
