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
from pink.tasks import FrameTask
from magicsim.Env.Robot.mdp.pink_ik import NullSpacePostureTask
from magicsim.Env.Robot.mdp.pink_actions_cfg import (
    PinkIKControllerCfg,
    PinkInverseKinematicsActionCfg,
)
from magicsim.Env.Robot.mdp.curobo_ik_cfg import CuroboIKActionCfg


PIPER_X_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/piper_x.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0,
            "joint3": 0,
            "joint4": 0.0,
            "joint5": 0,
            "joint6": 0.0,
            "joint7": 0,
            "joint8": 0,
        },
    ),
    actuators={
        "piper_arm_base": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-4]"],
            effort_limit_sim=200.0,
            stiffness=1500.0,
            damping=120.0,
        ),
        "piper_arm_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[56]"],
            effort_limit_sim=200.0,
            stiffness=600.0,
            damping=120.0,
        ),
        "piper_gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint[78]"],
            effort_limit_sim=200.0,
            stiffness=5000.0,
            damping=200.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


PIPER_X_PINK_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="robot",
    base_link_name="root",
    num_hand_joints=0,
    show_ik_warnings=True,
    urdf_path=f"{MAGICSIM_ASSETS}/Robots/URDF/piper_x.urdf",
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        FrameTask(
            "link6",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        NullSpacePostureTask(
            cost=0.5,
            lm_damping=1,
            controlled_frames=["link6"],
            controlled_joints=[
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
            ],
            gain=0.3,
        ),
    ],
    fixed_input_tasks=[],
    amplify_factor=1.0,
)


@configclass
class PiperXActionsCfg(ManipulatorActionsCfg):
    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "arm_action": {
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=["joint[1-6]"],
            ),
            "ik_abs": mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["joint[1-6]"],
                body_name="gripper_base",
                controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="pinv"
                ),
                body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
                    pos=[0.0, 0.0, 0.0]
                ),
                action_space=torch.tensor(
                    [
                        [-0.5, -0.5, -0.5, -1.0, -1.0, -1.0, -1.0],
                        [0.5, 0.5, 0.8, 1.0, 1.0, 1.0, 1.0],
                    ]
                ),
            ),
            "joint_pos_vel": mdp.JointPositionVelocityToLimitsActionCfg(
                joint_names=["joint[1-6]"],
            ),
            "ik_pink": PinkInverseKinematicsActionCfg(
                pink_controlled_joint_names=["joint[1-6]"],
                num_joints=6,
                hand_joint_names=None,
                target_eef_link_names={"eef": "gripper_base"},
                action_space=torch.tensor(
                    [
                        [-0.5, -0.5, 0.0, -1.0, -1.0, -1.0, -1.0],
                        [0.5, 0.5, 0.8, 1.0, 1.0, 1.0, 1.0],
                    ]
                ),
                controller=PIPER_X_PINK_IK_CONTROLLER_CFG,
                relative_to_base=False,
            ),
            "ik_curobo": CuroboIKActionCfg(
                joint_names=["joint[1-6]"],
                robot_cfg_file="magicsim_piper_x.yml",
                # tool_frames is read from the YAML's kinematics.tool_frames.
                action_space=torch.tensor(
                    [
                        [-0.5, -0.5, 0.0, -1.0, -1.0, -1.0, -1.0],
                        [0.5, 0.5, 0.8, 1.0, 1.0, 1.0, 1.0],
                    ]
                ),
                num_seeds=20,
                position_threshold=0.005,
                rotation_threshold=0.05,
                decimation=4,
                diff_ik_method="pinv",
            ),
        },
        "eef_action": {
            "binary": mdp.BinaryJointPositionActionCfg(
                joint_names=["joint[78]"],
                open_command_expr={"joint7": 0.05, "joint8": -0.05},
                close_command_expr={"joint7": 0.0, "joint8": 0.0},
            ),
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=["joint[78]"],
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()


class PiperXFrameCfg(FrameSensorCfg):
    def __post_init__(self):
        self.target_frames = [
            FrameTransformerCfg.FrameCfg(
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                name="tool_rightfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                name="tool_leftfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                ),
            ),
        ]
        self.visualizer_cfg.prim_path = (
            self.robot_prim_path + "/Visuals/FrameTransformer"
        )
        self.target_frames[0].prim_path = self.robot_prim_path + "/gripper_base"
        self.target_frames[1].prim_path = self.robot_prim_path + "/link7"
        self.target_frames[2].prim_path = self.robot_prim_path + "/link8"
        self.prim_path = self.robot_prim_path + "/base_link"


@configclass
class PiperXObsCfg(ManipulatorObsCfg):
    gripper_pos: ObsTerm = MISSING

    def __post_init__(self):
        self.gripper_pos = ObsTerm(
            func=mdp.gripper_pos, params={"robot_cfg": SceneEntityCfg(self.asset_name)}
        )
        super().__post_init__()


@configclass
class PiperXPlannerCfg(ManipulatorPlannerCfg):
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
class PiperXCfg(ManipulatorCfg):
    """Configuration for the AgileX Piper X (6-DoF + parallel gripper) robot."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    arm_action_name: str = "joint_pos"
    eef_action_name: str = "binary"
    frame_name: str = "ee_frame"
    robot: ArticulationCfg = MISSING
    action: PiperXActionsCfg = MISSING
    ee_frame: PiperXFrameCfg = MISSING
    obs: PiperXObsCfg = MISSING
    planner: PiperXPlannerCfg = PiperXPlannerCfg()

    def __post_init__(self):
        self.robot: ArticulationCfg = PIPER_X_CFG
        self.robot.prim_path = self.prim_path
        self.action: PiperXActionsCfg = PiperXActionsCfg(
            asset_name=self.asset_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = PiperXFrameCfg(robot_prim_path=self.prim_path)
        self.obs: PiperXObsCfg = PiperXObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        super().__post_init__()
