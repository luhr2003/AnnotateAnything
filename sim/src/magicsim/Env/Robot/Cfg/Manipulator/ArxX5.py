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


ARX_X5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/arx_x5.usd",
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
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.022,
            "joint8": 0.022,
        },
    ),
    actuators={
        "arx_x5_arm_base": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-4]"],
            effort_limit_sim=200.0,
            stiffness=1500.0,
            damping=120.0,
        ),
        "arx_x5_arm_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[56]"],
            effort_limit_sim=200.0,
            stiffness=600.0,
            damping=120.0,
        ),
        "arx_x5_gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint[78]"],
            effort_limit_sim=200.0,
            stiffness=5000.0,
            damping=200.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


ARX_X5_PINK_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="robot",
    base_link_name="root",
    num_hand_joints=0,
    show_ik_warnings=True,
    urdf_path=f"{MAGICSIM_ASSETS}/Robots/URDF/arx_x5.urdf",
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
class ArxX5ActionsCfg(ManipulatorActionsCfg):
    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "arm_action": {
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=["joint[1-6]"],
            ),
            "joint_pos_vel": mdp.JointPositionVelocityToLimitsActionCfg(
                joint_names=["joint[1-6]"],
            ),
            "ik_pink": PinkInverseKinematicsActionCfg(
                pink_controlled_joint_names=["joint[1-6]"],
                num_joints=6,
                hand_joint_names=None,
                target_eef_link_names={"eef": "link6"},
                action_space=torch.tensor(
                    [
                        [-0.5, -0.5, 0.0, -1.0, -1.0, -1.0, -1.0],
                        [0.5, 0.5, 0.8, 1.0, 1.0, 1.0, 1.0],
                    ]
                ),
                controller=ARX_X5_PINK_IK_CONTROLLER_CFG,
                relative_to_base=False,
            ),
            "ik_curobo": CuroboIKActionCfg(
                joint_names=["joint[1-6]"],
                robot_cfg_file="magicsim_arx_x5.yml",
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
                open_command_expr={"joint7": 0.044, "joint8": 0.044},
                close_command_expr={"joint7": 0.0, "joint8": 0.0},
            ),
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=["joint[78]"],
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()


class ArxX5FrameCfg(FrameSensorCfg):
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
        self.target_frames[0].prim_path = self.robot_prim_path + "/link6"
        self.target_frames[1].prim_path = self.robot_prim_path + "/link7"
        self.target_frames[2].prim_path = self.robot_prim_path + "/link8"
        self.prim_path = self.robot_prim_path + "/base_link"


@configclass
class ArxX5ObsCfg(ManipulatorObsCfg):
    gripper_pos: ObsTerm = MISSING

    def __post_init__(self):
        self.gripper_pos = ObsTerm(
            func=mdp.gripper_pos, params={"robot_cfg": SceneEntityCfg(self.asset_name)}
        )
        super().__post_init__()


@configclass
class ArxX5PlannerCfg(ManipulatorPlannerCfg):
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
class ArxX5Cfg(ManipulatorCfg):
    """Configuration for the ARX-X5 (6-DoF + parallel gripper) robot."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    arm_action_name: str = "joint_pos"
    eef_action_name: str = "binary"
    frame_name: str = "ee_frame"
    robot: ArticulationCfg = MISSING
    action: ArxX5ActionsCfg = MISSING
    ee_frame: ArxX5FrameCfg = MISSING
    obs: ArxX5ObsCfg = MISSING
    planner: ArxX5PlannerCfg = ArxX5PlannerCfg()

    def __post_init__(self):
        self.robot: ArticulationCfg = ARX_X5_CFG
        self.robot.prim_path = self.prim_path
        self.action: ArxX5ActionsCfg = ArxX5ActionsCfg(
            asset_name=self.asset_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = ArxX5FrameCfg(robot_prim_path=self.prim_path)
        self.obs: ArxX5ObsCfg = ArxX5ObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        super().__post_init__()
