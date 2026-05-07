from typing import Dict

import torch
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

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

MAGIC_FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
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
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -1.3,
            "panda_joint3": 0.0,
            "panda_joint4": -2.5,
            "panda_joint5": 0.0,
            "panda_joint6": 1.0,
            "panda_joint7": 0.0,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FRANKA_PINK_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="robot",
    base_link_name="root",
    num_hand_joints=0,
    show_ik_warnings=True,
    urdf_path=f"{MAGICSIM_ASSETS}/Robots/URDF/franka_panda.urdf",
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        FrameTask(
            "panda_hand",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        NullSpacePostureTask(
            cost=0.5,
            lm_damping=1,
            controlled_frames=["panda_hand"],
            controlled_joints=[
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
            ],
            gain=0.3,
        ),
    ],
    fixed_input_tasks=[],
    amplify_factor=1.0,
)


@configclass
class FrankaActionsCfg(ManipulatorActionsCfg):
    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "arm_action": {
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=["panda_joint.*"],
            ),
            "ik_abs": mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="pinv"
                ),
                body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
                    pos=[0.0, 0.0, 0]
                ),
                action_space=torch.tensor(
                    [
                        [-0.6, -0.6, -1.0, -1.0, -1.0, -1.0, -1.0],
                        [0.6, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0],
                    ]
                ),
            ),
            "joint_pos_vel": mdp.JointPositionVelocityToLimitsActionCfg(
                joint_names=["panda_joint.*"],
            ),
            "ik_pink": PinkInverseKinematicsActionCfg(
                pink_controlled_joint_names=["panda_joint.*"],
                num_joints=7,
                hand_joint_names=None,
                target_eef_link_names={"eef": "panda_hand"},
                action_space=torch.tensor(
                    [
                        [-0.6, -0.6, 0.0, -1.0, -1.0, -1.0, -1.0],
                        [0.6, 0.6, 1.2, 1.0, 1.0, 1.0, 1.0],
                    ]
                ),
                controller=FRANKA_PINK_IK_CONTROLLER_CFG,
                relative_to_base=False,
            ),
            "ik_curobo": CuroboIKActionCfg(
                joint_names=["panda_joint.*"],
                robot_cfg_file="magicsim_franka.yml",
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
                diff_ik_method="pinv",
            ),
        },
        "eef_action": {
            "binary": mdp.BinaryJointPositionActionCfg(
                joint_names=["panda_finger.*"],
                open_command_expr={"panda_finger_.*": 0.04},
                close_command_expr={"panda_finger_.*": 0.0},
            ),
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=["panda_finger.*"],
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()


class FrankaFrameCfg(FrameSensorCfg):
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
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                name="tool_leftfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
        ]
        self.visualizer_cfg.prim_path = (
            self.robot_prim_path + "/Visuals/FrameTransformer"
        )
        self.target_frames[0].prim_path = self.robot_prim_path + "/panda_hand"
        self.target_frames[1].prim_path = self.robot_prim_path + "/panda_rightfinger"
        self.target_frames[2].prim_path = self.robot_prim_path + "/panda_leftfinger"
        self.prim_path = self.robot_prim_path + "/panda_link0"


@configclass
class FrankaObsCfg(ManipulatorObsCfg):
    gripper_pos: ObsTerm = MISSING

    def __post_init__(self):
        self.gripper_pos = ObsTerm(
            func=mdp.gripper_pos, params={"robot_cfg": SceneEntityCfg(self.asset_name)}
        )
        super().__post_init__()


@configclass
class FrankaPlannerCfg(ManipulatorPlannerCfg):
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
                [-1, -1, -1, 1, 1, 1, 1],  # x y z + quaternion
                [1, 1, 1, 1, 1, 1, 1],
            ]
        ),
        "default": None,
    }
    eef_action_space: Dict[str, torch.Tensor] = {
        "default": None,
    }


@configclass
class FrankaCfg(ManipulatorCfg):
    """Configuration for the Franka robot."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    arm_action_name: str = "joint_pos"
    eef_action_name: str = "binary"
    frame_name: str = "ee_frame"
    robot: ArticulationCfg = MISSING
    action: FrankaActionsCfg = MISSING
    ee_frame: FrankaFrameCfg = MISSING
    obs: FrankaObsCfg = MISSING
    planner: FrankaPlannerCfg = FrankaPlannerCfg()

    def __post_init__(self):
        self.robot: ArticulationCfg = MAGIC_FRANKA_PANDA_CFG
        self.robot.prim_path = self.prim_path
        self.action: FrankaActionsCfg = FrankaActionsCfg(
            asset_name=self.asset_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = FrankaFrameCfg(robot_prim_path=self.prim_path)
        self.obs: FrankaObsCfg = FrankaObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        super().__post_init__()
