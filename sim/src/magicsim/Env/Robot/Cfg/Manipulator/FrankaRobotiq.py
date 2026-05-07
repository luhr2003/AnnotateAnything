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


# Robotiq 2F-85 mechanics (mirrors ``FRANKA_ROBOTIQ_GRIPPER_CFG`` in
# ``isaaclab_assets.robots.franka``):
#
# * ``finger_joint`` is the *only* actively PD-controlled joint.
# * ``right_outer_knuckle_joint`` and the two ``*_inner_knuckle_joint`` joints
#   are passive — the USD encodes a closed kinematic loop, so PhysX propagates
#   ``finger_joint`` motion through them with zero PD.
# * ``*_inner_finger_joint`` get a very low PD so the finger pads can flex
#   parallel to the palm when they touch an object (enables parallel grasp).
#
# The joint names below follow this USD's own URDF (``left_inner_knuckle_joint``
# etc., without IsaacLab's ``_finger_`` infix), not IsaacLab canonical names.
_FINGER_OPEN = 0.0
_FINGER_CLOSE = 0.725


MAGIC_FRANKA_ROBOTIQ_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/frankarobotiq/robot.usd",
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
            "panda_joint1": 0.0,
            "panda_joint2": -1.3,
            "panda_joint3": 0.0,
            "panda_joint4": -2.5,
            "panda_joint5": 0.0,
            "panda_joint6": 1.0,
            "panda_joint7": 0.0,
            "finger_joint": _FINGER_OPEN,
            ".*_inner_finger_joint": _FINGER_OPEN,
            ".*_inner_knuckle_joint": _FINGER_OPEN,
            "right_outer_knuckle_joint": _FINGER_OPEN,
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
        # Active drive — ``right_outer_knuckle_joint`` is its closed-loop mimic.
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=1650.0,
            velocity_limit_sim=10.0,
            stiffness=17.0,
            damping=0.02,
        ),
        # Low-stiffness parallel-grasp enablement on the inner finger pads.
        "gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=10.0,
            stiffness=0.2,
            damping=0.001,
        ),
        # Passive loop-closure joints — zero PD, driven kinematically.
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_knuckle_joint", "right_outer_knuckle_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
    # The frankarobotiq USD declares ArticulationRootAPI on ``/Root/arm`` and
    # its defaultPrim is ``/Root``; ``articulation_root_prim_path=None`` lets
    # Isaac Lab auto-detect the authored articulation root after spawn.
    articulation_root_prim_path=None,
    soft_joint_pos_limit_factor=1.0,
)


FRANKA_ROBOTIQ_PINK_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="robot",
    base_link_name="root",
    num_hand_joints=0,
    show_ik_warnings=True,
    urdf_path=f"{MAGICSIM_ASSETS}/Robots/URDF/frankarobotiq.urdf",
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        FrameTask(
            "panda_link8",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        NullSpacePostureTask(
            cost=0.5,
            lm_damping=1,
            controlled_frames=["panda_link8"],
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
class FrankaRobotiqActionsCfg(ManipulatorActionsCfg):
    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "arm_action": {
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=["panda_joint.*"],
            ),
            "ik_abs": mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_link8",
                controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="pinv"
                ),
                body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
                    pos=[0.0, 0.0, 0.0]
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
                target_eef_link_names={"eef": "panda_link8"},
                action_space=torch.tensor(
                    [
                        [-0.6, -0.6, 0.0, -1.0, -1.0, -1.0, -1.0],
                        [0.6, 0.6, 1.2, 1.0, 1.0, 1.0, 1.0],
                    ]
                ),
                controller=FRANKA_ROBOTIQ_PINK_IK_CONTROLLER_CFG,
                relative_to_base=False,
            ),
            "ik_curobo": CuroboIKActionCfg(
                joint_names=["panda_joint.*"],
                robot_cfg_file="magicsim_frankarobotiq.yml",
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
            # Only ``finger_joint`` is actively PD-driven; the loop-closure
            # propagates motion to the passive mimic joints.
            "binary": mdp.BinaryJointPositionActionCfg(
                joint_names=["finger_joint"],
                open_command_expr={"finger_joint": _FINGER_OPEN},
                close_command_expr={"finger_joint": _FINGER_CLOSE},
            ),
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=["finger_joint"],
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()


class FrankaRobotiqFrameCfg(FrameSensorCfg):
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
        # The frankarobotiq USD nests links under ``/Root/arm/...``; after
        # spawn the articulation appears at ``{robot_prim_path}/arm/...``.
        self.target_frames[0].prim_path = self.robot_prim_path + "/arm/panda_link8"
        self.prim_path = self.robot_prim_path + "/arm/panda_link0"


@configclass
class FrankaRobotiqObsCfg(ManipulatorObsCfg):
    gripper_pos: ObsTerm = MISSING

    def __post_init__(self):
        self.gripper_pos = ObsTerm(
            func=mdp.gripper_pos, params={"robot_cfg": SceneEntityCfg(self.asset_name)}
        )
        super().__post_init__()


@configclass
class FrankaRobotiqPlannerCfg(ManipulatorPlannerCfg):
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
class FrankaRobotiqCfg(ManipulatorCfg):
    """Configuration for the Franka Panda with a Robotiq 2F-85 gripper."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    arm_action_name: str = "joint_pos"
    eef_action_name: str = "binary"
    frame_name: str = "ee_frame"
    robot: ArticulationCfg = MISSING
    action: FrankaRobotiqActionsCfg = MISSING
    ee_frame: FrankaRobotiqFrameCfg = MISSING
    obs: FrankaRobotiqObsCfg = MISSING
    planner: FrankaRobotiqPlannerCfg = FrankaRobotiqPlannerCfg()

    def __post_init__(self):
        self.robot: ArticulationCfg = MAGIC_FRANKA_ROBOTIQ_CFG
        self.robot.prim_path = self.prim_path
        self.action: FrankaRobotiqActionsCfg = FrankaRobotiqActionsCfg(
            asset_name=self.asset_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = FrankaRobotiqFrameCfg(robot_prim_path=self.prim_path)
        self.obs: FrankaRobotiqObsCfg = FrankaRobotiqObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        super().__post_init__()
