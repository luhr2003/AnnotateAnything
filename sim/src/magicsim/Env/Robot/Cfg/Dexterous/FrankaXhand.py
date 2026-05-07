from typing import Dict

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from pink.tasks import FrameTask
from magicsim.Env.Robot.mdp.pink_ik import NullSpacePostureTask
from magicsim.Env.Robot.mdp.pink_actions_cfg import (
    PinkIKControllerCfg,
    PinkInverseKinematicsActionCfg,
)
from magicsim.Env.Robot.Cfg.Manipulator.Franka import FrankaObsCfg
from magicsim.Env.Robot.Cfg.Dexterous.Dexterous import (
    FrameSensorCfg,
    DexterousCfg,
    DexterousActionsCfg,
    DexterousPlannerCfg,
)
import magicsim.Env.Robot.mdp as mdp
from dataclasses import MISSING
from magicsim import MAGICSIM_ASSETS


FRANKA_XHAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/panda_xhand_right.usd",
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
            # Panda arm joints
            "panda_joint1": 0.0,
            "panda_joint2": -1.3,
            "panda_joint3": 0.0,
            "panda_joint4": -2.5,
            "panda_joint5": 0.0,
            "panda_joint6": 1.0,
            "panda_joint7": 0.0,
            # XHand right hand joints (12 DOF)
            "right_hand_thumb_rota_joint1": 0.0,
            "right_hand_thumb_rota_joint2": 0.0,
            "right_hand_thumb_bend_joint": 0.0,
            "right_hand_index_bend_joint": 0.0,
            "right_hand_index_joint1": 0.0,
            "right_hand_index_joint2": 0.0,
            "right_hand_mid_joint1": 0.0,
            "right_hand_mid_joint2": 0.0,
            "right_hand_ring_joint1": 0.0,
            "right_hand_ring_joint2": 0.0,
            "right_hand_pinky_joint1": 0.0,
            "right_hand_pinky_joint2": 0.0,
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
        "xhand_fingers": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_.*"],
            effort_limit_sim=5.0,
            stiffness=200.0,
            damping=20.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# Pink IK: target frame is xhand_right_right_hand_link (XHand palm)
FRANKA_XHAND_PINK_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="robot",
    base_link_name="root",
    num_hand_joints=0,
    show_ik_warnings=True,
    urdf_path=f"{MAGICSIM_ASSETS}/Robots/URDF/panda_xhand.urdf",
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        FrameTask(
            "right_hand_link",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        NullSpacePostureTask(
            cost=0.5,
            lm_damping=1,
            controlled_frames=["right_hand_link"],
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
class FrankaXhandActionsCfg(DexterousActionsCfg):
    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "arm_action": {
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=["panda_joint.*"],
            ),
            "ik_abs": mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="right_hand_link",
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
            "ik_pink": PinkInverseKinematicsActionCfg(
                pink_controlled_joint_names=["panda_joint.*"],
                num_joints=7,
                hand_joint_names=None,
                target_eef_link_names={"eef": "right_hand_link"},
                action_space=torch.tensor(
                    [
                        [-0.6, -0.6, 0.0, -1.0, -1.0, -1.0, -1.0],
                        [0.6, 0.6, 1.2, 1.0, 1.0, 1.0, 1.0],
                    ]
                ),
                controller=FRANKA_XHAND_PINK_IK_CONTROLLER_CFG,
                relative_to_base=False,
            ),
        },
        "eef_action": {
            "joint_pos": mdp.JointPositionToLimitsActionCfg(
                joint_names=[
                    "right_hand_index_bend_joint",
                    "right_hand_index_joint1",
                    "right_hand_index_joint2",
                    "right_hand_mid_joint1",
                    "right_hand_mid_joint2",
                    "right_hand_pinky_joint1",
                    "right_hand_pinky_joint2",
                    "right_hand_ring_joint1",
                    "right_hand_ring_joint2",
                    "right_hand_thumb_bend_joint",
                    "right_hand_thumb_rota_joint1",
                    "right_hand_thumb_rota_joint2",
                ],
                preserve_order=True,
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()


class FrankaXhandFrameCfg(FrameSensorCfg):
    """Frame sensor for Franka XHand. Only end_effector (right_hand_link) is tracked.

    Note: FrameTransformer orders target_frames alphabetically by body name, so
    target_pos_w[:, 0, :] is end_effector only when it's the sole target frame.
    """

    def __post_init__(self):
        self.target_frames = [
            FrameTransformerCfg.FrameCfg(
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                ),
            ),
        ]
        self.visualizer_cfg.prim_path = (
            self.robot_prim_path + "/Visuals/FrameTransformer"
        )
        self.target_frames[0].prim_path = self.robot_prim_path + "/right_hand_link"
        self.prim_path = self.robot_prim_path + "/base_link"


@configclass
class FrankaXhandPlannerCfg(DexterousPlannerCfg):
    """Fixed-base: base_action_dim=0. Arm and eef use joint_pos by default."""

    base_action_dim: Dict[str, int] = {"default": 0}
    base_action_space: Dict[str, torch.Tensor] = {"default": None}
    arm_action_dim: Dict[str, int] = {
        "curobo": 7,
        "default": 7,
        "ik_pink": 7,
    }
    eef_action_dim: Dict[str, int] = {
        "default": 12,
    }
    arm_action_space: Dict[str, torch.Tensor] = {
        "curobo": torch.tensor(
            [
                [-1, -1, -1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ]
        ),
        "default": None,
        "ik_pink": torch.tensor(
            [
                [-0.6, -0.6, 0.0, -1.0, -1.0, -1.0, -1.0],
                [0.6, 0.6, 1.2, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
    }
    eef_action_space: Dict[str, torch.Tensor] = {
        "default": None,
    }


@configclass
class FrankaXhandCfg(DexterousCfg):
    """Configuration for the Franka arm with XHand dexterous hand (fixed base)."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    base_action_name: str | None = None
    arm_action_name: str = "joint_pos"
    eef_action_name: str = "joint_pos"
    frame_name: str = "ee_frame"
    robot: ArticulationCfg = MISSING
    action: FrankaXhandActionsCfg = MISSING
    ee_frame: FrankaXhandFrameCfg = MISSING
    obs: FrankaObsCfg = MISSING
    planner: FrankaXhandPlannerCfg = FrankaXhandPlannerCfg()

    def __post_init__(self):
        self.robot: ArticulationCfg = FRANKA_XHAND_CFG
        self.robot.prim_path = self.prim_path
        self.action: FrankaXhandActionsCfg = FrankaXhandActionsCfg(
            asset_name=self.asset_name,
            base_action_name=self.base_action_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = FrankaXhandFrameCfg(robot_prim_path=self.prim_path)
        self.obs: FrankaObsCfg = FrankaObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        super().__post_init__()
