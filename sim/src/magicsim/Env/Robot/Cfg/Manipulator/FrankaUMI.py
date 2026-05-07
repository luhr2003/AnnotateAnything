from typing import Dict

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from magicsim.Env.Robot.Cfg.Manipulator.Franka import FrankaActionsCfg, FrankaObsCfg
from magicsim.Env.Robot.Cfg.Manipulator.Manipulator import (
    FrameSensorCfg,
    ManipulatorCfg,
    ManipulatorPlannerCfg,
)
from dataclasses import MISSING
from magicsim import MAGICSIM_ASSETS


FRANKA_UMI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/franka_umi.usd",
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
            effort_limit_sim=20.0,
            stiffness=400.0,
            damping=80.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


class FrankaUMIFrameCfg(FrameSensorCfg):
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
class FrankaPlannerCfg(ManipulatorPlannerCfg):
    arm_action_dim: Dict[str, int] = {
        "curobo": 7,
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
class FrankaUMICfg(ManipulatorCfg):
    """Configuration for the Franka robot."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    arm_action_name: str = "joint_pos"
    eef_action_name: str = "binary"
    frame_name: str = "ee_frame"
    robot: ArticulationCfg = MISSING
    action: FrankaActionsCfg = MISSING
    ee_frame: FrankaUMIFrameCfg = MISSING
    obs: FrankaObsCfg = MISSING
    planner: FrankaPlannerCfg = FrankaPlannerCfg()

    def __post_init__(self):
        self.robot: ArticulationCfg = FRANKA_UMI_CFG
        self.robot.prim_path = self.prim_path
        self.action: FrankaActionsCfg = FrankaActionsCfg(
            asset_name=self.asset_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.ee_frame = FrankaUMIFrameCfg(robot_prim_path=self.prim_path)
        self.obs: FrankaObsCfg = FrankaObsCfg(
            asset_name=self.asset_name, frame_name=self.frame_name
        )
        super().__post_init__()
