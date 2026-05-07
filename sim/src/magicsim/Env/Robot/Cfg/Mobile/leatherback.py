from typing import Dict
import torch
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.managers import ActionTermCfg as ActionTerm

from magicsim.Env.Robot.Cfg.Mobile.Mobile import (
    MobileActionsCfg,
    MobileCfg,
    MobileObsCfg,
    MobilePlannerCfg,
)
import magicsim.Env.Robot.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

LEATHERBACK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Leatherback/leatherback.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            max_linear_velocity=50.0,
            max_angular_velocity=50.0,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
        joint_pos={
            # 轮子旋转关节
            "Wheel__Knuckle__Front_Left": 0.0,
            "Wheel__Knuckle__Front_Right": 0.0,
            "Wheel__Upright__Rear_Right": 0.0,
            "Wheel__Upright__Rear_Left": 0.0,
            # 转向关节（初始直行）
            "Knuckle__Upright__Front_Right": 0.0,
            "Knuckle__Upright__Front_Left": 0.0,
            # --- [修复] 为避震器关节添加有效的初始位置 ---
            "Shock__Rear_Right": -0.03,  # 有效范围: [-0.050, -0.010]
            "Shock__Rear_Left": -0.03,  # 有效范围: [-0.050, -0.010]
            "Shock__Front_Right": 0.03,  # 有效范围: [0.010, 0.050]
            "Shock__Front_Left": 0.03,  # 有效范围: [0.010, 0.050]
            # --- 修复结束 ---
        },
    ),
    actuators={
        # 四轮驱动执行器（速度控制）
        "throttle": ImplicitActuatorCfg(
            joint_names_expr=["Wheel.*"],
            effort_limit_sim=100.0,  # 驱动扭矩 (N⋅m)
            velocity_limit_sim=50.0,  # 最大角速度 (rad/s)
            stiffness=0.0,  # 速度控制模式
            damping=10.0,  # 阻尼
        ),
        # 前轮转向执行器（位置控制）
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["Knuckle__Upright__Front.*"],
            effort_limit_sim=50.0,  # 转向扭矩 (N⋅m)
            velocity_limit_sim=10.0,  # 转向速度 (rad/s)
            stiffness=100.0,  # 位置控制刚度
            damping=5.0,  # 阻尼
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of NVIDIA Leatherback 4-wheel steering robot."""


# ================================
#  Action Configuration
# ================================


@configclass
class leatherbackActionsCfg(MobileActionsCfg):
    """Action specifications for the MDP."""

    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "base_action": {
            "ackermann_drive": mdp.AckermannSteeringActionCfg(
                asset_name="robot",
                action_space=torch.tensor(
                    [
                        # Minima
                        [
                            -0.5,  # 0: 油门命令 [-1, 1]，负值表示倒车
                            -2.0,  # 1: 转向命令 [-1, 1]，正值表示左转
                        ],
                        # Maxima
                        [
                            0.5,  # 0: 油门命令 [-1, 1]，正值表示前进
                            2.0,  # 1: 转向命令 [-1, 1]，负值表示右转
                        ],
                    ],
                ),
                wheel_joint_names=[
                    "Wheel__Knuckle__Front_Left",
                    "Wheel__Knuckle__Front_Right",
                    "Wheel__Upright__Rear_Right",
                    "Wheel__Upright__Rear_Left",
                ],
                steering_joint_names=[
                    "Knuckle__Upright__Front_Right",
                    "Knuckle__Upright__Front_Left",
                ],
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()


# ================================
#  Composite Robot Configuration
# ================================
@configclass
class LeatherbackPlannerCfg(MobilePlannerCfg):
    base_action_dim: Dict[str, int] = {
        "dwb": 2,
    }
    base_action_space: Dict[str, torch.Tensor] = {
        "dwb": torch.tensor(
            [
                [-0.5, -2.0],
                [0.5, 2.0],
            ],
        ),
    }


@configclass
class LeatherbackObsCfg(MobileObsCfg):
    front_steer: ObsTerm = MISSING

    def __post_init__(self):
        self.front_steer = ObsTerm(
            func=mdp.front_steer, params={"robot_cfg": SceneEntityCfg(self.asset_name)}
        )
        super().__post_init__()


@configclass
class LeatherbackCfg(MobileCfg):
    """Configuration for a mobile robot: leatherback"""

    prim_path: str = MISSING
    asset_name: str = "robot"
    base_action_name: str = "ackermann_drive"

    robot: ArticulationCfg = MISSING
    planner: LeatherbackPlannerCfg = LeatherbackPlannerCfg()
    # Subconfigs
    action: leatherbackActionsCfg = MISSING
    obs: LeatherbackObsCfg = MISSING

    def __post_init__(self):
        self.robot: ArticulationCfg = LEATHERBACK_CFG

        # set prim paths
        self.robot.prim_path = self.prim_path
        # define actions and frames
        self.action = leatherbackActionsCfg(
            asset_name=self.asset_name,
            base_action_name=self.base_action_name,
        )

        # observation configuration
        self.obs = LeatherbackObsCfg(asset_name=self.asset_name)
