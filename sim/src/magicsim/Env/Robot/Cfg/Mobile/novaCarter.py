from typing import Dict
import torch
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.managers import ActionTermCfg as ActionTerm

from magicsim.Env.Robot.Cfg.Mobile.Mobile import (
    MobileActionsCfg,
    MobileCfg,
    novaCarterObsCfg,
    MobilePlannerCfg,
)
import magicsim.Env.Robot.mdp as mdp

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg

NOVA_CARTER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/NovaCarter/nova_carter.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            # 使用实际存在的关节名称
            "joint_wheel_left": 0.0,
            "joint_wheel_right": 0.0,
            # 可选：如果需要初始化其他关节
            # "joint_caster_base": 0.0,
            # "joint_swing_left": 0.0,
            # "joint_swing_right": 0.0,
            # "joint_caster_left": 0.0,
            # "joint_caster_right": 0.0,
        },
    ),
    actuators={
        "wheel_drive": ImplicitActuatorCfg(
            # 使用正则表达式匹配左右轮子
            joint_names_expr=[
                "joint_wheel_.*"
            ],  # 匹配 joint_wheel_left 和 joint_wheel_right
            effort_limit_sim=100.0,
            velocity_limit_sim=20.0,
            stiffness=0.0,  # 差速轮子通常不使用高刚度控制
            damping=10.0,  # 添加一些阻尼以提高稳定性
        ),
        # 可选：如果需要控制脚轮
        # "caster": ImplicitActuatorCfg(
        #     joint_names_expr=["joint_caster_.*", "joint_swing_.*"],
        #     effort_limit_sim=50.0,
        #     velocity_limit_sim=10.0,
        #     stiffness=0.0,
        #     damping=5.0,
        # ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of the Nova Carter mobile robot."""


# ================================
#  Action Configuration
# ================================


@configclass
class NovaCarterActionsCfg(MobileActionsCfg):
    """Action specifications for the MDP."""

    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "base_action": {
            "differential_drive": mdp.DifferentialActionCfg(
                action_space=torch.tensor(
                    [
                        [-1, -0.4],
                        [1, 0.4],
                    ]
                ),
                joint_names=["joint_wheel_left", "joint_wheel_right"],
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()


# ================================
#  Composite Robot Configuration
# ================================


@configclass
class NovaCarterPlannerCfg(MobilePlannerCfg):
    base_action_dim: Dict[str, int] = {
        "dwb_differential": 2,
    }
    base_action_space: Dict[str, torch.Tensor] = {
        "dwb_differential": torch.tensor(
            [
                [-1, -0.4],
                [1, 0.4],
            ],
        ),
    }


@configclass
class NovaCarterCfg(MobileCfg):
    """Configuration for a mobile manipulator: Ridgeback + Franka"""

    prim_path: str = MISSING
    asset_name: str = "novaCarter"
    base_action_name: str = MISSING
    robot: ArticulationCfg = MISSING
    # Subconfigs
    action: NovaCarterActionsCfg = MISSING
    obs: novaCarterObsCfg = MISSING
    planner: NovaCarterPlannerCfg = NovaCarterPlannerCfg()

    def __post_init__(self):
        self.robot: ArticulationCfg = NOVA_CARTER_CFG
        # set prim paths
        self.robot.prim_path = self.prim_path
        # define actions and frames
        self.action = NovaCarterActionsCfg(
            asset_name=self.asset_name,
            base_action_name=self.base_action_name,
        )
        # observation configuration
        self.obs = novaCarterObsCfg(asset_name=self.asset_name)
