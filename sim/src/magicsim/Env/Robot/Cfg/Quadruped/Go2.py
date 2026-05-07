from typing import Dict

import torch
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from dataclasses import MISSING
from isaaclab.actuators import DCMotorCfg
from magicsim.Env.Robot.Cfg.Quadruped.Quadruped import (
    QuadrupedActionsCfg,
    QuadrupedCfg,
    QuadrupedPlannerCfg,
)
from magicsim.Env.Robot.mdp.actions_cfg import (
    HolonomicForQuadrupedActionCfg,
    JointPositionToLimitsActionCfg,
)
from magicsim.Env.Robot.Cfg.Quadruped.mdp.go2_wbc_action_cfg import (
    Go2WBCActionCfg,
)
from magicsim.Env.Robot.Cfg.Quadruped.mdp.agile_quadruped_action_cfg import (
    AgileQuadrupedActionCfg,
)
from magicsim.Env.Robot import mdp
from magicsim.Env.Robot.Cfg.Base import RobotObsCfg

print("ISAACLAB_NUCLEUS_DIR", ISAACLAB_NUCLEUS_DIR)
UNITREE_GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)


# ================================
#  Action Configuration
# ================================


@configclass
class Go2ActionsCfg(QuadrupedActionsCfg):
    """Action specifications for the MDP."""

    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "base_action": {
            "wbc": Go2WBCActionCfg(
                joint_names=[
                    ".*_hip_joint",
                    ".*_thigh_joint",
                    ".*_calf_joint",
                ],
                num_wbc_joints=12,
                action_space=torch.tensor(
                    [
                        [
                            -2.0,  # vx_min
                            -2.0,  # vy_min
                            0.2,  # base_height_min
                            -2.0,  # yaw_rate_min
                        ],
                        [
                            2.0,  # vx_max
                            2.0,  # vy_max
                            0.6,  # base_height_max
                            2.0,  # yaw_rate_max
                        ],
                    ]
                ),
                wbc_joint_yaml_path="/home/magic/shuyang/MagicSim/src/magicsim/Env/Robot/Cfg/Quadruped/mdp/go2_wbc.yaml",
            ),
            "agile": AgileQuadrupedActionCfg(
                joint_names=[
                    ".*_hip_joint",
                    ".*_thigh_joint",
                    ".*_calf_joint",
                ],
                num_wbc_joints=12,
                action_space=torch.tensor(
                    [
                        [
                            -1.0,  # vx_min
                            -1.0,  # vy_min
                            -1.0,  # wz_min
                            0.2,  # base_height_min
                        ],
                        [
                            1.0,  # vx_max
                            1.0,  # vy_max
                            1.0,  # wz_max
                            0.6,  # base_height_max
                        ],
                    ]
                ),
                policy_output_scale=0.25,
                policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/Quadruped/go2_locomotion.pt",
            ),
            "holonomic": HolonomicForQuadrupedActionCfg(
                joint_names=[
                    ".*_hip_joint",
                    ".*_thigh_joint",
                    ".*_calf_joint",
                ],
                action_space=torch.tensor(
                    [
                        [-2.0, -2.0],  # [vx_min, vy_min]
                        [2.0, 2.0],  # [vx_max, vy_max]
                    ]
                ),
            ),
            "joint_pos": JointPositionToLimitsActionCfg(
                joint_names=[
                    ".*_hip_joint",
                    ".*_thigh_joint",
                    ".*_calf_joint",
                ],
                num_joints=12,
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()


# ================================
#  Observation Configuration
# ================================


@configclass
class Go2ObsCfg(RobotObsCfg):
    """Observation specifications for the MDP."""

    base_ang_vel: ObsTerm = MISSING
    base_lin_vel: ObsTerm = MISSING

    def __post_init__(self):
        """Observations for quadruped robot."""
        # Base velocity observations
        self.base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg(self.asset_name)},
        )
        self.base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg(self.asset_name)},
        )
        super().__post_init__()


# ================================
#  Planner Configuration
# ================================


@configclass
class Go2PlannerCfg(QuadrupedPlannerCfg):
    """Planner configuration for Go2 quadruped robot."""

    quadruped_action_dim: Dict[str, int] = {
        "dwb_quadruped": 2,  # [vx, vy]
    }
    quadruped_action_space: Dict[str, torch.Tensor] = {
        "dwb_quadruped": torch.tensor(
            [
                [-2.0, -2.0],  # [vx_min, vy_min]
                [2.0, 2.0],  # [vx_max, vy_max]
            ],
        ),
    }
    eef_action_dim: Dict[str, int] = {}
    eef_action_space: Dict[str, torch.Tensor] = {}


# ================================
#  Composite Robot Configuration
# ================================


@configclass
class Go2Cfg(QuadrupedCfg):
    """Configuration for the Go2 robot."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    base_action_name: str = MISSING
    robot: ArticulationCfg = MISSING
    base_action: Go2ActionsCfg = MISSING
    action: Go2ActionsCfg = MISSING
    obs: Go2ObsCfg = MISSING
    planner: Go2PlannerCfg = Go2PlannerCfg()

    def __post_init__(self):
        self.robot: ArticulationCfg = UNITREE_GO2_CFG
        self.robot.prim_path = self.prim_path
        self.action: Go2ActionsCfg = Go2ActionsCfg(
            asset_name=self.asset_name,
            base_action_name=self.base_action_name,
        )
        self.obs: Go2ObsCfg = Go2ObsCfg(asset_name=self.asset_name)
        super().__post_init__()
