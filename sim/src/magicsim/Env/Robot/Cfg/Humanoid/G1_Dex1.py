"""G1 humanoid with dex1_1 parallel-jaw grippers replacing dex3-1 hands.

Aligned with ``g1_new.usd`` everywhere except the wrists:

* spawn USD ``g1_dex1.usd`` — built by transplanting the dex1_1 hand subtree
  twice (left + right) into a copy of g1_new.usd via
  ``Script/Robot/_build_g1_dex1_usd.py``. All g1_new body materials/visuals
  are preserved exactly. The dex1 ``base_link`` is renamed to
  ``<side>_hand_palm_link`` so existing G1 plumbing (Pink IK, obs cfg) works
  unchanged.
* DOF list: 29 G1 body joints + 4 dex1 prismatic joints
  (``<side>_dex1_finger_joint_{1,2}``) — replaces the 14 dex3 finger joints.
* Actions: subclass :class:`G1ActionsCfg`; reuse ``base_action`` and
  ``arm_action`` (Pink IK / wbc / joint_pos). ``eef_action`` is replaced with
  a 2-dim ``binary`` action (one float per side, >=0 open / <0 close — Franka
  pattern) over the 4 dex1 prismatic joints.
* Obs / Pink IK / planner / head camera: reused from :mod:`...G1` directly.
"""

from __future__ import annotations

import copy
from dataclasses import MISSING, field, fields
from typing import Dict

from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from magicsim import MAGICSIM_ASSETS
from magicsim.Env.Robot.Cfg.Humanoid.Humanoid import HumanoidCfg
from magicsim.Env.Robot.mdp.actions_cfg import (
    BinaryJointPositionActionCfg,
    MultipleBinaryJointPositionActionCfg,
)

from magicsim.Env.Robot.Cfg.Humanoid.G1 import (
    G1ActionsCfg,
    G1HeadCameraCfg,
    G1ObsCfg,
    G1PlannerCfg,
    G1_PINK_IK_CONTROLLER_CFG,
    MAGIC_G1_CFG,  # used as the actuator template
    G1_DOF_NAMES as _G1_DOF_NAMES_FULL,
)


# Pink IK URDF for the dex1 articulation (mesh-stripped, like ``g1.urdf``).
# Same kinematic chain as G1 but with the 14 dex3 finger joints replaced by
# the 4 dex1 prismatic joints — required so pinocchio doesn't enumerate
# joints that no longer exist in the live IsaacLab articulation.
G1_DEX1_PINK_IK_CONTROLLER_CFG = copy.copy(G1_PINK_IK_CONTROLLER_CFG)
G1_DEX1_PINK_IK_CONTROLLER_CFG.urdf_path = f"{MAGICSIM_ASSETS}/Robots/URDF/g1_dex1.urdf"


# =========================================================================
# 1. DOF names — body 29 + 4 dex1 prismatic (drops the 14 dex3 finger joints)
# =========================================================================
G1_DEX1_BODY_DOF_NAMES = [n for n in _G1_DOF_NAMES_FULL if "_hand_" not in n]
G1_DEX1_GRIPPER_DOF_NAMES = [
    "left_dex1_finger_joint_1",
    "left_dex1_finger_joint_2",
    "right_dex1_finger_joint_1",
    "right_dex1_finger_joint_2",
]
G1_DEX1_DOF_NAMES = G1_DEX1_BODY_DOF_NAMES + G1_DEX1_GRIPPER_DOF_NAMES


# =========================================================================
# 2. ArticulationCfg — clone MAGIC_G1_CFG, swap USD + actuator dict
# =========================================================================
def _build_articulation_cfg() -> ArticulationCfg:
    body = MAGIC_G1_CFG.actuators["all"]

    def _filter(d):
        return {k: v for k, v in d.items() if "_hand_" not in k}

    body_stiff = _filter(body.stiffness)
    body_damp = _filter(body.damping)
    body_arm = _filter(body.armature)
    body_fric = _filter(body.friction)
    body_eff = _filter(body.effort_limit_sim)
    body_vel = _filter(body.velocity_limit_sim)

    grip = {n: 2000.0 for n in G1_DEX1_GRIPPER_DOF_NAMES}
    grip_d = {n: 100.0 for n in G1_DEX1_GRIPPER_DOF_NAMES}
    grip_a = {n: 0.01 for n in G1_DEX1_GRIPPER_DOF_NAMES}
    grip_f = {n: 0.0 for n in G1_DEX1_GRIPPER_DOF_NAMES}
    grip_e = {n: 20.0 for n in G1_DEX1_GRIPPER_DOF_NAMES}
    grip_v = {n: 0.2 for n in G1_DEX1_GRIPPER_DOF_NAMES}

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{MAGICSIM_ASSETS}/Robots/g1_dex1.usd",
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
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, -1.38, 0.78),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                **MAGIC_G1_CFG.init_state.joint_pos,
                # Grippers open (joint range [-0.02, 0.0245])
                "left_dex1_finger_joint_1": 0.0245,
                "left_dex1_finger_joint_2": 0.0245,
                "right_dex1_finger_joint_1": 0.0245,
                "right_dex1_finger_joint_2": 0.0245,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=G1_DEX1_DOF_NAMES,
                stiffness={**body_stiff, **grip},
                damping={**body_damp, **grip_d},
                armature={**body_arm, **grip_a},
                friction={**body_fric, **grip_f},
                effort_limit_sim={**body_eff, **grip_e},
                velocity_limit_sim={**body_vel, **grip_v},
            ),
        },
    )


MAGIC_G1_DEX1_CFG = _build_articulation_cfg()


# =========================================================================
# 3. Actions — subclass G1ActionsCfg, override only eef_action
# =========================================================================
_DEX1_OPEN = 0.0245
_DEX1_CLOSE = -0.02


# configclass converts mutable defaults to ``field(default_factory=...)``, so
# the class attribute ``G1ActionsCfg.available_action`` is no longer a dict —
# we have to fetch it via the dataclass field and call its factory.
_G1_AVAIL_FACTORY = next(
    f for f in fields(G1ActionsCfg) if f.name == "available_action"
).default_factory


def _g1_dex1_available_action() -> Dict[str, Dict[str, ActionTerm]]:
    """Fresh ``available_action`` per instance — inherits G1's base/arm
    options, retargets Pink IK to the dex1 URDF, swaps in a 2-dim binary
    eef for the dex1 jaws."""
    avail = _G1_AVAIL_FACTORY()
    # Point Pink IK at the dex1 URDF (G1's points to ``g1.urdf`` which has the
    # 14 dex3 finger joints — those are gone from the dex1 articulation).
    if "ik_abs" in avail.get("arm_action", {}):
        avail["arm_action"]["ik_abs"].controller = G1_DEX1_PINK_IK_CONTROLLER_CFG
    # Route WBC through the dex1-aware Homie config (33-joint table).
    if "wbc" in avail.get("base_action", {}):
        avail["base_action"]["wbc"].robot_type = "g1_dex1"
    avail["eef_action"] = {
        # 2-dim: one float per side. >= 0 -> open, < 0 -> close.
        "binary": MultipleBinaryJointPositionActionCfg(
            joint_groups=[
                BinaryJointPositionActionCfg(
                    joint_names=[
                        "left_dex1_finger_joint_1",
                        "left_dex1_finger_joint_2",
                    ],
                    open_command_expr={
                        "left_dex1_finger_joint_1": _DEX1_OPEN,
                        "left_dex1_finger_joint_2": _DEX1_OPEN,
                    },
                    close_command_expr={
                        "left_dex1_finger_joint_1": _DEX1_CLOSE,
                        "left_dex1_finger_joint_2": _DEX1_CLOSE,
                    },
                ),
                BinaryJointPositionActionCfg(
                    joint_names=[
                        "right_dex1_finger_joint_1",
                        "right_dex1_finger_joint_2",
                    ],
                    open_command_expr={
                        "right_dex1_finger_joint_1": _DEX1_OPEN,
                        "right_dex1_finger_joint_2": _DEX1_OPEN,
                    },
                    close_command_expr={
                        "right_dex1_finger_joint_1": _DEX1_CLOSE,
                        "right_dex1_finger_joint_2": _DEX1_CLOSE,
                    },
                ),
            ],
        ),
    }
    return avail


@configclass
class G1Dex1ActionsCfg(G1ActionsCfg):
    """Inherits ``base_action`` (wbc / joint_pos) and ``arm_action`` (ik_abs /
    joint_pos / joint_pos_vel) from G1; replaces ``eef_action`` with 2-dim
    binary open/close on the dex1 prismatic jaws (Franka style)."""

    available_action: Dict[str, Dict[str, ActionTerm]] = field(
        default_factory=_g1_dex1_available_action
    )

    def __post_init__(self):
        super().__post_init__()


# =========================================================================
# 4. Top-level robot cfg — reuses G1's obs / planner / head camera unchanged
# =========================================================================
@configclass
class G1_Dex1Cfg(HumanoidCfg):
    """G1 humanoid with dex1_1 grippers (drop-in replacement for G1Cfg)."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    base_action_name: str = MISSING
    arm_action_name: str = MISSING
    eef_action_name: str = MISSING
    robot: ArticulationCfg = MISSING
    action: G1Dex1ActionsCfg = MISSING
    sensor: G1HeadCameraCfg = G1HeadCameraCfg()
    obs: G1ObsCfg = MISSING
    planner: G1PlannerCfg = G1PlannerCfg()

    def __post_init__(self):
        self.robot = MAGIC_G1_DEX1_CFG
        self.robot.prim_path = self.prim_path
        self.action = G1Dex1ActionsCfg(
            asset_name=self.asset_name,
            base_action_name=self.base_action_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.sensor = G1HeadCameraCfg(robot_prim_path=self.prim_path)
        self.obs = G1ObsCfg(
            asset_name=self.asset_name, sensor_name=f"{self.asset_name}_head_camera"
        )
        super().__post_init__()


__all__ = [
    "MAGIC_G1_DEX1_CFG",
    "G1_DEX1_DOF_NAMES",
    "G1_DEX1_BODY_DOF_NAMES",
    "G1_DEX1_GRIPPER_DOF_NAMES",
    "G1_DEX1_PINK_IK_CONTROLLER_CFG",
    "G1Dex1ActionsCfg",
    "G1_Dex1Cfg",
]
