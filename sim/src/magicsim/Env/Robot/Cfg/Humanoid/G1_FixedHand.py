"""G1 humanoid with rigid rubber_hand end-effectors (no articulated fingers).

Aligned with ``g1_new.usd`` everywhere except the wrists:

* spawn USD ``g1_fixed_hand.usd`` — built by transplanting the rubber_hand
  link bodies twice into a copy of g1_new.usd via
  ``Script/Robot/_build_g1_fixed_hand_usd.py``. All g1_new body materials/
  visuals are preserved exactly. The rubber_hand prim is renamed to
  ``<side>_hand_palm_link`` so existing G1 plumbing (Pink IK, obs cfg) works
  unchanged. Material binding is overridden to a new pure-black
  ``RubberBlack`` material (matching the user's "黑色" requirement).
* DOF list: 29 G1 body joints exactly — no gripper or finger joints.
* Actions: subclass :class:`G1ActionsCfg`; reuse ``base_action`` (wbc /
  joint_pos) and ``arm_action`` (ik_abs / joint_pos / joint_pos_vel) verbatim.
  ``eef_action`` is removed — there are no end-effector DoFs to drive.
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
from magicsim.Env.Robot.mdp.actions_cfg import JointPositionToLimitsActionCfg

from magicsim.Env.Robot.Cfg.Humanoid.G1 import (
    G1ActionsCfg,
    G1HeadCameraCfg,
    G1ObsCfg,
    G1PlannerCfg,
    G1_PINK_IK_CONTROLLER_CFG,
    MAGIC_G1_CFG,
    G1_DOF_NAMES as _G1_DOF_NAMES_FULL,
    g1_move_strategy as _g1_move_strategy,
)


# =========================================================================
# G1_FixedHand-specific RetractMoveL move_strategy
# =========================================================================
# Same algorithm as G1's stock ``g1_move_strategy`` with one LocoBox-tuned
# default:
#   * ``lock_fwd_offset = 0.45`` — pull the locked-XY anchor 45 cm
#     **behind** the grasp point (formula: ``locked_xy = grasp_xy -
#     lock_fwd_offset * fwd``; positive = behind, gives the bent torso
#     clearance). Stock G1 default is 0.3.
# ``clip_height`` is intentionally not passed — pre_grasp is full-body
# locomotion and shouldn't be height-clamped; the bend phase handles
# the squat-and-reach via locked-base MotionGen with a wider waist URDF.
# Other planner channels (PController helper, base/arm/eef dims, dehatch)
# are inherited from G1PlannerCfg unchanged. Stock G1 is untouched.
def g1_fixed_hand_move_strategy(
    trajectory,
    robot_state,
    hand_id: int = -1,
    lock_xy_steps: int = 10,
    num_rotation_steps: int = 50,
    lock_fwd_offset: float = 0.45,
):
    return _g1_move_strategy(
        trajectory,
        robot_state,
        hand_id=hand_id,
        lock_xy_steps=lock_xy_steps,
        num_rotation_steps=num_rotation_steps,
        lock_fwd_offset=lock_fwd_offset,
    )


@configclass
class G1FixedHandPlannerCfg(G1PlannerCfg):
    """G1 planner cfg with the fixed-hand move_strategy bound."""

    move_strategy = staticmethod(g1_fixed_hand_move_strategy)


# Pink IK URDF for the fixed-hand articulation (mesh-stripped, like ``g1.urdf``).
# Same kinematic chain as G1 minus the 14 dex3 finger joints — required so
# pinocchio doesn't enumerate joints that no longer exist in the live
# articulation.
G1_FIXED_HAND_PINK_IK_CONTROLLER_CFG = copy.copy(G1_PINK_IK_CONTROLLER_CFG)
G1_FIXED_HAND_PINK_IK_CONTROLLER_CFG.urdf_path = (
    f"{MAGICSIM_ASSETS}/Robots/URDF/g1_fixed_hand.urdf"
)


# =========================================================================
# 1. DOF names — body 29 only (drops the 14 dex3 finger joints)
# =========================================================================
G1_FIXED_HAND_DOF_NAMES = [n for n in _G1_DOF_NAMES_FULL if "_hand_" not in n]


# =========================================================================
# 2. ArticulationCfg — clone MAGIC_G1_CFG, swap USD + filter actuator dict
# =========================================================================
def _build_articulation_cfg() -> ArticulationCfg:
    body = MAGIC_G1_CFG.actuators["all"]

    def _filter(d):
        return {k: v for k, v in d.items() if "_hand_" not in k}

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{MAGICSIM_ASSETS}/Robots/g1_fixed_hand.usd",
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
                k: v
                for k, v in MAGIC_G1_CFG.init_state.joint_pos.items()
                if "_hand_" not in k
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=G1_FIXED_HAND_DOF_NAMES,
                stiffness=_filter(body.stiffness),
                damping=_filter(body.damping),
                armature=_filter(body.armature),
                friction=_filter(body.friction),
                effort_limit_sim=_filter(body.effort_limit_sim),
                velocity_limit_sim=_filter(body.velocity_limit_sim),
            ),
        },
    )


MAGIC_G1_FIXED_HAND_CFG = _build_articulation_cfg()


# =========================================================================
# 3. Actions — subclass G1ActionsCfg, drop eef_action, retarget Pink IK +
# WBC to the fixed-hand variants.
# =========================================================================
_G1_AVAIL_FACTORY = next(
    f for f in fields(G1ActionsCfg) if f.name == "available_action"
).default_factory


def _g1_fixed_hand_available_action() -> Dict[str, Dict[str, ActionTerm]]:
    """Fresh ``available_action`` per instance — inherits G1's base/arm,
    retargets Pink IK to the fixed-hand URDF, retargets WBC to the
    fixed-hand Homie config, replaces ``eef_action`` with a 1-dim no-op
    placeholder.

    The placeholder is required because :class:`PlannerManager` registers
    one gym Box per action channel (base / arm / eef) and chokes on a
    missing ``eef_action_space``. The placeholder targets ``waist_yaw_joint``
    with NaN-fallback semantics — passing NaN from the driver makes the
    action term hold the joint at its current position, which is then
    overwritten the next physics step by the WBC / Pink IK channels that
    also drive that joint. Net effect: zero behaviour change vs. having
    no eef channel.
    """
    avail = _G1_AVAIL_FACTORY()
    if "ik_abs" in avail.get("arm_action", {}):
        avail["arm_action"]["ik_abs"].controller = G1_FIXED_HAND_PINK_IK_CONTROLLER_CFG
    if "wbc" in avail.get("base_action", {}):
        avail["base_action"]["wbc"].robot_type = "g1_fixed_hand"
    avail["eef_action"] = {
        # 2-dim no-op so PlannerManager has a real Box space to register.
        # Width must be divisible by ``max_eef_num=2`` (one slot per hand).
        # NaN-fallback in :class:`JointPositionToLimitsAction._handle_nan_actions`
        # short-circuits the whole term when every entry is NaN, so this
        # writes nothing and never collides with WBC / Pink IK targets that
        # also touch the waist joints.
        "placeholder": JointPositionToLimitsActionCfg(
            joint_names=["waist_yaw_joint", "waist_roll_joint"],
            num_joints=2,
        ),
    }
    return avail


@configclass
class G1FixedHandActionsCfg(G1ActionsCfg):
    """Inherits G1's base/arm options, drops eef_action."""

    available_action: Dict[str, Dict[str, ActionTerm]] = field(
        default_factory=_g1_fixed_hand_available_action
    )

    def __post_init__(self):
        super().__post_init__()


# =========================================================================
# 4. Top-level robot cfg
# =========================================================================
@configclass
class G1_FixedHandCfg(HumanoidCfg):
    """G1 humanoid with rigid rubber_hand end-effectors (no finger DoFs)."""

    prim_path: str = MISSING
    asset_name: str = "robot"
    base_action_name: str = MISSING
    arm_action_name: str = MISSING
    # Default to the placeholder; override in YAML if you ever wire real
    # finger DoFs onto a hand variant.
    eef_action_name: str | None = "placeholder"
    robot: ArticulationCfg = MISSING
    action: G1FixedHandActionsCfg = MISSING
    sensor: G1HeadCameraCfg = G1HeadCameraCfg()
    obs: G1ObsCfg = MISSING
    planner: G1FixedHandPlannerCfg = G1FixedHandPlannerCfg()

    def __post_init__(self):
        self.robot = MAGIC_G1_FIXED_HAND_CFG
        self.robot.prim_path = self.prim_path
        self.action = G1FixedHandActionsCfg(
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
    "MAGIC_G1_FIXED_HAND_CFG",
    "G1_FIXED_HAND_DOF_NAMES",
    "G1_FIXED_HAND_PINK_IK_CONTROLLER_CFG",
    "G1FixedHandActionsCfg",
    "G1FixedHandPlannerCfg",
    "G1_FixedHandCfg",
    "g1_fixed_hand_move_strategy",
]
