"""G1_Sonic robot configuration —— SONIC hybrid IK pipeline 专用的 G1 变种。

和 MagicSim 原版 `G1.py` 的区别（**仅**这几处；其余全部复用）：
    1. Spawn 仍用 MagicSim 的 `g1_new.usd`（USD 路径保持）
    2. Actuator PD/armature/effort/velocity **改用 sonic 公式**（`ARMATURE × (10 Hz
       × 2π)² × 2` 的 stiffness，对应 sonic `gear_sonic/envs/manager_env/robots/g1.py:7-26`）
    3. Actuator 分 **5 组**（legs/feet/waist/waist_yaw/arms）+ 1 组 dex_fingers
       —— 与 sonic 训练侧一致；dex_fingers 保留 MagicSim 的原值
    4. `init_state.joint_pos` 改用 **sonic 训练初值**（hip_pitch=-0.312, knee=0.669,
       ankle_pitch=-0.363, elbow=0.6, shoulder_roll=±0.2, shoulder_pitch=0.2）——
       因为 SONIC decoder 是在这组 default_angles 下训练，换值会 OOD
    5. `G1SonicActionsCfg.base_action.wbc` = `SonicWBCActionCfg`（跑 SONIC hybrid
       pipeline，lower 12 DOF）
    6. `G1SonicActionsCfg.arm_action.ik_abs` = `SonicPinkInverseKinematicsActionCfg`
       —— Pink IK 子类，`apply_actions()` 末尾把 17-DOF 解写进 `SonicArmBuffer`

其他（Pink IK task 拓扑 / obs / strategies / eef_action）**全部复用** MagicSim G1。
"""

from __future__ import annotations

import math
from typing import Dict

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.utils import configclass

from magicsim import MAGICSIM_ASSETS
from dataclasses import MISSING as _MISSING

from magicsim.Env.Robot import mdp
from magicsim.Env.Robot.Cfg.Humanoid.Humanoid import (
    HumanoidActionsCfg,
    HumanoidCfg,
    HumanoidPlannerCfg,
)
from magicsim.Env.Robot.Cfg.Humanoid.mdp.sonic_wbc_action_cfg import SonicWBCActionCfg
from magicsim.Env.Robot.Cfg.Humanoid.mdp.sonic_pink_ik_action import (
    SonicPinkInverseKinematicsActionCfg,
)
from magicsim.Env.Robot.mdp.actions_cfg import (
    MultipleJointPositionToLimitsActionCfg,
    MultipleJointPositionToLimitsActionGroupCfg,
)
from magicsim.Env.Robot.Cfg.Humanoid.whole_body_controller.Sonic.G1.configs import (
    LEG_JOINTS_IL,
    PINK_CONTROLLED_JOINTS_IL,
)

# ---- 从 MagicSim G1 原样复用的东西（pink ik cfg / obs cfg / strategies）-----
from magicsim.Env.Robot.Cfg.Humanoid.G1 import (
    G1_DOF_NAMES,
    G1_PINK_IK_CONTROLLER_CFG,
    G1HeadCameraCfg,
    G1ObsCfg,
    g1_dehatch_strategy,
    g1_move_strategy,
    g1_postprocess_p_controller_action,
    wrap_to_pi,
)


# =========================================================================
# 1. sonic actuator PD 公式 —— 和 gear_sonic/envs/manager_env/robots/g1.py:7-26 对齐
# =========================================================================
ARMATURE_5020: float = 0.003609725
ARMATURE_7520_14: float = 0.010177520
ARMATURE_7520_22: float = 0.025101925
ARMATURE_4010: float = 0.00425

NATURAL_FREQ: float = 10 * 2.0 * math.pi  # 10 Hz
DAMPING_RATIO: float = 2.0

STIFFNESS_5020: float = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14: float = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22: float = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010: float = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020: float = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14: float = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22: float = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010: float = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ


# =========================================================================
# 2. Sonic init_state.joint_pos（训练侧 default_angles 的来源）
# =========================================================================
# 未列出的 joint 默认 0。和 `gear_sonic/envs/manager_env/robots/g1.py:225-233`
# 完全一致：hip_pitch/knee/ankle_pitch/elbow 非零，肩 roll/pitch = ±0.2 / 0.2。
_SONIC_G1_INIT_JOINT_POS: dict[str, float] = {
    # legs
    "left_hip_pitch_joint": -0.312,
    "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669,
    "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_ankle_pitch_joint": -0.363,
    # arms
    "left_elbow_joint": 0.6,
    "right_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_pitch_joint": 0.2,
    # 其余全部默认 0（waist / hip_roll / hip_yaw / ankle_roll / wrist / shoulder_yaw / dex fingers）
}


# =========================================================================
# 3. Dex finger actuator values（原样复用 MagicSim G1 的，不动）
# =========================================================================
_DEX_FINGER_JOINT_REGEX = [
    ".*_hand_index_\\d_joint",
    ".*_hand_middle_\\d_joint",
    ".*_hand_thumb_\\d_joint",
]
_DEX_FINGER_NAMES: list[str] = [n for n in G1_DOF_NAMES if "_hand_" in n]
assert len(_DEX_FINGER_NAMES) == 14


# =========================================================================
# 4. G1_SONIC_CFG: ArticulationCfg
# =========================================================================
G1_SONIC_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MAGICSIM_ASSETS}/Robots/g1_new.usd",
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
        # 位姿：0.78 保持和原 G1 一致。场景 reset 时由调用方再覆盖即可。
        pos=(0.8, -1.38, 0.78),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos=_SONIC_G1_INIT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # ---------- (1) Legs: 6 hip+knee joints on each side (sonic split) ----------
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 139.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 20.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_22,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_22,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_22,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        # ---------- (2) Feet ----------
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        # ---------- (3) Waist roll + pitch ----------
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        # ---------- (4) Waist yaw（独立一组，PD 和 waist roll/pitch 不同）----------
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        # ---------- (5) Arms（shoulders + elbows + wrists）----------
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
                ".*_wrist_pitch_joint": STIFFNESS_4010,
                ".*_wrist_yaw_joint": STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
                ".*_wrist_pitch_joint": DAMPING_4010,
                ".*_wrist_yaw_joint": DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
        # ---------- (6) Dex fingers（保留 MagicSim 原值）----------
        "dex_fingers": ImplicitActuatorCfg(
            joint_names_expr=_DEX_FINGER_JOINT_REGEX,
            effort_limit_sim=300.0,
            velocity_limit_sim=100.0,
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
            friction=0.0,
        ),
    },
)


# =========================================================================
# 5. G1SonicActionsCfg —— 用 SonicWBC + SonicPinkIK 替换 base_action / arm_action
# =========================================================================
# arm_action 必须插入在前，保证 Pink IK 先跑、SonicArmBuffer 先被写，
# 然后 base_action 的 SonicWBCAction.process_actions 读到新鲜 buffer。
# IsaacLab `ActionManager` 按 dict 插入顺序执行 action terms。


@configclass
class G1SonicActionsCfg(HumanoidActionsCfg):
    """SONIC hybrid IK 的 action 配置。

    只保留 base_action + arm_action 两类（和 G1 decoupled WBC 同）；
    eef_action / dex 等按需用时再从 G1.py 复制或继承。
    """

    available_action: Dict[str, Dict[str, ActionTerm]] = {
        # ---------- 先跑 arm_action：Pink IK 写 SonicArmBuffer ----------
        "arm_action": {
            "ik_abs": SonicPinkInverseKinematicsActionCfg(
                # 显式 17-joint list，按 `PINK_CONTROLLED_JOINTS_IL` 顺序 —— 不用 regex，
                # 减少 `find_joints()` 顺序不确定性（plan.md 关节序不变式 4）
                pink_controlled_joint_names=list(PINK_CONTROLLED_JOINTS_IL),
                num_joints=17,
                hand_joint_names=None,
                target_eef_link_names={
                    "right_wrist": "right_hand_palm_link",
                    "left_wrist": "left_hand_palm_link",
                },
                action_space=torch.tensor(
                    [
                        # Lower limits: 双手 xyz+xyzw 各 7 = 14
                        [
                            # Right wrist (xyz, xyzw)
                            0.2,
                            -0.6,
                            0.4,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            # Left wrist (xyz, xyzw)
                            0.2,
                            -0.6,
                            0.4,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                        ],
                        # Upper limits
                        [
                            0.8,
                            0.6,
                            1.2,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.8,
                            0.6,
                            1.2,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                    ]
                ),
                controller=G1_PINK_IK_CONTROLLER_CFG,
                # **True**：action 里的 xyz+xyzw 已经是 base_link_frame_name
                # (pelvis_contour_link) 坐标系下的目标（和 sonic `PinkIKDriver.solve`
                # 参数 `left_target_pelvis / right_target_pelvis` 语义一致）。
                # False 会触发 `_transform_poses_to_base_link_frame` 把 input 当
                # world 系再转进 pelvis 系，相当于双重变换 → 手会飘到后背。
                relative_to_base=True,
                # NaN 时由 Pink IK skip（fallback_to_current=False）。对应的 fallback 逻辑
                # 在 `SonicPinkInverseKinematicsAction.apply_actions()` 里实现：把
                # `_processed_actions` 填成**当前 joint_pos**，再写进 SonicArmBuffer。
                fallback_to_current=False,
                decimation=4,
            ),
        },
        # ---------- 后跑 base_action：SONIC lower-body 12 DOF ----------
        "base_action": {
            "wbc": SonicWBCActionCfg(
                # 15 个 = 12 legs + 3 waist（`HomieWBCActionCfg` 同口径；最后 apply 只写 12 legs）
                joint_names=[
                    ".*_hip_.*_joint",
                    ".*_knee_joint",
                    ".*_ankle_.*_joint",
                    "waist_.*_joint",
                ],
                num_wbc_joints=15,
                action_space=torch.tensor(
                    [
                        # Lower:  [vx, vy, ang_vel, height, mode]
                        [-2.0, -2.0, -1.5, -1.0, -1.0],
                        # Upper
                        [2.0, 2.0, 1.5, 1.0, 7.0],
                    ]
                ),
                decimation=4,
            ),
        },
        # ---------- eef_action：14 dex finger joints，原样复用 MagicSim G1 ----------
        # 两个选项：`interpolated`（插值 open↔close 手势）和 `joint_pos`（直接给 14 个
        # 关节角）。SONIC 不管手指，这里只是为了和 MagicSim env 框架兼容
        # （yaml 指 `eef_action: joint_pos` 时不 KeyError）。
        "eef_action": {
            "interpolated": mdp.MultipleInterpolatedJointChoicePositionActionCfg(
                joint_groups=[
                    mdp.InterpolatedJointChoiceActionCfg(
                        joint_names=[
                            "left_hand_index_0_joint",
                            "left_hand_index_1_joint",
                            "left_hand_middle_0_joint",
                            "left_hand_middle_1_joint",
                            "left_hand_thumb_0_joint",
                            "left_hand_thumb_1_joint",
                            "left_hand_thumb_2_joint",
                        ],
                        open_command_expr={
                            "left_hand_index_0_joint": 0.0,
                            "left_hand_index_1_joint": 0.0,
                            "left_hand_middle_0_joint": 0.0,
                            "left_hand_middle_1_joint": 0.0,
                            "left_hand_thumb_0_joint": 0.0,
                            "left_hand_thumb_1_joint": 0.0,
                            "left_hand_thumb_2_joint": 0.0,
                        },
                        close_command_exprs=[
                            {  # wbc close
                                "left_hand_index_0_joint": -0.6,
                                "left_hand_index_1_joint": -1.2,
                                "left_hand_middle_0_joint": -0.6,
                                "left_hand_middle_1_joint": -1.2,
                                "left_hand_thumb_0_joint": 0.0,
                                "left_hand_thumb_1_joint": 0.7,
                                "left_hand_thumb_2_joint": 0.7,
                            },
                            {  # index close
                                "left_hand_index_0_joint": -1.5,
                                "left_hand_index_1_joint": -1.5,
                                "left_hand_middle_0_joint": -0.6,
                                "left_hand_middle_1_joint": -1.5,
                                "left_hand_thumb_0_joint": -0.5,
                                "left_hand_thumb_1_joint": 0.7,
                                "left_hand_thumb_2_joint": 0.7,
                            },
                            {  # middle close
                                "left_hand_index_0_joint": -1.0,
                                "left_hand_index_1_joint": -1.5,
                                "left_hand_middle_0_joint": -1.0,
                                "left_hand_middle_1_joint": -1.5,
                                "left_hand_thumb_0_joint": 0.0,
                                "left_hand_thumb_1_joint": 0.7,
                                "left_hand_thumb_2_joint": 0.7,
                            },
                            {  # ring close
                                "left_hand_index_0_joint": -0.6,
                                "left_hand_index_1_joint": -1.5,
                                "left_hand_middle_0_joint": -1.5,
                                "left_hand_middle_1_joint": -1.5,
                                "left_hand_thumb_0_joint": 0.5,
                                "left_hand_thumb_1_joint": 0.7,
                                "left_hand_thumb_2_joint": 0.7,
                            },
                        ],
                    ),
                    mdp.InterpolatedJointChoiceActionCfg(
                        joint_names=[
                            "right_hand_index_0_joint",
                            "right_hand_index_1_joint",
                            "right_hand_middle_0_joint",
                            "right_hand_middle_1_joint",
                            "right_hand_thumb_0_joint",
                            "right_hand_thumb_1_joint",
                            "right_hand_thumb_2_joint",
                        ],
                        open_command_expr={
                            "right_hand_index_0_joint": 0.0,
                            "right_hand_index_1_joint": 0.0,
                            "right_hand_middle_0_joint": 0.0,
                            "right_hand_middle_1_joint": 0.0,
                            "right_hand_thumb_0_joint": 0.0,
                            "right_hand_thumb_1_joint": 0.0,
                            "right_hand_thumb_2_joint": 0.0,
                        },
                        close_command_exprs=[
                            {  # wbc close (negated)
                                "right_hand_index_0_joint": 0.6,
                                "right_hand_index_1_joint": 1.2,
                                "right_hand_middle_0_joint": 0.6,
                                "right_hand_middle_1_joint": 1.2,
                                "right_hand_thumb_0_joint": 0.0,
                                "right_hand_thumb_1_joint": -0.7,
                                "right_hand_thumb_2_joint": -0.7,
                            },
                            {  # index close (negated)
                                "right_hand_index_0_joint": 1.5,
                                "right_hand_index_1_joint": 1.5,
                                "right_hand_middle_0_joint": 0.6,
                                "right_hand_middle_1_joint": 1.5,
                                "right_hand_thumb_0_joint": -0.5,
                                "right_hand_thumb_1_joint": -0.7,
                                "right_hand_thumb_2_joint": -0.7,
                            },
                            {  # middle close (negated)
                                "right_hand_index_0_joint": 1.0,
                                "right_hand_index_1_joint": 1.5,
                                "right_hand_middle_0_joint": 1.0,
                                "right_hand_middle_1_joint": 1.5,
                                "right_hand_thumb_0_joint": 0.0,
                                "right_hand_thumb_1_joint": -0.7,
                                "right_hand_thumb_2_joint": -0.7,
                            },
                            {  # ring close (negated)
                                "right_hand_index_0_joint": 0.6,
                                "right_hand_index_1_joint": 1.5,
                                "right_hand_middle_0_joint": 1.5,
                                "right_hand_middle_1_joint": 1.5,
                                "right_hand_thumb_0_joint": 0.5,
                                "right_hand_thumb_1_joint": -0.7,
                                "right_hand_thumb_2_joint": -0.7,
                            },
                        ],
                    ),
                ],
            ),
            "joint_pos": MultipleJointPositionToLimitsActionCfg(
                joint_groups=[
                    MultipleJointPositionToLimitsActionGroupCfg(
                        joint_names=[
                            "right_hand_index_0_joint",
                            "right_hand_index_1_joint",
                            "right_hand_middle_0_joint",
                            "right_hand_middle_1_joint",
                            "right_hand_thumb_0_joint",
                            "right_hand_thumb_1_joint",
                            "right_hand_thumb_2_joint",
                        ],
                        num_joints=7,
                        preserve_order=True,
                    ),
                    MultipleJointPositionToLimitsActionGroupCfg(
                        joint_names=[
                            "left_hand_index_0_joint",
                            "left_hand_index_1_joint",
                            "left_hand_middle_0_joint",
                            "left_hand_middle_1_joint",
                            "left_hand_thumb_0_joint",
                            "left_hand_thumb_1_joint",
                            "left_hand_thumb_2_joint",
                        ],
                        num_joints=7,
                        preserve_order=True,
                    ),
                ],
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()


# =========================================================================
# 6. Re-export MagicSim G1 辅助（ObsCfg / HeadCamera / strategies / utils）
# =========================================================================
# 这样下游 `from magicsim.Env.Robot.Cfg.Humanoid.G1_Sonic import G1SonicObsCfg`
# 直接可用，不需要跨文件 import 两处。
G1SonicObsCfg = G1ObsCfg
G1SonicHeadCameraCfg = G1HeadCameraCfg


# =========================================================================
# 7. G1_SonicPlannerCfg / G1_SonicCfg —— magicsim Planner 层的默认透传 cfg
# =========================================================================
# SONIC 自己的 planner 在 `SonicPolicy` 内部跑（planner_sonic.onnx）。
# magicsim 外层 PlannerManager 对 base/arm/eef 三路都走 `default`（pass-through），
# 让 5D(SonicWBC) + 14D(Pink IK) action tensor 直接流到 ActionTerm。
# `base_action_dim` / `arm_action_dim` / `eef_action_dim` 里 "default" 键的值
# 用于 PlannerManager 的 action slice offset 计算。


@configclass
class G1SonicPlannerCfg(HumanoidPlannerCfg):
    """G1_Sonic 用的 planner cfg，全部 default 透传。"""

    max_eef_num: int = 2

    # 5D sonic action: [vx, vy, ang_vel, height, mode]
    base_action_dim: Dict[str, int] = {"default": 5}
    base_action_space: Dict[str, torch.Tensor] = {
        # 和 G1SonicActionsCfg.base_action.wbc.action_space 对齐
        "default": torch.tensor(
            [
                [-2.0, -2.0, -1.5, -1.0, -1.0],
                [2.0, 2.0, 1.5, 1.0, 7.0],
            ]
        ),
    }

    # 14D Pink IK: 2 × (xyz + xyzw)
    arm_action_dim: Dict[str, int] = {"default": 14}
    arm_action_space: Dict[str, torch.Tensor] = {
        "default": torch.tensor(
            [
                # right (xyz xyzw) + left (xyz xyzw)  —— 和 G1SonicActionsCfg 里一致
                [
                    0.2,
                    -0.6,
                    0.4,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    0.2,
                    -0.6,
                    0.4,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ],
                [0.8, 0.6, 1.2, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 1.2, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
    }

    # eef: 14 dex finger joints (2 × 7)
    eef_action_dim: Dict[str, int] = {"default": 14}
    eef_action_space: Dict[str, torch.Tensor] = {"default": None}

    # 复用 MagicSim G1 的 RetractMoveL move_strategy（轨迹多段化）
    move_strategy = staticmethod(g1_move_strategy)
    move_strategy_distance_threshold: float = 0.3


@configclass
class G1_SonicCfg(HumanoidCfg):
    """Robot cfg for the G1-Sonic variant (SONIC hybrid IK pipeline)."""

    prim_path: str = _MISSING
    asset_name: str = "robot"
    base_action_name: str = _MISSING
    arm_action_name: str = _MISSING
    eef_action_name: str | None = None  # SONIC 默认不配 eef
    robot: ArticulationCfg = _MISSING
    action: G1SonicActionsCfg = _MISSING
    sensor: G1HeadCameraCfg = G1HeadCameraCfg()
    obs: G1ObsCfg = _MISSING
    planner: G1SonicPlannerCfg = G1SonicPlannerCfg()

    def __post_init__(self):
        self.robot: ArticulationCfg = G1_SONIC_CFG
        self.robot.prim_path = self.prim_path
        self.action: G1SonicActionsCfg = G1SonicActionsCfg(
            asset_name=self.asset_name,
            base_action_name=self.base_action_name,
            arm_action_name=self.arm_action_name,
            eef_action_name=self.eef_action_name,
        )
        self.sensor: G1HeadCameraCfg = G1HeadCameraCfg(
            robot_prim_path=self.prim_path,
        )
        self.obs: G1ObsCfg = G1ObsCfg(
            asset_name=self.asset_name, sensor_name=f"{self.asset_name}_head_camera"
        )
        super().__post_init__()


__all__ = [
    # Robot cfg
    "G1_SONIC_CFG",
    "G1_SonicCfg",
    "G1SonicPlannerCfg",
    # Actions
    "G1SonicActionsCfg",
    # Obs / camera
    "G1SonicObsCfg",
    "G1SonicHeadCameraCfg",
    # Strategies (re-export)
    "g1_move_strategy",
    "g1_dehatch_strategy",
    "g1_postprocess_p_controller_action",
    # Utils
    "wrap_to_pi",
    # PD constants (expose for downstream inspection / unit tests)
    "ARMATURE_5020",
    "ARMATURE_7520_14",
    "ARMATURE_7520_22",
    "ARMATURE_4010",
    "STIFFNESS_5020",
    "STIFFNESS_7520_14",
    "STIFFNESS_7520_22",
    "STIFFNESS_4010",
    "DAMPING_5020",
    "DAMPING_7520_14",
    "DAMPING_7520_22",
    "DAMPING_4010",
    "LEG_JOINTS_IL",
    "PINK_CONTROLLED_JOINTS_IL",
]
