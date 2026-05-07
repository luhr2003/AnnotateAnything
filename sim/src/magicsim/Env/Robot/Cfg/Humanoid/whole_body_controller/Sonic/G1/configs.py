"""G1SonicV1Config — SONIC hybrid IK pipeline 的 WBC 配置。

和 `G1HomieV2Config` 平行，但 SONIC 专用：
  - 3 个 ONNX（planner + g1_encoder + decoder）而非 homie 的 2 个（stand + walk）
  - 额外的 MJ↔IL / IL↔MJ joint DOF 置换常量（sonic 训练时写死的 29-int 数组）
  - `PINK_CONTROLLED_JOINTS_IL`：Pink IK 输出 17 DOF 的 sonic canonical 序

**绝对不动**：两组 29-int 置换数组原样从 sonic 源码拷来，
见 `gear_sonic/envs/manager_env/robots/g1.py:61-123`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from magicsim import MAGICSIM_ASSETS
from magicsim.Env.Robot.Cfg.Humanoid.whole_body_controller.wbc_base_policy import (
    BaseConfig,
)


# ---------------- sonic 训练时写死的 DOF 置换 -------------------------
# 语义（stage_hybrid_eval.py:245-254 中的注释）：
#     mj_data = il_data[G1_ISAACLAB_TO_MUJOCO_DOF]  (IL→MJ gather)
#     il_data = mj_data[G1_MUJOCO_TO_ISAACLAB_DOF]  (MJ→IL gather)
# MagicSim g1_new.usd 下已通过 stage_hybrid_eval_magicsim.py 验证可直接复用。
G1_ISAACLAB_TO_MUJOCO_DOF: list[int] = [
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
]
G1_MUJOCO_TO_ISAACLAB_DOF: list[int] = [
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
]

# ---------------- SONIC canonical PINK IK 17 DOF 顺序 -----------------
# 来源 sonic `g1_pink_ik_cfg.py:31-49`。`SonicArmBuffer` 按此顺序存储。
PINK_CONTROLLED_JOINTS_IL: list[str] = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# ---------------- Body (29 DOF) IL 顺序 -------------------------------
# MagicSim g1_new.usd 过滤 14 个 dex finger 后剩下 29 关节，IsaacLab 按**广度优先**
# 加载（所有 hip_pitch → waist_yaw → 所有 hip_roll → waist_roll → ...），这也正是
# sonic 训练时的 IL 顺序（通过 `G1_ISAACLAB_TO_MUJOCO_DOF` 反推可验证：
# `mj_data = il_data[IL→MJ]` 把广度优先的 IL 置换成 sonic deploy 用的链式 MJ）。
#
# `SonicPolicy.bind_articulation` 会和 runtime `robot.data.joint_names` 过滤结果
# 做 assert；如果 MagicSim 之后换 USD 导致顺序变了，会立刻在启动时报错。
G1_BODY_JOINTS_IL: list[str] = [
    "left_hip_pitch_joint",  # 0
    "right_hip_pitch_joint",  # 1
    "waist_yaw_joint",  # 2
    "left_hip_roll_joint",  # 3
    "right_hip_roll_joint",  # 4
    "waist_roll_joint",  # 5
    "left_hip_yaw_joint",  # 6
    "right_hip_yaw_joint",  # 7
    "waist_pitch_joint",  # 8
    "left_knee_joint",  # 9
    "right_knee_joint",  # 10
    "left_shoulder_pitch_joint",  # 11
    "right_shoulder_pitch_joint",  # 12
    "left_ankle_pitch_joint",  # 13
    "right_ankle_pitch_joint",  # 14
    "left_shoulder_roll_joint",  # 15
    "right_shoulder_roll_joint",  # 16
    "left_ankle_roll_joint",  # 17
    "right_ankle_roll_joint",  # 18
    "left_shoulder_yaw_joint",  # 19
    "right_shoulder_yaw_joint",  # 20
    "left_elbow_joint",  # 21
    "right_elbow_joint",  # 22
    "left_wrist_roll_joint",  # 23
    "right_wrist_roll_joint",  # 24
    "left_wrist_pitch_joint",  # 25
    "right_wrist_pitch_joint",  # 26
    "left_wrist_yaw_joint",  # 27
    "right_wrist_yaw_joint",  # 28
]
assert len(G1_BODY_JOINTS_IL) == 29

# Leg joint 名（12 个）—— 在 G1_BODY_JOINTS_IL 中按广度优先出现的顺序
# （和 _leg_indices_il regex 匹配结果一致）。
LEG_JOINTS_IL: list[str] = [
    "left_hip_pitch_joint",  # IL 0
    "right_hip_pitch_joint",  # IL 1
    "left_hip_roll_joint",  # IL 3
    "right_hip_roll_joint",  # IL 4
    "left_hip_yaw_joint",  # IL 6
    "right_hip_yaw_joint",  # IL 7
    "left_knee_joint",  # IL 9
    "right_knee_joint",  # IL 10
    "left_ankle_pitch_joint",  # IL 13
    "right_ankle_pitch_joint",  # IL 14
    "left_ankle_roll_joint",  # IL 17
    "right_ankle_roll_joint",  # IL 18
]
assert len(LEG_JOINTS_IL) == 12
assert set(LEG_JOINTS_IL) | set(PINK_CONTROLLED_JOINTS_IL) == set(G1_BODY_JOINTS_IL)


@dataclass
class G1SonicV1Config(BaseConfig):
    """G1 SONIC hybrid v1 config.

    `BaseConfig` requires `wbc_model_path` (str) 和 `policy_config_path` (str)。
    SONIC 有 3 个 ONNX，我们用额外字段分别存；`wbc_model_path` 填 planner 路径
    保持 base 契约满足，其余两个用单独字段。
    """

    wbc_version: Literal["sonic_v1"] = "sonic_v1"

    # --- ONNX paths ------------------------------------------------------
    planner_onnx_path: str = f"{MAGICSIM_ASSETS}/WBC/models/sonic_v1/planner_sonic.onnx"
    g1_encoder_onnx_path: str = (
        f"{MAGICSIM_ASSETS}/WBC/models/sonic_v1/g1_encoder_dyn.onnx"
    )
    decoder_onnx_path: str = f"{MAGICSIM_ASSETS}/WBC/models/sonic_v1/decoder_dyn.onnx"

    # BaseConfig 契约：`wbc_model_path` 不为 MISSING。填 planner 主路径（其余两个独立字段）。
    wbc_model_path: str = f"{MAGICSIM_ASSETS}/WBC/models/sonic_v1/planner_sonic.onnx"
    # BaseConfig 契约：`policy_config_path` 不为 MISSING。没有 yaml，填 __file__ 占位。
    policy_config_path: str = __file__

    # --- Joint metadata -------------------------------------------------
    # BaseConfig 契约：`wbc_joints_order` 不为 MISSING。给全 43-DOF USD-full 顺序
    # （和 G1HomieV2Config 同形式，方便下游通用代码复用；SonicPolicy 内部
    # 用 body_joint_names 29 个过滤结果，不依赖这个 43-map）。
    wbc_joints_order: dict[str, int] = field(
        default_factory=lambda: {
            "left_hip_pitch_joint": 0,
            "left_hip_roll_joint": 1,
            "left_hip_yaw_joint": 2,
            "left_knee_joint": 3,
            "left_ankle_pitch_joint": 4,
            "left_ankle_roll_joint": 5,
            "right_hip_pitch_joint": 6,
            "right_hip_roll_joint": 7,
            "right_hip_yaw_joint": 8,
            "right_knee_joint": 9,
            "right_ankle_pitch_joint": 10,
            "right_ankle_roll_joint": 11,
            "waist_yaw_joint": 12,
            "waist_roll_joint": 13,
            "waist_pitch_joint": 14,
            "left_shoulder_pitch_joint": 15,
            "left_shoulder_roll_joint": 16,
            "left_shoulder_yaw_joint": 17,
            "left_elbow_joint": 18,
            "left_wrist_roll_joint": 19,
            "left_wrist_pitch_joint": 20,
            "left_wrist_yaw_joint": 21,
            "left_hand_index_0_joint": 22,
            "left_hand_index_1_joint": 23,
            "left_hand_middle_0_joint": 24,
            "left_hand_middle_1_joint": 25,
            "left_hand_thumb_0_joint": 26,
            "left_hand_thumb_1_joint": 27,
            "left_hand_thumb_2_joint": 28,
            "right_shoulder_pitch_joint": 29,
            "right_shoulder_roll_joint": 30,
            "right_shoulder_yaw_joint": 31,
            "right_elbow_joint": 32,
            "right_wrist_roll_joint": 33,
            "right_wrist_pitch_joint": 34,
            "right_wrist_yaw_joint": 35,
            "right_hand_index_0_joint": 36,
            "right_hand_index_1_joint": 37,
            "right_hand_middle_0_joint": 38,
            "right_hand_middle_1_joint": 39,
            "right_hand_thumb_0_joint": 40,
            "right_hand_thumb_1_joint": 41,
            "right_hand_thumb_2_joint": 42,
        },
    )

    # --- Robot Configuration -------------------------------------------
    enable_waist: bool = True
    """SONIC Pink IK 管 3 waist + 14 arms = 17 DOF。"""
