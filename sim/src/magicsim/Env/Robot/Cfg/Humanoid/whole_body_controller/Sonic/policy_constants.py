"""SONIC WBC policy 常量。

Action 语义（5 维，和 Homie 不兼容、专用）：
    [vx, vy, ang_vel, height, mode]

    vx, vy    : 本体系线速度命令 (m/s)，正向 vx = 前进
    ang_vel   : yaw rate 命令 (rad/s)
    height    : 目标身高 (m)，-1 = planner 默认（~0.789 m）
    mode      : 整数锁
                -1 = AUTO：按 |v| 自动切 IDLE / SLOW_WALK / WALK / RUN
                 0 = IDLE
                 1 = SLOW_WALK
                 2 = WALK
                 3 = RUN
                 (其他 sonic planner 原生 mode 如 SQUAT/KNEEL 也直接透传)

Rates（必须和训练对齐，不可调）：
    sim       : 200 Hz (sim.dt = 0.005)
    policy    : 50 Hz  (decimation = 4)
    planner   : 10 Hz  (每 5 个 policy step 跑一次 planner ONNX)
    future win: 10 frames @ dt=0.1s for G1 encoder (frame_skip=5 at 50Hz)
"""

from __future__ import annotations

# ------------------------------- Action ---------------------------------
NUM_VX: int = 1
NUM_VY: int = 1
NUM_ANG_VEL: int = 1
NUM_HEIGHT: int = 1
NUM_MODE: int = 1
ACTION_DIM: int = NUM_VX + NUM_VY + NUM_ANG_VEL + NUM_HEIGHT + NUM_MODE  # 5

# Action slice offsets
OFFSET_VX: int = 0
OFFSET_VY: int = 1
OFFSET_ANG_VEL: int = 2
OFFSET_HEIGHT: int = 3
OFFSET_MODE: int = 4

# ------------------------------- Timing ---------------------------------
DT_POLICY: float = 0.02  # 1 / 50 Hz
DT_PLANNER_NATIVE: float = 1.0 / 30.0  # planner ONNX 输出是 30 Hz
PLANNER_EVERY_K_POLICY_STEPS: int = 5  # 每 5 个 policy tick 跑一次 planner
G1_FUTURE_FRAME_SKIP: int = (
    5  # G1 encoder future window 的 frame_skip（50Hz × 5 = 0.1 s）
)
G1_NUM_FUTURE_FRAMES: int = 10

# ------------------------------- Mode codes -----------------------------
# 对齐 sonic deploy (localmotion_kplanner.hpp:527)
MODE_AUTO: int = -1
MODE_IDLE: int = 0
MODE_SLOW_WALK: int = 1
MODE_WALK: int = 2
MODE_RUN: int = 3

# AUTO 模式下按 |v| 切 mode 的阈值 (m/s)
AUTO_MODE_IDLE_MAX: float = 0.05
AUTO_MODE_SLOW_WALK_MAX: float = 0.8
AUTO_MODE_WALK_MAX: float = 2.5
# >= AUTO_MODE_WALK_MAX → RUN

# Planner default height sentinel: -1 means "use planner's trained default (~0.789 m)"
HEIGHT_PLANNER_DEFAULT: float = -1.0
PLANNER_CONTEXT_DEFAULT_HEIGHT: float = 0.788740  # localmotion_kplanner.hpp:216

# ------------------------------- DOF counts ----------------------------
# G1 body (29 DOF) = lower 12 legs + 3 waist + 14 arms. Dex fingers (14) 不归 WBC 管。
NUM_BODY_DOF: int = 29
NUM_LEG_DOF: int = (
    12  # 2 × (hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll)
)
NUM_UPPER_DOF: int = 17  # 3 waist + 14 arm — Pink IK 管，来源于 SonicArmBuffer
NUM_WAIST_DOF: int = 3
NUM_ARM_DOF: int = 14

# 最小速度阈值：|v| < EPS_TARGET_VEL 时 movement_direction fallback 到 facing_direction
EPS_TARGET_VEL: float = 1e-4

# Hand joint 的正则，用于从 USD-full 43 joints 里过滤出 29 body joints。
# 匹配 `left_hand_index_0_joint` / `right_hand_thumb_2_joint` 等。
HAND_JOINT_REGEX: str = r".*_hand_(index|middle|thumb)_\d+_joint"

# Leg joint 的正则——sonic 参考实现分 3 条独立匹配（knee joint 没有中间段，
# 像 `left_knee_joint`，所以不能和 hip/ankle 合并成 `.*_(hip|knee|ankle)_.*_joint`）。
# 见 `stage_hybrid_eval_magicsim.py:130-134`。
LEG_JOINT_REGEXES: tuple[str, ...] = (
    r".*_hip_.*_joint",
    r".*_knee_joint",
    r".*_ankle_.*_joint",
)

# Planner ONNX 输入 allowed_pred_num_tokens mask（localmotion_kplanner_onnx.hpp:155-163）。
# 11 维 int64，前 6 位允许、后 5 位屏蔽。
ALLOWED_PRED_NUM_TOKENS: tuple[int, ...] = (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0)
