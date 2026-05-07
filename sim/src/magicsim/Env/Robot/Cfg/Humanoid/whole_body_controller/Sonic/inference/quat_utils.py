"""Quaternion / rotation / resample utilities for SONIC inference.

**从 `sonic_python_inference/sonic_inference.py` 抠出**。约定：
    - 四元数一律 wxyz（和 sonic 训练 / MagicSim 一致）
    - batch shape `[..., 4]` for quats, `[..., 3]` for vectors
    - `yaw_from_quat` 是本项目新加的工具（SonicPolicy 每 tick 从 sim 读 base_quat_w 算 current_yaw 用）

共享常量同样搬过来（`ACTION_DIM`, `NUM_JOINTS`, `HIST_LEN`, 等 `sonic_g1_inference.py`
会 `from .quat_utils import ACTION_DIM, ...`）。
"""

from __future__ import annotations

import math
import numpy as np
import torch


# ---------------------------- 常量 ------------------------------------
ACTION_DIM: int = 29
NUM_JOINTS: int = 29
HIST_LEN: int = 10
NUM_LOWER_BODY_JOINTS: int = 12

PLANNER_NATIVE_HZ: float = 30.0
POLICY_HZ: float = 50.0
# RESAMPLED_FRAMES = ceil(64 * 50 / 30) + 1 = 108 + 1 = 109  (sonic_inference.py:87 行原样)
# 但 stage_hybrid_eval 里的 RESAMPLED_FRAMES 是 250；原因是 scripts 里独立定义
# 了更长的 cache（见 stage_hybrid_eval.py 顶部）。SONIC G1 hybrid 只用到 10 Hz
# 对 10 future frames step=5 的采样，即 50 frames，109 够用。这里保留 sonic_inference.py 原值。
from .planner_pool import PLANNER_FRAME_DIM, PLANNER_OUTPUT_FRAMES  # re-export

RESAMPLED_FRAMES: int = (
    int(math.ceil(PLANNER_OUTPUT_FRAMES * POLICY_HZ / PLANNER_NATIVE_HZ)) + 1
)

# Planner locomotion mode ids (localmotion_kplanner.hpp:527).
PLANNER_MODE_IDLE: int = 0
PLANNER_MODE_SLOW_WALK: int = 1
PLANNER_MODE_WALK: int = 2
PLANNER_MODE_RUN: int = 3

# allowed_pred_num_tokens mask (localmotion_kplanner_onnx.hpp:155-163).
ALLOWED_PRED_NUM_TOKENS: np.ndarray = np.array(
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int64
)
PLANNER_HEIGHT_DEFAULT: float = -1.0
PLANNER_CONTEXT_DEFAULT_HEIGHT: float = 0.788740


# ---------------------------- Quat utils ------------------------------
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion conjugate."""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by wxyz quaternion q. Shapes: q[..., 4], v[..., 3]."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return torch.stack([rx, ry, rz], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion multiply."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_to_6d(q: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion → 6D rot (row-wise flatten of first 2 columns of R).

    Matches training/deploy:
      - training: gear_sonic/envs/manager_env/mdp/commands.py:1961-1962
      - deploy:   g1_deploy_onnx_ref.cpp:679-683
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    return torch.stack([r00, r01, r10, r11, r20, r21], dim=-1)


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """wxyz → yaw (world-frame, ZYX 欧拉 yaw).

    `q` shape [..., 4]. Returns [...].
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def slerp_torch(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Batched wxyz slerp. q0, q1: [..., 4], t: [...] or broadcastable."""
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    small = sin_theta < 1e-6
    w0 = torch.where(
        small,
        1.0 - t.unsqueeze(-1),
        torch.sin((1 - t.unsqueeze(-1)) * theta) / sin_theta,
    )
    w1 = torch.where(
        small, t.unsqueeze(-1), torch.sin(t.unsqueeze(-1) * theta) / sin_theta
    )
    return w0 * q0 + w1 * q1


def resample_traj_30_to_50hz(
    traj_30hz: torch.Tensor, num_output: int = RESAMPLED_FRAMES
) -> torch.Tensor:
    """`traj_30hz: [N, 64, 36]` (root_pos 3 | root_quat_wxyz 4 | joints 29).
    Returns [N, num_output, 36] resampled to 50 Hz via linear + slerp.
    """
    N, T_in, D = traj_30hz.shape
    assert D == PLANNER_FRAME_DIM
    device = traj_30hz.device
    t_out = torch.arange(num_output, device=device, dtype=torch.float32) * (
        PLANNER_NATIVE_HZ / POLICY_HZ
    )
    idx0 = torch.clamp(t_out.long(), 0, T_in - 1)
    idx1 = torch.clamp(idx0 + 1, 0, T_in - 1)
    alpha = (t_out - idx0.float()).clamp(0.0, 1.0)

    pos0 = traj_30hz[:, idx0, 0:3]
    pos1 = traj_30hz[:, idx1, 0:3]
    pos = pos0 + (pos1 - pos0) * alpha.view(1, -1, 1)

    q0 = traj_30hz[:, idx0, 3:7]
    q1 = traj_30hz[:, idx1, 3:7]
    a_q = alpha.view(1, -1).expand(N, -1)
    quat = slerp_torch(q0, q1, a_q)

    j0 = traj_30hz[:, idx0, 7:]
    j1 = traj_30hz[:, idx1, 7:]
    joints = j0 + (j1 - j0) * alpha.view(1, -1, 1)

    return torch.cat([pos, quat, joints], dim=-1)
