"""SonicPolicy 辅助函数：从 articulation 抽取 robot state。

和 Homie 的 `prepare_observations` 平行，但返回的是 **torch 张量**（SonicPolicy
全链路 torch，避免 CPU↔GPU 来回），且包含 SONIC hybrid 需要的字段：
    - joint_pos, joint_vel            [N, 29]  IL body 顺序（dex 已过滤）
    - base_ang_vel                    [N, 3]   body frame
    - gravity_in_base                 [N, 3]   projected_gravity_b
    - root_quat_wxyz                  [N, 4]   world frame
"""

from __future__ import annotations

import torch

from isaaclab.assets.articulation.articulation_data import ArticulationData


def prepare_observations(
    articulation_data: ArticulationData,
    body_idx_full: list[int] | torch.Tensor,
) -> dict[str, torch.Tensor]:
    """从 Isaac Lab articulation.data 抽取 SONIC 需要的 6 个张量，返回 dict。

    Args:
        articulation_data: `robot.data`（Articulation 的 data 属性）
        body_idx_full: USD-full 43 joints 里 29 个 body joint 的 index list
            （由 SonicPolicy 在 `__init__` 时一次性解析 —— 见
            SonicPolicy._resolve_body_indices）
    """
    if not isinstance(body_idx_full, torch.Tensor):
        body_idx_full = torch.as_tensor(
            body_idx_full, dtype=torch.long, device=articulation_data.joint_pos.device
        )

    return {
        "joint_pos": articulation_data.joint_pos[:, body_idx_full],  # [N, 29]
        "joint_vel": articulation_data.joint_vel[:, body_idx_full],  # [N, 29]
        "base_ang_vel": articulation_data.root_ang_vel_b,  # [N, 3]
        "gravity_in_base": articulation_data.projected_gravity_b,  # [N, 3]
        "root_quat_wxyz": articulation_data.root_state_w[:, 3:7],  # [N, 4] wxyz
    }
