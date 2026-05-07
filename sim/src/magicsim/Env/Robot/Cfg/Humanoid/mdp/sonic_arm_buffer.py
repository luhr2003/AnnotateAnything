"""Shared buffer: Pink IK 解出的 17-DOF arm+waist 目标，供 SONIC WBC lower-body 读取。

拓扑：
    SonicPinkInverseKinematicsAction.apply_actions()
        → 计算出 17-DOF 关节目标 (3 waist + 14 arm, `PINK_CONTROLLED_JOINTS_IL` 顺序)
        → 施加到 sim（set_joint_position_target）
        → 同步写入本 buffer

    SonicWBCAction.process_actions()
        → 从本 buffer 读最新 17-DOF
        → 塞进 SonicPolicy 的 g1_encoder 输入 (joint_pos_future 的 upper 17 slot)
        → g1_encoder → decoder → 输出 29-DOF，只取 lower 12 施加

并发约束：**env 级批并行，非多线程**。Buffer 的 `targets` 是 [N, 17] torch 张量，
写入/读取都是 tensor indexing，天生支持 env 并行；不涉及 Python 线程。

单例：由 `SonicArmBuffer.get_or_create(num_envs, device)` 获取。同一个 process
中同一个 (num_envs, device) 组合返回同一个实例，保证 Pink IK action 和
SONIC WBC action 看到同一个 tensor；重建（num_envs 变了）时抛错，避免残留。
"""

from __future__ import annotations

import torch


class SonicArmBuffer:
    """Per-env 17-DOF arm+waist 目标缓存，Pink IK 写、SONIC 读。"""

    _instance: "SonicArmBuffer | None" = None

    NUM_ARM_DOF: int = 17
    """3 waist + 7 left arm + 7 right arm, 顺序严格等于
    `G1SonicV1Config.PINK_CONTROLLED_JOINTS_IL`（sonic canonical order）。"""

    def __init__(self, num_envs: int, device: torch.device | str):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.targets = torch.zeros(
            self.num_envs, self.NUM_ARM_DOF, device=self.device, dtype=torch.float32
        )
        # True 表示该 env 的 targets 是 Pink IK 新鲜解；False 表示是 reset 填的 fallback 值。
        # SONIC 读时可以据此降级（如 fallback 到自己 default_angles 的 upper 17），v1 先不用。
        self.ready = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # Monotonic tick counter per env，给 SONIC 侧做 debug / 检测 stale 用。
        self.tick = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    # --------------------------- Singleton ------------------------------
    @classmethod
    def get_or_create(
        cls, num_envs: int, device: torch.device | str
    ) -> "SonicArmBuffer":
        if cls._instance is None:
            cls._instance = cls(num_envs, device)
            return cls._instance

        inst = cls._instance
        if inst.num_envs != num_envs:
            raise RuntimeError(
                f"SonicArmBuffer already initialized with num_envs={inst.num_envs}, "
                f"but asked for num_envs={num_envs}. Call reset_singleton() first."
            )
        if inst.device != torch.device(device):
            raise RuntimeError(
                f"SonicArmBuffer device mismatch: existing={inst.device}, "
                f"requested={torch.device(device)}. Call reset_singleton() first."
            )
        return inst

    @classmethod
    def reset_singleton(cls) -> None:
        """清理单例 —— 仅测试或 env 重建时使用。"""
        cls._instance = None

    # --------------------------- I/O ------------------------------------
    def write(self, env_ids: torch.Tensor | slice, targets_17: torch.Tensor) -> None:
        """Pink IK 每 tick 调用；`targets_17` 形状 [len(env_ids), 17]，
        **必须已经在 `PINK_CONTROLLED_JOINTS_IL` 顺序**。置换由上游
        `SonicPinkInverseKinematicsAction` 做完。"""
        self.targets[env_ids] = targets_17.to(self.device, dtype=torch.float32)
        self.ready[env_ids] = True
        if isinstance(env_ids, slice):
            self.tick += 1
        else:
            self.tick[env_ids] += 1

    def read_latest(self, env_ids: torch.Tensor | slice | None = None) -> torch.Tensor:
        """SONIC 每 tick 调用。返回最新 17-DOF target（PINK_CONTROLLED_JOINTS_IL 顺序）。"""
        if env_ids is None:
            return self.targets
        return self.targets[env_ids]

    def is_ready(self, env_ids: torch.Tensor | slice | None = None) -> torch.Tensor:
        if env_ids is None:
            return self.ready
        return self.ready[env_ids]

    def reset(
        self,
        env_ids: torch.Tensor | slice,
        default_arm_17: torch.Tensor,
    ) -> None:
        """env reset 时调用，用 init_state 的 arm joint_pos 填 buffer。

        避免第一个 SONIC tick 读到零向量导致 encoder OOD —— 先放 rest pose，
        等 Pink IK 第一次 solve 完 `ready` 自然翻 True。
        """
        self.targets[env_ids] = default_arm_17.to(self.device, dtype=torch.float32)
        self.ready[env_ids] = False
        if isinstance(env_ids, slice):
            self.tick[:] = 0
        else:
            self.tick[env_ids] = 0
