"""SonicWBCActionCfg —— SONIC lower-body 12 DOF action term 的 cfg dataclass。"""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from magicsim.Env.Robot.Cfg.Humanoid.mdp.sonic_wbc_action import SonicWBCAction


@configclass
class SonicWBCActionCfg(ActionTermCfg):
    """SONIC whole-body-control action cfg（hybrid IK pipeline 的 lower-body 12 DOF）。

    参考 `HomieWBCActionCfg` 的结构，语义上：
      - `joint_names`   —— 匹配下半身 12 legs + 3 waist（waist 最终会被丢，只施加 12；和 Homie 同）
      - `num_wbc_joints` —— legs + waist 共 15（和 Homie v2 同）
      - `action_space`  —— [2, 5]，5D action 的上下限（`[vx, vy, ang_vel, height, mode]`）
      - `wbc_version`   —— "sonic_v1"，由 `wbc_policy_factory.get_wbc_policy` 分派
      - `decimation`    —— 必须是 4（和 sonic 训练的 policy 50Hz / sim 200Hz 对齐）
    """

    class_type: type[ActionTerm] = SonicWBCAction

    robot_type: str = "g1"

    preserve_order: bool = False
    """Pass to `_asset.find_joints(..., preserve_order=...)`. SONIC legs / waist 在
    USD-full IL 顺序里已经连续，默认 False 即可。"""

    joint_names: list[str] = MISSING
    """Legs + waist regex/list，同 Homie 的 15 joint。"""

    wbc_version: str = "sonic_v1"

    num_wbc_joints: int = MISSING
    """15 = 12 legs + 3 waist（最终 apply_actions 只写 12 legs）。"""

    action_space: torch.Tensor = MISSING
    """`[2, 5]` 张量，上下限 = `[[vx_lo, vy_lo, ang_vel_lo, height_lo, mode_lo],
                                  [vx_hi, vy_hi, ang_vel_hi, height_hi, mode_hi]]`。
    典型取值参考 G1_Sonic.py 里构造时填的 `G1SonicActionsCfg`。"""

    decimation: int = 4
