"""SONIC G1 forward-walking test —— 以 0.3 m/s 向前走。

Action 组成 (总 33 维):
    base_action (SonicWBC, 5):  [vx, vy, ang_vel, height, mode] = [0.3, 0, 0, -1, -1]
    arm_action (Pink IK, 14):   [right(xyz+wxyz), left(xyz+wxyz)] — **rest pose**
        双手固定在 sonic canonical rest pose（pelvis_contour_link 帧）：
            LEFT_WRIST_REST_POSE_PELVIS  = (0.24127,  0.15165, 0.14523, identity quat)
            RIGHT_WRIST_REST_POSE_PELVIS = (0.24127, -0.15164, 0.14523, identity quat)
        MagicSim Pink IK 用的 quat 顺序是 (qx, qy, qz, qw)，identity = (0, 0, 0, 1)。
        这和 sonic 参考 `stage_hybrid_eval_magicsim.py` 的做法**完全一致** ——
        每 tick Pink IK 解到同一个 rest pose，给 SONIC encoder 一个稳定的外部 arm ref。
    eef_action (joint_pos, 14): 14 dex finger joints —— NaN 透传保持初始角度

期望：机器人以约 0.3 m/s 前进且**手臂自然摆动**（配合腿部步态），root_z 维持
在 ~0.78 m 以上不摔。
"""

from __future__ import annotations

import time

import gymnasium as gym
import hydra
import torch
from loguru import logger as log
from omegaconf import DictConfig

from magicsim import MAGICSIM_CONF
from magicsim.Env.Environment.SyncRobotEnv import SyncRobotEnv
from magicsim.Env.Utils.file import Logger


# Sonic canonical 双手 rest pose（pelvis_contour_link 帧，**xyz + wxyz**）
# 来源：`sonic_python_inference/g1_pink_ik_cfg.py:52-57`，和 `pink_ik_driver.py:15`
# 契约一致：`Targets: [N, 7] = (x, y, z, qw, qx, qy, qz) (wxyz quat)`。
#
# MagicSim Pink IK 用 `isaaclab.utils.math.matrix_from_quat` 解 action 里的
# 四元数，后者文档明确指定 **(w, x, y, z)**（IsaacLab/math.py:169）。所以 action
# tensor 的 quat 段必须是 wxyz，identity = [1, 0, 0, 0]，**不是** [0,0,0,1]。
RIGHT_WRIST_REST_POSE_WXYZ: list[float] = [
    0.24127,
    -0.15164,
    0.14523,
    1.0,
    0.0,
    0.0,
    0.0,
]
LEFT_WRIST_REST_POSE_WXYZ: list[float] = [
    0.24127,
    0.15165,
    0.14523,
    1.0,
    0.0,
    0.0,
    0.0,
]


def build_forward_walk_action(
    num_envs: int,
    device: torch.device | str,
    vx: float = 0.6,
    vy: float = 0.0,
    ang_vel: float = 0.0,
    height: float = -1.0,
    mode: float = -1.0,
) -> torch.Tensor:
    """Build a [num_envs, 33] action tensor for SONIC hybrid forward walking.

    Layout (按 HumanoidActionsCfg 的 dataclass 字段顺序：base_action → arm_action → eef_action)：
        cols  0..4   : SonicWBC 5D `[vx, vy, ang_vel, height, mode]`
        cols  5..11  : Pink IK right wrist pose  (xyz + wxyz)  — **rest pose**
        cols 12..18  : Pink IK left wrist pose   (xyz + wxyz)  — **rest pose**
        cols 19..32  : eef 14D  (14 dex finger joint_pos)       — NaN 透传

    默认 `vx=0.3` 本体系前进，`mode=-1=AUTO` → planner 按 |v| 自动切 SLOW_WALK。
    Pink IK 每 tick 解到 rest pose → SonicArmBuffer 写入稳定外部 arm ref。
    """
    wbc = (
        torch.tensor(
            [vx, vy, ang_vel, height, mode],
            device=device,
            dtype=torch.float32,
        )
        .unsqueeze(0)
        .expand(num_envs, -1)
        .contiguous()
    )

    # Pink IK: right 7 + left 7 = 14 固定 rest pose
    right = torch.tensor(RIGHT_WRIST_REST_POSE_WXYZ, device=device, dtype=torch.float32)
    left = torch.tensor(LEFT_WRIST_REST_POSE_WXYZ, device=device, dtype=torch.float32)
    ik = torch.cat([right, left], dim=0).unsqueeze(0).expand(num_envs, -1).contiguous()

    # eef (dex fingers) 14D: NaN 透传保持初始
    eef_pad = torch.full(
        (num_envs, 14), float("nan"), device=device, dtype=torch.float32
    )

    action = torch.cat([wbc, ik, eef_pad], dim=-1)  # [num_envs, 33]
    return action


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="sonic_g1_config")
def main(cfg: DictConfig):
    new_seed = int(time.time())
    print(f"Initializing with seed: {new_seed}")
    cfg.sim.seed = new_seed

    logger = Logger("Env", log)
    env: SyncRobotEnv = gym.make(
        "SyncRobotEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    num_envs = getattr(env, "num_envs", 1) or 1
    # 以 0.3 m/s 向前走；AUTO mode 在 |v|=0.3 时会自动选 SLOW_WALK
    action = build_forward_walk_action(num_envs, env.device, vx=0.3)

    print("\n" + "=" * 64)
    print("SONIC G1 forward-walking test  (vx=0.3 m/s, AUTO → SLOW_WALK)")
    print(f"  num_envs = {num_envs}")
    print(f"  action   = {action.shape}  (base_5 | ik_14 | eef_14)")
    print(f"    wbc   [vx, vy, ang_vel, height, mode] = {action[0, :5].tolist()}")
    print("    ik_14/eef_14  —— NaN 透传")
    print("=" * 64 + "\n")

    # ------- Warm-up (5 ticks 让 planner 先跑一遍 + buffer 有效值) -------
    for _ in range(5):
        env.step(action=action)

    # ------- 主循环，每 50 tick (= 1 s) 打印 root 高度 / 位移 / 实测速度 -------
    start_xy: torch.Tensor | None = None
    prev_xy: torch.Tensor | None = None
    for t in range(10_000):
        env.step(action=action)

        if t % 50 == 0:
            try:
                art = env.sim.scene.articulations.get("robot")
                if art is None:
                    art = next(iter(env.sim.scene.articulations.values()))
                root_xy = art.data.root_pos_w[:, :2].clone()
                root_z = art.data.root_pos_w[:, 2]
                root_quat = art.data.root_state_w[:, 3:7]  # wxyz
                # yaw = atan2(2(w*z + x*y), 1 - 2(y² + z²))
                w, x, y, z_ = root_quat.unbind(-1)
                root_yaw = torch.atan2(
                    2.0 * (w * z_ + x * y), 1.0 - 2.0 * (y * y + z_ * z_)
                )

                if start_xy is None:
                    start_xy = root_xy.clone()
                    prev_xy = root_xy.clone()
                    measured_vel = torch.zeros(num_envs, device=root_xy.device)
                else:
                    measured_vel = (root_xy - prev_xy).norm(dim=-1)  # m/s
                    prev_xy = root_xy.clone()

                dx = root_xy[:, 0] - start_xy[:, 0]
                dy = root_xy[:, 1] - start_xy[:, 1]

                fallen = [bool(zz < 0.4) for zz in root_z.cpu().tolist()]
                # 单 env 信息（N=1 最常见；多 env 只看 env 0）
                print(
                    f"[t={t / 50:6.2f}s]  "
                    f"dx={dx[0]:+.3f}  dy={dy[0]:+.3f}  "
                    f"yaw={root_yaw[0]:+.3f}rad  "
                    f"z={root_z[0]:.3f}  "
                    f"vel_1s={measured_vel[0]:+.3f}m/s  "
                    f"fallen={fallen[0]}"
                )
                if any(fallen):
                    print("!! Fallen detected — terminating.")
                    break
            except Exception as e:
                print(f"[warn] failed to read robot state: {e}")


if __name__ == "__main__":
    main()
