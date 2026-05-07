from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from magicsim.Collect.AtomicSkill.AtomicSkill import AtomicSkill
from magicsim.Env.Utils.mesh_utils import (
    compute_reach_offset_from_bbox,
    get_world_bbox_half_extents,
)
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv


class Push(AtomicSkill):
    """
    Push AtomicSkill: 先 Reach 到物体附近，再沿给定方向推动一定距离。

    支持两种用法：
    - 仅 (obj_type, obj_name, obj_id)：使用配置的 push_direction 和 push_distance。
    - 带物体目标 (obj_type, obj_name, obj_id, target_x, target_y, target_z)：推动方向为
      当前物体位置指向目标点，推动距离取 min(配置值, 到目标点距离)，从而将物体推向目标点。
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.current_target_pose = None
        self.robot_id = int(getattr(config, "robot_id", 0))
        self.hand_id = int(getattr(config, "hand_id", 0))

        # 是否隔离模式：True 时不做 Reach，直接在当前末端位置开始 push
        self.isolate: bool = bool(getattr(config, "isolate", False))

        # 从配置中读取推的参数
        self.push_distance: float = float(getattr(config, "push_distance", 0.2))
        # 轴对齐推动时，每段至少推动该距离（米），避免剩余距离很小时轨迹过短、看起来“变慢”
        self.min_segment_push_distance: float = float(
            getattr(config, "min_segment_push_distance", 0.03)
        )
        direction = getattr(config, "push_direction", [1.0, 0.0, 0.0])
        self._config_push_direction = torch.tensor(direction, dtype=torch.float32)
        if torch.norm(self._config_push_direction) > 0:
            self._config_push_direction = self._config_push_direction / torch.norm(
                self._config_push_direction
            )
        else:
            self._config_push_direction = torch.tensor(
                [1.0, 0.0, 0.0], dtype=torch.float32
            )
        # 当前段的推动方向（可能由物体目标点动态设置）
        self.push_direction = self._config_push_direction.clone()

        # Reach 阶段目标相对于物体位置的偏移（世界坐标系，米）
        # reach_offset_auto=True 时根据目标物体 bbox 自动计算，避免接近时碰撞
        self.reach_offset_auto: bool = bool(getattr(config, "reach_offset_auto", False))
        self.reach_offset_margin: float = float(
            getattr(config, "reach_offset_margin", 0.02)
        )
        reach_offset = getattr(config, "reach_offset", [0.0, 0.0, 0.0])
        self.reach_offset = torch.tensor(reach_offset, dtype=torch.float32)

        # 推动分成多少步完成
        self.push_steps_total: int = int(getattr(config, "push_steps", 50))

        # 内部状态机：approach -> push -> done
        self._phase: str = "approach"
        self._push_step_count: int = 0

        # 记录起始 push pose（在接近物体完成时的末端目标 pose）
        self._push_start_pos: torch.Tensor | None = None
        self._push_start_quat: torch.Tensor | None = None
        self._logged_reach_offset_mode: bool = False
        # 本段推动距离（当由物体目标点驱动时可能小于 push_distance）
        self._segment_push_distance: float | None = None

    def _get_object_position_raw(self):
        """当前物体中心位置 (3,) 世界坐标，无偏移。"""
        scene_mgr = self.env.scene.scene_manager
        env_id = int(self.env_id)
        rigid_env = scene_mgr.rigid_objects[env_id]
        geo_env = scene_mgr.geometry_objects[env_id]
        obj_list = rigid_env.get(self.obj_name, [])
        if not obj_list:
            obj_list = geo_env.get(self.obj_name, [])
        if not obj_list or self.obj_id >= len(obj_list):
            raise RuntimeError(
                f"Push: no valid target object obj_name='{self.obj_name}' "
                f"obj_id={self.obj_id} env_id={env_id}"
            )
        obj = obj_list[self.obj_id]
        pos, _ = obj.get_local_pose()
        return pos.squeeze()

    def _get_object_pose(self):
        """Get current 7D pose (pos+quat) of target object with configured reach offset."""
        # 兼容 rigid / geometry 两种存储
        scene_mgr = self.env.scene.scene_manager
        env_id = int(self.env_id)

        rigid_env = scene_mgr.rigid_objects[env_id]
        geo_env = scene_mgr.geometry_objects[env_id]

        obj_list = rigid_env.get(self.obj_name, [])
        if not obj_list:
            obj_list = geo_env.get(self.obj_name, [])

        if not obj_list or self.obj_id >= len(obj_list):
            raise RuntimeError(
                f"Push: no valid target object found for "
                f"obj_name='{self.obj_name}', obj_id={self.obj_id}, "
                f"env_id={env_id}. rigid_keys={list(rigid_env.keys())}, "
                f"geo_keys={list(geo_env.keys())}"
            )

        obj = obj_list[self.obj_id]
        pos, quat = obj.get_local_pose()  # pos(3), quat(4)

        # 应用 reach 偏移（世界坐标系）：自动根据 bbox 或使用配置的固定值
        if self.reach_offset_auto and hasattr(obj, "prim") and obj.prim is not None:
            half_ext = get_world_bbox_half_extents(obj.prim)
            if half_ext is not None:
                if not self._logged_reach_offset_mode:
                    self.logger.info(
                        f"Push env_id={self.env_id} reach_offset: auto (from bbox), "
                        f"half_extents={half_ext}, margin={self.reach_offset_margin}"
                    )
                    self._logged_reach_offset_mode = True
                d = (
                    self.push_direction.numpy()
                    if isinstance(self.push_direction, torch.Tensor)
                    else np.array(self.push_direction)
                )
                offset_np = compute_reach_offset_from_bbox(
                    half_ext, d, self.reach_offset_margin
                )
                offset = torch.tensor(offset_np, device=pos.device, dtype=pos.dtype)
            else:
                if not self._logged_reach_offset_mode:
                    self.logger.info(
                        f"Push env_id={self.env_id} reach_offset: manual (bbox failed), "
                        f"offset={self.reach_offset.tolist()}"
                    )
                    self._logged_reach_offset_mode = True
                offset = self.reach_offset.to(pos.device, dtype=pos.dtype)
        else:
            if not self._logged_reach_offset_mode:
                self.logger.info(
                    f"Push env_id={self.env_id} reach_offset: manual, "
                    f"offset={self.reach_offset.tolist()}"
                )
                self._logged_reach_offset_mode = True
            offset = self.reach_offset.to(pos.device, dtype=pos.dtype)
        pos = pos + offset
        return pos, quat  # (pos(3), quat(4))

    def _get_eef_pose(self):
        """Get current 7D pose (pos+quat) of end effector for this env."""
        eef_pose = self.env.get_eef_pose([self.env_id])
        pos = eef_pose[0, :3]
        quat = eef_pose[0, 3:7]
        return pos, quat

    def reset(self, action: list[Any]):
        # action: ["Push", robot_id, hand_id, obj_type, obj_name, obj_id]
        #   或 ["Push", robot_id, hand_id, ..., target_x, target_y, target_z]
        #   或 ["Push", robot_id, hand_id, ..., axis, target_value] 轴对齐
        self.robot_id = int(action[1])
        self.hand_id = int(action[2])
        self.obj_type = action[3]
        self.obj_name = action[4]
        self.obj_id = action[5]
        self.current_state = "ready"
        self.current_command = list(action[:6])
        self._push_step_count = 0
        self._push_start_pos = None
        self._push_start_quat = None
        self._logged_reach_offset_mode = False
        self._segment_push_distance = None

        # 8 元组：轴对齐推动 (axis, target_value)，先 x 再 y 再 z，直线推
        if len(action) == 8 and action[6] in (0, 1, 2):
            axis = int(action[6])
            target_value = float(action[7])
            self.current_command = list(action[:8])
            try:
                current_pos = self._get_object_position_raw()
                self.push_direction = torch.zeros(
                    3, dtype=torch.float32, device=self.env.device
                )
                cur_val = current_pos[axis].item()
                if target_value > cur_val:
                    self.push_direction[axis] = 1.0
                    remain = abs(target_value - cur_val)
                    self._segment_push_distance = max(
                        self.min_segment_push_distance,
                        min(self.push_distance, remain),
                    )
                elif target_value < cur_val:
                    self.push_direction[axis] = -1.0
                    remain = abs(target_value - cur_val)
                    self._segment_push_distance = max(
                        self.min_segment_push_distance,
                        min(self.push_distance, remain),
                    )
                else:
                    self.push_direction = self._config_push_direction.to(
                        self.env.device
                    )
                    self._segment_push_distance = 0.0  # 已到该轴目标，本段不位移
            except Exception:
                self.push_direction = self._config_push_direction.to(self.env.device)
        # 9 元组：目标点 (target_x, target_y, target_z)，方向为 当前->目标
        elif len(action) >= 9:
            target_xyz = torch.tensor(
                [float(action[6]), float(action[7]), float(action[8])],
                dtype=torch.float32,
                device=self.env.device,
            )
            self.current_command = list(action[:9])
            try:
                current_pos = self._get_object_position_raw()
                if current_pos.device != target_xyz.device:
                    target_xyz = target_xyz.to(current_pos.device)
                diff = target_xyz - current_pos
                dist = torch.norm(diff).item()
                if dist > 1e-6:
                    self.push_direction = (diff / dist).to(
                        dtype=torch.float32, device=self.env.device
                    )
                    self._segment_push_distance = min(self.push_distance, float(dist))
                else:
                    self.push_direction = self._config_push_direction.to(
                        self.env.device
                    )
            except Exception:
                self.push_direction = self._config_push_direction.to(self.env.device)
        else:
            self.push_direction = self._config_push_direction.to(self.env.device)

        if self.isolate:
            eef_pos, eef_quat = self._get_eef_pose()
            self._push_start_pos = eef_pos
            self._push_start_quat = eef_quat
            self._phase = "push"
            self.current_target_pose = torch.cat([eef_pos, eef_quat], dim=0)
        else:
            self.current_target_pose = self._get_object_pose()
            self._phase = "approach"

    def refresh(self, action: list[Any]):
        self.robot_id = int(action[1])
        self.hand_id = int(action[2])
        self.obj_type = action[3]
        self.obj_name = action[4]
        self.obj_id = action[5]
        self.current_command = list(action[:6])
        if len(action) == 8 and action[6] in (0, 1, 2):
            self.current_command = list(action[:8])
            axis = int(action[6])
            target_value = float(action[7])
            try:
                current_pos = self._get_object_position_raw()
                self.push_direction = torch.zeros(
                    3, dtype=torch.float32, device=self.env.device
                )
                cur_val = current_pos[axis].item()
                if target_value > cur_val:
                    self.push_direction[axis] = 1.0
                    remain = abs(target_value - cur_val)
                    self._segment_push_distance = max(
                        self.min_segment_push_distance,
                        min(self.push_distance, remain),
                    )
                elif target_value < cur_val:
                    self.push_direction[axis] = -1.0
                    remain = abs(target_value - cur_val)
                    self._segment_push_distance = max(
                        self.min_segment_push_distance,
                        min(self.push_distance, remain),
                    )
                else:
                    self.push_direction = self._config_push_direction.to(
                        self.env.device
                    )
                    self._segment_push_distance = 0.0
            except Exception:
                self.push_direction = self._config_push_direction.to(self.env.device)
                self._segment_push_distance = None
        elif len(action) >= 9:
            self.current_command = list(action[:9])
            target_xyz = torch.tensor(
                [float(action[6]), float(action[7]), float(action[8])],
                dtype=torch.float32,
                device=self.env.device,
            )
            try:
                current_pos = self._get_object_position_raw()
                if current_pos.device != target_xyz.device:
                    target_xyz = target_xyz.to(current_pos.device)
                diff = target_xyz - current_pos
                dist = torch.norm(diff).item()
                if dist > 1e-6:
                    self.push_direction = (diff / dist).to(
                        dtype=torch.float32, device=self.env.device
                    )
                    self._segment_push_distance = min(self.push_distance, float(dist))
                else:
                    self.push_direction = self._config_push_direction.to(
                        self.env.device
                    )
                    self._segment_push_distance = None
            except Exception:
                self.push_direction = self._config_push_direction.to(self.env.device)
                self._segment_push_distance = None
        else:
            self.push_direction = self._config_push_direction.to(self.env.device)
            self._segment_push_distance = None
        self.current_target_pose = self._get_object_pose()

    def _build_movl_action(self, pos: torch.Tensor, quat: torch.Tensor):
        """Helper to build MoveL action dict with [pos, quat, gripper_flag]."""
        # 这里沿用 Reach 的约定：最后一维 1.0 代表 gripper 闭合
        return {
            "MoveL": (
                (self.robot_id, self.hand_id, -1),
                torch.cat(
                    [
                        pos,
                        quat,
                        torch.tensor([1.0], device=self.env.device),
                    ],
                    dim=0,
                ),
            )
        }

    def step(self):
        # 根据当前 phase 构造 action
        if self._phase == "done":
            # 已经完成，不再输出新动作
            return None

        if self._phase == "approach":
            # 阶段 1：像 Reach 一样先到达物体 pose
            self.current_target_pose = self._get_object_pose()
            target_pos, target_quat = self.current_target_pose
            self.current_state = "running"
            self.current_action = self._build_movl_action(target_pos, target_quat)
            return self.current_action

        # phase == "push"：沿设定方向均匀推进
        if self._push_start_pos is None or self._push_start_quat is None:
            # 理论上不会走到这里（_phase 切到 push 时已经设置），但加个保护
            self.current_target_pose = self._get_object_pose()
            self._push_start_pos, self._push_start_quat = self.current_target_pose

        # 线性插值推进：t 从 0 -> 1；本段距离可能由物体目标点截断
        step_dist = (
            self._segment_push_distance
            if self._segment_push_distance is not None
            else self.push_distance
        )
        t = min(
            float(self._push_step_count + 1) / float(max(self.push_steps_total, 1)), 1.0
        )
        total_offset = self.push_direction.to(self.env.device) * step_dist
        cur_offset = total_offset * t
        push_pos = self._push_start_pos.to(self.env.device) + cur_offset
        push_quat = self._push_start_quat.to(self.env.device)  # 推动时保持姿态不变

        self.current_state = "running"
        self.current_action = self._build_movl_action(push_pos, push_quat)
        self._push_step_count += 1

        # 推动完成
        if self._push_step_count >= self.push_steps_total:
            self._phase = "done"

        return self.current_action

    def update(self, info):
        """
        根据 global_planner_info 更新 Push 的状态机：
        - approach 阶段：等待全局 planner 报 finished，然后切到 push 阶段；
        - push 阶段：由内部 step 的 _phase 控制，done 后返回 finished=True；
        """
        if self.current_state == "failed":
            self.current_state = "failed: atomicskill failed to plan"
            return {
                "atomic_skill_type": "Push",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
            }

        gp_info = info["global_planner_info"][self.env_id]

        if self._phase == "approach":
            if gp_info["finished"]:
                # 到达目标，切换到 push 阶段，并记录起始 pose
                self._phase = "push"
                self._push_step_count = 0
                # 使用当前物体 pose 作为起始 push pose
                self.current_target_pose = self._get_object_pose()
                self._push_start_pos, self._push_start_quat = self.current_target_pose
                self.current_state = "running"
                return {
                    "type": "Push",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": "push_started",
                    "truncated": 0,
                }
            elif gp_info["truncated"] == 1:
                self.current_state = "truncated: env terminated first"
                return {
                    "type": "Push",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": True,
                    "state": self.current_state,
                    "truncated": 1,
                }
            elif gp_info["truncated"] == 2:
                self.current_state = "truncated: env truncated first"
                return {
                    "type": "Push",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 2,
                }
            elif gp_info["truncated"] == 3:
                self.current_state = "failed: global planner failed to plan"
                return {
                    "type": "Push",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 3,
                }
            else:
                # 仍在接近阶段
                self.current_state = "running"
                return {
                    "type": "Push",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 0,
                }

        # push 阶段或 done：根据 _phase 决定是否完成
        if self._phase == "done":
            self.current_state = "finished"
            return {
                "type": "Push",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 0,
            }

        # 仍在 push 中
        self.current_state = "running"
        return {
            "type": "Push",
            "command": self.current_command,
            "action": self.current_action,
            "finished": False,
            "state": f"push_running_{self._push_step_count}",
            "truncated": 0,
        }
