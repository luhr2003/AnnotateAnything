from typing import Any, Dict, List, Tuple

import torch
from magicsim.Collect.Command.Task import Task
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


def plan_segments_axis_aligned(
    start_xyz: torch.Tensor,
    goal_xyz: torch.Tensor,
    use_z: bool = False,
) -> List[Tuple[int, float]]:
    """
    规划轴对齐的推动段：每段只沿一个轴直线推，先 x 再 y 再 z，避免斜推导致偏移。

    Returns:
        List of (axis, target_value): axis 0=x, 1=y, 2=z；target_value 为该轴目标坐标。
        例如 [0,0,0]->[0.6,0.6,1.2] 得到 [(0, 0.6), (1, 0.6), (2, 1.2)]（若 use_z=True）。
    """
    s = start_xyz.squeeze()
    g = goal_xyz.squeeze()
    if s.dim() == 0:
        s = s.unsqueeze(0)
    if g.dim() == 0:
        g = g.unsqueeze(0)
    if s.numel() < 3:
        s = torch.nn.functional.pad(s, (0, 3 - s.numel()), value=0.0)
    if g.numel() < 3:
        g = torch.nn.functional.pad(g, (0, 3 - g.numel()), value=0.0)
    sx, sy, sz = s[0].item(), s[1].item(), s[2].item()
    gx, gy, gz = g[0].item(), g[1].item(), g[2].item()
    segments: List[Tuple[int, float]] = []
    if abs(gx - sx) > 1e-6:
        segments.append((0, gx))
    if abs(gy - sy) > 1e-6:
        segments.append((1, gy))
    if use_z and abs(gz - sz) > 1e-6:
        segments.append((2, gz))
    return segments


class Push(Task):
    """
    Push task: 将目标物体通过机械臂反复推动，从当前位置移动到目标位置。

    MPC 式规划：根据物体当前位置与 goal_position 生成轴对齐的路径点序列
    （例如 [0,0,0]->[1,1,0] 先到 [1,0,0] 再到 [1,1,0]），
    每次向 AtomicSkill 下发当前路径点作为物体目标，AtomicSkill 完成一段后再下发下一路径点，
    直到物体到达最终目标。
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.obj_type: str = getattr(config, "obj_type", "geometry")
        self.obj_name: str = getattr(config, "obj_name", "cube")
        self.obj_id: int = int(getattr(config, "obj_id", 0))
        # 物体目标位置 [x, y, z]（世界坐标系）
        goal = getattr(config, "goal_position", [1.0, 1.0, 0.0])
        if isinstance(goal, (list, tuple)):
            self.goal_position = torch.tensor(goal, dtype=torch.float32)
        else:
            self.goal_position = torch.tensor(goal, dtype=torch.float32)
        self.goal_tolerance: float = float(getattr(config, "goal_tolerance", 0.03))
        self.waypoint_tolerance: float = float(
            getattr(config, "waypoint_tolerance", 0.04)
        )
        self.use_z: bool = bool(getattr(config, "goal_use_z", False))

        # MPC 状态：轴对齐段 (axis, target_value)，每段只沿一轴直线推
        self.segments: List[Tuple[int, float]] = []
        self.current_segment_index: int = 0

    def _get_current_object_position(self) -> torch.Tensor:
        """获取当前物体位置 (3,) 或 (1,3)。"""
        pose = self.env.get_object_pose(
            [self.env_id], self.obj_type, self.obj_name, self.obj_id
        )
        return pose[0, :3]

    def _plan_segments(self) -> None:
        """根据当前物体位置与 goal_position 规划轴对齐段：先 x 再 y 再 z。"""
        current = self._get_current_object_position()
        goal = self.goal_position.to(device=current.device, dtype=current.dtype)
        self.segments = plan_segments_axis_aligned(current, goal, use_z=self.use_z)
        self.current_segment_index = 0

    def reset(self):
        self.current_state = "ready"
        self.current_action = None
        self.last_action = None
        self.segments = []
        self.current_segment_index = 0

    def step(self):
        """
        每次返回当前段的 Push 命令：沿单轴推到目标坐标。
        返回 ["Push", obj_type, obj_name, obj_id, axis, target_value]，axis 0=x,1=y,2=z。
        """
        self.current_state = "running"

        if not self.segments:
            self._plan_segments()

        if self.current_segment_index >= len(self.segments):
            self.current_action = None
            return None

        axis, target_value = self.segments[self.current_segment_index]
        self.current_action = [
            "Push",
            0,  # robot_id
            0,  # hand_id
            self.obj_type,
            self.obj_name,
            self.obj_id,
            int(axis),
            float(target_value),
        ]
        self.last_action = None
        return self.current_action

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据 atomic_skill_info 更新任务状态。
        当一段 Push 完成时：若物体已接近当前路径点则推进到下一路径点；
        若已接近最终目标则任务完成。
        """
        if self.current_state == "failed: task max attempt reached":
            return {
                "type": "Push",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 5,
            }
        atomic_info = info["atomic_skill_info"][self.env_id]

        if atomic_info["finished"]:
            # 本段推动结束，检查是否到达当前路径点并推进；仅当到达最终目标或全部路径点才 task finished=True
            try:
                current_pos = self._get_current_object_position()
            except Exception:
                current_pos = None
            goal_xyz = self.goal_position.to(
                device=current_pos.device if current_pos is not None else "cpu",
                dtype=current_pos.dtype if current_pos is not None else torch.float32,
            )
            if current_pos is not None and self.segments:
                axis, target_value = self.segments[self.current_segment_index]
                dist_along_axis = abs(current_pos[axis].item() - target_value)
                dist_to_goal = torch.norm(current_pos - goal_xyz).item()
                obj_xyz = [round(current_pos[i].item(), 4) for i in range(3)]
                goal_list = [round(goal_xyz[i].item(), 4) for i in range(3)]
                axis_name = "xyz"[axis]
                self.logger.info(
                    f"Push env_id={self.env_id} object_xyz={obj_xyz} goal={goal_list} "
                    f"segment={axis_name}={target_value} dist_axis={dist_along_axis:.4f} dist_goal={dist_to_goal:.4f} "
                    f"(tol_wp={self.waypoint_tolerance} tol_goal={self.goal_tolerance}) "
                    f"segment_idx={self.current_segment_index}/{len(self.segments)}"
                )
                if dist_to_goal <= self.goal_tolerance:
                    self.current_state = "success: object at goal"
                    self.logger.info(
                        f"Push env_id={self.env_id} task finished: object at {obj_xyz}, goal reached."
                    )
                    return {
                        "type": "Push",
                        "last_action": self.last_action,
                        "current_action": self.current_action,
                        "finished": True,
                        "state": self.current_state,
                        "truncated": 1,
                        "object_xyz": obj_xyz,
                        "goal_xyz": goal_list,
                    }
                if dist_along_axis <= self.waypoint_tolerance:
                    self.current_segment_index += 1
                    if self.current_segment_index >= len(self.segments):
                        self.current_state = "success: atomic finished (all segments)"
                        self.logger.info(
                            f"Push env_id={self.env_id} task finished: object at {obj_xyz}, all segments done."
                        )
                        return {
                            "type": "Push",
                            "last_action": self.last_action,
                            "current_action": self.current_action,
                            "finished": True,
                            "state": self.current_state,
                            "truncated": 1,
                            "object_xyz": obj_xyz,
                            "goal_xyz": goal_list,
                        }
                    self.current_state = "running: next segment"
                    return {
                        "type": "Push",
                        "last_action": self.last_action,
                        "current_action": self.current_action,
                        "finished": False,
                        "state": self.current_state,
                        "truncated": 0,
                        "object_xyz": obj_xyz,
                        "goal_xyz": goal_list,
                    }
                self.current_state = "running: retry segment"
                return {
                    "type": "Push",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 0,
                    "object_xyz": obj_xyz,
                    "goal_xyz": goal_list,
                }
            # 无法取到物体位置或 segments 为空：按 env 是否终止决定是否结束
            if info["env_info"][2][self.env_id]:
                self.current_state = "success: env terminated"
            else:
                self.current_state = "success: atomic finished"
            return {
                "type": "Push",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
            }
        elif atomic_info["truncated"] == 1:
            self.current_state = "success: env terminated first"
            return {
                "type": "Push",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
            }
        elif atomic_info["truncated"] == 2:
            self.current_state = "truncated: env truncated first"
            return {
                "type": "Push",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
            }
        elif atomic_info["truncated"] == 3:
            self.current_state = "failed: atomic skill failed to plan"
            return {
                "type": "Push",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
            }
        elif atomic_info["truncated"] == 4:
            self.current_state = "truncated: atomic skill failed to plan"
            return {
                "type": "Push",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
            }
        else:
            self.current_state = "running"
            return {
                "type": "Push",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }
