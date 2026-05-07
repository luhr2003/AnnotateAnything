from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import DictConfig

from magicsim.Collect.GlobalPlanner.GlobalPlanner import (
    GlobalPlanner,
)
from magicsim.Env.Nav.NavManager import NavManager
from magicsim.Env.Planner.Utils import angle_diff
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv


def quat_wxyz_to_yaw(q):
    # q: (..., 4) in wxyz
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return yaw


class NavTo(GlobalPlanner):
    """
    Global Planner that moves a robot to a target pose.
    Similar to MoveL for robots, but for robot.
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        self.current_target = None
        self.robot_id = -1
        self.robot_name = None

        super().__init__(config, env, env_id, logger)
        # Default thresholds if not specified in config
        self.translation_threshold = float(
            getattr(config, "translation_threshold", 0.2)
        )
        self.rotation_threshold = float(getattr(config, "rotation_threshold", 0.1))
        self.segment_targets: List[torch.Tensor] = []
        # self.current_segment_idx: int = 0
        self.goal_reached: bool = False
        self.visualize_path: bool = bool(getattr(config, "visualize_path", True))
        # Optional flag to disable NavMesh-based pathing for the mobile.
        # When False, the camera will move directly in 3D from current pose to target pose.
        self.use_navmesh: bool = bool(getattr(config, "use_navmesh", True))
        # Path simplification parameters
        # Minimum distance between path segments (in meters). Points closer than this will be skipped.
        self.min_path_segment_distance: float = float(
            getattr(config, "min_path_segment_distance", 0.1)
        )
        # Maximum number of path segments. If set > 0, path will be downsampled to this many segments.
        self.num_path_segments: int = int(getattr(config, "num_path_segments", 5))
        self.start_forward_offset: float = float(
            getattr(config, "start_forward_offset", 0.35)
        )

    def _get_robot_name_list(self):
        return list(self.env.scene.robot_manager.robots.keys())

    def _set_robot_by_id(self, robot_id: int) -> bool:
        robot_id = int(robot_id)
        if robot_id == self.robot_id and self.robot_name is not None:
            return False
        robot_name_list = self._get_robot_name_list()
        if robot_id < 0 or robot_id >= len(robot_name_list):
            raise ValueError(
                f"robot_id {robot_id} out of range for robot_name_list={robot_name_list}"
            )
        self.robot_id = robot_id
        self.robot_name = robot_name_list[robot_id]
        return True

    def _parse_action(self, action: torch.Tensor):
        # Action format: (robot_id, target_tensor) or just target_tensor.
        robot_id = self.robot_id
        target = action
        if isinstance(action, (list, tuple)) and len(action) == 2:
            robot_id = int(action[0])
            target = action[1]
        return robot_id, target

    def reset(self, action: torch.Tensor):
        """
        Reset the planner with a new target.

        Args:
            action: Dictionary containing robot_name and target_pose
                   Format: {"robot_name": str, "target_pose": torch.Tensor [7]}
        """
        print("action: ", action)
        robot_id, target_action = self._parse_action(action)
        self._set_robot_by_id(robot_id)
        self.current_state = "ready"
        self.current_target = target_action
        self.current_command = [
            "NavTo",
            self.robot_name,
            self.current_target.clone(),
        ]

        self.segment_targets = []
        # self.current_segment_idx = 0
        self.goal_reached = False

        self._build_nav_path()

    def refresh(self, action: torch.Tensor):
        robot_id, target_action = self._parse_action(action)
        robot_changed = self._set_robot_by_id(robot_id)
        new_target = target_action

        # Rebuild threshold widened from 1e-2 -> 0.5m to avoid replanning the
        # NavMesh path on every centimeter of robot drift (the AtomicSkill recomputes
        # ``target`` each step → otherwise A* fires every frame and dominates wall-clock).
        rebuild = (
            self.current_target is None
            or new_target.shape != self.current_target.shape
            or torch.norm(new_target - self.current_target) > 0.5
        )
        self.current_target = new_target

        if rebuild:
            self.segment_targets = []
            # self.current_segment_idx = 0
            self.goal_reached = False
            self._build_nav_path()
            if not self.segment_targets:
                print(
                    f"[NavTo] Failed to generate nav path for env {self.env_id}, falling back to direct target"
                )
                self.segment_targets = [self.current_target.clone()]

        # Always set current_target to the active segment start
        # self.current_target = self.segment_targets[0]
        self.current_command = [
            "NavTo",
            self.robot_name,
            self.current_target.clone(),
        ]

    def step(self) -> Dict[str, torch.Tensor]:
        """
        Step the planner and return the current target pose.

        Returns:
            Dictionary with mobile_name and target_pose
            Format: {"robot_name": str, "target_pose": torch.Tensor [7]}
        """
        if self.current_target is None:
            raise RuntimeError("Current Target Is Not Set, Please Call Reset First")
        if self.robot_name is None:
            raise RuntimeError("Robot Name Is Not Set, Please Call Reset First")
        # self._advance_segment_if_reached()
        self.current_state = "running"
        # print("self.segment_targets: ", self.segment_targets)
        self.current_action = ("NavTo", self.segment_targets)
        path_tensor = torch.stack(self.segment_targets, dim=0)  # [T, 3] (x, y, yaw)

        # Match PlannerManager's expected action width. The mode_flag must land
        # on the **base slice**'s last column (that's what Nav.step reads), not
        # the full action's last column. Layout:
        #     [x, y, yaw][NaN pad -> base_dim-1][2 = Dwb flag][NaN pad -> total_dim]
        planner_mgr = getattr(self.env.scene, "planner_manager", None)
        if planner_mgr is not None:
            total_dim = int(planner_mgr.total_action_dim)
            base_slice = planner_mgr.planner_slice_dict.get(self.robot_name, {}).get(
                "base"
            )
            base_dim = (
                int(base_slice[1] - base_slice[0])
                if base_slice is not None
                else total_dim
            )

            T = path_tensor.shape[0]
            cur_width = path_tensor.shape[-1]

            # 1) pad within the base slice up to (base_dim - 1)
            inner_pad = max(0, (base_dim - 1) - cur_width)
            if inner_pad > 0:
                path_tensor = torch.cat(
                    [
                        path_tensor,
                        torch.full(
                            (T, inner_pad),
                            float("nan"),
                            device=path_tensor.device,
                            dtype=path_tensor.dtype,
                        ),
                    ],
                    dim=-1,
                )

            # 2) append flag=2 → base slice last column = Dwb route.
            path_tensor = torch.cat(
                [
                    path_tensor,
                    torch.full(
                        (T, 1),
                        2.0,
                        device=path_tensor.device,
                        dtype=path_tensor.dtype,
                    ),
                ],
                dim=-1,
            )  # [T, base_dim]

            # 3) pad the remaining arm/eef slots with NaN up to total_dim.
            outer_pad = max(0, total_dim - path_tensor.shape[-1])
            if outer_pad > 0:
                path_tensor = torch.cat(
                    [
                        path_tensor,
                        torch.full(
                            (T, outer_pad),
                            float("nan"),
                            device=path_tensor.device,
                            dtype=path_tensor.dtype,
                        ),
                    ],
                    dim=-1,
                )  # [T, total_dim]
        # print("path_tensor: ", path_tensor)
        return path_tensor

    def get_done(self) -> bool:
        """
        Check if the camera has reached the target pose.

        Returns:
            True if robot is within thresholds of target, False otherwise
        """
        if self.robot_name is None or self.current_target is None:
            return False

        self._update_goal_reached()
        if not self.goal_reached:
            return False

        current_pos, current_quat = self._get_robot_pose()
        target_xy = self.segment_targets[-1][:2]
        pos_distance = torch.norm(current_pos[:2] - target_xy)
        if pos_distance >= self.translation_threshold:
            return False

        target_yaw = self.current_target[2]
        if current_quat.shape[0] == 1:
            current_yaw = current_quat.squeeze(0)
        else:
            current_yaw = quat_wxyz_to_yaw(current_quat)
        yaw_error = torch.abs(angle_diff(current_yaw, target_yaw))
        return bool(yaw_error < self.rotation_threshold)

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the planner state based on environment feedback.

        Args:
            info: Environment information dictionary

        Returns:
            Dictionary with planner status information
        """
        if self.current_state == "failed":
            self.current_state = "failed: robot global planner failed to plan"
            result = {
                "type": "NavTo",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
            }
        elif self.get_done():
            self.current_state = "finished"
            result = {
                "type": "NavTo",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 0,
            }
        elif info.get("env_info") is not None and len(info["env_info"]) > 2:
            env_info = info["env_info"]
            if env_info[2][self.env_id]:
                self.current_state = "truncated: env terminated first"
                result = {
                    "type": "NavTo",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": True,
                    "state": self.current_state,
                    "truncated": 1,
                }
            elif len(env_info) > 3 and env_info[3][self.env_id]:
                self.current_state = "truncated: env truncated first"
                result = {
                    "type": "NavTo",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 2,
                }
            else:
                self.current_state = "running"
                result = {
                    "type": "NavTo",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 0,
                }
        else:
            self.current_state = "running"
            result = {
                "type": "NavTo",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }
        return result

    def _build_nav_path(self):
        """
        Use NavManager to plan a segmented path between the robot and the target pose.
        """
        tgt = self.current_target
        tgt_str = (
            tgt.detach().cpu().tolist() if isinstance(tgt, torch.Tensor) else str(tgt)
        )
        print(
            f"[NavTo GP] env={self.env_id} robot={self.robot_name} _build_nav_path target={tgt_str}"
        )
        if not self.use_navmesh:
            self.segment_targets = [self.current_target.clone()]
            return

        nav_manager: NavManager = getattr(self.env.scene, "nav_manager", None)
        if nav_manager is None or getattr(nav_manager, "navmesh_manager", None) is None:
            self.segment_targets = [self.current_target.clone()]
            return

        robot_pos, robot_quat = self._get_robot_pose()
        pos = robot_pos.detach()
        if robot_quat.shape[0] == 1:
            yaw = robot_quat.squeeze(0)
        else:
            yaw = quat_wxyz_to_yaw(robot_quat)

        dx = torch.cos(yaw)
        dy = torch.sin(yaw)

        start_pos = pos.clone()
        start_pos[0] += self.start_forward_offset * dx
        start_pos[1] += self.start_forward_offset * dy
        start_local = start_pos.cpu().numpy().copy()

        goal_local = self.current_target.detach().cpu().numpy().copy()
        goal_local[2] = start_local[2]

        coords = goal_local[:3].copy()
        paths = nav_manager.generate_path(
            start_point=[start_local],
            coords=[[coords]],
            env_ids=[int(self.env_id)],
            visualize=self.visualize_path,
        )

        if (
            not paths
            or not isinstance(paths, list)
            or not paths[0]
            or len(paths[0]) == 0
        ):
            self.segment_targets = [self.current_target.clone()]
            return

        path_points: List[torch.Tensor] = []
        device = self.current_target.device

        # Third component of atomic NavTo target is goal yaw (rad), not z.
        yaw_goal = float(self.current_target[2].detach().cpu())
        for point in paths[0]:
            point_tensor = torch.tensor(point, dtype=torch.float32, device=device)
            point_tensor[2] = yaw_goal
            path_points.append(point_tensor)

        if (
            len(path_points) == 0
            or torch.norm(path_points[-1][:2] - self.current_target[:2]) > 1e-3
        ):
            path_points.append(self.current_target.clone())
            path_points[-1][2] = yaw_goal

        idx = np.linspace(0, len(path_points) - 1, self.num_path_segments).astype(int)
        path_points = [path_points[i] for i in idx]
        path_points[-1] = self.current_target.clone()
        path_points[-1][2] = yaw_goal

        self.segment_targets = path_points or [self.current_target.clone()]
        # print("self.segment_targets: ", self.segment_targets)

    def _update_goal_reached(self):
        if not self.segment_targets or self.current_target is None:
            return

        current_pos, _ = self._get_robot_pose()
        target_xy = self.current_target[:2]
        distance = torch.norm(current_pos[:2] - target_xy)
        if distance <= self.translation_threshold:
            self.goal_reached = True
            return

    def _get_robot_pose(self):
        robot_state = self.env.scene.robot_manager.get_robot_state(noise_flag=False)[0][
            self.robot_name
        ]

        current_pos = robot_state["base_pos"][self.env_id].to(
            self.current_target.device
        )
        if current_pos.numel() == 2:
            current_pos = torch.cat(
                [current_pos, torch.tensor([0.0], device=current_pos.device)]
            )

        current_quat = robot_state["base_quat"][self.env_id].to(
            self.current_target.device
        )
        return current_pos, current_quat
