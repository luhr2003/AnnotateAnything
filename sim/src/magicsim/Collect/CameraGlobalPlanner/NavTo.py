from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import DictConfig

from magicsim.Collect.CameraGlobalPlanner.CameraGlobalPlanner import (
    CameraGlobalPlanner,
)
from magicsim.Env.Nav.NavManager import NavManager
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv


class NavTo(CameraGlobalPlanner):
    """
    Camera Global Planner that moves a camera to a target pose.
    Similar to MoveL for robots, but for cameras.
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        # Default thresholds if not specified in config
        self.translation_threshold = float(
            getattr(config, "translation_threshold", 0.02)
        )
        self.rotation_threshold = float(getattr(config, "rotation_threshold", 0.1))
        self.segment_targets: List[torch.Tensor] = []
        self.current_segment_idx: int = 0
        self.goal_reached: bool = False
        self.visualize_path: bool = bool(getattr(config, "visualize_path", True))
        # Optional flag to disable NavMesh-based pathing for the camera.
        # When False, the camera will move directly in 3D from current pose to target pose.
        self.use_navmesh: bool = bool(getattr(config, "use_navmesh", True))
        # Path simplification parameters
        # Minimum distance between path segments (in meters). Points closer than this will be skipped.
        self.min_path_segment_distance: float = float(
            getattr(config, "min_path_segment_distance", 0.1)
        )
        # Maximum number of path segments. If set > 0, path will be downsampled to this many segments.
        self.max_path_segments: int = int(getattr(config, "max_path_segments", 0))

    def reset(self, action: Dict[str, Any]):
        """
        Reset the planner with a new target.

        Args:
            action: Dictionary containing camera_name and target_pose
                   Format: {"camera_name": str, "target_pose": torch.Tensor [7]}
        """
        if "camera_name" not in action or "target_pose" not in action:
            raise ValueError("Action must contain 'camera_name' and 'target_pose' keys")
        self.camera_name = action["camera_name"]
        self.current_target = action["target_pose"]
        if not isinstance(self.current_target, torch.Tensor):
            self.current_target = torch.tensor(self.current_target, dtype=torch.float32)
        self.current_target = self.current_target.to(torch.float32)
        # Clear all previous state to ensure clean reset
        self.segment_targets = []
        self.current_segment_idx = 0
        self.goal_reached = False
        # Ensure we get the current camera pose (which should be reset to initial position)
        # before building the nav path
        camera_pos, camera_quat = self._get_camera_pose()
        target_pos = self.current_target[:3]
        self._build_nav_path()
        # Always start from the first segment target
        self.current_segment_idx = 0
        self.current_target = self.segment_targets[0]
        self.current_command = [
            "NavTo",
            self.camera_name,
            self.current_target.clone(),
        ]
        self.current_state = "ready"
        return {"state": self.current_state}

    def refresh(self, action: Dict[str, Any]):
        """Update target while the planner is running, similar to robot MoveL.refresh."""
        if "camera_name" not in action or "target_pose" not in action:
            raise ValueError("Action must contain 'camera_name' and 'target_pose' keys")
        self.camera_name = action["camera_name"]
        new_target = action["target_pose"]
        if not isinstance(new_target, torch.Tensor):
            new_target = torch.tensor(new_target, dtype=torch.float32)
        new_target = new_target.to(torch.float32)

        # Rebuild path only if target changed
        rebuild = (
            self.current_target is None
            or new_target.shape != self.current_target.shape
            or torch.norm(new_target - self.current_target) > 1e-2
        )
        self.current_target = new_target

        if rebuild:
            self.segment_targets = []
            self.current_segment_idx = 0
            self.goal_reached = False
            self._build_nav_path()
            if not self.segment_targets:
                print(
                    f"[NavTo] Failed to generate nav path for env {self.env_id}, falling back to direct target"
                )
                self.segment_targets = [self.current_target.clone()]

        # Always set current_target to the active segment start
        self.current_target = self.segment_targets[0]
        self.current_command = [
            "NavTo",
            self.camera_name,
            self.current_target.clone(),
        ]

    def step(self) -> Dict[str, torch.Tensor]:
        """
        Step the planner and return the current target pose.

        Returns:
            Dictionary with camera_name and target_pose
            Format: {"camera_name": str, "target_pose": torch.Tensor [7]}
        """
        if self.current_target is None:
            raise RuntimeError("Current Target Is Not Set, Please Call Reset First")
        if self.camera_name is None:
            raise RuntimeError("Camera Name Is Not Set, Please Call Reset First")
        self._advance_segment_if_reached()
        self.current_state = "running"
        self.current_action = {
            "camera_name": self.camera_name,
            "target_pose": self.current_target,
        }
        return self.current_action

    def get_done(self) -> bool:
        """
        Check if the camera has reached the target pose.

        Returns:
            True if camera is within thresholds of target, False otherwise
        """
        if self.camera_name is None or self.current_target is None:
            return False

        self._advance_segment_if_reached()
        if not self.goal_reached:
            return False

        current_pos, current_quat = self._get_camera_pose()
        target_pos = self.segment_targets[-1][:3]
        target_quat = self.segment_targets[-1][3:]

        pos_distance = torch.norm(current_pos - target_pos)
        quat_distance = torch.norm(current_quat - target_quat)

        return bool(
            pos_distance < self.translation_threshold
            and quat_distance < self.rotation_threshold
        )

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the planner state based on environment feedback.

        Args:
            info: Environment information dictionary

        Returns:
            Dictionary with planner status information
        """
        result: Dict[str, Any]
        if self.current_state == "failed":
            self.current_state = "failed: camera global planner failed to plan"
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
        Use NavManager to plan a segmented path between the camera and the target pose.
        """
        # If NavMesh-based planning is disabled, just go straight to the target.
        if not self.use_navmesh:
            self.segment_targets = [self.current_target.clone()]
            return

        nav_manager: NavManager = getattr(self.env.scene, "nav_manager", None)
        if nav_manager is None or getattr(nav_manager, "navmesh_manager", None) is None:
            # Nav is not enabled for this environment
            self.segment_targets = [self.current_target.clone()]
            return

        camera_pos, _ = self._get_camera_pose()
        target_pos = self.current_target[:3]
        start_local = camera_pos.detach().cpu().numpy()
        goal_local = target_pos.detach().cpu().numpy()

        try:
            paths = nav_manager.generate_path(
                start_point=[start_local],
                coords=[[goal_local]],
                env_ids=[int(self.env_id)],
                visualize=self.visualize_path,
            )
        except Exception as exc:
            print(f"[NavTo] Failed to generate nav path for env {self.env_id}: {exc}")
            self.segment_targets = [self.current_target.clone()]
            return

        if (
            not paths
            or not isinstance(paths, list)
            or not paths[0]
            or len(paths[0]) == 0
        ):
            # Fallback to direct target if nav path is empty
            self.segment_targets = [self.current_target.clone()]
            return

        path_points = []
        device = self.current_target.device
        orientation = self.current_target[3:]
        # Use target Z coordinate for all path points, so camera moves at a fixed height
        # while following NavMesh's X, Y plan on the ground
        target_z = target_pos[2]
        for i, point in enumerate(paths[0]):
            point_tensor = (
                torch.tensor(point, dtype=torch.float32, device=device)
                if not isinstance(point, torch.Tensor)
                else point.to(device, dtype=torch.float32)
            )
            # Keep NavMesh's X, Y coordinates, but replace Z with target Z
            elevated_point = torch.tensor(
                [point_tensor[0], point_tensor[1], target_z],
                device=device,
                dtype=torch.float32,
            )
            # Apply minimum distance threshold to reduce path segments
            if (
                not path_points
                or torch.norm(elevated_point - path_points[-1][:3])
                > self.min_path_segment_distance
            ):
                path_points.append(torch.cat([elevated_point, orientation], dim=0))

        # Ensure final pose is the exact requested target
        if len(path_points) == 0 or torch.norm(path_points[-1][:3] - target_pos) > 1e-3:
            path_points.append(self.current_target.clone())

        # Drop the first point if it is identical to the current camera pose
        if len(path_points) > 1 and torch.norm(path_points[0][:3] - camera_pos) < 1e-4:
            path_points = path_points[1:]

        # Downsample path if max_path_segments is set
        if self.max_path_segments > 0 and len(path_points) > self.max_path_segments:
            # Uniformly sample points from the path
            indices = np.linspace(
                0, len(path_points) - 1, self.max_path_segments, dtype=int
            )
            path_points = [path_points[i] for i in indices]
            # Always keep the last point (final target)
            if torch.norm(path_points[-1][:3] - target_pos) > 1e-3:
                path_points[-1] = self.current_target.clone()

        self.segment_targets = path_points or [self.current_target.clone()]

    def _advance_segment_if_reached(self):
        if not self.segment_targets or self.current_target is None:
            return

        current_pos, _ = self._get_camera_pose()
        target_pos = self.current_target[:3]
        distance = torch.norm(current_pos - target_pos)
        if distance >= self.translation_threshold:
            return

        if self.current_segment_idx < len(self.segment_targets) - 1:
            self.current_segment_idx += 1
            self.current_target = self.segment_targets[self.current_segment_idx]
        else:
            self.goal_reached = True

    def _get_camera_pose(self):
        camera_state = self.env.scene.camera_manager.get_camera_state(
            camera_name=self.camera_name, env_ids=[int(self.env_id)]
        )
        current_pos = camera_state["pos"][0].to(self.current_target.device)
        current_quat = camera_state["quat"][0].to(self.current_target.device)
        return current_pos, current_quat
