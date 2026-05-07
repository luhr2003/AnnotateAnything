from typing import Any, Dict
import torch
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class CameraGlobalPlanner:
    """
    Global Planner for camera tasks.
    Base class for camera global planners (mirrors robot GlobalPlanner abstraction).
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        self.config = config
        self.env = env
        self.env_id = env_id
        self.logger = logger
        self.current_target = None
        self.current_state = None
        self.current_command: list[Any] | None = None
        self.current_action: Dict[str, torch.Tensor] | None = None
        self.camera_name = None

    def reset(self, action: Dict[str, Any]):
        """
        Reset the planner with a new target.

        Args:
            action: Dictionary containing camera_name and target_pose
                   Format: {"camera_name": str, "target_pose": torch.Tensor [7]}
        """
        raise NotImplementedError

    def step(self) -> Dict[str, torch.Tensor]:
        """
        Step the planner and return the current target pose.

        Returns:
            Dictionary with camera_name and target_pose
            Format: {"camera_name": str, "target_pose": torch.Tensor [7]}
        """
        if self.current_target is None:
            raise RuntimeError("Current Target Is Not Set, Please Call Reset First")
        raise NotImplementedError

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the planner state based on environment feedback.

        Args:
            info: Environment information dictionary

        Returns:
            Dictionary with planner status information
        """
        raise NotImplementedError
