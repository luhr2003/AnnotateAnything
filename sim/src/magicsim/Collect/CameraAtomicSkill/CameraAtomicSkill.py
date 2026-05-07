from typing import Any, Dict

from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class CameraAtomicSkill:
    """Base class for camera atomic skills (mirrors robot AtomicSkill abstraction)."""

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        self.config = config
        self.env = env
        self.env_id = env_id
        self.logger = logger
        self.camera_name: str | None = None
        self.current_state: str | None = None
        self.current_command: list[Any] | None = None
        self.current_action: Dict[str, Any] | None = None

    def reset(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def step(self) -> Dict[str, Any] | None:
        raise NotImplementedError

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
