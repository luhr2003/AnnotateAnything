from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class Task:
    """
    Base class for all tasks.
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        self.config = config
        self.env = env
        self.env_id = env_id
        self.logger = logger
        self.current_state = (
            None  # should be one of "ready", "running", "failed", "truncated"
        )
        self.current_action = None  # the current output action
        self.last_action = None  # the last output action

    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError
