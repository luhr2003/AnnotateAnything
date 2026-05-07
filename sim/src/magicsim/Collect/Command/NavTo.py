from typing import Any, Dict
from magicsim.Collect.Command.Task import Task
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class NavTo(Task):
    """
    NavTo task (robot navigation to target).
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        # Extract camera policy from config
        # robot_policy = config.get("robot_policy", {})
        # self.robot_name = robot_policy.get("robot_name", "Robot1")
        # self.obj_type = robot_policy.get("obj_type", "geometry")
        # self.obj_name = robot_policy.get("obj_name", "cube")
        self.robot_id = int(getattr(config, "robot_id", 0))
        self.obj_type = getattr(config, "obj_type", "geometry")
        self.obj_name = getattr(config, "obj_name", "cube")
        self.obj_id = int(getattr(config, "obj_id", 0))
        # self.init_count = 0
        # self.init_limit = getattr(config, "init_limit", 0)

    def reset(self):
        self.current_state = "ready"
        self.current_action = None
        self.last_action = None

    def step(self):
        """
        This function is used as MPC policy state transition function.
        In NavTo Task, we just do NavTo every time.
        """
        self.current_state = "running"
        self.current_action = [
            "NavTo",
            self.robot_id,
            self.obj_type,
            self.obj_name,
            self.obj_id,
        ]
        self.last_action = None
        return self.current_action

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        This function is used to update the task state.
        """
        if self.current_state == "failed: task max attempt reached":
            return {
                "type": "NavTo",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 5,
            }
        if info["atomic_skill_info"][self.env_id]["finished"]:
            self.current_state = "success"
            return {
                "type": "NavTo",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 0,
            }
        elif info["atomic_skill_info"][self.env_id]["truncated"] == 1:
            self.current_state = "truncated: env terminated first"
            return {
                "type": "NavTo",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
            }
        elif info["atomic_skill_info"][self.env_id]["truncated"] == 2:
            self.current_state = "truncated: env truncated first"
            return {
                "type": "NavTo",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
            }
        elif info["atomic_skill_info"][self.env_id]["truncated"] == 3:
            self.current_state = "failed: atomic skill failed to plan"
            return {
                "type": "NavTo",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
            }
        elif info["atomic_skill_info"][self.env_id]["truncated"] == 4:
            self.current_state = "truncated: atomic skill failed to plan"
            return {
                "type": "NavTo",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
            }
        else:
            self.current_state = "running"
            return {
                "type": "NavTo",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }
