from typing import Any, Dict

from magicsim.Collect.Command.Task import Task
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class DualReach(Task):
    """
    DualReach task for dual-arm TableTop.
    Sends Reach commands with hand_id=-1 (dual-arm mode),
    right arm -> cube_right, left arm -> cube_left.
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.right_obj_type = getattr(config, "right_obj_type", "geometry")
        self.right_obj_name = getattr(config, "right_obj_name", "cube_right")
        self.right_obj_id = int(getattr(config, "right_obj_id", 0))
        self.left_obj_type = getattr(config, "left_obj_type", "geometry")
        self.left_obj_name = getattr(config, "left_obj_name", "cube_left")
        self.left_obj_id = int(getattr(config, "left_obj_id", 0))

    def reset(self):
        self.current_state = "ready"
        self.current_action = None
        self.last_action = None

    def step(self):
        self.current_state = "running"
        # Reach atomic skill with 9-element dual-arm action
        self.current_action = [
            "Reach",
            0,
            -1,
            self.right_obj_type,
            self.right_obj_name,
            self.right_obj_id,
            self.left_obj_type,
            self.left_obj_name,
            self.left_obj_id,
        ]
        self.last_action = None
        return self.current_action

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        if self.current_state == "failed: task max attempt reached":
            return {
                "type": "DualReach",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 5,
            }
        if info["atomic_skill_info"][self.env_id]["finished"]:
            if info["env_info"][2][self.env_id]:
                self.current_state = "success: env terminated"
                return {
                    "type": "DualReach",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": True,
                    "state": self.current_state,
                    "truncated": 1,
                }
            else:
                self.current_state = "running"
                return {
                    "type": "DualReach",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 0,
                }
        elif info["atomic_skill_info"][self.env_id]["truncated"] == 1:
            self.current_state = "success: env terminated first"
            return {
                "type": "DualReach",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
            }
        elif info["atomic_skill_info"][self.env_id]["truncated"] == 2:
            self.current_state = "truncated: env truncated first"
            return {
                "type": "DualReach",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
            }
        elif info["atomic_skill_info"][self.env_id]["truncated"] == 3:
            self.current_state = "failed: atomic skill failed to plan"
            return {
                "type": "DualReach",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
            }
        elif info["atomic_skill_info"][self.env_id]["truncated"] == 4:
            self.current_state = "truncated: atomic skill failed to plan"
            return {
                "type": "DualReach",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
            }
        else:
            self.current_state = "running"
            return {
                "type": "DualReach",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }
