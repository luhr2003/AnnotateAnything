from typing import Any, Dict

from magicsim.Collect.Command.Task import Task
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class Fold(Task):
    """Bimanual two-stage garment-fold high-level task.

    Hands a single ``["Fold", robot_id, obj_type, obj_name, obj_id]``
    command to the ``Fold`` atomic skill, which drives the bimanual
    DualFranka robot through the sleeve fold followed by the
    bottom-to-shoulder fold.

    Config keys:
        robot_id:    int, robot index (dual-arm; hand_id fixed to -1)
        obj_type:    str, scene_manager category type (e.g. "garment")
        obj_name:    str, category key (e.g. "garment_items")
        obj_id:      int, index within that category (default 0)
        init_limit:  int, idle steps before the first command
        max_attempt: int, max retries before giving up
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.init_count = 0
        self.init_limit = getattr(config, "init_limit", 50)
        self.max_attempt = getattr(config, "max_attempt", 2)
        self.attempt_count = 0

        self.robot_id = int(getattr(config, "robot_id", 0))
        self.obj_type = str(getattr(config, "obj_type", "garment"))
        self.obj_name = str(getattr(config, "obj_name", "garment_items"))
        self.obj_id = int(getattr(config, "obj_id", 0))

    def reset(self):
        self.current_state = "ready"
        self.current_action = None
        self.last_action = None
        self.attempt_count = 0
        self.init_count = 0

    def step(self):
        if self.current_state == "failed: task max attempt reached":
            return "Failed"
        if self.init_count < self.init_limit:
            self.init_count += 1
            self.current_state = "initializing"
            self.current_action = None
            self.last_action = None
            return None

        self.current_state = "running"
        self.current_action = [
            "Fold",
            self.robot_id,
            self.obj_type,
            self.obj_name,
            self.obj_id,
        ]
        self.last_action = None
        if self.attempt_count >= self.max_attempt:
            self.current_state = "failed: task max attempt reached"
            return "Failed"
        return self.current_action

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        if self.current_state == "initializing":
            return {
                "type": "Fold",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }
        if self.current_state == "failed: task max attempt reached":
            return {
                "type": "Fold",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 5,
            }

        atomic_info = info["atomic_skill_info"][self.env_id]
        if atomic_info["finished"]:
            self.current_state = "success: atomic skill finished"
            self.current_state = "running"
            self.attempt_count += 1
            if self.attempt_count > self.max_attempt:
                self.current_state = "failed: task max attempt reached"
                return {
                    "type": "Fold",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 5,
                    "attempt_count": self.attempt_count,
                }
            self.init_count = 0
            self.current_state = "initializing"
            self.current_action = None
            return {
                "type": "Fold",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
                "attempt_count": self.attempt_count,
            }
        if atomic_info["truncated"] == 1:
            self.current_state = "success: env terminated first"
            return {
                "type": "Fold",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
                "attempt_count": self.attempt_count,
            }
        if atomic_info["truncated"] == 2:
            self.current_state = "truncated: env truncated first"
            return {
                "type": "Fold",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
                "attempt_count": self.attempt_count,
            }
        if atomic_info["truncated"] == 3:
            self.current_state = "failed: global planner failed to plan"
            return {
                "type": "Fold",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
                "attempt_count": self.attempt_count,
            }
        if atomic_info["truncated"] == 4:
            self.current_state = "truncated: atomic skill failed to plan"
            return {
                "type": "Fold",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
                "attempt_count": self.attempt_count,
            }

        self.current_state = "running"
        return {
            "type": "Fold",
            "last_action": self.last_action,
            "current_action": self.current_action,
            "finished": False,
            "state": self.current_state,
            "truncated": 0,
            "attempt_count": self.attempt_count,
        }
