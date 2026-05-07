from typing import Any, Dict
from magicsim.Collect.Command.Task import Task
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class DexGrasp(Task):
    """
    Dexterous grasp task using XHand 3-phase grasp annotation.
    Dispatches to the DexGrasp atomic skill.
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
        self.hand_id = int(getattr(config, "hand_id", 0))
        self.obj_type = getattr(config, "obj_type", "rigid")
        self.obj_name = getattr(config, "obj_name", "bottle")
        self.obj_id = getattr(config, "obj_id", 0)
        self.functional_grasp = getattr(config, "functional_grasp", True)
        self.functional_part = getattr(config, "functional_part", None)

    def reset(self):
        self.current_state = "ready"
        self.current_action = None
        self.last_action = None
        self.attempt_count = 0

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
            "DexGrasp",
            self.robot_id,
            self.hand_id,
            self.obj_type,
            self.obj_name,
            self.obj_id,
            self.functional_grasp,
            self.functional_part,
        ]
        self.last_action = None
        if self.attempt_count >= self.max_attempt:
            self.current_state = "failed: task max attempt reached"
            return "Failed"
        return self.current_action

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        if self.current_state == "initializing":
            return {
                "type": "DexGrasp",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }

        if self.current_state == "failed: task max attempt reached":
            return {
                "type": "DexGrasp",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 5,
            }
        else:
            self.current_state = "running"

            atomic_info = info["atomic_skill_info"][self.env_id]

            if atomic_info["finished"]:
                self.current_state = "success: atomic skill finished"
                if info["env_info"][2][self.env_id]:
                    return {
                        "type": "DexGrasp",
                        "last_action": self.last_action,
                        "current_action": self.current_action,
                        "finished": True,
                        "state": self.current_state,
                        "truncated": 0,
                        "attempt_count": self.attempt_count,
                    }
                else:
                    self.attempt_count += 1
                    if self.attempt_count > self.max_attempt:
                        self.current_state = "failed: task max attempt reached"
                        return {
                            "type": "DexGrasp",
                            "last_action": self.last_action,
                            "current_action": self.current_action,
                            "finished": False,
                            "state": self.current_state,
                            "truncated": 5,
                            "attempt_count": self.attempt_count,
                        }
                    else:
                        # Re-initialize and retry
                        self.init_count = 0
                        self.current_state = "initializing"
                        self.current_action = None
                        return {
                            "type": "DexGrasp",
                            "last_action": self.last_action,
                            "current_action": self.current_action,
                            "finished": False,
                            "state": self.current_state,
                            "truncated": 0,
                            "attempt_count": self.attempt_count,
                        }

            trunc = atomic_info.get("truncated", 0)
            if trunc == 1:
                self.current_state = "success: env terminated first"
                return {
                    "type": "DexGrasp",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": True,
                    "state": self.current_state,
                    "truncated": 1,
                    "attempt_count": self.attempt_count,
                }
            if trunc == 2:
                self.current_state = "truncated: env truncated first"
                return {
                    "type": "DexGrasp",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 2,
                    "attempt_count": self.attempt_count,
                }
            if trunc == 3:
                self.current_state = "failed: global planner failed to plan"
                return {
                    "type": "DexGrasp",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 3,
                    "attempt_count": self.attempt_count,
                }
            if trunc == 4:
                self.current_state = "truncated: atomic skill failed to plan"
                return {
                    "type": "DexGrasp",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 4,
                    "attempt_count": self.attempt_count,
                }

            self.current_state = "running"
            return {
                "type": "DexGrasp",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
                "attempt_count": self.attempt_count,
            }
