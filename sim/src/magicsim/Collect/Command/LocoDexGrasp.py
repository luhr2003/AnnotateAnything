from typing import Any, Dict

from magicsim.Collect.Command.Task import Task
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig, OmegaConf


class LocoDexGrasp(Task):
    """
    LocoDexGrasp task for mobile dexterous grasping.
    Mimics LocoReach: init phase sends placeholder (None obj), then real DexGrasp command.
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.obj_type = OmegaConf.select(config, "obj_type", default="rigid")
        self.obj_name = OmegaConf.select(config, "obj_name", default="bottle")
        self.obj_id = int(OmegaConf.select(config, "obj_id", default=0))
        self.functional_grasp = OmegaConf.select(
            config, "functional_grasp", default=True
        )
        self.functional_part = OmegaConf.select(config, "functional_part", default=None)
        self.init_count = 0
        self.init_limit = int(OmegaConf.select(config, "init_limit", default=150))
        self.max_attempt = int(OmegaConf.select(config, "max_attempt", default=2))
        self.attempt_count = 0

    def reset(self):
        self.current_state = "ready"
        self.current_action = None
        self.last_action = None
        self.init_count = 0
        self.attempt_count = 0

    def step(self):
        if self.current_state == "failed: task max attempt reached":
            return "Failed"
        if self.init_count < self.init_limit:
            self.init_count += 1
            self.current_state = "initializing"
            self.current_action = [
                "DexGrasp",
                0,  # robot_id
                0,  # hand_id
                None,
                None,
                None,
                None,
                None,
            ]
            self.last_action = None
            return self.current_action
        else:
            self.current_state = "running"
            self.current_action = [
                "DexGrasp",
                0,  # robot_id
                0,  # hand_id
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
                "type": "LocoDexGrasp",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }
        if self.current_state == "failed: task max attempt reached":
            return {
                "type": "LocoDexGrasp",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 5,
            }
        elif info["atomic_skill_info"][self.env_id]["finished"]:
            if info["env_info"][2][self.env_id]:
                self.current_state = "success: env terminated"
                return {
                    "type": "LocoDexGrasp",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": True,
                    "state": self.current_state,
                    "truncated": 1,
                }
            else:
                # Atomic skill finished but env not terminated - retry with limit
                self.attempt_count += 1
                if self.attempt_count > self.max_attempt:
                    self.current_state = "failed: task max attempt reached"
                    return {
                        "type": "LocoDexGrasp",
                        "last_action": self.last_action,
                        "current_action": self.current_action,
                        "finished": False,
                        "state": self.current_state,
                        "truncated": 5,
                    }
                # Re-initialize and retry
                self.init_count = 0
                self.current_state = "initializing"
                return {
                    "type": "LocoDexGrasp",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 0,
                }
        elif info["atomic_skill_info"][self.env_id]["truncated"] == 1:
            self.current_state = "success: env terminated first"
            return {
                "type": "LocoDexGrasp",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
            }
        elif info["atomic_skill_info"][self.env_id]["truncated"] == 2:
            self.current_state = "truncated: env truncated first"
            return {
                "type": "LocoDexGrasp",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
            }
        elif info["atomic_skill_info"][self.env_id]["truncated"] == 3:
            self.current_state = "failed: atomic skill failed to plan"
            return {
                "type": "LocoDexGrasp",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
            }
        elif info["atomic_skill_info"][self.env_id]["truncated"] == 4:
            self.current_state = "truncated: atomic skill failed to plan"
            return {
                "type": "LocoDexGrasp",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
            }
        else:
            self.current_state = "running"
            return {
                "type": "LocoDexGrasp",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }
