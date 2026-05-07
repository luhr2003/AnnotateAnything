"""LocoOpenDoor task: open door with mobile base (g1) + arm. Uses LocoOpenDoorEnv."""

from typing import Any, Dict
from magicsim.Collect.Command.Task import Task
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class LocoOpenDoor(Task):
    """
    LocoOpenDoor task: open door in LocoOpenDoorEnv (g1 at 0,0,0, door at 0,1.5,0).
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.init_count = 0
        self.init_limit = getattr(config, "init_limit", 20)
        self.max_attempt = getattr(config, "max_attempt", 2)
        self.attempt_count = 0
        self.robot_id = int(getattr(config, "robot_id", 0))
        self.hand_id = int(getattr(config, "hand_id", 0))
        self.obj_type = getattr(config, "obj_type", "articulation")
        self.obj_name = getattr(config, "obj_name", "articulation_items")
        self.obj_id = int(getattr(config, "obj_id", 0))
        self.joint_id = int(getattr(config, "joint_id", -1))

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
            "LocoOpenDoor",
            self.robot_id,
            self.hand_id,
            self.obj_type,
            self.obj_name,
            self.obj_id,
            self.joint_id,
        ]
        self.last_action = None
        if self.attempt_count >= self.max_attempt:
            self.current_state = "failed: task max attempt reached"
            return "Failed"
        return self.current_action

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        if self.current_state == "initializing":
            return {
                "type": "LocoOpenDoor",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }
        if self.current_state == "failed: task max attempt reached":
            return {
                "type": "LocoOpenDoor",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 5,
            }
        if info["atomic_skill_info"][self.env_id]["finished"]:
            self.current_state = "success: atomic skill finished"
            if info["env_info"][2][self.env_id]:
                return {
                    "type": "LocoOpenDoor",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": True,
                    "state": self.current_state,
                    "truncated": 0,
                    "attempt_count": self.attempt_count,
                }
            self.current_state = "running"
            self.attempt_count += 1
            if self.attempt_count > self.max_attempt:
                self.current_state = "failed: task max attempt reached"
                return {
                    "type": "LocoOpenDoor",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 5,
                    "attempt_count": self.attempt_count,
                }
            return {
                "type": "LocoOpenDoor",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
                "attempt_count": self.attempt_count,
            }
        if info["atomic_skill_info"][self.env_id]["truncated"] == 1:
            self.current_state = "success: env terminated first"
            return {
                "type": "LocoOpenDoor",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
                "attempt_count": self.attempt_count,
            }
        if info["atomic_skill_info"][self.env_id]["truncated"] == 2:
            self.current_state = "truncated: env truncated first"
            return {
                "type": "LocoOpenDoor",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
                "attempt_count": self.attempt_count,
            }
        if info["atomic_skill_info"][self.env_id]["truncated"] == 3:
            self.current_state = "failed: global planner failed to plan"
            return {
                "type": "LocoOpenDoor",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
                "attempt_count": self.attempt_count,
            }
        if info["atomic_skill_info"][self.env_id]["truncated"] == 4:
            self.current_state = "truncated: atomic skill failed to plan"
            return {
                "type": "LocoOpenDoor",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
                "attempt_count": self.attempt_count,
            }
        self.current_state = "running"
        return {
            "type": "LocoOpenDoor",
            "last_action": self.last_action,
            "current_action": self.current_action,
            "finished": False,
            "state": self.current_state,
            "truncated": 0,
            "attempt_count": self.attempt_count,
        }
