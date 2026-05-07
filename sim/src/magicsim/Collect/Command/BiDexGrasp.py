"""
BiDexGrasp command (Task layer): bimanual dexterous grasp.

Mirrors :class:`LocoDexBiGrasp` but issues the dedicated ``BiDexGrasp``
atomic skill (single-hand vs paired branches are now separate skills).
"""

from typing import Any, Dict

from magicsim.Collect.Command.Task import Task
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig, OmegaConf


class BiDexGrasp(Task):
    """Bimanual dex grasp (e.g. lift a bin with both Sharpa hands)."""

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.obj_type = OmegaConf.select(config, "obj_type", default="rigid")
        self.obj_name = OmegaConf.select(config, "obj_name", default="bin")
        self.obj_id = int(OmegaConf.select(config, "obj_id", default=0))
        self.functional_grasp = OmegaConf.select(
            config, "functional_grasp", default=True
        )
        self.functional_part = OmegaConf.select(config, "functional_part", default=None)
        self.robot_id = int(OmegaConf.select(config, "robot_id", default=0))
        self.init_count = 0
        self.init_limit = int(OmegaConf.select(config, "init_limit", default=100))
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
                "BiDexGrasp",
                self.robot_id,
                None,
                None,
                None,
                None,
                None,
            ]
            self.last_action = None
            return self.current_action

        self.current_state = "running"
        self.current_action = [
            "BiDexGrasp",
            self.robot_id,
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
        base = {
            "type": "BiDexGrasp",
            "last_action": self.last_action,
            "current_action": self.current_action,
        }
        if self.current_state == "initializing":
            return {
                **base,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }
        if self.current_state == "failed: task max attempt reached":
            return {
                **base,
                "finished": False,
                "state": self.current_state,
                "truncated": 5,
            }
        if info["atomic_skill_info"][self.env_id]["finished"]:
            if info["env_info"][2][self.env_id]:
                self.current_state = "success: env terminated"
                return {
                    **base,
                    "finished": True,
                    "state": self.current_state,
                    "truncated": 1,
                }
            self.attempt_count += 1
            if self.attempt_count > self.max_attempt:
                self.current_state = "failed: task max attempt reached"
                return {
                    **base,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 5,
                }
            self.init_count = 0
            self.current_state = "initializing"
            return {
                **base,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }

        trunc = info["atomic_skill_info"][self.env_id]["truncated"]
        if trunc == 1:
            self.current_state = "success: env terminated first"
            return {
                **base,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
            }
        if trunc == 2:
            self.current_state = "truncated: env truncated first"
            return {
                **base,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
            }
        if trunc == 3:
            self.current_state = "failed: atomic skill failed to plan"
            return {
                **base,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
            }
        if trunc == 4:
            self.current_state = "truncated: atomic skill failed to plan"
            return {
                **base,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
            }
        self.current_state = "running"
        return {**base, "finished": False, "state": self.current_state, "truncated": 0}
