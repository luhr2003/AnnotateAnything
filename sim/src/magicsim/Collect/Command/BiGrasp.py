"""BiGrasp command: issues the bimanual ``BiGrasp`` atomic skill.

The atomic skill drives both arms synchronously through pre_grasp → grasp
→ close_gripper → retrieval; this command just wraps that as a top-level
task so the AutoCollect framework can schedule it like any other task.
"""

from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from magicsim.Collect.Command.Task import Task
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv


class BiGrasp(Task):
    """Top-level wrapper around the BiGrasp atomic skill."""

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.robot_id = int(OmegaConf.select(config, "robot_id", default=0))
        self.obj_type = OmegaConf.select(config, "obj_type", default="rigid")
        self.obj_name = OmegaConf.select(config, "obj_name", default="basket")
        self.obj_id = int(OmegaConf.select(config, "obj_id", default=0))
        self.functional_grasp = bool(
            OmegaConf.select(config, "functional_grasp", default=True)
        )
        self.functional_part = OmegaConf.select(config, "functional_part", default=None)

        self.init_count = 0
        self.init_limit = int(OmegaConf.select(config, "init_limit", default=50))
        self.max_attempt = int(OmegaConf.select(config, "max_attempt", default=2))
        self.attempt_count = 0

    def reset(self):
        self.current_state = "ready"
        self.current_action = None
        self.last_action = None
        self.init_count = 0
        self.attempt_count = 0

    def _bigrasp_action(self) -> list:
        # AtomicSkill BiGrasp.reset signature:
        #   ["BiGrasp", robot_id, obj_type, obj_name, obj_id,
        #    functional_grasp, part]
        return [
            "BiGrasp",
            self.robot_id,
            self.obj_type,
            self.obj_name,
            self.obj_id,
            self.functional_grasp,
            self.functional_part,
        ]

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
        self.current_action = self._bigrasp_action()
        self.last_action = None
        if self.attempt_count >= self.max_attempt:
            self.current_state = "failed: task max attempt reached"
            return "Failed"
        return self.current_action

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        base = {
            "type": "BiGrasp",
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

        atomic = info["atomic_skill_info"][self.env_id]
        env_terminated = info["env_info"][2][self.env_id]

        if atomic["finished"]:
            if env_terminated:
                self.current_state = "success: env terminated"
                return {
                    **base,
                    "finished": True,
                    "state": self.current_state,
                    "truncated": 0,
                    "attempt_count": self.attempt_count,
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
                "state": "retrying: bigrasp finished but env not terminated",
                "truncated": 0,
                "attempt_count": self.attempt_count,
            }

        trunc = atomic["truncated"]
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
                "finished": True,
                "state": self.current_state,
                "truncated": 2,
            }
        if trunc in (3, 4):
            # Mark the task FINISHED (not just truncated) so AutoCollect
            # resets the env before the next attempt. Without this the
            # atomic skill gets re-created in place — same robot/basket
            # state, same paired-IK failure, infinite loop.
            self.current_state = (
                "failed: global planner failed to plan"
                if trunc == 3
                else "truncated: atomic skill failed to plan"
            )
            return {
                **base,
                "finished": True,
                "state": self.current_state,
                "truncated": trunc,
            }

        self.current_state = "running"
        return {**base, "finished": False, "state": self.current_state, "truncated": 0}
