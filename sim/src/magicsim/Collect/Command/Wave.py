from typing import Any, Dict
from magicsim.Collect.Command.Task import Task
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class Wave(Task):
    """
    Wave task.

    Similar to Grasp task, but uses Wave AtomicSkill which performs
    grasp followed by random jittering/waving.
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.init_count = 0
        self.init_limit = getattr(config, "init_limit", 20)
        self.max_attempt = getattr(config, "max_attempt", 2)
        self.attempt_count = 0

    def reset(self):
        self.current_state = "ready"
        self.current_action = None
        self.last_action = None
        self.attempt_count = 0

    def step(self):
        """
        This function is used as MPC policy state transition function.
        In Wave Task, we just do wave (grasp + jitter) every time.
        """
        if self.current_state == "failed: task max attempt reached":
            return "Failed"
        if self.init_count < self.init_limit:
            self.init_count += 1
            self.current_state = "initializing"
            self.current_action = None
            self.last_action = None
            return None
        else:
            self.current_state = "running"
            self.current_action = ["Wave", "rigid", "mug", 0]
            self.last_action = None
            if self.attempt_count >= self.max_attempt:
                self.current_state = "failed: task max attempt reached"
                return "Failed"
            return self.current_action

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        This function is used to update the task state.
        """
        if self.current_state == "initializing":
            return {
                "type": "Wave",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }
        elif self.current_state == "failed: task max attempt reached":
            return {
                "type": "Wave",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 5,
            }
        else:
            if info["atomic_skill_info"][self.env_id]["finished"]:
                # Wave 任务：AtomicSkill 完成（含抓取+晃动）即视为成功，不依赖 env terminate。
                # WaveEnv 不会因成功而 terminate，由存盘后 need_reset 触发 reset_idx。
                self.current_state = "success: atomic skill finished"
                return {
                    "type": "Wave",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": True,
                    "state": self.current_state,
                    "truncated": 0,
                    "attempt_count": self.attempt_count,
                }
            elif info["atomic_skill_info"][self.env_id]["truncated"] == 1:
                self.current_state = "success: env terminated first"
                return {
                    "type": "Wave",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": True,
                    "state": self.current_state,
                    "truncated": 1,
                    "attempt_count": self.attempt_count,
                }
            elif info["atomic_skill_info"][self.env_id]["truncated"] == 2:
                self.current_state = "truncated: env truncated first"
                return {
                    "type": "Wave",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 2,
                    "attempt_count": self.attempt_count,
                }
            elif info["atomic_skill_info"][self.env_id]["truncated"] == 3:
                self.current_state = "failed: global planner failed to plan"
                return {
                    "type": "Wave",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 3,
                    "attempt_count": self.attempt_count,
                }
            elif info["atomic_skill_info"][self.env_id]["truncated"] == 4:
                self.current_state = "truncated: atomic skill failed to plan"
                return {
                    "type": "Wave",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 4,
                    "attempt_count": self.attempt_count,
                }
            else:
                self.current_state = "running"
                return {
                    "type": "Wave",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 0,
                    "attempt_count": self.attempt_count,
                }
