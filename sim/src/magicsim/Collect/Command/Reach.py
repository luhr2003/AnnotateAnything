from typing import Any, Dict
from magicsim.Collect.Command.Task import Task
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class Reach(Task):
    """
    Reach task.
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        # 目标物体配置（可在 task/default.yaml 中的 Reach 下设置）
        # obj_type: 对应 SceneManager.get_category 的对象类型，如 "rigid" / "geometry"
        # obj_name: 在该对象集合中的类别 key，如 "mug" / "cube"
        # obj_id:   该类别下的第几个实例，一般为 0
        self.obj_type: str = getattr(config, "obj_type", "geometry")
        self.obj_name: str = getattr(config, "obj_name", "cube")
        self.obj_id: int = int(getattr(config, "obj_id", 0))

    def reset(self):
        self.current_state = "ready"
        self.current_action = None
        self.last_action = None

    def step(self):
        """
        This function is used as MPC policy state transition function.
        In Reach Task, we just do reach every time.

        """
        self.current_state = "running"
        # 目标由配置决定（默认为 geometry/cube/0）
        self.current_action = ["Reach", 0, 0, self.obj_type, self.obj_name, self.obj_id]
        self.last_action = None
        return self.current_action

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        This function is used to update the task state.
        """
        if self.current_state == "failed: task max attempt reached":
            return {
                "type": "Reach",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 5,
            }
        if info["atomic_skill_info"][self.env_id]["finished"]:
            if info["env_info"][2][self.env_id]:  # env terminated, real success
                self.current_state = "success: env terminated"
                return {
                    "type": "Reach",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": True,
                    "state": self.current_state,
                    "truncated": 1,
                }
            else:  # env do not terminated, but action finished, try last action again
                self.current_state = "running"
                return {
                    "type": "Reach",
                    "last_action": self.last_action,
                    "current_action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 0,
                }
        elif info["atomic_skill_info"][self.env_id]["truncated"] == 1:
            self.current_state = "success: env terminated first"
            return {
                "type": "Reach",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
            }
        elif info["atomic_skill_info"][self.env_id]["truncated"] == 2:
            self.current_state = "truncated: env truncated first"
            return {
                "type": "Reach",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
            }
        elif info["atomic_skill_info"][self.env_id]["truncated"] == 3:
            self.current_state = "failed: atomic skill failed to plan"
            return {
                "type": "Reach",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
            }
        elif info["atomic_skill_info"][self.env_id]["truncated"] == 4:
            self.current_state = "truncated: atomic skill failed to plan"
            return {
                "type": "Reach",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
            }
        else:
            self.current_state = "running"
            return {
                "type": "Reach",
                "last_action": self.last_action,
                "current_action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }
