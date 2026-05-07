from typing import Any, Dict, List, Sequence
import torch
from magicsim.Collect.Command import STR2TASK
from magicsim.Collect.Command.Task import Task
from magicsim.Env.Utils.file import Logger
from omegaconf import DictConfig
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
import numpy as np


class AutoCollectManager:
    def __init__(
        self,
        env: TaskBaseEnv,
        num_envs: int,
        task_string: Dict[str, float],
        task_config: DictConfig,
        auto_collect_config: DictConfig,
        device=torch.device("cpu"),
        logger: Logger = None,
    ):
        self.env = env
        self.num_envs = num_envs
        self.auto_collect_config = auto_collect_config
        self.task_config = task_config
        self.task_string = task_string
        self.device = device
        self.logger = logger
        self.task_list: List[Task] = [None] * num_envs
        self.task_type_list: List[str] = [None] * num_envs
        self.info_list: List[Dict[str, Any]] = [None] * self.num_envs

    def get_next_task(self):
        task_type = np.random.choice(
            list(self.task_string.keys()), p=list(self.task_string.values())
        )
        return task_type

    def step(self, env_ids: Sequence[int]):
        robot_actions = []  # Note here we only return the action for env_ids
        valid_env_ids = []
        failed_env_ids = []
        for i, env_id in enumerate(env_ids):
            # Don't create new task if previous task just finished (info_list still has finished info)
            # Wait for AutoCollectEnv to reset first
            if self.task_type_list[env_id] is None:
                # No previous task or previous task was reset, create new task
                task_type = self.get_next_task()
                self.task_type_list[env_id] = task_type
                self.task_list[env_id] = STR2TASK[task_type](
                    self.task_config[task_type], self.env, env_id, self.logger
                )
                action = self.task_list[env_id].step()
            else:
                action = self.task_list[env_id].step()
            if action == "Failed":
                failed_env_ids.append(env_id)
            elif action is not None:
                valid_env_ids.append(env_id)
                # Robot tasks only return robot actions
                robot_actions.append(action)
        return robot_actions, valid_env_ids, failed_env_ids

    def update(self, info: Dict[str, Any]):
        # Note here we update the info for all environments
        for i, env_id in enumerate(range(self.num_envs)):
            if self.task_type_list[env_id] is None:
                # Task already finished, but keep the info_list entry so AutoCollectEnv can detect it
                # Don't clear info_list[env_id] here, let AutoCollectEnv handle it after reset
                continue

            self.info_list[env_id] = self.task_list[env_id].update(info)
            if (
                self.info_list[env_id]["finished"]
                or self.info_list[env_id]["truncated"] > 0
            ):
                print(
                    f"Task of env {env_id} is finished or truncated, state: {self.info_list[env_id].get('state', 'unknown')}"
                )
                # Clear task but keep info_list so AutoCollectEnv can detect completion
                self.task_type_list[env_id] = None
                self.task_list[env_id] = None
        return self.info_list

    def get_manager_info(self):
        return {"task_type_list": self.task_type_list, "task_list": self.task_list}

    def reset(self):
        return [None] * self.num_envs
