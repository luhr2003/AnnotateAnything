from typing import Any, List, Sequence

import torch
import gymnasium as gym
from magicsim.Collect.AtomicSkill.AtomicSkillManager import AtomicSkillManager
from magicsim.Collect.GlobalPlanner.GlobalPlannerManager import GlobalPlannerManager
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv


class AsyncRobotEnv(gym.Env):
    def __init__(self, config, cli_args, logger):
        self.env_string = config.env_string
        self.env_config = config.env
        # Pass task config to env so it can access task-specific settings (e.g., obj_name)
        if hasattr(config, "task"):
            from omegaconf import OmegaConf

            struct_flag = OmegaConf.is_struct(self.env_config)
            if struct_flag:
                OmegaConf.set_struct(self.env_config, False)
            self.env_config.task = config.task
            if struct_flag:
                OmegaConf.set_struct(self.env_config, True)
        self.env: TaskBaseEnv = gym.make(
            self.env_string, config=self.env_config, cli_args=cli_args, logger=logger
        )
        self.num_envs = self.env.num_envs
        self.scene = self.env.scene
        self.device = self.env.device
        self.logger = logger
        self.global_planner_config = config.global_planner
        self.global_planner_manager = GlobalPlannerManager(
            self.env,
            self.num_envs,
            self.global_planner_config,
            self.device,
            self.logger,
        )
        self.atomic_skill_config = config.atomic_skill
        self.atomic_skill_manager = AtomicSkillManager(
            self.env,
            self.num_envs,
            self.atomic_skill_config,
            self.device,
            self.logger,
        )

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        env_info = self.env.reset(seed, options)
        global_planner_info = self.global_planner_manager.reset()
        atomic_skill_info = self.atomic_skill_manager.reset()
        return {
            "env_info": env_info,
            "global_planner_info": global_planner_info,
            "atomic_skill_info": atomic_skill_info,
        }

    def step(
        self,
        actions: List[List[str]],
        env_ids: Sequence[int] | None = None,
        failed_env_ids: Sequence[int] | None = None,
    ):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        valid_env_ids = env_ids

        # Process robot actions
        actions, valid_env_ids, failed_env_at_atomic_skill = (
            self.atomic_skill_manager.step(actions, env_ids)
        )
        # print("actions: ", actions)
        actions, valid_env_ids, failed_env_at_global_planner = (
            self.global_planner_manager.step(actions, valid_env_ids)
        )
        failed_env_ids = (
            failed_env_at_atomic_skill + failed_env_at_global_planner + failed_env_ids
        )

        record_info = {}

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
            pending_env_ids,
        ) = self.env.step(
            action=actions,
            env_ids=valid_env_ids,
            failed_env_ids=failed_env_ids,
        )
        env_info = (obs, reward, terminated, truncated, info)
        # print("info in AsyncRobotEnv step: ", info)
        valid_env_ids = [x for x in valid_env_ids if x not in pending_env_ids]
        record_info["env_info"] = env_info
        global_planner_info = self.global_planner_manager.update(record_info)
        assert len(global_planner_info) == self.num_envs, (
            "Global planner info length should be equal to num_envs"
        )
        record_info["global_planner_info"] = global_planner_info
        atomic_skill_info = self.atomic_skill_manager.update(record_info)
        assert len(atomic_skill_info) == self.num_envs, (
            "Atomic skill info length should be equal to num_envs"
        )
        record_info["atomic_skill_info"] = atomic_skill_info

        return record_info, valid_env_ids
