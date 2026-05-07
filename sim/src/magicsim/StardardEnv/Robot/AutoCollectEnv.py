from typing import Any, Dict, List, Sequence

from magicsim.Collect.Command.AutoCollectManager import AutoCollectManager
from magicsim.Collect.Record.RecordManager import RecordManager
from magicsim.StardardEnv.Robot.AsyncRobotEnv import AsyncRobotEnv
from omegaconf import DictConfig
from magicsim.Env.Utils.file import Logger


class AutoCollectEnv(AsyncRobotEnv):
    def __init__(
        self,
        task_string: Dict[str, float],
        config: DictConfig,
        cli_args: DictConfig,
        logger: Logger,
    ):
        super().__init__(config, cli_args, logger)
        self.task_config = config.task
        self.task_string = task_string
        self.auto_collect_config = config.auto_collect
        self.auto_collect_manager = AutoCollectManager(
            self.env,
            self.num_envs,
            self.task_string,
            self.task_config,
            self.auto_collect_config,
            self.device,
            self.logger,
        )
        self.record_config = config.record
        self.record_manager = RecordManager(
            self.env,
            self.num_envs,
            self.record_config,
            self.device,
            self.logger,
        )

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        info = super().reset(seed, options)
        info["auto_collect_info"] = self.auto_collect_manager.reset()
        self.record_manager.reset(info)  # we record the reset obs here
        return info

    def step(
        self,
        info: List[Dict[str, Any]],
        env_ids: Sequence[int] | None = None,
    ):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        robot_actions, valid_env_ids, failed_env_ids = self.auto_collect_manager.step(
            env_ids
        )

        # Always call super().step() so that:
        # 1) update() runs (after step returns), and can clear task when state is "failed"
        # 2) when failed_env_ids is non-empty, env.step(..., failed_env_ids) runs and triggers reset_idx()
        # If we early-return when valid_env_ids==[], we never call update() and never pass failed_env_ids to env.
        info, valid_env_ids = super().step(robot_actions, valid_env_ids, failed_env_ids)

        record_env_ids = []
        if valid_env_ids is not None:
            record_env_ids = valid_env_ids
        if failed_env_ids is not None:
            record_env_ids = record_env_ids + failed_env_ids

        auto_collect_info = self.auto_collect_manager.update(info)
        info["auto_collect_info"] = auto_collect_info

        if len(record_env_ids) > 0:
            # Record step obs before checking for task completion
            self.record_manager.step(
                info, record_env_ids
            )  # we record the step obs here

            # Update record manager first (this will save to disk for finished tasks)
            info["record_info"] = self.record_manager.update(info)

        return info

    def start_collect(self):
        info = self.reset()
        while True:
            info = self.step(info)
