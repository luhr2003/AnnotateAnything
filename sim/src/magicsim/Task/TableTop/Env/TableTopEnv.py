from typing import Any, Dict, Sequence

import torch
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
import gymnasium as gym


class TableTopEnv(TaskBaseEnv):
    """
    Table Top Environment for Robot Tasks.
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)

    def get_obs_space(self) -> gym.spaces.Dict:
        """
        Get the observation space for the environment.
        This method should be overridden by subclasses to define specific observation spaces.
        """
        return gym.spaces.Dict({})

    def get_policy_obs(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        """
        Get the policy observation for the environment.
        This method should be overridden by subclasses to define specific policy observations.
        """
        pass

    def get_privilege_obs(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        """
        Get the privilege observation for the environment.
        This method should be overridden by subclasses to define specific privilege observations.
        """
        pass

    def process_action(self, action: torch.Tensor | list[Dict]):
        """
        Process the action for the environment.
        This method should be overridden by subclasses to define specific action processing.
        """
        return action

    def get_info(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        """
        Get the info dictionary for the environment.
        This method should be overridden by subclasses to define specific info retrieval.
        """
        pass

    def get_reward(
        self,
        obs: Dict[str, Any],
        action: torch.Tensor | list[Dict],
        info: Dict[str, Any],
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        pass

    def get_termination(
        self,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        env_ids: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
