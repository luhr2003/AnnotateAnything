# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import MISSING, asdict, dataclass


@dataclass
class ArgsConfig:
    """Args Config for running the data collection loop."""

    def update(
        self, config_dict: dict, strict: bool = False, skip_keys: list[str] = []
    ):
        for k, v in config_dict.items():
            if k in skip_keys:
                continue
            if strict and not hasattr(self, k):
                raise ValueError(f"Config {k} not found in {self.__class__.__name__}")
            if not strict and not hasattr(self, k):
                continue
            setattr(self, k, v)

    @classmethod
    def from_dict(
        cls, config_dict: dict, strict: bool = False, skip_keys: list[str] = []
    ):
        instance = cls()
        instance.update(config_dict=config_dict, strict=strict, skip_keys=skip_keys)
        return instance

    def to_dict(self):
        return asdict(self)


@dataclass
class BaseConfig(ArgsConfig):
    """Base config inherited by all G1 control loops"""

    # WBC Configuration
    wbc_version: str = MISSING
    """Version of the whole body controller."""

    wbc_model_path: str = MISSING
    """Path to WBC model file"""

    policy_config_path: str = MISSING
    """Policy related configuration to specify inputs/outputs dim"""

    wbc_joints_order: dict[str, int] = MISSING
    """Order of the WBC joints"""

    # Robot Configuration
    enable_waist: bool = False
    """Whether to include waist joints in IK."""


class WBCPolicy(ABC):
    """Base class for implementing control policies in the Gear'WBC framework.

    A Policy defines how an agent should behave in an environment by mapping observations
    to actions. This abstract base class provides the interface that all concrete policy
    implementations must follow.
    """

    wbc_config: BaseConfig

    def set_goal(self, goal: dict[str, any]):
        """Set the command from the planner that the policy should follow.

        Args:
            goal: Dictionary containing high-level commands or goals from the planner
        """
        pass

    def set_observation(self, observation: dict[str, any]):
        """Update the policy's current observation of the environment.

        Args:
            observation: Dictionary containing the current state/observation of the environment
        """
        self.observation = observation

    @abstractmethod
    def get_action(self, time: float | None = None) -> dict[str, any]:
        """Compute and return the next action at the specified time, based on current observation
        and planner command.

        Args:
            time: Optional "monotonic time" for time-dependent policies

        Returns:
            Dictionary containing the action to be executed
        """

    def close(self):
        """Clean up any resources used by the policy."""
        pass
