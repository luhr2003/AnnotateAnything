"""Whole body controller configurations for quadruped robots."""

from .wbc_base_policy import WBCPolicy, BaseConfig
from .wbc_policy_factory import get_wbc_policy

__all__ = [
    "WBCPolicy",
    "BaseConfig",
    "get_wbc_policy",
]
