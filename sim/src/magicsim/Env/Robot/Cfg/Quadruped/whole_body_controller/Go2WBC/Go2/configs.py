# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
import os
from typing import Literal

from magicsim import MAGICSIM_ASSETS
from magicsim.Env.Robot.Cfg.Quadruped.whole_body_controller.wbc_base_policy import (
    BaseConfig,
)


@dataclass
class Go2WBCV1Config(BaseConfig):
    """Base config inherited by all Go2 control loops"""

    # WBC Configuration
    wbc_version: Literal["go2_v1"] = "go2_v1"
    """Version of the whole body controller."""

    wbc_model_path: str = f"{MAGICSIM_ASSETS}/WBC/models/go2_v1/policy.onnx"
    """Path to WBC model file"""

    policy_config_path: str = os.path.join(os.path.dirname(__file__), "go2_wbc_v1.yaml")
    """Policy related configuration to specify inputs/outputs dim"""

    wbc_joints_order: dict[str, int] = field(
        default_factory=lambda: {
            # Actual joint order in IsaacLab: all hips, then all thighs, then all calves
            # This matches the order returned by find_joints:
            # ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
            #  'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
            #  'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
            "FL_hip_joint": 0,
            "FR_hip_joint": 1,
            "RL_hip_joint": 2,
            "RR_hip_joint": 3,
            "FL_thigh_joint": 4,
            "FR_thigh_joint": 5,
            "RL_thigh_joint": 6,
            "RR_thigh_joint": 7,
            "FL_calf_joint": 8,
            "FR_calf_joint": 9,
            "RL_calf_joint": 10,
            "RR_calf_joint": 11,
        },
    )

    body_ids: list[int] = field(
        default_factory=lambda: [
            # These are the indices in the actual joint order (all hips, then all thighs, then all calves)
            0,  # FL_hip_joint
            1,  # FR_hip_joint
            2,  # RL_hip_joint
            3,  # RR_hip_joint
            4,  # FL_thigh_joint
            5,  # FR_thigh_joint
            6,  # RL_thigh_joint
            7,  # RR_thigh_joint
            8,  # FL_calf_joint
            9,  # FR_calf_joint
            10,  # RL_calf_joint
            11,  # RR_calf_joint
        ]
    )

    # Robot Configuration
    enable_waist: bool = False
    """Whether to include waist joints (not applicable for quadruped)."""
