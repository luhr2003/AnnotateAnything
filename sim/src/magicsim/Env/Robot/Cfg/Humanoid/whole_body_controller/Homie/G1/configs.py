# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
import os
from typing import Literal

from magicsim import MAGICSIM_ASSETS
from magicsim.Env.Robot.Cfg.Humanoid.whole_body_controller.wbc_base_policy import (
    BaseConfig,
)


@dataclass
class G1HomieV2Config(BaseConfig):
    """Base config inherited by all G1 control loops"""

    # WBC Configuration
    wbc_version: Literal["homie_v2"] = "homie_v2"
    """Version of the whole body controller."""

    wbc_model_path: str = f"{MAGICSIM_ASSETS}/WBC/models/homie_v2/stand.onnx,{MAGICSIM_ASSETS}/WBC/models/homie_v2/walk.onnx"
    """Path to WBC model file"""

    policy_config_path: str = os.path.join(
        os.path.dirname(__file__), "g1_homie_v2.yaml"
    )
    """Policy related configuration to specify inputs/outputs dim"""

    wbc_joints_order: dict[str, int] = field(
        default_factory=lambda: {
            "left_hip_pitch_joint": 0,
            "left_hip_roll_joint": 1,
            "left_hip_yaw_joint": 2,
            "left_knee_joint": 3,
            "left_ankle_pitch_joint": 4,
            "left_ankle_roll_joint": 5,
            "right_hip_pitch_joint": 6,
            "right_hip_roll_joint": 7,
            "right_hip_yaw_joint": 8,
            "right_knee_joint": 9,
            "right_ankle_pitch_joint": 10,
            "right_ankle_roll_joint": 11,
            "waist_yaw_joint": 12,
            "waist_roll_joint": 13,
            "waist_pitch_joint": 14,
            "left_shoulder_pitch_joint": 15,
            "left_shoulder_roll_joint": 16,
            "left_shoulder_yaw_joint": 17,
            "left_elbow_joint": 18,
            "left_wrist_roll_joint": 19,
            "left_wrist_pitch_joint": 20,
            "left_wrist_yaw_joint": 21,
            "left_hand_index_0_joint": 22,
            "left_hand_index_1_joint": 23,
            "left_hand_middle_0_joint": 24,
            "left_hand_middle_1_joint": 25,
            "left_hand_thumb_0_joint": 26,
            "left_hand_thumb_1_joint": 27,
            "left_hand_thumb_2_joint": 28,
            "right_shoulder_pitch_joint": 29,
            "right_shoulder_roll_joint": 30,
            "right_shoulder_yaw_joint": 31,
            "right_elbow_joint": 32,
            "right_wrist_roll_joint": 33,
            "right_wrist_pitch_joint": 34,
            "right_wrist_yaw_joint": 35,
            "right_hand_index_0_joint": 36,
            "right_hand_index_1_joint": 37,
            "right_hand_middle_0_joint": 38,
            "right_hand_middle_1_joint": 39,
            "right_hand_thumb_0_joint": 40,
            "right_hand_thumb_1_joint": 41,
            "right_hand_thumb_2_joint": 42,
        },
    )

    body_ids: list[int] = field(
        default_factory=lambda: [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
        ]
    )

    # Robot Configuration
    enable_waist: bool = False
    """Whether to include waist joints in IK."""


@dataclass
class G1Dex1HomieV2Config(G1HomieV2Config):
    """Homie WBC config for G1 with dex1_1 grippers (33-joint articulation:
    29 G1 body joints + 4 dex1 prismatic).

    The pretrained Homie ONNX policy outputs only the 15 leg+waist joints,
    so we keep those indices identical to ``G1HomieV2Config`` and just shrink
    the rest of the joint table to what the dex1 articulation actually has.
    The 4 dex1 finger joints take indices 22-23 (left) and 31-32 (right) —
    where the original dex3 ``hand_index_*_joint`` slots used to be — so the
    arm-joint indices stay at 15-21 / 29-35 and ``body_ids`` (which the
    policy reads to build its observation slice) is unchanged.
    """

    wbc_joints_order: dict[str, int] = field(
        default_factory=lambda: {
            "left_hip_pitch_joint": 0,
            "left_hip_roll_joint": 1,
            "left_hip_yaw_joint": 2,
            "left_knee_joint": 3,
            "left_ankle_pitch_joint": 4,
            "left_ankle_roll_joint": 5,
            "right_hip_pitch_joint": 6,
            "right_hip_roll_joint": 7,
            "right_hip_yaw_joint": 8,
            "right_knee_joint": 9,
            "right_ankle_pitch_joint": 10,
            "right_ankle_roll_joint": 11,
            "waist_yaw_joint": 12,
            "waist_roll_joint": 13,
            "waist_pitch_joint": 14,
            "left_shoulder_pitch_joint": 15,
            "left_shoulder_roll_joint": 16,
            "left_shoulder_yaw_joint": 17,
            "left_elbow_joint": 18,
            "left_wrist_roll_joint": 19,
            "left_wrist_pitch_joint": 20,
            "left_wrist_yaw_joint": 21,
            "left_dex1_finger_joint_1": 22,
            "left_dex1_finger_joint_2": 23,
            "right_shoulder_pitch_joint": 24,
            "right_shoulder_roll_joint": 25,
            "right_shoulder_yaw_joint": 26,
            "right_elbow_joint": 27,
            "right_wrist_roll_joint": 28,
            "right_wrist_pitch_joint": 29,
            "right_wrist_yaw_joint": 30,
            "right_dex1_finger_joint_1": 31,
            "right_dex1_finger_joint_2": 32,
        },
    )

    body_ids: list[int] = field(
        default_factory=lambda: [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
        ]
    )


@dataclass
class G1FixedHandHomieV2Config(G1HomieV2Config):
    """Homie WBC config for G1 with rigid rubber_hand (no finger DoFs).

    The pretrained Homie ONNX policy still outputs the same 15 leg+waist
    joints, so we keep those indices identical to ``G1HomieV2Config`` and
    just drop the 14 dex3 finger entries entirely. The arm joints stay at
    indices 15-21 / 22-28 (renumbered to fill the gap left by the missing
    fingers).
    """

    wbc_joints_order: dict[str, int] = field(
        default_factory=lambda: {
            "left_hip_pitch_joint": 0,
            "left_hip_roll_joint": 1,
            "left_hip_yaw_joint": 2,
            "left_knee_joint": 3,
            "left_ankle_pitch_joint": 4,
            "left_ankle_roll_joint": 5,
            "right_hip_pitch_joint": 6,
            "right_hip_roll_joint": 7,
            "right_hip_yaw_joint": 8,
            "right_knee_joint": 9,
            "right_ankle_pitch_joint": 10,
            "right_ankle_roll_joint": 11,
            "waist_yaw_joint": 12,
            "waist_roll_joint": 13,
            "waist_pitch_joint": 14,
            "left_shoulder_pitch_joint": 15,
            "left_shoulder_roll_joint": 16,
            "left_shoulder_yaw_joint": 17,
            "left_elbow_joint": 18,
            "left_wrist_roll_joint": 19,
            "left_wrist_pitch_joint": 20,
            "left_wrist_yaw_joint": 21,
            "right_shoulder_pitch_joint": 22,
            "right_shoulder_roll_joint": 23,
            "right_shoulder_yaw_joint": 24,
            "right_elbow_joint": 25,
            "right_wrist_roll_joint": 26,
            "right_wrist_pitch_joint": 27,
            "right_wrist_yaw_joint": 28,
        },
    )

    body_ids: list[int] = field(
        default_factory=lambda: [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
        ]
    )
