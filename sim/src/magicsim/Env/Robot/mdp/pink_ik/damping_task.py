# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Damping task implementation for IsaacLab.

This module provides an IsaacLab-compatible version of the Pink DampingTask.
The damping task minimizes joint velocities to create smoother robot motion.
"""

from pink.configuration import Configuration
from pink.tasks.damping_task import DampingTask as PinkDampingTask


class DampingTask(PinkDampingTask):
    r"""Minimize joint velocities (IsaacLab-compatible version).

    The damping task minimizes :math:`\| v \|_2` with :math:`v` the joint
    velocity resulting from differential IK. The word "damping" is used here by
    analogy with forces that fight against motion, and bring the robot to a
    rest if nothing else drives it.

    This is an IsaacLab-compatible wrapper around Pink's DampingTask that adds
    the `set_target_from_configuration` method required by PinkIKController.
    """

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set task target from a robot configuration.

        For DampingTask, this is a no-op since the damping task always aims
        to minimize velocity (error is always zero). This method is provided
        for compatibility with PinkIKController initialization.

        Args:
            configuration: Robot configuration (unused, but required for interface compatibility).
        """
        # DampingTask doesn't need a target - it always minimizes velocity
        # This method is provided for compatibility with PinkIKController.init()
        pass

    def __repr__(self) -> str:
        """Human-readable representation of the task."""
        return f"DampingTask(cost={self.cost})"
