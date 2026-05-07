from typing import Sequence

import torch

from magicsim.Task.TableTop.Env.CloseDrawerEnv import CloseDrawerEnv


class MobileCloseDrawerEnv(CloseDrawerEnv):
    """
    CloseDrawer environment for Mobile Manipulation (ridgebackFranka).

    Inherits from :class:`CloseDrawerEnv` to reuse drawer trajectory /
    object-pose helpers, ``set_drawer_open`` (used by the atomic skill at
    reset), and the ``progress.min() <= 5%`` close-drawer termination.
    Only ``process_action`` is overridden: the mobile base + arm + gripper
    layout is wider than the fixed-base Franka's 8D vector, so the full
    action passes through untouched (same as :class:`MobileGraspEnv` /
    :class:`MobileOpenDrawerEnv`).
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)

    def process_action(self, action):
        return action

    def get_eef_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        robot_state = list(self.scene.robot_manager.get_robot_state()[0].values())[0]
        eef_pos = robot_state["eef_pos"]
        eef_quat = robot_state["eef_quat"]
        if eef_pos.dim() == 2:
            eef_pose = torch.cat([eef_pos, eef_quat], dim=1)
        else:
            eef_pose = torch.cat([eef_pos, eef_quat], dim=-1)
        return eef_pose[env_ids]
