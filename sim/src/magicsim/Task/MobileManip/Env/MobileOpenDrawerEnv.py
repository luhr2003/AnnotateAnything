from typing import Sequence

import torch

from magicsim.Task.TableTop.Env.OpenDrawerEnv import OpenDrawerEnv


class MobileOpenDrawerEnv(OpenDrawerEnv):
    """
    OpenDrawer environment for Mobile Manipulation (ridgebackFranka).

    Inherits from :class:`OpenDrawerEnv` to reuse drawer trajectory / object-pose
    helpers and the 20%-joint-open termination logic. The only deviation is
    ``process_action``: the mobile base + arm + gripper layout is wider than
    the fixed-base Franka's 8D vector, so we pass the full action through
    untouched (same pattern as :class:`MobileGraspEnv`).
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
