from typing import Any, Dict, Sequence

import torch
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
import gymnasium as gym


class MobileDualReachEnv(TaskBaseEnv):
    """Dual-arm mobile reach environment.

    Scene spawns two cubes (``cube_1`` = red, ``cube_2`` = blue). The test
    script reads both cube poses and drives the Vega 1P + Sharpa pink IK
    with them as left / right wrist targets while the holonomic base stays
    stationary and the Sharpa fingers hold at zero.
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)

    def get_obs_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

    def get_policy_obs(self) -> Dict[str, Any]:
        robot_state = self.scene.robot_manager.get_robot_state()
        camera_info = self.scene.capture_manager.step()
        return {"robot_state": robot_state, "camera_info": camera_info}

    def get_privilege_obs(self) -> Dict[str, Any]:
        cube_pose = self.get_cube_pose()
        return {"cube_pose": cube_pose}

    def get_cube_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Return dual cube pose tensor ``[N, 2, 7]`` (index 0 = red, 1 = blue)."""
        if env_ids is None:
            env_ids = torch.arange(
                self.scene.robot_manager.num_envs, device=self.device
            )
        out = []
        for env_id in env_ids:
            cubes = self.scene.scene_manager.geometry_objects[env_id]["cube"]
            per_env = []
            for cube in cubes[:2]:
                translation, orientation = cube.get_local_pose()
                per_env.append(torch.cat([translation, orientation], dim=0))
            out.append(torch.stack(per_env, dim=0))
        return torch.stack(out, dim=0)

    def process_action(self, action: torch.Tensor | list[Dict]):
        return action

    def get_eef_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        eef_pos = list(self.scene.robot_manager.get_robot_state()[0].values())[0][
            "eef_pos"
        ]
        eef_quat = list(self.scene.robot_manager.get_robot_state()[0].values())[0][
            "eef_quat"
        ]
        eef_pose = torch.cat([eef_pos, eef_quat], dim=-1)
        return eef_pose[env_ids]

    def get_info(self) -> Dict[str, Any]:
        state = self.get_state()
        description = self.get_description()
        return {"state": state, "description": description}

    def get_description(self) -> str:
        return (
            "Dual-arm reach: right hand tracks the blue cube, "
            "left hand tracks the red cube."
        )

    def get_state(self) -> Dict[str, Any]:
        robot_state = self.scene.robot_manager.get_robot_state()
        cube_state = self.get_cube_pose()
        return {
            "robot_state": robot_state,
            "scene_state": {"cube_pose": cube_state},
            "camera_state": self.scene.camera_manager.get_all_camera_state(),
        }

    def get_reward(
        self,
        action: torch.Tensor | list[Dict],
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        return [0] * self.num_envs

    def get_termination(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor([False] * self.num_envs, dtype=torch.bool),
            torch.tensor([False] * self.num_envs, dtype=torch.bool),
        )
