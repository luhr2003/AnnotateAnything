from typing import Any, Dict, Sequence

import torch
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
import gymnasium as gym


class LocoReachEnv(TaskBaseEnv):
    """
    Loco reach environment for robot tasks.
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)

    def get_obs_space(self) -> gym.spaces.Dict:
        """
        Get the observation space for the environment.
        This method should be overridden by subclasses to define specific observation spaces.
        """
        return gym.spaces.Dict({})

    def get_policy_obs(
        self,
    ) -> Dict[str, Any]:
        """
        Get the policy observation for the environment.
        This method should be overridden by subclasses to define specific policy observations.
        """
        robot_state = self.scene.robot_manager.get_robot_state()
        camera_info = self.scene.capture_manager.step()
        return {"robot_state": robot_state, "camera_info": camera_info}

    def get_privilege_obs(
        self,
    ) -> Dict[str, Any]:
        """
        Get the privilege observation for the environment.
        This method should be overridden by subclasses to define specific privilege observations.
        """
        cube_pose = self.get_cube_pose()
        return {"cube_pose": cube_pose}

    def get_cube_pose(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        """
        Get the pose of the cube for the environment.
        This method should be overridden by subclasses to define specific cube pose retrieval.
        """
        if env_ids is None:
            env_ids = torch.arange(
                self.scene.robot_manager.num_envs, device=self.device
            )
        cube_pose = []
        for env_id in env_ids:
            translation, orientation = self.scene.scene_manager.geometry_objects[
                env_id
            ]["cube"][0].get_local_pose()
            cube_pose.append(torch.cat([translation, orientation], dim=0))
        return torch.stack(cube_pose, dim=0)

    def process_action(self, action: torch.Tensor | list[Dict]):
        """
        Process the action for the environment.
        This method should be overridden by subclasses to define specific action processing.
        """
        # padding 1 at end of each action to mimic gripper close
        if action is None:
            return None
        if action.shape[1] == 7:
            action = torch.cat(
                [action, torch.ones((action.shape[0], 1), device=self.device)], dim=1
            )
        return action

    def get_eef_pose(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        """
        Get the pose of the end effector for the environment.
        This method should be overridden by subclasses to define specific end effector pose retrieval.
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        eef_pos = list(self.scene.robot_manager.get_robot_state()[0].values())[0][
            "eef_pos"
        ]
        eef_quat = list(self.scene.robot_manager.get_robot_state()[0].values())[0][
            "eef_quat"
        ]
        # eef_pos: [num_envs, num_eef, 3], eef_quat: [num_envs, num_eef, 4]
        # -> [num_envs, num_eef, 7]
        eef_pose = torch.cat([eef_pos, eef_quat], dim=-1)
        return eef_pose[env_ids]

    def get_info(
        self,
    ) -> Dict[str, Any]:
        """
        Get the info dictionary for the environment.
        This method should be overridden by subclasses to define specific info retrieval.
        """
        state = self.get_state()
        description = self.get_description()
        return {"state": state, "description": description}

    def get_description(self) -> Dict[str, Any]:
        """
        Get the description of the environment.
        This method should be overridden by subclasses to define specific description retrieval.
        """
        return "Reaching the red cube with the end effector"

    def get_state(self) -> Dict[str, Any]:
        """
        Get the state of the environment.
        This method should be overridden by subclasses to define specific state retrieval.
        """
        robot_state = self.scene.robot_manager.get_robot_state()
        cube_state = self.get_cube_pose()
        state = {
            "robot_state": robot_state,
            "scene_state": {
                "cube_pose": cube_state,
            },
            "camera_state": self.scene.camera_manager.get_all_camera_state(),
        }
        return state

    def get_reward(
        self,
        action: torch.Tensor | list[Dict],
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        reward = [None] * self.num_envs
        for env_id in range(self.num_envs):
            reward[env_id] = 0
        return reward

    def get_termination(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eef_pose = self.get_eef_pose()  # [num_envs, num_eef, 7]
        # Use first EEF (left hand) for termination check
        eef_pos = eef_pose[:, 0, :3]
        eef_quat = eef_pose[:, 0, 3:7]
        cube_pose = self.get_cube_pose()
        cube_pos = cube_pose[:, :3]
        cube_quat = cube_pose[:, 3:7]
        distance = torch.norm(eef_pos - cube_pos, dim=1)
        eef_norm = torch.linalg.norm(eef_quat, dim=1)
        cube_norm = torch.linalg.norm(cube_quat, dim=1)
        denom = (eef_norm * cube_norm).clamp_min(1e-9)
        cos = torch.abs(torch.sum(eef_quat * cube_quat, dim=1) / denom)
        quat_error = 1.0 - cos
        termination = distance < 0.05
        return torch.tensor([False] * self.num_envs, dtype=torch.bool), torch.tensor(
            [False] * self.num_envs, dtype=torch.bool
        )
