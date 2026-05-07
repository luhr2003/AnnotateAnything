from typing import Any, Dict, Sequence

import torch
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
import gymnasium as gym


class DualReachEnv(TaskBaseEnv):
    """
    Dual-arm Reach Environment for Robot Tasks.
    Left arm reaches cube_left (red), right arm reaches cube_right (blue).
    Uses scene with cube_left and cube_right (reach_dual_env.yaml).
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)
        self.target_category_left: str = "cube_left"
        self.target_category_right: str = "cube_right"

    def get_obs_space(self) -> gym.spaces.Dict:
        """
        Get the observation space for the environment.
        """
        return gym.spaces.Dict({})

    def get_policy_obs(self) -> Dict[str, Any]:
        """
        Get the policy observation for the environment.
        """
        robot_state = self.scene.robot_manager.get_robot_state()
        camera_info = self.scene.capture_manager.step()
        return {"robot_state": robot_state, "camera_info": camera_info}

    def get_privilege_obs(self) -> Dict[str, Any]:
        """
        Get the privilege observation for the environment.
        """
        cube_left_pose = self.get_cube_pose_left()
        cube_right_pose = self.get_cube_pose_right()
        return {
            "cube_right_pose": cube_right_pose,
            "cube_left_pose": cube_left_pose,
        }

    def _get_cube_pose_by_category(
        self, target_category: str, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Get pose of objects in target_category for given env_ids."""
        if env_ids is None:
            env_ids = torch.arange(
                self.scene.robot_manager.num_envs, device=self.device
            )
        cube_pose = []
        scene_mgr = self.scene.scene_manager
        for env_id in env_ids:
            e_id = int(env_id)
            rigid_env = scene_mgr.rigid_objects[e_id]
            geo_env = scene_mgr.geometry_objects[e_id]

            obj_list = rigid_env.get(target_category, [])
            if not obj_list:
                obj_list = geo_env.get(target_category, [])

            if not obj_list:
                raise RuntimeError(
                    f"DualReachEnv: no objects found for target_category='{target_category}' "
                    f"in env_id={e_id}. 请检查 Scene 配置是否在该类别下创建了实例。"
                )

            translation, orientation = obj_list[0].get_local_pose()
            cube_pose.append(torch.cat([translation, orientation], dim=0))
        return torch.stack(cube_pose, dim=0)

    def get_cube_pose_left(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get pose of cube_left (red, left side)."""
        return self._get_cube_pose_by_category(self.target_category_left, env_ids)

    def get_cube_pose_right(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get pose of cube_right (blue, right side)."""
        return self._get_cube_pose_by_category(self.target_category_right, env_ids)

    def process_action(self, action: torch.Tensor | list[Dict]):
        """
        Process the action for dual-arm environment.
        ik_pink: arm 14D (left 7D + right 7D), eef 2D -> total 16D.
        If action is 14D, pad with eef open (ones).
        """
        if action is None:
            return None
        if action.shape[1] == 14:
            action = torch.cat(
                [
                    action,
                    torch.ones((action.shape[0], 2), device=self.device),
                ],
                dim=1,
            )
        return action

    def get_eef_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get the pose of both end effectors for dual-arm.
        Returns [num_envs, 2, 7] (right eef at index 0, left eef at index 1).
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        robot_state = self.scene.robot_manager.get_robot_state()[0]
        state_dict = list(robot_state.values())[0]
        eef_pos = state_dict["eef_pos"]  # [num_envs, num_eef, 3]
        eef_quat = state_dict["eef_quat"]  # [num_envs, num_eef, 4]
        eef_pose = torch.cat([eef_pos, eef_quat], dim=-1)  # [num_envs, num_eef, 7]
        return eef_pose[env_ids]

    def get_info(self) -> Dict[str, Any]:
        """
        Get the info dictionary for the environment.
        """
        state = self.get_state()
        description = self.get_description()
        return {"state": state, "description": description}

    def get_description(self) -> str:
        """
        Get the description of the environment.
        """
        return "Dual-arm reach: left arm to red cube (cube_left), right arm to blue cube (cube_right)"

    def get_state(self) -> Dict[str, Any]:
        """
        Get the state of the environment.
        """
        robot_state = self.scene.robot_manager.get_robot_state()
        cube_left = self.get_cube_pose_left()
        cube_right = self.get_cube_pose_right()
        state = {
            "robot_state": robot_state,
            "scene_state": {
                "cube_left_pose": cube_left,
                "cube_right_pose": cube_right,
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
        """
        Termination when both arms reach their respective cubes.
        eef_pose: [num_envs, 2, 7], index 0=right, 1=left (frame order: right first, left second).
        """
        eef_pose = self.get_eef_pose()  # [num_envs, 2, 7]
        cube_left = self.get_cube_pose_left()  # [num_envs, 7]
        cube_right = self.get_cube_pose_right()  # [num_envs, 7]

        # Right arm (index 0) -> cube_right
        right_eef_pos = eef_pose[:, 0, :3]
        right_cube_pos = cube_right[:, :3]
        right_dist = torch.norm(right_eef_pos - right_cube_pos, dim=1)
        right_reached = right_dist < 0.02

        # Left arm (index 1) -> cube_left
        left_eef_pos = eef_pose[:, 1, :3]
        left_cube_pos = cube_left[:, :3]
        left_dist = torch.norm(left_eef_pos - left_cube_pos, dim=1)
        left_reached = left_dist < 0.03
        termination = left_reached & right_reached
        truncated = torch.tensor(
            [False] * self.num_envs, dtype=torch.bool, device=self.device
        )
        return termination, truncated
