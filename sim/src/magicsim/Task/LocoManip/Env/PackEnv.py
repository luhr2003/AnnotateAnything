from typing import Any, Dict, Sequence

import torch
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
import gymnasium as gym


class PackEnv(TaskBaseEnv):
    """
    Reach Environment for Robot Tasks.
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
        mug_pose = self.get_mug_pose()
        return {
            "mug_pose": mug_pose,
        }

    def get_mug_pose(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        """
        Get the pose of the mug for the environment.
        This method should be overridden by subclasses to define specific mug pose retrieval.
        """
        if env_ids is None:
            env_ids = torch.arange(
                self.scene.robot_manager.num_envs, device=self.device
            )
        mug_pose = []
        for env_id in env_ids:
            translation, orientation = self.scene.scene_manager.rigid_objects[env_id][
                "mug"
            ][0].get_local_pose()
            mug_pose.append(torch.cat([translation, orientation], dim=0))
        return torch.stack(mug_pose, dim=0)

    def get_mug_grasp_pose(
        self, env_ids: Sequence[int] | None = None
    ) -> Dict[str, Any]:
        """
        Get the grasp pose of the mug for the environment.
        This method should be overridden by subclasses to define specific mug grasp pose retrieval.

        Note: Applies a coordinate frame transformation from grasp pose frame to gripper frame.
        The transformation rotates the grasp pose orientation to match gripper coordinate system.
        """
        if env_ids is None:
            env_ids = torch.arange(
                self.scene.robot_manager.num_envs, device=self.device
            )
        mug_grasp_pose_list = []
        for env_id in env_ids:
            mug_grasp_pose = self.scene.scene_manager.rigid_objects[env_id]["mug"][
                0
            ].get_grasp_poses(transform_to_world=True, device=self.device)
            mug_grasp_pose_list.append(mug_grasp_pose)
        return mug_grasp_pose_list

    def process_action(self, action: torch.Tensor | list[Dict]):
        """
        Process the action for the environment.
        This method should be overridden by subclasses to define specific action processing.

        Note:
            Action format: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, ...]
            - action[0:3]: Position (x, y, z)
            - action[3:7]: Quaternion rotation (w, x, y, z) that represents a rotation
                          relative to the base orientation [0, 1, 0, 0] (gripper pointing down).
                          The quaternion action[3:7] is applied on top of [0, 1, 0, 0] to get
                          the final gripper orientation.
        """
        # padding 1 at end of each action to mimic gripper close
        if action is None:
            return None
        if action.shape[1] < 8:
            action = torch.cat(
                [action, torch.zeros((action.shape[0], 1), device=self.device)], dim=1
            )
        else:
            action = action[:, :8]
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
        return {
            "state": state,
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Get the state of the environment.
        This method should be overridden by subclasses to define specific state retrieval.
        """
        robot_state = self.scene.robot_manager.get_robot_state()
        mug_state = self.get_mug_pose()
        state = {
            "robot_state": robot_state,
            "scene_state": {
                "mug_pose": mug_state,
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
        eef_pos = self.get_eef_pose()[:, 0, :3]  # first EEF
        mug_pos = self.get_mug_pose()[:, :3]
        distance = torch.norm(eef_pos - mug_pos, dim=1)
        mug_z = mug_pos[:, 2]

        # termination: eef与mug距离小于0.3 并且 mug z轴大于0.2
        termination = (distance < 0.3) & (mug_z > 1.2)
        # print(f"distance: {distance}, mug_z: {mug_z}, termination: {termination}")

        # truncated: mug掉下桌子，z轴小于0.8
        truncated = mug_z < 0.8

        return termination, truncated
