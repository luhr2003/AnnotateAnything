from typing import Any, Dict, Sequence

import torch
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
import gymnasium as gym


class ReachEnv(TaskBaseEnv):
    """
    Reach Environment for Robot Tasks.
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)
        # 目标物体类别名称（对应 scene_manager 中的类别 key）
        # 默认为 "cube"，也可以在对应的 env yaml（如 reach_env.yaml）里通过 target_category 覆盖。
        self.target_category: str = getattr(config, "target_category", "cube")

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
        scene_mgr = self.scene.scene_manager
        for env_id in env_ids:
            e_id = int(env_id)
            rigid_env = scene_mgr.rigid_objects[e_id]
            geo_env = scene_mgr.geometry_objects[e_id]

            obj_list = rigid_env.get(self.target_category, [])
            if not obj_list:
                obj_list = geo_env.get(self.target_category, [])

            if not obj_list:
                print(
                    f"[ReachEnv Debug] env_id={e_id} target_category={self.target_category} "
                    f"has no instances; rigid_keys={list(rigid_env.keys())}, "
                    f"geo_keys={list(geo_env.keys())}"
                )
                raise RuntimeError(
                    f"ReachEnv: no objects found for target_category='{self.target_category}' "
                    f"in env_id={e_id}. 请检查 Scene 配置是否在该类别下创建了实例。"
                )

            translation, orientation = obj_list[0].get_local_pose()
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

    def get_eef_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get the pose of the end effector for the environment.
        Single-arm: returns [num_envs, 7] (pos+quat).
        Dual-arm: returns [num_envs, 2, 7] (right at 0, left at 1).
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        robot_state = list(self.scene.robot_manager.get_robot_state()[0].values())[0]
        eef_pos = robot_state["eef_pos"]
        eef_quat = robot_state["eef_quat"]
        # Single-arm: [N, 3] + [N, 4] -> cat dim=1 -> [N, 7]
        # Dual-arm: [N, 2, 3] + [N, 2, 4] -> cat dim=-1 -> [N, 2, 7]
        if eef_pos.dim() == 2:
            eef_pose = torch.cat([eef_pos, eef_quat], dim=1)
        else:
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
        """
        Termination when any eef reaches the cube (distance < 0.008).
        Single-arm: one eef; dual-arm: either eef reaches.
        """
        eef_pose = self.get_eef_pose()
        cube_pose = self.get_cube_pose()
        cube_pos = cube_pose[:, :3]

        if eef_pose.dim() == 2:
            # Single-arm: eef_pose [N, 7]
            eef_pos = eef_pose[:, :3]
            distance = torch.norm(eef_pos - cube_pos, dim=1)
        else:
            # Dual-arm: eef_pose [N, 2, 7], 任意 eef 到了就结束
            right_dist = torch.norm(eef_pose[:, 0, :3] - cube_pos, dim=1)
            left_dist = torch.norm(eef_pose[:, 1, :3] - cube_pos, dim=1)
            distance = torch.minimum(right_dist, left_dist)

        termination = distance < 0.005
        truncated = torch.tensor(
            [False] * self.num_envs, dtype=torch.bool, device=self.device
        )
        return termination, truncated
