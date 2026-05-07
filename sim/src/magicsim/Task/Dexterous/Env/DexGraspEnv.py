from typing import Any, Dict, Sequence

import torch
from magicsim.Task.TableTop.Env.GraspEnv import GraspEnv
import gymnasium as gym


class DexGraspEnv(GraspEnv):
    """
    Environment for dexterous grasping with XHand.

    Inherits from GraspEnv. Use get_grasp_pose(hand_type="xhand") to retrieve
    xhand_grasp_pose annotations (functional_grasp/grasp with coarse/fine/final phases).
    """

    def get_obs_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

    def get_policy_obs(self) -> Dict[str, Any]:
        robot_state = self.scene.robot_manager.get_robot_state()
        camera_info = self.scene.capture_manager.step()
        return {"robot_state": robot_state, "camera_info": camera_info}

    def get_privilege_obs(self) -> Dict[str, Any]:
        return {"object_pose": self.get_object_pose()}

    # ------------------------------------------------------------------
    # EEF pose
    # ------------------------------------------------------------------

    def get_eef_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        eef_pos = list(self.scene.robot_manager.get_robot_state()[0].values())[0][
            "eef_pos"
        ]
        eef_quat = list(self.scene.robot_manager.get_robot_state()[0].values())[0][
            "eef_quat"
        ]
        eef_pose = torch.cat([eef_pos, eef_quat], dim=1)
        return eef_pose[env_ids]

    # ------------------------------------------------------------------
    # Reward / termination
    # ------------------------------------------------------------------

    def process_action(self, action):
        return action

    def get_reward(self, action, env_ids: Sequence[int] | None = None):
        return [0] * self.num_envs

    def get_termination(self):
        eef_pos = self.get_eef_pose()[:, :3]
        object_poses_dict = self.get_object_pose()

        obj_name = next(iter(object_poses_dict.keys())) if object_poses_dict else None
        if obj_name is None:
            zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return zeros, zeros

        object_pos = object_poses_dict[obj_name][:, :3]
        distance = torch.norm(eef_pos - object_pos, dim=1)
        object_z = object_pos[:, 2]

        termination = (distance < 0.3) & (object_z > 1.2)
        truncated = object_z < 0.8
        return termination, truncated

    def get_info(self) -> Dict[str, Any]:
        return {"state": self.get_state()}

    def get_state(self) -> Dict[str, Any]:
        return {
            "robot_state": self.scene.robot_manager.get_robot_state(),
            "scene_state": {"object_pose": self.get_object_pose()},
            "camera_state": self.scene.camera_manager.get_all_camera_state(),
        }
