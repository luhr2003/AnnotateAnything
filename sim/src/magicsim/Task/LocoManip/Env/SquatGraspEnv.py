from typing import Any, Dict, Sequence

import torch
from magicsim.Task.TableTop.Env.GraspEnv import GraspEnv
import gymnasium as gym


class SquatGraspEnv(GraspEnv):
    """
    Environment for squat grasping: bottle placed on floor, robot squats to grasp.

    Inherits from GraspEnv. Same grasp API as LocoGraspEnv (get_grasp_pose with dex3_1).
    Termination: EEF close to object AND object lifted above floor (object_z > lift_threshold).
    Truncated: object fell through floor (object_z < floor_threshold).
    """

    def get_obs_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

    def get_policy_obs(self) -> Dict[str, Any]:
        robot_state = self.scene.robot_manager.get_robot_state()
        camera_info = self.scene.capture_manager.step()
        return {"robot_state": robot_state, "camera_info": camera_info}

    def get_privilege_obs(self) -> Dict[str, Any]:
        return {"object_pose": self.get_object_pose()}

    def get_grasp_pose(
        self,
        env_ids=None,
        obj_name=None,
        hand_type: str | None = "dex3_1",
        grasp_type=None,
        obj_id: int = 0,
        transform_to_world: bool = True,
    ) -> list:
        """Override: default hand_type is dex3_1 for G1/Dex3 hand."""
        return super().get_grasp_pose(
            env_ids=env_ids,
            obj_name=obj_name,
            hand_type=hand_type,
            grasp_type=grasp_type,
            obj_id=obj_id,
            transform_to_world=transform_to_world,
        )

    # ------------------------------------------------------------------
    # EEF pose
    # ------------------------------------------------------------------

    def get_grasp_success(
        self,
        env_ids: Sequence[int] | None = None,
        obj_name: str | None = None,
        height_above_floor: float = 0.18,
    ) -> torch.Tensor:
        """
        Per-env success: target object is lifted at least ``height_above_floor`` (m) above the floor.

        Uses world-frame object position z; floor is taken as z == 0 (same convention as
        :meth:`get_termination`).
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        object_poses_dict = self.get_object_pose(env_ids)
        name = obj_name
        if name is None:
            name = self.target_obj_name if hasattr(self, "target_obj_name") else None
        if name is None or name not in object_poses_dict:
            name = next(iter(object_poses_dict.keys())) if object_poses_dict else None
        if name is None:
            return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        object_pos = object_poses_dict[name][:, :3]
        object_z = object_pos[:, 2]
        return object_z > height_above_floor

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

    # ------------------------------------------------------------------
    # Reward / termination (floor-specific)
    # ------------------------------------------------------------------

    def process_action(self, action):
        return action

    def get_reward(self, action, env_ids: Sequence[int] | None = None):
        return [0] * self.num_envs

    def get_termination(self):
        """
        Squat grasp on floor:
        - termination: EEF close to object AND object lifted (z > 0.5)
        - truncated: object fell through floor (z < 0.0)
        """
        eef_pose = self.get_eef_pose()
        object_poses_dict = self.get_object_pose()

        obj_name = self.target_obj_name if hasattr(self, "target_obj_name") else None
        if obj_name is None or obj_name not in object_poses_dict:
            obj_name = (
                next(iter(object_poses_dict.keys())) if object_poses_dict else None
            )
            if obj_name is None:
                zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                return zeros, zeros

        object_pos = object_poses_dict[obj_name][:, :3]
        if eef_pose.dim() == 2:
            distance = torch.norm(eef_pose[:, :3] - object_pos, dim=1)
        else:
            right_dist = torch.norm(eef_pose[:, 0, :3] - object_pos, dim=1)
            left_dist = torch.norm(eef_pose[:, 1, :3] - object_pos, dim=1)
            distance = torch.minimum(right_dist, left_dist)
        object_z = object_pos[:, 2]

        # termination: EEF close AND object lifted (same height as :meth:`get_grasp_success`)
        lift_threshold = 0.25
        termination = (distance < 0.6) & (object_z > lift_threshold)

        # truncated: object fell through floor
        floor_threshold = 0.0
        truncated = object_z < floor_threshold

        return termination, truncated

    def get_info(self) -> Dict[str, Any]:
        return {"state": self.get_state()}

    def get_state(self) -> Dict[str, Any]:
        return {
            "robot_state": self.scene.robot_manager.get_robot_state(),
            "scene_state": {"object_pose": self.get_object_pose()},
            "camera_state": self.scene.camera_manager.get_all_camera_state(),
        }
