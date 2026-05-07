from typing import Any, Dict, Sequence

import torch
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from magicsim.Env.Scene.Object.Rigid import RigidObject
import gymnasium as gym


class GraspEnv(TaskBaseEnv):
    """
    Reach Environment for Robot Tasks.
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)
        # Get object name from task config, default to "mug" for backwards compatibility
        task_config = getattr(config, "task", None)
        print(f"task_config: {task_config}")
        grasp_config = (
            getattr(task_config, "Grasp", None)
            or getattr(task_config, "DexGrasp", None)
            if task_config
            else None
        )

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
        object_pose = self.get_object_pose()
        return {
            "object_pose": object_pose,
        }

    def get_object_pose(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        """
        Get the pose of all rigid objects for the environment.
        Returns a dictionary mapping object names to their poses.
        """
        if env_ids is None:
            env_ids = torch.arange(
                self.scene.robot_manager.num_envs, device=self.device
            )

        # Dictionary to store poses: {object_name: tensor of shape (num_envs, 7)}
        object_poses_dict = {}

        # Get all object names from the first env_id (assuming all envs have same objects)
        if len(env_ids) > 0:
            first_env_id = (
                env_ids[0].item() if isinstance(env_ids, torch.Tensor) else env_ids[0]
            )
            all_obj_names = list(
                self.scene.scene_manager.rigid_objects[first_env_id].keys()
            )
            # For each object name, collect poses across all env_ids
            for obj_name in all_obj_names:
                # skip simple_desk
                if obj_name == "simple_desk":
                    continue
                obj_poses = []
                for env_id in env_ids:
                    translation, orientation = self.scene.scene_manager.rigid_objects[
                        env_id
                    ][obj_name][0].get_local_pose()

                    obj_poses.append(torch.cat([translation, orientation], dim=0))

                # Stack poses: shape (num_envs, 7)
                object_poses_dict[obj_name] = torch.stack(obj_poses, dim=0)

        return object_poses_dict

    def get_grasp_pose(
        self,
        env_ids: Sequence[int] | None = None,
        obj_name: str | None = None,
        hand_type: str | None = None,
        grasp_type: str | None = None,
        obj_id: int = 0,
        transform_to_world: bool = True,
    ) -> list:
        """
        Get grasp poses for each env. Returns list of dicts (one per env).

        Args:
            env_ids: Environment IDs. If None, use all.
            obj_name: Object name. If None, use first non-desk object.
            hand_type: None = use "grasp_pose" annotation;
                      "xhand" = use "xhand_grasp_pose" (functional_grasp/grasp with coarse/fine/final).
            grasp_type: Optional filter (e.g. "functional_grasp", "grasp", or part name like "body").
            obj_id: Object index when multiple instances of same type. Default 0.
            transform_to_world: If True, poses in world frame; if False, in object frame.

        Returns:
            list: One element per env_id. Each element is dict from get_grasp_poses, or None.
        """
        if env_ids is None:
            env_ids = torch.arange(
                self.scene.robot_manager.num_envs, device=self.device
            )
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        grasp_pose_list = []
        for env_id in env_ids:
            rigid_objs = self.scene.scene_manager.rigid_objects[env_id]
            obj_name_used = obj_name
            if obj_name_used is None or obj_name_used not in rigid_objs:
                obj_name_used = next(
                    (k for k in rigid_objs if k != "simple_desk"), None
                )
            if obj_name_used is None:
                grasp_pose_list.append(None)
                continue
            obj_list = rigid_objs[obj_name_used]
            if not obj_list or obj_id >= len(obj_list):
                grasp_pose_list.append(None)
                continue
            rigid_obj = obj_list[obj_id]
            grasp_pose = rigid_obj.get_grasp_poses(
                grasp_type=grasp_type,
                transform_to_world=transform_to_world,
                device=self.device,
                hand_type=hand_type,
            )
            grasp_pose_list.append(grasp_pose)
        return grasp_pose_list

    def get_grasp_pose_updated(
        self,
        env_ids: Sequence[int],
        obj_name: str,
        obj_id: int,
        obj_type: str,
        hand_type: str,
        selected_idx: int,
        functional_grasp: bool = True,
        part: str | None = None,
    ) -> dict | None:
        """
        Get selected grasp pose (coarse/fine/final) in world frame using current object pose.
        For reactive mode: updates grasp target when object moves.

        Args:
            env_ids: Environment IDs.
            obj_name: Object name.
            obj_id: Object index.
            obj_type: "rigid" or "geometry".
            hand_type: e.g. "xhand", "dex3_1".
            selected_idx: Index of selected grasp candidate.
            functional_grasp: Prefer functional_grasp over grasp.
            part: Optional part filter.

        Returns:
            Dict with coarse_pose, fine_pose, final_pose, coarse_joints, fine_joints, final_joints
            (each torch.Tensor), or None if failed.
        """
        grasp_list = self.get_grasp_pose(
            env_ids=env_ids,
            obj_name=obj_name,
            hand_type=hand_type,
            obj_id=obj_id,
            transform_to_world=False,
        )
        if not grasp_list or grasp_list[0] is None:
            return None
        candidates = self._extract_grasp_candidates(
            grasp_list[0], functional_grasp, part
        )
        if not candidates or selected_idx >= len(candidates):
            return None
        obj_pose = super().get_object_pose(env_ids, obj_type, obj_name, obj_id)
        if obj_pose.shape[0] < 1:
            return None
        obj_pos = obj_pose[0, :3].to(self.device)
        obj_quat = obj_pose[0, 3:7].to(self.device)
        chosen = candidates[selected_idx]
        dev = self.device

        def _t(k, key):
            phase = k.get(key)
            if phase is None:
                return None
            pos = (
                phase["position"]
                if isinstance(phase["position"], torch.Tensor)
                else torch.tensor(phase["position"], dtype=torch.float32, device=dev)
            )
            ori = (
                phase["orientation"]
                if isinstance(phase["orientation"], torch.Tensor)
                else torch.tensor(phase["orientation"], dtype=torch.float32, device=dev)
            )
            if pos.device != dev:
                pos = pos.to(dev)
            if ori.device != dev:
                ori = ori.to(dev)
            return torch.cat([pos.flatten()[:3], ori.flatten()[:4]], dim=0)

        coarse_local = _t(chosen, "coarse_grasp")
        fine_local = _t(chosen, "fine_grasp")
        final_local = _t(chosen, "final_grasp")
        if coarse_local is None:
            return None

        coarse_pose = RigidObject.transform_pose_to_world(
            coarse_local, obj_pos, obj_quat
        )
        fine_pose = (
            RigidObject.transform_pose_to_world(fine_local, obj_pos, obj_quat)
            if fine_local is not None
            else None
        )
        final_pose = (
            RigidObject.transform_pose_to_world(final_local, obj_pos, obj_quat)
            if final_local is not None
            else None
        )
        coarse_joints = chosen["coarse_grasp"]["joints"]
        fine_joints = (
            chosen["fine_grasp"]["joints"] if chosen.get("fine_grasp") else None
        )
        final_joints = (
            chosen["final_grasp"]["joints"] if chosen.get("final_grasp") else None
        )
        if not isinstance(coarse_joints, torch.Tensor):
            coarse_joints = torch.tensor(coarse_joints, dtype=torch.float32, device=dev)
        if fine_joints is not None and not isinstance(fine_joints, torch.Tensor):
            fine_joints = torch.tensor(fine_joints, dtype=torch.float32, device=dev)
        if final_joints is not None and not isinstance(final_joints, torch.Tensor):
            final_joints = torch.tensor(final_joints, dtype=torch.float32, device=dev)
        return {
            "coarse_pose": coarse_pose,
            "fine_pose": fine_pose,
            "final_pose": final_pose,
            "coarse_joints": coarse_joints,
            "fine_joints": fine_joints,
            "final_joints": final_joints,
        }

    @staticmethod
    def _extract_grasp_candidates(
        grasp_dict: dict,
        functional_grasp: bool = True,
        part: str | None = None,
    ) -> list:
        """Extract flat candidate list from grasp dict (same ordering as DexGrasp)."""
        if not grasp_dict or not isinstance(grasp_dict, dict):
            return []
        func_dict = grasp_dict.get("functional_grasp", {})
        grasp_only = grasp_dict.get("grasp", {})

        def _from_parts(d, part_name):
            out = []
            if part_name and part_name in d and isinstance(d[part_name], list):
                out.extend(d[part_name])
            if not out:
                for v in d.values():
                    if isinstance(v, list):
                        out.extend(v)
            return out

        if functional_grasp:
            out = _from_parts(func_dict, part)
            if not out:
                out = _from_parts(grasp_only, part)
        else:
            out = _from_parts(grasp_only, part)
            if not out:
                out = _from_parts(func_dict, part)
        return out

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
        if action is None:
            return None
        expected_dim = self.scene.robot_manager.total_action_dim
        assert action.shape[1] == expected_dim, (
            f"action.shape[1]: {action.shape[1]} does not match expected_dim: {expected_dim}"
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
        return {
            "state": state,
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Get the state of the environment.
        This method should be overridden by subclasses to define specific state retrieval.
        """
        robot_state = self.scene.robot_manager.get_robot_state()
        object_state = self.get_object_pose()
        state = {
            "robot_state": robot_state,
            "scene_state": {
                "object_pose": object_state,  # object_pose is a dictionary of object_name and object_pose
            },  # object_state is a dictionary of object_name and object_pose
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
        eef_pose = self.get_eef_pose()
        object_poses_dict = self.get_object_pose()

        # Get object name from action if available, otherwise use first object or default
        obj_name = "apple"

        # print("obj_name: ", obj_name)
        object_pos = object_poses_dict[obj_name][:, :3]
        # Single-arm: eef_pose [N, 7]; dual-arm: eef_pose [N, 2, 7], 任意 eef 到了就结束
        if eef_pose.dim() == 2:
            distance = torch.norm(eef_pose[:, :3] - object_pos, dim=1)
        else:
            right_dist = torch.norm(eef_pose[:, 0, :3] - object_pos, dim=1)
            left_dist = torch.norm(eef_pose[:, 1, :3] - object_pos, dim=1)
            distance = torch.minimum(right_dist, left_dist)
        object_z = object_pos[:, 2]

        # termination: eef与object距离小于0.3 并且 object z轴大于0.2
        termination = (distance < 0.3) & (object_z > 1.2)
        # print(f"distance: {distance}, object_z: {object_z}, termination: {termination}")

        # truncated: object掉下桌子，z轴小于0.8
        truncated = object_z < 0.8

        return termination, truncated
