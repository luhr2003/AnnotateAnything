"""
LocoManip Open Door environment: mobile base (g1) at (0,0,0), door at (0, 1.5, 0).
Combines LocoGraspEnv-style obs/action with OpenDrawerEnv-style door/articulation API.
Annotation: two-phase per trajectory — "approach" and "trajectory" (pull).
"""

from typing import Any, Dict, List, Sequence

import torch
import gymnasium as gym
from magicsim.Env.Planner.Utils import quat_mul
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from magicsim.Env.Utils.rotations import quat_to_rot_matrix


class LocoOpenDoorEnv(TaskBaseEnv):
    """
    Open Door environment for LocoManip: g1 at origin, door at (0, 1.5, 0).
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)

    def _waypoints_to_world(
        self,
        waypoints_list: List[List[float]],
        obj_pos: torch.Tensor,
        obj_quat: torch.Tensor,
        obj_rot: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Transform local waypoints to world frame, preserving trailing joint cols.

        Each waypoint is (x,y,z,qw,qx,qy,qz, j0..jK-1). The pose prefix is
        transformed by (obj_pos, obj_quat); trailing joint-angle columns are
        kept as-is (they are target hand-joint positions, already in joint space).

        Returns:
            [N, 7+K] tensor. Rows have a uniform width = min waypoint length
            across the input list (≥ 7). Returns [0, 7] on malformed input.
        """
        if not waypoints_list or not isinstance(waypoints_list[0], (list, tuple)):
            return torch.zeros(0, 7, device=device, dtype=torch.float32)
        width = min(len(w) for w in waypoints_list)
        if width < 7:
            return torch.zeros(0, 7, device=device, dtype=torch.float32)
        n = len(waypoints_list)
        arr = torch.tensor(
            [[w[i] for i in range(width)] for w in waypoints_list],
            device=device,
            dtype=torch.float32,
        )
        local_pos = arr[:, :3]
        local_quat = arr[:, 3:7]
        world_pos = (obj_rot @ local_pos.T).T + obj_pos.unsqueeze(0)
        obj_quat_exp = obj_quat.unsqueeze(0).expand(n, -1)
        world_quat = quat_mul(obj_quat_exp, local_quat)
        pose_world = torch.cat([world_pos, world_quat], dim=1)
        if width > 7:
            return torch.cat([pose_world, arr[:, 7:]], dim=1)
        return pose_world

    def get_obs_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

    def get_policy_obs(self) -> Dict[str, Any]:
        robot_state = self.scene.robot_manager.get_robot_state()
        camera_info = self.scene.capture_manager.step()
        return {"robot_state": robot_state, "camera_info": camera_info}

    def get_privilege_obs(self) -> Dict[str, Any]:
        articulation_pose = self._get_articulation_pose()
        return {"articulation_pose": articulation_pose}

    def _get_articulation_pose(
        self, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(
                self.scene.robot_manager.num_envs, device=self.device
            )
        poses = []
        for env_id in env_ids:
            translation, orientation = self.scene.scene_manager.articulation_objects[
                env_id
            ]["articulation_items"][0].get_local_pose()
            poses.append(torch.cat([translation, orientation], dim=0))
        return torch.stack(poses, dim=0)

    def process_action(self, action: torch.Tensor | list[Dict]):
        """Pass through action; LocoManip g1 uses full action dim (e.g. 43), do not truncate."""
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
        eef_pose = torch.cat([eef_pos, eef_quat], dim=1)
        return eef_pose[env_ids]

    def get_info(self) -> Dict[str, Any]:
        state = self.get_state()
        return {"state": state}

    def get_state(self) -> Dict[str, Any]:
        robot_state = self.scene.robot_manager.get_robot_state()
        articulation_pose = self._get_articulation_pose()
        state = {
            "robot_state": robot_state,
            "scene_state": {
                "articulation_pose": articulation_pose,
            },
            "camera_state": self.scene.camera_manager.get_all_camera_state(),
        }
        return state

    def get_reward(
        self,
        action: torch.Tensor | list[Dict],
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        reward = [0] * self.num_envs
        return reward

    # ------------------------------------------------------------------ #
    # Door data-access helpers (used by AtomicSkill / OpenDoor)
    # ------------------------------------------------------------------ #

    # Segment-level keys that are metadata, not waypoint trajectories.
    _ANNOTATION_META_KEYS = {
        "rotate_joint",
        "push_target_angle_deg",
        "rotate_target_angle_deg",
    }

    def get_door_trajectories(
        self,
        env_id: int,
        annotation_name: str = "dex3_1_open_by_handle_trajectory",
        joint_id: int = -1,
    ) -> dict:
        """Load world-frame door trajectories from annotation, generic over phases.

        Returns each trajectory as a dict mapping phase_name → world-frame [N,7]
        tensor. Supported annotation variants:
          - open-by-handle (pull):  {"approach": ..., "pull"/"trajectory": ...}
          - rotate-and-push:        {"approach": ..., "rotate": ..., "push": ...}

        Each waypoint can be 7 (pose) or longer (pose + joint angles); only the
        first 7 elements (x,y,z,w,qx,qy,qz) are used here.

        joint_id < 0 disables joint filtering; joint_id >= 0 filters to the
        annotation key "joint_{joint_id}" (legacy Door/9280 style).

        Returns:
            dict: {traj_key: {phase_name: Tensor [N, 7], ...}}
        """
        if not hasattr(self.scene, "scene_manager") or self.scene.scene_manager is None:
            return {}
        sm = self.scene.scene_manager
        if not hasattr(sm, "articulation_objects") or env_id >= len(
            sm.articulation_objects
        ):
            return {}
        art_by_env = sm.articulation_objects[env_id]
        if not isinstance(art_by_env, dict) or "articulation_items" not in art_by_env:
            return {}
        items = art_by_env["articulation_items"]
        if not items or len(items) == 0:
            return {}
        obj = items[0]
        joint_name = None
        if joint_id >= 0:
            num_joints = obj.num_joints
            if joint_id >= num_joints:
                raise ValueError(
                    f"joint_id={joint_id} out of range: articulation has {num_joints} joints"
                )
            joint_name = f"joint_{joint_id}"

        annotation_data = obj.get_annotation(annotation_name)
        if annotation_data is None:
            return {}

        trajs = annotation_data.get("trajectories")
        if not isinstance(trajs, dict):
            return {}

        pos, quat = obj.get_local_pose()
        device = pos.device if isinstance(pos, torch.Tensor) else self.device
        if not isinstance(pos, torch.Tensor):
            pos = torch.tensor(pos, dtype=torch.float32, device=device)
        if not isinstance(quat, torch.Tensor):
            quat = torch.tensor(quat, dtype=torch.float32, device=device)
        obj_pos = pos.squeeze()[:3]
        obj_quat = quat.squeeze()[:4]
        obj_rot = quat_to_rot_matrix(obj_quat.unsqueeze(0))[0]

        empty_traj = torch.zeros(0, 7, device=device, dtype=torch.float32)

        result = {}
        for joint, joint_trajs in trajs.items():
            if joint_name is not None and joint != joint_name:
                continue
            if not isinstance(joint_trajs, dict):
                continue
            for traj_id, seg in joint_trajs.items():
                try:
                    key = int(traj_id)
                except (TypeError, ValueError):
                    key = traj_id
                if isinstance(seg, dict):
                    phase_dict: dict = {}
                    for phase_name, wps in seg.items():
                        if phase_name in self._ANNOTATION_META_KEYS:
                            continue
                        if not isinstance(wps, list) or len(wps) == 0:
                            phase_dict[phase_name] = empty_traj.clone()
                            continue
                        phase_dict[phase_name] = self._waypoints_to_world(
                            wps, obj_pos, obj_quat, obj_rot, device
                        )
                    if any(t.shape[0] > 0 for t in phase_dict.values()):
                        result[key] = phase_dict
                elif (
                    isinstance(seg, torch.Tensor)
                    and seg.ndim == 2
                    and seg.shape[1] >= 7
                ):
                    t = seg[:, :7].to(device=device)
                    result[key] = {"approach": t.clone(), "pull": t.clone()}
        return result

    def get_door_object_pose(self, env_id: int):
        """Return (pos, quat, scale) for the door articulation object."""
        obj = self.scene.scene_manager.articulation_objects[env_id][
            "articulation_items"
        ][0]
        pos, quat = obj.get_local_pose()
        scale = torch.tensor(obj.init_scale, dtype=torch.float32)
        return pos, quat, scale

    def get_termination(self) -> tuple[torch.Tensor, torch.Tensor]:
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for env_id in range(self.num_envs):
            obj = self.scene.scene_manager.articulation_objects[env_id][
                "articulation_items"
            ][0]
            current_pos = obj.get_current_joint_positions()
            lower = torch.as_tensor(obj.lower_joint_positions, dtype=torch.float32)
            upper = torch.as_tensor(obj.upper_joint_positions, dtype=torch.float32)
            joint_range = upper - lower
            valid = joint_range.abs() > 1e-6
            if valid.any():
                progress = (current_pos[valid] - lower[valid]) / joint_range[valid]
                if progress.max() >= 0.2:
                    termination[env_id] = True
        return termination, truncated
