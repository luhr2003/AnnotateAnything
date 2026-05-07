from typing import Any, Dict, Sequence

import torch
import gymnasium as gym
from magicsim.StardardEnv.Camera.TaskCameraBaseEnv import TaskCameraBaseEnv


class GoToEnv(TaskCameraBaseEnv):
    """
    Camera Move environment. Mirrors ReachEnv but termination is based on camera pose.
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)
        self._primary_camera_name: str | None = None

    def get_obs_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

    def get_policy_obs(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        if env_ids is None:
            env_ids = torch.arange(self.scene.num_envs, device=self.device)
        camera_pose = self.get_camera_pose(env_ids)
        # Get camera_info from capture_manager
        camera_info = self.scene.capture_manager.step(env_ids=env_ids)
        return {
            "camera_pose": camera_pose,
            "camera_info": camera_info,
        }

    def get_privilege_obs(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        target_pose = self.get_target_pose(env_ids)
        return {"target_pose": target_pose}

    def get_target_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(
                self.scene.num_envs, device=self.device, dtype=torch.long
            )
        target_pose = []
        for env_id in env_ids:
            translation, orientation = self.scene.scene_manager.geometry_objects[
                env_id
            ]["cube"][0].get_local_pose()
            target_pose.append(torch.cat([translation, orientation], dim=0))
        return torch.stack(target_pose, dim=0)

    def get_camera_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(
                self.scene.num_envs, device=self.device, dtype=torch.long
            )
        camera_state = self.scene.camera_manager.get_camera_state(env_ids=env_ids)
        return torch.cat([camera_state["pos"], camera_state["quat"]], dim=1)

    def process_camera_action(
        self,
        camera_action: Any | None,
        env_ids: Sequence[int] | None = None,  # noqa: ARG002
    ) -> Dict[str, Any] | None:
        return camera_action

    def _get_primary_camera_name(self) -> str:
        if self._primary_camera_name is None:
            camera_names = list(self.scene.camera_manager.camera_config.keys())
            if not camera_names:
                raise RuntimeError("No camera configured in camera_manager.")
            self._primary_camera_name = camera_names[0]
        return self._primary_camera_name

    def get_info(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        return {}

    def get_reward(
        self,
        camera_action: Dict[str, torch.Tensor] | None,
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def get_termination(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(
                self.scene.num_envs, device=self.device, dtype=torch.long
            )
        camera_pose = self.get_camera_pose(env_ids)[:, :3]
        target_pose = self.get_target_pose(env_ids)[:, :3]
        distance = torch.norm(camera_pose - target_pose, dim=1)
        termination_mask = distance < 0.02
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            env_indices = env_ids.to(torch.long)
        else:
            env_indices = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        reached_indices = env_indices[termination_mask]
        if reached_indices.numel() > 0:
            termination[reached_indices] = True
        truncated = torch.zeros_like(termination)
        return termination, truncated
