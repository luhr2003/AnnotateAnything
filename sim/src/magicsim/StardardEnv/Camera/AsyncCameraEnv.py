from typing import Any, List, Sequence

import torch
import gymnasium as gym
from magicsim.Collect.CameraAtomicSkill.CameraAtomicSkillManager import (
    CameraAtomicSkillManager,
)
from magicsim.Collect.CameraGlobalPlanner.CameraGlobalPlannerManager import (
    CameraGlobalPlannerManager,
)
from magicsim.StardardEnv.Camera.TaskCameraBaseEnv import TaskCameraBaseEnv


class AsyncCameraEnv(gym.Env):
    def __init__(self, config, cli_args, logger):
        self.env_string = config.env_string
        self.env_config = config.env
        self.env: TaskCameraBaseEnv = gym.make(
            self.env_string, config=self.env_config, cli_args=cli_args, logger=logger
        )
        self.num_envs = self.env.num_envs
        self.scene = self.env.scene
        self.device = self.env.device
        self.logger = logger

        # Initialize camera managers
        self.camera_atomic_skill_config = config.camera_atomic_skill
        if self.camera_atomic_skill_config is not None:
            self.camera_atomic_skill_manager = CameraAtomicSkillManager(
                self.env,
                self.num_envs,
                self.camera_atomic_skill_config,
                self.device,
                self.logger,
            )
        else:
            self.camera_atomic_skill_manager = None

        # Global planner config (for NavTo, etc.)
        # Historical configs used key 'lanner'; newer configs use 'camera_global_planner'.
        self.camera_global_planner_config = getattr(config, "lanner", None)
        if self.camera_global_planner_config is None:
            self.camera_global_planner_config = getattr(
                config, "camera_global_planner", None
            )
        if self.camera_global_planner_config is not None:
            self.camera_global_planner_manager = CameraGlobalPlannerManager(
                self.env,
                self.num_envs,
                self.camera_global_planner_config,
                self.device,
                self.logger,
            )
        else:
            self.camera_global_planner_manager = None

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        env_info = self.env.reset(seed, options)
        camera_global_planner_info = (
            self.camera_global_planner_manager.reset()
            if self.camera_global_planner_manager is not None
            else [None] * self.num_envs
        )
        camera_atomic_skill_info = (
            self.camera_atomic_skill_manager.reset()
            if self.camera_atomic_skill_manager is not None
            else [None] * self.num_envs
        )
        return {
            "env_info": env_info,
            "camera_global_planner_info": camera_global_planner_info,
            "camera_atomic_skill_info": camera_atomic_skill_info,
        }

    def sim_step(self):
        self.env.sim_step()

    def step(
        self,
        camera_actions: List[List[Any]] | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        failed_env_ids = []
        valid_env_ids = env_ids

        # Process camera actions if camera managers are available
        camera_action_for_planner = None
        if (
            camera_actions is not None
            and self.camera_atomic_skill_manager is not None
            and self.camera_global_planner_manager is not None
        ):
            camera_actions_dict, camera_valid_env_ids, camera_failed_at_atomic = (
                self.camera_atomic_skill_manager.step(camera_actions, env_ids)
            )
            camera_action_for_planner, camera_valid_env_ids, camera_failed_at_global = (
                self.camera_global_planner_manager.step(
                    camera_actions_dict, camera_valid_env_ids
                )
            )
            failed_env_ids = camera_failed_at_atomic + camera_failed_at_global
            valid_env_ids = camera_valid_env_ids

            # Adapt planner output to the format expected by CameraManager/scene:
            # CameraManager.step expects:
            #   actions: Dict[camera_name, poses_tensor[N,7]]
            # while CameraGlobalPlannerManager.step returns:
            #   {camera_name: {"env_ids": [...], "poses": tensor[N,7]}}
            if isinstance(camera_action_for_planner, dict):
                simple_actions = {}
                for cam_name, payload in camera_action_for_planner.items():
                    if isinstance(payload, dict) and "poses" in payload:
                        simple_actions[cam_name] = payload["poses"]
                    else:
                        simple_actions[cam_name] = payload
                camera_action_for_planner = simple_actions

        info = {}

        env_info = self.env.step(
            camera_action=camera_action_for_planner,
            env_ids=valid_env_ids,
            failed_env_ids=failed_env_ids,
        )
        info["env_info"] = env_info
        # Update camera-related infos
        camera_global_planner_info = self.camera_global_planner_manager.update(info)
        info["camera_global_planner_info"] = camera_global_planner_info
        camera_atomic_skill_info = self.camera_atomic_skill_manager.update(info)
        info["camera_atomic_skill_info"] = camera_atomic_skill_info

        return info, valid_env_ids
