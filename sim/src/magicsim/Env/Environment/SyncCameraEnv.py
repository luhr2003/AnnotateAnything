"""
This is a camera base environment class for sync camera tasks and 3D/4D data collection.
All the action here is synchronous and atomic. The action here is atomic, meaning that the action will be executed in a single step.
This environment only handles camera actions, not robot actions.

"""

from typing import Any, Dict, Sequence

import torch
from magicsim.Env.Environment.SyncCollectEnv import SyncCollectEnv
from magicsim.Env.Planner.CameraPlannerManager import CameraPlannerManager


class SyncCameraEnv(SyncCollectEnv):
    """
    This is a camera base environment class for sync camera tasks and 3D/4D data collection.
    Only handles camera actions, not robot actions.
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)
        self.camera_config = config.camera
        # Initialize CameraPlannerManager (camera-only)
        self.planner_manager = CameraPlannerManager(
            num_envs=self.num_envs,
            camera_config=self.camera_config,
            device=self.device,
            logger=logger,
        )

    def _setup_scene(self, sim):
        """
        Initialize the environment.
        This function will be called before simulation context create and be called by isaaclab _setup_scene function
        """
        self.planner_manager.initialize(
            camera_manager=self.camera_manager,
            camera_config=self.camera_config,
        )
        super()._setup_scene(sim)

    def step(
        self,
        camera_action: torch.Tensor | list[Dict] = None,
        env_ids: Sequence[int] | None = None,
    ):
        """
        Args:
            camera_action: Camera action, can be:
                - torch.Tensor: [N, 7] target pose [x,y,z,w,x,y,z] for single camera
                - dict: {camera_name: [N, 7]} target poses for multiple cameras
        """
        if camera_action is None:
            super().step()
            return {}

        camera_info = {}  # Note here we only return the camera info for env_ids

        camera_info["command"] = camera_action

        # Process camera actions through planner
        planner_result = self.planner_manager.step(
            camera_action=camera_action,
            env_ids=env_ids,
        )

        # Extract and apply camera poses
        if planner_result is not None:
            camera_action_for_manager = planner_result
            # Convert planner result format to camera_manager format
            # planner_result is dict {camera_name: {"env_ids": ..., "poses": ...}}
            # camera_manager.step expects dict {camera_name: [N, 7]} or torch.Tensor
            camera_action_dict = {}
            for camera_name, camera_data in camera_action_for_manager.items():
                if isinstance(camera_data, dict) and "poses" in camera_data:
                    camera_action_dict[camera_name] = camera_data["poses"]
                else:
                    camera_action_dict[camera_name] = camera_data

            # If only one camera, use tensor format for backward compatibility
            if len(camera_action_dict) == 1:
                camera_action_for_manager = list(camera_action_dict.values())[0]
            else:
                camera_action_for_manager = camera_action_dict

            camera_info["camera_action"] = camera_action_for_manager
            step_camera_info = self.camera_manager.step(
                camera_action_for_manager, env_ids=env_ids
            )
            camera_info.update(step_camera_info)

        # Advance simulation
        super().step()

        return camera_info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        Reset the environment.
        This should only be called once at the beginning of the environment.
        In this function, we will call scene_manager.reset(soft=False) to load all the objects managed in scene manager
        It will also reset the reset count.
        """
        super().reset(
            seed=seed, options=options
        )  # Lab Reset: Will Reset All Object Managed By Lab
        print("Scene Reset Finished")
        self.planner_manager.reset()  # Reset all planners

    def reset_idx(
        self,
        env_ids: Sequence[int] = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """
        Reset specific environments.
        """
        if env_ids is None:
            env_id_list = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_id_list = env_ids.detach().cpu().tolist()
        else:
            env_id_list = [int(i) for i in env_ids]

        super().reset_idx(env_ids=env_id_list, seed=seed, options=options)

        self.planner_manager.reset_idx(
            env_ids=env_id_list
        )  # Reset the planners in the specific environments

    def _update_seed_managers(self):
        super()._update_seed_managers()
        # Camera managers are updated in SyncCollectEnv._update_seed_managers
