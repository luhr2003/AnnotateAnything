from typing import Dict, Sequence
from omegaconf import DictConfig
from magicsim.Env.Planner.Planner import Planner
from magicsim.Env.Planner.FlyingCamera import (
    LinearFlyingCameraPlanner,
    GimbalFlyingCameraPlanner,
)
from magicsim.Env.Utils.file import Logger
import torch
from magicsim.Env.Sensor.CameraManager import CameraManager


class CameraPlannerManager:
    """
    Camera Planner Manager for camera-only environments.
    This manager only handles camera planners, not robot planners.
    """

    def __init__(
        self,
        num_envs: int,
        camera_config: DictConfig,
        device: torch.device,
        logger: Logger,
    ):
        """
        Initialize camera planner manager.

        Args:
            num_envs: Number of environments
            camera_config: Camera configuration dict
            device: Device to run on
            logger: Logger instance
        """
        self.camera_manager = None
        self.num_envs = num_envs
        self.camera_config = camera_config
        self.device = device
        self.logger = logger
        # Camera planners
        self.camera_planners: Dict[str, Planner] = {}
        self.camera_planner_configs: Dict[str, DictConfig] = {}

    def initialize(
        self,
        camera_manager: CameraManager,
        camera_config: DictConfig = None,
    ):
        """
        Initialize the camera planner manager after the world is initialized.

        Args:
            camera_manager: CameraManager instance
            camera_config: Camera configuration dict (optional, for camera planners)
        """
        self.camera_manager = camera_manager
        if camera_config is not None:
            self.camera_config = camera_config

    def step(
        self,
        camera_action: torch.Tensor | dict[str, torch.Tensor] = None,
        env_ids: Sequence[int] = None,
    ):
        """
        Process camera planners.

        Args:
            camera_action: Camera action, can be:
                - torch.Tensor: [N, 7] target pose [x,y,z,w,x,y,z] for single camera
                - dict: {camera_name: [N, 7]} target poses for multiple cameras
                - dict: {camera_name: {"env_ids": [...], "poses": tensor}} for multiple cameras with env_ids
            env_ids: List of environment IDs

        Returns:
            dict: {camera_name: {"env_ids": [...], "poses": tensor}} planned camera poses, or None if no action
        """
        if camera_action is None:
            return None

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, device=self.device)
            else:
                env_ids = env_ids.to(self.device)

        planned_poses = {}

        # Handle single camera tensor input
        if isinstance(camera_action, torch.Tensor):
            if len(self.camera_planners) == 1:
                camera_name = list(self.camera_planners.keys())[0]
                planner = self.camera_planners[camera_name]
                planned_pose = planner.forward(camera_action, env_ids)
                planned_poses[camera_name] = {
                    "env_ids": env_ids,
                    "poses": planned_pose,
                }
            else:
                raise ValueError(
                    "Multiple camera planners exist, please provide dict with camera names."
                )
        elif isinstance(camera_action, dict):
            for camera_name, payload in camera_action.items():
                if camera_name not in self.camera_planners:
                    continue
                planner = self.camera_planners[camera_name]
                if isinstance(payload, dict) and "poses" in payload:
                    cam_env_ids = payload.get("env_ids", env_ids)
                    target_pose = payload["poses"]
                    planned_pose = planner.forward(target_pose, cam_env_ids)
                    planned_poses[camera_name] = {
                        "env_ids": cam_env_ids,
                        "poses": planned_pose,
                    }
                else:
                    planned_pose = planner.forward(payload, env_ids)
                    planned_poses[camera_name] = {
                        "env_ids": env_ids,
                        "poses": planned_pose,
                    }

        return planned_poses if planned_poses else None

    def reset_idx(self, env_ids):
        """
        Reset camera planners for specific environments.

        Args:
            env_ids: List of environment IDs to reset
        """
        for camera_name, planner in self.camera_planners.items():
            if planner is not None:
                planner.reset_idx(env_ids)

    def reset(self):
        """
        Reset all camera planners.
        This will setup planners and reset them for all environments.
        """
        self.setup_planner()
        env_idx = list(range(self.num_envs))
        for camera_name, planner in self.camera_planners.items():
            if planner is not None:
                planner.reset_idx(env_idx)

    def setup_planner(self):
        """
        Setup camera planners based on camera configuration.
        """
        if self.camera_config is not None:
            for camera_name, cam_cfg in self.camera_config.items():
                # Skip non-dict entries (e.g., global camera settings like output_dir)
                if not isinstance(cam_cfg, (dict, DictConfig)):
                    continue
                planner_cfg = cam_cfg.get("planner", None)
                if planner_cfg is not None:
                    planner_type = planner_cfg.get("type", "linear").lower()
                    self.camera_planner_configs[camera_name] = planner_cfg

                    # Get planner parameters
                    planner_params = planner_cfg.get("params", {})

                    # Create camera planner
                    if planner_type == "linear":
                        camera_planner = LinearFlyingCameraPlanner(
                            camera_manager=self.camera_manager,
                            camera_name=camera_name,
                            device=self.device,
                            **planner_params,
                        )
                    elif planner_type == "gimbal":
                        camera_planner = GimbalFlyingCameraPlanner(
                            camera_manager=self.camera_manager,
                            camera_name=camera_name,
                            device=self.device,
                            **planner_params,
                        )
                    else:
                        raise NotImplementedError(
                            f"Camera planner type {planner_type} not supported."
                        )

                    self.camera_planners[camera_name] = camera_planner
