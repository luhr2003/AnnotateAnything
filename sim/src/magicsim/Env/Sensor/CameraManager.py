"""
This is the camera manager responsible for managing all cameras in the environment.
It handles camera creation, configuration, and management of camera-related operations ie. camera predefined action here.

1. Create Camera
2. Camera Predefined Action

Note it is not responsible get data from camera or handle annotation, which is handled by CaptureManager.
"""

from typing import Dict, Sequence, List, Tuple
from magicsim.Env.Environment.Isaac.IsaacRLEnv import IsaacRLEnv
from omegaconf import DictConfig, OmegaConf
import torch
import os
import numpy as np
from numpy import ndarray
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.semantics import add_labels, remove_labels
from magicsim.Env.Utils.rotations import (
    euler_angles_to_quat,
)
from magicsim import MAGICSIM_ASSETS
from magicsim.Env.Utils.constants import (
    KINECT_NFOV_CAMERA_PATH,
    KINECT_NFOV_FREQUENCY,
    KINECT_NFOV_RESOLUTION,
    KINECT_RGB_CAMERA_PATH,
    KINECT_RGB_FREQUENCY,
    KINECT_RGB_RESOLUTION,
    KINECT_WFOV_CAMERA_PATH,
    KINECT_WFOV_FREQUENCY,
    KINECT_WFOV_RESOLUTION,
    REALSENSE_RGB_CAMERA_PATH,
    REALSENSE_RGB_FREQUENCY,
    REALSENSE_RGB_RESOLUTION,
    REALSENSE_DEPTH_CAMERA_PATH,
    REALSENSE_DEPTH_FREQUENCY,
    REALSENSE_DEPTH_RESOLUTION,
)
from magicsim.Env.Environment.Utils.Basic import seed_everywhere

COLLISION_CHECK_FRAME = 5


class CameraManager:
    """
    This manager is responsible for managing all cameras in the environment.
    It handles camera creation, configuration, and management of camera-related operations.
    """

    def __init__(
        self,
        num_envs: int,
        camera_config: DictConfig,
        device: torch.device,
        seeds_per_env: Sequence[int] | None = None,
    ):
        """
        Initialize the CameraManager with the number of environments, camera configuration and device.
        We also handle camera_predefined_action here.
        """
        self.num_envs = num_envs
        self.num_cams = 1
        self.camera_config = camera_config
        self.device = device
        self.sim: IsaacRLEnv = None
        self.cameras: List[List[Camera]] = []  # List of cameras for all environments
        self.cameras_xform_path: List[List[str]] = []  # List of camera xforms path
        self.cameras_xform: List[List[SingleGeometryPrim]] = []  # List of camera xforms
        self.action_noise: List[
            List[Tuple[ndarray]]
        ] = []  # Describes noise in coordinates and orientation when a camera moves, read from config
        self.camera_poses: List[List[ndarray]] = []
        self._seeds_per_env: List[int] | None = None
        self.update_env_seeds(seeds_per_env)

    def update_env_seeds(self, seeds: Sequence[int] | None):
        """Update per-environment seed list used for camera randomness."""
        if seeds is None:
            self._seeds_per_env = None
            return
        seed_list = [int(s) for s in seeds]
        if len(seed_list) != self.num_envs:
            raise ValueError(
                f"seed list length {len(seed_list)} does not match num_envs {self.num_envs}."
            )
        self._seeds_per_env = seed_list

    def _set_env_seed(self, env_id: int):
        if self._seeds_per_env is None:
            return
        if env_id < 0 or env_id >= len(self._seeds_per_env):
            raise IndexError(
                f"Requested env_id {env_id} exceeds configured seeds (len={len(self._seeds_per_env)})."
            )
        seed_everywhere(self._seeds_per_env[env_id])

    def initialize(self, sim: IsaacRLEnv):
        """
        Initialize the camera manager.
        This method should be called before the simulation context is created.

        """
        self.create_cameras()
        self.sim = sim
        if self._seeds_per_env:
            seed_everywhere(self._seeds_per_env[0])

    def post_init(self):
        """
        This method should be called after the simulation context is created.
        """
        self.physics_sim_view = SimulationManager.get_physics_sim_view()
        for env_id in range(self.num_envs):
            for camera_xform in self.cameras_xform[env_id]:
                camera_xform.initialize(self.physics_sim_view)

    def create_cameras(self):
        """
        Create cameras based on the configuration. We create camera prims here and set camera parameters here.
        We do not set camera pose here, which is handled by init_camera_pose.
        We do not init camera here(create render product), which is handled by init_cameras.
        This method should be called at the first reset of the environment.
        """
        # This could involve creating camera instances and setting their properties
        for env_id in range(self.num_envs):
            env_name = f"/World/envs/env_{env_id}"
            env_camera_name = env_name + "/Cameras"
            self.cameras.append([])
            self.cameras_xform_path.append([])
            self.cameras_xform.append([])
            self.action_noise.append([])
            cam_id = 0
            config = OmegaConf.to_container(self.camera_config, resolve=True)
            if "output_dir" in config:
                config.pop("output_dir")
            if "disable_render_product" in config:
                config.pop("disable_render_product")
            if "render_frame_num" in config:
                config.pop("render_frame_num")
            if "enable_tiled" in config:
                config.pop("enable_tiled")
            if "colorize_depth" in config:
                self.colorize_depth = config.pop("colorize_depth")
            self.camera_config = OmegaConf.create(config)
            for camera_name, camera_config in self.camera_config.items():
                if camera_config.camera.get("mount_link") is not None:
                    cur_camera_xform_path = find_unique_string_name(
                        env_name
                        + "/"
                        + camera_config.camera.mount_link
                        + "/"
                        + f"Camera_{cam_id}",
                        is_unique_fn=lambda x: not is_prim_path_valid(x),
                    )
                else:
                    cur_camera_xform_path = find_unique_string_name(
                        env_name + f"/Camera_{cam_id}",
                        is_unique_fn=lambda x: not is_prim_path_valid(x),
                    )
                self.cameras_xform_path[env_id].append(cur_camera_xform_path)

                if hasattr(camera_config.camera, "action_noise"):
                    action_noise_pos = (
                        torch.tensor(camera_config.camera.action_noise.pos.min),
                        torch.tensor(camera_config.camera.action_noise.pos.max),
                    )
                    action_noise_ori = (
                        torch.tensor(camera_config.camera.action_noise.ori.min),
                        torch.tensor(camera_config.camera.action_noise.ori.max),
                    )
                    self.action_noise[env_id].append(
                        (action_noise_pos, action_noise_ori)
                    )
                else:
                    self.action_noise[env_id].append(
                        (
                            (torch.zeros(3), torch.zeros(3)),
                            (torch.zeros(3), torch.zeros(3)),
                        )
                    )

                if camera_config.camera.mesh == "kinect":
                    add_reference_to_stage(
                        usd_path=os.path.join(
                            MAGICSIM_ASSETS, "Sensor/Camera/kinect.usd"
                        ),
                        prim_path=cur_camera_xform_path,
                    )
                    cur_camera_xform = SingleGeometryPrim(
                        prim_path=cur_camera_xform_path,
                        name=find_unique_string_name(
                            camera_name,
                            is_unique_fn=lambda x: not is_prim_path_valid(x),
                        ),
                        collision=False,
                    )
                    # apply semantic label to camera xform
                    self._set_camera_semantics(
                        cur_camera_xform, camera_name, camera_config
                    )
                    self.cameras_xform[env_id].append(cur_camera_xform)
                    if camera_config.camera.mode == "WFOV":
                        cur_camera_path = (
                            cur_camera_xform_path + KINECT_WFOV_CAMERA_PATH
                        )
                        cur_camera_resolution = KINECT_WFOV_RESOLUTION
                        cur_camera = Camera(
                            prim_path=cur_camera_path,
                            resolution=cur_camera_resolution,
                            frequency=KINECT_WFOV_FREQUENCY,
                        )
                        self.cameras[env_id].append(cur_camera)

                    elif camera_config.camera.mode == "NFOV":
                        cur_camera_path = (
                            cur_camera_xform_path + KINECT_NFOV_CAMERA_PATH
                        )
                        cur_camera_resolution = KINECT_NFOV_RESOLUTION
                        cur_camera = Camera(
                            prim_path=cur_camera_path,
                            resolution=cur_camera_resolution,
                            frequency=KINECT_NFOV_FREQUENCY,
                        )
                        self.cameras[env_id].append(cur_camera)

                    elif camera_config.camera.mode == "RGB":
                        cur_camera_path = cur_camera_xform_path + KINECT_RGB_CAMERA_PATH
                        cur_camera_resolution = KINECT_RGB_RESOLUTION
                        cur_camera = Camera(
                            prim_path=cur_camera_path,
                            resolution=cur_camera_resolution,
                            frequency=KINECT_RGB_FREQUENCY,
                        )
                        self.cameras[env_id].append(cur_camera)

                    else:
                        raise ValueError("Invalid camera mode")

                elif camera_config.camera.mesh == "realsense":
                    add_reference_to_stage(
                        usd_path=os.path.join(
                            MAGICSIM_ASSETS, "Sensor/Camera/realsense.usd"
                        ),
                        prim_path=cur_camera_xform_path,
                    )
                    cur_camera_xform = SingleGeometryPrim(
                        prim_path=cur_camera_xform_path,
                        name=find_unique_string_name(
                            camera_name,
                            is_unique_fn=lambda x: not is_prim_path_valid(x),
                        ),
                        collision=False,
                    )
                    # apply semantic label to camera xform
                    self._set_camera_semantics(
                        cur_camera_xform, camera_name, camera_config
                    )
                    self.cameras_xform[env_id].append(cur_camera_xform)
                    if camera_config.camera.mode == "RGB":
                        cur_camera_path = (
                            cur_camera_xform_path + REALSENSE_RGB_CAMERA_PATH
                        )
                        cur_camera_resolution = REALSENSE_RGB_RESOLUTION
                        cur_camera = Camera(
                            prim_path=cur_camera_path,
                            resolution=cur_camera_resolution,
                            frequency=REALSENSE_RGB_FREQUENCY,
                        )
                        self.cameras[env_id].append(cur_camera)

                    elif camera_config.camera.mode == "DEPTH":
                        cur_camera_path = (
                            cur_camera_xform_path + REALSENSE_DEPTH_CAMERA_PATH
                        )
                        cur_camera_resolution = REALSENSE_DEPTH_RESOLUTION
                        cur_camera = Camera(
                            prim_path=cur_camera_path,
                            resolution=cur_camera_resolution,
                            frequency=REALSENSE_DEPTH_FREQUENCY,
                        )
                        self.cameras[env_id].append(cur_camera)

                    else:
                        raise ValueError("Invalid camera mode")
                elif camera_config.camera.mesh == "pinhole":
                    get_current_stage().DefinePrim(cur_camera_xform_path, "Xform")
                    cur_camera_xform = SingleGeometryPrim(
                        prim_path=cur_camera_xform_path,
                        name=find_unique_string_name(
                            camera_name,
                            is_unique_fn=lambda x: not is_prim_path_valid(x),
                        ),
                        collision=False,
                    )
                    self.cameras_xform[env_id].append(cur_camera_xform)
                    cur_camera_path = cur_camera_xform_path + "/PinholeCamera"
                    cur_camera = Camera(
                        prim_path=cur_camera_path,
                        resolution=list(camera_config.camera.resolution),
                        frequency=camera_config.camera.frequency,
                    )
                    cur_camera.set_lens_distortion_model("pinhole")
                    cur_camera.set_local_pose([0, 0, 0], [1, 0, 0, 0], "usd")
                    if camera_config.camera.get("focal_length") is not None:
                        cur_camera.set_focal_length(camera_config.camera.focal_length)
                    if camera_config.camera.get("horizontal_aperture") is not None:
                        cur_camera.set_horizontal_aperture(
                            camera_config.camera.horizontal_aperture, False
                        )
                    if camera_config.camera.get("vertical_aperture") is not None:
                        cur_camera.set_vertical_aperture(
                            camera_config.camera.vertical_aperture, False
                        )
                    if camera_config.camera.get("clipping_range") is not None:
                        cur_camera.set_clipping_range(
                            near_distance=camera_config.camera.clipping_range[0],
                            far_distance=camera_config.camera.clipping_range[1],
                        )
                    self.cameras[env_id].append(cur_camera)

                else:
                    raise NotImplementedError("Camera type not implemented")
                cam_id += 1
            self.num_cams = cam_id
        return

    def _set_camera_semantics(
        self,
        camera_xform: SingleGeometryPrim,
        camera_name: str,
        camera_config: DictConfig,
    ):
        """Clear existing labels and apply semantic label to camera prim.

        Priority: camera_config.semantic_label (if provided) else camera_name.
        """
        try:
            # remove existing labels on xform and its descendants
            remove_labels(camera_xform.prim, include_descendants=True)
            semantic_label = None
            # support optional semantic label from config
            try:
                if (
                    hasattr(camera_config, "semantic_label")
                    and camera_config.semantic_label
                ):
                    semantic_label = str(camera_config.semantic_label)
            except Exception:
                semantic_label = None
            if not semantic_label:
                semantic_label = str(camera_name)
            add_labels(camera_xform.prim, [semantic_label])
        except Exception:
            pass

    def step(self, actions: Dict | torch.Tensor = None, env_ids: Sequence[int] = None):
        """
        Step the camera manager to update camera poses or perform any necessary operations.
        This method should be called every step of the environment

        Args:
            actions: Camera action, can be:
                - torch.Tensor: [N, 7] target pose [x,y,z,w,x,y,z] for single camera
                - dict: {camera_name: [N, 7]} target poses for multiple cameras
            env_ids: List of environment IDs to update

        Returns:
            dict: Camera info dictionary (currently empty, camera_info should come from capture_manager)
        """
        if actions is None:
            return {}

        if env_ids is None:
            env_ids = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        else:
            env_ids = [int(e) for e in env_ids]

        # Update camera poses based on actions
        if isinstance(actions, torch.Tensor):
            # Single camera, tensor format [N, 7]
            if len(actions.shape) == 2 and actions.shape[1] == 7:
                # Get first camera name
                camera_name = list(self.camera_config.keys())[0]
                self.set_camera_pose(camera_name, actions, env_ids)
        elif isinstance(actions, dict):
            # Multiple cameras, dict format {camera_name: [N, 7]}
            for camera_name, poses in actions.items():
                if (
                    isinstance(poses, torch.Tensor)
                    and len(poses.shape) == 2
                    and poses.shape[1] == 7
                ):
                    self.set_camera_pose(camera_name, poses, env_ids)

        # Return empty dict (camera_info should come from capture_manager.step())
        return {}

    def init_camera_pose(
        self, env_ids: Sequence[int] = None, use_randomization: bool = True
    ):
        """
        Initialize camera poses based on the configuration.
        We do camera pose domain randomization here (if use_randomization=True).
        This method should be called at the beginning of the environment.

        Args:
            env_ids: Environment IDs to initialize. If None, initialize all.
            use_randomization: If True, apply domain randomization. If False, use exact base position from config.
                               Set to False during reset to ensure camera returns to fixed initial position.
        """
        # Implementation for initializing camera poses
        if env_ids is None:
            env_id_iter = range(self.num_envs)
        else:
            env_id_iter = [int(env_id) for env_id in env_ids]

        if not self.camera_poses or len(self.camera_poses) != self.num_envs:
            self.camera_poses = [[] for _ in range(self.num_envs)]

        for env_id in env_id_iter:
            self._set_env_seed(env_id)
            self.camera_poses[env_id] = []
            cam_id = 0
            for camera_name, camera_config in self.camera_config.items():
                cur_camera_xform = self.cameras_xform[env_id][cam_id]
                random_pos = torch.zeros(3)
                random_ori = torch.zeros(3)
                # Only apply randomization if explicitly enabled (for initial setup)
                # During reset, use_randomization=False to ensure exact base position
                if use_randomization and (
                    hasattr(camera_config.camera, "random")
                    and camera_config.camera.random is not None
                ):
                    if (
                        hasattr(camera_config.camera.random, "pos")
                        and camera_config.camera.random.pos is not None
                    ):
                        random_pos_min = torch.tensor(
                            camera_config.camera.random.pos.min
                        )
                        random_pos_max = torch.tensor(
                            camera_config.camera.random.pos.max
                        )
                        random_pos = torch.from_numpy(
                            np.random.uniform(random_pos_min, random_pos_max)
                        )
                    if (
                        hasattr(camera_config.camera.random, "ori")
                        and camera_config.camera.random.ori is not None
                    ):
                        random_ori_min = torch.tensor(
                            camera_config.camera.random.ori.min
                        )
                        random_ori_max = torch.tensor(
                            camera_config.camera.random.ori.max
                        )
                        random_ori = torch.from_numpy(
                            np.random.uniform(random_ori_min, random_ori_max)
                        )
                if (
                    hasattr(camera_config.camera, "ori")
                    and camera_config.camera.ori is not None
                ):
                    if torch.tensor(camera_config.camera.ori).shape[0] == 3:
                        orientation = euler_angles_to_quat(
                            torch.tensor(camera_config.camera.ori) + random_ori,
                            degrees=True,
                        )
                    else:
                        orientation = torch.tensor(camera_config.camera.ori)
                else:
                    orientation = euler_angles_to_quat(torch.tensor([0, 0, 0]))
                if (
                    hasattr(camera_config.camera, "pos")
                    and camera_config.camera.pos is not None
                ):
                    position = torch.tensor(camera_config.camera.pos)
                else:
                    position = torch.tensor([0, 0, 1])
                final_position = position + random_pos
                cur_camera_xform.set_local_pose(
                    final_position,
                    orientation,
                )
                self.camera_poses[env_id].append(cur_camera_xform.get_local_pose())
                cam_id += 1

        return

    def reset_idx(self, env_ids: Sequence[int] = None):
        """
        Reset specific environments.
        We just reset camera pose for the specified environments.
        This method should be called when resetting specific environments.
        """
        if env_ids is None:
            env_id_list = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_id_list = [int(idx) for idx in env_ids.detach().cpu().tolist()]
        else:
            env_id_list = [int(idx) for idx in env_ids]

        # Reset Camera Pose
        # Use use_randomization=False to ensure camera returns to exact base position from YAML
        self.init_camera_pose(env_ids=env_id_list, use_randomization=False)

    def reset(self):
        """
        Reset the camera manager for specific environments.
        This method should only be called when resetting the environment.
        """
        # Implementation for resetting camera configurations
        self.post_init()
        env_id_list = list(range(self.num_envs))
        self.init_camera_pose(env_ids=env_id_list)
        self.sim.sim_step()

    def get_camera_state(self, camera_name: str = None, env_ids: Sequence[int] = None):
        """
        Get current camera state (position and quaternion)

        Args:
            camera_name: Camera name, if None returns the first camera
            env_ids: List of environment IDs, if None returns all environments

        Returns:
            dict: {
                "pos": [N, 3] position
                "quat": [N, 4] quaternion (w,x,y,z)
            }
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        else:
            env_ids = [int(e) for e in env_ids]

        # Determine camera index
        if camera_name is None:
            cam_id = 0
        else:
            cam_id = list(self.camera_config.keys()).index(camera_name)

        # Collect camera poses for all environments
        pos_list = []
        quat_list = []

        for env_id in env_ids:
            camera_xform = self.cameras_xform[env_id][cam_id]
            pos, quat = camera_xform.get_local_pose()
            pos_list.append(pos.cpu())
            quat_list.append(quat.cpu())

        pos_tensor = torch.stack(pos_list, dim=0).to(self.device)  # [N, 3]
        quat_tensor = torch.stack(quat_list, dim=0).to(self.device)  # [N, 4]

        return {"pos": pos_tensor, "quat": quat_tensor}

    def get_all_camera_state(self, env_ids: Sequence[int] = None):
        """
        Get current state (position and quaternion) for all cameras

        Args:
            env_ids: List of environment IDs, if None returns all environments

        Returns:
            dict: {
                cam_id: [N, 7] pose tensor [x, y, z, w, x, y, z] (pos + quat concatenated),
                ...
            }
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        else:
            env_ids = [int(e) for e in env_ids]

        result = {}

        # Iterate through all cameras
        for cam_id in range(self.num_cams):
            pos_list = []
            quat_list = []

            for env_id in env_ids:
                camera_xform = self.cameras_xform[env_id][cam_id]
                pos, quat = camera_xform.get_local_pose()
                pos_list.append(pos.cpu())
                quat_list.append(quat.cpu())

            pos_tensor = torch.stack(pos_list, dim=0).to(self.device)  # [N, 3]
            quat_tensor = torch.stack(quat_list, dim=0).to(self.device)  # [N, 4]

            # Concatenate pos and quat: [N, 3] + [N, 4] -> [N, 7]
            pose_tensor = torch.cat([pos_tensor, quat_tensor], dim=1)  # [N, 7]

            result[cam_id] = pose_tensor

        return result

    def set_camera_pose(
        self, camera_name: str, poses: torch.Tensor, env_ids: Sequence[int] = None
    ):
        """
        Set camera pose

        Args:
            camera_name: Camera name
            poses: [N, 7] pose [x,y,z,w,x,y,z]
            env_ids: List of environment IDs
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        else:
            env_ids = [int(e) for e in env_ids]

        # Determine camera index
        cam_id = list(self.camera_config.keys()).index(camera_name)

        # Separate position and quaternion
        pos = poses[:, :3]  # [N, 3]
        quat = poses[:, 3:]  # [N, 4]

        # Apply to each environment
        for i, env_id in enumerate(env_ids):
            camera_xform = self.cameras_xform[env_id][cam_id]
            camera_xform.set_local_pose(pos[i], quat[i])
