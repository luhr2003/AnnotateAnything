"""
This is the capture manager in magicsim.
We will import camera, initialize camera, initialize replicator writer, record video and write naive flying camera trajectory in this file.
"""

from typing import List
from collections.abc import Sequence
from magicsim.Env.Environment.Isaac.IsaacRLEnv import IsaacRLEnv
import torch
from omni.replicator.core.scripts.annotators import Annotator
from magicsim.Env.Sensor.CameraManager import CameraManager
from omegaconf import DictConfig, OmegaConf
from isaacsim.sensors.camera import Camera
from magicsim.Env.Capture.camera_view import CameraView

TILED_AVAILABLE_ANNOTATORS = [
    "rgb",
    "rgba",
    "depth",
    "distance_to_image_plane",
    "distance_to_camera",
    "normals",
    "motion_vectors",
    "semantic_segmentation",
    "instance_segmentation_fast",
    "instance_id_segmentation_fast",
]


class TiledCaptureManager:
    """
    Main Class for managing the capture in MagicSim environment.
    This class is responsible for initializing cameras, replicator writers, and handling video recording.
    """

    def __init__(
        self,
        num_envs: int,
        config: DictConfig,
        camera_manager: CameraManager,
        device: torch.device,
    ):
        print("-----------------Tiled Capture Initialized--------------")
        self.sim: IsaacRLEnv = None
        self.config = config
        self.camera_manager = camera_manager
        self.num_envs = num_envs
        self.device = device
        self.tiled_render_products: List[str] = []  # all the render products
        self.replicator_writer = None  # Placeholder for replicator writer object
        self.annotator: List[
            List[List[Annotator]]
        ] = []  # Defines the types and devices of annotator. See yaml for camera for more detail
        self.annotator_type: List[List[List[str]]] = []
        self.annotator_device: List[List[List[str]]] = []
        self.cameras: List[List[Camera]] = camera_manager.cameras
        self.tiled_cameras: List[CameraView] = []
        self.camera_prim_paths: List[List[str]] = []
        # Pre-allocated output buffers for each camera and annotator to reduce memory allocation
        # Format: {cam_id: {annotator_name: wp.array}}
        self._output_buffers: dict = {}

    def initialize(self, sim: IsaacRLEnv):
        """
        Initialize the capture manager.
        This method should be called before the simulation context is created.
        """
        self.num_cams = len(self.cameras[0])
        self.sim = sim

    def init_cameras(self):
        """
        Initialize cameras based on the configuration.
        1. First Initialize the cameras using our magicsim camera
        2. Get all render_product control
        3. Attach Annotator
        """

        for env_id in range(self.num_envs):
            self.camera_prim_paths.append([])
            for camera in self.cameras[env_id]:
                self.camera_prim_paths[env_id].append(camera.prim_path)

        for cam_id in range(self.num_cams):
            self.annotator.append([])
            self.annotator_type.append([])
            self.annotator_device.append([])

        config = OmegaConf.to_container(self.config, resolve=True)
        if "enable_tiled" in config:
            config.pop("enable_tiled")
        if "colorize_depth" in config:
            self.colorize_depth = config["colorize_depth"]
            config.pop("colorize_depth")

        self.config = OmegaConf.create(config)

        for i, (capture_name, capture_config) in enumerate(self.config.items()):
            if capture_config.annotator.enabled:
                annotator_config = OmegaConf.to_container(
                    capture_config.annotator, resolve=True
                )
                annotator_config.pop("enabled")
                for annotator_name, annotator_setting in annotator_config.items():
                    device = annotator_setting.get("device", "cpu")
                    type_annotator = annotator_setting["type"]
                    self.annotator_type[cam_id].append(type_annotator)

            prim_paths_by_cam_id = [row[cam_id] for row in self.camera_prim_paths]
            tiled_camera = CameraView(
                prim_paths_by_cam_id,
                camera_resolution=self.cameras[0][cam_id]._resolution,
                output_annotators=self.annotator_type[cam_id],
            )
            self.tiled_cameras.append(tiled_camera)
            self.tiled_render_products.append(tiled_camera._render_product)

            # Pre-allocate output buffers for this camera's annotators
            self._output_buffers[cam_id] = {}
            camera_resolution = self.cameras[0][cam_id]._resolution
            height, width = camera_resolution[1], camera_resolution[0]

            for annotator_name in self.annotator_type[cam_id]:
                # Get annotator spec to determine shape and dtype
                from magicsim.Env.Capture.camera_view import ANNOTATOR_SPEC

                spec = ANNOTATOR_SPEC.get(annotator_name)
                if spec is None:
                    continue

                channels = spec["channels"]
                # rgb uses rgba (4 channels), not 3
                output_channels = (
                    channels  # Use channels directly (rgb has 4 channels from rgba)
                )
                shape = (self.num_envs, height, width, output_channels)

                # Pre-allocate warp array on CUDA to reuse memory
                import warp as wp

                self._output_buffers[cam_id][annotator_name] = wp.zeros(
                    shape, dtype=spec["dtype"], device="cuda:0"
                )
                print(
                    f"[TiledCaptureManager] Pre-allocated buffer for cam_{cam_id}, {annotator_name}: shape={shape}, dtype={spec['dtype']}"
                )

    def step(
        self, env_ids: List[int] = None, cam_ids: List[int] = None
    ) -> List[List[List[any]]]:
        """
        Step the annotator. When env_id and cam_id is given, use the given. Otherwise apply to all cameras.
        Args:
            env_ids: List[int] - list of environment IDs to process
            cam_ids: List[int] - list of camera IDs to process
        Returns:
            List[List[List[Any]]] : returns the required data from annotators for required env_ids and cam_ids
            Format: data[cam_id][annotator_name] = [env_0_data, env_1_data, ...]
            where each env_i_data is {data: numpy_array, info: dict}
        """

        if env_ids is None:
            env_ids = list(range(self.num_envs))
        if cam_ids is None:
            cam_ids = list(range(len(self.cameras[0])))

        data = []
        for cam_id in cam_ids:
            cam_data = {}
            # Get annotator names/types for this camera
            annotator_names = self.annotator_type[cam_id]
            for annotator_name in annotator_names:
                # Get pre-allocated output buffer for this camera and annotator
                pre_allocated_out = None
                if (
                    cam_id in self._output_buffers
                    and annotator_name in self._output_buffers[cam_id]
                ):
                    pre_allocated_out = self._output_buffers[cam_id][annotator_name]

                # Get batched data from tiled camera using pre-allocated buffer
                # out shape: (num_envs, height, width, channels)
                out, info = self.tiled_cameras[cam_id].get_data(
                    annotator_name, out=pre_allocated_out
                )

                # Convert out to numpy if it's a warp array (only convert once, reuse buffer)
                if hasattr(out, "numpy"):
                    out_np = out.numpy()
                elif hasattr(out, "cpu"):
                    out_np = out.cpu().numpy()
                else:
                    out_np = out

                # Build env_list: [env_0_data, env_1_data, ...]
                # where env_i_data = {data: out[env_id], info: info}
                env_list = []
                for env_id in env_ids:
                    # Extract data for this specific env_id
                    # out[env_id] shape: (height, width, channels)
                    env_data = out_np[env_id]
                    env_list.append({"data": env_data, "info": info})

                cam_data[annotator_name] = env_list
            data.append(cam_data)
        return data

    def reset(
        self,
    ):
        """
        Soft Reset do not need to reset the replicator writer and camera.
        Only Hard Reset need which means if we reset simulation backend we need to initialize camera again
        Since Render product change, we also need to attch a new writer maybe
        """
        self.init_cameras()

    def reset_idx(self, env_ids: Sequence[int] = None, output_dirs: List[str] = None):
        """
        Reset the capture manager for specific environment indices.
        This function will be called when we reset the environment.

        Args:
            env_ids: The indices of the environments to reset.
        """
        pass

    def destroy(self):
        """
        Destroy the capture manager.
        This function will be called when we close the environment.
        """

        self.annotator.clear()
        self.annotator_type.clear()
        self.annotator_device.clear()
        self.annotator_name.clear()
        self.tiled_cameras.clear()
        for rp in self.tiled_render_products:
            rp.destroy()
        self.camera_prim_paths.clear()
