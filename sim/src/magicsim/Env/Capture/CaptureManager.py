"""
This is the capture manager in magicsim.
We will import camera, initialize camera, initialize replicator writer, record video and write naive flying camera trajectory in this file.
"""

from typing import List, Dict
from collections.abc import Sequence
import torch
import omni.replicator.core as rep
from omni.replicator.core.scripts.annotators import Annotator
from magicsim.Env.Sensor.CameraManager import CameraManager
from omegaconf import DictConfig, OmegaConf
from isaacsim.sensors.camera import Camera
from omni.syntheticdata.scripts.SyntheticData import SyntheticData
from magicsim.Env.Environment.Isaac.IsaacRLEnv import IsaacRLEnv


class CaptureManager:
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
        self.sim: IsaacRLEnv = None
        self.config = config
        self.camera_manager = camera_manager
        self.num_envs = num_envs
        self.device = device
        self.render_products: List[List[str]] = []  # all the render products
        self.annotator: List[
            List[List[Annotator]]
        ] = []  # Defines the types and devices of annotator. See yaml for camera for more detail
        self.annotator_type: List[List[List[str]]] = []
        self.annotator_device: List[List[List[str]]] = []
        self.annotator_name: List[List[List[str]]] = []
        self.cameras: List[List[Camera]] = camera_manager.cameras

    def initialize(self, sim: IsaacRLEnv):
        """
        Initialize the capture manager.
        This method should be called before the simulation context is created.
        It will create cameras, render products, and initialize annotators.
        """
        self.sim = sim
        self.num_cams = len(self.cameras[0])

    def init_cameras(self):
        """
        Initialize cameras based on the configuration.
        1. First Initialize the cameras using our magicsim camera
        2. Get all render_product control
        3. Attach Annotator
        """
        self.physics_sim_view = self.sim.sim.physics_sim_view
        for env_id in range(self.num_envs):
            self.render_products.append([])
            for cur_camera in self.cameras[env_id]:
                cur_camera.initialize(self.physics_sim_view, attach_rgb_annotator=False)
                self.render_products[env_id].append(cur_camera._render_product)
        cam_id = 0
        for env_id in range(self.num_envs):
            self.annotator.append([])
            self.annotator_type.append([])
            self.annotator_device.append([])
            self.annotator_name.append([])
            for cam_id in range(self.num_cams):
                self.annotator[env_id].append([])
                self.annotator_type[env_id].append([])
                self.annotator_device[env_id].append([])
                self.annotator_name[env_id].append([])
        config = OmegaConf.to_container(self.config, resolve=True)

        # Strip global / non-camera keys from camera config to avoid treating
        # strings/ints/bools as per-camera capture configs.
        if "colorize_depth" in config:
            self.colorize_depth = config["colorize_depth"]
            config.pop("colorize_depth")
        if "enable_tiled" in config:
            config.pop("enable_tiled")

        self.config = OmegaConf.create(config)

        cam_id = 0
        for capture_name, capture_config in self.config.items():
            if capture_config.annotator.enabled:
                annotator_config = OmegaConf.to_container(
                    capture_config.annotator, resolve=True
                )
                annotator_config.pop("enabled")
                for annotator_name, annotator_setting in annotator_config.items():
                    device = annotator_setting.get("device", "cpu")
                    type_annotator = annotator_setting["type"]
                    for env_id in range(self.num_envs):
                        if (
                            type_annotator == "semantic_segmentation"
                            or type_annotator == "instance_id_segmentation"
                            or type_annotator == "instance_segmentation"
                        ):
                            semantic_types = annotator_setting.get(
                                "semantic_types", None
                            )
                            semantic_filter_predicate = annotator_setting.get(
                                "semantic_filter_predicate", None
                            )
                            if semantic_types == "None":
                                semantic_types = None
                            if semantic_filter_predicate == "None":
                                semantic_filter_predicate = None
                            if semantic_types is not None:
                                if semantic_filter_predicate is None:
                                    semantic_filter_predicate = (
                                        ":*; ".join(semantic_types) + ":*"
                                    )
                                else:
                                    raise ValueError(
                                        "`semantic_types` and `semantic_filter_predicate` are mutually exclusive. Please choose only one."
                                    )
                            elif semantic_filter_predicate is None:
                                semantic_filter_predicate = "class:*"

                            # Set the global semantic filter predicate
                            if semantic_filter_predicate is not None:
                                SyntheticData.Get().set_instance_mapping_semantic_filter(
                                    semantic_filter_predicate
                                )
                            self.annotator[env_id][cam_id].append(
                                rep.AnnotatorRegistry.get_annotator(
                                    type_annotator,
                                    device=device,
                                    init_params={
                                        "colorize": annotator_setting.get(
                                            "colorize", True
                                        )
                                    },
                                )
                            )
                        elif type_annotator == "pointcloud":
                            self.annotator[env_id][cam_id].append(
                                rep.AnnotatorRegistry.get_annotator(
                                    type_annotator,
                                    device=device,
                                    init_params={
                                        "includeUnlabelled": annotator_setting.get(
                                            "includeUnlabelled", False
                                        )
                                    },
                                )
                            )
                        elif type_annotator == "skeleton_data":
                            self.annotator[env_id][cam_id].append(
                                rep.AnnotatorRegistry.get_annotator(
                                    type_annotator,
                                    device=device,
                                    init_params={
                                        "useSkelJoints": annotator_setting.get(
                                            "useSkelJoints", False
                                        )
                                    },
                                )
                            )
                        else:
                            self.annotator[env_id][cam_id].append(
                                rep.AnnotatorRegistry.get_annotator(
                                    type_annotator, device=device
                                )
                            )
                        self.annotator_type[env_id][cam_id].append(type_annotator)
                        self.annotator_device[env_id][cam_id].append(device)
                        self.annotator_name[env_id][cam_id].append(annotator_name)
            else:
                pass

            cam_id += 1
        return

    def init_replicator_annotator(self):
        """
        Initialize the annotators and attach them to render products
        """
        rep.orchestrator.set_capture_on_play(False)
        for env_id in range(self.num_envs):
            for cam_id in range(len(self.render_products[0])):
                for annotator in self.annotator[env_id][cam_id]:
                    annotator.attach(self.render_products[env_id][cam_id])
        return

    def step(
        self,
        env_ids: List[int] = None,
        cam_ids: List[int] = None,
    ) -> List[Dict[str, List[any]]]:
        """
        Step the capture pipeline for selected environments/cameras.

        Args:
            env_ids: Environment indices to capture. Defaults to all environments.
            cam_ids: Camera indices to capture. Defaults to all cameras.
            write_to_disk: Whether to run the registered replicator annotators.
            capture_info: Optional dictionary describing additional per-env data to
                write this step. Expected format:

                {
                    env_id: {
                        "trigger_outputs": {...},   # optional
                        "annotators": {
                            "joint_positions": {
                                "robot_env_<env_id>": {"data": ..., "joint_names": ...}
                            },
                            ...
                        }
                    },
                    ...
                }

                The helper ``convert_robot_data_to_writer_format`` already returns
                the above structure for a given env_id and can be passed directly.

        Returns:
            Batched data structure: List[Dict[str, List[any]]]
            - Outer list indexed by cam_id
            - Dict keys are annotator names
            - Inner list contains data for each env_id in order
            Format: data[cam_id][annotator_name][env_index] = annotator_data
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        if cam_ids is None:
            cam_ids = list(range(self.num_cams))

        data = []
        for cam_id in cam_ids:
            cam_data = {}
            # Get annotator names from first env (all envs should have same annotators)
            annotator_names = self.annotator_type[env_ids[0]][cam_id]
            for i, annotator_name in enumerate(annotator_names):
                env_list = []
                for env_id in env_ids:
                    annotator = self.annotator[env_id][cam_id][i]
                    env_list.append(annotator.get_data())
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
        self.init_replicator_annotator()
        self.sim.sim_step()

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
        self.sim = None
        for rp in self.render_products:
            for render_product in rp:
                render_product.destroy()
