"""
This file is the BaseEnv of magicsim. It will launch an empty isaaclab simulation environment.
In this file, we will set physics and render setting for the simulation backbone by setting isaacRLCfg.
"""

from typing import Any, Sequence
from collections.abc import Sequence as SequenceABC

import torch
from magicsim.Launch.MagicLauncher import MagicLauncher

# from isaaclab.app import AppLauncher
from omegaconf import DictConfig
import argparse
import gymnasium as gym
from magicsim.Env.Utils.file import Logger

parser = argparse.ArgumentParser()
MagicLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = MagicLauncher(args)
simulation_app = app_launcher.app


from magicsim.Env.Environment.Utils.Basic import seed_everywhere
from isaaclab_tasks.utils import parse_env_cfg  # this need dynamic import
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from magicsim.Env.Environment.Isaac.IsaacRLEnv import IsaacRLEnv
from omni.physx import acquire_physx_interface
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import get_current_stage

NVIDIA_ASSETS = get_assets_root_path()


class BaseEnv(gym.Env):
    def __init__(
        self, config: DictConfig, cli_args, logger: Logger, app=simulation_app
    ):
        """
        Args:
            config (DictConfig): The configuration for the environment. This comes from hydra yaml.
            cli_arg: Command line arguments. This comes from the tyro.
        """
        self.app = app
        self.config = config
        self.cli_args = cli_args
        self.logger = logger
        self.device = self.config.device
        self.use_fabric = self.config.use_fabric
        self._seed_list: list[int] | None = None
        self._global_seed: int | None = None
        self.env_seeds: list[int] | None = None
        self._configure_seed_config(self.config.seed)
        if self.device != "cpu" and not self.use_fabric:
            self.logger.warning("use cuda device must launch use_fabric=True")
            self.use_fabric = True
        self.sim = None
        self.stage = get_current_stage()

    def _enable_flow_rendering(self):
        """Enable Flow rendering for fire/smoke/fluid effects using carb settings."""
        import carb.settings

        settings = carb.settings.get_settings()
        settings.set("/rtx/flow/enabled", True)
        settings.set("/rtx/flow/pathTracingEnabled", True)
        settings.set("/rtx/flow/rayTracedReflectionsEnabled", True)
        settings.set("/rtx/flow/rayTracedTranslucencyEnabled", True)

        print("[INFO] Flow rendering enabled for fire/smoke/fluid effects")
        print("       - /rtx/flow/enabled: True")
        print("       - /rtx/flow/pathTracingEnabled: True")
        print("       - /rtx/flow/rayTracedReflectionsEnabled: True")
        print("       - /rtx/flow/rayTracedTranslucencyEnabled: True")

    def launch_sim(self, config):
        """
        This function will launch the isaaclab simulation with the given configuration.

        Args:
            config (DictConfig): The configuration for the environment. This comes from hydra yaml.
        """
        self.sim_cfg: DirectRLEnvCfg = parse_env_cfg(
            "IsaacRLEnv-V0", device=self.device, use_fabric=self.use_fabric
        )

        # Isaaclab setting
        self.sim_cfg.decimation = config.decimation
        self.sim_cfg.sim.dt = config.get("dt", 1 / 60)
        self.sim_cfg.seed = (
            self._global_seed if self._global_seed is not None else config.seed
        )

        # Isaaclab scene setting
        self.sim_cfg.scene = InteractiveSceneCfg(
            num_envs=config.scene.num_envs,
            env_spacing=config.scene.env_spacing,
            replicate_physics=False,
        )

        self.setup_sim_cfg(config)

        self.sim: IsaacRLEnv = gym.make(
            "IsaacRLEnv-V0",
            cfg=self.sim_cfg,
            render_mode="rgb_array",
            func={
                "_setup_scene": self._setup_scene,
                "_reset_idx": self._reset_idx,
                "_apply_action": self._apply_action,
                "_pre_physics_step": self._pre_physics_step,
                "_post_setup_scene": self._post_setup_scene,
            },
        )
        self.env_spacing = config.scene.env_spacing
        self.sim.env_spacing = self.env_spacing
        self.env_origins = self.sim.scene.env_origins

        # Apply Flow render settings directly after simulation is created
        self._enable_flow_rendering()

    def setup_sim_cfg(self, config):
        """
        Setup the simulation configuration.
        This function will be called before launching the simulation.
        You can modify the sim_cfg here.

        Args:
            config (DictConfig): The configuration for the environment. This comes from hydra yaml.
        """
        physx_config = self.config.get("physx", {}) or {}
        scene_config = self.config.get("render", {}) or {}

        self.sim_cfg.sim.render.enable_translucency = scene_config.get(
            "enable_translucency", True
        )
        self.sim_cfg.sim.render.enable_reflections = scene_config.get(
            "enable_reflections", True
        )
        self.sim_cfg.sim.render.enable_global_illumination = scene_config.get(
            "enable_global_illumination", True
        )
        self.sim_cfg.sim.render.antialiasing_mode = scene_config.get(
            "antialiasing_mode", "TAA"
        )
        self.sim_cfg.sim.render.enable_dlssg = scene_config.get("enable_dlssg", True)
        self.sim_cfg.sim.render.enable_dl_denoiser = scene_config.get(
            "enable_dl_denoiser", True
        )
        self.sim_cfg.sim.render.dlss_mode = scene_config.get("dlss_mode", "Balanced")
        self.sim_cfg.sim.render.enable_direct_lighting = scene_config.get(
            "enable_direct_lighting", True
        )
        self.sim_cfg.sim.render.samples_per_pixel = scene_config.get(
            "samples_per_pixel", 4
        )
        self.sim_cfg.sim.render.enable_shadows = scene_config.get(
            "enable_shadows", True
        )
        self.sim_cfg.sim.render.enable_ambient_occlusion = scene_config.get(
            "enable_ambient_occlusion", True
        )
        self.sim_cfg.sim.render.rendering_mode = scene_config.get(
            "rendering_mode", "quality"
        )

        # Get carb_settings and add Flow rendering settings for fire simulation
        carb_settings = scene_config.get("carb_settings", {})
        if carb_settings is None:
            carb_settings = {}

        # Enable Flow rendering for fire/smoke/fluid effects
        flow_settings = {
            "/rtx/flow/enabled": True,
            "/rtx/flow/pathTracingEnabled": True,
            "/rtx/flow/rayTracedReflectionsEnabled": True,
            "/rtx/flow/rayTracedTranslucencyEnabled": True,
        }
        carb_settings.update(flow_settings)

        self.sim_cfg.sim.render.carb_settings = carb_settings
        self.sim_cfg.sim.physx.solver_type = physx_config.get("solver_type", 1)
        self.sim_cfg.sim.physx.min_position_iteration_count = physx_config.get(
            "min_position_iteration_count", 1
        )
        self.sim_cfg.sim.physx.max_position_iteration_count = physx_config.get(
            "max_position_iteration_count", 255
        )
        self.sim_cfg.sim.physx.min_velocity_iteration_count = physx_config.get(
            "min_velocity_iteration_count", 0
        )
        self.sim_cfg.sim.physx.max_velocity_iteration_count = physx_config.get(
            "max_velocity_iteration_count", 255
        )
        self.sim_cfg.sim.physx.enable_ccd = physx_config.get("enable_ccd", False)
        self.sim_cfg.sim.physx.enable_stabilization = physx_config.get(
            "enable_stabilization", False
        )
        self.sim_cfg.sim.physx.enable_enhanced_determinism = physx_config.get(
            "enable_enhanced_determinism", False
        )
        self.sim_cfg.sim.physx.bounce_threshold_velocity = physx_config.get(
            "bounce_threshold_velocity", 0.5
        )
        self.sim_cfg.sim.physx.friction_offset_threshold = physx_config.get(
            "friction_offset_threshold", 0.04
        )
        self.sim_cfg.sim.physx.friction_correlation_distance = physx_config.get(
            "friction_correlation_distance", 0.025
        )
        self.sim_cfg.sim.physx.gpu_max_rigid_contact_count = physx_config.get(
            "gpu_max_rigid_contact_count", 8388608
        )
        self.sim_cfg.sim.physx.gpu_max_rigid_patch_count = physx_config.get(
            "gpu_max_rigid_patch_count", 163840
        )
        self.sim_cfg.sim.physx.gpu_found_lost_pairs_capacity = physx_config.get(
            "gpu_found_lost_pairs_capacity", 2097152
        )
        self.sim_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = (
            physx_config.get("gpu_found_lost_aggregate_pairs_capacity", 33554432)
        )
        self.sim_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = physx_config.get(
            "gpu_total_aggregate_pairs_capacity", 2097152
        )
        self.sim_cfg.sim.physx.gpu_collision_stack_size = physx_config.get(
            "gpu_collision_stack_size", 100000000
        )
        self.sim_cfg.sim.physx.gpu_heap_capacity = physx_config.get(
            "gpu_heap_capacity", 67108864
        )
        self.sim_cfg.sim.physx.gpu_temp_buffer_capacity = physx_config.get(
            "gpu_temp_buffer_capacity", 16777216
        )
        self.sim_cfg.sim.physx.gpu_max_num_partitions = physx_config.get(
            "gpu_max_num_partitions", 8
        )
        self.sim_cfg.sim.physx.gpu_max_soft_body_contacts = physx_config.get(
            "gpu_max_soft_body_contacts", 1048576
        )
        self.sim_cfg.sim.physx.gpu_max_particle_contacts = physx_config.get(
            "gpu_max_particle_contacts", 1048576
        )

    def step(self):
        """
        sim backend step
        """
        self.sim.sim_step()

    def setup_physics(self, sim: IsaacRLEnv):
        """
        Setup the physics for the simulation.
        This function will be called after simulation context is created
        """
        # enable cpu garment and deformable
        self.physics_interface = acquire_physx_interface()
        self.physics_interface.overwrite_gpu_setting(1)

        # expose physics context
        self.physics_context = sim.sim.get_physics_context()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        Reset All Environment.
        isaaclab soft reset function. This will not reset the simulation backend.
        """
        global_seed, _ = self._process_seed_input(seed)
        if global_seed is not None:
            seed_everywhere(global_seed)  # Set seed for all envs
        if self.sim is None:
            self.launch_sim(self.config)
        self.sim.reset(options=options)

    def reset_idx(
        self,
        env_ids: Sequence[int] = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """
        BaseEnv reset_idx function, this will call directrlenv reset_idx function.
        Currently this function only used to reset all lab managed assets
        """
        if env_ids is None:
            env_id_list = list(range(self.sim.num_envs))
        else:
            env_id_list = [int(env_id) for env_id in env_ids]
        env_ids_tensor = torch.tensor(
            env_id_list, device=self.sim.device, dtype=torch.int32
        )

        global_seed, env_seed_list = self._process_seed_input(seed, env_id_list)
        if seed is not None:
            if self._is_sequence_seed(seed):
                if env_seed_list and env_id_list:
                    first_env = env_id_list[0]
                    first_seed = env_seed_list[first_env]
                    seed_everywhere(first_seed)
            elif global_seed is not None:
                seed_everywhere(global_seed)

        self.sim._reset_idx(env_ids_tensor)

    def get_dones(self, env_ids: Sequence[int] = None):
        """
        Get dones for the environment.
        This function will return the dones for the environment.
        """
        raise NotImplementedError(
            "BaseEnv does not implement get_dones function. Please implement it in your own environment."
        )

    def get_observations(self, env_ids: Sequence[int] = None):
        """
        Get observations for the environment.
        This function will return the observations for the environment.
        """
        raise NotImplementedError(
            "BaseEnv does not implement get_observations function. Please implement it in your own environment."
        )

    def _setup_scene(self, sim: IsaacRLEnv):
        """
        This is the entry point for spawn all isaaclab object into the scene.
        Here we use isaaclab to spawn ground plane. In Robotbase env we spawn a robot arm and a desk
        You should add all object you want isaaclab to spawn in this function.
        This function will be called by the DirectRLEnv.__init__() function.
        """
        self.setup_physics(sim)

    def _reset_idx(self, env, env_ids=None):
        """
        This is soft reset for robotics arm interface and other isaaclab related things.
        This function will not call simulation backend reset. It is only a soft reset entry point for robotics arm.
        This function is called by our own reset in robot base env. The collectbase env will never call this function.
        """
        pass  # Baseenv do not need reset robot and will rewrite in robot base env

    def _apply_action(self, env):
        """
        This is apply actions function for robotics arm and is called by directrlenv.step(action).
        This serve as the simulation backbone function to control the robot arm.

        ! Important Function !: This function will be called by the step function in the robot base env.
        """
        pass  # baseenv do not need apply action for robot arm and will rewrite in robot base env

    def _pre_physics_step(self, env, actions):
        """
        This is pre physics step function for robotics arm interface and is called by directrlenv.step(action).
        This serve as the simulation backbone function to control the robot arm.
        ! Important Function !: This function will be called by the step function in the robot base env.
        It will not be called by the collectbase env.
        """
        pass  # baseenv do not need pre physics step for robot arm and will rewrite in robot base env

    def _post_setup_scene(self, sim: IsaacRLEnv):
        """
        This function will be called after official cloner work but before simulation start.
        You can add additional objects to the scene here.
        """
        pass

    def close(self):
        """
        Close the environment and release resources.
        This function will be called when the environment is closed.
        """
        if self.sim is not None:
            self.sim.close()
            self.sim = None
        simulation_app.close()
        self.logger.info("BaseEnv closed.")

    def _configure_seed_config(self, seed_cfg: Any):
        if seed_cfg is None:
            self._global_seed = None
            self._seed_list = None
            self.env_seeds = None
            return
        if self._is_sequence_seed(seed_cfg):
            seeds = [int(s) for s in seed_cfg]
            self._set_env_seeds_full(seeds)
        else:
            self._set_global_seed(int(seed_cfg))

    def get_env_seed(self, env_id: int) -> int | None:
        if self._seed_list is not None:
            if env_id >= len(self._seed_list):
                raise IndexError(
                    f"Requested env_id {env_id} exceeds configured seeds (len={len(self._seed_list)})."
                )
            return self._seed_list[env_id]
        return self._global_seed

    # ------------------------------------------------------------------
    # Seed management helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_sequence_seed(seed: Any) -> bool:
        return isinstance(seed, SequenceABC) and not isinstance(seed, (str, bytes))

    def _ensure_env_seed_storage(self):
        if self.env_seeds is None:
            num_envs = int(self.config.scene.num_envs)
            default_seed = self._global_seed if self._global_seed is not None else 0
            self.env_seeds = [default_seed for _ in range(num_envs)]
            self._seed_list = list(self.env_seeds)

    def _set_global_seed(self, seed_value: int):
        self._global_seed = int(seed_value)
        if self.env_seeds is not None:
            self.env_seeds = [self._global_seed for _ in self.env_seeds]
            self._seed_list = list(self.env_seeds)
        else:
            self._seed_list = None

    def _set_env_seeds_full(self, seeds: Sequence[int]):
        num_envs = int(self.config.scene.num_envs)
        seed_list = [int(s) for s in seeds]
        if len(seed_list) != num_envs:
            raise ValueError(
                f"Seed list length {len(seed_list)} does not match num_envs {num_envs}."
            )
        self.env_seeds = list(seed_list)
        self._seed_list = list(seed_list)
        self._global_seed = seed_list[0] if seed_list else None

    def _set_env_seeds_subset(
        self, env_ids: Sequence[int], seeds: Sequence[int]
    ) -> list[int]:
        env_id_list = [int(env_id) for env_id in env_ids]
        seed_values = [int(s) for s in seeds]
        if len(env_id_list) != len(seed_values):
            raise ValueError(
                f"Seed list length {len(seed_values)} does not match env_ids length {len(env_id_list)}."
            )
        self._ensure_env_seed_storage()
        num_envs = len(self.env_seeds)
        for env_id, seed_val in zip(env_id_list, seed_values, strict=True):
            if env_id < 0 or env_id >= num_envs:
                raise IndexError(
                    f"Requested env_id {env_id} exceeds configured num_envs {num_envs}."
                )
            self.env_seeds[env_id] = seed_val
        self._seed_list = list(self.env_seeds)
        if self.env_seeds:
            first_seed = self.env_seeds[0]
            if first_seed is not None:
                self._global_seed = first_seed
        return list(self.env_seeds)

    def _process_seed_input(
        self, seed: Any, env_ids: Sequence[int] | None = None
    ) -> tuple[int | None, list[int] | None]:
        if seed is None:
            return None, self.env_seeds if self.env_seeds is not None else None

        if self._is_sequence_seed(seed):
            if env_ids is None:
                self._set_env_seeds_full(seed)
            else:
                self._set_env_seeds_subset(env_ids, seed)
            return (
                self._global_seed,
                list(self.env_seeds) if self.env_seeds is not None else None,
            )

        self._set_global_seed(int(seed))
        return self._global_seed, None
