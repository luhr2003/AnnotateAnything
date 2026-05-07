"""
This is a base environment class for MagicSim that serves as a foundation for sync systhetic data generation and sync low-level rl training
! Warning !: This is a synchronized env, meaning that each subenv will step and reset simultaneously.

If you are using a robotic arm for high-level actions that must be implemented with atomic actions, please use asyncaseenv.
All the action of this env should be atomic action, meaning that it will be called everytime before sim_step.

In a sync env, the user calls the sim_step function, and the simulator runs in the same process as the user program.
This is a synchronized environment, meaning that all subenv actions must be generated before the next step can be invoked.
This environment can be directly initialized and interacted with in the user's code. Therefore, this environment does not include async communication module.

"""

from typing import Any, Sequence, List
from magicsim.Env.Environment.BaseEnv import BaseEnv
from magicsim.Env.Scene.SceneManager import SceneManager
from magicsim.Env.Nav.NavManager import NavManager
import torch
from magicsim.Env.Animation.AnimationManager import AnimationManager


class SyncBaseEnv(BaseEnv):
    """
    This is a base environment class for MagicSim that serves as a foundation for sync synthetic data generation and sync action env
    It inherits from BaseEnv and provides the basic structure for the environment.
    In this class we will implement reset logic.
    We implement all scene related logics in scene manager including animation, scene understanding and scene randomization.
    """

    def __init__(self, config, cli_args, logger):
        self.sim_config = config.sim
        self.scene_config = config.scene
        self.anim_config = config.get("anim", None)
        self.nav_config = config.get("nav", None)
        self.num_envs = self.sim_config.scene.num_envs
        self.device = self.sim_config.device
        super().__init__(self.sim_config, cli_args, logger)
        self.env_seed_list = self.env_seeds
        self.num_envs = self.sim_config.scene.num_envs
        self.device = self.sim_config.device

        # Initialize SceneManager with LayoutManager reference
        self.scene_manager = SceneManager(
            self.num_envs,
            self.scene_config,
            self.device,
            self.sim_config.scene.env_spacing,
            use_fabric=self.use_fabric,
            seeds_per_env=self.env_seed_list,
            nav_enable=self.nav_config is not None,
        )

        if self.anim_config is not None:
            self.animation_manager = AnimationManager(
                self.num_envs,
                self.anim_config,
                self.device,
                self.scene_manager.layout_manager,
            )
        else:
            self.animation_manager = None

        # init reset count
        self.soft_reset_times = self.sim_config.soft_resets  # This is the number of times the environment can be soft reset before a hard reset is
        self.reset_count = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        if self.nav_config is not None:
            self.nav_manager = NavManager(
                self.num_envs,
                self.nav_config,
                self.device,
            )
        else:
            self.nav_manager = None

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        if self.nav_config is not None:
            self.nav_manager._post_setup_scene(sim)

    def _setup_scene(self, sim):
        """
        Initialize the environment.
        This function will be called before simulation context create
        !!! Please put everything that can not be  dynamicly imported here!!!
        """
        self.scene_manager.initialize(sim)
        if self.anim_config is not None:
            self.animation_manager.initialize(sim)
        if self.nav_config is not None:
            self.nav_manager.initialize(sim)
        super()._setup_scene(sim)

    def step(self):
        """
        1/ Update animation manager (if exists)
        2/ Step the simulation backend.
        """
        # Update avatars before physics step
        if hasattr(self, "animation_manager") and self.animation_manager is not None:
            self.animation_manager.on_update()

        self.sim.sim_step()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        Reset the environment.
        This should only be called once at the beginning of the environment.
        In this function, we will call scene_manager.reset(soft=False) to load all the objects managed in scene manager
        It will also reset the reset count.
        """
        super().reset(
            seed=seed, options=options
        )  # Lab Reset: Will Reset All Object Managed By Lab (Robot arm)
        self.reset_count.fill_(0)
        if self.scene_manager is not None and self.env_seeds is not None:
            self.scene_manager.update_env_seeds(self.env_seeds)
        self.scene_manager.reset(soft=False)
        if self.anim_config is not None:
            self.animation_manager.reset(
                soft=False
            )  # animation manager hard reset, meaning it will delete all animations in the scene and load new animations
        if self.nav_config is not None:
            self.nav_manager.reset(soft=False)
        # scene_manager hard reset, meaning it will delete all objects in the scene and load new objects

    def reset_idx(
        self,
        env_ids: Sequence[int] = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """
        Reset individual environments.
        Hard Soft Reset Logics Implementation Here
        """
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.sim.device)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.sim.device, dtype=torch.int32)
        else:
            env_ids_tensor = torch.tensor(
                list(env_ids), device=self.sim.device, dtype=torch.int32
            )

        seed_is_sequence = self._is_sequence_seed(seed)

        super().reset_idx(
            env_ids=env_ids_tensor, seed=seed, options=options
        )  # Lab Reset Will Reset All Object Managered By Lab(Robot arm)

        if self.scene_manager is not None and self.env_seeds is not None:
            self.scene_manager.update_env_seeds(self.env_seeds)

        self.reset_count[env_ids_tensor] += 1

        force_hard_mask = torch.zeros_like(env_ids_tensor, dtype=torch.bool)
        if seed_is_sequence:
            force_hard_mask[:] = True
        # 调用方可通过 options.force_hard_reset=True 强制本次按 env 做完整 hard reset（与首次 reset() 一致）
        if options and options.get("force_hard_reset", False):
            force_hard_mask[:] = True

        hard_reset_mask = (
            self.reset_count[env_ids_tensor] > self.soft_reset_times
        ) | force_hard_mask
        if hard_reset_mask.any():
            # If any environment has reached the soft reset limit, perform a hard reset
            hard_reset_ids = env_ids_tensor[hard_reset_mask]
            self.reset_count[hard_reset_ids] = (
                0  # Reset the count for these environments
            )
            self.scene_manager.reset_idx(
                env_ids=hard_reset_ids, soft=False
            )  # Soft reset the scene manager for the specified environments
            if self.anim_config is not None:
                self.animation_manager.reset_idx(
                    env_ids=hard_reset_ids, soft=False
                )  # Hard reset the animation manager for the specified environments
        soft_reset_mask = ~hard_reset_mask
        if soft_reset_mask.any():
            # If any environment is still within the soft reset limit, perform a soft reset
            soft_reset_ids = env_ids_tensor[soft_reset_mask]
            self.scene_manager.reset_idx(
                env_ids=soft_reset_ids, soft=True
            )  # Hard reset the scene manager for the specified environments
            if self.anim_config is not None:
                self.animation_manager.reset_idx(
                    env_ids=soft_reset_ids, soft=True
                )  # Soft reset the animation manager for the specified environments

    def get(
        self,
        env_ids: List[int] = None,
        object_name: str = None,
        object_id: List[int] = None,
    ):
        return self.scene_manager.get(
            env_ids=env_ids, object_name=object_name, object_id=object_id
        )
