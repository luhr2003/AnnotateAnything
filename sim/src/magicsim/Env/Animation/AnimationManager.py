from typing import List, Sequence, Dict, Optional
from magicsim.Env.Animation.Avatar_Utils import CharacterUtil
import torch
import numpy as np
import os
from magicsim.Env.Utils.path import resolve_path
from omegaconf import DictConfig
from isaacsim.core.utils.stage import get_current_stage
from magicsim.Env.Environment.Isaac.IsaacRLEnv import IsaacRLEnv
from magicsim.Env.Animation.Avatar import Avatar
from isaacsim.core.utils.prims import delete_prim
import omni.timeline


class AnimationManager:
    def __init__(
        self,
        num_envs: int,
        config: DictConfig,
        device: torch.device,
        layout_manager=None,
    ):
        """Initialize the SceneManager for managing multiple parallel environments and their objects.

        Args:
            num_envs: Number of parallel environments to manage
            config: Configuration dictionary containing environment and object parameters
            device: PyTorch device (CPU/GPU) for computations
            layout_manager: LayoutManager instance for position management
        """
        self.config = config
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.stage = get_current_stage()
        self.sim: Optional[IsaacRLEnv] = None
        self.layout_manager = layout_manager
        self.env_roots = [f"/World/envs/env_{env_id}" for env_id in range(num_envs)]
        self.avatars: Dict[int, List[Avatar]] = {
            env_id: [] for env_id in range(num_envs)
        }

    def initialize(self, sim: IsaacRLEnv):
        """Initialize the scene manager.
        This function will be called before simulation context creation.
        Put components that cannot be dynamically imported here.
        For CUDA devices, initialize rigid and articulation objects here.
        """
        self.sim = sim
        self.env_origins = self.sim.scene.env_origins

    def post_init(self):
        """Post-initialization logic.
        Initialize objects created in the initialize() method here.
        For CUDA devices, initialize rigid and articulation objects here.
        """
        # Load biped first (creates animation graph)
        self.biped_prim = CharacterUtil.load_default_biped_to_stage()

        # Then setup custom commands (requires animation graph to exist)
        Avatar.init_custom_commands_once()

    def reset(self, soft: bool = False):
        """Reset all environments (batch processing).

        Args:
            soft: If True, perform soft reset; If False, perform hard reset
        """
        self.post_init()
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids, soft=soft)

    def reset_idx(self, env_ids: Sequence[int], soft: bool = True):
        """Reset specified environments (batch processing).

        Args:
            env_ids: Sequence of environment IDs to reset
            soft: If True, perform soft reset; If False, perform hard reset
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.int32)

        for env_id in env_ids:
            env_id = env_id.item()
            print(f"Processing environment {env_id}")
            if soft:
                self._soft_reset(env_id)
            else:
                self._hard_reset(env_id)
            self.sim.sim_step()

    def _soft_reset(self, env_id: int):
        """Perform a soft reset on avatars.
        Resets avatar states without recreating them."""
        if not self.avatars[env_id]:
            print(f"No avatars found in env {env_id} for soft reset")
            return

        if not self.layout_manager:
            raise RuntimeError(
                "LayoutManager is required for avatar reset. Please provide a layout_manager when initializing AvatarManager."
            )

        for avatar in self.avatars[env_id]:
            avatar.reset(soft=True)

    def _hard_reset(self, env_id: int):
        """Perform a hard reset on the specified environment.
        Resets lights, room, and recreates avatars.

        Args:
            env_id: ID of the environment to reset
        """

        if self.avatars[env_id]:
            for avatar in self.avatars[env_id]:
                delete_prim(avatar.prim_path)
            self.avatars[env_id].clear()

        avatar_config = self.config.Avatar
        available_categories = [cat for cat in avatar_config.keys() if cat != "common"]

        for category in available_categories:
            category_config = avatar_config[category]
            common_config = category_config.common
            num_per_env = category_config.num_per_env
            usd_paths = category_config.usd

            if usd_paths and usd_paths[0]:
                usd_root = usd_paths[0]

            category_folder = resolve_path(usd_root)
            available_usd_files = []

            if os.path.exists(category_folder):
                for root, dirs, files in os.walk(category_folder):
                    dirs[:] = [d for d in dirs if d != ".thumbs"]
                    for file in files:
                        if file.endswith(".usd"):
                            usd_path = os.path.join(root, file)
                            available_usd_files.append(usd_path)

            for i in range(num_per_env):
                selected_usd = np.random.choice(available_usd_files)
                prim_path = f"/World/envs/env_{env_id}/Avatar_{category}_{i}"

                # Get layout info from LayoutManager
                if not self.layout_manager:
                    raise RuntimeError(
                        "LayoutManager is required for avatar creation. Please provide a layout_manager when initializing AvatarManager."
                    )

                # Build cat_spec in the same shape LayoutManager expects (with a 'common' section)

                layout_info = self.layout_manager.register_object_and_get_layout(
                    env_id=env_id,
                    prim_path=prim_path,
                    cat_name=category,
                    inst_cfg={},
                    cat_spec=category_config,
                    asset_to_spawn=selected_usd,
                )

                # Get collision setting from config (default: True)
                collision = common_config.get("collision", True)

                avatar = Avatar(
                    prim_path=prim_path,
                    usd_path=selected_usd,
                    config=self.config,
                    env_origin=self.env_origins[env_id],
                    layout_manager=self.layout_manager,
                    layout_info=layout_info,
                    collision=collision,
                )

                # Store avatar in LayoutManager
                self.layout_manager._assign_object_to_category(category, env_id, avatar)

                self.avatars[env_id].append(avatar)

        self.get_all_skel_root_prims()

        CharacterUtil.setup_animation_graph_to_character(
            self.skel_root_prims_list,
            CharacterUtil.get_anim_graph_from_character(self.biped_prim),
        )

    def get_all_skel_root_prims(self):
        self.skel_root_prims_dict = {}
        self.skel_root_prims_list = []
        for env_id, avatar_list in self.avatars.items():
            self.skel_root_prims_dict[env_id] = []
            for avatar in avatar_list:
                if avatar.skelroot_prim.IsValid():
                    self.skel_root_prims_dict[env_id].append(avatar.skelroot_prim)
                    self.skel_root_prims_list.append(avatar.skelroot_prim)
        return self.skel_root_prims_dict, self.skel_root_prims_list

    def on_update(self):
        """Update all characters every frame"""
        # Get current time and delta time from timeline
        timeline = omni.timeline.get_timeline_interface()
        current_time = timeline.get_current_time()
        delta_time = self.sim.physics_dt if self.sim else 0.016

        for env_id, avatar_list in self.avatars.items():
            for avatar in avatar_list:
                avatar.on_update(current_time, delta_time)
