import gymnasium as gym
from magicsim.Env.Environment.SyncCameraEnv import SyncCameraEnv
from typing import Any, Dict, Sequence
import torch
from magicsim.Env.Environment.Utils.Basic import seed_everywhere
from collections.abc import Sequence as SequenceABC


class TaskCameraBaseEnv(gym.Env):
    """
    Base Environment for Camera Tasks.
    """

    def __init__(self, config, cli_args, logger):
        self.config = config
        self.Scene_Config = config.Scene
        # ensure nav config is propagated down to scene level for SyncCameraEnv
        if hasattr(config, "Nav") and "nav" not in self.Scene_Config:
            from omegaconf import OmegaConf

            struct_flag = OmegaConf.is_struct(self.Scene_Config)
            if struct_flag:
                OmegaConf.set_struct(self.Scene_Config, False)
            self.Scene_Config.nav = config.Nav
            if struct_flag:
                OmegaConf.set_struct(self.Scene_Config, True)
        self.scene: SyncCameraEnv = gym.make(
            "SyncCameraEnv-V0",
            config=self.Scene_Config,
            cli_args=cli_args,
            logger=logger,
        )
        self.device = self.scene.device
        self.num_envs = self.scene.num_envs
        self.config = config
        self.cli_args = cli_args
        self.logger = logger

        self._reward_mode = None

        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.reset_terminated = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.reset_truncated = torch.zeros_like(self.reset_terminated)
        self.reset_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.last_camera_action = None
        self._configure_gym_env_spaces()

    def sim_step(self):
        self.scene.sim.sim_step()

    @property
    def reward_mode(self):
        return self._reward_mode

    def _configure_gym_env_spaces(self):
        """
        Configure the observation and action spaces for the Gym environment.
        This method should be overridden by subclasses to define specific spaces.
        """
        self.action_space = gym.spaces.Dict(
            {}
        )  # Camera tasks don't have robot action space
        self.observation_space = self.get_obs_space()

    def get_obs(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        obs_dict = {}
        obs_dict["policy_obs"] = self.get_policy_obs(env_ids)
        obs_dict["privilege_obs"] = self.get_privilege_obs(env_ids)
        # Attach last camera action for policy debugging / history
        obs_dict["policy_obs"]["last_camera_action"] = self.last_camera_action
        return obs_dict

    def get_policy_obs(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        raise NotImplementedError

    def get_privilege_obs(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        raise NotImplementedError

    def process_camera_action(
        self,
        camera_action: torch.Tensor | list[Dict] | None,
        env_ids: Sequence[int] | None = None,
    ) -> Dict[str, torch.Tensor] | None:
        """
        Process the camera action from the input action.
        Override this method in subclasses to generate camera actions.

        Args:
            camera_action: The input camera action
            env_ids: Environment IDs

        Returns:
            Camera action dict: {camera_name: [N, 7]} target poses for cameras, or None
        """
        return camera_action

    def step(
        self,
        camera_action: Dict[str, torch.Tensor] | None = None,
        env_ids: Sequence[int] | None = None,
        failed_env_ids: Sequence[int] | None = None,
    ):
        """
        Args:
            camera_action: Optional camera action dict: {camera_name: [N, 7]} target poses
        """
        # Normalize env_ids to a tensor on the correct device, like TaskBaseEnv
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, device=self.device)
            else:
                env_ids = env_ids.to(self.device)

        camera_action = self.process_camera_action(
            camera_action=camera_action, env_ids=env_ids
        )
        camera_info = self.scene.step(camera_action=camera_action, env_ids=env_ids)

        # Pad camera_info to num_envs so downstream code can assume batched layout.
        padded_camera_info = self._pad_camera_info_to_num_envs(
            camera_info, env_ids, self.num_envs
        )
        self.last_camera_action = padded_camera_info

        reward = self.get_reward(camera_action, env_ids)
        terminated, truncated = self.get_termination()
        self.reset_terminated[terminated] = 1
        self.reset_truncated[truncated] = 1
        if failed_env_ids is not None:
            self.reset_truncated[failed_env_ids] = 1
        self.episode_length_buf[env_ids] += 1
        self.reset_buf = self.reset_terminated | self.reset_truncated

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # Basic sanity checks to ensure shapes are consistent with num_envs
        assert len(reward) == self.num_envs, "Reward length should be equal to num_envs"
        assert len(terminated) == self.num_envs, (
            "Terminated length should be equal to num_envs"
        )
        assert len(truncated) == self.num_envs, (
            "Truncated length should be equal to num_envs"
        )

        # For consistency with TaskBaseEnv, get full-batch obs/info
        info = self.get_info()
        obs = self.get_obs()

        # Check that all innermost values in obs and info have length num_envs
        self._check_dict_values_length(obs, self.num_envs, "obs")
        self._check_dict_values_length(info, self.num_envs, "info")

        return obs, reward, terminated, truncated, info

    def get_info(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        raise NotImplementedError

    def get_reward(
        self,
        camera_action: Dict[str, torch.Tensor] | None,
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_termination(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def get_obs_space(self) -> gym.spaces.Space:
        raise NotImplementedError

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        Reset the environment.
        This should only be called once at the beginning of the environment.
        In this function, we will call scene.reset(soft=False) to load all the objects managed in scene manager
        It will also reset the reset count.
        """
        if seed is not None:
            seed_everywhere(seed)
        self.scene.reset(options=options)
        self.scene.sim.sim_step()
        self.episode_length_buf[:] = 0
        self.reset_terminated[:] = 0
        self.reset_truncated[:] = 0
        self.reset_buf[:] = 0
        self.last_camera_action = None
        self.scene.sim.sim_step()
        obs = self.get_obs()
        info = self.get_info()
        return obs, info

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
            env_ids = torch.arange(self.num_envs, device=self.scene.sim.device)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(
                env_ids, device=self.scene.sim.device, dtype=torch.int32
            )

        # Handle seed input similar to BaseEnv.reset_idx()
        # Check if seed is a sequence (list/tuple) to support per-env seeds
        if seed is not None:
            # Check if seed is a sequence (but not string/bytes)
            is_sequence_seed = isinstance(seed, SequenceABC) and not isinstance(
                seed, (str, bytes)
            )
            if is_sequence_seed:
                # If seed is a sequence, use the first env's seed for global seed
                env_id_list = (
                    env_ids.detach().cpu().tolist()
                    if isinstance(env_ids, torch.Tensor)
                    else list(env_ids)
                )
                if env_id_list and len(seed) > 0:
                    first_env = env_id_list[0]
                    if len(seed) > first_env:
                        first_seed = seed[first_env]
                    else:
                        first_seed = seed[0]
                    seed_everywhere(first_seed)
            else:
                seed_everywhere(seed)

        self.scene.reset_idx(env_ids=env_ids, seed=seed, options=options)
        self.episode_length_buf[env_ids] = 0
        self.reset_terminated[env_ids] = 0
        self.reset_truncated[env_ids] = 0
        self.reset_buf[env_ids] = 0
        # For simplicity, clear last actions when partial reset happens
        self.last_camera_action = None
        obs = self.get_obs(env_ids)
        info = self.get_info(env_ids)
        return obs, info

    def _pad_camera_info_to_num_envs(
        self, camera_info: Dict[str, Any], env_ids: torch.Tensor, num_envs: int
    ) -> Dict[str, Any]:
        """Pad camera_info from len(env_ids) to num_envs length.

        This mirrors TaskBaseEnv._pad_action_info_to_num_envs but is named for camera info.

        Args:
            camera_info: Dictionary containing camera information for env_ids
            env_ids: Tensor of environment IDs that have actual data
            num_envs: Total number of environments

        Returns:
            Padded camera_info with shape (num_envs, ...) for all tensors,
            with torch.nan used for padding.
        """
        if isinstance(env_ids, torch.Tensor):
            env_ids_list = env_ids.detach().cpu().tolist()
            if isinstance(env_ids_list, int):
                env_ids_list = [env_ids_list]
        else:
            env_ids_list = list(env_ids)

        padded_info: Dict[str, Any] = {}
        for key, value in camera_info.items():
            if isinstance(value, torch.Tensor):
                # Pad tensor: create (num_envs, ...) tensor filled with nan,
                # then fill env_ids positions with actual values.
                if value.ndim == 0:
                    padded_tensor = torch.full(
                        (num_envs,),
                        torch.nan,
                        device=value.device,
                        dtype=value.dtype,
                    )
                    padded_tensor[env_ids_list] = value.expand(len(env_ids_list))
                else:
                    shape = list(value.shape)
                    shape[0] = num_envs
                    padded_tensor = torch.full(
                        shape, torch.nan, device=value.device, dtype=value.dtype
                    )
                    padded_tensor[env_ids_list] = value
                padded_info[key] = padded_tensor
            elif isinstance(value, dict):
                padded_info[key] = self._pad_camera_info_to_num_envs(
                    value, env_ids, num_envs
                )
            else:
                padded_info[key] = value

        return padded_info

    def _check_dict_values_length(
        self, data: Dict[str, Any], expected_length: int, path: str = ""
    ) -> None:
        """Recursively check that all innermost values in a dictionary have the expected length.

        Args:
            data: Dictionary to check (can be nested)
            expected_length: Expected length (num_envs)
            path: Current path in the dictionary (for error messages)
        """
        if not isinstance(data, dict):
            return

        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, torch.Tensor):
                if value.ndim > 0:
                    actual_length = value.shape[0]
                    assert actual_length == expected_length, (
                        f"Value at path '{current_path}' has length {actual_length}, "
                        f"expected {expected_length}. Shape: {value.shape}"
                    )
            elif isinstance(value, dict):
                self._check_dict_values_length(value, expected_length, current_path)
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, torch.Tensor):
                        if item.ndim > 0:
                            actual_item_length = item.shape[0]
                            assert actual_item_length == expected_length, (
                                f"Value at path '{current_path}[{i}]' has length {actual_item_length}, "
                                f"expected {expected_length}. Shape: {item.shape}"
                            )
                    elif isinstance(item, dict):
                        self._check_dict_values_length(
                            item, expected_length, f"{current_path}[{i}]"
                        )
            # Other scalar / None / str etc. are allowed.
