import gymnasium as gym
from magicsim.Env.Environment.SyncRobotEnv import SyncRobotEnv
from typing import Any, Dict, Sequence
import torch
from magicsim.Env.Environment.Utils.Basic import seed_everywhere


class TaskBaseEnv(gym.Env):
    """
    Base Environment for Robot Tasks.
    """

    def __init__(self, config, cli_args, logger):
        self.config = config
        self.Scene_Config = config.Scene
        # ensure nav config is propagated down to scene level for SyncRobotEnv
        if hasattr(config, "Nav") and "nav" not in self.Scene_Config:
            from omegaconf import OmegaConf

            struct_flag = OmegaConf.is_struct(self.Scene_Config)
            if struct_flag:
                OmegaConf.set_struct(self.Scene_Config, False)
            self.Scene_Config.nav = config.Nav
            if struct_flag:
                OmegaConf.set_struct(self.Scene_Config, True)
        self.scene: SyncRobotEnv = gym.make(
            "SyncRobotEnv-V0",
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
        self.last_action = None
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

        self.action_space = self.scene.robot_manager.action_space
        self.observation_space = self.get_obs_space()

    def sample_actions(
        self, batched: bool = True, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor | list[Dict]:
        """
        Sample random actions for the robot.
        Args:
            batched (bool): If True, sample actions for all environments. If False, sample for a single environment.
        """
        return self.scene.robot_manager.sample_actions(batched=batched, env_ids=env_ids)

    def get_obs(
        self,
    ) -> Dict[str, Any]:
        obs_dict = {}
        obs_dict["policy_obs"] = self.get_policy_obs()
        obs_dict["privilege_obs"] = self.get_privilege_obs()
        # Attach last robot action for policy debugging / history
        obs_dict["policy_obs"]["last_action"] = self.last_action
        return obs_dict

    def get_policy_obs(
        self,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def get_privilege_obs(
        self,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def process_action(self, action: torch.Tensor | list[Dict]):
        raise NotImplementedError

    def step(
        self,
        action: torch.Tensor | list[Dict],
        env_ids: Sequence[int] | None = None,
        failed_env_ids: Sequence[int] = [],
    ):
        """
        Args:
            action (torch.Tensor): The action to be executed by the robot.
                The shape of the action should be (num_envs, action_dim).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.int32)
            else:
                env_ids = env_ids.to(self.device, dtype=torch.int32)
        if action is None or len(env_ids) == 0:
            self.scene.sim.sim_step()
            self.last_action = None
            pending_env_ids = []
        else:
            self.processed_action = self.process_action(action)
            action_info, pending_env_ids = self.scene.step(
                self.processed_action, env_ids=env_ids
            )
            # Pad action_info and step_success_flags to num_envs length
            padded_action_info = self._pad_action_info_to_num_envs(
                action_info, env_ids, self.num_envs
            )
            self.last_action = padded_action_info

        reward = self.get_reward(action, env_ids)
        terminated, truncated = self.get_termination()
        # print("terminated: ", terminated)
        self.reset_terminated[terminated] = 1
        self.reset_truncated[truncated] = 1

        failed_env_ids = torch.tensor(
            failed_env_ids, device=self.device, dtype=torch.int32
        )

        self.reset_truncated[failed_env_ids] = 1
        truncated[failed_env_ids] = 1

        self.episode_length_buf[env_ids] += 1
        self.reset_buf = self.reset_terminated | self.reset_truncated

        assert len(reward) == self.num_envs, "Reward length should be equal to num_envs"
        assert len(terminated) == self.num_envs, (
            "Terminated length should be equal to num_envs"
        )
        assert len(truncated) == self.num_envs, (
            "Truncated length should be equal to num_envs"
        )

        # 在 reset 之前采样 obs/info，保证本步返回的是「终止瞬间」的状态，供上层 task/atomic_skill 正确判定 success 并存轨迹
        info = self.get_info()
        obs = self.get_obs()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
            self.scene.sim.sim_step()

        # Check that all innermost values in obs and info have length num_envs
        self._check_dict_values_length(obs, self.num_envs, "obs")
        self._check_dict_values_length(info, self.num_envs, "info")

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
            pending_env_ids,
        )

    def get_info(
        self,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def get_reward(
        self,
        action: torch.Tensor | list[Dict],
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_termination(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def get_obs_space(self) -> gym.spaces.Space:
        raise NotImplementedError

    def get_object_pose(
        self,
        env_ids: Sequence[int],
        obj_type: str,
        obj_name: str,
        obj_id: int,
    ) -> torch.Tensor:
        """
        Get current 7D pose (pos + quat) of target object for the given env_ids.
        Used by tasks (e.g. Push) to read object position for MPC/waypoint planning.

        Args:
            env_ids: list or tensor of environment indices
            obj_type: "rigid" or "geometry" (which dict to try first)
            obj_name: category key in scene_manager (e.g. "cube", "mug")
            obj_id: index within that category list

        Returns:
            Tensor of shape (len(env_ids), 7) with [x, y, z, qw, qx, qy, qz] per row.
        """
        if not hasattr(self.scene, "scene_manager") or self.scene.scene_manager is None:
            raise RuntimeError(
                "TaskBaseEnv.get_object_pose: scene.scene_manager is not available."
            )
        scene_mgr = self.scene.scene_manager
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.cpu().tolist()
        if isinstance(env_ids, int):
            env_ids = [env_ids]
        poses = []
        for eid in env_ids:
            rigid_env = scene_mgr.rigid_objects[eid]
            geo_env = scene_mgr.geometry_objects[eid]
            obj_list = rigid_env.get(obj_name, [])
            if not obj_list:
                obj_list = geo_env.get(obj_name, [])
            if not obj_list or obj_id >= len(obj_list):
                raise RuntimeError(
                    f"get_object_pose: no object found env_id={eid} "
                    f"obj_name='{obj_name}' obj_id={obj_id}; "
                    f"rigid_keys={list(rigid_env.keys())}, geo_keys={list(geo_env.keys())}"
                )
            obj = obj_list[obj_id]
            pos, quat = obj.get_local_pose()
            poses.append(torch.cat([pos.squeeze(), quat.squeeze()], dim=0))
        return torch.stack(poses, dim=0)

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
        for i in range(3):
            self.scene.sim.sim_step()
        self.episode_length_buf[:] = 0
        self.reset_terminated[:] = 0
        self.reset_truncated[:] = 0
        self.reset_buf[:] = 0
        self.last_action = None
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
        if seed is not None:
            seed_everywhere(seed)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.scene.sim.device)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(
                env_ids, device=self.scene.sim.device, dtype=torch.int32
            )

        print(f"Resetting env {env_ids} with seed {seed} and options {options}")
        self.scene.reset_idx(env_ids=env_ids, seed=seed, options=options)
        # SceneManager.reset_idx only re-initializes scene objects (bin,
        # lights, etc.); it does NOT touch the robot's action_manager state.
        # Without this call, Pink IK's ``_last_passed_action`` retains the
        # previous episode's wrist target, so the next episode's init-phase
        # NaN action falls back onto that latched pose and the arms snap
        # back to the previous squeeze/lift instead of the rest pose.
        # See ``RobotManager.reset_idx`` → ``ActionManager.reset`` →
        # ``PinkInverseKinematicsAction.reset`` (clears ``_last_passed_action``).
        if (
            hasattr(self.scene, "robot_manager")
            and self.scene.robot_manager is not None
        ):
            self.scene.robot_manager.reset_idx(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.reset_terminated[env_ids] = 0
        self.reset_truncated[env_ids] = 0
        self.reset_buf[env_ids] = 0
        obs = self.get_obs()
        info = self.get_info()
        return obs, info

    def _pad_action_info_to_num_envs(
        self, action_info: Dict[str, Any], env_ids: torch.Tensor, num_envs: int
    ) -> Dict[str, Any]:
        """Pad action_info from len(env_ids) to num_envs length.

        Args:
            action_info: Dictionary containing action information for env_ids
            env_ids: Tensor of environment IDs that have actual data
            num_envs: Total number of environments

        Returns:
            Padded action_info with shape (num_envs, ...) for all tensors,
            with torch.nan used for padding.
        """
        # Convert env_ids to list
        if isinstance(env_ids, torch.Tensor):
            env_ids_list = env_ids.detach().cpu().tolist()
            if isinstance(env_ids_list, int):
                env_ids_list = [env_ids_list]
        else:
            env_ids_list = list(env_ids)

        padded_info = {}
        for key, value in action_info.items():
            if isinstance(value, torch.Tensor):
                # Pad tensor: create (num_envs, ...) tensor filled with nan
                # Then fill env_ids positions with actual values
                if value.ndim == 0:
                    # Scalar tensor - expand to (num_envs,)
                    padded_tensor = torch.full(
                        (num_envs,), torch.nan, device=value.device, dtype=value.dtype
                    )
                    padded_tensor[env_ids_list] = value.expand(len(env_ids_list))
                else:
                    # Multi-dimensional tensor: (len(env_ids), ...)
                    if value.shape[0] == num_envs:
                        # Already padded for all envs
                        padded_tensor = value
                    else:
                        # Handle values provided only for env_ids
                        if value.shape[0] == len(env_ids_list):
                            value_to_pad = value
                        elif value.shape[0] == 1 and len(env_ids_list) > 1:
                            # Broadcast single entry across env_ids if needed
                            value_to_pad = value.expand(
                                (len(env_ids_list), *value.shape[1:])
                            )
                        else:
                            raise ValueError(
                                "action_info tensor batch dimension does not match "
                                f"env_ids or num_envs for key '{key}'. "
                                f"Got {value.shape[0]}, env_ids={len(env_ids_list)}, "
                                f"num_envs={num_envs}."
                            )
                        shape = list(value.shape)
                        shape[0] = num_envs
                        padded_tensor = torch.full(
                            shape, torch.nan, device=value.device, dtype=value.dtype
                        )
                        padded_tensor[env_ids_list] = value_to_pad
                padded_info[key] = padded_tensor
            elif isinstance(value, dict):
                # Recursively pad nested dictionaries
                padded_info[key] = self._pad_action_info_to_num_envs(
                    value, env_ids, num_envs
                )
            else:
                # Non-tensor, non-dict values are kept as-is
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
                # Check tensor first dimension
                if value.ndim > 0:
                    actual_length = value.shape[0]
                    assert actual_length == expected_length, (
                        f"Value at path '{current_path}' has length {actual_length}, "
                        f"expected {expected_length}. Shape: {value.shape}"
                    )
                # Scalar tensors are allowed (they don't have a first dimension)
            elif isinstance(value, dict):
                # Recursively check nested dictionaries
                self._check_dict_values_length(value, expected_length, current_path)
            elif isinstance(value, (list, tuple)):
                # For lists/tuples, check each item
                # Note: The list itself may not have length num_envs (e.g., list of observation managers)
                # but the items inside (if dicts or tensors) should have num_envs length
                for i, item in enumerate(value):
                    if isinstance(item, torch.Tensor):
                        # Check tensor first dimension
                        if item.ndim > 0:
                            actual_item_length = item.shape[0]
                            assert actual_item_length == expected_length, (
                                f"Value at path '{current_path}[{i}]' has length {actual_item_length}, "
                                f"expected {expected_length}. Shape: {item.shape}"
                            )
                    elif isinstance(item, dict):
                        # Recursively check nested dicts in list
                        self._check_dict_values_length(
                            item, expected_length, f"{current_path}[{i}]"
                        )
                    # Other types in list are allowed (e.g., None, strings, etc.)
            # Other types (None, int, float, str, etc.) are allowed and skipped
