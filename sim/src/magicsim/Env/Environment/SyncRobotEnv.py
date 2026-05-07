"""
This is a robot base environment class for sync rl training(low-level rl training) and 3D/4D data sync data with robot.
All the action here is synchronous and atomic. If you require high-level action please use AsyncRobotEnv.
The action here is atomic, meaning that the action will be executed in a single step.

"""

from typing import Any, Dict, Sequence

import torch
from magicsim.Env.Environment.SyncCollectEnv import SyncCollectEnv
from magicsim.Env.Robot.RobotManager import RobotManager
from magicsim.Env.Planner.PlannerManager import PlannerManager


class SyncRobotEnv(SyncCollectEnv):
    """
    This is a robot base environment class for sync rl training(low-level rl training) and 3D/4D data sync data with robot.
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)
        self.robot_config = config.robot
        self.robot_manager = RobotManager(
            num_envs=self.num_envs,
            config=self.robot_config,
            device=self.device,
            logger=logger,
            seeds_per_env=self.env_seed_list,
        )
        self.planner_manager = PlannerManager(
            num_envs=self.num_envs,
            robot_config=self.robot_config,
            device=self.device,
            logger=logger,
        )

    def _setup_scene(self, sim):
        """
        Initialize the environment.
        This function will be called before simulation context create and be called by isaaclab _setup_scene function
        """
        self.robot_manager.initialize(sim)
        if self.nav_manager is not None:
            occupancy_manager = self.nav_manager.occupancy_manager
        else:
            occupancy_manager = None
        self.planner_manager.initialize(
            self.robot_manager,
            occupancy_manager=occupancy_manager,
        )
        super()._setup_scene(sim)

    def step(
        self,
        action: torch.Tensor | list[Dict] = None,
        env_ids: Sequence[int] | None = None,
    ):
        """
        Args:
            action (torch.Tensor): The action to be executed by the robot.
                The shape of the action should be (num_envs, action_dim).
        """

        if env_ids is None:
            env_ids = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        else:
            env_ids = [int(i) for i in env_ids]

        if action is None:
            self.sim.sim_step()
            return {}, None, None, None

        action_info = {}  # Note here we only return the action info for env_ids

        action_info["command"] = action
        pending_env_ids = self._get_pending_env_ids(action, env_ids)

        # Process robot actions through planner step
        processed_action = self.planner_manager.step(action=action, env_ids=env_ids)
        # print("processed_action in SyncRobotEnv: ", processed_action)
        action_info["robot_action"] = processed_action
        # print(action_info["robot_action"])

        step_info = self.robot_manager.step(
            action=action_info["robot_action"], env_ids=env_ids
        )
        action_info.update(step_info)

        return action_info, pending_env_ids

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        Reset the environment.
        This should only be called once at the beginning of the environment.
        In this function, we will call scene_manager.reset(soft=False) to load all the objects managed in scene manager
        It will also reset the reset count.
        """
        super().reset(
            seed=seed, options=options
        )  # Lab Reset: Will Reset All Object Managed By Lab(Robot arm)
        print("Scene Reset Finished")
        self.robot_manager.reset()
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
        self.robot_manager.reset_idx(
            env_ids=env_id_list
        )  # Reset the robots in the specific environments
        self.planner_manager.reset_idx(
            env_ids=env_id_list
        )  # Reset the planners in the specific environments

    def _update_seed_managers(self):
        super()._update_seed_managers()
        seeds = self.env_seeds
        if hasattr(self, "robot_manager") and self.robot_manager is not None:
            self.robot_manager.update_env_seeds(seeds)

    def _apply_action(self, sim):
        self.robot_manager._apply_action(sim)

    def _pre_physics_step(self, sim, actions, env_ids):
        self.robot_manager.pre_physics_step(sim, actions, env_ids)

    def _get_pending_env_ids(
        self,
        action: torch.Tensor | dict[str, dict[str, torch.Tensor]] | None,
        env_ids: Sequence[int] | None,
    ) -> list[int]:
        """
        Get environment IDs where all action fields are NaN (pending actions).

        Args:
            action: Action tensor or dictionary of actions
            env_ids: List of environment IDs to check

        Returns:
            List of environment IDs where all actions are NaN
        """
        if action is None:
            return []

        if env_ids is None:
            env_ids = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        else:
            env_ids = [int(i) for i in env_ids]

        pending_mask = None

        if isinstance(action, dict):
            pending_mask = self._check_all_nan_dict(action, env_ids)
        elif isinstance(action, torch.Tensor):
            # Flatten to (N, -1) for 2D/3D actions; clone to avoid modifying original
            action_flat = action.clone().reshape(action.shape[0], -1)
            pending_mask = action_flat.isnan().all(dim=1)
            # If action tensor has shape (num_envs, ...) but we're checking subset
            if action.shape[0] == self.num_envs and len(env_ids) != self.num_envs:
                env_ids_tensor = torch.tensor(env_ids, device=action.device)
                pending_mask = pending_mask[env_ids_tensor]
        else:
            # Unknown action type, return empty list
            return []

        if pending_mask is None or not pending_mask.any():
            return []

        env_ids_tensor = torch.tensor(env_ids, device=pending_mask.device)
        return env_ids_tensor[pending_mask].tolist()

    def _check_all_nan_dict(
        self, actions: dict[str, dict[str, torch.Tensor]], env_ids: Sequence[int]
    ) -> torch.Tensor:
        """
        Check if all fields in dictionary actions are NaN for each environment.

        Args:
            actions: Dictionary of actions with structure {robot_name: {action_key: tensor}}
            env_ids: List of environment IDs to check

        Returns:
            Boolean tensor indicating which environments have all NaN actions
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        else:
            env_ids = [int(i) for i in env_ids]

        batch_size = len(env_ids)
        device = self.device

        # Collect all tensors from the dictionary structure
        all_tensors = []
        for rname, rspace in self.planner_manager.single_action_space.spaces.items():
            if rname not in actions:
                # If robot name not in actions, all fields for this robot are considered NaN
                # Create NaN tensors for all action keys of this robot
                for k in rspace.spaces.keys():
                    # Get expected shape from action space
                    expected_shape = rspace.spaces[k].shape
                    total_dim = 1
                    for dim in expected_shape:
                        total_dim *= dim
                    # Create NaN tensor with batch dimension
                    nan_tensor = torch.full(
                        (batch_size, total_dim),
                        float("nan"),
                        device=device,
                        dtype=torch.float32,
                    )
                    all_tensors.append(nan_tensor)
                continue

            for k in rspace.spaces.keys():
                if k not in actions[rname]:
                    # If action key not in actions, consider as NaN
                    # Get expected shape from action space
                    expected_shape = rspace.spaces[k].shape
                    total_dim = 1
                    for dim in expected_shape:
                        total_dim *= dim
                    # Create NaN tensor with batch dimension
                    nan_tensor = torch.full(
                        (batch_size, total_dim),
                        float("nan"),
                        device=device,
                        dtype=torch.float32,
                    )
                    all_tensors.append(nan_tensor)
                    continue

                t = torch.as_tensor(
                    actions[rname][k], dtype=torch.float32, device=device
                )
                # Select env subset if tensor matches total env dimension
                if t.shape[0] == self.num_envs and batch_size != self.num_envs:
                    t = t[env_ids]
                # Ensure batch dimension exists
                if t.ndim == 1:
                    if t.shape[0] == batch_size:
                        t = t.unsqueeze(1)  # Treat as per-env scalar -> [N, 1]
                    elif t.shape[0] == self.num_envs:
                        t = t[env_ids].unsqueeze(1)
                    else:
                        t = t.unsqueeze(0)  # Fallback: treat as single sample
                if t.ndim > 2:
                    t = t.reshape(t.shape[0], -1)  # Flatten feature dimensions
                all_tensors.append(t)

        if len(all_tensors) == 0:
            # No valid tensors found, consider all as NaN
            return torch.ones(batch_size, dtype=torch.bool, device=device)

        # Check if all tensors are NaN for each environment
        # Concatenate all tensors along feature dimension
        combined = torch.cat(all_tensors, dim=1)  # [batch_size, total_features]
        # Check if all features are NaN for each environment
        pending_mask = combined.isnan().all(dim=1)  # [batch_size]

        return pending_mask
