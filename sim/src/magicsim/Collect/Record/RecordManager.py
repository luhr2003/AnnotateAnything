from typing import Any, Dict, List
from omegaconf import DictConfig
import torch
import os
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from magicsim.Env.Utils.file import Logger
from magicsim.Collect.Record.CameraWriter import write_annotator_step
import magicsim.Collect.Record.io_functions as F
from omni.replicator.core.scripts.backends import BackendDispatch
from magicsim.Env.Utils.path import resolve_path
from magicsim.Collect.Record.utils import (
    to_serializable,
    extract_env_from_dict,
    check_dict_values_length,
    is_image_annotator,
    is_json_annotator,
    is_numpy_annotator,
    extract_image_data,
    extract_json_data,
    extract_json_metadata,
    extract_numpy_data,
    convert_annotator_data_to_payload,
)

JSON_INDENT = 4  # pretty-print saved json artifacts for easier inspection


class RecordManager:
    def __init__(
        self,
        env: TaskBaseEnv,
        num_envs: int,
        record_config: DictConfig,
        device: torch.device,
        logger: Logger,
    ):
        self.env = env
        self.num_envs = num_envs
        self.record_config = record_config
        self.device = device
        self.logger = logger
        self.record_obs_buffer: List[List[Dict[str, Any]]] = []
        self.record_action_buffer: List[List[Dict[str, Any]]] = []
        self.record_collect_buffer: List[List[Dict[str, Any]]] = []
        self.trajectory_id = 0
        # Track if we've already saved for a completed task (to avoid multiple saves)
        self._saved_for_completed_task: List[bool] = [False] * num_envs
        # Allow overriding default OUTPUT_PATH via record_config (if provided).
        # Expected usage (Collect line): record_config.output_dir
        # Task line can also pass a config with output_dir field to customize.
        self.output_path = resolve_path(record_config.output_dir)
        # Save in video mode: merge frames into videos and consolidated JSONs
        self.save_in_video = record_config.get("save_in_video", True)

    def _convert_batched_env_info_to_per_env(
        self, batched_env_info: Dict[str, Any], env_ids: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Convert batched env_info format to per-env format.

        env_info should contain: obs, reward, terminated, truncated, info
        where obs contains: policy_obs, privilege_obs, camera_info

        Args:
            batched_env_info: Batched format where camera_info is organized by cam_id
            env_ids: List of environment IDs to process

        Returns:
            Dictionary mapping env_id to env_info dictionary
        """
        # Check that all innermost values have length num_envs
        # But skip "state" field as it has special structure (list of per-env state dicts)
        state_backup = None
        if "info" in batched_env_info and isinstance(batched_env_info["info"], dict):
            state_backup = batched_env_info["info"].pop("state", None)

        check_dict_values_length(batched_env_info, self.num_envs, "batched_env_info")

        # Restore state field
        if state_backup is not None and "info" in batched_env_info:
            batched_env_info["info"]["state"] = state_backup

        # Check terminated and truncated if they exist
        if (
            "terminated" in batched_env_info
            and batched_env_info["terminated"] is not None
        ):
            terminated = batched_env_info["terminated"]
            if isinstance(terminated, torch.Tensor):
                assert len(terminated) == self.num_envs, (
                    f"terminated has length {len(terminated)}, expected {self.num_envs}"
                )
        if (
            "truncated" in batched_env_info
            and batched_env_info["truncated"] is not None
        ):
            truncated = batched_env_info["truncated"]
            if isinstance(truncated, torch.Tensor):
                assert len(truncated) == self.num_envs, (
                    f"truncated has length {len(truncated)}, expected {self.num_envs}"
                )

        per_env_info = {}

        if len(batched_env_info) == 5:
            obs = batched_env_info["obs"]
            reward = batched_env_info["reward"]
            terminated = batched_env_info["terminated"]
            truncated = batched_env_info["truncated"]
            info = batched_env_info["info"]
        else:
            obs = batched_env_info["obs"]
            info = batched_env_info["info"]
            reward = None
            terminated = None
            truncated = None

        # Extract obs components
        policy_obs = obs.get("policy_obs", {})
        camera_info = policy_obs.get("camera_info", [])
        # Extract both last_action and last_camera_action, but keep them conceptually separate.
        # For pure camera tasks, we still need a last_action for the generic pipeline,
        # so we fall back to last_camera_action when last_action is missing.
        action_info = policy_obs.pop("last_action", None)
        camera_action_info = policy_obs.pop("last_camera_action", None)
        if action_info is None and camera_action_info is not None:
            # Use camera actions as the generic last_action when robot actions are absent.
            action_info = camera_action_info
        privilege_obs = obs.get("privilege_obs", {})

        if len(batched_env_info) == 5:
            if action_info is None:
                # If action_info is None, create a default empty dict to avoid assertion error
                # This can happen when atomic skill fails and no action was generated
                action_info = {}

        # Convert camera_info from cam_id-based to env_id-based
        # camera_info format: [cam_id][annotator_name][env_id]
        # Target format: [env_id][annotator_name][cam_id]
        env_camera_info = {}
        if camera_info:
            # Get annotator names from first camera (all cameras should have same annotators)
            num_cams = len(camera_info)
            if num_cams > 0:
                annotator_names = list(camera_info[0].keys())

                # Initialize camera_info for each requested env
                for env_id in env_ids:
                    env_cam_data = {}
                    for annotator_name in annotator_names:
                        cam_list = []
                        for cam_id in range(num_cams):
                            if annotator_name in camera_info[cam_id]:
                                annotator_data = camera_info[cam_id][annotator_name]
                                # Check if annotator_data is a list/tensor and has enough elements
                                if isinstance(annotator_data, (list, torch.Tensor)):
                                    if env_id < len(annotator_data):
                                        cam_list.append(annotator_data[env_id])
                                    else:
                                        # If env_id is out of range, append None
                                        cam_list.append(None)
                                else:
                                    # If not a list/tensor, use as-is (shouldn't happen normally)
                                    cam_list.append(annotator_data)
                            else:
                                cam_list.append(None)
                        env_cam_data[annotator_name] = cam_list

                    # For camera_params, add local pose information at recording time
                    # This ensures each step records its own pose, not the final pose
                    if "camera_params" in env_cam_data:
                        env_cam_data["camera_params"] = (
                            self._add_local_pose_to_camera_params_at_record_time(
                                env_cam_data["camera_params"], env_id
                            )
                        )

                    env_camera_info[env_id] = env_cam_data

        # Convert for each requested environment
        for env_id in env_ids:
            # Convert policy_obs
            env_policy_obs = {}

            # Special handling for robot_state
            if "robot_state" in policy_obs:
                env_robot_state = {}
                for robot_name, robot_data in policy_obs["robot_state"][0].items():
                    env_robot_state[robot_name] = {}
                    for key, value in robot_data.items():
                        if isinstance(value, torch.Tensor):
                            # Extract single env from batched tensor
                            env_robot_state[robot_name][key] = value[env_id]
                        else:
                            env_robot_state[robot_name][key] = value
                env_policy_obs["robot_state"] = [env_robot_state]

            # Special handling for camera_info
            if env_id in env_camera_info:
                env_policy_obs["camera_info"] = env_camera_info[env_id]

            # Special handling for last_action
            # Format: {
            #   'command': tensor,  # direct tensor (padded to num_envs)
            #   'robot_action': tensor,  # direct tensor (padded to num_envs)
            #   'raw_action': {robot_name: {term_name: tensor}},  # nested dict (padded to num_envs)
            #   'processed_action': {robot_name: {term_name: tensor}}  # nested dict (padded to num_envs)
            # }
            # Now action_info is padded to num_envs, so we can use env_id directly.
            if action_info:
                env_action_info = {}
                for key, value in action_info.items():
                    if isinstance(value, torch.Tensor):
                        # Direct tensor (e.g., 'command', 'robot_action')
                        env_value = value[env_id]
                        env_action_info[key] = env_value
                    elif isinstance(value, dict):
                        # Nested dict (e.g., 'raw_action', 'processed_action')
                        # Expected format: {robot_name: {term_name: tensor}}
                        # However, for some camera-only tasks, value may be {robot_name: tensor}
                        env_action_info[key] = {}
                        for robot_name, robot_actions in value.items():
                            # If robot_actions is a tensor, treat it as a direct per-env tensor
                            if isinstance(robot_actions, torch.Tensor):
                                env_action_info[key][robot_name] = robot_actions[env_id]
                                continue

                            env_action_info[key][robot_name] = {}
                            for term_name, term_actions in robot_actions.items():
                                if isinstance(term_actions, torch.Tensor):
                                    term_value = term_actions[env_id]
                                    # Extract single env from batched tensor (now padded to num_envs)
                                    env_action_info[key][robot_name][term_name] = (
                                        term_value
                                    )
                                else:
                                    env_action_info[key][robot_name][term_name] = (
                                        term_actions
                                    )
                    else:
                        # Other types (shouldn't happen, but handle gracefully)
                        env_action_info[key] = value
                env_policy_obs["last_action"] = env_action_info

            # Special handling for last_camera_action (camera-only or camera-augmented tasks)
            # We keep it separate from last_action, mirroring the old behavior.
            if camera_action_info:
                env_camera_action_info: Dict[str, Any] = {}
                for key, value in camera_action_info.items():
                    if isinstance(value, torch.Tensor):
                        # Extract single env from batched tensor; we don't enforce NaN here,
                        # assuming upstream camera env already passed padding checks.
                        env_camera_action_info[key] = value[env_id]
                    elif isinstance(value, dict):
                        # Use generic extractor for nested dicts.
                        env_camera_action_info[key] = extract_env_from_dict(
                            value, env_id
                        )
                    else:
                        env_camera_action_info[key] = value
                env_policy_obs["last_camera_action"] = env_camera_action_info

            for key, value in policy_obs.items():
                # Skip already processed items
                if key in ["robot_state", "camera_info", "last_action"]:
                    continue
                # Process like privilege_obs: extract tensor or recursively handle nested structures
                if isinstance(value, torch.Tensor):
                    env_policy_obs[key] = value[env_id]
                elif isinstance(value, dict):
                    env_policy_obs[key] = extract_env_from_dict(value, env_id)
                else:
                    env_policy_obs[key] = value

            # Convert privilege_obs (unchanged)
            env_privilege_obs = {}
            for key, value in privilege_obs.items():
                if isinstance(value, torch.Tensor):
                    # Extract single env from batched tensor
                    env_privilege_obs[key] = value[env_id]
                elif isinstance(value, dict):
                    # Recursively handle nested dicts
                    env_privilege_obs[key] = extract_env_from_dict(value, env_id)
                else:
                    env_privilege_obs[key] = value

            env_obs = {
                "policy_obs": env_policy_obs,
                "privilege_obs": env_privilege_obs,
            }

            # Convert reward, terminated, truncated (tensors)
            env_reward = (
                reward[env_id]
                if reward is not None and isinstance(reward, torch.Tensor)
                else None
            )
            env_terminated = (
                terminated[env_id]
                if terminated is not None and isinstance(terminated, torch.Tensor)
                else None
            )
            env_truncated = (
                truncated[env_id]
                if truncated is not None and isinstance(truncated, torch.Tensor)
                else None
            )

            # Convert info (may be dict or list)
            env_info_dict = {}
            if info:
                # Keep original handling for list-type info (rare, but supported)
                if isinstance(info, list):
                    # info is a list [info_env_0, info_env_1, ...]
                    if env_id < len(info):
                        env_info_dict = info[env_id] if info[env_id] is not None else {}
                    else:
                        env_info_dict = {}
                # Main case: info is a dict (e.g. ReachEnv.get_info)
                elif isinstance(info, dict):
                    for key, value in info.items():
                        # Special handling for "state" from ReachEnv.get_info or TestBallAndBlockEnv
                        # state is a nested dict with per-env tensors; we want per-env state
                        if key == "state":
                            if isinstance(value, dict):
                                env_info_dict[key] = extract_env_from_dict(
                                    value, env_id
                                )
                            elif isinstance(value, list):
                                # If state is a list, it should be [env_0_state, env_1_state, ...]
                                # where each env_i_state is a dict containing that env's state
                                if env_id < len(value):
                                    env_info_dict[key] = value[env_id]
                                else:
                                    env_info_dict[key] = {}
                            else:
                                env_info_dict[key] = value
                            continue

                        # Default handling for other keys (same as original logic)
                        if isinstance(value, torch.Tensor):
                            # Extract single env from batched tensor
                            env_info_dict[key] = value[env_id]
                        elif isinstance(value, dict):
                            # Recursively handle nested dicts
                            env_info_dict[key] = extract_env_from_dict(value, env_id)
                        else:
                            env_info_dict[key] = value
                else:
                    env_info_dict = {}

            # Combine into per-env info (matching env.step() return format)
            per_env_info[env_id] = {
                "obs": env_obs,
                "reward": env_reward,
                "terminated": env_terminated,
                "truncated": env_truncated,
                "info": env_info_dict,
            }

        return per_env_info

    def reset(self, info: Dict[str, Any]):
        # Convert batched env_info to per-env format
        batched_env_info = info["env_info"]
        # Check if env_info is a tuple (obs, info) from reset() or a dict from step()
        assert isinstance(batched_env_info, tuple), "env_info must be a tuple"
        assert len(batched_env_info) == 2, "env_info must be a tuple of length 2"
        # reset() returns (obs, info) tuple
        obs_dict, info_list = batched_env_info
        # Convert to dict format for processing
        batched_env_info = {
            "obs": obs_dict,
            "info": info_list,  # info is a list [None, None, ...]
        }

        # Process all environments for reset
        env_ids = list(range(self.num_envs))
        per_env_info = self._convert_batched_env_info_to_per_env(
            batched_env_info, env_ids
        )

        for env_id in env_ids:
            if len(self.record_obs_buffer) <= env_id:
                self.record_obs_buffer.append([])
            if len(self.record_action_buffer) <= env_id:
                self.record_action_buffer.append([])
            if len(self.record_collect_buffer) <= env_id:
                self.record_collect_buffer.append([])

            # When resetting, clear buffers and start fresh with the reset obs
            # This ensures buffer lengths are consistent (obs buffer has 1 more than action/collect buffers)
            self.record_obs_buffer[env_id] = []
            self.record_action_buffer[env_id] = []
            self.record_collect_buffer[env_id] = []
            self.record_obs_buffer[env_id].append(per_env_info[env_id])
            # Reset saved flag for new trajectory
            if env_id < len(self._saved_for_completed_task):
                self._saved_for_completed_task[env_id] = False

    def _convert_collect_info_to_per_env(
        self, collect_info: Dict[str, Any], env_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Convert collect_info to per-env format.
        collect_info is a dict of {key: value}
        value is a dict of {env_id: value}
        return a list of {key: value}
        """
        per_env_info = {}
        for env_id in env_ids:
            cur_env_info = {}
            for key, value in collect_info.items():
                cur_env_info[key] = value[env_id]
            per_env_info[env_id] = cur_env_info
        return per_env_info

    def step(self, info: Dict[str, Any], ready_env_ids: List[int]):
        # We only record the info for ready_env_ids
        batched_env_info = info["env_info"]
        collect_info = info.copy()
        collect_info.pop("env_info")
        assert isinstance(batched_env_info, tuple), "env_info must be a tuple"
        assert len(batched_env_info) == 5, "env_info must be a tuple of length 5"
        obs_dict, reward_tensor, terminated_tensor, truncated_tensor, info_dict = (
            batched_env_info
        )
        # Convert to dict format for processing
        batched_env_info = {
            "obs": obs_dict,
            "reward": reward_tensor,
            "terminated": terminated_tensor,
            "truncated": truncated_tensor,
            "info": info_dict,
        }
        per_env_info = self._convert_batched_env_info_to_per_env(
            batched_env_info, ready_env_ids
        )
        per_collect_info = self._convert_collect_info_to_per_env(
            collect_info, ready_env_ids
        )
        for env_id in ready_env_ids:
            assert len(self.record_obs_buffer[env_id]) - 1 == len(
                self.record_action_buffer[env_id]
            ), (
                f"Record Obs Buffer length{len(self.record_obs_buffer[env_id])}, Action Buffer length{len(self.record_action_buffer[env_id])}"
            )
            assert len(self.record_obs_buffer[env_id]) - 1 == len(
                self.record_collect_buffer[env_id]
            ), (
                f"Record Obs Buffer length{len(self.record_obs_buffer[env_id])}, Collect Buffer length{len(self.record_collect_buffer[env_id])}"
            )
            self.record_obs_buffer[env_id].append(per_env_info[env_id])

            # Merge last_action and last_camera_action for action.json
            action_data = per_env_info[env_id]["obs"]["policy_obs"].get(
                "last_action", {}
            )

            self.record_action_buffer[env_id].append(action_data)
            self.record_collect_buffer[env_id].append(per_collect_info[env_id])

    def update(self, info: Dict[str, Any]) -> List[int]:
        """Update record state; save trajectories for completed tasks. Returns env_ids that were saved this call."""
        saved_env_ids: List[int] = []
        for env_id in range(self.num_envs):
            ac_info = info["auto_collect_info"][env_id]
            if ac_info is None:
                continue
            if ac_info["state"].split(":")[0] == "success" and ac_info["finished"]:
                # Only save once per task completion to avoid multiple saves
                if not self._saved_for_completed_task[env_id]:
                    # Should add write to disk or save to buffer here
                    self.save_to_disk(env_id)
                    saved_env_ids.append(env_id)
                    new_obs = self.record_obs_buffer[env_id][-1]
                    self.record_obs_buffer[env_id] = []
                    self.record_obs_buffer[env_id].append(new_obs)
                    self.record_action_buffer[env_id] = []
                    self.record_collect_buffer[env_id] = []
                    # Mark as saved to prevent multiple saves for the same task completion
                    self._saved_for_completed_task[env_id] = True

            elif (
                ac_info["state"].split(":")[0] == "truncated"
                or ac_info["state"].split(":")[0] == "failed"
                and not ac_info["finished"]
            ):
                print(f"Env {env_id} failed, We will flush our buffer now")
                new_obs = self.record_obs_buffer[env_id][-1]
                self.record_obs_buffer[env_id] = []
                self.record_obs_buffer[env_id].append(new_obs)
                self.record_action_buffer[env_id] = []
                self.record_collect_buffer[env_id] = []
                # Reset saved flag on failure
                self._saved_for_completed_task[env_id] = False
            else:
                # Reset the saved flag when task is not finished (new task started)
                if not ac_info.get("finished", False):
                    self._saved_for_completed_task[env_id] = False
        return saved_env_ids

    def save_to_disk(self, env_id: int):
        """Save trajectory data to disk.

        Creates a folder structure:
        {OUTPUT_PATH}/{trajectory_id}/
            action/
            collect/
        """
        # Get trajectory data
        action_buffer = self.record_action_buffer[env_id]
        collect_buffer = self.record_collect_buffer[env_id]
        assert len(action_buffer) == len(collect_buffer), (
            f"Action buffer length {len(action_buffer)}, collect buffer length {len(collect_buffer)}"
        )
        assert len(action_buffer) == len(self.record_obs_buffer[env_id]) - 1, (
            f"Action buffer length {len(action_buffer)}, obs buffer length {len(self.record_obs_buffer[env_id])}"
        )

        if len(action_buffer) == 0:
            self.logger.warning(f"No action data to save for env {env_id}")
            return

        # Create trajectory folder with incremental ID
        trajectory_id = self.trajectory_id
        self.trajectory_id += 1
        trajectory_dir = os.path.join(self.output_path, str(trajectory_id))

        # Create subdirectories
        action_dir = os.path.join(trajectory_dir, "action")
        collect_dir = os.path.join(trajectory_dir, "collect")
        env_dir = os.path.join(trajectory_dir, "env")
        info_dir = os.path.join(env_dir, "info")
        camera_dir = os.path.join(env_dir, "camera")
        os.makedirs(action_dir, exist_ok=True)
        os.makedirs(collect_dir, exist_ok=True)
        os.makedirs(info_dir, exist_ok=True)
        os.makedirs(camera_dir, exist_ok=True)

        # Create backend for saving (paths are relative to trajectory_dir)
        backend = BackendDispatch(output_dir=trajectory_dir)

        if self.save_in_video:
            # Save action data as single merged JSON with frame_id as outer key
            action_merged = {
                f"frame_{step_idx:04d}": to_serializable(action_data)
                for step_idx, action_data in enumerate(action_buffer)
            }
            backend.schedule(
                F.write_json,
                data=action_merged,
                path=os.path.join("action", "action_merged.json"),
                indent=JSON_INDENT,
            )

            # Save collect data as single merged JSON with frame_id as outer key
            collect_merged = {
                f"frame_{step_idx:04d}": to_serializable(collect_data)
                for step_idx, collect_data in enumerate(collect_buffer)
            }
            backend.schedule(
                F.write_json,
                data=collect_merged,
                path=os.path.join("collect", "collect_merged.json"),
                indent=JSON_INDENT,
            )
        else:
            # Save action data
            for step_idx, action_data in enumerate(action_buffer):
                # Convert action data to serializable format
                serializable_action = to_serializable(action_data)
                rel_path = os.path.join("action", f"action_{step_idx:04d}.json")
                backend.schedule(
                    F.write_json,
                    data=serializable_action,
                    path=rel_path,
                    indent=JSON_INDENT,
                )

            # Save collect data
            for step_idx, collect_data in enumerate(collect_buffer):
                # Convert collect data to serializable format
                serializable_collect = to_serializable(collect_data)
                rel_path = os.path.join("collect", f"collect_{step_idx:04d}.json")
                backend.schedule(
                    F.write_json,
                    data=serializable_collect,
                    path=rel_path,
                    indent=JSON_INDENT,
                )

        # Save env info (obs / reward / terminated / truncated / info + camera data)
        self._save_env_info(env_id, backend, trajectory_dir)

        self.logger.info(
            f"Saved trajectory {trajectory_id} for env {env_id} to {trajectory_dir}"
        )

    def _save_env_info(
        self,
        env_id: int,
        backend: BackendDispatch,
        trajectory_dir: str,
    ):
        """Save per-step env info for a given env.

        Folder structure under trajectory_dir:
            env/
                camera/
                    cam_0/
                        <annotator_type>/step_0000.*
                        ...
                    cam_1/
                        ...
                info/
                    env_<env_id>_step_0000.json (or env_<env_id>_merged.json if save_in_video)
                    env_<env_id>_step_0001.json
                    ...
        """
        env_infos = self.record_obs_buffer[env_id]

        # Ensure base folders exist
        env_base = os.path.join(trajectory_dir, "env")
        os.makedirs(os.path.join(env_base, "info"), exist_ok=True)
        os.makedirs(os.path.join(env_base, "camera"), exist_ok=True)

        if self.save_in_video:
            # Collect all camera info first for video mode
            all_camera_info = []
            valid_step_indices = []
            for step_idx, env_info in enumerate(env_infos):
                if step_idx == len(env_infos) - 1:
                    continue  # We do not save the last step obs because it is the new obs
                obs = env_info.get("obs", {})
                policy_obs = obs.get("policy_obs", {})
                camera_info = policy_obs.get("camera_info", {})
                all_camera_info.append(camera_info)
                valid_step_indices.append(step_idx)

            # Save all camera data in video mode
            camera_paths_merged = self._save_camera_info_in_video_mode(
                all_camera_info=all_camera_info,
                backend=backend,
                trajectory_dir=trajectory_dir,
                env_id=env_id,
            )

            # Build merged env_info structure
            env_info_merged = {}
            for step_idx, env_info in zip(valid_step_indices, env_infos):
                obs = env_info.get("obs", {})
                policy_obs = obs.get("policy_obs", {})
                privilege_obs = obs.get("privilege_obs", {})

                # Build obs structure with camera_info replaced by paths
                policy_obs_serializable = {}
                for key, value in policy_obs.items():
                    if key == "camera_info":
                        # Use merged camera paths for this step
                        policy_obs_serializable["camera_info"] = camera_paths_merged
                    else:
                        policy_obs_serializable[key] = to_serializable(value)

                obs_serializable = {
                    "policy_obs": policy_obs_serializable,
                    "privilege_obs": to_serializable(privilege_obs),
                }

                # Other fields can be directly serialized
                reward_serializable = to_serializable(env_info.get("reward", None))
                terminated_serializable = to_serializable(
                    env_info.get("terminated", None)
                )
                truncated_serializable = to_serializable(
                    env_info.get("truncated", None)
                )
                info_serializable = to_serializable(env_info.get("info", {}))

                env_info_serializable = {
                    "obs": obs_serializable,
                    "reward": reward_serializable,
                    "terminated": terminated_serializable,
                    "truncated": truncated_serializable,
                    "info": info_serializable,
                }

                env_info_merged[f"frame_{step_idx:04d}"] = env_info_serializable

            # Save as single merged JSON
            rel_path = os.path.join("env", "info", f"env_{env_id}_merged.json")
            backend.schedule(
                F.write_json,
                data=env_info_merged,
                path=rel_path,
                indent=JSON_INDENT,
            )
        else:
            # Original per-step saving logic
            for step_idx, env_info in enumerate(env_infos):
                if step_idx == len(env_infos) - 1:
                    continue  # We do not save the last step obs because it is the new obs
                obs = env_info.get("obs", {})
                policy_obs = obs.get("policy_obs", {})
                privilege_obs = obs.get("privilege_obs", {})

                camera_info = policy_obs.get("camera_info", {})

                # Save camera data and get relative paths structure
                camera_paths = self._save_camera_info_for_step(
                    camera_info=camera_info,
                    step_idx=step_idx,
                    backend=backend,
                    trajectory_dir=trajectory_dir,
                    env_id=env_id,
                )

                # Build obs structure with camera_info replaced by paths
                policy_obs_serializable = {}
                for key, value in policy_obs.items():
                    if key == "camera_info":
                        policy_obs_serializable["camera_info"] = camera_paths
                    else:
                        policy_obs_serializable[key] = to_serializable(value)

                obs_serializable = {
                    "policy_obs": policy_obs_serializable,
                    "privilege_obs": to_serializable(privilege_obs),
                }

                # Other fields can be directly serialized
                reward_serializable = to_serializable(env_info.get("reward", None))
                terminated_serializable = to_serializable(
                    env_info.get("terminated", None)
                )
                truncated_serializable = to_serializable(
                    env_info.get("truncated", None)
                )
                info_serializable = to_serializable(env_info.get("info", {}))

                env_info_serializable = {
                    "obs": obs_serializable,
                    "reward": reward_serializable,
                    "terminated": terminated_serializable,
                    "truncated": truncated_serializable,
                    "info": info_serializable,
                }

                # Save as JSON (relative path under trajectory_dir)
                rel_path = os.path.join(
                    "env", "info", f"env_{env_id}_step_{step_idx:04d}.json"
                )
                backend.schedule(
                    F.write_json,
                    data=env_info_serializable,
                    path=rel_path,
                    indent=JSON_INDENT,
                )

    def _save_camera_info_in_video_mode(
        self,
        all_camera_info: List[Dict[str, Any]],
        backend: BackendDispatch,
        trajectory_dir: str,
        env_id: int,
    ) -> Dict[str, Any]:
        """Save camera_info in video mode: images as videos, JSONs/numpy merged.

        Args:
            all_camera_info: List of camera_info dicts, one per step
            backend: Backend for saving
            trajectory_dir: Trajectory directory
            env_id: Environment ID

        Returns:
            Dictionary with same structure as camera_info but with paths to videos/merged files
        """
        if not all_camera_info or not all_camera_info[0]:
            return {}

        # Infer number of cameras from first annotator of first step
        first_step_camera_info = all_camera_info[0]
        first_value = next(iter(first_step_camera_info.values()))
        if not isinstance(first_value, list):
            # Unexpected format, just serialize to JSON-compatible structure
            return to_serializable(first_step_camera_info)

        num_cams = len(first_value)
        num_steps = len(all_camera_info)

        # Prepare output structure
        camera_paths: Dict[str, Any] = {
            annotator_name: [None] * num_cams
            for annotator_name in first_step_camera_info.keys()
        }

        # Process each camera and annotator
        for cam_id in range(num_cams):
            cam_base_rel = os.path.join("env", "camera", f"cam_{cam_id}")
            cam_base_abs = os.path.join(trajectory_dir, cam_base_rel)
            os.makedirs(cam_base_abs, exist_ok=True)

            for annotator_name in first_step_camera_info.keys():
                annotator_dir_rel = os.path.join(cam_base_rel, annotator_name)
                annotator_dir_abs = os.path.join(trajectory_dir, annotator_dir_rel)
                os.makedirs(annotator_dir_abs, exist_ok=True)

                # Collect all frames for this annotator
                all_frames_data = []
                all_frames_json = []
                all_frames_numpy = []

                for step_idx, camera_info in enumerate(all_camera_info):
                    if cam_id >= len(camera_info.get(annotator_name, [])):
                        continue

                    anno_data = camera_info[annotator_name][cam_id]
                    payload = convert_annotator_data_to_payload(
                        annotator_name, anno_data
                    )

                    # For camera_params, add local pose information
                    if annotator_name == "camera_params":
                        payload = self._add_local_pose_to_camera_params(
                            payload, env_id, cam_id
                        )

                    # Determine annotator type and collect data
                    if is_image_annotator(annotator_name):
                        # Collect image data for video
                        img_data = extract_image_data(annotator_name, payload)
                        if img_data is not None:
                            all_frames_data.append(img_data)
                        # Also collect JSON metadata if present (e.g., idToLabels for segmentation)
                        json_metadata = extract_json_metadata(annotator_name, payload)
                        if json_metadata:
                            all_frames_json.append((step_idx, json_metadata))
                    elif is_json_annotator(annotator_name):
                        # Collect JSON data for merging
                        json_data = extract_json_data(annotator_name, payload)
                        if json_data is not None:
                            all_frames_json.append((step_idx, json_data))
                    elif is_numpy_annotator(annotator_name):
                        # Collect numpy data and JSON metadata for merging
                        numpy_data = extract_numpy_data(annotator_name, payload)
                        if numpy_data is not None:
                            all_frames_numpy.append((step_idx, numpy_data))

                # Save based on annotator type
                if all_frames_data:
                    # Save as video
                    video_path = os.path.join(
                        annotator_dir_rel, f"{annotator_name}.mp4"
                    )
                    backend.schedule(
                        F.write_mp4,
                        data=all_frames_data,
                        path=video_path,
                        fps=30.0,  # Default fps, can be made configurable
                    )
                    # If there's JSON metadata (e.g., labels for segmentation), save it separately
                    if all_frames_json:
                        merged_metadata = {
                            f"frame_{step_idx:04d}": json_data
                            for step_idx, json_data in all_frames_json
                        }
                        metadata_path = os.path.join(
                            annotator_dir_rel, f"{annotator_name}_metadata_merged.json"
                        )
                        backend.schedule(
                            F.write_json,
                            data=merged_metadata,
                            path=metadata_path,
                            indent=JSON_INDENT,
                        )
                    camera_paths[annotator_name][cam_id] = video_path
                elif all_frames_json:
                    # Merge JSONs with frame_id as outer key
                    merged_json = {
                        f"frame_{step_idx:04d}": json_data
                        for step_idx, json_data in all_frames_json
                    }
                    json_path = os.path.join(
                        annotator_dir_rel, f"{annotator_name}_merged.json"
                    )
                    backend.schedule(
                        F.write_json,
                        data=merged_json,
                        path=json_path,
                        indent=JSON_INDENT,
                    )
                    camera_paths[annotator_name][cam_id] = json_path
                elif all_frames_numpy:
                    # Merge numpy arrays with frame_id as outer key
                    merged_numpy = {
                        f"frame_{step_idx:04d}": to_serializable(numpy_data)
                        for step_idx, numpy_data in all_frames_numpy
                    }
                    json_path = os.path.join(
                        annotator_dir_rel, f"{annotator_name}_merged.json"
                    )
                    backend.schedule(
                        F.write_json,
                        data=merged_numpy,
                        path=json_path,
                        indent=JSON_INDENT,
                    )
                    camera_paths[annotator_name][cam_id] = json_path

        return camera_paths

    def _save_camera_info_for_step(
        self,
        camera_info: Dict[str, Any],
        step_idx: int,
        backend: BackendDispatch,
        trajectory_dir: str,
        env_id: int,
    ) -> Dict[str, Any]:
        """Save camera_info for a single step and return same-structured paths.

        Input format (per env, from _convert_batched_env_info_to_per_env):
            camera_info: {annotator_name: [per_cam_data_0, per_cam_data_1, ...]}

        Output format:
            {annotator_name: [relative_folder_cam0, relative_folder_cam1, ...]}
            where each folder is: camera/cam_{id}/{annotator_name}/
        """
        if not camera_info:
            return {}

        # Infer number of cameras from first annotator
        first_value = next(iter(camera_info.values()))
        if not isinstance(first_value, list):
            # Unexpected format, just serialize to JSON-compatible structure
            return to_serializable(camera_info)

        num_cams = len(first_value)

        # Prepare output structure with same keys / list lengths
        camera_paths: Dict[str, Any] = {
            annotator_name: [None] * num_cams for annotator_name in camera_info.keys()
        }

        for cam_id in range(num_cams):
            cam_base_rel = os.path.join("env", "camera", f"cam_{cam_id}")
            cam_base_abs = os.path.join(trajectory_dir, cam_base_rel)
            os.makedirs(cam_base_abs, exist_ok=True)

            for annotator_name, per_cam_list in camera_info.items():
                if cam_id >= len(per_cam_list):
                    continue

                anno_data = per_cam_list[cam_id]
                # Each annotator has its own folder under the camera
                annotator_dir_rel = os.path.join(cam_base_rel, annotator_name)
                annotator_dir_abs = os.path.join(trajectory_dir, annotator_dir_rel)
                os.makedirs(annotator_dir_abs, exist_ok=True)

                # Convert anno_data to payload format matching TiledCaptureManager logic
                # anno_data from annotator.get_data() may have {"data": ..., "info": {...}} structure
                payload = convert_annotator_data_to_payload(annotator_name, anno_data)

                # For camera_params, add local pose information
                if annotator_name == "camera_params":
                    payload = self._add_local_pose_to_camera_params(
                        payload, env_id, cam_id
                    )

                # Delegate per-annotator writing to shared helper (MyWriter-style)
                write_annotator_step(
                    backend=backend,
                    annotator_name=annotator_name,
                    payload=payload,
                    annotator_dir_rel=annotator_dir_rel,
                    step_idx=step_idx,
                )

                # In camera_info, store the annotator folder path (relative)
                camera_paths[annotator_name][cam_id] = annotator_dir_rel

        return camera_paths

    def _add_local_pose_to_camera_params_at_record_time(
        self, camera_params_list: List[Any], env_id: int
    ) -> List[Any]:
        """Add local pose to camera_params at recording time (for each camera).

        This ensures each step records its own pose, not the final pose when saving.
        camera_params_list format: [cam_0_data, cam_1_data, ...]
        """
        try:
            camera_manager = getattr(self.env, "camera_manager", None) or getattr(
                self.env.scene, "camera_manager", None
            )
            if camera_manager is None:
                return camera_params_list

            result_list = []
            for cam_id, cam_params in enumerate(camera_params_list):
                if cam_params is None:
                    result_list.append(None)
                    continue

                # Convert to dict if needed
                if isinstance(cam_params, dict):
                    payload = cam_params.copy()
                else:
                    payload = {"data": cam_params}

                # Get local pose for this camera at recording time
                camera_xform = camera_manager.cameras_xform[env_id][cam_id]
                local_pos, local_quat = camera_xform.get_local_pose()

                # Add local pose to payload
                payload["pos_local"] = to_serializable(local_pos)
                payload["ori_local"] = to_serializable(local_quat)

                result_list.append(payload)

            return result_list
        except Exception:
            # If anything fails, return original list
            return camera_params_list

    def _add_local_pose_to_camera_params(
        self, payload: Dict[str, Any], env_id: int, cam_id: int
    ) -> Dict[str, Any]:
        """Add local pose (pos and ori relative to parent prim) to camera_params payload.

        Note: This is called during save time. If pos_local and ori_local already exist
        (from recording time), they will be preserved. Otherwise, current pose is used.
        """
        # If pos_local and ori_local already exist (from recording time), keep them
        if "pos_local" in payload and "ori_local" in payload:
            return payload

        # Otherwise, get current pose (fallback for backward compatibility)
        try:
            camera_manager = getattr(self.env, "camera_manager", None) or getattr(
                self.env.scene, "camera_manager", None
            )
            if camera_manager is None:
                return payload

            camera_xform = camera_manager.cameras_xform[env_id][cam_id]
            local_pos, local_quat = camera_xform.get_local_pose()

            # Use existing serialization method
            payload["pos_local"] = to_serializable(local_pos)
            payload["ori_local"] = to_serializable(local_quat)
        except Exception:
            pass

        return payload

    def get_record_buffer(self):
        return self.record_obs_buffer, self.record_action_buffer
