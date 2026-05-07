from typing import Any, Dict, List, Sequence
import torch
from magicsim.Collect.CameraGlobalPlanner.CameraGlobalPlanner import (
    CameraGlobalPlanner,
)
from magicsim.Env.Utils.file import Logger
from omegaconf import DictConfig
from magicsim.Collect.CameraGlobalPlanner.NavTo import NavTo
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv


class CameraGlobalPlannerManager:
    """
    Global Planner Manager for camera tasks.
    Manages camera global planners for all environments.
    """

    def __init__(
        self,
        env: TaskBaseEnv,
        num_envs: int,
        camera_global_planner_config: DictConfig,
        device=torch.device("cpu"),
        logger: Logger = None,
    ):
        self.num_envs = num_envs
        self.env = env
        self.camera_global_planner_config = camera_global_planner_config
        self.device = device
        self.logger = logger
        self.camera_global_planner_list: list[CameraGlobalPlanner] = [None] * num_envs
        self.camera_global_planner_type_list: list[str] = [None] * num_envs
        self.info_list: List[Dict[str, Any]] = [None] * self.num_envs

    def create_camera_global_planner(
        self, camera_global_planner_type: str, env_id: int
    ):
        """
        Create a camera global planner instance for a specific environment.

        Args:
            camera_global_planner_type: Type of planner to create (e.g., "NavTo")
            env_id: Environment ID
        """
        if camera_global_planner_type == "NavTo":
            self.camera_global_planner_list[env_id] = NavTo(
                self.camera_global_planner_config.NavTo,
                self.env,
                env_id,
                self.logger,
            )
            self.camera_global_planner_type_list[env_id] = "NavTo"
        else:
            raise ValueError(
                f"Camera global planner type {camera_global_planner_type} not supported."
            )

    def step(
        self, actions: List[Dict[str, Any]], env_ids: Sequence[int]
    ) -> tuple[Dict[str, Dict[str, torch.Tensor]] | None, List[int], List[int]]:
        """
        Step all camera global planners for the given environments.

        Args:
            actions: List of action dictionaries, each containing:
                    {"NavTo": {"camera_name": str, "target_pose": torch.Tensor [7]}} or None
            env_ids: List of environment IDs

        Returns:
            Tuple of (camera_action_dict, valid_env_ids, failed_env_ids)
            - camera_action_dict: Dict format {camera_name: {"env_ids": [...], "poses": tensor}} or None
            - valid_env_ids: List of environment IDs that successfully generated actions
            - failed_env_ids: List of environment IDs that failed to generate actions
        """
        output_action = []
        valid_env_ids = []
        failed_env_ids = []
        for i, env_id in enumerate(env_ids):
            action_spec = actions[i]
            # Align camera global planner logic with robot GlobalPlannerManager:
            # - None means no new command, keep using the current planner.
            # - If this env has no planner, create and reset.
            # - If the same planner type is running, refresh only when appropriate.
            # - If a different planner type is running, raise an error.
            if action_spec is not None:
                planner_type = list(action_spec.keys())[0]
                if self.camera_global_planner_type_list[env_id] is None:
                    payload = list(action_spec.values())[0]
                    camera_name = (
                        payload.get("camera_name")
                        if isinstance(payload, dict)
                        else None
                    )

                    # Before creating a new planner, ensure FlyingCamera planner is reset
                    # This ensures that when NavTo.reset() is called, it uses the correct
                    # camera position (after reset) for path planning
                    if (
                        hasattr(self.env.scene, "planner_manager")
                        and self.env.scene.planner_manager is not None
                    ):
                        planner_manager = self.env.scene.planner_manager
                        if hasattr(planner_manager, "camera_planners"):
                            if (
                                camera_name
                                and camera_name in planner_manager.camera_planners
                            ):
                                camera_planner = planner_manager.camera_planners[
                                    camera_name
                                ]
                                if camera_planner is not None:
                                    camera_planner.reset_idx([env_id])

                    self.create_camera_global_planner(planner_type, env_id)
                    self.camera_global_planner_list[env_id].reset(
                        list(action_spec.values())[0]
                    )
                else:
                    if planner_type == self.camera_global_planner_type_list[env_id]:
                        planner = self.camera_global_planner_list[env_id]
                        payload = list(action_spec.values())[0]
                        # New requirement:
                        # Only refresh the running planner when the camera has already
                        # reached the true NavTo target (i.e. NavTo.get_done() == True).
                        #
                        # This lets the current segmented Nav path be executed completely
                        # before re-planning towards a (possibly new) target above the cube.
                        refresh_allowed = False
                        if hasattr(planner, "get_done"):
                            try:
                                refresh_allowed = bool(planner.get_done())
                            except Exception:
                                # If get_done fails for any reason, fall back to not refreshing
                                refresh_allowed = False
                        # If planner has no get_done method, we keep the old behavior and always refresh.
                        else:
                            refresh_allowed = True

                        if refresh_allowed:
                            planner.refresh(payload)
                    else:
                        raise RuntimeError(
                            f"Camera global planner {self.camera_global_planner_type_list[env_id]} "
                            f"is running, but new planner type {planner_type} is given for env {env_id}."
                        )
            if self.camera_global_planner_list[env_id] is None:
                output_action.append(None)
                continue
            try:
                action = self.camera_global_planner_list[env_id].step()
                if action is None:
                    failed_env_ids.append(env_id)
                else:
                    output_action.append(action)
                    valid_env_ids.append(env_id)
            except Exception as e:
                self.logger.error(
                    f"Error in camera global planner step for env {env_id}: {e}"
                ) if self.logger else None
                failed_env_ids.append(env_id)
                output_action.append(None)

        # Convert output_action list to unified format: {camera_name: {"env_ids": [...], "poses": tensor}}
        if len(valid_env_ids) == 0:
            return None, valid_env_ids, failed_env_ids

        camera_action_dict: Dict[str, Dict[str, Any]] = {}
        for i, action in enumerate(output_action):
            if action is not None and "camera_name" in action:
                camera_name = action["camera_name"]
                target_pose = action["target_pose"]
                if not isinstance(target_pose, torch.Tensor):
                    target_pose = torch.tensor(
                        target_pose, dtype=torch.float32, device=self.device
                    )
                else:
                    target_pose = target_pose.to(self.device)

                if camera_name not in camera_action_dict:
                    camera_action_dict[camera_name] = {
                        "env_ids": [],
                        "poses": [],
                    }

                # Get the actual env_id from env_ids (original input)
                env_id_value = (
                    int(env_ids[i])
                    if isinstance(env_ids, torch.Tensor)
                    else int(env_ids[i])
                )
                camera_action_dict[camera_name]["env_ids"].append(env_id_value)
                camera_action_dict[camera_name]["poses"].append(target_pose)

        # Stack poses for each camera
        for camera_name in list(camera_action_dict.keys()):
            payload = camera_action_dict[camera_name]
            if payload["poses"]:
                payload["poses"] = torch.stack(payload["poses"])
            else:
                del camera_action_dict[camera_name]

        return (
            camera_action_dict if camera_action_dict else None,
            valid_env_ids,
            failed_env_ids,
        )

    def update(self, info: Dict[str, Any]):
        """
        Update all camera global planners based on environment feedback.

        Args:
            info: Environment information dictionary

        Returns:
            List of planner info dictionaries for each environment
        """
        camera_atomic_info = info.get("camera_atomic_skill_info")
        for env_id in range(self.num_envs):
            if self.camera_global_planner_type_list[env_id] is None:
                self.info_list[env_id] = None
                continue

            # First, update the underlying camera global planner.
            planner_info = self.camera_global_planner_list[env_id].update(info)
            self.info_list[env_id] = planner_info

            # If the planner itself reports finished or truncated, clear it.
            should_clear = bool(
                planner_info.get("finished") or planner_info.get("truncated", 0) > 0
            )

            # Additionally, if the corresponding camera atomic skill (e.g., GoTo)
            # reports finished/truncated, we also clear the global planner so it
            # doesn't keep driving the camera after the skill is done.
            if (
                not should_clear
                and camera_atomic_info is not None
                and env_id < len(camera_atomic_info)
            ):
                skill_info = camera_atomic_info[env_id]
                if skill_info is not None and (
                    skill_info.get("finished") or skill_info.get("truncated", 0) > 0
                ):
                    should_clear = True

            if should_clear:
                self.camera_global_planner_type_list[env_id] = None
                self.camera_global_planner_list[env_id] = None
        return self.info_list

    def reset(self):
        """
        Reset all camera global planners.
        This clears all planner instances to prevent using old targets after reset.

        Returns:
            List of None values for each environment
        """
        # Clear all planner instances to ensure clean state
        self.camera_global_planner_list = [None] * self.num_envs
        self.camera_global_planner_type_list = [None] * self.num_envs
        self.info_list = [None] * self.num_envs
        return [None] * self.num_envs
