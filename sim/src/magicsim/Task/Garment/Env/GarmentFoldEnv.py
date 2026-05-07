from typing import Any, Dict, Sequence

import numpy as np
import torch
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
import gymnasium as gym


class GarmentFoldEnv(TaskBaseEnv):
    """
    Garment Folding Environment with dual Franka arms.

    Two Franka robots face each other with a garment on a table between them.
    The task is to fold the garment by bringing keypoints together.
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)
        self.garment_category = "garment_items"
        self.timeout_steps = getattr(config, "timeout_steps", 10000)

        # Store initial coverage (unfolded area) for ratio computation
        self._initial_coverage = None

        # Recompute the ConvexHull-based coverage only every N steps per env.
        self._coverage_interval = int(getattr(config, "coverage_interval", 20))
        self._coverage_cache: list[float | None] = [None] * self.num_envs
        self._coverage_last_step: list[int] = [-self._coverage_interval] * self.num_envs

        # Per-step cache of get_done results so reward/termination/state share
        # a single expensive computation instead of running it three times.
        self._done_cache: list[dict | None] = [None] * self.num_envs
        self._done_cache_step: list[int] = [-1] * self.num_envs

    def _get_garment(self, env_id: int):
        """Get the first garment object for an environment."""
        garment_dict = self.scene.scene_manager.garment_objects[env_id]
        garment_list = garment_dict.get(self.garment_category, [])
        if garment_list:
            return garment_list[0]
        return None

    def get_keypoint_positions(self, env_id: int) -> dict:
        """Get current keypoint world positions for an environment."""
        garment = self._get_garment(env_id)
        if garment is None:
            return {}
        return garment.get_keypoint()

    def compute_coverage(self, env_id: int) -> float:
        """Compute garment coverage area using convex hull of projected mesh points.

        Returns the area of the 2D convex hull when projecting mesh points
        onto the horizontal (XY) plane.
        """
        garment = self._get_garment(env_id)
        if garment is None:
            return 0.0

        mesh_points, _, _, _ = garment.get_current_mesh_points()
        # Project onto XY plane (horizontal)
        points_2d = mesh_points[:, :2]

        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(points_2d)
            return float(hull.volume)  # In 2D, volume = area
        except Exception:
            # Fallback: bounding box area
            x_range = points_2d[:, 0].max() - points_2d[:, 0].min()
            y_range = points_2d[:, 1].max() - points_2d[:, 1].min()
            return float(x_range * y_range)

    def compute_coverage_ratio(self, env_id: int) -> float:
        """Compute the ratio of current coverage to initial (unfolded) coverage.

        Returns value in [0, 1]. Lower means more folded. Throttled to only
        rerun ConvexHull every ``self._coverage_interval`` steps per env.
        """
        if self._initial_coverage is None:
            return 1.0

        initial = self._initial_coverage[env_id]
        if initial <= 0:
            return 1.0

        current_step = int(self.episode_length_buf[env_id].item())
        if (
            self._coverage_cache[env_id] is None
            or current_step - self._coverage_last_step[env_id]
            >= self._coverage_interval
        ):
            self._coverage_cache[env_id] = self.compute_coverage(env_id)
            self._coverage_last_step[env_id] = current_step

        return float(self._coverage_cache[env_id] / initial)

    def get_done(self, env_id: int) -> dict:
        """Evaluate fold completion for one environment.

        For Tops: checks that bottom keypoints are close to shoulder keypoints
        (i.e., the garment is folded in half vertically).

        Result is cached per (env_id, episode_length_buf) so reward/termination/
        state share a single computation within the same TaskBaseEnv.step call.

        Returns:
            dict with:
                - "is_done": bool
                - "keypoint_distance": float, mean distance between paired keypoints
                - "coverage_ratio": float, current/initial area ratio
        """
        current_step = int(self.episode_length_buf[env_id].item())
        if (
            self._done_cache[env_id] is not None
            and self._done_cache_step[env_id] == current_step
        ):
            return self._done_cache[env_id]

        kp = self.get_keypoint_positions(env_id)
        if not kp:
            result = {
                "is_done": False,
                "keypoint_distance": float("inf"),
                "coverage_ratio": 1.0,
            }
            self._done_cache[env_id] = result
            self._done_cache_step[env_id] = current_step
            return result

        # Fold in half: bottom hem should be brought up to the shoulders
        distances = []
        if "bottom_left" in kp and "left_shoulder" in kp:
            distances.append(
                float(np.linalg.norm(kp["bottom_left"] - kp["left_shoulder"]))
            )
        if "bottom_right" in kp and "right_shoulder" in kp:
            distances.append(
                float(np.linalg.norm(kp["bottom_right"] - kp["right_shoulder"]))
            )

        mean_dist = float(np.mean(distances)) if distances else float("inf")
        coverage_ratio = self.compute_coverage_ratio(env_id)

        # Done if bottom is close to shoulders and coverage reduced
        fold_threshold = 0.05  # 5cm
        coverage_threshold = 0.7  # coverage reduced to 70%
        is_done = mean_dist < fold_threshold and coverage_ratio < coverage_threshold

        result = {
            "is_done": is_done,
            "keypoint_distance": mean_dist,
            "coverage_ratio": coverage_ratio,
        }
        self._done_cache[env_id] = result
        self._done_cache_step[env_id] = current_step
        return result

    # ----- TaskBaseEnv interface -----

    def get_obs_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

    def get_policy_obs(self) -> Dict[str, Any]:
        robot_state = self.scene.robot_manager.get_robot_state()
        camera_info = self.scene.capture_manager.step()
        return {"robot_state": robot_state, "camera_info": camera_info}

    def get_privilege_obs(self) -> Dict[str, Any]:
        per_env = []
        max_k = 0
        for env_id in range(self.num_envs):
            kp_dict = self.get_keypoint_positions(env_id)
            if kp_dict:
                arr = np.stack(list(kp_dict.values()), axis=0).astype(np.float32)
            else:
                arr = np.zeros((0, 3), dtype=np.float32)
            per_env.append(arr)
            max_k = max(max_k, arr.shape[0])

        if max_k == 0:
            max_k = 1
        stacked = np.zeros((self.num_envs, max_k, 3), dtype=np.float32)
        for i, arr in enumerate(per_env):
            if arr.shape[0] > 0:
                stacked[i, : arr.shape[0]] = arr
        return {
            "garment_keypoints": torch.from_numpy(stacked).to(self.device),
        }

    def process_action(self, action: torch.Tensor | list[Dict]):
        if action is None:
            return None
        # Dual arm: 14D arm + 2D eef = 16D
        if action.shape[1] == 14:
            action = torch.cat(
                [action, torch.ones((action.shape[0], 2), device=self.device)],
                dim=1,
            )
        return action

    def get_eef_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        robot_state = self.scene.robot_manager.get_robot_state()[0]
        state_dict = list(robot_state.values())[0]
        eef_pos = state_dict["eef_pos"]
        eef_quat = state_dict["eef_quat"]
        eef_pose = torch.cat([eef_pos, eef_quat], dim=-1)
        return eef_pose[env_ids]

    def get_reward(
        self,
        action: torch.Tensor | list[Dict],
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        rewards = []
        for env_id in range(self.num_envs):
            done_info = self.get_done(env_id)
            # Reward: negative keypoint distance + bonus for completion
            r = -done_info["keypoint_distance"]
            if done_info["is_done"]:
                r += 10.0
            rewards.append(r)
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def get_termination(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for env_id in range(self.num_envs):
            done_info = self.get_done(env_id)
            if done_info["is_done"]:
                terminated[env_id] = True

        # Truncate on timeout
        truncated = self.episode_length_buf >= self.timeout_steps
        return terminated, truncated

    def get_info(self) -> Dict[str, Any]:
        state = self.get_state()
        description = self.get_description()
        return {"state": state, "description": description}

    def get_description(self) -> str:
        return "Fold the garment using two Franka arms"

    def get_state(self) -> Dict[str, Any]:
        robot_state = self.scene.robot_manager.get_robot_state()
        done_infos = [self.get_done(i) for i in range(self.num_envs)]
        state = {
            "robot_state": robot_state,
            "scene_state": {
                "done_info": done_infos,
            },
            "camera_state": self.scene.camera_manager.get_all_camera_state(),
        }
        return state

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Record initial coverage after reset
        self._initial_coverage = [
            self.compute_coverage(i) for i in range(self.num_envs)
        ]
        # Invalidate per-step caches so stale values from super().reset()'s
        # get_info calls (before _initial_coverage was set) are not reused.
        self._coverage_cache = [None] * self.num_envs
        self._coverage_last_step = [-self._coverage_interval] * self.num_envs
        self._done_cache = [None] * self.num_envs
        self._done_cache_step = [-1] * self.num_envs
        return obs, info
