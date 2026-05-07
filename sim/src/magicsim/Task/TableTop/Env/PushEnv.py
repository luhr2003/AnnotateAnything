from typing import Any, Dict, Sequence

import torch
import gymnasium as gym

from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv


class PushEnv(TaskBaseEnv):
    """
    Push Environment for Robot Tasks.

    基本观测与 ReachEnv 相同：
    - policy_obs: robot_state / camera_info;
    - privilege_obs: cube_pose。

    终止逻辑（二选一，由配置决定）：
    - 若配置了 goal_position：以「目标物体到达目标位置」为准，即物体中心与 goal_position 距离 < object_goal_tolerance 时终止。
    - 否则沿用旧逻辑：以「末端到达目标位置」为准（目标 = cube 初始位置 + push_direction * push_distance）。
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)
        self.target_category: str = getattr(config, "target_category", "cube")

        # 物体目标位置 [x, y, z]；若配置则终止条件为「物体到达该位置」
        goal = getattr(config, "goal_position", None)
        if goal is not None:
            self.goal_position = torch.tensor(
                goal, dtype=torch.float32, device=self.device
            )
            if self.goal_position.dim() == 1:
                self.goal_position = self.goal_position.unsqueeze(0)
        else:
            self.goal_position = None
        self.object_goal_tolerance: float = float(
            getattr(config, "object_goal_tolerance", 0.03)
        )

        # 以下为旧逻辑（当未配置 goal_position 时使用）
        self.push_distance: float = float(getattr(config, "push_distance", 0.2))
        direction = getattr(config, "push_direction", [1.0, 0.0, 0.0])
        self.push_direction = torch.tensor(direction, dtype=torch.float32)
        if torch.norm(self.push_direction) > 0:
            self.push_direction = self.push_direction / torch.norm(self.push_direction)
        else:
            self.push_direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        self.eef_target_tolerance: float = float(
            getattr(config, "eef_target_tolerance", 0.04)
        )
        self.min_push_progress_ratio: float = float(
            getattr(config, "min_push_progress_ratio", 0.5)
        )
        self._cube_start_pos: torch.Tensor | None = None

    def get_obs_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

    def get_policy_obs(self) -> Dict[str, Any]:
        robot_state = self.scene.robot_manager.get_robot_state()
        camera_info = self.scene.capture_manager.step()
        return {"robot_state": robot_state, "camera_info": camera_info}

    def get_privilege_obs(self) -> Dict[str, Any]:
        cube_pose = self.get_cube_pose()
        return {"cube_pose": cube_pose}

    def get_cube_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(
                self.scene.robot_manager.num_envs, device=self.device
            )
        cube_pose = []
        for env_id in env_ids:
            e_id = int(env_id)
            scene_mgr = self.scene.scene_manager

            # 1）优先从刚体对象里找（physics.type == "dynamic" 的物体，例如当前的 mug）
            rigid_env = scene_mgr.rigid_objects[e_id]
            rigid_list = rigid_env.get(self.target_category, [])

            # 2）兼容旧逻辑：再从 geometry_objects 里找（如之前的 cube）
            geo_env = scene_mgr.geometry_objects[e_id]
            geo_list = geo_env.get(self.target_category, [])

            if rigid_list:
                target_obj = rigid_list[0]
            elif geo_list:
                target_obj = geo_list[0]
            else:
                # 调试信息，便于检查 Scene 配置问题
                print(
                    f"[PushEnv Debug] env_id={e_id} target_category={self.target_category} "
                    f"has no instances; rigid_keys={list(rigid_env.keys())}, "
                    f"geo_keys={list(geo_env.keys())}"
                )
                raise RuntimeError(
                    f"PushEnv: no objects found for target_category='{self.target_category}' "
                    f"in env_id={e_id}. 请检查 Scene.objects 中该类别是否创建了实例，"
                    f"以及 physics.type 是否为 'dynamic' 或 'geometry'。"
                )

            translation, orientation = target_obj.get_local_pose()
            cube_pose.append(torch.cat([translation, orientation], dim=0))
        return torch.stack(cube_pose, dim=0)

    def process_action(self, action: torch.Tensor | list[Dict]):
        # 与 ReachEnv 相同：末端 7D pose 自动补 1 维 gripper close
        if action is None:
            return None
        if isinstance(action, torch.Tensor) and action.shape[1] == 7:
            action = torch.cat(
                [action, torch.ones((action.shape[0], 1), device=self.device)], dim=1
            )
        return action

    def get_eef_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        robot_state = self.scene.robot_manager.get_robot_state()
        eef_pos = list(robot_state[0].values())[0]["eef_pos"]
        eef_quat = list(robot_state[0].values())[0]["eef_quat"]
        eef_pose = torch.cat([eef_pos, eef_quat], dim=1)
        return eef_pose[env_ids]

    def get_info(self) -> Dict[str, Any]:
        state = self.get_state()
        description = self.get_description()
        return {"state": state, "description": description}

    def get_description(self) -> Dict[str, Any]:
        return "Pushing the red cube with the end effector"

    def get_state(self) -> Dict[str, Any]:
        robot_state = self.scene.robot_manager.get_robot_state()
        cube_state = self.get_cube_pose()
        state = {
            "robot_state": robot_state,
            "scene_state": {
                "cube_pose": cube_state,
            },
            "camera_state": self.scene.camera_manager.get_all_camera_state(),
        }
        return state

    def get_reward(
        self,
        action: torch.Tensor | list[Dict],
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        reward = [0] * self.num_envs
        return torch.tensor(reward, device=self.device, dtype=torch.float32)

    def get_termination(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        若配置了 goal_position：物体中心与 goal_position 距离 < object_goal_tolerance 即终止。
        否则沿用旧逻辑：末端到达 _cube_start_pos + push_direction * push_distance 且推进足够距离后终止。
        """
        truncated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        if self.goal_position is not None:
            cube_pose = self.get_cube_pose()
            cube_pos = cube_pose[:, :3]
            goal = self.goal_position.to(self.device)
            if goal.dim() == 1:
                goal = goal.unsqueeze(0)
            if goal.shape[0] == 1 and cube_pos.shape[0] > 1:
                goal = goal.expand(cube_pos.shape[0], -1)
            dist = torch.norm(cube_pos - goal, dim=1)
            termination = dist < self.object_goal_tolerance
            return termination, truncated

        if not hasattr(self, "_cube_start_pos") or self._cube_start_pos is None:
            return torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            ), truncated

        push_dir = self.push_direction.to(self.device)
        if push_dir.dim() == 1:
            push_dir = push_dir.unsqueeze(0)
        cube_start = self._cube_start_pos.to(self.device)
        target_pos = cube_start + push_dir * self.push_distance

        eef_pose = self.get_eef_pose()
        eef_pos = eef_pose[:, :3]

        dist_to_target = torch.norm(eef_pos - target_pos, dim=1)
        progress = (eef_pos - cube_start).matmul(push_dir.T).squeeze(-1)
        min_progress = self.min_push_progress_ratio * self.push_distance

        at_target = dist_to_target < self.eef_target_tolerance
        pushed_enough = progress >= min_progress
        termination = at_target & pushed_enough

        return termination, truncated

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """重置环境并记录新的 cube 初始位置。"""
        obs, info = super().reset(seed=seed, options=options)
        cube_pose = self.get_cube_pose()
        self._cube_start_pos = cube_pose[:, :3].clone()
        return obs, info

    def reset_idx(
        self,
        env_ids: Sequence[int] = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """按 env_id 局部重置时做完整 hard reset（与首次 reset 一致），再同步 cube 初始位置。"""
        opts = dict(options) if options else {}
        opts["force_hard_reset"] = True
        obs, info = super().reset_idx(env_ids=env_ids, seed=seed, options=opts)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        cube_pose = self.get_cube_pose()
        if (
            not hasattr(self, "_cube_start_pos")
            or self._cube_start_pos.shape != cube_pose[:, :3].shape
        ):
            self._cube_start_pos = cube_pose[:, :3].clone()
        else:
            self._cube_start_pos[env_ids] = cube_pose[env_ids, :3]
        return obs, info
