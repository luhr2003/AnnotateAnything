from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch

from magicsim.Env.Planner.Utils import angle_diff
from magicsim.Env.Utils.mesh_utils import (
    get_local_bbox_min_max,
    get_world_bbox_half_extents,
    ray_aabb_entry_face_center_outward,
)
from magicsim.Env.Utils.rotations import quat_to_rot_matrix
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv


class LocoNavEnv(TaskBaseEnv):
    """Navigation-only loco env: one randomized object, NavTo command, done on arrival.

    Scene is expected to contain a single target (e.g. a table) whose pose is
    randomized per reset via ``initial_pos_range`` / ``initial_ori_range``. The
    AutoCollect NavTo command drives the robot toward that object. The env's
    ``get_termination`` mirrors :class:`magicsim.Collect.AtomicSkill.NavTo`'s
    goal computation (face-center of the local bbox side entered from the
    robot, offset outward by ``position_offset``) and reports success when
    the base is within ``position_threshold`` (xy) and ``angle_threshold``
    (yaw) of that goal.
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)

        # NavTo target is defined by the (top-level) task config, which
        # AsyncRobotEnv stuffs into env_config.task for us.
        task_cfg = getattr(config, "task", None)
        navto_cfg = getattr(task_cfg, "NavTo", None) if task_cfg is not None else None
        self.target_obj_type: str = str(
            getattr(navto_cfg, "obj_type", "geometry") if navto_cfg else "geometry"
        )
        self.target_obj_name: Optional[str] = (
            getattr(navto_cfg, "obj_name", None) if navto_cfg else None
        )
        self.target_obj_id: int = int(
            getattr(navto_cfg, "obj_id", 0) if navto_cfg else 0
        )
        self.robot_id: int = int(getattr(navto_cfg, "robot_id", 0) if navto_cfg else 0)

        # Success thresholds / standoff. Wider than the AtomicSkill's own
        # ``position_threshold`` / ``angle_threshold`` (0.15 / 0.15) on purpose:
        # env-level done should fire *before* the AtomicSkill internally claims
        # finished, so we always reset cleanly via the env→AtomicSkill.update
        # path (avoids the stale-target loop).
        nav_term_cfg = getattr(config, "nav_termination", None)
        self.nav_position_threshold: float = float(
            getattr(nav_term_cfg, "position_threshold", 0.65) if nav_term_cfg else 0.65
        )
        self.nav_angle_threshold: float = float(
            getattr(nav_term_cfg, "angle_threshold", 0.1) if nav_term_cfg else 0.1
        )
        self.nav_position_offset: float = float(
            getattr(nav_term_cfg, "position_offset", 0.05) if nav_term_cfg else 0.05
        )

        # Per-env cached nav goal (set lazily on first get_termination after reset,
        # then reused — the goal depends only on (object pose, robot pose at the time
        # the cache was filled), and re-running quat→matrix + ray-AABB every sim step
        # was the dominant cost in this env's step.
        self._target_cache: list = [None] * self.num_envs

    # ------------------------------------------------------------------
    # Obs / reward / info
    # ------------------------------------------------------------------

    def get_obs_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

    def get_policy_obs(self) -> Dict[str, Any]:
        robot_state = self.scene.robot_manager.get_robot_state()
        camera_info = self.scene.capture_manager.step()
        return {"robot_state": robot_state, "camera_info": camera_info}

    def get_privilege_obs(self) -> Dict[str, Any]:
        return {"target_pose": self._get_all_target_poses()}

    def process_action(self, action):
        return action

    def get_reward(
        self,
        action: torch.Tensor | list[Dict],
        env_ids: Sequence[int] | None = None,
    ) -> list:
        return [0] * self.num_envs

    def get_info(self) -> Dict[str, Any]:
        return {"state": self.get_state()}

    def get_state(self) -> Dict[str, Any]:
        return {
            "robot_state": self.scene.robot_manager.get_robot_state(),
            "scene_state": {"target_pose": self._get_all_target_poses()},
            "camera_state": self.scene.camera_manager.get_all_camera_state(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_robot_name(self) -> str:
        return list(self.scene.robot_manager.robots.keys())[self.robot_id]

    def _get_target_obj(self, env_id: int):
        """Resolve the target object for ``env_id``; falls back to first entry."""
        try:
            cat = self.scene.scene_manager.get_category(self.target_obj_type)[env_id]
        except (AttributeError, KeyError, IndexError):
            return None
        obj_name = self.target_obj_name
        if obj_name is None or obj_name not in cat:
            obj_name = next(iter(cat.keys())) if cat else None
        if obj_name is None:
            return None
        obj_list = cat.get(obj_name, [])
        if not obj_list or self.target_obj_id >= len(obj_list):
            return None
        return obj_list[self.target_obj_id]

    def _get_all_target_poses(self) -> torch.Tensor:
        poses = []
        for eid in range(self.num_envs):
            obj = self._get_target_obj(eid)
            if obj is None:
                poses.append(torch.zeros(7, device=self.device))
                continue
            pos, quat = obj.get_local_pose()
            poses.append(
                torch.cat([pos.reshape(-1)[:3], quat.reshape(-1)[:4]], dim=0).to(
                    self.device
                )
            )
        return torch.stack(poses, dim=0)

    def _compute_nav_target(self, env_id: int) -> Optional[torch.Tensor]:
        """Goal ``[gx, gy, yaw]`` — mirrors :meth:`NavTo._compute_target_pose`."""
        target_obj = self._get_target_obj(env_id)
        if target_obj is None:
            return None

        robot_name = self._get_robot_name()
        robot_state = self.scene.robot_manager.get_robot_state()[0][robot_name]
        robot_base = robot_state["base_pos"][env_id].clone()

        obj_pos, obj_quat = target_obj.get_local_pose()
        obj_pos = obj_pos.reshape(-1)[:3]
        obj_quat = obj_quat.reshape(-1)[:4]

        R = quat_to_rot_matrix(obj_quat)
        O_np = obj_pos.detach().cpu().numpy().reshape(3)
        B_np = robot_base.detach().cpu().numpy().reshape(3)
        R_np = R.detach().cpu().numpy().reshape(3, 3)

        v_bo = O_np - B_np
        dist_bo = float(np.linalg.norm(v_bo))
        u_world = (
            np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if dist_bo < 1e-8
            else v_bo / dist_bo
        )

        goal_pos: Optional[torch.Tensor] = None
        yaw_face: Optional[Tuple[np.ndarray, np.ndarray]] = None
        prim = getattr(target_obj, "prim", None)
        if prim is not None:
            box = get_local_bbox_min_max(prim)
            if box is not None:
                box_min, box_max = box
                b_l = R_np.T @ (B_np - O_np)
                u_l = R_np.T @ u_world
                hit = ray_aabb_entry_face_center_outward(b_l, u_l, box_min, box_max)
                if hit is not None:
                    _t, fc_l, n_l = hit
                    fc_w = O_np + R_np @ fc_l
                    n_w = R_np @ n_l
                    nw = float(np.linalg.norm(n_w))
                    if nw > 1e-9:
                        n_w = n_w / nw
                    g_np = fc_w + n_w * self.nav_position_offset
                    goal_pos = torch.as_tensor(
                        g_np, device=obj_pos.device, dtype=obj_pos.dtype
                    )
                    yaw_face = (n_w.copy(), fc_w.copy())

        if goal_pos is None:
            half_ext = get_world_bbox_half_extents(prim) if prim is not None else None
            if half_ext is not None:
                r = float(np.dot(np.array(half_ext, dtype=np.float64), np.abs(u_world)))
                dist = r + self.nav_position_offset
            else:
                dist = self.nav_position_offset
            g_np = O_np - u_world * dist
            goal_pos = torch.as_tensor(g_np, device=obj_pos.device, dtype=obj_pos.dtype)

        if yaw_face is not None:
            n_w, fc_w = yaw_face
            look = -np.asarray(n_w, dtype=np.float64).reshape(3)
            lx, ly = float(look[0]), float(look[1])
            xy = float(np.hypot(lx, ly))
            if xy > 1e-6:
                target_yaw = torch.as_tensor(
                    float(np.arctan2(ly / xy, lx / xy)),
                    device=obj_pos.device,
                    dtype=obj_pos.dtype,
                )
            else:
                g_np = goal_pos.detach().cpu().numpy().reshape(3)
                to_face = fc_w[:2] - g_np[:2]
                tf = float(np.linalg.norm(to_face))
                if tf > 1e-6:
                    target_yaw = torch.as_tensor(
                        float(np.arctan2(to_face[1] / tf, to_face[0] / tf)),
                        device=obj_pos.device,
                        dtype=obj_pos.dtype,
                    )
                else:
                    target_yaw = torch.zeros(
                        (), device=obj_pos.device, dtype=obj_pos.dtype
                    )
        else:
            ux, uy = float(u_world[0]), float(u_world[1])
            uxy = float(np.hypot(ux, uy))
            if uxy > 1e-8:
                target_yaw = torch.as_tensor(
                    float(np.arctan2(uy / uxy, ux / uxy)),
                    device=obj_pos.device,
                    dtype=obj_pos.dtype,
                )
            else:
                target_yaw = torch.zeros((), device=obj_pos.device, dtype=obj_pos.dtype)

        return torch.stack([goal_pos[0], goal_pos[1], target_yaw])

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _get_or_compute_nav_target(self, env_id: int) -> Optional[torch.Tensor]:
        """Return the cached nav goal for ``env_id``; compute & cache on first miss."""
        cached = self._target_cache[env_id]
        if cached is not None:
            return cached
        target = self._compute_nav_target(env_id)
        self._target_cache[env_id] = target
        return target

    def reset(self, seed: int | None = None, options: dict | None = None):
        # Full reset → drop every per-env nav-target cache.
        self._target_cache = [None] * self.num_envs
        return super().reset(seed=seed, options=options)

    def reset_idx(
        self,
        env_ids: Sequence[int] = None,
        seed: int | None = None,
        options: dict | None = None,
    ):
        # Per-env reset → only drop the affected env caches.
        super().reset_idx(env_ids=env_ids, seed=seed, options=options)
        if env_ids is None:
            self._target_cache = [None] * self.num_envs
            return
        if isinstance(env_ids, torch.Tensor):
            ids = env_ids.detach().cpu().tolist()
        else:
            ids = list(env_ids)
        for eid in ids:
            if 0 <= int(eid) < self.num_envs:
                self._target_cache[int(eid)] = None

    def get_termination(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Success: base xy ≤ ``nav_position_threshold`` AND |yaw_err| ≤ ``nav_angle_threshold``."""
        robot_name = self._get_robot_name()
        robot_state = self.scene.robot_manager.get_robot_state()[0][robot_name]
        base_pos = robot_state["base_pos"].to(self.device)
        base_quat = robot_state["base_quat"].to(self.device)

        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for env_id in range(self.num_envs):
            target = self._get_or_compute_nav_target(env_id)
            if target is None:
                continue
            tgt = target.to(self.device)
            cur_xy = base_pos[env_id, :2]
            q = base_quat[env_id]
            if q.numel() == 4:
                qw, qx, qy, qz = q[0], q[1], q[2], q[3]
                cur_yaw = torch.atan2(
                    2.0 * (qw * qz + qx * qy),
                    1.0 - 2.0 * (qy * qy + qz * qz),
                )
            else:
                cur_yaw = q.squeeze(-1)

            distance = torch.norm(cur_xy - tgt[:2])
            yaw_err = torch.abs(angle_diff(cur_yaw, tgt[2]))

            pos_ok = distance < self.nav_position_threshold
            yaw_ok = yaw_err < self.nav_angle_threshold
            if pos_ok and yaw_ok:
                terminated[env_id] = True
                print(
                    f"[LocoNavEnv] env={env_id} TERMINATE | dist={distance.item():.3f}<"
                    f"{self.nav_position_threshold:.3f}, |yaw_err|={yaw_err.item():.3f}<"
                    f"{self.nav_angle_threshold:.3f} | cur_xy=({cur_xy[0].item():.3f},"
                    f"{cur_xy[1].item():.3f}) cur_yaw={cur_yaw.item():.3f} | "
                    f"tgt_xy=({tgt[0].item():.3f},{tgt[1].item():.3f}) "
                    f"tgt_yaw={tgt[2].item():.3f}"
                )
            else:
                # Diagnostic: log progress every step so you can see which check
                # is blocking termination (pos vs yaw).
                print(
                    f"[LocoNavEnv] env={env_id} not done | dist={distance.item():.3f} "
                    f"(<{self.nav_position_threshold:.3f}? {bool(pos_ok)}), "
                    f"|yaw_err|={yaw_err.item():.3f} "
                    f"(<{self.nav_angle_threshold:.3f}? {bool(yaw_ok)}) | "
                    f"cur_xy=({cur_xy[0].item():.3f},{cur_xy[1].item():.3f}) "
                    f"tgt_xy=({tgt[0].item():.3f},{tgt[1].item():.3f}) "
                    f"tgt_yaw={tgt[2].item():.3f} cur_yaw={cur_yaw.item():.3f}"
                )

        return terminated, truncated
