from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from magicsim.Collect.AtomicSkill.AtomicSkill import AtomicSkill
from magicsim.Env.Planner.Utils import angle_diff
from magicsim.Env.Utils.mesh_utils import (
    get_local_bbox_min_max,
    get_world_bbox_half_extents,
    ray_aabb_entry_face_center_outward,
)
from magicsim.Env.Utils.rotations import quat_to_rot_matrix
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from magicsim.Env.Utils.file import Logger
from omegaconf import DictConfig


class NavTo(AtomicSkill):
    """Mobile base goal ``[gx, gy, yaw]`` for GlobalPlanner.

    **Primary path**

    1. ``B = robot_base``, ``O = obj_pos`` (pose origin), ``u = normalize(O - B)`` (world).
    2. Transform the ray into the **prim-local** frame: origin ``b_l = R^T (B - O)``,
       direction ``u_l = R^T u``. Intersect with the **local axis-aligned bbox**
       ``[box_min, box_max]`` (slab method); take the **first face entered from outside**.
    3. Face center ``fc_l`` → world: ``fc_w = O + R @ fc_l``.
    4. Outward face normal ``n_l`` → world ``n_w`` (unit, points out of the box, toward
       the robot side). **Goal** ``G = fc_w + n_w * position_offset``.
    5. **Yaw**: face the entry face; forward ≈ ``-n_w`` (into the face). Planar yaw uses
       ``-n_w`` projected to XY, with fallbacks if that projection vanishes.

    **Fallback** (no local hit / no box): ``G = O - u * (world_AABB_radius_along_u +
    position_offset)``; yaw = ``u`` in XY toward the object center.

    **Rotation (180° and general):** The primary path uses **mesh-local** bounds and
    intersection in **body frame**, then maps to world with ``R``. For fixed ``B`` and ``O``
    in world, a rigid reorientation of the object changes ``R`` so that ``fc_w`` and ``n_w``
    update **together**; the standoff stays **consistent relative to the object**. For many
    symmetric layouts (e.g. 180° yaw about vertical through ``O``), the **world** goal
    often stays the same because entry face and offset flip coherently. The **fallback**
    path uses **world** AABB extent along ``u``, which **can** change when the object
    rotates if the mesh is not symmetric in world axes.
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.current_target_pose = None
        self.robot_id = int(getattr(config, "robot_id", 0))

        self._position_margin = float(getattr(config, "position_offset", 0.05))

        self.position_threshold = float(getattr(config, "position_threshold", 0.2))
        self.angle_threshold = float(getattr(config, "angle_threshold", 0.15))

        self._set_robot_by_id(self.robot_id)

    def _get_robot_name_list(self):
        return list(self.env.scene.robot_manager.robots.keys())

    def _set_robot_by_id(self, robot_id: int) -> bool:
        robot_id = int(robot_id)
        robot_name_list = self._get_robot_name_list()
        if robot_id < 0 or robot_id >= len(robot_name_list):
            raise ValueError(
                f"robot_id {robot_id} out of range for robot_name_list={robot_name_list}"
            )
        self.robot_id = robot_id
        self.robot_name = robot_name_list[robot_id]

    def _get_robot_base_pos_local(self) -> torch.Tensor:
        """Robot base position in subenv (local) coordinates."""
        robot_state = self.env.scene.robot_manager.get_robot_state()[0][self.robot_name]
        return robot_state["base_pos"][self.env_id].clone()

    def reset(self, action: list[Any]):
        if len(action) == 5:
            self.robot_id = int(action[1])
            self.obj_type = action[2]
            self.obj_name = action[3]
            self.obj_id = action[4]
            self._set_robot_by_id(self.robot_id)
        else:
            self.obj_type = action[1]
            self.obj_name = action[2]
            self.obj_id = action[3]

        self.current_state = "ready"
        self.current_command = [
            "NavTo",
            self.robot_id,
            self.obj_type,
            self.obj_name,
            self.obj_id,
        ]

        self.current_target_pose = self._compute_target_pose()
        self._update_target_marker()

    def refresh(self, action: list[Any]):
        # Track previous target object so we can skip the (expensive) bbox / face-entry
        # recompute when the user just refreshes with the same object.
        prev_key = (
            getattr(self, "obj_type", None),
            getattr(self, "obj_name", None),
            getattr(self, "obj_id", None),
            self.robot_id,
        )
        if len(action) == 5:
            self.robot_id = int(action[1])
            self.obj_type = action[2]
            self.obj_name = action[3]
            self.obj_id = action[4]
            self._set_robot_by_id(self.robot_id)
        else:
            self.obj_type = action[1]
            self.obj_name = action[2]
            self.obj_id = action[3]
        self.current_command = [
            "NavTo",
            self.robot_id,
            self.obj_type,
            self.obj_name,
            self.obj_id,
        ]
        new_key = (self.obj_type, self.obj_name, self.obj_id, self.robot_id)
        if new_key != prev_key or self.current_target_pose is None:
            self.current_target_pose = self._compute_target_pose()
            self._update_target_marker()

    @staticmethod
    def _yaw_toward_face(
        n_w: np.ndarray,
        goal_np: np.ndarray,
        fc_w: np.ndarray,
        obj_xy: torch.Tensor,
        goal_pos: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        n_w: unit outward normal (world) of the entry face — robot should face **−n** (into the face).

        If −n is vertical in XY, fall back to goal→face_center, then goal→object in XY.
        """
        look = -np.asarray(n_w, dtype=np.float64).reshape(3)
        lx, ly = float(look[0]), float(look[1])
        xy = float(np.hypot(lx, ly))
        if xy > 1e-6:
            return torch.atan2(
                torch.tensor(ly / xy, device=device, dtype=dtype),
                torch.tensor(lx / xy, device=device, dtype=dtype),
            )

        to_face = fc_w[:2] - goal_np[:2]
        tf = float(np.linalg.norm(to_face))
        if tf > 1e-6:
            return torch.atan2(
                torch.tensor(to_face[1] / tf, device=device, dtype=dtype),
                torch.tensor(to_face[0] / tf, device=device, dtype=dtype),
            )

        to_obj = obj_xy - goal_pos[:2]
        if torch.norm(to_obj) < 1e-6:
            return torch.zeros((), device=device, dtype=dtype)
        return torch.atan2(to_obj[1], to_obj[0])

    def _compute_target_pose(self) -> torch.Tensor:
        # Computes goal G and yaw; see class docstring.
        target_obj = self.env.scene.scene_manager.get_category(self.obj_type)[
            self.env_id
        ][self.obj_name][self.obj_id]
        obj_pos, obj_quat = target_obj.get_local_pose()
        obj_pos = obj_pos.reshape(-1)[:3]
        obj_quat = obj_quat.reshape(-1)[:4]

        robot_base = self._get_robot_base_pos_local()

        R = quat_to_rot_matrix(obj_quat)
        O_np = obj_pos.detach().cpu().numpy().reshape(3)
        B_np = robot_base.detach().cpu().numpy().reshape(3)
        R_np = R.detach().cpu().numpy().reshape(3, 3)

        v_bo = O_np - B_np
        dist_bo = float(np.linalg.norm(v_bo))
        if dist_bo < 1e-8:
            u_world = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            u_world = v_bo / dist_bo

        goal_pos: torch.Tensor
        yaw_face: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (n_w, fc_w) in world

        if hasattr(target_obj, "prim") and target_obj.prim is not None:
            box = get_local_bbox_min_max(target_obj.prim)
            if box is not None:
                box_min, box_max = box
                b_l = R_np.T @ (B_np - O_np)
                u_l = R_np.T @ u_world
                hit = ray_aabb_entry_face_center_outward(b_l, u_l, box_min, box_max)
                if hit is not None:
                    # hit: (t_enter, fc_l, n_l) — face center and outward normal in local
                    _t, fc_l, n_l = hit
                    fc_w = O_np + R_np @ fc_l
                    n_w = R_np @ n_l
                    nw = float(np.linalg.norm(n_w))
                    if nw > 1e-9:
                        n_w = n_w / nw
                    # G = face_center_world + outward_normal * position_offset
                    g_np = fc_w + n_w * self._position_margin
                    goal_pos = torch.as_tensor(
                        g_np, device=obj_pos.device, dtype=obj_pos.dtype
                    )
                    yaw_face = (n_w.copy(), fc_w.copy())
                else:
                    goal_pos = self._fallback_goal_from_center(
                        obj_pos, u_world, target_obj.prim
                    )
            else:
                goal_pos = self._fallback_goal_from_center(
                    obj_pos, u_world, target_obj.prim
                )
        else:
            goal_pos = self._fallback_goal_from_center(obj_pos, u_world, None)

        if yaw_face is not None:
            n_w, fc_w = yaw_face
            g_np = goal_pos.detach().cpu().numpy().reshape(3)
            target_yaw = self._yaw_toward_face(
                n_w,
                g_np,
                fc_w,
                obj_pos[:2],
                goal_pos,
                obj_pos.device,
                obj_pos.dtype,
            )
        else:
            # Fallback: face along base → object center in XY
            ux, uy = float(u_world[0]), float(u_world[1])
            uxy = float(np.hypot(ux, uy))
            if uxy > 1e-8:
                target_yaw = torch.atan2(
                    torch.tensor(uy / uxy, device=obj_pos.device, dtype=obj_pos.dtype),
                    torch.tensor(ux / uxy, device=obj_pos.device, dtype=obj_pos.dtype),
                )
            else:
                target_yaw = torch.zeros((), device=obj_pos.device, dtype=obj_pos.dtype)

        return torch.stack([goal_pos[0], goal_pos[1], target_yaw])

    def _fallback_goal_from_center(
        self,
        obj_pos: torch.Tensor,
        u_world: np.ndarray,
        prim,
    ) -> torch.Tensor:
        """``O - u * (world_AABB_extent_along_u + position_offset)``; margin only if no bbox."""
        O_np = obj_pos.detach().cpu().numpy().reshape(3)
        if prim is not None:
            half_ext = get_world_bbox_half_extents(prim)
            if half_ext is not None:
                half = np.array(half_ext, dtype=np.float64)
                r = float(np.dot(half, np.abs(u_world)))
                dist = r + self._position_margin
            else:
                dist = self._position_margin
        else:
            dist = self._position_margin
        g = O_np - u_world * dist
        return torch.as_tensor(g, device=obj_pos.device, dtype=obj_pos.dtype)

    def _update_target_marker(self) -> None:
        """Draw a red sphere + yaw arrow at the current NavTo target.

        Lazy-creates ``VisualSphere`` + ``VisualCuboid`` prims under the env root,
        then updates their world poses on each recompute. No-op if Isaac primitives
        aren't importable (e.g. unit tests) — failures are swallowed.
        """
        if self.current_target_pose is None:
            return
        try:
            from isaacsim.core.api.objects import VisualCuboid, VisualSphere
            from isaacsim.core.utils.prims import is_prim_path_valid

            g = self.current_target_pose.detach().cpu().numpy()
            gx, gy, gyaw = float(g[0]), float(g[1]), float(g[2])
            sphere_z = 0.05
            sphere_path = f"/World/envs/env_{self.env_id}/nav_target_sphere"
            arrow_path = f"/World/envs/env_{self.env_id}/nav_target_arrow"

            if not hasattr(self, "_target_sphere") or self._target_sphere is None:
                self._target_sphere = (
                    VisualSphere(
                        prim_path=sphere_path,
                        radius=0.08,
                        color=np.array([1.0, 0.0, 0.0]),
                    )
                    if not is_prim_path_valid(sphere_path)
                    else VisualSphere(prim_path=sphere_path)
                )
            if not hasattr(self, "_target_arrow") or self._target_arrow is None:
                # Thin elongated cuboid as an arrow shaft (length along local x).
                self._target_arrow = (
                    VisualCuboid(
                        prim_path=arrow_path,
                        scale=np.array([0.5, 0.04, 0.04]),
                        color=np.array([1.0, 0.2, 0.2]),
                    )
                    if not is_prim_path_valid(arrow_path)
                    else VisualCuboid(prim_path=arrow_path)
                )

            # Sphere at the goal (xy).
            sphere_pos = np.array([gx, gy, sphere_z], dtype=np.float32)
            self._target_sphere.set_world_pose(
                position=sphere_pos,
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )

            # Arrow shaft: place its center half-length out along yaw, oriented by yaw.
            arrow_len = 0.5
            ax = gx + (arrow_len / 2.0) * np.cos(gyaw)
            ay = gy + (arrow_len / 2.0) * np.sin(gyaw)
            half = gyaw / 2.0
            arrow_quat = np.array(
                [np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32
            )  # wxyz, rotation about world z
            self._target_arrow.set_world_pose(
                position=np.array([ax, ay, sphere_z], dtype=np.float32),
                orientation=arrow_quat,
            )
        except Exception as e:
            # Don't let a viz failure break the skill.
            print(f"[NavTo AS] _update_target_marker skipped (env={self.env_id}): {e}")

    def step(self):
        # Target depends only on the (object, robot_id) pair; both are fixed across
        # consecutive ``step`` calls within a single skill instance, so reuse the
        # cached value computed in ``reset`` / ``refresh`` and skip the per-frame
        # quat→matrix + ray-AABB intersection.
        if self.current_target_pose is None:
            self.current_target_pose = self._compute_target_pose()
            self._update_target_marker()
        self.current_state = "running"

        self.current_action = {"NavTo": (self.robot_id, self.current_target_pose)}
        return self.current_action

    def update(self, info: Dict[str, Any]):
        # Detect env-level reset (terminated / truncated). Without this, the cached
        # ``current_target_pose`` survives the env reset and we keep navigating to
        # the *old* (now-stale) goal — the AtomicSkillManager would never clear us,
        # so the Task command never reports success and RecordManager never saves.
        # ``terminated`` = env-level success criterion met (e.g. LocoNavEnv arrival),
        # so we report ``finished=True`` (Task → ``state="success"`` → RecordManager
        # saves). ``truncated`` = timeout / failure → report ``truncated=1`` instead.
        env_info = info.get("env_info", None)
        if env_info is not None and len(env_info) > 3:
            terminated = env_info[2]
            truncated = env_info[3]

            def _flag_for(env_id, t):
                if t is None:
                    return False
                try:
                    val = t[env_id]
                    return bool(val.item() if hasattr(val, "item") else val)
                except (IndexError, TypeError):
                    return False

            env_terminated = _flag_for(self.env_id, terminated)
            env_truncated = _flag_for(self.env_id, truncated)
            if env_terminated or env_truncated:
                # Drop the cached target so the next instance (created after
                # AtomicSkillManager clears us) recomputes against the post-reset
                # robot/object pose.
                self.current_target_pose = None
                if env_terminated:
                    self.current_state = "finished"
                    return {
                        "type": "NavTo",
                        "command": self.current_command,
                        "action": self.current_action,
                        "finished": True,
                        "state": self.current_state,
                        "truncated": 0,
                    }
                self.current_state = "truncated: env truncated first"
                return {
                    "type": "NavTo",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 1,
                }

        robot_state = self.env.scene.robot_manager.get_robot_state()[0][self.robot_name]
        current_pos = robot_state["base_pos"][self.env_id]
        base_quat = robot_state["base_quat"][self.env_id]

        if base_quat.shape[0] == 4:
            qw, qx, qy, qz = base_quat[0], base_quat[1], base_quat[2], base_quat[3]
            base_yaw = torch.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
        else:
            base_yaw = base_quat.squeeze(-1)

        target_xy = self.current_target_pose[:2]
        target_yaw = self.current_target_pose[2]
        distance = torch.norm(current_pos[:2] - target_xy)
        angle_error = torch.abs(angle_diff(base_yaw, target_yaw))

        if distance < self.position_threshold and angle_error < self.angle_threshold:
            self.current_state = "finished"
            return {
                "type": "NavTo",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 0,
            }

        self.current_state = "running"
        return {
            "type": "NavTo",
            "command": self.current_command,
            "action": self.current_action,
            "finished": False,
            "state": self.current_state,
            "truncated": 0,
        }
