"""
LocoBoxEnv: bimanual ground-box squeeze + lift, standalone (does NOT
inherit from any Loco*Env / Grasp*Env). Sub-classes :class:`TaskBaseEnv`
directly so the task lives entirely in this file — easier to read,
explicit about what's needed, and decoupled from desk / annotation
assumptions baked into LocoLiftEnv / LocoBiGraspEnv.

Provides exactly what the LocoBox AtomicSkill consumes:

* ``get_target_bbox_half_extents(env_id, obj_name, obj_id)`` — scaled
  local-AABB half-extents read directly off the prim.
* ``get_target_world_pose(env_id, obj_name, obj_id)`` — env-local
  ``[pos(3), quat_wxyz(4)]`` of the target.
* ``get_termination()`` — running-min-z baseline; episode terminates when
  both EEFs are within 0.6 m of the box and the box has been lifted at
  least ``lift_offset`` above the baseline.

Plus the boilerplate :class:`TaskBaseEnv` requires:
``get_obs_space / get_policy_obs / get_privilege_obs / get_object_pose /
get_eef_pose / process_action / get_reward / get_info``.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import torch
from isaacsim.core.utils.prims import get_prim_at_path

from magicsim.Env.Utils.mesh_utils import get_local_bbox_half_extents
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv


class LocoBoxEnv(TaskBaseEnv):
    """Standalone bimanual ground-box squeeze env."""

    # Lift threshold above the running-min-z baseline. Box settles ~0.05 m
    # on the ground; we want 15 cm of unambiguous lift before declaring
    # success.
    lift_offset: float = 0.15
    # Truncate when the box drops far below the ground (hard-fail).
    fallen_z_threshold: float = -0.05
    # EEF-to-box proximity required for termination success (both arms).
    eef_proximity: float = 0.6

    # ------------------------------------------------------------------
    # Gym / TaskBaseEnv plumbing — minimal stubs, no policy obs needed.
    # ------------------------------------------------------------------
    def get_obs_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

    def get_policy_obs(self) -> Dict[str, Any]:
        robot_state = self.scene.robot_manager.get_robot_state()
        camera_info = self.scene.capture_manager.step()
        return {"robot_state": robot_state, "camera_info": camera_info}

    def get_privilege_obs(self) -> Dict[str, Any]:
        return {"object_pose": self.get_object_pose()}

    def process_action(self, action):
        return action

    def get_reward(self, action, env_ids: Sequence[int] | None = None):
        return [0] * self.num_envs

    def get_info(self) -> Dict[str, Any]:
        return {
            "state": {
                "robot_state": self.scene.robot_manager.get_robot_state(),
                "scene_state": {"object_pose": self.get_object_pose()},
                "camera_state": self.scene.camera_manager.get_all_camera_state(),
            }
        }

    # ------------------------------------------------------------------
    # Object / EEF queries
    # ------------------------------------------------------------------
    def get_object_pose(
        self, env_ids: Sequence[int] | None = None
    ) -> Dict[str, torch.Tensor]:
        """``{obj_name: tensor(num_envs, 7)}``. Skips ``simple_desk`` if
        present (none expected in this scene, but harmless)."""
        if env_ids is None:
            env_ids = torch.arange(
                self.scene.robot_manager.num_envs, device=self.device
            )

        object_poses_dict: Dict[str, torch.Tensor] = {}
        if len(env_ids) == 0:
            return object_poses_dict

        first_env_id = (
            env_ids[0].item() if isinstance(env_ids, torch.Tensor) else env_ids[0]
        )
        all_obj_names = list(
            self.scene.scene_manager.rigid_objects[first_env_id].keys()
        )
        for obj_name in all_obj_names:
            if obj_name == "simple_desk":
                continue
            poses = []
            for env_id in env_ids:
                t, q = self.scene.scene_manager.rigid_objects[env_id][obj_name][
                    0
                ].get_local_pose()
                poses.append(torch.cat([t, q], dim=0))
            object_poses_dict[obj_name] = torch.stack(poses, dim=0)
        return object_poses_dict

    def get_eef_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Read ``eef_pos / eef_quat`` from the (single) robot's state.

        Single-arm: returns ``[N, 7]``. Dual-arm: returns ``[N, 2, 7]``
        with ``[right, left]`` as the inner index.
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        robot_state = list(self.scene.robot_manager.get_robot_state()[0].values())[0]
        eef_pos = robot_state["eef_pos"]
        eef_quat = robot_state["eef_quat"]
        if eef_pos.dim() == 2:
            eef_pose = torch.cat([eef_pos, eef_quat], dim=1)
        else:
            eef_pose = torch.cat([eef_pos, eef_quat], dim=-1)
        return eef_pose[env_ids]

    # ------------------------------------------------------------------
    # Termination — running-min-z baseline + lift offset
    # ------------------------------------------------------------------
    def get_termination(self):
        eef_pose = self.get_eef_pose()
        object_poses_dict = self.get_object_pose()

        obj_name = self.target_obj_name if hasattr(self, "target_obj_name") else None
        if obj_name is None or obj_name not in object_poses_dict:
            obj_name = (
                next(iter(object_poses_dict.keys())) if object_poses_dict else None
            )
            if obj_name is None:
                zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                return zeros, zeros

        object_pos = object_poses_dict[obj_name][:, :3]
        object_z = object_pos[:, 2]

        baseline = getattr(self, "_box_min_z", None)
        if baseline is None or baseline.shape[0] != object_z.shape[0]:
            baseline = object_z.clone().detach()
        else:
            baseline = torch.minimum(baseline, object_z).detach()
        self._box_min_z = baseline

        if eef_pose.dim() == 2:
            distance = torch.norm(eef_pose[:, :3] - object_pos, dim=1)
            both_close = distance < self.eef_proximity
        else:
            right_dist = torch.norm(eef_pose[:, 0, :3] - object_pos, dim=1)
            left_dist = torch.norm(eef_pose[:, 1, :3] - object_pos, dim=1)
            both_close = (right_dist < self.eef_proximity) & (
                left_dist < self.eef_proximity
            )

        lift = object_z - baseline
        terminated = both_close & (lift > self.lift_offset)
        truncated = object_z < self.fallen_z_threshold
        return terminated, truncated

    # ------------------------------------------------------------------
    # Helpers consumed by the LocoBox AtomicSkill
    # ------------------------------------------------------------------
    def _resolve_target_obj(
        self, env_id: int, obj_name: str | None, obj_id: int
    ) -> Tuple[str | None, Any]:
        """Pick the target rigid object by name (default = first non-desk)."""
        rigid_objs = self.scene.scene_manager.rigid_objects[env_id]
        name = obj_name
        if name is None or name not in rigid_objs:
            name = next((k for k in rigid_objs if k != "simple_desk"), None)
        if name is None:
            return None, None
        obj_list = rigid_objs[name]
        if not obj_list or obj_id >= len(obj_list):
            return name, None
        return name, obj_list[obj_id]

    def get_target_bbox_half_extents(
        self,
        env_id: int = 0,
        obj_name: str | None = None,
        obj_id: int = 0,
    ) -> Optional[Tuple[float, float, float]]:
        """Scaled local-AABB half-extents of the target prim.

        Multiplies the geometry's intrinsic half-extents by the prim's
        own xformOp:scale so per-axis scales authored in the scene config
        (e.g. ``primitive_scale: [0.25, 0.25, 0.25]`` on the cube) are
        baked into the result.
        """
        name, rigid_obj = self._resolve_target_obj(env_id, obj_name, obj_id)
        if rigid_obj is None:
            return None
        prim = get_prim_at_path(rigid_obj._prim_path)
        if prim is None:
            return None
        half = get_local_bbox_half_extents(prim)
        if half is None:
            return None

        from pxr import UsdGeom

        sx = sy = sz = 1.0
        try:
            xformable = UsdGeom.Xformable(prim)
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    v = op.Get()
                    if v is not None:
                        sx, sy, sz = float(v[0]), float(v[1]), float(v[2])
        except Exception:
            pass
        return (half[0] * sx, half[1] * sy, half[2] * sz)

    def get_target_world_pose(
        self,
        env_id: int = 0,
        obj_name: str | None = None,
        obj_id: int = 0,
    ) -> Optional[torch.Tensor]:
        """Env-local ``[pos(3), quat_wxyz(4)]`` of the target object."""
        name, rigid_obj = self._resolve_target_obj(env_id, obj_name, obj_id)
        if rigid_obj is None:
            return None
        trans, ori = rigid_obj.get_local_pose()
        pos = torch.as_tensor(trans, dtype=torch.float32, device=self.device).flatten()[
            :3
        ]
        quat = torch.as_tensor(ori, dtype=torch.float32, device=self.device).flatten()[
            :4
        ]
        return torch.cat([pos, quat], dim=0)
