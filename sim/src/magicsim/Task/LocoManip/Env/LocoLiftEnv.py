"""
LocoLiftEnv: bimanual box (bin) squeeze grasp driven by bounding-box geometry.

Instead of reading ``<hand_type>_bimanual_grasp_pose`` annotations,
:meth:`get_target_bbox_half_extents` returns the target object's local AABB
half-extents via :mod:`magicsim.Env.Utils.mesh_utils`. The open-loop test
computes per-arm IK targets from the bbox (±y face of the bin plus a small
inward gap), transforms them into world frame using the bin's current pose,
and commands both arms to squeeze and lift.
"""

from typing import Optional, Tuple
from magicsim.Task.LocoManip.Env.LocoBiGraspEnv import LocoBiGraspEnv
import torch
from isaacsim.core.utils.prims import get_prim_at_path

from magicsim.Env.Utils.mesh_utils import get_local_bbox_half_extents


class LocoLiftEnv(LocoBiGraspEnv):
    """Bimanual box squeeze + lift driven by the target object's bounding box."""

    # ------------------------------------------------------------------
    # Termination — relative lift from the settled baseline
    # ------------------------------------------------------------------
    # ``LocoBiGraspEnv.get_termination`` compares ``object_z`` to an absolute
    # threshold (``0.75``). With the taller table + higher bin spawn in this
    # env the bin already satisfies that before settling, so episodes trigger
    # termination at startup. Track the bin's running minimum z across steps
    # (the settled baseline) and require the object to be lifted at least
    # :attr:`lift_offset` above it.
    lift_offset: float = 0.15
    fallen_z_threshold: float = 0.2

    def get_termination(self):
        import torch as _torch

        eef_pose = self.get_eef_pose()
        object_poses_dict = self.get_object_pose()

        obj_name = self.target_obj_name if hasattr(self, "target_obj_name") else None
        if obj_name is None or obj_name not in object_poses_dict:
            obj_name = (
                next(iter(object_poses_dict.keys())) if object_poses_dict else None
            )
            if obj_name is None:
                zeros = _torch.zeros(
                    self.num_envs, dtype=_torch.bool, device=self.device
                )
                return zeros, zeros

        object_pos = object_poses_dict[obj_name][:, :3]
        object_z = object_pos[:, 2]

        # Baseline = running min z across the episode. Captures the settled
        # height regardless of how high the bin spawned initially.
        baseline = getattr(self, "_bin_min_z", None)
        if baseline is None or baseline.shape[0] != object_z.shape[0]:
            baseline = object_z.clone().detach()
        else:
            baseline = _torch.minimum(baseline, object_z).detach()
        self._bin_min_z = baseline

        if eef_pose.dim() == 2:
            distance = _torch.norm(eef_pose[:, :3] - object_pos, dim=1)
            both_close = distance < 0.6
        else:
            right_dist = _torch.norm(eef_pose[:, 0, :3] - object_pos, dim=1)
            left_dist = _torch.norm(eef_pose[:, 1, :3] - object_pos, dim=1)
            both_close = (right_dist < 0.6) & (left_dist < 0.6)

        lift = object_z - baseline
        termination = both_close & (lift > self.lift_offset)
        truncated = object_z < self.fallen_z_threshold
        return termination, truncated

    def get_target_bbox_half_extents(
        self,
        env_id: int = 0,
        obj_name: str | None = None,
        obj_id: int = 0,
    ) -> Optional[Tuple[float, float, float]]:
        """Return ``(half_x, half_y, half_z)`` of the target's scaled local AABB.

        :func:`get_local_bbox_half_extents` (aka ``ComputeUntransformedBound``)
        returns the geometry's intrinsic bound *without* the prim's own
        xform — so per-axis scales set in the scene config (e.g.
        ``scale: [0.6, 0.35, 1.0]`` on the bin) are missing. Multiply them
        back in using the scale factors extracted from the prim's
        local-to-world matrix (column norms of the upper-3×3 block). This
        keeps the bbox orientation-invariant but scale-aware, which is
        what callers (forearm-squeeze targets) actually need.
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

        # Walk the prim's own xform ops for the explicit scale op. This
        # ignores anything inherited from ancestors (env xforms, scaled
        # parents) — we only want the scale configured on this object.
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
        """Return ``[pos(3), quat_wxyz(4)]`` of the target object at ``env_id``."""
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
