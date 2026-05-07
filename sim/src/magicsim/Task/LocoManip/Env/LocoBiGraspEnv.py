"""
LocoBiGraspEnv: bimanual loco grasp (e.g. lift a bin with both Dex3 hands).

Adds bimanual grasp annotation helpers on top of :class:`LocoGraspEnv`:

* :meth:`get_bimanual_grasp_pose` reads ``bimanual_grasp_pose.json`` directly
  from the asset's USD directory (the new bin asset stores the file alongside
  ``Object.usd``, NOT under the ``Annotation/`` subdir that
  :meth:`Rigid._load_annotations` scans). Quaternions are already in the
  codebase-canonical ``[w, x, y, z]`` order — no conversion needed. When
  ``transform_to_world``, each candidate's ``left_hand`` / ``right_hand``
  phases get position + orientation transformed by the object's world pose.
* :meth:`get_bimanual_grasp_pose_updated` returns a single paired candidate
  dict so the reactive loop in :class:`DexGrasp` (bimanual branch) can refresh
  both arms' targets.

Termination: both EEFs close to the bin and bin lifted past ``lift_threshold``.
"""

from magicsim.Task.LocoManip.Env.LocoGraspEnv import LocoGraspEnv
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from magicsim.Env.Scene.Object.Rigid import RigidObject


# Filename next to the asset's ``Object.usd``. Convention chosen by the
# annotator pipeline; not a generic naming.
_BIMANUAL_GRASP_FILE = "bimanual_grasp_pose.json"


# Annotation orientation is already stored as [w, x, y, z] — verified
# empirically with scripts/check_g1_paired_ik_standalone.py: treating the
# values as wxyz lets paired curobo IK solve 3/4 pairs to 0mm/0° error
# against the actual G1 kinematics; treating as xyzw makes every pair
# unreachable. No conversion needed at load time.


class LocoBiGraspEnv(LocoGraspEnv):
    """Bimanual loco grasp: two Dex3 hands lift a symmetric object (bin)."""

    # ------------------------------------------------------------------
    # Bimanual grasp annotation loading
    # ------------------------------------------------------------------

    def _resolve_target_obj(self, env_id: int, obj_name: str | None, obj_id: int):
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

    def _transform_phase(
        self,
        phase: Dict[str, Any] | None,
        obj_pos: torch.Tensor,
        obj_quat: torch.Tensor,
        transform_to_world: bool,
    ) -> Dict[str, torch.Tensor] | None:
        if phase is None:
            return None
        dev = self.device
        pos = torch.as_tensor(
            phase["position"], dtype=torch.float32, device=dev
        ).flatten()[:3]
        # Annotation orientation is already wxyz (see module-level note).
        ori = torch.as_tensor(
            phase["orientation"], dtype=torch.float32, device=dev
        ).flatten()[:4]
        if transform_to_world:
            world = RigidObject.transform_pose_to_world(
                torch.cat([pos, ori], dim=0), obj_pos, obj_quat
            )
            pos = world[:3]
            ori = world[3:7]
        out = {"position": pos, "orientation": ori}
        if "joints" in phase:
            out["joints"] = torch.as_tensor(
                phase["joints"], dtype=torch.float32, device=dev
            ).flatten()
        return out

    def _load_bimanual_raw(self, rigid_obj) -> Dict[str, Any] | None:
        """Read ``bimanual_grasp_pose.json`` from the asset USD's parent dir.

        The file lives next to ``Object.usd`` (NOT in the ``Annotation/``
        subdir scanned by :meth:`Rigid._load_annotations`), so we bypass
        ``rigid_obj.get_annotation`` and read it directly.
        """
        usd_path = getattr(rigid_obj, "usd_path", None)
        if not usd_path:
            return None
        json_path = Path(usd_path).parent / _BIMANUAL_GRASP_FILE
        if not json_path.exists():
            return None
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def get_bimanual_grasp_pose(
        self,
        env_ids: Sequence[int] | None = None,
        obj_name: str | None = None,
        hand_type: str = "dex3_1",
        obj_id: int = 0,
        transform_to_world: bool = True,
    ) -> list:
        """Return paired bimanual grasp pose per env.

        Each element is either ``None`` (annotation missing) or a dict::

            {
              "functional_grasp": {part: [{"left_hand": {...}, "right_hand": {...}}, ...]},
              "grasp": {...}
            }

        Each hand's phase entry has ``position``/``orientation``/``joints`` as
        tensors. ``orientation`` is converted from the asset's ``[x, y, z, w]``
        to the codebase-canonical ``[w, x, y, z]``. World transform is applied
        when ``transform_to_world`` is True.

        ``hand_type`` is accepted for API compat with the single-hand env path
        but ignored — the new asset stores a single hand-agnostic file.
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        results: list = []
        for env_id in env_ids:
            name, rigid_obj = self._resolve_target_obj(env_id, obj_name, obj_id)
            if rigid_obj is None:
                results.append(None)
                continue
            raw = self._load_bimanual_raw(rigid_obj)
            if raw is None:
                results.append(None)
                continue

            obj_trans, obj_ori = rigid_obj.get_local_pose()
            obj_pos = torch.as_tensor(
                obj_trans, dtype=torch.float32, device=self.device
            ).flatten()[:3]
            obj_quat = torch.as_tensor(
                obj_ori, dtype=torch.float32, device=self.device
            ).flatten()[:4]

            out: Dict[str, Dict[str, list]] = {"functional_grasp": {}, "grasp": {}}
            for top_key in ("functional_grasp", "grasp"):
                top = raw.get(top_key, {})
                if not isinstance(top, dict):
                    continue
                for part_name, pair_list in top.items():
                    if not isinstance(pair_list, list):
                        continue
                    transformed_pairs = []
                    for pair in pair_list:
                        if not (
                            isinstance(pair, dict)
                            and "left_hand" in pair
                            and "right_hand" in pair
                        ):
                            continue
                        new_pair = {}
                        for side in ("left_hand", "right_hand"):
                            side_raw = pair[side]
                            side_out = {}
                            for phase_key in (
                                "coarse_grasp",
                                "fine_grasp",
                                "final_grasp",
                            ):
                                t = self._transform_phase(
                                    side_raw.get(phase_key),
                                    obj_pos,
                                    obj_quat,
                                    transform_to_world,
                                )
                                if t is not None:
                                    side_out[phase_key] = t
                            new_pair[side] = side_out
                        transformed_pairs.append(new_pair)
                    if transformed_pairs:
                        out[top_key][part_name] = transformed_pairs
            if not out["functional_grasp"] and not out["grasp"]:
                results.append(None)
            else:
                results.append(out)
        return results

    def get_bimanual_grasp_pose_updated(
        self,
        env_ids: Sequence[int],
        obj_name: str,
        obj_id: int,
        obj_type: str,
        hand_type: str,
        selected_idx: int,
        functional_grasp: bool = True,
        part: str | None = None,
    ) -> dict | None:
        """Return the selected paired candidate in world frame at the current object pose."""
        grasp_list = self.get_bimanual_grasp_pose(
            env_ids=env_ids,
            obj_name=obj_name,
            hand_type=hand_type,
            obj_id=obj_id,
            transform_to_world=True,
        )
        if not grasp_list or grasp_list[0] is None:
            return None
        grasp_dict = grasp_list[0]
        primary = "functional_grasp" if functional_grasp else "grasp"
        fallback = "grasp" if functional_grasp else "functional_grasp"
        candidates: list = []

        def _collect(parts: dict) -> list:
            out = []
            if part and part in parts and isinstance(parts[part], list):
                out.extend(parts[part])
            if not out:
                for v in parts.values():
                    if isinstance(v, list):
                        out.extend(v)
            return out

        candidates = _collect(grasp_dict.get(primary, {}))
        if not candidates:
            candidates = _collect(grasp_dict.get(fallback, {}))
        if not candidates or selected_idx >= len(candidates):
            return None
        return candidates[selected_idx]

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def get_termination(self):
        """Both EEFs near the bin AND bin lifted past ``lift_threshold``."""
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

        if eef_pose.dim() == 2:
            distance = torch.norm(eef_pose[:, :3] - object_pos, dim=1)
            both_close = distance < 0.6
        else:
            right_dist = torch.norm(eef_pose[:, 0, :3] - object_pos, dim=1)
            left_dist = torch.norm(eef_pose[:, 1, :3] - object_pos, dim=1)
            # Both hands must be near (bimanual lift).
            both_close = (right_dist < 0.6) & (left_dist < 0.6)

        # Bin spawns at z=0.85, settles on the table top (≈ 0.7m) at center
        # z ≈ 0.77. Threshold must be well above the resting height so we
        # only fire after a real lift, not the moment both arms approach.
        lift_threshold = 1.0
        termination = both_close & (object_z > lift_threshold)
        # Truncated if bin fell off the table (well below table top).
        truncated = object_z < 0.4
        return termination, truncated
