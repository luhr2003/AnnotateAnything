"""BiGraspEnv: bimanual parallel-gripper grasp on a tabletop basket.

DualFranka rig (L/R panda arms, parallel grippers) + a basket asset that
ships a paired-gripper annotation at ``Annotation/bi_gripper_grasp_pose.json``.

Annotation format (one file per basket, see ``Assets/Object/basket/basket_*``)::

    {
      "type": "basket",
      "bottom_center": [x, y, z],
      "functional_grasp": {
        "body": [
          {"left":  [x, y, z, qx, qy, qz, qw],
           "right": [x, y, z, qx, qy, qz, qw]},
          ...
        ]
      },
      "grasp": { ... same shape ... }   # optional
    }

Each pose is a flat 7-float ``[pos(3), quat(4)]`` in the basket-local frame.
Quaternions are stored as ``[w, x, y, z]`` — the codebase-canonical order,
NOT the same as the bin's bimanual annotation (which is xyzw). Sanity
check that confirms the convention: across 534 pairs the gripper z-axis
(approach direction) under ``wxyz`` interpretation is dominantly
``(small, small, ≈-0.98)`` — top-down approach toward an upright basket
on a table. Under ``xyzw`` interpretation the same quats decode to
sideways approaches that don't match the asset geometry. So we read the
quat slice straight in, no shuffle.

Unlike Dex hand bimanual annotations there is no ``coarse/fine/final``
split — parallel grippers approach in one shot.
"""

from typing import Any, Dict, Sequence

import torch
from magicsim.Task.TableTop.Env.GraspEnv import GraspEnv
from magicsim.Env.Scene.Object.Rigid import RigidObject


_BIGRIPPER_ANNOT_KEY = "bi_gripper_grasp_pose"


class BiGraspEnv(GraspEnv):
    """Dual-Franka bimanual basket grasp."""

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)
        self.target_obj_name = getattr(config, "target_obj_name", "basket")
        # World z basket bottom must reach for "lifted". Settled z ≈ 0.99
        # (basket on table with prim origin near bottom_center); the
        # BiGrasp retrieval offset moves the wrists 0.30 m up, so basket
        # caps near 1.29. 1.20 ⇒ ~21 cm carry — a real lift, not a
        # first-cm twitch off the table.
        self.lift_threshold = float(getattr(config, "lift_threshold", 1.20))
        # Per-eef distance to basket prim origin (the prim sits near the
        # basket BOTTOM per ``bottom_center`` annotation). With handles at
        # basket-local ``|x|≤0.27, |y|≤0.35, z≤0.36`` a properly grasping
        # eef is ~0.57 m from the origin, so 0.7 m is the right ballpark
        # for "both grippers are on the basket".
        self.eef_close_threshold = float(getattr(config, "eef_close_threshold", 0.70))
        # Basket fell off the table — terminate as truncated.
        self.fall_threshold = float(getattr(config, "fall_threshold", 0.5))

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

    def _split_pose7(self, raw, device):
        if raw is None or len(raw) != 7:
            return None
        pos = torch.tensor(raw[:3], dtype=torch.float32, device=device)
        # Quat is already [w, x, y, z] in this annotation — see module docstring
        # for the convention sanity check.
        quat = torch.tensor(raw[3:7], dtype=torch.float32, device=device)
        return pos, quat

    def get_bigripper_grasp_pose(
        self,
        env_ids: Sequence[int] | None = None,
        obj_name: str | None = None,
        obj_id: int = 0,
        transform_to_world: bool = True,
    ) -> list:
        """Return paired ``{"left": pose7, "right": pose7}`` candidates per env.

        Each element of the outer list (one per env) is either ``None`` or::

            {
              "functional_grasp": {part: [{"left": [7], "right": [7]}, ...]},
              "grasp": {...}
            }

        with each ``[7]`` a tensor ``[x, y, z, qw, qx, qy, qz]``. World-frame
        when ``transform_to_world`` is True (default).
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        target = obj_name if obj_name is not None else self.target_obj_name
        results: list = []
        for env_id in env_ids:
            _, rigid_obj = self._resolve_target_obj(env_id, target, obj_id)
            if rigid_obj is None:
                results.append(None)
                continue
            raw = rigid_obj.get_annotation(_BIGRIPPER_ANNOT_KEY)
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
                    transformed = []
                    for pair in pair_list:
                        if not (
                            isinstance(pair, dict)
                            and "left" in pair
                            and "right" in pair
                        ):
                            continue
                        new_pair = {}
                        valid = True
                        for side in ("left", "right"):
                            split = self._split_pose7(pair[side], self.device)
                            if split is None:
                                valid = False
                                break
                            pos, quat = split
                            pose7 = torch.cat([pos, quat], dim=0)
                            if transform_to_world:
                                pose7 = RigidObject.transform_pose_to_world(
                                    pose7, obj_pos, obj_quat
                                )
                            new_pair[side] = pose7
                        if valid:
                            transformed.append(new_pair)
                    if transformed:
                        out[top_key][part_name] = transformed

            if not out["functional_grasp"] and not out["grasp"]:
                results.append(None)
            else:
                results.append(out)
        return results

    def get_grasp_pose(
        self,
        env_ids: Sequence[int] | None = None,
        obj_name: str | None = None,
        hand_type: str | None = None,
        grasp_type: str | None = None,
        obj_id: int = 0,
        transform_to_world: bool = True,
    ) -> list:
        """Override of GraspEnv.get_grasp_pose so the single-arm Grasp atomic
        skill can drive each arm via the bigripper annotation.

        The basket asset ships ONLY ``bi_gripper_grasp_pose.json`` (paired
        ``{left, right}`` per candidate). It has no standard
        ``grasp_pose.json``, so :meth:`GraspEnv.get_grasp_pose` (which reads
        that file via :meth:`Rigid.get_grasp_poses`) would return ``None``
        and the atomic skill would fail with "no grasp annotation".

        We flatten every paired candidate into a SINGLE pool of individual
        7-D poses under the ``body`` part. Both sides go into the same pool
        — when :class:`BiGrasp` issues :class:`Grasp` for ``hand_id=0``
        (right arm) the single-arm IK only succeeds on poses whose world
        ``y < 0``; for ``hand_id=1`` (left arm) it only succeeds on
        ``y > 0``. The pairing between the two arms' picks is therefore
        emergent (each arm independently picks a reachable handle), not
        enforced by JSON pair index — fine for an MVP closed-loop test.

        Returned dict matches what :meth:`AtomicSkill.Grasp._build_pose_tensor_from_grasp_dict`
        expects::

            {"functional_grasp": {"body": tensor[N, 7]}, "grasp": {...}}
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        bi_results = self.get_bigripper_grasp_pose(
            env_ids=env_ids,
            obj_name=obj_name,
            obj_id=obj_id,
            transform_to_world=transform_to_world,
        )

        out: list = []
        for bi in bi_results:
            if bi is None:
                out.append(None)
                continue
            collapsed: Dict[str, Dict[str, torch.Tensor]] = {
                "functional_grasp": {},
                "grasp": {},
            }
            for top_key in ("functional_grasp", "grasp"):
                top = bi.get(top_key, {})
                if not isinstance(top, dict):
                    continue
                pool: list = []
                for pair_list in top.values():
                    for pair in pair_list:
                        if "right" in pair and isinstance(pair["right"], torch.Tensor):
                            pool.append(pair["right"])
                        if "left" in pair and isinstance(pair["left"], torch.Tensor):
                            pool.append(pair["left"])
                if pool:
                    collapsed[top_key]["body"] = torch.stack(pool, dim=0)
            if not collapsed["functional_grasp"] and not collapsed["grasp"]:
                out.append(None)
            else:
                out.append(collapsed)
        return out

    def get_bigripper_pairs_flat(
        self,
        env_id: int = 0,
        obj_name: str | None = None,
        obj_id: int = 0,
        functional_grasp: bool = True,
        part: str | None = None,
    ) -> list:
        """Flat list of ``{"left": pose7, "right": pose7}`` for one env.

        Honors the same primary/fallback selection rule as the other grasp
        helpers: prefer ``functional_grasp[part]``, fall back to all parts,
        then fall back to the ``grasp`` block.
        """
        results = self.get_bigripper_grasp_pose(
            env_ids=[env_id],
            obj_name=obj_name,
            obj_id=obj_id,
            transform_to_world=True,
        )
        if not results or results[0] is None:
            return []
        grasp_dict = results[0]
        primary = "functional_grasp" if functional_grasp else "grasp"
        fallback = "grasp" if functional_grasp else "functional_grasp"

        def _collect(parts):
            flat = []
            if part and part in parts and isinstance(parts[part], list):
                flat.extend(parts[part])
            if not flat:
                for v in parts.values():
                    if isinstance(v, list):
                        flat.extend(v)
            return flat

        flat = _collect(grasp_dict.get(primary, {}))
        if not flat:
            flat = _collect(grasp_dict.get(fallback, {}))
        return flat

    # ------------------------------------------------------------------
    # Termination: both grippers near the basket AND basket lifted.
    # ------------------------------------------------------------------

    def get_termination(self) -> tuple[torch.Tensor, torch.Tensor]:
        eef_pose = self.get_eef_pose()
        object_poses = self.get_object_pose()
        obj_name = self.target_obj_name
        if obj_name not in object_poses:
            obj_name = next(iter(object_poses.keys()), None)
            if obj_name is None:
                zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                return zeros, zeros
        object_pos = object_poses[obj_name][:, :3]
        object_z = object_pos[:, 2]

        if eef_pose.dim() == 2:
            distance = torch.norm(eef_pose[:, :3] - object_pos, dim=1)
            both_close = distance < self.eef_close_threshold
        else:
            right_dist = torch.norm(eef_pose[:, 0, :3] - object_pos, dim=1)
            left_dist = torch.norm(eef_pose[:, 1, :3] - object_pos, dim=1)
            both_close = (right_dist < self.eef_close_threshold) & (
                left_dist < self.eef_close_threshold
            )

        terminated = both_close & (object_z > self.lift_threshold)
        truncated = object_z < self.fall_threshold
        return terminated, truncated

    def get_state(self) -> Dict[str, Any]:
        return {
            "robot_state": self.scene.robot_manager.get_robot_state(),
            "scene_state": {"object_pose": self.get_object_pose()},
            "camera_state": self.scene.camera_manager.get_all_camera_state(),
        }

    def get_info(self) -> Dict[str, Any]:
        return {"state": self.get_state(), "description": self.get_description()}

    def get_description(self) -> str:
        return "Lift the basket using two Franka arms with parallel grippers."
