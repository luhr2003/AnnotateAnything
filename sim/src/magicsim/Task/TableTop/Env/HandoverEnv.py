"""HandoverEnv: bimanual handover of a tabletop object on DualFranka.

Designed for the closed-loop ``Handover`` atomic skill (see
``Collect/AtomicSkill/Handover.py``). Generic w.r.t. the object: any rigid
asset that ships ``Annotation/grasp_pose.json`` works — single-part
(only ``functional_grasp.handle``) or multi-part (handle + body + ...).

The skill flow is:

    1. Left arm picks ANY reachable grasp candidate from the full pool
       (functional_grasp + grasp, all parts merged) → grasps + lifts.
    2. The skill computes a handover pose biased toward the right arm
       (``handover_center`` defaults to ``y=-0.10`` so the mug ends up
       closer to ``R_panda_hand`` at world ``y=-0.5``).
    3. Cross-product of {handover mug poses} × {right-arm grasp candidates
       from the same pool} is submitted as a single PAIRED IK goalset:
         slot 0 (right) = handover_mug_world ⊙ right_local[j]
         slot 1 (left)  = handover_mug_world ⊙ left_grasp_local
       Curobo argmins jointly — one solve picks the best (mug pose,
       right grasp) tuple where BOTH arms are reachable.
    4. Right arm grasps; left arm releases.

This env's job is only to expose the data the skill needs: the grasp
pool (local + world frame) and a candidate generator for handover mug
poses. The state-machine and IK plumbing live in the atomic skill.
"""

from typing import Any, Dict, List, Sequence, Tuple

import torch

from magicsim.Task.TableTop.Env.GraspEnv import GraspEnv
from magicsim.Env.Scene.Object.Rigid import RigidObject
from magicsim.Env.Utils.rotations import euler_angles_to_quat


class HandoverEnv(GraspEnv):
    """Dual-Franka bimanual handover for arbitrary rigid objects."""

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)
        self.target_obj_name = getattr(config, "target_obj_name", "mug")
        # Handover mug-pose center, world frame. Right arm base at y=-0.5,
        # left at y=+0.5 (after the dual_franka.usd / URDF spacing reduction
        # from ±0.7 → ±0.5). Right yaw=+90° so forward = +y; sweet spot
        # ≈ 0.4m forward of base = world y∈[-0.3, 0.0]. Default y=-0.10
        # places the mug ≈40 cm from right base (mid-workspace) and
        # ≈60 cm from left base (still inside reach of the wrist + lift).
        center = getattr(config, "handover_center", None)
        self.handover_center: Tuple[float, float, float] = (
            tuple(center) if center is not None else (0.0, -0.10, 1.25)
        )
        # Lifted off the table for "left has it"; falling = truncate.
        self.lift_threshold = float(getattr(config, "lift_threshold", 1.05))
        self.fall_threshold = float(getattr(config, "fall_threshold", 0.5))
        # Distance from right eef → object that signals "right has taken it".
        self.right_close_threshold = float(
            getattr(config, "right_close_threshold", 0.20)
        )
        # Distance left eef must be FROM the object to count as released.
        # Pair with the atomic skill's ``retract_offset`` (default 0.30):
        # threshold MUST be smaller, else terminated never fires and the
        # env doesn't reset between attempts.
        self.left_release_threshold = float(
            getattr(config, "left_release_threshold", 0.20)
        )

    # ------------------------------------------------------------------
    # Object resolution
    # ------------------------------------------------------------------

    def _resolve_target_obj(self, env_id: int, obj_name: str | None, obj_id: int = 0):
        rigid_objs = self.scene.scene_manager.rigid_objects[env_id]
        name = obj_name or self.target_obj_name
        if name not in rigid_objs:
            name = next((k for k in rigid_objs if k != "simple_desk"), None)
        if name is None:
            return None, None
        obj_list = rigid_objs[name]
        if not obj_list or obj_id >= len(obj_list):
            return name, None
        return name, obj_list[obj_id]

    def get_object_world_pose(
        self, env_id: int, obj_name: str | None = None, obj_id: int = 0
    ) -> torch.Tensor | None:
        _, obj = self._resolve_target_obj(env_id, obj_name, obj_id)
        if obj is None:
            return None
        pos, quat = obj.get_local_pose()
        pos = torch.as_tensor(pos, dtype=torch.float32, device=self.device).flatten()[
            :3
        ]
        quat = torch.as_tensor(quat, dtype=torch.float32, device=self.device).flatten()[
            :4
        ]
        return torch.cat([pos, quat], dim=0)

    # ------------------------------------------------------------------
    # Grasp pool
    # ------------------------------------------------------------------

    def get_grasp_pool(
        self,
        env_id: int,
        obj_name: str | None = None,
        obj_id: int = 0,
        transform_to_world: bool = True,
        part: str | None = None,
        top_key: str | None = None,
    ) -> torch.Tensor | None:
        """Flatten grasp annotation entries into ``Tensor[N, 7]``.

        Args:
            part: Optional part name filter (e.g. ``"handle"``, ``"body"``).
                ``None`` pools every part. Useful for hard-coding the
                Handover test to "left grabs body, right grabs handle".
            top_key: Optional top-level filter, ``"functional_grasp"`` or
                ``"grasp"``. ``None`` searches both. ``handle`` typically
                lives under ``functional_grasp``; ``body`` under ``grasp``.
        """
        _, obj = self._resolve_target_obj(env_id, obj_name, obj_id)
        if obj is None:
            return None
        grasp_dict = obj.get_grasp_poses(
            transform_to_world=transform_to_world,
            device=self.device,
            hand_type=None,
        )
        if grasp_dict is None:
            return None
        top_keys = (top_key,) if top_key is not None else ("functional_grasp", "grasp")
        pool: List[torch.Tensor] = []
        for tk in top_keys:
            top = grasp_dict.get(tk, {})
            if not isinstance(top, dict):
                continue
            for part_name, part_poses in top.items():
                if part is not None and part_name != part:
                    continue
                if isinstance(part_poses, torch.Tensor) and part_poses.numel() > 0:
                    if part_poses.ndim == 1:
                        part_poses = part_poses.unsqueeze(0)
                    if part_poses.shape[-1] == 7:
                        pool.append(part_poses.to(self.device))
        if not pool:
            return None
        return torch.cat(pool, dim=0)

    # GraspEnv override: same as bigrasp's pattern — Grasp atomic skill
    # falls back to ``get_grasp_pose`` and our env exposes the flattened
    # pool under ``functional_grasp.body`` so the single-arm IK has data.
    # (Not strictly used by Handover skill — kept for compatibility with
    # the standard Grasp atomic if someone reuses this env.)
    def get_grasp_pose(
        self,
        env_ids: Sequence[int] | None = None,
        obj_name: str | None = None,
        hand_type: str | None = None,
        grasp_type: str | None = None,
        obj_id: int = 0,
        transform_to_world: bool = True,
    ) -> list:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        out: list = []
        for env_id in env_ids:
            pool = self.get_grasp_pool(
                env_id, obj_name=obj_name, obj_id=obj_id, transform_to_world=True
            )
            if pool is None:
                out.append(None)
            else:
                out.append({"functional_grasp": {"body": pool}, "grasp": {}})
        return out

    # ------------------------------------------------------------------
    # Handover mug-pose candidate generator
    # ------------------------------------------------------------------

    def generate_handover_mug_poses(
        self,
        n_yaws: int = 10,
        z_offsets: Tuple[float, ...] = (-0.05, 0.0, 0.05),
        y_offsets: Tuple[float, ...] = (-0.10, 0.0, 0.10),
        x_offsets: Tuple[float, ...] = (0.0,),
        pitch_degs: Tuple[float, ...] = (-45.0, -20.0, 0.0, 20.0, 45.0),
        # Mug ROLL is the lever that gives the LEFT arm extra reach to
        # the right side. With left base at y=+0.5 and the handover at
        # y=-0.10, left wrist sits at left_eef = mug ⊙ left_grasp_local
        # — no roll → wrist y ≈ mug y, ≈0.6m from left base (just inside
        # Franka's 0.85m reach but unstable). Tilting the mug ±20-45° about
        # its world x-axis swings left_grasp_local's local y-component
        # into a world +y component, "borrowing" several cm back into
        # left's workspace. Without this dimension paired IK fails on
        # nearly every (mug_pose, right_local) combo because the LEFT
        # slot is unreachable.
        roll_degs: Tuple[float, ...] = (-45.0, -20.0, 0.0, 20.0, 45.0),
        # Full ±180° sweep: lets the IK pick a yaw that turns the handle
        # toward the receiving arm regardless of how the mug ended up
        # rotated after random spawn + left grasp. With handle in mug-local
        # ``(-x, -y) @ 45°``, world yaw=+45° points handle to world -y
        # (right arm side); world yaw=-135° points handle to world +y
        # (left arm side). A narrow sweep misses both.
        yaw_range_deg: Tuple[float, float] = (-180.0, 180.0),
    ) -> torch.Tensor:
        """Return ``Tensor[M, 7]`` candidate mug world poses near ``handover_center``.

        Default config: ``3 z × 3 y × 1 x × 10 yaw × 5 pitch × 5 roll = 2250`` poses.
        Each is a 7-vec ``[x, y, z, qw, qx, qy, qz]`` in the same frame the
        rest of the codebase calls "world" (env-local — see
        ``Rigid.transform_pose_to_world`` for the convention).

        Yaw is 10 samples on ``[-180°, +180°)`` (endpoint excluded so the
        wraparound doesn't double-count) — 36° resolution. Pitch and roll
        are 5 samples each ``(-45, -20, 0, 20, 45)``: the joint pitch/roll
        sweep gives the mug full SO(3)-like coverage so paired IK has the
        flexibility to put both wrists in their respective workspaces.
        """
        cx, cy, cz = self.handover_center
        device = self.device
        # Endpoint-excluded so a full ±180° sweep doesn't double-count -180/+180.
        if (yaw_range_deg[1] - yaw_range_deg[0]) >= 359.99:
            yaws = torch.linspace(yaw_range_deg[0], yaw_range_deg[1], n_yaws + 1)[:-1]
        else:
            yaws = torch.linspace(yaw_range_deg[0], yaw_range_deg[1], n_yaws)
        candidates: List[List[float]] = []
        for dz in z_offsets:
            for dy in y_offsets:
                for dx in x_offsets:
                    for roll in roll_degs:
                        for pitch in pitch_degs:
                            for yaw in yaws:
                                candidates.append(
                                    [
                                        cx + float(dx),
                                        cy + float(dy),
                                        cz + float(dz),
                                        float(roll),
                                        float(pitch),
                                        float(yaw),
                                    ]
                                )
        if not candidates:
            return torch.zeros((0, 7), dtype=torch.float32, device=device)
        arr = torch.tensor(candidates, dtype=torch.float32)
        positions = arr[:, :3].to(device)
        # XYZ intrinsic, degrees=True. Returns scalar-first quat (w, x, y, z).
        quats = euler_angles_to_quat(arr[:, 3:6], degrees=True, device=device)
        return torch.cat([positions, quats], dim=1)

    @staticmethod
    def transform_pool_by_object_pose(
        local_pool: torch.Tensor,
        obj_pos: torch.Tensor,
        obj_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a single mug world pose to ``[N, 7]`` local poses → ``[N, 7]`` world."""
        out = torch.empty_like(local_pool)
        for i in range(local_pool.shape[0]):
            out[i] = RigidObject.transform_pose_to_world(
                local_pool[i], obj_pos, obj_quat
            )
        return out

    # ------------------------------------------------------------------
    # State / termination
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
        if eef_pose.dim() != 3:
            # Single-arm scene: degenerate; just check lift.
            terminated = object_z > self.lift_threshold
            truncated = object_z < self.fall_threshold
            return terminated, truncated
        # Slot 0 = right, slot 1 = left (DualFranka convention).
        right_dist = torch.norm(eef_pose[:, 0, :3] - object_pos, dim=1)
        left_dist = torch.norm(eef_pose[:, 1, :3] - object_pos, dim=1)
        right_holds = right_dist < self.right_close_threshold
        left_released = left_dist > self.left_release_threshold
        lifted = object_z > self.lift_threshold
        terminated = right_holds & left_released & lifted
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
        return (
            f"Bimanual handover: left Franka grasps {self.target_obj_name}, "
            "rotates it to a right-arm-comfortable pose, right Franka takes "
            "over, left releases."
        )
