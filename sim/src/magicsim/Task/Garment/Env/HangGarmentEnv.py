"""HangGarmentEnv: single-arm Franka picks a garment top off the table
and carries it over to a clothes rack — gestures the "hung up" motion.

Structurally mirrors :class:`GarmentFoldEnv` / :class:`FlingEnv` (subclass
:class:`TaskBaseEnv` directly, expose the garment via the
``garment_objects`` dict, drive keypoint-based picks). The task itself is
single-arm — the cloth is grasped at the midpoint between the
``top_left`` and ``top_right`` keypoints (the natural "hanger neck" of a
Tops asset) and carried over the rack apex.

Two side helpers are scene-specific:

* :meth:`get_hanger_pose` reads the rigid clothes-rack pose from
  ``rigid_objects`` so the test driver can derive the apex world position
  even if the rack drifts during settle.

* :meth:`set_garment_gravity_scale` flips the particle-material
  ``gravity_scale`` post-close so the parallel-gripper carry stays glued
  through the lift+push without a real cloth-on-hanger interaction
  (user spec: "closegripper后gravityu scale要调小，不用真的挂上，
  有个意思就可以了").

No automatic termination — like :class:`FlingEnv` this is open-loop and
the test driver dictates phase length.
"""

from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import gymnasium as gym

from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv


class HangGarmentEnv(TaskBaseEnv):
    """Single-arm Franka grasps a garment by its top and carries it to a rack."""

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)
        # Category names match keys in the scene yaml (under ``objects``).
        self.garment_category: str = getattr(
            config, "garment_category", "garment_items"
        )
        self.hanger_category: str = getattr(config, "hanger_category", "hanger")
        self.timeout_steps: int = int(getattr(config, "timeout_steps", 10000))

        # Asset bbox upper-z bound from Assets/Object/Hanger/Object.usdc
        # (≈+1.22 in unscaled object frame). Times the configured scale
        # gives the rack-apex offset above its origin. Default 0.58
        # matches scene scale=0.5 (0.5 · 1.22 - 0.03 inward).
        self.hanger_top_z_offset: float = float(
            getattr(config, "hanger_top_z_offset", 0.58)
        )
        # Wrist offset above the fingertip target along the approach
        # direction. ``ik_abs`` commands the panda_hand world pose. The
        # franka_umi gripper is a long custom-printed cloth-manipulation
        # tool — panda_hand → fingertip is noticeably longer than the
        # regular Franka panda gripper (Fold's 0.2 was tuned for the
        # standard gripper, and at 0.2 the UMI panda_hand still drives
        # itself into the table on grasp). Bumped to 0.30 so the wrist
        # sits ≈28 cm above the keypoint at grasp and ≈40 cm above at
        # lift — clear of the desk through every phase.
        self.gripper_length: float = float(getattr(config, "gripper_length", 0.3))
        # How far past the keypoint the fingertip pushes when grasping.
        # Aligned with TestFlingEnv (``REACH_Z_OFFSET = -0.022``) and the
        # canonical ``Collect/Task/Garment/Conf/atomic_skill/default.yaml``
        # (``reach_z_offset: -0.022``). TestFoldEnv uses 0.015 — that's
        # the open-loop outlier; we follow Fling + the production yaml.
        self.insertion_depth: float = float(getattr(config, "insertion_depth", 0.022))

    # ------------------------------------------------------------------
    # Garment / hanger accessors (mirror GarmentFoldEnv._get_garment)
    # ------------------------------------------------------------------
    def _get_garment(self, env_id: int):
        garment_dict = self.scene.scene_manager.garment_objects[env_id]
        garment_list = garment_dict.get(self.garment_category, [])
        return garment_list[0] if garment_list else None

    def _get_hanger(self, env_id: int):
        # The rack is configured as ``physics.type: "geometry"`` in the
        # scene yaml — it lives under ``geometry_objects``, not
        # ``rigid_objects``. Static geometry keeps the rack pinned, gives
        # the dropped garment a collision target, and removes the need
        # for a post-close gravity-disable hack.
        geom_dict = self.scene.scene_manager.geometry_objects[env_id]
        hanger_list = geom_dict.get(self.hanger_category, [])
        return hanger_list[0] if hanger_list else None

    # ------------------------------------------------------------------
    # Keypoints (same shape as GarmentFoldEnv.get_keypoint_positions)
    # ------------------------------------------------------------------
    def get_keypoint_positions(self, env_id: int) -> dict:
        """Return ``{name: np.ndarray(3,)}`` from the garment's mesh keypoints.

        Calls into :meth:`Garment.get_keypoint`. The test driver should
        run :meth:`Garment.update_keypoint` first so the indices are
        cached against the settled cloth.
        """
        garment = self._get_garment(env_id)
        if garment is None:
            return {}
        return garment.get_keypoint()

    def get_garment_top_keypoint(self, env_id: int = 0) -> torch.Tensor | None:
        """Return the bare midpoint of ``top_left`` / ``top_right`` (no offset).

        Returned tensor is ``Tensor[3]`` (xyz). Used by the test driver to
        visualize the raw pick keypoint in red so the offset wrist target
        is distinguishable from the cloth-anchored grasp point.
        """
        kp = self.get_keypoint_positions(env_id)
        if "top_left" not in kp or "top_right" not in kp:
            return None
        tl = np.asarray(kp["top_left"], dtype=np.float32)
        tr = np.asarray(kp["top_right"], dtype=np.float32)
        mid = 0.5 * (tl + tr)
        return torch.tensor(
            [float(mid[0]), float(mid[1]), float(mid[2])],
            dtype=torch.float32,
            device=self.device,
        )

    def get_garment_top_grasp(
        self,
        env_id: int = 0,
        lift_z: float = 0.0,
    ) -> torch.Tensor | None:
        """Synthesize a top-down WRIST grasp 7-vec for the garment "neck".

        Returned pose is the panda_hand world target — fingertip lands at
        ``midpoint(top_left, top_right) + (0, 0, -insertion_depth)`` and
        the wrist sits ``gripper_length`` above the fingertip along the
        approach direction (top-down ⇒ +z). Net z above the keypoint is
        therefore ``gripper_length - insertion_depth`` (≈ 0.115 m at the
        UMI defaults).

        Quat is hard-coded to ``[0, 1, 0, 0]`` (user spec:
        方向就固定死是0100) — for ``ik_abs`` this is panda_hand
        rotated 180° about world x, i.e. its +z axis points world -z so
        the gripper approaches straight down.
        """
        kp_pos = self.get_garment_top_keypoint(env_id)
        if kp_pos is None:
            return None
        # Approach direction = -z (top-down). Fingertip target =
        # kp + insertion_depth · (-z) = kp + (0, 0, -insertion_depth).
        # Wrist target = fingertip - gripper_length · (-z) =
        # fingertip + (0, 0, gripper_length). Net wrist z above kp is
        # gripper_length - insertion_depth.
        wrist_z_offset = self.gripper_length - self.insertion_depth + float(lift_z)
        return torch.tensor(
            [
                float(kp_pos[0]),
                float(kp_pos[1]),
                float(kp_pos[2]) + wrist_z_offset,
                0.0,
                1.0,
                0.0,
                0.0,
            ],
            dtype=torch.float32,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Hanger pose (rigid; world frame via local_pose since envs are flat)
    # ------------------------------------------------------------------
    def get_hanger_pose(self, env_id: int = 0) -> torch.Tensor | None:
        """Return rack origin pose ``[x, y, z, qw, qx, qy, qz]`` or None."""
        hanger = self._get_hanger(env_id)
        if hanger is None:
            return None
        pos, ori = hanger.get_local_pose()
        pos_t = torch.as_tensor(pos, dtype=torch.float32, device=self.device).flatten()[
            :3
        ]
        ori_t = torch.as_tensor(ori, dtype=torch.float32, device=self.device).flatten()[
            :4
        ]
        return torch.cat([pos_t, ori_t], dim=0)

    def get_hanger_top_pose(self, env_id: int = 0) -> torch.Tensor | None:
        """Rack-apex WRIST pose: panda_hand parked above the rack apex.

        Same top-down quat as the grasp pose. The z target is
        ``rack_origin_z + hanger_top_z_offset + gripper_length`` so the
        wrist sits ``gripper_length`` above the apex (fingers right at the
        apex). Test driver further pulls back / hovers from here so the
        wrist never collides with the rack.
        """
        pose = self.get_hanger_pose(env_id)
        if pose is None:
            return None
        target = pose[:7].clone()
        target[2] = target[2] + self.hanger_top_z_offset + self.gripper_length
        target[3:7] = torch.tensor(
            [0.0, 1.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        )
        return target

    # ------------------------------------------------------------------
    # Gravity helpers — both work post-close so the carry stays clean
    # ------------------------------------------------------------------
    def set_garment_gravity_scale(
        self, value: float, env_ids: Sequence[int] | None = None
    ) -> None:
        """Scale the particle-material gravity for the cloth (1.0 default).

        Setting to 0.0 makes the cloth float in place — the user wanted
        the gripper-carry to stay glued without modeling real
        cloth-on-hanger physics. Mirrors what TestFoldEnv does for
        ``adhesion`` / ``stretch_stiffness`` (live PhysX writes after
        ``env.reset``).
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        for env_id in env_ids:
            garment = self._get_garment(int(env_id))
            if garment is None:
                continue
            pm = getattr(garment, "particle_material", None)
            if pm is None:
                continue
            pm.set_gravity_scale(float(value))

    # ------------------------------------------------------------------
    # TaskBaseEnv interface (mirrors FlingEnv: open-loop, no termination)
    # ------------------------------------------------------------------
    def get_obs_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

    def get_policy_obs(self) -> Dict[str, Any]:
        robot_state = self.scene.robot_manager.get_robot_state()
        camera_info = self.scene.capture_manager.step()
        return {"robot_state": robot_state, "camera_info": camera_info}

    def get_privilege_obs(self) -> Dict[str, Any]:
        per_env: List[np.ndarray] = []
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
        # Single-arm: 7D arm + 1D gripper = 8D. Pad with closed gripper if
        # the test driver only supplies the arm pose.
        if action.shape[1] == 7:
            action = torch.cat(
                [action, torch.ones((action.shape[0], 1), device=self.device)],
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
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def get_termination(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Open-loop demo — never auto-reset. Phase plan in the test driver
        # spans hundreds of sim steps; in-flight termination would loop
        # the episode and the user would see the env constantly reset.
        zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return zeros, zeros.clone()

    def get_state(self) -> Dict[str, Any]:
        # ``_check_dict_values_length`` walks list items and fails any
        # tensor whose first dim ≠ num_envs. Stack the per-env hanger
        # poses into a single ``Tensor[num_envs, 7]`` so it satisfies
        # that check.
        hanger_rows = []
        for env_id in range(self.num_envs):
            pose = self.get_hanger_pose(env_id)
            if pose is None:
                pose = torch.zeros(7, dtype=torch.float32, device=self.device)
            hanger_rows.append(pose)
        hanger_tensor = torch.stack(hanger_rows, dim=0)
        # Garment keypoints stay as a list-of-dict-of-numpy — the
        # validator skips numpy arrays since they're neither tensors,
        # dicts, nor lists.
        keypoints_per_env = [
            self.get_keypoint_positions(i) for i in range(self.num_envs)
        ]
        return {
            "robot_state": self.scene.robot_manager.get_robot_state(),
            "scene_state": {
                "garment_keypoints": keypoints_per_env,
                "hanger_pose": hanger_tensor,
            },
            "camera_state": self.scene.camera_manager.get_all_camera_state(),
        }

    def get_info(self) -> Dict[str, Any]:
        return {"state": self.get_state(), "description": self.get_description()}

    def get_description(self) -> str:
        return (
            "Single-arm Franka grasps the garment by its top-shoulder midpoint, "
            "lifts it off the table, and carries it over a clothes rack."
        )
