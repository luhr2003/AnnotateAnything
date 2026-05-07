import math
from typing import Any

import numpy as np
import torch
from magicsim.Collect.AtomicSkill.AtomicSkill import AtomicSkill
from magicsim.Env.Utils.file import Logger
from magicsim.Env.Utils.viz import draw_grasp_samples_as_axes, draw_waypoints
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


# Mapping of group name -> (left_keypoint, right_keypoint).
# Left goes to hand_id=1 (left arm), right to hand_id=0 (right arm).
KEYPOINT_GROUPS = {
    "bottom": ("bottom_left", "bottom_right"),
    "sleeve": ("top_left", "top_right"),
    "shoulder": ("left_shoulder", "right_shoulder"),
}


class Fling(AtomicSkill):
    """Bimanual fling skill for garments.

    Both Franka arms grasp a paired (left, right) keypoint on the target
    garment and execute the fling motion in sync. The two robots face each
    other along world Y, so the fling direction (perpendicular bisector) is
    world +X.

    Phases::

        reach -> close_gripper -> lift_up -> fling_forward -> drop -> open_gripper

    Action format (Grasp-style object targeting):

        # group resolved via KEYPOINT_GROUPS
        ["Fling", robot_id, obj_type, obj_name, obj_id, group_name]

        # explicit left/right keypoint names
        ["Fling", robot_id, obj_type, obj_name, obj_id, left_kp, right_kp]

    ``obj_type`` / ``obj_name`` / ``obj_id`` pick the garment through
    ``scene_manager.get_category(obj_type)[env_id][obj_name][obj_id]``
    (e.g. ``"garment"`` / ``"garment_items"`` / ``0``).

    Each MoveL phase emits a dual-arm 14D target ``[right_7d, left_7d]`` with
    ``hand_id=-1``. Each gripper phase emits a 2D target ``[right, left]``
    with ``hand_id=-1``.
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.robot_id = int(getattr(config, "robot_id", 0))
        # Always dual-arm
        self.hand_id = -1

        self.lift_height = float(getattr(config, "lift_height", 0.35))
        self.fling_distance = float(getattr(config, "fling_distance", 0.45))
        # Drop continues past the fling end-point along the same X axis
        # (further forward, toward the neckline) and lower in z — the
        # arms lay the cloth out on the table rather than releasing it
        # mid-air at the fling apex.
        self.drop_distance = float(
            getattr(config, "drop_distance", 2.0 * self.fling_distance)
        )
        self.fling_apex = float(getattr(config, "fling_apex", 0.1))
        self.drop_height = float(getattr(config, "drop_height", 0.05))
        # Extra z offset added to the reach pose so the gripper hovers just
        # above the keypoint instead of colliding with the cloth/table.
        self.reach_z_offset = float(getattr(config, "reach_z_offset", 0.02))
        # Distance from panda_hand origin to the fingertip along the
        # approach axis. Every world-frame waypoint has this added to z
        # so that when the gripper is top-down the FINGERTIP lands at
        # ``keypoint_z + phase_offset`` instead of the hand origin.
        # Franka UMI gripper ≈ 0.2 m tip-out.
        self.gripper_length = float(getattr(config, "gripper_length", 0.2))
        # ``grasp_quat`` is the **world-frame** panda_hand orientation
        # for a single Franka at its home pose ("standard" forward-facing
        # grip, wxyz = [0, 1, 0, 0]). For dual_franka, L_panda_link0 and
        # R_panda_link0 are yawed -90° / +90° vs base_link, so each
        # arm's world-frame target is ``grasp_quat`` rotated by that
        # arm's yaw — L panda_hand ends up facing world -Y, R ends up
        # facing world +Y, which is the natural "forward" for each arm.
        self.grasp_quat = list(getattr(config, "grasp_quat", [0.0, 1.0, 0.0, 0.0]))
        self.left_arm_yaw_deg = float(getattr(config, "left_arm_yaw_deg", -90.0))
        self.right_arm_yaw_deg = float(getattr(config, "right_arm_yaw_deg", 90.0))
        self.grasp_quat_left = self._rotate_world_quat_by_arm_yaw(
            self.grasp_quat, self.left_arm_yaw_deg
        )
        self.grasp_quat_right = self._rotate_world_quat_by_arm_yaw(
            self.grasp_quat, self.right_arm_yaw_deg
        )
        self.debug = bool(getattr(config, "debug", True))
        self.visualize = bool(getattr(config, "visualize", True))

        # ----- Keypoint adjustments (mirrors TestFlingEnv) -----
        # Pull the two grasp anchors toward the LR midline so the fingers
        # land on fabric body, not the sleeve-tip seam.
        self.inward_shift = float(getattr(config, "inward_shift", 0.05))
        # Push each grasp anchor along world +X (away from neckline) so
        # the fingers wrap around the sleeve body, not the upper hem.
        self.grasp_kp_x_shift = float(getattr(config, "grasp_kp_x_shift", 0.02))

        # ----- Per-phase dynamic gravity scale -----
        # During lift_up / fling_forward / drop the cloth is quasi-
        # suspended (0.05× gravity) so the dynamic motion isn't fought
        # by full gravity drop; restored to default outside those phases.
        self.gravity_scale_default = float(
            getattr(config, "gravity_scale_default", 1.0)
        )
        self.gravity_scale_fling = float(getattr(config, "gravity_scale_fling", 0.05))
        self.phases_with_low_gravity = set(
            getattr(
                config,
                "phases_with_low_gravity",
                ["lift_up", "fling_forward", "drop"],
            )
        )  # drop kept here so the cloth trails along the diagonal drop

        # Trajectory smoothing now lives entirely in MoveL (its
        # ``interp_steps`` hyperparameter); this skill just emits the
        # final target and trusts MoveL to lerp + signal completion.
        self._cached_grav_scale: float | None = None

        self.group_name: str | None = None
        self.left_keypoint_name: str | None = None
        self.right_keypoint_name: str | None = None
        self.current_phase: str | None = None

        # Target garment identifiers (Grasp-style).
        self.obj_type: str | None = None
        self.obj_name: str | None = None
        self.obj_id: int | None = None

        # Per-arm waypoints (xyz only, env-local).
        self.reach_xyz_left: np.ndarray | None = None
        self.lift_xyz_left: np.ndarray | None = None
        self.fling_xyz_left: np.ndarray | None = None
        self.drop_xyz_left: np.ndarray | None = None
        self.reach_xyz_right: np.ndarray | None = None
        self.lift_xyz_right: np.ndarray | None = None
        self.fling_xyz_right: np.ndarray | None = None
        self.drop_xyz_right: np.ndarray | None = None

        # Per-arm 7D poses (pos + wxyz quat), device tensors.
        self.reach_pose_left: torch.Tensor | None = None
        self.lift_pose_left: torch.Tensor | None = None
        self.fling_pose_left: torch.Tensor | None = None
        self.drop_pose_left: torch.Tensor | None = None
        self.reach_pose_right: torch.Tensor | None = None
        self.lift_pose_right: torch.Tensor | None = None
        self.fling_pose_right: torch.Tensor | None = None
        self.drop_pose_right: torch.Tensor | None = None

        # Raw keypoint positions cached for debug viz (populated in
        # _build_waypoints).
        self.left_kp: np.ndarray | None = None
        self.right_kp: np.ndarray | None = None

    # ---------- helpers ----------

    @staticmethod
    def _quat_mul(q1, q2):
        """Hamilton product of two wxyz quaternions. Returns a list."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]

    @classmethod
    def _rotate_world_quat_by_arm_yaw(cls, q_std, arm_yaw_deg: float):
        """Rotate the world-frame standard panda_hand orientation ``q_std``
        by the arm's root-link yaw in world:

            q_world_target = q_yaw(arm_yaw_deg) * q_std

        ``q_std = [0, 1, 0, 0]`` is the panda_hand world quat at the
        single-Franka home pose (gripper pointing along world +X). For
        dual_franka, L_panda_link0 is yawed -90° and R_panda_link0 +90°
        vs base_link, so the desired world-frame panda_hand quats are
        ``q_std`` rotated by those same yaws (pointing world -Y for L,
        world +Y for R). No inverse is taken — targets stay in world.
        """
        half = math.radians(arm_yaw_deg) * 0.5
        q_yaw = [math.cos(half), 0.0, 0.0, math.sin(half)]
        return cls._quat_mul(q_yaw, list(q_std))

    def _build_pose(self, pos_xyz: np.ndarray, quat_wxyz) -> torch.Tensor:
        device = self.env.device
        pos = torch.as_tensor(pos_xyz, device=device, dtype=torch.float32)
        quat = torch.as_tensor(quat_wxyz, device=device, dtype=torch.float32)
        return torch.cat([pos, quat], dim=0)

    def _dual_target(
        self, right_pose_7d: torch.Tensor, left_pose_7d: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate [right_7d, left_7d] for hand_id=-1 MoveL."""
        return torch.cat([right_pose_7d, left_pose_7d], dim=0)

    def _get_target_garment(self):
        """Resolve the target garment from (obj_type, obj_name, obj_id)."""
        scene_mgr = self.env.scene.scene_manager
        cat = scene_mgr.get_category(self.obj_type)[self.env_id]
        if self.obj_name not in cat:
            raise RuntimeError(
                f"[Fling] obj_name '{self.obj_name}' not in category "
                f"'{self.obj_type}' of env {self.env_id}. "
                f"Available: {list(cat.keys())}"
            )
        obj_list = cat[self.obj_name]
        if self.obj_id >= len(obj_list):
            raise RuntimeError(
                f"[Fling] obj_id {self.obj_id} out of range for "
                f"'{self.obj_type}/{self.obj_name}' (size={len(obj_list)})."
            )
        return obj_list[self.obj_id]

    def _build_waypoints(self):
        garment = self._get_target_garment()
        kp_dict = garment.get_keypoint()
        if not kp_dict:
            raise RuntimeError(
                f"[Fling] no keypoints available for garment "
                f"{self.obj_type}/{self.obj_name}[{self.obj_id}] in env {self.env_id}."
            )
        if self.left_keypoint_name not in kp_dict:
            raise RuntimeError(
                f"[Fling] left keypoint '{self.left_keypoint_name}' not found. "
                f"Available: {list(kp_dict.keys())}"
            )
        if self.right_keypoint_name not in kp_dict:
            raise RuntimeError(
                f"[Fling] right keypoint '{self.right_keypoint_name}' not found. "
                f"Available: {list(kp_dict.keys())}"
            )

        raw_a = np.asarray(kp_dict[self.left_keypoint_name], dtype=np.float32)
        raw_b = np.asarray(kp_dict[self.right_keypoint_name], dtype=np.float32)

        # Garment kp labels (top_left / top_right etc) follow the
        # garment's body frame and don't necessarily match the ROBOT's
        # L/R arms. dual_franka has L_panda_link0 at world +Y and
        # R_panda_link0 at -Y, so route the larger-Y kp to the left
        # arm and the smaller-Y kp to the right arm.
        if raw_a[1] >= raw_b[1]:
            left_kp, right_kp = raw_a, raw_b
            l_label, r_label = self.left_keypoint_name, self.right_keypoint_name
        else:
            left_kp, right_kp = raw_b, raw_a
            l_label, r_label = self.right_keypoint_name, self.left_keypoint_name

        # 1) Pull each kp inward along the LR segment (toward midline).
        lr_vec = right_kp - left_kp
        lr_dist = float(np.linalg.norm(lr_vec))
        if lr_dist > 1e-4 and self.inward_shift > 0:
            step = self.inward_shift / lr_dist
            left_kp = left_kp + lr_vec * step
            right_kp = right_kp - lr_vec * step
        # 2) Push each kp along world +X (away from neckline at -X).
        if self.grasp_kp_x_shift != 0.0:
            left_kp = left_kp.copy()
            right_kp = right_kp.copy()
            left_kp[0] += self.grasp_kp_x_shift
            right_kp[0] += self.grasp_kp_x_shift

        # Panda_hand origin sits `gripper_length` above the fingertip when
        # the gripper is top-down, so every phase adds it to Z.
        gl = self.gripper_length
        reach_offset = np.array([0.0, 0.0, self.reach_z_offset + gl], dtype=np.float32)
        lift_offset = np.array([0.0, 0.0, self.lift_height + gl], dtype=np.float32)
        fling_offset = np.array(
            [self.fling_distance, 0.0, self.lift_height + self.fling_apex + gl],
            dtype=np.float32,
        )
        drop_offset = np.array(
            [self.drop_distance, 0.0, self.drop_height + gl], dtype=np.float32
        )

        self.reach_xyz_left = left_kp + reach_offset
        self.lift_xyz_left = left_kp + lift_offset
        self.fling_xyz_left = left_kp + fling_offset
        self.drop_xyz_left = left_kp + drop_offset

        self.reach_xyz_right = right_kp + reach_offset
        self.lift_xyz_right = right_kp + lift_offset
        self.fling_xyz_right = right_kp + fling_offset
        self.drop_xyz_right = right_kp + drop_offset

        self.reach_pose_left = self._build_pose(
            self.reach_xyz_left, self.grasp_quat_left
        )
        self.lift_pose_left = self._build_pose(self.lift_xyz_left, self.grasp_quat_left)
        self.fling_pose_left = self._build_pose(
            self.fling_xyz_left, self.grasp_quat_left
        )
        self.drop_pose_left = self._build_pose(self.drop_xyz_left, self.grasp_quat_left)

        self.reach_pose_right = self._build_pose(
            self.reach_xyz_right, self.grasp_quat_right
        )
        self.lift_pose_right = self._build_pose(
            self.lift_xyz_right, self.grasp_quat_right
        )
        self.fling_pose_right = self._build_pose(
            self.fling_xyz_right, self.grasp_quat_right
        )
        self.drop_pose_right = self._build_pose(
            self.drop_xyz_right, self.grasp_quat_right
        )

        self.left_kp = left_kp
        self.right_kp = right_kp

        if self.debug:
            print(
                f"[Fling][env={self.env_id}] waypoints for group='{self.group_name}' "
                f"L_arm←'{l_label}'(y={left_kp[1]:.3f}) "
                f"R_arm←'{r_label}'(y={right_kp[1]:.3f}) "
                f"inward={self.inward_shift:.3f} kp_x_shift={self.grasp_kp_x_shift:+.3f} "
                f"reach_z_offset={self.reach_z_offset:.3f} "
                f"grasp_left={self.grasp_quat_left} "
                f"grasp_right={self.grasp_quat_right}"
            )
            print(
                f"  [left ] kp={left_kp.tolist()} "
                f"reach={self.reach_xyz_left.tolist()} "
                f"lift={self.lift_xyz_left.tolist()} "
                f"fling={self.fling_xyz_left.tolist()} "
                f"drop={self.drop_xyz_left.tolist()}"
            )
            print(
                f"  [right] kp={right_kp.tolist()} "
                f"reach={self.reach_xyz_right.tolist()} "
                f"lift={self.lift_xyz_right.tolist()} "
                f"fling={self.fling_xyz_right.tolist()} "
                f"drop={self.drop_xyz_right.tolist()}"
            )

    def _parse_action(self, action: list[Any]):
        # Grasp-style: action[0]=skill, action[1]=robot_id,
        # action[2]=obj_type, action[3]=obj_name, action[4]=obj_id,
        # action[5:] = group_name OR (left_kp, right_kp).
        if len(action) < 6:
            raise ValueError(
                f"[Fling] action too short: {action}. "
                f"Expected ['Fling', robot_id, obj_type, obj_name, obj_id, group] "
                f"or ['Fling', robot_id, obj_type, obj_name, obj_id, left_kp, right_kp]."
            )
        self.robot_id = int(action[1])
        self.hand_id = -1  # always dual-arm
        self.obj_type = str(action[2])
        self.obj_name = str(action[3])
        self.obj_id = int(action[4])

        if len(action) == 6:
            group = str(action[5])
            if group not in KEYPOINT_GROUPS:
                raise ValueError(
                    f"[Fling] unknown keypoint group '{group}'. "
                    f"Known groups: {list(KEYPOINT_GROUPS.keys())}"
                )
            self.group_name = group
            self.left_keypoint_name, self.right_keypoint_name = KEYPOINT_GROUPS[group]
        elif len(action) >= 7:
            self.group_name = None
            self.left_keypoint_name = str(action[5])
            self.right_keypoint_name = str(action[6])

    def _command_signature(self, action: list[Any]):
        head = (int(action[1]), str(action[2]), str(action[3]), int(action[4]))
        if len(action) == 6:
            return ("group", *head, str(action[5]))
        return ("pair", *head, str(action[5]), str(action[6]))

    # ---------- AtomicSkill interface ----------

    def reset(self, action: list[Any]):
        self._parse_action(action)
        self.current_state = "ready"
        self.current_command = list(action)
        self.current_phase = "reach"
        self._cached_grav_scale = None
        self._build_waypoints()

    def refresh(self, action: list[Any]):
        new_sig = self._command_signature(action)
        old_sig = (
            None
            if self.current_command is None
            else self._command_signature(self.current_command)
        )
        command_changed = new_sig != old_sig
        self._parse_action(action)
        self.current_command = list(action)
        if command_changed or self.current_phase is None:
            self.current_phase = "reach"
            self._cached_grav_scale = None
            self._build_waypoints()

    def _submit_dual_movel(
        self,
        right_pose: torch.Tensor,
        left_pose: torch.Tensor,
        phase_label: str,
        interp: bool = False,
    ) -> dict:
        # planner_mode = 0 → snap (no motiongen, no interp).
        # planner_mode = 2 → MoveL.INTERP_MODE: lerps from current eef
        #   pose to target over MoveL's config'd ``interp_steps`` calls.
        target = self._dual_target(right_pose, left_pose)
        mode = 2 if interp else 0
        action = {"MoveL": ((self.robot_id, -1, mode), target)}
        if self.debug:
            right = right_pose.detach().cpu().numpy().tolist()
            left = left_pose.detach().cpu().numpy().tolist()
            print(
                f"[Fling][env={self.env_id}] submit MoveL phase={phase_label} "
                f"robot_id={self.robot_id} hand_id=-1 mode={mode} "
                f"right_7d={right} left_7d={left}"
            )
        return action

    def _submit_dual_gripper(self, close: bool, phase_label: str) -> dict:
        value = 1.0 if close else 0.0
        target = torch.tensor(
            [value, value], device=self.env.device, dtype=torch.float32
        )
        action = {"ParallelGripper": ((self.robot_id, -1, 0), target)}
        if self.debug:
            print(
                f"[Fling][env={self.env_id}] submit ParallelGripper phase={phase_label} "
                f"robot_id={self.robot_id} hand_id=-1 target={target.tolist()}"
            )
        return action

    # ---------- visualization ----------

    _PHASE_COLORS = {
        "reach": (1.0, 0.8, 0.1, 0.9),
        "close_gripper": (1.0, 0.8, 0.1, 0.9),
        "lift_up": (0.2, 0.9, 0.2, 0.9),
        "fling_forward": (0.2, 0.6, 1.0, 0.9),
        "drop": (0.9, 0.4, 0.9, 0.9),
        "open_gripper": (0.9, 0.4, 0.9, 0.9),
    }

    def _current_target_xyz(self):
        mapping = {
            "reach": (self.reach_xyz_right, self.reach_xyz_left),
            "close_gripper": (self.reach_xyz_right, self.reach_xyz_left),
            "lift_up": (self.lift_xyz_right, self.lift_xyz_left),
            "fling_forward": (self.fling_xyz_right, self.fling_xyz_left),
            "drop": (self.drop_xyz_right, self.drop_xyz_left),
            "open_gripper": (self.drop_xyz_right, self.drop_xyz_left),
        }
        return mapping.get(self.current_phase)

    def _current_target_pose_7d(self):
        mapping = {
            "reach": (self.reach_pose_right, self.reach_pose_left),
            "close_gripper": (self.reach_pose_right, self.reach_pose_left),
            "lift_up": (self.lift_pose_right, self.lift_pose_left),
            "fling_forward": (self.fling_pose_right, self.fling_pose_left),
            "drop": (self.drop_pose_right, self.drop_pose_left),
            "open_gripper": (self.drop_pose_right, self.drop_pose_left),
        }
        pair = mapping.get(self.current_phase)
        if pair is None:
            return None
        return torch.stack([pair[0], pair[1]], dim=0)

    def _draw_viz(self):
        """Red pick keypoints + phase-colored waypoint points + target axes.
        Renders only for env_id 0 to avoid multi-env clobbering the buffer."""
        if not self.visualize or self.env_id != 0:
            return
        if self.left_kp is None or self.right_kp is None:
            return

        draw_waypoints(
            [self.left_kp.tolist(), self.right_kp.tolist()],
            point_size=14.0,
            color=(1.0, 0.0, 0.0, 1.0),
            clear_existing=True,
        )

        tgt = self._current_target_xyz()
        if tgt is not None:
            draw_waypoints(
                [tgt[0].tolist(), tgt[1].tolist()],
                point_size=10.0,
                color=self._PHASE_COLORS.get(self.current_phase, (1.0, 1.0, 1.0, 0.9)),
                clear_existing=False,
            )

        pose_pair = self._current_target_pose_7d()
        if pose_pair is not None:
            draw_grasp_samples_as_axes(
                pose_pair,
                axis_length=0.06,
                line_thickness=2,
                line_opacity=0.9,
                clear_existing=True,
            )

    # ---------- gravity scale helper ----------

    def _set_gravity_scale_for_phase(self):
        """Toggle the garment particle_material gravity_scale based on
        the current phase (lift / fling → low, others → default)."""
        target = (
            self.gravity_scale_fling
            if self.current_phase in self.phases_with_low_gravity
            else self.gravity_scale_default
        )
        if self._cached_grav_scale == target:
            return
        try:
            garment = self._get_target_garment()
        except Exception:
            return
        pm = getattr(garment, "particle_material", None)
        if pm is None or not hasattr(pm, "set_gravity_scale"):
            return
        try:
            pm.set_gravity_scale(float(target))
            self._cached_grav_scale = target
            if self.debug:
                print(
                    f"[Fling][env={self.env_id}] gravity_scale -> {target}"
                    f" (phase={self.current_phase})"
                )
        except Exception as e:
            if self.debug:
                print(f"[Fling][env={self.env_id}] gravity_scale set FAIL: {e}")

    # ---------- step ----------

    def step(self):
        if self.current_state == "failed":
            self.current_action = "Failed"
            return "Failed"

        self.current_state = "running"
        self._set_gravity_scale_for_phase()
        self._draw_viz()

        # MoveL decides whether to lerp or snap based on its own config
        # (``interp_steps`` hyperparameter in global_planner/default.yaml);
        # the skill just emits the final target and lets MoveL handle the
        # trajectory + completion signal.
        if self.current_phase == "reach":
            # Reach goes straight down to grab the cloth — snap, no interp.
            self.current_action = self._submit_dual_movel(
                self.reach_pose_right, self.reach_pose_left, "reach", interp=False
            )
        elif self.current_phase == "close_gripper":
            self.current_action = self._submit_dual_gripper(
                close=True, phase_label="close_gripper"
            )
        elif self.current_phase == "lift_up":
            self.current_action = self._submit_dual_movel(
                self.lift_pose_right, self.lift_pose_left, "lift_up", interp=True
            )
        elif self.current_phase == "fling_forward":
            self.current_action = self._submit_dual_movel(
                self.fling_pose_right,
                self.fling_pose_left,
                "fling_forward",
                interp=True,
            )
        elif self.current_phase == "drop":
            self.current_action = self._submit_dual_movel(
                self.drop_pose_right, self.drop_pose_left, "drop", interp=True
            )
        elif self.current_phase == "open_gripper":
            self.current_action = self._submit_dual_gripper(
                close=False, phase_label="open_gripper"
            )
        else:
            self.current_state = "failed"
            self.current_action = None
            return None
        return self.current_action

    def _advance_phase(self) -> str | None:
        order = [
            "reach",
            "close_gripper",
            "lift_up",
            "fling_forward",
            "drop",
            "open_gripper",
        ]
        try:
            i = order.index(self.current_phase)
        except ValueError:
            return None
        if i + 1 < len(order):
            return order[i + 1]
        return None

    def update(self, info):
        if self.current_state == "failed":
            self.current_state = "failed: atomicskill failed to plan"
            return {
                "type": "Fling",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
                "phase": self.current_phase,
            }

        # All phases (incl. lift / fling / drop with their interp) advance
        # via gp_info["finished"] — MoveL signals finished when its own
        # interp substep counter reaches its config'd ``interp_steps``,
        # so this skill no longer needs to track substeps itself.

        gp_info = info.get("global_planner_info", None)
        if gp_info is None or gp_info[self.env_id] is None:
            return {
                "type": "Fling",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state or "ready",
                "truncated": 0,
                "phase": self.current_phase,
            }

        if gp_info[self.env_id]["finished"]:
            next_phase = self._advance_phase()
            if next_phase is None:
                self.current_state = "finished"
                print(f"[Fling] env_id={self.env_id} phase=completed")
                return {
                    "type": "Fling",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": True,
                    "state": self.current_state,
                    "truncated": 0,
                    "phase": "completed",
                }
            print(f"[Fling] env_id={self.env_id} phase={next_phase}")
            self.current_phase = next_phase
            return {
                "type": "Fling",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": f"running: {next_phase}",
                "truncated": 0,
                "phase": next_phase,
            }
        elif gp_info[self.env_id]["truncated"] == 1:
            self.current_state = "truncated: env terminated first"
            return {
                "type": "Fling",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
                "phase": self.current_phase,
            }
        elif gp_info[self.env_id]["truncated"] == 2:
            self.current_state = "truncated: env truncated first"
            return {
                "type": "Fling",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
                "phase": self.current_phase,
            }
        elif gp_info[self.env_id]["truncated"] == 3:
            self.current_state = "failed: global planner failed to plan"
            return {
                "type": "Fling",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
                "phase": self.current_phase,
            }
        else:
            self.current_state = "running"
            return {
                "type": "Fling",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
                "phase": self.current_phase,
            }
