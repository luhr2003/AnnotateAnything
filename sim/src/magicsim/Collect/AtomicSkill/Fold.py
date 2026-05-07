from typing import Any

import math

import numpy as np
import torch
from magicsim.Collect.AtomicSkill.AtomicSkill import AtomicSkill
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


# Ordered phases executed by the Fold skill. Each MoveL phase emits a
# dual-arm 14D target ``[right_7d, left_7d]`` (hand_id=-1); each gripper
# phase emits a 2D target ``[right, left]``.
PHASE_ORDER = [
    # ---- sleeve fold ----
    # pre_reach_sleeve hovers back along the approach direction so that
    # reach_sleeve becomes a pure slide-in along the gripper +Z (the
    # "insertion" feel when the gripper is tilted).
    "pre_reach_sleeve",
    "reach_sleeve",
    "close_gripper_sleeve",
    "lift_sleeve",
    "move_sleeve",
    "drop_sleeve",
    "open_gripper_sleeve",
    "retract_sleeve",
    # ---- bottom-to-shoulder fold ----
    "pre_reach_bottom",
    "reach_bottom",
    "close_gripper_bottom",
    "lift_bottom",
    "move_bottom",
    "drop_bottom",
    "open_gripper_bottom",
    "retract_bottom",
]


class Fold(AtomicSkill):
    """Bimanual two-stage garment fold on the DualFranka scene.

    Scene: one ``dualmanipulator`` robot (``DualFranka_0``) with two EEFs
    — L_panda (left arm, at +Y side, faces -Y) and R_panda (right arm,
    at -Y side, faces +Y). Hand ids follow the repo convention:
    ``hand_id=0`` → right, ``hand_id=1`` → left, ``hand_id=-1`` → both.

    Stage 1 — sleeve fold:
        pick the sleeve tips (``top_left`` / ``top_right``) and place each
        at its reflection across the shoulder→bottom line on the same side::

            left arm : pick=top_left,  place=reflect(top_left,  left_shoulder,  bottom_left)
            right arm: pick=top_right, place=reflect(top_right, right_shoulder, bottom_right)

    Stage 2 — bottom-to-shoulder fold::

            left arm : pick=bottom_left,  place=left_shoulder
            right arm: pick=bottom_right, place=right_shoulder

    Phases::

        (sleeve)  reach → close → lift → move → drop → open → retract →
        (bottom)  reach → close → lift → move → drop → open → retract

    Targets are emitted in **world frame** (MoveL accepts world targets
    and goes via IK abs; see ``Env/Planner/Services/README.md`` §2 / §5).
    Both arms perform the same top-down grasp, but because L_panda_link0
    / R_panda_link0 are yawed in world (-90° / +90° around Z in the
    dual_franka URDF), an arm-local top-down quat (``Rx(180°)``) lifts
    to two different world-frame quaternions. The skill composes those
    from ``arm_local_grasp_quat`` + ``left_arm_yaw_deg`` /
    ``right_arm_yaw_deg`` (URDF values) so the two arms just come
    straight down onto their keypoints.

    Action format::

        ["Fold", robot_id, obj_type, obj_name, obj_id]
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.robot_id = int(getattr(config, "robot_id", 0))
        # Always dual-arm.
        self.hand_id = -1

        # Heights (meters). Applied on top of keypoint z (vertical-only).
        # Aligned with ``Task/Garment/Env/Test/TestFoldEnv.py`` constants.
        self.lift_height = float(getattr(config, "lift_height", 0.1))
        self.drop_height = float(getattr(config, "drop_height", 0.03))
        # Clearance above the place target after release.
        self.retract_height = float(getattr(config, "retract_height", 0.15))
        # Franka gripper length from panda_hand frame origin (wrist) to
        # fingertip, applied along the per-arm approach direction (the
        # gripper's +Z rotated into world). With a tilted gripper this is
        # NOT pure +Z, so the wrist offset is computed as
        # ``-gripper_length * approach_dir`` per arm — keeps the
        # fingertip on the commanded keypoint regardless of tilt.
        self.gripper_length = float(getattr(config, "gripper_length", 0.2))
        # Insertion knobs. With a slanted gripper the fingertip slides
        # in along the approach direction. ``pre_reach_distance`` is how
        # far back along the approach the gripper starts; the slide-in
        # phase then advances by ``pre_reach_distance + insertion_depth``
        # to reach a fingertip ``insertion_depth`` past the keypoint
        # (positive = pressed into the cloth).
        self.pre_reach_distance = float(getattr(config, "pre_reach_distance", 0.06))
        self.insertion_depth = float(getattr(config, "insertion_depth", 0.015))
        # Pull each pick keypoint inward toward its arm's partner before
        # commanding the arms; lands the fingers on fabric body instead
        # of the hem / seam edge. Applied to sleeve and bottom picks
        # independently.
        self.inward_shift = float(getattr(config, "inward_shift", 0.05))
        # World +X nudge of the grasp anchor (Fling-style knob). Pushes
        # each pick by this much along world X after inward_shift.
        self.grasp_kp_x_shift = float(getattr(config, "grasp_kp_x_shift", 0.02))
        # Minimum world-Y separation between the left/right place targets.
        # The sleeve reflection (and the shoulder targets for the bottom
        # fold) can fall close to / past the body midline, which would
        # collide the two panda_hands. Place pairs are clamped so each
        # arm stays on its own side with this much total Y-gap.
        self.min_lr_separation = float(getattr(config, "min_lr_separation", 0.20))

        # Top-down grasp orientation expressed in each arm's root-link
        # frame (wxyz). Default [0, 1, 0, 0] = Rx(180°): the panda_hand
        # Z axis, which points forward from each arm's root, is flipped
        # to point "down" relative to that arm. The world-frame target
        # submitted to MoveL is then ``Rz(arm_yaw) ⊗ arm_local_quat`` —
        # because L_panda_link0 / R_panda_link0 are yawed in world, the
        # identical top-down grasp corresponds to two different
        # world-frame quaternions.
        self.arm_local_grasp_quat = list(
            getattr(
                config,
                "arm_local_grasp_quat",
                getattr(config, "grasp_quat", [0.0, 1.0, 0.0, 0.0]),
            )
        )
        # URDF convention (dual_franka): R_panda_link0 yaw=+90°,
        # L_panda_link0 yaw=-90° around world Z.
        self.right_arm_yaw_deg = float(getattr(config, "right_arm_yaw_deg", 90.0))
        self.left_arm_yaw_deg = float(getattr(config, "left_arm_yaw_deg", -90.0))
        # Slant the gripper instead of pointing straight down. Composed
        # in arm-local frame BEFORE the per-arm world yaw lift, so the
        # same ``tilt_deg`` produces a symmetric outward tilt for the two
        # arms (each tips forward along its own reach direction).
        self.tilt_deg = float(getattr(config, "tilt_deg", 45.0))
        self._arm_local_grasp_quat_tilted = self._tilt_arm_local(
            self.arm_local_grasp_quat, self.tilt_deg
        )
        # Direct per-arm overrides (in world frame) win if provided.
        self.grasp_quat_right = (
            list(config.grasp_quat_right)
            if hasattr(config, "grasp_quat_right")
            and getattr(config, "grasp_quat_right", None) is not None
            else self._arm_quat_to_world(
                self._arm_local_grasp_quat_tilted, self.right_arm_yaw_deg
            )
        )
        self.grasp_quat_left = (
            list(config.grasp_quat_left)
            if hasattr(config, "grasp_quat_left")
            and getattr(config, "grasp_quat_left", None) is not None
            else self._arm_quat_to_world(
                self._arm_local_grasp_quat_tilted, self.left_arm_yaw_deg
            )
        )
        # Per-arm approach directions in world (gripper +Z). Used to
        # offset wrist by ``-gripper_length * approach_dir`` so the
        # fingertip lands on the commanded keypoint regardless of tilt,
        # and to drive the pre_reach → reach slide-in along the approach.
        self.approach_dir_right = self._approach_dir_from_quat(self.grasp_quat_right)
        self.approach_dir_left = self._approach_dir_from_quat(self.grasp_quat_left)

        self.debug = bool(getattr(config, "debug", True))

        # Phase-waypoint visualization.
        self.visualize = bool(getattr(config, "visualize", False))
        self.viz_radius = float(getattr(config, "viz_radius", 0.025))
        self.viz_pick_color = list(getattr(config, "viz_pick_color", [1.0, 0.2, 0.2]))
        self.viz_place_color = list(getattr(config, "viz_place_color", [0.2, 1.0, 0.2]))
        self._last_viz_phase: str | None = None

        self.current_phase: str | None = None

        # Target garment identifiers (Grasp-style).
        self.obj_type: str | None = None
        self.obj_name: str | None = None
        self.obj_id: int | None = None

        # Per-phase per-arm xyz waypoints (env-local, computed at reset).
        self._wp_xyz: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # Per-phase per-arm 7D poses (device tensors).
        self._wp_pose: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    # ---------- geometry helpers ----------

    @staticmethod
    def _quat_mul(q1, q2):
        """Hamilton product of two wxyz quaternions, returns a list."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]

    @classmethod
    def _arm_quat_to_world(cls, q_arm_local, arm_yaw_deg: float):
        """Lift an arm-local quaternion to world frame.

        The arm root (L/R_panda_link0) is yawed by ``arm_yaw_deg`` around
        world Z relative to the robot base. Given a target orientation
        expressed in that arm-local frame, the equivalent world-frame
        orientation is ``Rz(arm_yaw) ⊗ q_arm_local``.
        """
        half = math.radians(arm_yaw_deg) * 0.5
        q_yaw = [math.cos(half), 0.0, 0.0, math.sin(half)]
        return cls._quat_mul(q_yaw, list(q_arm_local))

    @classmethod
    def _tilt_arm_local(cls, q_arm_local, tilt_deg: float):
        """Slant an arm-local quat: ``Rx(tilt_deg) ⊗ q_arm_local``.

        Composed before ``_arm_quat_to_world``. Positive ``tilt_deg``
        tips the approach (the panda_hand's down-pointing Z after
        Rx(180°)) toward arm-local +Y (the arm's reach direction).
        """
        half = math.radians(tilt_deg) * 0.5
        q_tilt = [math.cos(half), math.sin(half), 0.0, 0.0]  # Rx(tilt_deg)
        return cls._quat_mul(q_tilt, list(q_arm_local))

    @staticmethod
    def _approach_dir_from_quat(q) -> np.ndarray:
        """Rotate (0,0,1) by world quat (wxyz) → unit gripper +Z (approach).

        For Franka's ``panda_hand``, +Z runs from wrist to between the
        fingertips, i.e. the approach direction. With a tilted gripper
        this picks up a horizontal component along the arm's reach side.
        """
        w, x, y, z = q
        return np.array(
            [
                2.0 * (x * z + y * w),
                2.0 * (y * z - x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
            dtype=np.float32,
        )

    # Maps each phase to the ``_wp_xyz`` key whose waypoint marks the
    # "current gripper location" for that phase, plus a tag telling
    # whether this step is about the pick keypoint or the place target.
    # Gripper phases reuse the preceding MoveL waypoint.
    _PHASE_VIZ = {
        "pre_reach_sleeve": ("pre_reach_sleeve", "pick"),
        "reach_sleeve": ("reach_sleeve", "pick"),
        "close_gripper_sleeve": ("reach_sleeve", "pick"),
        "lift_sleeve": ("lift_sleeve", "pick"),
        "move_sleeve": ("move_sleeve", "place"),
        "drop_sleeve": ("drop_sleeve", "place"),
        "open_gripper_sleeve": ("drop_sleeve", "place"),
        "retract_sleeve": ("retract_sleeve", "place"),
        "pre_reach_bottom": ("pre_reach_bottom", "pick"),
        "reach_bottom": ("reach_bottom", "pick"),
        "close_gripper_bottom": ("reach_bottom", "pick"),
        "lift_bottom": ("lift_bottom", "pick"),
        "move_bottom": ("move_bottom", "place"),
        "drop_bottom": ("drop_bottom", "place"),
        "open_gripper_bottom": ("drop_bottom", "place"),
        "retract_bottom": ("retract_bottom", "place"),
    }

    @staticmethod
    def _reflect_across_line(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Reflect ``p`` across the line through ``a`` and ``b`` in 3D."""
        d = b - a
        denom = float(np.dot(d, d))
        if denom < 1e-9:
            return 2.0 * a - p
        t = float(np.dot(p - a, d)) / denom
        foot = a + t * d
        return 2.0 * foot - p

    @staticmethod
    def _shift_inward(
        left_xyz: np.ndarray, right_xyz: np.ndarray, dist: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pull two opposite-side picks toward each other by ``dist`` (m)."""
        lr = right_xyz - left_xyz
        lr_dist = float(np.linalg.norm(lr))
        if lr_dist < 1e-4 or dist <= 0.0:
            return left_xyz.copy(), right_xyz.copy()
        step = dist / lr_dist
        return left_xyz + lr * step, right_xyz - lr * step

    @staticmethod
    def _clamp_place_to_sides(
        L_place: np.ndarray,
        R_place: np.ndarray,
        midline_y: float,
        min_separation: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keep L_place at +Y side and R_place at -Y side of ``midline_y``.

        L_place is forced to ``y >= midline_y + min_separation/2``; R_place
        to ``y <= midline_y - min_separation/2``. Anything that would
        cross past the midline (e.g. an over-aggressive sleeve reflection,
        or shoulders that sit too close together) gets snapped back to
        the half-separation boundary on its own side. XZ untouched.
        """
        half = float(min_separation) * 0.5
        L_out = L_place.copy()
        R_out = R_place.copy()
        L_min_y = float(midline_y) + half
        R_max_y = float(midline_y) - half
        if L_out[1] < L_min_y:
            L_out[1] = L_min_y
        if R_out[1] > R_max_y:
            R_out[1] = R_max_y
        return L_out, R_out

    @staticmethod
    def _resolve_garment_side_mapping(kp: dict) -> tuple[str, dict]:
        """Decide which garment side belongs to which robot arm.

        Garment labels (``*_left`` / ``*_right``) follow the cloth's own
        body frame and are NOT a safe match for the ROBOT's L/R arms.
        dual_franka has L_panda_link0 at world +Y and R_panda_link0 at
        -Y, so assign the keypoint with the larger world Y to the left
        arm. Decision is made once from ``top_*`` and applied
        consistently to bottom + shoulder.
        """
        tl = np.asarray(kp["top_left"], dtype=np.float32)
        tr = np.asarray(kp["top_right"], dtype=np.float32)
        if tl[1] >= tr[1]:
            labels_for_arm = {
                "L": ("top_left", "bottom_left", "left_shoulder"),
                "R": ("top_right", "bottom_right", "right_shoulder"),
            }
            decision = "garment_left → L_arm"
        else:
            labels_for_arm = {
                "L": ("top_right", "bottom_right", "right_shoulder"),
                "R": ("top_left", "bottom_left", "left_shoulder"),
            }
            decision = "garment_right → L_arm (flipped)"
        return decision, labels_for_arm

    def _build_pose(self, pos_xyz: np.ndarray, quat_wxyz: list) -> torch.Tensor:
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
        scene_mgr = self.env.scene.scene_manager
        cat = scene_mgr.get_category(self.obj_type)[self.env_id]
        if self.obj_name not in cat:
            raise RuntimeError(
                f"[Fold] obj_name '{self.obj_name}' not in category "
                f"'{self.obj_type}' of env {self.env_id}. "
                f"Available: {list(cat.keys())}"
            )
        obj_list = cat[self.obj_name]
        if self.obj_id >= len(obj_list):
            raise RuntimeError(
                f"[Fold] obj_id {self.obj_id} out of range for "
                f"'{self.obj_type}/{self.obj_name}' (size={len(obj_list)})."
            )
        return obj_list[self.obj_id]

    # ---------- planning ----------

    def _build_waypoints(self):
        garment = self._get_target_garment()
        kp_dict = garment.get_keypoint()
        required = (
            "top_left",
            "top_right",
            "left_shoulder",
            "right_shoulder",
            "bottom_left",
            "bottom_right",
        )
        missing = [k for k in required if k not in kp_dict]
        if missing:
            raise RuntimeError(
                f"[Fold] garment {self.obj_type}/{self.obj_name}[{self.obj_id}] "
                f"in env {self.env_id} missing keypoints {missing}. "
                f"Available: {list(kp_dict.keys())}"
            )

        # Arm-side assignment by world Y (consistent across sleeve /
        # bottom / shoulder); see TestOpenLoopFoldEnv.
        decision, labels_for_arm = self._resolve_garment_side_mapping(kp_dict)
        L_top, L_bot, L_shoulder_lbl = labels_for_arm["L"]
        R_top, R_bot, R_shoulder_lbl = labels_for_arm["R"]

        L_pick_sleeve = np.asarray(kp_dict[L_top], dtype=np.float32)
        R_pick_sleeve = np.asarray(kp_dict[R_top], dtype=np.float32)
        L_pick_bottom = np.asarray(kp_dict[L_bot], dtype=np.float32)
        R_pick_bottom = np.asarray(kp_dict[R_bot], dtype=np.float32)
        L_shoulder = np.asarray(kp_dict[L_shoulder_lbl], dtype=np.float32)
        R_shoulder = np.asarray(kp_dict[R_shoulder_lbl], dtype=np.float32)

        # Place targets (computed BEFORE inward shift on picks so the
        # reflection axis stays the original shoulder→bottom line).
        L_place_sleeve = self._reflect_across_line(
            L_pick_sleeve, L_shoulder, L_pick_bottom
        )
        R_place_sleeve = self._reflect_across_line(
            R_pick_sleeve, R_shoulder, R_pick_bottom
        )
        L_place_bottom = L_shoulder.copy()
        R_place_bottom = R_shoulder.copy()

        # Clamp place targets to each arm's own side of the body
        # midline (with min_lr_separation gap) so the two panda_hands
        # don't collide near the centerline.
        sleeve_midline_y = 0.5 * (L_pick_sleeve[1] + R_pick_sleeve[1])
        bottom_midline_y = 0.5 * (L_pick_bottom[1] + R_pick_bottom[1])
        L_place_sleeve_raw = L_place_sleeve.copy()
        R_place_sleeve_raw = R_place_sleeve.copy()
        L_place_bottom_raw = L_place_bottom.copy()
        R_place_bottom_raw = R_place_bottom.copy()
        L_place_sleeve, R_place_sleeve = self._clamp_place_to_sides(
            L_place_sleeve, R_place_sleeve, sleeve_midline_y, self.min_lr_separation
        )
        L_place_bottom, R_place_bottom = self._clamp_place_to_sides(
            L_place_bottom, R_place_bottom, bottom_midline_y, self.min_lr_separation
        )

        # Inward shift on picks (sleeve + bottom independently).
        L_pick_sleeve, R_pick_sleeve = self._shift_inward(
            L_pick_sleeve, R_pick_sleeve, self.inward_shift
        )
        L_pick_bottom, R_pick_bottom = self._shift_inward(
            L_pick_bottom, R_pick_bottom, self.inward_shift
        )

        # World +X nudge on picks (Fling-style grasp_kp_x_shift).
        if self.grasp_kp_x_shift != 0.0:
            for arr in (
                L_pick_sleeve,
                R_pick_sleeve,
                L_pick_bottom,
                R_pick_bottom,
            ):
                arr[0] += self.grasp_kp_x_shift

        # World-Z fingertip clearances (no gripper_length here — the
        # wrist offset is applied along the per-arm approach direction).
        z_lift = np.array([0.0, 0.0, self.lift_height], dtype=np.float32)
        z_drop = np.array([0.0, 0.0, self.drop_height], dtype=np.float32)
        z_retract = np.array([0.0, 0.0, self.retract_height], dtype=np.float32)

        gl = self.gripper_length
        appR = self.approach_dir_right
        appL = self.approach_dir_left
        ins = self.insertion_depth
        pre = self.pre_reach_distance

        def wrist(R_finger, L_finger):
            """fingertip → wrist via per-arm approach direction."""
            return (R_finger - gl * appR, L_finger - gl * appL)

        self._wp_xyz = {}

        # ---- sleeve fold ----
        # pre_reach: fingertip ``pre_reach_distance`` back along approach
        # from the reach target. reach: fingertip ``insertion_depth``
        # past the keypoint along approach (positive = into the cloth).
        self._wp_xyz["pre_reach_sleeve"] = wrist(
            R_pick_sleeve + (ins - pre) * appR,
            L_pick_sleeve + (ins - pre) * appL,
        )
        self._wp_xyz["reach_sleeve"] = wrist(
            R_pick_sleeve + ins * appR,
            L_pick_sleeve + ins * appL,
        )
        self._wp_xyz["lift_sleeve"] = wrist(
            R_pick_sleeve + z_lift,
            L_pick_sleeve + z_lift,
        )
        self._wp_xyz["move_sleeve"] = wrist(
            R_place_sleeve + z_lift,
            L_place_sleeve + z_lift,
        )
        self._wp_xyz["drop_sleeve"] = wrist(
            R_place_sleeve + z_drop,
            L_place_sleeve + z_drop,
        )
        self._wp_xyz["retract_sleeve"] = wrist(
            R_place_sleeve + z_retract,
            L_place_sleeve + z_retract,
        )

        # ---- bottom fold ----
        self._wp_xyz["pre_reach_bottom"] = wrist(
            R_pick_bottom + (ins - pre) * appR,
            L_pick_bottom + (ins - pre) * appL,
        )
        self._wp_xyz["reach_bottom"] = wrist(
            R_pick_bottom + ins * appR,
            L_pick_bottom + ins * appL,
        )
        self._wp_xyz["lift_bottom"] = wrist(
            R_pick_bottom + z_lift,
            L_pick_bottom + z_lift,
        )
        self._wp_xyz["move_bottom"] = wrist(
            R_place_bottom + z_lift,
            L_place_bottom + z_lift,
        )
        self._wp_xyz["drop_bottom"] = wrist(
            R_place_bottom + z_drop,
            L_place_bottom + z_drop,
        )
        self._wp_xyz["retract_bottom"] = wrist(
            R_place_bottom + z_retract,
            L_place_bottom + z_retract,
        )

        # Cache for debug.
        self._side_mapping = decision

        # Precompute device pose tensors for each MoveL phase with
        # per-arm world-frame quaternions (no arm-frame yaw correction —
        # MoveL accepts world targets directly).
        self._wp_pose = {
            phase: (
                self._build_pose(right, self.grasp_quat_right),
                self._build_pose(left, self.grasp_quat_left),
            )
            for phase, (right, left) in self._wp_xyz.items()
        }

        if self.debug:
            print(
                f"[Fold][env={self.env_id}] side mapping: {decision}; "
                f"L picks sleeve='{L_top}' bottom='{L_bot}' shoulder='{L_shoulder_lbl}'; "
                f"R picks sleeve='{R_top}' bottom='{R_bot}' shoulder='{R_shoulder_lbl}'"
            )
            print(
                f"[Fold][env={self.env_id}] sleeve places (raw → clamped) "
                f"min_sep={self.min_lr_separation:.3f}: "
                f"L y {L_place_sleeve_raw[1]:.3f} → {L_place_sleeve[1]:.3f}; "
                f"R y {R_place_sleeve_raw[1]:.3f} → {R_place_sleeve[1]:.3f}"
            )
            print(
                f"[Fold][env={self.env_id}] bottom places (raw → clamped): "
                f"L y {L_place_bottom_raw[1]:.3f} → {L_place_bottom[1]:.3f}; "
                f"R y {R_place_bottom_raw[1]:.3f} → {R_place_bottom[1]:.3f}; "
                f"inward_shift={self.inward_shift:.3f} m"
            )
            print(
                f"[Fold][env={self.env_id}] grasp_quat (world, wxyz) "
                f"right={self.grasp_quat_right} left={self.grasp_quat_left} "
                f"(arm_local={self.arm_local_grasp_quat}, "
                f"yaw right={self.right_arm_yaw_deg} "
                f"left={self.left_arm_yaw_deg})"
            )

    def _parse_action(self, action: list[Any]):
        # ["Fold", robot_id, obj_type, obj_name, obj_id]
        if len(action) < 5:
            raise ValueError(
                f"[Fold] action too short: {action}. "
                f"Expected ['Fold', robot_id, obj_type, obj_name, obj_id]."
            )
        self.robot_id = int(action[1])
        self.hand_id = -1
        self.obj_type = str(action[2])
        self.obj_name = str(action[3])
        self.obj_id = int(action[4])

    def _command_signature(self, action: list[Any]):
        return (int(action[1]), str(action[2]), str(action[3]), int(action[4]))

    # ---------- AtomicSkill interface ----------

    def reset(self, action: list[Any]):
        self._parse_action(action)
        self.current_state = "ready"
        self.current_command = list(action)
        self.current_phase = PHASE_ORDER[0]
        self._last_viz_phase = None
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
            self.current_phase = PHASE_ORDER[0]
            self._last_viz_phase = None
            self._build_waypoints()

    # ---------- visualization ----------

    def _visualize_phase(self, phase: str) -> None:
        """Draw right/left target markers for the current phase.

        Pick-stage phases (reach / close / lift) show the pick keypoint's
        current hover target in ``viz_pick_color``; place-stage phases
        (move / drop / open / retract) show the place target in
        ``viz_place_color``. Re-created on phase change to refresh color.
        """
        if not self.visualize:
            return
        if phase == self._last_viz_phase:
            return
        viz = self._PHASE_VIZ.get(phase)
        if viz is None:
            return
        wp_key, kind = viz
        if wp_key not in self._wp_xyz:
            return

        # Lazy import — isaacsim isn't importable at module load in some
        # test contexts.
        from isaacsim.core.api.objects import VisualSphere
        import isaacsim.core.utils.prims as prims_utils

        right_xyz, left_xyz = self._wp_xyz[wp_key]
        env_origin = self.env.scene.env_origins[self.env_id].detach().cpu().numpy()
        color = self.viz_pick_color if kind == "pick" else self.viz_place_color
        color_arr = np.asarray(color, dtype=np.float32)

        root = f"/debug_fold/env_{self.env_id}"
        for arm_name, xyz in (("right", right_xyz), ("left", left_xyz)):
            prim_path = f"{root}/{arm_name}_target"
            # Recreate so color follows the current phase-stage.
            if prims_utils.is_prim_path_valid(prim_path):
                prims_utils.delete_prim(prim_path)
            sphere = VisualSphere(
                prim_path=prim_path,
                name=f"fold_env{self.env_id}_{arm_name}_target",
                radius=self.viz_radius,
                color=color_arr,
            )
            world_pos = np.asarray(xyz, dtype=np.float32) + env_origin
            sphere.set_world_pose(position=world_pos)

        if self.debug:
            print(
                f"[Fold][env={self.env_id}] viz phase={phase} kind={kind} "
                f"wp_key={wp_key} right={right_xyz.tolist()} left={left_xyz.tolist()}"
            )
        self._last_viz_phase = phase

    # ---------- submit helpers ----------

    # Fold phases that ride MoveL's INTERP_MODE (smooth lerp). The
    # ``reach_*`` phases stay snap (planner_mode=0) so the gripper
    # presses straight down onto the cloth without ramp-in.
    _INTERP_PHASES: set = {
        "lift_sleeve",
        "move_sleeve",
        "drop_sleeve",
        "retract_sleeve",
        "lift_bottom",
        "move_bottom",
        "drop_bottom",
        "retract_bottom",
    }

    def _submit_dual_movel(self, phase_label: str) -> dict:
        right_pose, left_pose = self._wp_pose[phase_label]
        target = self._dual_target(right_pose, left_pose)
        mode = 2 if phase_label in self._INTERP_PHASES else 0
        action = {"MoveL": ((self.robot_id, -1, mode), target)}
        if self.debug:
            right = right_pose.detach().cpu().numpy().tolist()
            left = left_pose.detach().cpu().numpy().tolist()
            print(
                f"[Fold][env={self.env_id}] submit MoveL phase={phase_label} "
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
                f"[Fold][env={self.env_id}] submit ParallelGripper phase={phase_label} "
                f"robot_id={self.robot_id} hand_id=-1 target={target.tolist()}"
            )
        return action

    def step(self):
        if self.current_state == "failed":
            self.current_action = "Failed"
            return "Failed"

        self.current_state = "running"
        phase = self.current_phase
        self._visualize_phase(phase)

        if phase in (
            "pre_reach_sleeve",
            "reach_sleeve",
            "lift_sleeve",
            "move_sleeve",
            "drop_sleeve",
            "retract_sleeve",
            "pre_reach_bottom",
            "reach_bottom",
            "lift_bottom",
            "move_bottom",
            "drop_bottom",
            "retract_bottom",
        ):
            self.current_action = self._submit_dual_movel(phase)
            return self.current_action
        if phase in ("close_gripper_sleeve", "close_gripper_bottom"):
            self.current_action = self._submit_dual_gripper(
                close=True, phase_label=phase
            )
            return self.current_action
        if phase in ("open_gripper_sleeve", "open_gripper_bottom"):
            self.current_action = self._submit_dual_gripper(
                close=False, phase_label=phase
            )
            return self.current_action

        self.current_state = "failed"
        self.current_action = None
        return None

    def _advance_phase(self) -> str | None:
        try:
            i = PHASE_ORDER.index(self.current_phase)
        except ValueError:
            return None
        if i + 1 < len(PHASE_ORDER):
            return PHASE_ORDER[i + 1]
        return None

    def update(self, info):
        if self.current_state == "failed":
            self.current_state = "failed: atomicskill failed to plan"
            return {
                "type": "Fold",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
                "phase": self.current_phase,
            }

        gp_info = info.get("global_planner_info", None)
        if gp_info is None or gp_info[self.env_id] is None:
            return {
                "type": "Fold",
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
                print(f"[Fold] env_id={self.env_id} phase=completed")
                return {
                    "type": "Fold",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": True,
                    "state": self.current_state,
                    "truncated": 0,
                    "phase": "completed",
                }
            print(f"[Fold] env_id={self.env_id} phase={next_phase}")
            self.current_phase = next_phase
            return {
                "type": "Fold",
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
                "type": "Fold",
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
                "type": "Fold",
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
                "type": "Fold",
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
                "type": "Fold",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
                "phase": self.current_phase,
            }
