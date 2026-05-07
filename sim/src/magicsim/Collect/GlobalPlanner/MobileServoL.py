"""
MobileServoL: trajectory-following global planner with mobile base.

**Input formats**

Base-embedded (direct)
    Each row already contains base + arm, optional EEF tail, optional lock column.
    Skips IK / MotionGen; streams rows directly.

    Row widths (EEF tail is optional, like ServoL):
        mdb + arm                       → lock filled with 0
        mdb + arm + matched_eef         → lock filled with 0  (only when matched > 0 and != base+arm)
        base + arm         (= mdb+arm+1)→ lock from row, layout per ``embedded_direct_lock_layout``
        base + arm + matched_eef        → lock from row + EEF tail

    Lock layout config ``embedded_direct_lock_layout``:
        "trailing" (default, matches ``g1_dehatch_strategy``):
            input  row = [mdb_pose | arm (+eef) | lock]
        "in_base":
            input  row = [mdb_pose | lock | arm (+eef)]

    Output action is always ``[mdb_pose | lock] | arm | eef`` — lock sits as the
    **last scalar of ``base_dim``**, before the arm segment.

Arm-only (no base in command; ``base_dim == 0``)
    Waypoint arm block width is ``7 * eef_num_from_hand_id(hand_id)``, optionally plus a gripper
    tail of length ``per_eef_dim * eef_num`` when ``eef_dim > 0``. If there is no EEF in the action
    space, only the arm block is allowed.

    Fixed-base IK on the last waypoint first.
    - IK success → stream input arm waypoints, base all NaN, lock = -1.
    - IK fail    → ``planner_mode`` decides:
        0 = linear base offset (no MotionGen), lock = 0
        1 = MotionGen (free base), lock = 0
       -1 = MotionGen only if EEF distance > threshold, else linear offset
"""

import traceback
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import torch
from magicsim.Collect.GlobalPlanner.GlobalPlanner import GlobalPlanner
from magicsim.Env.Utils.viz import draw_waypoints
from magicsim.Env.Planner.Utils import quat_angle_between, quat_normalize
from magicsim.Env.Planner.Services.MotionGenServer import MotionGenPlanRequest
from magicsim.Env.Planner.Services.DualMotionGenServer import DualMotionGenPlanRequest
from magicsim.Env.Planner.Services.IKServer import IKPlanRequest
from magicsim.Env.Planner.Services.DualIKServer import DualIKPlanRequest
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig
from magicsim.Env.Utils.file import Logger
from isaacsim.core.api.objects.sphere import VisualSphere


class MobileServoL(GlobalPlanner):
    """
    Global planner for mobile base + arm trajectory following.

    Command header: ``((robot_id, hand_id, planner_mode), trajectory)``.
    ``planner_mode`` (arm-only after IK fail): 0=no MG, 1=MG, -1=MG if far.
    """

    _LOCK_FLAG_LOCK_SKIP = -1.0
    _LOCK_FLAG_NAV = 0.0

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.robot_id = -1
        self.hand_id = 0
        self.robot_name: Optional[str] = None
        self.motiongen_server = None
        self.ik_server = None
        self._trajectory: Optional[torch.Tensor] = None
        self._use_locked_base = False
        self.inflight_ik_future = None
        self.inflight_ik_target: Optional[torch.Tensor] = None
        self.inflight_motiongen_future = None
        self.inflight_motiongen_target: Optional[torch.Tensor] = None
        self.plan_actions: Optional[torch.Tensor] = None
        self.plan_step = 0
        self.current_command = None
        self.current_target = None
        self.current_eef_target = None
        self._waypoint_max_steps = int(getattr(config, "waypoint_max_steps", 5))
        self._waypoint_step_counter = 0
        self._timeout_steps = int(getattr(config, "timeout_steps", 200))
        self._step_count = 0
        self._post_playback_timeout_counter = 0
        self._done_threshold = float(
            getattr(
                config,
                "done_threshold",
                getattr(config, "translation_threshold", 0.03),
            )
        )
        self._eef_pos_threshold = float(getattr(config, "eef_pos_threshold", 0.03))
        self._eef_rot_threshold = float(getattr(config, "eef_rot_threshold", 0.1))
        self._base_pos_threshold = float(getattr(config, "base_pos_threshold", 0.15))
        self._force_free_base_distance = float(
            getattr(config, "force_free_base_distance", 0.4)
        )
        self.debug = bool(getattr(config, "debug", False))
        self._direct_mode: bool = False
        self._direct_trajectory: Optional[torch.Tensor] = None
        self._direct_step: int = 0
        self._direct_waypoint_counter: int = 0
        self.debug_viz = bool(getattr(config, "debug_viz", True))
        self.debug_sphere = bool(getattr(config, "debug_sphere", False))
        self._viz_spheres: Optional[list] = None
        self.planner_mode: int = -1
        self._eef_trajectory: Optional[torch.Tensor] = None
        self._direct_base_embedded: bool = False
        self._embedded_lock_layout: str = str(
            getattr(config, "embedded_direct_lock_layout", "trailing")
        )
        self._stream_arm_after_ik_success: bool = False
        self._arm_stream_step: int = 0
        self._arm_stream_waypoint_counter: int = 0
        # Raw input tensor from the last ``reset``; used by ``refresh`` to
        # identity-short-circuit when the atomic skill re-sends the *same*
        # trajectory object every tick (OpenDrawer pull, Dehatch standup ...).
        # Without this the arm-stream path would ``self.reset`` every tick,
        # clobbering the inflight IK future and the stream step counters.
        self._last_raw_input: Any = None

    # ------------------------------------------------------------------ #
    # Environment / robot helpers
    # ------------------------------------------------------------------ #

    def _get_env_origin(self) -> torch.Tensor:
        return self.env.scene.env_origins[self.env_id].clone()

    def _get_planner_manager(self):
        pm = getattr(self.env.scene, "planner_manager", None)
        if pm is None:
            raise RuntimeError("PlannerManager is not available in the environment.")
        return pm

    def _get_robot_name_list(self) -> List[str]:
        return list(self.env.scene.robot_manager.robots.keys())

    def _set_robot_by_id(self, robot_id: int, hand_id: int = 0) -> bool:
        robot_id = int(robot_id)
        hand_id = int(hand_id)
        if (
            robot_id == self.robot_id
            and hand_id == self.hand_id
            and self.robot_name is not None
        ):
            return False
        name_list = self._get_robot_name_list()
        if robot_id < 0 or robot_id >= len(name_list):
            raise ValueError(
                f"robot_id {robot_id} out of range for robot_name_list={name_list}"
            )
        self.robot_id = robot_id
        self.hand_id = hand_id
        self.robot_name = name_list[robot_id]
        pm = self._get_planner_manager()
        motiongen_dict = getattr(pm, "motiongen_server", None)
        ik_dict = getattr(pm, "ik_server", None)
        if not motiongen_dict or self.robot_name not in motiongen_dict:
            raise RuntimeError(
                f"MotionGenServer not found for robot '{self.robot_name}'."
            )
        if not ik_dict or self.robot_name not in ik_dict:
            raise RuntimeError(f"IKServer not found for robot '{self.robot_name}'.")
        # Single server per robot (MERGE_LEFT_RIGHT §1–§8). ``hand_id`` no
        # longer selects a server; it only drives target packing.
        self.motiongen_server = motiongen_dict[self.robot_name]
        self.ik_server = ik_dict[self.robot_name]
        return True

    def _get_action_dims(self) -> Tuple[int, int, int]:
        if self.robot_name is None:
            raise RuntimeError("Robot name not set. Call _set_robot_by_id first.")
        pm = self._get_planner_manager()
        info = pm.get_info()
        robot_info = info.get(self.robot_name, {})
        base_dim = int(robot_info.get("base", {}).get("action_dim", 0))
        arm_dim = int(robot_info.get("arm", {}).get("action_dim", 0))
        eef_dim = int(robot_info.get("eef", {}).get("action_dim", 0))
        if base_dim == 0 and arm_dim == 0 and eef_dim == 0:
            raise RuntimeError(f"No action dims found for robot {self.robot_name}")
        return base_dim, arm_dim, eef_dim

    def _get_robot_state(self) -> dict:
        robot_states = self.env.scene.robot_manager.get_robot_state(noise_flag=False)[0]
        if isinstance(robot_states, dict):
            if self.robot_name and self.robot_name in robot_states:
                return robot_states[self.robot_name]
            return next(iter(robot_states.values()))
        return robot_states

    def _get_current_base_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self._get_robot_state()
        base_pos = state["base_pos"]
        base_quat = state["base_quat"]
        if base_pos.ndim == 2:
            base_pos = base_pos[self.env_id]
            base_quat = base_quat[self.env_id]
        return base_pos, base_quat

    def _get_current_eef(self) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self._get_robot_state()
        eef_pos = state["eef_pos"]
        eef_quat = state["eef_quat"]
        if eef_pos.ndim == 2:
            eef_pos = eef_pos[self.env_id]
            eef_quat = eef_quat[self.env_id]
        return eef_pos, eef_quat

    def _select_current_eef_for_target(
        self, target_pose_all: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if target_pose_all.ndim == 1:
            target_pose_all = target_pose_all.view(-1, 7)
        target_eef_num = target_pose_all.shape[0]
        cur_pos, cur_quat = self._get_current_eef()
        if cur_pos.ndim == 1:
            cur_pos = cur_pos.unsqueeze(0)
            cur_quat = cur_quat.unsqueeze(0)
        if target_eef_num == 1 and cur_pos.shape[0] > 1:
            eef_idx = self.hand_id if 0 <= self.hand_id < cur_pos.shape[0] else 0
            cur_pos = cur_pos[eef_idx : eef_idx + 1]
            cur_quat = cur_quat[eef_idx : eef_idx + 1]
        else:
            cur_pos = cur_pos[:target_eef_num]
            cur_quat = cur_quat[:target_eef_num]
        return cur_pos, cur_quat

    # ------------------------------------------------------------------ #
    # Action expand helpers
    # ------------------------------------------------------------------ #

    def _expand_eef_action(self, eef_action: torch.Tensor) -> torch.Tensor:
        _, _, eef_dim = self._get_action_dims()
        eef_action = eef_action.view(-1)
        if eef_action.shape[0] == eef_dim:
            return eef_action
        per_gripper = eef_action.shape[0]
        full_eef = torch.full(
            (eef_dim,), torch.nan, device=eef_action.device, dtype=eef_action.dtype
        )
        if self.hand_id == 0:
            full_eef[:per_gripper] = eef_action
        elif self.hand_id == 1:
            full_eef[eef_dim - per_gripper :] = eef_action
        return full_eef

    def _expand_arm_action(self, arm_action: torch.Tensor) -> torch.Tensor:
        _, arm_dim, _ = self._get_action_dims()
        arm_action = arm_action.view(-1)
        if arm_action.shape[0] == arm_dim:
            return arm_action
        full_arm = torch.full(
            (arm_dim,), torch.nan, device=arm_action.device, dtype=arm_action.dtype
        )
        if self.hand_id == 0:
            full_arm[: arm_action.shape[0]] = arm_action
        elif self.hand_id == 1:
            full_arm[arm_dim - arm_action.shape[0] :] = arm_action
        else:
            full_arm[: arm_action.shape[0]] = arm_action
        return full_arm

    # ``_format_target_for_submit`` removed — submit sites use
    # ``_expand_arm_action(target.view(-1)).view(1, -1)`` for hand_id NaN.

    # ------------------------------------------------------------------ #
    # Arrival checking
    # ------------------------------------------------------------------ #

    def _check_base_waypoint_arrived(self, target_pos3: torch.Tensor) -> bool:
        if torch.isnan(target_pos3).any():
            return True
        base_pos, _ = self._get_current_base_pose()
        diff = torch.linalg.norm(base_pos - target_pos3).item()
        return diff < self._base_pos_threshold

    def _check_eef_waypoint_arrived(self, waypoint_row: torch.Tensor) -> bool:
        """EEF arrival check for arm-only streaming (like ServoL)."""
        w = waypoint_row.view(-1)
        seg = w[:14] if w.shape[0] >= 14 else w[:7]
        tp = seg.view(-1, 7)
        if tp.shape[0] == 0:
            return False
        cur_pos, cur_quat = self._select_current_eef_for_target(tp)
        for i in range(tp.shape[0]):
            eef_pos = cur_pos[i] if cur_pos.ndim > 1 else cur_pos
            eef_quat = cur_quat[i] if cur_quat.ndim > 1 else cur_quat
            pos_diff = torch.linalg.norm(eef_pos - tp[i, :3]).item()
            rot_diff = quat_angle_between(
                eef_quat.unsqueeze(0), tp[i, 3:7].unsqueeze(0)
            ).item()
            if (
                pos_diff >= self._eef_pos_threshold
                or rot_diff >= self._eef_rot_threshold
            ):
                return False
        return True

    def _check_direct_row_eef_arrived(self, row: torch.Tensor) -> bool:
        """EEF arrival for direct-mode rows (embedded or arm-only linear), aligned with
        ``MobileMoveL._check_waypoint_arrived`` locked-base branch."""
        wp = row.view(-1)
        arm_flat: torch.Tensor
        if self._direct_base_embedded:
            _, _, arm_flat, _ = self._parse_embedded_row(row)
        else:
            mdb, _, arm_dim, _ = self._embedded_dims()
            if wp.shape[0] < mdb:
                return False
            arm_flat = wp[mdb : mdb + arm_dim]
        arm_data = arm_flat.view(-1)
        if arm_data.numel() < 7:
            return False
        if arm_data.shape[0] >= 7 and arm_data.shape[0] % 7 == 0:
            eef_targets = arm_data.view(-1, 7)
        else:
            eef_targets = arm_data[:7].unsqueeze(0)
        cur_pos, cur_quat = self._select_current_eef_for_target(eef_targets)
        pair_count = min(cur_pos.shape[0], eef_targets.shape[0])
        for eef_idx in range(pair_count):
            if torch.isnan(eef_targets[eef_idx, :3]).any():
                continue
            eef_pos_diff = torch.linalg.norm(
                cur_pos[eef_idx] - eef_targets[eef_idx, :3]
            ).item()
            eef_rot_diff = quat_angle_between(
                cur_quat[eef_idx].unsqueeze(0),
                eef_targets[eef_idx, 3:].unsqueeze(0),
            ).item()
            if (
                eef_pos_diff >= self._eef_pos_threshold
                or eef_rot_diff >= self._eef_rot_threshold
            ):
                return False
        return True

    def _motiongen_plan_row_has_base(self, wp: torch.Tensor) -> bool:
        """Whether ``wp`` is a MotionGen row ``[mdb_pose | arm]`` (mobile base present)."""
        base_dim, arm_dim, _ = self._get_action_dims()
        motiongen_base_dim = base_dim - 1 if base_dim > 0 else 0
        w = wp.view(-1)
        return motiongen_base_dim > 0 and w.shape[0] == motiongen_base_dim + arm_dim

    def _done_last_base_pos_arrived(self, target_pos3: torch.Tensor) -> bool:
        """Last waypoint: base position within ``_done_threshold``."""
        if torch.isnan(target_pos3).any():
            return True
        base_pos, _ = self._get_current_base_pose()
        return torch.linalg.norm(base_pos - target_pos3).item() < self._done_threshold

    def _done_last_eef_poses_arrived(self, target_poses: torch.Tensor) -> bool:
        """Last waypoint: every active EEF position within ``_done_threshold``."""
        if target_poses.ndim == 1:
            target_poses = target_poses.view(-1, 7)
        cur_pos, _ = self._select_current_eef_for_target(target_poses)
        pair_count = min(cur_pos.shape[0], target_poses.shape[0])
        thr = self._done_threshold
        for eef_idx in range(pair_count):
            if torch.isnan(target_poses[eef_idx, :3]).any():
                continue
            if (
                torch.linalg.norm(cur_pos[eef_idx] - target_poses[eef_idx, :3]).item()
                >= thr
            ):
                return False
        return True

    def _arm_waypoint_row_to_target_poses(
        self, waypoint_row: torch.Tensor
    ) -> torch.Tensor:
        """Arm-only trajectory row → ``[eef_num, 7]`` (matches stream / IK target layout)."""
        w = waypoint_row.view(-1)
        if w.shape[0] >= 14:
            return w[:14].view(2, 7)
        return w[:7].view(1, 7)

    def _direct_linear_row_to_eef_target_poses(self, row: torch.Tensor) -> torch.Tensor:
        """IK-fail linear direct row ``[mdb | arm]`` → ``[N, 7]`` for ``get_done``."""
        wp = row.view(-1)
        mdb, _, arm_dim, _ = self._embedded_dims()
        if wp.shape[0] < mdb + 7:
            return torch.empty(0, 7, device=wp.device, dtype=wp.dtype)
        arm_flat = wp[mdb : mdb + arm_dim]
        arm_data = arm_flat.view(-1)
        if arm_data.shape[0] >= 7 and arm_data.shape[0] % 7 == 0:
            return arm_data.view(-1, 7)
        return arm_data[:7].unsqueeze(0)

    def _motiongen_row_arm_target_poses_for_done(
        self, wp: torch.Tensor
    ) -> torch.Tensor:
        """MotionGen plan row → arm pose targets ``[N, 7]`` (strips mdb when present)."""
        w = wp.view(-1)
        base_dim, arm_dim, _ = self._get_action_dims()
        motiongen_base_dim = base_dim - 1 if base_dim > 0 else 0
        arm_data = w[motiongen_base_dim:] if self._motiongen_plan_row_has_base(w) else w
        if arm_data.numel() < 7:
            return torch.empty(0, 7, device=w.device, dtype=w.dtype)
        if arm_data.shape[0] >= 7 and arm_data.shape[0] % 7 == 0:
            return arm_data.view(-1, 7)
        return arm_data[:7].unsqueeze(0)

    def _check_motiongen_plan_row_eef_arrived(self, waypoint: torch.Tensor) -> bool:
        """EEF-first arrival for one MotionGen plan row (``mdb|arm`` pose targets)."""
        base_dim, arm_dim, _ = self._get_action_dims()
        motiongen_base_dim = base_dim - 1 if base_dim > 0 else 0
        wp = waypoint.view(-1)
        has_base_data = (
            motiongen_base_dim > 0 and wp.shape[0] == motiongen_base_dim + arm_dim
        )
        eef_offset = motiongen_base_dim if has_base_data else 0
        arm_data = wp[eef_offset:]
        if arm_data.numel() < 7:
            return False
        if arm_data.shape[0] >= 7 and arm_data.shape[0] % 7 == 0:
            eef_targets = arm_data.view(-1, 7)
        else:
            eef_targets = arm_data[:7].unsqueeze(0)
        cur_pos, cur_quat = self._select_current_eef_for_target(eef_targets)
        pair_count = min(cur_pos.shape[0], eef_targets.shape[0])
        for eef_idx in range(pair_count):
            if torch.isnan(eef_targets[eef_idx, :3]).any():
                continue
            eef_pos_diff = torch.linalg.norm(
                cur_pos[eef_idx] - eef_targets[eef_idx, :3]
            ).item()
            eef_rot_diff = quat_angle_between(
                cur_quat[eef_idx].unsqueeze(0),
                eef_targets[eef_idx, 3:].unsqueeze(0),
            ).item()
            if (
                eef_pos_diff >= self._eef_pos_threshold
                or eef_rot_diff >= self._eef_rot_threshold
            ):
                return False
        return True

    def _is_playback_fully_streamed(self) -> bool:
        """True when the active trajectory/plan row index has reached the end (playback done)."""
        if self._direct_mode and self._direct_trajectory is not None:
            tdir = self._direct_trajectory.shape[0]
            return tdir > 0 and self._direct_step >= tdir
        if self._stream_arm_after_ik_success and self._trajectory is not None:
            ta = self._trajectory.shape[0]
            return ta > 0 and self._arm_stream_step >= ta
        if (
            not self._direct_mode
            and not self._stream_arm_after_ik_success
            and self.plan_actions is not None
            and self.plan_actions.shape[0] > 0
        ):
            tp = self.plan_actions.shape[0]
            return self.plan_step >= tp
        return False

    def _should_force_free_base(self, target_pose: torch.Tensor) -> bool:
        if target_pose.ndim == 1:
            target_pose = target_pose.view(-1, 7)
        cur_pos, _ = self._select_current_eef_for_target(target_pose)
        pair_count = min(cur_pos.shape[0], target_pose.shape[0])
        max_diff = 0.0
        for eef_idx in range(pair_count):
            pos_diff = torch.linalg.norm(
                cur_pos[eef_idx] - target_pose[eef_idx, :3]
            ).item()
            max_diff = max(max_diff, pos_diff)
        return max_diff > self._force_free_base_distance

    def _should_use_motiongen_servo(self, target_pose: torch.Tensor) -> bool:
        if self.planner_mode == 0:
            return False
        if self.planner_mode == 1:
            return True
        if target_pose.ndim == 1:
            target_pose = target_pose.view(-1, 7)
        cur_pos, _ = self._select_current_eef_for_target(target_pose)
        pair_count = min(cur_pos.shape[0], target_pose.shape[0])
        threshold = float(
            getattr(
                self.config,
                "motiongen_distance_threshold",
                getattr(self.config, "translation_threshold", 0.05),
            )
        )
        max_diff = 0.0
        for eef_idx in range(pair_count):
            pos_diff = torch.linalg.norm(
                cur_pos[eef_idx] - target_pose[eef_idx, :3]
            ).item()
            max_diff = max(max_diff, pos_diff)
        return max_diff > threshold

    # ------------------------------------------------------------------ #
    # Embedded direct row: dimensions, parsing, building
    # ------------------------------------------------------------------ #

    def _eef_tail_width_for_hand(self) -> int:
        """Gripper segment width in a waypoint row when ``eef_dim > 0``.

        Uses ``GlobalPlanner._eef_layout_from_robot()`` → ``(max_eef_num, per_eef_dim)`` from
        ``RobotManager.get_info()`` — **same** ``per_eef_dim`` source as ``MobileMoveL._parse_target``
        (passed into ``GlobalPlanner.parse_target_vector``). Active tail length is
        ``per_eef_dim * eef_num_from_hand_id(hand_id)``.
        """
        _, _, eef_dim = self._get_action_dims()
        if eef_dim <= 0:
            return 0
        _, per_eef_dim = self._eef_layout_from_robot()
        n_eef = GlobalPlanner.eef_num_from_hand_id(self.hand_id)
        return int(per_eef_dim * n_eef)

    def _embedded_dims(self) -> Tuple[int, int, int, int]:
        """Return ``(mdb, base_dim, arm_dim, matched_eef)`` for base-embedded rows (``base_dim > 0``)."""
        base_dim, arm_dim, eef_dim = self._get_action_dims()
        mdb = base_dim - 1 if base_dim > 0 else 0
        _, per_eef_dim = self._eef_layout_from_robot()
        active = GlobalPlanner.eef_num_from_hand_id(self.hand_id)
        matched = per_eef_dim * active if eef_dim > 0 else 0
        arm_dim = active * 7
        return mdb, base_dim, arm_dim, matched

    def _embedded_valid_widths(self) -> Set[int]:
        """All valid row widths for base-embedded trajectories."""
        mdb, base_dim, arm_dim, matched = self._embedded_dims()
        widths: Set[int] = set()
        if mdb > 0:
            widths.add(mdb + arm_dim)
            widths.add(base_dim + arm_dim)
            if matched > 0:
                w_mg_eef = mdb + arm_dim + matched
                w_bd_eef = base_dim + arm_dim + matched
                widths.add(w_mg_eef)
                widths.add(w_bd_eef)
        return widths

    def _parse_embedded_row(
        self, row: torch.Tensor
    ) -> Tuple[torch.Tensor, float, torch.Tensor, Optional[torch.Tensor]]:
        """Parse one embedded row → ``(base_pose, lock_flag, arm, eef_or_none)``.

        Lock handling:
        - Row width ``mdb + arm [+ eef]``  → lock = **0.0**
        - Row width ``base + arm [+ eef]`` → lock extracted per ``_embedded_lock_layout``

        Output order is always ``[mdb_pose] [lock] [arm] [eef?]``, but here we return
        them as separate tensors; ``_build_action_from_embedded`` assembles the final
        ``base|arm|eef`` vector.
        """
        mdb, base_dim, arm_dim, matched = self._embedded_dims()
        row = row.view(-1)
        n = row.shape[0]
        dev, dtype = row.device, row.dtype

        if mdb <= 0:
            return torch.tensor([], device=dev, dtype=dtype), 0.0, row, None

        has_lock = (n == base_dim + arm_dim) or (
            matched > 0 and n == base_dim + arm_dim + matched
        )
        has_eef_no_lock = (
            matched > 0 and n == mdb + arm_dim + matched and n != base_dim + arm_dim
        )

        if has_lock:
            if self._embedded_lock_layout == "in_base":
                base_pose = row[:mdb]
                lock = float(row[mdb].item())
                arm_flat = row[base_dim : base_dim + arm_dim]
                eef = (
                    row[base_dim + arm_dim :]
                    if matched > 0 and n > base_dim + arm_dim
                    else None
                )
            else:
                base_pose = row[:mdb]
                lock = float(row[-1].item())
                body = row[mdb:-1]
                arm_flat = body[:arm_dim]
                eef = body[arm_dim:] if body.shape[0] > arm_dim else None
            if eef is not None and eef.numel() == 0:
                eef = None
            return base_pose, lock, arm_flat, eef

        if has_eef_no_lock:
            base_pose = row[:mdb]
            arm_flat = row[mdb : mdb + arm_dim]
            eef = row[mdb + arm_dim :]
            return base_pose, 0.0, arm_flat, eef

        if n == mdb + arm_dim:
            base_pose = row[:mdb]
            arm_flat = row[mdb : mdb + arm_dim]
            return base_pose, 0.0, arm_flat, None

        raise ValueError(
            f"Embedded row length {n} not valid "
            f"(mdb={mdb}, base={base_dim}, arm={arm_dim}, matched_eef={matched})."
        )

    def _build_action_from_embedded(
        self,
        row: torch.Tensor,
        lock_override: Optional[float] = None,
        eef_external: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build ``[base_dim | arm_dim | eef_dim]`` from one embedded row.

        ``base_action = [mdb_pose | lock]``: lock comes from the row (see
        ``_parse_embedded_row``) or ``lock_override``.
        EEF prefers ``eef_external`` > in-row eef > NaN fill.
        """
        base_dim, arm_dim, eef_dim = self._get_action_dims()
        dev, dtype = row.device, row.dtype

        base_pose, lock_parsed, arm_flat, eef_from_row = self._parse_embedded_row(row)
        lock = lock_parsed if lock_override is None else float(lock_override)
        arm_action = self._expand_arm_action(arm_flat)

        if base_dim > 0:
            base_action = torch.cat(
                [base_pose, torch.tensor([lock], device=dev, dtype=dtype)]
            )
        else:
            base_action = torch.tensor([], device=dev, dtype=dtype)

        eef_src = eef_external if eef_external is not None else eef_from_row
        if eef_dim > 0:
            if eef_src is not None:
                eef_action = self._expand_eef_action(eef_src.view(-1))
            else:
                eef_action = torch.full((eef_dim,), torch.nan, device=dev, dtype=dtype)
        else:
            eef_action = torch.tensor([], device=dev, dtype=dtype)

        return torch.cat([base_action, arm_action, eef_action])

    # ------------------------------------------------------------------ #
    # Arm-only action building (delegates to GlobalPlanner._build_full_action_mobile)
    # ------------------------------------------------------------------ #

    def _build_full_action(
        self,
        arm_action_flat: torch.Tensor,
        lock_flag_override: Optional[float] = None,
    ) -> torch.Tensor:
        return self._build_full_action_mobile(
            arm_action_flat, lock_flag_override=lock_flag_override
        )

    # ------------------------------------------------------------------ #
    # Trajectory parsing
    # ------------------------------------------------------------------ #

    def _parse_servo_action(self, action):
        return GlobalPlanner.parse_planner_header(
            action,
            default_robot_id=self.robot_id if self.robot_id >= 0 else 0,
            default_hand_id=self.hand_id,
        )

    def _arm_only_valid_waypoint_widths(self) -> Set[int]:
        """Valid last-dim sizes for arm-only trajectories (``base_dim == 0``): ``7*eef_num`` [+ EEF tail].

        ``eef_num`` is ``GlobalPlanner.eef_num_from_hand_id(hand_id)``. Optional tail is
        ``_eef_tail_width_for_hand()`` (``per_eef_dim * eef_num`` when ``eef_dim > 0``), same as
        ``parse_target_vector`` / ``MobileMoveL``.
        """
        eef_num = GlobalPlanner.eef_num_from_hand_id(self.hand_id)
        arm_w = 7 * eef_num
        tail = self._eef_tail_width_for_hand()
        widths: Set[int] = {arm_w}
        if tail > 0:
            widths.add(arm_w + tail)
        return widths

    def _assert_arm_trajectory_shape_ok(self, action: torch.Tensor) -> None:
        base_dim, _, _ = self._get_action_dims()
        if base_dim == 0:
            valid = self._arm_only_valid_waypoint_widths()
            if action.ndim == 1:
                n = int(action.numel())
                if n == 0:
                    return
                if any(w > 0 and n % w == 0 for w in valid):
                    return
                raise ValueError(
                    f"MobileServoL arm-only trajectory length {n} is not divisible by any of "
                    f"the valid waypoint widths {sorted(valid)} "
                    f"(7×eef_num or 7×eef_num+EEF tail from hand_id)."
                )
            if action.ndim == 2:
                w = action.shape[1]
                if w == 0 or w in valid:
                    return
                raise ValueError(
                    f"MobileServoL arm-only trajectory last dim {w} is invalid: "
                    f"expected one of {sorted(valid)}."
                )
            return

        # Mobile base present: keep legacy heuristics for mixed waypoint layouts.
        if action.ndim == 1:
            n = int(action.numel())
            if n == 0:
                return
            if any(n % w == 0 for w in (19, 8, 7)):
                return
            raise ValueError(
                f"MobileServoL arm trajectory length {n} cannot be split into "
                f"N×7, N×8, or N×19 waypoints."
            )
        if action.ndim == 2:
            w = action.shape[1]
            if w == 0 or w in (7, 8, 19) or w % 7 == 0:
                return
            raise ValueError(
                f"MobileServoL trajectory last dim {w} is invalid: expected 7, 8, 19, "
                f"or a multiple of 7."
            )

    def _parse_servo_arm_trajectory(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Parse ``(arm trajectory rows, optional per-row EEF tail)``; widths from ``hand_id`` + layout."""
        action = (
            torch.as_tensor(action, device=self.env.device, dtype=torch.float32)
            if not isinstance(action, torch.Tensor)
            else action.to(self.env.device, dtype=torch.float32)
        )
        if action.ndim not in (1, 2):
            raise ValueError(
                f"MobileServoL arm trajectory must be 1-D or 2-D, got shape {tuple(action.shape)}"
            )

        _, _, eef_dim = self._get_action_dims()
        max_eef_num, per_eef_dim = self._eef_layout_from_robot()
        if eef_dim > 0 and per_eef_dim * max_eef_num != eef_dim:
            raise ValueError(
                f"eef_dim {eef_dim} inconsistent with layout "
                f"max_eef_num={max_eef_num}, per_eef_dim={per_eef_dim}."
            )
        n_eef = GlobalPlanner.eef_num_from_hand_id(self.hand_id)
        if n_eef > max_eef_num:
            raise ValueError(
                f"hand_id={self.hand_id} implies {n_eef} active EEFs but "
                f"max_eef_num is {max_eef_num}."
            )
        arm_w = 7 * n_eef
        tail = self._eef_tail_width_for_hand()
        full_w = arm_w + tail

        if action.ndim == 1:
            L = int(action.shape[0])
            if tail > 0:
                if L % full_w == 0:
                    wp = full_w
                elif L % arm_w == 0:
                    wp = arm_w
                else:
                    raise ValueError(
                        f"MobileServoL flat trajectory length {L} must divide {full_w} "
                        f"(arm+EEF) or {arm_w} (arm only); hand_id={self.hand_id}, tail={tail}."
                    )
            elif L % arm_w != 0:
                raise ValueError(
                    f"MobileServoL flat trajectory length {L} must divide {arm_w} "
                    f"(hand_id={self.hand_id})."
                )
            else:
                wp = arm_w
            action = action.view(-1, wp)

        w = int(action.shape[1])
        eef_target = None
        if tail > 0:
            if w == full_w:
                traj = action[:, :arm_w].clone()
                eef_target = action[:, arm_w:].clone()
            elif w == arm_w:
                traj = action.clone()
            else:
                raise ValueError(
                    f"MobileServoL waypoint width {w} must be {arm_w} or {full_w} "
                    f"(hand_id={self.hand_id})."
                )
        elif w != arm_w:
            raise ValueError(
                f"MobileServoL waypoint width {w} must be {arm_w} (hand_id={self.hand_id})."
            )
        else:
            traj = action.clone()

        b, col = traj.shape
        n_poses = col // 7
        t3 = traj.view(b, n_poses, 7)
        traj = torch.cat(
            [
                t3[..., :3],
                quat_normalize(t3[..., 3:].reshape(-1, 4)).view(b, n_poses, 4),
            ],
            dim=-1,
        ).reshape(b, -1)
        return traj, eef_target

    # ------------------------------------------------------------------ #
    # Base-embedded detection and 1-D reshape
    # ------------------------------------------------------------------ #

    def _is_base_embedded_trajectory(self, traj_raw) -> bool:
        valid = self._embedded_valid_widths()
        if isinstance(traj_raw, torch.Tensor):
            if traj_raw.ndim == 2:
                return traj_raw.shape[1] in valid
            if traj_raw.ndim == 1:
                return any(w > 0 and traj_raw.shape[0] % w == 0 for w in valid)
        return False

    def _reshape_flat_embedded(self, flat: torch.Tensor) -> torch.Tensor:
        """Reshape 1-D tensor to ``[N, w]`` using the widest matching stride."""
        for w in sorted(self._embedded_valid_widths(), reverse=True):
            if w > 0 and flat.shape[0] % w == 0:
                return flat.view(-1, w)
        mdb, _, arm_dim, _ = self._embedded_dims()
        return flat.view(-1, mdb + arm_dim)

    # ------------------------------------------------------------------ #
    # Arm-only linear base synthesis
    # ------------------------------------------------------------------ #

    def _motiongen_base_vector_from_state(self) -> torch.Tensor:
        base_dim, _, _ = self._get_action_dims()
        mdb = base_dim - 1 if base_dim > 0 else 0
        if mdb <= 0:
            return torch.tensor([], device=self.env.device, dtype=torch.float32)
        cur_pos, cur_quat = self._get_current_base_pose()
        if cur_pos.ndim > 1:
            cur_pos = cur_pos[self.env_id]
            cur_quat = cur_quat[self.env_id]
        pelvis7 = torch.cat([cur_pos.view(-1)[:3], cur_quat.view(-1)[:4]])
        if mdb <= 7:
            return pelvis7[:mdb].clone()
        rest = torch.full(
            (mdb - 7,), torch.nan, device=pelvis7.device, dtype=pelvis7.dtype
        )
        return torch.cat([pelvis7[:7], rest])

    def _eef_delta_pos_from_trajectory(self, traj: torch.Tensor) -> torch.Tensor:
        """EEF translation delta from **current** EEF position to the last
        trajectory waypoint, averaged across active EEFs.

        Used by the linear-base synth so the base translates by exactly the
        residual that EEF still needs to cover, not by the trajectory's
        first→last span. This makes the base end up aligned with the *real*
        post-grasp arm pose regardless of any drift between ``traj[0]`` and
        ``current_eef`` introduced by grasp_handle timeout / residual.
        """
        last_row = traj[-1].view(-1)
        n = last_row.shape[0] // 7
        if n < 1:
            return torch.zeros(3, device=traj.device, dtype=traj.dtype)
        last_poses = last_row.view(n, 7)
        last_pos = last_poses[:, :3]
        cur_pos, _ = self._select_current_eef_for_target(last_pos)
        cur_pos = cur_pos.to(device=traj.device, dtype=traj.dtype)
        if cur_pos.shape[0] != last_pos.shape[0]:
            # Defensive: pad/truncate by repeating first row to match.
            cur_pos = cur_pos[:1].expand(last_pos.shape[0], -1)
        return (last_pos - cur_pos).mean(dim=0)

    def _build_arm_only_linear_direct_trajectory(
        self, traj: torch.Tensor
    ) -> torch.Tensor:
        """``[N, mdb + arm_dim]``: linearly interpolated base + input arm waypoints.

        **Downsampling** (``linear_synth_max_waypoints``, default 50): if the
        input arm trajectory has more rows than this, subsample by stride.
        Streaming this synth trajectory advances one row per tick (no arrive
        gate), so smaller N → larger per-tick base target deltas → higher
        target velocity. This matters because the PController's
        ``linear_dead_zone`` is a *velocity* output cutoff (not a position
        threshold): when PD output drops below dead_zone (kp · lag <
        dead_zone) it gets clamped to zero. With 200 wp at 60 Hz spread over
        30 cm, target moves only 9 cm/s and the PD output dips into dead
        zone whenever lag < 1.25 cm, producing visible stop-go motion. 50 wp
        gives 36 cm/s target velocity and ~22 cm lag — well clear of the
        dead-zone clamp, so base tracks smoothly.

        **Base target lookahead** (``linear_synth_lookahead``, default 0.10 m):
        each row's base target is pushed ``lookahead`` metres ahead of the
        arm-waypoint position along the EEF translation direction.

        **NaN quat for base**: the synthesized base pose carries NaN in
        ``[3:7]`` so the downstream PController helper interprets it as
        "preserve current heading" — sidesteps the panda_mobile→base_link
        frame conversion that would otherwise re-rotate a base_link-frame
        quat.
        """
        mdb, _, arm_dim, _ = self._embedded_dims()
        N_in = traj.shape[0]
        if N_in == 0:
            return torch.empty(0, mdb + arm_dim, device=traj.device)

        # Downsample input arm trajectory if too long. Always include the
        # first and last waypoints so endpoints are exact.
        max_wp = int(getattr(self.config, "linear_synth_max_waypoints", 50))
        if max_wp > 0 and N_in > max_wp:
            idxs = (
                torch.linspace(0, N_in - 1, steps=max_wp, device=traj.device)
                .round()
                .long()
            )
            traj_use = traj[idxs]
        else:
            traj_use = traj
        N = traj_use.shape[0]

        base_start = self._motiongen_base_vector_from_state()
        delta = self._eef_delta_pos_from_trajectory(traj_use)
        end_base = base_start.clone()
        end_base[:3] = base_start[:3] + delta

        # Lookahead fraction so base target leads arm progress by
        # ``lookahead`` metres along ``delta`` direction.
        lookahead = float(getattr(self.config, "linear_synth_lookahead", 0.10))
        delta_mag = float(torch.linalg.norm(delta).item())
        if delta_mag > 1e-6:
            lookahead_frac = lookahead / delta_mag
        else:
            lookahead_frac = 0.0

        # Mark the orientation portion of the synthesized base pose as NaN so
        # the PController helper doesn't apply its panda_mobile→base_link
        # offset to a value that's already in base_link frame.
        if base_start.shape[0] > 3:
            base_start = base_start.clone()
            base_start[3:] = float("nan")
            end_base[3:] = float("nan")
        rows = []
        for i in range(N):
            alpha_arm = float(i) / float(max(N - 1, 1))
            alpha_base = min(1.0, alpha_arm + lookahead_frac)
            b = base_start.clone()
            b[:3] = (1.0 - alpha_base) * base_start[:3] + alpha_base * end_base[:3]
            arm_flat = self._expand_arm_action(traj_use[i].view(-1))
            rows.append(torch.cat([b, arm_flat]))
        return torch.stack(rows)

    # ------------------------------------------------------------------ #
    # Stream alignment (MotionGen plan + eef trajectory)
    # ------------------------------------------------------------------ #

    def _pad_last_row_to_length(
        self, t: torch.Tensor, target_rows: int
    ) -> torch.Tensor:
        if t.shape[0] >= target_rows or t.shape[0] == 0:
            return t
        pad = t[-1:].expand(target_rows - t.shape[0], -1).clone()
        return torch.cat([t, pad])

    def _align_plan_streams(self) -> None:
        if self.plan_actions is None or self.plan_actions.numel() == 0:
            return
        T = self.plan_actions.shape[0]
        Nt = (
            self._trajectory.shape[0]
            if self._trajectory is not None and self._trajectory.ndim == 2
            else 0
        )
        Ne = self._eef_trajectory.shape[0] if self._eef_trajectory is not None else 0
        L = max(T, Nt, Ne)
        if L == 0:
            return
        if T < L:
            self.plan_actions = self._pad_last_row_to_length(self.plan_actions, L)
        if self._eef_trajectory is not None and Ne < L:
            self._eef_trajectory = self._pad_last_row_to_length(self._eef_trajectory, L)

    # ------------------------------------------------------------------ #
    # IK / MotionGen submission (aligned with MobileMoveL)
    # ------------------------------------------------------------------ #

    def _submit_ik(self, target_pose: torch.Tensor) -> None:
        """Submit IK with fixed base (``lock_base=True`` when dual). Same contract as ``MobileMoveL._submit_ik``."""
        robot_state = self._get_robot_state()
        robot_states_dict = {
            "base_pos": robot_state["base_pos"],
            "base_quat": robot_state["base_quat"],
            "joint_pos": robot_state["joint_pos"],
            "joint_vel": robot_state["joint_vel"],
        }
        # IKPlanRequest single-mode 2-D ``(1, L*7)`` via ``_expand_arm_action``.
        target_flat = self._expand_arm_action(target_pose.view(-1)).view(1, -1)

        is_dual_mode = getattr(self.ik_server, "dual_mode", False)
        if is_dual_mode:
            req = DualIKPlanRequest(
                env_ids=[self.env_id],
                target_pos=target_flat,
                robot_states=robot_states_dict,
                mode="single",
                lock_base=True,
            )
        else:
            req = IKPlanRequest(
                env_ids=[self.env_id],
                target_pos=target_flat,
                robot_states=robot_states_dict,
                mode="single",
            )
        self.inflight_ik_future = self.ik_server.submit_ik(req)
        self.inflight_ik_target = target_pose.clone()
        if self.debug:
            lock_str = " lock_base=True (fixed base check)" if is_dual_mode else ""
            print(
                "Submit IK",
                "env_id=",
                self.env_id,
                "robot=",
                self.robot_name,
                "target=",
                self.inflight_ik_target,
                lock_str,
            )
        self.inflight_motiongen_future = None
        self.inflight_motiongen_target = None
        self.plan_actions = None
        self.plan_step = 0

    def _submit_plan(
        self,
        target_pose_all: torch.Tensor,
        lock_base: Optional[bool] = None,
    ) -> None:
        """Submit a MotionGen request. Same layout as ``MobileMoveL._submit_plan`` (no reactive path)."""
        robot_state = self._get_robot_state()
        robot_states_dict = {
            "base_pos": robot_state["base_pos"],
            "base_quat": robot_state["base_quat"],
            "joint_pos": robot_state["joint_pos"],
            "joint_vel": robot_state["joint_vel"],
        }
        # Pack into the unified (Dual)MotionGenPlanRequest 2-D shape via
        # ``_expand_arm_action`` (single source of truth for hand_id NaN).
        target_flat = self._expand_arm_action(target_pose_all.view(-1)).view(1, -1)

        use_locked = lock_base if lock_base is not None else self._use_locked_base
        is_dual_mode = getattr(self.motiongen_server, "dual_mode", False)
        if is_dual_mode:
            req = DualMotionGenPlanRequest(
                env_ids=[self.env_id],
                target_pos=target_flat,
                robot_states=robot_states_dict,
                lock_base=use_locked,
            )
        else:
            req = MotionGenPlanRequest(
                env_ids=[self.env_id],
                target_pos=target_flat,
                robot_states=robot_states_dict,
            )
        self.inflight_motiongen_future = self.motiongen_server.submit_plan(req)
        self.inflight_motiongen_target = target_pose_all.clone()
        if self.debug:
            lock_str = f" lock_base={use_locked}" if is_dual_mode else ""
            print(
                "Submit MotionGen",
                "env_id=",
                self.env_id,
                "robot=",
                self.robot_name,
                "target=",
                self.inflight_motiongen_target,
                lock_str,
            )
        self.inflight_ik_future = None
        self.inflight_ik_target = None

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #

    def _visualize_pose_rows(
        self, traj: torch.Tensor, clear_existing: bool = True
    ) -> None:
        if not self.debug_viz or traj is None or traj.numel() == 0:
            return
        origin_cpu = self._get_env_origin().detach().cpu()
        points: List[List[float]] = []
        t = traj.detach().cpu()
        if t.shape[1] >= 19:
            for i in range(t.shape[0]):
                for sl in (slice(0, 7), slice(7, 14)):
                    pose = t[i, sl]
                    if not torch.isnan(pose[:3]).any():
                        points.append((pose[:3] + origin_cpu).tolist())
        elif t.shape[1] >= 7:
            for i in range(t.shape[0]):
                pose = t[i, :7]
                if not torch.isnan(pose[:3]).any():
                    points.append((pose[:3] + origin_cpu).tolist())
        if points:
            draw_waypoints(points, clear_existing=clear_existing)

    def _visualize_plan_rows(
        self, plan: torch.Tensor, clear_existing: bool = True
    ) -> None:
        if not self.debug_viz or plan is None or plan.numel() == 0:
            return
        try:
            base_dim, arm_dim, _ = self._get_action_dims()
        except Exception:
            base_dim, arm_dim = 0, 7
        mdb = base_dim - 1 if base_dim > 0 else 0
        origin_cpu = self._get_env_origin().detach().cpu()
        points: List[List[float]] = []
        if plan.ndim == 1:
            plan = plan.unsqueeze(0)
        for ri in range(plan.shape[0]):
            row = plan[ri].reshape(-1)
            arm_flat = (
                row[mdb : mdb + arm_dim]
                if row.numel() >= mdb + arm_dim and mdb + arm_dim > 0
                else row
            )
            if arm_flat.numel() % 7 != 0:
                continue
            poses = arm_flat.view(-1, 7)
            for pi in range(poses.shape[0]):
                if not torch.isnan(poses[pi, :3]).any():
                    points.append((poses[pi, :3] + origin_cpu).tolist())
        if points:
            draw_waypoints(points, clear_existing=clear_existing)

    def _visualize_direct_traj(self, traj: torch.Tensor) -> None:
        """Draw base-embedded direct waypoints: mdb poses (pelvis/torso/…) + all EEF 7D poses."""
        if not self.debug_viz or traj is None or traj.numel() == 0:
            return
        origin_cpu = self._get_env_origin().detach().cpu()
        points: List[List[float]] = []
        t = traj.detach().cpu()
        if t.ndim == 1:
            t = self._reshape_flat_embedded(t).cpu()
        if t.ndim != 2:
            return
        for i in range(t.shape[0]):
            row = t[i]
            try:
                base_pose, _, arm_flat, _ = self._parse_embedded_row(row)
            except (ValueError, RuntimeError):
                p3 = row[:3]
                if not torch.isnan(p3).any():
                    points.append((p3 + origin_cpu).tolist())
                continue
            if base_pose.numel() >= 7:
                for bi in range(base_pose.numel() // 7):
                    pose7 = base_pose[bi * 7 : (bi + 1) * 7]
                    if not torch.isnan(pose7[:3]).any():
                        points.append((pose7[:3] + origin_cpu).tolist())
            elif base_pose.numel() >= 3:
                if not torch.isnan(base_pose[:3]).any():
                    points.append((base_pose[:3] + origin_cpu).tolist())
            if arm_flat.numel() > 0 and arm_flat.numel() % 7 == 0:
                poses = arm_flat.view(-1, 7)
                for pi in range(poses.shape[0]):
                    if not torch.isnan(poses[pi, :3]).any():
                        points.append((poses[pi, :3] + origin_cpu).tolist())
        if points:
            draw_waypoints(points, clear_existing=True)

    def _visualize_robot_spheres(self, joint_pos: torch.Tensor) -> None:
        if self.motiongen_server is None:
            return
        if joint_pos.ndim > 1:
            joint_pos = joint_pos[self.env_id]
        try:
            sphere_list = self.motiongen_server.get_robot_spheres(joint_pos)
        except Exception:
            return
        if not sphere_list or len(sphere_list) == 0:
            return
        origin = self._get_env_origin().detach().cpu().numpy()
        spheres = sphere_list[0]
        if self._viz_spheres is None:
            self._viz_spheres = []
            for si, s in enumerate(spheres):
                pos = np.ravel(s.position)
                if np.any(np.isnan(pos)):
                    pos = np.zeros(3)
                pos = pos + origin
                sp = VisualSphere(
                    prim_path=f"/curobo/mobile_servo_sphere_env{self.env_id}_{si}",
                    position=pos,
                    radius=float(s.radius),
                    color=np.array([0.0, 0.8, 0.2]),
                )
                self._viz_spheres.append(sp)
        else:
            for si, s in enumerate(spheres):
                pos = np.ravel(s.position)
                if np.any(np.isnan(pos)):
                    if si < len(self._viz_spheres):
                        continue
                    else:
                        break
                pos = pos + origin
                if si >= len(self._viz_spheres):
                    sp = VisualSphere(
                        prim_path=f"/curobo/mobile_servo_sphere_env{self.env_id}_{si}",
                        position=pos,
                        radius=float(s.radius),
                        color=np.array([0.0, 0.8, 0.2]),
                    )
                    self._viz_spheres.append(sp)
                else:
                    self._viz_spheres[si].set_world_pose(position=pos)
                    self._viz_spheres[si].set_radius(float(s.radius))

    # ------------------------------------------------------------------ #
    # Step helpers (shared code for setting action & advancing index)
    # ------------------------------------------------------------------ #

    def _set_action_result(self, full_action: torch.Tensor) -> torch.Tensor:
        self.current_target = full_action
        self.current_command = ["MobileServoL", self.robot_name, self.current_target]
        self.current_action = {"MobileServoL": full_action}
        return full_action

    def _nan_action(self) -> torch.Tensor:
        base_dim, arm_dim, eef_dim = self._get_action_dims()
        return torch.full(
            (base_dim + arm_dim + eef_dim,),
            torch.nan,
            device=self.env.device,
            dtype=torch.float32,
        )

    # ================================================================== #
    # Core interface: reset / step / get_done / update
    # ================================================================== #

    def reset(self, action):
        robot_id, hand_id, planner_mode, traj_raw = self._parse_servo_action(action)
        self.planner_mode = planner_mode
        self._set_robot_by_id(robot_id, hand_id)
        # Remember the raw input reference so ``refresh`` can identity-short-circuit.
        self._last_raw_input = traj_raw
        self.plan_actions = None
        self.plan_step = 0
        self._waypoint_step_counter = 0
        self.current_command = ["MobileServoL", self.robot_name, None]
        self.current_action = None
        self._eef_trajectory = None
        self._direct_base_embedded = False
        self.inflight_ik_future = None
        self.inflight_ik_target = None
        self.inflight_motiongen_future = None
        self.inflight_motiongen_target = None
        self._stream_arm_after_ik_success = False
        self._arm_stream_step = 0
        self._arm_stream_waypoint_counter = 0
        self._step_count = 0
        self._post_playback_timeout_counter = 0

        # ---- Path A: base-embedded trajectory → direct streaming ----
        if self._is_base_embedded_trajectory(traj_raw):
            if not isinstance(traj_raw, torch.Tensor):
                traj_raw = torch.as_tensor(
                    traj_raw, device=self.env.device, dtype=torch.float32
                )
            else:
                traj_raw = traj_raw.to(self.env.device, dtype=torch.float32)
            if traj_raw.ndim == 1:
                traj_raw = self._reshape_flat_embedded(traj_raw)
            self._direct_mode = True
            self._direct_base_embedded = True
            self._direct_trajectory = traj_raw
            self._direct_step = 0
            self._direct_waypoint_counter = 0
            self._trajectory = None
            if self.debug:
                print(
                    f"[MobileServoL] base-embedded direct mode "
                    f"env_id={self.env_id} shape={traj_raw.shape}"
                )
            if self.debug_viz:
                self._visualize_direct_traj(traj_raw)
            return

        # ---- Path B: arm-only trajectory → IK then decide ----
        self._direct_mode = False
        self._direct_trajectory = None
        traj_tensor = traj_raw
        if not isinstance(traj_tensor, torch.Tensor):
            traj_tensor = torch.as_tensor(
                traj_tensor, device=self.env.device, dtype=torch.float32
            )
        else:
            traj_tensor = traj_tensor.to(self.env.device, dtype=torch.float32)

        traj_parsed, eef_t = self._parse_servo_arm_trajectory(traj_tensor)

        self._trajectory = traj_parsed
        self._eef_trajectory = eef_t
        if (
            self.debug_viz
            and self._trajectory is not None
            and self._trajectory.numel() > 0
        ):
            self._visualize_pose_rows(self._trajectory)
        if self._trajectory is None or self._trajectory.shape[0] == 0:
            return

        last_wp = self._trajectory[-1].view(-1)
        if last_wp.shape[0] >= 14:
            target_pose = last_wp[:14].view(2, 7)
        else:
            target_pose = last_wp[:7].view(1, 7)

        self._submit_ik(target_pose)
        if self.debug:
            print(
                f"[MobileServoL] arm-only: submitted fixed-base IK "
                f"env_id={self.env_id} robot={self.robot_name}"
            )

    def refresh(self, action) -> None:
        """Re-bind when atomic task re-sends the same planner key each step.

        ``GlobalPlannerManager`` calls ``refresh`` if ``MobileServoL`` is already
        active.  A no-op when robot ids and trajectory are unchanged so direct
        streaming indices (``_direct_step``, etc.) are not reset every tick.
        """
        robot_id, hand_id, _, traj_raw = self._parse_servo_action(action)
        if self._set_robot_by_id(robot_id, hand_id):
            self.reset(action)
            return

        # Fast path: identical raw input tensor as last reset → fully no-op.
        # Atomic skills (OpenDrawer pull, Dehatch standup ...) re-send the same
        # trajectory object every tick; without this short-circuit we would
        # re-parse, clobber the inflight IK future and reset the stream step,
        # trapping the planner in a "submit IK → success → reset → submit IK"
        # loop that never streams a single waypoint.
        if traj_raw is not None and traj_raw is self._last_raw_input:
            return

        # Direct streaming: same tensor re-sent (typical Dehatch / DexGrasp loop).
        if self._direct_mode and self._direct_trajectory is not None:
            if traj_raw is self._direct_trajectory:
                return
            if not isinstance(traj_raw, torch.Tensor):
                t_new = torch.as_tensor(
                    traj_raw, device=self.env.device, dtype=torch.float32
                )
            else:
                t_new = traj_raw.to(self.env.device, dtype=torch.float32)
            if self._direct_base_embedded and self._is_base_embedded_trajectory(t_new):
                if t_new.ndim == 1:
                    t_new = self._reshape_flat_embedded(t_new)
                if t_new.shape == self._direct_trajectory.shape and torch.allclose(
                    t_new, self._direct_trajectory
                ):
                    return
            elif (
                not self._direct_base_embedded
                and t_new.shape == self._direct_trajectory.shape
                and torch.allclose(t_new, self._direct_trajectory)
            ):
                return

        # Arm stream after IK: same underlying arm trajectory object.
        if self._stream_arm_after_ik_success and self._trajectory is not None:
            if traj_raw is self._trajectory:
                return

        self.reset(action)

    def step(self) -> torch.Tensor:
        self._step_count += 1
        if self._is_playback_fully_streamed():
            self._post_playback_timeout_counter += 1
        else:
            self._post_playback_timeout_counter = 0
        nan_action = self._nan_action()

        if self.debug_sphere:
            try:
                st = self._get_robot_state()
                jp = st.get("joint_pos")
                if jp is not None:
                    self._visualize_robot_spheres(jp.clone())
            except Exception:
                pass

        # ---- Direct streaming (base-embedded or arm-only linear) ----
        if self._direct_mode and self._direct_trajectory is not None:
            traj = self._direct_trajectory
            T = traj.shape[0]
            if T == 0:
                self.current_action = {"MobileServoL": nan_action}
                return nan_action
            idx = min(self._direct_step, T - 1)
            eef_row = None
            if not self._direct_base_embedded and self._eef_trajectory is not None:
                eef_row = self._eef_trajectory[
                    min(idx, self._eef_trajectory.shape[0] - 1)
                ]
            full_action = self._build_action_from_embedded(
                traj[idx], eef_external=eef_row
            )
            # Base-embedded rows → advance on base.
            # Arm-only direct (synthesized base from linear synth) → advance every
            # tick: this branch is the IK-fail-fallback path where we coupled
            # the base translation with arm waypoint progress upfront. Pacing
            # the row advance by EEF arrival drops the target advance rate to
            # the timeout cadence (3 cm/s with waypoint_max_steps=3 + 200 wp /
            # 30 cm), which falls below the PController dead zone's
            # steady-state lag and freezes the base mid-motion. Streaming
            # every tick (~9 cm/s for 200 wp / 30 cm at 60 Hz) keeps the base
            # PD comfortably above its dead zone so it tracks smoothly.
            if self._direct_base_embedded:
                arrived = self._check_base_waypoint_arrived(traj[idx].view(-1)[:3])
                self._direct_waypoint_counter += 1
                if arrived or self._direct_waypoint_counter >= self._waypoint_max_steps:
                    self._direct_step += 1
                    self._direct_waypoint_counter = 0
            else:
                self._direct_step += 1
                self._direct_waypoint_counter = 0
            return self._set_action_result(full_action)

        # ---- Arm-only: IK succeeded → stream arm with base NaN + lock -1 ----
        if self._stream_arm_after_ik_success and self._trajectory is not None:
            T = self._trajectory.shape[0]
            if T == 0:
                self.current_action = {"MobileServoL": nan_action}
                return nan_action
            idx = min(self._arm_stream_step, T - 1)
            wp = self._trajectory[idx]
            self.current_eef_target = None
            if self._eef_trajectory is not None and self._eef_trajectory.shape[0] > 0:
                self.current_eef_target = self._eef_trajectory[
                    min(idx, self._eef_trajectory.shape[0] - 1)
                ]
            full_action = self._build_full_action(
                wp.view(-1), lock_flag_override=self._LOCK_FLAG_LOCK_SKIP
            )
            arrived = self._check_eef_waypoint_arrived(wp)
            self._arm_stream_waypoint_counter += 1
            if arrived or self._arm_stream_waypoint_counter >= self._waypoint_max_steps:
                prev_step = self._arm_stream_step
                self._arm_stream_step += 1
                if self.debug and prev_step % max(1, T // 10) == 0:
                    reason = "arrived" if arrived else "timeout"
                    print(
                        f"[MobileServoL] env_id={self.env_id} arm-stream "
                        f"wp {prev_step}/{T} → {self._arm_stream_step} ({reason})"
                    )
                self._arm_stream_waypoint_counter = 0
            return self._set_action_result(full_action)

        if self._trajectory is None or self._trajectory.shape[0] == 0:
            self.current_action = {"MobileServoL": nan_action}
            return nan_action

        # ---- 1) IK result handling (fixed-base check; same flow as MobileMoveL.step) ----
        if self.inflight_ik_future is not None and self.inflight_ik_future.done():
            inflight_target = self.inflight_ik_target
            try:
                success, _, env_ids = self.inflight_ik_future.result()
            except Exception as e:
                if self.debug:
                    print(
                        "IK result exception",
                        "env_id=",
                        self.env_id,
                        "robot=",
                        self.robot_name,
                        "target=",
                        inflight_target,
                        "error=",
                        e,
                    )
                success = []
                env_ids = []
            self.inflight_ik_future = None
            self.inflight_ik_target = None
            if len(env_ids) >= 1:
                assert len(env_ids) == 1, (
                    f"Expected 1 env_id, got {len(env_ids)}, env_ids: {env_ids}"
                )
                assert env_ids[0] == self.env_id, (
                    f"Expected env_id {self.env_id}, got {env_ids[0]}, env_ids: {env_ids}"
                )

            ik_success = len(success) > 0 and success[0]

            if ik_success:
                self._stream_arm_after_ik_success = True
                self._arm_stream_step = 0
                self._arm_stream_waypoint_counter = 0
                if self.debug:
                    print(
                        "IK (fixed base) SUCCESS -> stream arm trajectory, base NaN lock=-1",
                        "env_id=",
                        self.env_id,
                        "robot=",
                        self.robot_name,
                        "target=",
                        inflight_target,
                    )
                self.current_action = {"MobileServoL": nan_action}
                return nan_action

            if self.debug:
                print(
                    "IK (fixed base) FAILED -> planner_mode branch",
                    "env_id=",
                    self.env_id,
                    "robot=",
                    self.robot_name,
                    "target=",
                    inflight_target,
                )
            last_wp = self._trajectory[-1].view(-1)
            target_pose = (
                last_wp[:14].view(2, 7)
                if last_wp.shape[0] >= 14
                else last_wp[:7].view(1, 7)
            )
            if not self._should_use_motiongen_servo(target_pose):
                self._direct_mode = True
                self._direct_base_embedded = False
                self._direct_trajectory = self._build_arm_only_linear_direct_trajectory(
                    self._trajectory
                )
                self._direct_step = 0
                self._direct_waypoint_counter = 0
                if self.debug:
                    print(
                        f"[MobileServoL] IK fail -> linear base offset N="
                        f"{self._direct_trajectory.shape[0]} planner_mode={self.planner_mode}"
                    )
                if self.debug_viz and self._direct_trajectory.numel() > 0:
                    self._visualize_direct_traj(self._direct_trajectory)
            else:
                self._use_locked_base = False
                if inflight_target is not None:
                    self._submit_plan(inflight_target, lock_base=False)
            self.current_action = {"MobileServoL": nan_action}
            return nan_action

        # ---- 2) MotionGen result handling ----
        if (
            self.inflight_motiongen_future is not None
            and self.inflight_motiongen_future.done()
        ):
            inflight_target = self.inflight_motiongen_target
            try:
                actions_list, success, env_ids = self.inflight_motiongen_future.result()
            except Exception as e:
                print(
                    f"MotionGen result failed for request {id(self.inflight_motiongen_future)}: {e}\n{traceback.format_exc()}"
                )
                assert False, (
                    f"MotionGen result failed for request {id(self.inflight_motiongen_future)}: {e}\n{traceback.format_exc()}"
                )
            self.inflight_motiongen_future = None
            self.inflight_motiongen_target = None
            assert len(env_ids) == 1, (
                f"Expected 1 env_id, got {len(env_ids)}, env_ids: {env_ids}"
            )
            assert env_ids[0] == self.env_id, (
                f"Expected env_id {self.env_id}, got {env_ids[0]}, env_ids: {env_ids}"
            )
            if actions_list and success and success[0]:
                traj_raw = actions_list[0]
                if self.debug:
                    print(
                        "MotionGen result success",
                        "env_id=",
                        self.env_id,
                        "robot=",
                        self.robot_name,
                        "target=",
                        inflight_target,
                    )
                self.plan_actions = (
                    traj_raw.to(self.env.device, dtype=torch.float32)
                    if isinstance(traj_raw, torch.Tensor)
                    else torch.as_tensor(
                        traj_raw, device=self.env.device, dtype=torch.float32
                    )
                )
                if self.plan_actions.ndim == 1:
                    self.plan_actions = self.plan_actions.unsqueeze(0)
                self._align_plan_streams()
                self.plan_step = 0
                if self.debug_viz:
                    self._visualize_plan_rows(self.plan_actions)
            else:
                if self.debug:
                    print(
                        "MotionGen result failed",
                        "env_id=",
                        self.env_id,
                        "robot=",
                        self.robot_name,
                        "target=",
                        inflight_target,
                    )

        # Block when no plan yet but IK or MotionGen still running
        if self.plan_actions is None or self.plan_actions.shape[0] == 0:
            if (
                self.inflight_ik_future is not None
                or self.inflight_motiongen_future is not None
            ):
                self.current_action = {"MobileServoL": nan_action}
                return nan_action
            self.current_action = {"MobileServoL": nan_action}
            return nan_action

        # ---- Stream MotionGen plan (lock = 0 / NAV) ----
        # Advance every tick (no arrive gate). Same rationale as the
        # arm-only-linear branch in direct mode: the MG plan is a synthesized
        # base+arm trajectory whose pacing is decided by ServoL, not by
        # downstream physics tracking. Gating on EEF arrival drops the row
        # advance rate to the timeout cadence (3 ticks/row) which falls
        # below the PController dead-zone steady-state lag and freezes the
        # base mid-motion. The downstream PD controllers (base PController +
        # arm IK) follow as fast as physics allows; we don't wait for them.
        T = self.plan_actions.shape[0]
        row: torch.Tensor
        if self.plan_step < T:
            row = self.plan_actions[self.plan_step]
            if row.ndim > 1:
                row = row.view(-1)
            self.plan_step += 1
            self._waypoint_step_counter = 0
            if self.plan_step < T:
                row = self.plan_actions[self.plan_step].view(-1)
        else:
            row = self.plan_actions[-1].view(-1)

        self.current_eef_target = None
        if self._eef_trajectory is not None and self._eef_trajectory.shape[0] > 0:
            eef_idx = min(self.plan_step, self._eef_trajectory.shape[0] - 1)
            self.current_eef_target = self._eef_trajectory[eef_idx]

        full_action = self._build_full_action(
            row, lock_flag_override=self._LOCK_FLAG_NAV
        )
        return self._set_action_result(full_action)

    def get_done(self) -> bool:
        # ``_timeout_steps`` applies only after playback: ``_post_playback_timeout_counter``
        # increments once per ``step()`` while the traj/plan is fully streamed.
        if self._direct_mode:
            if self._direct_trajectory is None:
                return False
            tdir = self._direct_trajectory.shape[0]
            if tdir == 0 or self._direct_step < tdir:
                return False
            if self._post_playback_timeout_counter >= self._timeout_steps:
                return True
            last_row = self._direct_trajectory[-1]
            if self._direct_base_embedded:
                base_pose, _, _, _ = self._parse_embedded_row(last_row)
                if base_pose.numel() < 3:
                    return False
                return self._done_last_base_pos_arrived(base_pose[:3])
            tp = self._direct_linear_row_to_eef_target_poses(last_row)
            if tp.numel() == 0:
                return False
            return self._done_last_eef_poses_arrived(tp)
        if self._stream_arm_after_ik_success:
            if self._trajectory is None:
                return False
            ta = self._trajectory.shape[0]
            if ta == 0 or self._arm_stream_step < ta:
                return False
            if self._post_playback_timeout_counter >= self._timeout_steps:
                return True
            poses = self._arm_waypoint_row_to_target_poses(self._trajectory[-1])
            return self._done_last_eef_poses_arrived(poses)
        if self._trajectory is None or self.plan_actions is None:
            return False
        tp = self.plan_actions.shape[0]
        if tp == 0 or self.plan_step < tp:
            return False
        if self._post_playback_timeout_counter >= self._timeout_steps:
            return True
        last = self.plan_actions[-1].view(-1)
        if self._motiongen_plan_row_has_base(last):
            return self._done_last_base_pos_arrived(last[:3])
        poses = self._motiongen_row_arm_target_poses_for_done(last)
        if poses.numel() == 0:
            return False
        return self._done_last_eef_poses_arrived(poses)

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        # Check env terminated / truncated BEFORE playback-done so the
        # atomic skill and task layers see a ``truncated=1/2`` signal the
        # same tick the env resets. Without this MobileServoL keeps
        # streaming while the scene has already been reset, and
        # AutoCollectManager never clears the stale task → whole pipeline
        # appears to "stop" after a reset.
        env_info = info.get("env_info") if isinstance(info, dict) else None
        if env_info is not None and len(env_info) >= 4:
            terminated = env_info[2]
            truncated = env_info[3]
            if terminated is not None and bool(terminated[self.env_id]):
                return {
                    "type": "MobileServoL",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": True,
                    "state": "truncated: env terminated first",
                    "truncated": 1,
                }
            if truncated is not None and bool(truncated[self.env_id]):
                return {
                    "type": "MobileServoL",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": "truncated: env truncated first",
                    "truncated": 2,
                }

        done = self.get_done()
        return {
            "type": "MobileServoL",
            "command": self.current_command,
            "action": self.current_action,
            "finished": done,
            "state": "running" if not done else "finished",
            "truncated": 0,
        }
