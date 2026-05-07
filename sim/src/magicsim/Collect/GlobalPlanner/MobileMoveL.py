import traceback
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from magicsim.Collect.GlobalPlanner.GlobalPlanner import GlobalPlanner
from magicsim.Env.Utils.file import Logger
from magicsim.Env.Utils.viz import draw_waypoints
from magicsim.Env.Planner.Services.IKServer import IKPlanRequest
from magicsim.Env.Planner.Services.MotionGenServer import MotionGenPlanRequest
from magicsim.Env.Planner.Services.DualMotionGenServer import DualMotionGenPlanRequest
from magicsim.Env.Planner.Services.DualIKServer import DualIKPlanRequest
from magicsim.Env.Planner.Utils import quat_angle_between
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig
from isaacsim.core.api.objects.sphere import VisualSphere


class MobileMoveL(GlobalPlanner):
    """Global planner for mobile base + end-effector pose targets.

    G1 / PController: last dim of ``base_action`` is ``lock_flag``. Call sites pass it explicitly
    via ``lock_flag_override`` to ``_build_full_action`` (see ``_LOCK_FLAG_*``).

    The third header field ``planner_mode`` (together with ``robot_id`` and ``hand_id``) selects whether to run IK,
    whether the base is locked vs free, and whether to use MotionGen. **Free base always uses MotionGen when
    planning is required.**

    ``planner_mode`` summary (distance gate uses ``motiongen_distance_threshold`` / ``translation_threshold``)::

        +--------+----------+---------------+-----------------------------------------------+
        |  mode  | Submit IK |  Base mode   | MotionGen                                     |
        +--------+----------+---------------+-----------------------------------------------+
        |   0    |    No    | locked (fix)  | None; single-step arm target                  |
        |   1    |    No    | locked (fix)  | Always MotionGen                              |
        |   2    |    No    | locked (fix)  | MG if EEF distance > threshold, else direct   |
        |   3    |    No    | free (mobile) | Always MotionGen                              |
        |   4    |    No    | locked (fix)  | None; linear-interp from current EEF to target|
        |        |          |               | over ``config.interp_steps`` calls of step()  |
        |  -1    |   Yes    | from IK       | Free: always MG; locked: MG if distance > thr.  |
        |  -2    |   Yes    | from IK       | Free: always MG; locked: always MG (no distance)|
        +--------+----------+---------------+-----------------------------------------------+

    Default when the header is omitted (plain target tensor or legacy ``(robot_id, target)``) is ``planner_mode=-1``
    (IK first, then plan).

    Reactive re-planning applies only in **free base** and is controlled by ``reactive`` / ``reactive_step`` in config,
    not by the third header field.

    ``min_steps_before_done`` (default 50): ``get_done()`` stays false until ``step_count`` reaches this
    (applies to pose arrival and timeout).
    """

    # G1 / PController: ``base_action`` last dim (explicit at each ``_build_full_action`` call).
    _LOCK_FLAG_LOCK_SKIP = -1.0
    _LOCK_FLAG_NAV = 0.0

    # Mode that activates locked-base + EEF linear-interp (see docstring table).
    INTERP_MODE: int = 4

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        self.current_target = None
        self.current_base_target = None
        self.current_arm_targets = None  # [eef_num, 7]
        self.current_eef_target = None
        self.arm_eef_num = 1
        self.robot_id = -1
        self.hand_id = 0  # 0 = right (default), 1 = left
        # Third header value: planner_mode (see class docstring table).
        self.planner_mode = -1
        self.robot_name = None
        self.motiongen_server = None
        self.ik_server = None
        self.inflight_motiongen_future = None
        self.inflight_motiongen_target = None
        self.inflight_ik_future = None
        self.inflight_ik_target = None
        self.pending_target = None
        self.plan_actions = None
        self.plan_step = 0
        self.last_action = None
        self.step_count = 0  # Track number of steps
        self.debug = config.get("debug", True)
        self.debug_viz = config.get("debug_viz", True)
        self.debug_sphere = config.get("debug_sphere", False)
        self._viz_spheres = None
        # For dual mode: tracks whether locked base IK succeeded
        # If IK succeeds -> use locked base MotionGen
        # If IK fails -> use free base MotionGen
        self._use_locked_base = False

        # Position thresholds for arrival detection (position-based waypoint advance)
        self._base_pos_threshold = config.get("base_pos_threshold", 0.15)
        self._base_rot_threshold = config.get("base_rot_threshold", 0.1)
        self._eef_pos_threshold = config.get("eef_pos_threshold", 0.03)
        self._eef_rot_threshold = config.get("eef_rot_threshold", 0.1)
        # Max steps to wait at a single waypoint before forcing advance
        self._waypoint_max_steps = config.get("waypoint_max_steps", 5)
        self._waypoint_step_counter = 0

        # Reactive mode: re-plan every reactive_step when using free-base MotionGen only
        # (IK failed -> free base). Reactive is disabled for locked base.
        self.reactive = config.get("reactive", True)
        # 50 ticks between successive reactive MG submits. With the
        # MotionGenServer ``num_trajopt_seeds`` dropped from 12 back to 4
        # (matches v1 MagicSim + curobo default), a single MG solve is
        # ~3× faster, so the GPU contention that previously motivated a
        # longer cooldown is gone.
        self.reactive_step = int(config.get("reactive_step", 50))
        self._reactive_last_override_step = -1
        self._reactive_last_submit_step = -1
        self._reactive_has_plan = False  # True after first MotionGen result
        self._reactive_inflight = (
            False  # True when current inflight is from reactive_submit
        )
        self._pending_silent_finish = False
        # Do not report done (pose/timeout) until at least this many step() ticks.
        self._min_steps_before_done = int(config.get("min_steps_before_done", 50))

        # Linear-interp hyperparameter for ``planner_mode == INTERP_MODE`` (4).
        # Default 0 disables interp (mode 4 falls back to single-step snap).
        self.interp_steps_default: int = int(config.get("interp_steps", 0))
        self.interp_steps: int = 0
        self.interp_substep: int = 0
        self.interp_start: torch.Tensor | None = None  # [eef_num, 7]
        self.interp_end: torch.Tensor | None = None  # [eef_num, 7]

        super().__init__(config, env, env_id, logger)

    def _reactive_active(self) -> bool:
        """Reactive re-planning only in free base; toggled by config ``reactive``."""
        if self._use_locked_base:
            return False
        return self.reactive

    def _mode_skip_ik(self) -> bool:
        """Modes 0–4 do not submit IK; base mode is fixed by mode (not IK)."""
        return self.planner_mode in (0, 1, 2, 3, 4)

    def _set_use_locked_from_mode(self) -> None:
        """Set ``_use_locked_base`` for planner_mode 0–4 (skip-IK paths)."""
        if self.planner_mode == 3:
            self._use_locked_base = False
        else:
            # 0, 1, 2, 4 are all locked-base paths.
            self._use_locked_base = True

    def _locked_base_distance_exceeds_threshold(
        self, target_pose: torch.Tensor
    ) -> bool:
        if target_pose.ndim == 1:
            target_pose = target_pose.view(-1, 7)
        cur_pos, _ = self._select_current_eef_for_target(target_pose)
        pair_count = min(cur_pos.shape[0], target_pose.shape[0])
        threshold = float(
            getattr(
                self.config,
                "motiongen_distance_threshold",
                self.config.motiongen_distance_threshold,
            )
        )
        max_diff = 0.0
        for eef_idx in range(pair_count):
            pos_diff = torch.linalg.norm(
                cur_pos[eef_idx] - target_pose[eef_idx, :3]
            ).item()
            max_diff = max(max_diff, pos_diff)
        return max_diff > threshold

    def _begin_plan_after_mode_resolution(self, target_targets: torch.Tensor) -> None:
        """Start MotionGen or single-step plan after targets and _use_locked_base are set."""
        if self._should_use_motiongen(target_targets):
            self._reactive_has_plan = False
            self._submit_plan(self.current_arm_targets)
        else:
            self.plan_actions = target_targets.view(-1).unsqueeze(0)
            self.plan_step = 0

    def _replan_from_pending_mismatch(self) -> None:
        """New target arrived while IK/MotionGen was in flight; replan from ``current_arm_targets``."""
        assert self.pending_target is not None
        if self._mode_skip_ik():
            self._set_use_locked_from_mode()
            self._begin_plan_after_mode_resolution(self.current_arm_targets)
        else:
            self._submit_ik(self.current_arm_targets)
        self.pending_target = None

    def _get_robot_name_list(self):
        return list(self.env.scene.robot_manager.robots.keys())

    def _get_planner_manager(self):
        planner_manager = getattr(self.env.scene, "planner_manager", None)
        if planner_manager is None:
            raise RuntimeError("PlannerManager is not available in the environment.")
        return planner_manager

    def _set_robot_by_id(self, robot_id: int, hand_id: int = 0) -> bool:
        robot_id = int(robot_id)
        hand_id = int(hand_id)
        if (
            robot_id == self.robot_id
            and hand_id == self.hand_id
            and self.robot_name is not None
        ):
            return False
        robot_name_list = self._get_robot_name_list()
        if robot_id < 0 or robot_id >= len(robot_name_list):
            raise ValueError(
                f"robot_id {robot_id} out of range for robot_name_list={robot_name_list}"
            )
        self.robot_id = robot_id
        self.hand_id = hand_id
        self.robot_name = robot_name_list[robot_id]
        # Single server per robot (MERGE_LEFT_RIGHT §1–§8). ``hand_id``
        # only drives target packing now.
        planner_manager = self._get_planner_manager()
        motiongen_dict = getattr(planner_manager, "motiongen_server", None)
        ik_dict = getattr(planner_manager, "ik_server", None)
        if not motiongen_dict or self.robot_name not in motiongen_dict:
            raise RuntimeError(
                f"MotionGenServer not found for robot '{self.robot_name}'."
            )
        if not ik_dict or self.robot_name not in ik_dict:
            raise RuntimeError(f"IKServer not found for robot '{self.robot_name}'.")
        self.motiongen_server = motiongen_dict[self.robot_name]
        self.ik_server = ik_dict[self.robot_name]
        return True

    def _parse_action(self, action: torch.Tensor):
        """``((robot_id, hand_id, planner_mode), target_tensor)`` — see ``parse_planner_header``."""
        robot_id, hand_id, planner_mode, target = GlobalPlanner.parse_planner_header(
            action,
            default_robot_id=self.robot_id if self.robot_id >= 0 else 0,
            default_hand_id=self.hand_id,
        )
        self.planner_mode = planner_mode
        return robot_id, hand_id, target

    def _parse_target(self, action: torch.Tensor):
        base_dim, _, eef_dim = self._get_action_dims()
        max_eef_num, per_eef_dim = self._eef_layout_from_robot()
        return GlobalPlanner.parse_target_vector(
            action,
            device=self.env.device,
            base_dim=base_dim,
            eef_dim=eef_dim,
            hand_id=self.hand_id,
            allow_base=False,
            max_eef_num=max_eef_num,
            per_eef_dim=per_eef_dim,
        )

    def _build_full_action(
        self,
        arm_action_flat: torch.Tensor,
        lock_flag_override: Optional[float] = None,
    ) -> torch.Tensor:
        return self._build_full_action_mobile(
            arm_action_flat, lock_flag_override=lock_flag_override
        )

    def _reshape_eef_targets(self, target: torch.Tensor) -> torch.Tensor:
        if target is None:
            return None
        if target.ndim == 1:
            if target.shape[0] % 7 != 0:
                raise ValueError(
                    f"EEF target length {target.shape[0]} not multiple of 7."
                )
            return target.view(-1, 7)
        return target

    def _nan_action(self) -> torch.Tensor:
        base_dim, arm_dim, eef_dim = self._get_action_dims()
        total_dim = base_dim + arm_dim + eef_dim
        return torch.full(
            (total_dim,),
            torch.nan,
            device=self.env.device,
            dtype=torch.float32,
        )

    def _targets_close(self, target_a: torch.Tensor, target_b: torch.Tensor) -> bool:
        if target_a is None or target_b is None:
            return False
        target_a = self._reshape_eef_targets(target_a)
        target_b = self._reshape_eef_targets(target_b)
        if target_a.shape != target_b.shape:
            return False
        for eef_idx in range(target_a.shape[0]):
            pos_diff = torch.linalg.norm(
                target_a[eef_idx, :3] - target_b[eef_idx, :3]
            ).item()
            rot_diff = quat_angle_between(
                target_a[eef_idx, 3:].unsqueeze(0),
                target_b[eef_idx, 3:].unsqueeze(0),
            ).item()
            if pos_diff >= float(
                self.config.translation_threshold
            ) or rot_diff >= float(self.config.rotation_threshold):
                return False
        return True

    def _submit_plan(
        self,
        target_pose_all: torch.Tensor,
        lock_base: bool = None,
        reactive_submit: bool = False,
    ):
        """Submit a MotionGen planning request.

        Args:
            target_pose_all: Target poses
            lock_base: Override lock_base setting. If None, uses self._use_locked_base
            reactive_submit: If True, do not clear plan_actions (for reactive re-planning)
        """
        robot_state = self._get_robot_state()
        robot_states_dict = {
            "base_pos": robot_state["base_pos"],
            "base_quat": robot_state["base_quat"],
            "joint_pos": robot_state["joint_pos"],
            "joint_vel": robot_state["joint_vel"],
        }
        # Pack into the unified (Dual)MotionGenPlanRequest 2-D shape
        # ``(1, eef_num*7)`` via ``_expand_arm_action`` (single source of
        # truth for hand_id NaN-padding). NaN rows trigger the Server's
        # per-env-tool disable.
        target_flat = self._expand_arm_action(target_pose_all.view(-1)).view(1, -1)

        # Determine lock_base setting
        use_locked = lock_base if lock_base is not None else self._use_locked_base

        # Check if server supports dual mode
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
        self._reactive_inflight = reactive_submit
        if not reactive_submit:
            self.plan_actions = None
            self.plan_step = 0
            self.last_action = None
            self.pending_target = None

    def _submit_ik(self, target_pose: torch.Tensor):
        """Submit IK check with lock_base=True (fixed base).

        This checks if the target is reachable without moving the base.
        The result determines whether to use locked or free base MotionGen.
        """
        robot_state = self._get_robot_state()
        robot_states_dict = {
            "base_pos": robot_state["base_pos"],
            "base_quat": robot_state["base_quat"],
            "joint_pos": robot_state["joint_pos"],
            "joint_vel": robot_state["joint_vel"],
        }
        # IKPlanRequest single-mode 2-D shape ``(1, L*7)``. ``_expand_arm_action``
        # packs hand_id NaN rows for the Server's per-env-tool disable.
        target_flat = self._expand_arm_action(target_pose.view(-1)).view(1, -1)

        # Check if server supports dual mode
        # Always use lock_base=True for IK check (fixed base reachability test)
        is_dual_mode = getattr(self.ik_server, "dual_mode", False)
        if is_dual_mode:
            req = DualIKPlanRequest(
                env_ids=[self.env_id],
                target_pos=target_flat,
                robot_states=robot_states_dict,
                mode="single",
                lock_base=True,  # Always check with fixed base
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
        self._reactive_inflight = False
        self.plan_actions = None
        self.plan_step = 0
        self.last_action = None
        self.pending_target = None

    def _get_robot_state(self) -> Dict[str, torch.Tensor]:
        robot_states = self.env.scene.robot_manager.get_robot_state(noise_flag=False)[0]
        if isinstance(robot_states, dict) and self.robot_name in robot_states:
            return robot_states[self.robot_name]
        if isinstance(robot_states, dict):
            return next(iter(robot_states.values()))
        return robot_states

    def _get_action_dims(self) -> tuple[int, int, int]:
        if self.robot_name is None:
            raise RuntimeError("Robot name not set. Call _set_robot_by_id first.")
        planner_manager = self._get_planner_manager()
        info = planner_manager.get_info()
        robot_info = info.get(self.robot_name, {})
        base_dim = int(robot_info.get("base", {}).get("action_dim", 0))
        arm_dim = int(robot_info.get("arm", {}).get("action_dim", 0))
        eef_dim = int(robot_info.get("eef", {}).get("action_dim", 0))
        if base_dim == 0 and arm_dim == 0 and eef_dim == 0:
            raise RuntimeError("No action dims found for robot {self.robot_name}")
        return base_dim, arm_dim, eef_dim

    def _expand_eef_action(self, eef_action: torch.Tensor) -> torch.Tensor:
        """Expand a per-gripper action to full eef_dim, filling the other with NaN.

        For hand_id 0 (right): action goes into first slot.
        For hand_id 1 (left):  action goes into last slot.
        For hand_id -1 (both): returned as-is.
        """
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

    def _build_fallback_plan(self, arm_target: torch.Tensor) -> torch.Tensor:
        """Build a single-step fallback plan when MotionGen fails.

        Locked-base only: base dims are omitted (``_build_full_action``
        fills them with NaN + ``lock_flag=-1`` / ``lock_skip``). The
        expanded arm target is returned so the arm still attempts to
        reach, though without a proper trajectory.

        Free-base failures do NOT use this fallback — they surface the
        failure to the caller via ``current_state = "failed"`` (see
        ``step()``'s MotionGen-result branch). An unreachable free-base
        target with 10-DOF search means the goal is genuinely infeasible;
        a base-interpolation fallback would mislead the caller.

        Args:
            arm_target: Flat arm target tensor (e.g. [7] for single arm or [14] for dual).

        Returns:
            [1, D] tensor suitable for storage in self.plan_actions.
        """
        assert self._use_locked_base, (
            "_build_fallback_plan is locked-base only. Free-base failures "
            "should set current_state='failed' instead of falling back."
        )
        arm_target = arm_target.view(-1)
        # Locked base failed: just send the arm target directly.
        # _build_full_action will fill base with NaN + lock_flag=-1 (lock_skip).
        combined = self._expand_arm_action(arm_target)
        return combined.unsqueeze(0)

    def _expand_arm_action(self, arm_action: torch.Tensor) -> torch.Tensor:
        """Expand a per-arm action to full arm_dim.

        For hand_id 0 (right): arm pose goes into first 7 dims.
        For hand_id 1 (left):  arm pose goes into last 7 dims.
        For hand_id -1 (both): returned as-is.

        If arm_dim > len(arm_action) (e.g. arm_dim=14 for single arm with dex hand),
        the remaining dims are filled from current_eef_target (e.g. hand joints) if
        the sizes match, otherwise NaN.
        """
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
    # ``_expand_arm_action(target.view(-1)).view(1, -1)`` for hand_id NaN
    # padding (single source of truth).

    def _get_current_base_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot_state = self._get_robot_state()
        return (
            robot_state["base_pos"][self.env_id],
            robot_state["base_quat"][self.env_id],
        )

    def _get_current_eef(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot_state = self._get_robot_state()
        return (
            robot_state["eef_pos"][self.env_id],
            robot_state["eef_quat"][self.env_id],
        )

    def _select_current_eef_for_target(
        self, target_pose_all: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the current EEF pos/quat matching the target's EEF count.

        For single-arm target on a dual-arm robot, selects by hand_id.
        For dual-arm target, returns all EEFs up to target count.
        """
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

    def _get_env_origin(self) -> torch.Tensor:
        return self.env.scene.env_origins[self.env_id].clone()

    def _visualize_waypoints(self, actions: List[torch.Tensor]) -> None:
        actions_cpu = [action.detach().cpu() for action in actions]
        origin_cpu = self._get_env_origin().detach().cpu()
        points: list[list[float]] = []
        for step in actions_cpu:
            step = step.view(-1)
            if step.numel() % 7 != 0:
                continue
            poses = step.view(-1, 7)
            for pose in poses:
                pos = (pose[:3] + origin_cpu).tolist()
                points.append(pos)
        draw_waypoints(points, clear_existing=True)

    def _visualize_robot_spheres(self, joint_pos: torch.Tensor) -> None:
        if self.motiongen_server is None:
            return

        if joint_pos.ndim > 1:
            joint_pos = joint_pos[self.env_id]
        try:
            sphere_list = self.motiongen_server.get_robot_spheres(joint_pos)
        except Exception as e:
            if self.debug:
                print("Failed to get robot spheres:", e)
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
                    prim_path=f"/curobo/robot_sphere_env{self.env_id}_{si}",
                    position=pos,
                    radius=float(s.radius),
                    color=np.array([0.0, 0.8, 0.2]),
                )
                self._viz_spheres.append(sp)
        else:
            for si, s in enumerate(spheres):
                if si >= len(self._viz_spheres):
                    pos = np.ravel(s.position)
                    if np.any(np.isnan(pos)):
                        continue
                    pos = pos + origin
                    sp = VisualSphere(
                        prim_path=f"/curobo/robot_sphere_env{self.env_id}_{si}",
                        position=pos,
                        radius=float(s.radius),
                        color=np.array([0.0, 0.8, 0.2]),
                    )
                    self._viz_spheres.append(sp)
                    continue
                pos = np.ravel(s.position)
                if np.any(np.isnan(pos)):
                    continue
                pos = pos + origin
                self._viz_spheres[si].set_world_pose(position=pos)
                self._viz_spheres[si].set_radius(float(s.radius))

    def _should_use_motiongen(self, target_pose: torch.Tensor) -> bool:
        # Free base: always MotionGen (mobile planning required).
        if not self._use_locked_base:
            return True
        m = self.planner_mode
        if m == 0:
            return False
        if m == 1:
            return True
        if m == 2:
            return self._locked_base_distance_exceeds_threshold(target_pose)
        if m == 3:
            return True
        if m == 4:
            # Locked base + linear-interp: never use MotionGen.
            return False
        if m == -1:
            return self._locked_base_distance_exceeds_threshold(target_pose)
        if m == -2:
            return True
        return self._locked_base_distance_exceeds_threshold(target_pose)

    def _check_waypoint_arrived(self, waypoint: torch.Tensor) -> bool:
        """Check if the robot has arrived at the given waypoint.

        For locked base (fix_base): check all EEF arrivals.
        For free base: only check base position (ignore rotation).
        """
        base_dim, arm_dim, eef_dim = self._get_action_dims()
        motiongen_base_dim = base_dim - 1 if base_dim > 0 else 0

        wp = waypoint.view(-1)
        # Determine if waypoint contains base data
        has_base_data = (
            motiongen_base_dim > 0 and wp.shape[0] == motiongen_base_dim + arm_dim
        )

        # If locked base (fix_base), check all EEFs
        if self._use_locked_base:
            eef_offset = motiongen_base_dim if has_base_data else 0
            arm_data = wp[eef_offset:]
            # arm_data may contain multiple 7D EEF targets
            if arm_data.shape[0] >= 7 and arm_data.shape[0] % 7 == 0:
                eef_targets = arm_data.view(-1, 7)
            else:
                eef_targets = arm_data[:7].unsqueeze(0)

            cur_pos, cur_quat = self._select_current_eef_for_target(eef_targets)
            pair_count = min(cur_pos.shape[0], eef_targets.shape[0])
            for eef_idx in range(pair_count):
                # Skip NaN targets (inactive arm)
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

        # Free base: only check base position (ignore rotation)
        assert has_base_data, "Free base mode must have base data in waypoint"
        target_base_pos = wp[:3]
        cur_base_pos, _ = self._get_current_base_pose()
        base_pos_diff = torch.linalg.norm(cur_base_pos - target_base_pos).item()
        return base_pos_diff < self._base_pos_threshold

    def _is_placeholder_target(self, arm_targets: torch.Tensor) -> bool:
        """Check if arm targets are all NaN (placeholder during init phase)."""
        return torch.isnan(arm_targets).all()

    def _find_closest_waypoint_idx(self, plan: torch.Tensor) -> int:
        """Find the waypoint index in plan closest to current base position.

        Only used in reactive mode (free base). Free-base plans have base pos in first 3 dims.
        Returns index in [0, plan.shape[0]).
        """
        plan = plan.view(plan.shape[0], -1)
        cur_base_pos, _ = self._get_current_base_pose()
        best_idx = 0
        best_dist = float("inf")
        for i in range(plan.shape[0]):
            target_pos = plan[i, :3]
            if torch.isnan(target_pos).any():
                continue
            dist = torch.linalg.norm(cur_base_pos - target_pos).item()
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def _trim_plan_from_current_state(self, plan: torch.Tensor) -> torch.Tensor:
        """Trim plan: remove waypoints before the one closest to current state.

        Returns trimmed plan [M, D] with M >= 1 (at least the last waypoint).
        """
        idx = self._find_closest_waypoint_idx(plan)
        trimmed = plan[idx:].clone()
        if trimmed.shape[0] == 0:
            trimmed = plan[-1:].clone()
        return trimmed

    def reset(self, action: torch.Tensor):
        # Reset expects action as ((robot_id, hand_id, planner_mode), target_tensor).
        robot_id, hand_id, target_action = self._parse_action(action)
        self._set_robot_by_id(robot_id, hand_id)
        (
            self.current_base_target,
            self.current_arm_targets,
            self.current_eef_target,
        ) = self._parse_target(target_action)
        self.arm_eef_num = int(self.current_arm_targets.shape[0])
        self.step_count = 0  # Reset step count
        self._waypoint_step_counter = 0

        self.inflight_motiongen_future = None
        self.inflight_motiongen_target = None
        self.inflight_ik_future = None
        self.inflight_ik_target = None
        self.pending_target = None
        self._use_locked_base = False
        # Clear interp state on every reset; ``_begin_plan_after_mode_resolution``
        # re-arms it for mode 4 below.
        self.interp_steps = 0
        self.interp_substep = 0
        self.interp_start = None
        self.interp_end = None
        self._reactive_last_override_step = -1
        self._reactive_last_submit_step = -1
        self._reactive_has_plan = False
        self._reactive_inflight = False
        self._pending_silent_finish = False

        if not self._is_placeholder_target(self.current_arm_targets):
            if self._mode_skip_ik():
                self._set_use_locked_from_mode()

        self.current_state = "ready"
        _lf = (
            self._LOCK_FLAG_LOCK_SKIP if self._use_locked_base else self._LOCK_FLAG_NAV
        )
        self.current_target = self._build_full_action(
            self.current_arm_targets.view(-1), lock_flag_override=_lf
        )
        self.current_command = ["MobileMoveL", self.robot_name, self.current_target]
        self.current_action = None

        # Skip planning if target is placeholder (all NaN during init)
        if not self._is_placeholder_target(self.current_arm_targets):
            if self._mode_skip_ik():
                self._begin_plan_after_mode_resolution(self.current_arm_targets)
            else:
                self._submit_ik(self.current_arm_targets)

    def step(self) -> torch.Tensor:
        if self.current_target is None or self.current_arm_targets is None:
            raise RuntimeError("Current Target Is Not Set, Please Call Reset First")

        # If target is placeholder (all NaN during init), return nan_action directly
        if self._is_placeholder_target(self.current_arm_targets):
            self.current_state = "waiting"
            return self._nan_action()

        # planner_mode 4 (locked base + EEF linear-interp): no IK, no MG, no
        # plan_actions queue. Each step() emits one lerped sub-target.
        if self.planner_mode == self.INTERP_MODE and self.interp_steps > 0:
            return self._step_interp()

        robot_state = self._get_robot_state()
        joint_pos = robot_state["joint_pos"].clone()
        if joint_pos is not None and self.debug_sphere:
            self._visualize_robot_spheres(joint_pos)

        # ---- 1) IK result handling ----
        # IK check is done with lock_base=True (fixed base)
        # If IK succeeds: use locked base MotionGen
        # If IK fails: use free base MotionGen (mobile base planning)
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
            self.inflight_ik_future = None
            self.inflight_ik_target = None
            assert len(env_ids) == 1, (
                f"Expected 1 env_id, got {len(env_ids)}, env_ids: {env_ids}"
            )
            assert env_ids[0] == self.env_id, (
                f"Expected env_id {self.env_id}, got {env_ids[0]}, env_ids: {env_ids}"
            )

            # Handle pending target change
            if self.pending_target is not None and not self._targets_close(
                self.pending_target, inflight_target
            ):
                self._replan_from_pending_mismatch()
                self.current_state = "planning"
                return self._nan_action()
            if self.pending_target is not None:
                self.pending_target = None

            target_targets = (
                inflight_target
                if inflight_target is not None
                else self.current_arm_targets
            )

            ik_success = len(success) > 0 and success[0]

            if ik_success:
                # IK succeeded with fixed base -> use locked base MotionGen
                self._use_locked_base = True
                if self.debug:
                    print(
                        "IK (fixed base) SUCCESS -> using locked base MotionGen",
                        "env_id=",
                        self.env_id,
                        "robot=",
                        self.robot_name,
                        "target=",
                        inflight_target,
                    )
            else:
                # IK failed with fixed base -> use free base MotionGen
                self._use_locked_base = False
                if self.debug:
                    print(
                        "IK (fixed base) FAILED -> using free base MotionGen",
                        "env_id=",
                        self.env_id,
                        "robot=",
                        self.robot_name,
                        "target=",
                        inflight_target,
                    )

            self._begin_plan_after_mode_resolution(target_targets)
            if self.inflight_motiongen_future is not None:
                self.current_state = "planning"
                return self._nan_action()

        # ---- 2) MotionGen result handling ----
        # If MotionGen fails, fall back to target pose directly.
        if (
            self.inflight_motiongen_future is not None
            and self.inflight_motiongen_future.done()
        ):
            inflight_target = self.inflight_motiongen_target
            try:
                actions, success, env_ids = self.inflight_motiongen_future.result()
            except Exception as e:
                print(
                    f"MotionGen result failed for request {id(self.inflight_motiongen_future)}: {e}\n{traceback.format_exc()}"
                )
                assert False, (
                    f"MotionGen result failed for request {id(self.inflight_motiongen_future)}: {e}\n{traceback.format_exc()}"
                )
                actions, success = [], []
            self.inflight_motiongen_future = None
            self.inflight_motiongen_target = None
            assert len(env_ids) == 1, (
                f"Expected 1 env_id, got {len(env_ids)}, env_ids: {env_ids}"
            )
            assert env_ids[0] == self.env_id, (
                f"Expected env_id {self.env_id}, got {env_ids[0]}, env_ids: {env_ids}"
            )
            if self.pending_target is not None and not self._targets_close(
                self.pending_target, inflight_target
            ):
                self._replan_from_pending_mismatch()
                self.current_state = "planning"
                return self._nan_action()
            if self.pending_target is not None:
                self.pending_target = None

            if len(actions) > 0 and len(success) > 0 and success[0]:
                plan = actions[0]
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
                if isinstance(plan, torch.Tensor):
                    plan = plan.to(self.env.device, dtype=torch.float32)
                else:
                    plan = torch.as_tensor(
                        plan, device=self.env.device, dtype=torch.float32
                    )
                if plan.ndim == 1:
                    plan = plan.unsqueeze(0)
                # Reactive: trim plan if we already have one and have walked enough
                should_override = (
                    not self._reactive_has_plan
                    or self.step_count - self._reactive_last_override_step
                    >= self.reactive_step
                )
                if (
                    should_override
                    and self._reactive_active()
                    and self._reactive_has_plan
                ):
                    plan = self._trim_plan_from_current_state(plan)
                if should_override:
                    self.plan_actions = plan
                    self.plan_step = 0
                    self._reactive_last_override_step = self.step_count
                    if self.debug_viz:
                        self._visualize_waypoints([self.plan_actions])
                self._reactive_has_plan = True
                self._reactive_inflight = False
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
                # Reactive re-plan failed: keep original plan, do not clear, do not visualize
                if self._reactive_inflight and self._reactive_has_plan:
                    self._reactive_inflight = False
                elif not self._use_locked_base:
                    # Free base + first-time (non-reactive) MotionGen
                    # failure ⇒ fail hard. No base-interpolation fallback:
                    # an IK-unreachable free-base target means the base
                    # itself is infeasible, and the old interpolation
                    # fallback would lie about that. Propagate failure up
                    # to the caller via ``current_state == "failed"`` (see
                    # ``update()`` at the bottom of this file).
                    self._reactive_inflight = False
                    self.current_state = "failed"
                    return self._nan_action()
                else:
                    # Locked-base first-time failure: keep the existing
                    # single-step fallback (send the arm target directly).
                    arm_fallback = (
                        inflight_target.view(-1)
                        if inflight_target is not None
                        else self.current_arm_targets.view(-1)
                    )
                    fallback_plan = self._build_fallback_plan(arm_fallback)
                    should_override = (
                        not self._reactive_has_plan
                        or self.step_count - self._reactive_last_override_step
                        >= self.reactive_step
                    )
                    if (
                        should_override
                        and self._reactive_active()
                        and self._reactive_has_plan
                    ):
                        fallback_plan = self._trim_plan_from_current_state(
                            fallback_plan
                        )
                    if should_override:
                        self.plan_actions = fallback_plan
                        self.plan_step = 0
                        self._reactive_last_override_step = self.step_count
                        if self.debug_viz:
                            self._visualize_waypoints([self.plan_actions])
                    self._reactive_has_plan = True
                self._reactive_inflight = False

            # Reactive: record when we received this result; submit next after reactive_step
            if self._reactive_active():
                self._reactive_last_submit_step = self.step_count

        # Block only when we have no plan and are waiting for IK/MotionGen
        # In reactive mode, we keep walking plan_actions even when next MotionGen is in flight
        if self.plan_actions is None:
            if (
                self.inflight_ik_future is not None
                or self.inflight_motiongen_future is not None
            ):
                self.current_state = "planning"
                return self._nan_action()
            self.current_state = "planning"
            return self._nan_action()

        # Reactive: submit next MotionGen every reactive_step (free base only)
        if (
            self._reactive_active()
            and self.inflight_motiongen_future is None
            and self._should_use_motiongen(self.current_arm_targets)
            and self.pending_target is None
            and self.step_count - self._reactive_last_submit_step >= self.reactive_step
        ):
            self._reactive_last_submit_step = self.step_count
            self._submit_plan(
                self.current_arm_targets,
                lock_base=False,
                reactive_submit=True,
            )

        # ---- 3) Stream action trajectory (position-based advancement) ----
        # Advance to next waypoint only when arrived:
        #   - fix_base (locked): only check EEF arrival
        #   - free_base: check both base and EEF arrival
        if self.plan_step < self.plan_actions.shape[0]:
            action_pose = self.plan_actions[self.plan_step]
            # Check if arrived at current waypoint or exceeded max steps
            arrived = self._check_waypoint_arrived(action_pose)
            self._waypoint_step_counter += 1
            if arrived or self._waypoint_step_counter >= self._waypoint_max_steps:
                self.plan_step += 1
                self._waypoint_step_counter = 0
                if self.plan_step < self.plan_actions.shape[0]:
                    action_pose = self.plan_actions[self.plan_step]
        else:
            # Repeat the final action once the plan is exhausted
            action_pose = self.plan_actions[-1]

        if action_pose.ndim != 1:
            action_pose = action_pose.view(-1)

        _lf = (
            self._LOCK_FLAG_LOCK_SKIP if self._use_locked_base else self._LOCK_FLAG_NAV
        )
        action = self._build_full_action(action_pose, lock_flag_override=_lf)

        self.step_count += 1  # Count only after a plan is ready (no waiting time)
        self.last_action = action
        self.current_state = "running"
        self.current_action = {"MobileMoveL": action}
        return action

    def refresh(self, action: torch.Tensor):
        # Support both 7D and 8D actions
        robot_id, hand_id, target_action = self._parse_action(action)
        robot_changed = self._set_robot_by_id(robot_id, hand_id)
        (
            new_base_target,
            new_arm_targets,
            new_eef_target,
        ) = self._parse_target(target_action)

        # New target still placeholder: end planner quietly if we were already on
        # placeholder (e.g. user changed grasp kind) so manager can swap planner type.
        if self._is_placeholder_target(new_arm_targets):
            # During the atomic skill's init phase ``refresh`` is called
            # every tick with an all-NaN target. Don't use ``silent_finish``
            # / ``state="finished"`` here — that would make the GP manager
            # destroy the planner each tick, recreate it next tick, and the
            # cycle would re-enter this branch indefinitely (log spam +
            # wasted plan work). Instead keep the planner alive and idle;
            # the placeholder→real transition below will reset state cleanly
            # when the real target finally arrives.
            return

        old_is_placeholder = self._is_placeholder_target(self.current_arm_targets)

        # Determine if target actually changed
        if old_is_placeholder:
            # Transitioning from placeholder to real target
            target_changed = True
            # Init-phase placeholder→placeholder refreshes can have
            # latched ``_pending_silent_finish=True`` + ``current_state="finished"``
            # (see the placeholder branch above). Clear them now that a
            # real target is coming in, otherwise the very first
            # ``update()`` after the swap returns ``truncated=5`` and the
            # atomic skill stays stuck on this phase for another full
            # MoveL run.
            if self._pending_silent_finish or self.current_state == "finished":
                self._pending_silent_finish = False
                self.current_state = "ready"
        else:
            # Both are real targets, compare normally
            target_changed = not self._targets_close(
                new_arm_targets, self.current_arm_targets
            )

        self.current_base_target = new_base_target
        self.current_arm_targets = new_arm_targets
        self.current_eef_target = new_eef_target
        _lf = (
            self._LOCK_FLAG_LOCK_SKIP if self._use_locked_base else self._LOCK_FLAG_NAV
        )
        self.current_target = self._build_full_action(
            new_arm_targets.view(-1), lock_flag_override=_lf
        )
        self.current_command = ["MobileMoveL", self.robot_name, self.current_target]
        if target_changed or robot_changed:
            self.pending_target = new_arm_targets
            self.plan_actions = None
            self.plan_step = 0
            self.last_action = None
            if (
                self.inflight_motiongen_future is None
                and self.inflight_ik_future is None
            ):
                if self._mode_skip_ik():
                    self._set_use_locked_from_mode()
                    self._begin_plan_after_mode_resolution(self.current_arm_targets)
                    self.pending_target = None
                else:
                    self._use_locked_base = False
                    self._submit_ik(new_arm_targets)
                    self.pending_target = None
            else:
                if not self._mode_skip_ik():
                    self._use_locked_base = False

    def get_done(self) -> bool:
        if self.current_arm_targets is None:
            return False

        # Placeholder target (all NaN during init): never consider done
        if self._is_placeholder_target(self.current_arm_targets):
            return False

        # If we have a plan, wait until all planned actions are consumed.
        if (
            self.plan_actions is not None
            and self.plan_step < self.plan_actions.shape[0]
        ):
            return False

        if self.step_count < self._min_steps_before_done:
            return False

        timeout_steps = int(getattr(self.config, "timeout_steps", 100))
        if self.step_count >= timeout_steps:
            # Log EEF distance at timeout so we can see if premature
            target_poses = self.current_arm_targets
            cur_pos, cur_quat = self._select_current_eef_for_target(target_poses)
            pair_count = min(cur_pos.shape[0], target_poses.shape[0])
            for eef_idx in range(pair_count):
                if torch.isnan(target_poses[eef_idx, :3]).any():
                    continue
                pos_diff = torch.norm(
                    cur_pos[eef_idx] - target_poses[eef_idx, :3]
                ).item()
                print(
                    f"[MobileMoveL] Env {self.env_id}: TIMEOUT done at step_count={self.step_count}, "
                    f"eef_pos_diff={pos_diff:.4f}, threshold={float(self.config.translation_threshold):.4f}"
                )
            return True

        translation_threshold = float(self.config.translation_threshold)
        rotation_threshold = float(self.config.rotation_threshold)

        # Check all EEFs against their targets (supports dual-arm)
        target_poses = self.current_arm_targets  # [eef_num, 7]
        cur_pos, cur_quat = self._select_current_eef_for_target(target_poses)
        pair_count = min(cur_pos.shape[0], target_poses.shape[0])

        for eef_idx in range(pair_count):
            if torch.isnan(target_poses[eef_idx, :3]).any():
                continue
            pos_diff = torch.norm(cur_pos[eef_idx] - target_poses[eef_idx, :3]).item()
            if pos_diff >= translation_threshold:
                return False
            # Quaternion comparison: q and -q represent the same rotation
            quat_diff_1 = torch.norm(cur_quat[eef_idx] - target_poses[eef_idx, 3:7])
            quat_diff_2 = torch.norm(cur_quat[eef_idx] + target_poses[eef_idx, 3:7])
            quat_diff = torch.min(quat_diff_1, quat_diff_2).item()
            if quat_diff >= rotation_threshold:
                return False

        return True

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        if self.current_state == "failed":
            self.current_state = "failed: global planner failed to plan"
            return {
                "type": "MobileMoveL",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
            }
        elif self.current_state == "finished" or self.get_done():
            self.current_state = "finished"
            if self._pending_silent_finish:
                self._pending_silent_finish = False
                print(
                    f"[MobileMoveL] env_id={self.env_id} silent_finish path "
                    f"→ truncated=5 (atomic skill won't advance phase). "
                    f"step_count={self.step_count} "
                    f"current_arm_targets[0]={self.current_arm_targets[0].tolist() if self.current_arm_targets is not None else None}"
                )
                return {
                    "type": "MobileMoveL",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 5,
                }
            return {
                "type": "MobileMoveL",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 0,
            }
        elif info["env_info"][2][self.env_id]:
            self.current_state = "truncated: env terminated first"
            return {
                "type": "MobileMoveL",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
            }
        elif info["env_info"][3][self.env_id]:
            self.current_state = "truncated: env truncated first"
            return {
                "type": "MobileMoveL",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
            }
        else:
            self.current_state = "running"
            return {
                "type": "MobileMoveL",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }
