import traceback
from typing import Any, Callable, Dict, List, Optional
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


class RetractMoveL(GlobalPlanner):
    """
    Global Planner with retract behavior: stand up -> move -> squat down -> arm motion.

    Header: ``((robot_id, hand_id, planner_mode), target_tensor)``. For **locked base**, ``planner_mode``
    behaves like MoveL (table below). For **free base**, planning always uses MotionGen; the
    "no" rows in the table do not apply.

    ``planner_mode`` when locked base (``_should_use_motiongen``; same distance thresholds as MoveL)::

        +--------+------------------------------+
        |  mode  | MotionGen (locked base)      |
        +--------+------------------------------+
        |   0    | No; single-step target       |
        |   1    | Always MotionGen             |
        | other  | Same as -1: MG if dist > thr |
        |  -1    | Auto: MG if EEF dist > thr   |
        +--------+------------------------------+

    This is not MobileMoveL's full IK/base/MotionGen mode table; see MobileMoveL docs for the
    analogous reference.

    This planner differs from MobileMoveL in that:
    1. First checks IK with fixed base
    2. If IK succeeds: use fixed-base MotionGen directly
    3. If IK fails: use free-base MotionGen
    4. For free-base MotionGen results:
       - If base movement is small: execute trajectory directly
       - If base movement is large: use move_strategy to create multi-segment base trajectory
         (stand up -> move horizontally -> squat down), then do fixed-base MotionGen for arms
    5. Execution is position-based (wait for arrival) rather than fixed-interval
    """

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
        # See class docstring (locked-base branch; same subset as MoveL, not Mobile full table).
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
        self.step_count = 0
        self.debug = config.get("debug", True)
        self.debug_viz = config.get("debug_viz", True)
        self.debug_sphere = config.get("debug_sphere", False)
        self._viz_spheres = None

        # For dual mode: tracks whether locked base IK succeeded
        self._use_locked_base = False

        # Move strategy related (populated from PlannerManager)
        self._move_strategy: Optional[Callable] = None
        self._move_strategy_distance_threshold: float = config.get(
            "move_strategy_distance_threshold", 0.1
        )

        # Execution phases
        # "idle": waiting for target
        # "ik_check": waiting for IK result
        # "base_moving": executing move_strategy unified trajectory (base+EEF+lock per row)
        # "arm_planning": waiting for arm MotionGen result
        # "arm_moving": executing MotionGen arm trajectory (no move_strategy)
        # "hold": move_strategy trajectory finished; repeat last full waypoint until done
        self._phase = "idle"

        # Base movement trajectory (from move_strategy)
        self._base_trajectory: Optional[torch.Tensor] = None
        self._base_step = 0
        # Last row [D+1] from move_strategy (motiongen row + lock_flag); used in hold
        self._hold_motiongen_row: Optional[torch.Tensor] = None

        # Arm trajectory (from MotionGen)
        self._arm_trajectory: Optional[torch.Tensor] = None
        self._arm_step = 0

        # Position thresholds for arrival detection (same as MobileMoveL)
        self._base_pos_threshold = config.get("base_pos_threshold", 0.15)
        self._base_rot_threshold = config.get("base_rot_threshold", 0.1)
        self._eef_pos_threshold = config.get("eef_pos_threshold", 0.03)
        self._eef_rot_threshold = config.get("eef_rot_threshold", 0.1)
        # Max steps to wait at a single waypoint before forcing advance
        self._base_waypoint_max_steps = config.get("base_waypoint_max_steps", 20)
        self._arm_waypoint_max_steps = config.get("arm_waypoint_max_steps", 5)
        self._base_waypoint_counter = 0
        self._arm_waypoint_counter = 0
        self._pending_silent_finish = False

        super().__init__(config, env, env_id, logger)

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

        # Single server per robot (MERGE_LEFT_RIGHT §1–§8).
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

        # Get move_strategy from PlannerManager
        self._move_strategy, self._move_strategy_distance_threshold = (
            planner_manager.get_move_strategy(self.robot_name)
        )
        return True

    def _parse_action(self, action: torch.Tensor):
        """Parse action format (third header value: ``planner_mode``). See ``parse_planner_header``."""
        robot_id, hand_id, mode, target = GlobalPlanner.parse_planner_header(
            action,
            default_robot_id=self.robot_id if self.robot_id >= 0 else 0,
            default_hand_id=self.hand_id,
        )
        self.planner_mode = mode
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
            # Skip inactive EEFs (nan pos): inflight uses nan for inactive arm, pending may use zeros
            if (
                torch.isnan(target_a[eef_idx, :3]).any()
                or torch.isnan(target_b[eef_idx, :3]).any()
            ):
                continue
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

    def _submit_plan_blocked_in_hold(self) -> bool:
        """Returns True (caller should skip) if we're in hold."""
        return self._phase == "hold"

    def _submit_plan(self, target_pose_all: torch.Tensor, lock_base: bool = None):
        """Submit a MotionGen planning request.

        Same hard guard as ``_submit_ik``: once we hit the ``hold`` phase
        (move_strategy trajectory done), there is no second-stage planning;
        every subsequent control tick just repeats the last waypoint.
        """
        if self._phase == "hold":
            return
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
                "RetractMoveL: Submit MotionGen",
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

    def _submit_ik(self, target_pose: torch.Tensor):
        """Submit IK check with lock_base=True (fixed base).

        Hard guard: once ``_phase == "hold"`` (move_strategy trajectory
        finished), this is a no-op. Some control paths (IK/MG result
        handlers, refresh) historically tried to submit a "follow-up" IK
        after move_strategy to land the wrist precisely; we want to treat
        ``hold`` as terminal — repeat the last waypoint until the upstream
        skill switches planner. Without this guard the box-pose drift in
        LocoBox keeps re-triggering full IK→MG→move_strategy cycles.
        """
        if self._phase == "hold":
            return
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
                "RetractMoveL: Submit IK",
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
            input_cfg = getattr(self.config, "input", None)
            base_dim = int(getattr(input_cfg, "base_dim", 0)) if input_cfg else 0
            arm_dim = int(getattr(input_cfg, "arm_dim", 0)) if input_cfg else 0
            eef_dim = int(getattr(input_cfg, "eef_dim", 0)) if input_cfg else 0
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

    def _expand_arm_action(self, arm_action: torch.Tensor) -> torch.Tensor:
        """Expand a per-arm action to full arm_dim, filling the other arm with NaN.

        For hand_id 0 (right): action goes into first 7 dims.
        For hand_id 1 (left):  action goes into last 7 dims.
        For hand_id -1 (both): returned as-is.
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
    # ``_expand_arm_action(target.view(-1)).view(1, -1)`` for hand_id NaN.

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

    def _visualize_base_trajectory(
        self, trajectory: torch.Tensor, clear_existing: bool = True
    ) -> None:
        """Visualize the base trajectory from move_strategy (each row is [D+1] with lock_flag)."""
        origin_cpu = self._get_env_origin().detach().cpu()
        points: list[list[float]] = []
        for i in range(trajectory.shape[0]):
            wp = trajectory[i].detach().cpu()
            pos = wp[:3]
            if torch.isnan(pos).any():
                continue
            points.append((pos + origin_cpu).tolist())
        draw_waypoints(points, clear_existing=clear_existing)

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

    def _check_base_waypoint_arrived(
        self, waypoint: torch.Tensor, height_only: bool = False
    ) -> bool:
        """Check if base has arrived at waypoint.

        Args:
            waypoint: Waypoint tensor, first 3 dims are (x, y, z).
            height_only: If True, only check z (height) arrival, ignore x/y.
        """
        target_base_pos = waypoint[:3]
        target_base_ori = waypoint[3:7]
        if torch.isnan(target_base_pos).any():
            return True  # Skip NaN waypoints
        cur_base_pos, cur_base_ori = self._get_current_base_pose()
        if height_only:
            height_diff = abs(cur_base_pos[2].item() - target_base_pos[2].item())
            return height_diff < self._base_pos_threshold
        base_pos_diff = torch.linalg.norm(cur_base_pos - target_base_pos).item()
        return base_pos_diff < self._base_pos_threshold

    def _check_arm_waypoint_arrived(self, waypoint: torch.Tensor) -> bool:
        """Check if EEF has arrived at waypoint (position + rotation).

        Same logic as MobileMoveL locked-base check.
        Extracts EEF pose from waypoint (skipping base dims if present).
        Supports multiple EEFs for dual-arm mode.
        """
        base_dim, arm_dim, eef_dim = self._get_action_dims()
        motiongen_base_dim = base_dim - 1 if base_dim > 0 else 0

        wp = waypoint.view(-1)
        # Determine if waypoint contains base data
        has_base_data = (
            motiongen_base_dim > 0 and wp.shape[0] == motiongen_base_dim + arm_dim
        )
        eef_offset = motiongen_base_dim if has_base_data else 0
        arm_data = wp[eef_offset:]

        # Parse arm data into per-EEF targets
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

    def _compute_base_distance(self, trajectory: torch.Tensor) -> float:
        """Compute base movement distance from trajectory's first 7 dims."""
        if trajectory.shape[0] < 2:
            return 0.0
        start_pos = trajectory[0, :3]
        end_pos = trajectory[-1, :3]
        return torch.linalg.norm(end_pos - start_pos).item()

    def _is_placeholder_target(self, arm_targets: torch.Tensor) -> bool:
        """Check if arm targets are all NaN (placeholder during init phase)."""
        return torch.isnan(arm_targets).all()

    def _should_use_motiongen(self, target_pose: torch.Tensor) -> bool:
        # Free base always uses motiongen
        if not self._use_locked_base:
            return True
        # Locked base: same as class doc / MoveL (not MobileMoveL full mode table).
        if self.planner_mode == 0:
            return False
        if self.planner_mode == 1:
            return True
        # Auto: decide based on max distance across all EEFs.
        if target_pose.ndim == 1:
            target_pose = target_pose.view(-1, 7)
        cur_pos, _ = self._select_current_eef_for_target(target_pose)
        pair_count = min(cur_pos.shape[0], target_pose.shape[0])
        threshold = float(
            getattr(
                self.config,
                "motiongen_distance_threshold",
                self.config.translation_threshold,
            )
        )
        max_diff = 0.0
        for eef_idx in range(pair_count):
            pos_diff = torch.linalg.norm(
                cur_pos[eef_idx] - target_pose[eef_idx, :3]
            ).item()
            max_diff = max(max_diff, pos_diff)
        return max_diff > threshold

    def reset(self, action: torch.Tensor):
        robot_id, hand_id, target_action = self._parse_action(action)
        self._set_robot_by_id(robot_id, hand_id)
        (
            self.current_base_target,
            self.current_arm_targets,
            self.current_eef_target,
        ) = self._parse_target(target_action)
        self.arm_eef_num = int(self.current_arm_targets.shape[0])
        self.step_count = 0
        self._base_waypoint_counter = 0
        self._arm_waypoint_counter = 0

        self.current_target = self._build_full_action(self.current_arm_targets.view(-1))
        self.current_command = ["RetractMoveL", self.robot_name, self.current_target]
        self.current_action = None
        self.inflight_motiongen_future = None
        self.inflight_motiongen_target = None
        self.inflight_ik_future = None
        self.inflight_ik_target = None
        self.pending_target = None
        self._use_locked_base = False
        self._base_trajectory = None
        self._base_step = 0
        self._hold_motiongen_row = None
        self._arm_trajectory = None
        self._arm_step = 0
        self.plan_actions = None
        self.plan_step = 0
        self.last_action = None
        self._pending_silent_finish = False

        # Skip IK submission if target is placeholder (all NaN during init)
        if self._is_placeholder_target(self.current_arm_targets):
            self._phase = "idle"
            self.current_state = "waiting"
        else:
            self._phase = "ik_check"
            self.current_state = "ready"
            self._submit_ik(self.current_arm_targets)

    def step(self) -> torch.Tensor:
        """One control tick: drain async IK/MotionGen, then run execution by ``self._phase``.

        **Phases (when each applies)**

        * ``idle`` — Set in ``reset`` when arm targets are placeholder (all NaN); not used in the
          main execution path below once a real target exists.

        * ``ik_check`` — After ``reset``/``refresh`` with a real target; we submitted IK and wait
          for ``inflight_ik_future``. Not a return branch here; transitions when IK handler runs.

        * ``arm_planning`` — IK decided we need MotionGen; ``_submit_plan`` was called. Wait for
          ``inflight_motiongen_future`` (``current_state`` may show ``planning``).

        * ``base_moving`` — Only when: IK **failed** (free-base MotionGen), plan **succeeds**, base
          displacement exceeds ``_move_strategy_distance_threshold``, and ``_move_strategy`` is set.
          Plays ``move_strategy`` output ``[M, D+1]`` (pelvis arrival; per-waypoint ``lock_flag``).

        * ``hold`` — Only after ``base_moving`` finishes the full move_strategy trajectory. Repeats
          the last waypoint row (``_hold_motiongen_row``) until ``get_done()``; no second arm-only
          rollout.

        * ``arm_moving`` — Any of: IK **succeeded** and ``_should_use_motiongen`` is **False** (single
          IK row); OR MotionGen **succeeded** with **locked** base; OR free-base MotionGen but
          displacement small / no ``move_strategy``; OR MotionGen **failed** (fallback single row).
          Advances by **EEF** arrival on ``_arm_trajectory``.

        **Planning wait** (not a stored ``_phase``): while ``inflight_ik_future`` or
        ``inflight_motiongen_future`` is pending, return ``_nan_action()`` regardless of label above.
        """
        if self.current_target is None or self.current_arm_targets is None:
            raise RuntimeError("Current Target Is Not Set, Please Call Reset First")

        # If target is placeholder (all NaN during init), return nan_action directly
        if self._is_placeholder_target(self.current_arm_targets):
            self.current_state = "waiting"
            return self._nan_action()

        self.step_count += 1  # Count every step for timeout (including planning)
        robot_state = self._get_robot_state()
        joint_pos = robot_state["joint_pos"].clone()
        if joint_pos is not None and self.debug_sphere:
            self._visualize_robot_spheres(joint_pos)

        # ---- Async: IK result → ``arm_planning`` (submit MotionGen) or ``arm_moving`` (skip MG) ----
        if self.inflight_ik_future is not None and self.inflight_ik_future.done():
            inflight_target = self.inflight_ik_target
            try:
                success, _, env_ids = self.inflight_ik_future.result()
            except Exception as e:
                if self.debug:
                    print(
                        "RetractMoveL: IK exception",
                        "env_id=",
                        self.env_id,
                        e,
                    )
                success = []
            self.inflight_ik_future = None
            self.inflight_ik_target = None

            # Handle pending target
            if self.pending_target is not None and not self._targets_close(
                self.pending_target, inflight_target
            ):
                self._submit_ik(self.pending_target)
                self.pending_target = None
                self._phase = "ik_check"
                self.current_state = "planning"
                return self._nan_action()
            if self.pending_target is not None:
                self.pending_target = None

            target = (
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
                        "RetractMoveL: IK SUCCESS -> locked base MotionGen",
                        "env_id=",
                        self.env_id,
                    )
                if self._should_use_motiongen(target):
                    self._submit_plan(self.current_arm_targets, lock_base=True)
                    self._phase = "arm_planning"
                    self.current_state = "planning"
                    return self._nan_action()
                # Close enough, direct action
                self._arm_trajectory = target.view(-1).unsqueeze(0)
                self._arm_step = 0
                self._phase = "arm_moving"
            else:
                # IK failed -> use free base MotionGen
                self._use_locked_base = False
                if self.debug:
                    print(
                        "RetractMoveL: IK FAILED -> free base MotionGen",
                        "env_id=",
                        self.env_id,
                    )
                self._submit_plan(self.current_arm_targets, lock_base=False)
                self._phase = "arm_planning"
                self.current_state = "planning"
                return self._nan_action()

        # ---- Async: MotionGen result (may set base_moving, arm_moving, or arm_moving fallback) ----
        if (
            self.inflight_motiongen_future is not None
            and self.inflight_motiongen_future.done()
        ):
            inflight_target = self.inflight_motiongen_target
            try:
                actions, success, env_ids = self.inflight_motiongen_future.result()
            except Exception as e:
                print(f"RetractMoveL: MotionGen failed: {e}\n{traceback.format_exc()}")
                actions, success = [], []
            self.inflight_motiongen_future = None
            self.inflight_motiongen_target = None

            # Handle pending target
            if self.pending_target is not None and not self._targets_close(
                self.pending_target, inflight_target
            ):
                self._submit_ik(self.pending_target)
                self.pending_target = None
                self._phase = "ik_check"
                self.current_state = "planning"
                return self._nan_action()
            if self.pending_target is not None:
                self.pending_target = None

            if len(actions) > 0 and len(success) > 0 and success[0]:
                plan = actions[0]
                if self.debug:
                    print(
                        "RetractMoveL: MotionGen success",
                        "env_id=",
                        self.env_id,
                        "locked=",
                        self._use_locked_base,
                    )
                if self.debug_viz:
                    self._visualize_waypoints(actions)
                if isinstance(plan, torch.Tensor):
                    trajectory = plan.to(self.env.device, dtype=torch.float32)
                else:
                    trajectory = torch.as_tensor(
                        plan, device=self.env.device, dtype=torch.float32
                    )

                # Check if this was a free-base plan with significant base movement
                if not self._use_locked_base:
                    base_distance = self._compute_base_distance(trajectory)
                    if (
                        base_distance > self._move_strategy_distance_threshold
                        and self._move_strategy is not None
                    ):
                        # Use move_strategy to create base trajectory
                        if self.debug:
                            print(
                                f"RetractMoveL: Base distance {base_distance:.3f} > threshold, using move_strategy"
                            )
                        robot_state = self._get_robot_state()
                        # move_strategy returns [M, D+1] with lock_flag as last dim
                        self._base_trajectory = self._move_strategy(
                            trajectory,
                            robot_state,
                            hand_id=self.hand_id,
                        )
                        if self.debug_viz:
                            self._visualize_base_trajectory(
                                self._base_trajectory, clear_existing=False
                            )
                        self._base_step = 0
                        self._phase = "base_moving"
                        self.current_state = "running"
                    else:
                        # Small base movement, execute directly
                        self._arm_trajectory = trajectory
                        self._arm_step = 0
                        self._phase = "arm_moving"
                else:
                    # Locked base plan, execute arm trajectory directly
                    self._arm_trajectory = trajectory
                    self._arm_step = 0
                    self._phase = "arm_moving"
            else:
                if self.debug:
                    print(
                        "RetractMoveL: MotionGen failed",
                        "env_id=",
                        self.env_id,
                    )
                fallback = (
                    inflight_target.view(-1)
                    if inflight_target is not None
                    else self.current_arm_targets.view(-1)
                )
                self._arm_trajectory = fallback.unsqueeze(0)
                self._arm_step = 0
                self._phase = "arm_moving"

        # ---- Planning wait: IK or MotionGen still in flight (arm_planning / ik_check continuation) ----
        if (
            self.inflight_ik_future is not None
            or self.inflight_motiongen_future is not None
        ):
            self.current_state = "planning"
            return self._nan_action()

        # ---- ``base_moving``: unified move_strategy trajectory [M, D+1] ----
        if self._phase == "base_moving" and self._base_trajectory is not None:
            if self._base_step < self._base_trajectory.shape[0]:
                action_pose = self._base_trajectory[self._base_step]
                # lock_flag -1 means lock_skip (height change only) -> height-only arrival if wired
                lock_flag = action_pose[-1].item()
                # Advance waypoints using **pelvis pose only** (first 7 dims); EEF in the row
                # is still sent in the action but does not affect arrival (see g1_move_strategy).
                arrived = self._check_base_waypoint_arrived(action_pose[:-1][:7])
                self._base_waypoint_counter += 1

                # Non-last waypoints: advance on arrival or timeout
                if (
                    arrived
                    or self._base_waypoint_counter >= self._base_waypoint_max_steps
                ):
                    self._base_step += 1
                    self._base_waypoint_counter = 0
                    if self._base_step < self._base_trajectory.shape[0]:
                        action_pose = self._base_trajectory[self._base_step]

                if self._base_step < self._base_trajectory.shape[0]:
                    # lock_flag is the last dim of the waypoint
                    lock_flag = action_pose[-1].item()
                    # G1: lock_flag 1 = turning (in-place rotation padding)
                    action = self._build_full_action(
                        action_pose[:-1], lock_flag_override=lock_flag
                    )
                    self.last_action = action
                    self.current_state = "running"
                    self.current_action = {"RetractMoveL": action}
                    return action

            # Unified move_strategy trajectory is complete (terminal EEF == MotionGen last frame).
            # No separate arm_moving phase — hold repeating last full waypoint until get_done().
            if self.debug:
                print(
                    "RetractMoveL: move_strategy trajectory complete -> hold (no arm-only phase)"
                )
            if self._base_trajectory is not None and self._base_trajectory.shape[0] > 0:
                self._hold_motiongen_row = self._base_trajectory[-1].clone()
            self._base_trajectory = None
            self._base_step = 0
            self._base_waypoint_counter = 0
            self._use_locked_base = True
            self._arm_trajectory = None
            self._arm_step = 0
            self._phase = "hold"

        # ---- ``hold``: last move_strategy waypoint until task done (see ``step`` docstring) ----
        if self._phase == "hold":
            self.current_state = "running"
            if self._hold_motiongen_row is not None:
                row = self._hold_motiongen_row
                action = self._build_full_action(
                    row[:-1], lock_flag_override=row[-1].item()
                )
            else:
                action = self._build_full_action(self.current_arm_targets.view(-1))
            self.last_action = action
            self.current_action = {"RetractMoveL": action}
            return action

        # ---- ``arm_moving``: MotionGen/IK trajectory [T, D] or single row; EEF arrival ----
        if self._phase == "arm_moving" and self._arm_trajectory is not None:
            if self._arm_step < self._arm_trajectory.shape[0]:
                action_pose = self._arm_trajectory[self._arm_step]
                # Check if arrived at current waypoint (EEF pos + rot) or timeout
                arrived = self._check_arm_waypoint_arrived(action_pose)
                self._arm_waypoint_counter += 1
                if (
                    arrived
                    or self._arm_waypoint_counter >= self._arm_waypoint_max_steps
                ):
                    self._arm_step += 1
                    self._arm_waypoint_counter = 0
                    if self._arm_step < self._arm_trajectory.shape[0]:
                        action_pose = self._arm_trajectory[self._arm_step]

            if self._arm_step < self._arm_trajectory.shape[0]:
                action = self._build_full_action(action_pose)
            else:
                # Trajectory exhausted, repeat final action
                action = self._build_full_action(self._arm_trajectory[-1])

            self.last_action = action
            self.current_state = "running"
            self.current_action = {"RetractMoveL": action}
            return action

        # Unhandled phase (e.g. idle/ik_check with no trajectory yet) or missing trajectory
        self.current_state = "planning"
        return self._nan_action()

    def refresh(self, action: torch.Tensor):
        robot_id, hand_id, target_action = self._parse_action(action)
        robot_changed = self._set_robot_by_id(robot_id, hand_id)
        (
            new_base_target,
            new_arm_targets,
            new_eef_target,
        ) = self._parse_target(target_action)

        if self._is_placeholder_target(new_arm_targets):
            if self._is_placeholder_target(self.current_arm_targets):
                self._pending_silent_finish = True
                self.current_state = "finished"
            return

        old_is_placeholder = self._is_placeholder_target(self.current_arm_targets)
        # Placeholder → real target = upstream skill is **activating** us
        # for the first real planning round. Clear any "silent-finish" flag
        # accumulated during the placeholder window (e.g. LocoBox's long
        # async paired-IK selection sends placeholder every tick, which
        # otherwise leaves ``_pending_silent_finish=True`` so the eventual
        # finish reports ``truncated=5`` and the upstream skill never
        # advances phase — see LocoBox bug 2026-04-28).
        if old_is_placeholder:
            self._pending_silent_finish = False

        # Determine if target actually changed
        if old_is_placeholder:
            # Transitioning from placeholder to real target
            target_changed = True
        else:
            # Both are real targets, compare normally
            target_changed = not self._targets_close(
                new_arm_targets, self.current_arm_targets
            )

        self.current_base_target = new_base_target
        self.current_arm_targets = new_arm_targets
        self.current_eef_target = new_eef_target
        self.current_target = self._build_full_action(new_arm_targets.view(-1))
        self.current_command = ["RetractMoveL", self.robot_name, self.current_target]

        if target_changed or robot_changed:
            # Once we're in ``hold`` (move_strategy trajectory finished),
            # a subsequent target nudge from the upstream skill must NOT
            # trigger another IK submit — there's no longer an "arm only"
            # phase to invoke; we just keep repeating the last waypoint
            # until ``get_done()`` reports finished. Update the stored
            # target so observers see the latest value, but stay in hold.
            if self._phase == "hold":
                self.pending_target = None
                return
            # When IK is inflight: if new target is close to inflight target, don't set
            # pending_target to avoid repeated IK resubmit loop (IK result -> resubmit -> repeat).
            if (
                self.inflight_ik_future is not None
                and self.inflight_ik_target is not None
                and self._targets_close(new_arm_targets, self.inflight_ik_target)
            ):
                self.pending_target = None
                return
            # Target changed, restart with IK check
            self.pending_target = new_arm_targets
            self._base_trajectory = None
            self._base_step = 0
            self._base_waypoint_counter = 0
            self._hold_motiongen_row = None
            self._arm_trajectory = None
            self._arm_step = 0
            self._arm_waypoint_counter = 0
            self.plan_actions = None
            self.plan_step = 0
            self.last_action = None
            self._use_locked_base = False
            self._phase = "ik_check"
            self.current_state = "ready"

            if (
                self.inflight_motiongen_future is None
                and self.inflight_ik_future is None
            ):
                self._submit_ik(new_arm_targets)

    def get_done(self) -> bool:
        if self.current_arm_targets is None:
            return False

        # Placeholder target (all NaN during init): never consider done
        if self._is_placeholder_target(self.current_arm_targets):
            return False

        # Async planning still in flight: do not report done (otherwise timeout / empty EEF
        # loop can mark finished while MotionGen is pending and step() returns nan_action).
        if (
            self.inflight_ik_future is not None
            or self.inflight_motiongen_future is not None
        ):
            return False
        if self._phase == "arm_planning":
            return False

        # Check if all trajectories are consumed
        if self._phase == "base_moving" and self._base_trajectory is not None:
            if self._base_step < self._base_trajectory.shape[0]:
                return False

        if self._phase == "arm_moving" and self._arm_trajectory is not None:
            if self._arm_step < self._arm_trajectory.shape[0]:
                return False

        timeout_steps = int(getattr(self.config, "timeout_steps", 120))
        if self.step_count >= timeout_steps:
            return True

        translation_threshold = float(self.config.translation_threshold)
        rotation_threshold = float(self.config.rotation_threshold)

        # Check all EEFs against their targets (supports dual-arm)
        target_poses = self.current_arm_targets  # [eef_num, 7]
        cur_pos, cur_quat = self._select_current_eef_for_target(target_poses)
        pair_count = min(cur_pos.shape[0], target_poses.shape[0])
        if pair_count == 0:
            return False

        for eef_idx in range(pair_count):
            # Skip inactive arms (single-arm commands often pad the other EEF with NaN).
            if torch.isnan(target_poses[eef_idx, :3]).any():
                continue
            pos_diff = torch.norm(cur_pos[eef_idx] - target_poses[eef_idx, :3]).item()
            if pos_diff >= translation_threshold:
                return False
            # Quaternion comparison: q and -q represent the same rotation
            quat_diff_1 = torch.norm(cur_quat[eef_idx] - target_poses[eef_idx, 3:7])
            quat_diff_2 = torch.norm(cur_quat[eef_idx] + target_poses[eef_idx, 3:7])
            quat_diff = torch.min(quat_diff_1, quat_diff_2).item()
            if self.debug:
                print(
                    f"RetractMoveL get_done: quat_diff={quat_diff}, "
                    f"rotation_threshold={rotation_threshold}"
                )
            if quat_diff >= rotation_threshold:
                return False

        return True

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        if self.current_state == "failed":
            self.current_state = "failed: global planner failed to plan"
            return {
                "type": "RetractMoveL",
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
                return {
                    "type": "RetractMoveL",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": self.current_state,
                    "truncated": 5,
                }
            return {
                "type": "RetractMoveL",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 0,
            }
        elif info["env_info"][2][self.env_id]:
            self.current_state = "truncated: env terminated first"
            return {
                "type": "RetractMoveL",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
            }
        elif info["env_info"][3][self.env_id]:
            self.current_state = "truncated: env truncated first"
            return {
                "type": "RetractMoveL",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
            }
        else:
            self.current_state = "running"
            return {
                "type": "RetractMoveL",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }
