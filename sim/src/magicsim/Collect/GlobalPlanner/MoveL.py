from typing import Any, Dict, List
import numpy as np
import torch
from magicsim.Collect.GlobalPlanner.GlobalPlanner import GlobalPlanner
from magicsim.Env.Utils.file import Logger
from magicsim.Env.Utils.viz import draw_waypoints
from magicsim.Env.Planner.Services.IKServer import IKPlanRequest
from magicsim.Env.Planner.Services.MotionGenServer import MotionGenPlanRequest
from magicsim.Env.Planner.Utils import quat_angle_between
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig
from isaacsim.core.api.objects.sphere import VisualSphere


class MoveL(GlobalPlanner):
    """
    Global Planner for all tasks.

    Header: ``((robot_id, hand_id, planner_mode), target_tensor)`` (see ``_parse_action``).
    Whether IK runs is controlled only by config ``ik_check``, not by ``planner_mode``.

    ``planner_mode`` affects **MotionGen + interpolation** (via ``_should_use_motiongen``;
    distance gate: ``motiongen_distance_threshold`` / ``translation_threshold``)::

        +--------+------------------------------------------------------+
        |  mode  | Behavior                                             |
        +--------+------------------------------------------------------+
        |   0    | No MotionGen; single-step snap arm pose              |
        |   1    | Always MotionGen                                     |
        |   2    | Linear-interp from current eef pose to target over   |
        |        | ``config.interp_steps`` calls of step(); bypasses    |
        |        | motiongen / IK servers                               |
        | other  | Same as -1: MG if dist > thr                         |
        |  -1    | Auto: MG if EEF distance > thr                       |
        +--------+------------------------------------------------------+

    Similar in spirit to MobileMoveL's **locked base** branch, but this planner does **not**
    interpret base mode, and does not implement the full Mobile table (e.g. ``0–3 / -2``).
    """

    # planner_mode value that activates linear-interp (see table above).
    INTERP_MODE: int = 2

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
        # See class docstring ``planner_mode`` table (subset vs MobileMoveL full table).
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
        self.debug = config.get("debug", False)
        self.debug_viz = config.get("debug_viz", True)
        self.debug_sphere = config.get("debug_sphere", False)
        self._viz_spheres = None
        self.ik_check = config.get("ik_check", False)
        # Linear-interpolation hyperparameter (config-time, not runtime).
        # When ``interp_steps > 0`` MoveL lerps from the eef pose at the
        # moment ``reset()`` is called to the commanded target over this
        # many ``step()`` invocations, bypassing motiongen / IK servers.
        # Same value applies to every phase that issues a MoveL command;
        # callers don't override per-call. Set to 0 in yaml to disable.
        self.interp_steps_default: int = int(config.get("interp_steps", 0))
        self.interp_substep: int = 0
        self.interp_start: torch.Tensor | None = None  # [eef_num, 7]
        self.interp_end: torch.Tensor | None = None  # [eef_num, 7]
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
        # Resolve per-robot servers from planner_manager (single server per
        # robot post-MERGE_LEFT_RIGHT §1–§8). ``hand_id`` still drives
        # target packing (NaN-for-inactive-arm — see §3), not server
        # selection.
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
        """Parse action format (third header value: ``planner_mode``).

        Supported formats:
            - target_tensor → defaults for robot/hand; ``planner_mode`` unchanged
            - (robot_id, target_tensor) → legacy
            - ((robot_id, hand_id, planner_mode), target_tensor)
        """
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
            allow_base=True,
            max_eef_num=max_eef_num,
            per_eef_dim=per_eef_dim,
        )

    def _init_interp(self):
        """Snapshot the current eef pose as the lerp start. Interp only
        engages when ``planner_mode == INTERP_MODE`` AND the configured
        ``interp_steps`` hyperparameter is > 0."""
        steps = int(self.interp_steps_default)
        if (
            self.planner_mode == self.INTERP_MODE
            and steps > 0
            and self.current_arm_targets is not None
            and not self._is_placeholder_target(self.current_arm_targets)
        ):
            cur_pos, cur_quat = self._select_current_eef_for_target(
                self.current_arm_targets
            )
            start = torch.cat([cur_pos, cur_quat], dim=-1).to(self.env.device)
            end = self.current_arm_targets.to(self.env.device).clone()
            self.interp_steps = steps
            self.interp_substep = 0
            self.interp_start = start
            self.interp_end = end
            if self.debug:
                print(
                    f"[MoveL][env={self.env_id}] interp ON steps={steps} "
                    f"start_eef[0]={start[0].cpu().tolist()} "
                    f"end_eef[0]={end[0].cpu().tolist()}"
                )
        else:
            self.interp_steps = 0
            self.interp_substep = 0
            self.interp_start = None
            self.interp_end = None

    def _step_interp(self) -> torch.Tensor:
        """Send one lerped sub-target. Marks ``current_state="finished"``
        on the call that emits the final (alpha=1.0) sub-target."""
        n_total = max(1, int(self.interp_steps))
        # alpha goes 1/n_total, 2/n_total, ..., 1.0 across n_total calls.
        self.interp_substep += 1
        alpha = min(1.0, self.interp_substep / n_total)
        cur_arm = self.interp_start + alpha * (self.interp_end - self.interp_start)
        # Re-normalize quaternions per-eef so PD doesn't drift on lerped quats.
        if cur_arm.shape[-1] >= 7:
            quat = cur_arm[..., 3:7]
            n = torch.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)
            cur_arm = torch.cat([cur_arm[..., :3], quat / n], dim=-1)
        self.current_action = self._build_full_action(cur_arm.reshape(-1))
        if self.interp_substep >= n_total:
            self.current_state = "finished"
            if self.debug:
                print(
                    f"[MoveL][env={self.env_id}] interp DONE "
                    f"({self.interp_substep}/{n_total})"
                )
        else:
            self.current_state = "running"
        self.step_count += 1
        return self.current_action

    def _robot_slot_offset(self) -> int:
        """Offset (in the global action vector) where this robot's slots start.

        For dual-robot scenes the planner's own output only covers its own
        ``base | arm | eef`` slice — this method plus ``_get_total_action_dim``
        let us NaN-pad the other robots' slices.
        """
        planner_manager = self._get_planner_manager()
        info = planner_manager.get_info()
        if self.robot_name not in info:
            raise RuntimeError(f"Robot '{self.robot_name}' not found in planner info.")
        robot_info = info[self.robot_name]
        offsets = []
        for slot in ("base", "arm", "eef"):
            slc = robot_info.get(slot, {}).get("action_slice")
            if slc is not None:
                offsets.append(int(slc[0]))
        return min(offsets) if offsets else 0

    def _get_total_action_dim(self) -> int:
        return int(self._get_planner_manager().total_action_dim)

    def _build_full_action(self, arm_action_flat: torch.Tensor) -> torch.Tensor:
        """Full robot action vector; NaN-padded to cover all robots in the scene."""
        single_robot = self._build_full_action_manipulator(arm_action_flat)
        total_dim = self._get_total_action_dim()
        if single_robot.shape[0] == total_dim:
            return single_robot
        full = torch.full(
            (total_dim,),
            torch.nan,
            device=single_robot.device,
            dtype=single_robot.dtype,
        )
        offset = self._robot_slot_offset()
        full[offset : offset + single_robot.shape[0]] = single_robot
        return full

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

    def _is_placeholder_target(self, arm_targets: torch.Tensor) -> bool:
        """Check if arm targets are all NaN (placeholder during init phase)."""
        return torch.isnan(arm_targets).all()

    def _nan_action(self) -> torch.Tensor:
        try:
            total_dim = self._get_total_action_dim()
        except Exception:
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

    def _submit_plan(self, target_pose_all: torch.Tensor):
        robot_state = self._get_robot_state()
        robot_states_dict = {
            "base_pos": robot_state["base_pos"],
            "base_quat": robot_state["base_quat"],
            "joint_pos": robot_state["joint_pos"],
            "joint_vel": robot_state["joint_vel"],
        }
        # Pack into the unified MotionGenPlanRequest 2-D shape ``(1, eef_num*7)``.
        # ``_expand_arm_action`` is the single source of truth for hand_id
        # NaN-padding (right → [right_7d, nan*7], left → [nan*7, left_7d],
        # both → as-is). NaN rows trigger the Server's per-env-tool disable.
        target_flat = self._expand_arm_action(target_pose_all.view(-1)).view(1, -1)
        req = MotionGenPlanRequest(
            env_ids=[self.env_id],
            target_pos=target_flat,
            robot_states=robot_states_dict,
        )
        self.inflight_motiongen_future = self.motiongen_server.submit_plan(req)
        self.inflight_motiongen_target = target_pose_all.clone()
        if self.debug:
            print(
                "Submit MotionGen",
                "env_id=",
                self.env_id,
                "robot=",
                self.robot_name,
                "target=",
                self.inflight_motiongen_target,
            )
        self.inflight_ik_future = None
        self.inflight_ik_target = None
        self.plan_actions = None
        self.plan_step = 0
        self.last_action = None
        self.pending_target = None

    def _submit_ik(self, target_pose: torch.Tensor):
        robot_state = self._get_robot_state()
        robot_states_dict = {
            "base_pos": robot_state["base_pos"],
            "base_quat": robot_state["base_quat"],
            "joint_pos": robot_state["joint_pos"],
            "joint_vel": robot_state["joint_vel"],
        }
        # IKPlanRequest single-mode 2-D shape ``(1, L*7)`` — see Services
        # README §2.1. ``_expand_arm_action`` packs hand_id NaN rows.
        target_flat = self._expand_arm_action(target_pose.view(-1)).view(1, -1)
        req = IKPlanRequest(
            env_ids=[self.env_id],
            target_pos=target_flat,
            robot_states=robot_states_dict,
            mode="single",
        )
        self.inflight_ik_future = self.ik_server.submit_ik(req)
        self.inflight_ik_target = target_pose.clone()
        if self.debug:
            print(
                "Submit IK",
                "env_id=",
                self.env_id,
                "robot=",
                self.robot_name,
                "target=",
                self.inflight_ik_target,
            )
        self.inflight_motiongen_future = None
        self.inflight_motiongen_target = None
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
        if self.robot_name not in info:
            raise RuntimeError(
                f"Robot '{self.robot_name}' not found in planner info. "
                f"Available: {list(info.keys())}"
            )
        robot_info = info[self.robot_name]
        for key in ("base", "arm", "eef"):
            if key not in robot_info:
                raise RuntimeError(
                    f"Robot '{self.robot_name}' info missing '{key}'. "
                    f"Keys: {list(robot_info.keys())}"
                )
        base_dim = int(robot_info["base"].get("action_dim", 0))
        arm_dim = int(robot_info["arm"].get("action_dim", 0))
        eef_dim = int(robot_info["eef"].get("action_dim", 0))
        if base_dim == 0 and arm_dim == 0 and eef_dim == 0:
            raise RuntimeError(
                f"Robot '{self.robot_name}' has all zero action dims "
                f"(base={base_dim}, arm={arm_dim}, eef={eef_dim}). "
                "Check planner/action configuration."
            )
        return base_dim, arm_dim, eef_dim

    def _expand_eef_target(self, eef_target: torch.Tensor) -> torch.Tensor:
        """Expand per-eef target to full eef_dim, filling the other eef with NaN.

        For hand_id 0 (right): action goes into first per_eef_dim.
        For hand_id 1 (left):  action goes into last per_eef_dim.
        For hand_id -1 (both): returned as-is.
        """
        _, _, eef_dim = self._get_action_dims()
        if eef_dim == 0:
            return eef_target
        flat = eef_target.view(-1)
        if flat.shape[0] == eef_dim:
            return flat
        full_eef = torch.full(
            (eef_dim,), torch.nan, device=flat.device, dtype=flat.dtype
        )
        if self.hand_id == 0:
            full_eef[: flat.shape[0]] = flat
        elif self.hand_id == 1:
            full_eef[eef_dim - flat.shape[0] :] = flat
        else:
            full_eef[: flat.shape[0]] = flat
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

    # ``_format_target_for_submit`` removed — ``_expand_arm_action`` is the
    # single source of truth for hand_id NaN-padding. Submit sites call
    # ``_expand_arm_action(target.view(-1)).view(1, -1)`` directly.

    def _get_current_eef(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot_state = self._get_robot_state()
        return (
            robot_state["eef_pos"][self.env_id],
            robot_state["eef_quat"][self.env_id],
        )

    def _get_env_origin(self) -> torch.Tensor:
        return self.env.scene.env_origins[self.env_id].clone()

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
                if torch.isnan(pose[:3]).any():
                    continue
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
        # Matches the class-doc table (not MobileMoveL's full 0–3 / -2 mode table).
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
        # Reset expects action as ((robot_id, hand_id, planner_mode), target_tensor).
        # target_tensor can be 7D pose or 8D pose+gripper.
        robot_id, hand_id, target_action = self._parse_action(action)
        self._set_robot_by_id(robot_id, hand_id)
        (
            self.current_base_target,
            self.current_arm_targets,
            self.current_eef_target,
        ) = self._parse_target(target_action)
        self.arm_eef_num = int(self.current_arm_targets.shape[0])
        # Dual-arm: when eef_num > 1, hand_id becomes -1 (both arms).
        # Post-MERGE_LEFT_RIGHT flatten there is ONE server per robot —
        # no re-resolve needed; ``self.motiongen_server`` / ``self.ik_server``
        # already point at it from ``_set_robot_by_id``.
        if self.arm_eef_num > 1 and self.hand_id != -1:
            self.hand_id = -1
        self.step_count = 0  # Reset step count

        self.current_state = "ready"
        self.current_target = self._build_full_action(self.current_arm_targets.view(-1))
        self.current_command = ["MoveL", self.robot_name, self.current_target]
        self.current_action = None
        self.inflight_motiongen_future = None
        self.inflight_motiongen_target = None
        self.inflight_ik_future = None
        self.inflight_ik_target = None
        self.pending_target = None
        # Set up linear-interp state from the config hyperparameter.
        self._init_interp()
        # Skip IK submission if target is placeholder (all NaN during init).
        # Also skip when running in interp mode (we drive the arm directly).
        if (
            self.ik_check
            and not self._is_placeholder_target(self.current_arm_targets)
            and self.interp_steps == 0
        ):
            self._submit_ik(self.current_arm_targets)

    def step(self) -> torch.Tensor:
        if self.current_target is None or self.current_arm_targets is None:
            raise RuntimeError("Current Target Is Not Set, Please Call Reset First")

        # If target is placeholder (all NaN during init), return nan_action directly
        if self._is_placeholder_target(self.current_arm_targets):
            self.current_state = "waiting"
            return self._nan_action()

        # Linear-interpolation mode: only when the caller selected
        # planner_mode = INTERP_MODE (=2) AND interp is initialized.
        if self.planner_mode == self.INTERP_MODE and self.interp_steps > 0:
            return self._step_interp()

        if self.debug_sphere:
            robot_state = self._get_robot_state()
            joint_pos = robot_state["joint_pos"].clone()
            if joint_pos is not None:
                self._visualize_robot_spheres(joint_pos)

        if not self.ik_check:
            if self.inflight_motiongen_future is None and self.plan_actions is None:
                target_targets = self.current_arm_targets
                if self.pending_target is not None:
                    target_targets = self.pending_target
                    self.pending_target = None
                if self._should_use_motiongen(target_targets):
                    self._submit_plan(target_targets)
                    self.current_state = "planning"
                    return None
                self.plan_actions = target_targets.view(-1).unsqueeze(0)
                self.plan_step = 0

        # ---- 1) IK result handling ----
        # Always solve IK first; IK failure => Failed.
        # If IK success and target is far => submit MotionGen; otherwise use target directly.
        if self.inflight_ik_future is not None and self.inflight_ik_future.done():
            inflight_target = self.inflight_ik_target
            try:
                success, _, env_ids = self.inflight_ik_future.result()
            except Exception as e:
                if self.debug:
                    print(
                        "IK result failed",
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
            if self.pending_target is not None and not self._targets_close(
                self.pending_target, inflight_target
            ):
                if self.ik_check:
                    self._submit_ik(self.pending_target)
                    self.current_state = "planning"
                    return None
            if self.pending_target is not None:
                self.pending_target = None

            if len(success) == 0 or not success[0]:
                if self.debug:
                    print(
                        "IK result failed",
                        "env_id=",
                        self.env_id,
                        "robot=",
                        self.robot_name,
                        "target=",
                        inflight_target,
                    )
                self.current_state = "failed"
                return "Failed"
            if self.debug:
                print(
                    "IK result success",
                    "env_id=",
                    self.env_id,
                    "robot=",
                    self.robot_name,
                    "target=",
                    inflight_target,
                )

            target_targets = (
                inflight_target
                if inflight_target is not None
                else self.current_arm_targets
            )
            if self._should_use_motiongen(target_targets):
                self._submit_plan(self.current_arm_targets)
                self.current_state = "planning"
                return None
            self.plan_actions = target_targets.view(-1).unsqueeze(0)
            self.plan_step = 0

        # ---- 2) MotionGen result handling ----
        # If MotionGen fails, fall back to target pose directly.
        if (
            self.inflight_motiongen_future is not None
            and self.inflight_motiongen_future.done()
        ):
            inflight_target = self.inflight_motiongen_target
            try:
                actions, success, env_ids = self.inflight_motiongen_future.result()
            except Exception as ex:
                # Server-side raised (most commonly preprocessing / planning
                # error). Preserve ``env_ids`` so the asserts + downstream
                # fallback path still work; the empty ``success``/``actions``
                # below will trigger the "MotionGen failed → use target pose
                # directly" branch.
                if self.debug:
                    print(f"[MoveL] env_id={self.env_id} MotionGen future raised: {ex}")
                actions, success, env_ids = [], [], [self.env_id]
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
                # Target changed while planning; replan with latest target.
                if self.ik_check:
                    self._submit_ik(self.pending_target)
                    self.current_state = "planning"
                    return None
                if self._should_use_motiongen(self.pending_target):
                    self._submit_plan(self.pending_target)
                    self.current_state = "planning"
                    return None
                self.plan_actions = self.pending_target.view(-1).unsqueeze(0)
                self.plan_step = 0
                self.pending_target = None
            if self.pending_target is not None:
                self.pending_target = None

            if len(actions) > 0 and len(success) > 0 and success[0]:
                plan = actions[0]
                if self.debug_viz:
                    self._visualize_waypoints(actions)
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
                    self.plan_actions = plan.to(self.env.device, dtype=torch.float32)
                else:
                    self.plan_actions = torch.as_tensor(
                        plan, device=self.env.device, dtype=torch.float32
                    )
                self.plan_step = 0
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
                fallback = (
                    inflight_target.view(-1)
                    if inflight_target is not None
                    else self.current_arm_targets.view(-1)
                )
                self.plan_actions = fallback.unsqueeze(0)
                self.plan_step = 0

        if self.inflight_ik_future is not None:
            self._wait_count = getattr(self, "_wait_count", 0) + 1
            self.current_state = "planning"
            return None

        if self.inflight_motiongen_future is not None:
            self._wait_count = getattr(self, "_wait_count", 0) + 1
            self.current_state = "planning"
            return None

        if self.plan_actions is None:
            self._wait_count = getattr(self, "_wait_count", 0) + 1
            self.current_state = "planning"
            return None

        # ---- 3) Stream action trajectory (one step per call) ----
        self._wait_count = 0  # reset wait counter since we have a plan
        if self.plan_step < self.plan_actions.shape[0]:
            action_pose = self.plan_actions[self.plan_step]
            self.plan_step += 1
        else:
            # Repeat the final action once the plan is exhausted
            action_pose = self.plan_actions[-1]

        if action_pose.ndim != 1:
            action_pose = action_pose.view(-1)

        action = self._build_full_action(action_pose)

        self.step_count += 1  # Count only after a plan is ready (no waiting time)
        self.last_action = action
        self.current_state = "running"
        self.current_action = {"MoveL": action}
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

        # If new target is still placeholder (all NaN), do nothing
        if self._is_placeholder_target(new_arm_targets):
            return

        old_is_placeholder = self._is_placeholder_target(self.current_arm_targets)
        if old_is_placeholder:
            target_changed = True  # Transitioning from placeholder to real target
        else:
            target_changed = not self._targets_close(
                new_arm_targets, self.current_arm_targets
            )
        self.current_base_target = new_base_target
        self.current_arm_targets = new_arm_targets
        self.current_eef_target = new_eef_target
        self.current_target = self._build_full_action(new_arm_targets.view(-1))
        self.current_command = ["MoveL", self.robot_name, self.current_target]

        # Linear-interp refresh handling: only when the caller picked
        # planner_mode == INTERP_MODE. Same target → keep existing
        # interp progress (don't restart every frame). New target →
        # re-snapshot the start pose and restart the lerp.
        if self.planner_mode == self.INTERP_MODE and self.interp_steps_default > 0:
            if target_changed or robot_changed:
                self._init_interp()
            return  # interp drives the arm directly; skip motiongen/IK below
        # Switched out of interp mode: clear leftover interp state.
        if self.interp_steps > 0 and self.planner_mode != self.INTERP_MODE:
            self.interp_steps = 0
            self.interp_substep = 0
            self.interp_start = None
            self.interp_end = None

        if target_changed or robot_changed:
            # Replan: always restart from IK with the latest target.
            self.pending_target = new_arm_targets
            self.plan_actions = None
            self.plan_step = 0
            self.last_action = None
            if (
                self.inflight_motiongen_future is None
                and self.inflight_ik_future is None
            ):
                if self.ik_check:
                    self._submit_ik(new_arm_targets)
                else:
                    if self._should_use_motiongen(new_arm_targets):
                        self._submit_plan(new_arm_targets)
                        self.current_state = "planning"
                    else:
                        self.plan_actions = new_arm_targets.view(-1).unsqueeze(0)
                        self.plan_step = 0

    def get_done(self) -> bool:
        self._done_reason = None
        if self.current_arm_targets is None:
            return False

        # Placeholder target (all NaN during init): never consider done
        if self._is_placeholder_target(self.current_arm_targets):
            return False

        # Interpolation mode: completion is driven exclusively by the
        # substep counter — the eef may be near the START pose at the
        # beginning, which would otherwise trigger threshold-based done.
        if self.planner_mode == self.INTERP_MODE and self.interp_steps > 0:
            done = self.interp_substep >= self.interp_steps
            if done:
                self._done_reason = (
                    f"interp finished ({self.interp_substep}/{self.interp_steps})"
                )
            return done

        # If we have a plan, wait until all planned actions are consumed.
        if (
            self.plan_actions is not None
            and self.plan_step < self.plan_actions.shape[0]
        ):
            return False

        timeout_steps = int(getattr(self.config, "timeout_steps", 50))
        if self.step_count >= timeout_steps:
            self._done_reason = (
                f"timeout (step_count={self.step_count} >= "
                f"timeout_steps={timeout_steps})"
            )
            return True

        translation_threshold = float(self.config.translation_threshold)
        rotation_threshold = float(self.config.rotation_threshold)

        # Check all EEFs against their targets (supports dual-arm)
        target_poses = self.current_arm_targets  # [eef_num, 7]
        cur_pos, cur_quat = self._select_current_eef_for_target(target_poses)
        pair_count = min(cur_pos.shape[0], target_poses.shape[0])

        pos_diffs: list[float] = []
        rot_diffs: list[float] = []
        for eef_idx in range(pair_count):
            pos_diff = torch.norm(cur_pos[eef_idx] - target_poses[eef_idx, :3]).item()
            pos_diffs.append(pos_diff)
            if pos_diff >= translation_threshold:
                return False
            # Quaternion comparison: q and -q represent the same rotation
            quat_diff_1 = torch.norm(cur_quat[eef_idx] - target_poses[eef_idx, 3:7])
            quat_diff_2 = torch.norm(cur_quat[eef_idx] + target_poses[eef_idx, 3:7])
            quat_diff = torch.min(quat_diff_1, quat_diff_2).item()
            rot_diffs.append(quat_diff)
            if quat_diff >= rotation_threshold:
                return False

        self._done_reason = (
            f"reached target (pos_diff={['%.4f' % d for d in pos_diffs]}<"
            f"{translation_threshold:.4f}, rot_diff={['%.4f' % d for d in rot_diffs]}<"
            f"{rotation_threshold:.4f}, step_count={self.step_count})"
        )
        return True

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        if self.current_state == "failed":
            self.current_state = "failed: global planner failed to plan"
            print(
                f"[MoveL] env_id={self.env_id} TERMINATE reason=failed "
                f"(global planner failed to plan) truncated=3"
            )
            return {
                "type": "MoveL",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
            }
        elif self.current_state == "finished" or self.get_done():
            was_already_finished = self.current_state == "finished"
            self.current_state = "finished"
            reason = (
                "already-finished (held from previous update)"
                if was_already_finished
                else (getattr(self, "_done_reason", None) or "get_done()=True")
            )
            print(
                f"[MoveL] env_id={self.env_id} TERMINATE reason={reason} "
                f"finished=True truncated=0"
            )
            return {
                "type": "MoveL",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 0,
            }
        elif info["env_info"][2][self.env_id]:
            self.current_state = "truncated: env terminated first"
            print(
                f"[MoveL] env_id={self.env_id} TERMINATE reason=env terminated first "
                f"truncated=1"
            )
            return {
                "type": "MoveL",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
            }
        elif info["env_info"][3][self.env_id]:
            self.current_state = "truncated: env truncated first"
            print(
                f"[MoveL] env_id={self.env_id} TERMINATE reason=env truncated first "
                f"truncated=2"
            )
            return {
                "type": "MoveL",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
            }
        else:
            self.current_state = "running"
            return {
                "type": "MoveL",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }
