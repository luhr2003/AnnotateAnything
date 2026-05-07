from typing import Any, Dict
import torch
from magicsim.Collect.GlobalPlanner.GlobalPlanner import GlobalPlanner
from magicsim.Env.Utils.file import Logger
from magicsim.Env.Utils.viz import draw_waypoints
from magicsim.Env.Planner.Utils import quat_angle_between, quat_normalize
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class ServoL(GlobalPlanner):
    """
    Global Planner that follows a pre-computed trajectory waypoint by waypoint,
    directly sending each waypoint to the robot (no MotionGen).

    Advances to the next waypoint only when the EEF has arrived at the current
    waypoint or when timeout (_waypoint_max_steps) is reached.

    Action format for reset/refresh (aligned with MoveL/MobileMoveL; third field is ``planner_mode``)::

        ((robot_id, hand_id, planner_mode), trajectory_tensor)

    Legacy forms still work: plain ``trajectory_tensor``, ``(robot_id, trajectory_tensor)``,
    or ``((robot_id, hand_id), traj)``.

    ``hand_id``: 0=right, 1=left, -1=both arms.
    ``trajectory_tensor``: ``[N, 7]`` single-arm or ``[N, 14]`` dual-arm; optional extra eef dims per waypoint.

    This class does **not** run IK or MotionGen; ``planner_mode`` is parsed and stored for command-format
    consistency only (no branching in control).

    ``planner_mode`` (documentation only; not MobileMoveL's full table; no execution switch)::

        +--------+------------------------------+
        |  mode  | ServoL                       |
        +--------+------------------------------+
        |  any   | Stored only; playback unchanged |
        +--------+------------------------------+
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        # Trajectory management
        self.trajectory = None  # [N, 7] full waypoint trajectory
        self.trajectory_idx = 0  # Current target waypoint index in trajectory
        self.current_target = None
        self.current_base_target = None
        self.current_eef_target = None
        self.arm_eef_num = 1
        self.robot_id = -1
        self.hand_id = 0  # 0 = right, 1 = left, -1 = both
        self.planner_mode = -1  # See class docstring; not used for trajectory control.
        self.robot_name = None
        self.stride = int(config.get("stride", 1))

        self.last_action = None
        self.step_count = 0
        self.eef_targets = None  # [N, eef_dim] per-waypoint gripper, or None

        # Arrival checking thresholds
        self._eef_pos_threshold = config.get("eef_pos_threshold", 0.03)
        self._eef_rot_threshold = config.get("eef_rot_threshold", 0.1)
        self._waypoint_max_steps = config.get("waypoint_max_steps", 20)
        self._waypoint_step_counter = 0

        self.debug = config.get("debug", False)
        super().__init__(config, env, env_id, logger)

    # ------------------------------------------------------------------ #
    # Robot & server helpers (from MobileMoveL)
    # ------------------------------------------------------------------ #

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
        return True

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

    def _expand_eef_target(self, eef_target: torch.Tensor) -> torch.Tensor:
        """Expand per-eef target to full eef_dim. hand_id 0: first, 1: last, -1: as-is."""
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
        """Expand per-arm action to full arm_dim. hand_id 0: first 7, 1: last 7, -1: as-is."""
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

    def _get_current_eef(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot_state = self._get_robot_state()
        return (
            robot_state["eef_pos"][self.env_id],
            robot_state["eef_quat"][self.env_id],
        )

    def _select_current_eef_for_target(
        self, target_pose_all: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select EEF pos/quat matching target's EEF count. For single-arm on dual-arm robot, use hand_id."""
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

    # ------------------------------------------------------------------ #
    # Visualization (following MobileMoveL pattern)
    # ------------------------------------------------------------------ #

    def _visualize_trajectory_waypoints(self):
        """Draw all trajectory waypoints as green points in the viewport."""
        if self.trajectory is None:
            return
        origin_cpu = self._get_env_origin().detach().cpu()
        points: list[list[float]] = []
        n_poses = self.trajectory.shape[1] // 7
        for i in range(self.trajectory.shape[0]):
            for j in range(n_poses):
                pos = (
                    self.trajectory[i, j * 7 : j * 7 + 3].detach().cpu() + origin_cpu
                ).tolist()
                points.append(pos)
        print(
            f"[ServoL] Drawing {len(points)} trajectory waypoints (green), "
            f"env_id={self.env_id}, first={points[0]}, last={points[-1]}"
        )
        draw_waypoints(
            points, point_size=8.0, color=(0.0, 1.0, 0.0, 0.8), clear_existing=True
        )

    # ------------------------------------------------------------------ #
    # Action building
    # ------------------------------------------------------------------ #

    def _nan_action(self) -> torch.Tensor:
        base_dim, arm_dim, eef_dim = self._get_action_dims()
        total_dim = base_dim + arm_dim + eef_dim
        return torch.full(
            (total_dim,),
            torch.nan,
            device=self.env.device,
            dtype=torch.float32,
        )

    def _build_full_action(self, arm_action_flat: torch.Tensor) -> torch.Tensor:
        """Full robot action vector; same as MoveL via ``GlobalPlanner._build_full_action_manipulator``."""
        return self._build_full_action_manipulator(arm_action_flat)

    # ------------------------------------------------------------------ #
    # Trajectory parsing
    # ------------------------------------------------------------------ #

    def _parse_action(self, action):
        robot_id, hand_id, mode, target = GlobalPlanner.parse_planner_header(
            action,
            default_robot_id=self.robot_id if self.robot_id >= 0 else 0,
            default_hand_id=self.hand_id,
        )
        self.planner_mode = mode
        return robot_id, hand_id, target

    def _parse_trajectory(self, action: torch.Tensor):
        """Parse arm-only trajectory into ``[N, 7*active_eef_num]`` (+ optional per-waypoint EEF tail).

        Width is determined strictly by ``hand_id`` (``eef_num_from_hand_id``), matching
        ``MoveL.parse_target_vector`` / ``MobileServoL._parse_servo_arm_trajectory``:
            - ``hand_id=0/1`` (single arm): arm block width ``7``, optional tail ``per_eef_dim``
            - ``hand_id=-1`` (both arms):  arm block width ``14``, optional tail ``2*per_eef_dim``

        Feeding a width that doesn't match the active-arm layout is rejected (no more
        auto single/dual detection).
        """
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(
                action, device=self.env.device, dtype=torch.float32
            )
        else:
            action = action.to(self.env.device, dtype=torch.float32)
        if action.ndim not in (1, 2):
            raise ValueError(
                f"ServoL trajectory must be 1-D or 2-D, got shape {tuple(action.shape)}"
            )

        _, _, eef_dim = self._get_action_dims()
        max_eef_num, per_eef_dim = self._eef_layout_from_robot()
        if eef_dim > 0:
            if per_eef_dim * max_eef_num != eef_dim:
                raise ValueError(
                    f"eef_dim {eef_dim} inconsistent with RobotManager layout "
                    f"max_eef_num={max_eef_num}, per_eef_dim={per_eef_dim}."
                )
        elif per_eef_dim != 0:
            raise ValueError(
                f"eef_dim is 0 but per_eef_dim={per_eef_dim} (RobotManager.get_info)."
            )

        n_eef = GlobalPlanner.eef_num_from_hand_id(self.hand_id)
        if n_eef > max_eef_num:
            raise ValueError(
                f"hand_id={self.hand_id} implies {n_eef} active EEFs but "
                f"max_eef_num is {max_eef_num}."
            )
        arm_w = 7 * n_eef
        tail = per_eef_dim * n_eef if eef_dim > 0 else 0
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
                        f"ServoL flat trajectory length {L} must divide {full_w} "
                        f"(arm+EEF) or {arm_w} (arm only); hand_id={self.hand_id}, "
                        f"tail={tail}."
                    )
            elif L % arm_w != 0:
                raise ValueError(
                    f"ServoL flat trajectory length {L} must divide {arm_w} "
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
                    f"ServoL waypoint width {w} must be {arm_w} or {full_w} "
                    f"(hand_id={self.hand_id})."
                )
        elif w != arm_w:
            raise ValueError(
                f"ServoL waypoint width {w} must be {arm_w} (hand_id={self.hand_id})."
            )
        else:
            traj = action.clone()

        # Normalize quaternions (each 7D pose in traj)
        n_poses = traj.shape[1] // 7
        parts = []
        for i in range(n_poses):
            p = traj[:, i * 7 : (i + 1) * 7]
            parts.append(torch.cat([p[:, :3], quat_normalize(p[:, 3:])], dim=1))
        traj = torch.cat(parts, dim=1)
        return traj, eef_target

    # ------------------------------------------------------------------ #
    # Arrival checking
    # ------------------------------------------------------------------ #

    def _check_waypoint_arrived(self, target_pose: torch.Tensor) -> bool:
        """Check if the EEF(s) have arrived at the target waypoint. Supports dual-arm."""
        cur_pos, cur_quat = self._select_current_eef_for_target(target_pose)

        tp = target_pose.view(-1, 7)
        if tp.shape[0] == 0:
            return False

        # Check each EEF (single or both arms)
        for i in range(tp.shape[0]):
            target_pos = tp[i, :3]
            target_quat = tp[i, 3:7]
            eef_pos = cur_pos[i] if cur_pos.ndim > 1 else cur_pos
            eef_quat = cur_quat[i] if cur_quat.ndim > 1 else cur_quat
            pos_diff = torch.linalg.norm(eef_pos - target_pos).item()
            rot_diff = quat_angle_between(
                eef_quat.unsqueeze(0), target_quat.unsqueeze(0)
            ).item()
            if (
                pos_diff >= self._eef_pos_threshold
                or rot_diff >= self._eef_rot_threshold
            ):
                return False
        return True

    def _targets_close(self, target_a: torch.Tensor, target_b: torch.Tensor) -> bool:
        if target_a is None or target_b is None:
            return False
        if target_a.shape != target_b.shape:
            return False
        pos_diff = torch.linalg.norm(target_a[:3] - target_b[:3]).item()
        return pos_diff < float(self.config.translation_threshold)

    # ------------------------------------------------------------------ #
    # Trajectory waypoint management
    # ------------------------------------------------------------------ #

    def _get_current_waypoint(self) -> torch.Tensor:
        idx = min(self.trajectory_idx, self.trajectory.shape[0] - 1)
        return self.trajectory[idx]

    # ------------------------------------------------------------------ #
    # Core interface: reset / step / refresh / get_done / update
    # ------------------------------------------------------------------ #

    def reset(self, action):
        robot_id, hand_id, traj_action = self._parse_action(action)
        self._set_robot_by_id(robot_id, hand_id)

        self.trajectory, eef_targets = self._parse_trajectory(traj_action)
        self.eef_targets = eef_targets  # [N, eef_dim] or None
        if eef_targets is not None:
            self.current_eef_target = eef_targets[-1]
        else:
            self.current_eef_target = None

        self.current_base_target = None
        self.arm_eef_num = (
            self.trajectory.shape[1] // 7
        )  # 1 for single-arm, 2 for dual-arm
        self.trajectory_idx = 0
        self.step_count = 0
        self.last_action = None
        self._waypoint_step_counter = 0

        # Current target is the final waypoint (for done checking)
        final_wp = self.trajectory[-1]
        self.current_target = self._build_full_action(final_wp)
        self.current_command = ["ServoL", self.robot_name, self.current_target]
        self.current_action = None
        self.current_state = "ready"

        # Visualize full trajectory waypoints (green)
        if self.debug:
            self._visualize_trajectory_waypoints()

        if self.debug:
            print(
                f"ServoL reset: env_id={self.env_id}, robot={self.robot_name}, "
                f"trajectory={self.trajectory.shape}, stride={self.stride}"
            )

    def step(self) -> torch.Tensor:
        if self.trajectory is None:
            raise RuntimeError("Trajectory not set. Call reset first.")

        # Directly follow trajectory waypoints (no MotionGen).
        # Advance to next waypoint only when arrived or timeout.
        if self.trajectory_idx < self.trajectory.shape[0]:
            action_pose = self.trajectory[self.trajectory_idx]
            arrived = self._check_waypoint_arrived(action_pose)
            self._waypoint_step_counter += 1
            if arrived or self._waypoint_step_counter >= self._waypoint_max_steps:
                self.trajectory_idx += self.stride
                self._waypoint_step_counter = 0
                if self.trajectory_idx < self.trajectory.shape[0]:
                    action_pose = self.trajectory[self.trajectory_idx]
                else:
                    action_pose = self.trajectory[-1]
        else:
            # Trajectory exhausted, repeat last waypoint
            action_pose = self.trajectory[-1]

        # Use per-waypoint gripper if available
        if self.eef_targets is not None:
            idx = min(self.trajectory_idx, self.eef_targets.shape[0] - 1)
            self.current_eef_target = self.eef_targets[idx]

        if action_pose.ndim != 1:
            action_pose = action_pose.view(-1)

        action = self._build_full_action(action_pose)
        self.step_count += 1
        self.last_action = action
        self.current_state = "running"
        self.current_action = {"ServoL": action}

        return action

    def refresh(self, action):
        robot_id, hand_id, traj_action = self._parse_action(action)
        robot_changed = self._set_robot_by_id(robot_id, hand_id)

        new_traj, eef_targets = self._parse_trajectory(traj_action)

        # Check if trajectory changed (compare final waypoints)
        traj_changed = True
        if self.trajectory is not None and new_traj.shape == self.trajectory.shape:
            diff = torch.linalg.norm(new_traj[-1, :3] - self.trajectory[-1, :3]).item()
            if diff < float(self.config.translation_threshold):
                traj_changed = False

        if traj_changed or robot_changed:
            # Restart with new trajectory
            self.trajectory = new_traj
            self.eef_targets = eef_targets
            if eef_targets is not None:
                self.current_eef_target = eef_targets[-1]

            final_wp = self.trajectory[-1]
            self.current_target = self._build_full_action(final_wp)
            self.current_command = ["ServoL", self.robot_name, self.current_target]

            self.trajectory_idx = 0
            self.last_action = None
            self._waypoint_step_counter = 0

    def get_done(self) -> bool:
        if self.trajectory is None:
            return False

        # Global timeout bail-out: must be checked before the waypoint-progress
        # gate, otherwise a robot that fails to track intermediate waypoints
        # keeps ServoL alive for waypoint_max_steps * num_waypoints total steps.
        timeout_steps = int(getattr(self.config, "timeout_steps", 300))
        if self.step_count >= timeout_steps:
            return True

        # Not done if still advancing through waypoints
        if self.trajectory_idx < self.trajectory.shape[0] - 1:
            return False

        # Check if EEF(s) have reached the final waypoint (supports dual-arm)
        cur_pos, cur_quat = self._select_current_eef_for_target(self.trajectory[-1])
        target_poses = self.trajectory[-1].view(-1, 7)

        translation_threshold = self.config.translation_threshold
        rotation_threshold = self.config.rotation_threshold
        pose_done = True
        for i in range(target_poses.shape[0]):
            eef_pos = cur_pos[i] if cur_pos.ndim > 1 else cur_pos
            eef_quat = cur_quat[i] if cur_quat.ndim > 1 else cur_quat
            target_pos = target_poses[i, :3]
            target_quat = target_poses[i, 3:7]
            quat_diff_1 = torch.norm(eef_quat - target_quat)
            quat_diff_2 = torch.norm(eef_quat + target_quat)
            quat_diff = torch.min(quat_diff_1, quat_diff_2)
            if (
                torch.norm(eef_pos - target_pos) >= translation_threshold
                or quat_diff >= rotation_threshold
            ):
                pose_done = False
                break

        return pose_done

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        if self.current_state == "failed":
            self.current_state = "failed: global planner failed to plan"
            return {
                "type": "ServoL",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
            }
        elif self.current_state == "finished" or self.get_done():
            self.current_state = "finished"
            return {
                "type": "ServoL",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 0,
            }
        elif info["env_info"][2][self.env_id]:
            self.current_state = "truncated: env terminated first"
            return {
                "type": "ServoL",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
            }
        elif info["env_info"][3][self.env_id]:
            self.current_state = "truncated: env truncated first"
            return {
                "type": "ServoL",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
            }
        else:
            self.current_state = "running"
            return {
                "type": "ServoL",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }
