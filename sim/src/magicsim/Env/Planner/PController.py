from typing import Sequence, Type
from magicsim.Env.Robot.RobotManager import RobotManager
from magicsim.Env.Planner.Planner import Planner
import torch


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _get_cfg_val(obj, name: str, default):
    """从 OmegaConf / dict 取属性，兼容 getattr 和 []。"""
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    if hasattr(obj, "get"):
        return obj.get(name, default)
    return default


class PController(Planner):
    """PD-controller for humanoid body navigation and turning (open-loop).

    **Preprocessed ``mode_flag``** (last element of the vector ``[x, y, heading, *extra, mode_flag]``):

    * ``-2`` **Skip** — Output command is identical to ``last_command`` (full vector including
      extras). No PD update this step; used for invalid/placeholder targets (e.g. all-NaN).
    * ``-1`` **Lock skip** — Base linear and angular velocities are zero; ``extra_dims`` (e.g.
      height, torso RPY) pass through from the preprocessed action unchanged.
    * ``0`` **Navigate** — ``navigation_pd_controller``: track ``(x,y)`` and yaw toward target.
    * ``1`` **Turning** — ``turning_pd_controller`` only; yaw rate from heading error; no xy
      navigation (line velocities stay zero in the turning branch).

    G1 maps 15-dim ``lock_flag`` to this ``mode_flag`` in ``G1PControllerHelper.preprocess``.
    """

    def __init__(
        self,
        robot_manager: RobotManager,
        robot_type: str = "g1",
        robot_name: str = "g1",
        device: torch.device = torch.device("cpu"),
        num_envs: int = 1,
        n_extra_dims: int = 4,
        p_controller_helper: Type | None = None,
        p_controller_config=None,
    ):
        """
        Initialize PD-controller for humanoid body.

        Args:
            robot_manager: Robot manager instance
            robot_type: Type of robot
            robot_name: Name of robot
            device: Torch device
            num_envs: Number of environments
            n_extra_dims: Number of extra pass-through dimensions between heading and mode_flag
                in the preprocessed action. E.g. G1 has 4 (height, torso_roll, torso_pitch,
                torso_yaw), RidgebackFranka has 0.
            p_controller_helper: Class with preprocess, optional postprocess, and reset_idx.
                Instantiated with (num_envs, device). preprocess returns
                [N, 3 + n_extra_dims + 1]: [x, y, heading, *extra, mode_flag].
            p_controller_config: Optional config dict / DictConfig containing PD gains and limits.
                Supported keys (with defaults):
                    kp_angular_turning_only (0.4), kd_angular_turning_only (0.1),
                    kp_linear_x (2.0), kd_linear_x (0.5),
                    kp_linear_y (2.0), kd_linear_y (0.5),
                    kp_angular (0.05), kd_angular (0.1),
                    min_vel (-0.4), max_vel (0.4),
                    linear_dead_zone (0.1), angular_dead_zone (0.1).
        """
        self.robot_manager = robot_manager
        self.robot_name = robot_name
        self.robot_type = robot_type
        self.device = device
        self.num_envs = num_envs

        # --- parse PD gains / limits from config (like Dwb does) ---
        cfg = p_controller_config
        self.kp_angular_turning_only = _get_cfg_val(cfg, "kp_angular_turning_only", 0.4)
        self.kd_angular_turning_only = _get_cfg_val(cfg, "kd_angular_turning_only", 0.1)
        self.kp_linear_x = _get_cfg_val(cfg, "kp_linear_x", 2.0)
        self.kd_linear_x = _get_cfg_val(cfg, "kd_linear_x", 0.5)
        self.kp_linear_y = _get_cfg_val(cfg, "kp_linear_y", 2.0)
        self.kd_linear_y = _get_cfg_val(cfg, "kd_linear_y", 0.5)
        self.kp_angular = _get_cfg_val(cfg, "kp_angular", 0.05)
        self.kd_angular = _get_cfg_val(cfg, "kd_angular", 0.1)
        self.min_vel = _get_cfg_val(cfg, "min_vel", -0.4)
        self.max_vel = _get_cfg_val(cfg, "max_vel", 0.4)
        self.linear_dead_zone = _get_cfg_val(cfg, "linear_dead_zone", 0.1)
        self.angular_dead_zone = _get_cfg_val(cfg, "angular_dead_zone", 0.1)
        # Nav-mode heading target selector in [-1, 1]:
        #   t =  1 → desired = atan2(dy, dx)  (face the goal point)
        #   t =  0 → desired = current_yaw    (no yaw correction)
        #   t = -1 → desired = target_heading (face the planner yaw)
        #   t ∈ (0, 1) → slerp(current_yaw, atan2, t)
        #   t ∈ (-1, 0) → slerp(current_yaw, target_heading, |t|)
        self.nav_heading_target = float(_get_cfg_val(cfg, "nav_heading_target", 1.0))

        self.n_extra_dims = n_extra_dims
        self._preprocessor = None
        if p_controller_helper is not None:
            self._preprocessor = p_controller_helper(num_envs, device)
            self.preprocess_fn = self._preprocessor.preprocess
            self.postprocess_fn = getattr(self._preprocessor, "postprocess", None)
        else:
            self.preprocess_fn = None
            self.postprocess_fn = None
        # Preprocessed action width: [x, y, heading, *extra, mode_flag]
        self.preprocessed_width = 3 + n_extra_dims + 1
        # Output command width: [vel_x, vel_y, ang_vel, *extra]
        self.output_width = 3 + n_extra_dims

        # Command buffer to store last command for each environment
        # Shape: [num_envs, output_width] - [x_vel, y_vel, rz_vel, *extra_passthrough]
        self.last_command = torch.zeros(num_envs, self.output_width, device=device)

    def get_pos_diff(
        self, target_xy: torch.Tensor, current_xy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the position difference between the target and current position."""
        # target_xy: [N, 2] or [N, 1, 2]
        # current_xy: [N, 2]
        if target_xy.ndim == 3:
            target_xy = target_xy.squeeze(1)  # [N, 2]

        dx = target_xy[:, 0] - current_xy[:, 0]
        dy = target_xy[:, 1] - current_xy[:, 1]
        distance_error = torch.sqrt(dx**2 + dy**2)
        return dx, dy, distance_error

    def get_heading_diff(
        self, target_heading: torch.Tensor, current_heading: torch.Tensor
    ) -> torch.Tensor:
        """Get the heading difference between the target and current heading."""
        # target_heading: [N] or [N, 1]
        # current_heading: [N]
        if target_heading.ndim > 1:
            target_heading = target_heading.squeeze(-1)
        heading_error = wrap_to_pi(target_heading - wrap_to_pi(current_heading))
        return heading_error

    def turning_pd_controller(
        self,
        target_heading: torch.Tensor,
        current_heading: torch.Tensor,
        current_angular_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """PD-controller for in-place turning."""
        heading_error = wrap_to_pi(target_heading - current_heading)
        # PD control: u = kp * error - kd * current_velocity
        # The derivative term uses negative current velocity to provide damping
        angular_velocity = (
            self.kp_angular_turning_only * heading_error
            - self.kd_angular_turning_only * current_angular_velocity
        )
        # Clamp to velocity limits
        angular_velocity = torch.clamp(angular_velocity, self.min_vel, self.max_vel)
        return angular_velocity

    def navigation_pd_controller(
        self,
        target_xy: torch.Tensor,
        current_xy: torch.Tensor,
        current_theta: torch.Tensor,
        current_lin_vel: torch.Tensor,
        current_ang_vel: torch.Tensor,
        target_heading: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PD-controller for x & y & yaw control.

        The yaw set-point is selected by :attr:`nav_heading_target` ``t``:
        ``t=1`` face the goal point (``atan2(dy, dx)``), ``t=0`` hold
        current yaw, ``t=-1`` use ``target_heading``; intermediate values
        interpolate along the shortest angular path.
        """
        dx, dy, distance_error = self.get_pos_diff(target_xy, current_xy)
        wrapped_theta = wrap_to_pi(current_theta)

        # Translate dx, dy in world frame to robot local frame
        dx_local = dx * torch.cos(wrapped_theta) + dy * torch.sin(wrapped_theta)
        dy_local = -dx * torch.sin(wrapped_theta) + dy * torch.cos(wrapped_theta)

        # Desired yaw: interpolate based on nav_heading_target ∈ [-1, 1].
        t = self.nav_heading_target
        if t >= 1.0 - 1e-6:
            desired_angle = torch.atan2(dy, dx)
        elif t <= -1.0 + 1e-6:
            assert target_heading is not None, (
                "nav_heading_target=-1 requires target_heading to be supplied."
            )
            desired_angle = wrap_to_pi(target_heading)
        elif abs(t) < 1e-6:
            desired_angle = wrapped_theta
        elif t > 0.0:
            # (0, 1): slerp(current_yaw → atan2(dy, dx), t)
            face = torch.atan2(dy, dx)
            delta = wrap_to_pi(face - wrapped_theta)
            desired_angle = wrap_to_pi(wrapped_theta + t * delta)
        else:
            # (-1, 0): slerp(current_yaw → target_heading, |t|)
            assert target_heading is not None, (
                "nav_heading_target<0 requires target_heading to be supplied."
            )
            s = abs(t)
            delta = wrap_to_pi(wrap_to_pi(target_heading) - wrapped_theta)
            desired_angle = wrap_to_pi(wrapped_theta + s * delta)

        # Angular error (difference between desired and current orientation)
        angle_error = wrap_to_pi(desired_angle - wrapped_theta)

        # Transform current velocity from world frame to robot local frame
        # current_lin_vel: [N, 3] in world frame
        current_vx_world = current_lin_vel[:, 0]  # [N]
        current_vy_world = current_lin_vel[:, 1]  # [N]
        current_vx_local = current_vx_world * torch.cos(
            wrapped_theta
        ) + current_vy_world * torch.sin(wrapped_theta)  # [N]
        current_vy_local = -current_vx_world * torch.sin(
            wrapped_theta
        ) + current_vy_world * torch.cos(wrapped_theta)  # [N]

        # PD control for linear velocities: u = kp * error - kd * current_velocity
        # The derivative term uses negative current velocity to provide damping

        vx = self.kp_linear_x * dx_local - self.kd_linear_x * current_vx_local
        vy = self.kp_linear_y * dy_local - self.kd_linear_y * current_vy_local

        # print(f"kp_linear_x: {self.kp_linear_x}, kd_linear_x: {self.kd_linear_x}, kp_linear_y: {self.kp_linear_y}, kd_linear_y: {self.kd_linear_y}")

        # PD control for angular velocity: u = kp * error - kd * current_angular_velocity
        # Extract z-component of angular velocity (yaw rate)
        current_ang_vel_z = (
            current_ang_vel[:, 2] if current_ang_vel.ndim > 1 else current_ang_vel
        )  # [N]
        angular_velocity = (
            self.kp_angular * angle_error - self.kd_angular * current_ang_vel_z
        )

        # Clamp to velocity limits
        lin_vel_x = torch.clamp(vx, self.min_vel, self.max_vel)
        lin_vel_y = torch.clamp(vy, self.min_vel, self.max_vel)
        ang_vel = torch.clamp(angular_velocity, self.min_vel, self.max_vel)

        return lin_vel_x, lin_vel_y, ang_vel

    def step(
        self,
        action: torch.Tensor,
        env_ids: torch.Tensor | Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Step function for P-controller (open-loop).

        **Input:** ``action`` is robot-specific; if ``preprocess_fn`` is set, it becomes
        ``[x, y, heading, *extra, mode_flag]`` (width ``3 + n_extra_dims + 1``).

        **Output:** ``[vel_x, vel_y, ang_vel, *extra]`` — same width as ``output_width``.

        **Per-env ``mode_flag`` (last column of preprocessed action):**

        * ``-2`` — **Skip**: return previous ``last_command`` for this env (no PD).
        * ``-1`` — **Lock skip**: velocities ``(vx, vy, wz) = 0``; ``extra`` copied from input.
        * ``0`` — **Nav**: ``navigation_pd_controller`` fills ``vx, vy, wz``.
        * ``1`` — **Turning**: ``turning_pd_controller`` sets ``wz``; ``vx=vy=0`` (turning mask).

        Non-skip steps update ``last_command`` so the next skip can replay them.

        Args:
            action: Raw command; preprocessed to generic tensor when ``preprocess_fn`` is set.
            env_ids: Environment indices for this batch.

        Returns:
            ``[N, 3 + n_extra_dims]`` command tensor. For ``mode_flag == -2``, each row equals
            the stored ``last_command`` for that env.
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(self.device)

        # Ensure action is on correct device and has correct shape
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device, dtype=torch.float32)
        else:
            action = action.to(self.device)

        if action.ndim == 1:
            action = action.unsqueeze(0)

        N = action.shape[0]
        assert N == len(env_ids), (
            f"Action batch size {N} != env_ids length {len(env_ids)}"
        )

        # Get current robot state (needed for both preprocessing and control)
        robot_state = self.robot_manager.get_robot_state(noise_flag=False)[0][
            self.robot_name
        ]

        W = self.preprocessed_width  # 3 + n_extra_dims + 1
        if self.preprocess_fn is not None:
            processed_action = self.preprocess_fn(
                action=action,
                robot_state=robot_state,
                env_ids=env_ids,
                device=self.device,
            )
            assert processed_action.shape == (N, W), (
                f"Expected preprocessed shape [{N}, {W}], got {processed_action.shape}"
            )
        else:
            # No helper, expect correct width input directly
            assert action.shape[1] == W, (
                f"Expected action shape [N, {W}], got {action.shape}"
            )
            processed_action = action.to(self.device)

        # Extract components: [x, y, heading, *extra(n_extra_dims), mode_flag]
        target_x = processed_action[:, 0]  # [N]
        target_y = processed_action[:, 1]  # [N]
        target_heading = processed_action[:, 2]  # [N]
        extra_dims = processed_action[:, 3 : 3 + self.n_extra_dims]  # [N, n_extra_dims]
        mode_flag = processed_action[
            :, -1
        ]  # [N] -2: skip, -1: lock_skip, 0: nav, 1: turning

        # Check for full skip mode (-2): use last command entirely
        skip_mask = torch.abs(mode_flag + 2.0) < 0.5  # mode_flag ≈ -2
        # Active PD control mask: mode_flag == 0 (nav) or 1 (turning)
        # mode_flag -1 (lock_skip) is NOT active: zero velocity, but keeps height + torso RPY
        active_mask = mode_flag >= -0.5  # True for mode_flag 0 and 1

        # Get current robot pose and velocities
        base_pos = robot_state["base_pos"][env_ids]  # [N, 3]
        base_quat = robot_state["base_quat"][env_ids]  # [N, 4] (w, x, y, z)
        base_lin_vel = robot_state.get("base_lin_vel", None)
        base_ang_vel = robot_state.get("base_ang_vel", None)
        current_lin_vel = (
            base_lin_vel[env_ids]
            if base_lin_vel is not None
            else torch.zeros(N, 3, device=self.device)
        )
        current_ang_vel = (
            base_ang_vel[env_ids]
            if base_ang_vel is not None
            else torch.zeros(N, 3, device=self.device)
        )
        current_xy = base_pos[:, :2]  # [N, 2]

        # Initialize output velocities
        lin_vel_x = torch.zeros(N, device=self.device)
        lin_vel_y = torch.zeros(N, device=self.device)
        ang_vel = torch.zeros(N, device=self.device)

        # Process active environments (not skipping)
        if torch.any(active_mask):
            # Extract current heading (yaw) from quaternion for active envs
            qw = base_quat[:, 0]
            qx = base_quat[:, 1]
            qy = base_quat[:, 2]
            qz = base_quat[:, 3]
            current_heading = torch.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )  # [N]

            # Prepare target tensors
            target_xy = torch.stack([target_x, target_y], dim=1)  # [N, 2]

            # Determine mode: 0 = nav, 1 = turning
            turning_mask = (mode_flag > 0.5) & active_mask  # [N] bool
            nav_mask = (mode_flag <= 0.5) & active_mask  # [N] bool

            # Navigation mode: control x, y, heading
            if torch.any(nav_mask):
                nav_vx, nav_vy, nav_av = self.navigation_pd_controller(
                    target_xy[nav_mask],
                    current_xy[nav_mask],
                    current_heading[nav_mask],
                    current_lin_vel[nav_mask],
                    current_ang_vel[nav_mask],
                    target_heading=target_heading[nav_mask],
                )
                # Use index assignment since nav_vx/nav_vy/nav_av have shape [M] where M = sum(nav_mask)
                lin_vel_x[nav_mask] = nav_vx
                lin_vel_y[nav_mask] = nav_vy
                ang_vel[nav_mask] = nav_av

            # Turning mode: only control heading
            if torch.any(turning_mask):
                # Extract z-component of angular velocity (yaw rate)
                turning_ang_vel_z = current_ang_vel[turning_mask, 2]  # [M]
                turning_ang_vel = self.turning_pd_controller(
                    target_heading[turning_mask],
                    current_heading[turning_mask],
                    turning_ang_vel_z,
                )
                # Use index assignment since turning_ang_vel has shape [M] where M = sum(turning_mask)
                ang_vel[turning_mask] = turning_ang_vel

            # Apply dead zone for very small velocities to avoid jitter
            # Only set to zero if velocity is extremely small (below dead zone)
            # This allows natural deceleration near target without forcing minimum speed
            lin_vel_x = torch.where(
                (torch.abs(lin_vel_x) < self.linear_dead_zone) & active_mask,
                torch.zeros_like(lin_vel_x),
                lin_vel_x,
            )
            lin_vel_y = torch.where(
                (torch.abs(lin_vel_y) < self.linear_dead_zone) & active_mask,
                torch.zeros_like(lin_vel_y),
                lin_vel_y,
            )
            ang_vel = torch.where(
                (torch.abs(ang_vel) < self.angular_dead_zone) & active_mask,
                torch.zeros_like(ang_vel),
                ang_vel,
            )

        # Construct output: [vel_x, vel_y, ang_vel, *extra_passthrough]
        # The 3 velocity dims are computed by PD control; extra dims are passed through.
        vel_part = torch.stack([lin_vel_x, lin_vel_y, ang_vel], dim=1)  # [N, 3]
        if self.n_extra_dims > 0:
            new_command = torch.cat([vel_part, extra_dims], dim=1)  # [N, output_width]
        else:
            new_command = vel_part  # [N, 3]
        assert new_command.shape == (N, self.output_width), (
            f"New command shape should be [{N}, {self.output_width}], got {new_command.shape}"
        )

        # Postprocess: scale velocities (e.g., amplify when squatting)
        if self.postprocess_fn is not None:
            new_command = self.postprocess_fn(
                action=new_command,
                mode_flag=mode_flag,
                robot_state=robot_state,
                env_ids=env_ids,
            )

        # Get last command for the corresponding env_ids
        last_cmd_for_envs = self.last_command[env_ids]  # [N, output_width]

        # For skip mode, use last command; for active mode, use new command
        # skip_mask: [N] bool
        skip_mask_expanded = skip_mask.unsqueeze(1).expand(-1, self.output_width)
        processed_action = torch.where(
            skip_mask_expanded, last_cmd_for_envs, new_command
        )
        # If skip would replay an uninitialized last_command (still all zeros), use
        # new_command instead so extras (e.g. target height) are not forced to zero.
        if torch.any(skip_mask):
            last_uninit = torch.sum(torch.abs(last_cmd_for_envs), dim=1) < 1e-5
            fix_skip = skip_mask & last_uninit
            if torch.any(fix_skip):
                fix_exp = fix_skip.unsqueeze(1).expand(-1, self.output_width)
                processed_action = torch.where(fix_exp, new_command, processed_action)

        # Update last_command buffer for non-skip environments (active + lock_skip)
        # Need to use env_ids to index into the global buffer
        non_skip_mask = ~skip_mask  # Everything except mode_flag == -2
        if torch.any(non_skip_mask):
            non_skip_env_ids = env_ids[non_skip_mask]  # env indices for non-skip envs
            self.last_command[non_skip_env_ids] = new_command[non_skip_mask]

        return processed_action

    def reset_idx(self, env_ids: Sequence[int]):
        """Reset planner state for specified environments."""
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        # Reset command buffer for specified environments
        self.last_command[env_ids] = 0.0
        # Reset preprocessor state
        if self._preprocessor is not None:
            self._preprocessor.reset_idx(env_ids)

    def update_obstacles(
        self,
        obstacle_avoidance_path_list: list = None,
        obstacle_ignore_path_list: list = None,
        env_ids: list = None,
    ):
        """Update obstacles (not used for P-controller)."""
        # P-controller doesn't use obstacles
        pass
