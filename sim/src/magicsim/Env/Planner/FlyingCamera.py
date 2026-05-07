import torch
from typing import Sequence
from magicsim.Env.Planner.Utils import (
    quat_normalize,
    quat_slerp_batch,
    quat_mul,
    quat_inv,
    quat_error_to_rotvec,
    integrate_quat_with_omega,
    quat_angle_between,
)
from magicsim.Env.Planner.Planner import Planner


class LinearFlyingCamera:
    def __init__(
        self,
        num_env,
        steps=120,
        pos_threshold=1e-3,
        rot_threshold_deg=1.0,
        device=torch.device("cuda"),
    ):
        """
        Simple linear interpolation + quaternion slerp camera controller (Torch + GPU + env_idxs)

        pose format: [x, y, z, w, x, y, z]  total 7 dimensions

        State:
            pos:           [num_env, 3]
            quat:          [num_env, 4]
            start_pos:     [num_env, 3]  starting position of current interpolation trajectory
            start_quat:    [num_env, 4]  starting orientation of current interpolation trajectory
            target_pos:    [num_env, 3]  active target position
            target_quat:   [num_env, 4]  active target orientation
            last_prop_pos: [num_env, 3]  position proposed in last forward call
            last_prop_quat:[num_env, 4]  orientation proposed in last forward call
            step_idx:      [num_env]     current interpolation step count
        """
        self.num_env = num_env
        self.device = torch.device(device)

        self.steps = steps
        self.pos_threshold = float(pos_threshold)
        self.rot_threshold = torch.deg2rad(
            torch.tensor(rot_threshold_deg, device=self.device)
        )

        self.pos = torch.zeros(num_env, 3, device=device)
        self.quat = torch.zeros(num_env, 4, device=device)
        self.start_pos = torch.zeros(num_env, 3, device=device)
        self.start_quat = torch.zeros(num_env, 4, device=device)
        self.target_pos = torch.zeros(num_env, 3, device=device)
        self.target_quat = torch.zeros(num_env, 4, device=device)
        self.last_prop_pos = torch.zeros(num_env, 3, device=device)
        self.last_prop_quat = torch.zeros(num_env, 4, device=device)
        self.step_idx = torch.zeros(num_env, dtype=torch.long, device=device)
        self.initialized = torch.zeros(num_env, dtype=torch.bool, device=device)

    # ----------------------------------------
    # Initialize current state for env
    # ----------------------------------------
    def set_current_state(self, env_idxs, cur_pos, cur_quat):
        """
        env_idxs: 1D index
        cur_pos:  [3] or [N,3]
        cur_quat: [4] or [N,4] (w,x,y,z)
        """
        env_idxs = torch.as_tensor(env_idxs, dtype=torch.long, device=self.device)

        cur_pos = torch.as_tensor(cur_pos, dtype=torch.float32, device=self.device)
        if cur_pos.ndim == 1:
            cur_pos = cur_pos.unsqueeze(0).expand(env_idxs.shape[0], -1)

        cur_quat = torch.as_tensor(cur_quat, dtype=torch.float32, device=self.device)
        if cur_quat.ndim == 1:
            cur_quat = cur_quat.unsqueeze(0).expand(env_idxs.shape[0], -1)
        cur_quat = quat_normalize(cur_quat)

        # Write state
        self.pos[env_idxs] = cur_pos
        self.quat[env_idxs] = cur_quat

        self.start_pos[env_idxs] = cur_pos
        self.start_quat[env_idxs] = cur_quat

        self.target_pos[env_idxs] = cur_pos
        self.target_quat[env_idxs] = cur_quat

        self.last_prop_pos[env_idxs] = cur_pos
        self.last_prop_quat[env_idxs] = cur_quat

        self.step_idx[env_idxs] = 0
        self.initialized[env_idxs] = True

    # ----------------------------------------
    # forward: env_idxs + target (pos+quat)
    # ----------------------------------------
    def forward(self, env_idxs, target_pos_quat):
        """
        env_idxs: 1D index
        target_pos_quat: [7] or [N,7] → [x,y,z,w,x,y,z]

        Logic:
          - Extract pos_in, quat_in from target_pos_quat
          - Compare with last_prop_(pos,quat)
              pos_diff = L2(pos_in - last_prop_pos)
              rot_diff = angle_between_quat(last_prop_quat, quat_in)
            If pos_diff < pos_threshold and rot_diff < rot_threshold:
              - Consider user input stable → switch target and reset trajectory for this env:
                  start = current pos/quat
                  target = new target
                  step_idx = 0
          - Update last_prop_ regardless of whether target is switched
          - Use start & target + step_idx for linear interpolation + slerp
          - step_idx += 1
        """
        env_idxs = torch.as_tensor(env_idxs, dtype=torch.long, device=self.device)

        if not torch.all(self.initialized[env_idxs]):
            bad = env_idxs[~self.initialized[env_idxs]]
            raise RuntimeError(f"env {bad.tolist()} not initialized")

        target_pos_quat = torch.as_tensor(
            target_pos_quat, dtype=torch.float32, device=self.device
        )
        if target_pos_quat.ndim == 1:
            target_pos_quat = target_pos_quat.unsqueeze(0).expand(env_idxs.shape[0], -1)

        pos_in = target_pos_quat[:, :3]
        quat_in = target_pos_quat[:, 3:]
        quat_in = quat_normalize(quat_in)

        # Check if target has changed significantly
        pos_diff = torch.norm(pos_in - self.last_prop_pos[env_idxs], dim=-1)
        rot_diff = quat_angle_between(quat_in, self.last_prop_quat[env_idxs])

        # Only reset trajectory if target changed significantly
        target_changed = (pos_diff > self.pos_threshold) | (
            rot_diff > self.rot_threshold
        )

        # Update last_prop_* for all envs
        self.last_prop_pos[env_idxs] = pos_in
        self.last_prop_quat[env_idxs] = quat_in

        # Reset trajectory only for envs where target changed
        reset_mask = target_changed
        if torch.any(reset_mask):
            reset_env_idxs = env_idxs[reset_mask]
            self.target_pos[reset_env_idxs] = pos_in[reset_mask]
            self.target_quat[reset_env_idxs] = quat_in[reset_mask]
            self.start_pos[reset_env_idxs] = self.pos[reset_env_idxs]
            self.start_quat[reset_env_idxs] = self.quat[reset_env_idxs]
            self.step_idx[reset_env_idxs] = 0

        # Calculate progress
        step = self.step_idx[env_idxs].float()
        progress = torch.clamp(step / float(self.steps), 0.0, 1.0).unsqueeze(
            -1
        )  # [N,1]

        # Linear interpolate position
        start_pos = self.start_pos[env_idxs]
        target_pos = self.target_pos[env_idxs]
        pos = start_pos + progress * (target_pos - start_pos)

        # Slerp interpolate orientation
        start_quat = self.start_quat[env_idxs]
        target_quat = self.target_quat[env_idxs]
        quat = quat_slerp_batch(start_quat, target_quat, progress.squeeze(-1))

        # Write back current pose
        self.pos[env_idxs] = pos
        self.quat[env_idxs] = quat

        # Increment step_idx (not exceeding steps)
        self.step_idx[env_idxs] = torch.clamp(
            self.step_idx[env_idxs] + 1, 0, self.steps
        )

        # Return [N,7]: [x,y,z,w,x,y,z]
        return torch.cat([pos, quat], dim=-1)


# ----------------------------------------
# Gimbal controller with position + quaternion
# ----------------------------------------
class GimbalFlyingCamera:
    """
    Multi-env gimbal controller with:
      - Second-order damped dynamics for position.
      - Second-order damped dynamics for orientation (quaternion + angular velocity).
      - Target debouncing in both position and orientation.

    Pose format per env: [x, y, z, w, x, y, z].
    """

    def __init__(
        self,
        num_env: int,
        pos_stiffness: float = 20.0,
        pos_damping: float = 8.0,
        rot_stiffness: float = 10.0,
        rot_damping: float = 4.0,
        dt: float = 1 / 60.0,
        pos_target_threshold: float = 0.05,
        rot_target_threshold_deg: float = 2.0,
        device=torch.device("cuda"),
    ):
        """
        Args:
            num_env: number of parallel environments.
            pos_stiffness: spring coefficient k for position.
            pos_damping:  damping coefficient d for position.
            rot_stiffness: spring coefficient k for rotation (on rotation vector).
            rot_damping:   damping coefficient d for rotation.
            dt:            timestep size in seconds (time advance per forward step).
            pos_target_threshold: position debouncing threshold (L2 in meters/units).
            rot_target_threshold_deg: rotation debouncing threshold (degrees).
            device:        "cuda" or "cpu".
        """
        self.num_env = num_env
        self.device = torch.device(device)

        # Position dynamics parameters
        self.k_pos = torch.tensor(float(pos_stiffness), device=self.device)
        self.d_pos = torch.tensor(float(pos_damping), device=self.device)

        # Rotation dynamics parameters
        self.k_rot = torch.tensor(float(rot_stiffness), device=self.device)
        self.d_rot = torch.tensor(float(rot_damping), device=self.device)

        # Time step
        self.dt = torch.tensor(float(dt), device=self.device)

        # Debounce thresholds
        self.pos_target_threshold = torch.tensor(
            float(pos_target_threshold), device=self.device
        )
        self.rot_target_threshold = torch.deg2rad(
            torch.tensor(float(rot_target_threshold_deg), device=self.device)
        )

        # States: position & linear velocity
        self.pos = torch.zeros(num_env, 3, device=self.device)
        self.vel = torch.zeros(num_env, 3, device=self.device)

        # States: orientation (quaternion) & angular velocity
        self.quat = torch.zeros(num_env, 4, device=self.device)
        self.ang_vel = torch.zeros(num_env, 3, device=self.device)

        # Active target (debounced)
        self.target_pos = torch.zeros(num_env, 3, device=self.device)
        self.target_quat = torch.zeros(num_env, 4, device=self.device)

        # Last proposed (raw) target
        self.last_prop_pos = torch.zeros(num_env, 3, device=self.device)
        self.last_prop_quat = torch.zeros(num_env, 4, device=self.device)

        # Initialization mask
        self.initialized = torch.zeros(num_env, dtype=torch.bool, device=self.device)

    # ---------------------------------------------------------
    # Initialize current state for a batch of environments
    # ---------------------------------------------------------
    def set_current_state(self, env_idxs, cur_pos, cur_quat):
        """
        Initialize position and orientation for a set of envs.

        Args:
            env_idxs: 1D list/array/tensor of env indices.
            cur_pos:  [3] or [N,3] position.
            cur_quat: [4] or [N,4] quaternion (w,x,y,z).
        """
        env_idxs = torch.as_tensor(env_idxs, dtype=torch.long, device=self.device)

        cur_pos = torch.as_tensor(cur_pos, dtype=torch.float32, device=self.device)
        if cur_pos.ndim == 1:
            cur_pos = cur_pos.unsqueeze(0).expand(env_idxs.shape[0], -1)

        cur_quat = torch.as_tensor(cur_quat, dtype=torch.float32, device=self.device)
        if cur_quat.ndim == 1:
            cur_quat = cur_quat.unsqueeze(0).expand(env_idxs.shape[0], -1)
        cur_quat = quat_normalize(cur_quat)

        # Position state
        self.pos[env_idxs] = cur_pos
        self.vel[env_idxs] = 0.0

        # Orientation state
        self.quat[env_idxs] = cur_quat
        self.ang_vel[env_idxs] = 0.0

        # Targets initially equal to current pose
        self.target_pos[env_idxs] = cur_pos
        self.target_quat[env_idxs] = cur_quat

        # Last proposed also starts as current pose
        self.last_prop_pos[env_idxs] = cur_pos
        self.last_prop_quat[env_idxs] = cur_quat

        self.initialized[env_idxs] = True

    # ---------------------------------------------------------
    # Forward one timestep for selected environments
    # ---------------------------------------------------------
    def forward(self, env_idxs, target_pos_quat):
        """
        Advance one timestep for specified envs.

        Args:
            env_idxs:        1D list/array/tensor of env indices.
            target_pos_quat: [7] or [N,7] = [x,y,z, w,x,y,z] per env.

        Debouncing logic (per env):
            - Compare proposed target to last proposed:
                pos_diff = ||pos_in - last_prop_pos||
                rot_diff = angle_between(last_prop_quat, quat_in)
            - If pos_diff < pos_threshold AND rot_diff < rot_threshold:
                accept new target (target_pos/target_quat)
            - Always update last_prop_* to current proposed target.

        Dynamics:
            - Position: second-order damped system.
            - Orientation: second-order damped system in rotation-vector space,
              integrated via angular velocity and quaternion integration.

        Returns:
            poses_next: [N, 7] = [x,y,z, w,x,y,z]
        """
        env_idxs = torch.as_tensor(env_idxs, dtype=torch.long, device=self.device)

        # Check initialization
        if not torch.all(self.initialized[env_idxs]):
            not_init = env_idxs[~self.initialized[env_idxs]]
            raise RuntimeError(f"Envs not initialized: {not_init.tolist()}")

        # Normalize target input shape
        target_pos_quat = torch.as_tensor(
            target_pos_quat, dtype=torch.float32, device=self.device
        )
        if target_pos_quat.ndim == 1:
            target_pos_quat = target_pos_quat.unsqueeze(0).expand(env_idxs.shape[0], -1)

        pos_in = target_pos_quat[:, :3]
        quat_in = quat_normalize(target_pos_quat[:, 3:])

        # Always accept new targets, no debouncing
        self.target_pos[env_idxs] = pos_in
        self.target_quat[env_idxs] = quat_in

        # ---- Position dynamics (second-order) ----
        dt = self.dt
        k_pos = self.k_pos
        d_pos = self.d_pos

        pos = self.pos[env_idxs]
        vel = self.vel[env_idxs]
        t_pos = self.target_pos[env_idxs]

        acc_pos = -k_pos * (pos - t_pos) - d_pos * vel
        vel = vel + acc_pos * dt
        pos = pos + vel * dt

        # Write back position state
        self.vel[env_idxs] = vel
        self.pos[env_idxs] = pos

        # ---- Orientation dynamics (second-order) ----
        k_rot = self.k_rot
        d_rot = self.d_rot

        q = self.quat[env_idxs]
        w = self.ang_vel[env_idxs]
        q_tgt = self.target_quat[env_idxs]

        # Quaternion error and rotation-vector error
        q_err = quat_mul(q_tgt, quat_inv(q))
        rot_vec = quat_error_to_rotvec(q_err)  # [N,3]

        # Second-order system for angular motion: w_dot = -k*r - d*w
        ang_acc = -k_rot * rot_vec - d_rot * w

        w = w + ang_acc * dt
        q = integrate_quat_with_omega(q, w, float(dt))

        # Write back orientation state
        self.ang_vel[env_idxs] = w
        self.quat[env_idxs] = q

        # Return concatenated pose [x,y,z, w,x,y,z]
        return torch.cat([pos, q], dim=-1)


# ----------------------------------------
# Planner wrappers for FlyingCamera controllers
# ----------------------------------------
class LinearFlyingCameraPlanner(Planner):
    """
    Linear interpolation camera planner, wrapping LinearFlyingCamera controller.
    Inherits from Planner interface to work with PlannerManager.
    """

    def __init__(
        self,
        camera_manager,
        camera_name: str,
        steps=120,
        pos_threshold=1e-3,
        rot_threshold_deg=1.0,
        device=torch.device("cuda"),
    ):
        """
        Args:
            camera_manager: CameraManager instance
            camera_name: Camera name (key in config)
            steps: Number of interpolation steps
            pos_threshold: Position stability threshold
            rot_threshold_deg: Rotation stability threshold (degrees)
            device: Device
        """
        self.camera_manager = camera_manager
        self.camera_name = camera_name
        self.device = device
        self.num_envs = camera_manager.num_envs

        # Create internal FlyingCamera controller
        self._controller = LinearFlyingCamera(
            num_env=self.num_envs,
            steps=steps,
            pos_threshold=pos_threshold,
            rot_threshold_deg=rot_threshold_deg,
            device=device,
        )
        self._initialized = False

    def forward(self, target_pos_quat: torch.Tensor, env_ids: Sequence[int] = None):
        """
        Planner interface: input target pose, return smoothed pose

        Args:
            target_pos_quat: [N, 7] target pose [x,y,z,w,x,y,z]
            env_ids: List of environment IDs

        Returns:
            planned_pose: [N, 7] planned pose
        """
        if not self._initialized:
            self._initialize()

        if env_ids is None:
            env_ids = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        else:
            env_ids = [int(e) for e in env_ids]

        # Ensure input format is correct
        if target_pos_quat.ndim == 1:
            target_pos_quat = target_pos_quat.unsqueeze(0)

        # Call internal controller
        planned_pose = self._controller.forward(env_ids, target_pos_quat)
        return planned_pose

    def reset_idx(self, env_ids: Sequence[int]):
        """Reset planner state"""
        if not self._initialized:
            return

        # Get current state from CameraManager and reinitialize
        camera_states = self.camera_manager.get_camera_state(
            self.camera_name, env_ids=env_ids
        )

        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        else:
            env_ids = [int(e) for e in env_ids]

        self._controller.set_current_state(
            env_ids, camera_states["pos"], camera_states["quat"]
        )

    def _initialize(self):
        """Initialize: get current camera state from CameraManager"""
        camera_states = self.camera_manager.get_camera_state(self.camera_name)
        all_env_ids = list(range(self.num_envs))
        self._controller.set_current_state(
            all_env_ids, camera_states["pos"], camera_states["quat"]
        )
        self._initialized = True


class GimbalFlyingCameraPlanner(Planner):
    """
    Gimbal dynamics camera planner, wrapping GimbalFlyingCamera controller.
    Inherits from Planner interface to work with PlannerManager.
    """

    def __init__(
        self,
        camera_manager,
        camera_name: str,
        pos_stiffness: float = 20.0,
        pos_damping: float = 8.0,
        rot_stiffness: float = 10.0,
        rot_damping: float = 4.0,
        dt: float = 1 / 60.0,
        pos_target_threshold: float = 0.05,
        rot_target_threshold_deg: float = 2.0,
        device=torch.device("cuda"),
    ):
        """
        Args:
            camera_manager: CameraManager instance
            camera_name: Camera name (key in config)
            pos_stiffness: Position stiffness coefficient
            pos_damping: Position damping coefficient
            rot_stiffness: Rotation stiffness coefficient
            rot_damping: Rotation damping coefficient
            dt: Time step
            pos_target_threshold: Position stability threshold
            rot_target_threshold_deg: Rotation stability threshold (degrees)
            device: Device
        """
        self.camera_manager = camera_manager
        self.camera_name = camera_name
        self.device = device
        self.num_envs = camera_manager.num_envs

        # Create internal FlyingCamera controller
        self._controller = GimbalFlyingCamera(
            num_env=self.num_envs,
            pos_stiffness=pos_stiffness,
            pos_damping=pos_damping,
            rot_stiffness=rot_stiffness,
            rot_damping=rot_damping,
            dt=dt,
            pos_target_threshold=pos_target_threshold,
            rot_target_threshold_deg=rot_target_threshold_deg,
            device=device,
        )
        self._initialized = False

    def forward(self, target_pos_quat: torch.Tensor, env_ids: Sequence[int] = None):
        """
        Planner interface: input target pose, return smoothed pose

        Args:
            target_pos_quat: [N, 7] target pose [x,y,z,w,x,y,z]
            env_ids: List of environment IDs

        Returns:
            planned_pose: [N, 7] planned pose
        """
        if not self._initialized:
            self._initialize()

        if env_ids is None:
            env_ids = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        else:
            env_ids = [int(e) for e in env_ids]

        # Ensure input format is correct
        if target_pos_quat.ndim == 1:
            target_pos_quat = target_pos_quat.unsqueeze(0)

        # Call internal controller
        planned_pose = self._controller.forward(env_ids, target_pos_quat)
        return planned_pose

    def reset_idx(self, env_ids: Sequence[int]):
        """Reset planner state"""
        if not self._initialized:
            return

        # Get current state from CameraManager and reinitialize
        camera_states = self.camera_manager.get_camera_state(
            self.camera_name, env_ids=env_ids
        )

        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        else:
            env_ids = [int(e) for e in env_ids]

        self._controller.set_current_state(
            env_ids, camera_states["pos"], camera_states["quat"]
        )

    def _initialize(self):
        """Initialize: get current camera state from CameraManager"""
        camera_states = self.camera_manager.get_camera_state(self.camera_name)
        all_env_ids = list(range(self.num_envs))
        self._controller.set_current_state(
            all_env_ids, camera_states["pos"], camera_states["quat"]
        )
        self._initialized = True
